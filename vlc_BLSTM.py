#!/usr/bin/env python
# coding: utf-8

import os
import re
import glob

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

BASE_DIR = '/content/drive/MyDrive/IQ_imbalance_BLSTM/predistortion'   # 데이터/모델 위치

# ── 논문 III-B 와 맞춘 하이퍼파라미터 ──
UNITS_L1   = 70      # 첫 번째 BLSTM 은닉층 유닛 수
UNITS_L2   = 90      # 두 번째 BLSTM 은닉층 유닛 수
BATCH_SIZE = 32
N_FEATURES = 4       # [I, Q, eps, phi]

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.set_logical_device_configuration(
            gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=2048)])
    except RuntimeError as e:
        print(e)


def convert_to_float(data):
    if isinstance(data, pd.DataFrame):
        data = data.to_numpy()
    if isinstance(data, np.ndarray):
        if data.dtype.type is np.str_ or data.dtype.type is np.object_:
            data = data.astype(np.float64)
    else:
        raise TypeError("Input data must be a pandas DataFrame or numpy array.")
    return data


def find_latest_blstm_checkpoint(model_dir, epoch_num2, prefix):
    pattern = os.path.join(model_dir, f'{prefix}_*_r.h5')
    best_epoch, best_r, best_i = -1, None, None
    for path_r in glob.glob(pattern):
        m = re.search(rf'{prefix}_(\d+)_r\.h5$', os.path.basename(path_r))
        if not m:
            continue
        ep = int(m.group(1))
        if ep not in epoch_num2:
            continue
        path_i = path_r[:-len('_r.h5')] + '_i.h5'
        if not os.path.exists(path_i):
            print(f"경고: {os.path.basename(path_r)}의 짝(_i.h5)이 없어 건너뜁니다.")
            continue
        if ep > best_epoch:
            best_epoch, best_r, best_i = ep, path_r, path_i
    if best_epoch < 0:
        return 0, None, None
    return epoch_num2.index(best_epoch) + 1, best_r, best_i


# ============================================================
# 메타 로드 (조건화 상수 + 프레임별 파라미터)
# ============================================================
N = 32
meta = np.load(os.path.join(BASE_DIR, 'blstm_train_meta.npz'))
eps_frames   = meta['eps_frames']
phi_frames   = meta['phi_frames']
frame_len    = int(meta['frame_len'])
eps_mid, eps_half = float(meta['eps_mid']), float(meta['eps_half'])
phi_mid, phi_half = float(meta['phi_mid']), float(meta['phi_half'])
assert frame_len % N == 0
seq_per_frame = frame_len // N

# ============================================================
# 데이터 로드 및 시퀀스화 (기저대역 I/Q)
# ============================================================
sig_in_I = convert_to_float(pd.read_csv(os.path.join(BASE_DIR, 'input_iq_I.csv'), header=None).to_numpy()).reshape(-1, N)
sig_in_Q = convert_to_float(pd.read_csv(os.path.join(BASE_DIR, 'input_iq_Q.csv'), header=None).to_numpy()).reshape(-1, N)
tgt_I    = convert_to_float(pd.read_csv(os.path.join(BASE_DIR, 'input_I_r.csv'),  header=None).to_numpy()).reshape(-1, N)
tgt_Q    = convert_to_float(pd.read_csv(os.path.join(BASE_DIR, 'input_Q_r.csv'),  header=None).to_numpy()).reshape(-1, N)

num_seq = len(sig_in_I)
assert len(tgt_I) == num_seq, "입력과 target의 시퀀스 수가 다릅니다!"
split = int(0.8 * num_seq)
print(f"total sequences: {num_seq}, train: {split}, validation: {num_seq - split}")
if split == 0:
    raise ValueError("훈련 시퀀스가 0개입니다. 데이터 파일을 확인하세요.")

# ── 조건화 채널: 프레임별 (eps, phi) → 시퀀스별로 확장 ──
eps_seq = np.repeat((eps_frames - eps_mid) / eps_half, seq_per_frame)
phi_seq = np.repeat((phi_frames - phi_mid) / phi_half, seq_per_frame)
assert len(eps_seq) == num_seq

ch_e = np.repeat(eps_seq[:, None], N, axis=1)   # (num_seq, N)
ch_p = np.repeat(phi_seq[:, None], N, axis=1)

# ============================================================
# 정규화: train 통계 하나로 train/validation 모두 정규화 (추론과 동일 규약)
# ============================================================
train_input_std  = np.std(np.sqrt(sig_in_I[:split]**2 + sig_in_Q[:split]**2))
train_target_std = np.std(np.sqrt(tgt_I[:split]**2  + tgt_Q[:split]**2))
print(f"input_std={train_input_std:.6f}, target_std={train_target_std:.6f}, "
      f"ratio={train_target_std/train_input_std:.4f}")

sig_in_I /= train_input_std
sig_in_Q /= train_input_std
tgt_I    /= train_target_std
tgt_Q    /= train_target_std

# ── 추론용 스케일/상수 저장 (추론은 이 파일 하나만 읽으면 됨) ──
np.savez(os.path.join(BASE_DIR, 'blstm_norm_scale.npz'),
         input_std=train_input_std, target_std=train_target_std,
         eps_mid=eps_mid, eps_half=eps_half,
         phi_mid=phi_mid, phi_half=phi_half)
print("saved: blstm_norm_scale.npz")

# ============================================================
# (N, 4) 입력 구성: [I, Q, eps 채널, phi 채널]
#  I_PD 가 Q 에도 의존하므로 두 branch 를 모두 입력한다.
#  두 네트워크는 같은 입력을 받고 서로 다른 성분을 출력한다.
# ============================================================
x_all = np.stack([sig_in_I, sig_in_Q, ch_e, ch_p], axis=-1).astype(np.float32)
x_train, x_val = x_all[:split], x_all[split:]

y_train_r = tgt_I[:split].astype(np.float32)
y_val_r   = tgt_I[split:].astype(np.float32)
y_train_i = tgt_Q[:split].astype(np.float32)
y_val_i   = tgt_Q[split:].astype(np.float32)

print("train shapes:", x_train.shape, y_train_r.shape)
print("valid shapes:", x_val.shape,   y_val_r.shape)


# ============================================================
# BLSTM 모델 정의 (입력 4채널, 은닉층 70/90)
# ============================================================
def build_model():
    m = keras.Sequential()
    m.add(layers.Bidirectional(
        layers.LSTM(UNITS_L1, return_sequences=True),
        input_shape=(N, N_FEATURES)))
    m.add(layers.Bidirectional(layers.LSTM(UNITS_L2)))
    m.add(layers.Dense(N))
    m.compile(loss='mse', optimizer='rmsprop', metrics=['mse'])
    return m


model2_r = build_model()
model2_i = build_model()
model2_r.summary()

# ============================================================
# 훈련 스케줄 + Resume (새 prefix라 옛 3-feature 모델과 섞이지 않음)
# ============================================================
epoch_num  = [1, 1, 1, 2, 5, 10, 10, 10, 10, 10]
epoch_num2 = [1, 2, 3, 5, 10, 20, 30, 40, 50, 60]
PREFIX = 'vlc_lstm_cond4'

resume_idx, resume_path_r, resume_path_i = \
    find_latest_blstm_checkpoint(BASE_DIR, epoch_num2, PREFIX)

if resume_path_r is not None:
    print(f"\n저장된 가중치 발견:\n  real: {resume_path_r}\n  imag: {resume_path_i}")
    keras.backend.clear_session()
    model2_r = keras.models.load_model(resume_path_r, compile=False)
    model2_i = keras.models.load_model(resume_path_i, compile=False)
    model2_r.compile(loss='mse', optimizer='rmsprop', metrics=['mse'])
    model2_i.compile(loss='mse', optimizer='rmsprop', metrics=['mse'])
    if resume_idx >= len(epoch_num):
        print(f"모든 스테이지({epoch_num2[-1]} epochs)가 이미 완료되었습니다.")
    else:
        print(f"훈련 재개: stage {resume_idx}")
else:
    print("\n저장된 가중치 없음 — stage 0부터 새로 훈련합니다.")

# ============================================================
# 훈련 루프
# ============================================================
for idx in range(resume_idx, len(epoch_num)):
    print(f"\n===== Stage {idx} (누적 목표 {epoch_num2[idx]} epochs) =====")
    print("In-phase component")
    model2_r.fit(x_train, y_train_r, epochs=epoch_num[idx], batch_size=BATCH_SIZE,
                 validation_data=(x_val, y_val_r))
    print("Quadrature component")
    model2_i.fit(x_train, y_train_i, epochs=epoch_num[idx], batch_size=BATCH_SIZE,
                 validation_data=(x_val, y_val_i))

    name_r = os.path.join(BASE_DIR, f'{PREFIX}_{epoch_num2[idx]}_r.h5')
    name_i = os.path.join(BASE_DIR, f'{PREFIX}_{epoch_num2[idx]}_i.h5')
    print(name_r, name_i)
    model2_r.save(name_r)
    model2_i.save(name_i)

    model2_r.evaluate(x_val, y_val_r)
    model2_i.evaluate(x_val, y_val_i)

# ============================================================
# 최종 진단
#  1) 조건화: eps 가 작은 그룹의 gain 이 큰 그룹보다 커야 함
#     (predistortion gain 1/(1+eps/2) 는 eps 에 대해 감소)
#  2) 교차항: phi 가 큰 그룹에서 Q -> I_PD 결합 계수가 커야 함
#     (해석해의 교차항 계수는 sin(phi/2)/((1+eps/2) cos phi))
# ============================================================
yhat_r = model2_r.predict(x_val, batch_size=512)
I_val, Q_val = x_val[..., 0], x_val[..., 1]
eps_val, phi_val = eps_seq[split:], phi_seq[split:]


def lsq_coeffs(rows):
    """yhat_r ~ a*I + b*Q 의 최소제곱 계수."""
    A = np.stack([I_val[rows].ravel(), Q_val[rows].ravel()], axis=1)
    y = yhat_r[rows].ravel()
    return np.linalg.lstsq(A, y, rcond=None)[0]


a_lo, b_lo = lsq_coeffs(eps_val < 0)
a_hi, b_hi = lsq_coeffs(eps_val >= 0)
print(f"\n[조건화 진단] I 계수  eps 낮은 그룹 {a_lo:.4f} / 높은 그룹 {a_hi:.4f}")
print("  -> 왼쪽이 뚜렷이 크면 eps 조건화 성공")

_, b_plo = lsq_coeffs(phi_val < 0)
_, b_phi = lsq_coeffs(phi_val >= 0)
print(f"[교차항 진단] Q 계수  phi 낮은 그룹 {b_plo:.4f} / 높은 그룹 {b_phi:.4f}")
print("  -> 오른쪽이 뚜렷이 크고 둘 다 0이 아니면 교차항 학습 성공")

print("Evaluate : {}".format(np.average((yhat_r - y_val_r) ** 2)))
