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

BASE_DIR = '/content/drive/MyDrive'   # 데이터/모델 위치

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
frame_branch = int(meta['frame_branch'])
eps_mid, eps_half = float(meta['eps_mid']), float(meta['eps_half'])
phi_mid, phi_half = float(meta['phi_mid']), float(meta['phi_half'])
assert frame_branch % N == 0
seq_per_frame = frame_branch // N

# ============================================================
# 데이터 로드 및 시퀀스화
# ============================================================
sig_in_I = convert_to_float(pd.read_csv(os.path.join(BASE_DIR, 'input_iq_I.csv'), header=None).to_numpy()).reshape(-1, N)
sig_in_Q = convert_to_float(pd.read_csv(os.path.join(BASE_DIR, 'input_iq_Q.csv'), header=None).to_numpy()).reshape(-1, N)
tgt_I    = convert_to_float(pd.read_csv(os.path.join(BASE_DIR, 'input_I_r.csv'),  header=None).to_numpy()).reshape(-1, N)
tgt_Q    = convert_to_float(pd.read_csv(os.path.join(BASE_DIR, 'input_Q_r.csv'),  header=None).to_numpy()).reshape(-1, N)

num_seq = len(sig_in_I)
assert len(tgt_I) == num_seq, "입력과 target의 시퀀스 수가 다릅니다!"
split = int(0.8 * num_seq)
print(f"total sequences: {num_seq}, train: {split}, test: {num_seq - split}")
if split == 0:
    raise ValueError("훈련 시퀀스가 0개입니다. 데이터 파일을 확인하세요.")

# ── 조건화 채널: 프레임별 (eps, phi) → 시퀀스별로 확장 ──
eps_seq = np.repeat((eps_frames - eps_mid) / eps_half, seq_per_frame)
phi_seq = np.repeat((phi_frames - phi_mid) / phi_half, seq_per_frame)
assert len(eps_seq) == num_seq

ch_e = np.repeat(eps_seq[:, None], N, axis=1)   # (num_seq, N)
ch_p = np.repeat(phi_seq[:, None], N, axis=1)

# ============================================================
# 정규화: train 통계 하나로 train/test 모두 정규화 (추론과 동일 규약)
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
# (N, 3) 입력 구성: [신호, eps 채널, phi 채널]
# ============================================================
x_train_r = np.stack([sig_in_I[:split], ch_e[:split], ch_p[:split]], axis=-1).astype(np.float32)
x_test_r  = np.stack([sig_in_I[split:], ch_e[split:], ch_p[split:]], axis=-1).astype(np.float32)
x_train_i = np.stack([sig_in_Q[:split], ch_e[:split], ch_p[:split]], axis=-1).astype(np.float32)
x_test_i  = np.stack([sig_in_Q[split:], ch_e[split:], ch_p[split:]], axis=-1).astype(np.float32)

y_train_r = tgt_I[:split].astype(np.float32)
y_test_r  = tgt_I[split:].astype(np.float32)
y_train_i = tgt_Q[:split].astype(np.float32)
y_test_i  = tgt_Q[split:].astype(np.float32)

print("train shapes:", x_train_r.shape, y_train_r.shape)
print("test shapes :", x_test_r.shape,  y_test_r.shape)

# ============================================================
# BLSTM 모델 정의 (입력 3채널)
# ============================================================
model2_r = keras.Sequential()
model2_i = keras.Sequential()
model2_r.add(layers.Bidirectional(layers.LSTM(90, return_sequences=True, input_shape=(N, 3))))
model2_i.add(layers.Bidirectional(layers.LSTM(90, return_sequences=True, input_shape=(N, 3))))
model2_r.add(layers.Bidirectional(layers.LSTM(90)))
model2_i.add(layers.Bidirectional(layers.LSTM(90)))
model2_r.add(layers.Dense(N))
model2_i.add(layers.Dense(N))
model2_r.compile(loss='mse', optimizer='rmsprop', metrics=['mse'])
model2_i.compile(loss='mse', optimizer='rmsprop', metrics=['mse'])

# ============================================================
# 훈련 스케줄 + Resume (새 prefix라 옛 모델과 섞이지 않음)
# ============================================================
epoch_num  = [1, 1, 1, 2, 5, 10, 10, 10, 10, 10, 10, 10, 10, 10]
epoch_num2 = [1, 2, 3, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
PREFIX = 'vlc_lstm_cond'

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
    print("Real Part")
    model2_r.fit(x_train_r, y_train_r, epochs=epoch_num[idx], batch_size=128)
    print("Imaginary Part")
    model2_i.fit(x_train_i, y_train_i, epochs=epoch_num[idx], batch_size=128)

    name_r = os.path.join(BASE_DIR, f'{PREFIX}_{epoch_num2[idx]}_r.h5')
    name_i = os.path.join(BASE_DIR, f'{PREFIX}_{epoch_num2[idx]}_i.h5')
    print(name_r, name_i)
    model2_r.save(name_r)
    model2_i.save(name_i)

    model2_r.evaluate(x_test_r, y_test_r)
    model2_i.evaluate(x_test_i, y_test_i)

# ============================================================
# 최종 진단: 조건화가 실제로 동작하는지 확인
#  - eps가 작은 그룹과 큰 그룹에서 m=0(cos 에너지 위치) gain 비교
#  - 조건화가 동작하면: eps 작은 그룹 gain > eps 큰 그룹 gain
#    (predistortion gain 2/(2+eps)는 eps에 대해 감소하므로)
# ============================================================
yhat_r = model2_r.predict(x_test_r, batch_size=512)
sig_test = x_test_r[..., 0]
cols = np.arange(0, N, 4)          # X=4 기준 cos 에너지 위치
eps_test = eps_seq[split:]

def group_gain(rows):
    xs = sig_test[rows][:, cols]
    ps = yhat_r[rows][:, cols]
    return np.sum(ps * xs) / np.sum(xs * xs)

low, high = eps_test < 0, eps_test >= 0
print(f"\n[조건화 진단] gain(m=0)  eps<{eps_mid} 그룹: {group_gain(low):.4f}   "
      f"eps>{eps_mid} 그룹: {group_gain(high):.4f}")
print("→ 왼쪽이 오른쪽보다 뚜렷이 크면 조건화 성공, 두 값이 같으면 조건화 실패")
print("Evaluate : {}".format(np.average((yhat_r - y_test_r) ** 2)))