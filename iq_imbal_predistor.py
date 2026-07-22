"""
BLSTM 훈련 데이터 생성 (조건화 버전)
- 입력  : IQ imbalance 없는 원본 branch 신호
- Target: 프레임별 (eps, phi)로 predistortion한 신호
- 프레임별 (eps, phi)와 조건화 정규화 상수를 meta npz로 저장
"""
import os
import csv
import numpy as np

OUT_DIR = '.'   # Colab이면 '/content/drive/MyDrive' 등으로 변경

# ── 시뮬레이션과 동일한 함수들 ──
def gen_mapping(M, D, rng):
    rand_I = rng.random((1, D))
    rand_Q = rng.random((1, D))
    if M == 2:
        s = 2 * np.fix(rand_I * 2) - 1
    elif M == 4:
        s = (2 * np.fix(rand_I * 2) - 1) + 1j * (2 * np.fix(rand_Q * 2) - 1)
    elif M == 16:
        s = (-2 * np.fix(rand_I * 4) + 3) + 1j * (-2 * np.fix(rand_Q * 4) + 3)
    return s

def cos_sampling(X, Fq, C, upsilon):
    a = np.empty(X)
    for t in range(X):
        a[t] = (1 + upsilon / 2) * np.cos((2 * Fq * np.pi * t) / X + np.pi * C / 360)
    return a.reshape(1, -1)

def sin_sampling(X, Fq, C, upsilon):
    a = np.empty(X)
    for t in range(X):
        a[t] = (1 - upsilon / 2) * np.sin((2 * Fq * np.pi * t) / X - np.pi * C / 360)
    return a.reshape(1, -1)

def cos_predistor(data, X, Fq, epsilon, phi):
    a_Ir = 2 / (2 + epsilon) * data
    m = np.arange(0, X)
    S_IC = 1 / (np.cos((2 * Fq * np.pi * m) / X + np.pi * phi / 360))
    return a_Ir * S_IC

def sin_predistor(data, X, Fq, epsilon, phi):
    a_Qr = 2 / (2 - epsilon) * data
    m = np.arange(0, X)
    S_QC = 1 / (np.sin((2 * Fq * np.pi * m) / X - np.pi * phi / 360))
    return a_Qr * S_QC

# ── 파라미터 ──
N, M, X, Fq = 32, 4, 4, 1

EPS_MIN, EPS_MAX = 0.1, 0.6      # 테스트(0.2~0.5)보다 넓게
PHI_MIN, PHI_MAX = 1.0, 6.0      # 테스트(2~5°)보다 넓게
EPS_MID,  EPS_HALF = (EPS_MIN + EPS_MAX) / 2, (EPS_MAX - EPS_MIN) / 2
PHI_MID,  PHI_HALF = (PHI_MIN + PHI_MAX) / 2, (PHI_MAX - PHI_MIN) / 2

num_frames        = 2000
symbols_per_frame = 50
SEED = 42
rng = np.random.default_rng(SEED)

frame_qam    = N * symbols_per_frame     # 프레임당 QAM 심볼
frame_branch = frame_qam * X             # 프레임당 branch 샘플 (N의 배수 → 시퀀스가 프레임 경계를 넘지 않음)
total_branch = num_frames * frame_branch
assert frame_branch % N == 0

input_I  = np.zeros(total_branch)
input_Q  = np.zeros(total_branch)
target_I = np.zeros(total_branch)
target_Q = np.zeros(total_branch)
eps_log  = np.zeros(num_frames)
phi_log  = np.zeros(num_frames)

org_up_cos = cos_sampling(X, Fq, 0, 0)
org_up_sin = sin_sampling(X, Fq, 0, 0)

for f in range(num_frames):
    eps = rng.uniform(EPS_MIN, EPS_MAX)
    phi = rng.uniform(PHI_MIN, PHI_MAX)
    eps_log[f], phi_log[f] = eps, phi

    b = gen_mapping(M, frame_qam, rng)
    sp = b.reshape(-1, N)
    ps = np.fft.ifft(sp).reshape(-1, 1)
    I_r, Q_r = np.real(ps), np.imag(ps)

    org_I = np.dot(I_r, org_up_cos)
    org_Q = np.dot(Q_r, org_up_sin)

    # target: 참값 (eps, phi)로 predistortion (추정 오차 없는 정답)
    I_comp = cos_predistor(org_I, X, Fq, eps, phi)
    Q_comp = sin_predistor(org_Q, X, Fq, eps, phi)

    s0, s1 = f * frame_branch, (f + 1) * frame_branch
    input_I[s0:s1]  = org_I.reshape(-1)
    input_Q[s0:s1]  = org_Q.reshape(-1)
    target_I[s0:s1] = I_comp.reshape(-1)
    target_Q[s0:s1] = Q_comp.reshape(-1)

    if (f + 1) % 200 == 0:
        print(f"frame {f+1}/{num_frames}  (eps={eps:.3f}, phi={phi:.2f}°)")

print("\n=== 생성 데이터 검증 ===")
print(f"eps: {eps_log.min():.3f} ~ {eps_log.max():.3f} (평균 {eps_log.mean():.3f})")
print(f"phi: {phi_log.min():.2f}° ~ {phi_log.max():.2f}° (평균 {phi_log.mean():.2f}°)")
print(f"총 branch 샘플: {total_branch}, 시퀀스 수: {total_branch // N}")

def save_row_csv(filename, arr):
    path = os.path.join(OUT_DIR, filename)
    with open(path, mode='w', newline='') as fp:
        w = csv.writer(fp, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        w.writerow(arr.tolist())
    print(f"saved: {path}")

save_row_csv('input_iq_I.csv', input_I)
save_row_csv('input_iq_Q.csv', input_Q)
save_row_csv('input_I_r.csv',  target_I)
save_row_csv('input_Q_r.csv',  target_Q)

# ── 훈련용 메타 저장: 프레임별 파라미터 + 조건화 정규화 상수 ──
np.savez(os.path.join(OUT_DIR, 'blstm_train_meta.npz'),
         eps_frames=eps_log, phi_frames=phi_log,
         N=N, X=X, frame_branch=frame_branch,
         eps_mid=EPS_MID, eps_half=EPS_HALF,
         phi_mid=PHI_MID, phi_half=PHI_HALF,
         eps_range=[EPS_MIN, EPS_MAX], phi_range=[PHI_MIN, PHI_MAX])
print("saved: blstm_train_meta.npz")