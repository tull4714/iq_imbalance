"""
BLSTM 훈련 데이터 생성 (조건화 버전)
- 입력  : IFFT/PS 직후의 기저대역 I(t), Q(t)
- Target: 프레임별 (eps, phi)로 predistortion 한 기저대역 I_PD, Q_PD
- 프레임별 (eps, phi)와 조건화 정규화 상수를 meta npz로 저장

주의: predistortion 은 기저대역에서 정의되며 반송파 위상에 의존하지 않는다.
      I_PD 가 Q 에도 의존하므로 두 branch 를 함께 다룬다.
"""
import os
import csv
import numpy as np

OUT_DIR = '/content/drive/MyDrive/IQ_imbalance_BLSTM/predistortion'   # Colab이면 '/content/drive/MyDrive' 등으로 변경

# ── 시뮬레이션과 동일한 함수 ──
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


def iq_predistort(I, Q, epsilon, phi):
    """기저대역 I/Q 사전왜곡 (M^-1 적용).

    [I; Q] = M(eps, phi) [I_PD; Q_PD] 의 폐형해.
    epsilon : 이득 오차 ((1 +- eps/2) 컨벤션)
    phi     : 위상 오차 [degree]
    """
    h = np.deg2rad(phi) / 2.0          # varphi / 2
    c, s = np.cos(h), np.sin(h)
    cos_phi = np.cos(2.0 * h)          # cos(varphi), det M 의 공통 인자
    I_pd = (c * I + s * Q) / ((1.0 + epsilon / 2.0) * cos_phi)
    Q_pd = (s * I + c * Q) / ((1.0 - epsilon / 2.0) * cos_phi)
    return I_pd, Q_pd


# ── 파라미터 ──
N, M = 32, 4

EPS_MIN, EPS_MAX = 0.1, 0.6      # 테스트(0.2~0.5)보다 넓게
PHI_MIN, PHI_MAX = 1.0, 6.0      # 테스트(2~5°)보다 넓게
EPS_MID,  EPS_HALF = (EPS_MIN + EPS_MAX) / 2, (EPS_MAX - EPS_MIN) / 2
PHI_MID,  PHI_HALF = (PHI_MIN + PHI_MAX) / 2, (PHI_MAX - PHI_MIN) / 2

num_frames        = 2000
symbols_per_frame = 50
SEED = 42
rng = np.random.default_rng(SEED)

frame_qam  = N * symbols_per_frame     # 프레임당 QAM 심볼 = 기저대역 샘플 수
frame_len  = frame_qam                 # 프레임당 시퀀스 샘플 (N의 배수)
total_len  = num_frames * frame_len
assert frame_len % N == 0

input_I  = np.zeros(total_len)
input_Q  = np.zeros(total_len)
target_I = np.zeros(total_len)
target_Q = np.zeros(total_len)
eps_log  = np.zeros(num_frames)
phi_log  = np.zeros(num_frames)

for f in range(num_frames):
    eps = rng.uniform(EPS_MIN, EPS_MAX)
    phi = rng.uniform(PHI_MIN, PHI_MAX)
    eps_log[f], phi_log[f] = eps, phi

    b = gen_mapping(M, frame_qam, rng)
    sp = b.reshape(-1, N)
    ps = np.fft.ifft(sp).reshape(-1)
    I_r, Q_r = np.real(ps), np.imag(ps)

    # target: 참값 (eps, phi)로 predistortion (추정 오차 없는 정답)
    I_comp, Q_comp = iq_predistort(I_r, Q_r, eps, phi)

    s0, s1 = f * frame_len, (f + 1) * frame_len
    input_I[s0:s1]  = I_r
    input_Q[s0:s1]  = Q_r
    target_I[s0:s1] = I_comp
    target_Q[s0:s1] = Q_comp

    if (f + 1) % 200 == 0:
        print(f"frame {f+1}/{num_frames}  (eps={eps:.3f}, phi={phi:.2f}deg)")

print("\n=== 생성 데이터 검증 ===")
print(f"eps: {eps_log.min():.3f} ~ {eps_log.max():.3f} (평균 {eps_log.mean():.3f})")
print(f"phi: {phi_log.min():.2f} ~ {phi_log.max():.2f} deg (평균 {phi_log.mean():.2f})")
print(f"총 기저대역 샘플: {total_len}, OFDM 심볼(시퀀스) 수: {total_len // N}")

# ── 타깃 정합성 확인: 사전왜곡 후 상향변환하면 이상적 RF 와 일치해야 함 ──
_wt = np.linspace(0, 2 * np.pi, 401)[:, None]
_h  = np.deg2rad(phi_log[-1]) / 2.0
_e  = eps_log[-1]
_Ipd, _Qpd = target_I[-N:], target_Q[-N:]
_rf = ((1 + _e / 2) * _Ipd * np.cos(_wt + _h)
       + (1 - _e / 2) * _Qpd * np.sin(_wt - _h))
_ideal = input_I[-N:] * np.cos(_wt) + input_Q[-N:] * np.sin(_wt)
print(f"타깃 검증(마지막 프레임) max|RF - ideal| = {np.abs(_rf - _ideal).max():.3e}")


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

# ── 훈련용 메타 저장 ──
np.savez(os.path.join(OUT_DIR, 'blstm_train_meta.npz'),
         eps_frames=eps_log, phi_frames=phi_log,
         N=N, frame_len=frame_len,
         eps_mid=EPS_MID, eps_half=EPS_HALF,
         phi_mid=PHI_MID, phi_half=PHI_HALF,
         eps_range=[EPS_MIN, EPS_MAX], phi_range=[PHI_MIN, PHI_MAX])
print("saved: blstm_train_meta.npz")
