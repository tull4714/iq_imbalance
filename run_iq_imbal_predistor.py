# @title
import numpy as np
import time
import os
import matplotlib.pyplot as plt
from scipy.special import erfc
from tensorflow import keras
from scipy.io import savemat

BASE_DIR = '/content/drive/MyDrive/IQ_imbalance_BLSTM/predistortion'
MODEL_EPOCH = 60                     # 사용할 체크포인트 (누적 epoch)

def gen_mapping(M, D):
    #np.random.seed(1)
    rand_I = np.random.rand(1, D)
    rand_Q = np.random.rand(1, D)
    s = None
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

def IQ_est(pilot0, pilot1, power_threshold=1e-6, eq_threshold=1e-3):
    I, Q = np.real(pilot0), np.imag(pilot0)
    If, Qf = np.real(pilot1), np.imag(pilot1)
    A = I**2 + Q**2
    D = I**2 - Q**2
    K = I*If - Q*Qf
    valid = A > power_threshold

    # [MSE 수정] 논문 식 (9): ||I|-|Q|| >= eta 인 샘플만 집합 T 로 사용.
    #  (이전: |I^2-Q^2| < 1e-3 샘플에 근사식 K/A 를 적용 -> 슬롯 기준에서 편향 발생)
    eps = np.zeros_like(I)
    far = np.abs(np.abs(I) - np.abs(Q)) >= eq_threshold
    Delta = 4*I**2*Q**2 + D*K                # 식 (7)의 판별식
    use = valid & far & (Delta >= 0)
    eps[use] = (-2*A[use] + 2*np.sqrt(Delta[use])) / D[use]   # 식 (8)
    if not np.any(use):
        return np.nan, np.nan
    eps_T = np.mean(eps[use])

    h = eps_T / 2.0
    denA = Q * (1 - h**2)                    # |Q| >= |I| 인 경우
    denB = I * (1 - h**2)                    # |Q| <  |I| 인 경우
    useA = use & (np.abs(Q) >= np.abs(I))
    useB = use & (np.abs(Q) < np.abs(I))
    s = np.full_like(I, np.inf)
    s[useA] = (I[useA]*(1+h)**2 - If[useA]) / denA[useA]
    s[useB] = (Q[useB]*(1-h)**2 - Qf[useB]) / denB[useB]
    m = (useA | useB) & (np.abs(s) <= 1.0)
    if not np.any(m):
        return eps_T, np.nan
    phi_T = np.rad2deg(np.mean(np.arcsin(s[m])))   # 식 (10), (11)
    return eps_T, phi_T

def iq_predistort(I, Q, epsilon, phi):
    """기저대역 I/Q 사전왜곡 (M^-1 적용).

    [I; Q] = M(eps, phi) [I_PD; Q_PD] 를 풀어 얻은 폐형해.
    반송파 위상에 의존하지 않으므로 DAC 앞단에서 그대로 구현 가능.

    I, Q    : 기저대역 성분 (같은 shape)
    epsilon : 이득 오차 ((1 +- eps/2) 컨벤션)
    phi     : 위상 오차 [degree]
    """
    h = np.deg2rad(phi) / 2.0          # varphi / 2
    c, s = np.cos(h), np.sin(h)
    cos_phi = np.cos(2.0 * h)          # cos(varphi), det M 의 공통 인자
    I_pd = (c * I + s * Q) / ((1.0 + epsilon / 2.0) * cos_phi)
    Q_pd = (s * I + c * Q) / ((1.0 - epsilon / 2.0) * cos_phi)
    return I_pd, Q_pd

def mirror_predistorter(sp, N, epsilon_c, phi_c_deg):
    epsilon = epsilon_c / 2
    theta = -phi_c_deg * np.pi / (2 * 180)   # 이 시스템(Q를 +sin에 싣는 컨벤션)에 맞춘 부호
    term1 = 0.5 * (np.exp(1j * theta) / (1 + epsilon) + np.exp(-1j * theta) / (1 - epsilon))
    term2 = 0.5 * (np.exp(-1j * theta) / (1 + epsilon) - np.exp(1j * theta) / (1 - epsilon))
    sp_comp = np.zeros_like(sp, dtype=complex)
    for k in range(N):
        k_mirror = (N - k) % N
        sp_comp[:, k] = term1 * sp[:, k] + term2 * np.conj(sp[:, k_mirror])
    return sp_comp

def hard_decision(in_vector, M):
    if M == 2:
        out_vector = np.sign(np.real(in_vector))
    elif M == 4:
        out_vector = np.sign(np.real(in_vector)) + 1j * np.sign(np.imag(in_vector))
    return out_vector

def ber_call_qpsk(a, b, M):
    NumOfBitError = 0
    A = a.flatten()
    B = b.flatten()
    D = np.size(A)
    if M == 2:
        return np.sum(~(B == A)) / D
    for q in range(D):
        decision_I = 1 if np.real(A[q]) > 0 else -1
        decision_Q = 1 if np.imag(A[q]) > 0 else -1
        if decision_I != np.real(B[q]):
            NumOfBitError += 1
        if decision_Q != np.imag(B[q]):
            NumOfBitError += 1
    return NumOfBitError / (2 * D)

def calculate_evm(rx_symbols, ideal_symbols):
    error_power = np.mean(np.abs(rx_symbols - ideal_symbols)**2)
    reference_power = np.mean(np.abs(ideal_symbols)**2)
    evm_rms = np.sqrt(error_power / reference_power)
    return evm_rms, evm_rms * 100, 20 * np.log10(evm_rms)

# ============================================================
# [MSE 추가] 피드백 루프백 + 피드백 경로 잡음
# ============================================================
def loopback(ps, up_cos, up_sin, X):
    """IQI 상향변환 -> (같은 IQI) 피드백 하향변환. ps: 기저대역 복소 신호."""
    ps = np.asarray(ps).reshape(-1, 1)
    rf = np.dot(np.real(ps), up_cos) + np.dot(np.imag(ps), up_sin)   # (n, X)
    fb_I = np.dot(rf, up_cos.T) * 2 / X
    fb_Q = np.dot(rf, up_sin.T) * 2 / X
    return (fb_I + 1j * fb_Q).flatten()

def add_fb_noise(fb, fb_snr_db):
    """피드백 관측 (I~, Q~)에 복소 AWGN 추가. fb_snr_db = inf 이면 무잡음."""
    if np.isinf(fb_snr_db):
        return fb
    p = np.mean(np.abs(fb)**2)
    sig = np.sqrt(p / 10**(fb_snr_db / 10) / 2)
    return fb + sig * (np.random.randn(*fb.shape) + 1j * np.random.randn(*fb.shape))

def estimation_reference(est_ref, b_slot=None):
    """추정에 쓰는 기준 신호. 'pilot': 4-point pilot, 'slot': 송신 슬롯(14 OFDM 심볼)."""
    if est_ref == 'pilot':
        b_r = np.array([1 + 1j, -1 - 1j])
        return np.fft.ifft(np.concatenate((b_r, -b_r)))
    if b_slot is None:
        b_slot = gen_mapping(M, change_block)
    return np.fft.ifft(b_slot.reshape(-1, N)).flatten()

def residual_evm(b_slot, eps, phi, est_eps, est_phi):
    """추정치로 predistortion 후 (채널/잡음 없이) 남는 EVM [%]."""
    sp_ = b_slot.reshape(-1, N)
    ps_ = np.fft.ifft(sp_).reshape(-1, 1)
    I_c, Q_c = iq_predistort(np.real(ps_), np.imag(ps_), est_eps, est_phi)
    uc, us = cos_sampling(X, Fq, phi, eps), sin_sampling(X, Fq, phi, eps)
    rf = np.dot(I_c, uc) + np.dot(Q_c, us)
    dc, ds = cos_sampling(X, Fq, 0, 0), sin_sampling(X, Fq, 0, 0)
    y = np.dot(rf, dc.T) * 2 / X + 1j * np.dot(rf, ds.T) * 2 / X
    Y = np.fft.fft(y.reshape(-1, N), N)
    return calculate_evm(Y.reshape(1, -1), b_slot.reshape(1, -1))[1]

# ============================================================
# 파라미터
# ============================================================
N = 1024
symb = 14
M = 4
X = 4
frames = 10000                 # 프레임 수
Fq = 1
change_block = N * symb
D = change_block * X * frames

np.random.seed(seed=int(time.time()))

# ── [MSE 추가] 피드백 잡음 / 추정 MSE 설정 ──
EPS_RANGE = (0.1, 0.6)         # 본 루프와 같은 epsilon 범위
PHI_RANGE = (1.0, 6.0)         # 본 루프와 같은 phi 범위 [deg]
EST_REF   = 'pilot'            # 'pilot': 4-point pilot (기존), 'slot': 송신 슬롯 14 OFDM 심볼
FB_SNR_MAIN = np.inf           # BER/EVM 루프의 피드백 SNR [dB] (inf = 무잡음 = 기존 결과)

mse_sweep   = 1                # 1: 피드백 SNR 대 추정 MSE 스윕 실행
run_ber     = 1                # 0: 스윕만 하고 BER/EVM 시뮬레이션은 생략
FB_SNR_LIST = np.arange(0, 62, 4).astype(float)
MSE_TRIALS  = 1000             # 피드백 SNR 점당 (eps, phi) 실현 수
MSE_REFS    = ['pilot', 'slot']

down_C = 0
down_upsilon = 0

blstm = 0
# 재학습된 모델은 기저대역 [I, Q, eps, phi] 4개 feature 를 받는다.
MODEL_PREFIX = 'vlc_lstm_cond4'

ch_mode = 2   # 0: AWGN, 1: Rayleigh, 2: Rician
K_list = [4, 8, 12, 16, 20]   # 테스트할 실제 Rician K factor

constel = 0
if constel == 1:
    CONST_SNR = 20
    frames = 1
    ch_mode = 0
    epsilon = 0.3
    phi = 5
    D = change_block * X * frames   # 추가

b_all = np.zeros((1, change_block * frames), dtype=complex)
SNR = np.arange(0, 20, 2)
org_ber = np.zeros((len(K_list), len(SNR)))
ber = np.zeros((len(K_list), len(SNR)))
pred_ber = np.zeros((len(K_list), len(SNR)))
mirror_ber = np.zeros((len(K_list), len(SNR)))
if blstm == 1:
    blstm_ber = np.zeros((len(K_list), len(SNR)))
theo_err = np.zeros(len(SNR))

org_IQ_out = np.zeros((1, D))
IQ_out_r = np.zeros((1, D))
pred_iq_out = np.zeros((1, D))
mirror_iq_out = np.zeros((1, D))
if blstm == 1:
    blstm_iq_out = np.zeros((1, D))

# ============================================================
# [조건화] BLSTM 리소스: 루프 밖에서 한 번만 로드
# ============================================================
if blstm == 1:
    scale = np.load(os.path.join(BASE_DIR, 'blstm_norm_scale.npz'))
    in_std  = float(scale['input_std'])
    tg_std  = float(scale['target_std'])
    eps_mid, eps_half = float(scale['eps_mid']), float(scale['eps_half'])
    phi_mid, phi_half = float(scale['phi_mid']), float(scale['phi_half'])
    keras.backend.clear_session()
    recovery_dl_r = keras.models.load_model(
        os.path.join(BASE_DIR, f'{MODEL_PREFIX}_{MODEL_EPOCH}_r.h5'), compile=False)
    recovery_dl_i = keras.models.load_model(
        os.path.join(BASE_DIR, f'{MODEL_PREFIX}_{MODEL_EPOCH}_i.h5'), compile=False)
    print(f"[BLSTM] loaded cond model (epoch {MODEL_EPOCH}), "
          f"in_std={in_std:.6f}, tg_std={tg_std:.6f}")

# ============================================================
# [MSE 추가] 피드백 SNR 대 파라미터 추정 MSE 스윕
#   MSE_eps = E[(eps_T - eps)^2],  MSE_phi = E[(phi_T - phi)^2] [deg^2]
#   residual EVM: 추정치로 predistortion 후 채널/수신잡음 없이 남는 EVM
# ============================================================
if mse_sweep == 1:
    # 무잡음 확인: 추정치가 참값과 일치해야 함
    _uc, _us = cos_sampling(X, Fq, 5.0, 0.3), sin_sampling(X, Fq, 5.0, 0.3)
    _ref = estimation_reference('pilot')
    print("[MSE] noiseless check (0.3, 5.0):", IQ_est(_ref, loopback(_ref, _uc, _us, X)))

    mse_eps = np.zeros((len(MSE_REFS), len(FB_SNR_LIST)))
    mse_phi = np.zeros_like(mse_eps)
    res_evm = np.zeros_like(mse_eps)
    fail    = np.zeros_like(mse_eps)
    for r_i, ref_name in enumerate(MSE_REFS):
        for s_i, fb_snr in enumerate(FB_SNR_LIST):
            se_e, se_p, ev = [], [], []
            for _ in range(MSE_TRIALS):
                eps_t = np.random.uniform(*EPS_RANGE)
                phi_t = np.random.uniform(*PHI_RANGE)
                b_slot = gen_mapping(M, change_block)
                ref = estimation_reference(ref_name, b_slot)
                uc, us = cos_sampling(X, Fq, phi_t, eps_t), sin_sampling(X, Fq, phi_t, eps_t)
                fb = add_fb_noise(loopback(ref, uc, us, X), fb_snr)
                with np.errstate(all='ignore'):
                    e_hat, p_hat = IQ_est(ref, fb)
                if not (np.isfinite(e_hat) and np.isfinite(p_hat)):
                    fail[r_i, s_i] += 1
                    continue
                se_e.append((e_hat - eps_t)**2)
                se_p.append((p_hat - phi_t)**2)
                ev.append(residual_evm(b_slot, eps_t, phi_t, e_hat, p_hat))
            mse_eps[r_i, s_i] = np.mean(se_e)
            mse_phi[r_i, s_i] = np.mean(se_p)
            res_evm[r_i, s_i] = np.mean(ev)
            print(f"[MSE] ref={ref_name:5s} fb SNR {fb_snr:4.0f} dB : "
                  f"MSE_eps {mse_eps[r_i, s_i]:.3e}, MSE_phi {mse_phi[r_i, s_i]:.3e} deg^2, "
                  f"residual EVM {res_evm[r_i, s_i]:.3f}%, fail {int(fail[r_i, s_i])}")

    os.makedirs(os.path.join(BASE_DIR, 'variable'), exist_ok=True)
    savemat(os.path.join(BASE_DIR, 'variable/est_mse.mat'), {
        'fb_snr': FB_SNR_LIST, 'refs': np.array(MSE_REFS, dtype=object),
        'mse_eps': mse_eps, 'mse_phi': mse_phi, 'res_evm': res_evm, 'fail': fail,
        'eps_range': np.array(EPS_RANGE), 'phi_range': np.array(PHI_RANGE),
        'trials': MSE_TRIALS})

    labels = {'pilot': '4-point pilot', 'slot': f'{symb} OFDM symbols (N={N})'}
    for data_, ylab, fname in [(mse_eps, r'MSE of $\hat{\varepsilon}$', 'MSE_eps'),
                               (mse_phi, r'MSE of $\hat{\varphi}$ [deg$^2$]', 'MSE_phi'),
                               (res_evm, 'Residual EVM [%]', 'Residual_EVM')]:
        plt.figure(figsize=(3.5, 2.7))
        for r_i, ref_name in enumerate(MSE_REFS):
            plt.semilogy(FB_SNR_LIST, data_[r_i], 'o-' if r_i == 0 else 's-',
                         ms=4, lw=1.1, label=labels[ref_name])
        plt.xlabel('Feedback SNR (dB)', fontsize=9)
        plt.ylabel(ylab, fontsize=9)
        plt.tick_params(labelsize=8)
        plt.grid(True, which='both', ls=':', lw=0.5)
        plt.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(fname + '.png', dpi=200)
        plt.savefig(fname + '.pdf')

if run_ber != 1:
    frames = 0

se_eps_main, se_phi_main = [], []   # 본 루프의 추정 제곱오차
for i in range(frames):
    org_up_cos = cos_sampling(X, Fq, 0, 0)
    org_up_sin = sin_sampling(X, Fq, 0, 0)
    if constel != 1:
        epsilon = np.random.uniform(*EPS_RANGE)
        phi = np.random.uniform(*PHI_RANGE)
    up_cos_r = cos_sampling(X, Fq, phi, epsilon)
    up_sin_r = sin_sampling(X, Fq, phi, epsilon)

    # ── 송신 데이터 ──
    b = gen_mapping(M, change_block)

    # ── 피드백 루프백으로 (eps, phi) 추정 (피드백 SNR = FB_SNR_MAIN) ──
    ref = estimation_reference(EST_REF, b)
    rx_out = add_fb_noise(loopback(ref, up_cos_r, up_sin_r, X), FB_SNR_MAIN)
    est_epsilon, est_phi = IQ_est(ref, rx_out)
    se_eps_main.append((est_epsilon - epsilon)**2)
    se_phi_main.append((est_phi - phi)**2)
    print(f"{i}/{frames}th - "
          f"Epsilon: {epsilon}, Estimated epsilon: {est_epsilon}, "
          f"Phi: {phi}, Estimated phi: {est_phi}\n")

    sp = b.reshape(-1, N)
    ps = np.fft.ifft(sp).reshape(-1, 1)
    I_r = np.real(ps)
    Q_r = np.imag(ps)

    b_all[:, i * change_block : (i+1) * change_block] = b
    
    org_I = np.dot(I_r, org_up_cos)
    org_Q = np.dot(Q_r, org_up_sin)
    org_IQ_out[:, i * change_block * X: (i + 1) * change_block * X] = \
        org_I.reshape(1, -1) + org_Q.reshape(1, -1)

    conv_I = np.dot(I_r, up_cos_r)
    conv_Q = np.dot(Q_r, up_sin_r)
    IQ_out_r[:, i * change_block * X: (i + 1) * change_block * X] = \
        conv_I.reshape(1, -1) + conv_Q.reshape(1, -1)

    # ── 기저대역 predistortion ──
    # 기저대역 I_r, Q_r 에 M^-1 을 적용한 뒤 상향변환한다.
    I_r_Comp, Q_r_Comp = iq_predistort(I_r, Q_r, est_epsilon, est_phi)
    pred_iq_out[:, i * change_block * X: (i + 1) * change_block * X] = \
        np.dot(I_r_Comp, up_cos_r).reshape(1, -1) + np.dot(Q_r_Comp, up_sin_r).reshape(1, -1)

    # ── 주파수영역(mirror) predistortion: 추정치 사용으로 복원 ──
    sp_mirror = mirror_predistorter(sp, N, 0.35, 3.5)
    ps_mirror = np.fft.ifft(sp_mirror).reshape(-1, 1)
    mirror_iq_out[:, i * change_block * X: (i + 1) * change_block * X] = \
        (np.dot(np.real(ps_mirror), up_cos_r)).reshape(1, -1) + \
        (np.dot(np.imag(ps_mirror), up_sin_r)).reshape(1, -1)

    # ── [조건화] BLSTM predistorter ──
    if blstm == 1:
        # 신호 채널: 훈련과 동일한 규약 (기저대역 I/Q 를 train input_std로 정규화)
        sig_r = (I_r.reshape(-1) / in_std).reshape(-1, N)
        sig_i = (Q_r.reshape(-1) / in_std).reshape(-1, N)

        # 조건화 채널: pilot 추정치를 훈련과 동일하게 정규화
        eps_n = (est_epsilon - eps_mid) / eps_half
        phi_n = (est_phi     - phi_mid) / phi_half
        if abs(eps_n) > 1.0 or abs(phi_n) > 1.0:
            print(f"경고: 추정 (eps={est_epsilon:.3f}, phi={est_phi:.2f})가 "
                  f"학습 범위 밖입니다. 외삽 성능 저하 가능.")
        ch_e = np.full_like(sig_r, eps_n)
        ch_p = np.full_like(sig_r, phi_n)

        # 두 네트워크 모두 [I, Q, eps, phi] 를 입력받는다
        x_in = np.stack([sig_r, sig_i, ch_e, ch_p], axis=-1).astype(np.float32)

        out_r = recovery_dl_r.predict(x_in, batch_size=512, verbose=0)
        out_i = recovery_dl_i.predict(x_in, batch_size=512, verbose=0)

        # 복원 후 기저대역 상태에서 상향변환
        blstm_I_r = (out_r.reshape(-1) * tg_std).reshape(-1, 1)
        blstm_Q_r = (out_i.reshape(-1) * tg_std).reshape(-1, 1)

        blstm_iq_out[i * change_block: (i + 1) * change_block, :] = \
            np.dot(blstm_I_r, up_cos_r).reshape(1, -1) + \
            np.dot(blstm_Q_r, up_sin_r).reshape(1, -1)

if run_ber != 1:
    raise SystemExit("[MSE] sweep finished (run_ber = 0)")

print(f"[MSE] main loop (ref={EST_REF}, fb SNR={FB_SNR_MAIN} dB, {frames} frames): "
      f"MSE_eps {np.nanmean(se_eps_main):.3e}, MSE_phi {np.nanmean(se_phi_main):.3e} deg^2")

# ============================================================
# 신호 파워 (스케일 정합 확인 포함)
# ============================================================
sigpwr_org = np.linalg.norm(org_IQ_out) ** 2 / org_IQ_out.shape[1]
sigpwr = np.linalg.norm(IQ_out_r) ** 2 / IQ_out_r.shape[1]
sigpwr_pred = np.linalg.norm(pred_iq_out) ** 2 / pred_iq_out.shape[1]
sigpwr_mirror = np.linalg.norm(mirror_iq_out) ** 2 / mirror_iq_out.shape[1]
if blstm == 1:
    sigpwr_blstm = np.linalg.norm(blstm_iq_out) ** 2 / blstm_iq_out.shape[1]
    print(f"sigpwr_pred  : {sigpwr_pred:.6f}")
    print(f"sigpwr_blstm : {sigpwr_blstm:.6f}")
    print(f"BLSTM/Pred power ratio: {sigpwr_blstm / sigpwr_pred:.4f} (1.0에 가까울수록 스케일 정합)")

for i_k in range(len(K_list)):
    print("Channel mode: ", ch_mode)
    for m in range(len(SNR)):
        snr_now = CONST_SNR if constel == 1 else SNR[m]
        print(f"SNR: {snr_now}\n")
        snr_wp = 10 ** (snr_now / 10)

        sgma_org = np.sqrt(X * sigpwr_org / snr_wp / 2 / np.log2(M))
        sgma = np.sqrt(X * sigpwr / snr_wp / 2 / np.log2(M))
        sgma_pred = np.sqrt(X * sigpwr_pred / snr_wp / 2 / np.log2(M))
        sgma_mirror = np.sqrt(X * sigpwr_mirror / snr_wp / 2 / np.log2(M))
        if blstm == 1:
            sgma_blstm = np.sqrt(X * sigpwr_blstm / snr_wp / 2 / np.log2(M))

        # 기법 간 공정 비교를 위해 노이즈를 공유 (분산은 각 신호 파워 기준으로 개별 계산)
        base = np.random.randn(1, np.size(org_IQ_out))
        n_org    = sgma_org    * base
        n_pred   = sgma_pred   * base
        n_mirror = sgma_mirror * base
        n        = sgma        * base
        if blstm == 1:
            n_blstm  = sgma_blstm  * base

        if ch_mode == 0:      # AWGN
            org_receive = org_IQ_out + n_org
            receive_data = IQ_out_r + n
            pred_rev = pred_iq_out + n_pred
            mirror_rev = mirror_iq_out + n_mirror
            if blstm == 1:
                blstm_rev = blstm_iq_out + n_blstm
        elif ch_mode == 1:    # Rayleigh
            h_Ideal = (np.random.randn(org_IQ_out.size) + 1j * np.random.randn(org_IQ_out.size)) / np.sqrt(2)
            h_normal = (np.random.randn(IQ_out_r.size) + 1j * np.random.randn(IQ_out_r.size)) / np.sqrt(2)
            h_comp = (np.random.randn(pred_iq_out.size) + 1j * np.random.randn(pred_iq_out.size)) / np.sqrt(2)
            h_mirror = (np.random.randn(mirror_iq_out.size) + 1j * np.random.randn(mirror_iq_out.size)) / np.sqrt(2)
            org_receive = (h_Ideal * org_IQ_out + n_org) / h_Ideal
            receive_data = (h_normal * IQ_out_r + n) / h_normal
            pred_rev = (h_comp * pred_iq_out + n_pred) / h_comp
            mirror_rev = (h_mirror * mirror_iq_out + n_mirror) / h_mirror
            if blstm == 1:
                h_blstm = (np.random.randn(blstm_iq_out.size) + 1j * np.random.randn(blstm_iq_out.size)) / np.sqrt(2)
                blstm_rev = (h_blstm * blstm_iq_out + n_blstm) / h_blstm
        else:                 # Rician
            # k_0 = 4 * (i_k + 1)
            k_0 = K_list[i_k]
            print("Rician K factor:", k_0)   # 실제 K가 출력됨
            frame = org_IQ_out.size
            h = np.sqrt(k_0 / (1 + k_0)) * np.ones((1, frame)) + \
                np.sqrt(1 / (1 + k_0)) * ((np.random.randn(1, frame) + 1j * np.random.randn(1, frame)) / np.sqrt(2))
            org_receive  = (h * org_IQ_out  + n_org)  / h
            receive_data = (h * IQ_out_r    + n)      / h
            pred_rev     = (h * pred_iq_out + n_pred) / h
            mirror_rev   = (h * mirror_iq_out + n_mirror) / h
            if blstm == 1:
                blstm_rev = (h * blstm_iq_out + n_blstm) / h

        down_cos_r = cos_sampling(X, Fq, down_C, down_upsilon)
        down_sin_r = sin_sampling(X, Fq, down_C, down_upsilon)
        org_down_cos = cos_sampling(X, Fq, 0, 0)
        org_down_sin = sin_sampling(X, Fq, 0, 0)

        def rx_chain(rev, dc, ds):
            rx_iq = rev.reshape(-1, X)
            r_i = np.dot(rx_iq, dc.T) * 2 / X
            r_q = np.dot(rx_iq, ds.T) * 2 / X
            return np.fft.fft((r_i + r_q * 1j).reshape(-1, N), N)

        org_fft_out = rx_chain(org_receive, org_down_cos, org_down_sin)
        fft_out_r = rx_chain(receive_data, down_cos_r, down_sin_r)
        pred_fft_out = rx_chain(pred_rev, down_cos_r, down_sin_r)
        mirror_fft_out = rx_chain(mirror_rev, down_cos_r, down_sin_r)
        if blstm == 1:
            blstm_fft_out = rx_chain(blstm_rev, down_cos_r, down_sin_r)

        mse_blstm_txt = ""
        if blstm == 1:
            mse_blstm = np.mean(np.abs(b_all - blstm_fft_out.reshape(1, -1)) ** 2)
            evm_blstm = calculate_evm(blstm_fft_out.reshape(1, -1), b_all)
            mse_blstm_txt = f", blstm MSE {mse_blstm:.5f}, EVM {evm_blstm[1]:.2f}%"
        mse_org = np.mean(np.abs(b_all - org_fft_out.reshape(1, -1)) ** 2)
        evm_org = calculate_evm(org_fft_out.reshape(1, -1), b_all)
        mse_iq = np.mean(np.abs(b_all - fft_out_r.reshape(1, -1)) ** 2)
        evm_iq = calculate_evm(fft_out_r.reshape(1, -1), b_all)
        mse_mirror = np.mean(np.abs(b_all - mirror_fft_out.reshape(1, -1)) ** 2)
        evm_mirror = calculate_evm(mirror_fft_out.reshape(1, -1), b_all)
        mse_pred = np.mean(np.abs(b_all - pred_fft_out.reshape(1, -1)) ** 2)
        evm_pred = calculate_evm(pred_fft_out.reshape(1, -1), b_all)
        print(f"SNR {snr_now}dB: ideal MSE {mse_org:.5f}, EVM {evm_org[1]:.2f}%, IQ imbalance MSE {mse_iq:.5f}, EVM {evm_iq[1]:.2f}%, mirror MSE {mse_mirror:.5f}, EVM {evm_mirror[1]:.2f}%")
        print(f"SNR {snr_now}dB: pred MSE {mse_pred:.5f}, EVM {evm_pred[1]:.2f}%{mse_blstm_txt}")
            
        org_ber[i_k, m] = ber_call_qpsk(hard_decision(org_fft_out.reshape(1, -1), M), b_all, M)
        ber[i_k, m] = ber_call_qpsk(hard_decision(fft_out_r.reshape(1, -1), M), b_all, M)
        pred_ber[i_k, m] = ber_call_qpsk(hard_decision(pred_fft_out.reshape(1, -1), M), b_all, M)
        mirror_ber[i_k, m] = ber_call_qpsk(hard_decision(mirror_fft_out.reshape(1, -1), M), b_all, M)
        if blstm == 1:
            blstm_ber[i_k, m] = ber_call_qpsk(hard_decision(blstm_fft_out.reshape(1, -1), M), b_all, M)
            print(f"org:{org_ber[i_k, m]:.5f}, imb:{ber[i_k, m]:.5f}, pred:{pred_ber[i_k, m]:.5f}, "
                  f"blstm:{blstm_ber[i_k, m]:.5f}, mirror:{mirror_ber[i_k, m]:.5f}\n")
        else:
            print(f"org:{org_ber[i_k, m]:.5f}, imb:{ber[i_k, m]:.5f}, "
                  f"pred:{pred_ber[i_k, m]:.5f}, mirror:{mirror_ber[i_k, m]:.5f}\n")

        if constel == 1:
            savemat(os.path.join(BASE_DIR, f'variable/constellation.mat'), {
                'iq':      fft_out_r.reshape(-1),
                'pred':    pred_fft_out.reshape(-1),
                'ref':     b_all.reshape(-1),
                'snr':     CONST_SNR,
                'epsilon': epsilon,
                'phi':     phi})
            print(f"[const] Eb/N0={CONST_SNR}dB, {change_block * frames} symbols saved")
            break
            
    if ch_mode == 0 or ch_mode == 1:
        break

for i in range(len(SNR)):
    t_snr = 10 ** (SNR[i] / 10)
    theo_err[i] = (1 / 2) * erfc(np.sqrt(t_snr)) - (1 / 8) * (erfc(np.sqrt(t_snr))) ** 2

if constel != 1:
    plt.figure(4)
    plt.semilogy(SNR, org_ber[4], 'g', label='No IQ imbalance BER')
    if ch_mode == 0:
        plt.semilogy(SNR, theo_err, 'k--', label='Theoretical BER')
    plt.semilogy(SNR, ber[4], 'b', label='Simulated BER (Imbalance)')
    plt.semilogy(SNR, pred_ber[4], 'r', label='Time-Domain Predistortion')
    if blstm == 1:
        plt.semilogy(SNR, blstm_ber[4], 'm', label='BLSTM (conditioned)')
    plt.semilogy(SNR, mirror_ber[4], 'c-x', label='Conv Freq-Domain Pre-dist')
    plt.xlabel('SNR (dB)')
    plt.ylabel('BER')
    plt.axis([0, 20, 1e-5, 1])
    plt.grid(True, which='both')
    plt.legend()
    plt.savefig('BER_plot_cond.png')
else:
    plt.plot(np.real(fft_out_r), np.imag(fft_out_r), 'b.');