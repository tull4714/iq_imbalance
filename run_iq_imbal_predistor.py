import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.special import erfc
from tensorflow import keras

# 훈련 시와 동일한 정규화 함수 추가
def normalize_with_rms(I_data, Q_data):
    magnitude = np.sqrt(I_data**2 + Q_data**2)
    rms_magnitude = np.sqrt(np.mean(magnitude**2))
    if rms_magnitude > 0:
        I_data_normalized = I_data / rms_magnitude
        Q_data_normalized = Q_data / rms_magnitude
    else:
        I_data_normalized = I_data
        Q_data_normalized = Q_data
    return I_data_normalized, Q_data_normalized, rms_magnitude

def denormalize_with_rms(I_data, Q_data, rms_magnitude):
    return I_data * rms_magnitude, Q_data * rms_magnitude

def gen_mapping(M, D):
    np.random.seed(1)
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

# ─────────────────────────────────────────────
#  1. 비선형 I/Q imbalance 모델 (AM-AM / AM-PM)
# ─────────────────────────────────────────────
def apply_nonlinear_iq(I, Q, eps0, eps2, phi0_deg, phi2_deg):
    """
    Signal-dependent (non-linear) I/Q imbalance.

    eps_inst = eps0 + eps2 * |x|^2     (AM-AM: 게인 오차의 진폭 의존)
    phi_inst = phi0 + phi2 * |x|^2     (AM-PM: 위상 오차의 진폭 의존)

    Parameters
    ----------
    I, Q      : 시간 도메인 기저대역 신호의 실수/허수부 (ndarray)
    eps0      : 정적 게인 오차 (예: 0.3)
    eps2      : 게인 오차의 진폭 의존 계수 (0이면 선형 모델과 동일)
    phi0_deg  : 정적 위상 오차 [deg] (예: 5)
    phi2_deg  : 위상 오차의 진폭 의존 계수 [deg/unit power]

    Returns
    -------
    I_out, Q_out : I/Q imbalance가 적용된 신호
    """
    amp2 = I**2 + Q**2                                  # 순시 전력 |x(t)|²

    eps_inst = eps0 + eps2 * amp2                       # 샘플별 게인 오차
    phi_inst = np.deg2rad(phi0_deg + phi2_deg * amp2)   # 샘플별 위상 오차

    I_out = (1 + eps_inst) * I - Q * np.sin(phi_inst)
    Q_out = (1 - eps_inst) * Q - I * np.sin(phi_inst)
    return I_out, Q_out

def IQ_est_approx(tx_signal, fb_signal, power_threshold=1e-6):
    I = np.real(tx_signal)
    Q = np.imag(tx_signal)
    I_fb = np.real(fb_signal)
    Q_fb = np.imag(fb_signal)

    denom = I**2 + Q**2

    # 전력이 충분한 샘플만 선택 (0 나눗셈 및 저SNR 샘플 배제)
    valid = denom > power_threshold

    eps_T = (I[valid] * I_fb[valid] - Q[valid] * Q_fb[valid]
             - (I[valid]**2 - Q[valid]**2)) / denom[valid]

    sin_phi = (2 * I[valid] * Q[valid]
               - (I[valid] * Q_fb[valid] + Q[valid] * I_fb[valid])) / denom[valid]
    sin_phi = np.clip(sin_phi, -1.0, 1.0)
    phi_T = np.arcsin(sin_phi)

    return np.mean(eps_T), np.rad2deg(np.mean(phi_T))

def IQ_est(pilot0, pilot1, power_threshold=1e-6):
    """
    I/Q imbalance estimation.

    Parameters
    ----------
    pilot0 : complex ndarray
        원본 기준 신호 (I = real, Q = imag)
    pilot1 : complex ndarray
        피드백 관측 신호 (I~ = real, Q~ = imag)

    Returns
    -------
    eps_T : float   추정 게인 오차
    phi_T : float   추정 위상 오차 [deg]
    """
    I  = np.real(pilot0)
    Q  = np.imag(pilot0)
    If = np.real(pilot1)   # I~
    Qf = np.imag(pilot1)   # Q~

    # ── 게인 오차: (I·I~ − Q·Q~) / (2(I² + Q²)) ──
    denom = I**2 + Q**2
    valid = denom > power_threshold                  # 0 나눗셈 방지
    com_p = (I[valid]*If[valid] - Q[valid]*Qf[valid]) / (2.0 * denom[valid])

    eps_T = np.mean(com_p)

    # ── 위상 오차: arcsin( [I·(1+ε)² − I~] / [Q·(1−ε²)] ) ──
    num = I[valid] * (1.0 + eps_T)**2 - If[valid]    # A(1+x)² − C
    den = Q[valid] * (1.0 - eps_T**2)                # B(1−x²)  ★ 변경

    safe = np.abs(den) > power_threshold             # Q ≈ 0 샘플 배제
    sin_phi = np.clip(num[safe] / den[safe], -1.0, 1.0)
    com_C = np.arcsin(sin_phi)

    phi_T = np.rad2deg(np.mean(com_C))

    return 2 * eps_T, phi_T

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

# --- [추가] 논문 1의 제안 기법: 주파수 영역 사전 왜곡 (Eq. 5, 6 기반) ---
def mirror_predistorter(sp, N, epsilon_c, phi_c_deg):
    """
    sp: 주파수 영역 심볼 (shape: num_symbols x N)
    epsilon_c: 추정된 진폭 오차
    phi_c_deg: 추정된 위상 오차 (도 단위)
    """
    # 사용자의 위상 모델(pi*C/360)에 맞춰 변환
    theta = phi_c_deg * np.pi / 360

    # 논문의 계수 계산
    term1 = 0.5 * (np.exp(1j * theta) / (1 + epsilon_c) + np.exp(-1j * theta) / (1 - epsilon_c))
    term2 = 0.5 * (np.exp(-1j * theta) / (1 + epsilon_c) - np.exp(1j * theta) / (1 - epsilon_c))

    sp_comp = np.zeros_like(sp, dtype=complex)

    for k in range(N):
        # 미러 부반송파 인덱스 계산 (-k)
        k_mirror = (N - k) % N
        # A[k] = term1 * S[k] + term2 * conj(S[-k])
        sp_comp[:, k] = term1 * sp[:, k] + term2 * np.conj(sp[:, k_mirror])

    return sp_comp
# -------------------------------------------------------------------

def hard_decision(in_vector, M):
    if M == 2:
        out_vector = np.sign(np.real(in_vector))
    elif M == 4:
        r_part = np.real(in_vector)
        i_part = np.imag(in_vector)
        out_vector = np.sign(r_part) + 1j * np.sign(i_part)
    return out_vector

def ber_call_qpsk(a, b, M):
    NumOfBitError = 0
    NumOfSymbolError = 0
    SymbolError = 0

    A = a.flatten()
    B = b.flatten()
    D = np.size(A) if isinstance(A, np.ndarray) else len(A)

    if M == 2:
        ber = np.sum(~(B == A)) / D
    elif M == 4:
        for q in range(D):
            integral_I, integral_Q = np.real(A[q]), np.imag(A[q])
            B_I, B_Q = np.real(B[q]), np.imag(B[q])

            decision_I = 1 if integral_I > 0 else -1
            decision_Q = 1 if integral_Q > 0 else -1

            if decision_I != B_I:
                NumOfBitError += 1
                SymbolError = 1
            if decision_Q != B_Q:
                NumOfBitError += 1
                SymbolError = 1
            if SymbolError == 1:
                NumOfSymbolError += 1
            SymbolError = 0

        ber = NumOfBitError / (2 * D)
    return ber

def calculate_evm(rx_symbols, ideal_symbols):
    """
    수신된 복소 심볼과 이상적인 복소 심볼을 비교하여 EVM(%)을 계산합니다.
    """
    # 1. 오차 벡터 계산 (수신값 - 이상적인 값)
    error_vector = rx_symbols - ideal_symbols

    # 2. 오차 벡터의 평균 전력 계산 (절댓값의 제곱)
    error_power = np.mean(np.abs(error_vector)**2)

    # 3. 레퍼런스(이상적 심볼)의 평균 전력 계산
    reference_power = np.mean(np.abs(ideal_symbols)**2)

    # 4. RMS EVM 계산 (%)
    evm_rms = np.sqrt(error_power / reference_power)
    evm_percent = evm_rms * 100
    evm_db = 20 * np.log10(evm_rms)
    return evm_rms, evm_percent, evm_db

N = 32
Block = 10*10000
M = 4
D = N * Block
X = 4
Fq = 1
Fs = 1
Fd = Fq
change_block = int(D / X)

np.random.seed(seed=int(time.time()))

up_C = 5
up_upsilon = 0.3
down_C = 0
down_upsilon = 0

blstm = 0

thermal_noise = 15
ch_mode = 2 # 0: AWGN, 1: Rayleigh, 2: Rician
K = 5	    # Rician K factor (0 ~ K)

SNR = np.arange(0, 20, 2)
org_ber = np.zeros((K, len(SNR)))
ber = np.zeros((K, len(SNR)))
pred_ber = np.zeros((K, len(SNR)))
if blstm == 1:
  blstm_ber = np.zeros((K, len(SNR)))
mirror_ber = np.zeros((K, len(SNR)))  # [추가] 논문 1 제안 기법 BER 저장 배열

theo_err = np.zeros(len(SNR))

org_IQ_out = np.zeros((1, D))
IQ_out_r = np.zeros((1, D))
pred_iq_out = np.zeros((1, D))
if blstm == 1:
  blstm_iq_out = np.zeros((1, D))
mirror_iq_out = np.zeros((1, D))      # [추가] 논문 1 제안 기법 출력 저장 배열

epsilon = up_upsilon
phi = up_C

for i in range(X):
    b = gen_mapping(M, change_block)

    b_r = np.array([1 + 1j, -1 - 1j])
    pilot = np.concatenate((b_r, -b_r))
    sp = pilot
    ifft_out = np.fft.ifft(sp)
    ps = ifft_out.reshape(1, -1)
    I = np.real(ps)
    Q = np.imag(ps)

    org_up_cos = cos_sampling(X, Fq, 0, 0)
    org_up_sin = sin_sampling(X, Fq, 0, 0)
    epsilon += 0.1
    phi += 1
    up_cos_r = cos_sampling(X, Fq, phi, epsilon)
    up_sin_r = sin_sampling(X, Fq, phi, epsilon)

    pilot_I = np.dot(I.reshape(-1, 1), up_cos_r)
    pilot_Q = np.dot(Q.reshape(-1, 1), up_sin_r)
    pilot_IQ_out = pilot_I.reshape(1, -1) + pilot_Q.reshape(1, -1)

    sigpwr_pilot = np.linalg.norm(pilot_IQ_out) ** 2 / pilot_IQ_out.shape[1]
    snr_wp = 10 ** (thermal_noise / 10)
    sgma_pilot = np.sqrt(X * sigpwr_pilot / snr_wp / 2 / np.log2(M))
    n_pilot = sgma_pilot * np.random.randn(1, np.size(pilot_IQ_out))
    pilot_IQ_out = pilot_IQ_out + n_pilot

    rx_I_out = np.dot(pilot_IQ_out.reshape(-1, X), up_cos_r.T)
    rx_Q_out = np.dot(pilot_IQ_out.reshape(-1, X), up_sin_r.T)
    rx_Or_I_out = rx_I_out * 2 / X
    rx_Or_Q_out = rx_Q_out * 2 / X
    rx_out = rx_Or_I_out + rx_Or_Q_out * 1j
    rx_fft = np.fft.fft(rx_out.reshape(1, -1))

    # ✅ 수정: FFT 제거, 시간 도메인 신호 사용
    est_epsilon, est_phi = IQ_est(ps.flatten(), rx_out.flatten())
    #                             ↑ IFFT 출력      ↑ 다운컨버전 출력 (FFT 전)
    print(f"Estimated epsilon: {est_epsilon}, Estimated phi: {est_phi}\n")

    # 기존 송신기 데이터
    sp = b.reshape(-1, N)
    ifft_out = np.fft.ifft(sp)
    ps = ifft_out.reshape(-1, 1)
    I_r = np.real(ps)
    Q_r = np.imag(ps)

    org_I = np.dot(I_r, org_up_cos)           # No IQ imbalance
    org_I_out = org_I.reshape(1, -1)
    org_Q = np.dot(Q_r, org_up_sin)           # No IQ imbalance
    org_Q_out = org_Q.reshape(1, -1)
    org_IQ_out[i * change_block: (i + 1) * change_block, :] = org_I_out + org_Q_out        # No IQ imbalance

    mirror_I_r = np.dot(I_r, up_cos_r)
    mirror_Iout_r = mirror_I_r.reshape(1, -1)
    mirror_Q_r = np.dot(Q_r, up_sin_r)
    mirror_Qout_r = mirror_Q_r.reshape(1, -1)
    IQ_out_r[i * change_block: (i + 1) * change_block, :] = mirror_Iout_r + mirror_Qout_r

    # 기존 시간 영역 사전왜곡
    I_r_Comp = cos_predistor(org_I, X, Fq, est_epsilon, est_phi) # Use est_epsilon directly
    Q_r_Comp = sin_predistor(org_Q, X, Fq, est_epsilon, est_phi) # Use est_phi directly
    pred_i = I_r_Comp * up_cos_r
    pred_i_out = pred_i.reshape(1, -1)
    pred_q = Q_r_Comp * up_sin_r
    pred_q_out = pred_q.reshape(1, -1)
    pred_iq_out[i * change_block: (i + 1) * change_block, :] = pred_i_out + pred_q_out

    # --- [추가] 기존 논문 제안 기법: 주파수 영역 사전왜곡 처리 ---
    sp_mirror = mirror_predistorter(sp, N, est_epsilon, est_phi) # Use est_epsilon, est_phi directly
    ifft_out_mirror = np.fft.ifft(sp_mirror)
    ps_mirror = ifft_out_mirror.reshape(-1, 1)

    I_p1 = np.real(ps_mirror)
    Q_p1 = np.imag(ps_mirror)
    mirror_I_p1 = np.dot(I_p1, up_cos_r)
    mirror_Q_p1 = np.dot(Q_p1, up_sin_r)
    mirror_iq_out[i * change_block: (i + 1) * change_block, :] = mirror_I_p1.reshape(1, -1) + mirror_Q_p1.reshape(1, -1)
    # --------------------------------------------------------

    if blstm == 1:
      # BLSTM Predistorter (기존 로직 유지)
      blstm_I_norm, blstm_Q_norm, rms_mag = normalize_with_rms(org_I.reshape(1,-1), org_Q.reshape(1,-1))
      model_corr_r = np.expand_dims(blstm_I_norm.reshape(-1, N), axis=-1)
      model_corr_i = np.expand_dims(blstm_Q_norm.reshape(-1, N), axis=-1)
      keras.backend.clear_session()
      try:
          name_r = '/content/drive/MyDrive/IQ_imbalance_BLSTM/predistortion/variable/vlc_lstm_model9_10_r.h5'
          name_i = '/content/drive/MyDrive/IQ_imbalance_BLSTM/predistortion/variable/vlc_lstm_model9_10_i.h5'
          recovery_dl_r = keras.models.load_model(name_r, custom_objects={'mse': 'mse'})
          recovery_dl_i = keras.models.load_model(name_i, custom_objects={'mse': 'mse'})
          sig_in_Linear_dl_r = recovery_dl_r.predict(model_corr_r)
          sig_in_Linear_dl_i = recovery_dl_i.predict(model_corr_i)

          blstm_I_r_norm = sig_in_Linear_dl_r.reshape(1, -1)
          blstm_Q_r_norm = sig_in_Linear_dl_i.reshape(1, -1)
          blstm_I_r, blstm_Q_r = denormalize_with_rms(blstm_I_r_norm.flatten(), blstm_Q_r_norm.flatten(), rms_mag)

          blstm_i = blstm_I_r.reshape(-1, X) * up_cos_r
          blstm_q = blstm_Q_r.reshape(-1, X) * up_sin_r
          blstm_iq_out[i * change_block: (i + 1) * change_block, :] = blstm_i.reshape(1, -1) + blstm_q.reshape(1, -1)
      except OSError:
          print("BLSTM 모델 파일을 찾을 수 없습니다. (BLSTM 로직 스킵)")
          blstm_iq_out[0, i * change_block: (i + 1) * change_block] = IQ_out_r[0, i * change_block: (i + 1) * change_block]

sigpwr_org = np.linalg.norm(org_IQ_out) ** 2 / org_IQ_out.shape[1]
sigpwr = np.linalg.norm(IQ_out_r) ** 2 / IQ_out_r.shape[1]
sigpwr_pred = np.linalg.norm(pred_iq_out) ** 2 / pred_iq_out.shape[1]
if blstm == 1:
  sigpwr_blstm = np.linalg.norm(blstm_iq_out) ** 2 / blstm_iq_out.shape[1]
sigpwr_mirror = np.linalg.norm(mirror_iq_out) ** 2 / mirror_iq_out.shape[1]

if ch_mode == 0 or ch_mode == 1:    # AWGN or Rayleigh Fading
    K = 1
for i_k in range(0, K):
    if ch_mode == 0 or ch_mode == 1:
        print("Channel mode: ", ch_mode)
    else:
        print("Rician K factor: ", i_k)
    for m in range(len(SNR)):
        snr_wp = 10 ** (SNR[m] / 10)

        sgma_org = np.sqrt(X * sigpwr_org / snr_wp / 2 / np.log2(M))
        sgma = np.sqrt(X * sigpwr / snr_wp / 2 / np.log2(M))
        sgma_pred = np.sqrt(X * sigpwr_pred / snr_wp / 2 / np.log2(M))
        if blstm == 1:
          sgma_blstm = np.sqrt(X * sigpwr_blstm / snr_wp / 2 / np.log2(M))
        sgma_mirror = np.sqrt(X * sigpwr_mirror / snr_wp / 2 / np.log2(M))

        n_org = sgma_org * np.random.randn(1, np.size(org_IQ_out))
        n = sgma * np.random.randn(1, np.size(IQ_out_r))
        n_pred = sgma_pred * np.random.randn(1, np.size(pred_iq_out))
        if blstm == 1:
          n_blstm = sgma_blstm * np.random.randn(1, np.size(blstm_iq_out))
        n_mirror = sgma_mirror * np.random.randn(1, np.size(mirror_iq_out))

        if ch_mode == 0:    # AWGN
            org_receive = org_IQ_out + n
            receive_data = IQ_out_r + n
            pred_rev = pred_iq_out + n
            if blstm == 1:
              blstm_rev = blstm_iq_out + n
            mirror_rev = mirror_iq_out + n_mirror
        elif ch_mode == 1:    # Rayleigh Fading
            h_Ideal = (np.random.randn(org_IQ_out.size) + 1j * np.random.randn(org_IQ_out.size)) / np.sqrt(2)
            h_normal = (np.random.randn(IQ_out_r.size) + 1j * np.random.randn(IQ_out_r.size)) / np.sqrt(2)
            h_comp = (np.random.randn(pred_iq_out.size) + 1j * np.random.randn(pred_iq_out.size)) / np.sqrt(2)
            if blstm == 1:
              h_blstm = (np.random.randn(blstm_iq_out.size) + 1j * np.random.randn(blstm_iq_out.size)) / np.sqrt(2)
            h_mirror = (np.random.randn(mirror_iq_out.size) + 1j * np.random.randn(mirror_iq_out.size)) / np.sqrt(2)

            org_receive = (h_Ideal * org_IQ_out + n_org) / h_Ideal
            receive_data = (h_normal * IQ_out_r + n) / h_normal
            pred_rev = (h_comp * pred_iq_out + n_pred) / h_comp
            if blstm == 1:
              blstm_rev = (h_blstm * blstm_iq_out + n_blstm) / h_blstm
            mirror_rev = (h_mirror * mirror_iq_out + n_mirror) / h_mirror
        else: # Rician Fading
            k_0 = 4 * (i_k + 1)
            frame = org_IQ_out.size
            h = np.sqrt(k_0 / (1 + k_0)) * np.ones((1, frame)) + np.sqrt(1 / (1 + k_0)) * ((np.random.randn(1, frame) + 1j * np.random.randn(1, frame)) / np.sqrt(2))

            org_receive = h * org_IQ_out + n_org
            receive_data = h * IQ_out_r + n
            pred_rev = h * pred_iq_out + n_pred
            if blstm == 1:
              blstm_rev = h * blstm_iq_out + n_blstm
            mirror_rev = h * mirror_iq_out + n_mirror

        org_rx_IQ_out = org_receive.reshape(-1, X)
        rx_IQ_out_r = receive_data.reshape(-1, X)
        pred_rx_iq = pred_rev.reshape(-1, X)
        if blstm == 1:
          blstm_rx_iq = blstm_rev.reshape(-1, X)
        mirror_rx_iq = mirror_rev.reshape(-1, X)

        down_cos_r = cos_sampling(X, Fq, down_C, down_upsilon)
        down_sin_r = sin_sampling(X, Fq, down_C, down_upsilon)
        org_down_cos = cos_sampling(X, Fq, 0, 0)
        org_down_sin = sin_sampling(X, Fq, 0, 0)

        org_rx_I_out = np.dot(org_rx_IQ_out, org_down_cos.T)
        org_rx_Q_out = np.dot(org_rx_IQ_out, org_down_sin.T)
        rx_I_out_r = np.dot(rx_IQ_out_r, down_cos_r.T)
        rx_Q_out_r = np.dot(rx_IQ_out_r, down_sin_r.T)
        pred_rx_i = np.dot(pred_rx_iq, down_cos_r.T)
        pred_rx_q = np.dot(pred_rx_iq, down_sin_r.T)
        if blstm == 1:
          blstm_rx_i = np.dot(blstm_rx_iq, down_cos_r.T)
          blstm_rx_q = np.dot(blstm_rx_iq, down_sin_r.T)
        mirror_rx_i = np.dot(mirror_rx_iq, down_cos_r.T)
        mirror_rx_q = np.dot(mirror_rx_iq, down_sin_r.T)

        org_rx_out = (org_rx_I_out * 2 / X) + (org_rx_Q_out * 2 / X) * 1j
        rx_out_r = (rx_I_out_r * 2 / X) + (rx_Q_out_r * 2 / X) * 1j
        pred_rx_out = (pred_rx_i * 2 / X) + (pred_rx_q * 2 / X) * 1j
        if blstm == 1:
          blstm_rx_out = (blstm_rx_i * 2 / X) + (blstm_rx_q * 2 / X) * 1j
        mirror_rx_out = (mirror_rx_i * 2 / X) + (mirror_rx_q * 2 / X) * 1j

        org_fft_out = np.fft.fft(org_rx_out.reshape(-1, N), N)
        fft_out_r = np.fft.fft(rx_out_r.reshape(-1, N), N)
        pred_fft_out = np.fft.fft(pred_rx_out.reshape(-1, N), N)
        if blstm == 1:
          blstm_fft_out = np.fft.fft(blstm_rx_out.reshape(-1, N), N)
        mirror_fft_out = np.fft.fft(mirror_rx_out.reshape(-1, N), N)

        # Mean Square Error (MSE)
        mse_ideal = np.mean(np.abs(b - org_fft_out.reshape(1, -1)) ** 2)
        mse_iq = np.mean(np.abs(b - fft_out_r.reshape(1, -1)) ** 2)
        mse_pred = np.mean(np.abs(b - pred_fft_out.reshape(1, -1)) ** 2)
        if blstm == 1:
          mse_blstm = np.mean(np.abs(b - blstm_fft_out.reshape(1, -1)) ** 2)
        mse_mirror = np.mean(np.abs(b - mirror_fft_out.reshape(1, -1)) ** 2)
        print(f"MSE of no IQ imbalance at SNR {SNR[m]}dB: {mse_ideal}")
        print(f"MSE of IQ imbalance at SNR {SNR[m]}dB: {mse_iq}")
        print(f"MSE of Predistortion at SNR {SNR[m]}dB: {mse_pred}")
        if blstm == 1:
          print(f"MSE of BLSTM at SNR {SNR[m]}dB: {mse_blstm}")
        print(f"MSE of Mirror at SNR {SNR[m]}dB: {mse_mirror}")

        # EVM (Error Vector Magnitude)
        evm_ideal = calculate_evm(org_fft_out.reshape(1, -1), b)
        evm_iq = calculate_evm(fft_out_r.reshape(1, -1), b)
        evm_pred = calculate_evm(pred_fft_out.reshape(1, -1), b)
        if blstm == 1:
          evm_blstm = calculate_evm(blstm_fft_out.reshape(1, -1), b)
        evm_mirror = calculate_evm(mirror_fft_out.reshape(1, -1), b)
        print(f"EVM of no IQ imbalance at SNR {SNR[m]}dB: {evm_ideal}")
        print(f"EVM of IQ imbalance at SNR {SNR[m]}dB: {evm_iq}")
        print(f"EVM of Predistortion at SNR {SNR[m]}dB: {evm_pred}")
        if blstm == 1:
          print(f"EVM of BLSTM at SNR {SNR[m]}dB: {evm_blstm}")
        print(f"EVM of Mirror at SNR {SNR[m]}dB: {evm_mirror}")
        # breakpoint()

        org_hard = hard_decision(org_fft_out.reshape(1, -1), M)
        out_hard = hard_decision(fft_out_r.reshape(1, -1), M)
        pred_hard = hard_decision(pred_fft_out.reshape(1, -1), M)
        if blstm == 1:
          blstm_hard = hard_decision(blstm_fft_out.reshape(1, -1), M)
        mirror_hard = hard_decision(mirror_fft_out.reshape(1, -1), M)

        org_ber[i_k, m] = ber_call_qpsk(org_hard, b, M)
        ber[i_k, m] = ber_call_qpsk(out_hard, b, M)
        pred_ber[i_k, m] = ber_call_qpsk(pred_hard, b, M)
        if blstm == 1:
          blstm_ber[i_k, m] = ber_call_qpsk(blstm_hard, b, M)
        mirror_ber[i_k, m] = ber_call_qpsk(mirror_hard, b, M)

        if blstm == 1:
          print(f"org:{org_ber[i_k, m]:.5f}, imb:{ber[i_k, m]:.5f}, pred:{pred_ber[i_k, m]:.5f}, blstm:{blstm_ber[i_k, m]:.5f}, mirror:{mirror_ber[i_k, m]:.5f}\n")
        else:
          print(f"org:{org_ber[i_k, m]:.5f}, imb:{ber[i_k, m]:.5f}, pred:{pred_ber[i_k, m]:.5f}, mirror:{mirror_ber[i_k, m]:.5f}\n")
for i in range(len(SNR)):
    t_snr = 10 ** (SNR[i] / 10)
    theo_err[i] = (1 / 2) * erfc(np.sqrt(t_snr)) - (1 / 8) * (erfc(np.sqrt(t_snr))) ** 2

plt.figure(4)
plt.semilogy(SNR, org_ber[4], 'g', label='No IQ imbalance BER')
if ch_mode == 0:
	plt.semilogy(SNR, theo_err, 'k--', label='Theoretical BER')
plt.semilogy(SNR, ber[4], 'b', label='Simulated BER (Imbalance)')
plt.semilogy(SNR, pred_ber[4], 'r', label='Time-Domain Predistortion')
if blstm == 1:
  plt.semilogy(SNR, blstm_ber[4], 'm', label='BLSTM BER')
plt.semilogy(SNR, mirror_ber[4], 'c-x', label='Conv Freq-Domain Pre-dist') # [추가] 논문 1 기법 플롯
plt.xlabel('SNR (dB)')
plt.ylabel('BER')
plt.axis([0, 20, 1e-5, 1])
plt.grid(True, which='both')
plt.legend()
# plt.savefig('BER_plot_with_mirror.png')
