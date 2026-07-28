# @title
import numpy as np
import time
import os
import matplotlib.pyplot as plt
from scipy.special import erfc
from tensorflow import keras

BASE_DIR = '/content/drive/MyDrive'
MODEL_EPOCH = 100                     # 사용할 체크포인트 (누적 epoch)

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

def IQ_est(pilot0, pilot1, power_threshold=1e-6):
    I  = np.real(pilot0)
    Q  = np.imag(pilot0)
    If = np.real(pilot1)
    Qf = np.imag(pilot1)
    denom = I**2 + Q**2
    valid = denom > power_threshold
    com_p = (I[valid]*If[valid] - Q[valid]*Qf[valid]) / (2.0 * denom[valid])
    eps_T = np.mean(com_p)
    num = I[valid] * (1.0 + eps_T)**2 - If[valid]
    den = Q[valid] * (1.0 - eps_T**2)
    safe = np.abs(den) > power_threshold
    sin_phi = np.clip(num[safe] / den[safe], -1.0, 1.0)
    com_C = np.arcsin(sin_phi)
    phi_T = np.rad2deg(np.mean(com_C))
    return 2 * eps_T, phi_T          # (1±eps/2) 컨벤션의 eps로 반환

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

def mirror_predistorter(sp, N, epsilon_c, phi_c_deg):
    theta = -phi_c_deg * np.pi / 360   # 이 시스템(Q를 +sin에 싣는 컨벤션)에 맞춘 부호
    term1 = 0.5 * (np.exp(1j * theta) / (1 + epsilon_c) + np.exp(-1j * theta) / (1 - epsilon_c))
    term2 = 0.5 * (np.exp(-1j * theta) / (1 + epsilon_c) - np.exp(1j * theta) / (1 - epsilon_c))
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
# 파라미터
# ============================================================
N = 32
Block = 10*10000
M = 4
D = N * Block
X = 4
Fq = 1
change_block = int(D / X)

np.random.seed(seed=int(time.time()))

# 테스트 지점: 루프 첫 블록에서 +0.1/+1 되므로 실제 평가는 (up_upsilon+0.1, up_C+1)
# 예) 0.1/1 → eps=0.2, phi=2.  다른 지점을 보려면 이 두 값을 바꿔서 재실행.
up_C = 1
up_upsilon = 0.1
down_C = 0
down_upsilon = 0

blstm = 1

ch_mode = 2   # 0: AWGN, 1: Rayleigh, 2: Rician
K_list = [4, 8, 12, 16, 20]   # 테스트할 실제 Rician K factor

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
        os.path.join(BASE_DIR, f'vlc_lstm_cond_{MODEL_EPOCH}_r.h5'), compile=False)
    recovery_dl_i = keras.models.load_model(
        os.path.join(BASE_DIR, f'vlc_lstm_cond_{MODEL_EPOCH}_i.h5'), compile=False)
    print(f"[BLSTM] loaded cond model (epoch {MODEL_EPOCH}), "
          f"in_std={in_std:.6f}, tg_std={tg_std:.6f}")

epsilon = up_upsilon
phi = up_C

for i in range(X):
    b = gen_mapping(M, change_block)

    # ── Pilot 루프백으로 (eps, phi) 추정 ──
    b_r = np.array([1 + 1j, -1 - 1j])
    pilot = np.concatenate((b_r, -b_r))
    sp = pilot
    ps = np.fft.ifft(sp).reshape(1, -1)
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

    rx_I_out = np.dot(pilot_IQ_out.reshape(-1, X), up_cos_r.T)
    rx_Q_out = np.dot(pilot_IQ_out.reshape(-1, X), up_sin_r.T)
    rx_out = (rx_I_out * 2 / X) + (rx_Q_out * 2 / X) * 1j

    est_epsilon, est_phi = IQ_est(ps.flatten(), rx_out.flatten())
    print(f"Epsilon: {epsilon}, Estimated epsilon: {est_epsilon}, "
          f"Phi: {phi}, Estimated phi: {est_phi}\n")

    # ── 송신 데이터 ──
    sp = b.reshape(-1, N)
    ps = np.fft.ifft(sp).reshape(-1, 1)
    I_r = np.real(ps)
    Q_r = np.imag(ps)

    org_I = np.dot(I_r, org_up_cos)
    org_Q = np.dot(Q_r, org_up_sin)
    org_IQ_out[i * change_block: (i + 1) * change_block, :] = \
        org_I.reshape(1, -1) + org_Q.reshape(1, -1)

    conv_I = np.dot(I_r, up_cos_r)
    conv_Q = np.dot(Q_r, up_sin_r)
    IQ_out_r[i * change_block: (i + 1) * change_block, :] = \
        conv_I.reshape(1, -1) + conv_Q.reshape(1, -1)

    # ── 시간영역 predistortion ──
    I_r_Comp = cos_predistor(org_I, X, Fq, est_epsilon, est_phi)
    Q_r_Comp = sin_predistor(org_Q, X, Fq, est_epsilon, est_phi)
    pred_iq_out[i * change_block: (i + 1) * change_block, :] = \
        (I_r_Comp * up_cos_r).reshape(1, -1) + (Q_r_Comp * up_sin_r).reshape(1, -1)

    # ── 주파수영역(mirror) predistortion: 추정치 사용으로 복원 ──
    sp_mirror = mirror_predistorter(sp, N, 0.35 / 2, 3.5)
    ps_mirror = np.fft.ifft(sp_mirror).reshape(-1, 1)
    mirror_iq_out[i * change_block: (i + 1) * change_block, :] = \
        (np.dot(np.real(ps_mirror), up_cos_r)).reshape(1, -1) + \
        (np.dot(np.imag(ps_mirror), up_sin_r)).reshape(1, -1)

    # ── [조건화] BLSTM predistorter ──
    if blstm == 1:
        # 신호 채널: 훈련과 동일한 규약 (train input_std로 정규화)
        sig_r = (org_I.reshape(-1) / in_std).reshape(-1, N)
        sig_i = (org_Q.reshape(-1) / in_std).reshape(-1, N)

        # 조건화 채널: pilot 추정치를 훈련과 동일하게 정규화
        eps_n = (est_epsilon - eps_mid) / eps_half
        phi_n = (est_phi     - phi_mid) / phi_half
        if abs(eps_n) > 1.0 or abs(phi_n) > 1.0:
            print(f"경고: 추정 (eps={est_epsilon:.3f}, phi={est_phi:.2f})가 "
                  f"학습 범위 밖입니다. 외삽 성능 저하 가능.")
        ch_e = np.full_like(sig_r, eps_n)
        ch_p = np.full_like(sig_r, phi_n)

        x_r = np.stack([sig_r, ch_e, ch_p], axis=-1).astype(np.float32)
        x_i = np.stack([sig_i, ch_e, ch_p], axis=-1).astype(np.float32)

        out_r = recovery_dl_r.predict(x_r, batch_size=512, verbose=0)
        out_i = recovery_dl_i.predict(x_i, batch_size=512, verbose=0)

        # 복원: 네트워크 출력(target_std로 정규화된 스케일) × target_std
        blstm_I_r = out_r.reshape(-1) * tg_std
        blstm_Q_r = out_i.reshape(-1) * tg_std

        blstm_i_wave = blstm_I_r.reshape(-1, X) * up_cos_r
        blstm_q_wave = blstm_Q_r.reshape(-1, X) * up_sin_r
        blstm_iq_out[i * change_block: (i + 1) * change_block, :] = \
            blstm_i_wave.reshape(1, -1) + blstm_q_wave.reshape(1, -1)

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
        snr_wp = 10 ** (SNR[m] / 10)

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
            org_receive = org_IQ_out + n
            receive_data = IQ_out_r + n
            pred_rev = pred_iq_out + n
            mirror_rev = mirror_iq_out + n_mirror
            if blstm == 1:
                blstm_rev = blstm_iq_out + n
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
            mse_blstm = np.mean(np.abs(b - blstm_fft_out.reshape(1, -1)) ** 2)
            evm_blstm = calculate_evm(blstm_fft_out.reshape(1, -1), b)
            mse_blstm_txt = f", blstm MSE {mse_blstm:.5f}, EVM {evm_blstm[1]:.2f}%"
        mse_org = np.mean(np.abs(b - org_fft_out.reshape(1, -1)) ** 2)
        evm_org = calculate_evm(org_fft_out.reshape(1, -1), b)
        mse_iq = np.mean(np.abs(b - fft_out_r.reshape(1, -1)) ** 2)
        evm_iq = calculate_evm(fft_out_r.reshape(1, -1), b)
        mse_mirror = np.mean(np.abs(b - mirror_fft_out.reshape(1, -1)) ** 2)
        evm_mirror = calculate_evm(mirror_fft_out.reshape(1, -1), b)
        mse_pred = np.mean(np.abs(b - pred_fft_out.reshape(1, -1)) ** 2)
        evm_pred = calculate_evm(pred_fft_out.reshape(1, -1), b)
        print(f"SNR {SNR[m]}dB: ideal MSE {mse_org:.5f}, EVM {evm_org[1]:.2f}%, IQ imbalance MSE {mse_iq:.5f}, EVM {evm_iq[1]:.2f}%, mirror MSE {mse_mirror:.5f}, EVM {evm_mirror[1]:.2f}%")
        print(f"SNR {SNR[m]}dB: pred MSE {mse_pred:.5f}, EVM {evm_pred[1]:.2f}%{mse_blstm_txt}")

        org_ber[i_k, m] = ber_call_qpsk(hard_decision(org_fft_out.reshape(1, -1), M), b, M)
        ber[i_k, m] = ber_call_qpsk(hard_decision(fft_out_r.reshape(1, -1), M), b, M)
        pred_ber[i_k, m] = ber_call_qpsk(hard_decision(pred_fft_out.reshape(1, -1), M), b, M)
        mirror_ber[i_k, m] = ber_call_qpsk(hard_decision(mirror_fft_out.reshape(1, -1), M), b, M)
        if blstm == 1:
            blstm_ber[i_k, m] = ber_call_qpsk(hard_decision(blstm_fft_out.reshape(1, -1), M), b, M)
            print(f"org:{org_ber[i_k, m]:.5f}, imb:{ber[i_k, m]:.5f}, pred:{pred_ber[i_k, m]:.5f}, "
                  f"blstm:{blstm_ber[i_k, m]:.5f}, mirror:{mirror_ber[i_k, m]:.5f}\n")
        else:
            print(f"org:{org_ber[i_k, m]:.5f}, imb:{ber[i_k, m]:.5f}, "
                  f"pred:{pred_ber[i_k, m]:.5f}, mirror:{mirror_ber[i_k, m]:.5f}\n")
                  
    if ch_mode == 0 or ch_mode == 1:
        break

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
    plt.semilogy(SNR, blstm_ber[4], 'm', label='BLSTM (conditioned)')
plt.semilogy(SNR, mirror_ber[4], 'c-x', label='Conv Freq-Domain Pre-dist')
plt.xlabel('SNR (dB)')
plt.ylabel('BER')
plt.axis([0, 20, 1e-5, 1])
plt.grid(True, which='both')
plt.legend()
plt.savefig('BER_plot_cond.png')