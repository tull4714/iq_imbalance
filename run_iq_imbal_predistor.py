import numpy as np
import time
import matplotlib.pyplot as plt
import csv
from scipy.special import erfc

from tensorflow import keras

# 훈련 시와 동일한 정규화 함수 추가
def normalize_with_rms(I_data, Q_data):
    """RMS 기반 정규화로 더 안정적인 정규화"""
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
    """RMS 기반 역정규화"""
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

def IQ_est(pilot0, pilot1):
    # Ratio calculation
    com_p = (np.abs(np.real(pilot1)) - np.abs(np.imag(pilot1))) / 2
    com_C = np.arcsin((2 - (np.real(pilot0) * np.imag(pilot1) + np.real(pilot1) * np.imag(pilot0))) / 2)

    return com_p, com_C * 180 / np.pi

def cos_predistor(data, X, Fq, epsilon, phi):
    # Amplitude compensation
    a_Ir = 2 / (2 + epsilon) * data

    # Phase compensation
    m = np.arange(0, X)
    S_IC = 1 / (np.cos((2 * Fq * np.pi * m) / X + np.pi * phi / 360))

    I_r = a_Ir * S_IC

    return I_r

def sin_predistor(data, X, Fq, epsilon, phi):
    # Amplitude compensation
    a_Qr = 2 / (2 - epsilon) * data

    # Phase compensation
    m = np.arange(0, X)
    S_QC = 1 / (np.sin((2 * Fq * np.pi * m) / X - np.pi * phi / 360))

    Q_r = a_Qr * S_QC

    return Q_r

def hard_decision(in_vector, M):
    if M == 2:
        out_vector = np.sign(np.real(in_vector))
    elif M == 4:
        r_part = np.real(in_vector)
        i_part = np.imag(in_vector)
        out_vector = np.sign(r_part) + 1j * np.sign(i_part)

    return out_vector

def ber_call_qpsk(a, b, M):
    numoferr = 0
    NumOfBitError = 0
    NumOfSymbolError = 0
    SymbolError = 0
    integral = 0

    A = a.flatten()
    B = b.flatten()
    D = np.size(A) if isinstance(A, np.ndarray) else len(A)

    if M == 2:
        ber = np.sum(~(B == A)) / D
    elif M == 4:
        for q in range(D):
            integral_I = np.real(A[q])
            integral_Q = np.imag(A[q])
            B_I = np.real(B[q])
            B_Q = np.imag(B[q])

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

        pb = NumOfBitError / (2 * D)
        ps = NumOfSymbolError / D
        ber = pb

    return ber

N = 32
Block = 10*10000
M = 4
D = N * Block
X = 4
Fq = 1
Fs = 1  # Sampling Frequency
Fd = Fq
change_block = int(D / X)

np.random.seed(seed=int(time.time()))

up_C = 5    # np.random.random_sample()
up_upsilon = 0.3    #np.random.random_sample()
down_C = 0  # np.random.randint(0, 15)
down_upsilon = 0 #np.random.random_sample() * 0.3
# print(f"Up C: {up_C}, Down C: {down_C}\n")
# print(f"Up Upsilon: {up_upsilon}, Down Upsilon: {down_upsilon}\n")

ch_mode = 2	# 0: AWGN, 1: Rayleigh, 2: Rician
K = 5	# Rician K factor (0 ~ K)

SNR = np.arange(0, 20, 2)
org_ber = np.zeros((K, len(SNR)))
ber = np.zeros((K, len(SNR)))
pred_ber = np.zeros((K, len(SNR)))	# Predistortion
blstm_ber = np.zeros((K, len(SNR)))	# BLSTM
theo_err = np.zeros(len(SNR))

org_IQ_out = np.zeros((1, D))
IQ_out_r = np.zeros((1, D))
pred_iq_out = np.zeros((len(SNR), 1, D))
blstm_iq_out = np.zeros((1, D))

epsilon = up_upsilon
phi = up_C
for i in range(X):
	b = gen_mapping(M, change_block)

	# 송신기
	sp = b.reshape(-1, N)
	ifft_out = np.fft.ifft(sp)
	ps = ifft_out.reshape(-1, 1)
	I_r = np.real(ps)
	Q_r = np.imag(ps)

	print(ps.shape)

	org_up_cos = cos_sampling(X, Fq, 0, 0)      # No IQ imbalance
	org_up_sin = sin_sampling(X, Fq, 0, 0)      # No IQ imbalance
	epsilon += 0.1
	phi += 1
	up_cos_r = cos_sampling(X, Fq, phi, epsilon)
	up_sin_r = sin_sampling(X, Fq, phi, epsilon)

	org_I = np.dot(I_r, org_up_cos)           # No IQ imbalance
	org_I_out = org_I.reshape(1, -1)
	org_Q = np.dot(Q_r, org_up_sin)           # No IQ imbalance
	org_Q_out = org_Q.reshape(1, -1)
	org_IQ_out[i * change_block: (i + 1) * change_block, :] = org_I_out + org_Q_out        # No IQ imbalance

	conv_I_r = np.dot(I_r, up_cos_r)
	conv_Iout_r = conv_I_r.reshape(1, -1)
	conv_Q_r = np.dot(Q_r, up_sin_r)
	conv_Qout_r = conv_Q_r.reshape(1, -1)
	IQ_out_r[i * change_block: (i + 1) * change_block, :] = conv_Iout_r + conv_Qout_r		# IQ imbalance

	# Information data
	b_r = np.array([1 + 1j, -1 - 1j])
	pilot = np.concatenate((b_r, -b_r))
	sp = pilot
	ifft_out = np.fft.ifft(sp)
	ps = ifft_out.reshape(1, -1)
	I = np.real(ps)
	Q = np.imag(ps)

	pilot_I = np.dot(I.reshape(-1, 1), up_cos_r)
	pilot_I_out = pilot_I.reshape(1, -1)
	pilot_Q = np.dot(Q.reshape(-1, 1), up_sin_r)
	pilot_Q_out = pilot_Q.reshape(1, -1)
	pilot_IQ_out = pilot_I_out + pilot_Q_out

	sigpwr_pilot = np.linalg.norm(pilot_IQ_out) ** 2 / pilot_IQ_out.shape[1]
	for m in range(len(SNR)):
		snr_wp = 10 ** (SNR[m] / 10)
		sgma_pilot = np.sqrt(X * sigpwr_pilot / snr_wp / 2 / np.log2(M))
		n_pilot = sgma_pilot * np.random.randn(1, np.size(pilot_IQ_out))
		pilot_IQ_out = pilot_IQ_out + n_pilot

		rx_I_out = np.dot(pilot_IQ_out.reshape(-1, X), up_cos_r.T)
		rx_Q_out = np.dot(pilot_IQ_out.reshape(-1, X), up_sin_r.T)
		rx_Or_I_out = rx_I_out * 2 / X
		rx_Or_Q_out = rx_Q_out * 2 / X
		rx_out = rx_Or_I_out + rx_Or_Q_out * 1j
		rx_ps = rx_out.reshape(1, -1)
		rx_fft = np.fft.fft(rx_ps)
		rx_sp = rx_fft

		est_epsilon, est_phi = IQ_est(pilot, rx_sp)
		print(f"Estimated epsilon: {est_epsilon}, Estimated phi: {est_phi}\n")

		I_r_Comp = cos_predistor(org_I, X, Fq, est_epsilon[0, 0], est_phi[0, 0])
		Q_r_Comp = sin_predistor(org_Q, X, Fq, est_epsilon[0, 0], est_phi[0, 0])
		pred_i = I_r_Comp * up_cos_r
		pred_i_out = pred_i.reshape(1, -1)
		pred_q = Q_r_Comp * up_sin_r
		pred_q_out = pred_q.reshape(1, -1)
		pred_iq_out[m, i * change_block: (i + 1) * change_block, :] = pred_i_out + pred_q_out

	#Correct Tx
	print("start")
	print(org_I_out.shape)
	print(org_Q_out.shape)

	# 수정: 훈련 시와 동일한 정규화 적용
	blstm_I_norm, blstm_Q_norm, rms_mag = normalize_with_rms(org_I_out, org_Q_out)

	# model_corr_r = org_I_out.reshape(-1, N)
	# model_corr_i = org_Q_out.reshape(-1, N)
	# BLSTM Predistorter
	model_corr_r = blstm_I_norm.reshape(-1, N)
	model_corr_i = blstm_Q_norm.reshape(-1, N)

	model_corr_r=np.expand_dims(model_corr_r, axis=-1)
	model_corr_i=np.expand_dims(model_corr_i, axis=-1)
	keras.backend.clear_session()
	name_r = '/content/drive/MyDrive/MachineLearning/Predistortion/vlc_lstm_model9_10_r.h5'
	name_i = '/content/drive/MyDrive/MachineLearning/Predistortion/vlc_lstm_model9_10_i.h5'
	recovery_dl_r = keras.models.load_model(name_r, custom_objects={'mse': 'mse'})
	recovery_dl_i = keras.models.load_model(name_i, custom_objects={'mse': 'mse'})

	sig_in_Linear_dl_r = recovery_dl_r.predict(model_corr_r)
	sig_in_Linear_dl_i = recovery_dl_i.predict(model_corr_i)
	blstm_I_r_norm = sig_in_Linear_dl_r.reshape(1, -1)
	blstm_Q_r_norm = sig_in_Linear_dl_i.reshape(1, -1)

	# 역정규화
	blstm_I_r, blstm_Q_r = denormalize_with_rms(blstm_I_r_norm.flatten(),
												blstm_Q_r_norm.flatten(),
												rms_mag)

	mse = np.mean((pred_i_out - blstm_I_r) ** 2)
	print(f"MSE: {mse}\n")
	print("start 2")
	blstm_i = blstm_I_r.reshape(-1, X) * up_cos_r
	blstm_i_out = blstm_i.reshape(1, -1)
	blstm_q = blstm_Q_r.reshape(-1, X) * up_sin_r
	blstma_q_out = blstm_q.reshape(1, -1)
	blstm_iq_out[i * change_block: (i + 1) * change_block, :] = blstm_i_out + blstma_q_out
	#------end-----

# IQ_out_r = ps
sigpwr_org = np.linalg.norm(org_IQ_out) ** 2 / org_IQ_out.shape[1]
sigpwr = np.linalg.norm(IQ_out_r) ** 2 / IQ_out_r.shape[1]
sigpwr_blstm = np.linalg.norm(blstm_iq_out) ** 2 / blstm_iq_out.shape[1]

for i_k in range(0, K):
	print("Rician K factor: ", i_k)
	for m in range(len(SNR)):
		sigpwr_pred = np.linalg.norm(pred_iq_out[m]) ** 2 / pred_iq_out[m].shape[1]
		snr_wp = 10 ** (SNR[m] / 10)
		sgma_org = np.sqrt(X * sigpwr_org / snr_wp / 2 / np.log2(M))
		sgma = np.sqrt(X * sigpwr / snr_wp / 2 / np.log2(M))
		sgma_pred = np.sqrt(X * sigpwr_pred / snr_wp / 2 / np.log2(M))
		sgma_blstm = np.sqrt(X * sigpwr_blstm / snr_wp / 2 / np.log2(M))

		n_org = sgma_org * np.random.randn(1, np.size(org_IQ_out))
		n = sgma * np.random.randn(1, np.size(IQ_out_r))
		n_pred = sgma_pred * np.random.randn(1, np.size(pred_iq_out[m]))
		n_blstm = sgma_blstm * np.random.randn(1, np.size(blstm_iq_out))

		if(ch_mode == 0):	# AWGN
			org_receive = org_IQ_out + n                          # No IQ imbalance
			receive_data = IQ_out_r + n
			pred_rev = pred_iq_out[m] + n
			blstm_rev = blstm_iq_out + n
		elif(ch_mode == 1):	# Rayleigh Fading
			# Ideal channel
			h_Ideal = (np.random.randn(org_IQ_out.size) + 1j * np.random.randn(org_IQ_out.size)) / np.sqrt(2)
			# Normal channel
			h_normal = (np.random.randn(IQ_out_r.size) + 1j * np.random.randn(IQ_out_r.size)) / np.sqrt(2)
			# IQ compensation channel
			h_comp = (np.random.randn(pred_iq_out[m].size) + 1j * np.random.randn(pred_iq_out[m].size)) / np.sqrt(2)
			# BLSTM channel
			h_blstm = (np.random.randn(blstm_iq_out.size) + 1j * np.random.randn(blstm_iq_out.size)) / np.sqrt(2)

			# Rayleigh + AWGN
			org_receive = h_Ideal * org_IQ_out + n_org			# Ideal
			receive_data = h_normal * IQ_out_r + n	# Normal
			pred_rev = h_comp * pred_iq_out[m] + n_pred		# IQ compensation
			blstm_rev = h_blstm * blstm_iq_out + n_blstm

			# Equalization
			org_receive = org_receive / h_Ideal			# Ideal
			receive_data = receive_data / h_normal	# Normal
			pred_rev = pred_rev / h_comp		# IQ compensation
			blstm_rev = blstm_rev / h_blstm
		else:	# Rician Fading
			k_0 = 4 * (i_k + 1)
			frame = org_IQ_out.size
			h = np.sqrt(k_0 / (1 + k_0)) * np.ones((1, frame)) + np.sqrt(1 / (1 + k_0)) * ((np.random.randn(1, frame) + 1j * np.random.randn(1, frame)) / np.sqrt(2))
			org_receive = h * org_IQ_out + n_org			# Ideal
			receive_data = h * IQ_out_r + n	# Normal
			pred_rev = h * pred_iq_out[m] + n_pred		# IQ compensation
			blstm_rev = h * blstm_iq_out + n_blstm

		org_rx_IQ_out = org_receive.reshape(-1, X)
		rx_IQ_out_r = receive_data.reshape(-1, X)
		pred_rx_iq = pred_rev.reshape(-1, X)
		blstm_rx_iq = blstm_rev.reshape(-1, X)

		org_down_cos = cos_sampling(X, Fq, 0, 0)              # No IQ imbalance
		org_down_sin = sin_sampling(X, Fq, 0, 0)              # No IQ imbalance
		down_cos_r = cos_sampling(X, Fq, down_C, down_upsilon)
		down_sin_r = sin_sampling(X, Fq, down_C, down_upsilon)

		org_rx_I_out = np.dot(org_rx_IQ_out, org_down_cos.T)  # No IQ imbalance
		org_rx_Q_out = np.dot(org_rx_IQ_out, org_down_sin.T)  # No IQ imbalance
		rx_I_out_r = np.dot(rx_IQ_out_r, down_cos_r.T)
		rx_Q_out_r = np.dot(rx_IQ_out_r, down_sin_r.T)
		pred_rx_i = np.dot(pred_rx_iq, down_cos_r.T)
		pred_rx_q = np.dot(pred_rx_iq, down_sin_r.T)
		blstm_rx_i = np.dot(blstm_rx_iq, down_cos_r.T)
		blstm_rx_q = np.dot(blstm_rx_iq, down_sin_r.T)

		org_rx_Or_I_out = org_rx_I_out * 2 / X                # No IQ imbalance
		org_rx_Or_Q_out = org_rx_Q_out * 2 / X                # No IQ imbalance
		rx_Or_I_out_r = rx_I_out_r * 2 / X  # cos(theta) ** 2 = 1 / 2 + cos(2 * theta)
		rx_Or_Q_out_r = rx_Q_out_r * 2 / X  # sin(theta) ** 2 = 1 / 2 - cos(2 * theta)
		scale_pred_i = pred_rx_i * 2 / X
		scale_pred_q = pred_rx_q * 2 / X
		scale_blstm_i = blstm_rx_i * 2 / X
		scale_blstm_q = blstm_rx_q * 2 / X

		org_rx_out = org_rx_Or_I_out + org_rx_Or_Q_out * 1j   # No IQ imbalance
		rx_out_r = rx_Or_I_out_r + rx_Or_Q_out_r * 1j	      # IQ imbalance
		pred_rx_out = scale_pred_i + scale_pred_q * 1j	      # Predistortion
		blstm_rx_out = scale_blstm_i + scale_blstm_q * 1j	  # BLSTM

		org_rx_sp = org_rx_out.reshape(-1, N)                 # No IQ imbalance
		rx_sp_r = rx_out_r.reshape(-1, N)		      # IQ imabalance
		pred_rx_sp = pred_rx_out.reshape(-1, N)		      # Predistortion
		blstm_rx_sp = blstm_rx_out.reshape(-1, N)			  # BLSTM

		org_fft_out = np.fft.fft(org_rx_sp, N)                # No IQ imbalance
		fft_out_r = np.fft.fft(rx_sp_r, N)		      # IQ imabalance
		pred_fft_out = np.fft.fft(pred_rx_sp, N)	      # Predistortion
		blstm_fft_out = np.fft.fft(blstm_rx_sp, N)			  # BLSTM

		org_rx_ps = org_fft_out.reshape(1, -1)                # No IQ imbalance
		rx_ps_r = fft_out_r.reshape(1, -1)		      # IQ imabalance
		pred_rx_ps = pred_fft_out.reshape(1, -1)	      # Predistortion
		blstm_rx_ps = blstm_fft_out.reshape(1, -1)			  # BLSTM

		org_hard = hard_decision(org_rx_ps, M)                # No IQ imbalance
		out_hard = hard_decision(rx_ps_r, M)		      # IQ imabalance
		pred_hard = hard_decision(pred_rx_ps, M)	      # Predistortion
		blstm_hard = hard_decision(blstm_rx_ps, M)			  # BLSTM

		org_ber[i_k, m] = ber_call_qpsk(org_hard, b, M)            # No IQ imbalance
		ber[i_k, m] = ber_call_qpsk(out_hard, b, M)		      # IQ imabalance
		pred_ber[i_k, m] = ber_call_qpsk(pred_hard, b, M)	      # Predistortion
		blstm_ber[i_k, m] = ber_call_qpsk(blstm_hard, b, M)		  # BLSTM
		print(org_ber[i_k, m], ber[i_k, m], pred_ber[i_k, m], blstm_ber[i_k, m])

print(f"Ideal BER: {org_ber}\n")
print(f"IQ imbalance BER: {ber}\n")
print(f"Predistortion BER: {pred_ber}\n")
print(f"BLSTM BER: {blstm_ber}\n")

for i in range(len(SNR)):
    t_snr = 10 ** (SNR[i] / 10)
    theo_err[i] = (1 / 2) * erfc(np.sqrt(t_snr)) - (1 / 8) * (erfc(np.sqrt(t_snr))) ** 2

# plt.figure(3)
# plt.plot(rx_ps_comp.real[:, 0: 1000], rx_ps_comp.imag[:, 0: 1000], 'b.', label='Contallation')
# plt.savefig('output1.png', bbox_inches='tight')

plt.figure(4)
plt.semilogy(SNR, org_ber[4], 'g', label='No IQ imabalance BER')
if ch_mode == 0:
	plt.semilogy(SNR, theo_err[4], 'k', label='Theoretical BER')
plt.semilogy(SNR, ber[4], 'b', label='Simulated BER')
plt.semilogy(SNR, pred_ber[4], 'r', label='Predistortion BER')
plt.semilogy(SNR, blstm_ber[4], 'm', label='BLSTM BER')
plt.xlabel('SNR (dB)')
plt.ylabel('BER')
plt.axis([0, 12, 1e-5, 1])
plt.legend()
plt.savefig('BER_plot.png')