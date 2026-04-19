import numpy as np
import time
import matplotlib.pyplot as plt
import csv
from scipy.special import erfc

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

VAR = 100

N = 32
Block = 1 * 1000000
M = 4
D = N * Block
X = 4
Fq = 1
change_block = int(D / X)

np.random.seed(seed=int(time.time()))

up_C = 4    # np.random.random_sample()
up_upsilon = 0.2    #np.random.random_sample()
down_C = 0  # np.random.randint(0, 15)
down_upsilon = 0 #np.random.random_sample() * 0.3
# print(f"Up C: {up_C}, Down C: {down_C}\n")
# print(f"Up Upsilon: {up_upsilon}, Down Upsilon: {down_upsilon}\n")

SNR = np.arange(0, 20, 2)
org_ber = np.zeros(len(SNR))
ber = np.zeros(len(SNR))
pred_ber = np.zeros(len(SNR))	# Predistortion
theo_err = np.zeros(len(SNR))
I_r0 = np.zeros((D, X))
Q_r0 = np.zeros((D, X))
conv_I_r = np.zeros((D, X))
conv_Q_r = np.zeros((D, X))
pred_i = np.zeros((D, X))
pred_q = np.zeros((D, X))

b = gen_mapping(M, D)

# 송신기
sp = b.reshape(-1, N)
ifft_out = np.fft.ifft(sp)
ps = ifft_out.reshape(-1, 1)
I_r = np.real(ps)
Q_r = np.imag(ps)
org_up_cos = cos_sampling(X, Fq, 0, 0)      # No IQ imbalance
org_up_sin = sin_sampling(X, Fq, 0, 0)      # No IQ imbalance		

# print(ps.shape)
# print(up_cos_r.shape)
# print(up_sin_r.shape)


org_I = np.dot(I_r, org_up_cos)           # No IQ imbalance
org_I_out = org_I.reshape(1, -1)
org_Q = np.dot(Q_r, org_up_sin)           # No IQ imbalance
org_Q_out = org_Q.reshape(1, -1)
org_IQ_out = org_I_out + org_Q_out        # No IQ imbalance
with open("input_iq_I.csv", mode='w') as go1_I:
    fidw1_I = csv.writer(go1_I, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for idx in range(len(org_I_out.squeeze())):
        fidw1_I.writerow([org_I_out[0, idx]])
with open("input_iq_Q.csv", mode='w') as go1_Q:
    fidw1_Q = csv.writer(go1_Q, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for idx in range(len(org_Q_out.squeeze())):
        fidw1_Q.writerow([org_Q_out[0, idx]])

if 'VAR' in globals():
	epsilon = up_upsilon
	phi = up_C
	for i in range(X):
		# up_upsilon = np.round(np.random.uniform(0.2, 0.4), 1)
		# up_C = np.random.randint(4, 7)
		epsilon += 0.1
		phi += 1
		print(i * change_block)
		I_r0[i * change_block: (i + 1) * change_block, :] = cos_predistor(org_I[i * change_block: (i + 1) * change_block, :], X, Fq, epsilon, phi)
		Q_r0[i * change_block: (i + 1) * change_block, :] = sin_predistor(org_Q[i * change_block: (i + 1) * change_block, :], X, Fq, epsilon, phi)
else:
	I_r0 = cos_predistor(org_I, X, Fq, up_upsilon, up_C)
	Q_r0 = sin_predistor(org_Q, X, Fq, up_upsilon, up_C)
input_I = I_r0.reshape(-1, 1)
input_Q = Q_r0.reshape(-1, 1)
with open("input_I_r.csv", mode='w') as go2_I:
	fidw2_I = csv.writer(go2_I, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
	for idx in range(len(input_I)):
		fidw2_I.writerow(input_I[idx])
with open("input_Q_r.csv", mode='w') as go2_Q:
	fidw2_Q = csv.writer(go2_Q, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
	for idx in range(len(input_Q)):
		fidw2_Q.writerow(input_Q[idx])	
		
if 'VAR' in globals():
	epsilon = up_upsilon
	phi = up_C
	for i in range(X):
		epsilon += 0.1
		phi += 1
		up_cos_r = cos_sampling(X, Fq, phi, epsilon)
		up_sin_r = sin_sampling(X, Fq, phi, epsilon)
		conv_I_r[i * change_block: (i + 1) * change_block, :] = np.dot(I_r[i * change_block: (i + 1) * change_block, :], up_cos_r)
		conv_Q_r[i * change_block: (i + 1) * change_block, :] = np.dot(Q_r[i * change_block: (i + 1) * change_block, :], up_sin_r)
		
		pred_i[i * change_block: (i + 1) * change_block, :] = I_r0[i * change_block: (i + 1) * change_block, :] * up_cos_r
		pred_q[i * change_block: (i + 1) * change_block, :] = Q_r0[i * change_block: (i + 1) * change_block, :] * up_sin_r
else:
	up_cos_r = cos_sampling(X, Fq, up_C, up_upsilon)
	up_sin_r = sin_sampling(X, Fq, up_C, up_upsilon)
	conv_I_r = np.dot(I_r, up_cos_r)
	conv_Q_r = np.dot(Q_r, up_sin_r)
	
	pred_i = I_r0 * up_cos_r
	pred_q = Q_r0 * up_sin_r
	
conv_Iout_r = conv_I_r.reshape(1, -1)
conv_Qout_r = conv_Q_r.reshape(1, -1)
IQ_out_r = conv_Iout_r + conv_Qout_r

pred_i_out = pred_i.reshape(1, -1)
pred_q_out = pred_q.reshape(1, -1)
pred_iq_out = pred_i_out + pred_q_out
		
# IQ_out_r = ps
sigpwr = np.linalg.norm(IQ_out_r) ** 2 / IQ_out_r.shape[1]

for m in range(len(SNR)):
	snr_wp = 10 ** (SNR[m] / 10)
	sgma = np.sqrt(X * sigpwr / snr_wp / 2 / np.log2(M))
	n = sgma * np.random.randn(1, np.size(IQ_out_r))
	org_receive = org_IQ_out + n                          # No IQ imbalance
	receive_data = IQ_out_r + n
	pred_rev = pred_iq_out + n
	org_rx_IQ_out = org_receive.reshape(-1, X)
	rx_IQ_out_r = receive_data.reshape(-1, X)
	pred_rx_iq = pred_rev.reshape(-1, X)
	
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
	
	org_rx_Or_I_out = org_rx_I_out * 2 / X                # No IQ imbalance
	org_rx_Or_Q_out = org_rx_Q_out * 2 / X                # No IQ imbalance
	rx_Or_I_out_r = rx_I_out_r * 2 / X  # cos(theta) ** 2 = 1 / 2 + cos(2 * theta)
	rx_Or_Q_out_r = rx_Q_out_r * 2 / X  # sin(theta) ** 2 = 1 / 2 - cos(2 * theta)
	scale_pred_i = pred_rx_i * 2 / X
	scale_pred_q = pred_rx_q * 2 / X
	
	org_rx_out = org_rx_Or_I_out + org_rx_Or_Q_out * 1j   # No IQ imbalance
	rx_out_r = rx_Or_I_out_r + rx_Or_Q_out_r * 1j	      # IQ imbalance
	pred_rx_out = scale_pred_i + scale_pred_q * 1j	      # Predistortion
	
	org_rx_sp = org_rx_out.reshape(-1, N)                 # No IQ imbalance
	rx_sp_r = rx_out_r.reshape(-1, N)		      # IQ imabalance
	pred_rx_sp = pred_rx_out.reshape(-1, N)		      # Predistortion
	
	org_fft_out = np.fft.fft(org_rx_sp, N)                # No IQ imbalance
	fft_out_r = np.fft.fft(rx_sp_r, N)		      # IQ imabalance
	pred_fft_out = np.fft.fft(pred_rx_sp, N)	      # Predistortion
	
	org_rx_ps = org_fft_out.reshape(1, -1)                # No IQ imbalance
	rx_ps_r = fft_out_r.reshape(1, -1)		      # IQ imabalance
	pred_rx_ps = pred_fft_out.reshape(1, -1)	      # Predistortion
	
	org_hard = hard_decision(org_rx_ps, M)                # No IQ imbalance
	out_hard = hard_decision(rx_ps_r, M)		      # IQ imabalance
	pred_hard = hard_decision(pred_rx_ps, M)	      # Predistortion
	
	org_ber[m] = ber_call_qpsk(org_hard, b, M)            # No IQ imbalance
	ber[m] = ber_call_qpsk(out_hard, b, M)		      # IQ imabalance
	pred_ber[m] = ber_call_qpsk(pred_hard, b, M)	      # Predistortion
print(f"BER: {ber}\n")
print(f"Predistortion BER: {pred_ber}\n")

for i in range(len(SNR)):
    t_snr = 10 ** (SNR[i] / 10)
    theo_err[i] = (1 / 2) * erfc(np.sqrt(t_snr)) - (1 / 8) * (erfc(np.sqrt(t_snr))) ** 2

# plt.figure(3)
# plt.plot(rx_ps_comp.real[:, 0: 1000], rx_ps_comp.imag[:, 0: 1000], 'b.', label='Contallation')
# plt.savefig('output1.png', bbox_inches='tight')

plt.figure(4)
plt.semilogy(SNR, org_ber, 'g', label='No IQ imabalance BER')
plt.semilogy(SNR, theo_err, 'k', label='Theoretical BER')
plt.semilogy(SNR, ber, 'b', label='Simulated BER')
plt.semilogy(SNR, pred_ber, 'r', label='Predistortion BER')
plt.xlabel('SNR (dB)')
plt.ylabel('BER')
plt.axis([0, 12, 1e-5, 1])
plt.legend()
plt.savefig('BER_plot.png')
