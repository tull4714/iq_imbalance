# @title
"""
Two-state RF-feedback TX I/Q imbalance estimation and predistortion with 3GPP NR polar coding
==============================================================================================
No LO frequency offset between the TX LO and the feedback LO is assumed.

Feedback observation (state k = 0, 1):
    [I~; Q~] = g_k * M_FB(eps_R, phi_R) R(theta0 + psi_k) M_TX(eps_T, phi_T) [I; Q] + noise
  - state 0: psi_0 = 0, gain g;  state 1: psi_1 = psi (FB-LO phase change), gain g * kappa
  - actual psi = STATE_PSI_NOM + STATE_PSI_ERR, kappa = STATE_GAIN_DB  (both unknown to 'prop')

Schemes
  'prop'       : proposed two-state closed form (det -> kappa, trace -> cos(psi), C -> eps_T, phi_T),
                 data-aided LS on the transmitted OFDM data, time-domain predistorter M^-1 (eq. (14)).
  'huang'      : X. Huang, Y. J. Guo, J. A. Zhang, IEEE TWC 2016, eqs. (17)-(25): dedicated training
                 A(f), B(f) in both states, RX-image removal with an ASSUMED phase-shifter response P
                 (HUANG_P_*, set arbitrarily), per-subcarrier beta_t(f), frequency-domain
                 pre-compensation X(f) - beta_t(f) X*(-f).
  'huang_flat' : same as 'huang', but beta_t(f) is averaged over the subcarriers (flat I/Q imbalance
                 is assumed, as in 'prop'), so that both methods use the same model.
  'uncomp'     : no compensation.          'ideal' : no I/Q imbalance (reference).

Channel coding: 3GPP TS 38.212 polar code for UCI (CRC11, n_max = 10, sub-block interleaver,
  shortening / puncturing / repetition, coded-bit interleaver), CRC-aided SCL decoding (numba).
Modulation: TS 38.211 QPSK / 16QAM / 64QAM mapping, max-log LLR demapping.
Channels: AWGN, and i.i.d. Rayleigh fading per resource element (subcarrier x OFDM symbol) with
  perfect CSI at the receiver (zero-forcing equalization, LLRs weighted by |H|^2).

Metrics
  Experiment 1 (estimation): MSE of eps_T and phi_T, CRB of the two-state model, residual image-to-signal
    ratio (ISR, exact), residual EVM (noise-free), estimation failure rate, MSE of psi_hat (proposed),
    versus feedback SNR, state phase error, and state gain difference.
  Experiment 2 (link): BLER, coded BER, uncoded (raw) BER, EVM, normalized throughput, and estimation
    statistics, versus Eb/N0 per information bit, for each modulation and channel.
Results are saved to .mat files (and .png / .pdf figures) in OUT_DIR.
"""
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc
from scipy.io import savemat
try:
    from numba import njit, prange
except ImportError as exc:                       # Colab ships numba; otherwise: pip install numba
    raise ImportError('numba is required: pip install numba') from exc
_NB_CACHE = os.path.isfile(globals().get('__file__', ''))   # numba cache only when run from a file


# ============================================================
# Variables saved in the .mat files
# ============================================================
# ---- two_state_polar_mse.mat (experiment 1) -----------------------------------------------------
# methods              : estimation methods = column order of the 2-D arrays below (prop, huang, huang_flat)
# trials, mse_mod      : Monte Carlo trials per point, modulation order of the data used for estimation
# state_psi_nom        : nominal phase change between the two states [deg]
# state_psi_err        : actual - nominal phase change used in the 'snr' sweep [deg]
# state_gain_db        : gain ratio kappa between the states used in the 'snr' sweep [dB]
# huang_p_phase_deg, huang_p_gain_db : phase-shifter response ASSUMED by Huang's method
# huang_train_rep      : repetitions of the training symbols A(f), B(f) per state
# mse_err_snr, fb_gain : feedback SNRs of the 'perr'/'gerr' sweeps [dB], common feedback gain g
# Sweeps, <tag> = snr      (x: feedback SNR [dB];                state error = state_psi_err, state_gain_db)
#                 perr<S>  (x: state phase error psi - 90 [deg]; gain ratio 0 dB, feedback SNR <S> dB)
#                 gerr<S>  (x: state gain ratio kappa [dB];      phase error 0,   feedback SNR <S> dB)
#                 with <S> in mse_err_snr, e.g. perr20, perr40, gerr20, gerr40
# <tag>_x              : x axis of the sweep
# <tag>_mse_eps        : MSE of eps_T                      (n_x by n_methods)
# <tag>_mse_phi        : MSE of phi_T [deg^2]              (n_x by n_methods)
# <tag>_isr_db         : residual image-to-signal ratio after compensation [dB] (n_x by n_methods)
# <tag>_res_evm        : noise-free residual EVM after compensation [%]         (n_x by n_methods)
# <tag>_fail_rate      : fraction of failed / out-of-domain estimates           (n_x by n_methods)
# <tag>_crb_eps, <tag>_crb_phi : CRB of eps_T and phi_T [deg^2] for the two-state model (n_x)
# <tag>_isr_uncomp_db  : image-to-signal ratio without compensation [dB]        (n_x)
# <tag>_mse_psi_prop   : MSE of the phase change psi estimated by 'prop' [deg^2] (n_x)
#
# ---- two_state_polar_link.mat (experiment 2) ----------------------------------------------------
# schemes, mod_list, channels : link schemes, modulation orders, channel names
# fb_snr_link          : feedback SNR used for the estimation in the link [dB]
# state_psi_nom, state_psi_err, state_gain_db, huang_p_phase_deg, huang_p_gain_db : as above
# polar_A, polar_K, polar_E, polar_N : payload bits, K = A + CRC bits, rate-matched length, mother code length
# polar_mode, polar_rate, polar_list : rate matching (shortening/puncturing/repetition), A/E, SCL list size
# target_block_err     : block errors after which a (scheme, Eb/N0) point stops being decoded
# min_frames           : minimum slots (I/Q imbalance realizations) decoded for every point
# rx_gain_eq           : 1 = receiver equalizes the desired-signal gain left by the compensation
# For each <tag> = <M>QAM_<channel> (e.g. 16QAM_Rayleigh), vectors over the Eb/N0 points:
# ebn0_<tag>           : Eb/N0 per information bit [dB]
# esn0_<tag>           : Es/N0 per modulation symbol [dB] = ebn0 + 10 log10(rate * log2 M)
# frames_<tag>, max_frames_<tag> : slots actually simulated, slot limit
# theory_raw_ber_<tag> : theoretical uncoded BER of M-QAM in AWGN (valid for the AWGN channel only)
# For each scheme <s> (ideal, uncomp, prop, huang, huang_flat):
# bler_<s>_<tag>       : block error rate = nblkerr / ncw
# ber_<s>_<tag>        : bit error rate of the payload bits after decoding
# rawber_<s>_<tag>     : uncoded bit error rate (hard decision before decoding, all slots)
# evm_<s>_<tag>        : EVM at the receiver including noise [%] (Rayleigh: error weighted by |H|^2)
# thr_<s>_<tag>        : normalized throughput (1 - BLER) * rate * log2 M [information bits / symbol]
# ncw_<s>_<tag>        : number of decoded codewords  (use with nblkerr for confidence intervals)
# nblkerr_<s>_<tag>    : number of block errors
# For each estimation method <m> (prop, huang, huang_flat), scalars averaged over the slots:
# est_mse_eps_<m>_<tag>, est_mse_phi_<m>_<tag> : MSE of eps_T and phi_T [deg^2] in the link
# est_isr_db_<m>_<tag> : residual image-to-signal ratio in the link [dB]
# Predictions (no decoding involved, computed from the residual image-to-signal ratio ISR of every slot):
# isr_db_<s>_<tag>         : mean ISR of scheme <s> over the slots [dB] (ideal: -inf)
# rawber_pred_<s>_<tag>    : theoretical uncoded BER, exact Gray M-QAM formula at the SINR with the image
#                            treated as Gaussian interference (Rayleigh: averaged over |H|^2), slot average
# snrloss_pred_<s>_<tag>   : predicted SNR loss against 'ideal' [dB] (mutual-information-equivalent SNR)
# bler_pred_<s>_<tag>      : predicted BLER = simulated 'ideal' BLER read at the equivalent SNR, slot average
# ber_pred_<s>_<tag>       : predicted coded BER, same mapping applied to the simulated 'ideal' coded BER
#                            (NaN where the 'ideal' reference curve has no observed errors)


# ============================================================
# Configuration
# ============================================================
SEED = 2026
SAVE_TO_DRIVE = True                                          # Colab: mount Google Drive and save there
DRIVE_DIR = '/content/drive/MyDrive/IQ_imbalance_BLSTM/predistortion/two_state'
LOCAL_DIR = './results_two_state_polar'

# ---- OFDM / carrier sampling (same as the base script)
N = 1024                        # subcarriers (all used)
SYMB = 14                       # OFDM symbols per slot (one estimation interval)
X = 4                           # RF samples per baseband sample
Fq = 1
SLOT = N * SYMB

# ---- I/Q imbalance and feedback path
EPS_RANGE = (0.1, 0.6)          # TX gain error
PHI_RANGE = (1.0, 6.0)          # TX phase error [deg]
EPS_R_RANGE = (0.1, 0.6)        # feedback-RX gain error
PHI_R_RANGE = (1.0, 6.0)        # feedback-RX phase error [deg]
FB_GAIN = 0.7                   # common feedback gain g
THETA0_DEG = None               # common phase of state 0 [deg]; None = U(0, 360) per interval

# ---- two-state switching (actual values; unknown to the proposed method)
STATE_PSI_NOM = 90.0            # nominal FB-LO phase change between the states [deg]
STATE_PSI_ERR = 20.0            # actual - nominal [deg]
STATE_GAIN_DB = 0.5             # gain difference between the states [dB]

# ---- Huang et al.: parameters that must be known in advance (set arbitrarily here)
HUANG_P_PHASE_DEG = 90.0        # assumed phase-shifter phase [deg]
HUANG_P_GAIN_DB = 0.0           # assumed phase-shifter gain [dB]
HUANG_TRAIN_REP = 7             # repetitions of A(f) and B(f) per state (2 x 2 x 7 = 28 OFDM symbols,
                                #  equal to the 2 x 14 data symbols used by 'prop')

# ---- methods
EST_METHODS = ['prop', 'huang', 'huang_flat']
LINK_SCHEMES = ['ideal', 'uncomp', 'prop', 'huang', 'huang_flat']   # 'ideal': no-I/Q-imbalance reference
HUANG_SET = ('huang', 'huang_flat')

# ---- polar code (3GPP UCI)
POLAR_A = 200                   # payload bits per codeword (20 <= A < 360)
POLAR_E = 432                   # rate-matched codeword length
POLAR_LIST = 8                  # CA-SCL list size

# ---- experiment 1: estimation accuracy
RUN_MSE = True
MSE_MOD = 16                    # modulation of the data used by the data-aided estimators
MSE_TRIALS = 300
MSE_FB_SNR = list(range(0, 41, 2))                          # feedback SNR sweep [dB]
MSE_ERR_SNR = [20.0, 40.0]                                  # feedback SNRs of the state-mismatch sweeps [dB]
MSE_PERR_LIST = list(range(0, 21))                          # state phase error psi - 90 [deg] (gain ratio 0 dB)
MSE_GERR_LIST = [round(0.1 * i, 1) for i in range(0, 21)]   # state gain ratio kappa [dB] (phase error 0)

# ---- experiment 2: coded link
RUN_LINK = True
MOD_LIST = [4, 16, 64]
CHANNELS = ['AWGN', 'Rayleigh']
EBN0_STEP = 0.5                 # [dB]
EBN0_RANGE = {                  # (first, last) Eb/N0 per information bit [dB] for (modulation, channel)
    (4, 'AWGN'): (0.0, 5.0), (4, 'Rayleigh'): (2.0, 8.0),
    (16, 'AWGN'): (3.0, 10.0), (16, 'Rayleigh'): (4.0, 12.5),
    (64, 'AWGN'): (5.5, 14.0), (64, 'Rayleigh'): (8.0, 17.0)}
FB_SNR_LINK = 40.0              # feedback SNR used for estimation in the link
MAX_FRAMES = {4: 1500, 16: 1000, 64: 700}   # max slots per (modulation, channel):
                                #  ~1e5 codewords per point -> BLER down to ~1e-3 with ~100 errors
MIN_FRAMES = 50                 # every point is decoded over at least this many slots (= I/Q imbalance
                                #  realizations), even if it reaches TARGET_BLOCK_ERR earlier
TARGET_BLOCK_ERR = 100          # stop decoding a (scheme, Eb/N0) point after this many block errors
SAVE_EVERY = 20                 # checkpoint interval [slots]
ZERO_STOP_CW = 20000            # a point with 0 block errors after this many codewords also stops the
                                #  higher Eb/N0 points of the same scheme
RX_GAIN_EQ = True               # receiver knows the end-to-end desired-signal gain per subcarrier
                                #  (perfect channel estimation); only the image interference remains

LABELS = {'prop': 'Proposed (unknown $\\psi$, $\\kappa$)', 'huang': 'Huang et al. (assumed P, per subcarrier)',
          'huang_flat': 'Huang et al. (assumed P, flat)', 'ideal': 'No I/Q imbalance', 'uncomp': 'Uncompensated'}
STYLE = {'prop': 'r-o', 'huang': 'b-s', 'huang_flat': 'g--^', 'ideal': 'k-', 'uncomp': 'c--'}
N0_UNIT = 4.0 * N / X                        # noise variance per subcarrier of the unit-variance RF noise
D2R = np.pi / 180
MIRROR_IDX = (N - np.arange(N)) % N          # index of -f for each subcarrier


# ============================================================
# 3GPP NR polar code (TS 38.212), uplink control information (UCI)
#   CRC attachment       : Sec. 5.1, 6.3.1.2.1 (CRC6 for 12<=A<=19, CRC11 for A>=20)
#   Polar encoding       : Sec. 5.3.1 (n_max = 10, I_IL = 0)
#   Rate matching        : Sec. 5.4.1 (sub-block interleaver, bit selection, I_BIL = 1)
#   Not supported        : code-block segmentation (A >= 360 with E >= 1088, or A >= 1013)
#                          and the 3 parity-check bits used only for 12 <= A <= 19
# Decoder: CRC-aided successive-cancellation list (CA-SCL) decoding, min-sum, numba.
# ============================================================
# Reliability sequence Q_0^{1023}, TS 38.212 Table 5.3.1.2-1 (ascending reliability)
NR_POLAR_Q = np.array([
    0, 1, 2, 4, 8, 16, 32, 3, 5, 64, 9, 6, 17, 10, 18, 128, 12, 33, 65, 20, 256, 34, 24, 36, 7, 129, 66, 512, 11, 40, 68, 130, 19, 13, 48, 14,
    72, 257, 21, 132, 35, 258, 26, 513, 80, 37, 25, 22, 136, 260, 264, 38, 514, 96, 67, 41, 144, 28, 69, 42, 516, 49, 74, 272, 160, 520,
    288, 528, 192, 544, 70, 44, 131, 81, 50, 73, 15, 320, 133, 52, 23, 134, 384, 76, 137, 82, 56, 27, 97, 39, 259, 84, 138, 145, 261, 29,
    43, 98, 515, 88, 140, 30, 146, 71, 262, 265, 161, 576, 45, 100, 640, 51, 148, 46, 75, 266, 273, 517, 104, 162, 53, 193, 152, 77, 164,
    768, 268, 274, 518, 54, 83, 57, 521, 112, 135, 78, 289, 194, 85, 276, 522, 58, 168, 139, 99, 86, 60, 280, 89, 290, 529, 524, 196,
    141, 101, 147, 176, 142, 530, 321, 31, 200, 90, 545, 292, 322, 532, 263, 149, 102, 105, 304, 296, 163, 92, 47, 267, 385, 546, 324,
    208, 386, 150, 153, 165, 106, 55, 328, 536, 577, 548, 113, 154, 79, 269, 108, 578, 224, 166, 519, 552, 195, 270, 641, 523, 275,
    580, 291, 59, 169, 560, 114, 277, 156, 87, 197, 116, 170, 61, 531, 525, 642, 281, 278, 526, 177, 293, 388, 91, 584, 769, 198, 172,
    120, 201, 336, 62, 282, 143, 103, 178, 294, 93, 644, 202, 592, 323, 392, 297, 770, 107, 180, 151, 209, 284, 648, 94, 204, 298, 400,
    608, 352, 325, 533, 155, 210, 305, 547, 300, 109, 184, 534, 537, 115, 167, 225, 326, 306, 772, 157, 656, 329, 110, 117, 212, 171,
    776, 330, 226, 549, 538, 387, 308, 216, 416, 271, 279, 158, 337, 550, 672, 118, 332, 579, 540, 389, 173, 121, 553, 199, 784, 179,
    228, 338, 312, 704, 390, 174, 554, 581, 393, 283, 122, 448, 353, 561, 203, 63, 340, 394, 527, 582, 556, 181, 295, 285, 232, 124,
    205, 182, 643, 562, 286, 585, 299, 354, 211, 401, 185, 396, 344, 586, 645, 593, 535, 240, 206, 95, 327, 564, 800, 402, 356, 307,
    301, 417, 213, 568, 832, 588, 186, 646, 404, 227, 896, 594, 418, 302, 649, 771, 360, 539, 111, 331, 214, 309, 188, 449, 217, 408,
    609, 596, 551, 650, 229, 159, 420, 310, 541, 773, 610, 657, 333, 119, 600, 339, 218, 368, 652, 230, 391, 313, 450, 542, 334, 233,
    555, 774, 175, 123, 658, 612, 341, 777, 220, 314, 424, 395, 673, 583, 355, 287, 183, 234, 125, 557, 660, 616, 342, 316, 241, 778,
    563, 345, 452, 397, 403, 207, 674, 558, 785, 432, 357, 187, 236, 664, 624, 587, 780, 705, 126, 242, 565, 398, 346, 456, 358, 405,
    303, 569, 244, 595, 189, 566, 676, 361, 706, 589, 215, 786, 647, 348, 419, 406, 464, 680, 801, 362, 590, 409, 570, 788, 597, 572,
    219, 311, 708, 598, 601, 651, 421, 792, 802, 611, 602, 410, 231, 688, 653, 248, 369, 190, 364, 654, 659, 335, 480, 315, 221, 370,
    613, 422, 425, 451, 614, 543, 235, 412, 343, 372, 775, 317, 222, 426, 453, 237, 559, 833, 804, 712, 834, 661, 808, 779, 617, 604,
    433, 720, 816, 836, 347, 897, 243, 662, 454, 318, 675, 618, 898, 781, 376, 428, 665, 736, 567, 840, 625, 238, 359, 457, 399, 787,
    591, 678, 434, 677, 349, 245, 458, 666, 620, 363, 127, 191, 782, 407, 436, 626, 571, 465, 681, 246, 707, 350, 599, 668, 790, 460,
    249, 682, 573, 411, 803, 789, 709, 365, 440, 628, 689, 374, 423, 466, 793, 250, 371, 481, 574, 413, 603, 366, 468, 655, 900, 805,
    615, 684, 710, 429, 794, 252, 373, 605, 848, 690, 713, 632, 482, 806, 427, 904, 414, 223, 663, 692, 835, 619, 472, 455, 796, 809,
    714, 721, 837, 716, 864, 810, 606, 912, 722, 696, 377, 435, 817, 319, 621, 812, 484, 430, 838, 667, 488, 239, 378, 459, 622, 627,
    437, 380, 818, 461, 496, 669, 679, 724, 841, 629, 351, 467, 438, 737, 251, 462, 442, 441, 469, 247, 683, 842, 738, 899, 670, 783,
    849, 820, 728, 928, 791, 367, 901, 630, 685, 844, 633, 711, 253, 691, 824, 902, 686, 740, 850, 375, 444, 470, 483, 415, 485, 905,
    795, 473, 634, 744, 852, 960, 865, 693, 797, 906, 715, 807, 474, 636, 694, 254, 717, 575, 913, 798, 811, 379, 697, 431, 607, 489,
    866, 723, 486, 908, 718, 813, 476, 856, 839, 725, 698, 914, 752, 868, 819, 814, 439, 929, 490, 623, 671, 739, 916, 463, 843, 381,
    497, 930, 821, 726, 961, 872, 492, 631, 729, 700, 443, 741, 845, 920, 382, 822, 851, 730, 498, 880, 742, 445, 471, 635, 932, 687,
    903, 825, 500, 846, 745, 826, 732, 446, 962, 936, 475, 853, 867, 637, 907, 487, 695, 746, 828, 753, 854, 857, 504, 799, 255, 964,
    909, 719, 477, 915, 638, 748, 944, 869, 491, 699, 754, 858, 478, 968, 383, 910, 815, 976, 870, 917, 727, 493, 873, 701, 931, 756,
    860, 499, 731, 823, 922, 874, 918, 502, 933, 743, 760, 881, 494, 702, 921, 501, 876, 847, 992, 447, 733, 827, 934, 882, 937, 963,
    747, 505, 855, 924, 734, 829, 965, 938, 884, 506, 749, 945, 966, 755, 859, 940, 830, 911, 871, 639, 888, 479, 946, 750, 969, 508,
    861, 757, 970, 919, 875, 862, 758, 948, 977, 923, 972, 761, 877, 952, 495, 703, 935, 978, 883, 762, 503, 925, 878, 735, 993, 885,
    939, 994, 980, 926, 764, 941, 967, 886, 831, 947, 507, 889, 984, 751, 942, 996, 971, 890, 509, 949, 973, 1000, 892, 950, 863, 759,
    1008, 510, 979, 953, 763, 974, 954, 879, 981, 982, 927, 995, 765, 956, 887, 985, 997, 986, 943, 891, 998, 766, 511, 988, 1001,
    951, 1002, 893, 975, 894, 1009, 955, 1004, 1010, 957, 983, 958, 987, 1012, 999, 1016, 767, 989, 1003, 990, 1005, 959, 1011, 1013,
    895, 1006, 1014, 1017, 1018, 991, 1020, 1007, 1015, 1019, 1021, 1022, 1023
], dtype=np.int64)

NR_SUBBLOCK_P = np.array([0, 1, 2, 4, 3, 5, 6, 7, 8, 16, 9, 17, 10, 18, 11, 19,
                          12, 20, 13, 21, 14, 22, 15, 23, 24, 25, 26, 28, 27, 29, 30, 31])   # Table 5.4.1.1-1

NR_CRC_POLY = {6: [6, 5, 0], 11: [11, 10, 9, 5, 0], 24: [24, 23, 21, 20, 17, 15, 13, 12, 8, 4, 2, 1, 0]}


def crc_poly_bits(L):
    """g(D) as a bit vector [g_L, ..., g_0] (MSB first)."""
    g = np.zeros(L + 1, dtype=np.uint8)
    g[[L - c for c in NR_CRC_POLY[L]]] = 1
    return g


def crc_parity(a, L):
    """CRC parity bits p_0..p_{L-1} of Sec. 5.1 (zero initial register) for a batch a: (B, A) -> (B, L)."""
    g = crc_poly_bits(L)[1:]
    reg = np.zeros((a.shape[0], L), dtype=np.uint8)
    for i in range(a.shape[1]):
        fb = reg[:, 0] ^ a[:, i]
        reg[:, :-1] = reg[:, 1:]
        reg[:, -1] = 0
        reg ^= fb[:, None] * g[None, :]
    return reg


def subblock_pattern(N):
    """J(n) of Sec. 5.4.1.1: y_n = d_{J(n)}."""
    n = np.arange(N)
    i = (32 * n) // N
    return NR_SUBBLOCK_P[i] * (N // 32) + n % (N // 32)


def triangular_interleaver(E):
    """Coded-bit interleaver of Sec. 5.4.1.3 (I_BIL = 1): returns idx with f_k = e_{idx[k]}."""
    T = 0
    while T * (T + 1) // 2 < E:
        T += 1
    v = -np.ones((T, T), dtype=np.int64)
    k = 0
    for i in range(T):
        for j in range(T - i):
            if k < E:
                v[i, j] = k
            k += 1
    out = [v[i, j] for j in range(T) for i in range(T - j) if v[i, j] >= 0]
    return np.array(out, dtype=np.int64)


def polar_transform(u):
    """x = u G_N, G_N = [[1,0],[1,1]]^{(x)n}, natural order. u: (B, N) uint8."""
    x = u.copy()
    N = x.shape[1]
    h = 1
    while h < N:
        x = x.reshape(x.shape[0], -1, 2, h)
        x[:, :, 0, :] ^= x[:, :, 1, :]
        x = x.reshape(x.shape[0], N)
        h *= 2
    return x


class NRPolarUCI:
    """3GPP NR polar code for UCI: A payload bits -> E rate-matched bits."""

    def __init__(self, A, E, list_size=8, ibil=True, n_max=10):
        if A < 20:
            raise ValueError('A >= 20 is required (CRC11, no PC bits).')
        if A >= 1013 or (A >= 360 and E >= 1088):
            raise ValueError('code-block segmentation is not implemented (use A < 360 or E < 1088).')
        self.A, self.E, self.Lcrc, self.list_size, self.ibil = A, E, 11, list_size, ibil
        K = A + self.Lcrc
        if K > E:
            raise ValueError('K > E')
        self.K = K
        # mother code length, Sec. 5.3.1
        lE = int(np.ceil(np.log2(E)))
        n1 = lE - 1 if (E <= (9 / 8) * 2 ** (lE - 1) and K / E < 9 / 16) else lE
        n2 = int(np.ceil(np.log2(K / (1 / 8))))
        n = max(min(n1, n2, n_max), 5)
        N = 2 ** n
        self.n, self.N = n, N
        J = subblock_pattern(N)
        # frozen set, Sec. 5.3.1.2
        Qn = NR_POLAR_Q[NR_POLAR_Q < N]
        qf = set()
        if E < N:
            if K / E <= 7 / 16:                                   # puncturing
                self.mode = 'puncturing'
                qf.update(J[:N - E].tolist())
                t = int(np.ceil(3 * N / 4 - E / 2)) if E >= 3 * N / 4 else int(np.ceil(9 * N / 16 - E / 4))
                qf.update(range(t))
            else:                                                 # shortening
                self.mode = 'shortening'
                qf.update(J[E:N].tolist())
        else:
            self.mode = 'repetition' if E > N else 'none'
        cand = np.array([q for q in Qn if q not in qf], dtype=np.int64)
        self.info_pos = np.sort(cand[-K:])
        self.frozen = np.ones(N, dtype=np.bool_)
        self.frozen[self.info_pos] = False
        # rate matching, Sec. 5.4.1.2: e_k = y_{sel[k]}, y_n = d_{J(n)}
        if E >= N:
            sel = np.arange(E) % N
        elif self.mode == 'puncturing':
            sel = np.arange(E) + (N - E)
        else:
            sel = np.arange(E)
        idx = J[sel]                                              # e_k = d_{idx[k]}
        if ibil:
            idx = idx[triangular_interleaver(E)]                  # f_k = e_{T(k)}
        self.idx_rm = idx.astype(np.int64)
        self.shortened = np.zeros(N, dtype=np.bool_)
        if self.mode == 'shortening':
            self.shortened[J[E:N]] = True
        # CRC parity-check helpers for the decoder
        self.crc_g = crc_poly_bits(self.Lcrc)

    @property
    def rate(self):
        return self.A / self.E

    def encode(self, a):
        """a: (B, A) uint8 payload -> (B, E) uint8 rate-matched coded bits."""
        a = np.asarray(a, dtype=np.uint8)
        c = np.concatenate([a, crc_parity(a, self.Lcrc)], axis=1)
        u = np.zeros((a.shape[0], self.N), dtype=np.uint8)
        u[:, self.info_pos] = c
        return polar_transform(u)[:, self.idx_rm]

    def decode(self, llr, list_size=None):
        """llr: (B, E) with LLR = log P(b=0)/P(b=1). Returns (a_hat (B, A), crc_ok (B,))."""
        llr = np.asarray(llr, dtype=np.float64)
        Lc = np.zeros((llr.shape[0], self.N))
        for k in range(self.E):                                   # rate recovery (repetition -> sum)
            Lc[:, self.idx_rm[k]] += llr[:, k]
        Lc[:, self.shortened] = 1e3                               # shortened bits are known zeros
        L = self.list_size if list_size is None else list_size
        c_hat, ok = _scl_batch(Lc, self.frozen, self.info_pos, L, self.crc_g, self.A, self.n)
        return c_hat[:, :self.A], ok


@njit(cache=_NB_CACHE)
def _crc_ok(c, A, g):
    """c: K = A + Lc bits; recompute parity of c[:A] with g (MSB first) and compare."""
    Lc = g.size - 1
    reg = np.zeros(Lc, dtype=np.uint8)
    for i in range(A):
        fb = reg[0] ^ c[i]
        for j in range(Lc - 1):
            reg[j] = reg[j + 1] ^ (fb & g[j + 1])
        reg[Lc - 1] = fb & g[Lc]
    for j in range(Lc):
        if reg[j] != c[A + j]:
            return False
    return True


@njit(cache=_NB_CACHE)
def _scl_one(llr, frozen, info_pos, L, g, A, n):
    N = llr.size
    alpha = np.zeros((L, n + 1, N))
    bl = np.zeros((L, n + 1, N), dtype=np.uint8)
    uh = np.zeros((L, N), dtype=np.uint8)
    PM = np.zeros(L)
    tmp = np.zeros(N, dtype=np.uint8)
    tmp2 = np.zeros(N, dtype=np.uint8)
    nP = 1
    cand_pm = np.zeros(2 * L)
    keep = np.zeros(2 * L, dtype=np.bool_)
    newbit = np.zeros(L, dtype=np.uint8)
    for i in range(N):
        # ---- LLRs down to leaf i for every active path
        if i == 0:
            d0 = 1
        else:
            tz = 0
            v = i
            while (v & 1) == 0:
                v >>= 1
                tz += 1
            d0 = n - tz
        for p in range(nP):
            for d in range(d0, n + 1):
                m = N >> d
                if d == d0 and i > 0:                          # g at the right child
                    for j in range(m):
                        a = llr[j] if d == 1 else alpha[p, d - 1, j]
                        b = llr[j + m] if d == 1 else alpha[p, d - 1, j + m]
                        alpha[p, d, j] = b + a if bl[p, d, j] == 0 else b - a
                else:                                          # f (min-sum) at the left child
                    for j in range(m):
                        a = llr[j] if d == 1 else alpha[p, d - 1, j]
                        b = llr[j + m] if d == 1 else alpha[p, d - 1, j + m]
                        s = 1.0 if (a >= 0) == (b >= 0) else -1.0
                        alpha[p, d, j] = s * min(abs(a), abs(b))
        # ---- decisions
        if frozen[i]:
            for p in range(nP):
                lam = alpha[p, n, 0] if n > 0 else llr[0]
                if lam < 0:
                    PM[p] -= lam
                newbit[p] = 0
        else:
            for p in range(nP):
                lam = alpha[p, n, 0] if n > 0 else llr[0]
                cand_pm[2 * p] = PM[p] + (-lam if lam < 0 else 0.0)
                cand_pm[2 * p + 1] = PM[p] + (lam if lam > 0 else 0.0)
            nc = 2 * nP
            for c in range(nc):
                keep[c] = False
            if nc <= L:
                for c in range(nc):
                    keep[c] = True
            else:
                order = np.argsort(cand_pm[:nc])
                for r in range(L):
                    keep[order[r]] = True
            # free slots: paths with no surviving child, and unused slots
            free = np.empty(L, dtype=np.int64)
            nf = 0
            for p in range(nP):
                if not keep[2 * p] and not keep[2 * p + 1]:
                    free[nf] = p
                    nf += 1
            for p in range(nP, L):
                free[nf] = p
                nf += 1
            fi = 0
            newP = 0
            alive = np.zeros(L, dtype=np.bool_)
            pm_new = np.zeros(L)
            for p in range(nP):
                k0, k1 = keep[2 * p], keep[2 * p + 1]
                if k0 and k1:
                    q = free[fi]
                    fi += 1
                    for d in range(1, n + 1):
                        m = N >> d
                        for j in range(m):
                            alpha[q, d, j] = alpha[p, d, j]
                            bl[q, d, j] = bl[p, d, j]
                    for j in range(i):
                        uh[q, j] = uh[p, j]
                    newbit[p] = 0
                    pm_new[p] = cand_pm[2 * p]
                    alive[p] = True
                    newbit[q] = 1
                    pm_new[q] = cand_pm[2 * p + 1]
                    alive[q] = True
                elif k0 or k1:
                    newbit[p] = 0 if k0 else 1
                    pm_new[p] = cand_pm[2 * p] if k0 else cand_pm[2 * p + 1]
                    alive[p] = True
            # compact alive paths to indices 0..newP-1
            for p in range(L):
                if alive[p]:
                    if p != newP:
                        for d in range(1, n + 1):
                            m = N >> d
                            for j in range(m):
                                alpha[newP, d, j] = alpha[p, d, j]
                                bl[newP, d, j] = bl[p, d, j]
                        for j in range(i):
                            uh[newP, j] = uh[p, j]
                        newbit[newP] = newbit[p]
                    PM[newP] = pm_new[p]
                    newP += 1
            nP = newP
        # ---- partial sums
        for p in range(nP):
            b = newbit[p]
            uh[p, i] = b
            tmp[0] = b
            ln = 1
            j = i
            d = n
            while d > 0 and (j & 1) == 1:
                for t in range(ln):
                    tmp2[t] = bl[p, d, t] ^ tmp[t]
                    tmp2[t + ln] = tmp[t]
                ln *= 2
                for t in range(ln):
                    tmp[t] = tmp2[t]
                d -= 1
                j >>= 1
            if d > 0:
                for t in range(ln):
                    bl[p, d, t] = tmp[t]
    # ---- final selection (CRC-aided)
    order = np.argsort(PM[:nP])
    K = info_pos.size
    c = np.zeros(K, dtype=np.uint8)
    for r in range(nP):
        p = order[r]
        for k in range(K):
            c[k] = uh[p, info_pos[k]]
        if _crc_ok(c, A, g):
            return c, True
    p = order[0]
    for k in range(K):
        c[k] = uh[p, info_pos[k]]
    return c, False


@njit(parallel=True, cache=_NB_CACHE)
def _scl_batch(llr, frozen, info_pos, L, g, A, n):
    B = llr.shape[0]
    out = np.zeros((B, info_pos.size), dtype=np.uint8)
    ok = np.zeros(B, dtype=np.bool_)
    for b in prange(B):
        c, f = _scl_one(llr[b], frozen, info_pos, L, g, A, n)
        out[b] = c
        ok[b] = f
    return out, ok


# ============================================================
# Modulation (TS 38.211 Sec. 5.1), max-log demapping
# ============================================================
def _axis_level(c):
    """PAM level of one axis from its bits c (n, m) following TS 38.211 (unnormalized)."""
    m = c.shape[1]
    t = 1.0 - 2.0 * c[:, m - 1]
    for j in range(m - 2, -1, -1):
        t = (1.0 - 2.0 * c[:, j]) * (2 ** (m - 1 - j) - t)
    return t


def qam_map(bits, M):
    """bits (n*log2M,) -> unit-power symbols (n,). Bits b0, b2, ... on I and b1, b3, ... on Q."""
    k = int(np.log2(M))
    b = np.asarray(bits, dtype=np.float64).reshape(-1, k)
    return (_axis_level(b[:, 0::2]) + 1j * _axis_level(b[:, 1::2])) / np.sqrt(2 * (M - 1) / 3)


_AXIS_TAB = {}


def _axis_table(M):
    if M not in _AXIS_TAB:
        m = int(np.log2(M)) // 2
        lab = (np.arange(2 ** m)[:, None] >> np.arange(m - 1, -1, -1)[None, :]) & 1
        _AXIS_TAB[M] = (_axis_level(lab.astype(float)) / np.sqrt(2 * (M - 1) / 3), lab)
    return _AXIS_TAB[M]


def qam_llr(Y, M, N0):
    """Max-log LLR = log P(b=0)/P(b=1); Y (n,), N0 (n,) complex noise variance. Returns (n*log2M,)."""
    lv, lab = _axis_table(M)
    k, m = int(np.log2(M)), int(np.log2(M)) // 2
    out = np.empty((Y.size, k))
    for ax, y in ((0, Y.real), (1, Y.imag)):
        d2 = (y[:, None] - lv[None, :]) ** 2
        for j in range(m):
            out[:, ax + 2 * j] = (d2[:, lab[:, j] == 1].min(1) - d2[:, lab[:, j] == 0].min(1)) / N0
    return out.ravel()


def qam_theory_ber(M, ebn0_db):
    """AWGN BER of Gray-coded square M-QAM vs Eb/N0 per coded bit."""
    g = 10 ** (np.asarray(ebn0_db, float) / 10)
    k = np.log2(M)
    if M == 4:
        return 0.5 * erfc(np.sqrt(g))
    return (2 / k) * (1 - 1 / np.sqrt(M)) * erfc(np.sqrt(3 * k * g / (2 * (M - 1))))


# ============================================================
# Performance prediction from the residual image-to-signal ratio (ISR)
# ============================================================
# Received symbol after desired-gain equalization: Y(k) = X(k) + rho X*(-k) + noise, |rho|^2 = ISR. The image
# is an independent data symbol of the same power, so it is treated as Gaussian interference:
#     SINR = |H|^2 / ((1 + ISR) / (Es/N0) + |H|^2 ISR)          (AWGN: |H|^2 = 1)
# The factor (1 + ISR) appears because Es/N0 is defined on the total transmit power (signal + image), and
# the image fades with the signal because the I/Q imbalance is at the transmitter.
def qam_ber_exact(M, snr):
    """Exact hard-decision BER of Gray-coded square M-QAM in Gaussian noise vs Es/N0 (linear)."""
    snr = np.asarray(snr, float)
    m = int(np.log2(M)) // 2
    L = 2 ** m
    a = np.sqrt(3 * snr / (2 * (M - 1)))
    pb = np.zeros_like(snr)
    for j in range(1, m + 1):
        for i in range(int((1 - 2.0 ** -j) * L)):
            w = (-1) ** (i * 2 ** (j - 1) // L) * (2 ** (j - 1) - int(i * 2 ** (j - 1) / L + 0.5))
            pb += w * erfc((2 * i + 1) * a)
    return pb / (L * m)


_PRED_DB = np.arange(-30.0, 60.01, 0.25)                 # Es/N0 grid of the mutual-information tables [dB]
_tq, _wq = np.polynomial.legendre.leggauss(96)           # Rayleigh average: |H|^2 = t^2, t in [0, 6]
_H2_Q = (3 * (_tq + 1)) ** 2
_H2_W = 3 * _wq * 2 * (3 * (_tq + 1)) * np.exp(-_H2_Q)
_H2_W /= _H2_W.sum()
_MI_TAB = {}


def bicm_mi_table(M):
    """BICM mutual information [bits/symbol] of M-QAM (TS 38.211 labels) in AWGN on the grid _PRED_DB."""
    if M not in _MI_TAB:
        lv, lab = _axis_table(M)
        z, wz = np.polynomial.hermite_e.hermegauss(64)
        wz = wz / wz.sum()
        mi = np.zeros(_PRED_DB.size)
        for n, sdb in enumerate(_PRED_DB):
            sg = np.sqrt(0.5 / 10 ** (sdb / 10))                         # noise std per real dimension
            y = lv[:, None] + sg * z[None, :]                            # (levels, nodes)
            ll = -(y[:, :, None] - lv[None, None, :]) ** 2 / (2 * sg * sg)
            for j in range(lab.shape[1]):
                l0 = np.logaddexp.reduce(ll[:, :, lab[:, j] == 0], axis=2)
                l1 = np.logaddexp.reduce(ll[:, :, lab[:, j] == 1], axis=2)
                d = np.where(lab[:, j][:, None] == 0, l1 - l0, l0 - l1)   # log P(wrong) / P(right)
                mi[n] += 2 * (1 - np.mean(np.logaddexp(0, d) @ wz) / np.log(2))
        _MI_TAB[M] = mi
    return _MI_TAB[M]


def avg_mi(M, sinr):
    """BICM mutual information at the given SINR (linear, any shape)."""
    return np.interp(10 * np.log10(np.maximum(sinr, 1e-12)), _PRED_DB, bicm_mi_table(M))


def predict_from_isr(M, fading, esn0_db, isr):
    """Uncoded BER and MI-equivalent Es/N0 [dB] for every slot.
    esn0_db (n_snr,), isr (n_slot,) linear -> raw_ber (n_slot, n_snr), esn0_eq_db (n_slot, n_snr)."""
    g = 10 ** (np.asarray(esn0_db, float) / 10)[None, :, None]
    r = np.asarray(isr, float)[:, None, None]
    h2, wq = (_H2_Q, _H2_W) if fading else (np.ones(1), np.ones(1))
    sinr = h2[None, None, :] / ((1 + r) / g + h2[None, None, :] * r)
    raw = qam_ber_exact(M, sinr) @ wq
    mi = avg_mi(M, sinr) @ wq
    ref = avg_mi(M, 10 ** (_PRED_DB / 10)[:, None] * h2[None, :]) @ wq   # 'ideal' MI vs Es/N0
    ref = ref + 1e-12 * np.arange(ref.size)                              # strictly increasing for the inversion
    return raw, np.interp(mi, ref, _PRED_DB)


def read_ref_curve(x_ref, y_ref, n_err, x, cap):
    """Reference curve y_ref(x_ref) read at x (any shape), log-linear inter/extrapolation.
    Only the points with observed errors are used; returns (values, smallest supported value)."""
    v = (np.asarray(n_err) > 0) & (np.asarray(y_ref) > 0)
    if v.sum() < 2:
        return np.full(np.shape(x), np.nan), np.nan
    xv, lv = np.asarray(x_ref)[v], np.log10(np.asarray(y_ref)[v])
    out = np.interp(x, xv, lv)
    lo, hi = x < xv[0], x > xv[-1]
    out = np.where(lo, lv[0] + (x - xv[0]) * (lv[1] - lv[0]) / (xv[1] - xv[0]), out)
    out = np.where(hi, lv[-1] + (x - xv[-1]) * (lv[-1] - lv[-2]) / (xv[-1] - xv[-2]), out)
    return 10 ** np.minimum(out, np.log10(cap)), 10 ** lv.min()


def random_slot(M):
    """Random M-QAM slot (SYMB, N) for the data-aided estimators."""
    return qam_map(np.random.randint(0, 2, SLOT * int(np.log2(M))), M).reshape(SYMB, N)


# ============================================================
# Carrier-level chain
# ============================================================
def cos_sampling(Xs, F, C, upsilon, theta_deg=0.0):
    t = np.arange(Xs)
    return ((1 + upsilon / 2) * np.cos(2 * F * np.pi * t / Xs + np.pi * C / 360 + np.deg2rad(theta_deg))).reshape(1, -1)


def sin_sampling(Xs, F, C, upsilon, theta_deg=0.0):
    t = np.arange(Xs)
    return ((1 - upsilon / 2) * np.sin(2 * F * np.pi * t / Xs - np.pi * C / 360 + np.deg2rad(theta_deg))).reshape(1, -1)


def modulate(I, Q, eps, phi, theta_deg=0.0):
    """Baseband (column vectors) -> RF samples (n, X) with I/Q imbalance and extra RF phase."""
    return np.dot(I, cos_sampling(X, Fq, phi, eps, theta_deg)) + np.dot(Q, sin_sampling(X, Fq, phi, eps, theta_deg))


def demodulate(rf, eps, phi):
    """RF samples (n, X) -> baseband complex (n,) with I/Q imbalance."""
    return (np.dot(rf, cos_sampling(X, Fq, phi, eps).T) * 2 / X +
            1j * np.dot(rf, sin_sampling(X, Fq, phi, eps).T) * 2 / X).ravel()


def iq_predistort(I, Q, epsilon, phi):
    """Baseband predistortion, inverse of M(eps, phi) (eq. (14))."""
    h = np.deg2rad(phi) / 2.0
    c, s, cp = np.cos(h), np.sin(h), np.cos(2.0 * h)
    return (c * I + s * Q) / ((1.0 + epsilon / 2.0) * cp), (s * I + c * Q) / ((1.0 - epsilon / 2.0) * cp)


def feedback(ps, eT, pT, eR, pR, g, theta0, psi=0.0, gs=1.0, snr_db=np.inf):
    """Feedback of the modulator input ps (complex): TX (eT, pT) -> common phase theta0 + psi,
    gain g * gs -> FB demodulator (eR, pR) -> AWGN at snr_db (relative to the feedback power)."""
    ps = np.asarray(ps).reshape(-1, 1)
    fb = demodulate(g * gs * modulate(np.real(ps), np.imag(ps), eT, pT, theta0 + psi), eR, pR)
    if np.isfinite(snr_db):
        s2 = np.mean(np.abs(fb) ** 2) / 10 ** (snr_db / 10)
        fb = fb + np.sqrt(s2 / 2) * (np.random.randn(fb.size) + 1j * np.random.randn(fb.size))
    return fb


def draw_feedback_path():
    th0 = np.random.uniform(0, 360) if THETA0_DEG is None else THETA0_DEG
    return np.random.uniform(*EPS_R_RANGE), np.random.uniform(*PHI_R_RANGE), th0


def probe_matrix(fun):
    """2x2 real matrix of a real-linear map (I, Q) -> complex output, by probing unit inputs."""
    cols = []
    for x in (1.0, 1j):
        y = np.asarray(fun(x)).ravel()[0]
        cols.append([y.real, y.imag])
    return np.array(cols).T


def tx_matrix(eT, pT):
    return probe_matrix(lambda x: demodulate(modulate(np.array([[x.real]]), np.array([[x.imag]]), eT, pT), 0, 0))


def pd_matrix(e, p):
    def f(x):
        I, Q = iq_predistort(x.real, x.imag, e, p)
        return np.array([I + 1j * Q])
    return probe_matrix(f)


def widely_linear(T):
    """2x2 real T acting on [I; Q] -> (mu, nu) with y = mu x + nu x*."""
    return (T[0, 0] + T[1, 1]) / 2 + 1j * (T[1, 0] - T[0, 1]) / 2, (T[0, 0] - T[1, 1]) / 2 + 1j * (T[1, 0] + T[0, 1]) / 2


# ============================================================
# Estimators
# ============================================================
def fit2x2(ref, fb):
    """LS observation matrix per state (7 accumulated sums): [I~; Q~] = A [I; Q]."""
    Z = np.c_[np.real(ref).ravel(), np.imag(ref).ravel()]
    return np.linalg.lstsq(Z, np.c_[np.real(fb).ravel(), np.imag(fb).ravel()], rcond=None)[0].T


def est_proposed(A0, A1):
    """Proposed: B = A0^-1 A1 = kappa M^-1 R(psi) M. det -> kappa, trace -> cos(psi),
    C = (B/kappa - cos(psi) I)/sin(psi), sign of sin(psi) from C12 > 0. Returns (eps, phi[deg], psi_hat[deg])."""
    B = np.linalg.solve(A0, A1)
    dB = np.linalg.det(B)
    if not np.isfinite(dB) or dB <= 0:
        return np.nan, np.nan, np.nan
    B = B / np.sqrt(dB)
    cpsi = np.trace(B) / 2
    if abs(cpsi) >= 1:
        return np.nan, np.nan, np.nan
    spsi = np.sign(B[0, 1]) * np.sqrt(1 - cpsi ** 2)
    C = (B - cpsi * np.eye(2)) / spsi
    r = np.sqrt(-C[1, 0] / C[0, 1])
    return 2 * (r - 1) / (r + 1), np.rad2deg(np.arctan(C[1, 1])), np.rad2deg(np.arctan2(spsi, cpsi))


def beta_to_param(beta, iters=30):
    """beta = nu/mu of the TX model y = mu x + nu x*  ->  (eps, phi[deg]); vectorized."""
    beta = np.asarray(beta, dtype=complex)
    e, t = np.zeros(beta.shape), np.zeros(beta.shape)
    for _ in range(iters):
        t = -beta.imag / (1 - beta.real * e / 2)
        e = 2 * beta.real / (1 - beta.imag * t)
    return e, np.rad2deg(2 * np.arctan(t))


def huang_training():
    """Unit-magnitude A(f) and B(f) = A(f) (f > 0), -A(f) (f < 0) (Huang eqs. (17)-(19))."""
    A = np.exp(1j * (np.pi / 4 + np.pi / 2 * np.random.randint(0, 4, N)))
    B = A.copy()
    B[N // 2 + 1:] = -A[N // 2 + 1:]
    return A, B


def est_huang(eT, pT, eR, pR, th0, psi_true, gs_true, fb_snr):
    """Huang et al. (TWC 2016): training A, B in direct and phase-shifted loopback, RX-image removal
    with the ASSUMED P (eq. (23)), beta_t(f) per subcarrier (eq. (25)). Returns beta (N,)."""
    A, B = huang_training()
    Z = {}
    for st, (ps, gs) in enumerate(((0.0, 1.0), (psi_true, gs_true))):
        for name, T in (('A', A), ('B', B)):
            acc = np.zeros(N, complex)
            x = np.fft.ifft(T)
            for _ in range(HUANG_TRAIN_REP):
                acc += np.fft.fft(feedback(x, eT, pT, eR, pR, FB_GAIN, th0, ps, gs, fb_snr))
            Z[name, st] = acc / HUANG_TRAIN_REP
    P = 10 ** (HUANG_P_GAIN_DB / 20) * np.exp(-1j * np.deg2rad(HUANG_P_PHASE_DEG))   # code convention: R(psi) <-> e^{-j psi}
    den = np.conj(P) - P
    ZA = (np.conj(P) * Z['A', 0] - Z['A', 1]) / den
    ZB = (np.conj(P) * Z['B', 0] - Z['B', 1]) / den
    beta = (ZA * A[MIRROR_IDX] + ZB * B[MIRROR_IDX]) / (ZA * np.conj(A) + ZB * np.conj(B))
    other = np.ones(N, bool)
    other[[0, N // 2]] = False                       # self-mirrored bins: eq. (19) does not hold
    beta[~other] = np.mean(beta[other])
    if not np.all(np.isfinite(beta)):
        beta = np.zeros(N, complex)
    return beta


def safe(e, p):
    """Out-of-domain or failed estimate -> no compensation."""
    if not (np.isfinite(e) and np.isfinite(p)) or abs(e) >= 2 or abs(p) >= 90:
        return (0.0, 0.0), True
    return (e, p), False


def measure_and_estimate(methods, eT, pT, fb_snr, psi_err, gain_db, M_ref):
    """One estimation interval. Returns est (dict), fail (dict), aux (dict)."""
    eR, pR, th0 = draw_feedback_path()
    psi_true, gs_true = STATE_PSI_NOM + psi_err, 10 ** (gain_db / 20)
    est, fail, aux = {}, {}, {'eR': eR, 'pR': pR, 'th0': th0, 'psi': psi_true, 'gs': gs_true}
    with np.errstate(all='ignore'):
        if 'prop' in methods:
            r0 = np.fft.ifft(random_slot(M_ref)).ravel()
            r1 = np.fft.ifft(random_slot(M_ref)).ravel()
            f0 = feedback(r0, eT, pT, eR, pR, FB_GAIN, th0, 0.0, 1.0, fb_snr)
            f1 = feedback(r1, eT, pT, eR, pR, FB_GAIN, th0, psi_true, gs_true, fb_snr)
            aux.update(r0=r0, r1=r1)
            e, p, aux['psi_hat'] = est_proposed(fit2x2(r0, f0), fit2x2(r1, f1))
            est['prop'], fail['prop'] = safe(e, p)
        if any(m in HUANG_SET for m in methods):
            beta = est_huang(eT, pT, eR, pR, th0, psi_true, gs_true, fb_snr)
            bad = not np.any(beta)
            if 'huang' in methods:
                est['huang'], fail['huang'] = beta, bad
            if 'huang_flat' in methods:        # flat model: one beta for all subcarriers
                est['huang_flat'], fail['huang_flat'] = np.full(N, np.mean(np.delete(beta, [0, N // 2]))), bad
    return est, fail, aux


def isr_of(method, est, eT, pT):
    """Exact residual image-to-signal ratio (linear) after compensation."""
    mu, nu = widely_linear(tx_matrix(eT, pT))
    if method in HUANG_SET:
        b = est
        return np.mean(np.abs(nu - mu * b) ** 2 / np.abs(mu - nu * np.conj(b[MIRROR_IDX])) ** 2)
    if method == 'uncomp':
        return abs(nu) ** 2 / abs(mu) ** 2
    mu, nu = widely_linear(tx_matrix(eT, pT) @ pd_matrix(*est))
    return abs(nu) ** 2 / abs(mu) ** 2


def param_errors(method, est, eT, pT):
    """Squared errors of (eps, phi[deg]); Huang: averaged over subcarriers of the per-subcarrier conversion."""
    if method in HUANG_SET:
        e, p = beta_to_param(np.delete(est, [0, N // 2]))
        return np.mean((e - eT) ** 2), np.mean((p - pT) ** 2)
    return (est[0] - eT) ** 2, (est[1] - pT) ** 2


def crb_two_state(eT, pT, aux, fb_snr):
    """CRB of (eps_T, phi_T[deg]) for the two-state data-aided model with the 8 unknowns
    (eT, pT, eR, pR, g, theta0, psi, kappa) and the actual data of both states."""
    th = np.array([eT, pT, aux['eR'], aux['pR'], FB_GAIN, aux['th0'], aux['psi'], aux['gs']])

    def A_of(t, state):
        ps, gs = (0.0, 1.0) if state == 0 else (t[6], t[7])
        return probe_matrix(lambda x: feedback(np.array([x]), t[0], t[1], t[2], t[3], t[4], t[5], ps, gs))
    F = np.zeros((8, 8))
    for st, r in ((0, aux['r0']), (1, aux['r1'])):
        Zr = np.c_[r.real, r.imag]
        S = Zr.T @ Zr
        A = A_of(th, st)
        s2 = np.mean(np.sum((Zr @ A.T) ** 2, 1)) / 10 ** (fb_snr / 10) / 2      # per real dimension
        J = np.zeros((2, 2, 8))
        for i in range(8):
            h = 1e-6 * max(1.0, abs(th[i]))
            tp, tm = th.copy(), th.copy()
            tp[i] += h
            tm[i] -= h
            J[:, :, i] = (A_of(tp, st) - A_of(tm, st)) / (2 * h)
        for row in range(2):
            F += J[row].T @ S @ J[row] / s2
    try:
        C = np.linalg.inv(F)
        return C[0, 0], C[1, 1]
    except np.linalg.LinAlgError:
        return np.nan, np.nan


# ============================================================
# Transmitter / receiver of the link
# ============================================================
def tx_signal(scheme, sp, eT, pT, est):
    """Slot sp (SYMB, N) -> real RF samples (1, SLOT*X) for each scheme."""
    if scheme in HUANG_SET:
        sp = sp - est[scheme][None, :] * np.conj(sp[:, MIRROR_IDX])
    ps = np.fft.ifft(sp).reshape(-1, 1)
    I, Q = np.real(ps), np.imag(ps)
    if scheme == 'prop':
        I, Q = iq_predistort(I, Q, *est[scheme])
    if scheme == 'ideal':
        return modulate(I, Q, 0, 0).reshape(1, -1)
    return modulate(I, Q, eT, pT).reshape(1, -1)


_DC, _DS = cos_sampling(X, Fq, 0, 0), sin_sampling(X, Fq, 0, 0)


def rx_chain(rev):
    rx_iq = rev.reshape(-1, X)
    y = np.dot(rx_iq, _DC.T).ravel() * 2 / X + 1j * np.dot(rx_iq, _DS.T).ravel() * 2 / X
    return np.fft.fft(y.reshape(-1, N), N)


def desired_gain(scheme, est, eT, pT):
    """End-to-end gain of the desired signal per subcarrier, c(k) in Y(k) = c(k) X(k) + d(k) X*(-k)."""
    if scheme == 'ideal' or not RX_GAIN_EQ:
        return np.ones(N, complex)
    mu, nu = widely_linear(tx_matrix(eT, pT))
    if scheme == 'uncomp':
        return np.full(N, mu, complex)
    if scheme in HUANG_SET:
        return mu - nu * np.conj(est[scheme][MIRROR_IDX])
    return np.full(N, widely_linear(tx_matrix(eT, pT) @ pd_matrix(*est[scheme]))[0], complex)


def residual_evm(scheme, est, eT, pT, M):
    """Noise-free EVM [%] after compensation (no channel), after desired-gain equalization."""
    sp = random_slot(M)
    Y = rx_chain(tx_signal(scheme, sp, eT, pT, est)) / desired_gain(scheme, est, eT, pT)[None, :]
    return 100 * np.sqrt(np.mean(np.abs(Y - sp) ** 2) / np.mean(np.abs(sp) ** 2))


# ============================================================
# Output directory
# ============================================================
def resolve_out_dir():
    if SAVE_TO_DRIVE:
        try:
            from google.colab import drive
            if not os.path.isdir('/content/drive/MyDrive'):
                drive.mount('/content/drive')
            os.makedirs(DRIVE_DIR, exist_ok=True)
            return DRIVE_DIR
        except Exception:
            pass
    os.makedirs(LOCAL_DIR, exist_ok=True)
    return LOCAL_DIR


# ============================================================
# Sanity checks
# ============================================================
def sanity_checks(code):
    print('[check] polar code: A=%d, K=%d, E=%d, N=%d, %s, rate %.3f, list %d'
          % (code.A, code.K, code.E, code.N, code.mode, code.rate, code.list_size))
    a = np.random.randint(0, 2, (64, code.A)).astype(np.uint8)
    ah, ok = code.decode(20.0 * (1.0 - 2.0 * code.encode(a).astype(float)))
    print('[check] polar noiseless decode:', bool(np.array_equal(ah, a) and ok.all()))
    for M in (4, 16, 64):
        b = np.random.randint(0, 2, 6000 * int(np.log2(M)))
        sy = qam_map(b, M)
        llr = qam_llr(sy, M, np.full(sy.size, 0.01))
        print(f'[check] {M}-QAM: mean |s|^2 = {np.mean(np.abs(sy) ** 2):.3f}, '
              f'noiseless demap errors = {int(np.sum((llr < 0) != b))}')
    w = rx_chain(np.random.randn(1, SLOT * X))
    print(f'[check] noise variance per subcarrier: measured {np.mean(np.abs(w) ** 2):.1f}, N0_UNIT {N0_UNIT:.1f}')
    eT, pT = 0.3, 4.0
    methods = ['prop', 'huang', 'huang_flat']
    est, fail, aux = measure_and_estimate(methods, eT, pT, np.inf, 7.0, 0.9, 16)
    e_h, p_h = beta_to_param(est['huang_flat'][1])
    print(f'[check] noiseless, TX ({eT}, {pT}), state error 7 deg / 0.9 dB, Huang assumes '
          f'{HUANG_P_PHASE_DEG:g} deg / {HUANG_P_GAIN_DB:g} dB:')
    print(f'        prop {np.round(est["prop"], 6)} (psi_hat {aux["psi_hat"]:.3f} deg, true {aux["psi"]:.1f}) | '
          f'huang_flat ({e_h:.4f}, {p_h:.4f})')
    _ps = (HUANG_P_PHASE_DEG, HUANG_P_GAIN_DB)
    globals()['HUANG_P_PHASE_DEG'], globals()['HUANG_P_GAIN_DB'] = STATE_PSI_NOM + 7.0, 0.9
    est2, _, _ = measure_and_estimate(['huang'], eT, pT, np.inf, 7.0, 0.9, 16)
    globals()['HUANG_P_PHASE_DEG'], globals()['HUANG_P_GAIN_DB'] = _ps
    e_h, p_h = beta_to_param(np.delete(est2['huang'], [0, N // 2]))
    print(f'        huang with the true P: ({np.mean(e_h):.6f}, {np.mean(p_h):.6f}), '
          f'ISR {10 * np.log10(isr_of("huang", est2["huang"], eT, pT)):.1f} dB')
    print('[check] residual ISR [dB]: ' + ', '.join(
        f'{m} {10 * np.log10(isr_of(m, est[m], eT, pT) + 1e-30):.1f}' for m in methods) +
        f', uncomp {10 * np.log10(isr_of("uncomp", None, eT, pT)):.1f}')


# ============================================================
# Experiment 1: estimation accuracy
# ============================================================
def mse_point(fb_snr, psi_err, gain_db, trials):
    acc = {m: np.zeros(6) for m in EST_METHODS}   # se_eps, se_phi, isr, evm, fail, se_psi
    crb = np.zeros(2)
    isr_u = 0.0
    for _ in range(trials):
        eT, pT = np.random.uniform(*EPS_RANGE), np.random.uniform(*PHI_RANGE)
        est, fail, aux = measure_and_estimate(EST_METHODS, eT, pT, fb_snr, psi_err, gain_db, MSE_MOD)
        for m in EST_METHODS:
            se_e, se_p = param_errors(m, est[m], eT, pT)
            acc[m][:5] += [se_e, se_p, isr_of(m, est[m], eT, pT), residual_evm(m, est, eT, pT, MSE_MOD), fail[m]]
            if m == 'prop' and np.isfinite(aux.get('psi_hat', np.nan)):
                acc[m][5] += (aux['psi_hat'] - aux['psi']) ** 2
        if 'r0' in aux:
            crb += crb_two_state(eT, pT, aux, fb_snr)
        isr_u += isr_of('uncomp', None, eT, pT)
    out = {m: acc[m] / trials for m in EST_METHODS}
    return out, crb / trials, isr_u / trials


def run_mse(out_dir):
    err_snrs = list(np.atleast_1d(MSE_ERR_SNR))
    xl_p, xl_g = r'State phase error $\psi-90^\circ$ [deg]', r'State gain ratio $\kappa$ [dB]'
    sweeps = [('snr', MSE_FB_SNR, lambda v: (v, STATE_PSI_ERR, STATE_GAIN_DB), 'Feedback SNR [dB]')]
    for es in err_snrs:
        sweeps.append((f'perr{es:g}', MSE_PERR_LIST, lambda v, es=es: (es, v, 0.0), xl_p))
    for es in err_snrs:
        sweeps.append((f'gerr{es:g}', MSE_GERR_LIST, lambda v, es=es: (es, 0.0, v), xl_g))
    save = {'methods': np.array(EST_METHODS, dtype=object), 'trials': MSE_TRIALS, 'mse_mod': MSE_MOD,
            'state_psi_nom': STATE_PSI_NOM, 'state_psi_err': STATE_PSI_ERR, 'state_gain_db': STATE_GAIN_DB,
            'huang_p_phase_deg': HUANG_P_PHASE_DEG, 'huang_p_gain_db': HUANG_P_GAIN_DB,
            'huang_train_rep': HUANG_TRAIN_REP, 'mse_err_snr': np.array(err_snrs, float), 'fb_gain': FB_GAIN}
    res = {}
    for tag, xs, fn, _ in sweeps:
        rows, crbs, isru = [], [], []
        t0 = time.time()
        for v in xs:
            o, c, iu = mse_point(*fn(v), MSE_TRIALS)
            rows.append(np.array([o[m] for m in EST_METHODS]))
            crbs.append(c)
            isru.append(iu)
            print(f'[MSE] {tag} = {v}: ' + ' | '.join(
                f'{m} {o[m][0]:.2e}/{o[m][1]:.2e} ISR {10 * np.log10(o[m][2]):.1f} dB'
                for m in EST_METHODS) + f' | CRB {c[0]:.2e}/{c[1]:.2e}  ({time.time() - t0:.0f} s)')
        arr = np.array(rows)                       # (n_x, n_methods, 6)
        res[tag] = (np.array(xs, float), arr, np.array(crbs), np.array(isru))
        save.update({f'{tag}_x': res[tag][0], f'{tag}_mse_eps': arr[:, :, 0], f'{tag}_mse_phi': arr[:, :, 1],
                     f'{tag}_isr_db': 10 * np.log10(arr[:, :, 2]), f'{tag}_res_evm': arr[:, :, 3],
                     f'{tag}_fail_rate': arr[:, :, 4], f'{tag}_crb_eps': res[tag][2][:, 0],
                     f'{tag}_crb_phi': res[tag][2][:, 1], f'{tag}_isr_uncomp_db': 10 * np.log10(res[tag][3])})
        if 'prop' in EST_METHODS:
            save[f'{tag}_mse_psi_prop'] = arr[:, EST_METHODS.index('prop'), 5]
        savemat(os.path.join(out_dir, 'two_state_polar_mse.mat'), save)

    # figure: column 1 = feedback SNR sweep, column 2 = phase error, column 3 = gain ratio;
    # in columns 2-3 the highest feedback SNR is drawn with solid lines and filled markers,
    # lower ones with dotted lines and open markers
    cols = [[('snr', None)], [(f'perr{es:g}', es) for es in err_snrs], [(f'gerr{es:g}', es) for es in err_snrs]]
    xlabels = ['Feedback SNR [dB]', xl_p, xl_g]
    fig, ax = plt.subplots(4, 3, figsize=(13, 11.5), squeeze=False)
    for c, group in enumerate(cols):
        for tag, es in group:
            x, arr, crb, isru = res[tag]
            main = es is None or es == max(err_snrs)
            suffix = '' if es is None or len(err_snrs) == 1 else f', {es:g} dB'
            for r, (k, yl, logy) in enumerate([(0, r'MSE of $\varepsilon_T$', True),
                                               (1, r'MSE of $\varphi_T$ [deg$^2$]', True),
                                               (2, 'Residual ISR [dB]', False), (3, 'Residual EVM [%]', True)]):
                a = ax[r, c]
                draw = a.semilogy if logy else a.plot
                for j, m in enumerate(EST_METHODS):
                    yv = 10 * np.log10(arr[:, j, k]) if k == 2 else arr[:, j, k]
                    st = STYLE[m]
                    draw(x, yv, color=st[0], marker=st[-1], ls='-' if main else ':', ms=3.5, lw=1,
                         mfc=st[0] if main else 'none', markevery=2, label=LABELS[m] + suffix)
                if k in (0, 1) and np.all(np.isfinite(crb[:, k])):
                    a.semilogy(x, crb[:, k], color='k', ls='--' if main else '-.', lw=1, label='CRB' + suffix)
                if k == 2 and main:
                    a.plot(x, 10 * np.log10(isru), 'c--', lw=1, label='Uncompensated')
                a.set_xlabel(xlabels[c], fontsize=8)
                a.set_ylabel(yl, fontsize=8)
                a.grid(True, which='both', ls=':', lw=.5)
                a.tick_params(labelsize=7)
    ax[0, 0].legend(fontsize=6)
    ax[2, 0].legend(fontsize=6)
    fig.suptitle(f'Huang assumes P = {HUANG_P_PHASE_DEG:g} deg / {HUANG_P_GAIN_DB:g} dB; '
                 f'col. 1: state mismatch {STATE_PSI_ERR:g} deg / {STATE_GAIN_DB:g} dB; '
                 f'col. 2-3: feedback SNR ' + ', '.join(f'{e:g}' for e in err_snrs) + ' dB', fontsize=9)
    h_, l_ = ax[0, 1].get_legend_handles_labels()
    fig.legend(h_, l_, loc='lower center', ncol=4, fontsize=7, title='Columns 2-3', title_fontsize=7)
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(out_dir, f'two_state_polar_mse.{ext}'), dpi=160)
    return res


# ============================================================
# Experiment 2: coded link (BLER / BER / EVM / throughput)
# ============================================================
def run_link(out_dir, code):
    est_methods = [m for m in EST_METHODS if m in LINK_SCHEMES]
    for m in LINK_SCHEMES:
        if m in ('prop',) + HUANG_SET and m not in est_methods:
            est_methods.append(m)
    save = {'schemes': np.array(LINK_SCHEMES, dtype=object), 'mod_list': np.array(MOD_LIST),
            'channels': np.array(CHANNELS, dtype=object), 'fb_snr_link': FB_SNR_LINK,
            'state_psi_nom': STATE_PSI_NOM, 'state_psi_err': STATE_PSI_ERR, 'state_gain_db': STATE_GAIN_DB,
            'huang_p_phase_deg': HUANG_P_PHASE_DEG, 'huang_p_gain_db': HUANG_P_GAIN_DB,
            'polar_A': code.A, 'polar_K': code.K, 'polar_E': code.E, 'polar_N': code.N,
            'polar_mode': code.mode, 'polar_rate': code.rate, 'polar_list': code.list_size,
            'target_block_err': TARGET_BLOCK_ERR, 'min_frames': MIN_FRAMES, 'rx_gain_eq': RX_GAIN_EQ}
    R = code.rate
    for M in MOD_LIST:
        k = int(np.log2(M))
        n_cw = SLOT * k // code.E
        for ch_name in CHANNELS:
            tag = f'{M}QAM_{ch_name}'
            lo, hi = EBN0_RANGE[(M, ch_name)]
            ebn0 = np.arange(lo, hi + EBN0_STEP / 2, EBN0_STEP)
            fading = ch_name.lower().startswith('rayleigh')
            S = {s: {key: np.zeros(ebn0.size) for key in
                     ('bit_err', 'blk_err', 'n_cw', 'raw_err', 'err_pow')} for s in LINK_SCHEMES}
            active = {s: np.ones(ebn0.size, bool) for s in LINK_SCHEMES}
            n_raw, frames = 0, 0
            ref_pow = np.zeros(ebn0.size)
            est_stat = {m: [] for m in est_methods}
            isr_slot = {s: [] for s in LINK_SCHEMES}       # residual ISR of every slot (prediction)
            t0 = time.time()

            def dump():
                d = {f'ebn0_{tag}': ebn0, f'esn0_{tag}': ebn0 + 10 * np.log10(R * k), f'frames_{tag}': frames,
                     f'max_frames_{tag}': max_frames,
                     f'theory_raw_ber_{tag}': qam_theory_ber(M, ebn0 + 10 * np.log10(R))}
                for s in LINK_SCHEMES:
                    ncw = np.maximum(S[s]['n_cw'], 1)
                    bler = S[s]['blk_err'] / ncw
                    d[f'bler_{s}_{tag}'] = bler
                    d[f'ber_{s}_{tag}'] = S[s]['bit_err'] / (ncw * code.A)
                    d[f'rawber_{s}_{tag}'] = S[s]['raw_err'] / max(n_raw, 1)
                    d[f'evm_{s}_{tag}'] = 100 * np.sqrt(S[s]['err_pow'] / np.maximum(ref_pow, 1e-30))
                    d[f'thr_{s}_{tag}'] = (1 - bler) * R * k
                    d[f'ncw_{s}_{tag}'] = S[s]['n_cw']
                    d[f'nblkerr_{s}_{tag}'] = S[s]['blk_err']
                for m in est_methods:
                    if est_stat[m]:
                        v = np.mean(est_stat[m], 0)
                        d[f'est_mse_eps_{m}_{tag}'], d[f'est_mse_phi_{m}_{tag}'] = v[0], v[1]
                        d[f'est_isr_db_{m}_{tag}'] = 10 * np.log10(v[2])
                if frames:                      # predictions from the per-slot ISR
                    esn0 = ebn0 + 10 * np.log10(R * k)
                    for s in LINK_SCHEMES:
                        isr = np.array(isr_slot[s])
                        raw, eq = predict_from_isr(M, fading, esn0, isr)
                        d[f'isr_db_{s}_{tag}'] = 10 * np.log10(max(isr.mean(), 1e-30)) if isr.mean() > 0 else -np.inf
                        d[f'rawber_pred_{s}_{tag}'] = raw.mean(0)
                        d[f'snrloss_pred_{s}_{tag}'] = (esn0[None, :] - eq).mean(0)
                        for key, cnt, cap in (('bler', 'blk_err', 1.0), ('ber', 'bit_err', 0.5)):
                            val, floor = read_ref_curve(esn0, d[f'{key}_ideal_{tag}'], S['ideal'][cnt], eq, cap)
                            val = val.mean(0)
                            d[f'{key}_pred_{s}_{tag}'] = np.where(val >= floor, val, np.nan)
                save.update(d)
                savemat(os.path.join(out_dir, 'two_state_polar_link.mat'), save)

            max_frames = MAX_FRAMES[M] if isinstance(MAX_FRAMES, dict) else MAX_FRAMES
            while frames < max_frames:
                eT, pT = np.random.uniform(*EPS_RANGE), np.random.uniform(*PHI_RANGE)
                est, fail, aux = measure_and_estimate(est_methods, eT, pT, FB_SNR_LINK, STATE_PSI_ERR,
                                                      STATE_GAIN_DB, M)
                for m in est_methods:
                    est_stat[m].append((*param_errors(m, est[m], eT, pT), isr_of(m, est[m], eT, pT)))
                for s in LINK_SCHEMES:
                    isr_slot[s].append(0.0 if s == 'ideal' else isr_of(s, est.get(s), eT, pT))
                a = np.random.randint(0, 2, (n_cw, code.A)).astype(np.uint8)
                cw = code.encode(a)
                bits = np.concatenate([cw.ravel(), np.random.randint(0, 2, SLOT * k - n_cw * code.E).astype(np.uint8)])
                sp = qam_map(bits, M).reshape(SYMB, N)
                sig, pwr, cg = {}, {}, {}
                for s in LINK_SCHEMES:
                    tx = tx_signal(s, sp, eT, pT, est)
                    pwr[s] = np.mean(tx ** 2)
                    cg[s] = desired_gain(s, est, eT, pT)[None, :]
                    sig[s] = rx_chain(tx) / cg[s]
                n_raw += bits.size
                for i, eb in enumerate(ebn0):
                    # unit noise per subcarrier (equivalent to white RF noise through rx_chain)
                    W = np.sqrt(N0_UNIT / 2) * (np.random.randn(SYMB, N) + 1j * np.random.randn(SYMB, N))
                    if fading:      # i.i.d. Rayleigh per resource element, perfect CSI, zero forcing
                        H = (np.random.randn(SYMB, N) + 1j * np.random.randn(SYMB, N)) / np.sqrt(2)
                        W = W / H
                        h2 = np.abs(H) ** 2
                    else:
                        h2 = np.ones((SYMB, N))
                    ref_pow[i] += np.sum(h2 * np.abs(sp) ** 2)
                    for s in LINK_SCHEMES:
                        sg2 = X * pwr[s] / (2 * R * k * 10 ** (eb / 10))
                        Y = sig[s] + np.sqrt(sg2) * W / cg[s]
                        # EVM referred to the receiver input (error weighted by |H|^2)
                        S[s]['err_pow'][i] += np.sum(h2 * np.abs(Y - sp) ** 2)
                        llr = qam_llr(Y.ravel(), M, (sg2 * N0_UNIT / (h2 * np.abs(cg[s]) ** 2)).ravel())
                        S[s]['raw_err'][i] += np.sum((llr < 0) != bits)
                        if active[s][i]:
                            ah, _ = code.decode(llr[:n_cw * code.E].reshape(n_cw, code.E))
                            e_cw = np.any(ah != a, axis=1)
                            S[s]['bit_err'][i] += np.sum(ah != a)
                            S[s]['blk_err'][i] += np.sum(e_cw)
                            S[s]['n_cw'][i] += n_cw
                            if frames + 1 >= MIN_FRAMES:
                                if S[s]['blk_err'][i] >= TARGET_BLOCK_ERR:
                                    active[s][i] = False
                                elif S[s]['blk_err'][i] == 0 and S[s]['n_cw'][i] >= ZERO_STOP_CW:
                                    active[s][i:] = False
                frames += 1
                if frames % SAVE_EVERY == 0:
                    dump()
                    print(f'[link {tag}] slot {frames}  ({time.time() - t0:.0f} s), active points: '
                          + ', '.join(f'{s} {int(active[s].sum())}' for s in LINK_SCHEMES))
                if frames >= MIN_FRAMES and not any(active[s].any() for s in LINK_SCHEMES):
                    break
            dump()
            print(f'[link {tag}] done: {frames} slots, {time.time() - t0:.0f} s')
            for m in est_methods:
                v = np.mean(est_stat[m], 0)
                print(f'    estimation {m:10s}: MSE eps {v[0]:.2e}, MSE phi {v[1]:.2e} deg^2, ISR {10 * np.log10(v[2]):.1f} dB')
            print('    predicted SNR loss vs ideal at the highest Eb/N0: ' + ', '.join(
                f'{s} {save[f"snrloss_pred_{s}_{tag}"][-1]:.3f} dB' for s in LINK_SCHEMES if s != 'ideal'))
            for i, eb in enumerate(ebn0):
                print(f'    Eb/N0 {eb:4.1f} dB: ' + ' | '.join(
                    f'{s} BLER {save[f"bler_{s}_{tag}"][i]:.2e} EVM {save[f"evm_{s}_{tag}"][i]:.2f}%'
                    for s in LINK_SCHEMES))

            fig, ax = plt.subplots(2, 3, figsize=(13, 7))
            for s in LINK_SCHEMES:
                kw = dict(ms=3.5, lw=1, label=LABELS[s])
                ax[0, 0].semilogy(ebn0, np.maximum(save[f'bler_{s}_{tag}'], 1e-6), STYLE[s], **kw)
                ax[0, 1].semilogy(ebn0, np.maximum(save[f'ber_{s}_{tag}'], 1e-8), STYLE[s], **kw)
                ax[0, 2].semilogy(ebn0, np.maximum(save[f'rawber_{s}_{tag}'], 1e-8), STYLE[s], **kw)
                ax[1, 0].plot(ebn0, save[f'evm_{s}_{tag}'], STYLE[s], **kw)
                ax[1, 1].plot(ebn0, save[f'thr_{s}_{tag}'], STYLE[s], **kw)
                c_ = STYLE[s][0]                 # predictions: dotted line of the same colour, no markers
                ax[0, 2].semilogy(ebn0, save[f'rawber_pred_{s}_{tag}'], c_ + ':', lw=1.6)
                if s != 'ideal':
                    ax[0, 0].semilogy(ebn0, save[f'bler_pred_{s}_{tag}'], c_ + ':', lw=1.6)
                    ax[0, 1].semilogy(ebn0, save[f'ber_pred_{s}_{tag}'], c_ + ':', lw=1.6)
            for a_, yl in zip(ax.ravel()[:5], ['BLER', 'Coded BER', 'Uncoded (raw) BER', 'EVM [%]',
                                              'Throughput [info bits / symbol]']):
                a_.set_xlabel('$E_b/N_0$ per information bit [dB]', fontsize=8)
                a_.set_ylabel(yl, fontsize=8)
                a_.grid(True, which='both', ls=':', lw=.5)
                a_.tick_params(labelsize=7)
            ax[0, 0].set_ylim(1e-4, 1.1)
            ax[0, 1].set_ylim(1e-6, 1)
            ax[0, 2].set_ylim(1e-6, 1)
            ax[1, 2].axis('off')
            h_, l_ = ax[0, 0].get_legend_handles_labels()
            h_.append(plt.Line2D([], [], color='gray', ls=':', lw=1.6))
            l_.append('Prediction from ISR (dotted, same colour)')
            ax[1, 2].legend(h_, l_, fontsize=8, loc='center')
            fig.suptitle(f'{M}-QAM, {ch_name}, polar (A={code.A}, E={code.E}, N={code.N}, CA-SCL L={code.list_size}), '
                         f'feedback SNR {FB_SNR_LINK:g} dB, state error {STATE_PSI_ERR:g} deg / {STATE_GAIN_DB:g} dB, '
                         f'Huang P = {HUANG_P_PHASE_DEG:g} deg / {HUANG_P_GAIN_DB:g} dB', fontsize=8)
            fig.tight_layout()
            for ext in ('png', 'pdf'):
                fig.savefig(os.path.join(out_dir, f'two_state_polar_link_{tag}.{ext}'), dpi=160)
            plt.close(fig)
    return save


# ============================================================
# Main
# ============================================================
def main():
    np.random.seed(SEED)
    out_dir = resolve_out_dir()
    print('[info] results ->', os.path.abspath(out_dir))
    code = NRPolarUCI(POLAR_A, POLAR_E, list_size=POLAR_LIST)
    sanity_checks(code)
    if RUN_MSE:
        run_mse(out_dir)
    if RUN_LINK:
        run_link(out_dir, code)
    plt.show()


if __name__ == '__main__':
    main()
