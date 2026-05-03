#%%
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import torch

# Import custom modules
from modulation import modulate_16qam, demodulate_16qam
from dbp import dbp_fullband_to_subband
from utils import rcosdesign
from fiber import fiber_prop_DP


# Main script to simulate optical WDM transmission system
# Set random seed for reproducibility
np.random.seed(42)

#%%
# ==========================================
# Parameters settings
# ==========================================

# Symbols setting
Nsym = 2**12             # number of QAM symbols per channel
M = 16                   # 16-QAM modulation
k = int(np.log2(M))      # bits per symbol
Rs = 32e9                # symbol rate [symbols/s]
Ts = 1 / Rs              # symbol duration [s]
rrc_rolloff = 0.20       # pulse roll-off for rough channel bandwidth estimate
B_ch = Rs * (1 + rrc_rolloff) # Single-channel occupied bandwidth [Hz]

# WDM layout
Nch = 1                  # number of channels (set >1 for WDM)
DeltaF = 50e9            # channel spacing [Hz]
ch_idx = np.arange(-(Nch-1)/2, (Nch-1)/2 + 1) # channel indices around center
margin = 0.1 * B_ch      # minimum margin in Hz
minDeltaF = B_ch + margin  
if DeltaF < minDeltaF:
    DeltaF = minDeltaF

# Total occupied extent [Hz]
B_wdm = (Nch - 1) * DeltaF + B_ch  

# Sampling setting
guard_factor = 2.0       # 1.0 means 50% guard band on each side of B_wdm 
fs_target = B_wdm * (1 + guard_factor) # required sampling frequency
sps = int(np.ceil(fs_target / Rs))     # samples per symbol
fs = sps * Rs            # actual sampling rate used [Hz]

# Ensure enough points and FFT efficiency (equivalent to nextpow2 in MATLAB)
Nt = 2 ** int(np.ceil(np.log2(Nsym * sps + 500)))
dt = 1 / fs              # interval length of time grid
t = np.arange(0, Nt).reshape(-1, 1) * dt # time grid (Nt x 1)
df = fs / Nt             # interval length of frequency grid
f = np.arange(-Nt/2, Nt/2).reshape(-1, 1) * df # frequency grid (fftshift ordering)

# Physical parameters
h_planck = 6.62607015e-34 # Planck constant [J*s] (renamed to avoid conflict with filter h)
lambda0 = 1550e-9         # reference wavelength [m]
c = 299792458             # speed of light [m/s]
nu0 = c / lambda0         # optical frequency

# Fiber parameters
L_span = 100e3           # fiber length per span [m]
Nspans = 1               # number of spans
alpha_dBpm = 0.2e-3      # loss [dB/m]
GroupRef = 1.47          # group refractive index
Dispersion = 16.5e-6     # Dispersion D in [s/m^2]
Dis_S = 0.08e3           # Dispersion slope [s/m^3]
n2 = 2.6e-20             # nonlinear index [m^2/W] 
Aeff = 80e-12            # effective area [m^2] 

# step-control scheme
method = 'constant'
if method == 'global_error':
    control_param = 1e-0
elif method == 'local_error':
    control_param = 1e-3
else:
    control_param = 1e3
dz = 2e3                 # backup SSF step size

# NLSE parameters
beta1 = GroupRef / c
beta2 = -Dispersion * lambda0**2 / (2 * np.pi * c)
beta3 = (lambda0**2 / (2 * np.pi * c))**2 * (Dis_S + 2 * Dispersion / lambda0)
gamma = 2 * np.pi * n2 / (lambda0 * Aeff)
pmd_coeff = 0 * 0.1e-12 / np.sqrt(1000)

# DSP parameters
eta = 0.0 * 1e-2
beta1_DSP = (1 + eta) * beta1
beta2_DSP = (1 + eta) * beta2
beta3_DSP = (1 + eta) * beta3
gamma_DSP = (1 + eta) * gamma

# Launch power per channel
Pdbm_ch = 0
PinW_ch = 10 ** ((Pdbm_ch - 30) / 10)

# Amplifier parameters
G_dB = alpha_dBpm * L_span
G_lin = 10 ** (G_dB / 10)
NF_dB = 5
nsp = 0.5 * 10 ** (NF_dB / 10)

#%%
# ==========================================
# TX: generate WDM composite signal
# ==========================================

# Pre-allocate combined fields for Nt samples (IMPORTANT: dtype=complex)
E_total = np.zeros((Nt, 2), dtype=complex)  
avg_pow_tx_up = np.zeros((2, Nch))

# Python lists acting as MATLAB cell arrays
tx_bits = [None] * Nch
tx_symbols = [None] * Nch
tx_wave = [None] * Nch
tx_start_idx = np.zeros(Nch, dtype=int)

# RRC pulse-shaping parameters
rrc_span = 8
rrc_taps = rcosdesign(rrc_rolloff, rrc_span, sps)
L_rrc = len(rrc_taps)
rrc_delay = (L_rrc - 1) // 2

for m in range(Nch):
    # --- Symbol and Waveform Generation ---
    bits_x = np.random.randint(0, 2, Nsym * k)
    symbols_x = modulate_16qam(bits_x, M)
    bits_y = np.random.randint(0, 2, Nsym * k)
    symbols_y = modulate_16qam(bits_y, M)

    # Upsample by inserting zeros (Ensure dtype=complex)
    up_len = Nsym * sps
    tx_up_z_x = np.zeros(up_len, dtype=complex)
    tx_up_z_x[0::sps] = symbols_x  # Python uses 0-based indexing
    
    tx_up_z_y = np.zeros(up_len, dtype=complex)
    tx_up_z_y[0::sps] = symbols_y

    # RRC Filter X & Y (mode='full' keeps all samples)
    tx_shaped_x = np.convolve(tx_up_z_x, rrc_taps, mode='full')
    tx_shaped_y = np.convolve(tx_up_z_y, rrc_taps, mode='full')
    
    # --- Power Normalization ---
    p_raw_x = np.mean(np.abs(tx_shaped_x)**2)
    p_raw_y = np.mean(np.abs(tx_shaped_y)**2)
    
    avg_pow_tx_up[0, m] = p_raw_x
    avg_pow_tx_up[1, m] = p_raw_y

    PinW_X = 0.5 * PinW_ch
    PinW_Y = 0.5 * PinW_ch

    tx_shaped_x = tx_shaped_x * np.sqrt(PinW_X / p_raw_x)
    tx_shaped_y = tx_shaped_y * np.sqrt(PinW_Y / p_raw_y)

    # --- Placement in Time Grid ---
    tx_up_block = np.zeros((Nt, 2), dtype=complex)
    L0 = len(tx_shaped_x)
    # Python is 0-indexed, no "+ 1" needed for center calculation
    start_tx = int(np.floor((Nt - L0) / 2)) 
    
    # Assign shaped signals
    tx_up_block[start_tx : start_tx + L0, 0] = tx_shaped_x  # X-pol
    tx_up_block[start_tx : start_tx + L0, 1] = tx_shaped_y  # Y-pol

    tx_wave[m] = tx_up_block
    tx_start_idx[m] = start_tx
    
    # Store bits/symbols (using dictionaries or lists for structure)
    tx_bits[m] = {'X': bits_x, 'Y': bits_y}
    tx_symbols[m] = {'X': symbols_x, 'Y': symbols_y}
    
    # --- Frequency Upconversion ---
    fch = ch_idx[m] * DeltaF
    # Carrier vector (Nt x 1)
    carrier = np.exp(1j * 2 * np.pi * fch * t) 
    
    # NumPy broadcasting automatically aligns (Nt, 2) * (Nt, 1)
    E_total = E_total + tx_up_block * carrier

print("Transmission generation complete. E_total shape:", E_total.shape)

#%%
# ==========================================
# Channel propagation: Nspans of SSF + EDFA
# ==========================================
E = E_total.copy()

for span in range(Nspans):
    # Propagate whole WDM composite field through fiber span using SSF
    E = fiber_prop_DP(E, L_span, alpha_dBpm, beta1, beta2, beta3, gamma, 
                      dz, fs, control_param, method, B_wdm, pmd_coeff)
    
    # Apply amplifier gain and add ASE noise
    ASE_PSD = nsp * h_planck * nu0 * (G_lin - 1) # W/Hz (Note: used renamed h_planck)
    # Generate complex Gaussian noise
    noise = np.sqrt(ASE_PSD * fs / 2) * (np.random.randn(Nt, 2) + 1j * np.random.randn(Nt, 2))
    noise = 0
    
    E = np.sqrt(G_lin) * E + noise

# plotting
# Visualization: Waveforms
plt.figure('Time-Domain Waveforms', figsize=(10, 8))

plt.subplot(4, 1, 1);plt.plot(t, np.abs(E_total[:, 0]));plt.title('Tx Waveform (X-Pol)')
plt.grid(True);plt.xlabel('Time (s)');plt.ylabel('|E|')

plt.subplot(4, 1, 2);plt.plot(t, np.abs(E_total[:, 1]));plt.title('Tx Waveform (Y-Pol)')
plt.grid(True);plt.xlabel('Time (s)');plt.ylabel('|E|')

plt.subplot(4, 1, 3);plt.plot(t, np.abs(E[:, 0]));plt.title('Rx Waveform (X-Pol)')
plt.grid(True);plt.xlabel('Time (s)');plt.ylabel('|E|')

plt.subplot(4, 1, 4);plt.plot(t, np.abs(E[:, 1]));plt.title('Rx Waveform (Y-Pol)')
plt.grid(True);plt.xlabel('Time (s)');plt.ylabel('|E|')

plt.tight_layout()

# Visualization: Spectra
plt.figure('Frequency Spectra', figsize=(10, 8))
# Compute spectra (magnitude)
Spec_Tx = np.fft.fftshift(np.abs(np.fft.fft(E_total, axis=0)), axes=0)
Spec_Rx = np.fft.fftshift(np.abs(np.fft.fft(E, axis=0)), axes=0)
f_GHz = f / 1e9

plt.subplot(4, 1, 1);plt.plot(f_GHz, Spec_Tx[:, 0]);plt.title('Tx Spectrum (X-Pol)')
plt.grid(True);plt.xlabel('Freq (GHz)');plt.ylabel('Magnitude');plt.xlim([np.min(f_GHz), np.max(f_GHz)])

plt.subplot(4, 1, 2);plt.plot(f_GHz, Spec_Tx[:, 1]);plt.title('Tx Spectrum (Y-Pol)')
plt.grid(True);plt.xlabel('Freq (GHz)');plt.ylabel('Magnitude');plt.xlim([np.min(f_GHz), np.max(f_GHz)])

plt.subplot(4, 1, 3);plt.plot(f_GHz, Spec_Rx[:, 0]);plt.title('Rx Spectrum (X-Pol)')
plt.grid(True);plt.xlabel('Freq (GHz)');plt.ylabel('Magnitude');plt.xlim([np.min(f_GHz), np.max(f_GHz)])

plt.subplot(4, 1, 4);plt.plot(f_GHz, Spec_Rx[:, 1]);plt.title('Rx Spectrum (Y-Pol)')
plt.grid(True);plt.xlabel('Freq (GHz)');plt.ylabel('Magnitude');plt.xlim([np.min(f_GHz), np.max(f_GHz)])

plt.tight_layout()

# plt.show()

#%%
# ==========================================
# RX: per-channel downconvert, sample, demodulate
# ==========================================

BER = np.zeros((2, Nch))
b_match = rrc_taps.reshape(-1, 1) # column vector

rx_bits = [None] * Nch
rx_symbols = [None] * Nch
rx_wave = [None] * Nch

# Plot settings
plots_per_fig = Nch
nrows = int(np.floor(np.sqrt(plots_per_fig)))
ncols = int(np.ceil(plots_per_fig / nrows))

plt.figure('X-Polarization Constellations', figsize=(12, 8))
plt.suptitle('X-Polarization')
fig_x = plt.gcf()

plt.figure('Y-Polarization Constellations', figsize=(12, 8))
plt.suptitle('Y-Polarization')
fig_y = plt.gcf()

for m in range(Nch):
    fch = ch_idx[m] * DeltaF
    
    # Downconvert
    carrier_rx = np.exp(-1j * 2 * np.pi * fch * t)
    r = E * carrier_rx
    
    # --- DBP (Dual Polarization) ---
    BW = DeltaF
    guard_factor_dbp = 10.0
    
    print(f"Running DBP for Channel {m+1}/{Nch}...")
    r_dbp = dbp_fullband_to_subband(r, BW, guard_factor_dbp, L_span, alpha_dBpm,
                           beta1_DSP, beta2_DSP, beta3_DSP, gamma_DSP, 
                           dz, fs, Nspans, G_lin, fch)
                           
    # --- Matched Filter ---
    # np.convolve only works on 1D arrays, so we process columns separately
    r_filt_x = np.convolve(r_dbp[:, 0], b_match.flatten(), mode='full')
    r_filt_y = np.convolve(r_dbp[:, 1], b_match.flatten(), mode='full')
    r_filt = np.column_stack((r_filt_x, r_filt_y))
    
    rx_wave[m] = r_filt
    
    # --- Sampling ---
    beta1_virtual = (beta2 - beta2_DSP) * fch + (beta3 - beta3_DSP) * (fch**2) / 2
    beta_delay_samples_DSP = int(np.round(2 * np.pi * beta1_virtual * Nspans * L_span * fs))
    
    # Python is 0-indexed, start_tx matches tx_start_idx array
    start_tx = tx_start_idx[m] + rrc_delay * 2 + beta_delay_samples_DSP
    
    # Create sampling index grid
    idx_end = start_tx + Nsym * sps
    idx = np.arange(start_tx, idx_end, sps)
    
    # Safety bounds check
    idx = idx[idx < r_filt.shape[0]]
    
    y = r_filt[idx, :]
    
    # Normalize X and Y using TX power records
    # avg_pow_tx_up has shape (2, Nch) -> Row 0 is X, Row 1 is Y
    p_raw_x = avg_pow_tx_up[0, m]
    p_raw_y = avg_pow_tx_up[1, m]
    
    y[:, 0] = y[:, 0] * np.sqrt(p_raw_x / PinW_X)
    y[:, 1] = y[:, 1] * np.sqrt(p_raw_y / PinW_Y)
    
    rx_symbols[m] = y
    
    # --- Demodulation & BER ---
    bits_hat_x = demodulate_16qam(y[:, 0], M)
    bits_hat_y = demodulate_16qam(y[:, 1], M)
    
    rx_bits[m] = {'X': bits_hat_x, 'Y': bits_hat_y}
    tx_bits_ref = tx_bits[m]
    
    # Calculate BER (mean of boolean array where bits don't match)
    BER[0, m] = np.mean(bits_hat_x != tx_bits_ref['X'][:len(bits_hat_x)])
    BER[1, m] = np.mean(bits_hat_y != tx_bits_ref['Y'][:len(bits_hat_y)])
    print(f'Channel {m+1} BER: X = {BER[0, m]:.3g}, Y = {BER[1, m]:.3g}')
    
    # --- Plotting ---
    # matplotlib subplots are 1-indexed (unlike arrays)
    plt.figure(fig_x.number)
    plt.subplot(nrows, ncols, m + 1)
    plt.scatter(np.real(y[:, 0]), np.imag(y[:, 0]), s=8, alpha=0.6)
    plt.axis('square')
    plt.grid(True)
    plt.title(f'Ch {m+1} X')
    
    plt.figure(fig_y.number)
    plt.subplot(nrows, ncols, m + 1)
    plt.scatter(np.real(y[:, 1]), np.imag(y[:, 1]), s=8, alpha=0.6)
    plt.axis('square')
    plt.grid(True)
    plt.title(f'Ch {m+1} Y')

plt.show()
# %%


