# ==========================================
# rx_DSP.py
# Receiver Digital Signal Processing functions
# ==========================================
import numpy as np
from scipy import signal
from scipy.interpolate import CubicSpline
from scipy.signal import resample

def apply_matched_filter(r_in, rrc_taps):
    """Applies the matched RRC filter independently to X and Y polarizations."""
    b_match = rrc_taps.reshape(-1, 1)
    # np.convolve only works on 1D arrays, process columns separately
    r_filt_x = np.convolve(r_in[:, 0], b_match.flatten(), mode='full')
    r_filt_y = np.convolve(r_in[:, 1], b_match.flatten(), mode='full')
    return np.column_stack((r_filt_x, r_filt_y))

def compensate_phase(rx_aligned, sync_info):
    """
    Extracts the phase angle from the cross-correlation peak 
    and applies derotation to compensate for beta2 mismatch bulk phase.
    """
    peak_complex = sync_info['correlation_array'][np.argmax(np.abs(sync_info['correlation_array']))]
    phase_offset = np.angle(peak_complex)
    return rx_aligned * np.exp(1j * phase_offset)

def normalize_rx_power(rx_derotated, p_raw, PinW_ch):
    """
    Normalizes the RX power using TX power records to ensure 
    the constellation matches the expected decision boundaries.
    """
    PinW_pol = 0.5 * PinW_ch
    return rx_derotated * np.sqrt(p_raw / PinW_pol)

def addskew(sigin, skew, Fs=1.0):
    """
    @Sub-function for adding a time-domain fractional delay (skew).
    It uses the Fourier shift theorem: shifting in time equals a phase ramp in frequency.
    For complex signals, it strictly replicates MATLAB's logic by processing 
    real and imaginary parts independently.
    
    Args:
        sigin (ndarray): Input complex or real signal (1D array).
        skew (float): Fractional delay to apply.
        Fs (float): Sampling frequency (defaults to 1.0 as in Decimator call).
        
    Returns:
        ndarray: Time-shifted signal.
    """
    # Ensure input is a flat 1D array
    sigin = np.asarray(sigin).flatten()
    is_complex = np.iscomplexobj(sigin)
    
    # Split into real and imaginary columns to match MATLAB's separation logic
    if is_complex:
        sig = np.column_stack((np.real(sigin), np.imag(sigin)))
    else:
        sig = sigin.reshape(-1, 1)
        
    Npoints = sig.shape[0]
    
    # Generate the exact frequency grid used by MATLAB's standard FFT.
    # np.fft.fftfreq generates [0, 1, ..., N/2-1, -N/2, ..., -1] / N points
    # This exactly matches the MATLAB expression: [0:(Npoints/2) (-Npoints/2+1):1:-1]/Npoints
    freqGrid = np.fft.fftfreq(Npoints)
    
    # Construct the phase shift operator
    SkewPhase = np.exp(1j * 2 * np.pi * skew * Fs * freqGrid)
    
    # Pre-allocate output array
    sigout = np.zeros_like(sig, dtype=float)
    
    for i in range(sig.shape[1]):
        temp_sig = sig[:, i]
        # Perform FFT
        SIGIN = np.fft.fft(temp_sig)
        # Apply fractional delay phase shift
        SIGOUT = SIGIN * SkewPhase
        # Inverse FFT and take the real part (since input to FFT was real)
        sigo = np.real(np.fft.ifft(SIGOUT))
        sigout[:, i] = sigo

    # Recombine real and imaginary parts if the original signal was complex
    if is_complex:
        return sigout[:, 0] + 1j * sigout[:, 1]
    else:
        return sigout[:, 0]

def decimator(E_m, Nss):
    """
    @Sub-function for variance-based timing extraction and decimation.
    Finds the optimal fractional sampling phase by maximizing the signal variance,
    applies the fractional delay, and decimates the signal.
    
    Args:
        E_m (ndarray): Input complex oversampled signal.
        Nss (int): Oversampling factor (Samples per Symbol).
        
    Returns:
        ndarray: Downsampled signal at the optimal symbol phase.
    """
    E_m = np.asarray(E_m).flatten()
    r = int(Nss)
    N = r * (len(E_m) // r)  # Equivalent to fix(numel(E_m)/Nss) * Nss
    
    SYMBOLS_LIMIT = int(1e5)
    LIM = SYMBOLS_LIMIT * r
    LIM = min(LIM, N)
    
    E_m_trunc = E_m[:LIM]
    
    # Reshape signal into a matrix where each column is a phase offset.
    # Python's default row-major (-1, r) directly achieves the same matrix 
    # structure as MATLAB's reshape(..., Nss, []).' (which does column-major then transpose).
    symbols = E_m_trunc.reshape(-1, r)
    
    # Calculate variance along the symbol axis (axis=0).
    # NOTE: ddof=1 is critical here to match MATLAB's default sample variance definition.
    S = np.var(symbols, axis=0, ddof=1)
    
    N2 = 10
    Nsst = 5000
    
    # Replicate the variance profile to simulate periodicity (equivalent to repmat)
    v = np.tile(S, N2)
    L = N2 * Nsst
    
    # Spline interpolation to find the fine-grained maximum
    # x2 uses 0-based coordinate system
    x2 = np.linspace(0, len(v) - 1, L)
    orig_x = np.arange(len(v))
    
    # CubicSpline represents MATLAB's interp1(..., 'spline')
    cs = CubicSpline(orig_x, v)
    v2 = cs(x2)
    
    # Search for the peak in the second simulated period to avoid edge artifacts
    # MATLAB: 1*Nsst to 2*Nsst (inclusive). Python: Nsst to 2*Nsst (inclusive -> +1 on upper bound).
    start_idx = Nsst
    end_idx = 2 * Nsst + 1
    idx_range = np.arange(start_idx, end_idx)
    
    # Find index of maximum variance in the interpolated grid
    ptr_offset = np.argmax(v2[idx_range])
    samplingPoint = x2[idx_range[ptr_offset]]
    
    # Extract fractional time skew
    skew = samplingPoint - np.floor(samplingPoint)
    
    # Apply fractional skew using addskew (assuming Fs=1 as in MATLAB call)
    E_m_skewed = addskew(E_m[:N], skew, Fs=1.0)
    
    # Determine the integer downsampling start point (0-based indexing)
    # mod(floor(samplingPoint) - 1, r) + 1 in MATLAB becomes floor(sp) % r in Python
    ptr_idx = int(np.floor(samplingPoint)) % r
    
    # Decimate the skewed signal
    idx = np.arange(ptr_idx, N, r)
    E_out = E_m_skewed[idx]
    
    # if draw:
    #     import matplotlib.pyplot as plt
    #     plt.figure()
    #     plt.plot(v, label='Original Variance')
    #     plt.plot(x2, v2, label='Spline Interpolation')
    #     plt.title('Clock Recovery Interpolation Curve')
    #     plt.legend()
    #     plt.show()
    #
    #     plt.figure()
    #     plt.hist(np.abs(E_out), bins=50)
    #     plt.title('Amplitude Probability Distribution')
    #     plt.show()
        
    return E_out

def demodulate_16qam(symbols, M=16):
    """
    Demodulates 16-QAM complex symbols back to bits.
    Uses vectorized distance calculation instead of loops for efficiency.
    """
    symbols = np.asarray(symbols).flatten()
    k = int(np.log2(M))
    
    # Standard 16-QAM constellation mapping (same as modulate_16qam)
    qam16_map = np.array([
        -3+3j, -3+1j, -3-3j, -3-1j,
        -1+3j, -1+1j, -1-3j, -1-1j,
         3+3j,  3+1j,  3-3j,  3-1j,
         1+3j,  1+1j,  1-3j,  1-1j
    ])
    
    # Vectorized distance calculation: shape (N_symbols, 16)
    # Broadcasting computes the distance from each symbol to all 16 ideal points
    distances = np.abs(symbols[:, np.newaxis] - qam16_map[np.newaxis, :])
    
    # Find the index of the minimum distance (equivalent to the decimal symbol value)
    dec_symbols = np.argmin(distances, axis=1)
    
    # Convert decimal back to binary bits (left-msb)
    # Equivalent to MATLAB's de2bi(..., 'left-msb')
    bit_matrix = np.zeros((len(dec_symbols), k), dtype=int)
    for i in range(k):
        # Shift and mask to get each bit
        bit_matrix[:, k - 1 - i] = (dec_symbols >> i) & 1
        
    bits = bit_matrix.flatten()
    return bits

def subband_convert(r_in, target_sps_sub, fs, Rs):
    """
    Extracts a subband from the fullband signal by targeting a specific integer SPS.
    This entirely avoids fractional SPS issues and eliminates the need for fullband reconstruction.
    
    Args:
        r_in: Downconverted fullband signal (channel of interest is at DC).
        target_sps_sub: The desired integer samples per symbol for the subband (e.g., 2).
        fs: Original fullband sampling rate.
        Rs: Symbol rate.
    """
    Nt = r_in.shape[0]
    
    # 1. Calculate the required subband sampling rate
    fs_sub_target = target_sps_sub * Rs
    
    # 2. Calculate the exact number of subband samples needed
    # Nsub / Nt = fs_sub_target / fs  =>  Nsub = Nt * (fs_sub_target / fs)
    Nsub = int(np.round(Nt * (fs_sub_target / fs)))
    
    # Ensure Nsub is an even number for standard FFT symmetry
    if Nsub % 2 != 0:
        Nsub += 1
        
    # 3. Transform to frequency domain
    R_shifted = np.fft.fftshift(np.fft.fft(r_in, axis=0), axes=0) / np.sqrt(Nt)
    
    # 4. Extract the central Nsub bins (since the signal is already at baseband)
    start_idx = (Nt - Nsub) // 2
    end_idx = start_idx + Nsub
    idx = np.arange(start_idx, end_idx)
    
    R_sub_shifted = R_shifted[idx, :]
    
    # 5. Transform back to time domain
    R_sub = np.fft.ifftshift(R_sub_shifted, axes=0)
    r_time = np.fft.ifft(R_sub, n=Nsub, axis=0) * np.sqrt(Nsub) 
    
    # Scale to maintain power due to sub-sampling
    r_time = r_time * np.sqrt(Nsub / Nt)
    
    # Recalculate exact parameters in case of minor rounding adjustments
    fs_sub_exact = fs * Nsub / Nt
    sps_sub_exact = fs_sub_exact / Rs
    
    return r_time, Nsub, Nt, fs_sub_exact, sps_sub_exact

def subband_convert_BW(r_in, BW, guard_factor, fs, Rs):
    """
    Extracts a subband from the fullband signal by targeting a specific bandwidth.
    Returns the downsampled signal and necessary parameters for later steps.
    """
    Nt = r_in.shape[0]
    df = fs / Nt
    f = np.arange(-Nt/2, Nt/2).reshape(-1, 1) * df 
    
    bw_eff = BW * (1 + guard_factor)
    f_low, f_high = -bw_eff / 2, bw_eff / 2
    
    R_shifted = np.fft.fftshift(np.fft.fft(r_in, axis=0), axes=0) / np.sqrt(Nt)
    
    # Find frequency indices within the subband
    # np.where returns a tuple, we take the first element (array of indices)
    idx_tuple = np.where((f >= f_low) & (f <= f_high))[0]
    if len(idx_tuple) == 0:
        raise ValueError("Selected BW too small relative to df; no bins selected.")
        
    idx_min, idx_max = np.min(idx_tuple), np.max(idx_tuple)
    idx = np.arange(idx_min, idx_max + 1)
    Nsub = len(idx)
    
    # Extract subband spectrum
    R_sub_shifted = R_shifted[idx, :]
    
    # Convert to time domain for subband
    R_sub = np.fft.ifftshift(R_sub_shifted, axes=0)
    r_sub = np.fft.ifft(R_sub, n=Nsub, axis=0) * np.sqrt(Nsub) 
    
    # Scale to maintain power due to sub-sampling
    r_time = r_sub * np.sqrt(Nsub / Nt)
    
    # Calculate new subband parameters
    fs_sub = fs * Nsub / Nt
    sps_sub = fs_sub / Rs
    

    
    return r_time, idx, Nsub, Nt, fs_sub, sps_sub

def fullband_convert(r_time, idx, Nt, Nsub):
    """
    Reconstructs the fullband signal by zero-padding the processed subband spectrum.
    (Optional step if demodulation is performed directly at subband sampling rate)
    """
    
    # Reconstruct Full Spectrum
    R_sub_proc = np.fft.fftshift(np.fft.fft(r_time, n=Nsub, axis=0), axes=0) / np.sqrt(Nsub)
    
    # Place back into full spectrum with zeros outside
    R_shifted_db = np.zeros((Nt, 2), dtype=complex)
    R_shifted_db[idx, :] = R_sub_proc
    
    r_out_full = np.fft.ifft(np.fft.ifftshift(R_shifted_db, axes=0), n=Nt, axis=0) * np.sqrt(Nt)
    # Scale back to maintain power
    r_out_full = r_out_full * np.sqrt(Nt / Nsub) 
    
    return r_out_full

