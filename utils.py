import numpy as np
from scipy import signal
from scipy.interpolate import CubicSpline

def sync_align(sig1, sig2):
    """
    Aligns two signals based on cross-correlation and standardizes length.
    
    This function performs the following:
    1. Calculates cross-correlation to find the delay between sig1 and sig2.
    2. Detects polarity (inverted or not) and corrects sig2 if necessary.
    3. Shifts the signals to align the start points.
    4. Truncates both signals to the length of the shorter signal (overlap region).
    
    Args:
        sig1 (ndarray): First complex signal (e.g., Reference or Captured).
        sig2 (ndarray): Second complex signal.
        
    Returns:
        sig1_out (ndarray): Aligned and truncated sig1.
        sig2_out (ndarray): Aligned and truncated sig2.
        sync_info (dict): Debug info including max correlation, lag, and polarity.
    """
    # Ensure 1D arrays
    sig1 = np.asarray(sig1).flatten()
    sig2 = np.asarray(sig2).flatten()
    
    # 1. Cross-Correlation
    # scipy.signal.correlate automatically conjugates sig2 when inputs are complex, 
    # perfectly matching MATLAB's xcorr behavior.
    xcorr_result = signal.correlate(sig1, sig2, mode='full')
    lags = signal.correlation_lags(len(sig1), len(sig2), mode='full')
    
    # 2. Find Peak and Lag
    max_xcorr_idx = np.argmax(np.abs(xcorr_result))
    best_lag = lags[max_xcorr_idx]
    peak_complex = xcorr_result[max_xcorr_idx]
    
    # 3. Polarity Detection
    # If the real part of the peak is negative, sig2 is inverted
    polarity = -1 if np.real(peak_complex) < 0 else 1
    
    # 4. Alignment Logic
    # 0-based indexing in Python means we do not need the '+ 1' used in MATLAB.
    if best_lag >= 0:
        # sig2 is delayed, skip the first 'best_lag' samples of sig1
        start_idx1 = best_lag
        start_idx2 = 0
    else:
        # sig2 is ahead, skip the first 'abs(best_lag)' samples of sig2
        start_idx1 = 0
        start_idx2 = abs(best_lag)
        
    # 5. Truncation (Equal Length)
    len1_remaining = len(sig1) - start_idx1
    len2_remaining = len(sig2) - start_idx2
    common_len = min(len1_remaining, len2_remaining)
    
    # Slicing in Python is end-exclusive: [start : start + length]
    sig1_out = sig1[start_idx1 : start_idx1 + common_len]
    sig2_out = sig2[start_idx2 : start_idx2 + common_len]
    
    # 6. Save Debug Info
    sync_info = {
        'lag': best_lag,
        'polarity': polarity,
        'max_corr': np.abs(peak_complex),
        'correlation_array': xcorr_result
    }
    
    return sig1_out, sig2_out, sync_info

def rcosdesign(beta, span, sps):
    """
    Generates a Root Raised Cosine (RRC) filter.
    Equivalent to MATLAB's rcosdesign(beta, span, sps, 'sqrt').
    """
    delay = span * sps / 2
    t = np.arange(-delay, delay + 1)
    h = np.zeros(len(t), dtype=float)
    
    # RRC mathematical formulation to handle singularities
    for i in range(len(t)):
        if t[i] == 0:
            h[i] = 1.0 - beta + 4 * beta / np.pi
        elif beta != 0 and abs(t[i]) == sps / (4 * beta):
            h[i] = beta / np.sqrt(2) * ((1 + 2 / np.pi) * np.sin(np.pi / (4 * beta)) + 
                                        (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta)))
        else:
            num = np.sin(np.pi * t[i] / sps * (1 - beta)) + \
                  4 * beta * t[i] / sps * np.cos(np.pi * t[i] / sps * (1 + beta))
            den = np.pi * t[i] / sps * (1 - (4 * beta * t[i] / sps)**2)
            h[i] = num / den
            
    # Normalize energy similar to MATLAB
    h = h / np.sqrt(sps)
    return h


def norm_matrix(matrix):
    """ norm to [-1, 1] """
    max_val = np.max(matrix)
    min_val = np.min(matrix)
    return 2 * (matrix - min_val) / (max_val - min_val) - 1