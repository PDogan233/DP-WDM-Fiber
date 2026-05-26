import numpy as np

# --- Local Helper Functions for DBP ---
def apply_dispersion_dbp(y_in, beta_omega, alpha_np, h):
    """
    APPLY_DISPERSION (Local DBP version)
    h is negative for back-propagation.
    """
    # Inverse loss (gain) and inverse dispersion
    H = np.exp((-alpha_np / 2) * h - 1j * (beta_omega * h))
    
    X = np.fft.fft(y_in, axis=0)
    X = np.fft.fftshift(X, axes=0)
    X = X * H # Broadcasting (N, 2) * (N, 1)
    X = np.fft.ifftshift(X, axes=0)
    y_out = np.fft.ifft(X, axis=0)
    return y_out

def apply_nonlinearity_dbp(y_in, gamma, h):
    """
    APPLY_NONLINEARITY (Local DBP version - Manakov)
    h is negative for back-propagation.
    """
    # keepdims=True is crucial for broadcasting!
    P_total = np.sum(np.abs(y_in)**2, axis=1, keepdims=True)
    gamma_manakov = (8/9) * gamma
    
    phi = gamma_manakov * P_total * h
    y_out = y_in * np.exp(-1j * phi)
    return y_out

def dbp_subband(r_time, Nsub, fs_sub, fch, L, alpha_dBpm, beta2, beta3, gamma, dz, Nspans, G_lin=1.0):
    """
    Core function that performs linear and nonlinear back-propagation 
    on the already downsampled subband signal.
    """
    # Subband physical parameters
    df_sub = fs_sub / Nsub
    f_sub = np.arange(-Nsub/2, Nsub/2).reshape(-1, 1) * df_sub
    omega_total_sub = 2 * np.pi * (f_sub + fch)
    
    alpha_np = np.log(10**(alpha_dBpm / 10))
    beta_omega_sub = 0.5 * beta2 * (omega_total_sub**2) + (1/6) * beta3 * (omega_total_sub**3)
    sqrtG = np.sqrt(G_lin)
    
    # Back-propagation Loop
    for spanIdx in range(Nspans - 1, -1, -1):
        # Remove EDFA Gain
        r_time = r_time / sqrtG
        z = 0
        while z < L - 1e-12:
            h_step = dz
            if z + h_step > L:
                h_step = L - z
            # Negative step for inverse propagation
            h = -h_step
            
            r_time = apply_nonlinearity_dbp(r_time, gamma, h/2)
            r_time = apply_dispersion_dbp(r_time, beta_omega_sub, alpha_np, h)
            r_time = apply_nonlinearity_dbp(r_time, gamma, h/2)
            
            z += h_step
            
    return r_time


def dbp_fullband_to_subband(r_in, BW, guard_factor, L, alpha_dBpm, beta1, beta2, beta3, gamma, dz, fs, Nspans, G_lin=1.0, fch=0.0):
    """
    DBP_SUBBAND_DP - Dual Polarization Subband DBP
    Extracts a subband, processes it via SSFM backwards, and maps it back.
    """
    Nt = r_in.shape[0]
    df = fs / Nt
    f = np.arange(-Nt/2, Nt/2).reshape(-1, 1) * df # (Nt, 1)
    
    bw_eff = BW * (1 + guard_factor)
    f_low = -bw_eff / 2
    f_high = bw_eff / 2
    
    R_shifted = np.fft.fftshift(np.fft.fft(r_in, axis=0), axes=0)/np.sqrt(Nt)
    
    # Find frequency indices within the subband
    # np.where returns a tuple, we take the first element (array of indices)
    idx_tuple = np.where((f >= f_low) & (f <= f_high))[0]
    if len(idx_tuple) == 0:
        raise ValueError("Selected BW too small relative to df; no bins selected.")
        
    # Enforce contiguous block
    idx_min, idx_max = np.min(idx_tuple), np.max(idx_tuple)
    idx = np.arange(idx_min, idx_max + 1)
    Nsub = len(idx)
    
    # Extract subband spectrum
    R_sub_shifted = R_shifted[idx, :]
    
    # Convert to time domain for subband
    R_sub = np.fft.ifftshift(R_sub_shifted, axes=0)
    r_sub = np.fft.ifft(R_sub, n=Nsub, axis=0)*np.sqrt(Nsub) # Scale to maintain energy
    
    # Power/Amplitude scaling (to conserve power, not energy, due to sub-sampling)
    r_time = r_sub * np.sqrt(Nsub / Nt)
    
    # Subband physical parameters
    fs_sub = fs * Nsub / Nt
    df_sub = fs_sub / Nsub
    f_sub = np.arange(-Nsub/2, Nsub/2).reshape(-1, 1) * df_sub
    omega_total_sub = 2 * np.pi * (f_sub + fch)
    
    # gamma_sub = gamma * Nsub / Nt
    alpha_np = np.log(10**(alpha_dBpm / 10))
    beta_omega_sub = 0.5 * beta2 * (omega_total_sub**2) + (1/6) * beta3 * (omega_total_sub**3)
    
    sqrtG = np.sqrt(G_lin)
    
    # Back-propagation Loop
    for spanIdx in range(Nspans - 1, -1, -1):
        # Remove EDFA Gain
        r_time = r_time / sqrtG
        
        z = 0
        while z < L - 1e-12:
            h_step = dz
            if z + h_step > L:
                h_step = L - z
                
            # Negative step for inverse propagation
            h = -h_step
            
            r_time = apply_nonlinearity_dbp(r_time, gamma, h/2)
            r_time = apply_dispersion_dbp(r_time, beta_omega_sub, alpha_np, h)
            r_time = apply_nonlinearity_dbp(r_time, gamma, h/2)
            
            z += h_step
            
    # Reconstruct Full Spectrum
    R_sub_proc = np.fft.fftshift(np.fft.fft(r_time, n=Nsub, axis=0), axes=0)/np.sqrt(Nsub)
    
    # Place back into full spectrum with zeros outside
    R_shifted_db = np.zeros((Nt, 2), dtype=complex)
    R_shifted_db[idx, :] = R_sub_proc
    
    r_out_full = np.fft.ifft(np.fft.ifftshift(R_shifted_db, axes=0), n=Nt, axis=0)*np.sqrt(Nt)
    
    # Scale back to maintain power
    r_out_full = r_out_full * np.sqrt(Nt / Nsub) 
    
    return r_out_full


