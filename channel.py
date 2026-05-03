import numpy as np


def channel_propagation(E_in, p):
    """
    Simulates the full WDM transmission link.
    Loops through N spans of fiber propagation followed by EDFA amplification.
    """
    E_out = E_in.copy()
    
    for span in range(p['Nspans']):
        # 1. Fiber Propagation (Split-Step Fourier Method)
        E_out = fiber_prop_DP(
            E_out, p['L_span'], p['alpha_dBpm'], p['beta1'], p['beta2'], p['beta3'], 
            p['gamma'], p['dz'], p['fs'], p['control_param'], p['method'], 
            p['B_wdm'], p['pmd_coeff']
        )
        
        # 2. EDFA Amplification and Noise compensation
        E_out = apply_edfa(E_out, p['G_lin'], p['nsp'], p['lambda0'], p['fs'])
        
    return E_out

def choose_step(method, dz, L, z, control_param, y, gamma, beta2, B_wdm):
    """
    CHOOSE_STEP - choose step length according to method
    """
    if method is None or method == '':
        method = 'constant'
        
    method = method.lower()
    
    if method == 'constant':
        h = dz
            
    elif method == 'local_error':
        LE_target = control_param
        # Pmax over the combined polarizations: sum along axis=1
        Pmax = np.max(np.sum(np.abs(y)**2, axis=1))
        if Pmax == 0:
            return L - z
        
        gamma_eff = (8/9) * gamma
        denominator = gamma_eff * Pmax * (4 * np.pi**2 * beta2**2) * (B_wdm**4)
        h = (LE_target / denominator)**(1/3)
        
    elif method == 'global_error':
        GE_target = control_param
        Pmax = np.max(np.sum(np.abs(y)**2, axis=1))
        if Pmax == 0:
            return L - z
            
        gamma_eff = (8/9) * gamma
        denominator = L * gamma_eff * Pmax * (4 * np.pi**2 * beta2**2) * (B_wdm**4)
        h = (GE_target / denominator)**(1/2)
        
    else:
        h = dz
            
    return h

def apply_nonlinearity(y_in, gamma, h):
    """
    APPLY_NONLINEARITY - apply Kerr nonlinearity for propagation length h
    Using Manakov equation factor (8/9) for randomly birefringent fibers.
    """
    # Sum along polarization dimension (axis=1), keepdims=True to maintain (Nt, 1) shape for broadcasting
    P_total = np.sum(np.abs(y_in)**2, axis=1, keepdims=True)
    
    gamma_manakov = (8/9) * gamma
    phi = gamma_manakov * P_total * h
    
    # Broadcast (Nt, 2) * (Nt, 1)
    y_out = y_in * np.exp(-1j * phi)
    return y_out

def apply_dispersion(y_in, beta_omega, alpha_np, h):
    """
    APPLY_DISPERSION - apply linear propagation operator of length h (Scalar/No PMD)
    """
    # H shape: (Nt, 1)
    H = np.exp((-alpha_np / 2) * h - 1j * (beta_omega * h))
    
    # FFT along time dimension (axis=0)
    X = np.fft.fft(y_in, axis=0)
    X = np.fft.fftshift(X, axes=0)
    
    X = X * H # Broadcasting (Nt, 2) * (Nt, 1)
    
    X = np.fft.ifftshift(X, axes=0)
    y_out = np.fft.ifft(X, axis=0)
    return y_out

def apply_dispersion_PMD(y_in, beta_omega, alpha_np, h, omega, pmd_coeff):
    """
    APPLY_DISPERSION_PMD - Linear step with random birefringence
    """
    H_CD = np.exp((-alpha_np / 2) * h - 1j * (beta_omega * h))
    
    delta_tau = pmd_coeff * np.sqrt(h)
    
    # Random unitary matrix generation
    theta = np.random.rand() * np.pi / 2
    phi = np.random.rand() * 2 * np.pi
    
    c = np.cos(theta)
    s = np.sin(theta)
    e_phi = np.exp(1j * phi)
    
    R11 = c
    R12 = s * np.conj(e_phi)
    R21 = -s * e_phi
    R22 = c
    
    # Delay matrix
    phase_dgd = omega * delta_tau / 2
    D11 = np.exp(-1j * phase_dgd)
    D22 = np.exp(+1j * phase_dgd)
    
    # Intermediate multiplications
    RD11 = R11 * D11
    RD12 = R12 * D22
    RD21 = R21 * D11
    RD22 = R22 * D22
    
    M11 = RD11 * np.conj(R11) + RD12 * np.conj(R12)
    M12 = RD11 * np.conj(R21) + RD12 * np.conj(R22)
    M21 = RD21 * np.conj(R11) + RD22 * np.conj(R12)
    M22 = RD21 * np.conj(R21) + RD22 * np.conj(R22)
    
    M11 = M11 * H_CD
    M12 = M12 * H_CD
    M21 = M21 * H_CD
    M22 = M22 * H_CD
    
    # Apply in frequency domain
    X = np.fft.fft(y_in, axis=0)
    X = np.fft.fftshift(X, axes=0)
    
    # Extract Ex and Ey, keeping them as (Nt, 1) to match M dimensions
    Ex = X[:, 0:1]
    Ey = X[:, 1:2]
    
    Ex_out = M11 * Ex + M12 * Ey
    Ey_out = M21 * Ex + M22 * Ey
    
    X_out = np.concatenate((Ex_out, Ey_out), axis=1)
    
    X_out = np.fft.ifftshift(X_out, axes=0)
    y_out = np.fft.ifft(X_out, axis=0)
    return y_out

def fiber_prop_DP(x, L, alpha_dBpm, beta1, beta2, beta3, gamma, dz, fs, control_param, method, B_wdm, pmd_coeff=0):
    """
    FIBER_PROP_DP - Manakov Equation Split-step propagation
    """
    Nt = x.shape[0]
    df = fs / Nt
    f = np.arange(-Nt/2, Nt/2).reshape(-1, 1) * df
    omega = 2 * np.pi * f
    
    alpha_np = np.log(10**(alpha_dBpm / 10))
    beta_omega = 0.5 * beta2 * (omega**2) + (1/6) * beta3 * (omega**3)
    
    y = x.copy()
    z = 0
    step_count = 0
    
    while z < L - 1e-12:
        step_count += 1
        h = choose_step(method, dz, L, z, control_param, y, gamma, beta2, B_wdm)
        # print("Step size chosen: {:.2f} m (z = {:.2f} m, step count = {})".format(h, z, step_count))
        
        if h <= 0 or z + h > L:
            h = L - z
            
        y = apply_nonlinearity(y, gamma, h/2)
        if pmd_coeff == 0:
            y = apply_dispersion(y, beta_omega, alpha_np, h)
        else:
            y = apply_dispersion_PMD(y, beta_omega, alpha_np, h, omega, pmd_coeff)
        y = apply_nonlinearity(y, gamma, h/2)
        
        z += h
        
    return y


def apply_edfa(E_in, G_lin, nsp, lambda0, fs):
    """
    Applies Erbium-Doped Fiber Amplifier (EDFA) gain and adds ASE noise.
    """
    # Planck constant and speed of light
    h_planck = 6.62607015e-34 
    c = 299792458
    nu0 = c / lambda0
    
    # 1. Apply linear gain
    E_out = np.sqrt(G_lin) * E_in
    
    # 2. Add ASE noise
    ASE_PSD = nsp * h_planck * nu0 * (G_lin - 1) # Noise Power Spectral Density [W/Hz]
    Nt = E_in.shape[0]
    # Generate complex Gaussian noise for both X and Y polarizations
    noise = 0*np.sqrt(ASE_PSD * fs / 2) * (np.random.randn(Nt, 2) + 1j * np.random.randn(Nt, 2))
    E_out = E_out + noise
        
    return E_out

