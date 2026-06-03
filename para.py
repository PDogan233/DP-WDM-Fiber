# ==========================================
# para.py
# Contains all parameters for the optical WDM system
# ==========================================
import numpy as np
from utils import rcosdesign

def get_parameters():
    """
    Generates and returns a dictionary containing all system parameters.
    Divided into Symbols, WDM, Sampling, Physical, and DSP settings.
    """
    p = {}
    
    # --- Symbols Setting ---
    p['Nsym'] = 2**15             # Number of QAM symbols per channel
    p['M'] = 16                   # 16-QAM modulation
    p['k'] = int(np.log2(p['M'])) # Bits per symbol
    p['Rs'] = 32e9                # Symbol rate [symbols/s]
    p['Ts'] = 1 / p['Rs']         # Symbol duration [s]
    p['rrc_rolloff'] = 0.20       # Pulse roll-off factor
    p['B_ch'] = p['Rs'] * (1 + p['rrc_rolloff']) # Channel bandwidth
    
    # --- WDM Layout ---
    p['Nch'] = 3                  # Number of channels
    p['DeltaF'] = 50e9            # Channel spacing [Hz]
    p['ch_idx'] = np.arange(-(p['Nch']-1)/2, (p['Nch']-1)/2 + 1)
    margin = 0.1 * p['B_ch']      # minimum margin in Hz
    minDeltaF = p['B_ch'] + margin  
    if p['DeltaF'] < minDeltaF:
        p['DeltaF'] = minDeltaF
    p['B_wdm'] = (p['Nch'] - 1) * p['DeltaF'] + p['B_ch'] # Total occupied extent [Hz]
    
    # --- Sampling Setting ---
    guard_factor = 2.0 # Guard factor to ensure sufficient oversampling beyond Nyquist rate
    fs_target = p['B_wdm'] * (1 + guard_factor)
    
    # Force tx sps to be a power of 2 to avoid fractional sps issues during function "subband_convert"
    p['sps'] = 2 ** int(np.ceil(np.log2(fs_target / p['Rs'])))  # Samples per symbol
    p['fs'] = p['sps'] * p['Rs']                 # Actual sampling rate [Hz]
    # Setting Rx sps to be an integer divisor of Tx sps to avoid fractional sps issues during resampling in RX DSP
    # This is the target SPS for DBP and Matched Filter. 
    # A higher value corresponds to a wider subband and more accurate DBP, but also higher computational cost.
    p['sps_rx'] = 8  
    # Prevent requesting a subband wider than the original fullband signal
    if p['sps_rx'] > p['sps']:
        p['sps_rx'] = p['sps'] 
    
    # Time and Frequency grids
    p['Nt'] = 2 ** int(np.ceil(np.log2((p['Nsym'] + 50) * p['sps'])))
    zero_padding_num = p['Nt'] - p['Nsym'] * p['sps']
    p['dt'] = 1 / p['fs'] # interval length of time grid
    p['t'] = np.arange(0, p['Nt']).reshape(-1, 1) * p['dt'] # time grid (Nt x 1)
    p['df'] = p['fs'] / p['Nt'] # interval length of freq grid
    p['f'] = np.arange(-p['Nt']/2, p['Nt']/2).reshape(-1, 1) * p['df'] # frequency grid (fftshift ordering)
    
    # RRC Filter Taps (TX)
    rrc_span = 4
    p['rrc_taps'] = rcosdesign(p['rrc_rolloff'], rrc_span, p['sps'])
    
    # RRC Filter Taps (RX) with amplitude scaling to match TX filter gain
    amplitude_scaling_factor = np.sqrt(p['sps'] / p['sps_rx'])
    p['rrc_taps_rx'] = rcosdesign(p['rrc_rolloff'], rrc_span, p['sps_rx']) * amplitude_scaling_factor
    
    
    # --- Physical Parameters ---
    h_planck = 6.62607015e-34 # Planck constant [J*s]
    p['lambda0'] = 1550e-9        # Reference wavelength [m]
    c = 299792458                 # Speed of light [m/s]
    nu0 = c / p['lambda0']         # optical frequency
    
    # Fiber parameters
    p['L_span'] = 100e3           # Fiber length per span [m]
    p['Nspans'] = 10              # Number of spans
    p['alpha_dBpm'] = 0.2e-3      # Loss [dB/m]
   
    GroupRef = 1.47               # group refractive index
    Dispersion = 16.5e-6          # Dispersion D [s/m^2]
    Dis_S = 0.08e3                # Dispersion slope [s/m^3]
    n2 = 2.6e-20                  # Nonlinear index [m^2/W]
    Aeff = 80e-12                 # Effective area [m^2]
    
    p['beta1'] = GroupRef / c # [s/m]
    p['beta2'] = 1 * -Dispersion * p['lambda0']**2 / (2 * np.pi * c) # [s^2/m]
    p['beta3'] = 0 * (p['lambda0']**2 / (2 * np.pi * c))**2 * (Dis_S + 2 * Dispersion / p['lambda0']) #  [s^3/m]
    p['gamma'] = 1 * 2 * np.pi * n2 / (p['lambda0'] * Aeff) # [1/(W*m)]
    p['pmd_coeff'] = 0 * 0.1e-12 / np.sqrt(1000)      # PMD coefficient [s/sqrt(m)]
        
    # Power
    Pdbm_ch = 0
    p['PinW_ch'] = 10 ** ((Pdbm_ch - 30) / 10)
    
    
    # --- Numerical Parameters ---
    
    # step-control scheme
    p['method'] = 'constant'  # 'constant', 'global_error', 'local_error'
    if p['method'] == 'global_error':
        p['control_param'] = 10e-0  # 1e-0
    elif p['method'] == 'local_error':
        p['control_param'] = 1e-3
    else:
        p['control_param'] = 1e3
    p['dz'] = 2e3                 # backup SSF step size
    
    # DSP Mismatch parameters (for testing robustness)
    p['eta1'] = 0.0 * 1e-2
    p['eta2'] = 18 * 1e-2
    p['eta3'] = 0.0 * 1e-2
    p['eta4'] = 0.0 * 1e-2
    p['beta1_DSP'] = (1 + p['eta1']) * p['beta1']
    p['beta2_DSP'] = (1 + p['eta2']) * p['beta2']
    p['beta3_DSP'] = (1 + p['eta3']) * p['beta3']
    p['gamma_DSP'] = (1 + p['eta4']) * p['gamma']
    p['dz_DBP'] = 1 * p['dz']
   
    
    # Amplifier parameters
    G_dB = p['alpha_dBpm'] * p['L_span']
    p['G_lin'] = 10 ** (G_dB / 10)
    NF_dB = 5
    p['nsp'] = 0.5 * 10 ** (NF_dB / 10)
    
    return p