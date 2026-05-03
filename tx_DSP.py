# ==========================================
# tx_DSP.py
# Transmitter Digital Signal Processing functions
# ==========================================
import numpy as np

def modulate_16qam(bits, M=16):
    """
    Modulates a 1D array of bits using 16-QAM.
    Matches MATLAB's default qammod behavior (left-msb).
    """
    # Ensure bits is a 1D numpy array
    bits = np.asarray(bits).flatten()
    k = int(np.log2(M))
    num_symbols = len(bits) // k
    
    # Reshape bits: (num_symbols, k)
    bit_matrix = bits.reshape((num_symbols, k))
    
    # Convert binary to decimal (left-msb)
    # Equivalent to MATLAB's bi2de(bit_matrix, 'left-msb')
    weights = 2 ** np.arange(k - 1, -1, -1)
    dec_symbols = np.sum(bit_matrix * weights, axis=1)
    
    # Standard 16-QAM constellation mapping
    # This lookup table aligns with typical Gray-coded mappings
    # For exact MATLAB qammod default mapping, we define the complex values directly
    qam16_map = np.array([
        -3+3j, -3+1j, -3-3j, -3-1j,
        -1+3j, -1+1j, -1-3j, -1-1j,
         3+3j,  3+1j,  3-3j,  3-1j,
         1+3j,  1+1j,  1-3j,  1-1j
    ])
    
    # Map decimal symbols to complex constellation points
    symbols = qam16_map[dec_symbols]
    
    return symbols


def generate_baseband_symbols(Nsym, k, M):
    """Generates random bits and modulates them into QAM symbols."""
    bits = np.random.randint(0, 2, Nsym * k)
    symbols = modulate_16qam(bits, M)
    return bits, symbols

def apply_pulse_shaping(symbols, sps, rrc_taps):
    """Upsamples the symbols and applies the RRC pulse shaping filter."""
    up_len = len(symbols) * sps
    tx_up_z = np.zeros(up_len, dtype=complex)
    tx_up_z[0::sps] = symbols  # Zero-insertion upsampling
    
    # Apply RRC filter
    tx_shaped = np.convolve(tx_up_z, rrc_taps, mode='full')
    return tx_shaped

def normalize_power(tx_shaped_x, tx_shaped_y, PinW_ch):
    """Normalizes the power of X and Y polarizations to the target launch power."""
    p_raw_x = np.mean(np.abs(tx_shaped_x)**2)
    p_raw_y = np.mean(np.abs(tx_shaped_y)**2)
    
    PinW_pol = 0.5 * PinW_ch # Half power per polarization
    
    tx_shaped_x = tx_shaped_x * np.sqrt(PinW_pol / p_raw_x)
    tx_shaped_y = tx_shaped_y * np.sqrt(PinW_pol / p_raw_y)
    
    return tx_shaped_x, tx_shaped_y, p_raw_x, p_raw_y

def multiplex_wdm_channels(p):
    """
    Generates all WDM channels, applies pulse shaping, and multiplexes 
    them into a single time-domain composite waveform.
    """
    E_total = np.zeros((p['Nt'], 2), dtype=complex)
    
    # Data storage for evaluation later
    tx_data = {
        'bits': [None] * p['Nch'],
        'symbols': [None] * p['Nch'],
        'wave': [None] * p['Nch'],
        'p_raw_x': np.zeros(p['Nch']),
        'p_raw_y': np.zeros(p['Nch'])
    }
    
    for m in range(p['Nch']):
        # 1. Generate QAM symbols for X and Y
        bits_x, sym_x = generate_baseband_symbols(p['Nsym'], p['k'], p['M'])
        bits_y, sym_y = generate_baseband_symbols(p['Nsym'], p['k'], p['M'])
        
        # 2. Pulse Shaping
        tx_shaped_x = apply_pulse_shaping(sym_x, p['sps'], p['rrc_taps'])
        tx_shaped_y = apply_pulse_shaping(sym_y, p['sps'], p['rrc_taps'])
        
        # 3. Power Normalization
        tx_shaped_x, tx_shaped_y, p_x, p_y = normalize_power(tx_shaped_x, tx_shaped_y, p['PinW_ch'])
        
        # 4. Place into padded time grid
        tx_up_block = np.zeros((p['Nt'], 2), dtype=complex)
        L0 = len(tx_shaped_x)
        start_tx = int(np.floor((p['Nt'] - L0) / 2)) 
        
        tx_up_block[start_tx : start_tx + L0, 0] = tx_shaped_x
        tx_up_block[start_tx : start_tx + L0, 1] = tx_shaped_y
        
        # 5. Upconversion (Frequency shift to specific channel center)
        fch = p['ch_idx'][m] * p['DeltaF']
        carrier = np.exp(1j * 2 * np.pi * fch * p['t'])
        
        # Accumulate WDM signal
        E_total += tx_up_block * carrier
        
        # Save records
        tx_data['bits'][m] = {'X': bits_x, 'Y': bits_y}
        tx_data['symbols'][m] = {'X': sym_x, 'Y': sym_y}
        tx_data['wave'][m] = tx_up_block
        tx_data['p_raw_x'][m] = p_x
        tx_data['p_raw_y'][m] = p_y
        
    return E_total, tx_data


