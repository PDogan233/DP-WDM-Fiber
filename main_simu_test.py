import numpy as np
import matplotlib.pyplot as plt

# Custom modules
from para import get_parameters
from tx_DSP import multiplex_wdm_channels
from rx_DSP import apply_matched_filter, compensate_phase, normalize_rx_power, addskew, decimator, demodulate_16qam, subband_convert, fullband_convert
from utils import sync_align, rcosdesign, norm_matrix
from channel import channel_propagation
from dbp import dbp_fullband_to_subband, dbp_subband
from visualize import plot_spectrum, plot_wav_spec, setup_constellation, single_ch_constellation


# --- Load Parameters --- 
p = get_parameters()

# --- Generate Train and Test Datasets --- 
# includes WDM composite signals and corresponding TX data for evaluation
print("Generating Training Set...")
np.random.seed(42)
E_tx_train, tx_data_train = multiplex_wdm_channels(p)

print("Generating Testing Set...")
np.random.seed(99)
E_tx_test, tx_data_test = multiplex_wdm_channels(p)


# --- Channel propagation: Nspans of SSF + EDFA --- 
print("Running Fiber Propagation (Train Set)...")

# Pass the combined signal through the entire channel (Fiber + EDFA).
E_rx_train = channel_propagation(E_tx_train, p)

plt.ion()
# Plot Waveforms and Spectra
plot_wav_spec(p['t'], p['f'], E_tx_train, E_rx_train)


# --- RX Processing (Targeting Central Channel) --- 
print(f"\n--- Processing All {p['Nch']} Channels (Baseline Test) ---")

fig_x, fig_y, nrows, ncols = setup_constellation(p['Nch'])
BER = np.zeros((2, p['Nch']))

# Loop over all channels to verify the modularized pipeline
for m in range(p['Nch']):
    fch = p['ch_idx'][m] * p['DeltaF']
    print(f"\nProcessing Channel {m+1}/{p['Nch']} (Center Freq: {fch/1e9:.2f} GHz)...")
    
    # Downconvert
    carrier_rx = np.exp(-1j * 2 * np.pi * fch * p['t'])
    r_baseband = E_rx_train * carrier_rx
    # plot_spectrum(p['f'], r_baseband)
    
    # Extract Subband & Downsample (Targeting exact integer sps)
    r_sub_in, p['Nsub'], p['Nt'], p['fs_sub'], p['sps_sub'] = subband_convert(
        r_baseband, p['sps_rx'], p['fs'], p['Rs']
    )
    print(f"  -> Subband Extracted: SPS reduced from {p['sps_rx']:.2f} to {p['sps_sub']:.2f}")
    
    
    # Core DBP Processing on Subband: Perform nonlinear and dispersion compensation
    print("  -> Running DBP on Subband...")
    r_dbp_sub = dbp_subband(
        r_sub_in, p['Nsub'], p['fs_sub'], fch, p['L_span'], p['alpha_dBpm'], 
        p['beta2_DSP'], p['beta3_DSP'], p['gamma_DSP'], 
        p['dz_DBP'], p['Nspans'], p['G_lin']
    )
    

    # Matched Filter (Using the RX-specific taps and the DBP output directly)
    r_filt = apply_matched_filter(r_dbp_sub, p['rrc_taps_rx'])
    
    # Clock Recovery (Decimate to 1 sps, using the subband sps)
    # We round sps_sub to integer just to be safe, though it should already be
    y_x_1sps = decimator(r_filt[:, 0], int(np.round(p['sps_sub'])))
    y_y_1sps = decimator(r_filt[:, 1], int(np.round(p['sps_sub'])))
    
    
    # Frame Synchronization
    tx_sym_x = tx_data_train['symbols'][m]['X']
    tx_sym_y = tx_data_train['symbols'][m]['Y']
    tx_align_x, rx_align_x, sync_x = sync_align(tx_sym_x, y_x_1sps)
    tx_align_y, rx_align_y, sync_y = sync_align(tx_sym_y, y_y_1sps)
    
    # Phase Derotation
    rx_derotated_x = compensate_phase(rx_align_x, sync_x)
    rx_derotated_y = compensate_phase(rx_align_y, sync_y)
    
    # Power Normalization
    rx_norm_x = normalize_rx_power(rx_derotated_x, tx_data_train['p_raw_x'][m], p['PinW_ch'])
    rx_norm_y = normalize_rx_power(rx_derotated_y, tx_data_train['p_raw_y'][m], p['PinW_ch'])
    
    # Save combined plot data
    y_plot = np.column_stack((rx_norm_x, rx_norm_y))
    single_ch_constellation(fig_x, fig_y, nrows, ncols, m, y_plot)
    
    # Demodulation
    bits_hat_x = demodulate_16qam(rx_norm_x, p['M'])
    bits_hat_y = demodulate_16qam(rx_norm_y, p['M'])
    
    tx_bits_aligned_x = demodulate_16qam(tx_align_x, p['M'])
    tx_bits_aligned_y = demodulate_16qam(tx_align_y, p['M'])
    
    # BER Calculation
    BER[0, m] = np.mean(bits_hat_x != tx_bits_aligned_x)
    BER[1, m] = np.mean(bits_hat_y != tx_bits_aligned_y)
    
    print(f"  -> Delay Lag: X={sync_x['lag']}, Y={sync_y['lag']} symbols")
    print(f"  -> BER: X = {BER[0, m]:.3g}, Y = {BER[1, m]:.3g}")

plt.ioff()
plt.show()
print("Baseline Simulation Complete.")

