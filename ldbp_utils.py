import numpy as np
import matplotlib.pyplot as plt
import torch

from rx_DSP import (apply_matched_filter, compensate_phase, normalize_rx_power,
                    decimator, demodulate_16qam, subband_convert)
from utils import sync_align
from dbp import dbp_subband


# =====================================================================
# Data conversion & pipeline helpers
# =====================================================================

def complex_np_to_torch(arr, device):
    """Convert complex numpy (N, 2) array to complex torch tensor on device."""
    arr = np.ascontiguousarray(arr)
    real = torch.from_numpy(arr.real.copy()).float().to(device)
    imag = torch.from_numpy(arr.imag.copy()).float().to(device)
    return torch.complex(real, imag)


def extract_subband(rx_wav_ch, p, fch):
    """
    Downconvert center channel to baseband and extract subband.
    rx_wav_ch: (Nt, 2) fullband waveform after channel.
    Returns (rx_wav_sub, Nsub, fs_sub, sps_sub).
    """
    carrier_rx = np.exp(-1j * 2 * np.pi * fch * p['t'])
    baseband = rx_wav_ch * carrier_rx
    rx_wav_sub, Nsub, Nt, fs_sub, sps_sub = subband_convert(
        baseband, p['sps_rx'], p['fs'], p['Rs']
    )
    return rx_wav_sub, Nsub, fs_sub, sps_sub


def run_rx_chain(rx_wav_in, tx_data, p, m, sps_sub, rrc_taps_rx):
    """
    Standard RX DSP chain on DBP/LDBP output (waveform level).
    Steps: matched filter -> decimator -> sync -> phase derotate -> power norm -> demod.
    Returns (ber_x, ber_y, rx_sym_xy).
    """
    # Matched filter (still at waveform sps)
    rx_wav_mf = apply_matched_filter(rx_wav_in, rrc_taps_rx)

    # Clock recovery: decimate to 1 sample/symbol
    rx_sym_x_dec = decimator(rx_wav_mf[:, 0], int(np.round(sps_sub)))
    rx_sym_y_dec = decimator(rx_wav_mf[:, 1], int(np.round(sps_sub)))

    tx_sym_x = tx_data['symbols'][m]['X']
    tx_sym_y = tx_data['symbols'][m]['Y']

    # Frame synchronization via cross-correlation
    tx_align_x, rx_align_x, sync_x = sync_align(tx_sym_x, rx_sym_x_dec)
    tx_align_y, rx_align_y, sync_y = sync_align(tx_sym_y, rx_sym_y_dec)

    # Bulk phase derotation
    rx_derot_x = compensate_phase(rx_align_x, sync_x)
    rx_derot_y = compensate_phase(rx_align_y, sync_y)

    # Power normalization to match TX constellation boundaries
    rx_norm_x = normalize_rx_power(rx_derot_x, tx_data['p_raw_x'][m], p['PinW_ch'])
    rx_norm_y = normalize_rx_power(rx_derot_y, tx_data['p_raw_y'][m], p['PinW_ch'])

    # 16-QAM hard-decision demodulation
    bits_hat_x = demodulate_16qam(rx_norm_x, p['M'])
    bits_hat_y = demodulate_16qam(rx_norm_y, p['M'])
    tx_bits_x = demodulate_16qam(tx_align_x, p['M'])
    tx_bits_y = demodulate_16qam(tx_align_y, p['M'])

    ber_x = np.mean(bits_hat_x != tx_bits_x)
    ber_y = np.mean(bits_hat_y != tx_bits_y)

    # Combine X and Y symbols for constellation plotting
    rx_sym_xy = np.column_stack((rx_norm_x, rx_norm_y))
    return ber_x, ber_y, rx_sym_xy


# =====================================================================
# Plotting helpers
# =====================================================================

def plot_constellation_grid(data, title='Constellations'):
    """
    2-row grid of constellation diagrams.
    data: list of dicts with keys 'x', 'y', 'label'.
    Row 0 = X-pol, Row 1 = Y-pol across all columns.
    """
    n = len(data)
    fig, axes = plt.subplots(2, n, figsize=(3 * n, 7), num=title)
    if n == 1:
        axes = axes.reshape(2, 1)

    for col, d in enumerate(data):
        for row, pol in enumerate(['X', 'Y']):
            ax = axes[row, col]
            sym = d['x'] if pol == 'X' else d['y']
            ax.scatter(np.real(sym), np.imag(sym), s=6, alpha=0.5)
            ax.set_title(f"{d['label']}  ({pol}-pol)")
            ax.axis('square')
            ax.grid(True)
            ax.set_xlim(-4, 4)
            ax.set_ylim(-4, 4)

    fig.tight_layout()


def plot_h_phase(model, Nsub, fs_sub, fch, L_span, alpha_dBpm, beta2, beta3,
                 steps_per_span, title='H Filter Phase vs Frequency'):
    """
    Overlay all linear layers' learned H phase on a single plot,
    with the ideal (physical-init) H phase as a thick black reference line.

    Dispersion compensation filters should have flat magnitude and parabolic
    phase across frequency. This plot reveals how LDBP adjusts the phase.
    """
    # Frequency grid
    df_sub = fs_sub / Nsub
    f_ghz = (np.arange(-Nsub / 2, Nsub / 2) * df_sub) / 1e9
    omega = 2 * np.pi * (np.arange(-Nsub / 2, Nsub / 2) * df_sub + fch)

    # Ideal H from physical formula
    alpha_np = np.log(10 ** (alpha_dBpm / 10))
    beta_omega = 0.5 * beta2 * omega**2 + (1.0 / 6.0) * beta3 * omega**3
    h_dbp = -L_span / steps_per_span
    H_ideal = np.exp((-alpha_np / 2) * h_dbp - 1j * beta_omega * h_dbp)
    ideal_phase = np.unwrap(np.angle(H_ideal))

    fig, ax = plt.subplots(figsize=(10, 5), num=title)

    # Ideal reference
    ax.plot(f_ghz, ideal_phase, 'k-', linewidth=2.5,
            label='Ideal (physical init)')

    # Overlay all learned layers
    colors = plt.cm.viridis(np.linspace(0, 1, 8))
    layer_idx = 0
    for ly in model.layers:
        if not (hasattr(ly, 'H_real') and hasattr(ly, 'H_imag')):
            continue
        H_learned = torch.complex(ly.H_real, ly.H_imag)
        H_np = H_learned.detach().cpu().numpy().flatten()
        learned_phase = np.unwrap(np.angle(H_np))
        ax.plot(f_ghz, learned_phase, color=colors[layer_idx % len(colors)],
                linewidth=0.6, alpha=0.7, label=f'Layer {layer_idx + 1}')
        layer_idx += 1

    ax.set_xlabel('Frequency (GHz)')
    ax.set_ylabel('Phase (rad)')
    ax.set_title(title)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True)
    fig.tight_layout()


def plot_ber_bars(ber_results, title='BER Comparison'):
    """
    Grouped bar chart of BER values in log scale.
    ber_results: list of (label, ber_x, ber_y) tuples.
    """
    labels = [r[0] for r in ber_results]
    ber_x_vals = [r[1] for r in ber_results]
    ber_y_vals = [r[2] for r in ber_results]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6, 4), num=title)
    ax.bar(x - width / 2, ber_x_vals, width,
           label='X-pol', color='steelblue')
    ax.bar(x + width / 2, ber_y_vals, width,
           label='Y-pol', color='darkorange')

    ax.set_yscale('log')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('BER')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    fig.tight_layout()
