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


def prepare_data(rx_wav_ch, p, fch, Nsub, fs_sub, sps_sub, device):
    """
    Subband extraction + true DBP + torch conversion for one dataset.
    Returns (rx_wav_sub_ts, rx_wav_dbp_label_np, rx_wav_dbp_label_ts).
    """
    # Extract subband
    rx_wav_sub, _, _, _ = extract_subband(rx_wav_ch, p, fch)

    # True DBP with matched physical params as label
    rx_wav_dbp_label = dbp_subband(
        rx_wav_sub, Nsub, fs_sub, fch,
        p['L_span'], p['alpha_dBpm'],
        p['beta2'], p['beta3'], p['gamma'],
        p['dz_DBP'], p['Nspans'], p['G_lin']
    )

    # Convert to torch
    rx_wav_sub_ts = complex_np_to_torch(rx_wav_sub, device)
    rx_wav_dbp_label_ts = complex_np_to_torch(rx_wav_dbp_label, device)

    return rx_wav_sub_ts, rx_wav_dbp_label, rx_wav_dbp_label_ts


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


def plot_weight_curves(model, fs_sub, Nsub, title='LDBP |H| vs Frequency'):
    """
    Plot |H| vs frequency for each linear layer in the LDBP model.
    One subplot per linear layer.
    """
    df_sub = fs_sub / Nsub
    f_ghz = (np.arange(-Nsub / 2, Nsub / 2) * df_sub) / 1e9

    # Count linear layers (those with H_real / H_imag attributes)
    n_linear = sum(1 for ly in model.layers
                   if hasattr(ly, 'H_real') and hasattr(ly, 'H_imag'))
    if n_linear == 0:
        return

    fig, axes = plt.subplots(n_linear, 1, figsize=(8, 3 * n_linear), num=title)
    if n_linear == 1:
        axes = [axes]

    idx = 0
    for ly in model.layers:
        if not (hasattr(ly, 'H_real') and hasattr(ly, 'H_imag')):
            continue
        H_learned = torch.complex(ly.H_real, ly.H_imag)
        H_abs = torch.abs(H_learned).detach().cpu().numpy().flatten()
        axes[idx].plot(f_ghz, H_abs, linewidth=0.5)
        axes[idx].set_xlabel('Frequency (GHz)')
        axes[idx].set_ylabel('|H|')
        axes[idx].set_title(f'Linear Layer {idx + 1}: Learned |H|')
        axes[idx].grid(True)
        idx += 1

    fig.tight_layout()


def plot_ber_bars(ber_results, title='BER Comparison'):
    """
    Grouped bar chart of BER values.
    ber_results: list of (label, ber_x, ber_y) tuples.
    """
    labels = [r[0] for r in ber_results]
    ber_x_vals = [r[1] for r in ber_results]
    ber_y_vals = [r[2] for r in ber_results]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6, 4), num=title)
    bars_x = ax.bar(x - width / 2, ber_x_vals, width,
                    label='X-pol', color='steelblue')
    bars_y = ax.bar(x + width / 2, ber_y_vals, width,
                    label='Y-pol', color='darkorange')

    for bar in bars_x:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., h * 1.1,
                f'{h:.2e}', ha='center', va='bottom', fontsize=7, rotation=90)
    for bar in bars_y:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., h * 1.1,
                f'{h:.2e}', ha='center', va='bottom', fontsize=7, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('BER')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    fig.tight_layout()
