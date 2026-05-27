import numpy as np
import matplotlib.pyplot as plt
import torch

from rx_DSP import (apply_matched_filter, compensate_phase, normalize_rx_power,
                    decimator, demodulate_16qam, subband_convert)
from utils import sync_align


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


def rx_after_dbp(rx_wav_in, tx_data, p, m, sps_sub, rrc_taps_rx):
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



# =====================================================================
# H-filter visualization helpers
# =====================================================================

def _compute_h_one_step(Nsub, fs_sub, fch, L_span, alpha_dBpm,
                        beta2, beta3, steps_per_span):
    """
    Compute the frequency-domain filter H for a single DBP step.

    H(omega) = exp((-alpha/2) * h - j * beta(omega) * h)
    where beta(omega) = 0.5*beta2*omega^2 + (1/6)*beta3*omega^3
    and h = -L_span / steps_per_span (negative = back-propagation).

    Returns (f_ghz, H) where f_ghz is length-Nsub in GHz.
    """
    df_sub = fs_sub / Nsub
    f_sub = np.arange(-Nsub / 2, Nsub / 2) * df_sub
    f_full = f_sub + fch  # absolute optical frequency
    omega = 2 * np.pi * f_full

    alpha_np = np.log(10 ** (alpha_dBpm / 10))
    beta_omega = 0.5 * beta2 * omega**2 + (1.0 / 6.0) * beta3 * omega**3

    h_dbp = -L_span / steps_per_span
    H = np.exp((-alpha_np / 2) * h_dbp - 1j * beta_omega * h_dbp)

    f_ghz = f_full / 1e9
    return f_ghz, H


def _extract_learned_h(model, layer_indices):
    """
    Extract learned H from specified linear layers.

    layer_indices: list of 1-based indices, e.g. [1, 2, 5].
                   If None, extract all linear layers.

    Returns list of (layer_idx, H_numpy) tuples.
    H_numpy is a 1D complex array of length Nsub.
    """
    result = []
    idx = 0
    for layer in model.layers:
        if hasattr(layer, 'H_real') and hasattr(layer, 'H_imag'):
            idx += 1
            if layer_indices is None or idx in layer_indices:
                H = torch.complex(layer.H_real, layer.H_imag)
                H_np = H.detach().cpu().numpy().flatten()
                result.append((idx, H_np))
    return result


# -------------------------------------------------------------------
# Small per-row helpers for the 3x2 H comparison figures.
# Each helper fills one row: left = raw overlay, right = residual.
# -------------------------------------------------------------------

def _block_downsample(arr, factor):
    """
    Downsample by averaging every `factor` contiguous points.

    This naturally smooths high-frequency numerical noise (e.g. phase
    wrapping artifacts at the 1e-3 rad level) that strided decimation
    would preserve.
    """
    n = len(arr) // factor * factor
    return np.mean(arr[:n].reshape(-1, factor), axis=1)


def _plot_phase_row(ax_raw, ax_res, f_ghz, ref_H, ref_label, items, colors, ds):
    """Row 1: unwrapped phase overlay (left) + phase residual (right)."""
    ref_phase = np.unwrap(np.angle(ref_H))

    # Left: raw phase
    ax_raw.plot(f_ghz, ref_phase, 'k--', linewidth=0.8, label=ref_label)
    for ci, (ly_idx, H_full) in enumerate(items):
        Hd = _block_downsample(H_full, ds)
        ax_raw.plot(f_ghz, np.unwrap(np.angle(Hd)), color=colors[ci],
                    linewidth=0.5, label=f'Layer {ly_idx}')
    ax_raw.set_ylabel('Phase (rad)')
    ax_raw.legend(fontsize=5, ncol=4)
    ax_raw.grid(True)

    # Right: phase residual.
    # Both curves are individually unwrapped (proven smooth in the
    # left panel), then subtracted.  This avoids any subtle issues
    # from computing the angle of a complex product.
    for ci, (ly_idx, H_full) in enumerate(items):
        Hd = _block_downsample(H_full, ds)
        residual = np.unwrap(np.angle(Hd)) - ref_phase
        ax_res.plot(f_ghz, residual, color=colors[ci], linewidth=0.5,
                    label=f'Layer {ly_idx}')
    ax_res.axhline(y=0, color='gray', linewidth=0.5, linestyle=':')
    ax_res.set_ylabel('Phase diff (rad)')
    ax_res.legend(fontsize=5, ncol=4)
    ax_res.grid(True)


def _plot_real_row(ax_raw, ax_res, f_ghz, ref_H, ref_label, items, colors, ds):
    """Row 2: real-part overlay (left) + real-part residual (right)."""
    # Left: raw real
    ax_raw.plot(f_ghz, ref_H.real, 'k--', linewidth=0.8, label=ref_label)
    for ci, (ly_idx, H_full) in enumerate(items):
        Hd = _block_downsample(H_full, ds)
        ax_raw.plot(f_ghz, Hd.real, color=colors[ci], linewidth=0.5,
                    label=f'Layer {ly_idx}')
    ax_raw.set_ylabel('Real(H)')
    ax_raw.legend(fontsize=5, ncol=4)
    ax_raw.grid(True)

    # Right: real residual = Re(learned) - Re(ref)
    for ci, (ly_idx, H_full) in enumerate(items):
        Hd = _block_downsample(H_full, ds)
        ax_res.plot(f_ghz, Hd.real - ref_H.real, color=colors[ci],
                    linewidth=0.5, label=f'Layer {ly_idx}')
    ax_res.axhline(y=0, color='gray', linewidth=0.5, linestyle=':')
    ax_res.set_ylabel('Real(H) diff')
    ax_res.legend(fontsize=5, ncol=4)
    ax_res.grid(True)


def _plot_imag_row(ax_raw, ax_res, f_ghz, ref_H, ref_label, items, colors, ds):
    """Row 3: imag-part overlay (left) + imag-part residual (right)."""
    # Left: raw imag
    ax_raw.plot(f_ghz, ref_H.imag, 'k--', linewidth=0.8, label=ref_label)
    for ci, (ly_idx, H_full) in enumerate(items):
        Hd = _block_downsample(H_full, ds)
        ax_raw.plot(f_ghz, Hd.imag, color=colors[ci], linewidth=0.5,
                    label=f'Layer {ly_idx}')
    ax_raw.set_xlabel('Frequency (GHz)')
    ax_raw.set_ylabel('Imag(H)')
    ax_raw.legend(fontsize=5, ncol=4)
    ax_raw.grid(True)

    # Right: imag residual = Im(learned) - Im(ref)
    for ci, (ly_idx, H_full) in enumerate(items):
        Hd = _block_downsample(H_full, ds)
        ax_res.plot(f_ghz, Hd.imag - ref_H.imag, color=colors[ci],
                    linewidth=0.5, label=f'Layer {ly_idx}')
    ax_res.axhline(y=0, color='gray', linewidth=0.5, linestyle=':')
    ax_res.set_xlabel('Frequency (GHz)')
    ax_res.set_ylabel('Imag(H) diff')
    ax_res.legend(fontsize=5, ncol=4)
    ax_res.grid(True)


def plot_h_vs_ideal(model, Nsub, fs_sub, fch, p, cfg,
                    layer_indices=None, downsample=20):
    """
    3x2 figure: learned H vs ideal (physical-params) H.

    Left column: raw overlay.  Right column: residual = learned - ideal.
    """
    f_ghz_full, H_ref = _compute_h_one_step(
        Nsub, fs_sub, fch,
        p['L_span'], p['alpha_dBpm'],
        p['beta2'], p['beta3'],
        cfg['steps_per_span'],
    )
    ds = downsample
    f_ghz = _block_downsample(f_ghz_full, ds)
    H_ref = _block_downsample(H_ref, ds)

    items = _extract_learned_h(model, layer_indices)
    if not items:
        print("  plot_h_vs_ideal: no linear layers found, skipping.")
        return
    colors = plt.cm.viridis(np.linspace(0, 1, len(items)))

    fig, axes = plt.subplots(3, 2, figsize=(10, 8),
                             num='H: Learned vs Ideal')

    _plot_phase_row(axes[0, 0], axes[0, 1], f_ghz, H_ref,
                    'Ideal', items, colors, ds)
    axes[0, 0].set_title('Phase: Learned vs Ideal')
    axes[0, 1].set_title('Phase Residual')

    _plot_real_row(axes[1, 0], axes[1, 1], f_ghz, H_ref,
                   'Ideal', items, colors, ds)
    axes[1, 0].set_title('Real Part: Learned vs Ideal')
    axes[1, 1].set_title('Real Residual')

    _plot_imag_row(axes[2, 0], axes[2, 1], f_ghz, H_ref,
                   'Ideal', items, colors, ds)
    axes[2, 0].set_title('Imag Part: Learned vs Ideal')
    axes[2, 1].set_title('Imag Residual')

    fig.tight_layout()


def plot_h_vs_init(model, Nsub, fs_sub, fch, p, cfg,
                   layer_indices=None, downsample=20):
    """
    3x2 figure: learned H vs initial (DSP-mismatched) H.

    Left column: raw overlay.  Right column: residual = learned - init.
    """
    f_ghz_full, H_ref = _compute_h_one_step(
        Nsub, fs_sub, fch,
        p['L_span'], p['alpha_dBpm'],
        p['beta2_DSP'], p['beta3_DSP'],
        cfg['steps_per_span'],
    )
    ds = downsample
    f_ghz = _block_downsample(f_ghz_full, ds)
    H_ref = _block_downsample(H_ref, ds)

    items = _extract_learned_h(model, layer_indices)
    if not items:
        print("  plot_h_vs_init: no linear layers found, skipping.")
        return
    colors = plt.cm.viridis(np.linspace(0, 1, len(items)))

    fig, axes = plt.subplots(3, 2, figsize=(10, 8),
                             num='H: Learned vs Initial')

    _plot_phase_row(axes[0, 0], axes[0, 1], f_ghz, H_ref,
                    'Init (DSP)', items, colors, ds)
    axes[0, 0].set_title('Phase: Learned vs Initial')
    axes[0, 1].set_title('Phase Residual')

    _plot_real_row(axes[1, 0], axes[1, 1], f_ghz, H_ref,
                   'Init (DSP)', items, colors, ds)
    axes[1, 0].set_title('Real Part: Learned vs Initial')
    axes[1, 1].set_title('Real Residual')

    _plot_imag_row(axes[2, 0], axes[2, 1], f_ghz, H_ref,
                   'Init (DSP)', items, colors, ds)
    axes[2, 0].set_title('Imag Part: Learned vs Initial')
    axes[2, 1].set_title('Imag Residual')

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
