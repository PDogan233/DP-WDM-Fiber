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
    Often used to generate H_ref.

    H(omega) = exp((-alpha/2) * h - j * beta(omega) * h)
    where beta(omega) = 0.5*beta2*omega^2 + (1/6)*beta3*omega^3
    and h = -L_span / steps_per_span (negative = back-propagation).

    Returns (f_ghz, H, h_dbp) where f_ghz is length-Nsub in GHz
    and h_dbp is the (negative) back-propagation step.
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
    return f_ghz, H, h_dbp


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


def _extract_learned_gamma(model, layer_indices=None):
    """
    Extract learned gamma from specified nonlinear layers.

    layer_indices: list of 1-based indices, e.g. [1, 2, 5].
                   If None, extract all nonlinear layers.

    Returns list of (layer_idx, gamma_value) tuples.
    """
    result = []
    idx = 0
    for layer in model.layers:
        if hasattr(layer, 'gamma'):
            idx += 1
            if layer_indices is None or idx in layer_indices:
                result.append((idx, layer.gamma.item()))
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
    ax_raw.set_xlabel('Frequency (GHz)')
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
    ax_res.set_xlabel('Frequency (GHz)')
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
    ax_raw.set_xlabel('Frequency (GHz)')
    ax_raw.set_ylabel('Real(H)')
    ax_raw.legend(fontsize=5, ncol=4)
    ax_raw.grid(True)

    # Right: real residual = Re(learned) - Re(ref)
    for ci, (ly_idx, H_full) in enumerate(items):
        Hd = _block_downsample(H_full, ds)
        ax_res.plot(f_ghz, Hd.real - ref_H.real, color=colors[ci],
                    linewidth=0.5, label=f'Layer {ly_idx}')
    ax_res.axhline(y=0, color='gray', linewidth=0.5, linestyle=':')
    ax_res.set_xlabel('Frequency (GHz)')
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
                    layer_indices=None, downsample=20, fig_suffix=''):
    """
    3x2 figure: learned H vs ideal (physical-params) H.

    Left column: raw overlay.  Right column: residual = learned - ideal.
    """
    f_ghz_full, H_ref, _ = _compute_h_one_step(
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

    fig_num = f'H: Learned vs Ideal{fig_suffix}'
    fig, axes = plt.subplots(3, 2, figsize=(12, 10),
                             num=fig_num,
                             constrained_layout=True)

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


def plot_h_vs_init(model, Nsub, fs_sub, fch, p, cfg,
                   layer_indices=None, downsample=20, fig_suffix=''):
    """
    3x2 figure: learned H vs initial (DSP-mismatched) H.

    Left column: raw overlay.  Right column: residual = learned - init.
    """
    f_ghz_full, H_ref, _ = _compute_h_one_step(
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

    fig_num = f'H: Learned vs Initial{fig_suffix}'
    fig, axes = plt.subplots(3, 2, figsize=(12, 10),
                             num=fig_num,
                             constrained_layout=True)

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


# -------------------------------------------------------------------
# Beta2 estimation helpers (shared pre-computation)
# -------------------------------------------------------------------

def _beta2_est_prep(model, Nsub, fs_sub, fch, p, cfg, layer_indices, downsample):
    """Shared setup: compute frequency grid, H_init, layer phases."""
    f_ghz_full, H_init, h_dbp = _compute_h_one_step(
        Nsub, fs_sub, fch,
        p['L_span'], p['alpha_dBpm'],
        p['beta2_DSP'], p['beta3_DSP'],
        cfg['steps_per_span'],
    )
    ds = downsample
    f_ghz = _block_downsample(f_ghz_full, ds)
    H_init_ds = _block_downsample(H_init, ds)

    items = _extract_learned_h(model, layer_indices)
    if not items:
        return None

    init_phase = np.unwrap(np.angle(H_init_ds))
    omega = 2 * np.pi * f_ghz * 1e9
    weight = (-0.5 * h_dbp) * (omega ** 2)
    domega = omega[1] - omega[0]

    phases = {}
    for ly_idx, H_full in items:
        Hd = _block_downsample(H_full, ds)
        phases[ly_idx] = np.unwrap(np.angle(Hd))

    return {
        'f_ghz': f_ghz, 'omega': omega, 'weight': weight, 'domega': domega,
        'h_dbp': h_dbp, 'H_init_ds': H_init_ds, 'init_phase': init_phase,
        'items': items, 'phases': phases,
    }


def estimate_beta2(model, Nsub, fs_sub, fch, p, cfg,
                   layer_indices=None, downsample=20,
                   fit_fmin_ghz=None, fit_fmax_ghz=None,
                   prev_beta2=None):
    """
    Estimate beta2 from learned H using quadratic fit on phi_learned(w).

    Method (M2): fit  phi_learned = a*w^2 + b*w + c  in [f_min, f_max]
    weighted by |H_init|^2, then  beta2_est = -2a/h_dbp.

    Frequency limits default to system-dependent values (when not set via cfg):
        f_min = Rs / 20       (1.6 GHz @ 32Gbaud)
        f_max = Rs / 3        (10.7 GHz @ 32Gbaud)

    prev_beta2: the beta2 estimate from the PREVIOUS outer loop.
        Used to compute acc_per_est (per-estimation accuracy improvement).
        On the first call, pass None (or omit).

    Returns (beta2_est, delta_est, beta2_std, beta2_acc_overall, beta2_acc_per_est).
    """
    Rs = p['Rs']
    if fit_fmin_ghz is None:
        fit_fmin_ghz = Rs / 20 / 1e9    # Rs / 20 / 1e9
    if fit_fmax_ghz is None:
        fit_fmax_ghz = Rs / 3 / 1e9     # Rs / 3 / 1e9
    if layer_indices is None:
        layer_indices = [1]

    prep = _beta2_est_prep(model, Nsub, fs_sub, fch, p, cfg,
                           layer_indices, downsample)
    if prep is None:
        print("  estimate_beta2: no linear layers found, skipping.")
        return None, None, None, None, None

    omega = prep['omega']; f_ghz = prep['f_ghz']
    h_dbp = prep['h_dbp']; H_init_ds = prep['H_init_ds']
    phases = prep['phases']; items = prep['items']

    mask = (np.abs(f_ghz) >= fit_fmin_ghz) & (np.abs(f_ghz) <= fit_fmax_ghz)
    if mask.sum() < 5:
        print("  estimate_beta2: too few points in fit window, skipping.")
        return None, None, None, None, None
    wt = np.abs(H_init_ds[mask]) ** 2

    # M2: phi_learned quadratic fit
    beta2_vals = []
    for ly_idx, _ in items:
        a, _, _ = np.polyfit(omega[mask], phases[ly_idx][mask], 2, w=wt)
        beta2_vals.append(-2.0 * a / h_dbp)
    beta2_arr = np.array(beta2_vals)
    beta2_est = np.mean(beta2_arr)
    beta2_std = np.std(beta2_arr)

    beta2_true = p['beta2']
    beta2_dsp = p['beta2_DSP']
    delta_true = beta2_true - beta2_dsp
    delta_est = beta2_est - beta2_dsp

    # Accuracy: overall (vs initial DSP) and per-estimation (vs previous estimate)
    err_init = abs(beta2_dsp - beta2_true)
    err_now = abs(beta2_est - beta2_true)
    beta2_acc_overall = (1.0 - err_now / err_init) * 100 if err_init > 0 else 0.0

    beta2_acc_per_est = None
    if prev_beta2 is not None:
        err_prev = abs(prev_beta2 - beta2_true)
        beta2_acc_per_est = (1.0 - err_now / err_prev) * 100 if err_prev > 0 else 0.0
        acc_str = "beta2_acc_overall={:+.2f}%  beta2_acc_per_est={:+.2f}%".format(
            beta2_acc_overall, beta2_acc_per_est)
    else:
        acc_str = "beta2_acc_overall={:+.2f}%  (first est.)".format(beta2_acc_overall)

    print()
    print("-" * 68)
    print("  Beta2 Estimation  (|f| in [{:.1f}, {:.1f}] GHz)".format(
        fit_fmin_ghz, fit_fmax_ghz))
    print("  eta2               = {:+.1f} %".format(p['eta2'] * 100))
    print("  beta2 (DSP init)   = {:.4e}  s^2/m".format(beta2_dsp))
    if prev_beta2 is not None:
        print("  beta2 (prev est)   = {:.4e}  s^2/m".format(prev_beta2))
    print("  beta2 (true)       = {:.4e}  s^2/m".format(beta2_true))
    print("  Delta_beta2 (true) = {:.4e}  s^2/m".format(delta_true))
    print("-" * 68)
    print("  beta2 (estimated)  = {:.4e} +/- {:.4e}  s^2/m".format(
        beta2_est, beta2_std))
    print("  Delta_beta2 (est)  = {:.4e}  s^2/m".format(delta_est))
    print("  {}".format(acc_str))
    print("-" * 68)
    print()

    return beta2_est, delta_est, beta2_std, beta2_acc_overall, beta2_acc_per_est


def estimate_gamma(model, p, layer_indices=None, prev_gamma=None):
    """
    Estimate gamma by directly reading learned NonlinearLayer gamma parameters.

    Computes layer-wise statistics and compares with gamma_DSP (init) and
    gamma_true (physical).

    prev_gamma: the gamma estimate from the PREVIOUS outer loop.
        Used to compute gamma_acc_per_est (per-estimation accuracy improvement).
        On the first call, pass None (or omit).

    Returns (gamma_est, delta_est, gamma_std, gamma_acc_overall, gamma_acc_per_est).
    """
    if layer_indices is None:
        layer_indices = [1]

    items = _extract_learned_gamma(model, layer_indices)
    if not items:
        print("  estimate_gamma: no nonlinear layers found, skipping.")
        return None, None, None, None, None

    gamma_vals = [v for _, v in items]
    gamma_arr = np.array(gamma_vals)
    gamma_est = np.mean(gamma_arr)
    gamma_std = np.std(gamma_arr)

    gamma_true = p['gamma']
    gamma_dsp = p['gamma_DSP']
    delta_true = gamma_true - gamma_dsp
    delta_est = gamma_est - gamma_dsp

    # Accuracy: overall (vs initial DSP) and per-estimation (vs previous estimate)
    err_init = abs(gamma_dsp - gamma_true)
    err_now = abs(gamma_est - gamma_true)
    gamma_acc_overall = (1.0 - err_now / err_init) * 100 if err_init > 0 else 0.0

    gamma_acc_per_est = None
    if prev_gamma is not None:
        err_prev = abs(prev_gamma - gamma_true)
        gamma_acc_per_est = (1.0 - err_now / err_prev) * 100 if err_prev > 0 else 0.0
        acc_str = "gamma_acc_overall={:+.2f}%  gamma_acc_per_est={:+.2f}%".format(
            gamma_acc_overall, gamma_acc_per_est)
    else:
        acc_str = "gamma_acc_overall={:+.2f}%  (first est.)".format(gamma_acc_overall)

    n_layers = len(gamma_arr)
    print()
    print("-" * 68)
    print("  Gamma Estimation  ({} nonlinear layers)".format(n_layers))
    print("  eta4               = {:+.1f} %".format(p['eta4'] * 100))
    print("  gamma (DSP init)   = {:.4e}  1/(W*m)".format(gamma_dsp))
    if prev_gamma is not None:
        print("  gamma (prev est)   = {:.4e}  1/(W*m)".format(prev_gamma))
    print("  gamma (true)       = {:.4e}  1/(W*m)".format(gamma_true))
    print("  Delta_gamma (true) = {:.4e}  1/(W*m)".format(delta_true))
    print("-" * 68)
    print("  gamma (estimated)  = {:.4e} +/- {:.4e}  1/(W*m)".format(
        gamma_est, gamma_std))
    print("  Delta_gamma (est)  = {:.4e}  1/(W*m)".format(delta_est))
    print("  {}".format(acc_str))
    print("-" * 68)
    print()

    return gamma_est, delta_est, gamma_std, gamma_acc_overall, gamma_acc_per_est


def estimate_beta2_m1(prep, p, fit_fmin_ghz, fit_fmax_ghz):
    """M1: regularized division  db2 = dphi / (denom*w^2 + eps)."""
    f_ghz = prep['f_ghz']; omega = prep['omega']; weight = prep['weight']
    init_phase = prep['init_phase']; phases = prep['phases']
    H_init_ds = prep['H_init_ds']; items = prep['items']

    eps_reg = np.max(np.abs(weight))
    mask = (np.abs(f_ghz) >= fit_fmin_ghz) & (np.abs(f_ghz) <= fit_fmax_ghz)
    wt = np.abs(H_init_ds[mask]) ** 2

    vals = []
    for ly_idx, _ in items:
        phase_diff = phases[ly_idx] - init_phase
        db2 = phase_diff[mask] / (weight[mask] + eps_reg)
        vals.append(np.average(db2, weights=wt))
    return np.array(vals)


def estimate_beta2_m3(prep, p, fit_fmin_ghz, fit_fmax_ghz):
    """M3: numerical 2nd-derivative  db2 = -d^2(dphi)/dw^2 / h."""
    f_ghz = prep['f_ghz']; omega = prep['omega']; domega = prep['domega']
    h_dbp = prep['h_dbp']; init_phase = prep['init_phase']
    phases = prep['phases']; H_init_ds = prep['H_init_ds']; items = prep['items']

    mask = (np.abs(f_ghz) >= fit_fmin_ghz) & (np.abs(f_ghz) <= fit_fmax_ghz)
    wt = np.abs(H_init_ds[mask]) ** 2

    vals = []
    for ly_idx, _ in items:
        phase_diff = phases[ly_idx] - init_phase
        d2phi = np.empty_like(phase_diff)
        d2phi[1:-1] = (phase_diff[2:] - 2*phase_diff[1:-1] + phase_diff[:-2]) / (domega**2)
        d2phi[0] = d2phi[1]; d2phi[-1] = d2phi[-2]
        db2 = d2phi[mask] / (-h_dbp)
        vals.append(np.average(db2, weights=wt))
    return np.array(vals)


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
