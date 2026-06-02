import os

import numpy as np
import matplotlib.pyplot as plt
import torch

from para import get_parameters
from tx_DSP import multiplex_wdm_channels
from channel import channel_propagation
from dbp import dbp_subband
from visualize import plot_wav_spec
from ldbp import LDBP
from ldbp_utils import (complex_np_to_torch, extract_subband, rx_after_dbp,
                        plot_constellation_grid, plot_ber_bars,
                        plot_h_vs_ideal, plot_h_vs_init,
                        estimate_beta2, estimate_gamma)
from data_cache import build_cache_path, save_sim_cache, load_sim_cache
from model_cache import (build_model_dir, build_model_filename_prdbp,
                        save_model_cache, load_model_cache, CURRENT_CKPT_VERSION)


# =====================================================================
# Configuration
# =====================================================================

cfg = {
    'steps_per_span': 10,

    'N_est': 1,                 # outer loop iterations
    'N_ep_per_est': 1000,        # epochs per inner loop
    'learning_rate': 1e-3,
    'learning_rate_min': 1e-4,

    'trainable_gamma': False,
    'trainable_beta2': True,
    'use_cache': True,

    # Max number of linear layers to select for H plot and estimation.
    # Layers are picked uniformly across all linear layers.
    'h_plot_max_layers': 10,
    'print_interval': 100,
    'h_plot_ds': 500,
    'debug_h_plot': 0,          # 1=plot H after each N_est, 0=only final
}

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# =====================================================================
# 1. Load parameters
# =====================================================================
p = get_parameters()

# Center channel only
m_center = p['Nch'] // 2
fch = p['ch_idx'][m_center] * p['DeltaF']
print(f"\nCenter channel: m={m_center}, fch={fch/1e9:.2f} GHz")

# =====================================================================
# 2. Generate or load cached datasets (channel SSFM and label DBP)
# =====================================================================
cache_path = build_cache_path(p, 42, 99)

if cfg.get('use_cache') and os.path.exists(cache_path):
    print(f"\nLoading cached data from {cache_path}...")
    c = load_sim_cache(cache_path, p)
    tx_wav_train = c['tx_wav_train']
    tx_wav_test = c['tx_wav_test']
    rx_wav_ch_train = c['rx_wav_ch_train']
    rx_wav_ch_test = c['rx_wav_ch_test']
    rx_wav_sub_train = c['rx_wav_sub_train']
    rx_wav_sub_test = c['rx_wav_sub_test']
    Nsub = c['Nsub'].item()
    fs_sub = c['fs_sub'].item()
    sps_sub = c['sps_sub'].item()
    rx_wav_dbp_train_label = c['rx_wav_dbp_train_label']
    rx_wav_dbp_test_label = c['rx_wav_dbp_test_label']
    tx_data_train = c['tx_data_train'].item()
    tx_data_test = c['tx_data_test'].item()
    print(f"  Nsub={Nsub}, fs_sub={fs_sub/1e9:.2f} GHz, sps_sub={sps_sub:.2f}")

else:
    if cfg.get('use_cache'): print("\nNo cache found. Generating data...")
    else: print("\nuse_cache=False. Generating data...")

    print("Generating Training Set (seed=42)...")
    np.random.seed(42)
    tx_wav_train, tx_data_train = multiplex_wdm_channels(p)
    rx_wav_ch_train = channel_propagation(tx_wav_train, p)

    print("Generating Test Set (seed=99)...")
    np.random.seed(99)
    tx_wav_test, tx_data_test = multiplex_wdm_channels(p)
    rx_wav_ch_test = channel_propagation(tx_wav_test, p)

    # Subband extraction
    print("\nExtracting subband...")
    rx_wav_sub_train, Nsub, fs_sub, sps_sub = extract_subband(rx_wav_ch_train, p, fch)
    rx_wav_sub_test, _, _, _ = extract_subband(rx_wav_ch_test, p, fch)
    print(f"  Nsub={Nsub}, fs_sub={fs_sub/1e9:.2f} GHz, sps_sub={sps_sub:.2f}")

    # True DBP labels
    print("Running True DBP for labels...")
    rx_wav_dbp_train_label = dbp_subband(rx_wav_sub_train, Nsub, fs_sub, fch,p['L_span'], p['alpha_dBpm'],
        p['beta2'], p['beta3'], p['gamma'],p['dz_DBP'], p['Nspans'], p['G_lin'])
    rx_wav_dbp_test_label = dbp_subband(rx_wav_sub_test, Nsub, fs_sub, fch,p['L_span'], p['alpha_dBpm'],
        p['beta2'], p['beta3'], p['gamma'],p['dz_DBP'], p['Nspans'], p['G_lin'])

    # Cache for next run
    if cfg.get('use_cache'):
        print("Caching simulation data...")
        save_sim_cache(cache_path, p,
            tx_wav_train=tx_wav_train,
            tx_wav_test=tx_wav_test,
            tx_data_train=tx_data_train,
            tx_data_test=tx_data_test,
            rx_wav_ch_train=rx_wav_ch_train,
            rx_wav_ch_test=rx_wav_ch_test,
            rx_wav_sub_train=rx_wav_sub_train,
            rx_wav_sub_test=rx_wav_sub_test,
            Nsub=np.array(Nsub), fs_sub=np.array(fs_sub), sps_sub=np.array(sps_sub),
            rx_wav_dbp_train_label=rx_wav_dbp_train_label,
            rx_wav_dbp_test_label=rx_wav_dbp_test_label,
        )

# =====================================================================
# 3. Convert to torch tensors
# =====================================================================
print("Converting to torch tensors...")
rx_wav_sub_train_ts = complex_np_to_torch(rx_wav_sub_train, DEVICE)
rx_wav_dbp_train_label_ts = complex_np_to_torch(rx_wav_dbp_train_label, DEVICE)
rx_wav_sub_test_ts = complex_np_to_torch(rx_wav_sub_test, DEVICE)
rx_wav_dbp_test_label_ts = complex_np_to_torch(rx_wav_dbp_test_label, DEVICE)
print(f"  rx_wav_sub_train_ts: {rx_wav_sub_train_ts.shape}, {rx_wav_sub_train_ts.dtype}")

# =====================================================================
# 4. Build model
# =====================================================================
print(f"\nPRDBP info: Nspans={p['Nspans']}, "
      f"steps_per_span={cfg['steps_per_span']}, "
      f"trainable_gamma={cfg['trainable_gamma']}, "
      f"trainable_beta2={cfg['trainable_beta2']}")

# Model is initialized with MISMATCHED DSP params (the "wrong" starting point).
# PRDBP must learn to converge toward the correct (physical) DBP label.
model = LDBP(
    Nsub=Nsub, fs_sub=fs_sub, fch=fch, L_span=p['L_span'], alpha_dBpm=p['alpha_dBpm'],
    beta2=p['beta2_DSP'], beta3=p['beta3_DSP'], gamma=p['gamma_DSP'], Nspans=p['Nspans'], 
    steps_per_span=cfg['steps_per_span'], G_lin=p['G_lin'],
    trainable_gamma=cfg['trainable_gamma'], trainable_beta2=cfg['trainable_beta2']
).to(DEVICE)

total_params = sum(pn.numel() for pn in model.parameters())
trainable_params = sum(pn.numel() for pn in model.parameters() if pn.requires_grad)
print(f"  Total parameters: {total_params:,}  |  Trainable: {trainable_params:,}")

# Auto-select linear layer indices for H plot and estimation
n_linear = sum(1 for ly in model.layers if hasattr(ly, 'H_real'))
max_layers = cfg['h_plot_max_layers']
if n_linear <= max_layers:
    cfg['h_plot_layers'] = list(range(1, n_linear + 1))
else:
    cfg['h_plot_layers'] = list(np.linspace(1, n_linear, max_layers, dtype=int))
print(f"  H-plot/estimation layers: {cfg['h_plot_layers']}  "
      f"(out of {n_linear} linear layers)")

# =====================================================================
# 5. Check for cached PRDBP model (skip training if found)
# =====================================================================
model_dir = build_model_dir(p, 42, 99)
model_filename = build_model_filename_prdbp(cfg, p)
model_path = os.path.join(model_dir, model_filename)

pretrained = False
if cfg.get('use_cache') and os.path.exists(model_path):
    print(f"\nLoading cached PRDBP model from {model_path}...")
    try:
        ckpt = load_model_cache(model_path, cfg, p)
        if ckpt.get('ckpt_version') != CURRENT_CKPT_VERSION:
            raise ValueError(f"version mismatch "
                             f"(got {ckpt.get('ckpt_version')}, "
                             f"need {CURRENT_CKPT_VERSION})")

        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
        pretrained = True

        all_loss_history = ckpt['all_loss_history']
        loss_per_est_history = ckpt['loss_per_est_history']
        final_loss_train = ckpt['final_loss_train']
        final_loss_test = ckpt['final_loss_test']
        ber_x_ldbp_init_train = ckpt['ber_x_ldbp_init_train']
        ber_y_ldbp_init_train = ckpt['ber_y_ldbp_init_train']
        ber_x_ldbp_init_test = ckpt['ber_x_ldbp_init_test']
        ber_y_ldbp_init_test = ckpt['ber_y_ldbp_init_test']
        ber_x_ldbp_train = ckpt['ber_x_ldbp_train']
        ber_y_ldbp_train = ckpt['ber_y_ldbp_train']
        ber_x_ldbp_test = ckpt['ber_x_ldbp_test']
        ber_y_ldbp_test = ckpt['ber_y_ldbp_test']
        ber_x_dbp_test_label = ckpt['ber_x_dbp_test_label']
        ber_y_dbp_test_label = ckpt['ber_y_dbp_test_label']
        rx_sym_ldbp_init_train = ckpt['rx_sym_ldbp_init_train']
        rx_sym_ldbp_init_test = ckpt['rx_sym_ldbp_init_test']

        beta2_est_history = ckpt['beta2_est_history']
        gamma_est_history = ckpt['gamma_est_history']
        beta2_acc_overall_history = ckpt['beta2_acc_overall_history']
        gamma_acc_overall_history = ckpt['gamma_acc_overall_history']
        beta2_acc_per_est_history = ckpt['beta2_acc_per_est_history']
        gamma_acc_per_est_history = ckpt['gamma_acc_per_est_history']

        print(f"  Init loss  (train): {loss_per_est_history[0]:.6e}")
        print(f"  Per-N_est loss:  "
              + "  |  ".join(f"N_est{i}={loss_per_est_history[i+1]:.4e}"
                              for i in range(len(loss_per_est_history) - 1)))
        print(f"  Final loss (train): {final_loss_train:.6e}")
        print(f"  Final loss (test) : {final_loss_test:.6e}")
        

        beta2_true = p['beta2']
        beta2_dsp = p['beta2_DSP']
        gamma_true = p['gamma']
        gamma_dsp = p['gamma_DSP']

        print(f"\n--- Parameter Estimation History (from cache) ---")
        N_est_loaded = len(beta2_est_history) - 1  # exclude DSP init entry
        for i in range(N_est_loaded):
            b2 = beta2_est_history[i + 1]       # skip DSP init at index 0
            g = gamma_est_history[i + 1]
            b2_acc = beta2_acc_overall_history[i]
            g_acc = gamma_acc_overall_history[i]
            b2_acc_per_est = beta2_acc_per_est_history[i]
            g_acc_per_est = gamma_acc_per_est_history[i]
            b2_per_est_str = f"acc_per_est={b2_acc_per_est:+.2f}%" if b2_acc_per_est is not None else "first est."
            g_per_est_str = f"acc_per_est={g_acc_per_est:+.2f}%" if g_acc_per_est is not None else "first est."
            print(f"  N_est {i+1}/{N_est_loaded}:\n"
                  f"    beta2={b2:.4e} (acc_overall={b2_acc:+.2f}%, {b2_per_est_str})\n"
                  f"    gamma={g:.4e} (acc_overall={g_acc:+.2f}%, {g_per_est_str})")

    except (ValueError, KeyError) as e:
        print(f"  Checkpoint incompatible ({e}), will re-train.")

if not pretrained:

    # =====================================================================
    # 6. PRDBP double loop: reinit -> train -> estimate -> repeat
    # =====================================================================
    print(f"\n{'='*68}")
    print(f"  PRDBP Training: N_est={cfg['N_est']}, "
          f"N_ep_per_est={cfg['N_ep_per_est']}")
    print(f"{'='*68}")

    # Pre-populate histories with DSP init as starting point.
    # This makes the loop body uniform: always reinitialize from history[-1],
    # train, then append new estimate.
    beta2_est_history = [p['beta2_DSP']]
    gamma_est_history = [p['gamma_DSP']]
    beta2_acc_overall_history = []
    gamma_acc_overall_history = []
    beta2_acc_per_est_history = []
    gamma_acc_per_est_history = []

    all_loss_history = []
    loss_per_est_history = []

    for n_est in range(cfg['N_est']):

        # ---- 6a. Reinitialize model (uniform for all iterations) ----
        if n_est == 0:
            # Model already built with DSP params in Section 4.
            # Run initial evaluation here (BER before any training).
            print(f"\n{'='*68}")
            print(" PRDBP at Initialization")
            print(f"{'='*68}")
            model.eval()
            with torch.no_grad():
                rx_wav_ldbp_init_train_ts = model(rx_wav_sub_train_ts)
                rx_wav_ldbp_init_train = rx_wav_ldbp_init_train_ts.cpu().numpy()
                rx_wav_ldbp_init_test_ts = model(rx_wav_sub_test_ts)
                rx_wav_ldbp_init_test = rx_wav_ldbp_init_test_ts.cpu().numpy()

            # Compute initial loss (before any training)
            loss_init = torch.mean(
                torch.abs(rx_wav_ldbp_init_train_ts - rx_wav_dbp_train_label_ts) ** 2).item()
            loss_per_est_history.append(loss_init)

            ber_x_ldbp_init_train, ber_y_ldbp_init_train, rx_sym_ldbp_init_train = rx_after_dbp(
                rx_wav_ldbp_init_train, tx_data_train, p, m_center, sps_sub, p['rrc_taps_rx'])
            ber_x_ldbp_init_test, ber_y_ldbp_init_test, rx_sym_ldbp_init_test = rx_after_dbp(
                rx_wav_ldbp_init_test, tx_data_test, p, m_center, sps_sub, p['rrc_taps_rx'])
            print(f"  Init loss (train)  : {loss_init:.6e}")
            print(f"  Init BER (train): X={ber_x_ldbp_init_train:.3g}, "
                  f"Y={ber_y_ldbp_init_train:.3g}")
            print(f"  Init BER (test) : X={ber_x_ldbp_init_test:.3g}, "
                  f"Y={ber_y_ldbp_init_test:.3g}")
        else:
            model.reinitialize_linear_layers(beta2_est_history[-1], p['beta3_DSP'])
            model.reinitialize_nonlinear_layers(gamma_est_history[-1])
            print(f"\n{'='*68}")
            print(f"  N_est {n_est+1}/{cfg['N_est']}: Reinitialized with "
                  f"beta2={beta2_est_history[-1]:.4e}, gamma={gamma_est_history[-1]:.4e}")
            print(f"{'='*68}")

        # ---- 6b. Train inner loop ----
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg['N_ep_per_est'], eta_min=cfg['learning_rate_min'])
        loss_history = []

        model.train()
        for epoch in range(cfg['N_ep_per_est']):
            optimizer.zero_grad()
            output = model(rx_wav_sub_train_ts)
            loss = torch.mean(torch.abs(output - rx_wav_dbp_train_label_ts) ** 2)
            loss.backward()
            optimizer.step()
            scheduler.step()
            loss_history.append(loss.item())

            if (epoch + 1) % cfg['print_interval'] == 0:
                print(f"  Epoch {epoch+1:4d}/{cfg['N_ep_per_est']}  |  "
                      f"Loss: {loss.item():.6e}  |  lr: {scheduler.get_last_lr()[0]:.2e}")
        all_loss_history.extend(loss_history)
        loss_per_est_history.append(loss_history[-1])

        total_ep = (n_est + 1) * cfg['N_ep_per_est']
        print(f"  Inner loop done. Loss: {loss_history[0]:.6e} -> {loss_history[-1]:.6e}  "
              f"(total epochs so far: {total_ep})")

        # ---- 6c. Parameter estimation ----
        model.eval()
        prev_b2 = beta2_est_history[-1]
        prev_g = gamma_est_history[-1]

        if cfg['trainable_beta2']:
            (beta2_est, delta_b2, beta2_std, beta2_acc_overall, beta2_acc_per_est) = \
                estimate_beta2(model, Nsub, fs_sub, fch, p, cfg,
                               layer_indices=cfg['h_plot_layers'],
                               downsample=cfg['h_plot_ds'],
                               fit_fmin_ghz=(p['Rs']/20/1e9), fit_fmax_ghz=(p['Rs']/5/1e9),
                               prev_beta2=prev_b2)
        else:
            beta2_est = prev_b2
            beta2_acc_overall = 0.0
            beta2_acc_per_est = None

        if cfg['trainable_gamma']:
            (gamma_est, delta_g, gamma_std, gamma_acc_overall, gamma_acc_per_est) = \
                estimate_gamma(model, p, layer_indices=cfg['h_plot_layers'],
                               prev_gamma=prev_g)
        else:
            gamma_est = prev_g
            gamma_acc_overall = 0.0
            gamma_acc_per_est = None

        # ---- 6d. Record ----
        beta2_est_history.append(beta2_est)
        gamma_est_history.append(gamma_est)
        beta2_acc_overall_history.append(beta2_acc_overall)
        gamma_acc_overall_history.append(gamma_acc_overall)
        beta2_acc_per_est_history.append(beta2_acc_per_est)
        gamma_acc_per_est_history.append(gamma_acc_per_est)

        # ---- 6e. Debug H plots (per-iteration) ----
        if cfg['debug_h_plot']:
            suffix = f' (N_est {n_est+1})'
            plot_h_vs_ideal(model, Nsub, fs_sub, fch, p, cfg,
                            layer_indices=cfg['h_plot_layers'],
                            downsample=cfg['h_plot_ds'],
                            fig_suffix=suffix)
            plot_h_vs_init(model, Nsub, fs_sub, fch, p, cfg,
                           layer_indices=cfg['h_plot_layers'],
                           downsample=cfg['h_plot_ds'],
                           fig_suffix=suffix)

        # ---- 6f. Iteration summary ----
        b2_str = f"beta2_est={beta2_est:.4e} (acc_overall={beta2_acc_overall:+.2f}%"
        if beta2_acc_per_est is not None:
            b2_str += f", acc_per_est={beta2_acc_per_est:+.2f}%)"
        else:
            b2_str += ")"
        g_str = f"gamma_est={gamma_est:.4e} (acc_overall={gamma_acc_overall:+.2f}%"
        if gamma_acc_per_est is not None:
            g_str += f", acc_per_est={gamma_acc_per_est:+.2f}%)"
        else:
            g_str += ")"
        print(f"\n  N_est {n_est+1}/{cfg['N_est']} summary:\n"
              f"    {b2_str}\n"
              f"    {g_str}")

    # =====================================================================
    # 7. Final evaluation (after all N_est iterations)
    # =====================================================================
    print(f"\n{'='*68}")
    print("  PRDBP Final Evaluation")
    print(f"{'='*68}")
    model.eval()
    with torch.no_grad():
        # Training set
        rx_wav_ldbp_train_ts = model(rx_wav_sub_train_ts)
        final_loss_train = torch.mean(
            torch.abs(rx_wav_ldbp_train_ts - rx_wav_dbp_train_label_ts) ** 2).item()
        rx_wav_ldbp_train = rx_wav_ldbp_train_ts.cpu().numpy()
        # Test set
        rx_wav_ldbp_test_ts = model(rx_wav_sub_test_ts)
        final_loss_test = torch.mean(
            torch.abs(rx_wav_ldbp_test_ts - rx_wav_dbp_test_label_ts) ** 2).item()
        rx_wav_ldbp_test = rx_wav_ldbp_test_ts.cpu().numpy()

    print(f"  Init loss  (train): {loss_per_est_history[0]:.6e}")
    print(f"  Per-N_est loss:  "
          + "  |  ".join(f"N_est{i}={loss_per_est_history[i+1]:.4e}"
                          for i in range(cfg['N_est'])))
    print(f"  Final loss (train): {final_loss_train:.6e}")
    print(f"  Final loss (test) : {final_loss_test:.6e}")
    

    # BER for PRDBP after training
    ber_x_ldbp_train, ber_y_ldbp_train, rx_sym_ldbp_train = rx_after_dbp(
        rx_wav_ldbp_train, tx_data_train, p, m_center, sps_sub, p['rrc_taps_rx'])
    ber_x_ldbp_test, ber_y_ldbp_test, rx_sym_ldbp_test = rx_after_dbp(
        rx_wav_ldbp_test, tx_data_test, p, m_center, sps_sub, p['rrc_taps_rx'])

    # BER for True DBP baseline (test set only)
    ber_x_dbp_test_label, ber_y_dbp_test_label, rx_sym_dbp_test_label = rx_after_dbp(
        rx_wav_dbp_test_label, tx_data_test, p, m_center, sps_sub, p['rrc_taps_rx'])

    # =====================================================================
    # 8. Save model checkpoint
    # =====================================================================
    print("\nSaving PRDBP model checkpoint...")
    save_model_cache(model_path, model, cfg, p,
        all_loss_history=all_loss_history,
        loss_per_est_history=loss_per_est_history,
        final_loss_train=final_loss_train,
        final_loss_test=final_loss_test,
        ber_x_ldbp_init_train=ber_x_ldbp_init_train,
        ber_y_ldbp_init_train=ber_y_ldbp_init_train,
        ber_x_ldbp_init_test=ber_x_ldbp_init_test,
        ber_y_ldbp_init_test=ber_y_ldbp_init_test,
        ber_x_ldbp_train=ber_x_ldbp_train,
        ber_y_ldbp_train=ber_y_ldbp_train,
        ber_x_ldbp_test=ber_x_ldbp_test,
        ber_y_ldbp_test=ber_y_ldbp_test,
        ber_x_dbp_test_label=ber_x_dbp_test_label,
        ber_y_dbp_test_label=ber_y_dbp_test_label,
        rx_sym_ldbp_init_train=rx_sym_ldbp_init_train,
        rx_sym_ldbp_init_test=rx_sym_ldbp_init_test,
        beta2_est_history=beta2_est_history,
        gamma_est_history=gamma_est_history,
        beta2_acc_overall_history=beta2_acc_overall_history,
        gamma_acc_overall_history=gamma_acc_overall_history,
        beta2_acc_per_est_history=beta2_acc_per_est_history,
        gamma_acc_per_est_history=gamma_acc_per_est_history,
    )

# =====================================================================
# 9. BER summary
# =====================================================================

print("\n--- BER Summary ---")
print(f"  LDBP init  (train): X={ber_x_ldbp_init_train:.3g}, "
      f"Y={ber_y_ldbp_init_train:.3g}")
print(f"  LDBP init  (test) : X={ber_x_ldbp_init_test:.3g}, "
      f"Y={ber_y_ldbp_init_test:.3g}")
print(f"  PRDBP final (train): X={ber_x_ldbp_train:.3g}, "
      f"Y={ber_y_ldbp_train:.3g}")
print(f"  PRDBP final (test) : X={ber_x_ldbp_test:.3g}, "
      f"Y={ber_y_ldbp_test:.3g}")
print(f"  True DBP    (test) : X={ber_x_dbp_test_label:.3g}, "
      f"Y={ber_y_dbp_test_label:.3g}")

# =====================================================================
# 10. Generate constellation data (re-run evaluation if pretrained)
# =====================================================================
if pretrained:
    print("\nRunning evaluation for plots...")
    model.eval()
    with torch.no_grad():
        rx_wav_ldbp_train_ts = model(rx_wav_sub_train_ts)
        rx_wav_ldbp_train = rx_wav_ldbp_train_ts.cpu().numpy()
        rx_wav_ldbp_test_ts = model(rx_wav_sub_test_ts)
        rx_wav_ldbp_test = rx_wav_ldbp_test_ts.cpu().numpy()

    _, _, rx_sym_ldbp_train = rx_after_dbp(
        rx_wav_ldbp_train, tx_data_train, p, m_center, sps_sub, p['rrc_taps_rx'])
    _, _, rx_sym_ldbp_test = rx_after_dbp(
        rx_wav_ldbp_test, tx_data_test, p, m_center, sps_sub, p['rrc_taps_rx'])
    _, _, rx_sym_dbp_test_label = rx_after_dbp(
        rx_wav_dbp_test_label, tx_data_test, p, m_center, sps_sub, p['rrc_taps_rx'])

# =====================================================================
# 11. Plots
# =====================================================================
plt.ion()

# a. Constellation grid
plot_constellation_grid([
    {'x': rx_sym_ldbp_init_train[:, 0], 'y': rx_sym_ldbp_init_train[:, 1],
     'label': 'LDBP init (train)'},
    {'x': rx_sym_ldbp_train[:, 0], 'y': rx_sym_ldbp_train[:, 1],
     'label': 'PRDBP final (train)'},
    {'x': rx_sym_ldbp_test[:, 0], 'y': rx_sym_ldbp_test[:, 1],
     'label': 'PRDBP final (test)'},
    {'x': rx_sym_dbp_test_label[:, 0], 'y': rx_sym_dbp_test_label[:, 1],
     'label': 'True DBP (test)'},
])

# b. Training loss curve (concatenated across all N_est)
plt.figure('PRDBP Training Loss', figsize=(10, 4))
plt.semilogy(all_loss_history)
# Vertical lines at N_est boundaries
for i in range(1, cfg['N_est']):
    plt.axvline(x=i * cfg['N_ep_per_est'], color='gray', linestyle='--',
                linewidth=0.8, alpha=0.6)
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title(f"PRDBP Training (N_est={cfg['N_est']}, "
          f"N_ep_per_est={cfg['N_ep_per_est']}, "
          f"steps_per_span={cfg['steps_per_span']}, "
          f"trainable_G={cfg['trainable_gamma']}, trainable_B2={cfg['trainable_beta2']})")
plt.grid(True)
plt.tight_layout()

# c. BER bar chart (log scale)
plot_ber_bars([
    ('LDBP init\n(train)', ber_x_ldbp_init_train, ber_y_ldbp_init_train),
    ('PRDBP final\n(train)', ber_x_ldbp_train, ber_y_ldbp_train),
    ('PRDBP final\n(test)', ber_x_ldbp_test, ber_y_ldbp_test),
    ('True DBP\n(test)', ber_x_dbp_test_label, ber_y_dbp_test_label),
])

# d. Parameter estimation curves (independent figures, only if trainable)
n_est_axis = np.arange(len(beta2_est_history))  # 0..N_est (0 = DSP init)

if cfg['trainable_beta2']:
    plt.figure('PRDBP Beta2 Estimation', figsize=(8, 4))
    plt.axhline(y=p['beta2'], color='green', linestyle='-', linewidth=1.0,
                label=f"True = {p['beta2']:.4e}")
    plt.axhline(y=p['beta2_DSP'], color='red', linestyle='--', linewidth=0.8,
                label=f"DSP = {p['beta2_DSP']:.4e}")
    plt.plot(n_est_axis, beta2_est_history, 'bo-', markersize=4, linewidth=1.0,
             label='Estimated')
    plt.xlabel('N_est iteration')
    plt.ylabel('beta2 (s^2/m)')
    plt.title('PRDBP Beta2 Estimation')
    plt.legend(fontsize=7)
    plt.grid(True)
    plt.xticks(n_est_axis)
    plt.tight_layout()

if cfg['trainable_gamma']:
    plt.figure('PRDBP Gamma Estimation', figsize=(8, 4))
    plt.axhline(y=p['gamma'], color='green', linestyle='-', linewidth=1.0,
                label=f"True = {p['gamma']:.4e}")
    plt.axhline(y=p['gamma_DSP'], color='red', linestyle='--', linewidth=0.8,
                label=f"DSP = {p['gamma_DSP']:.4e}")
    plt.plot(n_est_axis, gamma_est_history, 'bo-', markersize=4, linewidth=1.0,
             label='Estimated')
    plt.xlabel('N_est iteration')
    plt.ylabel('gamma (1/(W*m))')
    plt.title('PRDBP Gamma Estimation')
    plt.legend(fontsize=7)
    plt.grid(True)
    plt.xticks(n_est_axis)
    plt.tight_layout()

# e. H filter analysis
plot_h_vs_ideal(model, Nsub, fs_sub, fch, p, cfg,
                layer_indices=cfg['h_plot_layers'],
                downsample=cfg['h_plot_ds'])
plot_h_vs_init(model, Nsub, fs_sub, fch, p, cfg,
               layer_indices=cfg['h_plot_layers'],
               downsample=cfg['h_plot_ds'])

plt.ioff()
plt.show()
print("\nEvaluation Complete.")

