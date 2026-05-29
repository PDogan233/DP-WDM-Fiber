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
                        estimate_delta_beta2)
from data_cache import build_cache_path, save_sim_cache, load_sim_cache
from model_cache import build_model_dir, build_model_filename, save_model_cache, load_model_cache


# =====================================================================
# Configuration
# =====================================================================

cfg = {
    'steps_per_span': 5,
    'trainable_gamma': True,
    'num_epochs': 500,  # 6000
    'learning_rate': 1e-3,
    'learning_rate_min': 1e-4,
    'print_interval': 100,
    'use_cache': True,
    'h_plot_layers': [1],
    'h_plot_ds': 500,
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
# 4. Build LDBP model
# =====================================================================
print(f"\nBuilding LDBP: Nspans={p['Nspans']}, "
      f"steps_per_span={cfg['steps_per_span']}, "
      f"trainable_gamma={cfg['trainable_gamma']}")

# LDBP is initialized with MISMATCHED DSP params (the "wrong" starting point).
# It must learn to converge toward the correct (physical) DBP label.
model = LDBP(
    Nsub=Nsub, fs_sub=fs_sub, fch=fch, L_span=p['L_span'], alpha_dBpm=p['alpha_dBpm'],
    beta2=p['beta2_DSP'], beta3=p['beta3_DSP'], gamma=p['gamma_DSP'], Nspans=p['Nspans'], 
    steps_per_span=cfg['steps_per_span'], G_lin=p['G_lin'], trainable_gamma=cfg['trainable_gamma']
).to(DEVICE)

total_params = sum(pn.numel() for pn in model.parameters())
trainable_params = sum(pn.numel() for pn in model.parameters() if pn.requires_grad)
print(f"  Total parameters: {total_params:,}  |  Trainable: {trainable_params:,}")

# =====================================================================
# 5. Check for cached model (skip training if found)
# =====================================================================
model_dir = build_model_dir(p, 42, 99)
model_filename = build_model_filename(cfg, p)
model_path = os.path.join(model_dir, model_filename)

pretrained = False
if cfg.get('use_cache') and os.path.exists(model_path):
    print(f"\nLoading cached model from {model_path}...")
    ckpt = load_model_cache(model_path, cfg, p)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    pretrained = True

    loss_history = ckpt['loss_history']
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

    print(f"  Final loss (train): {final_loss_train:.6e}")
    print(f"  Final loss (test) : {final_loss_test:.6e}")

if not pretrained:

    # =====================================================================
    # 6. Evaluate LDBP at initialization (before any training)
    # =====================================================================
    print("\n--- LDBP at Initialization ---")
    model.eval()
    with torch.no_grad():
        rx_wav_ldbp_init_train_ts = model(rx_wav_sub_train_ts)
        rx_wav_ldbp_init_train = rx_wav_ldbp_init_train_ts.cpu().numpy()
        rx_wav_ldbp_init_test_ts = model(rx_wav_sub_test_ts)
        rx_wav_ldbp_init_test = rx_wav_ldbp_init_test_ts.cpu().numpy()

    ber_x_ldbp_init_train, ber_y_ldbp_init_train, rx_sym_ldbp_init_train = rx_after_dbp(
        rx_wav_ldbp_init_train, tx_data_train, p, m_center, sps_sub, p['rrc_taps_rx'])
    ber_x_ldbp_init_test, ber_y_ldbp_init_test, rx_sym_ldbp_init_test = rx_after_dbp(
        rx_wav_ldbp_init_test, tx_data_test, p, m_center, sps_sub, p['rrc_taps_rx'])

    # =====================================================================
    # 7. Training
    # =====================================================================
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg['num_epochs'], eta_min=cfg['learning_rate_min'])
    loss_history = []

    model.train()
    print(f"\nTraining for {cfg['num_epochs']} epochs...")
    for epoch in range(cfg['num_epochs']):
        optimizer.zero_grad()
        output = model(rx_wav_sub_train_ts)
        loss = torch.mean(torch.abs(output - rx_wav_dbp_train_label_ts) ** 2)
        loss.backward()
        optimizer.step()
        scheduler.step()
        loss_history.append(loss.item())

        if (epoch + 1) % cfg['print_interval'] == 0:
            print(f"  Epoch {epoch+1:4d}/{cfg['num_epochs']}  |  "
                  f"Loss: {loss.item():.6e}  |  lr: {scheduler.get_last_lr()[0]:.2e}")

    print(f"\nTraining done. Loss: {loss_history[0]:.6e} -> {loss_history[-1]:.6e}  "
          f"({loss_history[0]/loss_history[-1]:.1f}x reduction)")

    # =====================================================================
    # 8. Evaluate LDBP after training
    # =====================================================================
    print("\n--- LDBP After Training ---")
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

    print(f"  Final loss (train): {final_loss_train:.6e}")
    print(f"  Final loss (test) : {final_loss_test:.6e}")

    # BER for LDBP after training
    ber_x_ldbp_train, ber_y_ldbp_train, rx_sym_ldbp_train = rx_after_dbp(
        rx_wav_ldbp_train, tx_data_train, p, m_center, sps_sub, p['rrc_taps_rx'])
    ber_x_ldbp_test, ber_y_ldbp_test, rx_sym_ldbp_test = rx_after_dbp(
        rx_wav_ldbp_test, tx_data_test, p, m_center, sps_sub, p['rrc_taps_rx'])

    # BER for True DBP baseline (test set only)
    ber_x_dbp_test_label, ber_y_dbp_test_label, rx_sym_dbp_test_label = rx_after_dbp(
        rx_wav_dbp_test_label, tx_data_test, p, m_center, sps_sub, p['rrc_taps_rx'])

    # =====================================================================
    # 9. Save model checkpoint
    # =====================================================================
    print("\nSaving model checkpoint...")
    save_model_cache(model_path, model, cfg, p,
        loss_history=loss_history,
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
    )

# =====================================================================
# 10. BER summary
# =====================================================================

print("\n--- BER Summary ---")
print(f"  LDBP init  (train): X={ber_x_ldbp_init_train:.3g}, "
      f"Y={ber_y_ldbp_init_train:.3g}")
print(f"  LDBP init  (test) : X={ber_x_ldbp_init_test:.3g}, "
      f"Y={ber_y_ldbp_init_test:.3g}")
print(f"  LDBP final (train): X={ber_x_ldbp_train:.3g}, "
      f"Y={ber_y_ldbp_train:.3g}")
print(f"  LDBP final (test) : X={ber_x_ldbp_test:.3g}, "
      f"Y={ber_y_ldbp_test:.3g}")
print(f"  True DBP   (test) : X={ber_x_dbp_test_label:.3g}, "
      f"Y={ber_y_dbp_test_label:.3g}")

# =====================================================================
# 11. Generate constellation data (re-run evaluation if pretrained)
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
# 12. Plots
# =====================================================================
plt.ion()

# 12a. Waveform & spectrum before/after channel
plot_wav_spec(p['t'], p['f'], tx_wav_train, rx_wav_ch_train)

# 12b. Constellation grid
plot_constellation_grid([
    {'x': rx_sym_ldbp_init_train[:, 0], 'y': rx_sym_ldbp_init_train[:, 1],
     'label': 'LDBP init (train)'},
    {'x': rx_sym_ldbp_train[:, 0], 'y': rx_sym_ldbp_train[:, 1],
     'label': 'LDBP final (train)'},
    {'x': rx_sym_ldbp_test[:, 0], 'y': rx_sym_ldbp_test[:, 1],
     'label': 'LDBP final (test)'},
    {'x': rx_sym_dbp_test_label[:, 0], 'y': rx_sym_dbp_test_label[:, 1],
     'label': 'True DBP (test)'},
])

# 12c. Training loss curve
plt.figure('LDBP Training Loss', figsize=(8, 4))
plt.semilogy(loss_history)
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title(f"LDBP Training (steps_per_span={cfg['steps_per_span']}, "
          f"trainable_gamma={cfg['trainable_gamma']})")
plt.grid(True)
plt.tight_layout()

# 12d. BER bar chart (log scale)
plot_ber_bars([
    ('LDBP init\n(train)', ber_x_ldbp_init_train, ber_y_ldbp_init_train),
    ('LDBP final\n(train)', ber_x_ldbp_train, ber_y_ldbp_train),
    ('LDBP final\n(test)', ber_x_ldbp_test, ber_y_ldbp_test),
    ('True DBP\n(test)', ber_x_dbp_test_label, ber_y_dbp_test_label),
])

# 12e. H filter analysis: learned vs ideal (physical params)
plot_h_vs_ideal(model, Nsub, fs_sub, fch, p, cfg,
                layer_indices=cfg['h_plot_layers'],
                downsample=cfg['h_plot_ds'])

# 12f. H filter analysis: learned vs initial (DSP-mismatched params)
plot_h_vs_init(model, Nsub, fs_sub, fch, p, cfg,
               layer_indices=cfg['h_plot_layers'],
               downsample=cfg['h_plot_ds'])

# 12g. Beta2 estimation summary (three methods, printed table)
estimate_delta_beta2(model, Nsub, fs_sub, fch, p, cfg,
                     layer_indices=cfg['h_plot_layers'],
                     downsample=cfg['h_plot_ds'])

plt.ioff()
plt.show()
print("\nEvaluation Complete.")

