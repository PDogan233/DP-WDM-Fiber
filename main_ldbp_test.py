import numpy as np
import matplotlib.pyplot as plt
import torch

from para import get_parameters
from tx_DSP import multiplex_wdm_channels
from channel import channel_propagation
from dbp import dbp_subband
from visualize import plot_wav_spec
from ldbp import LDBP
from ldbp_utils import (complex_np_to_torch, extract_subband, run_rx_chain,
                        plot_constellation_grid, plot_h_phase, plot_ber_bars)


# =====================================================================
# Configuration
# =====================================================================

cfg = {
    'steps_per_span': 2,
    'trainable_gamma': True,
    'num_epochs': 300,
    'learning_rate': 1e-3,
    'print_interval': 20,
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
# 2. Generate datasets
# =====================================================================
print("\nGenerating Training Set (seed=42)...")
np.random.seed(42)
tx_wav_train, tx_data_train = multiplex_wdm_channels(p)
rx_wav_ch_train = channel_propagation(tx_wav_train, p)

print("Generating Test Set (seed=99)...")
np.random.seed(99)
tx_wav_test, tx_data_test = multiplex_wdm_channels(p)
rx_wav_ch_test = channel_propagation(tx_wav_test, p)

# =====================================================================
# 3. Subband extraction (once per dataset — captures dimensions)
# =====================================================================
print("\nExtracting subband...")
rx_wav_sub_train, Nsub, fs_sub, sps_sub = extract_subband(rx_wav_ch_train, p, fch)
rx_wav_sub_test, _, _, _ = extract_subband(rx_wav_ch_test, p, fch)
print(f"  Nsub={Nsub}, fs_sub={fs_sub/1e9:.2f} GHz, sps_sub={sps_sub:.2f}")

# =====================================================================
# 4. True DBP labels (matched physical params, fine steps)
# =====================================================================
print("\nRunning True DBP for labels...")
rx_wav_dbp_train_label = dbp_subband(
    rx_wav_sub_train, Nsub, fs_sub, fch,
    p['L_span'], p['alpha_dBpm'],
    p['beta2'], p['beta3'], p['gamma'],
    p['dz_DBP'], p['Nspans'], p['G_lin']
)
rx_wav_dbp_test_label = dbp_subband(
    rx_wav_sub_test, Nsub, fs_sub, fch,
    p['L_span'], p['alpha_dBpm'],
    p['beta2'], p['beta3'], p['gamma'],
    p['dz_DBP'], p['Nspans'], p['G_lin']
)

# =====================================================================
# 5. Convert to torch tensors
# =====================================================================
print("Converting to torch tensors...")
rx_wav_sub_train_ts = complex_np_to_torch(rx_wav_sub_train, DEVICE)
rx_wav_dbp_train_label_ts = complex_np_to_torch(rx_wav_dbp_train_label, DEVICE)
rx_wav_sub_test_ts = complex_np_to_torch(rx_wav_sub_test, DEVICE)
rx_wav_dbp_test_label_ts = complex_np_to_torch(rx_wav_dbp_test_label, DEVICE)
print(f"  rx_wav_sub_train_ts: {rx_wav_sub_train_ts.shape}, {rx_wav_sub_train_ts.dtype}")

# =====================================================================
# 6. Build LDBP model
# =====================================================================
print(f"\nBuilding LDBP: Nspans={p['Nspans']}, "
      f"steps_per_span={cfg['steps_per_span']}, "
      f"trainable_gamma={cfg['trainable_gamma']}")

model = LDBP(
    Nsub=Nsub, fs_sub=fs_sub, fch=fch,
    L_span=p['L_span'], alpha_dBpm=p['alpha_dBpm'],
    beta2=p['beta2'], beta3=p['beta3'], gamma=p['gamma'],
    Nspans=p['Nspans'], steps_per_span=cfg['steps_per_span'],
    G_lin=p['G_lin'], trainable_gamma=cfg['trainable_gamma']
).to(DEVICE)

total_params = sum(pn.numel() for pn in model.parameters())
trainable_params = sum(pn.numel() for pn in model.parameters() if pn.requires_grad)
print(f"  Total parameters: {total_params:,}  |  Trainable: {trainable_params:,}")

# =====================================================================
# 7. Evaluate LDBP at initialization (before any training)
# =====================================================================
print("\n--- LDBP at Initialization ---")
model.eval()
with torch.no_grad():
    rx_wav_ldbp_init_train_ts = model(rx_wav_sub_train_ts)
    rx_wav_ldbp_init_train = rx_wav_ldbp_init_train_ts.cpu().numpy()

    rx_wav_ldbp_init_test_ts = model(rx_wav_sub_test_ts)
    rx_wav_ldbp_init_test = rx_wav_ldbp_init_test_ts.cpu().numpy()

# BER for LDBP at initialization
ber_x_ldbp_init_train, ber_y_ldbp_init_train, rx_sym_ldbp_init_train = run_rx_chain(
    rx_wav_ldbp_init_train, tx_data_train, p, m_center, sps_sub, p['rrc_taps_rx'])
ber_x_ldbp_init_test, ber_y_ldbp_init_test, rx_sym_ldbp_init_test = run_rx_chain(
    rx_wav_ldbp_init_test, tx_data_test, p, m_center, sps_sub, p['rrc_taps_rx'])

# =====================================================================
# 8. Training
# =====================================================================
optimizer = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'])
loss_history = []

model.train()
print(f"\nTraining for {cfg['num_epochs']} epochs...")
for epoch in range(cfg['num_epochs']):
    optimizer.zero_grad()
    output = model(rx_wav_sub_train_ts)
    loss = torch.mean(torch.abs(output - rx_wav_dbp_train_label_ts) ** 2)
    loss.backward()
    optimizer.step()

    loss_history.append(loss.item())

    if (epoch + 1) % cfg['print_interval'] == 0:
        weight_norms = model.get_linear_weight_norms()
        wn_str = ', '.join([f'{wn:.4f}' for wn in weight_norms])
        print(f"  Epoch {epoch+1:4d}/{cfg['num_epochs']}  |  "
              f"Loss: {loss.item():.6e}  |  |H| mean: [{wn_str}]")

print(f"\nTraining done. Loss: {loss_history[0]:.6e} -> {loss_history[-1]:.6e}  "
      f"({loss_history[0]/loss_history[-1]:.1f}x reduction)")

# =====================================================================
# 9. Evaluate LDBP after training
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
ber_x_ldbp_train, ber_y_ldbp_train, rx_sym_ldbp_train = run_rx_chain(
    rx_wav_ldbp_train, tx_data_train, p, m_center, sps_sub, p['rrc_taps_rx'])
ber_x_ldbp_test, ber_y_ldbp_test, rx_sym_ldbp_test = run_rx_chain(
    rx_wav_ldbp_test, tx_data_test, p, m_center, sps_sub, p['rrc_taps_rx'])

# BER for True DBP baseline (test set only)
ber_x_dbp_test_label, ber_y_dbp_test_label, rx_sym_dbp_test_label = run_rx_chain(
    rx_wav_dbp_test_label, tx_data_test, p, m_center, sps_sub, p['rrc_taps_rx'])

# =====================================================================
# 10. BER summary
# =====================================================================
print("\n========== BER Summary ==========")
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
# 11. Save model
# =====================================================================
torch.save({
    'model_state_dict': model.state_dict(),
    'cfg': cfg,
    'loss_history': loss_history,
}, 'ldbp_checkpoint.pth')
print("\nModel saved to ldbp_checkpoint.pth")

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

# 12d. H filter phase: learned vs ideal (all layers overlaid)
plot_h_phase(model, Nsub, fs_sub, fch,
             p['L_span'], p['alpha_dBpm'],
             p['beta2'], p['beta3'], cfg['steps_per_span'])

# 12e. BER bar chart (log scale)
plot_ber_bars([
    ('LDBP init\n(train)', ber_x_ldbp_init_train, ber_y_ldbp_init_train),
    ('LDBP final\n(train)', ber_x_ldbp_train, ber_y_ldbp_train),
    ('LDBP final\n(test)', ber_x_ldbp_test, ber_y_ldbp_test),
    ('True DBP\n(test)', ber_x_dbp_test_label, ber_y_dbp_test_label),
])

plt.ioff()
plt.show()
print("\nLDBP Training Complete.")