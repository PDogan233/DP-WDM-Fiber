import numpy as np
import matplotlib.pyplot as plt
import torch

from para import get_parameters
from tx_DSP import multiplex_wdm_channels
from channel import channel_propagation
from visualize import plot_wav_spec
from ldbp import LDBP
from ldbp_utils import (extract_subband, prepare_data, run_rx_chain,
                        plot_constellation_grid, plot_weight_curves,
                        plot_ber_bars)


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
# 3. Determine subband dimensions from train set
# =====================================================================
print("\nExtracting subband to determine dimensions...")
_, Nsub, fs_sub, sps_sub = extract_subband(rx_wav_ch_train, p, fch)
print(f"  Nsub={Nsub}, fs_sub={fs_sub/1e9:.2f} GHz, sps_sub={sps_sub:.2f}")

# =====================================================================
# 4. Prepare data: subband + true DBP label + torch conversion
# =====================================================================
print("\nRunning True DBP (matched params) for labels...")

rx_wav_sub_train_ts, rx_wav_dbp_train_label, rx_wav_dbp_train_label_ts = \
    prepare_data(rx_wav_ch_train, p, fch, Nsub, fs_sub, sps_sub, DEVICE)

rx_wav_sub_test_ts, rx_wav_dbp_test_label, rx_wav_dbp_test_label_ts = \
    prepare_data(rx_wav_ch_test, p, fch, Nsub, fs_sub, sps_sub, DEVICE)

print(f"  rx_wav_sub_train_ts shape: {rx_wav_sub_train_ts.shape}, "
      f"dtype: {rx_wav_sub_train_ts.dtype}")

# =====================================================================
# 5. Build LDBP model
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
# 6. Evaluate LDBP BEFORE training
# =====================================================================
print("\n--- Evaluating LDBP BEFORE training ---")
model.eval()
with torch.no_grad():
    # Training set
    rx_wav_ldbp_init_train_ts = model(rx_wav_sub_train_ts)
    init_loss_train = torch.mean(
        torch.abs(rx_wav_ldbp_init_train_ts - rx_wav_dbp_train_label_ts) ** 2).item()
    rx_wav_ldbp_init_train = rx_wav_ldbp_init_train_ts.cpu().numpy()

    # Test set
    rx_wav_ldbp_init_test_ts = model(rx_wav_sub_test_ts)
    init_loss_test = torch.mean(
        torch.abs(rx_wav_ldbp_init_test_ts - rx_wav_dbp_test_label_ts) ** 2).item()
    rx_wav_ldbp_init_test = rx_wav_ldbp_init_test_ts.cpu().numpy()

print(f"  Initial loss (train): {init_loss_train:.6e}")
print(f"  Initial loss (test) : {init_loss_test:.6e}")

# RX chain on LDBP init output
ber_x_ldbp_init_train, ber_y_ldbp_init_train, rx_sym_ldbp_init_train = run_rx_chain(
    rx_wav_ldbp_init_train, tx_data_train, p, m_center, sps_sub, p['rrc_taps_rx'])
ber_x_ldbp_init_test, ber_y_ldbp_init_test, rx_sym_ldbp_init_test = run_rx_chain(
    rx_wav_ldbp_init_test, tx_data_test, p, m_center, sps_sub, p['rrc_taps_rx'])

# =====================================================================
# 7. Training
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

final_loss = loss_history[-1]
print(f"\nFinal loss: {final_loss:.6e}  |  "
      f"Reduction: {init_loss_train/final_loss:.2f}x")

# =====================================================================
# 8. Evaluate LDBP AFTER training
# =====================================================================
print("\n--- Evaluating LDBP AFTER training ---")
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

# RX chain on LDBP final output (train + test)
ber_x_ldbp_train, ber_y_ldbp_train, rx_sym_ldbp_train = run_rx_chain(
    rx_wav_ldbp_train, tx_data_train, p, m_center, sps_sub, p['rrc_taps_rx'])
ber_x_ldbp_test, ber_y_ldbp_test, rx_sym_ldbp_test = run_rx_chain(
    rx_wav_ldbp_test, tx_data_test, p, m_center, sps_sub, p['rrc_taps_rx'])

# RX chain for True DBP baseline (test set)
ber_x_dbp_test_label, ber_y_dbp_test_label, rx_sym_dbp_test_label = run_rx_chain(
    rx_wav_dbp_test_label, tx_data_test, p, m_center, sps_sub, p['rrc_taps_rx'])

# =====================================================================
# 9. BER summary
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
# 10. Save model
# =====================================================================
torch.save({
    'model_state_dict': model.state_dict(),
    'cfg': cfg,
    'loss_history': loss_history,
}, 'ldbp_checkpoint.pth')
print("\nModel saved to ldbp_checkpoint.pth")

# =====================================================================
# 11. Plots
# =====================================================================
plt.ion()

# 11a. Waveform & spectrum before/after channel
plot_wav_spec(p['t'], p['f'], tx_wav_train, rx_wav_ch_train)

# 11b. Constellation grid (X/Y-pol x 4 conditions)
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

# 11c. Training loss curve
plt.figure('LDBP Training Loss', figsize=(8, 4))
plt.semilogy(loss_history)
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title(f"LDBP Training (steps_per_span={cfg['steps_per_span']}, "
          f"trainable_gamma={cfg['trainable_gamma']})")
plt.grid(True)
plt.tight_layout()

# 11d. Weight |H| curves
plot_weight_curves(model, fs_sub, Nsub)

# 11e. BER bar chart
plot_ber_bars([
    ('LDBP init\n(train)', ber_x_ldbp_init_train, ber_y_ldbp_init_train),
    ('LDBP final\n(train)', ber_x_ldbp_train, ber_y_ldbp_train),
    ('LDBP final\n(test)', ber_x_ldbp_test, ber_y_ldbp_test),
    ('True DBP\n(test)', ber_x_dbp_test_label, ber_y_dbp_test_label),
])

plt.ioff()
plt.show()
print("\nLDBP Training Complete.")
