import numpy as np
import matplotlib.pyplot as plt
import torch


from para import get_parameters
from tx_DSP import multiplex_wdm_channels
from rx_DSP import (apply_matched_filter, compensate_phase, normalize_rx_power,
                    decimator, demodulate_16qam, subband_convert)
from utils import sync_align, rcosdesign
from channel import channel_propagation
from dbp import dbp_subband
from visualize import setup_constellation, single_ch_constellation
from ldbp import LDBP


# =====================================================================
# Helpers
# =====================================================================

def complex_np_to_torch(arr, device):
    """Convert complex numpy (N, 2) to complex torch tensor on device."""
    arr = np.ascontiguousarray(arr)
    real = torch.from_numpy(arr.real.copy()).float().to(device)
    imag = torch.from_numpy(arr.imag.copy()).float().to(device)
    return torch.complex(real, imag)


def run_rx_chain(r_dbp, tx_data, p, m, sps_sub, rrc_taps_rx):
    """Standard RX DSP chain: MF -> decimator -> sync -> phase -> normalize -> demod."""
    r_filt = apply_matched_filter(r_dbp, rrc_taps_rx)

    y_x_1sps = decimator(r_filt[:, 0], int(np.round(sps_sub)))
    y_y_1sps = decimator(r_filt[:, 1], int(np.round(sps_sub)))

    tx_sym_x = tx_data['symbols'][m]['X']
    tx_sym_y = tx_data['symbols'][m]['Y']

    tx_align_x, rx_align_x, sync_x = sync_align(tx_sym_x, y_x_1sps)
    tx_align_y, rx_align_y, sync_y = sync_align(tx_sym_y, y_y_1sps)

    rx_derot_x = compensate_phase(rx_align_x, sync_x)
    rx_derot_y = compensate_phase(rx_align_y, sync_y)

    rx_norm_x = normalize_rx_power(rx_derot_x, tx_data['p_raw_x'][m], p['PinW_ch'])
    rx_norm_y = normalize_rx_power(rx_derot_y, tx_data['p_raw_y'][m], p['PinW_ch'])

    bits_hat_x = demodulate_16qam(rx_norm_x, p['M'])
    bits_hat_y = demodulate_16qam(rx_norm_y, p['M'])

    tx_bits_x = demodulate_16qam(tx_align_x, p['M'])
    tx_bits_y = demodulate_16qam(tx_align_y, p['M'])

    ber_x = np.mean(bits_hat_x != tx_bits_x)
    ber_y = np.mean(bits_hat_y != tx_bits_y)

    y_plot = np.column_stack((rx_norm_x, rx_norm_y))
    return ber_x, ber_y, y_plot


# =====================================================================
# Configuration
# =====================================================================

STEPS_PER_SPAN = 2       # LDBP steps per fiber span (>= 1)
TRAINABLE_GAMMA = 1    # Whether nonlinear gamma is trainable
NUM_EPOCHS = 300
LEARNING_RATE = 1e-3
PRINT_INTERVAL = 20

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
E_tx_train, tx_data_train = multiplex_wdm_channels(p)
E_rx_train = channel_propagation(E_tx_train, p)

print("Generating Test Set (seed=99)...")
np.random.seed(99)
E_tx_test, tx_data_test = multiplex_wdm_channels(p)
E_rx_test = channel_propagation(E_tx_test, p)

# =====================================================================
# 3. Extract center channel -> subband
# =====================================================================
def extract_subband(E_rx, p, fch):
    """Downconvert center channel and extract subband. Returns (r_sub, Nsub, fs_sub, sps_sub)."""
    carrier_rx = np.exp(-1j * 2 * np.pi * fch * p['t'])
    r_baseband = E_rx * carrier_rx
    r_sub, Nsub, Nt, fs_sub, sps_sub = subband_convert(
        r_baseband, p['sps_rx'], p['fs'], p['Rs']
    )
    return r_sub, Nsub, fs_sub, sps_sub

print("\nExtracting subband (Train)...")
r_sub_train, Nsub, fs_sub, sps_sub = extract_subband(E_rx_train, p, fch)
print(f"  Nsub={Nsub}, fs_sub={fs_sub/1e9:.2f} GHz, sps_sub={sps_sub:.2f}")

print("Extracting subband (Test)...")
r_sub_test, Nsub_test, fs_sub_test, sps_sub_test = extract_subband(E_rx_test, p, fch)

# =====================================================================
# 4. True DBP targets (matched physical params, fine step resolution)
# =====================================================================
print("\nRunning True DBP (matched params, fine steps) for target...")
r_dbp_train_label = dbp_subband(
    r_sub_train, Nsub, fs_sub, fch,
    p['L_span'], p['alpha_dBpm'],
    p['beta2'], p['beta3'], p['gamma'],   # physical params (eta=0)
    p['dz_DBP'], p['Nspans'], p['G_lin']
)

r_dbp_test_label = dbp_subband(
    r_sub_test, Nsub_test, fs_sub_test, fch,
    p['L_span'], p['alpha_dBpm'],
    p['beta2'], p['beta3'], p['gamma'],
    p['dz_DBP'], p['Nspans'], p['G_lin']
)

# =====================================================================
# 5. Convert to torch tensors
# =====================================================================
print("\nConverting to torch tensors...")
input_train = complex_np_to_torch(r_sub_train, DEVICE)
target_train = complex_np_to_torch(r_dbp_train_label, DEVICE)

input_test = complex_np_to_torch(r_sub_test, DEVICE)
target_test = complex_np_to_torch(r_dbp_test_label, DEVICE)

print(f"  x_train shape: {input_train.shape}, dtype: {input_train.dtype}")

# =====================================================================
# 6. Build LDBP model
# =====================================================================
print(f"\nBuilding LDBP: Nspans={p['Nspans']}, steps_per_span={STEPS_PER_SPAN}, "
      f"trainable_gamma={TRAINABLE_GAMMA}")

model = LDBP(
    Nsub=Nsub, fs_sub=fs_sub, fch=fch,
    L_span=p['L_span'], alpha_dBpm=p['alpha_dBpm'],
    beta2=p['beta2'], beta3=p['beta3'], gamma=p['gamma'],
    Nspans=p['Nspans'], steps_per_span=STEPS_PER_SPAN,
    G_lin=p['G_lin'], trainable_gamma=TRAINABLE_GAMMA
).to(DEVICE)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Total parameters: {total_params:,}  |  Trainable: {trainable_params:,}")

# =====================================================================
# 7. Training
# =====================================================================
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_history = []

# Initial loss (before any training)
model.eval()
with torch.no_grad():
    out_init = model(input_train)
    init_loss = torch.mean(torch.abs(out_init - target_train) ** 2).item()
print(f"\nInitial loss (before training): {init_loss:.6e}")

model.train()
print(f"Training for {NUM_EPOCHS} epochs...")
for epoch in range(NUM_EPOCHS):
    optimizer.zero_grad()
    output = model(input_train)
    loss = torch.mean(torch.abs(output - target_train) ** 2)
    loss.backward()
    optimizer.step()

    loss_history.append(loss.item())

    if (epoch + 1) % PRINT_INTERVAL == 0:
        print(f"  Epoch {epoch+1:4d}/{NUM_EPOCHS}  |  Loss: {loss.item():.6e}")

final_loss = loss_history[-1]
print(f"Final loss: {final_loss:.6e}  |  Reduction: {init_loss/final_loss:.2f}x")

# =====================================================================
# 8. Evaluation on test set
# =====================================================================
print("\n--- Evaluation on Test Set ---")
model.eval()
with torch.no_grad():
    out_test = model(input_test)
    test_loss = torch.mean(torch.abs(out_test - target_test) ** 2).item()
    r_dbp_ldbp_test = out_test.cpu().numpy()

print(f"Test MSE: {test_loss:.6e}")

# RX DSP chain for LDBP output
print("\nRunning RX DSP chain...")
ber_x_ldbp, ber_y_ldbp, y_plot_ldbp = run_rx_chain(
    r_dbp_ldbp_test, tx_data_test, p, m_center, sps_sub_test, p['rrc_taps_rx']
)

# RX DSP chain for True DBP baseline
ber_x_dbp, ber_y_dbp, y_plot_dbp = run_rx_chain(
    r_dbp_test_label, tx_data_test, p, m_center, sps_sub_test, p['rrc_taps_rx']
)

print(f"\nBER Comparison (Test Set, Channel {m_center+1}):")
print(f"  LDBP      X: {ber_x_ldbp:.3g}, Y: {ber_y_ldbp:.3g}")
print(f"  True DBP  X: {ber_x_dbp:.3g}, Y: {ber_y_dbp:.3g}")

# =====================================================================
# 9. Plots
# =====================================================================
plt.ion()

# Loss curve
plt.figure('LDBP Training Loss', figsize=(8, 4))
plt.semilogy(loss_history)
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title(f'LDBP Training (steps_per_span={STEPS_PER_SPAN}, trainable_gamma={TRAINABLE_GAMMA})')
plt.grid(True)
plt.tight_layout()

# Constellations
fig_x, fig_y, nrows, ncols = setup_constellation(2)
single_ch_constellation(fig_x, fig_y, nrows, ncols, 0, y_plot_ldbp)
single_ch_constellation(fig_x, fig_y, nrows, ncols, 1, y_plot_dbp)

# Rename constellation titles
plt.figure(fig_x.number)
for i, ax in enumerate(fig_x.get_axes()):
    label = ['LDBP X', 'True DBP X'][i] if i < 2 else ''
    if label:
        ax.set_title(label)

plt.figure(fig_y.number)
for i, ax in enumerate(fig_y.get_axes()):
    label = ['LDBP Y', 'True DBP Y'][i] if i < 2 else ''
    if label:
        ax.set_title(label)

plt.ioff()
plt.show()
print("\nLDBP Training Complete.")
