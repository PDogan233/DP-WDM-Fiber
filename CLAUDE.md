# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview
This is a Python optical communication simulation for **Dual-Polarization WDM 16-QAM** systems. It models the full transceiver chain: TX pulse-shaping and WDM multiplexing → fiber propagation (Split-Step Fourier Method with Manakov equation) → EDFA amplification → RX DSP with Digital Back Propagation (DBP) for nonlinearity compensation.

Two DBP approaches are implemented: **analytical DBP** (physics-based, fine step resolution) and **Learned DBP / LDBP** (parameterized as a neural network in PyTorch, coarser steps, trainable frequency-domain filters).

The codebase is a MATLAB-to-Python port. NumPy/SciPy conventions are preferred over writing loops.

**Main scripts (`main_*.py`) must not contain `def` statements.** All helper functions go into dedicated modules.

## Running the simulation

```bash
# Original simulation (modular, all channels)
python main_simu_v4.py

# Original simulation (smaller scope)
python main_simu_test.py

# LDBP training + evaluation (center channel only, GPU)
python main_ldbp_test.py
```

Python environment: `E:\Anaconda3\envs\pytorch251` (PyTorch 2.5.1, CUDA 12.1, 2× RTX 3080).

## Architecture

### Signal format convention
Dual-polarization signals use `(Nt, 2)` complex NumPy arrays: column 0 = X-pol, column 1 = Y-pol. Frequency-domain arrays follow `fftshift` ordering (DC at center).

### Variable naming convention (LDBP scripts)
```
{domain}_{level}_{dsp}_{role}[_ts]

  domain: tx (transmitted) / rx (received)
   level: wav (waveform, before decimator) / sym (symbol, after decimator)
     dsp: ch (after channel) / sub (after subband) / dbp (after true DBP) / ldbp (after LDBP)
    role: train / test / label (ground truth)
      ts: suffix for torch tensor (omit for numpy)

Examples:
  tx_wav_train          TX fullband waveform, train set (numpy)
  rx_wav_ch_train       RX after channel, train set (numpy)
  rx_wav_sub_train      After subband extraction, train set (numpy)
  rx_wav_dbp_train_label  True DBP output = training label (numpy)
  rx_wav_sub_train_ts   Same as rx_wav_sub_train but torch tensor on GPU
  rx_sym_ldbp_test      Symbol-level after LDBP + RX chain, test set (numpy)
```

### Module map

| File | Role |
|---|---|
| `para.py` | Single `get_parameters()` function returning all system params as a dict. **Central configuration**. Includes DSP mismatch params (eta1-eta4) for robustness testing. |
| `tx_DSP.py` | 16-QAM modulation, RRC pulse shaping, power normalization, WDM frequency multiplexing. Entry point: `multiplex_wdm_channels(p)`. |
| `channel.py` | Fiber propagation via SSFM (Manakov equation) + EDFA amplification with ASE noise. Entry point: `channel_propagation(E_tx, p)`. Supports adaptive step-size control and optional PMD. |
| `dbp.py` | Analytical DBP — inverse SSFM with fine step resolution. `dbp_subband()` operates on the downsampled subband. `dbp_fullband_to_subband()` is a legacy fullband path. Uses symmetric SSF: NL(h/2) → Linear(h) → NL(h/2). |
| `rx_DSP.py` | Matched filtering, subband extraction (`subband_convert`), clock recovery (`decimator` — variance-based timing), frame sync, phase derotation, power normalization, 16-QAM demodulation. |
| `utils.py` | `sync_align()` (cross-correlation alignment), `rcosdesign()` (RRC filter — equivalent to MATLAB's `rcosdesign`). |
| `visualize.py` | Constellation diagrams, waveform plots, frequency spectra. |
| `ldbp.py` | **Learned DBP** — PyTorch `nn.Module`. Alternating `LDBP_LinearLayer` (learnable frequency-domain filter `H`) and `LDBP_NonlinearLayer` (Manakov Kerr, optionally learnable `gamma`). Structure per step: Linear → Nonlinear (no symmetric SSF). EDFA gain removal between spans is fixed. |
| `ldbp_utils.py` | Helpers for LDBP training: `complex_np_to_torch`, `extract_subband`, `rx_after_dbp`, `plot_constellation_grid`, `plot_ber_bars`, `plot_h_vs_ideal`, `plot_h_vs_init`. H visualization uses block-average downsampling and individual phase unwrapping (unwrap each curve, then subtract) to avoid aliasing artifacts. |
| `data_cache.py` | Simulation data caching: `build_cache_path`, `save_sim_cache`, `load_sim_cache`. Saves SSFM results to `data/` to skip recomputation when parameters haven't changed. |
| `model_cache.py` | Model checkpoint caching: `build_model_dir`, `build_model_filename`, `save_model_cache`, `load_model_cache`. Saves trained LDBP weights + results to `model/<system>/` for reuse. |
| `main_ldbp_test.py` | LDBP training + evaluation script. Center channel only. Trains LDBP (initialized with mismatched DSP params) to match true DBP output (matched physical params). Uses `data_cache.py` to skip SSFM and `model_cache.py` to skip training when results already exist. |

### Data flow (main_simu_v4.py / main_simu_test.py)
1. `get_parameters()` → params dict `p`
2. `multiplex_wdm_channels(p)` → TX composite signal + TX reference data
3. `channel_propagation(E_tx, p)` → RX signal after fiber + EDFA
4. Per-channel loop:
   - Downconvert to baseband
   - `subband_convert()` → extract narrower subband at lower SPS
   - `dbp_subband()` → nonlinearity compensation on subband
   - `apply_matched_filter()` → RRC matched filter
   - `decimator()` → clock recovery to 1 sample/symbol
   - `sync_align()` → frame synchronization
   - `compensate_phase()` → bulk phase derotation
   - `normalize_rx_power()` → power normalization
   - `demodulate_16qam()` → symbol-to-bit decision, BER calculation

### Data flow (main_ldbp_test.py)
1. `get_parameters()` → params dict `p`
2. `build_cache_path(p, 42, 99)` → check `data/` for cached `.npz`
3. If cache hit: `load_sim_cache()` with param validation → skip to step 5
4. If cache miss: generate train/test datasets (seed=42/99), extract subband, compute True DBP labels, then `save_sim_cache()`
5. `complex_np_to_torch()` → GPU tensors
6. Build LDBP model (initialized with **mismatched** DSP params `beta2_DSP`, `gamma_DSP`)
7. `build_model_dir()` + `build_model_filename()` → check `model/<system>/` for trained checkpoint
8. If model cache hit: `load_model_cache()` → restore weights, skip to step 10
9. If model cache miss: training loop (Adam + CosineAnnealingLR) → evaluation → `save_model_cache()`
10. BER summary
11. Plots: waveform/spectrum, constellation grid, loss curve, BER bars, H filter analysis (learned vs ideal and learned vs initial)

### LDBP vs analytical DBP design differences

| Aspect | Analytical DBP (`dbp.py`) | LDBP (`ldbp.py`) |
|---|---|---|
| Step structure | Symmetric SSF: NL/2 → Linear → NL/2 | Asymmetric: Linear → Nonlinear |
| Steps per span | ~50 (dz=2km) | Configurable, typ. 1-5 |
| Linear operator | `H = exp((-α/2)h - j·β(ω)·h)` | Learnable `H_real`, `H_imag` parameters |
| Nonlinear operator | Fixed `(8/9)*gamma` | Optionally learnable `gamma` |
| EDFA removal | `x / sqrt(G_lin)` | Same, fixed |
| Framework | NumPy | PyTorch (GPU, autograd) |

### Key design decisions
- **TX SPS is forced to a power of 2** so that `subband_convert` always produces an integer SPS, avoiding fractional-SPS timing drift. See `para.py`.
- **DBP operates on the subband** (not fullband), which is more computationally efficient.
- **LDBP is initialized from physical formula** (with mismatch if DSP params differ), then fine-tuned via gradient descent to compensate for both the mismatch and the coarse step approximation.
- **LDBP training target** is the true DBP output (matched physical params, fine steps), not the TX symbols, avoiding the need for differentiable clock recovery and demodulation.
- **Simulation data caching** (`data_cache.py`): SSFM results saved to `data/` with human-readable filenames encoding key physical params (Nsym, Nch, Nspans, L_span, PinW_ch, etc.). Internal param snapshots validate that loaded data matches current `para.py`. Eta (DSP mismatch) params are excluded from the data snapshot since they don't affect SSFM.
- **Model checkpoint caching** (`model_cache.py`): trained LDBP weights + BER results saved to `model/<system>/`. The system folder name reuses the data cache naming convention. Model filenames encode cfg params (steps_per_span, learning_rate, num_epochs) AND DSP mismatch (e.g. `e2+1.5` = eta2=1.5%). Both cfg and p_snapshot are validated on load.
- **H filter visualization**: block-average downsampling (not strided decimation) prevents aliasing of phase wrapping artifacts. Phase residual uses individually unwrapped curves subtracted (`unwrap(angle(learned)) - unwrap(angle(ref))`), which is more robust than computing the angle of the complex product.

## Notes
- PMD is implemented but disabled by default (`pmd_coeff = 0` in `para.py`).
- `beta3` is set to 0 in `para.py` for simplified testing.
- DSP mismatch params (eta1-eta4) control the initialization error for LDBP robustness testing. They are stored in `p` for access by `model_cache.py`.
- The `decimator` and `sync_align` functions use non-differentiable operations (argmax, spline interpolation) and are only used for evaluation, not during LDBP training.
- `h_plot_layers` and `h_plot_ds` in `cfg` control which linear layers appear in H filter plots and the block-average downsampling factor. A larger `h_plot_ds` (e.g. 100) produces cleaner plots by averaging out numerical noise at phase wrapping boundaries.
