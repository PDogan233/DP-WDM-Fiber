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
| `ldbp_utils.py` | Helpers for LDBP training: `complex_np_to_torch`, `extract_subband`, `rx_after_dbp`, `plot_constellation_grid`, `plot_ber_bars`, `plot_h_vs_ideal`, `plot_h_vs_init`, `estimate_beta2` (primary), `estimate_beta2_m1`, `estimate_beta2_m3` (reference). H visualization uses block-average downsampling and individual phase unwrapping. |
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

### Beta2 estimation from learned H filters

Three methods were developed to extract the effective β₂ from trained LDBP linear layers. **M2 is the primary method** — it directly fits the learned phase φ_learned(ω) rather than the phase residual Δφ, which makes it ~500× more numerically stable.

#### M2 (primary) — Direct φ_learned quadratic fit

```
φ_learned(ω) = a·ω² + b·ω + c     (fit in trusted region, weighted by |H_init|²)
β₂_eff = −2a / h_dbp               (derived from H(ω) = exp(−j·β(ω)·h))
Δβ₂_est = β₂_eff − β₂_DSP
```

- **Trusted region**: `f ∈ [Rs/20, Rs/3]` (e.g. 1.6–10.7 GHz @ 32Gbaud). Below Rs/20 the quadratic signal is too small vs noise; above Rs/3 the network has learned non-physical ripple that cancels the β₂ signal.
- **Weights**: `|H_init|²` — inverse-variance optimal weighting. Phase noise ∝ 1/|H|, so |H|² gives each frequency point weight proportional to its SNR.
- **b and c** (linear and constant terms) absorb time delay and phase rotation from optimizer drift, keeping `a` (the β₂ coefficient) unbiased.
- **Layer averaging**: estimate is averaged across all layers in `layer_indices` (default `[1]`). Use `list(range(1, N+1))` for all layers. Per-layer std is typically < 1e-31.
- **Accuracy**: typically 99.2%+ improvement over DSP init, 99.99%+ absolute accuracy.
- **State**: `prev_beta2` parameter enables step-wise accuracy tracking (`acc_curr`) alongside cumulative improvement (`acc_overall`).

#### M1 (reference) — Regularized division: `Δβ₂(ω) = Δφ(ω) / (denom·ω² + ε)`
Stable (avoids division-by-zero near DC) but systematically underestimates |Δβ₂| by ~100× because the regularization ε dominates the denominator in the trusted region.

#### M3 (reference) — Numerical 2nd derivative: `Δβ₂(ω) = −d²(Δφ)/dω² / h_dbp`
Uses explicit 3-point central finite differences. Correct order-of-magnitude but wrong sign for well-trained models (phase residual dominated by higher-order overfitting).

#### Key empirical findings
- For a well-trained LDBP (500+ epochs), the phase residual Δφ is NOT a clean quadratic — it oscillates with ~23 GHz period at the 0.02 rad level. The network has learned frequency-domain ripple that compensates for β₂ mismatch at the waveform level without learning the true β₂.
- Fitting φ_learned directly (~3.7 rad signal) gives 500× better SNR than fitting Δφ (~0.02 rad residual).
- M2 estimates converge to ~99.99% accuracy in a single fit when training is sufficient, but the H filter remains non-physical. This motivates PRDBP: periodic re-initialization forces H back to quadratic form.

### PRDBP (Physics-Regulated DBP) — design concept

PRDBP wraps LDBP training in a double loop to enforce physical constraints:

- **Outer loop** (N_est iterations): estimate β₂ (and optionally γ) from learned H → re-initialize all linear/nonlinear layers with estimated physical params
- **Inner loop** (N_epoch_per_est epochs): standard gradient descent training (Adam + CosineAnnealingLR)
- **Key insight**: LDBP can achieve low waveform MSE without H converging to true β₂. PRDBP projects H back onto the manifold of physically-valid filters after each training phase, preventing non-physical overfitting.

Missing infrastructure (in `ldbp.py`):
- `LDBP.reinitialize_linear_layers(beta2)` — recompute H_real/H_imag for all LinearLayer instances
- `LDBP.get_gamma_values()` — extract gamma from all NonlinearLayer instances

### Data flow (main_ldbp_test.py)
1. `get_parameters()` → params dict `p`
2. `build_cache_path(p, 42, 99)` → check `data/` for cached `.npz`
3. If cache hit: `load_sim_cache()` with param validation → skip to step 5
4. If cache miss: generate train/test datasets (seed=42/99), extract subband, compute True DBP labels, then `save_sim_cache()`
5. `complex_np_to_torch()` → GPU tensors
6. Build LDBP model (initialized with **mismatched** DSP params `beta2_DSP`, `gamma_DSP`)
7. `build_model_dir()` + `build_model_filename()` → check `model/<system>/` for trained checkpoint
8. If model cache hit: `load_model_cache()` → restore weights, skip to step 10
9. If model cache miss: training loop (Adam + CosineAnnealingLR, configurable epochs) → evaluation → `save_model_cache()`
10. `estimate_beta2()` — extract β₂ from learned H via M2 quadratic fit, print comparison table
11. BER summary
12. Constellation data (re-run if pretrained)
13. Plots: waveform/spectrum, constellation grid, loss curve, BER bars, H filter analysis (learned vs ideal, learned vs initial)

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
- **H filter visualization**: block-average downsampling (not strided decimation) prevents aliasing of phase wrapping artifacts. Phase residual uses individually unwrapped curves subtracted (`unwrap(angle(learned)) - unwrap(angle(ref))`), which is more robust than computing the angle of the complex product. All H plot rows now include x-axis labels ("Frequency (GHz)").
- **Beta2 estimation from learned H**: M2 (direct φ_learned quadratic fit) is the primary method. Fitting φ_learned (~3.7 rad signal) rather than Δφ (~0.02 rad residual) gives ~500× better SNR. Trusted frequency window defaults to `[Rs/20, Rs/3]` — these are anchored to the symbol rate Rs (not hardcoded GHz) so they auto-adapt to different system configurations. |H_init|² weighting provides inverse-variance optimal weights since phase noise ∝ 1/|H|.
- **Phase residual is non-quadratic after training**: the network learns frequency-domain ripple that compensates β₂ mismatch at the waveform level without converging to the true β₂ in H's phase. This is the core motivation for PRDBP's physics-regulation outer loop.
- **`estimate_beta2()` interface**: takes `prev_beta2` parameter for step-wise accuracy tracking; returns `(beta2_est, delta_est)`. Accuracy metrics: `acc_overall` = 1 − |err_now|/|err_init| (vs DSP init), `acc_curr` = 1 − |err_now|/|err_prev| (vs previous estimate). M1 and M3 are kept as standalone functions for future comparison.

## Notes
- PMD is implemented but disabled by default (`pmd_coeff = 0` in `para.py`).
- `beta3` is set to 0 in `para.py` for simplified testing.
- DSP mismatch params (eta1-eta4) control the initialization error for LDBP robustness testing. They are stored in `p` for access by `model_cache.py`.
- The `decimator` and `sync_align` functions use non-differentiable operations (argmax, spline interpolation) and are only used for evaluation, not during LDBP training.
- `h_plot_layers` and `h_plot_ds` in `cfg` control which linear layers appear in H filter plots and the block-average downsampling factor. Also controls which layers participate in `estimate_beta2()` layer averaging (e.g. `list(range(1, 41))` for all layers 1-40).
- `estimate_beta2()` frequency limits default to `fit_fmin_ghz=Rs/20`, `fit_fmax_ghz=Rs/3`. These can be overridden via keyword arguments. `estimate_beta2_m1()` and `estimate_beta2_m3()` are available as reference methods but not called by default.
