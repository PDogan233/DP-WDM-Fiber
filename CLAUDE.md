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

# PRDBP training + evaluation (physics-regulated LDBP, center channel only, GPU)
python main_prdbp_test.py
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
| `ldbp.py` | **Learned DBP** — PyTorch `nn.Module`. Alternating `LDBP_LinearLayer` (learnable frequency-domain filter `H`, optionally trainable via `trainable_beta2`) and `LDBP_NonlinearLayer` (Manakov Kerr, optionally trainable `gamma`). Both use `nn.Parameter(..., requires_grad=...)` for flexible freeze/unfreeze. Structure per step: Linear → Nonlinear (no symmetric SSF). EDFA gain removal between spans is fixed. **PRDBP support**: `reinitialize_linear_layers(beta2, beta3)` and `reinitialize_nonlinear_layers(gamma)` update Parameter data in-place to project H back to physical form. Stores `Nsub`, `fs_sub`, `fch`, `L_span`, `alpha_dBpm` as instance attributes for reinitialization use. |
| `ldbp_utils.py` | Helpers: `complex_np_to_torch`, `extract_subband`, `rx_after_dbp`, `plot_constellation_grid`, `plot_ber_bars`, `plot_h_vs_ideal`, `plot_h_vs_init` (with optional `fig_suffix` for unique figure windows), `estimate_beta2` (M2, primary), `estimate_gamma` (direct extraction), `_extract_learned_h`, `_extract_learned_gamma`, `estimate_beta2_m1`, `estimate_beta2_m3` (reference). H visualization uses block-average downsampling and individual phase unwrapping. |
| `data_cache.py` | Simulation data caching: `build_cache_path`, `save_sim_cache`, `load_sim_cache`. Saves SSFM results to `data/` to skip recomputation when parameters haven't changed. |
| `model_cache.py` | Model checkpoint caching: `build_model_dir`, `build_model_filename` (LDBP), `build_model_filename_prdbp` (PRDBP), `save_model_cache`, `load_model_cache`, `CURRENT_CKPT_VERSION` (currently **5**). PRDBP filename includes `Nest`, `fmin`, `fmax` fields. cfg validation list includes `fit_fmin_ghz`, `fit_fmax_ghz`. Version bump triggers re-training on incompatible checkpoints. |
| `main_ldbp_test.py` | LDBP training + evaluation script. Center channel only. Single training loop with parameter estimation at the end. |
| `main_prdbp_test.py` | **PRDBP** training + evaluation. Double loop: outer loop (N_est iterations) reinitializes model from estimated β₂/γ, inner loop (N_ep_per_est epochs) trains via Adam + CosineAnnealingLR. Tracks estimation histories and per-N_est loss. Supports conditional estimation (skips non-trainable params) and debug H plots. |

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

Three methods were developed to extract the effective β₂ from trained LDBP linear layers. **M2 is the primary method** — it directly fits the learned phase φ_learned(ω) rather than the phase residual Δφ, which makes it ~500× more numerically stable. For Non-trainable case, when `trainable_beta2=False`, estimation still runs and returns the (unchanged) DSP init value.

#### M2 (primary) — Direct φ_learned quadratic fit

```
φ_learned(ω) = a·ω² + b·ω + c     (fit in trusted region, weighted by |H_init|²)
β₂_eff = −2a / h_dbp               (derived from H(ω) = exp(−j·β(ω)·h))
Δβ₂_est = β₂_eff − β₂_DSP
```

- **Trusted region**: Configurable via `cfg['fit_fmin_ghz']` and `cfg['fit_fmax_ghz']` (defaults: `Rs/20` to `Rs/5`, e.g. 1.6–6.4 GHz @ 32Gbaud). Below Rs/20 the quadratic signal is too small vs noise; above Rs/5 the network has learned non-physical ripple that cancels the β₂ signal. Narrower windows (e.g. `[Rs/8, Rs/4]`) help at large mismatch by focusing on the highest-SNR region. Stored in checkpoint and model filename.
- **Weights**: `|H_init|²` — inverse-variance optimal weighting. Phase noise ∝ 1/|H|, so |H|² gives each frequency point weight proportional to its SNR.
- **b and c** (linear and constant terms) absorb time delay and phase rotation from optimizer drift, keeping `a` (the β₂ coefficient) unbiased.
- **Layer averaging**: estimate is averaged across all layers in `layer_indices` (default `[1]`). Use `list(range(1, N+1))` for all layers. Per-layer std is typically < 1e-31.
- **Accuracy**: typically 99.2%+ improvement over DSP init, 99.99%+ absolute accuracy.
- **State**: `prev_beta2` parameter enables per-estimation accuracy tracking (`acc_per_est`) alongside cumulative improvement (`acc_overall`).

#### M1 (reference) — Regularized division: `Δβ₂(ω) = Δφ(ω) / (denom·ω² + ε)`
Stable (avoids division-by-zero near DC) but systematically underestimates |Δβ₂| by ~100× because the regularization ε dominates the denominator in the trusted region.

#### M3 (reference) — Numerical 2nd derivative: `Δβ₂(ω) = −d²(Δφ)/dω² / h_dbp`
Uses explicit 3-point central finite differences. Correct order-of-magnitude but wrong sign for well-trained models (phase residual dominated by higher-order overfitting).

#### Key empirical findings
- For a well-trained LDBP (500+ epochs), the phase residual Δφ is NOT a clean quadratic — it oscillates with ~23 GHz period at the 0.02 rad level. The network has learned frequency-domain ripple that compensates for β₂ mismatch at the waveform level without learning the true β₂.
- Fitting φ_learned directly (~3.7 rad signal) gives 500× better SNR than fitting Δφ (~0.02 rad residual).
- M2 estimates converge to ~99.99% accuracy in a single fit when training is sufficient, but the H filter remains non-physical. This motivates PRDBP: periodic re-initialization forces H back to quadratic form.

### Gamma estimation from learned nonlinear layers

Gamma is estimated by **directly reading** the learned `gamma` parameter from each `LDBP_NonlinearLayer`. Unlike beta2 (which requires fitting a frequency-domain phase curve), gamma is a scalar per layer — estimation reduces to layer-wise extraction and averaging.

- **Method**: `_extract_learned_gamma(model, layer_indices)` → get per-layer gamma values → `mean`/`std` across layers
- **Comparison**: `gamma_est` vs `gamma_DSP` (init) vs `gamma_true` (physical), with `delta_est = gamma_est − gamma_DSP`
- **Accuracy**: `gamma_acc_overall = 1 − |err_now|/|err_init|` (vs DSP init), `gamma_acc_per_est` (vs `prev_gamma`, for PRDBP step-wise tracking)
- **Interface**: `estimate_gamma(model, p, layer_indices, prev_gamma)` returns `(gamma_est, delta_est, gamma_std, gamma_acc_overall, gamma_acc_per_est)`
- **Non-trainable case**: when `trainable_gamma=False`, estimation still runs and returns the (unchanged) DSP init value with `gamma_acc_overall=0%`
- **Layer indices**: reuses `cfg['h_plot_layers']` — same 1-based index selects corresponding linear and nonlinear layers at each step

### PRDBP (Physics-Regulated DBP)

PRDBP wraps LDBP training in a double loop to enforce physical constraints (`main_prdbp_test.py`):

- **Outer loop** (N_est iterations): train → estimate β₂ (and optionally γ) from learned H → re-initialize all linear/nonlinear layers with estimated physical params
- **Inner loop** (N_ep_per_est epochs): standard gradient descent training (Adam + CosineAnnealingLR). Optimizer and scheduler are **reset each N_est** — learning rate decays from `learning_rate` to `learning_rate_min` independently per iteration.
- **Re-initialization**: `LDBP.reinitialize_linear_layers(beta2, beta3)` recomputes `H_real`/`H_imag` from the physical formula and updates Parameters via `.data.copy_()`. `LDBP.reinitialize_nonlinear_layers(gamma)` updates gamma via `.data.fill_()`.
- **Key insight**: LDBP can achieve low waveform MSE without H converging to true β₂. PRDBP projects H back onto the manifold of physically-valid filters after each training phase, preventing non-physical overfitting.

**Estimation histories**: `beta2_est_history` and `gamma_est_history` are pre-populated with DSP init values (length N_est+1), making the outer loop uniform across all iterations. `acc_per_est` (per-estimation accuracy improvement vs previous estimate) and `acc_overall` (vs DSP init) are tracked.

**Conditional estimation**: When `trainable_beta2=False` or `trainable_gamma=False`, the estimate function is skipped entirely (no terminal output). The unchanged value is recorded with `acc_overall=0.0`.

**Debug H plots**: When `cfg['debug_h_plot']=1`, `plot_h_vs_ideal` and `plot_h_vs_init` are called after each N_est with unique `fig_suffix=' (N_est N)'` to create separate figure windows. Final evaluation always shows H plots regardless of flag.

**Fit window**: β₂ estimation uses `fit_fmin_ghz` and `fit_fmax_ghz` from cfg (defaults: `Rs/20` and `Rs/5`). These are stored in checkpoint and included in model filename. The window critically affects estimation accuracy — narrower windows (e.g. `[Rs/8, Rs/4]`) help at large mismatch by avoiding noisy edges.

**Steps per span considerations**: Lower `steps_per_span` means fewer layers with larger `h_dbp` per step. Each H filter has a 10× larger phase range → optimization is harder (stiffer gradient coupling, larger per-step approximation error vs true DBP). Compensation: more epochs, narrower fit window, gradient clipping, or symmetric SSF structure.

### Data flow (main_ldbp_test.py)
1. `get_parameters()` → params dict `p`
2. `build_cache_path(p, 42, 99)` → check `data/` for cached `.npz`
3. If cache hit: `load_sim_cache()` with param validation → skip to step 5
4. If cache miss: generate train/test datasets (seed=42/99), extract subband, compute True DBP labels, then `save_sim_cache()`
5. `complex_np_to_torch()` → GPU tensors
6. Build LDBP model (initialized with **mismatched** DSP params; `trainable_beta2`/`trainable_gamma` from cfg)
7. `build_model_dir()` + `build_model_filename()` → check `model/<system>/` for trained checkpoint
8. If model cache compatible: `load_model_cache()` → restore weights + results, print summary, skip to BER summary
9. If not: single training loop → parameter estimation → evaluation → save checkpoint
10. BER summary, constellation plots, H filter analysis

### Data flow (main_prdbp_test.py)
1. `p = get_parameters()` (before cfg, so `p['Rs']` is available for fit window defaults)
2. cfg includes: `N_est`, `N_ep_per_est`, `fit_fmin_ghz`, `fit_fmax_ghz`, `h_plot_max_layers`, `debug_h_plot`
3. SSFM data (same cache logic as LDBP, reuse `data_cache.py`)
4. Convert to torch tensors
5. Build model with DSP-mismatched params
6. Auto-select `h_plot_layers` from `h_plot_max_layers` (uniformly spaced across all linear layers)
7. Check model cache (`build_model_filename_prdbp`)
8. If cache hit: restore model + all histories, print per-N_est estimates, skip to plots
9. If not: **PRDBP double loop**:
   - **6a.** Reinitialize model (n_est=0: init eval with BER; n_est≥1: `reinitialize_linear_layers` + `reinitialize_nonlinear_layers`)
   - **6b.** Train inner loop (Adam + CosineAnnealingLR, reset per iteration)
   - **6c.** Parameter estimation (conditional: skip if non-trainable)
   - **6d.** Record estimates + acc values to histories
   - **6e.** Debug H plots (if `debug_h_plot=1`, with unique `fig_suffix`)
   - **6f.** Print per-iteration summary
10. Final evaluation (loss + BER), save checkpoint with all histories
11. BER summary
12. Constellation data (re-run if pretrained)
13. Plots: constellation, concatenated loss curve with N_est boundary lines, BER bars, parameter estimation curves (split by trainable flag), H filter analysis

### LDBP vs analytical DBP design differences

| Aspect | Analytical DBP (`dbp.py`) | LDBP (`ldbp.py`) |
|---|---|---|
| Step structure | Symmetric SSF: NL/2 → Linear → NL/2 | Asymmetric: Linear → Nonlinear |
| Steps per span | ~50 (dz=2km) | Configurable, typ. 1-5 |
| Linear operator | `H = exp((-α/2)h - j·β(ω)·h)` | Learnable `H_real`, `H_imag` parameters (optionally frozen via `trainable_beta2=False`) |
| Nonlinear operator | Fixed `(8/9)*gamma` | Optionally learnable `gamma` (via `trainable_gamma`) |
| EDFA removal | `x / sqrt(G_lin)` | Same, fixed |
| Framework | NumPy | PyTorch (GPU, autograd) |

### Key design decisions
- **TX SPS is forced to a power of 2** so that `subband_convert` always produces an integer SPS, avoiding fractional-SPS timing drift. See `para.py`.
- **DBP operates on the subband** (not fullband), which is more computationally efficient.
- **LDBP is initialized from physical formula** (with mismatch if DSP params differ), then fine-tuned via gradient descent to compensate for both the mismatch and the coarse step approximation.
- **LDBP training target** is the true DBP output (matched physical params, fine steps), not the TX symbols, avoiding the need for differentiable clock recovery and demodulation.
- **Simulation data caching** (`data_cache.py`): SSFM results saved to `data/` with human-readable filenames encoding key physical params (Nsym, Nch, Nspans, L_span, PinW_ch, etc.). Internal param snapshots validate that loaded data matches current `para.py`. Eta (DSP mismatch) params are excluded from the data snapshot since they don't affect SSFM.
- **Model checkpoint caching** (`model_cache.py`): LDBP uses `build_model_filename` → `stps{steps}_lr{lr}_lrmin{lrmin}_ep{epochs}_e2{eta2}_e3{eta3}_e4{eta4}_{GT|GF}_{B2T|B2F}.pth`. PRDBP uses `build_model_filename_prdbp` which adds `Nest{N_est}` (before `ep`) and `fmin{val}_fmax{val}` (GHz values). `CURRENT_CKPT_VERSION` (currently **5**) stored in each checkpoint — version mismatch triggers re-training. Both LDBP and PRDBP checkpoints are validated against current cfg (including fit window params for PRDBP); any mismatch raises ValueError.
- **H filter visualization**: block-average downsampling (not strided decimation) prevents aliasing of phase wrapping artifacts. Phase residual uses individually unwrapped curves subtracted (`unwrap(angle(learned)) - unwrap(angle(ref))`), which is more robust than computing the angle of the complex product. All H plot rows now include x-axis labels ("Frequency (GHz)").
- **Beta2 estimation from learned H**: M2 (direct φ_learned quadratic fit) is the primary method. Fitting φ_learned (~3.7 rad signal) rather than Δφ (~0.02 rad residual) gives ~500× better SNR. Trusted frequency window defaults to `[Rs/20, Rs/3]` — these are anchored to the symbol rate Rs (not hardcoded GHz) so they auto-adapt to different system configurations. |H_init|² weighting provides inverse-variance optimal weights since phase noise ∝ 1/|H|.
- **Phase residual is non-quadratic after training**: the network learns frequency-domain ripple that compensates β₂ mismatch at the waveform level without converging to the true β₂ in H's phase. This is the core motivation for PRDBP's physics-regulation outer loop.
- **`estimate_beta2()` interface**: takes `prev_beta2` parameter for step-wise accuracy tracking; returns `(beta2_est, delta_est, beta2_std, beta2_acc_overall, beta2_acc_per_est)`. Accuracy metrics: `beta2_acc_overall` = 1 − |err_now|/|err_init| (vs DSP init), `beta2_acc_per_est` = 1 − |err_now|/|err_prev| (vs previous estimate, None on first call). M1 and M3 are kept as standalone functions for future comparison.
- **`estimate_gamma()` interface**: mirrors `estimate_beta2()` — returns `(gamma_est, delta_est, gamma_std, gamma_acc_overall, gamma_acc_per_est)`. Takes `prev_gamma` for step-wise tracking.
- **Trainability control**: both `LDBP_LinearLayer` and `LDBP_NonlinearLayer` use `nn.Parameter(..., requires_grad=trainable)` (not `register_buffer`). `requires_grad` allows future dynamic freeze/unfreeze during training (e.g., alternating beta2/gamma optimization in PRDBP), while `register_buffer` would permanently exclude the tensor from the optimizer.
- **Checkpoint compatibility**: a single `try/except (ValueError, KeyError)` block handles all incompatibility cases — cfg mismatch, p mismatch, version mismatch, missing keys. Any failure → delete old file implicitly by overwriting after re-training.

## Notes
- PMD is implemented but disabled by default (`pmd_coeff = 0` in `para.py`).
- `beta3` is set to 0 in `para.py` for simplified testing.
- DSP mismatch params (eta1-eta4) control the initialization error for LDBP robustness testing. They are stored in `p` for access by `model_cache.py`. eta2 controls beta2 mismatch, eta4 controls gamma mismatch.
- The `decimator` and `sync_align` functions use non-differentiable operations (argmax, spline interpolation) and are only used for evaluation, not during LDBP training.
- **Layer selection**: `h_plot_max_layers` (default 10) determines how many linear layers are used for H plots and estimation. Layers are auto-selected uniformly across all linear layers after model creation. The computed `h_plot_layers` list is stored back to cfg for use by all estimation/plot functions.
- **Downsampling**: `h_plot_ds` controls block-average downsampling factor for BOTH H plots and β₂ estimation (via `downsample` parameter). The `_block_downsample` function does block averaging (not strided decimation).
- `estimate_beta2()` frequency limits come from `cfg['fit_fmin_ghz']` and `cfg['fit_fmax_ghz']` (derived from `p['Rs']`). Function defaults (`Rs/20` and `Rs/3`) are only used when not passed from cfg. `estimate_beta2_m1()` and `estimate_beta2_m3()` are available as reference methods but not called by default.
- `CURRENT_CKPT_VERSION` in `model_cache.py` (currently **5**) must be incremented whenever the checkpoint format changes (new required keys, new result fields). Bumping the version ensures old checkpoints trigger automatic re-training rather than crashing.
- `trainable_beta2` and `trainable_gamma` in `cfg` allow independent control of which physical parameters are learned. In PRDBP, setting one to False suppresses its estimation output but still records the unchanged value in histories (`acc_overall=0.0`). The corresponding visualization figure is also skipped.
- **Prefix convention for acc variables**: `beta2_acc_*` and `gamma_acc_*` keep estimation metrics separate. `acc_overall` = improvement vs DSP init. `acc_per_est` = improvement vs previous N_est estimate (None on first call).
- **PRDBP learning rate**: Each N_est iteration creates a fresh Adam optimizer + CosineAnnealingLR scheduler with `T_max=N_ep_per_est`. LR decays from `learning_rate` → `learning_rate_min` independently per outer loop.
- **PRDBP loss tracking**: `loss_per_est_history[0]` = initial loss (before any training), `[1]..[N_est]` = loss after each inner loop. All saved to checkpoint.
- Parameter estimation results are saved to checkpoint as histories (arrays of length N_est for acc values, N_est+1 for estimate values) so pretrained loading can reproduce all outputs without re-running estimation.
