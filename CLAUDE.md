# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview
This is a Python optical communication simulation for **Dual-Polarization WDM 16-QAM** systems. It models the full transceiver chain: TX pulse-shaping and WDM multiplexing → fiber propagation (Split-Step Fourier Method with Manakov equation) → EDFA amplification → RX DSP with Digital Back Propagation (DBP) for nonlinearity compensation.

The codebase is a MATLAB-to-Python port. NumPy/SciPy conventions are preferred over writing loops.

## Running the simulation

```bash
# Stable simulation (modular, current)
python main_simu_v4.py

# Test simulation (smaller scope)
python main_simu_test.py
```

There is no build step, no test runner, and no package manager required. Standard scientific Python stack: `numpy`, `scipy`, `matplotlib`, `torch`.

## Architecture

### Signal format convention
Dual-polarization signals use `(Nt, 2)` complex NumPy arrays: column 0 = X-pol, column 1 = Y-pol. Frequency-domain arrays follow `fftshift` ordering (DC at center).

### Module map

| File | Role |
|---|---|
| `para.py` | Single `get_parameters()` function returning all system params as a dict. This is the **central configuration** — simulation scripts import this first. |
| `tx_DSP.py` | 16-QAM modulation, RRC pulse shaping, power normalization, WDM frequency multiplexing. Entry point: `multiplex_wdm_channels(p)`. |
| `channel.py` | Fiber propagation via SSFM (Manakov equation) + EDFA amplification with ASE noise. Entry point: `channel_propagation(E_tx, p)`. Supports adaptive step-size control (`constant`, `local_error`, `global_error`) and optional PMD. |
| `dbp.py` | Digital Back Propagation — inverse fiber propagation for nonlinearity compensation. `dbp_subband()` operates on the already-downsampled subband (preferred path). `dbp_fullband_to_subband()` extracts a subband by bandwidth, runs DBP, and reconstructs the fullband (legacy path). |
| `rx_DSP.py` | Matched filtering, subband extraction (`subband_convert`), clock recovery (`decimator` — variance-based timing), frame sync, phase derotation, power normalization, 16-QAM demodulation. |
| `utils.py` | `sync_align()` (cross-correlation alignment), `rcosdesign()` (RRC filter — equivalent to MATLAB's `rcosdesign`). |
| `visualize.py` | Constellation diagrams, waveform plots, frequency spectra. |

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

### Key design decisions
- **TX SPS is forced to a power of 2** (`sps = 2**ceil(log2(...))`) so that `subband_convert` always produces an integer SPS after resampling, avoiding fractional-SPS timing drift in the matched filter and decimator stages. See `para.py` line 39.
- **DBP operates on the subband** (not the fullband), which is more computationally efficient. The subband SPS (e.g., 2 or 8) is the target resolution for DBP and matched filtering.
- The `decimator` uses variance-based timing extraction: it finds the sampling phase that maximizes signal variance via spline interpolation, then applies a fractional-delay skew before decimation.

## Notes
- `torch` is imported in `dbp.py` but not actively used; it's included for potential ML-based DBP training paths.
- PMD is implemented but disabled by default (`pmd_coeff = 0` in `para.py`).