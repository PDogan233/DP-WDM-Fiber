"""Model checkpoint caching — save/load trained LDBP models and results."""

import os
import numpy as np
import torch

from data_cache import build_cache_path, _param_snapshot

# Increment when checkpoint format changes (new required keys, etc.)
CURRENT_CKPT_VERSION = 4


def build_model_dir(p, seed_train, seed_test):
    """Build model directory path, reusing the same naming as data cache."""
    cache_path = build_cache_path(p, seed_train, seed_test)
    base = os.path.splitext(os.path.basename(cache_path))[0]
    model_dir = os.path.join('model', base)
    os.makedirs(model_dir, exist_ok=True)
    return model_dir


def build_model_filename(cfg, p):
    """Build model filename from cfg + DSP mismatch params."""
    def _pct_str(val):
        pct = round(val * 100 + 1e-12, 6)
        s = f"{pct:.6f}".rstrip('0').rstrip('.')
        return f"+{s}" if pct >= 0 else s

    e2 = _pct_str(p['eta2'])
    e3 = _pct_str(p['eta3'])
    e4 = _pct_str(p['eta4'])

    lr_str = f"{cfg['learning_rate']:.0e}".replace('-', 'm')
    lrmin_str = f"{cfg['learning_rate_min']:.0e}".replace('-', 'm')
    g_flag = 'GT' if cfg['trainable_gamma'] else 'GF'
    b2_flag = 'B2T' if cfg['trainable_beta2'] else 'B2F'
    name = (
        f"stps{cfg['steps_per_span']}_"
        f"lr{lr_str}_"
        f"lrmin{lrmin_str}_"
        f"ep{cfg['num_epochs']}_"
        f"e2{e2}_e3{e3}_e4{e4}_"
        f"{g_flag}_{b2_flag}"
    )
    return f'{name}.pth'


def build_model_filename_prdbp(cfg, p):
    """Build model filename for PRDBP (includes N_est)."""
    def _pct_str(val):
        pct = round(val * 100 + 1e-12, 6)
        s = f"{pct:.6f}".rstrip('0').rstrip('.')
        return f"+{s}" if pct >= 0 else s

    e2 = _pct_str(p['eta2'])
    e3 = _pct_str(p['eta3'])
    e4 = _pct_str(p['eta4'])

    lr_str = f"{cfg['learning_rate']:.0e}".replace('-', 'm')
    lrmin_str = f"{cfg['learning_rate_min']:.0e}".replace('-', 'm')
    g_flag = 'GT' if cfg['trainable_gamma'] else 'GF'
    b2_flag = 'B2T' if cfg['trainable_beta2'] else 'B2F'
    name = (
        f"stps{cfg['steps_per_span']}_"
        f"lr{lr_str}_"
        f"lrmin{lrmin_str}_"
        f"Nest{cfg['N_est']}_"
        f"ep{cfg['N_ep_per_est']}_"
        f"e2{e2}_e3{e3}_e4{e4}_"
        f"{g_flag}_{b2_flag}"
    )
    return f'{name}.pth'


def save_model_cache(path, model, cfg, p, **results):
    """Save model state + config + results to checkpoint file."""
    snap = _param_snapshot(p)
    snap['beta2_DSP'] = p['beta2_DSP']
    snap['beta3_DSP'] = p['beta3_DSP']
    snap['gamma_DSP'] = p['gamma_DSP']
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'cfg': cfg,
        'p_snapshot': snap,
        'ckpt_version': CURRENT_CKPT_VERSION,
        **results,
    }
    torch.save(checkpoint, path)
    size_mb = os.path.getsize(path) / 1e6
    print(f"  Model saved to {path}  ({size_mb:.1f} MB)")


def load_model_cache(path, cfg, p):
    """Load and validate checkpoint. Returns dict of saved results.

    Raises ValueError if current cfg/p don't match the saved snapshot.
    """
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)

    # Validate cfg
    for k in ['steps_per_span', 'trainable_gamma', 'trainable_beta2',
              'num_epochs', 'N_ep_per_est', 'N_est',
              'learning_rate', 'learning_rate_min']:
        if checkpoint['cfg'].get(k) != cfg.get(k):
            raise ValueError(
                f"cfg mismatch: {k} current={cfg.get(k)}  cached={checkpoint['cfg'].get(k)}\n"
                "Delete the model file to retrain."
            )

    # Validate p snapshot (including DSP params)
    snap = checkpoint['p_snapshot']
    current = _param_snapshot(p)
    current['beta2_DSP'] = p['beta2_DSP']
    current['beta3_DSP'] = p['beta3_DSP']
    current['gamma_DSP'] = p['gamma_DSP']
    mismatches = []
    for k in snap:
        cv = current.get(k)
        sv = snap[k]
        if cv is None:
            continue
        if isinstance(cv, list):
            if not np.allclose(cv, sv, rtol=1e-8):
                mismatches.append(f"  {k}: array mismatch")
        elif isinstance(cv, float):
            if not np.isclose(cv, float(sv), rtol=1e-8):
                mismatches.append(f"  {k}: current={cv:.6e}  cached={sv:.6e}")
        elif cv != sv:
            mismatches.append(f"  {k}: current={cv}  cached={sv}")

    if mismatches:
        raise ValueError(
            "Parameter mismatch between current para.py and cached model:\n"
            + '\n'.join(mismatches)
            + "\n\nDelete the model file to retrain."
        )

    return checkpoint