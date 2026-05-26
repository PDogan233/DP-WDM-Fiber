"""Simulation data caching — save/load SSFM results to skip re-computation."""

import os
import json
import numpy as np


def _param_snapshot(p):
    """Extract scalar/array params that affect cached simulation data."""
    keys = [
        'Nsym', 'M', 'k', 'Rs', 'rrc_rolloff',
        'Nch', 'DeltaF', 'sps', 'fs', 'Nt', 'sps_rx',
        'L_span', 'Nspans', 'alpha_dBpm',
        'beta1', 'beta2', 'beta3', 'gamma', 'pmd_coeff',
        'PinW_ch', 'G_lin', 'nsp', 'lambda0',
        'method', 'control_param', 'dz', 'dz_DBP',
    ]
    snap = {}
    for k in keys:
        v = p[k]
        if isinstance(v, np.ndarray):
            snap[k] = v.tolist()
        elif isinstance(v, (np.integer, np.floating)):
            snap[k] = v.item()
        else:
            snap[k] = v
    return snap


def build_cache_path(p, seed_train, seed_test):
    """Build cache filename from key parameters that change often."""
    data_dir = 'data'
    os.makedirs(data_dir, exist_ok=True)
    P_dBm = round(10 * np.log10(p['PinW_ch'] * 1e3))
    name = (
        f"N{p['Nsym']}_"
        f"Rs{int(p['Rs'] * 1e-9)}G_"
        f"Ch{p['Nch']}_"
        f"Ns{p['Nspans']}_"
        f"L{int(p['L_span'] * 1e-3)}km_"
        f"P{P_dBm}dBm_"
        f"spsrx{p['sps_rx']}_"
        f"dzdbp{p['dz_DBP'] * 1e-3:.0f}k_"
        f"sd{seed_train}-{seed_test}"
    )
    return os.path.join(data_dir, f'{name}.npz')


def save_sim_cache(path, p, **arrays):
    """Save simulation data + parameter snapshot to .npz."""
    snap = _param_snapshot(p)
    np.savez(path, params_json=json.dumps(snap), allow_pickle=True, **arrays)
    size_mb = os.path.getsize(path) / 1e6
    print(f"  Cached to {path}  ({size_mb:.1f} MB)")


def load_sim_cache(path, p):
    """Load and validate cached simulation data. Returns dict of arrays.

    Raises ValueError if current parameters don't match the cached snapshot.
    """
    data = np.load(path, allow_pickle=True)
    snap = json.loads(str(data['params_json']))

    current = _param_snapshot(p)
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
            "Parameter mismatch between current para.py and cached data:\n"
            + '\n'.join(mismatches)
            + "\n\nDelete the cache file or set use_cache=False to regenerate."
        )

    result = {}
    for k in data.files:
        if k == 'params_json':
            continue
        result[k] = data[k]
    data.close()
    return result
