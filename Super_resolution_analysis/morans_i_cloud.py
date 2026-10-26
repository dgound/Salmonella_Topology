"""
Per-walk "cloud" Moran's I between 3D localization positions and circular
genomic position (time-point), with a 100-permutation p-value.

Two tiers with identical numerics:
  - C++: the validated GenomicMoransICalculator (Morans_I/), compiled on
    first use into morans_i_genomic.so and called through ctypes.
  - NumPy: a blockwise reimplementation used when no compiler is available.

Walks above `max_points` localizations are stratified-subsampled per
time-point exactly as `stratified_sample` in Morans_I/main_genomic.cpp.
"""

import os
import sys
import ctypes
import shutil
import subprocess
from pathlib import Path

import numpy as np
from tqdm import tqdm

EPSILON = 1e-10
PERMUTATIONS = 100
_SOURCES = ('morans_i_genomic.cpp', 'morans_i_genomic.h', 'morans_i_genomic_api.cpp')
_HERE = os.path.dirname(os.path.abspath(__file__))


def find_cpp_dir(override=None):
    """Return the directory holding the C++ sources, or None if not found."""
    if override:
        candidates = [override]
    else:
        candidates = [os.path.join(_HERE, 'Morans_I')]
    for d in candidates:
        if all(os.path.isfile(os.path.join(d, f)) for f in _SOURCES):
            return os.path.abspath(d)
    return None


def _compiler_hint():
    if sys.platform == 'darwin':
        return "run 'xcode-select --install'"
    if os.environ.get('CONDA_PREFIX'):
        return "run 'conda install -c conda-forge cxx-compiler'"
    return "run 'apt install g++'"


def load_c_backend(cpp_dir):
    """Build (if missing or stale) and load morans_i_genomic.so; None on any failure."""
    try:
        if cpp_dir is None:
            return None
        so = os.path.join(cpp_dir, 'morans_i_genomic.so')
        srcs = [os.path.join(cpp_dir, f) for f in _SOURCES]
        if (not os.path.exists(so)
                or os.path.getmtime(so) < max(os.path.getmtime(s) for s in srcs)):
            cxx = os.environ.get('CXX') or next(
                (c for c in ('c++', 'g++', 'clang++') if shutil.which(c)), None)
            if cxx is None:
                print("WARNING: no C++ compiler found for the Moran's I backend "
                      f"({_compiler_hint()} to install one); "
                      "using the slower NumPy implementation.")
                return None
            print(f"Compiling Moran's I C++ backend ({so}) with {cxx}...")
            cmd = [cxx, '-std=c++17', '-O3', '-march=native', '-pthread', '-shared',
                   '-fPIC', '-o', so, srcs[0], srcs[2]]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                print("WARNING: failed to compile the Moran's I C++ backend "
                      f"({_compiler_hint()} to install a working C++ compiler); "
                      "using the slower NumPy implementation.")
                if proc.stderr.strip():
                    print(proc.stderr.strip())
                return None
        lib = ctypes.CDLL(so)
        lib.morans_i_cloud.argtypes = [
            ctypes.POINTER(ctypes.c_double),   # xyz (n*3)
            ctypes.POINTER(ctypes.c_int),      # tp  (n)
            ctypes.c_int,                      # n
            ctypes.c_double,                   # cycle_bins
            ctypes.c_ulonglong,                # seed (0 = random_device)
            ctypes.POINTER(ctypes.c_double),   # out (2)
        ]
        lib.morans_i_cloud.restype = ctypes.c_int
        return lib
    except Exception as e:
        print(f"WARNING: could not load the Moran's I C++ backend ({e}); "
              "using the slower NumPy implementation.")
        return None


def morans_i_cloud_c(lib, xyz, tp, cycle_bins, seed):
    """Moran's I and permutation p-value via the C++ calculator. Returns (I, p)."""
    out = np.empty(2, dtype=np.float64)
    status = lib.morans_i_cloud(
        xyz.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        tp.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        len(tp), float(cycle_bins), int(seed or 0),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
    if status != 0:
        raise RuntimeError("C++ Moran's I calculation failed")
    return float(out[0]), float(out[1])


def _circular_deviations(tp, cycle_bins):
    # Sequential sums and the C++ operation order: with equal counts per time-point the
    # circular mean is atan2 of rounding noise, so it must be rounded exactly as in C++.
    n = len(tp)
    theta = tp * 2.0 * np.pi / cycle_bins
    x_bar = np.arctan2(np.cumsum(np.sin(theta))[-1] / n, np.cumsum(np.cos(theta))[-1] / n)
    diff = theta - x_bar
    d = np.arctan2(np.sin(diff), np.cos(diff))
    return d, np.sqrt(np.cumsum(d * d)[-1] / n)


def morans_i_cloud_numpy(xyz, tp, cycle_bins, seed, n_perms=PERMUTATIONS):
    """NumPy port of compute_genomic_morans_i_with_pvalue. Returns (I, p)."""
    n = len(tp)
    if n < 3:
        raise ValueError("Need at least 3 points for Moran's I calculation")
    rng = np.random.default_rng(seed)
    tp = np.asarray(tp, dtype=np.float64)

    D = np.empty((n, n_perms + 1), dtype=np.float64)
    d, std_dev = _circular_deviations(tp, cycle_bins)
    if std_dev <= EPSILON:
        raise ValueError("Standard deviation of circular deviations is zero")
    D[:, 0] = d / std_dev
    perm = tp.copy()
    for k in range(1, n_perms + 1):
        rng.shuffle(perm)
        d, std_dev = _circular_deviations(perm, cycle_bins)
        D[:, k] = d / std_dev if std_dev > EPSILON else d

    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    B = max(1, int(2.5e8 // (8 * n)))
    num = np.zeros(n_perms + 1)
    W = 0.0
    for a in range(0, n, B):
        b = min(a + B, n)
        w = np.subtract.outer(x[a:b], x)
        w *= w
        t = np.subtract.outer(y[a:b], y)
        t *= t
        w += t
        np.subtract.outer(z[a:b], z, out=t)
        t *= t
        w += t
        del t
        w += EPSILON
        np.reciprocal(w, out=w)
        rows = np.arange(b - a)
        w[rows, a + rows] = 0.0
        r = w.sum(axis=1)
        contrib = D[a:b] * (w @ D)
        del w
        ok = r > EPSILON
        num += (contrib[ok] / r[ok, None]).sum(axis=0) + contrib[~ok].sum(axis=0)
        W += ok.sum() + r[~ok].sum()

    den = (D * D).sum(axis=0)
    if W <= EPSILON:
        raise ValueError("Sum of weights is zero or too small")
    if den[0] <= EPSILON:
        raise ValueError("Denominator in Moran's I calculation is zero")
    I_obs = (n / W) * num[0] / den[0]
    valid = den[1:] > EPSILON
    I_perm = (n / W) * num[1:][valid] / den[1:][valid]
    count_ge = int(np.sum(I_perm >= I_obs))
    return float(I_obs), (count_ge + 1) / (n_perms + 1)


def stratified_subsample(tp_values, max_points, rng):
    """Row indices of a per-time-point stratified sample of max_points rows (sorted). max_points=0 disables subsampling."""
    tp_values = np.asarray(tp_values)
    if max_points <= 0 or len(tp_values) <= max_points:
        return np.arange(len(tp_values))
    unique_tps = np.unique(tp_values)
    per_tp, remainder = divmod(max_points, len(unique_tps))
    selected = []
    for i, t in enumerate(unique_tps):
        idx = np.flatnonzero(tp_values == t)
        k = min(per_tp + (1 if i < remainder else 0), len(idx))
        selected.append(rng.choice(idx, size=k, replace=False))
    return np.sort(np.concatenate(selected))


def compute_for_walks(walk_groups, timepoints, max_points, seed, backend, cpp_dir=None,
                      filename=''):
    """Per-walk Moran's I. Returns (rows for write_morans_results_csv, morans_i_dict)."""
    lib = None
    if backend in ('auto', 'c'):
        lib = load_c_backend(find_cpp_dir(cpp_dir))
        if lib is None and backend == 'c':
            raise RuntimeError("C++ Moran's I backend requested but unavailable")
    if lib is not None:
        print("Moran's I backend: C++ (morans_i_genomic.so)")
    else:
        print("Moran's I backend: NumPy")

    cycle_bins = len(timepoints)
    stem = Path(filename).stem if filename else ''
    rows = []
    morans_i_dict = {}
    for i, (walk_id, walk_data) in enumerate(tqdm(walk_groups.items(), desc="Moran's I")):
        rng = np.random.default_rng(None if seed is None else seed + i)
        tp_all = walk_data['time-point'].to_numpy()
        sel = stratified_subsample(tp_all, max_points, rng)
        xyz = np.ascontiguousarray(walk_data[['x', 'y', 'z']].to_numpy(dtype=np.float64)[sel])
        tp = np.ascontiguousarray(tp_all[sel], dtype=np.int32)
        n_tp = len(np.unique(tp))
        try:
            if lib is not None:
                I, p = morans_i_cloud_c(lib, xyz, tp, cycle_bins,
                                        int(rng.integers(1, 2**63)))
            else:
                I, p = morans_i_cloud_numpy(xyz, tp, cycle_bins, rng)
        except Exception as e:
            print(f"Error computing Moran's I for Walk-ID {walk_id}: {e}")
            I, p = float('nan'), float('nan')
        rows.append((stem, walk_id, n_tp, I, p))
        morans_i_dict[walk_id] = {'I_value': I, 'P_value': p}
    return rows, morans_i_dict


def write_morans_results_csv(rows, path):
    """Write rows in the C++ CLI's <stem>_morans_i_results.csv format."""
    def fmt(v):
        return 'NaN' if np.isnan(v) else f'{v:.6f}'
    with open(path, 'w') as f:
        f.write('Filename,Walk-ID,Time-points,I-value,P-value\n')
        for stem, walk_id, n_tp, I, p in rows:
            f.write(f'{stem},{walk_id},{n_tp},{fmt(I)},{fmt(p)}\n')
