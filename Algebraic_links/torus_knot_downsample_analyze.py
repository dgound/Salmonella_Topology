#!/usr/bin/env python3
"""
De Witt downsampling + Alexander-polynomial-fingerprint analysis for
(p, q)-torus knot ensembles produced by generate_torus_knot_polygons.py.

For each knot polygon, each length L in --lengths, and each subsample:
  - De Witt subsample the 10k-gon to L vertices (one random point per
    variable-width chunk; --no-replacement caps subsamples to the min
    chunk size automatically).
  - Compute the simplified crossing count and the Alexander-polynomial
    fingerprint (|Delta(t)| at several real t) via knoodle.

The fingerprint is the Alexander polynomial evaluated at the canonical
invariant points (roots of unity) that knoodle exposes via
alexander_invariants(): we take the magnitudes at t = -1, i, and
omega = exp(2 pi i / 3), i.e. (|Delta(-1)|, |Delta(i)|, |Delta(omega)|).
These are the standard special values of the Alexander polynomial (the
determinant, the i-value, and the omega-value), not arbitrary points,
and they are realization-independent since knoodle returns the
symmetrically normalized polynomial. Using the full invariant fingerprint
instead of the single determinant |Delta(-1)| lets us include torus knots
whose determinant collides (T(3,4) and the trefoil both have det 3;
T(3,5), T(3,7) and the unknot all have det 1).

'still knotted'  <=>  crossing_count > 0
'correct type'   <=>  fingerprint == expected fingerprint of the knot

Output: CSV with columns knot_id, length, sample_id, crossing_count,
determinant, fingerprint, knotted, correct.
"""

import os
import sys
import argparse
import glob
import multiprocessing

CPU_COUNT = multiprocessing.cpu_count()
NUM_WORKERS = int(os.environ.get("TORUS_WORKERS", "0")) or max(1, CPU_COUNT - 2)
THREADS_PER_PROCESS = 1
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = str(THREADS_PER_PROCESS)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import pandas as pd
from multiprocessing import Pool
from tqdm import tqdm

MIN_LENGTH = 5
DEFAULT_NUM_SAMPLES = 100
# Canonical Alexander invariant points (roots of unity) from knoodle's
# alexander_invariants(); magnitudes at t = -1, i, omega.
FP_KEYS = ("at_minus_1", "at_i", "at_omega")
# Evaluation points for the fingerprint: t = -1, i, omega. Must go through
# the multi-point alexander() overload: the single-point overload (and
# alexander_invariants(), which uses it) discards its results due to a
# pass-by-value bug in Alexander_UMFPACK.hpp and always returns 0.
FP_POINTS = [complex(-1.0, 0.0), complex(0.0, 1.0), complex(-0.5, 3**0.5 / 2)]
FP_STORE_ROUND = 4                   # decimals kept in the stored CSV string
FP_RTOL = 2e-3                       # relative tolerance for fingerprint match
FP_ATOL = 1e-2                       # absolute tolerance for fingerprint match
EXPECTED_PROBE_L = 300               # faithful plateau length for expected fingerprint
EXPECTED_PROBE_FILES = 41            # files sampled to determine expected fingerprint


def load_polygon(filepath):
    try:
        pts = np.loadtxt(filepath, delimiter="\t")
        return pts if pts.size else None
    except Exception:
        return None


def flatten_coords(points):
    return [float(c) for p in points for c in p]


def fingerprint(analyzer):
    """Tuple of |Delta(t)| at the canonical invariant points FP_KEYS.
    None on failure."""
    try:
        res = analyzer.alexander(FP_POINTS)
        return tuple(abs(r.to_double()) for r in res)
    except Exception:
        return None


def fps_match(a, b, rtol=FP_RTOL, atol=FP_ATOL):
    """Tolerance comparison so floating-point jitter at rounding boundaries
    does not spuriously reject a genuinely-correct fingerprint."""
    if a is None or b is None or len(a) != len(b):
        return False
    return all(abs(x - y) <= atol + rtol * abs(y) for x, y in zip(a, b))


def systematic_indices(chain_length, target_L):
    """Deterministic every-k-th indices giving target_L points."""
    return np.linspace(0, chain_length, target_L, endpoint=False).astype(np.int64)


def compute_expected_fingerprint(files, chain_length):
    """Mode of the fingerprint over a faithful (L=EXPECTED_PROBE_L) systematic
    downsample of the first EXPECTED_PROBE_FILES knots."""
    import knoodle
    probe_L = min(EXPECTED_PROBE_L, chain_length)
    idx = systematic_indices(chain_length, probe_L)
    fps = []
    for f in files[:EXPECTED_PROBE_FILES]:
        pts = load_polygon(f)
        if pts is None:
            continue
        try:
            a = knoodle.KnotAnalyzer(flatten_coords(pts[idx]),
                                     simplify=True, simplify_level=5)
            fp = fingerprint(a)
            if fp is not None:
                fps.append(fp)
        except Exception:
            continue
    if not fps:
        raise RuntimeError("Could not compute expected fingerprint (all probes failed)")
    # Componentwise median is robust to occasional knoodle failures / jitter.
    return tuple(np.median(np.array(fps, dtype=float), axis=0))


def chunk_boundaries(chain_length, target_L):
    return np.floor(np.arange(target_L + 1) * chain_length / target_L).astype(np.int64)


def dewitt_indices(chain_length, target_L, num_samples, replace, rng):
    boundaries = chunk_boundaries(chain_length, target_L)
    sizes = boundaries[1:] - boundaries[:-1]
    min_chunk = int(sizes.min())
    if replace:
        idx = np.empty((num_samples, target_L), dtype=np.int64)
        for c in range(target_L):
            idx[:, c] = int(boundaries[c]) + rng.integers(0, int(sizes[c]), size=num_samples)
        return idx
    if num_samples > min_chunk:
        raise ValueError(f"num_samples={num_samples} > min chunk {min_chunk}")
    idx = np.empty((num_samples, target_L), dtype=np.int64)
    for c in range(target_L):
        perm = rng.permutation(int(sizes[c]))[:num_samples]
        idx[:, c] = int(boundaries[c]) + perm
    return idx


def uniform_indices(chain_length, target_L, num_samples, replace, rng):
    """target_L indices per sample drawn uniformly from the whole chain
    (no chunk structure), sorted. replace=False draws one distinct pool of
    num_samples*target_L indices and partitions it."""
    if replace:
        idx = np.empty((num_samples, target_L), dtype=np.int64)
        for s in range(num_samples):
            idx[s] = np.sort(rng.choice(chain_length, size=target_L, replace=False))
        return idx
    if num_samples * target_L > chain_length:
        raise ValueError(f"num_samples*L={num_samples*target_L} > chain {chain_length}")
    pool = rng.choice(chain_length, size=num_samples * target_L, replace=False)
    idx = np.empty((num_samples, target_L), dtype=np.int64)
    for s in range(num_samples):
        idx[s] = np.sort(pool[s * target_L:(s + 1) * target_L])
    return idx


def sample_indices(method, chain_length, target_L, num_samples, replace, rng):
    if method == "uniform":
        return uniform_indices(chain_length, target_L, num_samples, replace, rng)
    return dewitt_indices(chain_length, target_L, num_samples, replace, rng)


def analyze_knot(args):
    import knoodle
    (knot_idx, filepath, expected_fp, lengths, replace,
     chain_length, samples_per_length, method) = args
    points = load_polygon(filepath)
    rows = []
    if points is None:
        return rows
    rng = np.random.default_rng(42 + knot_idx * chain_length)

    for L in lengths:
        n_samp = samples_per_length[L]
        idx = sample_indices(method, chain_length, L, n_samp, replace, rng)
        for s_id in range(n_samp):
            sample = points[idx[s_id]]
            try:
                a = knoodle.KnotAnalyzer(flatten_coords(sample),
                                         simplify=True, simplify_level=5)
                cc = int(a.crossing_count)
                fp = fingerprint(a)
            except Exception:
                cc, fp = None, None
            det = round(fp[0]) if fp is not None else None
            fp_str = (str(tuple(round(v, FP_STORE_ROUND) for v in fp))
                      if fp is not None else None)
            rows.append({
                "knot_id": knot_idx + 1,
                "length": L,
                "sample_id": s_id,
                "crossing_count": cc,
                "determinant": det,
                "fingerprint": fp_str,
                "knotted": int(cc is not None and cc > 0),
                "correct": int(fps_match(fp, expected_fp)),
            })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-folder", "-i", required=True)
    parser.add_argument("--p", type=int, required=True)
    parser.add_argument("--q", type=int, required=True)
    parser.add_argument("--output", "-o", required=True)
    parser.add_argument("--num-samples", "-n", type=int, default=DEFAULT_NUM_SAMPLES)
    parser.add_argument("--no-replacement", action="store_true")
    parser.add_argument("--method", choices=["dewitt", "uniform"], default="dewitt",
                        help="Sampling: dewitt (one random point per chunk) or "
                             "uniform (L random points from the whole chain)")
    parser.add_argument("--lengths", default=None,
                        help="Comma-separated integer L values (need not divide N).")
    parser.add_argument("--chain-length", type=int, default=None)
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.input_folder, "file_*.tsv")))
    if not files:
        print(f"ERROR: no file_*.tsv in {args.input_folder}", file=sys.stderr)
        sys.exit(1)

    if args.chain_length is None:
        with open(files[0]) as f:
            args.chain_length = sum(1 for _ in f)

    # Expected fingerprint: mode over a faithful downsample of a probe subset.
    expected_fp = compute_expected_fingerprint(files, args.chain_length)

    if args.lengths:
        lengths = sorted({int(s) for s in args.lengths.split(",") if s.strip()})
        lengths = [L for L in lengths if MIN_LENGTH <= L <= args.chain_length]
    else:
        lengths = sorted({args.chain_length // s
                          for s in range(1, args.chain_length + 1)
                          if args.chain_length % s == 0 and args.chain_length // s >= MIN_LENGTH})
    lengths = sorted(lengths, reverse=True)

    replace = not args.no_replacement
    samples_per_length = {}
    capped = []
    for L in lengths:
        b = chunk_boundaries(args.chain_length, L)
        min_chunk = int((b[1:] - b[:-1]).min())
        n = min(args.num_samples, min_chunk) if not replace else args.num_samples
        samples_per_length[L] = n
        if not replace and n < args.num_samples:
            capped.append(f"L={L}: {args.num_samples}->{n}")

    print("=" * 60)
    print(f"Torus knot downsampling analysis  T({args.p}, {args.q})")
    print(f"  Expected fingerprint:  {expected_fp}   (|Delta(t)| at t={FP_KEYS})")
    print(f"  Knots:                 {len(files)}")
    print(f"  Vertices per knot:     {args.chain_length}")
    print(f"  Lengths ({len(lengths)}):        {lengths}")
    print(f"  Samples per length:    {args.num_samples}")
    print(f"  Sampling method:       {args.method}")
    print(f"  Replacement:           {'no' if args.no_replacement else 'yes'}")
    if capped:
        print(f"  Caps: {', '.join(capped)}")
    print(f"  Workers:               {NUM_WORKERS}")
    print(f"  Output:                {args.output}")
    print("=" * 60, flush=True)

    tasks = [(i, f, expected_fp, lengths, replace, args.chain_length, samples_per_length, args.method)
             for i, f in enumerate(files)]

    rows = []
    with Pool(processes=NUM_WORKERS, maxtasksperchild=10) as pool:
        for r in tqdm(pool.imap_unordered(analyze_knot, tasks),
                      total=len(tasks), desc="Analyzing", mininterval=2.0):
            rows.extend(r)

    df = pd.DataFrame(rows).sort_values(["knot_id", "length", "sample_id"])
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"\n[INFO] Saved {len(df)} rows to {args.output}")

    print("\nKnot topology preservation vs L:")
    for L in sorted(df["length"].unique(), reverse=True):
        sub = df[df["length"] == L]
        knotted = sub["knotted"].mean() * 100
        correct = sub["correct"].mean() * 100
        print(f"  L = {L:>5}: still knotted = {knotted:6.2f}%   correct type = {correct:6.2f}%")


if __name__ == "__main__":
    main()
