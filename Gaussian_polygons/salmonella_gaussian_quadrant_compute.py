#!/usr/bin/env python3
"""
Compute per-polygon crossing-number statistics for the Gaussian-random-polygon
quadrant analysis, with a length sweep.

For each Gaussian polygon TSV we compute:
  - the simplified crossing number of the full N-vertex polygon
  - for each downsampling length L in --lengths, the mean simplified
    crossing number across NUM_SAMPLES deWitt L-gons (chunk size = N // L,
    leaving floor(N - L * (N//L)) vertices unused at the tail).

The output CSV has columns:
    filename, crossing_full, error,
    mean_crossing_L<L1>, mean_crossing_L<L2>, ...

so the plot script can replay arbitrary threshold sweeps without recomputing.

Usage:
    python salmonella_gaussian_quadrant_compute.py \\
        --input-folder Gaussian_polygons \\
        --output Gaussian_polygons/quadrant_stats.csv \\
        --lengths 100,75,50,25,20,15,10,6
"""

import os
import sys
import argparse
import multiprocessing
import time

CPU_COUNT = multiprocessing.cpu_count()
NUM_WORKERS = max(1, CPU_COUNT - 2)
THREADS_PER_PROCESS = 1

os.environ["OMP_NUM_THREADS"] = str(THREADS_PER_PROCESS)
os.environ["OPENBLAS_NUM_THREADS"] = str(THREADS_PER_PROCESS)
os.environ["MKL_NUM_THREADS"] = str(THREADS_PER_PROCESS)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(THREADS_PER_PROCESS)
os.environ["NUMEXPR_NUM_THREADS"] = str(THREADS_PER_PROCESS)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import glob
import numpy as np
import pandas as pd
from multiprocessing import Pool
from tqdm import tqdm

PROGRESS_EVERY = 100
DEFAULT_LENGTHS = "100,75,50,25,20,15,10,6"
DEFAULT_NUM_SAMPLES = 10000


def load_polygon(filepath):
    try:
        pts = np.loadtxt(filepath, delimiter="\t")
        return pts if pts.size else None
    except Exception:
        return None


def flatten_coords(points):
    return [float(c) for p in points for c in p]


def mean_dewitt_crossings(points, target_length, num_samples, rng):
    """Mean simplified crossing number across num_samples deWitt L-gons.

    chunk_size = floor(N / L). The last (N - L*chunk_size) vertices are
    ignored to keep all chunks the same size.
    """
    import knoodle

    n = len(points)
    chunk = n // target_length
    if chunk < 1:
        return None, "polygon too short for target length"

    crossings = np.full(num_samples, -1, dtype=np.int32)
    for i in range(num_samples):
        idx = np.empty(target_length, dtype=np.int64)
        for c in range(target_length):
            idx[c] = c * chunk + int(rng.integers(0, chunk))
        sample = points[idx]
        try:
            a = knoodle.KnotAnalyzer(flatten_coords(sample),
                                    simplify=True, simplify_level=5)
            crossings[i] = a.crossing_count
        except Exception:
            continue

    valid = crossings[crossings >= 0]
    if valid.size == 0:
        return None, "all subsample analyses failed"
    return float(np.mean(valid)), None


def analyze_polygon(args):
    """Compute (crossing_full, dict of mean_crossing_L) for one polygon."""
    import knoodle

    filepath, filename, file_index, lengths, num_samples = args
    result = {"filename": filename, "crossing_full": None, "error": None}
    for L in lengths:
        result[f"mean_crossing_L{L}"] = None

    try:
        points = load_polygon(filepath)
        if points is None:
            result["error"] = "Failed to load polygon"
            return result

        analyzer = knoodle.KnotAnalyzer(flatten_coords(points),
                                       simplify=True, simplify_level=5)
        result["crossing_full"] = int(analyzer.crossing_count)

        for L in lengths:
            rng = np.random.default_rng(42 + file_index * 1000003 + L)
            mean_val, err = mean_dewitt_crossings(points, L, num_samples, rng)
            if mean_val is None:
                result["error"] = f"L={L}: {err}"
                continue
            result[f"mean_crossing_L{L}"] = mean_val

    except Exception as e:
        result["error"] = str(e)
    return result


def main():
    parser = argparse.ArgumentParser(description="Per-polygon Gaussian quadrant stats with length sweep")
    parser.add_argument("--input-folder", "-i", required=True)
    parser.add_argument("--output", "-o", required=True)
    parser.add_argument("--lengths", default=DEFAULT_LENGTHS,
                        help=f"Comma-separated downsampling lengths (default: {DEFAULT_LENGTHS})")
    parser.add_argument("--num-samples", type=int, default=DEFAULT_NUM_SAMPLES,
                        help=f"deWitt subsamples per length per polygon (default: {DEFAULT_NUM_SAMPLES})")
    args = parser.parse_args()

    lengths = sorted({int(s) for s in args.lengths.split(",") if s.strip()},
                     reverse=True)
    if not lengths:
        raise SystemExit("No lengths specified")

    files = sorted(glob.glob(os.path.join(args.input_folder, "*.tsv")))
    if not files:
        print(f"ERROR: no .tsv files in {args.input_folder}", file=sys.stderr)
        sys.exit(1)

    n_total = len(files)
    tasks = [(f, os.path.basename(f), i, lengths, args.num_samples)
             for i, f in enumerate(files)]

    print("=" * 60)
    print("Gaussian polygon quadrant compute (length sweep)")
    print(f"  Input folder:               {args.input_folder}")
    print(f"  Polygons:                   {n_total}")
    print(f"  Lengths:                    {lengths}")
    print(f"  deWitt samples per length:  {args.num_samples}")
    print(f"  Workers:                    {NUM_WORKERS}")
    print(f"  Output CSV:                 {args.output}")
    print("=" * 60, flush=True)

    start = time.monotonic()
    results = []
    last_log_idx = 0

    with Pool(processes=NUM_WORKERS, maxtasksperchild=10) as pool:
        for idx, res in enumerate(tqdm(pool.imap_unordered(analyze_polygon, tasks),
                                       total=n_total, desc="Computing",
                                       mininterval=2.0, smoothing=0.05), start=1):
            results.append(res)
            if idx - last_log_idx >= PROGRESS_EVERY or idx == n_total:
                elapsed = time.monotonic() - start
                rate = idx / elapsed if elapsed > 0 else 0.0
                remaining = (n_total - idx) / rate if rate > 0 else float("inf")
                print(f"[progress] polygon {idx} of {n_total} "
                      f"({100*idx/n_total:.1f}%) "
                      f"elapsed={elapsed/60:.1f} min  "
                      f"rate={rate:.2f}/s  "
                      f"eta={remaining/60:.1f} min",
                      flush=True)
                last_log_idx = idx

    column_order = ["filename", "crossing_full"] + [f"mean_crossing_L{L}" for L in lengths] + ["error"]
    df = pd.DataFrame(results)
    for col in column_order:
        if col not in df.columns:
            df[col] = None
    df = df[column_order].sort_values("filename").reset_index(drop=True)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"\n[INFO] Saved {len(df)} rows to {args.output}")

    failed = df["error"].notna().sum()
    if failed:
        print(f"[WARN] {failed} polygons had errors")

    valid = df[df["error"].isna()]
    if len(valid):
        print(f"\nSummary ({len(valid)} valid):")
        print(f"  Full-chain crossings: median={valid['crossing_full'].median():.1f}, "
              f"mean={valid['crossing_full'].mean():.2f}, "
              f"min={valid['crossing_full'].min():.0f}, "
              f"max={valid['crossing_full'].max():.0f}")
        for L in lengths:
            col = f"mean_crossing_L{L}"
            s = valid[col].dropna()
            if len(s):
                print(f"  L={L:>3}-gon mean:     median={s.median():.4f}, "
                      f"mean={s.mean():.4f}, "
                      f"min={s.min():.4f}, "
                      f"max={s.max():.4f}")


if __name__ == "__main__":
    main()
