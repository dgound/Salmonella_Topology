#!/usr/bin/env python3
"""
Compute per-object crossing-number statistics for the quadrant analysis.

For each object in the combined knotted + unknotted population:
  - Crossing number of the full 10k-gon
  - Median crossing number across 1000 deWitt 20-gons
    (deWitt = stride sampling: split the chain into 500-point chunks,
     pick one uniformly-random point per chunk, repeat 1000 times)

Output is a single CSV ready to feed into salmonella_quadrant_plot.py.

Usage:
    python salmonella_quadrant_compute.py \\
        --knotted-dir new_run_2026/rotated_tsv/knotted \\
        --unknotted-dir new_run_2026/rotated_tsv/unknotted \\
        --output new_run_2026/quadrant_stats.csv
"""

import os
import sys
import argparse
import multiprocessing

CPU_COUNT = multiprocessing.cpu_count()
NUM_WORKERS = max(1, CPU_COUNT - 2)
THREADS_PER_PROCESS = 1

# Set internal threading BEFORE other imports so each worker stays single-threaded
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

CHAIN_LENGTH = 10000
TARGET_LENGTH = 20
STRIDE = CHAIN_LENGTH // TARGET_LENGTH  # 500
NUM_20GON_SAMPLES = 10000


def load_polygon(filepath):
    try:
        points = np.loadtxt(filepath, delimiter='\t')
        if points.size == 0:
            return None
        return points
    except Exception:
        return None


def flatten_coords(points):
    return [float(c) for p in points for c in p]


def sample_dewitt_20gons(points, num_samples, rng):
    """One random point per 500-point chunk; repeated num_samples times."""
    num_chunks = len(points) // STRIDE
    samples = []
    for _ in range(num_samples):
        indices = [chunk * STRIDE + rng.integers(0, STRIDE) for chunk in range(num_chunks)]
        samples.append(points[indices])
    return samples


def analyze_object(args):
    """Return (summary_dict, crossings_array) for one object.

    The summary dict has the per-object stats (matches the summary CSV
    schema). The crossings_array is a length-NUM_20GON_SAMPLES int16 array
    of per-sample 20-gon crossing counts (or empty if everything failed).
    """
    import knoodle

    filepath, filename, population, file_index = args
    empty = np.empty(0, dtype=np.int16)
    try:
        points = load_polygon(filepath)
        if points is None:
            return ({"filename": filename, "population": population,
                     "crossing_10k": None, "median_crossing_20gon": None,
                     "error": "Failed to load polygon"}, empty)

        analyzer = knoodle.KnotAnalyzer(flatten_coords(points),
                                       simplify=True, simplify_level=5)
        crossing_10k = analyzer.crossing_count

        rng = np.random.default_rng(42 + file_index)
        samples = sample_dewitt_20gons(points, NUM_20GON_SAMPLES, rng)

        crossings = np.full(NUM_20GON_SAMPLES, -1, dtype=np.int16)
        for i, sample in enumerate(samples):
            try:
                a = knoodle.KnotAnalyzer(flatten_coords(sample),
                                        simplify=True, simplify_level=5)
                crossings[i] = a.crossing_count
            except Exception:
                continue

        valid = crossings[crossings >= 0]
        if valid.size == 0:
            return ({"filename": filename, "population": population,
                     "crossing_10k": crossing_10k, "median_crossing_20gon": None,
                     "error": "All 20-gon analyses failed"}, empty)

        return ({"filename": filename, "population": population,
                 "crossing_10k": int(crossing_10k),
                 "median_crossing_20gon": float(np.median(valid)),
                 "error": None}, valid)

    except Exception as e:
        return ({"filename": filename, "population": population,
                 "crossing_10k": None, "median_crossing_20gon": None,
                 "error": str(e)}, empty)


def build_task_list(knotted_dir, unknotted_dir, exclude_knotted, exclude_unknotted):
    """Collect file lists for both populations, filtering per-population basenames."""
    tasks = []
    for population, directory, excluded in [
        ("knotted", knotted_dir, set(exclude_knotted)),
        ("unknotted", unknotted_dir, set(exclude_unknotted)),
    ]:
        files = sorted(glob.glob(os.path.join(directory, "*.tsv")))
        files = [f for f in files if os.path.basename(f) not in excluded]
        for f in files:
            tasks.append((f, os.path.basename(f), population))
    return [(fp, fn, pop, idx) for idx, (fp, fn, pop) in enumerate(tasks)]


def main():
    parser = argparse.ArgumentParser(description="Compute quadrant analysis stats")
    parser.add_argument("--knotted-dir", required=True)
    parser.add_argument("--unknotted-dir", required=True)
    parser.add_argument("--output", "-o", required=True,
                        help="Output CSV path")
    parser.add_argument("--exclude-knotted", nargs="*", default=[],
                        help="Basenames to exclude from the knotted population")
    parser.add_argument("--exclude-unknotted", nargs="*", default=[],
                        help="Basenames to exclude from the unknotted population")
    parser.add_argument("--samples-output", default=None,
                        help="Parquet path for the per-sample 20-gon counts "
                             "(default: same dir as --output, suffix _samples.parquet)")
    args = parser.parse_args()

    tasks = build_task_list(args.knotted_dir, args.unknotted_dir,
                            args.exclude_knotted, args.exclude_unknotted)

    print("=" * 60)
    print("Quadrant analysis: compute crossing numbers")
    print(f"Knotted dir:   {args.knotted_dir}")
    print(f"Unknotted dir: {args.unknotted_dir}")
    print(f"Total objects: {len(tasks)}")
    print(f"Workers: {NUM_WORKERS} (threads per worker: {THREADS_PER_PROCESS})")
    print(f"deWitt samples per object: {NUM_20GON_SAMPLES} (stride {STRIDE} -> 20-gons)")
    print("=" * 60)

    with Pool(processes=NUM_WORKERS, maxtasksperchild=20) as pool:
        pairs = list(tqdm(
            pool.imap(analyze_object, tasks),
            total=len(tasks),
            desc="Computing",
        ))

    summaries = [p[0] for p in pairs]
    df = pd.DataFrame(summaries)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"\n[INFO] Saved {len(df)} summary rows to {args.output}")

    samples_path = args.samples_output
    if samples_path is None:
        root, _ = os.path.splitext(args.output)
        samples_path = root + "_samples.parquet"

    parts = []
    for summary, crossings in pairs:
        if crossings.size == 0:
            continue
        parts.append(pd.DataFrame({
            "filename": np.repeat(summary["filename"], crossings.size),
            "population": np.repeat(summary["population"], crossings.size),
            "crossing_count": crossings,
        }))
    if parts:
        samples_df = pd.concat(parts, ignore_index=True)
        samples_df["filename"] = samples_df["filename"].astype("category")
        samples_df["population"] = samples_df["population"].astype("category")
        samples_df.to_parquet(samples_path, index=False)
        print(f"[INFO] Saved {len(samples_df)} per-sample rows to {samples_path}")

    failed = df["error"].notna().sum()
    if failed:
        print(f"[WARN] {failed} objects failed")

    valid = df[df["error"].isna()]
    print(f"\nSummary ({len(valid)} valid):")
    print(f"  10k-gon crossing counts:   median={valid['crossing_10k'].median():.1f}, "
          f"mean={valid['crossing_10k'].mean():.2f}")
    print(f"  20-gon median crossings:   median={valid['median_crossing_20gon'].median():.1f}, "
          f"mean={valid['median_crossing_20gon'].mean():.2f}")


if __name__ == "__main__":
    main()
