#!/usr/bin/env python3
"""
Generate Gaussian random polygons (Python port of Mathematica's
GaussianRandomPolygon[n_]):

    Accumulate[
        Standardize[RandomVariate[NormalDistribution[], {n, 3}], Mean, 1 &]
    ]

For each polygon: draw an (n, 3) matrix of standard-normal samples, subtract
the column means so the increments sum to zero, then accumulate to obtain
the vertex positions of a closed random walk in 3D.

Each polygon is saved as a tab-delimited TSV file (one vertex per row,
3 columns, 6 decimal places) so it slots into the existing knot analysis
pipeline.
"""

import os
import sys
import argparse
import multiprocessing
from pathlib import Path

CPU_COUNT = multiprocessing.cpu_count()
NUM_WORKERS = max(1, CPU_COUNT - 2)
THREADS_PER_PROCESS = 1

os.environ["OMP_NUM_THREADS"] = str(THREADS_PER_PROCESS)
os.environ["OPENBLAS_NUM_THREADS"] = str(THREADS_PER_PROCESS)
os.environ["MKL_NUM_THREADS"] = str(THREADS_PER_PROCESS)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(THREADS_PER_PROCESS)
os.environ["NUMEXPR_NUM_THREADS"] = str(THREADS_PER_PROCESS)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
from multiprocessing import Pool
from tqdm import tqdm


def gaussian_random_polygon(n, rng):
    """Port of Mathematica's GaussianRandomPolygon[n_]."""
    v = rng.standard_normal((n, 3))
    v -= v.mean(axis=0)
    return np.cumsum(v, axis=0)


def worker(args):
    idx, n, seed, output_path = args
    rng = np.random.default_rng(seed)
    polygon = gaussian_random_polygon(n, rng)
    np.savetxt(output_path, polygon, delimiter="\t", fmt="%.6f")
    return idx


def main():
    parser = argparse.ArgumentParser(description="Generate Gaussian random polygons.")
    parser.add_argument("--n-vertices", type=int, default=25000,
                        help="Vertices per polygon (default: 25000)")
    parser.add_argument("--num-polygons", type=int, default=25000,
                        help="Number of polygons to generate (default: 25000)")
    parser.add_argument("--output-folder", "-o", default="Gaussian_polygons",
                        help="Output folder (default: Gaussian_polygons)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base seed; per-polygon seed = base + polygon index (default: 42)")
    args = parser.parse_args()

    out_dir = Path(args.output_folder)
    out_dir.mkdir(parents=True, exist_ok=True)

    digits = max(5, len(str(args.num_polygons - 1)))
    tasks = [
        (i, args.n_vertices, args.seed + i,
         out_dir / f"polygon_{i:0{digits}d}.tsv")
        for i in range(args.num_polygons)
    ]

    print("=" * 60)
    print("Generating Gaussian random polygons")
    print(f"  Vertices per polygon: {args.n_vertices}")
    print(f"  Polygon count:        {args.num_polygons}")
    print(f"  Output folder:        {out_dir}")
    print(f"  Workers:              {NUM_WORKERS}")
    print(f"  Base seed:            {args.seed}")
    print("=" * 60)

    with Pool(processes=NUM_WORKERS) as pool:
        for _ in tqdm(pool.imap_unordered(worker, tasks),
                      total=len(tasks), desc="Generating"):
            pass

    print(f"\n[INFO] Wrote {args.num_polygons} polygons to {out_dir}")


if __name__ == "__main__":
    main()
