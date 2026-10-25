#!/usr/bin/env python3
"""
DeWitt downsampling + pairwise linking-number analysis for (p, q)-torus link
ensembles produced by generate_torus_link_polygons.py.

For each link, each valid stride s (dividing the vertex count and giving
L >= MIN_LENGTH), and each subsample:
  - Subsample every component to L vertices via one-random-point-per-chunk
    ('stride' / De Witt method); optional --no-replacement caps subsamples to
    the chunk size s automatically.
  - Compute the discrete Gauss linking number for every ordered pair (i, j)
    of components.

Output: a single CSV of per-subsample lk values. The 'perfect classification'
criterion is |lk| == expected (= p*q / gcd(p,q)^2), rounded within a small
tolerance.
"""

import argparse
import glob
import multiprocessing
import os
import re
import sys
from collections import defaultdict
from math import gcd

import numpy as np
import pandas as pd
from multiprocessing import Pool
from tqdm import tqdm

MIN_LENGTH = 5
DEFAULT_NUM_SAMPLES = 100
LINK_RE = re.compile(r"link_(\d+)_comp_(\d+)\.tsv$")


def linking_number_gauss(g1, g2):
    """Midpoint-form discrete Gauss linking integral for polygonal loops."""
    dr1 = np.roll(g1, -1, axis=0) - g1
    dr2 = np.roll(g2, -1, axis=0) - g2
    m1 = 0.5 * (g1 + np.roll(g1, -1, axis=0))
    m2 = 0.5 * (g2 + np.roll(g2, -1, axis=0))
    r = m2[None, :, :] - m1[:, None, :]
    r_norm = np.linalg.norm(r, axis=2)
    cross = np.cross(dr1[:, None, :], dr2[None, :, :])
    with np.errstate(divide="ignore", invalid="ignore"):
        integrand = np.sum(cross * r, axis=2) / (r_norm ** 3)
    integrand[~np.isfinite(integrand)] = 0.0
    return float(np.sum(integrand) / (4.0 * np.pi))


def get_valid_strides(length, min_length):
    return sorted(s for s in range(1, length + 1)
                  if length % s == 0 and length // s >= min_length)


def chunk_boundaries(chain_length, target_L):
    """Return length-(target_L+1) array of chunk start indices.

    Chunk k covers indices [b[k], b[k+1]); sizes are floor(N/L) or
    ceil(N/L), covering all N vertices exactly (no leftover).
    """
    return np.floor(np.arange(target_L + 1) * chain_length / target_L).astype(np.int64)


def load_link(link_idx, comp_paths):
    return [np.loadtxt(p, delimiter="\t") for p in comp_paths]


def dewitt_indices(chain_length, target_L, num_samples, replace, rng):
    """Return an (num_samples, target_L) matrix of index sets, one random
    vertex per variable-width chunk.

    Uses variable chunk sizes so target_L need not divide chain_length.
    With replace=False, per chunk we take a distinct permutation entry
    per row (requires num_samples <= min chunk size).
    """
    boundaries = chunk_boundaries(chain_length, target_L)
    chunk_sizes = boundaries[1:] - boundaries[:-1]
    min_chunk = int(chunk_sizes.min())

    if replace:
        idx = np.empty((num_samples, target_L), dtype=np.int64)
        for c in range(target_L):
            start = int(boundaries[c])
            size = int(chunk_sizes[c])
            idx[:, c] = start + rng.integers(0, size, size=num_samples)
        return idx

    if num_samples > min_chunk:
        raise ValueError(f"num_samples={num_samples} > min chunk size {min_chunk} "
                         "with replace=False")
    idx = np.empty((num_samples, target_L), dtype=np.int64)
    for c in range(target_L):
        start = int(boundaries[c])
        size = int(chunk_sizes[c])
        perm = rng.permutation(size)[:num_samples]
        idx[:, c] = start + perm
    return idx


def uniform_indices(chain_length, target_L, num_samples, replace, rng):
    """Return an (num_samples, target_L) matrix; each row is target_L indices
    drawn uniformly at random from the whole chain (no chunk structure),
    sorted to preserve cyclic order.

    replace=True: each sample draws target_L distinct indices independently.
    replace=False: draw one pool of num_samples * target_L distinct indices
    from the chain and partition it into the samples (requires
    num_samples * target_L <= chain_length).
    """
    if replace:
        idx = np.empty((num_samples, target_L), dtype=np.int64)
        for s in range(num_samples):
            idx[s] = np.sort(rng.choice(chain_length, size=target_L, replace=False))
        return idx

    if num_samples * target_L > chain_length:
        raise ValueError(f"num_samples*L = {num_samples*target_L} > chain length "
                         f"{chain_length} with replace=False")
    pool = rng.choice(chain_length, size=num_samples * target_L, replace=False)
    idx = np.empty((num_samples, target_L), dtype=np.int64)
    for s in range(num_samples):
        idx[s] = np.sort(pool[s * target_L:(s + 1) * target_L])
    return idx


def sample_indices(method, chain_length, target_L, num_samples, replace, rng):
    if method == "uniform":
        return uniform_indices(chain_length, target_L, num_samples, replace, rng)
    return dewitt_indices(chain_length, target_L, num_samples, replace, rng)


def analyze_link(args):
    (link_idx, comp_paths, p, q, expected_abs_lk,
     lengths, num_samples, replace, chain_length,
     samples_per_length, method) = args
    components = load_link(link_idx, comp_paths)
    d = len(components)
    rng = np.random.default_rng(42 + link_idx * chain_length)

    rows = []
    for L in lengths:
        n_samp = samples_per_length[L]
        comp_idx = [sample_indices(method, chain_length, L, n_samp, replace, rng)
                    for _ in range(d)]
        for s_id in range(n_samp):
            sampled = [components[k][comp_idx[k][s_id]] for k in range(d)]
            for i in range(d):
                for j in range(i + 1, d):
                    lk = linking_number_gauss(sampled[i], sampled[j])
                    rows.append({
                        "link_id": link_idx + 1,
                        "length": L,
                        "sample_id": s_id,
                        "comp_i": i,
                        "comp_j": j,
                        "lk": lk,
                        "lk_abs_int": int(round(abs(lk))),
                        "preserved": int(abs(abs(lk) - expected_abs_lk) < 0.4),
                    })
    return rows


def discover_links(input_folder):
    """Return dict link_idx -> sorted list of component file paths."""
    files = glob.glob(os.path.join(input_folder, "link_*.tsv"))
    groups = defaultdict(list)
    for f in files:
        m = LINK_RE.search(f)
        if not m:
            continue
        link_idx = int(m.group(1))
        comp_idx = int(m.group(2))
        groups[link_idx].append((comp_idx, f))
    for k in groups:
        groups[k] = [f for _, f in sorted(groups[k])]
    return dict(sorted(groups.items()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-folder", "-i", required=True,
                        help="Folder with link_XXXX_comp_Y.tsv files")
    parser.add_argument("--p", type=int, required=True)
    parser.add_argument("--q", type=int, required=True)
    parser.add_argument("--output", "-o", required=True, help="Output CSV path")
    parser.add_argument("--num-samples", "-n", type=int, default=DEFAULT_NUM_SAMPLES)
    parser.add_argument("--no-replacement", action="store_true")
    parser.add_argument("--method", choices=["dewitt", "uniform"], default="dewitt",
                        help="Sampling: dewitt (one random point per chunk) or "
                             "uniform (L random points from the whole chain)")
    parser.add_argument("--max-length", type=int, default=None,
                        help="Optional cap on L (default: no cap). Ignored if --lengths is used.")
    parser.add_argument("--lengths", default=None,
                        help="Comma-separated integer L values (need not divide N). "
                             "Default: all divisors of N with L>=5 (subject to --max-length).")
    parser.add_argument("--chain-length", type=int, default=None,
                        help="Vertices per component (default: auto-detect)")
    args = parser.parse_args()

    d = gcd(args.p, args.q)
    expected_abs_lk = args.p * args.q / (d * d)

    links = discover_links(args.input_folder)
    if not links:
        print(f"ERROR: no link_*_comp_*.tsv in {args.input_folder}", file=sys.stderr)
        sys.exit(1)

    if args.chain_length is None:
        first_component = next(iter(links.values()))[0]
        with open(first_component) as f:
            args.chain_length = sum(1 for _ in f)

    if args.lengths:
        lengths = sorted({int(s) for s in args.lengths.split(",") if s.strip()})
        lengths = [L for L in lengths if L >= MIN_LENGTH and L <= args.chain_length]
    else:
        strides = get_valid_strides(args.chain_length, MIN_LENGTH)
        if args.max_length is not None:
            strides = [s for s in strides if args.chain_length // s <= args.max_length]
        lengths = sorted({args.chain_length // s for s in strides})
    lengths = sorted(lengths, reverse=True)

    replace = not args.no_replacement
    samples_per_length = {}
    capped_msgs = []
    for L in lengths:
        boundaries = chunk_boundaries(args.chain_length, L)
        min_chunk = int((boundaries[1:] - boundaries[:-1]).min())
        cap = min_chunk if not replace else None
        n = min(args.num_samples, cap) if cap is not None else args.num_samples
        samples_per_length[L] = n
        if replace is False and n < args.num_samples:
            capped_msgs.append(f"L={L} (min chunk {min_chunk}): {args.num_samples} -> {n}")

    print("=" * 60)
    print(f"Torus link downsampling analysis  T({args.p}, {args.q})")
    print(f"  d = gcd(p,q):                 {d}")
    print(f"  Expected pairwise |lk|:       {expected_abs_lk:g}")
    print(f"  Links:                        {len(links)}")
    print(f"  Vertices per component:       {args.chain_length}")
    print(f"  Lengths ({len(lengths)}):"
          f"                     {lengths}")
    print(f"  Samples per length (default): {args.num_samples}")
    print(f"  Sampling method:              {args.method}")
    print(f"  Replacement:                  {'no' if args.no_replacement else 'yes'}")
    if capped_msgs:
        print("  --no-replacement caps applied:")
        for m in capped_msgs:
            print(f"    {m}")
    print(f"  Output CSV:                   {args.output}")
    print("=" * 60)

    tasks = []
    for link_idx_1based, comps in links.items():
        tasks.append((
            link_idx_1based - 1, comps, args.p, args.q, expected_abs_lk,
            lengths, args.num_samples, replace, args.chain_length,
            samples_per_length, args.method,
        ))

    n_workers = int(os.environ.get("TORUS_WORKERS", "0")) or max(1, multiprocessing.cpu_count() - 2)
    rows = []
    with Pool(processes=n_workers, maxtasksperchild=20) as pool:
        for r in tqdm(pool.imap_unordered(analyze_link, tasks),
                      total=len(tasks), desc="Analyzing", mininterval=2.0):
            rows.extend(r)

    df = pd.DataFrame(rows).sort_values(["link_id", "length", "sample_id", "comp_i", "comp_j"])
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"\n[INFO] Saved {len(df)} rows to {args.output}")

    # Quick summary
    print("\nLinking-number preservation vs L:")
    for L in sorted(df["length"].unique(), reverse=True):
        sub = df[df["length"] == L]
        if len(sub) == 0:
            continue
        frac = sub["preserved"].mean() * 100
        distinct_lks = sub["lk_abs_int"].value_counts().to_dict()
        print(f"  L = {L:>5}: preserved = {frac:6.2f}%   "
              f"|lk| distribution: {distinct_lks}")


if __name__ == "__main__":
    main()
