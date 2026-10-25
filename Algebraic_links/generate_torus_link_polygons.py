#!/usr/bin/env python3
"""
Generate (p, q)-torus link 10k-gon ensembles from the algebraic construction
V = {x^p + y^q = 0} intersected with S^3.

Each link has d = gcd(p, q) components; each component is saved as its own
TSV so the existing analysis pipeline can read it. Per link we randomise
theta_0 and apply a shared random 3D rotation to all components (so the
relative link geometry is preserved).
"""

import argparse
import multiprocessing
from math import gcd
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.spatial.transform import Rotation
from multiprocessing import Pool
from tqdm import tqdm


def solve_u(p, q, r=1.0):
    """Real positive root of u^p + u^q = r^2."""
    return brentq(lambda u: u ** p + u ** q - r * r, 1e-6, 1.0 - 1e-9)


def stereo_project(pts_r4, radius=1.0):
    """Stereographic projection from north pole (0,0,0,radius) to R^3."""
    denom = radius - pts_r4[:, 3]
    return pts_r4[:, :3] * (radius / denom)[:, None]


def generate_link_components(n_vertices, p, q, u, theta0, rotation_matrix):
    """Generate the d = gcd(p, q) components of a (p, q)-torus link.

    Component k:
        t = sqrt(u) exp(i theta), theta in [theta0, theta0 + 2 pi / d)
        x = t^q                = u^(q/2) exp(i q theta)
        y = nu_k * t^p         = u^(p/2) exp(i (p theta + pi (2k+1)/q))
    """
    d = gcd(p, q)
    theta = theta0 + np.linspace(0.0, 2.0 * np.pi / d,
                                 n_vertices, endpoint=False)
    amp_x = u ** (q / 2.0)
    amp_y = u ** (p / 2.0)
    x = amp_x * np.exp(1j * q * theta)
    components = []
    for k in range(d):
        y = amp_y * np.exp(1j * (p * theta + np.pi * (2 * k + 1) / q))
        pts_r4 = np.column_stack([x.real, x.imag, y.real, y.imag])
        pts_r3 = stereo_project(pts_r4, radius=1.0)
        pts_r3 = pts_r3 @ rotation_matrix.T
        components.append(pts_r3)
    return components


def worker(task):
    link_idx, n_vertices, p, q, u, seed, out_dir, digits = task
    rng = np.random.default_rng(seed)
    theta0 = float(rng.uniform(0.0, 2.0 * np.pi))
    rotation = Rotation.random(random_state=rng).as_matrix()
    components = generate_link_components(n_vertices, p, q, u, theta0, rotation)
    for k, comp in enumerate(components):
        path = out_dir / f"link_{link_idx + 1:0{digits}d}_comp_{k}.tsv"
        np.savetxt(path, comp, delimiter="\t", fmt="%.6f")
    return link_idx


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p", type=int, required=True)
    parser.add_argument("--q", type=int, required=True)
    parser.add_argument("--num-files", type=int, default=1000)
    parser.add_argument("--num-vertices", type=int, default=10000)
    parser.add_argument("--output-folder", "-o", default=None,
                        help="Default: new_run_2026/rotated_tsv/torus_links/T_p_q")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    d = gcd(args.p, args.q)
    u = solve_u(args.p, args.q, r=1.0)

    out_dir = Path(args.output_folder or
                   f"new_run_2026/rotated_tsv/torus_links/T_{args.p}_{args.q}")
    out_dir.mkdir(parents=True, exist_ok=True)
    digits = max(4, len(str(args.num_files)))

    tasks = [(i, args.num_vertices, args.p, args.q, u,
              args.seed + i, out_dir, digits)
             for i in range(args.num_files)]

    print("=" * 60)
    print(f"Torus link T({args.p}, {args.q})")
    print(f"  gcd(p,q) = d:          {d}")
    print(f"  u = |t|^2:             {u:.6f}")
    print(f"  Expected pairwise |lk|: {args.p * args.q / (d * d):g}")
    print(f"  Files:                 {args.num_files}")
    print(f"  Vertices per component:{args.num_vertices}")
    print(f"  Output:                {out_dir}")
    print("=" * 60)

    n_workers = max(1, multiprocessing.cpu_count() - 2)
    with Pool(processes=n_workers) as pool:
        for _ in tqdm(pool.imap_unordered(worker, tasks),
                      total=len(tasks), desc="Generating"):
            pass

    print(f"\n[INFO] Wrote {args.num_files} links "
          f"({d * args.num_files} component files) to {out_dir}")


if __name__ == "__main__":
    main()
