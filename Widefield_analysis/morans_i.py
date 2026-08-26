#!/usr/bin/env python3
"""Genomic Moran's I for single-curve walks.

Ported from `~/knot_analysis_Nir/Knot_analysis_pipeline/morans_i_genomic.cpp`
and adapted for this project, where each WalkID has exactly one ordered 3D
point per TP after drift correction and filtering. No DeWitt resampling, no
weighted centroid: the spatial coordinates are the walk's own ordered
(x_nm, y_nm, z_nm) points, one per TP.

Algorithm (matches the C++ reference exactly)
---------------------------------------------
Inputs: N spatial points in 3D (nm) and N integer genomic positions (one
per point).

1. Pairwise inverse-squared-distance spatial weights W_{ij} = 1 / (d_{ij}^2 + eps).
2. W_{ii} = 0.
3. Row-standardize W: W_{ij} <- W_{ij} / sum_j W_{ij}  (per row).
4. Map each genomic position g_i to an angle theta_i = g_i * 2 pi / cycle_bins.
5. Circular mean theta_bar = atan2(mean sin(theta_i), mean cos(theta_i)).
6. Circular deviations d_i = atan2(sin(theta_i - theta_bar),
                                    cos(theta_i - theta_bar)).
7. Standardize d: d_i <- d_i / sqrt(mean(d_i^2)).
8. I = (N / W_total) * (d^T W d) / (d^T d),  with W_total = sum_{ij} W_{ij}.

One-tailed permutation test for positive spatial autocorrelation: shuffle
the genomic-position vector M times, recompute I for each shuffle, and
report p = (#{I_perm >= I_obs} + 1) / (M + 1).

Generalization away from the hardcoded 20
-----------------------------------------
The reference C++ uses `angles[i] = genomic_positions[i] * 2 pi / 20.0`
everywhere, which hardcodes a 20-TP cycle. Here `cycle_bins` defaults to
`len(genomic_positions)` (i.e. the walk's own vertex count) but can be
overridden (`genome_cycle_bins`). The genomic positions themselves can be
absolute 0-based TP indices (e.g. `[2, 3, ..., 22]` for TP002..TP022) or
range-relative `[0, 1, ..., N-1]`; the caller picks the policy explicitly
so the downstream tables can carry that metadata.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import pdist, squareform


EPSILON = 1e-10


def _circular_chain_offset_matrix(N):
    """Return an N x N int matrix of circular-chain offsets along the curve.

    Each point i is a genomic bin on a closed chain (the bacterial genome
    is circular), so the "chain distance" between two curve vertices is
    `min(|i-j|, N - |i-j|)`. This is independent of the per-walk genomic
    *labels*: it is the curve / polygon order used only to decide which
    pairs are direct chain neighbors.
    """
    idx = np.arange(N)
    abs_diff = np.abs(idx[:, None] - idx[None, :])
    return np.minimum(abs_diff, N - abs_diff)


def _inverse_distance_weights(spatial_coords,
                              exclude_circular_genomic_offsets_leq=0):
    """Return row-standardized inverse-squared-distance weights, N x N.

    When `exclude_circular_genomic_offsets_leq > 0`, pairs whose circular
    chain offset is at most that value are zeroed **before** row
    standardization. The diagonal is always zero; setting the argument to
    `1` additionally zeroes the two direct genomic-chain neighbors (and,
    because the chain is circular, the first and last vertices count as
    neighbors too). Larger values remove a wider neighborhood along the
    chain. This is NOT a spatial-nearest-neighbor exclusion; it is a
    sequence-adjacency exclusion.

    After the mask, each row is re-standardized to sum to 1 (or kept at
    zero if the row has no surviving entries). `_morans_i_core` will
    raise if the resulting W_total is unusably small.
    """
    N = spatial_coords.shape[0]
    D = squareform(pdist(spatial_coords))
    # W_{ij} = 1 / (d^2 + eps); W_{ii} = 0
    D2 = D * D + EPSILON
    W = 1.0 / D2
    np.fill_diagonal(W, 0.0)
    if exclude_circular_genomic_offsets_leq > 0:
        circ_off = _circular_chain_offset_matrix(N)
        W = np.where(circ_off <= int(exclude_circular_genomic_offsets_leq),
                     0.0, W)
    # Row standardize (post-mask so zeroed rows stay zero)
    row_sums = W.sum(axis=1, keepdims=True)
    safe = row_sums > EPSILON
    W = np.where(safe, W / np.where(safe, row_sums, 1.0), W)
    return W


def _circular_standardize(genomic_positions, cycle_bins):
    """Port of the C++ circular deviation + standardization steps.

    Returns a 1-D array of length N, or raises ValueError if the
    deviations have zero variance (cannot standardize).
    """
    angles = np.asarray(genomic_positions, dtype=float) * (2.0 * np.pi / cycle_bins)
    mean_cos = float(np.mean(np.cos(angles)))
    mean_sin = float(np.mean(np.sin(angles)))
    X_bar = np.arctan2(mean_sin, mean_cos)
    diff = angles - X_bar
    d = np.arctan2(np.sin(diff), np.cos(diff))
    std_dev = float(np.sqrt(np.mean(d * d)))
    if std_dev <= EPSILON:
        raise ValueError('Standard deviation of circular deviations is zero '
                         '(genomic positions fold to a single angle).')
    return d / std_dev


def _morans_i_core(W, d):
    """Compute (N / W_total) * (d^T W d) / (d^T d). Port of core formula."""
    W_total = float(W.sum())
    if W_total <= EPSILON:
        raise ValueError('Sum of spatial weights is zero or too small.')
    numer = float(d @ W @ d)
    denom = float(d @ d)
    if denom <= EPSILON:
        raise ValueError('Denominator (sum d^2) is zero.')
    return (len(d) / W_total) * (numer / denom)


def compute_morans_i(spatial_coords, genomic_positions, cycle_bins=None,
                     exclude_circular_genomic_offsets_leq=0):
    """Observed genomic Moran's I for one walk.

    Parameters
    ----------
    spatial_coords : (N, 3) array-like of nm coordinates, ordered by TP.
    genomic_positions : (N,) array-like of integer genomic positions.
        In this project these are either absolute 0-based TP indices
        (e.g. [2, 3, ..., 22]) or range-relative [0, 1, ..., N-1]; the
        caller chooses.
    cycle_bins : int or None
        Divisor used to map genomic position -> angle. When None, the
        walk's own vertex count N is used, which is the "fits in one
        period" policy. Matches the reference C++ literal `20` only when
        you pass `cycle_bins=20`.
    exclude_circular_genomic_offsets_leq : int, default 0
        Remove direct genomic-chain neighbors from the spatial weight
        matrix (see `_inverse_distance_weights`). 0 disables the
        exclusion; 1 removes the diagonal plus the two immediate chain
        neighbors (circular).

    Returns
    -------
    float: observed Moran's I.
    """
    spatial_coords = np.asarray(spatial_coords, dtype=float)
    genomic_positions = np.asarray(genomic_positions, dtype=int)
    N = spatial_coords.shape[0]
    if N < 3:
        raise ValueError("Need at least 3 points for Moran's I.")
    if genomic_positions.shape != (N,):
        raise ValueError('spatial_coords and genomic_positions must have '
                         'matching length.')
    if cycle_bins is None:
        cycle_bins = N
    if cycle_bins <= 0:
        raise ValueError('cycle_bins must be > 0.')
    if exclude_circular_genomic_offsets_leq < 0:
        raise ValueError('exclude_circular_genomic_offsets_leq must be >= 0.')
    W = _inverse_distance_weights(
        spatial_coords,
        exclude_circular_genomic_offsets_leq=exclude_circular_genomic_offsets_leq)
    d = _circular_standardize(genomic_positions, cycle_bins)
    return _morans_i_core(W, d)


def _permute_once(W, base_positions, cycle_bins, observed_i, rng):
    """One permutation of the genomic-position vector; return the permuted I.

    Returns None when the permutation is degenerate (can't standardize).
    """
    perm = rng.permutation(base_positions)
    try:
        d = _circular_standardize(perm, cycle_bins)
    except ValueError:
        return None
    return _morans_i_core(W, d)


def permutation_test(spatial_coords, genomic_positions,
                     cycle_bins=None, n_permutations=999, seed=None,
                     exclude_circular_genomic_offsets_leq=0):
    """One-tailed permutation test for positive spatial autocorrelation.

    Matches the C++ reference: shuffle the genomic-position vector, recompute
    I for each shuffle, and report p = (#{I_perm >= I_obs} + 1) / (M + 1).

    The chain-offset exclusion (if requested) is applied once to the
    spatial weight matrix. Because the mask is defined by curve order
    (not by the permuted genomic labels), it is fixed across permutations.

    Returns a dict with fields: I_observed, P_value, n_permutations_valid,
    empirical_mean, empirical_std.
    """
    spatial_coords = np.asarray(spatial_coords, dtype=float)
    genomic_positions = np.asarray(genomic_positions, dtype=int)
    N = spatial_coords.shape[0]
    if cycle_bins is None:
        cycle_bins = N
    if exclude_circular_genomic_offsets_leq < 0:
        raise ValueError('exclude_circular_genomic_offsets_leq must be >= 0.')

    # Precompute once: the (optionally masked) spatial weight matrix is
    # constant across permutations; only the genomic-position vector
    # changes between permutations.
    W = _inverse_distance_weights(
        spatial_coords,
        exclude_circular_genomic_offsets_leq=exclude_circular_genomic_offsets_leq)
    d_obs = _circular_standardize(genomic_positions, cycle_bins)
    observed_i = _morans_i_core(W, d_obs)

    rng = np.random.default_rng(seed)
    perm_values = []
    count_ge = 0
    for _ in range(n_permutations):
        val = _permute_once(W, genomic_positions, cycle_bins, observed_i, rng)
        if val is None:
            continue
        perm_values.append(val)
        if val >= observed_i:
            count_ge += 1

    p_value = (count_ge + 1) / (n_permutations + 1)
    if perm_values:
        emp_mean = float(np.mean(perm_values))
        emp_std = float(np.std(perm_values, ddof=0))
    else:
        emp_mean = float('nan')
        emp_std = float('nan')

    return {
        'I_observed': observed_i,
        'P_value': p_value,
        'n_permutations_valid': len(perm_values),
        'n_permutations_requested': n_permutations,
        'empirical_mean': emp_mean,
        'empirical_std': emp_std,
        'exclude_circular_genomic_offsets_leq':
            int(exclude_circular_genomic_offsets_leq),
    }


def morans_i_with_pvalue(spatial_coords, genomic_positions,
                         cycle_bins=None, n_permutations=999, seed=None,
                         alpha=0.05,
                         exclude_circular_genomic_offsets_leq=0):
    """Convenience wrapper: returns I, P, is_significant (one-tailed test).

    See `permutation_test` / `_inverse_distance_weights` for the meaning
    of `exclude_circular_genomic_offsets_leq`.
    """
    res = permutation_test(
        spatial_coords, genomic_positions,
        cycle_bins=cycle_bins,
        n_permutations=n_permutations, seed=seed,
        exclude_circular_genomic_offsets_leq=
        exclude_circular_genomic_offsets_leq)
    res['is_significant'] = bool(res['P_value'] < alpha)
    res['alpha'] = alpha
    return res


# ---------------------------------------------------------------------------
# High-level helper: compute Moran's I for a dict of single-curve walks.
# ---------------------------------------------------------------------------
def compute_for_walks(walks, tp_indices_0based, *,
                      absolute_tp=False, cycle_bins=None,
                      n_permutations=999, seed=None, alpha=0.05,
                      exclude_circular_genomic_offsets_leq=0,
                      progress=None):
    """Compute per-walk Moran's I for a set of single-curve walks.

    Parameters
    ----------
    walks : dict[int, np.ndarray]
        WalkID -> (N, 3) array of nm coordinates, ordered by TP.
    tp_indices_0based : list[int]
        The 0-based TP indices in play for this analysis (length N).
    absolute_tp : bool
        If True, genomic_positions is the list of absolute 0-based TP
        indices (e.g. [2, 3, ..., 22] for TP002..TP022). If False, it is
        the range-relative [0, 1, ..., N-1]. Default False.
    cycle_bins : int or None
        Cycle size for the angle mapping theta_i = g_i * 2 pi / cycle_bins.
        If None, uses the walk's vertex count N.
    n_permutations : int
        Permutations for the one-tailed p-value.
    seed : int or None
        RNG seed (deterministic when set).
    alpha : float
        Significance threshold for `is_significant`.
    progress : optional callable
        Called with (wid, index, total) per iteration, for logging.

    Returns
    -------
    rows : list[dict]
        One dict per walk with fields used by the per-walk CSV.
    by_walk : dict[int, dict]
        WalkID -> {'I_value': float, 'P_value': float, 'is_significant': bool,
                   'n_points': int}. Compatible with the legacy Moran's I
        dict consumed by the report-card panels.
    """
    rows = []
    by_walk = {}
    tp_indices_0based = list(tp_indices_0based)
    N = len(tp_indices_0based)
    cb = cycle_bins if cycle_bins is not None else N
    if absolute_tp:
        genomic_positions = np.array(tp_indices_0based, dtype=int)
        mapping = 'absolute_0based_tp_indices'
    else:
        genomic_positions = np.arange(N, dtype=int)
        mapping = 'range_relative_0_to_N-1'

    total = len(walks)
    for idx, (wid, curve) in enumerate(walks.items()):
        if progress is not None:
            progress(wid, idx, total)
        if curve.shape[0] != N:
            raise ValueError(f'Walk {wid}: curve has {curve.shape[0]} points '
                             f'but expected {N}')
        res = morans_i_with_pvalue(
            curve, genomic_positions, cycle_bins=cb,
            n_permutations=n_permutations, seed=seed, alpha=alpha,
            exclude_circular_genomic_offsets_leq=
            exclude_circular_genomic_offsets_leq)
        rows.append({
            'WalkID': wid,
            'n_points': int(N),
            'tp_start_0based': int(tp_indices_0based[0]),
            'tp_end_0based': int(tp_indices_0based[-1]),
            'tp_start_1based': int(tp_indices_0based[0]) + 1,
            'tp_end_1based': int(tp_indices_0based[-1]) + 1,
            'I_value': float(res['I_observed']),
            'P_value': float(res['P_value']),
            'empirical_mean': float(res['empirical_mean']),
            'empirical_std': float(res['empirical_std']),
            'n_permutations_requested': int(res['n_permutations_requested']),
            'n_permutations_valid': int(res['n_permutations_valid']),
            'is_significant': bool(res['is_significant']),
            'alpha': float(res['alpha']),
            'cycle_bins': int(cb),
            'genomic_position_mapping': mapping,
            'exclude_circular_genomic_offsets_leq':
                int(exclude_circular_genomic_offsets_leq),
        })
        by_walk[wid] = {
            'I_value': float(res['I_observed']),
            'P_value': float(res['P_value']),
            'is_significant': bool(res['is_significant']),
            'n_points': int(N),
            'exclude_circular_genomic_offsets_leq':
                int(exclude_circular_genomic_offsets_leq),
        }
    return rows, by_walk


# ---------------------------------------------------------------------------
# Cache helpers: reuse a previously computed per-walk Moran's I CSV.
# ---------------------------------------------------------------------------
def load_morans_cache_csv(path):
    """Load a previously saved per-walk Moran's I CSV.

    Supports two schemas:
    1. The canonical schema written by `compute_for_walks` here
       (columns: WalkID, I_value, P_value, is_significant, n_points, ...).
    2. The legacy external schema (columns: Walk-ID, I-value, P-value)
       used by the reference pipeline's morans_i_results.csv.

    Returns
    -------
    by_walk : dict[int, dict]
        Same format as `compute_for_walks` returns (best-effort; legacy
        schema cannot always populate `is_significant` / `n_points`).
    """
    import csv
    out = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        fields = set(reader.fieldnames or [])
        if {'WalkID', 'I_value', 'P_value'}.issubset(fields):
            for row in reader:
                wid = int(float(row['WalkID']))
                out[wid] = {
                    'I_value': float(row['I_value']),
                    'P_value': float(row['P_value']),
                    'is_significant': (row.get('is_significant', '').lower()
                                       in ('true', '1', 'yes')),
                    'n_points': int(float(row.get('n_points', 0) or 0)),
                }
        elif {'Walk-ID', 'I-value', 'P-value'}.issubset(fields):
            for row in reader:
                wid = int(float(row['Walk-ID']))
                out[wid] = {
                    'I_value': float(row['I-value']),
                    'P_value': float(row['P-value']),
                    'is_significant': float(row['P-value']) < 0.05,
                    'n_points': int(float(row.get('Time-points', 0) or 0)),
                }
        else:
            raise KeyError(
                f'{path}: cannot parse Moran\'s I schema. Expected columns '
                f'"WalkID, I_value, P_value" (new) or "Walk-ID, I-value, '
                f'P-value" (legacy); got {sorted(fields)}.')
    return out


def write_morans_results_csv(rows, out_path):
    """Write the canonical per-walk Moran's I CSV."""
    import csv
    if not rows:
        with open(out_path, 'w', newline='') as f:
            f.write('')
        return
    fields = ['WalkID', 'n_points',
              'tp_start_1based', 'tp_end_1based',
              'tp_start_0based', 'tp_end_0based',
              'I_value', 'P_value',
              'empirical_mean', 'empirical_std',
              'n_permutations_requested', 'n_permutations_valid',
              'is_significant', 'alpha',
              'cycle_bins', 'genomic_position_mapping',
              'exclude_circular_genomic_offsets_leq']
    with open(out_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, '') for k in fields})
