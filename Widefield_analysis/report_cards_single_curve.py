#!/usr/bin/env python3
"""Comprehensive report-card analysis for single-curve walks.

Adapted from `Super_resolution_analysis/comprehensive_report_cards.py`
for this project's data model.

Key adaptation
--------------
In the reference pipeline each Walk-ID carried many sampled "DeWitt curves" per
walk, and most panels showed distributions over those samples. In this project,
after drift correction + full-coverage + movement filters, each `WalkID` is
already one ordered curve of `n_vertices = len(selected_tp_range)` points in
(x_nm, y_nm, z_nm). The script therefore:

  - removes DeWitt resampling (`sample_trajectory_from_walk`),
  - removes weighted-centroid-over-clouds logic (`compute_weighted_centroids`),
  - removes per-walk distributions of knot type / ACN / writhe / Rg,
  - removes Vassiliev V2 computations (not useful for a single curve with
    a small, fixed number of vertices),
  - removes the P(unknot) convergence panel (meaningless with one curve);
    no replacement text panel either,
  - keeps the control distributions (random polygon in hull, optional globule),
  - replaces per-walk histograms by a control histogram + a single vertical
    line for the walk's value,
  - replaces the "localization cloud" 3D panel by the ordered walk curve
    connected by line, TP-colored markers.

Vendoring
---------
Pure-curve metric functions (ACN / writhe / Rg / xy-xz-yz-CN / KendallIsogon /
knot-type + determinant / KL / Wasserstein / classify_name / custom time-point
colormap / control-generation helpers) are copied inline in this file.

All `n_vertices == 20` assertions and hard-coded 20-TP loops in the reference
have been generalized to `n_vertices = len(selected_tp_range)`. The custom
20-color time-point colormap is preserved and linearly resampled to the actual
point count when it differs from 20.

Entry point
-----------
See `main()` / the `--help` output. Input is the final filtered CSV written by
RodRunner's `drift_correct.py` step (after `qc_post_filter` has already run on it).
"""

from __future__ import annotations

import argparse
import csv
import multiprocessing
import os
import sys
import warnings
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Third-party scientific imports (vendored helpers below depend on these)
# ---------------------------------------------------------------------------
from scipy.stats import entropy, wasserstein_distance
from scipy.spatial import ConvexHull, QhullError
from scipy.linalg import orthogonal_procrustes

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.backends.backend_pdf import PdfPages
plt.style.use('default')
plt.rcParams['path.simplify'] = True
plt.rcParams['path.simplify_threshold'] = 1.0
plt.rcParams['agg.path.chunksize'] = 10000

# ---------------------------------------------------------------------------
# Optional: Numba for fast parallel ACN/writhe kernels. Falls back to a
# pure-python implementation when numba is unavailable.
# ---------------------------------------------------------------------------
try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    print('Warning: numba not installed; ACN/writhe will use the slow path.')

# ---------------------------------------------------------------------------
# Knot analysis: KnotoID + knoodle are required for knot type + determinant.
# They are third-party packages, not vendored.
# ---------------------------------------------------------------------------
try:
    from KnotoID import KnotoID
    import knoodle
    KNOTID_AVAILABLE = True
except ImportError:
    KNOTID_AVAILABLE = False
    print('Warning: KnotoID / knoodle not installed. Knot-type and determinant '
          'computation will be skipped.')

try:
    from wurlitzer import pipes
    WURLITZER_AVAILABLE = True
except ImportError:
    WURLITZER_AVAILABLE = False

_KLUT_CACHE = {}


def _get_klut_table():
    """Lazily initialize KnotLookupTable per-process (fork-safe)."""
    pid = os.getpid()
    if pid not in _KLUT_CACHE:
        try:
            _KLUT_CACHE[pid] = knoodle.KnotLookupTable() if KNOTID_AVAILABLE else None
        except Exception:
            _KLUT_CACHE[pid] = None
    return _KLUT_CACHE[pid]

# ---------------------------------------------------------------------------
# Vendored helper imports (sibling files in this repo)
# ---------------------------------------------------------------------------
try:
    from qc_post_filter import _load_ordered_walks as _qc_load_ordered_walks
    from qc_post_filter import _tp_index as _qc_tp_index
    QC_POST_FILTER_AVAILABLE = True
except Exception:
    QC_POST_FILTER_AVAILABLE = False

from morans_i import (
    compute_for_walks as morans_compute_for_walks,
    write_morans_results_csv as morans_write_csv,
    load_morans_cache_csv as morans_load_cache,
)

# Figure builders from the QC module, used to embed the parabola and
# distance-matrix pages directly into the main report PDF in addition to
# the standalone PNGs that `qc_post_filter.run_post_filter_analysis` writes.
try:
    from qc_post_filter import (
        build_mean_distance_matrices_figure,
        build_mean_parabola_xyz_figure,
        build_mean_parabola_3d_only_figure,
        draw_per_walk_parabola as _qcpf_draw_per_walk_parabola,
        draw_per_walk_distance_matrices as _qcpf_draw_per_walk_distance_matrices,
    )
except Exception:
    build_mean_distance_matrices_figure = None
    build_mean_parabola_xyz_figure = None
    build_mean_parabola_3d_only_figure = None
    _qcpf_draw_per_walk_parabola = None
    _qcpf_draw_per_walk_distance_matrices = None


# ===========================================================================
# VENDORED: custom 20-color time-point colormap.
# Generalized so any n_vertices maps to a colormap by linear resampling.
# ===========================================================================
CUSTOM_20_COLORS_HEX = [
    '#ED2024', '#F05023', '#F8991D', '#FFE500', '#CADB2B',
    '#94C83D', '#72BF44', '#6ABD45', '#71BF4C', '#6FC59B',
    '#6FCCDD', '#43ABE1', '#4068B2', '#3B54A5', '#4652A4',
    '#6B52A2', '#97509F', '#C44D9C', '#ED2590', '#EE1C51',
]
CUSTOM_TIMEPOINT_CMAP_20 = ListedColormap(CUSTOM_20_COLORS_HEX, name='timepoint_20')


def timepoint_colormap(n_vertices):
    """Return a colormap + per-vertex color list for n_vertices points.

    Preserves the reference `CUSTOM_TIMEPOINT_CMAP` when n_vertices == 20;
    otherwise linearly resamples the 20 reference colors to n_vertices so
    the same red-to-magenta rainbow is preserved regardless of TP count.
    """
    base = LinearSegmentedColormap.from_list(
        'timepoint_custom', CUSTOM_20_COLORS_HEX, N=max(256, n_vertices))
    colors = [base(i / max(n_vertices - 1, 1)) for i in range(n_vertices)]
    return base, colors


# ===========================================================================
# VENDORED METRIC FUNCTIONS (from comprehensive_report_cards.py)
# ===========================================================================

def classify_name(name):
    """Classify knot type names into 3 categories ('0_1', '3_1', 'Complex')."""
    if name == '0_1':
        return '0_1'
    if name in ('3_1', '3_1m'):
        return '3_1'
    return 'Complex'


if NUMBA_AVAILABLE:

    @jit(nopython=True, cache=True)
    def _create_orthonormal_basis_numba(direction):
        if abs(direction[0]) < 0.9:
            v1 = np.array([1.0, 0.0, 0.0])
        else:
            v1 = np.array([0.0, 1.0, 0.0])
        dot = v1[0]*direction[0] + v1[1]*direction[1] + v1[2]*direction[2]
        u1 = v1 - dot * direction
        norm = np.sqrt(u1[0]**2 + u1[1]**2 + u1[2]**2)
        u1 = u1 / norm
        u2 = np.array([
            direction[1]*u1[2] - direction[2]*u1[1],
            direction[2]*u1[0] - direction[0]*u1[2],
            direction[0]*u1[1] - direction[1]*u1[0],
        ])
        return u1, u2

    @jit(nopython=True, cache=True)
    def _project_polygon_numba(polygon_3d, direction):
        u1, u2 = _create_orthonormal_basis_numba(direction)
        n = len(polygon_3d)
        projected = np.empty((n, 2))
        for i in range(n):
            projected[i, 0] = (polygon_3d[i, 0]*u1[0] + polygon_3d[i, 1]*u1[1]
                               + polygon_3d[i, 2]*u1[2])
            projected[i, 1] = (polygon_3d[i, 0]*u2[0] + polygon_3d[i, 1]*u2[1]
                               + polygon_3d[i, 2]*u2[2])
        return projected

    @jit(nopython=True, cache=True)
    def _orientation_numba(px, py, qx, qy, rx, ry):
        return (qx - px) * (ry - py) - (qy - py) * (rx - px)

    @jit(nopython=True, cache=True)
    def _segments_intersect_numba(p1x, p1y, p2x, p2y, q1x, q1y, q2x, q2y):
        o1 = _orientation_numba(p1x, p1y, p2x, p2y, q1x, q1y)
        o2 = _orientation_numba(p1x, p1y, p2x, p2y, q2x, q2y)
        o3 = _orientation_numba(q1x, q1y, q2x, q2y, p1x, p1y)
        o4 = _orientation_numba(q1x, q1y, q2x, q2y, p2x, p2y)
        if o1 * o2 < 0 and o3 * o4 < 0:
            return True
        return False

    @jit(nopython=True, cache=True)
    def _compute_crossing_sign_numba(polygon_3d, i, j, n_vertices):
        i_next = (i + 1) % n_vertices
        j_next = (j + 1) % n_vertices
        s_i = polygon_3d[i_next] - polygon_3d[i]
        s_j = polygon_3d[j_next] - polygon_3d[j]
        r_ij = polygon_3d[j] - polygon_3d[i]
        cross_x = s_j[1]*s_i[2] - s_j[2]*s_i[1]
        cross_y = s_j[2]*s_i[0] - s_j[0]*s_i[2]
        cross_z = s_j[0]*s_i[1] - s_j[1]*s_i[0]
        triple_product = cross_x*r_ij[0] + cross_y*r_ij[1] + cross_z*r_ij[2]
        if triple_product > 0:
            return 1
        if triple_product < 0:
            return -1
        return 0

    @jit(nopython=True, cache=True)
    def _count_crossings_with_signs_numba(polygon_3d, polygon_2d):
        n_vertices = len(polygon_2d)
        unsigned_count = 0
        signed_sum = 0
        for i in range(n_vertices):
            i_next = (i + 1) % n_vertices
            for j in range(i + 2, n_vertices):
                if j == i_next or (i == 0 and j == n_vertices - 1):
                    continue
                j_next = (j + 1) % n_vertices
                if _segments_intersect_numba(
                        polygon_2d[i, 0], polygon_2d[i, 1],
                        polygon_2d[i_next, 0], polygon_2d[i_next, 1],
                        polygon_2d[j, 0], polygon_2d[j, 1],
                        polygon_2d[j_next, 0], polygon_2d[j_next, 1]):
                    unsigned_count += 1
                    signed_sum += _compute_crossing_sign_numba(
                        polygon_3d, i, j, n_vertices)
        return unsigned_count, signed_sum

    @jit(nopython=True, cache=True)
    def _is_valid_projection_numba(polygon_2d, epsilon=1e-10):
        n = len(polygon_2d)
        for i in range(n):
            i_next = (i + 1) % n
            dx = polygon_2d[i_next, 0] - polygon_2d[i, 0]
            dy = polygon_2d[i_next, 1] - polygon_2d[i, 1]
            edge_len = np.sqrt(dx*dx + dy*dy)
            if edge_len < epsilon:
                return False
        return True

    @jit(nopython=True, parallel=True, cache=True)
    def _acn_writhe_parallel(polygon_3d, directions):
        n_projections = len(directions)
        crossings = np.empty(n_projections)
        writhes = np.empty(n_projections)
        valid = np.empty(n_projections, dtype=np.bool_)
        for p in prange(n_projections):
            projected = _project_polygon_numba(polygon_3d, directions[p])
            if _is_valid_projection_numba(projected):
                uc, ss = _count_crossings_with_signs_numba(polygon_3d, projected)
                crossings[p] = uc
                writhes[p] = ss
                valid[p] = True
            else:
                crossings[p] = 0
                writhes[p] = 0
                valid[p] = False
        return crossings, writhes, valid


def generate_random_projection_directions(n_projections):
    theta = np.random.uniform(0, 2 * np.pi, n_projections)
    phi = np.arccos(1 - 2 * np.random.uniform(0, 1, n_projections))
    return np.stack([
        np.sin(phi) * np.cos(theta),
        np.sin(phi) * np.sin(theta),
        np.cos(phi),
    ], axis=1)


def _create_orthonormal_basis(direction):
    if abs(direction[0]) < 0.9:
        v1 = np.array([1.0, 0.0, 0.0])
    else:
        v1 = np.array([0.0, 1.0, 0.0])
    u1 = v1 - np.dot(v1, direction) * direction
    u1 = u1 / np.linalg.norm(u1)
    u2 = np.cross(direction, u1)
    return u1, u2


def _project_polygon_to_plane(polygon_3d, direction):
    u1, u2 = _create_orthonormal_basis(direction)
    projected = np.zeros((len(polygon_3d), 2))
    for i, vertex in enumerate(polygon_3d):
        projected[i, 0] = np.dot(vertex, u1)
        projected[i, 1] = np.dot(vertex, u2)
    return projected


def _orientation(p, q, r):
    return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])


def _segments_intersect(p1, p2, q1, q2):
    o1 = _orientation(p1, p2, q1)
    o2 = _orientation(p1, p2, q2)
    o3 = _orientation(q1, q2, p1)
    o4 = _orientation(q1, q2, p2)
    if o1 * o2 < 0 and o3 * o4 < 0:
        return True
    return False


def _count_crossings_2d(polygon_2d):
    n_vertices = len(polygon_2d)
    crossings = 0
    for i in range(n_vertices):
        i_next = (i + 1) % n_vertices
        for j in range(i + 2, n_vertices):
            if j == i_next or (i == 0 and j == n_vertices - 1):
                continue
            j_next = (j + 1) % n_vertices
            if _segments_intersect(polygon_2d[i], polygon_2d[i_next],
                                   polygon_2d[j], polygon_2d[j_next]):
                crossings += 1
    return crossings


def _compute_crossing_sign(polygon_3d, i, j):
    n_vertices = len(polygon_3d)
    i_next = (i + 1) % n_vertices
    j_next = (j + 1) % n_vertices
    s_i = polygon_3d[i_next] - polygon_3d[i]
    s_j = polygon_3d[j_next] - polygon_3d[j]
    r_ij = polygon_3d[j] - polygon_3d[i]
    triple_product = np.dot(np.cross(s_j, s_i), r_ij)
    if triple_product > 0:
        return 1
    if triple_product < 0:
        return -1
    return 0


def _count_crossings_with_signs(polygon_3d, polygon_2d):
    n_vertices = len(polygon_2d)
    unsigned_count = 0
    signed_sum = 0
    for i in range(n_vertices):
        i_next = (i + 1) % n_vertices
        for j in range(i + 2, n_vertices):
            if j == i_next or (i == 0 and j == n_vertices - 1):
                continue
            j_next = (j + 1) % n_vertices
            if _segments_intersect(polygon_2d[i], polygon_2d[i_next],
                                   polygon_2d[j], polygon_2d[j_next]):
                unsigned_count += 1
                signed_sum += _compute_crossing_sign(polygon_3d, i, j)
    return unsigned_count, signed_sum


def _is_valid_projection(polygon_2d, epsilon=1e-10):
    edges = polygon_2d[1:] - polygon_2d[:-1]
    edges = np.vstack([edges, polygon_2d[0:1] - polygon_2d[-1:]])
    edge_lengths = np.linalg.norm(edges, axis=1)
    return not bool(np.any(edge_lengths < epsilon))


def average_crossing_number(polygon_3d, n_projections=1000):
    """Return (acn, writhe, rms_writhe) via Monte Carlo projection."""
    directions = generate_random_projection_directions(n_projections)
    if NUMBA_AVAILABLE:
        polygon_3d = np.ascontiguousarray(polygon_3d, dtype=np.float64)
        directions = np.ascontiguousarray(directions, dtype=np.float64)
        crossings, writhes, valid = _acn_writhe_parallel(polygon_3d, directions)
        vc = crossings[valid]
        vw = writhes[valid]
    else:
        vc, vw = [], []
        for direction in directions:
            projected = _project_polygon_to_plane(polygon_3d, direction)
            if _is_valid_projection(projected):
                uc, ss = _count_crossings_with_signs(polygon_3d, projected)
                vc.append(uc)
                vw.append(ss)
        vc = np.array(vc)
        vw = np.array(vw)
    if vc.size < n_projections * 0.9:
        warnings.warn(
            f'Only {vc.size} valid projections out of {n_projections}')
    acn = float(np.mean(vc)) if vc.size else float('nan')
    writhe = float(np.mean(vw)) if vw.size else float('nan')
    rms_writhe = float(np.sqrt(np.mean(vw ** 2))) if vw.size else float('nan')
    return acn, writhe, rms_writhe


def radius_of_gyration(vertices):
    com = np.mean(vertices, axis=0)
    squared = np.sum((vertices - com) ** 2, axis=1)
    return float(np.sqrt(np.mean(squared)))


def xy_crossing_number(polygon_3d):
    polygon_2d = polygon_3d[:, :2]
    if not _is_valid_projection(polygon_2d):
        return float('nan')
    return _count_crossings_2d(polygon_2d)


def xz_crossing_number(polygon_3d):
    polygon_2d = polygon_3d[:, [0, 2]]
    if not _is_valid_projection(polygon_2d):
        return float('nan')
    return _count_crossings_2d(polygon_2d)


def yz_crossing_number(polygon_3d):
    polygon_2d = polygon_3d[:, [1, 2]]
    if not _is_valid_projection(polygon_2d):
        return float('nan')
    return _count_crossings_2d(polygon_2d)


# ===========================================================================
# VENDORED: Kendall shape-space averager.
# Generalized: the reference asserted n_vertices == 20; that check is dropped,
# the class now works on any n_vertices consistently across the input isogons.
# ===========================================================================
class KendallIsogonAverager:
    """Compute Frechet mean of a family of n-gons in Kendall shape space.

    Accepts any n_vertices; the reference version hardcoded 20. The dimension
    (2 or 3) is auto-detected from input shapes.
    """

    def __init__(self, dim=3):
        self.dim = dim

    def center_isogon(self, X):
        return X - np.mean(X, axis=0)

    def to_preshape(self, X):
        Xc = self.center_isogon(X)
        norm = np.linalg.norm(Xc, 'fro')
        if norm > 0:
            return Xc / norm
        return Xc

    def align_to_reference(self, X, reference):
        Z = self.to_preshape(X)
        Zref = self.to_preshape(reference)
        R, _ = orthogonal_procrustes(Z, Zref)
        return Z @ R, R

    def compute_average(self, isogons, max_iter=100, tol=1e-6,
                        return_scale=False):
        if isinstance(isogons, list):
            isogons = np.array(isogons)
        if isogons.ndim == 2:
            isogons = isogons[np.newaxis, :]
        n_isogons, n_vertices, dim = isogons.shape
        assert dim in (2, 3), f'Expected 2D or 3D coordinates, got {dim}D'
        preshapes = np.array([self.to_preshape(X) for X in isogons])
        mean_shape = preshapes[0].copy()
        iteration = 0
        for iteration in range(max_iter):
            old = mean_shape.copy()
            aligned_sum = np.zeros_like(mean_shape)
            for i in range(n_isogons):
                aligned, _ = self.align_to_reference(preshapes[i], mean_shape)
                aligned_sum += aligned
            mean_shape = self.to_preshape(aligned_sum)
            change = np.linalg.norm(mean_shape - old, 'fro')
            if change < tol:
                break
        if return_scale:
            scales = [np.linalg.norm(self.center_isogon(X), 'fro')
                      for X in isogons]
            mean_shape = mean_shape * np.mean(scales)
        converged = iteration < max_iter - 1
        return mean_shape, converged, iteration + 1

    def procrustes_distance(self, X1, X2):
        Z1 = self.to_preshape(X1)
        Z2 = self.to_preshape(X2)
        singular = np.linalg.svd(Z1.T @ Z2, compute_uv=False)
        return float(np.arccos(np.clip(np.sum(singular), -1, 1)))


# ===========================================================================
# VENDORED: get_all_component_codes + calculate_knot_properties (KnotoID path)
# Also vendored verbatim-ish from the reference.
# ===========================================================================
def get_all_component_codes(analyzer):
    """Collect PD / Gauss codes from all components of a knot analyzer."""
    all_pd, all_gauss = [], []
    for comp in analyzer.prime_components:
        all_pd.append(comp.pd_code)
        all_gauss.append(comp.gauss_code)
    if analyzer.crossing_count > 0:
        all_pd.append(analyzer.pd_code)
        all_gauss.append(analyzer.gauss_code)
    elif analyzer.is_prime and not analyzer.is_unknot():
        all_pd.append(analyzer.pd_code)
        all_gauss.append(analyzer.gauss_code)
    return all_pd, all_gauss


def _knot_name_from_determinant(det):
    """Derive knot classification from Alexander polynomial determinant."""
    if det == 1:
        return '0_1'
    if det == 3:
        return '3_1'
    return 'Complex'


def calculate_knot_properties(curve, determinant_only=False):
    """Return (knot_name, determinant, knot_name_knoodle) for an (n_vertices, 3) curve.

    knot_name: from KnotoID (projection-based) or determinant-derived.
    determinant: |Alexander(-1)|.
    knot_name_knoodle: from KnotLookupTable (MacLeod code table lookup),
        or '' if the table is unavailable or the knot is not in the table.
    """
    if not KNOTID_AVAILABLE:
        return 'Unavailable', float('nan'), ''
    try:
        coords = [float(c) for pt in curve for c in pt]
        analyzer = knoodle.KnotAnalyzer(coords, simplify=True, simplify_level=5)
        get_all_component_codes(analyzer)
        alexander_value = analyzer.alexander([-1.0])[0].to_double()
        determinant = abs(alexander_value)
        rounded = round(determinant)
        if abs(determinant - rounded) < 1e-10:
            determinant = float(rounded)
        klut_name = ''
        klut = _get_klut_table()
        if klut is not None:
            try:
                result = klut.lookup(analyzer)
                if result is not None:
                    klut_name = result
            except Exception:
                pass
        if determinant_only:
            return _knot_name_from_determinant(determinant), determinant, klut_name
        if WURLITZER_AVAILABLE:
            with pipes():
                k = KnotoID(projections_list='internal',
                            closure_method='direct')
                k.analyze_curve(curve)
        else:
            k = KnotoID(projections_list='internal',
                        closure_method='direct')
            k.analyze_curve(curve)
        rez = k.curve_analysis
        knot_name = rez.loc[0, 'Name'] if len(rez) > 0 else 'Unknown'
        return knot_name, determinant, klut_name
    except Exception as exc:
        print(f'Error calculating knot properties: {exc}')
        return 'Error', float('nan'), ''


# ===========================================================================
# VENDORED: KL divergence / Wasserstein on knot-type distributions.
# ===========================================================================
def calculate_kl_divergence(p_dist, q_dist,
                            categories=('0_1', '3_1', 'Complex')):
    """KL divergence between two lists of (knot_name, determinant) tuples."""
    p_classified = [classify_name(n) for n, _ in p_dist]
    q_classified = [classify_name(n) for n, _ in q_dist]
    p_counts = Counter(p_classified)
    q_counts = Counter(q_classified)
    p_probs = np.array([p_counts.get(c, 0) for c in categories], dtype=float)
    q_probs = np.array([q_counts.get(c, 0) for c in categories], dtype=float)
    p_probs /= (p_probs.sum() + 1e-10)
    q_probs /= (q_probs.sum() + 1e-10)
    eps = 1e-10
    return float(entropy(p_probs + eps, q_probs + eps)), p_probs, q_probs


def calculate_emd(p_dist, q_dist):
    """Wasserstein distance using |det_i - det_j| as the ground metric."""
    try:
        import ot
        all_knots = sorted(set([n for n, _ in p_dist]
                               + [n for n, _ in q_dist]))
        knot_to_det = {}
        for n, d in list(p_dist) + list(q_dist):
            knot_to_det.setdefault(n, d)
        p_counts = Counter([n for n, _ in p_dist])
        q_counts = Counter([n for n, _ in q_dist])
        p_probs = np.array([p_counts.get(k, 0) / max(len(p_dist), 1)
                            for k in all_knots])
        q_probs = np.array([q_counts.get(k, 0) / max(len(q_dist), 1)
                            for k in all_knots])
        cost = np.array([[abs(knot_to_det[a] - knot_to_det[b])
                          for b in all_knots] for a in all_knots])
        return float(ot.emd2(p_probs, q_probs, cost))
    except ImportError:
        all_dets = sorted(set([d for _, d in p_dist] + [d for _, d in q_dist]))
        p_dc = Counter([d for _, d in p_dist])
        q_dc = Counter([d for _, d in q_dist])
        p_w = np.array([p_dc.get(d, 0) / max(len(p_dist), 1) for d in all_dets])
        q_w = np.array([q_dc.get(d, 0) / max(len(q_dist), 1) for d in all_dets])
        return float(wasserstein_distance(all_dets, all_dets, p_w, q_w))


# ===========================================================================
# VENDORED: Moran's I CSV loader.
# ===========================================================================
def load_morans_i_data(filepath):
    """Return {walk_id: {'I_value': float, 'P_value': float}} or {} on fail."""
    if not filepath or not os.path.exists(filepath):
        if filepath:
            print(f"Warning: Moran's I file not found: {filepath}")
        return {}
    try:
        df = pd.read_csv(filepath)
        if ('Walk-ID' in df.columns and 'I-value' in df.columns
                and 'P-value' in df.columns):
            out = {row['Walk-ID']: {'I_value': row['I-value'],
                                    'P_value': row['P-value']}
                   for _, row in df.iterrows()}
            print(f"Loaded Moran's I data for {len(out)} walks")
            return out
        cols = df.columns.tolist()
        if len(cols) >= 5:
            out = {row[cols[1]]: {'I_value': row[cols[3]],
                                  'P_value': row[cols[4]]}
                   for _, row in df.iterrows()}
            print(f"Loaded Moran's I data for {len(out)} walks (index fallback)")
            return out
        print(f"Error: expected >=5 columns, got {len(cols)}")
        return {}
    except Exception as exc:
        print(f"Error loading Moran's I data: {exc}")
        return {}


# ===========================================================================
# VENDORED: control generation (random polygon in hull, globule).
# All hardcoded n_vertices=20 defaults generalized to an explicit argument.
# ===========================================================================
def _is_inside_convex_hull(points, A, b):
    return np.all(np.dot(points, A.T) + b <= 1e-12, axis=1)


def _generate_random_points_in_convex_hull(convex_hull, num_points):
    A = convex_hull.equations[:, :-1]
    b = convex_hull.equations[:, -1]
    min_coords = np.min(convex_hull.points, axis=0)
    max_coords = np.max(convex_hull.points, axis=0)
    out = []
    generated = 0
    batch_size = max(num_points * 2, 1000)
    while generated < num_points:
        batch = np.random.uniform(min_coords, max_coords, size=(batch_size, 3))
        inside = _is_inside_convex_hull(batch, A, b)
        inside_points = batch[inside]
        remaining = num_points - generated
        if inside_points.shape[0] >= remaining:
            out.append(inside_points[:remaining])
            generated += remaining
        else:
            out.append(inside_points)
            generated += inside_points.shape[0]
    return np.vstack(out)


def compute_average_convex_hull(polygons):
    """Convex hull encompassing the typical spatial extent of the inputs."""
    all_points = []
    for poly in polygons:
        all_points.extend(poly)
    all_points = np.array(all_points)
    low = np.percentile(all_points, 5, axis=0)
    high = np.percentile(all_points, 95, axis=0)
    box = []
    for x in [low[0], high[0]]:
        for y in [low[1], high[1]]:
            for z in [low[2], high[2]]:
                box.append([x, y, z])
    sample_size = min(1000, len(all_points))
    idx = np.random.choice(len(all_points), sample_size, replace=False)
    box.extend(all_points[idx])
    try:
        return ConvexHull(np.array(box))
    except QhullError as exc:
        print(f'Warning: could not compute convex hull: {exc}')
        return None


def generate_random_polygon_in_hull(convex_hull, n_vertices):
    """Random `n_vertices`-gon with vertices inside the given convex hull."""
    if convex_hull is None:
        return np.random.uniform(-5e-7, 5e-7, (n_vertices, 3))
    return _generate_random_points_in_convex_hull(convex_hull, n_vertices)


def generate_random_polygons(n_polygons, n_vertices, convex_hull=None,
                             verbose=True):
    """Generate random polygons and compute (knot, determinant) for each.

    n_vertices is required. Reference hardcoded 20.
    """
    if verbose:
        if convex_hull is not None:
            print(f'Generating {n_polygons} random {n_vertices}-gons '
                  f'within convex hull...')
        else:
            print(f'Generating {n_polygons} random {n_vertices}-gons '
                  f'in a box...')
    results = []
    it = range(n_polygons)
    if verbose:
        it = tqdm(it, desc='Random polygons')
    for _ in it:
        if convex_hull is not None:
            poly = generate_random_polygon_in_hull(convex_hull, n_vertices)
        else:
            poly = np.random.uniform(-5e-7, 5e-7, (n_vertices, 3))
        name, det, _ = calculate_knot_properties(poly)
        results.append((name, det))
    return results


def load_unknotted_globule_file(filepath):
    """Load an N x 3 TSV file of globule coordinates."""
    return np.loadtxt(filepath, delimiter='\t')


def segment_and_sample_polygon_arc_length(polygon_nk, n_segments):
    """Partition `polygon_nk` into n_segments of equal arc length; one sample each."""
    n_points = len(polygon_nk)
    d = np.sqrt(np.sum(np.diff(polygon_nk, axis=0) ** 2, axis=1))
    cum = np.zeros(n_points)
    cum[1:] = np.cumsum(d)
    total = cum[-1]
    seg_len = total / n_segments
    out = []
    for i in range(n_segments):
        start = i * seg_len
        end = (i + 1) * seg_len
        if i == n_segments - 1:
            mask = (cum >= start) & (cum <= total)
        else:
            mask = (cum >= start) & (cum < end)
        idx = np.where(mask)[0]
        if idx.size:
            out.append(polygon_nk[np.random.choice(idx)])
        else:
            nearest = np.argmin(np.abs(cum - (start + seg_len / 2)))
            out.append(polygon_nk[nearest])
    return np.array(out)


def segment_and_sample_polygon_point_count(polygon_nk, n_segments):
    n_points = len(polygon_nk)
    seg = n_points // n_segments
    out = []
    for i in range(n_segments):
        s = i * seg
        e = s + seg if i < n_segments - 1 else n_points
        chunk = polygon_nk[s:e]
        out.append(chunk[np.random.randint(0, len(chunk))])
    return np.array(out)


def segment_and_sample_polygon(polygon_nk, n_segments, method='arc_length'):
    if method == 'arc_length':
        return segment_and_sample_polygon_arc_length(polygon_nk, n_segments)
    if method == 'point_count':
        return segment_and_sample_polygon_point_count(polygon_nk, n_segments)
    raise ValueError(f'Unknown segmentation method: {method}')


def generate_globule_polygons(samples_per_file, globule_dir, n_segments,
                              segment_method='arc_length'):
    """Generate n_segments-gons from globule TSVs; returns list of (name, det)."""
    import glob
    if globule_dir is None:
        raise ValueError('globule_dir must be specified')
    globule_dir = os.path.expanduser(globule_dir)
    tsvs = sorted(glob.glob(os.path.join(globule_dir, '*.tsv')))
    kind = 'knotted' if 'knotted' in globule_dir else 'unknotted'
    print(f'Found {len(tsvs)} {kind} globule files; sampling '
          f'{n_segments}-gons from each (segment method: {segment_method})')
    results = []
    for fp in tqdm(tsvs, desc=f'{kind} globule'):
        poly = load_unknotted_globule_file(fp)
        for _ in range(samples_per_file):
            seg = segment_and_sample_polygon(poly, n_segments, segment_method)
            name, det, _ = calculate_knot_properties(seg)
            results.append((name, det))
    return results


# ===========================================================================
# Globule-control cache: the unknotted/knotted globule knot-type
# distributions depend only on (globule dir, n_vertices, samples_per_globule,
# segment method) -- NOT on the walk set -- so they can be computed once and
# reused across datasets/ranges that share those settings.
# ===========================================================================
def load_globule_cache_csv(path):
    """Return (dists, meta) from a globule cache CSV, or (None, None).

    dists: {globule_type: [(knot_name, determinant), ...]}
    meta:  {n_vertices, samples_per_globule, segment_by, unknotted_dir,
            knotted_dir} taken from the file (constant across rows).
    """
    if not path or not os.path.exists(path):
        return None, None
    dists = defaultdict(list)
    meta = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw = (row.get('determinant') or '').strip()
            try:
                det = float(raw)
            except ValueError:
                det = float('nan')
            dists[row['globule_type']].append((row['knot_name'], det))
            if not meta:
                meta = {
                    'n_vertices': int(row['n_vertices']),
                    'samples_per_globule': int(row['samples_per_globule']),
                    'segment_by': row['segment_by'],
                    'unknotted_dir': row.get('unknotted_dir', ''),
                    'knotted_dir': row.get('knotted_dir', ''),
                }
    return dict(dists), meta


def write_globule_cache_csv(path, dists, n_vertices, samples_per_globule,
                            segment_by, unknotted_dir, knotted_dir):
    """Write globule (knot_name, determinant) lists + settings to a cache CSV."""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['globule_type', 'knot_name', 'determinant', 'n_vertices',
                    'samples_per_globule', 'segment_by', 'unknotted_dir',
                    'knotted_dir'])
        for gtype, lst in dists.items():
            for name, det in lst:
                w.writerow([gtype, name, det, n_vertices, samples_per_globule,
                            segment_by, unknotted_dir, knotted_dir])


# ===========================================================================
# NEW: single-curve data loader.
# One ordered (n_vertices, 3) curve per WalkID, built from the final filtered
# CSV over the user-supplied inclusive 1-based TP range.
# Reuses qc_post_filter._load_ordered_walks when available, otherwise
# re-implements the same logic inline.
# ===========================================================================
TP_RE_LOCAL = None  # filled lazily when qc_post_filter is not importable


def _fallback_tp_index(tp_name):
    import re
    global TP_RE_LOCAL
    if TP_RE_LOCAL is None:
        TP_RE_LOCAL = re.compile(r'^TP(\d+)_CYC(\d+)$')
    m = TP_RE_LOCAL.match(tp_name)
    if m is None:
        raise ValueError(f'Unrecognized TP name: {tp_name!r}')
    return int(m.group(1))


def _fallback_load_ordered_walks(filtered_csv, tp_start_0, tp_end_0,
                                 skip_incomplete=False):
    """Mirror of qc_post_filter._load_ordered_walks, inlined for standalone use.

    Honors `skip_incomplete` the same way as the canonical loader: empty
    or non-finite x_nm / y_nm / z_nm cells either drop the row (when
    True) or raise a clear ValueError (when False).
    """
    required_tp0 = list(range(tp_start_0, tp_end_0 + 1))
    required_set = set(required_tp0)
    per_wid = defaultdict(lambda: defaultdict(list))
    tp_names_seen = {}
    skipped_empty_coord_rows = 0

    def _row_xyz(row, tp_name, wid):
        out = []
        for k in ('x_nm', 'y_nm', 'z_nm'):
            raw = (row.get(k) or '').strip()
            if raw == '':
                if skip_incomplete:
                    return None
                raise ValueError(
                    f'{filtered_csv}: empty {k} for WalkID={wid} '
                    f'tp={tp_name!r}. Pass --skip_incomplete to drop '
                    'walks with incomplete coordinate rows.')
            try:
                v = float(raw)
            except ValueError as exc:
                raise ValueError(
                    f'{filtered_csv}: non-numeric {k}={raw!r} for '
                    f'WalkID={wid} tp={tp_name!r}: {exc}') from exc
            if not np.isfinite(v):
                if skip_incomplete:
                    return None
                raise ValueError(
                    f'{filtered_csv}: non-finite {k}={raw!r} for '
                    f'WalkID={wid} tp={tp_name!r}.')
            out.append(v)
        return out

    with open(filtered_csv) as f:
        reader = csv.DictReader(f)
        required_cols = {'WalkID', 'tp', 'x_nm', 'y_nm', 'z_nm'}
        missing = required_cols - set(reader.fieldnames or [])
        if missing:
            raise KeyError(f'{filtered_csv}: missing columns {sorted(missing)}')
        for row in reader:
            tp0 = _fallback_tp_index(row['tp'])
            if tp0 not in required_set:
                continue
            tp_names_seen[tp0] = row['tp']
            wid = int(row['WalkID'])
            xyz = _row_xyz(row, row['tp'], wid)
            if xyz is None:
                skipped_empty_coord_rows += 1
                continue
            per_wid[wid][tp0].append(np.array(xyz))
    missing_in_csv = [t for t in required_tp0 if t not in tp_names_seen]
    if missing_in_csv:
        raise ValueError(
            'Requested analysis TP range includes TP(s) not present in '
            f'{filtered_csv}: 0-based {missing_in_csv} '
            f'(1-based: {[t + 1 for t in missing_in_csv]}).')
    present_tp_names = [tp_names_seen[t] for t in required_tp0]
    walks_ok = {}
    skip_miss = 0
    skip_dupe = 0
    for wid, tp_map in per_wid.items():
        if not all(t in tp_map for t in required_tp0):
            skip_miss += 1
            continue
        if any(len(tp_map[t]) != 1 for t in required_tp0):
            skip_dupe += 1
            continue
        walks_ok[wid] = np.vstack([tp_map[t][0] for t in required_tp0])
    return (walks_ok, present_tp_names, required_tp0,
            skip_miss, skip_dupe, len(per_wid),
            skipped_empty_coord_rows)


def load_single_curve_walks(filtered_csv, tp_start_1based, tp_end_1based,
                            skip_incomplete=False):
    """Return one ordered curve per WalkID from the final filtered CSV.

    Walks with missing TPs or duplicate (WalkID, TP) rows are skipped and
    reported. Fails if any requested TP is absent from the CSV.
    When `skip_incomplete` is True, rows with empty/non-finite
    x_nm / y_nm / z_nm are treated as missing for that walk (the walk
    then falls into the missing-TP skip path). When False (default), the
    loader raises a clear ValueError on the first such row.
    """
    if not isinstance(tp_start_1based, int) or not isinstance(tp_end_1based, int):
        raise TypeError('tp_start/end must be ints')
    if tp_start_1based < 1:
        raise ValueError(f'tp_start_1based must be >= 1 (got {tp_start_1based})')
    if tp_end_1based < tp_start_1based:
        raise ValueError(
            f'tp_end_1based ({tp_end_1based}) must be >= tp_start_1based '
            f'({tp_start_1based})')
    tp_start_0 = tp_start_1based - 1
    tp_end_0 = tp_end_1based - 1
    if QC_POST_FILTER_AVAILABLE:
        (walks_ok, present_names, present_indices,
         skip_miss, skip_dupe, n_input,
         skip_empty) = _qc_load_ordered_walks(
            filtered_csv, tp_start_0, tp_end_0,
            skip_incomplete=skip_incomplete)
    else:
        (walks_ok, present_names, present_indices,
         skip_miss, skip_dupe, n_input,
         skip_empty) = _fallback_load_ordered_walks(
            filtered_csv, tp_start_0, tp_end_0,
            skip_incomplete=skip_incomplete)
    return {
        'walks': walks_ok,
        'tp_names': present_names,
        'tp_indices_0based': present_indices,
        'n_vertices': len(present_indices),
        'skipped_missing_tp': skip_miss,
        'skipped_duplicate_tp': skip_dupe,
        'skipped_empty_coord_rows': skip_empty,
        'n_walks_input_in_range': n_input,
        'tp_start_1based': tp_start_1based,
        'tp_end_1based': tp_end_1based,
    }


# ===========================================================================
# NEW: per-walk metric computation (single curve, no sampling loop).
# Parallelizable via --parallel/--workers. The top-level worker functions
# below are picklable by ProcessPoolExecutor; they take a small dict of
# flags instead of the full argparse Namespace to keep the payload tight.
# ===========================================================================
def _flags_from_args(args):
    """Return a minimal, picklable dict of metric flags for worker processes."""
    return {
        'acn': bool(args.acn),
        'n_acn_projections': int(args.n_acn_projections),
        'rog': bool(args.rog),
        'xyCN': bool(args.xyCN),
        'xzCN': bool(args.xzCN),
        'yzCN': bool(args.yzCN),
        'no_knot_type': bool(getattr(args, 'no_knot_type', False)),
    }


def _walk_metric_worker(payload):
    wid, curve, flags = payload
    out = {'WalkID': wid, 'n_vertices': int(curve.shape[0])}
    name, det, klut_name = calculate_knot_properties(
        curve, determinant_only=flags.get('no_knot_type', False))
    out['knot_type'] = name
    out['determinant'] = det
    out['knot_type_knoodle'] = klut_name
    out['classified'] = classify_name(name)
    out['is_unknot'] = int(name == '0_1')
    if flags['acn']:
        acn, writhe, rms = average_crossing_number(curve, flags['n_acn_projections'])
        out['acn'] = acn
        out['writhe'] = writhe
        out['rms_writhe'] = rms
    if flags['rog']:
        out['rog_nm'] = radius_of_gyration(curve)
    if flags['xyCN']:
        out['xyCN'] = xy_crossing_number(curve)
    if flags['xzCN']:
        out['xzCN'] = xz_crossing_number(curve)
    if flags['yzCN']:
        out['yzCN'] = yz_crossing_number(curve)
    return wid, out


def _control_metric_worker(payload):
    curve, flags = payload
    out = {}
    if flags['acn']:
        acn, wri, rms = average_crossing_number(curve, flags['n_acn_projections'])
        out['acn'] = acn
        out['writhe'] = wri
        out['rms_writhe'] = rms
    if flags['rog']:
        out['rog_nm'] = radius_of_gyration(curve)
    if flags['xyCN']:
        out['xyCN'] = xy_crossing_number(curve)
    if flags['xzCN']:
        out['xzCN'] = xz_crossing_number(curve)
    if flags['yzCN']:
        out['yzCN'] = yz_crossing_number(curve)
    return out


def _control_knot_worker(curve):
    """Pickle-safe wrapper so a pool can fan out calculate_knot_properties."""
    name, det, _ = calculate_knot_properties(curve)
    return name, det


def _morans_worker(payload):
    """Run Moran's I for a single walk in a worker process.

    Imported inside the worker to keep the top-level import cost down when
    running serially.
    """
    from morans_i import morans_i_with_pvalue
    (wid, curve, genomic_positions, cycle_bins, n_perms, seed, alpha,
     tp_indices_0based, mapping, exclude_circ_leq) = payload
    res = morans_i_with_pvalue(
        curve, genomic_positions,
        cycle_bins=cycle_bins,
        n_permutations=n_perms,
        seed=seed, alpha=alpha,
        exclude_circular_genomic_offsets_leq=exclude_circ_leq)
    row = {
        'WalkID': wid,
        'n_points': int(len(genomic_positions)),
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
        'cycle_bins': int(cycle_bins),
        'genomic_position_mapping': mapping,
        'exclude_circular_genomic_offsets_leq': int(exclude_circ_leq),
    }
    by_walk = {
        'I_value': float(res['I_observed']),
        'P_value': float(res['P_value']),
        'is_significant': bool(res['is_significant']),
        'n_points': int(len(genomic_positions)),
        'exclude_circular_genomic_offsets_leq': int(exclude_circ_leq),
    }
    return wid, row, by_walk


def _resolve_workers(args):
    if not args.parallel:
        return 1
    if args.workers and args.workers > 0:
        return int(args.workers)
    try:
        return max(1, multiprocessing.cpu_count())
    except NotImplementedError:
        return 1


def _map_serial_or_pool(fn, payloads, n_workers, desc):
    """Run `fn` over `payloads`, sequentially or via ProcessPoolExecutor.

    Returns results in the same order as `payloads`.
    """
    if n_workers <= 1 or len(payloads) <= 1:
        return [fn(p) for p in tqdm(payloads, desc=desc)]
    # Preserve input order: stash the index on each future.
    out = [None] * len(payloads)
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        future_to_idx = {ex.submit(fn, p): i for i, p in enumerate(payloads)}
        for fut in tqdm(as_completed(future_to_idx),
                        total=len(payloads), desc=desc):
            out[future_to_idx[fut]] = fut.result()
    return out


def compute_walk_metrics(walk_id, curve, args):
    """Serial single-walk metric computation.

    Kept for backward callers; the parallel path goes through
    _walk_metric_worker via _map_serial_or_pool.
    """
    _, out = _walk_metric_worker((walk_id, curve, _flags_from_args(args)))
    return out


def compute_control_metric_values(control_curves, args, n_workers=1):
    """Run the same per-curve metrics on a list of control polygons.

    Parallelized over controls when n_workers > 1.
    """
    fields = []
    if args.acn:
        fields += ['acn', 'writhe', 'rms_writhe']
    if args.rog:
        fields += ['rog_nm']
    if args.xyCN:
        fields += ['xyCN']
    if args.xzCN:
        fields += ['xzCN']
    if args.yzCN:
        fields += ['yzCN']
    flags = _flags_from_args(args)
    payloads = [(curve, flags) for curve in control_curves]
    results = _map_serial_or_pool(
        _control_metric_worker, payloads, n_workers, 'Control metrics')
    store = {f: [] for f in fields}
    for out in results:
        for f in fields:
            if f in out:
                store[f].append(out[f])
    return store


# ===========================================================================
# NEW: single-curve report-card panels.
# ===========================================================================
def _panel_3d_trajectory(ax, walk_id, curve, tp_names, morans_i_dict,
                         knot_name, determinant):
    """3D panel: ordered curve + TP-colored markers + line connecting them."""
    n = curve.shape[0]
    _, colors = timepoint_colormap(n)
    # Line connecting points
    ax.plot(curve[:, 0], curve[:, 1], curve[:, 2],
            color='0.3', alpha=0.5, linewidth=1.0)
    # Points with TP-colored markers
    for i in range(n):
        ax.scatter(curve[i, 0], curve[i, 1], curve[i, 2],
                   c=[colors[i]], s=40, alpha=0.9,
                   edgecolor='black', linewidths=0.3)
    morans_text = 'N/A'
    if walk_id in morans_i_dict:
        morans_text = f"{morans_i_dict[walk_id]['I_value']:.4f}"
    title = (f"Walk-ID {walk_id}: 3D trajectory "
             f"(Moran's I = {morans_text}, knot={knot_name})")
    ax.set_xlabel('X (nm)', fontsize=8)
    ax.set_ylabel('Y (nm)', fontsize=8)
    ax.set_zlabel('Z (nm)', fontsize=8)
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.view_init(elev=20, azim=45)
    ax.grid(True, alpha=0.3)


def _panel_metric_vs_controls(ax, walk_value, control_values,
                              metric_label, color='steelblue',
                              bins=30):
    """Control histogram + single vertical line for this walk's metric value."""
    cv = np.array([v for v in control_values if np.isfinite(v)])
    if cv.size:
        ax.hist(cv, bins=bins, alpha=0.55, color='0.55',
                edgecolor='white', label=f'controls (n={cv.size})')
        cmean = float(np.mean(cv))
        cstd = float(np.std(cv))
        ax.axvline(cmean, color='0.2', linestyle='--', linewidth=1.2,
                   label=f'control mean={cmean:.3g}')
    if walk_value is not None and np.isfinite(walk_value):
        ax.axvline(walk_value, color=color, linestyle='-', linewidth=2.5,
                   label=f'walk={walk_value:.3g}')
    ax.set_xlabel(metric_label, fontsize=9)
    ax.set_ylabel('control count', fontsize=9)
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)


def _panel_knot_type_bar(ax, walk_knot, control_dists):
    """Stacked-like bar: walk's classified knot + per-control distribution.

    control_dists: dict {label: list of (knot_name, determinant) tuples}.
    """
    categories = ['0_1', '3_1', 'Complex']
    colors = {'0_1': '#2ca02c', '3_1': '#ff7f0e', 'Complex': '#d62728'}
    labels = ['Walk (single curve)'] + list(control_dists.keys())
    fractions = []
    walk_vec = np.zeros(3)
    walk_vec[categories.index(classify_name(walk_knot))] = 1.0
    fractions.append(walk_vec)
    for ctrl_name, dist in control_dists.items():
        counts = Counter(classify_name(n) for n, _ in dist)
        total = sum(counts.values())
        if total == 0:
            fractions.append(np.zeros(3))
        else:
            fractions.append(
                np.array([counts.get(c, 0) / total for c in categories]))
    fractions = np.array(fractions)
    bottoms = np.zeros(len(labels))
    x = np.arange(len(labels))
    for i, cat in enumerate(categories):
        ax.bar(x, fractions[:, i], bottom=bottoms,
               color=colors[cat], label=cat, alpha=0.9)
        bottoms += fractions[:, i]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha='right', fontsize=8)
    ax.set_ylabel('fraction', fontsize=9)
    ax.set_ylim(0, 1)
    ax.legend(loc='best', fontsize=7)
    ax.grid(True, alpha=0.3, axis='y')


def create_walk_report_card(walk_id, curve, metrics_dict,
                            control_metric_values, control_dists,
                            morans_i_dict, tp_names, args,
                            parabola_curve=None,
                            parabola_tp_start_1based=None,
                            parabola_tp_end_1based=None):
    """Assemble a single report-card page for one walk.

    Layout (4 rows x 4 cols, one PDF page per walk, no extra pages):
        Row 0 cols 0..1  : 3D trajectory
        Row 0 cols 2..3  : knot-type stacked bar vs controls
        Row 1 col 0      : ACN distribution (control histogram + walk line)
        Row 1 col 1      : Planar CN (overlaid xy/xz/yz)
        Row 1 col 2      : RMS writhe (if --acn) or Radius of gyration (if --rog)
        Row 1 col 3      : Radius of gyration (when both --acn and --rog set)
        Rows 2..3 cols 0..1 : per-walk parabola (2x2 nested: 3D/X/Y/Z)
        Rows 2..3 cols 2..3 : per-walk distance matrices (2x2 nested: 3D/X/Y/Z)

    The per-walk distance-matrix content is the walk's OWN curve over the
    main TP range. The per-walk parabola uses `parabola_curve` when
    provided — which is the same walk re-sliced to the parabola-specific
    TP range — so the parabola panel can reflect a narrower window while
    the trajectory and matrices stay on the main range. If
    `parabola_curve` is None, the parabola panel falls back to the full
    `curve` so the card still renders.
    """
    fig = plt.figure(figsize=(18, 20))
    gs = fig.add_gridspec(4, 4, hspace=0.40, wspace=0.40,
                          left=0.05, right=0.97, top=0.94, bottom=0.05,
                          height_ratios=[1.0, 1.0, 1.1, 1.1])
    knot_name = metrics_dict.get('knot_type', 'N/A')
    det = metrics_dict.get('determinant', float('nan'))

    # Row 0 cols 0..1: 3D trajectory
    ax1 = fig.add_subplot(gs[0, 0:2], projection='3d')
    _panel_3d_trajectory(ax1, walk_id, curve, tp_names, morans_i_dict,
                         knot_name, det)

    # Row 0 cols 2..3: knot-type bar, walk vs controls
    ax2 = fig.add_subplot(gs[0, 2:4])
    _panel_knot_type_bar(ax2, knot_name, control_dists)
    morans_text = (f"{morans_i_dict[walk_id]['I_value']:.4f}"
                   if walk_id in morans_i_dict else 'N/A')
    ax2.set_title(
        f"Walk-ID {walk_id}: knot type vs controls "
        f"(knot={knot_name}, det={det}, n_vertices={curve.shape[0]}, "
        f"Moran's I={morans_text})",
        fontsize=10, fontweight='bold')

    # Row 1 col 0: ACN
    ax_acn = fig.add_subplot(gs[1, 0])
    if args.acn:
        _panel_metric_vs_controls(
            ax_acn, metrics_dict.get('acn'),
            control_metric_values.get('acn', []),
            'ACN (Average Crossing Number)', color='#1f77b4')
    else:
        ax_acn.axis('off')
        ax_acn.text(0.5, 0.5, 'ACN disabled (--acn)',
                    ha='center', va='center',
                    transform=ax_acn.transAxes, fontsize=11)
    ax_acn.set_title('ACN distribution', fontsize=10, fontweight='bold')

    # Row 1 col 1: planar CNs
    ax_planar = fig.add_subplot(gs[1, 1])
    planar_specs = []
    if args.xyCN:
        planar_specs.append(('xyCN', 'xy', '#2ca02c'))
    if args.xzCN:
        planar_specs.append(('xzCN', 'xz', '#ff7f0e'))
    if args.yzCN:
        planar_specs.append(('yzCN', 'yz', '#9467bd'))
    if planar_specs:
        for metric, short, color in planar_specs:
            cv = np.array([v for v in control_metric_values.get(metric, [])
                           if np.isfinite(v)])
            if cv.size:
                ax_planar.hist(cv, bins=20, alpha=0.4, color=color,
                               edgecolor='white',
                               label=f'{short} controls (n={cv.size})')
            wv = metrics_dict.get(metric)
            if wv is not None and np.isfinite(wv):
                ax_planar.axvline(wv, color=color, linestyle='-',
                                  linewidth=2.5,
                                  label=f'{short} walk={wv:.0f}')
        ax_planar.set_xlabel('Planar crossing number', fontsize=9)
        ax_planar.set_ylabel('control count', fontsize=9)
        ax_planar.legend(loc='best', fontsize=7)
        ax_planar.grid(True, alpha=0.3)
        ax_planar.set_title(
            'Planar CN: ' + ' / '.join(s[1] for s in planar_specs),
            fontsize=10, fontweight='bold')
    else:
        ax_planar.axis('off')
        ax_planar.text(0.5, 0.5, 'No planar CN enabled (--xyCN/--xzCN/--yzCN)',
                       ha='center', va='center',
                       transform=ax_planar.transAxes, fontsize=10)

    # Row 1 col 2: RMS writhe (if --acn), else Rg (if --rog)
    ax_third = fig.add_subplot(gs[1, 2])
    if args.acn:
        _panel_metric_vs_controls(
            ax_third, metrics_dict.get('rms_writhe'),
            control_metric_values.get('rms_writhe', []),
            'RMS writhe', color='#d62728')
        ax_third.set_title('RMS writhe distribution',
                           fontsize=10, fontweight='bold')
    elif args.rog:
        _panel_metric_vs_controls(
            ax_third, metrics_dict.get('rog_nm'),
            control_metric_values.get('rog_nm', []),
            'Radius of gyration (nm)', color='#17becf')
        ax_third.set_title('Radius of gyration',
                           fontsize=10, fontweight='bold')
    else:
        ax_third.axis('off')
        ax_third.text(0.5, 0.5, 'RMS writhe / Rg disabled (--acn / --rog)',
                      ha='center', va='center',
                      transform=ax_third.transAxes, fontsize=10)

    # Row 1 col 3: if both --acn and --rog, show Rg here; else placeholder.
    ax_fourth = fig.add_subplot(gs[1, 3])
    if args.acn and args.rog:
        _panel_metric_vs_controls(
            ax_fourth, metrics_dict.get('rog_nm'),
            control_metric_values.get('rog_nm', []),
            'Radius of gyration (nm)', color='#17becf')
        ax_fourth.set_title('Radius of gyration',
                            fontsize=10, fontweight='bold')
    else:
        ax_fourth.axis('off')

    # Rows 2..3 cols 0..1: per-walk parabola (2x2 nested).
    # Uses the parabola-range slice of this walk when available; otherwise
    # falls back to the main curve and labels the title accordingly.
    if _qcpf_draw_per_walk_parabola is not None:
        parabola_pts = parabola_curve if parabola_curve is not None else curve
        range_label = ''
        if parabola_tp_start_1based is not None and parabola_tp_end_1based is not None:
            range_label = (
                f' [parabola TP range '
                f'{parabola_tp_start_1based}..{parabola_tp_end_1based} '
                f'(1-based, n={parabola_pts.shape[0]})]')
        if parabola_curve is None:
            range_label += ' [fallback: main-range curve]'
        _qcpf_draw_per_walk_parabola(
            fig, gs[2:4, 0:2], parabola_pts,
            title=f'Walk-ID {walk_id}: per-walk parabola '
                  f'(3D / X / Y / Z){range_label}')
    else:
        ax_par = fig.add_subplot(gs[2:4, 0:2])
        ax_par.axis('off')
        ax_par.text(0.5, 0.5,
                    'qc_post_filter per-walk parabola unavailable',
                    ha='center', va='center',
                    transform=ax_par.transAxes, fontsize=11)

    # Rows 2..3 cols 2..3: per-walk distance matrices (2x2 nested)
    if _qcpf_draw_per_walk_distance_matrices is not None:
        _qcpf_draw_per_walk_distance_matrices(
            fig, gs[2:4, 2:4], curve,
            title=f'Walk-ID {walk_id}: per-walk distance matrices '
                  '(3D / X / Y / Z)',
            global_colorbar=getattr(args, 'global_colorbar', False))
    else:
        ax_mat = fig.add_subplot(gs[2:4, 2:4])
        ax_mat.axis('off')
        ax_mat.text(0.5, 0.5,
                    'qc_post_filter per-walk distance matrices unavailable',
                    ha='center', va='center',
                    transform=ax_mat.transAxes, fontsize=11)

    fig.suptitle(
        f'Single-curve report card, Walk-ID {walk_id} '
        f'(n_vertices={curve.shape[0]})',
        fontsize=13, fontweight='bold')
    return fig


def create_population_summary_page(walk_rows, control_dists,
                                   control_metric_values, morans_i_dict,
                                   args):
    """Build the population-level page-1 summary figure.

    Layout (adapted from comprehensive_report_cards.create_first_page_summary
    for the single-curve setting):

      [row 0, span 2]   stacked bar: knot-type proportions for the
                        experimental population (one curve per WalkID) and
                        each available control (Random / Unknotted Globule /
                        Knotted Globule).
      [row 1, col 0]    ACN violin plot: experimental population vs Random.
      [row 1, col 1]    Moran's I histogram across walks with KDE and mean.

    All summaries are population-level: one value per WalkID, never
    per-sample. No DeWitt convergence framing. Returns the Figure.
    """
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.30, wspace=0.30,
                          left=0.08, right=0.95, top=0.92, bottom=0.08,
                          height_ratios=[1, 1])
    # Compact mode uses a simpler 2-category scheme (unknotted vs
    # everything-else-knotted); full mode keeps the 3-category scheme
    # (0_1 / 3_1 / Complex). In both modes we REFUSE to bucket placeholder
    # values ('Unavailable', 'Error', 'Unknown') from calculate_knot_properties
    # -- those mean the knot call failed, not "Complex" / "Knotted". They
    # are dropped from the bar entirely and counted in the warning below.
    _UNCLASSIFIABLE = {'Unavailable', 'Error', 'Unknown', '', None}

    if getattr(args, 'compact', False):
        colors_map = {'Unknotted': '#2ecc71', 'Knotted': '#e74c3c'}
        categories = ['Unknotted', 'Knotted']
        category_labels = ['Unknotted', 'Knotted']

        def _bucket(knot_name):
            if knot_name in _UNCLASSIFIABLE:
                return None
            return 'Unknotted' if classify_name(knot_name) == '0_1' else 'Knotted'
    else:
        colors_map = {'0_1': '#2ecc71', '3_1': '#ff9800', 'Complex': '#e74c3c'}
        categories = ['0_1', '3_1', 'Complex']
        category_labels = ['Unknot (0_1)', 'Trefoil (3_1)', 'Complex']

        def _bucket(knot_name):
            if knot_name in _UNCLASSIFIABLE:
                return None
            return classify_name(knot_name)

    # -- Top panel: knot-type stacked bar (exp vs controls) --
    ax1 = fig.add_subplot(gs[0, :])
    try:
        exp_raw = len(walk_rows)
        exp_classified = [b for b in
                          (_bucket(r.get('knot_type', 'Unknown'))
                           for r in walk_rows)
                          if b is not None]
        exp_unavail = exp_raw - len(exp_classified)
        if exp_unavail and exp_classified:
            print(f"[summary page] {exp_unavail}/{exp_raw} walks had "
                  f'knot_type in {{Unavailable, Error, Unknown}} and were '
                  f'dropped from the knot-type bar chart. Install KnotoID / '
                  f'knoodle to classify them.')
        if not exp_classified:
            ax1.axis('off')
            ax1.text(0.5, 0.5,
                     f'knot_type unavailable for all {exp_raw} walks.\n'
                     'KnotoID / knoodle not installed in this environment?',
                     ha='center', va='center', transform=ax1.transAxes,
                     fontsize=13,
                     bbox=dict(boxstyle='round', facecolor='mistyrose',
                               alpha=0.85))
            raise RuntimeError('no classifiable walks')
        exp_counts = Counter(exp_classified)
        exp_total = max(len(exp_classified), 1)
        groups = [f'Experimental walks\n(n={exp_total}'
                  + (f', {exp_unavail} unclassified' if exp_unavail else '')
                  + ')']
        group_data = {cat: [exp_counts.get(cat, 0) / exp_total]
                      for cat in categories}
        preferred_order = ['Random', 'Unknotted Globule', 'Knotted Globule']
        control_labels = ([lbl for lbl in preferred_order if lbl in control_dists]
                          + [lbl for lbl in control_dists
                             if lbl not in preferred_order])
        # Compact mode: the knot-type bar chart compares the experimental
        # population to the globule controls only. The Random control is
        # still used for the ACN violin panel on the same page, but would
        # clutter the two-category bar (it is ~100% Knotted by design).
        if getattr(args, 'compact', False):
            control_labels = [lbl for lbl in control_labels if lbl != 'Random']
        for lbl in control_labels:
            dist = control_dists[lbl]
            ctrl_classified = [b for b in (_bucket(n) for n, _ in dist)
                               if b is not None]
            ctrl_total = max(len(ctrl_classified), 1)
            ctrl_counts = Counter(ctrl_classified)
            groups.append(f'{lbl}\n(n={ctrl_total})')
            for cat in categories:
                group_data[cat].append(ctrl_counts.get(cat, 0) / ctrl_total)
        x = np.arange(len(groups))
        bottom = np.zeros(len(groups))
        bar_handles = []
        for cat in categories:
            vals = np.array(group_data[cat])
            bar = ax1.bar(x, vals, 0.6, bottom=bottom,
                          color=colors_map[cat],
                          edgecolor='black', linewidth=1.4)
            bar_handles.append(bar)
            for i, (v, b) in enumerate(zip(vals, bottom)):
                if v > 0.05:
                    ax1.text(x[i], b + v / 2, f'{v*100:.1f}%',
                             ha='center', va='center',
                             fontweight='bold', fontsize=9)
            bottom += vals
        ax1.set_ylabel('Proportion', fontsize=12, fontweight='bold')
        ax1.set_title('Knot type distribution: experimental population vs controls',
                      fontsize=14, fontweight='bold', pad=12)
        ax1.set_xticks(x)
        ax1.set_xticklabels(groups, fontsize=11)
        ax1.set_ylim(0, 1)
        ax1.legend(bar_handles, category_labels, loc='upper right',
                   frameon=True, fancybox=True, fontsize=10)
        ax1.grid(True, axis='y', alpha=0.2)
        ax1.set_axisbelow(True)
    except Exception as exc:
        ax1.text(0.5, 0.5, f'Knot-type plot unavailable:\n{exc}',
                 ha='center', va='center', transform=ax1.transAxes,
                 fontsize=11)

    # -- Bottom-left: ACN violin (experimental vs Random) --
    ax2 = fig.add_subplot(gs[1, 0])
    try:
        exp_acn = np.array([r.get('acn') for r in walk_rows
                            if r.get('acn') is not None
                            and np.isfinite(r.get('acn'))], dtype=float)
        rnd_acn = np.array([v for v in control_metric_values.get('acn', [])
                            if v is not None and np.isfinite(v)], dtype=float)
        if not args.acn:
            ax2.axis('off')
            ax2.text(0.5, 0.5, 'ACN disabled (--acn not set)',
                     ha='center', va='center', transform=ax2.transAxes,
                     fontsize=12)
        elif exp_acn.size and rnd_acn.size:
            data = [rnd_acn, exp_acn]
            parts = ax2.violinplot(data, positions=[0, 1], showmeans=True,
                                    showmedians=True, widths=0.6)
            violin_colors = ['steelblue', 'coral']
            for i, body in enumerate(parts['bodies']):
                body.set_facecolor(violin_colors[i])
                body.set_alpha(0.6)
                body.set_edgecolor('black')
                body.set_linewidth(1.4)
            for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians', 'cmeans'):
                if partname in parts:
                    parts[partname].set_edgecolor('black')
                    parts[partname].set_linewidth(1.4)
            ax2.text(0, float(np.mean(rnd_acn)),
                     f'mu={np.mean(rnd_acn):.2f}',
                     ha='right', va='center', fontsize=9,
                     bbox=dict(boxstyle='round,pad=0.3',
                               facecolor='white', alpha=0.8))
            ax2.text(1, float(np.mean(exp_acn)),
                     f'mu={np.mean(exp_acn):.2f}',
                     ha='left', va='center', fontsize=9,
                     bbox=dict(boxstyle='round,pad=0.3',
                               facecolor='white', alpha=0.8))
            try:
                from scipy.stats import mannwhitneyu
                _, pvalue = mannwhitneyu(rnd_acn, exp_acn,
                                         alternative='two-sided')
                ttext = f'Mann-Whitney U\np = {pvalue:.4g}'
                ax2.text(0.98, 0.98, ttext, transform=ax2.transAxes,
                         fontsize=9, va='top', ha='right',
                         bbox=dict(boxstyle='round,pad=0.4',
                                   facecolor='yellow', alpha=0.7))
            except Exception:
                pass
            ax2.set_ylabel('Average Crossing Number (ACN)',
                           fontsize=11, fontweight='bold')
            ax2.set_title('ACN: experimental walks vs Random control',
                           fontsize=12, fontweight='bold')
            ax2.set_xticks([0, 1])
            ax2.set_xticklabels([f'Random\n(n={rnd_acn.size})',
                                 f'Experimental\n(n={exp_acn.size})'],
                                fontsize=10)
            ax2.grid(True, axis='y', alpha=0.2)
            ax2.set_axisbelow(True)
        else:
            ax2.axis('off')
            ax2.text(0.5, 0.5, 'Insufficient ACN data for violin',
                     ha='center', va='center', transform=ax2.transAxes,
                     fontsize=11)
    except Exception as exc:
        ax2.text(0.5, 0.5, f'ACN violin unavailable:\n{exc}',
                 ha='center', va='center', transform=ax2.transAxes,
                 fontsize=10)

    # -- Bottom-right: Moran's I histogram + KDE + mean --
    ax3 = fig.add_subplot(gs[1, 1])
    try:
        I_vals = np.array([morans_i_dict[w]['I_value']
                           for w in sorted(morans_i_dict.keys())
                           if np.isfinite(morans_i_dict[w].get('I_value',
                                                               float('nan')))],
                          dtype=float)
        P_vals = np.array([morans_i_dict[w].get('P_value', float('nan'))
                           for w in sorted(morans_i_dict.keys())],
                          dtype=float)
        if I_vals.size:
            bins = max(10, min(30, I_vals.size // 2))
            counts, bins_edges, patches = ax3.hist(
                I_vals, bins=bins, alpha=0.55, color='steelblue',
                edgecolor='black', linewidth=0.9)
            significant = (P_vals < args.morans_alpha) if P_vals.size else None
            if significant is not None and significant.size == I_vals.size:
                for i, patch in enumerate(patches):
                    lo = bins_edges[i]
                    hi = bins_edges[i + 1]
                    in_bin = ((I_vals >= lo) & (I_vals < hi))
                    if i == len(patches) - 1:
                        in_bin = ((I_vals >= lo) & (I_vals <= hi))
                    if np.any(significant[in_bin]):
                        patch.set_facecolor('coral')
                        patch.set_alpha(0.7)
            if I_vals.size > 1:
                try:
                    from scipy.stats import gaussian_kde
                    kde = gaussian_kde(I_vals)
                    x_range = np.linspace(I_vals.min(), I_vals.max(), 200)
                    density = kde(x_range)
                    bin_width = bins_edges[1] - bins_edges[0]
                    scaled = density * I_vals.size * bin_width
                    ax3.plot(x_range, scaled, color='darkblue',
                             linewidth=2, alpha=0.85, label='KDE')
                except Exception:
                    pass
            mean_val = float(np.mean(I_vals))
            ax3.axvline(mean_val, color='red', linestyle='--',
                         linewidth=2.5, alpha=0.9,
                         label=f'mean = {mean_val:.3f}')
            ax3.axvline(0, color='gray', linestyle=':', linewidth=1.5,
                         alpha=0.6, label='0 (no correlation)')
            n_sig = (int(np.sum(significant))
                     if significant is not None else 0)
            pct_sig = 100.0 * n_sig / max(I_vals.size, 1)
            stats_text = (
                f'N = {I_vals.size}\n'
                f'mean = {mean_val:.3f}\n'
                f'std = {float(np.std(I_vals)):.3f}\n'
                f'sig (p<{args.morans_alpha}): '
                f'{n_sig} ({pct_sig:.1f}%)')
            ax3.text(0.02, 0.98, stats_text, transform=ax3.transAxes,
                      fontsize=9, va='top',
                      bbox=dict(boxstyle='round,pad=0.5',
                                facecolor='white', alpha=0.85,
                                edgecolor='gray'))
            ax3.set_xlabel("Moran's I", fontsize=11, fontweight='bold')
            ax3.set_ylabel('Frequency', fontsize=11, fontweight='bold')
            ax3.set_title("Moran's I distribution across walks",
                           fontsize=12, fontweight='bold')
            ax3.legend(loc='upper right', frameon=True, fancybox=True,
                        fontsize=9)
            ax3.grid(True, alpha=0.2)
            ax3.set_axisbelow(True)
        else:
            ax3.axis('off')
            ax3.text(0.5, 0.5, "No Moran's I data available",
                     ha='center', va='center', transform=ax3.transAxes,
                     fontsize=11)
    except Exception as exc:
        ax3.text(0.5, 0.5, f"Moran's I plot unavailable:\n{exc}",
                 ha='center', va='center', transform=ax3.transAxes,
                 fontsize=10)

    fig.suptitle('Dataset summary overview Widefield',
                 fontsize=16, fontweight='bold', y=0.98)
    return fig


def create_morans_summary_figure(morans_i_dict, args):
    """Summary page across all walks for Moran's I.

    Left: histogram of I values with counts of significant vs not.
    Right: scatter of I vs P with the significance line.
    """
    if not morans_i_dict:
        return None
    wids = sorted(morans_i_dict.keys())
    I = np.array([morans_i_dict[w]['I_value'] for w in wids])
    P = np.array([morans_i_dict[w].get('P_value', float('nan'))
                  for w in wids])
    sig = np.array([bool(morans_i_dict[w].get('is_significant', False))
                    for w in wids])
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    if np.isfinite(I).any():
        axes[0].hist(I[np.isfinite(I)], bins=min(30, max(5, len(I) // 3)),
                     color='#1f77b4', alpha=0.75, edgecolor='white',
                     label=f'all walks (n={len(I)})')
    axes[0].axvline(0, color='0.3', linestyle='--', linewidth=1)
    axes[0].set_xlabel("Moran's I")
    axes[0].set_ylabel('count')
    axes[0].set_title(
        f"Moran's I across walks "
        f'(n_sig={int(sig.sum())}/{len(sig)} at alpha={args.morans_alpha})')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    if np.isfinite(P).any():
        axes[1].scatter(I[~sig], P[~sig], color='0.5', s=25,
                        alpha=0.8, label='not significant')
        axes[1].scatter(I[sig], P[sig], color='#d62728', s=35,
                        alpha=0.9, label='significant')
    axes[1].axhline(args.morans_alpha, color='0.3', linestyle='--',
                    linewidth=1, label=f'alpha={args.morans_alpha}')
    axes[1].set_xlabel("Moran's I")
    axes[1].set_ylabel('P-value (one-tailed)')
    axes[1].set_title("Moran's I vs P-value")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    fig.suptitle(
        f"Moran's I summary "
        f'(perms={args.morans_permutations}, '
        f'cycle_bins='
        f'{args.genome_cycle_bins if args.genome_cycle_bins is not None else "n_vertices"}, '
        f'mapping='
        f'{"absolute_tp" if args.morans_absolute_tp else "range_relative"})',
        fontsize=13, fontweight='bold')
    fig.tight_layout()
    return fig


def create_rog_side_figure(walk_metric_rows, control_values, args):
    """Separate figure when both --acn and --rog are set so Rg still gets its own page."""
    if not (args.acn and args.rog):
        return None
    fig, ax = plt.subplots(figsize=(10, 6))
    cv = np.array([v for v in control_values.get('rog_nm', [])
                   if np.isfinite(v)])
    walk_vals = np.array([m.get('rog_nm') for m in walk_metric_rows
                          if m.get('rog_nm') is not None
                          and np.isfinite(m.get('rog_nm'))])
    bins = 30
    if cv.size:
        ax.hist(cv, bins=bins, alpha=0.55, color='0.55',
                edgecolor='white', label=f'controls (n={cv.size})')
    if walk_vals.size:
        ax.hist(walk_vals, bins=bins, alpha=0.55, color='#17becf',
                edgecolor='white',
                label=f'walks (n={walk_vals.size}, one value per walk)')
    ax.set_xlabel('Radius of gyration (nm)')
    ax.set_ylabel('count')
    ax.set_title('Radius of gyration: walks vs controls', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig


# ===========================================================================
# NEW: CSV summary writer for the single-curve case.
# ===========================================================================
def write_results_csv(rows, control_dists, output_csv):
    """Write one row per walk to CSV plus per-walk KL / EMD vs each control."""
    if not rows:
        print('No walk rows to write.')
        return
    df = pd.DataFrame(rows)
    walk_dist = [(r['knot_type'], r['determinant']) for r in rows]
    for ctrl_name, ctrl in control_dists.items():
        safe = ctrl_name.replace(' ', '_').lower()
        try:
            kl, _, _ = calculate_kl_divergence(walk_dist, ctrl)
            df[f'kl_vs_{safe}'] = kl
        except Exception as exc:
            print(f'KL vs {ctrl_name}: {exc}')
            df[f'kl_vs_{safe}'] = float('nan')
        try:
            emd = calculate_emd(walk_dist, ctrl)
            df[f'emd_vs_{safe}'] = emd
        except Exception as exc:
            print(f'EMD vs {ctrl_name}: {exc}')
            df[f'emd_vs_{safe}'] = float('nan')
    df.to_csv(output_csv, index=False)
    print(f'Wrote {output_csv}')


# ===========================================================================
# CLI
# ===========================================================================
# Constrained PC-space helpers.
#
# Ported from walk_curator_lite.py::compute_constrained_axes (earlier walk-curation code)
# and the canonical-orientation logic in transform_to_pca. Important: PC2
# here is NOT the second PCA eigenvector; it is the XY-plane rotation of
# PC1's XY projection (`(x, y) -> (-y, x)`). PC3 is the right-handed cross
# product of PC1 and PC2. The resulting basis is always right-handed and
# canonically oriented (PC1 -> +Z, PC2 -> +X, PC3 by right-hand rule).
# ===========================================================================
def _constrained_axes_from_pc1(pc1):
    """Return (pc1, pc2, pc3) unit vectors using walk_curator_lite's
    constrained-axis construction (PC2 lives in the XY plane, PC3 is the
    cross product)."""
    pc1 = np.asarray(pc1, dtype=float)
    pc1 = pc1 / np.linalg.norm(pc1)
    xy_norm = np.linalg.norm(pc1[:2])
    if xy_norm < 1e-9:
        # Major axis is nearly vertical (along Z): fall back to (X, Y)
        pc2 = np.array([1.0, 0.0, 0.0])
        pc3 = np.array([0.0, 1.0, 0.0])
    else:
        u = pc1[:2] / xy_norm  # XY projection, normalized
        # (x, y) -> (-y, x): 90-degree rotation in the XY plane, z = 0
        pc2 = np.array([-u[1], u[0], 0.0])
        pc3 = np.cross(pc1, pc2)
        pc3 = pc3 / np.linalg.norm(pc3)
    return pc1, pc2, pc3


def _canonical_orient_pc(pc1, pc2, pc3):
    """Flip sign of PC1/PC2/PC3 to enforce walk_curator_lite's canonical
    orientation: PC1 has +Z, PC2 has +X, PC3 follows the right-hand rule
    from PC1 and PC2."""
    if pc1[2] < 0:
        pc1 = -pc1
    if pc2[0] < 0:
        pc2 = -pc2
    expected_pc3 = np.cross(pc1, pc2)
    if np.dot(pc3, expected_pc3) < 0:
        pc3 = -pc3
    return pc1, pc2, pc3


def _canonical_actualpc(pc1, pc2, pc3):
    """Deterministic sign convention for the unconstrained-PCA frame.

    Flip PC1 and PC2 so each has its largest-magnitude component positive,
    then set PC3 = PC1 x PC2 for a right-handed frame. Signs of PCA
    eigenvectors are otherwise arbitrary; this makes runs reproducible.
    """
    if pc1[int(np.argmax(np.abs(pc1)))] < 0:
        pc1 = -pc1
    if pc2[int(np.argmax(np.abs(pc2)))] < 0:
        pc2 = -pc2
    pc3 = np.cross(pc1, pc2)
    return pc1, pc2, pc3


def _walk_to_constrained_pc(curve, major_axis='walk3d'):
    """Project one walk's (N, 3) curve into a per-walk PC frame (PC1, PC2, PC3).

    `major_axis` selects the frame:
      - 'walk3d' (default): PC1 = largest-eigenvalue eigenvector of the walk's
        3D covariance (may tilt out of plane); PC2 constrained into the XY
        plane, PC3 = PC1 x PC2 (tilts with PC1, not aligned with Z).
      - 'walkxy': PC1 = walk's in-plane long axis (largest eigenvector of the
        2D XY covariance, z=0). PC2 stays in XY, PC3 = Z (up to sign),
        isolating the depth axis.
      - 'actualpc': unconstrained 3D PCA -- PC1/PC2/PC3 are the three
        eigenvectors of the 3D covariance (descending eigenvalue), with a
        deterministic sign convention. Nothing is forced.
    The returned array is centered at the walk's centroid.
    """
    centered = curve - curve.mean(axis=0)
    R = np.column_stack(_walk_pc_basis(curve, major_axis))  # basis columns
    return centered @ R


def _walk_pc_basis(curve, major_axis):
    """Return (pc1, pc2, pc3) unit basis vectors for one walk's PC frame.

    Same frame definitions as `_walk_to_constrained_pc`; returns the axes
    themselves (not the projected curve) so callers can average them.
    """
    centered = curve - curve.mean(axis=0)
    # eigh returns eigenvalues in ascending order; biggest is last column
    _, eigvecs = np.linalg.eigh(np.cov(centered, rowvar=False))
    if major_axis == 'walkxy':
        # Long axis from the XY plane only, embedded as an in-plane 3D vector.
        _, eig_xy = np.linalg.eigh(np.cov(centered[:, :2], rowvar=False))
        pc1 = np.array([eig_xy[0, -1], eig_xy[1, -1], 0.0])
        return _canonical_orient_pc(*_constrained_axes_from_pc1(pc1))
    if major_axis == 'walk3d':
        return _canonical_orient_pc(*_constrained_axes_from_pc1(eigvecs[:, -1]))
    if major_axis == 'actualpc':
        return _canonical_actualpc(
            eigvecs[:, -1], eigvecs[:, -2], eigvecs[:, -3])
    raise ValueError(f'Unknown major_axis={major_axis!r}')


def walks_to_constrained_pc(walks, major_axis='walk3d'):
    """Map a walks dict {wid -> (N, 3)} into constrained PC coordinates."""
    return {wid: _walk_to_constrained_pc(curve, major_axis=major_axis)
            for wid, curve in walks.items()}


# ===========================================================================
# PC-frame schematic cartoon (fixed illustration of where PC1/PC2/PC3 lie
# relative to XYZ, one per frame mode). Not data-derived.
# ===========================================================================
_PC_CARTOON_COL = {'PC1': '#d62728', 'PC2': '#2ca02c', 'PC3': '#1f77b4'}
_PC_CARTOON_TITLE = {
    'walk3d': 'walk3d  (PC1 = 3D major axis)',
    'walkxy': 'walkxy  (PC1 in XY, PC3 = Z)',
    'actualpc': 'actualpc  (3D PCA, unconstrained)',
}
_PC_CARTOON_DESC = {
    'walk3d': [('PC1', 'PC1: 3D long axis — tilts out of plane'),
               ('PC2', 'PC2: lies in the XY plane'),
               ('PC3', 'PC3: perpendicular to both (tilted)')],
    'walkxy': [('PC1', 'PC1: long axis, in the XY plane'),
               ('PC2', 'PC2: in the XY plane (perp. PC1)'),
               ('PC3', 'PC3: aligned with Z (depth)')],
    'actualpc': [('PC1', 'PC1: 1st eigenvector (major axis)'),
                 ('PC2', 'PC2: 2nd eigenvector (free)'),
                 ('PC3', 'PC3: 3rd eigenvector (free)')],
}
def _pc_median_axes(walks, major_axis):
    """Per-frame median-orientation arrow + per-cell axes for the cartoon.

    For each PC axis: the bold arrow is built from the MEDIAN tilt above the
    XY plane (robust to the Z-tilted tail, unlike a vector average which
    collapses onto the coherent minority) combined with the dominant in-plane
    azimuth. Returns (arrows, vecs, med_elev) where vecs holds the per-cell
    unit axes for the faint spread cloud and med_elev the median tilt (deg).
    """
    labs = ('PC1', 'PC2', 'PC3')
    vecs = {l: [] for l in labs}
    for curve in walks.values():
        b = _walk_pc_basis(curve, major_axis)
        for k, l in enumerate(labs):
            vecs[l].append(b[k])
    arrows, med_elev = {}, {}
    for l in labs:
        V = np.array(vecs[l])
        elev = np.degrees(np.arcsin(np.clip(np.abs(V[:, 2]), 0, 1)))
        eM = float(np.median(elev)); med_elev[l] = eM
        T2 = V[:, :2].T @ V[:, :2]                 # dominant in-plane azimuth
        _, v2 = np.linalg.eigh(T2)
        az = v2[:, -1]
        az = az / (np.linalg.norm(az) + 1e-12)
        er = np.radians(eM)
        arrows[l] = np.array([az[0]*np.cos(er), az[1]*np.cos(er), np.sin(er)])
    return arrows, vecs, med_elev


def build_pc_frame_cartoon_figure(mode, walks):
    """3D cartoon of the PC axes vs XYZ for `mode` (data-derived).

    Bold arrow per PC = MEDIAN orientation (median tilt above the XY plane +
    dominant in-plane azimuth); faint lines = the per-cell axis spread. The
    median tilt is used instead of a vector average, which for a broad in-plane
    distribution collapses onto a coherent Z-tilted minority and points where
    few walks lie. Each PC's median tilt (deg above plane) is annotated.
    """
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    arrows, vecs, med_elev = _pc_median_axes(walks, mode)
    n = len(walks)
    fig = plt.figure(figsize=(8.5, 8.5))
    ax = fig.add_subplot(111, projection='3d')
    sq = [[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]]
    ax.add_collection3d(Poly3DCollection([sq], facecolor='0.5', alpha=0.10,
                                         edgecolor='0.7', linewidths=0.5))
    L = 1.05
    for v, lab in [([L, 0, 0], 'X'), ([0, L, 0], 'Y'), ([0, 0, L], 'Z')]:
        ax.quiver(0, 0, 0, *v, color='0.6', lw=1.2, arrow_length_ratio=0.08)
        ax.text(*[x*1.15 for x in v], lab, color='0.4', fontsize=10)
    for lab in ('PC1', 'PC2', 'PC3'):
        col = _PC_CARTOON_COL[lab]
        a = arrows[lab]
        V = np.array(vecs[lab])
        step = max(1, len(V) // 350)
        for u in V[::step]:                        # faint per-cell spread
            s = u if np.dot(u, a) >= 0 else -u
            ax.plot([0, s[0]], [0, s[1]], [0, s[2]], color=col, lw=0.3, alpha=0.08)
        ax.quiver(0, 0, 0, *a, color=col, lw=2.8, arrow_length_ratio=0.12)
        ax.text(*[x*1.12 for x in a], lab, color=col, fontsize=10,
                fontweight='bold')
    ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.2, 1.2); ax.set_zlim(-1.2, 1.2)
    ax.set_box_aspect((1, 1, 1)); ax.view_init(elev=20, azim=35)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_title(f'PC frame: {_PC_CARTOON_TITLE[mode]}\n'
                 f'bold = median orientation, faint = per-cell spread ({n} cells)',
                 fontsize=12, fontweight='bold')
    for j, (lab, txt) in enumerate(_PC_CARTOON_DESC[mode]):
        fig.text(0.5, 0.11 - 0.032*j,
                 f'{txt}   [median {med_elev[lab]:.0f}° from plane]',
                 ha='center', va='top', fontsize=9.5, color=_PC_CARTOON_COL[lab])
    return fig


# ===========================================================================
# Compact-mode CSV writer.
# ===========================================================================
_COMPACT_COLUMNS = (
    'WalkID', 'knot_type', 'knot_type_knoodle', 'determinant', 'ACN', 'xyCN', 'yzCN', 'xzCN', 'RoG', 'RMS_writhe',
)


def write_compact_summary_csv(rows, output_csv):
    """Write the compact summary CSV with one row per WalkID.

    Columns: WalkID, knot_type, ACN, xyCN, yzCN, xzCN, RoG, RMS_writhe.
    Values are pulled from the per-walk metric dict produced by
    `compute_walk_metrics`. A missing metric is written as the empty
    string (reads as NaN in pandas), matching the spec's requirement
    that unavailable values not be silently placeholdered.
    """
    # Map internal metric names -> compact column names
    col_map = {
        'WalkID':            ('WalkID',            None),
        'knot_type':         ('knot_type',         None),
        'knot_type_knoodle': ('knot_type_knoodle', None),
        'determinant':       ('determinant',       None),
        'ACN':               ('acn',               None),
        'xyCN':        ('xyCN',        None),
        'yzCN':        ('yzCN',        None),
        'xzCN':        ('xzCN',        None),
        'RoG':         ('rog_nm',      None),
        'RMS_writhe':  ('rms_writhe',  None),
    }
    with open(output_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(_COMPACT_COLUMNS))
        w.writeheader()
        for row in rows:
            out = {}
            for col in _COMPACT_COLUMNS:
                src_key, _ = col_map[col]
                v = row.get(src_key)
                if v is None or (isinstance(v, float) and not np.isfinite(v)):
                    out[col] = ''
                else:
                    out[col] = v
            w.writerow(out)


def _positive_int(s):
    v = int(s)
    if v < 1:
        raise argparse.ArgumentTypeError(f'must be >= 1, got {v}')
    return v


def _parse_args():
    p = argparse.ArgumentParser(
        description='Single-curve comprehensive report cards, adapted from '
                    'comprehensive_report_cards.py for one curve per WalkID.')
    p.add_argument(
        '--input', required=True,
        help='Final filtered walks CSV (RodRunner output, produced by its drift_correct.py step).')
    p.add_argument(
        '--outdir', default='.',
        help='Output directory (default: current directory).')
    p.add_argument(
        '--output_prefix', default='report_cards',
        help='Output filename prefix (default: report_cards).')
    p.add_argument(
        '--tp_start_1based', type=_positive_int, default=3,
        help='Main analysis TP range start, 1-based inclusive. Used for '
             'report-card 3D trajectories, distance matrices, per-walk '
             'metrics, and Moran\'s I. Default 3.')
    p.add_argument(
        '--tp_end_1based', type=_positive_int, default=22,
        help='Main analysis TP range end, 1-based inclusive. Default 22.')
    p.add_argument(
        '--parabola_tp_start_1based', type=_positive_int, default=3,
        help='Parabola-specific TP range start, 1-based inclusive. Applied '
             'to BOTH the global parabola page and the per-walk parabola '
             'panels. Independent of --tp_start_1based. Default 3.')
    p.add_argument(
        '--parabola_tp_end_1based', type=_positive_int, default=22,
        help='Parabola-specific TP range end, 1-based inclusive. Default 22.')
    p.add_argument(
        '--n_random_controls', type=_positive_int, default=1000,
        help='Number of random n-gons in the convex-hull control. '
             'Default 1000.')
    p.add_argument(
        '--unknotted_globule', type=str, default=None,
        help='Directory of unknotted globule TSVs (optional).')
    p.add_argument(
        '--knotted_globule', type=str, default=None,
        help='Directory of knotted globule TSVs (optional).')
    p.add_argument(
        '--samples_per_globule', type=_positive_int, default=5,
        help='n-gons per 10k-gon file. Default 5.')
    p.add_argument(
        '--segment_by', choices=['arc_length', 'point_count'],
        default='arc_length',
        help='Globule segmentation method. Default arc_length.')
    p.add_argument(
        '--show_random', action='store_true',
        help='Include random-polygon control when a globule is enabled.')
    p.add_argument(
        '--globule_cache_csv', type=str, default=None,
        help='Path to a globule-control cache CSV. The unknotted/knotted '
             'globule knot-type distributions depend only on the globule '
             'dirs, n_vertices, samples_per_globule and segmentation (NOT on '
             'the walk set), so they can be computed once and reused across '
             'datasets/ranges with the same settings. If the file exists it '
             'is loaded (validated against the current settings); otherwise '
             'the globules are computed and written to it. The random control '
             'is always recomputed per dataset (its hull is walk-specific).')
    # ---- Moran's I (now computed internally by default) ----
    p.add_argument(
        '--morans_permutations', type=_positive_int, default=999,
        help="Permutations for Moran's I one-tailed p-value. Default 999.")
    p.add_argument(
        '--genome_cycle_bins', type=int, default=None,
        help='Cycle size for the angle mapping theta=g*2pi/cycle_bins. '
             'Default: use n_vertices (the current TP range length). '
             'Setting to 20 reproduces the reference C++ convention.')
    p.add_argument(
        '--morans_absolute_tp', action='store_true',
        help='Use absolute 0-based TP indices as genomic positions. '
             'Default uses range-relative 0..N-1.')
    p.add_argument(
        '--morans_seed', type=int, default=None,
        help="RNG seed for Moran's I permutations (default: non-deterministic).")
    p.add_argument(
        '--morans_alpha', type=float, default=0.05,
        help="Significance threshold for the is_significant flag. Default 0.05.")
    p.add_argument(
        '--morans_exclude_circular_offsets_leq', type=int, default=0,
        help="Zero out Moran's I weights for pairs whose circular GENOMIC "
             "chain offset is <= this value (direct chain neighbors, not "
             'spatial nearest neighbors). Uses curve order to define the '
             'chain; circular distance = min(|i-j|, N-|i-j|). 0 disables '
             'the exclusion (default, backward compatible). 1 removes the '
             'immediate chain neighbors on the circular bacterial genome.')
    p.add_argument(
        '--morans_output_csv', type=str, default=None,
        help="Where to write the per-walk Moran's I CSV. Default: "
             '{outdir}/{output_prefix}_morans_i_results.csv.')
    p.add_argument(
        '--skip_morans', action='store_true',
        help="Skip Moran's I computation entirely. Useful for large datasets "
             "where the permutation test is the bottleneck.")
    p.add_argument(
        '--no_knot_type', action='store_true',
        help="Skip KnotoID projection analysis (the slow knot step). "
             "Still computes knot determinant via knoodle and derives "
             "knotted/unknotted classification from it.")
    p.add_argument(
        '--morans_cache_csv', type=str, default=None,
        help='Optional path to a previously computed per-walk Moran\'s I CSV. '
             'If provided, values are loaded from this file instead of '
             'recomputing. Accepts both the new WalkID/I_value/P_value schema '
             'and the legacy Walk-ID/I-value/P-value schema.')
    p.add_argument(
        '--morans_i_file', type=str, default=None,
        help="DEPRECATED alias for --morans_cache_csv (external precomputed "
             "Moran's I). Kept only for backward compatibility.")
    p.add_argument('--acn', action='store_true', help='Enable ACN + writhe + RMS writhe.')
    p.add_argument('--n_acn_projections', type=_positive_int, default=1000)
    p.add_argument('--rog', action='store_true', help='Enable Radius of gyration.')
    p.add_argument('--xyCN', action='store_true')
    p.add_argument('--xzCN', action='store_true')
    p.add_argument('--yzCN', action='store_true')
    p.add_argument(
        '--kendall', action='store_true',
        help='Add a Kendall shape-space summary figure across all walks.')
    p.add_argument('--kendall_max_iter', type=_positive_int, default=100)
    p.add_argument('--kendall_tol', type=float, default=1e-6)
    p.add_argument(
        '--parallel', action='store_true',
        help='Enable parallel processing (ProcessPoolExecutor) for per-walk '
             "metrics, control knot-type, control metric loops, and Moran's I.")
    p.add_argument(
        '--workers', type=_positive_int, default=None,
        help='Number of worker processes when --parallel is set. '
             'Default: CPU count.')
    p.add_argument('--verbose', action='store_true')
    p.add_argument(
        '--skip_incomplete', action='store_true',
        help='If set, rows in the input CSV with empty or non-finite '
             'x_nm / y_nm / z_nm are treated as missing for that walk. '
             'The walk then falls into the existing missing-TP skip '
             'path (counted in skipped_missing_tp) instead of being '
             'used with bogus coordinates. Without this flag, the '
             'loader fails loudly on the first such row.')
    p.add_argument(
        '--global_colorbar', action='store_true',
        help='Share one colorbar across all panels of each distance-matrix '
             'figure (XYZ family scaled to the Z off-diagonal max, '
             'constrained-PC family scaled to PC3 off-diagonal max; '
             'also applied to the per-walk block scaled to Z). When this '
             'flag is not set, each panel uses its own per-panel color '
             'scale. Raw nm values are plotted in both cases -- no '
             '[0,1] normalization. Default: off.')
    p.add_argument(
        '--xyz_distance_caps_nm', type=str, default='500,300,300,210',
        help='Comma-separated 4-tuple (3D, X, Y, Z) of vmax caps in nm '
             'applied to the GLOBAL XYZ distance-matrix figure. Each '
             'panel saturates above its cap; matrix values are raw, not '
             'rescaled. Pass "auto" to disable caps and use each panel\'s '
             'off-diagonal max instead. Default: 500,300,300,210.')
    p.add_argument(
        '--pc_distance_caps_nm', type=str, default='500,300,300,210',
        help='Comma-separated 4-tuple (3D, PC1, PC2, PC3) of vmax caps '
             'in nm applied to the GLOBAL PC-space distance-matrix '
             'figure. Pass "auto" to disable caps. Default: '
             '500,300,300,210.')
    p.add_argument(
        '--pc_frames', nargs='+',
        choices=['walk3d', 'walkxy', 'actualpc'],
        default=['walk3d', 'walkxy', 'actualpc'],
        help='Which per-cell PC frames to render for the parabola and PC '
             'distance-matrix pages (the XYZ parabola + distance matrices are '
             'always kept). Default: all three. Each shown frame gets its own '
             'schematic cartoon page. walk3d: 3D major axis (PC3 tilts). '
             'walkxy: in-plane major axis, PC3 = Z. actualpc: unconstrained '
             '3D PCA. Give a subset to plot only those, e.g. '
             '--pc_frames walkxy actualpc.')
    p.add_argument(
        '--compact', action='store_true',
        help='Compact mode: write a 5-page summary PDF (population summary, '
             'XYZ parabola, constrained PC-space parabola, XYZ distance '
             'matrices, constrained PC-space distance matrices) plus a '
             'compact CSV with columns WalkID, knot_type, ACN, xyCN, yzCN, '
             'xzCN, RoG, RMS_writhe. No per-walk report cards are produced. '
             'Auto-enables --acn, --rog, --xyCN, --xzCN, --yzCN when not '
             'already set so the compact CSV has all metric columns.')
    p.add_argument(
        '--sections', nargs='+',
        choices=['summary', 'parabolas', 'distance_matrices', 'all'],
        default=['all'],
        help='Which compact-report sections to produce. Default all. '
             'Selecting only parabolas and/or distance_matrices (without '
             'summary) runs a fast path that SKIPS the globule/random '
             'controls, Moran\'s I, and per-walk metrics (none are needed for '
             'those pages) and writes <prefix>_<sections>.pdf. Requesting '
             'summary (or all) runs the full pipeline. Ignored in '
             'comprehensive (non-compact) mode.')
    p.add_argument(
        '--start_locus', type=_positive_int, default=1,
        help='1-based position within the TP range at which to START the '
             'ordered curve, circularly rolling the loci. Default 1 (no '
             'change). Use to re-anchor the parabola + distance matrices at a '
             'chosen locus, e.g. Ter: --start_locus 17 reorders a 20-locus '
             'range to 17,18,...,20,1,...,16. Applied to both the analysis '
             'and parabola ranges (must be valid for both). The parabola '
             'offset is linear, so this genuinely re-anchors it (not just a '
             'relabel); PC frames are unaffected (order-invariant) but their '
             'projected coordinates roll with the loci.')
    args = p.parse_args()
    if args.tp_end_1based < args.tp_start_1based:
        p.error('--tp_end_1based must be >= --tp_start_1based')
    if args.parabola_tp_end_1based < args.parabola_tp_start_1based:
        p.error('--parabola_tp_end_1based must be >= '
                '--parabola_tp_start_1based')

    def _parse_caps(raw, flag_name):
        s = (raw or '').strip().lower()
        if s in ('', 'auto', 'none', 'off'):
            return None
        parts = [v.strip() for v in raw.split(',')]
        if len(parts) != 4:
            p.error(f'{flag_name} must be 4 comma-separated nm values or '
                    f'"auto"; got {raw!r}.')
        try:
            caps = tuple(float(v) for v in parts)
        except ValueError:
            p.error(f'{flag_name} values must be numeric; got {raw!r}.')
        if any(c <= 0 for c in caps):
            p.error(f'{flag_name} values must be > 0; got {caps}.')
        return caps

    args.xyz_distance_caps_nm = _parse_caps(
        args.xyz_distance_caps_nm, '--xyz_distance_caps_nm')
    args.pc_distance_caps_nm = _parse_caps(
        args.pc_distance_caps_nm, '--pc_distance_caps_nm')

    if 'all' in args.sections:
        args.sections = ['summary', 'parabolas', 'distance_matrices']

    if args.compact:
        # Auto-enable metric flags that feed the compact CSV columns.
        args.acn = True
        args.rog = True
        args.xyCN = True
        args.xzCN = True
        args.yzCN = True
    return args


def _render_sections_subset(args, walks, parabola_walks, parabola_n_vertices,
                            sections):
    """Fast path for parabolas-only / distance-matrices-only compact runs.

    Renders just the requested geometry pages (XYZ + per-frame, each PC frame
    led by its cartoon) and, if parabolas are requested, the 3D parabola
    side-cars. Skips the population summary, so no globule/random controls,
    no Moran's I, and no per-walk metrics are computed. Writes
    <prefix>_<sections>.pdf. `sections` is a subset of
    {'parabolas', 'distance_matrices'}.
    """
    want_parab = 'parabolas' in sections
    want_mat = 'distance_matrices' in sections
    tag = '_'.join(s for s in ('parabolas', 'distance_matrices')
                   if s in sections)
    pdf_path = os.path.join(args.outdir, f'{args.output_prefix}_{tag}.pdf')
    pc_walks = {f: walks_to_constrained_pc(walks, major_axis=f)
                for f in args.pc_frames}
    pc_parab = {f: walks_to_constrained_pc(parabola_walks, major_axis=f)
                for f in args.pc_frames}
    print(f'\nSections mode ({tag}): skipping controls / Moran\'s / metrics.')
    print(f'Writing PDF: {pdf_path}')

    def _parab(w, names, label):
        fig, _, _ = build_mean_parabola_xyz_figure(
            w, args.parabola_tp_start_1based, args.parabola_tp_end_1based,
            n_tp_in_range=parabola_n_vertices, axis_names=names,
            space_label=label)
        return fig

    def _mat(w, names, label, caps):
        fig, _ = build_mean_distance_matrices_figure(
            w, args.tp_start_1based, args.tp_end_1based, axis_names=names,
            space_label=label, global_colorbar=args.global_colorbar,
            axis_caps_nm=caps)
        return fig

    with PdfPages(pdf_path) as pdf:
        if want_parab and build_mean_parabola_xyz_figure is not None:
            fig = _parab(parabola_walks, ('X', 'Y', 'Z'), 'XYZ')
            pdf.savefig(fig); plt.close(fig)
        if want_mat and build_mean_distance_matrices_figure is not None:
            fig = _mat(walks, ('X', 'Y', 'Z'), 'XYZ', args.xyz_distance_caps_nm)
            pdf.savefig(fig); plt.close(fig)
        for frame in args.pc_frames:
            cfig = build_pc_frame_cartoon_figure(frame, walks)
            pdf.savefig(cfig); plt.close(cfig)
            label = f'PC space ({frame})'
            if want_parab and build_mean_parabola_xyz_figure is not None:
                fig = _parab(pc_parab[frame], ('PC1', 'PC2', 'PC3'), label)
                pdf.savefig(fig); plt.close(fig)
            if want_mat and build_mean_distance_matrices_figure is not None:
                fig = _mat(pc_walks[frame], ('PC1', 'PC2', 'PC3'), label,
                           args.pc_distance_caps_nm)
                pdf.savefig(fig); plt.close(fig)
    print(f'Done. Sections PDF: {pdf_path}')

    if want_parab and build_mean_parabola_3d_only_figure is not None:
        sidecars = [('XYZ', 'xyz', parabola_walks)]
        for frame in args.pc_frames:
            sidecars.append((f'PC space ({frame})', f'pc_{frame}',
                             pc_parab[frame]))
        for space_label, sctag, slice_walks in sidecars:
            side_path = os.path.join(
                args.outdir, f'{args.output_prefix}_parabola_3d_{sctag}.pdf')
            fig_side, _, _ = build_mean_parabola_3d_only_figure(
                slice_walks, args.parabola_tp_start_1based,
                args.parabola_tp_end_1based, n_tp_in_range=parabola_n_vertices,
                space_label=space_label)
            fig_side.savefig(side_path); plt.close(fig_side)
            print(f'Wrote {side_path}')


# ===========================================================================
# main
# ===========================================================================
def main():
    args = _parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    n_workers = _resolve_workers(args)
    print('=' * 60)
    print('Single-curve comprehensive report cards')
    print('=' * 60)
    print(f'Input: {args.input}')
    print(f'Output dir: {args.outdir}')
    print(f'Output prefix: {args.output_prefix}')
    print(f'Analysis TP range (1-based, inclusive): '
          f'[{args.tp_start_1based}, {args.tp_end_1based}]')
    print(f'Parallel: {args.parallel} (workers={n_workers})')
    if not KNOTID_AVAILABLE:
        print('WARNING: KnotoID/knoodle unavailable. Knot type / determinant '
              'fields will be "Unavailable" and controls will not compute '
              'topology. Proceeding for metric-only QC.')

    # ---- Load the final filtered walks (main range) ----
    skip_incomplete = bool(getattr(args, 'skip_incomplete', False))
    loaded = load_single_curve_walks(
        args.input, args.tp_start_1based, args.tp_end_1based,
        skip_incomplete=skip_incomplete)
    walks = loaded['walks']
    tp_names = loaded['tp_names']
    n_vertices = loaded['n_vertices']
    print(f'Main TPs in range: {tp_names[0]} .. {tp_names[-1]} '
          f'(n_vertices={n_vertices})')
    print(f'Walks with full coverage: {len(walks)} '
          f'(skipped missing={loaded["skipped_missing_tp"]}, '
          f'skipped duplicate={loaded["skipped_duplicate_tp"]}, '
          f'input ids in range={loaded["n_walks_input_in_range"]})')
    if skip_incomplete and loaded.get('skipped_empty_coord_rows', 0):
        print(f'  --skip_incomplete: dropped '
              f'{loaded["skipped_empty_coord_rows"]} rows with empty/'
              f'non-finite x_nm / y_nm / z_nm; affected walks fell into '
              f'the missing-TP skip path.')
    if not walks:
        print('No walks to analyze. Exiting.')
        return 1

    # ---- Load the walks again over the parabola-specific TP range ----
    # Reused for the global parabola page (page 2) and each walk's
    # individual parabola panel. `load_single_curve_walks` fails loudly if
    # any TP in the parabola range is absent from the CSV, which is the
    # desired behavior per the spec. The same skip-rules (missing /
    # duplicate TPs) apply independently on this range.
    print(f'Parabola TP range (1-based, inclusive): '
          f'[{args.parabola_tp_start_1based}, {args.parabola_tp_end_1based}]')
    loaded_parabola = load_single_curve_walks(
        args.input,
        args.parabola_tp_start_1based,
        args.parabola_tp_end_1based,
        skip_incomplete=skip_incomplete)
    parabola_walks = loaded_parabola['walks']
    parabola_tp_names = loaded_parabola['tp_names']
    parabola_n_vertices = loaded_parabola['n_vertices']
    print(f'Parabola TPs resolved: {parabola_tp_names[0]} .. '
          f'{parabola_tp_names[-1]} (n_vertices={parabola_n_vertices})')
    print(f'Walks usable for parabola: {len(parabola_walks)}')
    if skip_incomplete and loaded_parabola.get('skipped_empty_coord_rows', 0):
        print(f'  --skip_incomplete (parabola range): dropped '
              f'{loaded_parabola["skipped_empty_coord_rows"]} rows with '
              f'empty/non-finite coordinates.')

    # Optional: re-anchor the ordered curve at a chosen locus (e.g. Ter) by
    # circularly rolling the loci. Affects the parabola (linear offset) and the
    # distance-matrix layout for XYZ and all PC frames.
    if args.start_locus != 1:
        n_min = min(n_vertices, parabola_n_vertices)
        if not (1 <= args.start_locus <= n_min):
            raise SystemExit(
                f'--start_locus must be in [1, {n_min}]; got {args.start_locus}')
        shift = args.start_locus - 1
        walks = {w: np.roll(c, -shift, axis=0) for w, c in walks.items()}
        parabola_walks = {w: np.roll(c, -shift, axis=0)
                          for w, c in parabola_walks.items()}
        print(f'Re-anchored curves at locus {args.start_locus} (rolled by '
              f'{shift}); order is now {args.start_locus}..N, 1..{shift}.')

    # Fast path: parabolas/distance-matrices only (no summary) -> skip the
    # globule/random controls, Moran's I, and per-walk metrics entirely.
    sections = set(args.sections)
    if args.compact and 'summary' not in sections:
        _render_sections_subset(args, walks, parabola_walks,
                                parabola_n_vertices, sections)
        return 0

    # ---- Moran's I: internal by default, optional cache override ----
    cache_path = args.morans_cache_csv or args.morans_i_file
    morans_rows = []
    if cache_path:
        if args.morans_i_file and not args.morans_cache_csv:
            print('Note: --morans_i_file is deprecated; prefer '
                  '--morans_cache_csv (same semantics).')
        print(f"\nLoading cached Moran's I from {cache_path} ...")
        try:
            morans_i_dict = morans_load_cache(cache_path)
            print(f"  Loaded Moran's I for {len(morans_i_dict)} walks.")
        except Exception as exc:
            print(f'  Cache load failed ({exc}); falling back to recompute.')
            morans_i_dict = {}
            cache_path = None
    else:
        morans_i_dict = {}

    if args.skip_morans:
        print("\nSkipping Moran's I computation (--skip_morans).")
        morans_i_dict = {}
    elif not morans_i_dict:
        cycle_bins = (args.genome_cycle_bins
                      if args.genome_cycle_bins is not None else n_vertices)
        mapping = ('absolute_0based_tp_indices' if args.morans_absolute_tp
                   else 'range_relative_0_to_N-1')
        print(f"\nComputing Moran's I per walk "
              f'(cycle_bins={cycle_bins}, mapping={mapping}, '
              f'permutations={args.morans_permutations}, '
              f'seed={args.morans_seed}, '
              f'exclude_circular_genomic_offsets_leq='
              f'{args.morans_exclude_circular_offsets_leq})...')
        tp_indices_0based = loaded['tp_indices_0based']
        N = len(tp_indices_0based)
        if args.morans_absolute_tp:
            genomic_positions = np.array(tp_indices_0based, dtype=int)
        else:
            genomic_positions = np.arange(N, dtype=int)
        # Derive per-walk seeds from the master seed so results are
        # reproducible regardless of worker count or completion order.
        exclude_circ_leq = int(args.morans_exclude_circular_offsets_leq)
        if exclude_circ_leq < 0:
            raise ValueError('--morans_exclude_circular_offsets_leq must be >= 0')
        payloads = []
        for i, (wid, curve) in enumerate(walks.items()):
            per_walk_seed = (None if args.morans_seed is None
                             else int(args.morans_seed) + i)
            payloads.append((wid, curve, genomic_positions, cycle_bins,
                             args.morans_permutations, per_walk_seed,
                             args.morans_alpha, tp_indices_0based, mapping,
                             exclude_circ_leq))
        results = _map_serial_or_pool(
            _morans_worker, payloads, n_workers, "Moran's I")
        morans_rows = []
        morans_i_dict = {}
        for wid, row, by_walk_entry in results:
            morans_rows.append(row)
            morans_i_dict[wid] = by_walk_entry
        morans_out = (args.morans_output_csv
                      or os.path.join(args.outdir,
                                      f'{args.output_prefix}_morans_i_results.csv'))
        morans_write_csv(morans_rows, morans_out)
        print(f"  Wrote Moran's I results: {morans_out}")

    # ---- Per-walk metrics (parallelizable) ----
    print('\nComputing per-walk metrics...')
    flags = _flags_from_args(args)
    walk_payloads = [(wid, curve, flags) for wid, curve in walks.items()]
    walk_results = _map_serial_or_pool(
        _walk_metric_worker, walk_payloads, n_workers, 'Walk metrics')
    rows = []
    walk_metric_map = {}
    for wid, m in walk_results:
        # Fold Moran's I into the main results row so the summary CSV carries
        # it too. The dedicated morans_i_results.csv remains the canonical
        # output; this is a convenience copy.
        if wid in morans_i_dict:
            m['morans_I_value'] = float(morans_i_dict[wid]['I_value'])
            m['morans_P_value'] = float(morans_i_dict[wid]['P_value'])
            m['morans_is_significant'] = int(
                bool(morans_i_dict[wid].get('is_significant', False)))
        rows.append(m)
        walk_metric_map[wid] = m

    # ---- Controls ----
    # Convex hull from the actual walk set, using the same n_vertices as walks.
    print('\nBuilding convex hull from the experimental walk set...')
    avg_hull = compute_average_convex_hull(list(walks.values()))

    control_dists = {}
    control_curves_for_metrics = []
    use_random = (not (args.unknotted_globule or args.knotted_globule)
                  or args.show_random)
    if use_random:
        print(f'\nRandom polygon controls (n={args.n_random_controls}, '
              f'n_vertices={n_vertices})...')
        # Generate curves first (cheap, NumPy), then fan out the slow
        # calculate_knot_properties call across workers.
        rand_curves = [generate_random_polygon_in_hull(avg_hull, n_vertices)
                       for _ in range(args.n_random_controls)]
        rand_results = _map_serial_or_pool(
            _control_knot_worker, rand_curves, n_workers,
            'Random polygons (knot type)')
        control_dists['Random'] = rand_results
        control_curves_for_metrics = rand_curves

    # Globule controls: reuse a shared cache when available (they don't depend
    # on the walk set). Random control above stays per-dataset.
    globule_cache, cache_meta = load_globule_cache_csv(args.globule_cache_csv)
    if globule_cache is not None:
        want = (n_vertices, int(args.samples_per_globule), args.segment_by,
                os.path.basename(os.path.normpath(args.unknotted_globule or '')),
                os.path.basename(os.path.normpath(args.knotted_globule or '')))
        got = (cache_meta['n_vertices'], cache_meta['samples_per_globule'],
               cache_meta['segment_by'],
               os.path.basename(os.path.normpath(cache_meta['unknotted_dir'])),
               os.path.basename(os.path.normpath(cache_meta['knotted_dir'])))
        if want != got:
            raise ValueError(
                f'globule cache {args.globule_cache_csv} settings {got} do not '
                f'match current {want}; use a matching cache or a fresh path.')
        print(f'\nLoaded globule controls from cache: {args.globule_cache_csv}')

    computed = {}
    if args.unknotted_globule:
        if globule_cache and 'Unknotted Globule' in globule_cache:
            control_dists['Unknotted Globule'] = globule_cache['Unknotted Globule']
        else:
            print(f'\nUnknotted globule controls ({n_vertices}-gons)...')
            control_dists['Unknotted Globule'] = generate_globule_polygons(
                args.samples_per_globule, args.unknotted_globule,
                n_vertices, args.segment_by)
            computed['Unknotted Globule'] = control_dists['Unknotted Globule']
    if args.knotted_globule:
        if globule_cache and 'Knotted Globule' in globule_cache:
            control_dists['Knotted Globule'] = globule_cache['Knotted Globule']
        else:
            print(f'\nKnotted globule controls ({n_vertices}-gons)...')
            control_dists['Knotted Globule'] = generate_globule_polygons(
                args.samples_per_globule, args.knotted_globule,
                n_vertices, args.segment_by)
            computed['Knotted Globule'] = control_dists['Knotted Globule']

    # Persist a freshly-computed cache so later runs can reuse it.
    if args.globule_cache_csv and globule_cache is None and computed:
        write_globule_cache_csv(
            args.globule_cache_csv, computed, n_vertices,
            int(args.samples_per_globule), args.segment_by,
            args.unknotted_globule or '', args.knotted_globule or '')
        print(f'Wrote globule cache: {args.globule_cache_csv}')

    # ---- Control metric values for the ACN/planar-CN/RMS/Rg panels ----
    print('\nComputing metrics on control curves...')
    control_metric_values = compute_control_metric_values(
        control_curves_for_metrics, args, n_workers=n_workers)

    # ---- Report output paths ----
    csv_path = os.path.join(args.outdir, f'{args.output_prefix}_results.csv')
    if args.compact:
        pdf_path = os.path.join(
            args.outdir, f'{args.output_prefix}_compact_report.pdf')
        compact_csv_path = os.path.join(
            args.outdir, f'{args.output_prefix}_compact_summary.csv')
    else:
        pdf_path = os.path.join(
            args.outdir, f'{args.output_prefix}_comprehensive_report.pdf')
        compact_csv_path = None

    print(f'\nWriting PDF: {pdf_path}')

    # Pre-compute the parabola walks transformed into constrained PC space.
    # The PC basis is per-walk (each walk fits its own covariance); after
    # transform the curves live in (PC1, PC2, PC3) coordinates centered
    # on the walk's own centroid. PC2 is NOT PCA-eigenvector-2 -- it's the
    # XY rotation of PC1's projection (see `_constrained_axes_from_pc1`).
    pc_walks = {f: walks_to_constrained_pc(walks, major_axis=f)
                for f in args.pc_frames}
    pc_parabola_walks = {f: walks_to_constrained_pc(parabola_walks, major_axis=f)
                         for f in args.pc_frames}

    with PdfPages(pdf_path) as pdf:
        # ----- Page 1: population summary (shared with full mode) -----
        summary_fig = create_population_summary_page(
            rows, control_dists, control_metric_values, morans_i_dict, args)
        pdf.savefig(summary_fig)
        plt.close(summary_fig)

        # ----- Page 2: XYZ parabola QC -----
        if build_mean_parabola_xyz_figure is not None:
            fig, _, _ = build_mean_parabola_xyz_figure(
                parabola_walks,
                args.parabola_tp_start_1based,
                args.parabola_tp_end_1based,
                n_tp_in_range=parabola_n_vertices,
                axis_names=('X', 'Y', 'Z'),
                space_label='XYZ')
            pdf.savefig(fig)
            plt.close(fig)
        else:
            placeholder = plt.figure(figsize=(10, 6))
            placeholder.text(0.5, 0.5,
                             'qc_post_filter not importable; parabola page skipped',
                             ha='center', va='center', fontsize=13)
            pdf.savefig(placeholder)
            plt.close(placeholder)

        # ----- XYZ distance matrices (mean nm, not normalized) -----
        if build_mean_distance_matrices_figure is not None:
            fig, _ = build_mean_distance_matrices_figure(
                walks, args.tp_start_1based, args.tp_end_1based,
                axis_names=('X', 'Y', 'Z'),
                space_label='XYZ',
                global_colorbar=args.global_colorbar,
                axis_caps_nm=args.xyz_distance_caps_nm)
            pdf.savefig(fig)
            plt.close(fig)
        else:
            placeholder = plt.figure(figsize=(10, 6))
            placeholder.text(0.5, 0.5,
                             'qc_post_filter not importable; distance-matrix page skipped',
                             ha='center', va='center', fontsize=13)
            pdf.savefig(placeholder)
            plt.close(placeholder)

        # ----- Per PC frame: cartoon page, then PC parabola + PC matrices -----
        for frame in args.pc_frames:
            cartoon_fig = build_pc_frame_cartoon_figure(frame, walks)
            pdf.savefig(cartoon_fig)
            plt.close(cartoon_fig)
            label = f'PC space ({frame})'
            if build_mean_parabola_xyz_figure is not None:
                fig, _, _ = build_mean_parabola_xyz_figure(
                    pc_parabola_walks[frame],
                    args.parabola_tp_start_1based,
                    args.parabola_tp_end_1based,
                    n_tp_in_range=parabola_n_vertices,
                    axis_names=('PC1', 'PC2', 'PC3'),
                    space_label=label)
                pdf.savefig(fig)
                plt.close(fig)
            if build_mean_distance_matrices_figure is not None:
                fig, _ = build_mean_distance_matrices_figure(
                    pc_walks[frame], args.tp_start_1based, args.tp_end_1based,
                    axis_names=('PC1', 'PC2', 'PC3'),
                    space_label=label,
                    global_colorbar=args.global_colorbar,
                    axis_caps_nm=args.pc_distance_caps_nm)
                pdf.savefig(fig)
                plt.close(fig)

        # ----- Pages 6+: per-walk report cards (full mode only) -----
        if not args.compact:
            for wid, curve in tqdm(walks.items(), desc='Report cards'):
                parabola_curve = parabola_walks.get(wid)
                fig = create_walk_report_card(
                    wid, curve, walk_metric_map[wid],
                    control_metric_values, control_dists, morans_i_dict,
                    tp_names, args,
                    parabola_curve=parabola_curve,
                    parabola_tp_start_1based=args.parabola_tp_start_1based,
                    parabola_tp_end_1based=args.parabola_tp_end_1based)
                pdf.savefig(fig)
                plt.close(fig)

    # ---- Side-car PDFs: 3D-only mean parabola in XYZ and PC space -----
    # These are written next to the main report so they can be embedded in
    # talks / paper figures without cropping the 4-panel layout. Same
    # underlying data, support filter, weighting, and fit as the 3D panel
    # of the global parabola page -- numbers are identical.
    if build_mean_parabola_3d_only_figure is not None:
        sidecars = [('XYZ', 'xyz', parabola_walks)]
        for frame in args.pc_frames:
            sidecars.append((f'PC space ({frame})', f'pc_{frame}',
                             pc_parabola_walks[frame]))
        for space_label, tag, slice_walks in sidecars:
            side_path = os.path.join(
                args.outdir, f'{args.output_prefix}_parabola_3d_{tag}.pdf')
            fig_side, _, _ = build_mean_parabola_3d_only_figure(
                slice_walks,
                args.parabola_tp_start_1based,
                args.parabola_tp_end_1based,
                n_tp_in_range=parabola_n_vertices,
                space_label=space_label)
            fig_side.savefig(side_path)
            plt.close(fig_side)
            print(f'Wrote {side_path}')

    if args.compact:
        write_compact_summary_csv(rows, compact_csv_path)
        print(f'Done. Compact PDF: {pdf_path}')
        print(f'Compact CSV:    {compact_csv_path}')
    else:
        write_results_csv(rows, control_dists, csv_path)
        print(f'Done. PDF: {pdf_path}')
        print(f'Summary CSV: {csv_path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
