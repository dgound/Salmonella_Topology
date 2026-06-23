#!/usr/bin/env python3
"""Plot a projected polygon with explicit over/under crossing annotations.

The plot intentionally does not use the classic knot-diagram convention of
breaking the under-passing strand. That convention becomes ambiguous for dense
or clustered polygons. Instead this script keeps every edge continuous, marks
each crossing with an indexed callout, locally highlights the over/under edge,
and writes an authoritative crossing table.

Input options:
  - final drift-corrected pipeline CSV with WalkID/tp/x_nm/y_nm/z_nm columns
  - .xyz text file with one x y z row per vertex
  - .npy file containing an (N, 3) array
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
from dataclasses import dataclass, replace
from typing import Iterable, Optional

import numpy as np

os.environ.setdefault(
    'MPLCONFIGDIR',
    os.path.join(os.environ.get('TMPDIR', '/tmp'), 'matplotlib'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.backends.backend_pdf import PdfPages


CUSTOM_20_COLORS_HEX = [
    '#ED2024', '#F05023', '#F8991D', '#FFE500', '#CADB2B',
    '#94C83D', '#72BF44', '#6ABD45', '#71BF4C', '#6FC59B',
    '#6FCCDD', '#43ABE1', '#4068B2', '#3B54A5', '#4652A4',
    '#6B52A2', '#97509F', '#C44D9C', '#ED2590', '#EE1C51',
]

TP_RE = re.compile(r'^TP(\d+)(?:_CYC(\d+))?$')


@dataclass(frozen=True)
class ProjectionSpec:
    name: str
    u: np.ndarray
    v: np.ndarray
    depth: np.ndarray
    sign_normal: np.ndarray
    xlabel: str
    ylabel: str
    depth_label: str
    score: Optional[float] = None
    min_vertex_distance: Optional[float] = None
    p05_vertex_distance: Optional[float] = None
    crossing_count: Optional[int] = None
    min_crossing_distance: Optional[float] = None
    p05_crossing_distance: Optional[float] = None
    min_crossing_vertex_distance: Optional[float] = None
    p05_crossing_vertex_distance: Optional[float] = None


@dataclass(frozen=True)
class Crossing:
    cid: int
    edge_a: int
    edge_b: int
    a_next: int
    b_next: int
    t_a: float
    t_b: float
    xy: np.ndarray
    depth_a: float
    depth_b: float
    over_edge: int
    under_edge: int
    over_next: int
    under_next: int
    depth_delta: float
    sign: int


PROJECTIONS = {
    'xy': ProjectionSpec(
        'xy',
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        np.array([0.0, 0.0, 1.0]),
        'X (nm)', 'Y (nm)', 'Z depth (nm)'),
    'xz': ProjectionSpec(
        'xz',
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, -1.0, 0.0]),
        'X (nm)', 'Z (nm)', 'Y depth (nm)'),
    'yz': ProjectionSpec(
        'yz',
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        'Y (nm)', 'Z (nm)', 'X depth (nm)'),
}


def _tp_index(tp_name: str) -> int:
    match = TP_RE.match(str(tp_name))
    if match is None:
        raise ValueError(f'Unrecognized TP name: {tp_name!r}')
    return int(match.group(1))


def _load_xyz(path: str) -> tuple[np.ndarray, list[str]]:
    points = []
    with open(path, newline='') as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text or text.startswith('#'):
                continue
            parts = text.replace(',', ' ').split()
            if len(parts) < 3:
                raise ValueError(f'{path}:{line_no}: expected at least 3 columns')
            points.append([float(parts[0]), float(parts[1]), float(parts[2])])
    curve = np.asarray(points, dtype=float)
    if curve.ndim != 2 or curve.shape[1] != 3 or curve.shape[0] < 3:
        raise ValueError(f'{path}: expected at least 3 xyz points')
    labels = [str(i + 1) for i in range(curve.shape[0])]
    return curve, labels


def _load_npy(path: str) -> tuple[np.ndarray, list[str]]:
    curve = np.asarray(np.load(path), dtype=float)
    if curve.ndim != 2 or curve.shape[1] != 3 or curve.shape[0] < 3:
        raise ValueError(f'{path}: expected an (N, 3) array with N >= 3')
    labels = [str(i + 1) for i in range(curve.shape[0])]
    return curve, labels


def _read_pipeline_csv(path: str,
                       walk_id: int,
                       location: Optional[str],
                       tp_start_1based: int,
                       tp_end_1based: int) -> tuple[np.ndarray, list[str]]:
    if tp_start_1based < 1:
        raise ValueError('--tp_start_1based must be >= 1')
    if tp_end_1based < tp_start_1based:
        raise ValueError('--tp_end_1based must be >= --tp_start_1based')

    required_tp = list(range(tp_start_1based - 1, tp_end_1based))
    required_set = set(required_tp)
    rows_by_tp: dict[int, list[dict[str, str]]] = {tp: [] for tp in required_tp}
    locations_seen = set()

    with open(path, newline='') as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        required = {'WalkID', 'tp', 'x_nm', 'y_nm', 'z_nm'}
        missing = required - fields
        if missing:
            raise KeyError(f'{path}: missing columns {sorted(missing)}')
        has_location = 'Location' in fields

        for row in reader:
            try:
                row_wid = int(row['WalkID'])
            except Exception:
                continue
            if row_wid != walk_id:
                continue
            row_location = row.get('Location', '')
            if has_location:
                locations_seen.add(row_location)
            if location is not None and row_location != location:
                continue
            tp0 = _tp_index(row['tp'])
            if tp0 in required_set:
                rows_by_tp[tp0].append(row)

    if location is None and len(locations_seen) > 1:
        choices = ', '.join(sorted(locations_seen))
        raise ValueError(
            f'WalkID {walk_id} appears in multiple locations: {choices}. '
            'Pass --location to choose one.')

    points = []
    labels = []
    for tp0 in required_tp:
        rows = rows_by_tp[tp0]
        if not rows:
            raise ValueError(
                f'No row for WalkID {walk_id}, TP {tp0} '
                f'(1-based {tp0 + 1}) in {path}')
        if len(rows) > 1:
            raise ValueError(
                f'Multiple rows for WalkID {walk_id}, TP {tp0} '
                f'(1-based {tp0 + 1}); add --location or inspect duplicates.')
        row = rows[0]
        points.append([float(row['x_nm']), float(row['y_nm']), float(row['z_nm'])])
        labels.append(row['tp'])

    return np.asarray(points, dtype=float), labels


def load_curve(args: argparse.Namespace) -> tuple[np.ndarray, list[str]]:
    lower = args.input.lower()
    if lower.endswith('.npy'):
        return _load_npy(args.input)
    if lower.endswith('.xyz') or lower.endswith('.txt'):
        return _load_xyz(args.input)
    if args.walk_id is None:
        raise ValueError('CSV input requires --walk_id')
    return _read_pipeline_csv(
        args.input, args.walk_id, args.location,
        args.tp_start_1based, args.tp_end_1based)


def _resample_colors(n: int) -> list:
    if n == len(CUSTOM_20_COLORS_HEX):
        return CUSTOM_20_COLORS_HEX
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        'timepoint_custom', CUSTOM_20_COLORS_HEX, N=max(256, n))
    return [cmap(i / max(n - 1, 1)) for i in range(n)]


def _normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    norm = float(np.linalg.norm(v))
    if norm <= 0:
        raise ValueError('Cannot normalize a zero vector')
    return v / norm


def _orthonormal_basis_from_depth(depth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    depth = _normalize(depth)
    ref = np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(ref, depth))) > 0.88:
        ref = np.array([0.0, 1.0, 0.0])
    u = ref - float(np.dot(ref, depth)) * depth
    u = _normalize(u)
    v = np.cross(depth, u)
    v = _normalize(v)
    return u, v


def project_curve(curve: np.ndarray, projection: ProjectionSpec) -> np.ndarray:
    return np.column_stack((curve @ projection.u, curve @ projection.v))


def depth_values(curve: np.ndarray, projection: ProjectionSpec) -> np.ndarray:
    return curve @ projection.depth


def _fibonacci_sphere(n: int) -> np.ndarray:
    idx = np.arange(n, dtype=float)
    z = 1.0 - 2.0 * (idx + 0.5) / float(n)
    theta = idx * (math.pi * (3.0 - math.sqrt(5.0)))
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack((r * np.cos(theta), r * np.sin(theta), z))


def _pair_vectors(curve: np.ndarray,
                  closed: bool,
                  pair_mode: str) -> np.ndarray:
    n = curve.shape[0]
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            if pair_mode == 'nonadjacent':
                if abs(i - j) == 1:
                    continue
                if closed and {i, j} == {0, n - 1}:
                    continue
            pairs.append(curve[j] - curve[i])
    if not pairs:
        raise ValueError('No vertex pairs available for projection scoring')
    return np.asarray(pairs, dtype=float)


def _projection_score(distances: np.ndarray, metric: str) -> np.ndarray:
    if metric == 'min':
        return np.min(distances, axis=0)
    if metric == 'p05':
        return np.percentile(distances, 5, axis=0)
    if metric == 'p10':
        return np.percentile(distances, 10, axis=0)
    if metric == 'median':
        return np.median(distances, axis=0)
    raise ValueError(f'Unknown projection metric: {metric}')


def _score_directions(pair_vecs: np.ndarray,
                      directions: np.ndarray,
                      metric: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pair_len2 = np.sum(pair_vecs * pair_vecs, axis=1)
    dots = pair_vecs @ directions.T
    projected2 = np.maximum(pair_len2[:, None] - dots * dots, 0.0)
    projected_dist = np.sqrt(projected2)
    scores = _projection_score(projected_dist, metric)
    min_d = np.min(projected_dist, axis=0)
    p05_d = np.percentile(projected_dist, 5, axis=0)
    return scores, min_d, p05_d


def _crossing_distance_score(crossings: list[Crossing],
                             metric: str) -> tuple[float, float, float]:
    """Return (score, min_distance, p05_distance) for crossing point spacing."""
    if len(crossings) < 2:
        return -float('inf'), 0.0, 0.0
    pts = np.vstack([c.xy for c in crossings])
    diffs = pts[:, None, :] - pts[None, :, :]
    distances = np.sqrt(np.sum(diffs * diffs, axis=2))
    tri = distances[np.triu_indices(len(crossings), k=1)]
    min_d = float(np.min(tri))
    p05_d = float(np.percentile(tri, 5))
    if metric == 'min':
        score = min_d
    elif metric == 'p05':
        score = p05_d
    elif metric == 'p10':
        score = float(np.percentile(tri, 10))
    elif metric == 'median':
        score = float(np.median(tri))
    else:
        raise ValueError(f'Unknown projection metric: {metric}')
    return score, min_d, p05_d


def _metric_value(values: np.ndarray, metric: str) -> float:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return 0.0
    if metric == 'min':
        return float(np.min(values))
    if metric == 'p05':
        return float(np.percentile(values, 5))
    if metric == 'p10':
        return float(np.percentile(values, 10))
    if metric == 'median':
        return float(np.median(values))
    raise ValueError(f'Unknown projection metric: {metric}')


def _point_segment_dist_matrix(points: np.ndarray,
                               seg_a: np.ndarray,
                               seg_b: np.ndarray) -> np.ndarray:
    """Vectorised distance matrix from each point in ``points`` (P, 2) to
    each line segment with endpoints (``seg_a``, ``seg_b``), shape (S, 2).

    Returns an (P, S) array. Points on a segment yield distance 0.
    """
    if points.size == 0 or seg_a.size == 0:
        return np.zeros((points.shape[0], seg_a.shape[0]), dtype=float)
    ab = (seg_b - seg_a)[None, :, :]                        # (1, S, 2)
    ap = points[:, None, :] - seg_a[None, :, :]             # (P, S, 2)
    ab_len2 = np.sum(ab * ab, axis=-1)                      # (1, S)
    safe_ab_len2 = np.where(ab_len2 > 0, ab_len2, 1.0)
    t = np.sum(ap * ab, axis=-1) / safe_ab_len2             # (P, S)
    t = np.clip(t, 0.0, 1.0)
    closest = seg_a[None, :, :] + t[..., None] * ab         # (P, S, 2)
    diff = points[:, None, :] - closest
    return np.sqrt(np.sum(diff * diff, axis=-1))


def _diagram_min_separation(projected: np.ndarray,
                            crossings: list[Crossing],
                            closed: bool) -> tuple[float, dict]:
    """Minimum separation across every legitimate element-pair in the 2D
    diagram. Used as the primary "readability" metric for the projection
    optimizer. Returns (overall_min_separation, per_category_breakdown).

    Pair categories and exclusions (rows are categories, exclusions in
    parentheses):

        vertex-vertex    (adjacent vertices — they are an edge by design)
        vertex-edge      (vertex's two incident edges)
        edge-edge        (adjacent edges, edges that legitimately cross)
        crossing-vertex  (none)
        crossing-edge    (the crossing's two incident edges)
        crossing-crossing(none, but excludes self)

    A pair category with no valid pairs contributes ``+inf`` (does not
    pull the overall minimum down).
    """
    n = projected.shape[0]
    n_edges = n if closed else max(0, n - 1)
    if n_edges < 1:
        return float('inf'), {}
    edges = [(i, (i + 1) % n if closed else i + 1) for i in range(n_edges)]
    seg_a = np.asarray([projected[a] for a, _ in edges], dtype=float)
    seg_b = np.asarray([projected[b] for _, b in edges], dtype=float)

    breakdown: dict[str, float] = {}

    # 1. vertex-vertex, excluding adjacent (and self).
    vv_diffs = projected[:, None, :] - projected[None, :, :]
    vv = np.sqrt(np.sum(vv_diffs * vv_diffs, axis=-1))
    vv_mask = np.ones_like(vv, dtype=bool)
    np.fill_diagonal(vv_mask, False)
    for a, b in edges:
        vv_mask[a, b] = False
        vv_mask[b, a] = False
    breakdown['vertex_vertex'] = float(vv[vv_mask].min()) if vv_mask.any() else float('inf')

    # 2. vertex-edge, excluding the vertex's own incident edges.
    ve = _point_segment_dist_matrix(projected, seg_a, seg_b)
    ve_mask = np.ones_like(ve, dtype=bool)
    for ei, (a, b) in enumerate(edges):
        ve_mask[a, ei] = False
        ve_mask[b, ei] = False
    breakdown['vertex_edge'] = float(ve[ve_mask].min()) if ve_mask.any() else float('inf')

    # 3. edge-edge, excluding adjacent edges and edges that cross.
    crossing_pairs: set[tuple[int, int]] = set()
    for c in crossings:
        ea, eb = int(c.edge_a), int(c.edge_b)
        if ea > eb:
            ea, eb = eb, ea
        crossing_pairs.add((ea, eb))

    ee_min = float('inf')
    for i in range(n_edges):
        for j in range(i + 1, n_edges):
            if _edges_adjacent(i, j, n, closed):
                continue
            if (i, j) in crossing_pairs:
                continue
            # Minimum endpoint-to-other-segment distance over the four ends.
            cand = min(
                ve[edges[i][0], j], ve[edges[i][1], j],
                ve[edges[j][0], i], ve[edges[j][1], i])
            if cand < ee_min:
                ee_min = cand
    breakdown['edge_edge'] = float(ee_min)

    # 4. crossing-vertex (all pairs)
    if crossings:
        cross_pts = np.asarray([c.xy for c in crossings], dtype=float)
        cv = np.sqrt(np.sum(
            (cross_pts[:, None, :] - projected[None, :, :]) ** 2, axis=-1))
        breakdown['crossing_vertex'] = float(cv.min())
    else:
        cross_pts = np.zeros((0, 2), dtype=float)
        breakdown['crossing_vertex'] = float('inf')

    # 5. crossing-edge, excluding the crossing's incident edges.
    if crossings:
        ce = _point_segment_dist_matrix(cross_pts, seg_a, seg_b)
        ce_mask = np.ones_like(ce, dtype=bool)
        for ci, c in enumerate(crossings):
            ce_mask[ci, int(c.over_edge)] = False
            ce_mask[ci, int(c.under_edge)] = False
        breakdown['crossing_edge'] = float(ce[ce_mask].min()) if ce_mask.any() else float('inf')
    else:
        breakdown['crossing_edge'] = float('inf')

    # 6. crossing-crossing, excluding self.
    if len(crossings) >= 2:
        cc = np.sqrt(np.sum(
            (cross_pts[:, None, :] - cross_pts[None, :, :]) ** 2, axis=-1))
        cc_mask = np.ones_like(cc, dtype=bool)
        np.fill_diagonal(cc_mask, False)
        breakdown['crossing_crossing'] = float(cc[cc_mask].min())
    else:
        breakdown['crossing_crossing'] = float('inf')

    overall = min(v for v in breakdown.values() if np.isfinite(v))
    if not np.isfinite(overall):
        overall = float('inf')
    return float(overall), breakdown


def _crossing_clearance_score(crossings: list[Crossing],
                              projected_vertices: np.ndarray,
                              metric: str) -> tuple[float, float, float, float, float]:
    """Score clearance around crossing points.

    Returns:
      composite_score, cc_min, cc_p05, cv_min, cv_p05

    The composite score is the smaller of the crossing-crossing metric and
    crossing-vertex metric. With fewer than two crossings, crossing-crossing
    spacing is not defined and the crossing-vertex metric becomes the score.
    """
    if not crossings:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    crossing_pts = np.vstack([c.xy for c in crossings])

    cv_diffs = crossing_pts[:, None, :] - projected_vertices[None, :, :]
    cv_dist = np.sqrt(np.sum(cv_diffs * cv_diffs, axis=2)).ravel()
    cv_min = float(np.min(cv_dist))
    cv_p05 = float(np.percentile(cv_dist, 5))
    cv_score = _metric_value(cv_dist, metric)

    if len(crossings) < 2:
        return cv_score, 0.0, 0.0, cv_min, cv_p05

    cc_diffs = crossing_pts[:, None, :] - crossing_pts[None, :, :]
    cc_dist = np.sqrt(np.sum(cc_diffs * cc_diffs, axis=2))
    cc_tri = cc_dist[np.triu_indices(len(crossings), k=1)]
    cc_min = float(np.min(cc_tri))
    cc_p05 = float(np.percentile(cc_tri, 5))
    cc_score = _metric_value(cc_tri, metric)
    return min(cc_score, cv_score), cc_min, cc_p05, cv_min, cv_p05


def _projection_from_depth(name: str,
                           depth: np.ndarray,
                           score: Optional[float] = None,
                           min_vertex_distance: Optional[float] = None,
                           p05_vertex_distance: Optional[float] = None,
                           crossing_count: Optional[int] = None,
                           min_crossing_distance: Optional[float] = None,
                           p05_crossing_distance: Optional[float] = None,
                           min_crossing_vertex_distance: Optional[float] = None,
                           p05_crossing_vertex_distance: Optional[float] = None) -> ProjectionSpec:
    depth = _normalize(depth)
    u, v = _orthonormal_basis_from_depth(depth)
    return ProjectionSpec(
        name=name,
        u=u,
        v=v,
        depth=depth,
        sign_normal=depth,
        xlabel='projection U (nm)',
        ylabel='projection V (nm)',
        depth_label='projection depth (nm)',
        score=score,
        min_vertex_distance=min_vertex_distance,
        p05_vertex_distance=p05_vertex_distance,
        crossing_count=crossing_count,
        min_crossing_distance=min_crossing_distance,
        p05_crossing_distance=p05_crossing_distance,
        min_crossing_vertex_distance=min_crossing_vertex_distance,
        p05_crossing_vertex_distance=p05_crossing_vertex_distance,
    )


def _score_one_crossing_projection(curve: np.ndarray,
                                   depth: np.ndarray,
                                   closed: bool,
                                   over_depth: str,
                                   metric: str,
                                   min_crossings: int) -> tuple[float, int, float, float, float, float]:
    projection = _projection_from_depth('candidate', depth)
    projected = project_curve(curve, projection)
    depths = depth_values(curve, projection)
    crossings = find_crossings(curve, projected, depths, projection,
                               closed, over_depth)
    if len(crossings) < min_crossings:
        return -float('inf'), len(crossings), 0.0, 0.0, 0.0, 0.0
    score, cc_min, cc_p05, cv_min, cv_p05 = _crossing_clearance_score(
        crossings, projected, metric)
    return score, len(crossings), cc_min, cc_p05, cv_min, cv_p05


def _score_one_min_crossing_projection(curve: np.ndarray,
                                       depth: np.ndarray,
                                       closed: bool,
                                       over_depth: str,
                                       metric: str,
                                       min_crossings: int) -> tuple[float, int, float, float, float, float, float]:
    projection = _projection_from_depth('candidate', depth)
    projected = project_curve(curve, projection)
    depths = depth_values(curve, projection)
    crossings = find_crossings(curve, projected, depths, projection,
                               closed, over_depth)
    n_cross = len(crossings)
    if n_cross < min_crossings:
        return -float('inf'), n_cross, 0.0, 0.0, 0.0, 0.0, 0.0
    score, cc_min, cc_p05, cv_min, cv_p05 = _crossing_clearance_score(
        crossings, projected, metric)
    overall_min, _ = _diagram_min_separation(projected, crossings, closed)
    return score, n_cross, cc_min, cc_p05, cv_min, cv_p05, overall_min


def _optimize_vertex_projection(curve: np.ndarray,
                                closed: bool,
                                metric: str,
                                pair_mode: str,
                                n_samples: int,
                                refine: int,
                                seed: int) -> ProjectionSpec:
    pair_vecs = _pair_vectors(curve, closed, pair_mode)
    directions = _fibonacci_sphere(max(int(n_samples), 64))
    scores, min_d, p05_d = _score_directions(pair_vecs, directions, metric)
    best_idx = int(np.lexsort((min_d, scores))[-1])
    best_dir = directions[best_idx]
    best_score = float(scores[best_idx])
    best_min = float(min_d[best_idx])
    best_p05 = float(p05_d[best_idx])

    rng = np.random.default_rng(seed)
    n_refine = max(int(refine), 0)
    for sigma in (0.28, 0.12, 0.045):
        if n_refine <= 0:
            break
        local = best_dir + rng.normal(0.0, sigma, size=(n_refine, 3))
        local /= np.linalg.norm(local, axis=1)[:, None]
        scores, min_d, p05_d = _score_directions(pair_vecs, local, metric)
        idx = int(np.lexsort((min_d, scores))[-1])
        if float(scores[idx]) > best_score:
            best_dir = local[idx]
            best_score = float(scores[idx])
            best_min = float(min_d[idx])
            best_p05 = float(p05_d[idx])

    u, v = _orthonormal_basis_from_depth(best_dir)
    return ProjectionSpec(
        name='auto_vertices',
        u=u,
        v=v,
        depth=_normalize(best_dir),
        sign_normal=_normalize(best_dir),
        xlabel='projection U (nm)',
        ylabel='projection V (nm)',
        depth_label='projection depth (nm)',
        score=best_score,
        min_vertex_distance=best_min,
        p05_vertex_distance=best_p05,
    )


def _optimize_crossing_projection(curve: np.ndarray,
                                  closed: bool,
                                  metric: str,
                                  pair_mode: str,
                                  n_samples: int,
                                  refine: int,
                                  seed: int,
                                  over_depth: str,
                                  min_crossings: int) -> ProjectionSpec:
    pair_vecs = _pair_vectors(curve, closed, pair_mode)
    directions = _fibonacci_sphere(max(int(n_samples), 64))
    vertex_scores, vertex_min, vertex_p05 = _score_directions(
        pair_vecs, directions, 'p05')

    best_dir = None
    best_tuple = (-float('inf'), -float('inf'), -float('inf'),
                  -float('inf'), -float('inf'))
    best_crossing_count = 0
    best_crossing_min = 0.0
    best_crossing_p05 = 0.0
    best_crossing_vertex_min = 0.0
    best_crossing_vertex_p05 = 0.0
    best_vertex_min = 0.0
    best_vertex_p05 = 0.0

    for i, direction in enumerate(directions):
        (score, n_cross, min_cross_d, p05_cross_d,
         min_cv_d, p05_cv_d) = _score_one_crossing_projection(
             curve, direction, closed, over_depth, metric, min_crossings)
        # Primary: crossing/vertex clearance. Tie-breakers: the two component
        # clearances, then projected vertex separation.
        # Final tie-breaker: fewer crossings, because fewer symbols are easier
        # to read if the spacing is otherwise equivalent.
        rank = (score, p05_cross_d, p05_cv_d, float(vertex_scores[i]),
                -float(n_cross))
        if rank > best_tuple:
            best_tuple = rank
            best_dir = direction
            best_crossing_count = n_cross
            best_crossing_min = min_cross_d
            best_crossing_p05 = p05_cross_d
            best_crossing_vertex_min = min_cv_d
            best_crossing_vertex_p05 = p05_cv_d
            best_vertex_min = float(vertex_min[i])
            best_vertex_p05 = float(vertex_p05[i])

    if best_dir is None or not np.isfinite(best_tuple[0]):
        raise ValueError(
            f'No sampled projection had at least {min_crossings} crossings. '
            'Try lowering --min_crossings or increasing --projection_samples.')

    rng = np.random.default_rng(seed)
    n_refine = max(int(refine), 0)
    for sigma in (0.28, 0.12, 0.045):
        if n_refine <= 0:
            break
        local = best_dir + rng.normal(0.0, sigma, size=(n_refine, 3))
        local /= np.linalg.norm(local, axis=1)[:, None]
        vertex_scores, vertex_min, vertex_p05 = _score_directions(
            pair_vecs, local, 'p05')
        for i, direction in enumerate(local):
            (score, n_cross, min_cross_d, p05_cross_d,
             min_cv_d, p05_cv_d) = _score_one_crossing_projection(
                 curve, direction, closed, over_depth, metric, min_crossings)
            rank = (score, p05_cross_d, p05_cv_d, float(vertex_scores[i]),
                    -float(n_cross))
            if rank > best_tuple:
                best_tuple = rank
                best_dir = direction
                best_crossing_count = n_cross
                best_crossing_min = min_cross_d
                best_crossing_p05 = p05_cross_d
                best_crossing_vertex_min = min_cv_d
                best_crossing_vertex_p05 = p05_cv_d
                best_vertex_min = float(vertex_min[i])
                best_vertex_p05 = float(vertex_p05[i])

    return _projection_from_depth(
        'auto_crossings', best_dir,
        score=float(best_tuple[0]),
        min_vertex_distance=best_vertex_min,
        p05_vertex_distance=best_vertex_p05,
        crossing_count=best_crossing_count,
        min_crossing_distance=best_crossing_min,
        p05_crossing_distance=best_crossing_p05,
        min_crossing_vertex_distance=best_crossing_vertex_min,
        p05_crossing_vertex_distance=best_crossing_vertex_p05)


def _optimize_min_crossing_projection(curve: np.ndarray,
                                      closed: bool,
                                      metric: str,
                                      pair_mode: str,
                                      n_samples: int,
                                      refine: int,
                                      seed: int,
                                      over_depth: str,
                                      min_crossings: int) -> ProjectionSpec:
    pair_vecs = _pair_vectors(curve, closed, pair_mode)
    directions = _fibonacci_sphere(max(int(n_samples), 64))
    vertex_scores, vertex_min, vertex_p05 = _score_directions(
        pair_vecs, directions, 'p05')

    best_dir = None
    # 7-tuple lex rank, biggest is best:
    #   (overall_min_separation,          # PRIMARY: min across ALL pair categories
    #    -crossing_count,                 # SECONDARY (fewer = better)
    #    min_crossing_vertex_distance,
    #    p05_crossing_vertex_distance,
    #    min_crossing_crossing_distance,
    #    p05_crossing_crossing_distance,
    #    vertex_pair_score)
    best_tuple = (-float('inf'), -float('inf'), -float('inf'),
                  -float('inf'), -float('inf'), -float('inf'),
                  -float('inf'))
    best_crossing_count = 0
    best_crossing_min = 0.0
    best_crossing_p05 = 0.0
    best_crossing_vertex_min = 0.0
    best_crossing_vertex_p05 = 0.0
    best_vertex_min = 0.0
    best_vertex_p05 = 0.0

    for i, direction in enumerate(directions):
        (sep_score, n_cross, min_cross_d, p05_cross_d,
         min_cv_d, p05_cv_d, overall_min) = (
            _score_one_min_crossing_projection(
                curve, direction, closed, over_depth, metric, min_crossings))
        if not np.isfinite(sep_score):
            continue
        # Primary: max the smallest pairwise distance among ALL element
        # pairs in the 2D diagram — vertices, edges, crossings — see
        # _diagram_min_separation for the exact categories. Secondary:
        # fewer crossings (lex tiebreak). Remaining tiers are the legacy
        # diagnostic clearances; they only matter when both primary and
        # secondary tie.
        rank = (overall_min, -float(n_cross),
                min_cv_d, p05_cv_d, min_cross_d, p05_cross_d,
                float(vertex_scores[i]))
        if rank > best_tuple:
            best_tuple = rank
            best_dir = direction
            best_crossing_count = n_cross
            best_crossing_min = min_cross_d
            best_crossing_p05 = p05_cross_d
            best_crossing_vertex_min = min_cv_d
            best_crossing_vertex_p05 = p05_cv_d
            best_vertex_min = float(vertex_min[i])
            best_vertex_p05 = float(vertex_p05[i])

    if best_dir is None:
        raise ValueError(
            f'No sampled projection had at least {min_crossings} crossings. '
            'Try lowering --min_crossings or increasing --projection_samples.')

    rng = np.random.default_rng(seed)
    n_refine = max(int(refine), 0)
    for sigma in (0.28, 0.12, 0.045):
        if n_refine <= 0:
            break
        local = best_dir + rng.normal(0.0, sigma, size=(n_refine, 3))
        local /= np.linalg.norm(local, axis=1)[:, None]
        vertex_scores, vertex_min, vertex_p05 = _score_directions(
            pair_vecs, local, 'p05')
        for i, direction in enumerate(local):
            (sep_score, n_cross, min_cross_d, p05_cross_d,
             min_cv_d, p05_cv_d, overall_min) = (
                _score_one_min_crossing_projection(
                    curve, direction, closed, over_depth, metric, min_crossings))
            if not np.isfinite(sep_score):
                continue
            rank = (overall_min, -float(n_cross),
                    min_cv_d, p05_cv_d, min_cross_d, p05_cross_d,
                    float(vertex_scores[i]))
            if rank > best_tuple:
                best_tuple = rank
                best_dir = direction
                best_crossing_count = n_cross
                best_crossing_min = min_cross_d
                best_crossing_p05 = p05_cross_d
                best_crossing_vertex_min = min_cv_d
                best_crossing_vertex_p05 = p05_cv_d
                best_vertex_min = float(vertex_min[i])
                best_vertex_p05 = float(vertex_p05[i])

    return _projection_from_depth(
        'auto_max_all_pair_clearance', best_dir,
        score=float(best_tuple[0]),
        min_vertex_distance=best_vertex_min,
        p05_vertex_distance=best_vertex_p05,
        crossing_count=best_crossing_count,
        min_crossing_distance=best_crossing_min,
        p05_crossing_distance=best_crossing_p05,
        min_crossing_vertex_distance=best_crossing_vertex_min,
        p05_crossing_vertex_distance=best_crossing_vertex_p05)


def optimize_projection(curve: np.ndarray,
                        closed: bool,
                        objective: str,
                        metric: str,
                        pair_mode: str,
                        n_samples: int,
                        refine: int,
                        seed: int,
                        over_depth: str,
                        min_crossings: int) -> ProjectionSpec:
    if objective == 'vertices':
        return _optimize_vertex_projection(
            curve, closed, metric, pair_mode, n_samples, refine, seed)
    if objective == 'crossings':
        return _optimize_crossing_projection(
            curve, closed, metric, pair_mode, n_samples, refine, seed,
            over_depth, min_crossings)
    if objective == 'min_crossings':
        return _optimize_min_crossing_projection(
            curve, closed, metric, pair_mode, n_samples, refine, seed,
            over_depth, min_crossings)
    raise ValueError(f'Unknown projection objective: {objective}')


def _edge_pairs(n_vertices: int, closed: bool) -> list[tuple[int, int]]:
    n_edges = n_vertices if closed else n_vertices - 1
    return [(i, (i + 1) % n_vertices) for i in range(n_edges)]


def _edges_adjacent(e1: int, e2: int, n_vertices: int, closed: bool) -> bool:
    if abs(e1 - e2) == 1:
        return True
    if closed and {e1, e2} == {0, n_vertices - 1}:
        return True
    return False


def _cross2(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def _segment_intersection_params(p: np.ndarray,
                                 p2: np.ndarray,
                                 q: np.ndarray,
                                 q2: np.ndarray,
                                 eps: float) -> Optional[tuple[float, float]]:
    r = p2 - p
    s = q2 - q
    denom = _cross2(r, s)
    if abs(denom) <= eps:
        return None
    qp = q - p
    t = _cross2(qp, s) / denom
    u = _cross2(qp, r) / denom
    if eps < t < 1.0 - eps and eps < u < 1.0 - eps:
        return t, u
    return None


def find_crossings(curve: np.ndarray,
                   projected: np.ndarray,
                   depth: np.ndarray,
                   projection: ProjectionSpec,
                   closed: bool,
                   over_depth: str,
                   eps: float = 1e-10) -> list[Crossing]:
    n_vertices = curve.shape[0]
    edges = _edge_pairs(n_vertices, closed)
    crossings: list[Crossing] = []
    cid = 1

    # This sign follows the oriented normal of the plotted u/v plane. The
    # over/under identity itself comes from the interpolated depth values.
    for edge_a, a_next in edges:
        for edge_b, b_next in edges:
            if edge_b <= edge_a:
                continue
            if _edges_adjacent(edge_a, edge_b, n_vertices, closed):
                continue
            params = _segment_intersection_params(
                projected[edge_a], projected[a_next],
                projected[edge_b], projected[b_next], eps)
            if params is None:
                continue
            t_a, t_b = params
            xy = projected[edge_a] + t_a * (projected[a_next] - projected[edge_a])
            depth_a = float(depth[edge_a] + t_a * (depth[a_next] - depth[edge_a]))
            depth_b = float(depth[edge_b] + t_b * (depth[b_next] - depth[edge_b]))

            a_is_over = depth_a >= depth_b
            if over_depth == 'low':
                a_is_over = not a_is_over
            if a_is_over:
                over_edge, under_edge = edge_a, edge_b
                over_next, under_next = a_next, b_next
            else:
                over_edge, under_edge = edge_b, edge_a
                over_next, under_next = b_next, a_next

            tangent_a = curve[a_next] - curve[edge_a]
            tangent_b = curve[b_next] - curve[edge_b]
            sign_value = float(
                np.dot(np.cross(tangent_a, tangent_b), projection.sign_normal))
            if not a_is_over:
                sign_value *= -1.0
            sign = 1 if sign_value > 0 else -1 if sign_value < 0 else 0

            crossings.append(Crossing(
                cid=cid,
                edge_a=edge_a,
                edge_b=edge_b,
                a_next=a_next,
                b_next=b_next,
                t_a=float(t_a),
                t_b=float(t_b),
                xy=xy,
                depth_a=depth_a,
                depth_b=depth_b,
                over_edge=over_edge,
                under_edge=under_edge,
                over_next=over_next,
                under_next=under_next,
                depth_delta=abs(depth_a - depth_b),
                sign=sign,
            ))
            cid += 1
    return crossings


def _first_pass_state(crossing: Crossing) -> str:
    """Return whether the first traversal encounter of this crossing is over.

    The walk starts at vertex 1/0 and follows increasing edge index. Because
    crossing search stores non-adjacent pairs with edge_a < edge_b, edge_a is
    the first time this crossing is encountered along that walk.
    """
    return 'over' if crossing.edge_a == crossing.over_edge else 'under'


def _renumber_crossings_by_walk(crossings: list[Crossing]) -> list[Crossing]:
    """Number crossings by first encounter while walking the polygon."""
    ordered = sorted(crossings, key=lambda c: (c.edge_a, c.t_a))
    return [replace(c, cid=i + 1) for i, c in enumerate(ordered)]


def _letter_label(i: int) -> str:
    """0 -> 'a', 1 -> 'b', ..., 25 -> 'z', 26 -> 'aa', 27 -> 'ab', ...

    Used for crossing labels so they read distinctly from the integer
    vertex labels in a polygon diagram.
    """
    if i < 0:
        raise ValueError(f'_letter_label index must be >= 0, got {i}')
    s = ''
    n = i
    while True:
        s = chr(ord('a') + (n % 26)) + s
        n = n // 26 - 1
        if n < 0:
            break
    return s


def _walk_ordering(n_vertices: int,
                   crossings: list[Crossing],
                   closed: bool,
                   projected: Optional[np.ndarray] = None,
                   ) -> tuple[list[int], dict[int, str]]:
    """Walk the polygon starting at vertex 0 in the **counter-clockwise**
    direction relative to the projection plane, numbering vertices and
    crossings as we meet them.

    Vertices receive integer labels (1, 2, 3, ...) and crossings receive
    lowercase letter labels (``a``, ``b``, ``c``, ..., ``z``, ``aa``,
    ``ab``, ...). The two label families are aligned by walk order but
    use different namespaces so a diagram can be read at a glance.

    Walk direction
    --------------
    If ``projected`` is supplied, we compute the signed area of the
    projected closed polygon and walk in the direction that yields CCW
    in those axes. If ``projected`` is omitted, or the polygon is open,
    we default to the natural vertex-index direction (V0 -> V1 -> ...).

    Crossings are encountered twice; only the first encounter assigns the
    label. The total label range is ``n_vertices`` integers plus
    ``len(crossings)`` letters.
    """
    # Decide the walk direction.
    reverse = False
    if closed and projected is not None and projected.shape[0] >= 3:
        x = projected[:, 0]
        y = projected[:, 1]
        signed_area = 0.5 * float(np.sum(
            x * np.roll(y, -1) - np.roll(x, -1) * y))
        # In standard math axes (y increases upward) CCW corresponds to a
        # positive signed area. If the natural walk is CW, reverse.
        reverse = signed_area < 0

    # Build the sequence of vertices visited along the walk.
    if reverse:
        if closed:
            walk_vertices = [0] + list(range(n_vertices - 1, 0, -1))
        else:
            walk_vertices = list(range(n_vertices - 1, -1, -1))
    else:
        walk_vertices = list(range(n_vertices))

    n_edges = n_vertices if closed else max(0, n_vertices - 1)
    walk_edges = [
        (walk_vertices[i],
         walk_vertices[(i + 1) % n_vertices] if closed else walk_vertices[i + 1])
        for i in range(n_edges)
    ]
    # Convenience: map physical edge (u, v) -> walk index.
    walk_edge_index: dict[tuple[int, int], int] = {}
    for we_idx, (u, v) in enumerate(walk_edges):
        walk_edge_index[(u, v)] = we_idx

    vertex_walk_label = [0] * n_vertices
    for i, v in enumerate(walk_vertices):
        vertex_walk_label[v] = i + 1

    # For each crossing, find its first-encounter position along the walk.
    # Each crossing has two natural edges (edge_a, a_next) and
    # (edge_b, b_next). Each natural edge corresponds to either the
    # forward or reversed walk edge; t along the walk edge is t_a / t_b
    # if forward, 1 - t_* if walking the edge in reverse.
    first_visit: dict[int, tuple[int, float]] = {}
    for c in crossings:
        for nat_u, nat_v, nat_t in (
                (int(c.edge_a), int(c.a_next), float(c.t_a)),
                (int(c.edge_b), int(c.b_next), float(c.t_b))):
            if (nat_u, nat_v) in walk_edge_index:
                we_idx = walk_edge_index[(nat_u, nat_v)]
                t_walk = nat_t
            elif (nat_v, nat_u) in walk_edge_index:
                we_idx = walk_edge_index[(nat_v, nat_u)]
                t_walk = 1.0 - nat_t
            else:
                continue
            cand = (we_idx, t_walk)
            cur = first_visit.get(int(c.cid))
            if cur is None or cand < cur:
                first_visit[int(c.cid)] = cand

    ordered_cids = sorted(first_visit, key=lambda cid: first_visit[cid])
    crossing_walk_label = {cid: _letter_label(i)
                           for i, cid in enumerate(ordered_cids)}

    return vertex_walk_label, crossing_walk_label


# Okabe-Ito colorblind-friendly palette. Blue + vermilion are distinguishable
# under protanopia, deuteranopia and tritanopia. These two replace the
# previous blue/purple convention used for "first pass over/under" diamonds.
_CB_FIRST_OVER  = '#0072B2'   # blue
_CB_FIRST_UNDER = '#D55E00'   # vermilion


def _crossing_dot_color(crossing: Crossing) -> str:
    """Color convention for numbered dots (colorblind-friendly).

    Blue (#0072B2) = first time the walk reaches this crossing it passes OVER.
    Vermilion (#D55E00) = first time the walk reaches this crossing it passes UNDER.
    Okabe-Ito palette; distinguishable for all common color-vision deficiencies.
    """
    return _CB_FIRST_OVER if _first_pass_state(crossing) == 'over' else _CB_FIRST_UNDER


def _axis_span(points: np.ndarray) -> float:
    span = np.ptp(points, axis=0)
    finite = span[np.isfinite(span)]
    if finite.size == 0:
        return 1.0
    value = float(max(finite.max(), 1.0))
    return value


def _short_edge_segment(projected: np.ndarray,
                        edge: int,
                        next_idx: int,
                        t: float,
                        half_len: float) -> np.ndarray:
    p0 = projected[edge]
    p1 = projected[next_idx]
    vec = p1 - p0
    length = float(np.linalg.norm(vec))
    if length <= 0:
        center = p0
        return np.vstack([center, center])
    frac = min(max(half_len / length, 0.01), 0.20)
    t0 = max(0.0, t - frac)
    t1 = min(1.0, t + frac)
    return np.vstack([p0 + t0 * vec, p0 + t1 * vec])


def _place_crossing_labels(crossings: list[Crossing],
                           projected: np.ndarray,
                           span: float) -> dict[int, np.ndarray]:
    """Return non-overlapping-ish label positions for crossing callouts.

    This lightweight placement avoids an extra dependency such as adjustText.
    It is intentionally conservative: labels may move away from the exact
    crossing, but leader lines keep the mapping explicit and the table remains
    authoritative.
    """
    if not crossings:
        return {}

    points_to_avoid = [np.asarray(p, dtype=float) for p in projected]
    points_to_avoid.extend(np.asarray(c.xy, dtype=float) for c in crossings)
    occupied: list[np.ndarray] = []
    positions: dict[int, np.ndarray] = {}
    base = 0.018 * span
    min_sep = 0.030 * span
    vertex_sep = 0.022 * span
    angles = np.deg2rad([25, 75, 125, 165, 205, 255, 305, 345])
    radii = [1.0, 1.7, 2.5, 3.4, 4.4]

    x_min, y_min = np.min(projected, axis=0)
    x_max, y_max = np.max(projected, axis=0)
    pad = 0.09 * span

    for crossing in crossings:
        best = None
        best_penalty = float('inf')
        origin = crossing.xy
        for radius in radii:
            for angle in angles:
                candidate = origin + base * radius * np.array(
                    [math.cos(angle), math.sin(angle)])
                penalty = 0.0

                # Softly keep labels near the plotted polygon area.
                if candidate[0] < x_min - pad or candidate[0] > x_max + pad:
                    penalty += 20.0
                if candidate[1] < y_min - pad or candidate[1] > y_max + pad:
                    penalty += 20.0

                for old in occupied:
                    dist = float(np.linalg.norm(candidate - old))
                    if dist < min_sep:
                        penalty += ((min_sep - dist) / min_sep) ** 2 * 30.0
                for avoid in points_to_avoid:
                    dist = float(np.linalg.norm(candidate - avoid))
                    if dist < vertex_sep:
                        penalty += ((vertex_sep - dist) / vertex_sep) ** 2 * 8.0
                # Prefer shorter leader lines when penalties are otherwise tied.
                penalty += 0.08 * radius

                if penalty < best_penalty:
                    best_penalty = penalty
                    best = candidate

        if best is None:
            best = origin + np.array([base, base])
        positions[crossing.cid] = best
        occupied.append(best)

    return positions


def _format_edge(edge: int, next_idx: int, one_based: bool) -> str:
    offset = 1 if one_based else 0
    return f'E{edge + offset} ({edge + offset}->{next_idx + offset})'


def write_crossing_csv(path: str,
                       crossings: Iterable[Crossing],
                       one_based: bool) -> None:
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', newline='') as handle:
        fieldnames = [
            'crossing_id',
            'edge_a', 'edge_b',
            'edge_a_vertices', 'edge_b_vertices',
            'first_pass',
            'over_edge', 'under_edge',
            'over_vertices', 'under_vertices',
            'projection_x', 'projection_y',
            'edge_a_depth', 'edge_b_depth',
            'depth_delta', 'sign',
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for c in crossings:
            offset = 1 if one_based else 0
            writer.writerow({
                'crossing_id': c.cid,
                'edge_a': c.edge_a + offset,
                'edge_b': c.edge_b + offset,
                'edge_a_vertices': f'{c.edge_a + offset}->{c.a_next + offset}',
                'edge_b_vertices': f'{c.edge_b + offset}->{c.b_next + offset}',
                'first_pass': _first_pass_state(c),
                'over_edge': c.over_edge + offset,
                'under_edge': c.under_edge + offset,
                'over_vertices': f'{c.over_edge + offset}->{c.over_next + offset}',
                'under_vertices': f'{c.under_edge + offset}->{c.under_next + offset}',
                'projection_x': f'{c.xy[0]:.6g}',
                'projection_y': f'{c.xy[1]:.6g}',
                'edge_a_depth': f'{c.depth_a:.6g}',
                'edge_b_depth': f'{c.depth_b:.6g}',
                'depth_delta': f'{c.depth_delta:.6g}',
                'sign': c.sign,
            })


def _draw_crossing_table(ax, crossings: list[Crossing], one_based: bool,
                         crossing_labels: dict[int, int] | None = None,
                         vertex_labels: list[int] | None = None) -> None:
    ax.axis('off')
    if not crossings:
        ax.text(0.0, 1.0, 'No projected crossings', va='top', ha='left',
                fontsize=11, fontweight='bold')
        return

    def _vlabel(v: int) -> str:
        if vertex_labels is not None:
            idx = vertex_labels[v] if one_based else vertex_labels[v] - 1
            return str(idx)
        return str(v + (1 if one_based else 0))

    def _edge_str(e: int, nxt: int) -> str:
        return f'{_vlabel(e)}->{_vlabel(nxt)}'

    max_rows = 15
    rows = []
    for c in crossings[:max_rows]:
        cl = (crossing_labels.get(int(c.cid), c.cid)
              if crossing_labels else c.cid)
        rows.append([str(cl), _first_pass_state(c),
                     _edge_str(c.over_edge, c.over_next),
                     _edge_str(c.under_edge, c.under_next),
                     f'{c.depth_delta:.3g}'])
    if len(crossings) > max_rows:
        rows.append(['...', f'+{len(crossings) - max_rows} more', '', '', ''])

    table = ax.table(
        cellText=rows,
        colLabels=['#', 'first', 'over', 'under', 'depth gap'],
        loc='upper left',
        cellLoc='left',
        colLoc='left',
        colWidths=[0.08, 0.14, 0.31, 0.31, 0.16],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.2 if len(rows) > 12 else 8.0)
    table.scale(1.0, 1.05)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor('0.82')
        if row == 0:
            cell.set_facecolor('0.92')
            cell.set_text_props(weight='bold')
    ax.set_title('Crossing table', loc='left', fontsize=11, fontweight='bold')


def _draw_crossing_map(ax,
                       crossings: list[Crossing],
                       projected: np.ndarray,
                       segments: np.ndarray,
                       xlim: tuple[float, float],
                       ylim: tuple[float, float],
                       crossing_labels: dict[int, int] | None = None,
                       label_crossings: bool = True) -> None:
    ax.set_title('Crossing Map', loc='left', fontsize=11, fontweight='bold')
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, color='0.88', linewidth=0.7)
    ax.tick_params(axis='both', labelsize=7)
    ax.set_xlabel('same U as polygon', fontsize=8)
    ax.set_ylabel('same V as polygon', fontsize=8)

    lc = LineCollection(segments, colors='0.82', linewidths=0.9, alpha=0.8,
                        zorder=1)
    ax.add_collection(lc)
    ax.scatter(projected[:, 0], projected[:, 1], s=12, facecolor='0.72',
               edgecolor='none', zorder=2)

    if not crossings:
        ax.text(0.5, 0.5, 'No projected crossings',
                transform=ax.transAxes, ha='center', va='center',
                fontsize=10)
        return

    for c in crossings:
        color = _crossing_dot_color(c)
        ax.scatter([c.xy[0]], [c.xy[1]], s=170, marker='D',
                   facecolor=color, edgecolor='black', linewidth=0.8,
                   zorder=3)
        if not label_crossings:
            continue
        label = (crossing_labels.get(int(c.cid), c.cid)
                 if crossing_labels else c.cid)
        ax.text(c.xy[0], c.xy[1], str(label),
                ha='center', va='center', color='white',
                fontsize=7, fontweight='bold', zorder=4)

    ax.text(0.02, -0.18,
            'Same projection/limits as polygon.\n'
            'Diamond: blue=first pass over; purple=first pass under.',
            transform=ax.transAxes, ha='left', va='top', fontsize=7.5)


def build_synced_map_figure(curve: np.ndarray,
                            projection: ProjectionSpec,
                            crossings: list[Crossing],
                            closed: bool,
                            title: str,
                            one_based: bool,
                            label_vertices: bool,
                            label_edges: bool,
                            crossing_overlay: str,
                            annotate_local: bool,
                            label_crossings: bool = True):
    # If diamond labels are explicitly suppressed, downgrade an explicit
    # 'labels' overlay to 'points' so the auxiliary label boxes don't draw.
    if not label_crossings and crossing_overlay == 'labels':
        crossing_overlay = 'points'
    projected = project_curve(curve, projection)
    colors = _resample_colors(curve.shape[0])
    fig = plt.figure(figsize=(14.5 if crossings else 9.0, 8.8),
                     constrained_layout=True)
    gs = fig.add_gridspec(
        2, 2,
        width_ratios=[3.8, 1.9 if crossings else 0.8],
        height_ratios=[1.0, 1.05],
        hspace=0.32,
        wspace=0.16)
    ax = fig.add_subplot(gs[:, 0])
    ax_crossing_map = fig.add_subplot(gs[0, 1])
    ax_table = fig.add_subplot(gs[1, 1])

    edges = _edge_pairs(curve.shape[0], closed)
    segments = np.asarray([[projected[i], projected[j]] for i, j in edges])
    lc = LineCollection(segments, colors='0.12', linewidths=1.6, alpha=0.9,
                        zorder=1)
    ax.add_collection(lc)

    # Optional local crossing overlays. The default is deliberately minimal:
    # the synced crossing map and table carry the over-under information, so dense geometry does
    # not have to carry dense labels.
    span = _axis_span(projected)
    marker_half_len = 0.018 * span
    if crossing_overlay in ('strokes', 'labels'):
        for c in crossings:
            over_t = c.t_a if c.over_edge == c.edge_a else c.t_b
            under_t = c.t_a if c.under_edge == c.edge_a else c.t_b
            over_seg = _short_edge_segment(
                projected, c.over_edge, c.over_next, over_t, marker_half_len)
            under_seg = _short_edge_segment(
                projected, c.under_edge, c.under_next, under_t, marker_half_len)
            ax.plot(under_seg[:, 0], under_seg[:, 1], color=_CB_FIRST_UNDER,
                    linewidth=4.0, linestyle=(0, (2, 2)),
                    solid_capstyle='round', alpha=0.95, zorder=3)
            ax.plot(over_seg[:, 0], over_seg[:, 1], color=_CB_FIRST_OVER,
                    linewidth=3.0, solid_capstyle='round', alpha=0.95, zorder=4)

    ax.scatter(projected[:, 0], projected[:, 1], c=colors, s=52,
               edgecolor='black', linewidths=0.8, zorder=5)

    vertex_labels, crossing_labels = _walk_ordering(
        curve.shape[0], crossings, closed, projected=projected)

    if label_vertices:
        offset = 0.007 * span
        for i, (x, y) in enumerate(projected):
            label = str(vertex_labels[i] if one_based else vertex_labels[i] - 1)
            ax.text(x + offset, y + offset, label, fontsize=7,
                    ha='left', va='bottom', zorder=7,
                    bbox=dict(boxstyle='round,pad=0.12',
                              facecolor='white', edgecolor='0.65',
                              alpha=0.82, linewidth=0.4))

    if label_edges:
        for edge, next_idx in edges:
            mid = 0.5 * (projected[edge] + projected[next_idx])
            ax.text(mid[0], mid[1], f'E{edge + (1 if one_based else 0)}',
                    fontsize=6.5, color='0.15', ha='center', va='center',
                    zorder=6,
                    bbox=dict(boxstyle='round,pad=0.10',
                              facecolor='white', edgecolor='none',
                              alpha=0.65))

    local_tag_offset = 0.028 * span
    if crossing_overlay in ('points', 'strokes', 'labels'):
        crossing_label_positions = (
            _place_crossing_labels(crossings, projected, span)
            if crossing_overlay == 'labels' else {})
        for c in crossings:
            x, y = c.xy
            ax.scatter([x], [y], s=34, marker='D', facecolor='white',
                       edgecolor='black', linewidths=0.9, zorder=8)
            if crossing_overlay == 'labels':
                label_pos = crossing_label_positions[c.cid]
                ax.plot([x, label_pos[0]], [y, label_pos[1]],
                        color='0.25', linewidth=0.7, alpha=0.75, zorder=7)
                walk_label = crossing_labels.get(int(c.cid), c.cid)
                ax.text(label_pos[0], label_pos[1], str(walk_label),
                        ha='center', va='center', fontsize=7.5,
                        fontweight='bold', zorder=9,
                        bbox=dict(boxstyle='round,pad=0.16',
                                  facecolor='white', edgecolor='black',
                                  alpha=0.96, linewidth=0.7))
            if annotate_local:
                over_tangent = projected[c.over_next] - projected[c.over_edge]
                under_tangent = projected[c.under_next] - projected[c.under_edge]
                for label, tangent, color, sign in (
                        ('O', over_tangent, _CB_FIRST_OVER, -1.0),
                        ('U', under_tangent, _CB_FIRST_UNDER, 1.0)):
                    norm = float(np.linalg.norm(tangent))
                    if norm <= 0:
                        direction = np.array([1.0, 0.0])
                    else:
                        direction = tangent / norm
                    pos = c.xy + sign * local_tag_offset * direction
                    ax.text(pos[0], pos[1], label, color='white', fontsize=7,
                            ha='center', va='center', fontweight='bold',
                            zorder=10,
                            bbox=dict(boxstyle='circle,pad=0.15',
                                      facecolor=color, edgecolor='white',
                                      linewidth=0.5, alpha=0.95))

    ax.set_aspect('equal', adjustable='datalim')
    ax.autoscale()
    xpad = 0.06 * max(float(np.ptp(projected[:, 0])), 1.0)
    ypad = 0.06 * max(float(np.ptp(projected[:, 1])), 1.0)
    ax.set_xlim(float(projected[:, 0].min()) - xpad,
                float(projected[:, 0].max()) + xpad)
    ax.set_ylim(float(projected[:, 1].min()) - ypad,
                float(projected[:, 1].max()) + ypad)
    ax.set_xlabel(projection.xlabel)
    ax.set_ylabel(projection.ylabel)
    ax.set_title(title, loc='left', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.18)
    if crossing_overlay in ('strokes', 'labels'):
        note = ('Blue local stroke = over, vermilion = under. '
                'Diamond color: blue first pass over, vermilion first pass under '
                '(Okabe-Ito colorblind-friendly). '
                'Vertices = numbers, crossings = letters; walk CCW from vertex 1.')
    elif crossing_overlay == 'points':
        note = ('Crossing positions are marked. Diamond color: blue first pass over, '
                'vermilion first pass under (Okabe-Ito colorblind-friendly). '
                'Vertices = numbers, crossings = letters; walk CCW from vertex 1.')
    else:
        note = 'Over/under is encoded in the map/table'
    ax.text(0.01, 0.01, note, transform=ax.transAxes,
            ha='left', va='bottom', fontsize=8,
            bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                      edgecolor='0.7', alpha=0.86))

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    _draw_crossing_map(ax_crossing_map, crossings, projected, segments,
                       xlim, ylim, crossing_labels=crossing_labels,
                       label_crossings=label_crossings)
    _draw_crossing_table(ax_table, crossings, one_based,
                         crossing_labels=crossing_labels,
                         vertex_labels=vertex_labels)
    return fig


def plot_diagram(curve: np.ndarray,
                 projection: ProjectionSpec,
                 crossings: list[Crossing],
                 output: str,
                 closed: bool,
                 title: str,
                 one_based: bool,
                 label_vertices: bool,
                 label_edges: bool,
                 crossing_overlay: str,
                 annotate_local: bool,
                 dpi: int) -> None:
    fig = build_synced_map_figure(
        curve, projection, crossings, closed, title, one_based,
        label_vertices, label_edges, crossing_overlay, annotate_local)
    os.makedirs(os.path.dirname(output) or '.', exist_ok=True)
    fig.savefig(output, dpi=dpi)
    plt.close(fig)


def _draw_projected_polygon(ax,
                            projected: np.ndarray,
                            edges: list[tuple[int, int]],
                            colors: list,
                            linewidth: float = 1.6,
                            alpha: float = 0.9) -> np.ndarray:
    segments = np.asarray([[projected[i], projected[j]] for i, j in edges])
    lc = LineCollection(segments, colors='0.12', linewidths=linewidth,
                        alpha=alpha, zorder=1)
    ax.add_collection(lc)
    ax.scatter(projected[:, 0], projected[:, 1], c=colors, s=52,
               edgecolor='black', linewidths=0.8, zorder=5)
    return segments


def _apply_projection_limits(ax, projected: np.ndarray) -> tuple[tuple[float, float],
                                                                 tuple[float, float]]:
    ax.set_aspect('equal', adjustable='datalim')
    ax.autoscale()
    xpad = 0.06 * max(float(np.ptp(projected[:, 0])), 1.0)
    ypad = 0.06 * max(float(np.ptp(projected[:, 1])), 1.0)
    xlim = (float(projected[:, 0].min()) - xpad,
            float(projected[:, 0].max()) + xpad)
    ylim = (float(projected[:, 1].min()) - ypad,
            float(projected[:, 1].max()) + ypad)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    return xlim, ylim


def _draw_numbered_crossing_dots(ax, crossings: list[Crossing],
                                 size: float = 170,
                                 crossing_labels: dict[int, int] | None = None,
                                 label_crossings: bool = True) -> None:
    for c in crossings:
        color = _crossing_dot_color(c)
        ax.scatter([c.xy[0]], [c.xy[1]], s=size, marker='D',
                   facecolor=color, edgecolor='black', linewidth=0.8,
                   zorder=8)
        if not label_crossings:
            continue
        label = (crossing_labels.get(int(c.cid), c.cid)
                 if crossing_labels else c.cid)
        ax.text(c.xy[0], c.xy[1], str(label),
                ha='center', va='center', color='white',
                fontsize=7, fontweight='bold', zorder=9)


def build_on_polygon_dots_figure(curve: np.ndarray,
                                 projection: ProjectionSpec,
                                 crossings: list[Crossing],
                                 closed: bool,
                                 title: str,
                                 one_based: bool,
                                 label_vertices: bool = True,
                                 label_crossings: bool = True):
    projected = project_curve(curve, projection)
    colors = _resample_colors(curve.shape[0])
    edges = _edge_pairs(curve.shape[0], closed)
    vertex_labels, crossing_labels = _walk_ordering(
        curve.shape[0], crossings, closed, projected=projected)

    fig = plt.figure(figsize=(14.5, 8.8), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[3.8, 1.45], wspace=0.16)
    ax = fig.add_subplot(gs[0, 0])
    ax_table = fig.add_subplot(gs[0, 1])

    _draw_projected_polygon(ax, projected, edges, colors)
    _draw_numbered_crossing_dots(ax, crossings, size=190,
                                 crossing_labels=crossing_labels,
                                 label_crossings=label_crossings)
    span = _axis_span(projected)
    offset = 0.007 * span
    if label_vertices:
        for i, (x, y) in enumerate(projected):
            ax.text(x + offset, y + offset,
                    str(vertex_labels[i] if one_based else vertex_labels[i] - 1),
                    fontsize=7, ha='left', va='bottom', zorder=7,
                    bbox=dict(boxstyle='round,pad=0.12',
                              facecolor='white', edgecolor='0.65',
                              alpha=0.82, linewidth=0.4))
    _apply_projection_limits(ax, projected)
    ax.set_xlabel(projection.xlabel)
    ax.set_ylabel(projection.ylabel)
    ax.set_title(title, loc='left', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.18)
    ax.text(0.01, 0.01,
            'Diamond color: blue=first pass over, vermilion=first pass under '
            '(Okabe-Ito CB-friendly). Vertices = numbers, crossings = letters; '
            'walk CCW from vertex 1; table gives both edges.',
            transform=ax.transAxes, ha='left', va='bottom', fontsize=8,
            bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                      edgecolor='0.7', alpha=0.86))

    _draw_crossing_table(ax_table, crossings, one_based,
                         crossing_labels=crossing_labels,
                         vertex_labels=vertex_labels)
    return fig


def _merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not intervals:
        return []
    intervals = sorted((max(0.0, a), min(1.0, b)) for a, b in intervals)
    merged = [intervals[0]]
    for lo, hi in intervals[1:]:
        prev_lo, prev_hi = merged[-1]
        if lo <= prev_hi:
            merged[-1] = (prev_lo, max(prev_hi, hi))
        else:
            merged.append((lo, hi))
    return merged


def _draw_broken_underpass_edges(ax,
                                 projected: np.ndarray,
                                 edges: list[tuple[int, int]],
                                 crossings: list[Crossing],
                                 span: float) -> None:
    under_by_edge: dict[int, list[float]] = {}
    for c in crossings:
        under_t = c.t_a if c.under_edge == c.edge_a else c.t_b
        under_by_edge.setdefault(c.under_edge, []).append(float(under_t))

    gap_len = 0.020 * span
    for edge, next_idx in edges:
        p0 = projected[edge]
        p1 = projected[next_idx]
        vec = p1 - p0
        length = float(np.linalg.norm(vec))
        if length <= 0:
            continue
        intervals = []
        for t in under_by_edge.get(edge, []):
            half_gap_t = min(max(gap_len / length, 0.012), 0.12)
            intervals.append((t - half_gap_t, t + half_gap_t))
        intervals = _merge_intervals(intervals)
        start = 0.0
        for lo, hi in intervals:
            if lo > start:
                seg = np.vstack([p0 + start * vec, p0 + lo * vec])
                ax.plot(seg[:, 0], seg[:, 1], color='0.12',
                        linewidth=1.8, solid_capstyle='round', zorder=2)
            start = max(start, hi)
        if start < 1.0:
            seg = np.vstack([p0 + start * vec, p1])
            ax.plot(seg[:, 0], seg[:, 1], color='0.12',
                    linewidth=1.8, solid_capstyle='round', zorder=2)


def build_broken_underpass_figure(curve: np.ndarray,
                                  projection: ProjectionSpec,
                                  crossings: list[Crossing],
                                  closed: bool,
                                  title: str,
                                  one_based: bool,
                                  label_vertices: bool = True,
                                  draw_table: bool = True,
                                  bare: bool = False):
    projected = project_curve(curve, projection)
    colors = _resample_colors(curve.shape[0])
    edges = _edge_pairs(curve.shape[0], closed)
    span = _axis_span(projected)
    vertex_labels, crossing_labels = _walk_ordering(
        curve.shape[0], crossings, closed, projected=projected)

    if bare:
        fig = plt.figure(figsize=(8.5, 8.5))
        ax = fig.add_subplot(1, 1, 1)
        ax_table = None
    else:
        fig = plt.figure(figsize=(14.5, 8.8), constrained_layout=True)
        if draw_table:
            gs = fig.add_gridspec(1, 2, width_ratios=[3.8, 1.45], wspace=0.16)
            ax = fig.add_subplot(gs[0, 0])
            ax_table = fig.add_subplot(gs[0, 1])
        else:
            ax = fig.add_subplot(1, 1, 1)
            ax_table = None

    _draw_broken_underpass_edges(ax, projected, edges, crossings, span)
    ax.scatter(projected[:, 0], projected[:, 1], c=colors, s=52,
               edgecolor='black', linewidths=0.8, zorder=5)
    offset = 0.007 * span
    if label_vertices:
        for i, (x, y) in enumerate(projected):
            ax.text(x + offset, y + offset,
                    str(vertex_labels[i] if one_based else vertex_labels[i] - 1),
                    fontsize=7, ha='left', va='bottom', zorder=7,
                    bbox=dict(boxstyle='round,pad=0.12',
                              facecolor='white', edgecolor='0.65',
                              alpha=0.82, linewidth=0.4))
    _apply_projection_limits(ax, projected)
    if bare:
        ax.set_axis_off()
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    else:
        ax.set_xlabel(projection.xlabel)
        ax.set_ylabel(projection.ylabel)
        ax.set_title(title, loc='left', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.18)
        ax.text(0.01, 0.01,
                'Traditional diagram: under-passing edge is broken at each crossing. '
                'Vertices = numbers, crossings = letters; walk CCW from vertex 1.',
                transform=ax.transAxes, ha='left', va='bottom', fontsize=8,
                bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                          edgecolor='0.7', alpha=0.86))

    if ax_table is not None:
        _draw_crossing_table(ax_table, crossings, one_based,
                             crossing_labels=crossing_labels,
                             vertex_labels=vertex_labels)
    return fig


def write_three_page_report_pdf(output_pdf: str,
                                curve: np.ndarray,
                                projection: ProjectionSpec,
                                crossings: list[Crossing],
                                closed: bool,
                                base_title: str,
                                one_based: bool,
                                label_vertices: bool,
                                label_edges: bool,
                                crossing_overlay: str,
                                annotate_local: bool,
                                label_crossings: bool = True) -> None:
    os.makedirs(os.path.dirname(output_pdf) or '.', exist_ok=True)
    with PdfPages(output_pdf) as pdf:
        fig = build_broken_underpass_figure(
            curve, projection, crossings, closed,
            f'{base_title} - broken underpasses',
            one_based, label_vertices=label_vertices)
        pdf.savefig(fig)
        plt.close(fig)

        fig = build_on_polygon_dots_figure(
            curve, projection, crossings, closed,
            f'{base_title} - crossings on polygon',
            one_based, label_vertices=label_vertices,
            label_crossings=label_crossings)
        pdf.savefig(fig)
        plt.close(fig)

        fig = build_synced_map_figure(
            curve, projection, crossings, closed,
            f'{base_title} - synced crossing map',
            one_based, label_vertices, label_edges,
            crossing_overlay, annotate_local,
            label_crossings=label_crossings)
        pdf.savefig(fig)
        plt.close(fig)


def plot_polygon_only(curve: np.ndarray,
                      projection: ProjectionSpec,
                      crossings: list[Crossing],
                      output: str,
                      closed: bool,
                      title: str,
                      one_based: bool,
                      label_vertices: bool,
                      dpi: int,
                      draw_table: bool = False) -> None:
    """Render the first ("polygon") page only: the broken-underpass diagram.

    Crossing diamonds in this view are colored but not labelled (over/under
    is read off the broken-edge style and the diamond color), so the
    --no_vertex_numbers / label_crossings interaction is moot here.
    """
    fig = build_broken_underpass_figure(
        curve, projection, crossings, closed, title, one_based,
        label_vertices=label_vertices, draw_table=draw_table, bare=True)
    os.makedirs(os.path.dirname(output) or '.', exist_ok=True)
    fig.savefig(output, dpi=dpi)
    plt.close(fig)


def _default_crossing_csv(output: str) -> str:
    base, _ = os.path.splitext(output)
    return f'{base}_crossings.csv'


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Plot a polygon with explicit over/under crossing calls.')
    parser.add_argument('--input', required=True,
                        help='Pipeline CSV, .xyz, .txt, or .npy curve file.')
    parser.add_argument('--output', required=True,
                        help='Output figure path (.png, .pdf, .svg, etc.).')
    parser.add_argument('--report_pdf', default=None,
                        help='Optional 3-page PDF report path. Pages: '
                             'traditional broken-underpass diagram, crossing '
                             'diamonds on polygon, and synced crossing map.')
    parser.add_argument('--crossing_csv', default=None,
                        help='Output crossing table CSV. Default: '
                             '<output stem>_crossings.csv.')
    parser.add_argument('--walk_id', type=int, default=None,
                        help='WalkID to load when --input is a pipeline CSV.')
    parser.add_argument('--walk_ids', default=None,
                        help='Comma-separated list of WalkIDs to process in '
                             'one invocation (e.g. "135,258,3295"). Overrides '
                             '--walk_id. Per-walk --output / --report_pdf / '
                             '--crossing_csv paths are auto-suffixed with '
                             '"_walk<id>" so files do not collide.')
    parser.add_argument('--location', default=None,
                        help='Location to load when a merged CSV has repeated '
                             'WalkID values, e.g. "Location 1".')
    parser.add_argument('--tp_start_1based', type=int, default=3,
                        help='First TP to include for pipeline CSV input. '
                             'Default 3.')
    parser.add_argument('--tp_end_1based', type=int, default=22,
                        help='Last TP to include for pipeline CSV input. '
                             'Default 22.')
    projection_choices = ['auto'] + sorted(PROJECTIONS)
    parser.add_argument('--projection', choices=projection_choices,
                        default='auto',
                        help='2D projection plane. Default auto searches for '
                             'a projection with few crossings and readable '
                             'crossing spacing.')
    parser.add_argument('--projection_objective',
                        choices=['min_crossings', 'crossings', 'vertices'],
                        default='min_crossings',
                        help='Objective for --projection auto. min_crossings '
                             'first minimizes crossing count, then maximizes '
                             'crossing spacing. crossings only maximizes '
                             'crossing spacing. vertices maximizes projected '
                             'vertex separation. Default min_crossings.')
    parser.add_argument('--projection_metric',
                        choices=['min', 'p05', 'p10', 'median'],
                        default='min',
                        help='Spacing metric inside the selected auto '
                             'objective. For min_crossings/crossings this is '
                             'computed on crossing-to-crossing distances. '
                             'For vertices it is computed on projected vertex '
                             'distances. Default min.')
    parser.add_argument('--projection_pairs',
                        choices=['all', 'nonadjacent'],
                        default='all',
                        help='Vertex pairs used by --projection auto. '
                             'Default all.')
    parser.add_argument('--projection_samples', type=int, default=24000,
                        help='Sphere samples for --projection auto. '
                             'Default 24000.')
    parser.add_argument('--projection_refine', type=int, default=3500,
                        help='Random local refinements per refinement round '
                             'for --projection auto. Default 3500.')
    parser.add_argument('--projection_seed', type=int, default=1,
                        help='Random seed for projection refinement. Default 1.')
    parser.add_argument('--min_crossings', type=int, default=0,
                        help='Minimum crossing count allowed for auto '
                             'projection. Default 0. Increase this if you '
                             'want a projection that preserves at least some '
                             'crossing information even when a zero-crossing '
                             'projection exists.')
    parser.add_argument('--over_depth', choices=['high', 'low'], default='high',
                        help='Which depth value is considered over-passing. '
                             'Default high.')
    parser.add_argument('--open', action='store_true',
                        help='Treat the curve as open. Default is closed.')
    parser.add_argument('--zero_based_edges', action='store_true',
                        help='Label vertices/edges as zero-based. Default is '
                             'one-based for human-readable figures.')
    parser.add_argument('--label_vertices', action='store_true',
                        help='Draw vertex number labels.')
    parser.add_argument('--label_edges', action='store_true',
                        help='Draw edge number labels at edge midpoints.')
    parser.add_argument('--crossing_overlay',
                        choices=['none', 'points', 'strokes', 'labels'],
                        default='points',
                        help='How much crossing information to draw directly '
                             'on the geometry. Default points; over/under is '
                             'encoded in the synced map/table.')
    parser.add_argument('--local_ou_tags', action='store_true',
                        help='Draw small local O/U tags near crossings. Off '
                             'by default because dense polygons are clearer '
                             'with crossing numbers plus the side table.')
    parser.add_argument('--polygon_only', action='store_true',
                        help='Render only the first page of the report '
                             '(broken-underpass polygon diagram). Single-axis '
                             'view, no side table, no on-polygon-dots page, '
                             'no synced-map page.')
    parser.add_argument('--no_vertex_numbers', action='store_true',
                        help='Suppress vertex number labels. When --polygon_only '
                             'is NOT set, also suppresses the letter labels '
                             'inside the crossing diamonds.')
    parser.add_argument('--title', default=None,
                        help='Custom figure title.')
    parser.add_argument('--dpi', type=int, default=200)
    parser.add_argument('--write_aligned_npy', default=None,
                        help='Optional path. After picking the projection, '
                             'rotate the polygon into the projection basis '
                             '(u -> +X, v -> +Y, depth -> +Z) and save the '
                             'resulting (N, 3) array to this .npy. Useful '
                             'for downstream 3D renderings that need to '
                             'align pixel-for-pixel with the 2D over/under '
                             'diagram (e.g. overlaying in Illustrator).')
    return parser.parse_args()


def _suffix_path(path: str, suffix: str) -> str:
    """Insert ``suffix`` before the file extension."""
    if not path:
        return path
    base, ext = os.path.splitext(path)
    return f'{base}{suffix}{ext}'


def main() -> int:
    args = _parse_args()
    closed = not args.open
    one_based = not args.zero_based_edges

    # Resolve the list of WalkIDs to process.
    if args.walk_ids:
        walk_id_list = [int(x) for x in args.walk_ids.split(',') if x.strip()]
        if not walk_id_list:
            print('No WalkIDs parsed from --walk_ids.', file=sys.stderr)
            return 1
    elif args.walk_id is not None:
        walk_id_list = [args.walk_id]
    else:
        walk_id_list = [None]  # for non-CSV inputs (.xyz, .npy)
    multi = len(walk_id_list) > 1

    # --no_vertex_numbers semantics:
    #   vertices: always off when the flag is set.
    #   crossings (diamond letter labels): off ONLY when --polygon_only is NOT
    #     set. In --polygon_only mode the broken-underpass diagram doesn't
    #     even include labelled crossings (only colored diamonds), so the
    #     toggle is a no-op there.
    effective_label_vertices = args.label_vertices and not args.no_vertex_numbers
    if args.polygon_only:
        effective_label_crossings = True
    else:
        effective_label_crossings = not args.no_vertex_numbers

    for wid in walk_id_list:
        if wid is not None:
            args.walk_id = wid
        curve, _vertex_labels = load_curve(args)
        if args.projection == 'auto':
            projection = optimize_projection(
                curve, closed, args.projection_objective,
                args.projection_metric, args.projection_pairs,
                args.projection_samples, args.projection_refine,
                args.projection_seed, args.over_depth, args.min_crossings)
            crossing_msg = ''
            if projection.crossing_count is not None:
                crossing_msg = (
                    f', crossing_count={projection.crossing_count}, '
                    f'min_crossing_distance='
                    f'{projection.min_crossing_distance:.3g}, '
                    f'p05_crossing_distance='
                    f'{projection.p05_crossing_distance:.3g}, '
                    f'min_crossing_vertex_distance='
                    f'{projection.min_crossing_vertex_distance:.3g}, '
                    f'p05_crossing_vertex_distance='
                    f'{projection.p05_crossing_vertex_distance:.3g}')
            wid_tag = f'WalkID {wid}: ' if wid is not None else ''
            print(
                f'{wid_tag}Auto projection: '
                f'depth=({projection.depth[0]:.5g}, '
                f'{projection.depth[1]:.5g}, {projection.depth[2]:.5g}), '
                f'objective={args.projection_objective}, '
                f'{args.projection_metric} score={projection.score:.3g}'
                f'{crossing_msg}, '
                f'min_vertex_distance={projection.min_vertex_distance:.3g}, '
                f'p05_vertex_distance={projection.p05_vertex_distance:.3g}')
        else:
            projection = PROJECTIONS[args.projection]

        projected = project_curve(curve, projection)
        depth = depth_values(curve, projection)
        crossings = find_crossings(
            curve, projected, depth, projection, closed, args.over_depth)
        crossings = _renumber_crossings_by_walk(crossings)

        # Per-walk output paths when multiple walks are processed.
        suffix = f'_walk{wid}' if (multi and wid is not None) else ''
        out_path = _suffix_path(args.output, suffix) if multi else args.output
        report_path = (_suffix_path(args.report_pdf, suffix)
                       if (args.report_pdf and multi) else args.report_pdf)
        crossing_csv = (args.crossing_csv if args.crossing_csv is not None
                        else _default_crossing_csv(out_path))
        if multi and args.crossing_csv is not None:
            crossing_csv = _suffix_path(args.crossing_csv, suffix)
        write_crossing_csv(crossing_csv, crossings, one_based)

        # Optional alignment side-car: rotate the polygon into the
        # projection basis so the (u, v) projection coincides with the
        # XY plane and `depth` becomes the new +Z axis. Saved as an
        # (N, 3) .npy file. Downstream tools (e.g. the random-icosagon
        # wrapper) can render a 3D scatter from this and stack it on
        # top of the 2D over/under diagram in Illustrator.
        if args.write_aligned_npy:
            R = np.column_stack(
                [projection.u, projection.v, projection.depth])
            aligned = curve @ R
            aligned_path = (_suffix_path(args.write_aligned_npy, suffix)
                            if multi else args.write_aligned_npy)
            np.save(aligned_path, aligned)
            print(f'Wrote aligned polygon: {aligned_path} '
                  f'(rotated so projection u->+X, v->+Y, depth->+Z)')

        if args.title is not None:
            title = args.title
        else:
            proj_label = ('AUTO' if args.projection == 'auto'
                          else args.projection.upper())
            if wid is not None:
                loc = f', {args.location}' if args.location else ''
                title = (f'WalkID {wid}{loc}: '
                         f'{proj_label} over/under diagram '
                         f'(n={curve.shape[0]}, crossings={len(crossings)})')
            else:
                source = os.path.basename(args.input)
                title = (f'{source}: {proj_label} over/under diagram '
                         f'(n={curve.shape[0]}, crossings={len(crossings)})')

        if args.polygon_only:
            plot_polygon_only(
                curve, projection, crossings, out_path, closed,
                title, one_based, effective_label_vertices, args.dpi)
            print(f'Wrote polygon-only figure: {out_path}')
            if report_path:
                print(f'[warn] --report_pdf ignored under --polygon_only',
                      file=sys.stderr)
        else:
            plot_diagram(
                curve, projection, crossings, out_path, closed,
                title, one_based, effective_label_vertices, args.label_edges,
                args.crossing_overlay, args.local_ou_tags, args.dpi)
            print(f'Wrote figure: {out_path}')
            if report_path:
                write_three_page_report_pdf(
                    report_path, curve, projection, crossings, closed,
                    title, one_based, effective_label_vertices,
                    args.label_edges, args.crossing_overlay,
                    args.local_ou_tags,
                    label_crossings=effective_label_crossings)
                print(f'Wrote report PDF: {report_path}')
        print(f'Wrote crossings: {crossing_csv} ({len(crossings)} rows)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
