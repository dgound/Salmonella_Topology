"""
Unified Jiggle Control Pipeline for Knot Topology Analysis.

One script consolidating three earlier variants:
  - jiggle_control.py            (KnotoID engine, 3-category output)
  - jiggle_control_det.py        (knoodle engine, 3-category output)
  - jiggle_control_det_2cat.py   (knoodle engine, 2-category output)

CLI selects the engine and the output category scheme:
  --engine knoodle|knotoid   (default: knoodle)
  --categories 2|3           (default: 2)

Engine semantics:
  knoodle : Alexander polynomial determinant at t = -1 via knoodle.
            det == 1 -> '0_1' (unknot)
            det == 3 -> '3_1' (trefoil)
            otherwise -> 'complex_det<N>' (gets bucketed as Complex/Knotted)
  knotoid : KnotoID's projection-based knot classifier, returning the
            dominant knot name (e.g., '0_1', '3_1', '3_1m', '4_1', ...).

Category schemes (column index of the H5 hierarchical DataFrame):
  2 : Unknotted (det==1 or KnotoID name == '0_1')   |  Knotted (anything else)
  3 : 0_1                                            |  3_1 (incl. '3_1m')  |  Complex

H5 layout (always):
  distance_NNN -> hierarchical Series with index (Control, KnotType)
    Control:  Raw, Permuted, Jiggled, Random Curves
    Values:   Percentages

Default output filename: rez_df_<engine>_<C>cat.h5
(e.g., rez_df_knoodle_2cat.h5; rez_df_knotoid_3cat.h5).

All other features of the recent jiggle pipeline are kept:
  --dewitt --sample_size N   : sample De Witt curves per walk instead of centroid
  -j NUM_JIGGLES             : jiggles per walk per distance (centroid mode)
  --stop_tolerance T         : early-stop the distance sweep once the jiggled
                               %Knotted is within T pp of Random Curves
  --min-tp/--max-tp          : time-point range (default 0..19)
  -s, --skip-chull           : skip convex-hull random control
  --workers N                : parallel workers across distances

Examples:
  python jiggle_control.py data.txt --engine knoodle --categories 2 --dewitt
  python jiggle_control.py data.txt --engine knotoid --categories 3
"""

#%%
# Optional engines: knoodle preferred (and required for the default), KnotoID
# imported lazily so the script still loads in envs that lack it.
try:
    import knoodle
except ImportError:
    knoodle = None
try:
    from KnotoID import KnotoID
except ImportError:
    KnotoID = None
import numpy as np
import pandas as pd
import random
import networkx as nx
from itertools import permutations, combinations
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import ConvexHull, QhullError
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from datetime import datetime
import multiprocessing as mp
import pickle
import argparse
import sys
import warnings
import os
from tqdm import tqdm

# Suppress library noise
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'


def _reflect_into_box(points, min_coords, max_coords):
    """Triangle-wave (infinite) reflection of points into [min_coords, max_coords].

    Used to bound jiggled vertices when the jiggle radius is much larger
    than the bounding-box dimensions. Each axis is folded back-and-forth so
    the asymptotic distribution at very large jiggles is uniform over the box.

    A single np.where reflection (the original behavior) caps points after
    one bounce; points that overshoot the opposite face stay outside.
    """
    L = max_coords - min_coords
    # Degenerate axes (L == 0) collapse to the box value
    safeL = np.where(L > 0, L, 1.0)
    shifted = np.mod(points - min_coords, 2 * safeL)
    folded = np.where(shifted > safeL, 2 * safeL - shifted, shifted)
    out = min_coords + folded
    return np.where(L > 0, out, min_coords)


def _walk_id_only(key):
    """Return the walk_id integer from a rotated_curves dict key.

    In De Witt mode keys are (walk_id, sample_idx) tuples; in centroid mode
    keys are scalar walk_ids. The H5 aggregation does not care about
    sample_idx, so we strip it here.
    """
    if isinstance(key, tuple):
        return key[0]
    return key


def _knoodle_det_name(curve):
    """Classify a 3D polygon by Alexander determinant at t = -1 (knoodle).

    Returns a knot-name string compatible with `JiggleAnalyzer.classify_name`:
      det == 1 -> '0_1'
      det == 3 -> '3_1'
      other    -> 'complex_det<N>'    (gets bucketed as Complex/Knotted)
    Returns 'complex_failed' on knoodle exceptions.
    """
    if knoodle is None:
        raise ImportError("knoodle is not installed; cannot use --engine knoodle")
    try:
        coords = np.asarray(curve, dtype=float).ravel().tolist()
        a = knoodle.KnotAnalyzer(coords, simplify=True, simplify_level=5)
        det = abs(a.alexander([-1.0])[0].to_double())
        r = round(det)
        if abs(det - r) < 1e-6:
            det = int(r)
        if det == 1:
            return '0_1'
        if det == 3:
            return '3_1'
        return f'complex_det{det}'
    except Exception:
        return 'complex_failed'


def _knotoid_name(curve, closure_method='direct'):
    """Classify a 3D polygon via KnotoID's dominant-knot identifier.

    Returns the knot name string from KnotoID (e.g., '0_1', '3_1', '3_1m', '4_1'),
    or 'complex_failed' on KnotoID exceptions.
    """
    if KnotoID is None:
        raise ImportError("KnotoID is not installed; cannot use --engine knotoid")
    # Suppress KnotoID's stdout/stderr noise
    old_stdout, old_stderr = sys.stdout, sys.stderr
    try:
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')
        try:
            k = KnotoID(projections_list='internal', closure_method=closure_method)
            k.analyze_curve(np.asarray(curve, dtype=float))
            row = k.curve_analysis.loc[0]
            return str(row['Name'])
        finally:
            sys.stdout.close()
            sys.stderr.close()
            sys.stdout = old_stdout
            sys.stderr = old_stderr
    except Exception:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        return 'complex_failed'


def _classify_curve(curve, engine='knoodle', closure_method='direct'):
    """Engine-dispatch helper returning a knot-name string."""
    if engine == 'knotoid':
        return _knotoid_name(curve, closure_method=closure_method)
    # default
    return _knoodle_det_name(curve)

#%%
class JiggleKnotDetector:
    """
    Streamlined knot detector for jiggle control analysis.

    Loads SMLM trajectory data, orders centroids, rotates curves to standard
    orientation, and performs knot analysis with various control methods.
    """

    def __init__(self, input_file: str, only_full_walks: bool = True,
                 by_timepoint: bool = True, exclude_walks: list = None, verbose: bool = True,
                 min_tp: int = 0, max_tp: int = 19,
                 dewitt: bool = False, sample_size: int = 1000,
                 engine: str = 'knoodle') -> None:
        """
        Initialize detector with data loading and preprocessing.

        Parameters:
            input_file: Path to CSV/TXT file with SMLM data
            only_full_walks: Only analyze walks with all timepoints in range
            by_timepoint: Group by timepoint only (not subclusters)
            exclude_walks: List of Walk-IDs to exclude from analysis
            verbose: Print initialization messages
            min_tp: Minimum timepoint (inclusive, default 0)
            max_tp: Maximum timepoint (inclusive, default 19)
        """
        self.input_file = input_file
        self.only_full_walks = only_full_walks
        self.by_tp_only = by_timepoint
        self.exclude_walks = exclude_walks if exclude_walks is not None else []
        self.verbose = verbose
        self.min_tp = min_tp
        self.max_tp = max_tp
        self.expected_timepoints = max_tp - min_tp + 1
        self.dewitt = dewitt
        self.sample_size = int(sample_size)
        if engine not in ('knoodle', 'knotoid'):
            raise ValueError(f"engine must be 'knoodle' or 'knotoid', got {engine!r}")
        self.engine = engine

        if self.verbose:
            print('Full walks only:', self.only_full_walks)
            print(f'Timepoint range: {self.min_tp} to {self.max_tp} ({self.expected_timepoints} timepoints)')
            if self.by_tp_only:
                print('Time-point Centroids Analysis: True')
            else:
                print('Using subclusters')
            if self.dewitt:
                print(f'De Witt sampling: ON, sample_size={self.sample_size} per walk')

        # Load and process data
        self.curves = self.load_data(input_file, by_tp_only=self.by_tp_only)

        # Convert from nm to meters (matching knotDetector_optimized behavior)
        # load_data multiplies by 1e9 to get nm, but we need meters for proper scaling
        self.curves[['weighted_centroid_x', 'weighted_centroid_y', 'weighted_centroid_z']] /= 1e9

        # In De Witt mode, also keep the raw per-(walk, TP) localizations
        # (in the same coordinate units as the centroids above).
        self.raw_locs_by_walk_tp = None
        if self.dewitt:
            self.raw_locs_by_walk_tp = self._load_raw_localizations(input_file)

        # Initialize results containers
        self.raw_results = []
        self.permute_results = []
        self.jiggle_results = []
        self.convex_hull_results = []

        # Rotate all curves to standard orientation
        self.transformations = {}
        self.rotated_curves = {}
        # In De Witt mode, rotated centroid kept separately for the convex hull control
        self.centroid_curves = {}
        self.rotate_all_curves(only_full_walks=self.only_full_walks)

    def _load_raw_localizations(self, data: str):
        """Return {walk_id: {tp: (n_locs, 3) ndarray}} of raw localizations.

        Same TP filter and exclude_walks logic as `load_data`. Coordinate
        units match the (post-`*1e9`/`/=1e9`) centroid units stored in
        `self.curves`, i.e. raw CSV coords (no scaling).
        """
        df = pd.read_csv(data, sep=",")
        if self.exclude_walks:
            df = df[~df['Walk-IDs'].isin(self.exclude_walks)]
        df = df[(df['time-point'] >= self.min_tp) & (df['time-point'] <= self.max_tp)]

        # Keep only walks with at least one row per TP in the range
        full_walks = self._full_walks_only(df) if self.only_full_walks else df['Walk-IDs'].unique()
        df = df[df['Walk-IDs'].isin(full_walks)]

        out = {}
        for walk_id, wdf in df.groupby('Walk-IDs'):
            out[int(walk_id)] = {
                int(tp): tpdf[['x', 'y', 'z']].values
                for tp, tpdf in wdf.groupby('time-point')
            }
        return out

    def _full_walks_only(self, df):
        """Return the set of Walk-IDs whose TP coverage spans the required range."""
        tp_counts = df.groupby('Walk-IDs')['time-point'].nunique()
        return tp_counts[tp_counts >= self.expected_timepoints].index

    def weighted_average_and_precision(self, group):
        """Calculate weighted centroids from raw localizations."""
        group['precisionx'] = group['precisionx'].replace(0, 1e-10)
        group['precisiony'] = group['precisiony'].replace(0, 1e-10)
        group['precisionz'] = group['precisionz'].replace(0, 1e-10)

        weights_x = 1 / group['precisionx']**2
        weights_y = 1 / group['precisiony']**2
        weights_z = 1 / group['precisionz']**2

        centroid_x = np.average(group['x'], weights=weights_x) * 1e9  # convert to nm
        centroid_y = np.average(group['y'], weights=weights_y) * 1e9
        centroid_z = np.average(group['z'], weights=weights_z) * 1e9

        precision_x = np.sqrt(1 / np.sum(weights_x))
        precision_y = np.sqrt(1 / np.sum(weights_y))
        precision_z = np.sqrt(1 / np.sum(weights_z))

        return pd.Series({
            'weighted_centroid_x': centroid_x,
            'weighted_centroid_y': centroid_y,
            'weighted_centroid_z': centroid_z,
            'combined_precision_x': precision_x,
            'combined_precision_y': precision_y,
            'combined_precision_z': precision_z
        })

    def load_data(self, data: str, by_tp_only: bool) -> pd.DataFrame:
        """Load SMLM data and compute weighted centroids."""
        df = pd.read_csv(data, sep=",")

        if self.exclude_walks:
            print("Excluding Walk-IDs:", self.exclude_walks)
            df = df[~df['Walk-IDs'].isin(self.exclude_walks)]

        # Filter to timepoint range (inclusive)
        df = df[(df['time-point'] >= self.min_tp) & (df['time-point'] <= self.max_tp)]

        if by_tp_only:
            results_df = df.groupby(['Walk-IDs', 'time-point']).apply(
                self.weighted_average_and_precision
            ).reset_index()
            results_df.columns = ['Walk-ID', 'time-point', 'weighted_centroid_x',
                                'weighted_centroid_y', 'weighted_centroid_z',
                                'combined_precision_x', 'combined_precision_y',
                                'combined_precision_z']
        else:
            results_df = df.groupby(['Walk-IDs', 'time-point', 'cluster-ID']).apply(
                self.weighted_average_and_precision
            ).reset_index()
            results_df.columns = ['Walk-ID', 'time-point', 'cluster-ID',
                                'weighted_centroid_x', 'weighted_centroid_y',
                                'weighted_centroid_z', 'combined_precision_x',
                                'combined_precision_y', 'combined_precision_z']

        # Filter incomplete walks if only_full_walks is True
        if self.only_full_walks:
            walk_counts = results_df.groupby('Walk-ID').size()
            full_walks = walk_counts[walk_counts >= self.expected_timepoints].index
            results_df = results_df[results_df['Walk-ID'].isin(full_walks)]
            incomplete_count = len(walk_counts) - len(full_walks)
            if incomplete_count > 0:
                print(f"Filtered out {incomplete_count} incomplete walks (need {self.expected_timepoints} timepoints)")

        return results_df

    def order_centroids(self):
        """
        Order multi-centroid timepoints using traveling salesman algorithm.

        Uses worm-like chain physics constraints (persistence length = 50nm)
        to enforce biologically plausible ordering.
        """
        L_p = 50e-9  # Persistence length in meters (50 nm)
        lambda_theta = 1.0  # Weight for angular penalty
        lambda_adj = 1.0    # Weight for alignment penalty

        self.curves[['weighted_centroid_x', 'weighted_centroid_y', 'weighted_centroid_z']] /= 1e9
        ordered_curves = pd.DataFrame()

        for walk_id, group in self.curves.groupby('Walk-ID'):
            group = group.sort_values('time-point').reset_index(drop=True)
            time_points = group['time-point'].unique()
            ordered_centroids = []
            prev_centroid = None

            for n in time_points:
                step_group = group[group['time-point'] == n].reset_index(drop=True)
                num_centroids = len(step_group)

                if num_centroids == 1:
                    centroid = step_group.iloc[0]
                    ordered_centroids.append(centroid)
                    prev_centroid = centroid
                else:
                    centroids = step_group[['weighted_centroid_x', 'weighted_centroid_y',
                                          'weighted_centroid_z']].values
                    cost_matrix = np.zeros((num_centroids, num_centroids))

                    distances = squareform(pdist(centroids))
                    L_s = np.mean(distances[distances > 0])
                    theta_max = np.sqrt(L_s / L_p)
                    cos_theta_min = np.cos(theta_max)

                    for i in range(num_centroids):
                        for j in range(num_centroids):
                            if i != j:
                                dist = np.linalg.norm(centroids[i] - centroids[j])
                                angle_penalty = 0
                                if prev_centroid is not None:
                                    v1 = centroids[i] - prev_centroid[['weighted_centroid_x',
                                                                       'weighted_centroid_y',
                                                                       'weighted_centroid_z']].values
                                    v2 = centroids[j] - centroids[i]
                                    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                                    cos_theta = np.clip(cos_theta, -1.0, 1.0)
                                    theta = np.arccos(cos_theta)
                                    if theta > theta_max:
                                        angle_penalty = lambda_theta * (1 - cos_theta)

                                alignment_penalty = 0
                                cost_matrix[i, j] = dist + angle_penalty + alignment_penalty
                            else:
                                cost_matrix[i, j] = np.inf

                    G = nx.complete_graph(num_centroids)
                    for u, v in G.edges():
                        G[u][v]['weight'] = cost_matrix[u, v]

                    if num_centroids <= 8:
                        min_cost = np.inf
                        best_order = None
                        for perm in permutations(range(num_centroids)):
                            cost = sum(cost_matrix[perm[i], perm[i + 1]] for i in range(num_centroids - 1))
                            if cost < min_cost:
                                min_cost = cost
                                best_order = perm
                    else:
                        cycle = nx.approximation.traveling_salesman_problem(G, weight='weight')
                        best_order = cycle[:-1]

                    for idx in best_order:
                        centroid = step_group.iloc[idx]
                        ordered_centroids.append(centroid)
                        prev_centroid = centroid

            ordered_df = pd.DataFrame(ordered_centroids)
            ordered_curves = pd.concat([ordered_curves, ordered_df], ignore_index=True)

        return ordered_curves

    def choose_random_vector(self, scaling_factor, accuracy):
        """Generate random 3D displacement within spherical volume."""
        effective_radius = accuracy * scaling_factor
        r = effective_radius * (random.uniform(0, 1)**(1/3))
        phi = random.uniform(0, 2 * np.pi)
        costheta = random.uniform(-1, 1)
        theta = np.arccos(costheta)

        dx = r * np.sin(theta) * np.cos(phi)
        dy = r * np.sin(theta) * np.sin(phi)
        dz = r * np.cos(theta)
        return dx, dy, dz

    def perturb_centroid(self, row, scaling_factor, accuracy):
        """Apply random perturbation to a point."""
        dx, dy, dz = self.choose_random_vector(scaling_factor, accuracy)

        if isinstance(row, pd.Series):
            perturbed_centroid_x = row['weighted_centroid_x'] + dx
            perturbed_centroid_y = row['weighted_centroid_y'] + dy
            perturbed_centroid_z = row['weighted_centroid_z'] + dz
        elif isinstance(row, np.ndarray):
            perturbed_centroid_x = row[0] + dx
            perturbed_centroid_y = row[1] + dy
            perturbed_centroid_z = row[2] + dz
        else:
            raise ValueError("Invalid input type. Expected pd.Series or np.ndarray.")

        return pd.Series({
            'perturbed_centroid_x': perturbed_centroid_x,
            'perturbed_centroid_y': perturbed_centroid_y,
            'perturbed_centroid_z': perturbed_centroid_z
        })

    def knots_dist(self, curve, closure, return_dominant):
        """Analyze knot topology via the selected engine.

        ``closure`` is used only when the engine is 'knotoid'; knoodle
        classifies the polygon directly. The return value mimics KnotoID's
        ``.curve_analysis.loc[0]`` access pattern (one-row Series with 'Name').
        """
        name = _classify_curve(curve, engine=self.engine, closure_method=closure)
        row = pd.Series({'Name': name})
        if return_dominant:
            return row
        return pd.DataFrame([row])

    def find_max_distance_pair(self, curve, exclude_pairs=None):
        """Find the pair of points with maximum distance."""
        max_dist = -1
        pair = (0, 1)
        for i, j in combinations(range(curve.shape[0]), 2):
            if exclude_pairs and ((i, j) in exclude_pairs or (j, i) in exclude_pairs):
                continue
            dist = np.linalg.norm(curve[i] - curve[j])
            if dist > max_dist:
                max_dist = dist
                pair = (i, j)
        return pair

    def compute_rotation_matrix(self, source_vector, target_vector):
        """Compute rotation matrix to align source vector with target vector."""
        source_vector = source_vector / np.linalg.norm(source_vector)
        target_vector = target_vector / np.linalg.norm(target_vector)
        v = np.cross(source_vector, target_vector)
        c = np.dot(source_vector, target_vector)

        if np.isclose(c, -1.0):
            orthogonal = np.array([1, 0, 0])
            if np.allclose(source_vector, orthogonal):
                orthogonal = np.array([0, 1, 0])
            v = np.cross(source_vector, orthogonal)
            v /= np.linalg.norm(v)
            rotation_matrix = -np.eye(3) + 2 * np.outer(v, v)
            return rotation_matrix
        elif np.isclose(c, 1.0):
            return np.eye(3)

        s = np.linalg.norm(v)
        K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        rotation_matrix = np.eye(3) + K + K @ K * ((1 - c) / (s ** 2))
        return rotation_matrix

    def rotate_curve(self, curve, rotation_matrix):
        """Apply rotation matrix to curve."""
        return curve @ rotation_matrix.T

    def is_inside_convex_hull(self, points, A, b):
        """Test if points are inside convex hull."""
        return np.all(np.dot(points, A.T) + b <= 1e-12, axis=1)

    def generate_random_points_in_convex_hull(self, convex_hull, num_points):
        """Generate random points uniformly within convex hull."""
        hull_vertices = convex_hull.points[convex_hull.vertices]
        A = convex_hull.equations[:, :-1]
        b = convex_hull.equations[:, -1]

        min_coords = np.min(convex_hull.points, axis=0)
        max_coords = np.max(convex_hull.points, axis=0)
        random_points = []
        generated = 0
        batch_size = max(num_points * 2, 1000)

        while generated < num_points:
            batch = np.random.uniform(min_coords, max_coords, size=(batch_size, 3))
            inside = self.is_inside_convex_hull(batch, A, b)
            inside_points = batch[inside]
            remaining = num_points - generated

            if inside_points.shape[0] >= remaining:
                random_points.append(inside_points[:remaining])
                generated += remaining
            else:
                random_points.append(inside_points)
                generated += inside_points.shape[0]

        return np.vstack(random_points)

    def rotate_all_curves(self, only_full_walks, walk_length: int = 20):
        """
        Rotate all curves to standard orientation using principal axes.

        Aligns longest axis to y-axis, then secondary axis to x-axis.
        Note: Incomplete walks are filtered in load_data(), so no skipping needed here.
        """
        groups = self.curves.groupby('Walk-ID')
        curves = {}

        for name, group in groups:
            c = group[['weighted_centroid_x', 'weighted_centroid_y',
                      'weighted_centroid_z']].values
            curves[int(name)] = np.array(c)

        for name, curve in curves.items():
            self.transformations[name] = {'rot1': np.identity(3), 'rot2': np.identity(3)}

            # First rotation: align longest axis to y-axis
            idx1, idx2 = self.find_max_distance_pair(curve)
            point1, point2 = curve[idx1], curve[idx2]
            major_vector = point2 - point1

            y_axis = np.array([0, 1, 0])
            rotation_matrix1 = self.compute_rotation_matrix(major_vector, y_axis)
            self.transformations[name]['rot1'] = rotation_matrix1
            rotated_curve = self.rotate_curve(curve, rotation_matrix1)

            # Second rotation: align secondary axis to x-axis
            exclude_pairs = {(idx1, idx2), (idx2, idx1)}
            idx3, idx4 = self.find_max_distance_pair(rotated_curve, exclude_pairs=exclude_pairs)

            if idx3 == -1:
                if self.verbose:
                    print(f"Warning: Could not find secondary axis for Walk-ID {name}. Skipping rotation 2.")
                rotation_matrix2 = np.identity(3)
            else:
                point3, point4 = rotated_curve[idx3], rotated_curve[idx4]
                secondary_vector = point4 - point3
                if np.linalg.norm(secondary_vector) < 1e-9:
                    if self.verbose:
                        print(f"Warning: Secondary axis has zero length for Walk-ID {name}. Skipping rotation 2.")
                    rotation_matrix2 = np.identity(3)
                else:
                    x_axis = np.array([1, 0, 0])
                    rotation_matrix2 = self.compute_rotation_matrix(secondary_vector, x_axis)

            self.transformations[name]['rot2'] = rotation_matrix2
            final_rotated_curve = self.rotate_curve(rotated_curve, rotation_matrix2)
            # Always keep the rotated centroid (used as canonical curve and for
            # the convex-hull control in De Witt mode).
            self.centroid_curves[name] = final_rotated_curve
            self.rotated_curves[name] = final_rotated_curve

        # In De Witt mode, replace the centroid-based rotated_curves with a
        # larger pool of sample_size De Witt curves per walk, each rotated by
        # the same (rot1, rot2) as the centroid of that walk.
        if self.dewitt and self.raw_locs_by_walk_tp is not None:
            self.rotated_curves = {}
            tps_required = list(range(self.min_tp, self.max_tp + 1))
            rng = np.random.default_rng()
            for walk_id, tp_dict in self.raw_locs_by_walk_tp.items():
                if any(tp not in tp_dict or len(tp_dict[tp]) == 0
                       for tp in tps_required):
                    continue
                if walk_id not in self.transformations:
                    continue
                rot1 = self.transformations[walk_id]['rot1']
                rot2 = self.transformations[walk_id]['rot2']
                sizes = [len(tp_dict[tp]) for tp in tps_required]
                for i in range(self.sample_size):
                    pts = np.array([tp_dict[tp][rng.integers(sizes[k])]
                                    for k, tp in enumerate(tps_required)])
                    rotated = self.rotate_curve(pts, rot1)
                    rotated = self.rotate_curve(rotated, rot2)
                    self.rotated_curves[(walk_id, i)] = rotated
            if self.verbose:
                print(f"De Witt curves prepared: "
                      f"{len(self.rotated_curves)} total "
                      f"({len(self.centroid_curves)} walks * "
                      f"{self.sample_size} samples).")

    def analyze_raw_curves(self, closure_method: str = 'direct'):
        """Analyze original curves without any perturbations."""
        if self.verbose:
            print("Analyzing raw curves...")
        items = list(self.rotated_curves.items())
        for name, final_rotated_curve in tqdm(items, desc="Raw curves", disable=not self.verbose):
            distribution = self.knots_dist(final_rotated_curve, closure=closure_method,
                                          return_dominant=True)
            self.raw_results.append({
                'Name': distribution['Name'],
                'WalkID': _walk_id_only(name)
            })

    def run_permutation_control(self, num_permutes: int = 0, closure_method: str = 'direct'):
        """
        Run permutation control by randomly shuffling curve points.

        NOTE: Disabled by default (num_permutes=0) as results are not used in plots_fig3.py.
        """
        if num_permutes <= 0:
            if self.verbose:
                print("Permutation control disabled (not used in plots_fig3.py)")
            return

        if self.verbose:
            print(f"Running permutation control with {num_permutes} iterations...")

        for name, final_rotated_curve in self.rotated_curves.items():
            for i in range(num_permutes):
                permuted_curve = final_rotated_curve.copy()
                permuted_curve = permuted_curve[np.random.permutation(permuted_curve.shape[0])]
                distribution = self.knots_dist(permuted_curve, closure=closure_method,
                                              return_dominant=True)
                self.permute_results.append({
                    'Name': distribution['Name'],
                    'WalkID': _walk_id_only(name)
                })

    def run_jiggle_control(self, num_jiggles: int = 100, scaling_factor: int = 1,
                          jiggle_distance: int = 40, boundary_expansion: int = 40,
                          closure_method: str = 'direct'):
        """
        Run jiggle control by adding small random perturbations to each point.

        Critical method for plots_fig3.py - generates main data curves.
        """
        if self.verbose:
            print(f"Running jiggle control with {num_jiggles} iterations at distance={jiggle_distance}nm...")
        if num_jiggles <= 0:
            return

        for name, final_rotated_curve in self.rotated_curves.items():
            min_coords = np.min(final_rotated_curve, axis=0) - boundary_expansion
            max_coords = np.max(final_rotated_curve, axis=0) + boundary_expansion

            for i in range(num_jiggles):
                jiggled = np.apply_along_axis(
                    lambda row: self.perturb_centroid(row, scaling_factor=scaling_factor,
                                                     accuracy=jiggle_distance),
                    axis=1,
                    arr=final_rotated_curve
                )

                # Apply triangle-wave (infinite) boundary reflections so large jiggles
                # uniformize into the bounding box
                new_jiggled_curve = _reflect_into_box(jiggled, min_coords, max_coords)

                distribution = self.knots_dist(new_jiggled_curve, closure=closure_method,
                                              return_dominant=True)
                self.jiggle_results.append({
                    'Name': distribution['Name'],
                    'WalkID': _walk_id_only(name)
                })

    def run_convex_hull_control(self, num_random_curves: int = 100, closure_method: str = 'direct'):
        """
        Generate random curves within the convex hull of original curves.

        Used in plots_fig3.py as "Random Curves" baseline.

        The convex hull is computed from the per-walk centroid curves (one
        per walk), even in De Witt mode, so we don't generate W * sample_size
        hulls.
        """
        if self.verbose:
            print(f"Running convex hull control with {num_random_curves} iterations...")
        if num_random_curves <= 0:
            return

        hull_curves_dict = self.centroid_curves if self.centroid_curves else self.rotated_curves
        n_walks = len(hull_curves_dict)
        total_iterations = n_walks * num_random_curves
        pbar = tqdm(total=total_iterations, desc="Random curves", disable=not self.verbose)

        for name, rot_crv in hull_curves_dict.items():
            try:
                convex_hull = ConvexHull(rot_crv)
            except QhullError as e:
                if self.verbose:
                    tqdm.write(f"Walk-ID {name}: Convex hull computation failed ({e}). Skipping.")
                pbar.update(num_random_curves)
                continue

            num_points = rot_crv.shape[0]

            for i in range(num_random_curves):
                try:
                    random_crv_chull = self.generate_random_points_in_convex_hull(convex_hull, num_points)
                    distribution = self.knots_dist(random_crv_chull, closure=closure_method,
                                                  return_dominant=True)
                    self.convex_hull_results.append({
                        'Name': distribution['Name'],
                        'WalkID': _walk_id_only(name)
                    })
                except QhullError as e:
                    if self.verbose:
                        tqdm.write(f"Walk-ID {name}: Random point generation failed ({e}). Skipping iteration {i}.")
                pbar.update(1)

        pbar.close()

    def run_timepoint_control(self, closure_method: str = 'direct', num_permutes: int = 0,
                             num_jiggles: int = 100, scaling_factor: int = 1, jiggle_distance: int = 40,
                             boundary_expansion: int = 40, num_random_curves: int = 100):
        """Main run method that executes all control analyses."""
        if self.verbose:
            print("Starting timepoint control analysis...")

        # Clear previous results
        self.raw_results = []
        self.permute_results = []
        self.jiggle_results = []
        self.convex_hull_results = []

        # Run all controls
        self.analyze_raw_curves(closure_method)
        self.run_permutation_control(num_permutes, closure_method)
        self.run_jiggle_control(num_jiggles, scaling_factor, jiggle_distance,
                               boundary_expansion, closure_method)
        self.run_convex_hull_control(num_random_curves, closure_method)

        if self.verbose:
            print("Timepoint control analysis completed.")

    def get_results(self):
        """Return all results in a dictionary format."""
        return {
            'raw_results': self.raw_results,
            'permute_results': self.permute_results,
            'jiggle_results': self.jiggle_results,
            'convex_hull_results': self.convex_hull_results
        }


#%%
def _analyze_single_walk_convex_hull(args):
    """
    Module-level helper for parallel convex hull analysis.

    Args:
        args: tuple of (walk_id, curve_data, num_random_curves, closure_method)
              or (walk_id, curve_data, num_random_curves, closure_method, engine)

    Returns:
        list of dicts with 'Name' and 'WalkID' for each random curve result
    """
    if len(args) == 5:
        walk_id, curve_data, num_random_curves, closure_method, engine = args
    else:
        walk_id, curve_data, num_random_curves, closure_method = args
        engine = 'knoodle'
    results = []

    try:
        # Compute convex hull
        convex_hull = ConvexHull(curve_data)
        num_points = curve_data.shape[0]

        # Get hull parameters for point generation
        A = convex_hull.equations[:, :-1]
        b = convex_hull.equations[:, -1]
        min_coords = np.min(curve_data, axis=0)
        max_coords = np.max(curve_data, axis=0)

        for i in range(num_random_curves):
            try:
                # Generate random points in convex hull
                random_points = []
                generated = 0
                batch_size = max(num_points * 2, 100)

                while generated < num_points:
                    batch = np.random.uniform(min_coords, max_coords, size=(batch_size, 3))
                    inside = np.all(np.dot(batch, A.T) + b <= 1e-12, axis=1)
                    inside_points = batch[inside]
                    remaining = num_points - generated

                    if inside_points.shape[0] >= remaining:
                        random_points.append(inside_points[:remaining])
                        generated += remaining
                    else:
                        random_points.append(inside_points)
                        generated += inside_points.shape[0]

                random_curve = np.vstack(random_points)

                # Analyze knot topology via the selected engine
                name = _classify_curve(random_curve, engine=engine,
                                       closure_method=closure_method)

                results.append({
                    'Name': name,
                    'WalkID': _walk_id_only(walk_id)
                })
            except (QhullError, Exception):
                # Skip failed iterations
                continue

    except QhullError:
        # Skip walks where convex hull fails
        pass

    return results


def _analyze_single_distance_jiggle(args):
    """
    Module-level helper for parallel jiggle analysis at a single distance.

    Args:
        args: tuple of (distance, rotated_curves_dict, num_jiggles,
              closure_method, boundary_expansion) or with a trailing engine string.

    Returns:
        dict with 'distance' and 'jiggle_results'
    """
    if len(args) == 6:
        distance, rotated_curves, num_jiggles, closure_method, boundary_expansion, engine = args
    else:
        distance, rotated_curves, num_jiggles, closure_method, boundary_expansion = args
        engine = 'knoodle'
    results = []

    # Create independent RNG seeded uniquely for this worker
    # Using numpy's modern Generator ensures complete independence from global state
    rng = np.random.default_rng(int(distance) * 1000000 + os.getpid())

    for walk_id, curve in rotated_curves.items():
        # Compute boundary limits for reflections
        min_coords = np.min(curve, axis=0) - boundary_expansion
        max_coords = np.max(curve, axis=0) + boundary_expansion

        for i in range(num_jiggles):
            # Perturb each point in the curve
            jiggled = np.zeros_like(curve)
            for pt_idx in range(curve.shape[0]):
                point = curve[pt_idx]
                # Generate random displacement within spherical volume using independent RNG
                effective_radius = float(distance)
                r = effective_radius * (rng.uniform(0, 1)**(1/3))
                phi = rng.uniform(0, 2 * np.pi)
                costheta = rng.uniform(-1, 1)
                theta = np.arccos(costheta)
                dx = r * np.sin(theta) * np.cos(phi)
                dy = r * np.sin(theta) * np.sin(phi)
                dz = r * np.cos(theta)
                jiggled[pt_idx] = [point[0] + dx, point[1] + dy, point[2] + dz]

            # Apply triangle-wave (infinite) boundary reflections so large jiggles
            # uniformize into the bounding box
            new_jiggled_curve = _reflect_into_box(jiggled, min_coords, max_coords)

            # Analyze knot topology via the selected engine
            name = _classify_curve(new_jiggled_curve, engine=engine,
                                   closure_method=closure_method)
            results.append({
                'Name': name,
                'WalkID': _walk_id_only(walk_id)
            })

    return {'distance': distance, 'jiggle_results': results}


#%%
class JiggleAnalyzer:
    """
    Pipeline orchestrator for jiggle control analysis.

    Manages parallel execution across multiple jiggle distances and formats
    output for plots_fig3.py compatibility.
    """

    def __init__(self, data_file: str, resolution: int = 100,
                 full_walks_only: bool = True, by_tp_only: bool = True,
                 exclude_walks: list = None, min_tp: int = 0, max_tp: int = 19,
                 dewitt: bool = False, sample_size: int = 1000,
                 engine: str = 'knoodle', categories: int = 2):
        """
        Initialize analyzer with data file and parameters.

        Parameters:
            data_file: Path to SMLM trajectory data
            resolution: Spatial resolution in nm (for naming)
            full_walks_only: Only analyze complete walks with all timepoints in range
            by_tp_only: Group by timepoint (not subclusters)
            exclude_walks: List of Walk-IDs to exclude
            min_tp: Minimum timepoint (inclusive, default 0)
            max_tp: Maximum timepoint (inclusive, default 19)
            dewitt: If True, sample `sample_size` De Witt curves per walk
                (one random localization per TP) instead of using one
                centroid per walk for raw/permuted/jiggle.
            sample_size: Number of De Witt curves per walk (only used when
                `dewitt=True`).
        """
        self.resolution = resolution
        self.data_file = data_file
        self.full_walks_only = full_walks_only
        self.by_tp_only = by_tp_only
        self.exclude_walks = exclude_walks if exclude_walks is not None else []
        self.min_tp = min_tp
        self.max_tp = max_tp
        self.dewitt = dewitt
        self.sample_size = int(sample_size)
        if engine not in ('knoodle', 'knotoid'):
            raise ValueError(f"engine must be 'knoodle' or 'knotoid', got {engine!r}")
        if categories not in (2, 3):
            raise ValueError(f"categories must be 2 or 3, got {categories!r}")
        self.engine = engine
        self.categories = int(categories)
        self.todate = datetime.today().strftime('%m_%d_%Y')

        # Initialize detector
        self.detector = JiggleKnotDetector(
            data_file,
            only_full_walks=full_walks_only,
            by_timepoint=by_tp_only,
            exclude_walks=exclude_walks,
            min_tp=min_tp,
            max_tp=max_tp,
            dewitt=self.dewitt,
            sample_size=self.sample_size,
            engine=self.engine,
        )

        # Results storage
        self.raw_results = []
        self.permute_results = []
        self.convex_hull_results = []

        # Distance-wise jiggle results storage
        self.jiggle_results_by_distance = {}

        # Distance array for analysis - starts at 1 (Raw serves as distance=0).
        # Extended to 100,000 nm so the jiggle saturates onto the Random Curves baseline.
        self.distances = np.array([1, 5, 10, 20, 30, 40] +
                                 list(range(50, 1050, 50)) +
                                 [2000, 2500, 3000, 4000, 5000,
                                  7500, 10000, 15000, 20000, 30000, 50000, 100000])

    def classify_name(self, name: str) -> str:
        """Map a raw knot-name string into the configured category scheme.

        2-cat:  '0_1' -> 'Unknotted', everything else -> 'Knotted'
        3-cat:  '0_1' -> '0_1', '3_1' or '3_1m' -> '3_1', else -> 'Complex'
        """
        if self.categories == 2:
            return 'Unknotted' if name == '0_1' else 'Knotted'
        if name == '0_1':
            return '0_1'
        if name in ('3_1', '3_1m'):
            return '3_1'
        return 'Complex'

    @property
    def category_labels(self):
        """Ordered category labels used as H5 columns."""
        return ['Unknotted', 'Knotted'] if self.categories == 2 else ['0_1', '3_1', 'Complex']

    def run_baseline_analysis(self, num_permutes: int = 0,
                            num_random_curves: int = 1000,
                            max_workers: int = None) -> None:
        """
        Run baseline analysis (raw curves and random curves in convex hull).

        Permutation disabled by default as it's not used in plots_fig3.py.
        Random curves are now computed in parallel for better performance.
        """
        print("Running baseline analysis...")

        # Run raw curve analysis (sequential, fast)
        self.detector.analyze_raw_curves()
        self.raw_results = self.detector.raw_results

        # Run permutation control if enabled (usually disabled)
        if num_permutes > 0:
            self.detector.run_permutation_control(num_permutes)
            self.permute_results = self.detector.permute_results

        # Run parallel convex hull analysis
        if num_random_curves > 0:
            self.run_parallel_convex_hull_analysis(
                num_random_curves=num_random_curves,
                max_workers=max_workers
            )

        print(f"Baseline analysis completed. Raw: {len(self.raw_results)}, "
              f"Permuted: {len(self.permute_results)}, "
              f"Random Curves: {len(self.convex_hull_results)}")

    def run_parallel_convex_hull_analysis(self, num_random_curves: int = 1000,
                                         closure_method: str = 'direct',
                                         max_workers: int = None) -> None:
        """
        Run convex hull random curve analysis in parallel across walks.

        Each walk is processed independently, generating num_random_curves
        random curves within its convex hull. Hulls are built from per-walk
        centroid curves (one per walk) even in De Witt mode.
        """
        rotated_curves = (self.detector.centroid_curves
                          if getattr(self.detector, 'centroid_curves', None)
                          else self.detector.rotated_curves)
        n_walks = len(rotated_curves)

        if max_workers is None:
            max_workers = min(mp.cpu_count(), n_walks)

        print(f"Running parallel convex hull analysis with {max_workers} workers...")
        print(f"Analyzing {n_walks} walks with {num_random_curves} random curves each...")

        # Prepare arguments for parallel execution
        args_list = [
            (walk_id, curve_data, num_random_curves, closure_method, self.engine)
            for walk_id, curve_data in rotated_curves.items()
        ]

        # Execute in parallel with progress bar
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            parallel_results = list(tqdm(
                executor.map(_analyze_single_walk_convex_hull, args_list),
                total=n_walks,
                desc="Random curves (walks)"
            ))

        # Combine results from all walks
        self.convex_hull_results = []
        for walk_results in parallel_results:
            self.convex_hull_results.extend(walk_results)

        print(f"Parallel convex hull analysis completed. Total: {len(self.convex_hull_results)} results")

    @staticmethod
    def _is_knotted(name):
        """A curve is 'knotted' iff its raw name is not '0_1' (works for both schemes)."""
        return name != '0_1'

    def _jiggle_knotted_pct(self, distance):
        """Return %Knotted for a single distance's jiggle results, or NaN if empty."""
        results = self.jiggle_results_by_distance.get(distance, [])
        if not results:
            return float('nan')
        n_knotted = sum(1 for r in results if self._is_knotted(r['Name']))
        return 100.0 * n_knotted / len(results)

    def _random_curves_knotted_pct(self):
        """Return %Knotted from the convex-hull random control."""
        if not self.convex_hull_results:
            return float('nan')
        n_knotted = sum(1 for r in self.convex_hull_results if self._is_knotted(r['Name']))
        return 100.0 * n_knotted / len(self.convex_hull_results)

    def run_parallel_distance_analysis(self, distances: np.ndarray = None,
                                     num_jiggles: int = 1, max_workers: int = None,
                                     stop_tolerance: float = None) -> None:
        """
        Run distance analysis with optional early stopping.

        Distances are processed in increasing-distance chunks of size
        `max_workers`. After each chunk, the largest distance's %Knotted is
        compared to the Random Curves %Knotted (from `self.convex_hull_results`).
        If `|jig_knotted - random_knotted| <= stop_tolerance`, the sweep
        stops and the remaining (larger) distances are skipped.

        Set `stop_tolerance=None` (the default for backwards-compat callers)
        to disable early stopping and run every distance.
        """
        if distances is None:
            distances = self.distances

        # Always work in increasing order so early-stop is meaningful.
        distances_to_compute = sorted(np.asarray(distances).tolist())

        if max_workers is None:
            max_workers = min(mp.cpu_count(), len(distances_to_compute))

        print(f"Running parallel distance analysis with {max_workers} workers...")
        print(f"Analyzing {len(distances_to_compute)} distances with {num_jiggles} jiggles each...")
        random_knotted = self._random_curves_knotted_pct()
        if stop_tolerance is not None:
            print(f"Early stop: tol={stop_tolerance:.2f}%, "
                  f"random_curves knotted={random_knotted:.2f}%")

        # Get pre-computed curves ONCE from main detector (avoids redundant loading)
        rotated_curves = self.detector.rotated_curves
        closure_method = 'direct'
        boundary_expansion = 10

        pbar = tqdm(total=len(distances_to_compute), desc="Jiggle distances")
        stopped_early = False
        for chunk_start in range(0, len(distances_to_compute), max_workers):
            chunk = distances_to_compute[chunk_start:chunk_start + max_workers]
            args_list = [
                (distance, rotated_curves, num_jiggles, closure_method, boundary_expansion, self.engine)
                for distance in chunk
            ]
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                for result in executor.map(_analyze_single_distance_jiggle, args_list):
                    distance = result['distance']
                    self.jiggle_results_by_distance[distance] = result['jiggle_results']
                    pbar.update(1)

            if stop_tolerance is not None and not np.isnan(random_knotted):
                # Check convergence using the largest distance in this chunk
                last_d = chunk[-1]
                jig_pct = self._jiggle_knotted_pct(last_d)
                if (not np.isnan(jig_pct)) and abs(jig_pct - random_knotted) <= stop_tolerance:
                    pbar.set_postfix_str(
                        f"converged @ d={last_d:.0f} ({jig_pct:.1f}% vs {random_knotted:.1f}%)"
                    )
                    print(f"\nEarly stop: jig_knotted={jig_pct:.2f}% within "
                          f"{stop_tolerance:.2f}% of random={random_knotted:.2f}% "
                          f"at d={last_d:.0f}; skipping larger distances.")
                    stopped_early = True
                    break
        pbar.close()

        n_done = len(self.jiggle_results_by_distance)
        if stopped_early:
            print(f"Parallel analysis stopped early after {n_done} of "
                  f"{len(distances_to_compute)} distances.")
        else:
            print(f"Parallel analysis completed. Analyzed {n_done} distances")

    def save_h5_compatible_format(self, filename: str = None) -> None:
        """
        Save results in H5 format with the configured category scheme.

        Schema:
            - Multiple DataFrames indexed by distance
            - Each DataFrame has hierarchical index: (Control, KnotType)
            - Controls:   Raw, Permuted (zero if disabled), Jiggled, Random Curves
            - KnotTypes:  2-cat -> Unknotted, Knotted
                          3-cat -> 0_1, 3_1, Complex
            - Values:     Percentages

        Default filename: rez_df_<engine>_<C>cat.h5.
        """
        if filename is None:
            filename = f'rez_df_{self.engine}_{self.categories}cat.h5'
        print(f"Saving H5 file: {filename}")

        # Debug: Check what's in jiggle_results_by_distance
        print(f"DEBUG: jiggle_results_by_distance has {len(self.jiggle_results_by_distance)} keys")
        from collections import Counter
        for dist in [1, 100, 500, 5000]:
            if dist in self.jiggle_results_by_distance:
                results = self.jiggle_results_by_distance[dist]
                names = [r['Name'] for r in results]
                counts = Counter(names)
                unknot_pct = counts.get('0_1', 0) / len(names) * 100 if names else 0
                print(f"  Distance {dist}nm: {len(results)} results, {unknot_pct:.1f}% unknot")

        labels = self.category_labels

        with pd.HDFStore(filename, 'w') as store:
            for i, distance in enumerate(self.distances):
                distance_jiggles = self.jiggle_results_by_distance.get(distance, [])
                raw_df = pd.DataFrame(self.raw_results)
                perm_df = pd.DataFrame(self.permute_results)
                jiggle_df = pd.DataFrame(distance_jiggles)
                random_df = pd.DataFrame(self.convex_hull_results)

                def get_percentages(df):
                    if df.empty:
                        return pd.Series({lab: 0.0 for lab in labels})
                    df['category'] = df['Name'].apply(self.classify_name)
                    counts = df['category'].value_counts()
                    total = len(df)
                    return pd.Series({lab: 100.0 * counts.get(lab, 0) / total
                                      for lab in labels})

                result_dict = {
                    'Raw':           get_percentages(raw_df),
                    'Permuted':      get_percentages(perm_df),
                    'Jiggled':       get_percentages(jiggle_df),
                    'Random Curves': get_percentages(random_df),
                }
                hierarchical_df = pd.concat(result_dict, keys=['Raw', 'Permuted', 'Jiggled',
                                                               'Random Curves'])
                store[f'/distance_{i:03d}'] = hierarchical_df

        print(f"Saved H5 file compatible with plots_fig3.py: {filename}")

    def run_complete_analysis(self, num_permutes: int = 0,
                            num_jiggles: int = 1, num_random_curves: int = 1000,
                            max_workers: int = None,
                            stop_tolerance: float = None) -> dict:
        """
        Run complete optimized analysis pipeline.

        Parameters
        ----------
        stop_tolerance : float or None
            If set, stop the distance sweep once the jiggled %Knotted is
            within `stop_tolerance` percentage points of the Random Curves
            %Knotted (computed during baseline_analysis). Distances larger
            than the convergence point are skipped.

        Returns:
            dict: Summary statistics of the analysis
        """
        print("Starting complete jiggle control analysis...")

        # In De Witt mode, variation already comes from the sample_size
        # De Witt curves per walk -- force num_jiggles=1 so we get
        # sample_size samples per walk per distance, not sample_size * num_jiggles.
        if self.dewitt and num_jiggles != 1:
            print(f"De Witt mode: forcing num_jiggles=1 (was {num_jiggles}).")
            num_jiggles = 1

        # Step 1: Baseline analysis (raw + random curves) - parallelized
        self.run_baseline_analysis(num_permutes, num_random_curves, max_workers)

        # Step 2: Parallel distance analysis (jiggle at multiple distances)
        self.run_parallel_distance_analysis(
            num_jiggles=num_jiggles,
            max_workers=max_workers,
            stop_tolerance=stop_tolerance,
        )

        # Step 3: Save H5 output for plots_fig3.py
        self.save_h5_compatible_format()

        # Return summary statistics
        total_jiggle_results = sum(len(v) for v in self.jiggle_results_by_distance.values())
        summary = {
            'total_raw_results': len(self.raw_results),
            'total_permute_results': len(self.permute_results),
            'total_jiggle_results': total_jiggle_results,
            'total_random_curves': len(self.convex_hull_results),
            'distances_analyzed': len(self.distances)
        }

        print("Complete analysis finished!")
        return summary


#%%
def main(input_file=None, max_workers=None):
    """
    Main execution script for jiggle control analysis.

    Parameters:
        input_file: Path to SMLM trajectory data file
        max_workers: Number of parallel workers (default: cpu_count - 2)
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Jiggle Control Pipeline for Knot Topology Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python jiggle_control.py
  python jiggle_control.py ../datasets/my_data.txt
  python jiggle_control.py ../datasets/my_data.txt --workers 8
  python jiggle_control.py --min-tp 0 --max-tp 19
  python jiggle_control.py -s           # skip convex hull for faster debugging
  python jiggle_control.py -j 100 -s    # quick test with 100 jiggles, no convex hull
        """
    )
    parser.add_argument(
        'input_file',
        nargs='?',
        default='../datasets/BiggieExp3_97walks_FIXED_10-20-25.txt',
        help='Path to SMLM trajectory data file (default: BiggieExp3_97walks_FIXED_10-20-25.txt)'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=None,
        help='Number of parallel workers (default: cpu_count - 2)'
    )
    parser.add_argument(
        '--min-tp',
        type=int,
        default=0,
        help='Minimum timepoint inclusive (default: 0)'
    )
    parser.add_argument(
        '--max-tp',
        type=int,
        default=19,
        help='Maximum timepoint inclusive (default: 19)'
    )
    parser.add_argument(
        '-s', '--skip-chull',
        action='store_true',
        help='Skip convex hull control (for debugging jiggle analysis)'
    )
    parser.add_argument(
        '-j', '--jiggles',
        type=int,
        default=1000,
        help='Number of jiggles per distance (default: 1000). '
             'Ignored when --dewitt is set (forced to 1; '
             'variation comes from --sample_size).'
    )
    parser.add_argument(
        '--dewitt',
        action='store_true',
        help='Use De Witt curves (sample one random localization per TP) '
             'instead of weighted centroids per walk. With this flag, every '
             'control (raw/permuted/jiggled) is run on sample_size De Witt '
             'curves per walk; jiggle distance still sweeps the full distance array.'
    )
    parser.add_argument(
        '--sample_size',
        type=int,
        default=1000,
        help='Number of De Witt curves per walk (default: 1000). '
             'Only used when --dewitt is set.'
    )
    parser.add_argument(
        '--stop_tolerance',
        type=float,
        default=1.0,
        help='Stop the distance sweep once jiggled %%Knotted is within this '
             'many percentage points of the Random Curves %%Knotted '
             '(default: 1.0). Set to 0 or a negative number to disable.'
    )
    parser.add_argument(
        '--engine',
        choices=['knoodle', 'knotoid'],
        default='knoodle',
        help='Knot classifier engine: "knoodle" uses the Alexander polynomial '
             'determinant at t=-1 (fast, deterministic); "knotoid" uses the '
             'KnotoID projection-based dominant-knot classifier. Default: knoodle.'
    )
    parser.add_argument(
        '--categories',
        type=int,
        choices=[2, 3],
        default=2,
        help='Output category scheme. 2 = Unknotted/Knotted (default). '
             '3 = 0_1 / 3_1 / Complex.'
    )

    args = parser.parse_args()

    # Override with function parameters if provided
    if input_file is not None:
        args.input_file = input_file
    if max_workers is not None:
        args.workers = max_workers

    # Set default workers if not specified
    if args.workers is None:
        args.workers = max(1, mp.cpu_count() - 2)

    # Configuration
    RESOLUTION = 100
    PERMUTES = 0  # Disabled (not used in plots_fig3.py)
    JIGGLES = args.jiggles
    RANDOM_CURVES = 0 if args.skip_chull else 1000
    FULL_WALKS_ONLY = True
    BY_TP_ONLY = True
    EXCLUDE_WALKS = []  # e.g., [3, 9, 12, 13, 15, 20, 21, 23, 24]

    # Access timepoint args (argparse converts - to _)
    min_tp = args.min_tp
    max_tp = args.max_tp

    print("\n" + "="*60)
    print("JIGGLE CONTROL ANALYSIS")
    print("="*60)
    print(f"Input file: {args.input_file}")
    print(f"Workers: {args.workers}")
    print(f"Timepoint range: {min_tp} to {max_tp}")
    print(f"Jiggles per distance: {JIGGLES}")
    print(f"Random curves: {RANDOM_CURVES}" + (" (SKIPPED)" if args.skip_chull else ""))
    print(f"Engine: {args.engine}    Categories: {args.categories}")
    if args.dewitt:
        print(f"De Witt sampling: ON  (sample_size={args.sample_size} per walk; "
              f"num_jiggles forced to 1)")
    print("="*60 + "\n")

    # Initialize analyzer
    analyzer = JiggleAnalyzer(
        data_file=args.input_file,
        resolution=RESOLUTION,
        full_walks_only=FULL_WALKS_ONLY,
        by_tp_only=BY_TP_ONLY,
        exclude_walks=EXCLUDE_WALKS,
        min_tp=min_tp,
        max_tp=max_tp,
        dewitt=args.dewitt,
        sample_size=args.sample_size,
        engine=args.engine,
        categories=args.categories,
    )

    # Run complete analysis
    stop_tol = args.stop_tolerance if args.stop_tolerance is not None and args.stop_tolerance > 0 else None
    summary = analyzer.run_complete_analysis(
        num_permutes=PERMUTES,
        num_jiggles=JIGGLES,
        num_random_curves=RANDOM_CURVES,
        max_workers=args.workers,
        stop_tolerance=stop_tol,
    )

    print("\n" + "="*60)
    print("ANALYSIS SUMMARY:")
    print("="*60)
    for key, value in summary.items():
        print(f"{key}: {value}")
    print("="*60)

    return analyzer


if __name__ == "__main__":
    analyzer = main()

    # Optional: Save analyzer object for later inspection
    # with open('jiggle_analyzer_results.pkl', 'wb') as f:
    #     pickle.dump(analyzer, f)
