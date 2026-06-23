#!/usr/bin/env python3
"""
Compare knot distributions between experimental SMLM data and random polygons.
This script analyzes the knot types and determinants of sampled trajectories
from experimental data and compares them to random 20-gons.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import Counter
from typing import Tuple
import warnings
warnings.filterwarnings('ignore')

# Knot analysis imports
from KnotoID import KnotoID
import knoodle

# Vassiliev V2 invariant
from vassiliev_v2_projection import compute_v2_invariant

# Import wurlitzer to suppress C++ output from KnotoID
try:
    from wurlitzer import pipes
    WURLITZER_AVAILABLE = True
except ImportError:
    print("Warning: wurlitzer not installed. KnotoID output will be verbose.")
    print("Install with: pip install wurlitzer")
    WURLITZER_AVAILABLE = False

# Statistical analysis imports
from scipy.stats import entropy, wasserstein_distance
from scipy.spatial import ConvexHull, QhullError

# Visualization imports
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_pdf import PdfPages
plt.style.use('default')
sns.set_palette("husl")

# Enable rasterization for all PDFs (reduces file size dramatically)
plt.rcParams['path.simplify'] = True
plt.rcParams['path.simplify_threshold'] = 1.0
plt.rcParams['agg.path.chunksize'] = 10000

# Shape analysis imports
from scipy.linalg import orthogonal_procrustes
from scipy.stats import gaussian_kde

# Numba JIT compilation for performance-critical functions
try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    print("Warning: Numba not installed. ACN/Writhe computation will be slower.")
    print("Install with: pip install numba")


# ============================================================================
# Custom 20-Color Colormap for Time-Points
# ============================================================================

# Create custom 20-color colormap for time-points (rainbow gradient)
CUSTOM_20_COLORS_HEX = [
    '#ED2024', '#F05023', '#F8991D', '#FFE500', '#CADB2B',
    '#94C83D', '#72BF44', '#6ABD45', '#71BF4C', '#6FC59B',
    '#6FCCDD', '#43ABE1', '#4068B2', '#3B54A5', '#4652A4',
    '#6B52A2', '#97509F', '#C44D9C', '#ED2590', '#EE1C51'
]

from matplotlib.colors import ListedColormap
CUSTOM_TIMEPOINT_CMAP = ListedColormap(CUSTOM_20_COLORS_HEX, name='timepoint_20')


def classify_name(name: str) -> str:
    """Classify knot types into three categories."""
    if name == '0_1':
        return '0_1'
    elif name in ['3_1', '3_1m']:
        return '3_1'
    else:
        return 'Complex'


# ============================================================================
# ACN (Average Crossing Number) Computation Functions
# ============================================================================

# Numba-optimized helper functions (defined conditionally)
if NUMBA_AVAILABLE:
    @jit(nopython=True, cache=True)
    def _create_orthonormal_basis_numba(direction):
        """Create two orthonormal vectors perpendicular to the given direction."""
        if abs(direction[0]) < 0.9:
            v1 = np.array([1.0, 0.0, 0.0])
        else:
            v1 = np.array([0.0, 1.0, 0.0])

        # First perpendicular vector
        dot = v1[0]*direction[0] + v1[1]*direction[1] + v1[2]*direction[2]
        u1 = v1 - dot * direction
        norm = np.sqrt(u1[0]**2 + u1[1]**2 + u1[2]**2)
        u1 = u1 / norm

        # Second perpendicular vector (cross product)
        u2 = np.array([
            direction[1]*u1[2] - direction[2]*u1[1],
            direction[2]*u1[0] - direction[0]*u1[2],
            direction[0]*u1[1] - direction[1]*u1[0]
        ])
        return u1, u2

    @jit(nopython=True, cache=True)
    def _project_polygon_numba(polygon_3d, direction):
        """Project 3D polygon onto plane perpendicular to direction."""
        u1, u2 = _create_orthonormal_basis_numba(direction)
        n = len(polygon_3d)
        projected = np.empty((n, 2))
        for i in range(n):
            projected[i, 0] = polygon_3d[i, 0]*u1[0] + polygon_3d[i, 1]*u1[1] + polygon_3d[i, 2]*u1[2]
            projected[i, 1] = polygon_3d[i, 0]*u2[0] + polygon_3d[i, 1]*u2[1] + polygon_3d[i, 2]*u2[2]
        return projected

    @jit(nopython=True, cache=True)
    def _orientation_numba(px, py, qx, qy, rx, ry):
        """Compute orientation of ordered triplet."""
        return (qx - px) * (ry - py) - (qy - py) * (rx - px)

    @jit(nopython=True, cache=True)
    def _segments_intersect_numba(p1x, p1y, p2x, p2y, q1x, q1y, q2x, q2y):
        """Check if line segments intersect."""
        o1 = _orientation_numba(p1x, p1y, p2x, p2y, q1x, q1y)
        o2 = _orientation_numba(p1x, p1y, p2x, p2y, q2x, q2y)
        o3 = _orientation_numba(q1x, q1y, q2x, q2y, p1x, p1y)
        o4 = _orientation_numba(q1x, q1y, q2x, q2y, p2x, p2y)

        if o1 * o2 < 0 and o3 * o4 < 0:
            return True
        return False

    @jit(nopython=True, cache=True)
    def _compute_crossing_sign_numba(polygon_3d, i, j, n_vertices):
        """Compute the sign of a crossing between segments i and j."""
        i_next = (i + 1) % n_vertices
        j_next = (j + 1) % n_vertices

        # Segment vectors
        s_i = polygon_3d[i_next] - polygon_3d[i]
        s_j = polygon_3d[j_next] - polygon_3d[j]

        # Vector between segment origins
        r_ij = polygon_3d[j] - polygon_3d[i]

        # Cross product s_j x s_i
        cross_x = s_j[1]*s_i[2] - s_j[2]*s_i[1]
        cross_y = s_j[2]*s_i[0] - s_j[0]*s_i[2]
        cross_z = s_j[0]*s_i[1] - s_j[1]*s_i[0]

        # Dot product with r_ij
        triple_product = cross_x*r_ij[0] + cross_y*r_ij[1] + cross_z*r_ij[2]

        if triple_product > 0:
            return 1
        elif triple_product < 0:
            return -1
        else:
            return 0

    @jit(nopython=True, cache=True)
    def _count_crossings_with_signs_numba(polygon_3d, polygon_2d):
        """Count crossings and compute their signs."""
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
                    polygon_2d[j_next, 0], polygon_2d[j_next, 1]
                ):
                    unsigned_count += 1
                    sign = _compute_crossing_sign_numba(polygon_3d, i, j, n_vertices)
                    signed_sum += sign

        return unsigned_count, signed_sum

    @jit(nopython=True, cache=True)
    def _is_valid_projection_numba(polygon_2d, epsilon=1e-10):
        """Check if 2D projection produces a valid knot diagram."""
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
        """Compute ACN and writhe using parallel projections."""
        n_projections = len(directions)
        crossings = np.empty(n_projections)
        writhes = np.empty(n_projections)
        valid = np.empty(n_projections, dtype=np.bool_)

        for p in prange(n_projections):
            projected = _project_polygon_numba(polygon_3d, directions[p])

            if _is_valid_projection_numba(projected):
                unsigned_count, signed_sum = _count_crossings_with_signs_numba(polygon_3d, projected)
                crossings[p] = unsigned_count
                writhes[p] = signed_sum
                valid[p] = True
            else:
                crossings[p] = 0
                writhes[p] = 0
                valid[p] = False

        return crossings, writhes, valid


def generate_random_projection_directions(n_projections: int) -> np.ndarray:
    """Generate uniformly distributed random directions on unit sphere."""
    theta = np.random.uniform(0, 2*np.pi, n_projections)
    phi = np.arccos(1 - 2*np.random.uniform(0, 1, n_projections))

    directions = np.stack([
        np.sin(phi) * np.cos(theta),
        np.sin(phi) * np.sin(theta),
        np.cos(phi)
    ], axis=1)

    return directions


def create_orthonormal_basis(direction: np.ndarray):
    """Create two orthonormal vectors perpendicular to the given direction."""
    # Find a vector not parallel to direction
    if abs(direction[0]) < 0.9:
        v1 = np.array([1, 0, 0])
    else:
        v1 = np.array([0, 1, 0])

    # First perpendicular vector
    u1 = v1 - np.dot(v1, direction) * direction
    u1 = u1 / np.linalg.norm(u1)

    # Second perpendicular vector
    u2 = np.cross(direction, u1)

    return u1, u2


def project_polygon_to_plane(polygon_3d: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Project 3D polygon onto plane perpendicular to direction."""
    u1, u2 = create_orthonormal_basis(direction)

    # Project each vertex
    projected = np.zeros((len(polygon_3d), 2))
    for i, vertex in enumerate(polygon_3d):
        projected[i, 0] = np.dot(vertex, u1)
        projected[i, 1] = np.dot(vertex, u2)

    return projected


def orientation(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> float:
    """Compute orientation of ordered triplet (p, q, r).
    Returns positive if counterclockwise, negative if clockwise, 0 if collinear."""
    return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])


def segments_intersect(p1: np.ndarray, p2: np.ndarray, q1: np.ndarray, q2: np.ndarray) -> bool:
    """Check if line segments (p1, p2) and (q1, q2) intersect."""
    o1 = orientation(p1, p2, q1)
    o2 = orientation(p1, p2, q2)
    o3 = orientation(q1, q2, p1)
    o4 = orientation(q1, q2, p2)

    # General case: segments intersect if endpoints are on opposite sides
    if o1 * o2 < 0 and o3 * o4 < 0:
        return True

    # Special cases (collinear points) - skip for simplicity
    return False


def count_crossings_2d(polygon_2d: np.ndarray) -> int:
    """Count crossings in a 2D polygon projection."""
    n_vertices = len(polygon_2d)
    crossings = 0

    # Check all pairs of non-adjacent edges
    for i in range(n_vertices):
        i_next = (i + 1) % n_vertices

        for j in range(i + 2, n_vertices):
            # Skip if edges are adjacent
            if j == i_next or (i == 0 and j == n_vertices - 1):
                continue

            j_next = (j + 1) % n_vertices

            if segments_intersect(polygon_2d[i], polygon_2d[i_next],
                                polygon_2d[j], polygon_2d[j_next]):
                crossings += 1

    return crossings


def compute_crossing_sign(polygon_3d: np.ndarray, i: int, j: int) -> int:
    """
    Compute the sign of a crossing between segments i and j.

    Based on Klenin & Langowski (2000) Eq. (32b):
    sign = sign((s_j x s_i) . (r_j - r_i))

    where s_i, s_j are segment vectors and r_i, r_j are segment start points.

    Parameters:
    -----------
    polygon_3d : np.ndarray
        3D polygon vertices (N, 3) array
    i, j : int
        Segment indices

    Returns:
    --------
    int : +1 for right-handed crossing, -1 for left-handed, 0 for degenerate
    """
    n_vertices = len(polygon_3d)
    i_next = (i + 1) % n_vertices
    j_next = (j + 1) % n_vertices

    # Segment vectors
    s_i = polygon_3d[i_next] - polygon_3d[i]
    s_j = polygon_3d[j_next] - polygon_3d[j]

    # Vector between segment origins
    r_ij = polygon_3d[j] - polygon_3d[i]

    # Triple product: (s_j x s_i) . r_ij
    cross = np.cross(s_j, s_i)
    triple_product = np.dot(cross, r_ij)

    if triple_product > 0:
        return 1
    elif triple_product < 0:
        return -1
    else:
        return 0  # Degenerate case


def count_crossings_with_signs(polygon_3d: np.ndarray, polygon_2d: np.ndarray) -> Tuple[int, int]:
    """
    Count crossings and compute their signs for writhe calculation.

    Parameters:
    -----------
    polygon_3d : np.ndarray
        3D polygon vertices (N, 3) array
    polygon_2d : np.ndarray
        2D projected polygon vertices (N, 2) array

    Returns:
    --------
    Tuple[int, int] : (unsigned_count, signed_sum)
        - unsigned_count: total number of crossings (for ACN)
        - signed_sum: sum of crossing signs (for writhe)
    """
    n_vertices = len(polygon_2d)
    unsigned_count = 0
    signed_sum = 0

    for i in range(n_vertices):
        i_next = (i + 1) % n_vertices

        for j in range(i + 2, n_vertices):
            if j == i_next or (i == 0 and j == n_vertices - 1):
                continue

            j_next = (j + 1) % n_vertices

            if segments_intersect(polygon_2d[i], polygon_2d[i_next],
                                polygon_2d[j], polygon_2d[j_next]):
                unsigned_count += 1
                sign = compute_crossing_sign(polygon_3d, i, j)
                signed_sum += sign

    return unsigned_count, signed_sum


def is_valid_projection(polygon_2d: np.ndarray, epsilon: float = 1e-10) -> bool:
    """Check if 2D projection produces a valid knot diagram."""
    # Check for zero-length edges
    edges = polygon_2d[1:] - polygon_2d[:-1]
    edges = np.vstack([edges, polygon_2d[0:1] - polygon_2d[-1:]])  # Close the loop
    edge_lengths = np.linalg.norm(edges, axis=1)

    if np.any(edge_lengths < epsilon):
        return False

    return True


def average_crossing_number(polygon_3d: np.ndarray, n_projections: int = 1000) -> Tuple[float, float, float]:
    """
    Compute average crossing number, writhe, and RMS writhe for a 3D polygon.

    All metrics are computed from the same projections for efficiency.
    ACN is the average of unsigned crossing counts.
    Writhe is the average of signed crossing sums (Klenin & Langowski 2000).
    RMS writhe is sqrt(mean(signed_crossings^2)) - measures magnitude of chirality.

    Uses Numba JIT compilation for ~10-50x speedup when available.

    Parameters:
    -----------
    polygon_3d : np.ndarray
        3D polygon vertices (N, 3) array
    n_projections : int
        Number of random projections to average over

    Returns:
    --------
    Tuple[float, float, float] : (acn, writhe, rms_writhe)
        - acn: Average Crossing Number (always >= 0)
        - writhe: Average signed crossing number (can be negative)
        - rms_writhe: Root mean square writhe (always >= 0)
    """
    directions = generate_random_projection_directions(n_projections)

    # Use Numba-optimized parallel version if available
    if NUMBA_AVAILABLE:
        # Ensure contiguous arrays for Numba
        polygon_3d = np.ascontiguousarray(polygon_3d, dtype=np.float64)
        directions = np.ascontiguousarray(directions, dtype=np.float64)

        crossings, writhes, valid = _acn_writhe_parallel(polygon_3d, directions)

        valid_crossings = crossings[valid]
        valid_writhes = writhes[valid]

        if len(valid_crossings) < n_projections * 0.9:
            warnings.warn(f"Only {len(valid_crossings)} valid projections out of {n_projections}")

        acn = np.mean(valid_crossings) if len(valid_crossings) > 0 else np.nan
        writhe = np.mean(valid_writhes) if len(valid_writhes) > 0 else np.nan
        rms_writhe = np.sqrt(np.mean(valid_writhes**2)) if len(valid_writhes) > 0 else np.nan
        return acn, writhe, rms_writhe

    # Fallback to pure Python implementation
    valid_crossings = []
    valid_writhes = []

    for direction in directions:
        projected = project_polygon_to_plane(polygon_3d, direction)

        if is_valid_projection(projected):
            unsigned_count, signed_sum = count_crossings_with_signs(polygon_3d, projected)
            valid_crossings.append(unsigned_count)
            valid_writhes.append(signed_sum)

    if len(valid_crossings) < n_projections * 0.9:
        warnings.warn(f"Only {len(valid_crossings)} valid projections out of {n_projections}")

    acn = np.mean(valid_crossings) if valid_crossings else np.nan
    writhe = np.mean(valid_writhes) if valid_writhes else np.nan
    rms_writhe = np.sqrt(np.mean(np.array(valid_writhes)**2)) if valid_writhes else np.nan
    return acn, writhe, rms_writhe


def radius_of_gyration(vertices: np.ndarray) -> float:
    """
    Compute radius of gyration for a polygon.

    The radius of gyration measures the spatial extent of the polymer chain
    from its center of mass. It is defined as:
    Rg = sqrt(mean(|r_i - r_cm|^2))

    Parameters:
    -----------
    vertices : np.ndarray
        3D polygon vertices (N, 3) array

    Returns:
    --------
    float : Radius of gyration
    """
    center_of_mass = np.mean(vertices, axis=0)
    squared_distances = np.sum((vertices - center_of_mass)**2, axis=1)
    return np.sqrt(np.mean(squared_distances))


def xy_crossing_number(polygon_3d: np.ndarray) -> int:
    """
    Compute crossing number from xy-plane projection.

    This is essentially ACN with a single fixed projection to the xy-plane,
    useful for analyzing planar topology.

    Parameters:
    -----------
    polygon_3d : np.ndarray
        3D polygon vertices (N, 3) array

    Returns:
    --------
    int : Number of crossings in xy-projection (or np.nan if invalid projection)
    """
    # Project to xy-plane (drop z-coordinate)
    polygon_2d = polygon_3d[:, :2]

    # Validate projection
    if not is_valid_projection(polygon_2d):
        return np.nan

    # Count crossings
    return count_crossings_2d(polygon_2d)


def xz_crossing_number(polygon_3d: np.ndarray) -> int:
    """
    Compute crossing number from xz-plane projection.

    Projects 3D curve to xz-plane and counts crossings in that 2D projection.

    Parameters:
    -----------
    polygon_3d : np.ndarray
        3D polygon vertices (N, 3) array

    Returns:
    --------
    int : Number of crossings in xz-projection (or np.nan if invalid projection)
    """
    # Project to xz-plane (drop y-coordinate, keep x and z)
    polygon_2d = polygon_3d[:, [0, 2]]

    # Validate projection
    if not is_valid_projection(polygon_2d):
        return np.nan

    # Count crossings
    return count_crossings_2d(polygon_2d)


def yz_crossing_number(polygon_3d: np.ndarray) -> int:
    """
    Compute crossing number from yz-plane projection.

    Projects 3D curve to yz-plane and counts crossings in that 2D projection.

    Parameters:
    -----------
    polygon_3d : np.ndarray
        3D polygon vertices (N, 3) array

    Returns:
    --------
    int : Number of crossings in yz-projection (or np.nan if invalid projection)
    """
    # Project to yz-plane (drop x-coordinate, keep y and z)
    polygon_2d = polygon_3d[:, [1, 2]]

    # Validate projection
    if not is_valid_projection(polygon_2d):
        return np.nan

    # Count crossings
    return count_crossings_2d(polygon_2d)


# ============================================================================
# Kendall Shape Space Implementation
# ============================================================================

class KendallIsogonAverager:
    """
    Compute average isogon from a family of 20-sided PL isogons using Kendall shape space.

    Input: List of n isogons, each as a 20×3 array (or 20×2 for 2D)
    Output: Average isogon as a 20×3 array (or 20×2 for 2D)
    """

    def __init__(self, dim=3):
        """
        Initialize the averager.

        Parameters:
        -----------
        dim : int
            Dimension of the space (2 or 3)
        """
        self.dim = dim

    def center_isogon(self, X):
        """
        Center an isogon by removing its centroid.

        Parameters:
        -----------
        X : np.ndarray, shape (20, 3) or (20, 2)
            Isogon vertices

        Returns:
        --------
        np.ndarray, shape (20, 3) or (20, 2)
            Centered isogon
        """
        return X - np.mean(X, axis=0)

    def to_preshape(self, X):
        """
        Convert isogon to preshape (centered and unit norm).

        Parameters:
        -----------
        X : np.ndarray, shape (20, 3) or (20, 2)
            Isogon vertices

        Returns:
        --------
        np.ndarray, shape (20, 3) or (20, 2)
            Preshape with unit Frobenius norm
        """
        X_centered = self.center_isogon(X)
        norm = np.linalg.norm(X_centered, 'fro')
        if norm > 0:
            return X_centered / norm
        return X_centered

    def align_to_reference(self, X, reference):
        """
        Align isogon X to reference using Procrustes.

        Parameters:
        -----------
        X : np.ndarray, shape (20, 3) or (20, 2)
            Isogon to align
        reference : np.ndarray, shape (20, 3) or (20, 2)
            Reference isogon

        Returns:
        --------
        aligned : np.ndarray, shape (20, 3) or (20, 2)
            Aligned isogon
        R : np.ndarray, shape (3, 3) or (2, 2)
            Rotation matrix used
        """
        Z = self.to_preshape(X)
        Z_ref = self.to_preshape(reference)

        # Find optimal rotation
        R, _ = orthogonal_procrustes(Z, Z_ref)

        # Apply rotation
        aligned = Z @ R

        return aligned, R

    def compute_average(self, isogons, max_iter=100, tol=1e-6, return_scale=False):
        """
        Compute the average isogon using Kendall shape space.

        Parameters:
        -----------
        isogons : list of np.ndarray or np.ndarray
            Input isogons. Can be:
            - List of n arrays, each shape (20, 3) or (20, 2)
            - Single array of shape (n, 20, 3) or (n, 20, 2)
        max_iter : int
            Maximum iterations for convergence
        tol : float
            Convergence tolerance
        return_scale : bool
            If True, return average with a canonical scale (default: False returns unit norm)

        Returns:
        --------
        average_isogon : np.ndarray, shape (20, 3) or (20, 2)
            Average isogon in Kendall shape space
        converged : bool
            Whether the algorithm converged
        """
        # Convert input to consistent format
        if isinstance(isogons, list):
            isogons = np.array(isogons)

        # Check dimensions
        if isogons.ndim == 2:
            # Single isogon provided
            isogons = isogons[np.newaxis, :]

        n_isogons, n_vertices, dim = isogons.shape

        assert n_vertices == 20, f"Expected 20 vertices, got {n_vertices}"
        assert dim in [2, 3], f"Expected 2D or 3D coordinates, got {dim}D"

        # Convert all to preshapes
        preshapes = np.array([self.to_preshape(X) for X in isogons])

        # Initialize with first preshape
        mean_shape = preshapes[0].copy()

        # Iterative Fréchet mean computation
        for iteration in range(max_iter):
            old_mean = mean_shape.copy()

            # Align all shapes to current mean
            aligned_sum = np.zeros_like(mean_shape)
            for i in range(n_isogons):
                aligned, _ = self.align_to_reference(preshapes[i], mean_shape)
                aligned_sum += aligned

            # Update mean (project back to preshape)
            mean_shape = self.to_preshape(aligned_sum)

            # Check convergence
            change = np.linalg.norm(mean_shape - old_mean, 'fro')

            if change < tol:
                break

        # Optionally rescale to a canonical size
        if return_scale:
            # Use average centroid size of input isogons as scale
            scales = []
            for isogon in isogons:
                centered = self.center_isogon(isogon)
                scales.append(np.linalg.norm(centered, 'fro'))
            avg_scale = np.mean(scales)
            mean_shape = mean_shape * avg_scale

        converged = iteration < max_iter - 1
        return mean_shape, converged, iteration + 1

    def procrustes_distance(self, X1, X2):
        """
        Compute Procrustes distance between two isogons.

        Parameters:
        -----------
        X1, X2 : np.ndarray, shape (20, 3) or (20, 2)
            Two isogons

        Returns:
        --------
        float
            Procrustes distance in shape space
        """
        Z1 = self.to_preshape(X1)
        Z2 = self.to_preshape(X2)

        # Compute the singular values of Z1^T @ Z2
        singular_values = np.linalg.svd(Z1.T @ Z2, compute_uv=False)

        # Full Procrustes distance (geodesic distance on shape manifold)
        distance = np.arccos(np.clip(np.sum(singular_values), -1, 1))

        return distance


# ============================================================================
# Weighted Centroid Computation
# ============================================================================

def weighted_average_and_precision(group):
    """
    Compute weighted average centroid and combined precision for a group of points.
    From knotDetector_optimized.py
    """
    group['precisionx'] = group['precisionx'].replace(0, 1e-10)
    group['precisiony'] = group['precisiony'].replace(0, 1e-10)
    group['precisionz'] = group['precisionz'].replace(0, 1e-10)

    weights_x = 1 / group['precisionx']**2
    weights_y = 1 / group['precisiony']**2
    weights_z = 1 / group['precisionz']**2

    centroid_x = np.average(group['x'], weights=weights_x)
    centroid_y = np.average(group['y'], weights=weights_y)
    centroid_z = np.average(group['z'], weights=weights_z)

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


def compute_weighted_centroids(walk_data):
    """
    Compute weighted centroid curve for a single walk.

    Parameters:
    -----------
    walk_data : pd.DataFrame
        Walk data with columns: x, y, z, precisionx, precisiony, precisionz, time-point

    Returns:
    --------
    np.ndarray, shape (20, 3)
        Weighted centroid curve (20 time-points)
    """
    # Group by time-point and compute weighted average
    centroids = walk_data.groupby('time-point').apply(weighted_average_and_precision).reset_index()

    # Extract coordinates
    centroid_curve = centroids[['weighted_centroid_x', 'weighted_centroid_y', 'weighted_centroid_z']].values

    return centroid_curve


def load_morans_i_data(filepath):
    """
    Load Moran's I data from CSV file.

    Parameters:
    -----------
    filepath : str
        Path to Moran's I results CSV file

    Returns:
    --------
    dict
        Dictionary mapping walk_id to dict with 'I_value' and 'P_value'
    """
    if not os.path.exists(filepath):
        print(f"Warning: Moran's I file not found: {filepath}")
        return {}

    try:
        df = pd.read_csv(filepath)
        # Expected header: Filename,Walk-ID,Time-points,I-value,P-value
        # We need columns 1 (Walk-ID), 3 (I-value), and 4 (P-value)

        if 'Walk-ID' in df.columns and 'I-value' in df.columns and 'P-value' in df.columns:
            morans_dict = {
                row['Walk-ID']: {
                    'I_value': row['I-value'],
                    'P_value': row['P-value']
                }
                for _, row in df.iterrows()
            }
            print(f"Loaded Moran's I data for {len(morans_dict)} walks")
        else:
            # Fallback: use column indices (1=Walk-ID, 3=I-value, 4=P-value)
            print(f"Note: Column names not found, using indices [1, 3, 4] as (Walk-ID, I-value, P-value)")
            col_names = df.columns.tolist()
            if len(col_names) >= 5:
                morans_dict = {
                    row[col_names[1]]: {
                        'I_value': row[col_names[3]],
                        'P_value': row[col_names[4]]
                    }
                    for _, row in df.iterrows()
                }
                print(f"Loaded Moran's I data for {len(morans_dict)} walks")
            else:
                print(f"Error: Expected at least 5 columns, got {len(col_names)}")
                return {}

        return morans_dict
    except Exception as e:
        print(f"Error loading Moran's I data: {e}")
        return {}


def get_all_component_codes(analyzer):
    """
    Collect PD and Gauss codes from all components of a knot.
    Returns lists of PD codes and Gauss codes for all components.
    """
    all_pd_codes = []
    all_gauss_codes = []

    # Add all split-off prime components
    for comp in analyzer.prime_components:
        all_pd_codes.append(comp.pd_code)
        all_gauss_codes.append(comp.gauss_code)

    # Add the remainder if it's non-trivial
    if analyzer.crossing_count > 0:
        all_pd_codes.append(analyzer.pd_code)
        all_gauss_codes.append(analyzer.gauss_code)
    elif analyzer.is_prime and not analyzer.is_unknot():
        # For prime knots, the main analyzer has the only component
        all_pd_codes.append(analyzer.pd_code)
        all_gauss_codes.append(analyzer.gauss_code)

    return all_pd_codes, all_gauss_codes


def calculate_knot_properties(curve):
    """
    Calculate knot type and determinant for a given curve.
    Returns: (knot_name, determinant)
    """
    try:
        # Calculate knot type using KnotoID (suppress output with wurlitzer if available)
        if WURLITZER_AVAILABLE:
            with pipes() as (out, err):
                k = KnotoID(projections_list='internal', closure_method='direct')
                k.analyze_curve(curve)
        else:
            k = KnotoID(projections_list='internal', closure_method='direct')
            k.analyze_curve(curve)

        rez = k.curve_analysis
        knot_name = rez.loc[0, 'Name'] if len(rez) > 0 else 'Unknown'

        # Flatten coordinates for knoodle
        coords = [float(coord) for point in curve for coord in point]

        # Calculate determinant using knoodle
        analyzer = knoodle.KnotAnalyzer(coords, simplify=True, simplify_level=5)

        # Get all component codes
        all_pd_codes, all_gauss_codes = get_all_component_codes(analyzer)

        # Calculate total crossings
        if analyzer.is_composite and analyzer.prime_components:
            total_crossings = sum(comp.crossing_count for comp in analyzer.prime_components)
            if analyzer.crossing_count > 0:
                total_crossings += analyzer.crossing_count
        else:
            total_crossings = analyzer.crossing_count

        # Calculate Alexander polynomial at -1
        alexander_result = analyzer.alexander([-1.0])[0]
        alexander_value = alexander_result.to_double()

        # Determinant is absolute value of Alexander polynomial at -1
        determinant = abs(alexander_value)

        # Handle floating-point precision issues
        rounded_det = round(determinant)
        if abs(determinant - rounded_det) < 1e-10:
            determinant = float(rounded_det)

        return knot_name, determinant

    except Exception as e:
        print(f"Error calculating knot properties: {e}")
        return 'Error', 0.0


def load_and_preprocess_data(filepath):
    """
    Load and preprocess the SMLM data.
    Returns: Dictionary with Walk-ID as keys and dataframes as values (only complete walks with all 20 time-points)
    """
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)

    # Filter for time-points 0-19 (0-indexed)
    df = df[df['time-point'] <= 19]
    df = df[df['time-point'] >= 0]

    # Group by Walk-ID
    walk_groups = {}
    for walk_id, group in df.groupby('Walk-IDs'):
        # Sort by time-point
        group = group.sort_values('time-point')
        walk_groups[walk_id] = group

    print(f"Found {len(walk_groups)} Walk-IDs")

    # Filter to only include walks with all 20 time-points
    complete_walks = {}
    incomplete_walks = []
    for walk_id, group in walk_groups.items():
        unique_timepoints = group['time-point'].unique()
        if len(unique_timepoints) == 20:
            complete_walks[walk_id] = group
        else:
            incomplete_walks.append((walk_id, len(unique_timepoints)))

    print(f"Complete walks (20 time-points): {len(complete_walks)}")
    if incomplete_walks:
        print(f"Incomplete walks excluded: {len(incomplete_walks)}")
        for walk_id, n_points in incomplete_walks[:5]:  # Show first 5
            print(f"  Walk-ID {walk_id}: {n_points} time-points (excluded)")

    print(f"Using {len(complete_walks)} complete walks for analysis")

    return complete_walks


def sample_trajectory_from_walk(walk_data, n_iterations, randomize_order=False):
    """
    Sample trajectories from a single Walk-ID.
    For each iteration, randomly select one point from each time-point.

    Parameters
    ----------
    walk_data : pd.DataFrame
        Localizations for one walk.
    n_iterations : int
        Number of De Witt curves to sample.
    randomize_order : bool, default False
        When True, shuffle the order of the 20 sampled vertices before
        computing the knot type (control for random rather than genomic
        connection order).

    Returns: List of (curve, knot_name, determinant) tuples
             where curve is np.ndarray of shape (20, 3)
    """
    results = []

    for _ in range(n_iterations):
        trajectory = []

        # For each time-point from 0 to 19 (0-indexed)
        for tp in range(0, 20):
            tp_data = walk_data[walk_data['time-point'] == tp]

            if len(tp_data) == 0:
                # If time-point is missing, skip this iteration
                trajectory = None
                break

            # Randomly select one point from this time-point
            selected = tp_data.sample(n=1).iloc[0]
            point = [selected['x'], selected['y'], selected['z']]
            trajectory.append(point)

        if trajectory is not None and len(trajectory) == 20:
            # Convert to numpy array
            curve = np.array(trajectory)
            if randomize_order:
                curve = curve[np.random.permutation(20)]
            # Scale to nm if needed (data appears to be in nm already based on the file)
            # curve = curve * 1e-9  # Convert to meters for consistency with other code

            # Calculate knot properties
            knot_name, determinant = calculate_knot_properties(curve)
            # Store curve along with knot properties for reuse in ACN, Kendall, etc.
            results.append((curve, knot_name, determinant))

    return results


def is_inside_convex_hull(points, A, b):
    """Check if points are inside convex hull defined by Ax + b <= 0"""
    return np.all(np.dot(points, A.T) + b <= 1e-12, axis=1)


def generate_random_points_in_convex_hull(convex_hull, num_points):
    """Generate random points uniformly distributed within a convex hull."""
    A = convex_hull.equations[:, :-1]
    b = convex_hull.equations[:, -1]

    min_coords = np.min(convex_hull.points, axis=0)
    max_coords = np.max(convex_hull.points, axis=0)
    random_points = []
    generated = 0
    batch_size = max(num_points * 2, 1000)

    while generated < num_points:
        batch = np.random.uniform(min_coords, max_coords, size=(batch_size, 3))
        inside = is_inside_convex_hull(batch, A, b)
        inside_points = batch[inside]
        remaining = num_points - generated

        if inside_points.shape[0] >= remaining:
            random_points.append(inside_points[:remaining])
            generated += remaining
        else:
            random_points.append(inside_points)
            generated += inside_points.shape[0]

    return np.vstack(random_points)


def compute_average_convex_hull(polygons):
    """
    Compute an 'average' convex hull from multiple polygons.
    This creates a convex hull that encompasses the typical spatial extent of the polygons.
    """
    # Collect all points from all polygons
    all_points = []
    for polygon in polygons:
        all_points.extend(polygon)

    all_points = np.array(all_points)

    # Compute percentiles to get a robust estimate of the spatial extent
    # Use 5th and 95th percentiles to avoid outliers
    percentile_low = np.percentile(all_points, 5, axis=0)
    percentile_high = np.percentile(all_points, 95, axis=0)

    # Create a box with these bounds and some representative interior points
    box_points = []

    # Add corners of the bounding box
    for x in [percentile_low[0], percentile_high[0]]:
        for y in [percentile_low[1], percentile_high[1]]:
            for z in [percentile_low[2], percentile_high[2]]:
                box_points.append([x, y, z])

    # Add some random sample points from the actual data
    sample_size = min(1000, len(all_points))
    sample_indices = np.random.choice(len(all_points), sample_size, replace=False)
    box_points.extend(all_points[sample_indices])

    box_points = np.array(box_points)

    # Compute convex hull
    try:
        hull = ConvexHull(box_points)
        return hull
    except QhullError as e:
        print(f"Warning: Could not compute convex hull: {e}")
        # Fallback: return a simple box
        return None


def generate_random_polygon_in_hull(convex_hull, n_vertices=20):
    """
    Generate a random polygon with vertices inside the given convex hull.
    """
    if convex_hull is None:
        # Fallback to box generation if hull computation failed
        box_size = 1e-6
        points = np.random.uniform(-box_size/2, box_size/2, (n_vertices, 3))
        return points

    # Generate n_vertices random points inside the convex hull
    points = generate_random_points_in_convex_hull(convex_hull, n_vertices)

    return points


def generate_random_polygons(n_polygons, n_vertices=20, convex_hull=None):
    """
    Generate multiple random polygons and calculate their knot properties.
    If convex_hull is provided, generate polygons within that hull.
    """
    if convex_hull is not None:
        print(f"Generating {n_polygons} random {n_vertices}-gons within convex hull...")
    else:
        print(f"Generating {n_polygons} random {n_vertices}-gons in a box...")

    results = []

    for _ in tqdm(range(n_polygons), desc="Random polygons"):
        if convex_hull is not None:
            polygon = generate_random_polygon_in_hull(convex_hull, n_vertices)
        else:
            # Fallback to box generation
            box_size = 1e-6
            polygon = np.random.uniform(-box_size/2, box_size/2, (n_vertices, 3))

        knot_name, determinant = calculate_knot_properties(polygon)
        results.append((knot_name, determinant))

    return results


def calculate_kl_divergence(p_dist, q_dist, categories=['0_1', '3_1', 'Complex']):
    """
    Calculate KL divergence between two distributions.
    p_dist: distribution from Walk-ID (list of knot names)
    q_dist: distribution from random polygons (list of knot names)
    """
    # Classify knots into categories
    p_classified = [classify_name(name) for name, _ in p_dist]
    q_classified = [classify_name(name) for name, _ in q_dist]

    # Count occurrences
    p_counts = Counter(p_classified)
    q_counts = Counter(q_classified)

    # Create probability distributions
    p_probs = np.array([p_counts.get(cat, 0) for cat in categories])
    q_probs = np.array([q_counts.get(cat, 0) for cat in categories])

    # Normalize
    p_probs = p_probs / (p_probs.sum() + 1e-10)
    q_probs = q_probs / (q_probs.sum() + 1e-10)

    # Add small epsilon to avoid log(0)
    epsilon = 1e-10
    p_probs = p_probs + epsilon
    q_probs = q_probs + epsilon

    # Calculate KL divergence
    kl_div = entropy(p_probs, q_probs)

    return kl_div, p_probs, q_probs


def calculate_emd(p_dist, q_dist):
    """
    Calculate Wasserstein distance between knot type distributions.
    Uses |det(K1) - det(K2)| as the ground metric between knot types.

    Args:
        p_dist: List of (knot_name, determinant) tuples from distribution P
        q_dist: List of (knot_name, determinant) tuples from distribution Q

    Returns:
        Wasserstein distance between the two knot distributions
    """
    try:
        # Try to use POT (Python Optimal Transport) for exact computation
        import ot

        # Get all unique knot types from both distributions
        all_knots = list(set([name for name, _ in p_dist] + [name for name, _ in q_dist]))

        # Create determinant mapping (use first occurrence of each knot type)
        knot_to_det = {}
        for name, det in p_dist + q_dist:
            if name not in knot_to_det:
                knot_to_det[name] = det

        # Sort knots for consistent ordering
        all_knots.sort()
        n_knots = len(all_knots)

        # Create probability distributions over knot types
        p_counts = Counter([name for name, _ in p_dist])
        q_counts = Counter([name for name, _ in q_dist])

        p_probs = np.array([p_counts.get(knot, 0) / len(p_dist) for knot in all_knots])
        q_probs = np.array([q_counts.get(knot, 0) / len(q_dist) for knot in all_knots])

        # Create cost matrix based on determinant differences
        cost_matrix = np.zeros((n_knots, n_knots))
        for i, knot1 in enumerate(all_knots):
            for j, knot2 in enumerate(all_knots):
                cost_matrix[i, j] = abs(knot_to_det[knot1] - knot_to_det[knot2])

        # Compute Wasserstein distance using optimal transport
        emd = ot.emd2(p_probs, q_probs, cost_matrix)

        return float(emd)

    except ImportError:
        # Fallback: Use scipy with determinant values
        # This groups knots by determinant value, which is less precise
        # but still uses the correct ground metric

        # Get all unique determinant values from both distributions
        all_dets = sorted(set([det for _, det in p_dist] + [det for _, det in q_dist]))

        # Create histograms over determinant values
        p_det_counts = Counter([det for _, det in p_dist])
        q_det_counts = Counter([det for _, det in q_dist])

        # Convert to probability arrays
        p_weights = np.array([p_det_counts.get(det, 0) / len(p_dist) for det in all_dets])
        q_weights = np.array([q_det_counts.get(det, 0) / len(q_dist) for det in all_dets])

        # Use scipy's wasserstein_distance with explicit weights
        # The determinant values serve as the "positions" and their differences as the metric
        emd = wasserstein_distance(all_dets, all_dets, p_weights, q_weights)

        return float(emd)


def extract_knot_properties(results):
    """
    Extract (knot_name, determinant) tuples from results that may contain curves.

    Parameters:
    -----------
    results : list
        List of either (curve, name, det) or (name, det) tuples

    Returns:
    --------
    list of (name, det) tuples
    """
    if not results:
        return []

    # Check first item to determine format
    first_item = results[0]
    if len(first_item) == 3:
        # Format: (curve, name, det) - extract name and det
        return [(name, det) for curve, name, det in results]
    elif len(first_item) == 2:
        # Format: (name, det) - already in correct format
        return results
    else:
        raise ValueError(f"Unexpected tuple length: {len(first_item)}")


def validate_knot_results(results, label="data"):
    """
    Validate that knot results have the correct structure: list of (curve, knot_name, determinant) tuples.

    Parameters:
    -----------
    results : any
        Data structure to validate
    label : str
        Label for error messages

    Returns:
    --------
    tuple : (is_valid, error_message)
        is_valid is True if structure is correct, False otherwise
        error_message describes the issue if invalid
    """
    # Check if it's a list or similar iterable (but not string)
    if isinstance(results, str):
        return False, f"{label}: Expected list, got string"

    try:
        # Try to convert to list if it's an iterable
        results_list = list(results) if not isinstance(results, list) else results
    except (TypeError, ValueError) as e:
        return False, f"{label}: Cannot convert to list: {e}"

    # Check if empty
    if len(results_list) == 0:
        return True, ""  # Empty list is valid, just has no data

    # Check ALL items for structure (not just first 5)
    for i, item in enumerate(results_list):
        # Check if item is a tuple/list with exactly 3 elements
        try:
            if not (isinstance(item, (tuple, list)) and len(item) == 3):
                return False, f"{label}: Item {i} has {len(item)} elements (expected 3): {item}"

            # Try to unpack to verify structure
            curve, name, det = item

            # Verify types are reasonable
            if not isinstance(curve, np.ndarray):
                return False, f"{label}: Item {i} has non-array curve: {type(curve)}"
            if curve.shape != (20, 3):
                return False, f"{label}: Item {i} curve has wrong shape: {curve.shape} (expected (20, 3))"
            if not isinstance(name, str):
                return False, f"{label}: Item {i} has non-string knot name: {type(name)}"
            if not isinstance(det, (int, float, np.integer, np.floating)):
                return False, f"{label}: Item {i} has non-numeric determinant: {type(det)}"

        except (TypeError, ValueError) as e:
            return False, f"{label}: Item {i} unpacking failed: {e}, item={item}"

    return True, ""


def create_single_walk_stacked_bar(walk_id, walk_results, controls, ax, metrics='all', show_legend=False,
                                   no_stacked=False):
    """
    Create a stacked bar plot for a single walk-id showing knot type distributions
    compared to all control distributions.

    Args:
        walk_id: ID of the walk to plot
        walk_results: Dictionary of all walk results
        controls: Dictionary of control results {label: results}
        ax: Matplotlib axis to plot on
        metrics: Which metrics to show ('all', 'kl', 'wasserstein')
        show_legend: Whether to show legend on this plot
        no_stacked: If True, only plot the walk's distribution as two side-by-side
            bars (Unknotted, Knotted) with percentages above each bar. Controls are
            omitted from the plot.
    """
    # Define colors and categories
    colors = {
        '0_1': '#2ecc71',      # Green for unknots
        '3_1': '#ff9800',      # Orange for trefoils
        'Complex': '#e74c3c'   # Red for complex knots
    }
    categories = ['0_1', '3_1', 'Complex']

    # Validate walk data first
    if walk_id not in walk_results:
        raise ValueError(f"Walk-ID {walk_id} not found in walk_results")

    walk_data = walk_results[walk_id]
    is_valid, error_msg = validate_knot_results(walk_data, f"Walk-ID {walk_id}")
    if not is_valid:
        raise ValueError(f"Invalid walk_data structure for Walk-ID {walk_id}: {error_msg}\n"
                        f"Data type: {type(walk_data)}, Data: {walk_data[:3] if len(walk_data) > 0 else walk_data}")

    if no_stacked:
        # Simple two-bar plot: Unknotted vs Knotted, walk only
        walk_knot_props = extract_knot_properties(walk_data)
        walk_classified = [classify_name(name) for name, _ in walk_knot_props]
        n_total = len(walk_classified)
        n_unknot = sum(1 for c in walk_classified if c == '0_1')
        n_knotted = n_total - n_unknot
        pct_unknot = 100.0 * n_unknot / n_total if n_total > 0 else 0.0
        pct_knotted = 100.0 * n_knotted / n_total if n_total > 0 else 0.0

        bar_x = np.arange(2)
        bar_heights = [pct_unknot, pct_knotted]
        ax.bar(bar_x, bar_heights, width=0.6, color='C0',
               edgecolor='black', linewidth=1.0)

        for xi, h in zip(bar_x, bar_heights):
            ax.text(xi, h + 1.5, f'{h:.1f}%', ha='center', va='bottom',
                    fontweight='bold', fontsize=10)

        ax.set_xticks(bar_x)
        ax.set_xticklabels(['Unknotted', 'Knotted'], fontsize=10)
        ax.set_ylabel('Percentage (%)', fontsize=10)
        ax.set_ylim(0, 110)
        ax.set_title(f'Walk-ID {walk_id}: Knot Type Distribution', fontweight='bold', fontsize=11)
        ax.grid(True, axis='y', alpha=0.2)
        ax.set_axisbelow(True)
        return

    # Calculate control distributions
    control_probs = {}
    control_labels = list(controls.keys())
    for label, control_results in controls.items():
        # Extract knot properties (handles both 2-tuple and 3-tuple formats)
        control_knot_props = extract_knot_properties(control_results)

        try:
            control_classified = [classify_name(name) for name, _ in control_knot_props]
        except (ValueError, TypeError) as e:
            raise ValueError(f"Error unpacking control '{label}' for Walk-ID {walk_id}: {e}\n"
                            f"First item: {control_results[0] if len(control_results) > 0 else 'empty'}")

        control_counts = Counter(control_classified)
        control_probs[label] = {cat: control_counts.get(cat, 0) / len(control_classified) for cat in categories}

    # Calculate walk distribution
    # Extract knot properties from walk data (removes curves if present)
    walk_knot_props = extract_knot_properties(walk_data)

    try:
        walk_classified = [classify_name(name) for name, _ in walk_knot_props]
    except (ValueError, TypeError) as e:
        raise ValueError(f"Error unpacking walk_data for Walk-ID {walk_id}: {e}\n"
                        f"Data type: {type(walk_data)}, First item: {walk_data[0] if len(walk_data) > 0 else 'empty'}")
    walk_counts = Counter(walk_classified)
    walk_probs = {cat: walk_counts.get(cat, 0) / len(walk_classified) for cat in categories}

    # Calculate metrics
    walk_stats = {}
    for control_label, control_results in controls.items():
        # Extract knot properties for metric calculations
        control_knot_props = extract_knot_properties(control_results)

        if metrics in ['all', 'kl']:
            kl_div, _, _ = calculate_kl_divergence(walk_knot_props, control_knot_props)
            walk_stats[f'kl_div_{control_label}'] = kl_div

        if metrics in ['all', 'wasserstein']:
            emd = calculate_emd(walk_knot_props, control_knot_props)
            walk_stats[f'emd_{control_label}'] = emd

    # Create stacked bars for walk and all controls
    n_controls = len(controls)
    n_bars = 1 + n_controls  # 1 for walk, rest for controls
    x = np.arange(n_bars)
    width = 0.6
    bottoms = [0] * n_bars

    # Plot stacked bars for each category
    for cat in categories:
        heights = []

        # Walk bar
        heights.append(walk_probs[cat])

        # Control bars
        for control_label in control_labels:
            heights.append(control_probs[control_label][cat])

        # Plot all bars for this category
        for i, height in enumerate(heights):
            ax.bar(x[i], height, width, bottom=bottoms[i],
                  color=colors[cat], edgecolor='black', linewidth=0.5)
            bottoms[i] += height

    # Customize subplot
    ax.set_ylim([0, 1])
    ax.set_xlim([-0.5, n_bars - 0.5])
    ax.set_xticks(x)

    # Create x-tick labels
    xlabels = [f'Walk-{walk_id}'] + [f'{label}' for label in control_labels]
    ax.set_xticklabels(xlabels, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Proportion', fontsize=10)
    ax.set_title(f'Walk-ID {walk_id}: Knot Type Distribution', fontweight='bold', fontsize=11)

    # Add statistics text in lower right
    stats_text_lines = []
    for control_label in control_labels:
        if metrics in ['all', 'kl']:
            kl_key = f'kl_div_{control_label}'
            if kl_key in walk_stats:
                stats_text_lines.append(f'KL vs {control_label}: {walk_stats[kl_key]:.4f}')

        if metrics in ['all', 'wasserstein']:
            emd_key = f'emd_{control_label}'
            if emd_key in walk_stats:
                stats_text_lines.append(f'Wasserstein vs {control_label}: {walk_stats[emd_key]:.4f}')

    textstr = '\n'.join(stats_text_lines)
    ax.text(0.98, 0.02, textstr, transform=ax.transAxes,
           fontsize=7, verticalalignment='bottom', horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Add legend if requested
    if show_legend:
        legend_elements = [plt.Rectangle((0, 0), 1, 1, fc=colors[cat], edgecolor='black')
                         for cat in categories]
        legend_labels = ['Unknot (0_1)', 'Trefoil (3_1)', 'Complex']
        ax.legend(legend_elements, legend_labels, loc='lower left', fontsize=8)


def create_kendall_single_panel(walk_id, walk_kendall_data, ax):
    """
    Create a Kendall isogon plot for a single walk-id on a provided 3D axis.

    Args:
        walk_id: ID of the walk to plot
        walk_kendall_data: Dictionary containing Kendall data for this walk
        ax: Matplotlib 3D axis to plot on (must have projection='3d')
    """
    if walk_id not in walk_kendall_data:
        ax.text(0.5, 0.5, 0.5, 'No Kendall data', ha='center', va='center')
        ax.set_title(f'Walk-ID {walk_id}', fontsize=11, fontweight='bold')
        return

    data = walk_kendall_data[walk_id]
    average_isogon = data['average_isogon']

    # Plot the isogon
    xs = average_isogon[:, 0]
    ys = average_isogon[:, 1]
    zs = average_isogon[:, 2]

    # Plot vertices with custom colormap (rainbow gradient for time-points)
    ax.scatter(xs, ys, zs, c=np.arange(20), cmap=CUSTOM_TIMEPOINT_CMAP, s=30, alpha=0.8, rasterized=True)

    # Plot edges
    for i in range(20):
        j = (i + 1) % 20
        ax.plot([xs[i], xs[j]], [ys[i], ys[j]], [zs[i], zs[j]],
               'k-', alpha=0.6, linewidth=1.2)

    # Labels
    ax.set_xlabel('X', fontsize=8)
    ax.set_ylabel('Y', fontsize=8)
    ax.set_zlabel('Z', fontsize=8)
    ax.set_title(f'Walk-ID {walk_id}: Kendall Icosagon', fontsize=11, fontweight='bold')

    # Statistics text
    stats_text = f"Knot: {data['knot_type']}\n"
    stats_text += f"Det: {data['determinant']:.1f}\n"
    stats_text += f"Mean Proc. Dist: {data['mean_procrustes_dist']:.3f}\n"
    stats_text += f"Converged: {data['converged']}"

    # Add text annotation
    ax.text2D(0.02, 0.98, stats_text, transform=ax.transAxes,
             fontsize=7, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))


def create_morans_info_panel(walk_id, morans_i_dict, distance, ax):
    """
    Create an info panel displaying Moran's I statistics for a single walk-id.

    Args:
        walk_id: ID of the walk
        morans_i_dict: Dictionary with walk_id as key and dict containing I-value and P-value
        distance: Not used (kept for compatibility)
        ax: Matplotlib axis to use for text display
    """
    # Turn off axis
    ax.axis('off')

    # Check if Moran's I data exists for this walk
    if walk_id in morans_i_dict:
        i_value = morans_i_dict[walk_id]['I_value']
        p_value = morans_i_dict[walk_id]['P_value']

        info_text = f"Walk-ID: {walk_id}\n\n"
        info_text += f"Moran's I: {i_value:.4f}"
        # info_text += f"P-value: {p_value:.4f}"
    else:
        info_text = f"Walk-ID: {walk_id}\n\n"
        info_text += "Moran's I: N/A"
        # info_text += "P-value: N/A"

    # Display text in center of panel
    ax.text(0.5, 0.5, info_text, transform=ax.transAxes,
           fontsize=12, ha='center', va='center',
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8, pad=1.5),
           family='monospace')

    ax.set_title('Spatial Statistics', fontsize=11, fontweight='bold', pad=10)


def create_3d_trajectory_panel(walk_id, walk_data, morans_i_dict, centroid_knot_types, ax):
    """
    Create a 3D scatter plot of raw trajectory localizations colored by time-point.

    Args:
        walk_id: ID of the walk
        walk_data: DataFrame with columns ['time-point', 'x', 'y', 'z']
        morans_i_dict: Dictionary with Moran's I values
        centroid_knot_types: Dictionary mapping walk_id to (knot_name, determinant) tuples
        ax: Matplotlib 3D axis
    """
    # Get Moran's I value
    if walk_id in morans_i_dict:
        i_value = morans_i_dict[walk_id]['I_value']
        morans_text = f"Moran's I = {i_value:.4f}"
    else:
        morans_text = "Moran's I = N/A"

    # Get centroid knot type
    if centroid_knot_types and walk_id in centroid_knot_types:
        knot_name, determinant = centroid_knot_types[walk_id]
        centroid_text = f"Centroid: {knot_name}"
    else:
        centroid_text = "Centroid: N/A"

    # Build title with both Moran's I and centroid knot type
    title_text = f"Walk-ID {walk_id}: 3D Trajectory ({morans_text}, {centroid_text})"

    # Plot each time-point with its corresponding color
    for tp in range(0, 20):
        tp_data = walk_data[walk_data['time-point'] == tp]

        if len(tp_data) > 0:
            # Get color from custom colormap (0-indexed)
            color = CUSTOM_20_COLORS_HEX[tp]

            # Plot all localizations for this time-point
            ax.scatter(tp_data['x'].values,
                      tp_data['y'].values,
                      tp_data['z'].values,
                      c=color,
                      s=20,
                      alpha=0.7,
                      label=f'TP{tp+1}' if tp in [0, 4, 9, 14, 19] else '',  # Only label some for cleaner legend (show as 1-indexed)
                      rasterized=True)

    # Set labels
    ax.set_xlabel('X (nm)', fontsize=8)
    ax.set_ylabel('Y (nm)', fontsize=8)
    ax.set_zlabel('Z (nm)', fontsize=8)
    ax.set_title(title_text, fontsize=11, fontweight='bold')

    # Add legend for selected time-points
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc='upper left', fontsize=6, framealpha=0.8)

    # Set viewing angle for better visualization
    ax.view_init(elev=20, azim=45)

    # Grid
    ax.grid(True, alpha=0.3)


def create_puknot_panel(walk_id, walk_results, centroid_knot_types, ax):
    """
    Create a P(unknot) convergence plot showing how the running fraction of unknots
    stabilizes as the number of DeWitt curves (N) grows.

    Args:
        walk_id: ID of the walk
        walk_results: Dictionary of all walk results
        centroid_knot_types: Dictionary mapping walk_id to (knot_name, determinant) tuples
        ax: matplotlib axes to plot on
    """
    walk_data = walk_results[walk_id]
    # walk_data is a list of (curve, knot_name, determinant) 3-tuples
    N = len(walk_data)
    if N == 0:
        ax.text(0.5, 0.5, 'No DeWitt curves', ha='center', va='center',
                transform=ax.transAxes, fontsize=11)
        ax.set_title(f'Walk-ID {walk_id}: P(unknot)', fontsize=11, fontweight='bold')
        return

    # Compute running P(unknot)
    running_punknot = []
    unknot_count = 0
    for i, item in enumerate(walk_data):
        knot_name = item[1]
        if knot_name == '0_1':
            unknot_count += 1
        running_punknot.append(unknot_count / (i + 1))

    n_values = np.arange(1, N + 1)

    # Plot running fraction
    ax.plot(n_values, running_punknot, color='#1f77b4', linewidth=1.5, label='P(unknot)')

    # Centroid comparison horizontal line
    if walk_id in centroid_knot_types:
        centroid_knot = centroid_knot_types[walk_id][0]
        centroid_y = 1.0 if centroid_knot == '0_1' else 0.0
        ax.axhline(y=centroid_y, color='#d62728', linestyle='--', linewidth=1.5,
                   alpha=0.8, label=f'Centroid ({centroid_knot})')

    ax.set_xlabel('Number of DeWitt curves', fontsize=10, fontweight='bold')
    ax.set_ylabel('P(unknot)', fontsize=10, fontweight='bold')
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(1, N)
    ax.set_title(f'Walk-ID {walk_id}: P(unknot) convergence', fontsize=11, fontweight='bold')
    ax.legend(loc='best', fontsize=8, frameon=True, fancybox=True)
    ax.grid(True, alpha=0.3)


def create_walk_report_card(walk_id, walk_results, controls, walk_acn_data, walk_kendall_data,
                           morans_i_dict, walk_tp1_tp21_distances, control_acn_values,
                           global_x_range, global_y_max, walk_groups, centroid_knot_types,
                           walk_v2_data=None, control_v2_values=None,
                           global_v2_x_range=None, global_v2_y_max=None,
                           walk_xycn_data=None, control_xycn_values=None,
                           global_xycn_x_range=None, global_xycn_y_max=None,
                           walk_xzcn_data=None, control_xzcn_values=None,
                           walk_yzcn_data=None, control_yzcn_values=None,
                           unified_planar_cn_x_range=None, unified_planar_cn_y_max=None,
                           walk_writhe_data=None, control_writhe_values=None,
                           global_writhe_x_range=None, global_writhe_y_max=None,
                           control_rms_writhe_values=None,
                           global_rms_writhe_x_range=None, global_rms_writhe_y_max=None,
                           walk_rog_data=None, control_rog_values=None,
                           global_rog_x_range=None, global_rog_y_max=None,
                           metrics='all', puknot=False, no_stacked=False,
                           show_centroids=False):
    """
    Create a comprehensive report card for a single walk-id with 2×2 layout.

    Args:
        walk_id: ID of the walk
        walk_results: Dictionary of all walk results
        controls: Dictionary of control results
        walk_acn_data: Dictionary of ACN data per walk
        walk_kendall_data: Dictionary of Kendall data per walk
        morans_i_dict: Dictionary of Moran's I data per walk
        walk_tp1_tp21_distances: Dictionary of TP1→TP21 distances per walk
        control_acn_values: List of ACN values for control
        global_x_range: Tuple (min, max) for ACN x-axis range
        global_y_max: Maximum y value for ACN y-axis
        walk_groups: Dictionary of raw DataFrames per walk-id (for 3D trajectory plot)
        centroid_knot_types: Dictionary mapping walk_id to (knot_name, determinant) tuples
        walk_v2_data: Dictionary of V2 data per walk (optional)
        control_v2_values: List of V2 values for control (optional)
        global_v2_x_range: Tuple (min, max) for V2 x-axis range (optional)
        global_v2_y_max: Maximum y value for V2 y-axis (optional)
        walk_xycn_data: Dictionary of xyCN data per walk (optional)
        control_xycn_values: List of xyCN values for control (optional)
        global_xycn_x_range: Tuple (min, max) for xyCN x-axis range (optional)
        global_xycn_y_max: Maximum y value for xyCN y-axis (optional)
        walk_xzcn_data: Dictionary of xzCN data per walk (optional)
        control_xzcn_values: List of xzCN values for control (optional)
        walk_yzcn_data: Dictionary of yzCN data per walk (optional)
        control_yzcn_values: List of yzCN values for control (optional)
        unified_planar_cn_x_range: Tuple (min, max) for unified planar CN x-axis range (optional)
        unified_planar_cn_y_max: Maximum y value for unified planar CN y-axis (optional)
        walk_writhe_data: Dictionary of Writhe data per walk (optional)
        control_writhe_values: List of Writhe values for control (optional)
        global_writhe_x_range: Tuple (min, max) for Writhe x-axis range (optional)
        global_writhe_y_max: Maximum y value for Writhe y-axis (optional)
        metrics: Which metrics to show

    Returns:
        matplotlib.figure.Figure: The report card figure
    """
    # Validate walk_id exists and has valid data
    if walk_id not in walk_results:
        raise ValueError(f"Walk-ID {walk_id} not found in walk_results")

    walk_data = walk_results[walk_id]
    is_valid, error_msg = validate_knot_results(walk_data, f"Walk-ID {walk_id} (report card)")
    if not is_valid:
        raise ValueError(f"Invalid data structure for Walk-ID {walk_id}: {error_msg}\n"
                        f"Type: {type(walk_data)}, Length: {len(walk_data) if hasattr(walk_data, '__len__') else 'N/A'}")

    # Note: Controls have 2-tuple format (name, det) while walks have 3-tuple format (curve, name, det)
    # We don't validate controls here - extract_knot_properties() handles both formats

    # Create figure with flexible grid layout
    # Use 6 columns for LCM flexibility: top row 2 panels (3+3), bottom row 2 or 3 panels
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 6, hspace=0.35, wspace=0.35,
                         left=0.08, right=0.95, top=0.95, bottom=0.08)

    # Panel 1 (top-left): Stacked bar plot
    ax1 = fig.add_subplot(gs[0, 0:3])
    create_single_walk_stacked_bar(walk_id, walk_results, controls, ax1,
                                   metrics=metrics, show_legend=True,
                                   no_stacked=no_stacked)

    # Panel 2 (top-right): Split horizontally for ACN (top) and V2/planar CNs (bottom)
    # Check if V2 data is available
    has_v2 = (walk_v2_data is not None and walk_id in walk_v2_data and
              control_v2_values is not None and len(control_v2_values) > 0)
    # Check if any planar CN data is available
    has_xycn = (walk_xycn_data is not None and walk_id in walk_xycn_data and
                control_xycn_values is not None and len(control_xycn_values) > 0)
    has_xzcn = (walk_xzcn_data is not None and walk_id in walk_xzcn_data and
                control_xzcn_values is not None and len(control_xzcn_values) > 0)
    has_yzcn = (walk_yzcn_data is not None and walk_id in walk_yzcn_data and
                control_yzcn_values is not None and len(control_yzcn_values) > 0)
    has_planar_cn = has_xycn or has_xzcn or has_yzcn
    has_rog = (walk_rog_data is not None and walk_id in walk_rog_data and
               control_rog_values is not None and len(control_rog_values) > 0)

    # Calculate number of rows needed for Panel 2
    n_rows = 1  # ACN always present
    if has_v2 or has_planar_cn:
        n_rows += 1  # V2 or planar CNs
    n_rows += 1  # RMS Writhe always present when ACN is enabled
    if has_rog:
        n_rows += 1  # RoG

    # Use 3 columns: shared y-label (5%), plots (75%), per-plot legends (20%)
    # Increase hspace significantly to prevent x-axis labels from being cut off
    gs_panel2 = gs[0, 3:6].subgridspec(n_rows, 3, hspace=0.55, wspace=0.02,
                                      width_ratios=[0.05, 0.75, 0.20])
    row_idx = 0

    # Create shared y-axis label on the left (spans all rows)
    ax2_ylabel = fig.add_subplot(gs_panel2[:, 0])
    ax2_ylabel.axis('off')
    ax2_ylabel.text(0.0, 0.5, 'Probability Density', rotation=90,
                    va='center', ha='center', fontsize=11, fontweight='bold',
                    transform=ax2_ylabel.transAxes)

    ax2_acn = fig.add_subplot(gs_panel2[row_idx, 1])  # ACN
    ax2_acn_legend = fig.add_subplot(gs_panel2[row_idx, 2])  # ACN legend
    ax2_acn_legend.axis('off')
    row_idx += 1

    if has_v2 or has_planar_cn:
        ax2_second = fig.add_subplot(gs_panel2[row_idx, 1])  # V2 or planar CNs
        ax2_second_legend = fig.add_subplot(gs_panel2[row_idx, 2])  # V2/planar legend
        ax2_second_legend.axis('off')
        row_idx += 1
    else:
        ax2_second = None
        ax2_second_legend = None

    ax2_writhe = fig.add_subplot(gs_panel2[row_idx, 1])  # RMS Writhe
    ax2_writhe_legend = fig.add_subplot(gs_panel2[row_idx, 2])  # Writhe legend
    ax2_writhe_legend.axis('off')
    row_idx += 1

    if has_rog:
        ax2_rog = fig.add_subplot(gs_panel2[row_idx, 1])  # RoG
        ax2_rog_legend = fig.add_subplot(gs_panel2[row_idx, 2])  # RoG legend
        ax2_rog_legend.axis('off')
    else:
        ax2_rog = None
        ax2_rog_legend = None

    # Plot ACN distribution
    if walk_id in walk_acn_data:
        # Get data
        sampled_acn = walk_acn_data[walk_id]['sampled_acn']
        centroid_acn = walk_acn_data[walk_id]['centroid_acn']

        # Remove NaN values
        sampled_acn_clean = [x for x in sampled_acn if not np.isnan(x)]
        control_acn_clean = [x for x in control_acn_values if not np.isnan(x)]

        # Convert global_x_range tuple to linspace for KDE
        if isinstance(global_x_range, tuple):
            x_range = np.linspace(global_x_range[0], global_x_range[1], 200)
        else:
            x_range = global_x_range

        # Define colors using seaborn palette (matching acn.py)
        colors = sns.color_palette("deep")
        raw_color = colors[0]  # blue
        control_color = colors[7]  # gray
        centroid_color = colors[3]  # red

        # Compute kernel density estimates
        if len(sampled_acn_clean) > 1:
            kde_raw = gaussian_kde(sampled_acn_clean)
            density_raw = kde_raw(x_range)
            ax2_acn.fill_between(x_range, density_raw, alpha=0.6, color=raw_color, label='Raw')
            ax2_acn.plot(x_range, density_raw, color=raw_color, linewidth=2, alpha=0.8)

        if len(control_acn_clean) > 1:
            kde_control = gaussian_kde(control_acn_clean)
            density_control = kde_control(x_range)
            ax2_acn.fill_between(x_range, density_control, alpha=0.4, color=control_color, label='Random Control')
            ax2_acn.plot(x_range, density_control, color=control_color, linewidth=2, alpha=0.8)

        # Add statistics
        if len(sampled_acn_clean) > 0:
            bio_mean = np.mean(sampled_acn_clean)
            bio_std = np.std(sampled_acn_clean)
        else:
            bio_mean = np.nan
            bio_std = np.nan

        if len(control_acn_clean) > 0:
            ctrl_mean = np.mean(control_acn_clean)
            ctrl_std = np.std(control_acn_clean)
        else:
            ctrl_mean = np.nan
            ctrl_std = np.nan

        # Add mean lines
        if not np.isnan(bio_mean):
            ax2_acn.axvline(bio_mean, color=raw_color, linestyle='--', linewidth=2.5, alpha=0.8)
        if not np.isnan(ctrl_mean):
            ax2_acn.axvline(ctrl_mean, color=control_color, linestyle='--', linewidth=2.5, alpha=0.8)

        # Add centroid ACN line if requested
        if show_centroids and centroid_acn is not None and not np.isnan(centroid_acn):
            ax2_acn.axvline(centroid_acn, color=centroid_color, linestyle='-', linewidth=3,
                            alpha=0.9, label=f'Centroid: {centroid_acn:.2f}')

        # Formatting
        ax2_acn.set_xlabel('Average Crossing Number', fontsize=10, fontweight='bold')
        # y-axis label removed - using shared label on left side
        # Simplified title
        ax2_acn.set_title(f'Walk-ID {walk_id}: Distributions', fontweight='bold', fontsize=11)

        # Set limits
        # global_x_range is a tuple (min, max), unpack it for set_xlim
        # Extract min and max as separate scalar values
        try:
            if hasattr(global_x_range, '__len__') and len(global_x_range) == 2:
                x_min = float(global_x_range[0])
                x_max = float(global_x_range[1])
            else:
                # Fallback to reasonable defaults
                x_min, x_max = 0, 10
            ax2_acn.set_xlim(x_min, x_max)
        except Exception as e:
            print(f"Warning: Could not set xlim for Walk-ID {walk_id}: {e}")
            print(f"  global_x_range type: {type(global_x_range)}, value: {global_x_range}")
            ax2_acn.set_xlim(0, 10)  # Safe default

        if global_y_max is not None:
            try:
                ax2_acn.set_ylim(0, float(global_y_max))
            except:
                ax2_acn.set_ylim(0, 1)  # Safe default

        # Add text box with statistics (positioned top-right)
        stats_text = ''
        if not np.isnan(bio_mean):
            stats_text += f'Raw: μ={bio_mean:.2f}, σ={bio_std:.2f}\n'
        if not np.isnan(ctrl_mean):
            stats_text += f'Control: μ={ctrl_mean:.2f}, σ={ctrl_std:.2f}'
        if centroid_acn is not None and not np.isnan(centroid_acn):
            stats_text += f'\nCentroid: {centroid_acn:.2f}'

        if stats_text:
            props = dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8, edgecolor='gray')
            ax2_acn.text(0.97, 0.97, stats_text, transform=ax2_acn.transAxes,
                    fontsize=8, verticalalignment='top', horizontalalignment='right', bbox=props)

        # Improve grid (matching acn.py)
        ax2_acn.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
        ax2_acn.set_axisbelow(True)

        # Add subtle spines (matching acn.py)
        for spine in ax2_acn.spines.values():
            spine.set_linewidth(0.5)
            spine.set_alpha(0.5)

        # Add legend to the right side
        handles, labels = ax2_acn.get_legend_handles_labels()
        if handles:
            ax2_acn_legend.legend(handles, labels, loc='center left', fontsize=7,
                                  frameon=True, fancybox=True)

    else:
        ax2_acn.text(0.5, 0.5, 'No ACN data', ha='center', va='center', transform=ax2_acn.transAxes)
        ax2_acn.set_title(f'Walk-ID {walk_id}: Distributions', fontweight='bold', fontsize=11)

    # Plot V2 distribution (if ax2_second exists and has_v2)
    if ax2_second is not None and has_v2 and walk_id in walk_v2_data:
        # Get data
        sampled_v2 = walk_v2_data[walk_id]['sampled_v2']
        centroid_v2 = walk_v2_data[walk_id]['centroid_v2']

        # Remove NaN values
        sampled_v2_clean = [x for x in sampled_v2 if not np.isnan(x)]
        control_v2_clean = [x for x in control_v2_values if not np.isnan(x)]

        # Convert global_v2_x_range tuple to linspace for KDE
        if isinstance(global_v2_x_range, tuple):
            v2_x_range = np.linspace(global_v2_x_range[0], global_v2_x_range[1], 200)
        else:
            v2_x_range = global_v2_x_range

        # Define colors using seaborn palette (same as ACN)
        colors = sns.color_palette("deep")
        raw_color = colors[0]  # blue
        control_color = colors[7]  # gray
        centroid_color = colors[3]  # red

        # Compute kernel density estimates
        if len(sampled_v2_clean) > 1:
            kde_raw = gaussian_kde(sampled_v2_clean)
            density_raw = kde_raw(v2_x_range)
            ax2_second.fill_between(v2_x_range, density_raw, alpha=0.6, color=raw_color, label='Raw')
            ax2_second.plot(v2_x_range, density_raw, color=raw_color, linewidth=2, alpha=0.8)

        if len(control_v2_clean) > 1:
            kde_control = gaussian_kde(control_v2_clean)
            density_control = kde_control(v2_x_range)
            ax2_second.fill_between(v2_x_range, density_control, alpha=0.4, color=control_color, label='Random Control')
            ax2_second.plot(v2_x_range, density_control, color=control_color, linewidth=2, alpha=0.8)

        # Add statistics
        if len(sampled_v2_clean) > 0:
            bio_mean = np.mean(sampled_v2_clean)
            bio_std = np.std(sampled_v2_clean)
        else:
            bio_mean = np.nan
            bio_std = np.nan

        if len(control_v2_clean) > 0:
            ctrl_mean = np.mean(control_v2_clean)
            ctrl_std = np.std(control_v2_clean)
        else:
            ctrl_mean = np.nan
            ctrl_std = np.nan

        # Add mean lines
        if not np.isnan(bio_mean):
            ax2_second.axvline(bio_mean, color=raw_color, linestyle='--', linewidth=2.5, alpha=0.8)
        if not np.isnan(ctrl_mean):
            ax2_second.axvline(ctrl_mean, color=control_color, linestyle='--', linewidth=2.5, alpha=0.8)

        # Add centroid V2 line if available
        if centroid_v2 is not None and not np.isnan(centroid_v2):
            ax2_second.axvline(centroid_v2, color=centroid_color, linestyle='-', linewidth=3,
                       alpha=0.9, label=f'Centroid: {centroid_v2:.6f}')

        # Formatting
        ax2_second.set_xlabel('Vassiliev V2 Invariant', fontsize=10, fontweight='bold')
        # y-axis label removed - using shared label on left side
        # No title for V2 plot - title is on ACN plot above

        # Set limits
        try:
            if hasattr(global_v2_x_range, '__len__') and len(global_v2_x_range) == 2:
                x_min = float(global_v2_x_range[0])
                x_max = float(global_v2_x_range[1])
            else:
                x_min, x_max = 0, 1
            ax2_second.set_xlim(x_min, x_max)
        except Exception as e:
            print(f"Warning: Could not set V2 xlim for Walk-ID {walk_id}: {e}")
            ax2_second.set_xlim(0, 1)  # Safe default

        if global_v2_y_max is not None:
            try:
                ax2_second.set_ylim(0, float(global_v2_y_max))
            except:
                ax2_second.set_ylim(0, 1)  # Safe default

        # Add text box with statistics
        stats_text = ''
        if not np.isnan(bio_mean):
            stats_text += f'Raw: μ={bio_mean:.6f}, σ={bio_std:.6f}\n'
        if not np.isnan(ctrl_mean):
            stats_text += f'Control: μ={ctrl_mean:.6f}, σ={ctrl_std:.6f}'
        if centroid_v2 is not None and not np.isnan(centroid_v2):
            stats_text += f'\nCentroid: {centroid_v2:.6f}'

        if stats_text:
            props = dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8, edgecolor='gray')
            ax2_second.text(0.97, 0.97, stats_text, transform=ax2_second.transAxes,
                    fontsize=8, verticalalignment='top', horizontalalignment='right', bbox=props)

        # Improve grid
        ax2_second.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
        ax2_second.set_axisbelow(True)

        # Add subtle spines
        for spine in ax2_second.spines.values():
            spine.set_linewidth(0.5)
            spine.set_alpha(0.5)

        # Add legend to the right side
        handles, labels = ax2_second.get_legend_handles_labels()
        if handles and ax2_second_legend is not None:
            ax2_second_legend.legend(handles, labels, loc='center left', fontsize=7,
                                     frameon=True, fancybox=True)

    # Plot combined planar CN distribution (if ax2_second exists and has_planar_cn)
    elif ax2_second is not None and has_planar_cn:
        # Use unified x-range for all projections
        if isinstance(unified_planar_cn_x_range, tuple):
            cn_x_range = np.linspace(unified_planar_cn_x_range[0], unified_planar_cn_x_range[1], 200)
        else:
            cn_x_range = np.linspace(0, 10, 200)  # Safe default

        # Define colors for each projection
        colors = sns.color_palette("deep")
        projection_colors = {
            'xy': colors[0],  # blue
            'xz': colors[2],  # green
            'yz': colors[4],  # purple
        }
        control_color = colors[7]  # gray

        # Plot each enabled projection
        if has_xycn and walk_id in walk_xycn_data:
            sampled_xycn = [x for x in walk_xycn_data[walk_id]['sampled_xycn'] if not np.isnan(x)]
            if len(sampled_xycn) > 1:
                kde = gaussian_kde(sampled_xycn)
                density = kde(cn_x_range)
                ax2_second.plot(cn_x_range, density, color=projection_colors['xy'],
                              linewidth=2, alpha=0.8, label='xy-CN')
                ax2_second.fill_between(cn_x_range, density, alpha=0.3,
                                       color=projection_colors['xy'])

            # Centroid line
            centroid_xycn = walk_xycn_data[walk_id]['centroid_xycn']
            if not np.isnan(centroid_xycn):
                ax2_second.axvline(centroid_xycn, color=projection_colors['xy'],
                                 linestyle='-', linewidth=2.5, alpha=0.9,
                                 label=f'xy Centroid: {centroid_xycn:.0f}')

        if has_xzcn and walk_id in walk_xzcn_data:
            sampled_xzcn = [x for x in walk_xzcn_data[walk_id]['sampled_xzcn'] if not np.isnan(x)]
            if len(sampled_xzcn) > 1:
                kde = gaussian_kde(sampled_xzcn)
                density = kde(cn_x_range)
                ax2_second.plot(cn_x_range, density, color=projection_colors['xz'],
                              linewidth=2, alpha=0.8, label='xz-CN')
                ax2_second.fill_between(cn_x_range, density, alpha=0.3,
                                       color=projection_colors['xz'])

            # Centroid line
            centroid_xzcn = walk_xzcn_data[walk_id]['centroid_xzcn']
            if not np.isnan(centroid_xzcn):
                ax2_second.axvline(centroid_xzcn, color=projection_colors['xz'],
                                 linestyle='-', linewidth=2.5, alpha=0.9,
                                 label=f'xz Centroid: {centroid_xzcn:.0f}')

        if has_yzcn and walk_id in walk_yzcn_data:
            sampled_yzcn = [x for x in walk_yzcn_data[walk_id]['sampled_yzcn'] if not np.isnan(x)]
            if len(sampled_yzcn) > 1:
                kde = gaussian_kde(sampled_yzcn)
                density = kde(cn_x_range)
                ax2_second.plot(cn_x_range, density, color=projection_colors['yz'],
                              linewidth=2, alpha=0.8, label='yz-CN')
                ax2_second.fill_between(cn_x_range, density, alpha=0.3,
                                       color=projection_colors['yz'])

            # Centroid line
            centroid_yzcn = walk_yzcn_data[walk_id]['centroid_yzcn']
            if not np.isnan(centroid_yzcn):
                ax2_second.axvline(centroid_yzcn, color=projection_colors['yz'],
                                 linestyle='-', linewidth=2.5, alpha=0.9,
                                 label=f'yz Centroid: {centroid_yzcn:.0f}')

        # Plot control distributions for each enabled projection (matching colors with dashed lines)
        if has_xycn and len(control_xycn_values) > 1:
            control_clean = [x for x in control_xycn_values if not np.isnan(x)]
            kde_control = gaussian_kde(control_clean)
            density_control = kde_control(cn_x_range)
            ax2_second.plot(cn_x_range, density_control, color=projection_colors['xy'],
                           linewidth=2, alpha=0.6, linestyle='--', label='xy Control')
            ax2_second.fill_between(cn_x_range, density_control, alpha=0.15,
                                   color=projection_colors['xy'])

        if has_xzcn and len(control_xzcn_values) > 1:
            control_clean = [x for x in control_xzcn_values if not np.isnan(x)]
            kde_control = gaussian_kde(control_clean)
            density_control = kde_control(cn_x_range)
            ax2_second.plot(cn_x_range, density_control, color=projection_colors['xz'],
                           linewidth=2, alpha=0.6, linestyle='--', label='xz Control')
            ax2_second.fill_between(cn_x_range, density_control, alpha=0.15,
                                   color=projection_colors['xz'])

        if has_yzcn and len(control_yzcn_values) > 1:
            control_clean = [x for x in control_yzcn_values if not np.isnan(x)]
            kde_control = gaussian_kde(control_clean)
            density_control = kde_control(cn_x_range)
            ax2_second.plot(cn_x_range, density_control, color=projection_colors['yz'],
                           linewidth=2, alpha=0.6, linestyle='--', label='yz Control')
            ax2_second.fill_between(cn_x_range, density_control, alpha=0.15,
                                   color=projection_colors['yz'])

        # Formatting
        ax2_second.set_xlabel('Planar Crossing Number', fontsize=10, fontweight='bold')
        # y-axis label removed - using shared label on left side
        # No title - title is on ACN plot above

        # Set limits using unified range
        try:
            if hasattr(unified_planar_cn_x_range, '__len__') and len(unified_planar_cn_x_range) == 2:
                x_min = float(unified_planar_cn_x_range[0])
                x_max = float(unified_planar_cn_x_range[1])
            else:
                x_min, x_max = 0, 10
            ax2_second.set_xlim(x_min, x_max)
        except Exception as e:
            print(f"Warning: Could not set planar CN xlim for Walk-ID {walk_id}: {e}")
            ax2_second.set_xlim(0, 10)  # Safe default

        if unified_planar_cn_y_max is not None:
            try:
                ax2_second.set_ylim(0, float(unified_planar_cn_y_max))
            except:
                ax2_second.set_ylim(0, 1)  # Safe default

        # Improve grid
        ax2_second.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
        ax2_second.set_axisbelow(True)

        # Add subtle spines
        for spine in ax2_second.spines.values():
            spine.set_linewidth(0.5)
            spine.set_alpha(0.5)

        # Add legend to the right side
        handles, labels = ax2_second.get_legend_handles_labels()
        if handles and ax2_second_legend is not None:
            ax2_second_legend.legend(handles, labels, loc='center left', fontsize=6,
                                     frameon=True, fancybox=True)

    # Plot Writhe distribution (bottom subplot of Panel 2)
    if walk_writhe_data is not None and walk_id in walk_writhe_data:
        # Get RMS writhe data (plot RMS writhe, not signed writhe)
        sampled_rms_writhe = walk_writhe_data[walk_id].get('sampled_rms_writhe', [])
        centroid_rms_writhe = walk_writhe_data[walk_id].get('centroid_rms_writhe', np.nan)

        # Remove NaN values
        sampled_rms_clean = [x for x in sampled_rms_writhe if not np.isnan(x)]
        control_rms_clean = [x for x in (control_rms_writhe_values or []) if not np.isnan(x)]

        # Convert global_rms_writhe_x_range tuple to linspace for KDE
        if global_rms_writhe_x_range is not None and isinstance(global_rms_writhe_x_range, tuple):
            rms_x_range = np.linspace(global_rms_writhe_x_range[0], global_rms_writhe_x_range[1], 200)
        else:
            # Safe default: RMS is always positive
            all_rms = sampled_rms_clean + control_rms_clean
            if all_rms:
                r_min = 0
                r_max = max(all_rms) + 1
            else:
                r_min, r_max = 0, 10
            rms_x_range = np.linspace(r_min, r_max, 200)

        # Define colors using seaborn palette (matching ACN)
        colors = sns.color_palette("deep")
        raw_color = colors[0]  # blue
        control_color = colors[7]  # gray
        centroid_color = colors[3]  # red

        # Compute kernel density estimates
        if len(sampled_rms_clean) > 1:
            kde_raw = gaussian_kde(sampled_rms_clean)
            density_raw = kde_raw(rms_x_range)
            ax2_writhe.fill_between(rms_x_range, density_raw, alpha=0.6, color=raw_color, label='Raw')
            ax2_writhe.plot(rms_x_range, density_raw, color=raw_color, linewidth=2, alpha=0.8)

        if len(control_rms_clean) > 1:
            kde_control = gaussian_kde(control_rms_clean)
            density_control = kde_control(rms_x_range)
            ax2_writhe.fill_between(rms_x_range, density_control, alpha=0.4, color=control_color, label='Random Control')
            ax2_writhe.plot(rms_x_range, density_control, color=control_color, linewidth=2, alpha=0.8)

        # Add statistics
        if len(sampled_rms_clean) > 0:
            bio_mean = np.mean(sampled_rms_clean)
            bio_std = np.std(sampled_rms_clean)
        else:
            bio_mean, bio_std = np.nan, np.nan

        if len(control_rms_clean) > 0:
            ctrl_mean = np.mean(control_rms_clean)
            ctrl_std = np.std(control_rms_clean)
        else:
            ctrl_mean, ctrl_std = np.nan, np.nan

        # Add centroid RMS writhe vertical line (solid red) if requested
        if show_centroids and centroid_rms_writhe is not None and not np.isnan(centroid_rms_writhe):
            ax2_writhe.axvline(centroid_rms_writhe, color=centroid_color, linestyle='-', linewidth=3,
                               alpha=0.9, label=f'Centroid: {centroid_rms_writhe:.2f}')

        # Formatting
        ax2_writhe.set_xlabel('RMS Writhe', fontsize=10, fontweight='bold')
        # y-axis label removed - using shared label on left side
        # No title - consistent with planar CNs

        # Set limits
        if global_rms_writhe_x_range is not None:
            try:
                if hasattr(global_rms_writhe_x_range, '__len__') and len(global_rms_writhe_x_range) == 2:
                    x_min = float(global_rms_writhe_x_range[0])
                    x_max = float(global_rms_writhe_x_range[1])
                    ax2_writhe.set_xlim(x_min, x_max)
            except Exception as e:
                print(f"Warning: Could not set RMS writhe xlim for Walk-ID {walk_id}: {e}")

        if global_rms_writhe_y_max is not None:
            try:
                ax2_writhe.set_ylim(0, float(global_rms_writhe_y_max))
            except:
                pass  # Use auto limits

        # Add text box with statistics
        stats_text = ''
        if not np.isnan(bio_mean):
            stats_text += f'Raw: {bio_mean:.2f} +/- {bio_std:.2f}\n'
        if not np.isnan(ctrl_mean):
            stats_text += f'Control: {ctrl_mean:.2f} +/- {ctrl_std:.2f}'
        if centroid_rms_writhe is not None and not np.isnan(centroid_rms_writhe):
            stats_text += f'\nCentroid: {centroid_rms_writhe:.2f}'

        if stats_text:
            props = dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8, edgecolor='gray')
            ax2_writhe.text(0.97, 0.97, stats_text, transform=ax2_writhe.transAxes,
                    fontsize=7, verticalalignment='top', horizontalalignment='right', bbox=props)

        # Improve grid
        ax2_writhe.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
        ax2_writhe.set_axisbelow(True)

        # Add subtle spines
        for spine in ax2_writhe.spines.values():
            spine.set_linewidth(0.5)
            spine.set_alpha(0.5)

        # Add legend to the right side
        handles, labels = ax2_writhe.get_legend_handles_labels()
        if handles:
            ax2_writhe_legend.legend(handles, labels, loc='center left', fontsize=7,
                                     frameon=True, fancybox=True)

    # Plot Radius of Gyration distribution (if enabled) - using log scale
    if ax2_rog is not None and walk_rog_data is not None and walk_id in walk_rog_data:
        # Get data
        sampled_rog = walk_rog_data[walk_id]['sampled_rog']
        centroid_rog = walk_rog_data[walk_id]['centroid_rog']

        # Remove NaN and zero/negative values (needed for log scale)
        sampled_rog_clean = [x for x in sampled_rog if not np.isnan(x) and x > 0]
        control_rog_clean = [x for x in control_rog_values if not np.isnan(x) and x > 0] if control_rog_values else []

        # Define colors using seaborn palette (matching ACN)
        colors = sns.color_palette("deep")
        raw_color = colors[0]  # blue
        control_color = colors[7]  # gray
        centroid_color = colors[3]  # red

        # Use log scale for x-axis
        ax2_rog.set_xscale('log')

        # Compute KDE on log-transformed data for proper log-scale display
        all_rog = sampled_rog_clean + control_rog_clean
        if all_rog:
            log_min = np.log10(max(1, min(all_rog) * 0.5))
            log_max = np.log10(max(all_rog) * 2)
            rog_x_range = np.logspace(log_min, log_max, 200)
        else:
            rog_x_range = np.logspace(0, 4, 200)

        # Compute kernel density estimates
        if len(sampled_rog_clean) > 1:
            # KDE on log-transformed data
            log_data = np.log10(sampled_rog_clean)
            kde_raw = gaussian_kde(log_data)
            log_x = np.log10(rog_x_range)
            # Transform density back: p(x) = p(log(x)) / (x * ln(10))
            density_raw = kde_raw(log_x) / (rog_x_range * np.log(10))
            ax2_rog.fill_between(rog_x_range, density_raw, alpha=0.6, color=raw_color, label='Raw')
            ax2_rog.plot(rog_x_range, density_raw, color=raw_color, linewidth=2, alpha=0.8)

        if len(control_rog_clean) > 1:
            log_data = np.log10(control_rog_clean)
            kde_control = gaussian_kde(log_data)
            log_x = np.log10(rog_x_range)
            density_control = kde_control(log_x) / (rog_x_range * np.log(10))
            ax2_rog.fill_between(rog_x_range, density_control, alpha=0.4, color=control_color, label='Random Control')
            ax2_rog.plot(rog_x_range, density_control, color=control_color, linewidth=2, alpha=0.8)

        # Add statistics
        if len(sampled_rog_clean) > 0:
            bio_mean = np.mean(sampled_rog_clean)
            bio_std = np.std(sampled_rog_clean)
        else:
            bio_mean, bio_std = np.nan, np.nan

        if len(control_rog_clean) > 0:
            ctrl_mean = np.mean(control_rog_clean)
            ctrl_std = np.std(control_rog_clean)
        else:
            ctrl_mean, ctrl_std = np.nan, np.nan

        # Add centroid RoG vertical line (solid red)
        if centroid_rog is not None and not np.isnan(centroid_rog) and centroid_rog > 0:
            ax2_rog.axvline(centroid_rog, color=centroid_color, linestyle='-', linewidth=3,
                           alpha=0.9, label=f'Centroid: {centroid_rog:.2f}')

        # Formatting
        ax2_rog.set_xlabel('Radius of Gyration (log scale)', fontsize=10, fontweight='bold')
        # y-axis label removed - using shared label on left side
        # No title - consistent with other subplots

        # Add text box with statistics
        stats_text = ''
        if not np.isnan(bio_mean):
            stats_text += f'Raw: {bio_mean:.2f} +/- {bio_std:.2f}\n'
        if not np.isnan(ctrl_mean):
            stats_text += f'Control: {ctrl_mean:.2f} +/- {ctrl_std:.2f}'
        if centroid_rog is not None and not np.isnan(centroid_rog):
            stats_text += f'\nCentroid: {centroid_rog:.2f}'

        if stats_text:
            props = dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8, edgecolor='gray')
            ax2_rog.text(0.97, 0.97, stats_text, transform=ax2_rog.transAxes,
                    fontsize=7, verticalalignment='top', horizontalalignment='right', bbox=props)

        # Improve grid
        ax2_rog.grid(True, alpha=0.2, linestyle='-', linewidth=0.5, which='both')
        ax2_rog.set_axisbelow(True)

        # Add subtle spines
        for spine in ax2_rog.spines.values():
            spine.set_linewidth(0.5)
            spine.set_alpha(0.5)

        # Add legend to the right side
        handles, labels = ax2_rog.get_legend_handles_labels()
        if handles and ax2_rog_legend is not None:
            ax2_rog_legend.legend(handles, labels, loc='center left', fontsize=7,
                                  frameon=True, fancybox=True)

    # Bottom row: 2 panels (no puknot) or 3 panels (with puknot)
    if puknot:
        # 3 panels in bottom row, each spanning 2 of 6 columns
        ax3 = fig.add_subplot(gs[1, 0:2], projection='3d')
        ax4 = fig.add_subplot(gs[1, 2:4], projection='3d')
        ax5 = fig.add_subplot(gs[1, 4:6])
    else:
        # 2 panels in bottom row, each spanning 3 of 6 columns
        ax3 = fig.add_subplot(gs[1, 0:3], projection='3d')
        ax4 = fig.add_subplot(gs[1, 3:6], projection='3d')

    # Panel 3 (bottom-left): Kendall icosagon
    if walk_kendall_data:
        create_kendall_single_panel(walk_id, walk_kendall_data, ax3)
    else:
        ax3.text(0.5, 0.5, 0.5, 'No Kendall data', ha='center', va='center')
        ax3.set_title(f'Walk-ID {walk_id}: Kendall Icosagon', fontsize=11, fontweight='bold')

    # Panel 4 (bottom-middle or bottom-right): 3D trajectory
    if walk_id in walk_groups:
        create_3d_trajectory_panel(walk_id, walk_groups[walk_id], morans_i_dict, centroid_knot_types, ax4)
    else:
        ax4.text(0.5, 0.5, 0.5, 'No trajectory data', ha='center', va='center')
        ax4.set_title(f'Walk-ID {walk_id}: 3D Trajectory', fontsize=11, fontweight='bold')

    # Panel 5 (bottom-right): P(unknot) convergence (only when --puknot is enabled)
    if puknot:
        create_puknot_panel(walk_id, walk_results, centroid_knot_types, ax5)

    # Overall title
    fig.suptitle(f'Walk-ID {walk_id} Report Card', fontsize=16, fontweight='bold')

    return fig


def create_summary_visualizations(walk_results, controls, results_df, metrics='all'):
    """
    Create summary visualizations for the entire dataset (KL vs EMD scatter, Moran's I, etc.).

    Args:
        walk_results: Dictionary of all walk results
        controls: Dictionary of control results
        results_df: DataFrame with all results including Moran's I
        metrics: Which metrics to show ('all', 'kl', 'wasserstein')

    Returns:
        list: List of summary figures
    """
    print("Creating summary visualizations...")
    figures = []

    # Set up colors and categories
    categories = ['0_1', '3_1', 'Complex']
    control_labels = list(controls.keys())
    n_controls = len(controls)

    # Calculate walk statistics for all walks
    walk_stats = {}
    for walk_id, walk_data in walk_results.items():
        # Extract knot properties
        walk_knot_props = extract_knot_properties(walk_data)
        walk_classified = [classify_name(name) for name, _ in walk_knot_props]
        walk_counts = Counter(walk_classified)
        walk_probs = {cat: walk_counts.get(cat, 0) / len(walk_classified) for cat in categories}

        walk_stats[walk_id] = {'probs': walk_probs}

        # Calculate metrics for each control
        for control_label, control_results in controls.items():
            # Extract knot properties from control
            control_knot_props = extract_knot_properties(control_results)

            if metrics in ['all', 'kl']:
                kl_div, _, _ = calculate_kl_divergence(walk_knot_props, control_knot_props)
                walk_stats[walk_id][f'kl_div_{control_label}'] = kl_div

            if metrics in ['all', 'wasserstein']:
                emd = calculate_emd(walk_knot_props, control_knot_props)
                walk_stats[walk_id][f'emd_{control_label}'] = emd

    # Create KL vs EMD scatter plot(s)
    if metrics == 'all' and n_controls == 1:
        # Single control: one scatter plot
        control_label = control_labels[0]
        kl_divergences = [stats[f'kl_div_{control_label}'] for stats in walk_stats.values()]
        emds = [stats[f'emd_{control_label}'] for stats in walk_stats.values()]

        fig, ax = plt.subplots(figsize=(8, 6))

        scatter = ax.scatter(kl_divergences, emds, alpha=0.6, s=100, c='steelblue', edgecolors='black', rasterized=True)

        # Add Walk-ID labels for outliers
        for i, (kl, emd, walk_id) in enumerate(zip(kl_divergences, emds, walk_stats.keys())):
            if kl > np.percentile(kl_divergences, 90) or emd > np.percentile(emds, 90):
                ax.annotate(f'W{walk_id}', (kl, emd), fontsize=8, alpha=0.7)

        ax.set_xlabel('KL Divergence', fontsize=12)
        ax.set_ylabel('Wasserstein Distance', fontsize=12)
        ax.set_title(f'KL Divergence vs Wasserstein Distance ({control_label})', fontsize=14)
        ax.grid(True, alpha=0.3)

        # Add correlation
        correlation = np.corrcoef(kl_divergences, emds)[0, 1]
        ax.text(0.05, 0.95, f'Correlation: {correlation:.3f}',
                transform=ax.transAxes, fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()
        figures.append(fig)

    elif metrics == 'all' and n_controls > 1:
        # Multiple controls: subplot for each
        fig, axes = plt.subplots(1, n_controls, figsize=(6*n_controls, 5))
        if n_controls == 2:
            axes = [axes[0], axes[1]]  # Ensure it's a list
        elif n_controls == 1:
            axes = [axes]

        control_colors = ['steelblue', 'coral', 'mediumseagreen', 'orange', 'purple', 'brown']

        for i, control_label in enumerate(control_labels):
            ax = axes[i] if n_controls > 1 else axes[0]
            kl_divergences = [stats[f'kl_div_{control_label}'] for stats in walk_stats.values()]
            emds = [stats[f'emd_{control_label}'] for stats in walk_stats.values()]

            scatter = ax.scatter(kl_divergences, emds, alpha=0.6, s=100,
                                c=control_colors[i % len(control_colors)], edgecolors='black', rasterized=True)

            # Add Walk-ID labels for outliers
            for j, (kl, emd, walk_id) in enumerate(zip(kl_divergences, emds, walk_stats.keys())):
                if kl > np.percentile(kl_divergences, 90) or emd > np.percentile(emds, 90):
                    ax.annotate(f'W{walk_id}', (kl, emd), fontsize=8, alpha=0.7)

            ax.set_xlabel('KL Divergence', fontsize=11)
            ax.set_ylabel('Wasserstein Distance', fontsize=11)
            ax.set_title(f'{control_label}', fontsize=12)
            ax.grid(True, alpha=0.3)

            # Add correlation
            correlation = np.corrcoef(kl_divergences, emds)[0, 1]
            ax.text(0.05, 0.95, f'Corr: {correlation:.3f}',
                    transform=ax.transAxes, fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        fig.suptitle('KL Divergence vs Wasserstein Distance', fontsize=14)
        plt.tight_layout()
        figures.append(fig)

    # Note: Moran's I correlation plot is skipped per user request

    print(f"  Created {len(figures)} summary visualization(s)")
    return figures


def create_first_page_summary(walk_results, controls, walk_acn_data,
                              control_acn_values, morans_i_dict, results_df, centroid_knot_types=None,
                              label='', two_categories=False, show_centroids=False):
    """
    Create first-page summary with 3 key plots:
    1. Overall knot distribution comparison
    2. ACN distribution summary
    3. Moran's I distribution

    Args:
        walk_results: Dictionary of all walk results
        controls: Dictionary of control results
        walk_acn_data: Dictionary of ACN data per walk
        control_acn_values: List of control ACN values
        morans_i_dict: Dictionary of Moran's I data per walk
        results_df: DataFrame with all results
        centroid_knot_types: Dictionary mapping walk_id to (knot_name, determinant) for centroids

    Returns:
        matplotlib.figure.Figure: Single figure with 3 subplots
    """
    print("  Creating first-page summary with 3 overview plots...")

    # Create figure with custom layout: top plot spans full width, bottom has 2 plots
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.30, wspace=0.30,
                         left=0.08, right=0.95, top=0.92, bottom=0.08,
                         height_ratios=[1, 1])

    # Define consistent colors
    colors = {
        '0_1': '#2ecc71',      # Green for unknots
        '3_1': '#ff9800',      # Orange for trefoils
        'Complex': '#e74c3c',  # Red for complex knots
        'Knotted': '#e74c3c'   # Red for combined knotted (2-cat mode)
    }
    if two_categories:
        categories = ['0_1', 'Knotted']
        category_labels = ['Unknotted', 'Knotted']
    else:
        categories = ['0_1', '3_1', 'Complex']
        category_labels = ['Unknot (0_1)', 'Trefoil (3_1)', 'Complex']

    def _classify_with_mode(name):
        c = classify_name(name)
        if two_categories and c != '0_1':
            return 'Knotted'
        return c

    # ========================================================================
    # Plot 1 (Top, spanning both columns): Overall Knot Distribution Comparison
    # ========================================================================
    ax1 = fig.add_subplot(gs[0, :])  # Span both columns

    try:
        # Aggregate all experimental walks
        all_exp_classified = []
        for walk_id, walk_data in walk_results.items():
            # Extract knot properties
            walk_knot_props = extract_knot_properties(walk_data)
            all_exp_classified.extend([_classify_with_mode(name) for name, _ in walk_knot_props])

        exp_counts = Counter(all_exp_classified)
        exp_total = len(all_exp_classified)

        # Prepare data for plotting
        groups = ['De Witt\nCurves']
        group_data = {cat: [exp_counts.get(cat, 0) / exp_total] for cat in categories}

        # Add centroid knot distribution if requested
        if show_centroids and centroid_knot_types:
            centroid_classified = [_classify_with_mode(name)
                                   for name, _ in centroid_knot_types.values()]
            centroid_counts = Counter(centroid_classified)
            centroid_total = len(centroid_classified)
            if centroid_total > 0:
                groups.append('Centroid\nCurves')
                for cat in categories:
                    group_data[cat].append(centroid_counts.get(cat, 0) / centroid_total)

        # Add controls
        for control_label, control_results in controls.items():
            # Extract knot properties from control
            control_knot_props = extract_knot_properties(control_results)
            control_classified = [_classify_with_mode(name) for name, _ in control_knot_props]
            control_counts = Counter(control_classified)
            control_total = len(control_classified)

            groups.append(control_label)
            for cat in categories:
                group_data[cat].append(control_counts.get(cat, 0) / control_total)

        # Create stacked bar chart
        x = np.arange(len(groups))
        width = 0.6

        bottom = np.zeros(len(groups))
        bars = []
        for cat, color in zip(categories, [colors[c] for c in categories]):
            values = group_data[cat]
            bar = ax1.bar(x, values, width, bottom=bottom,
                         color=color, edgecolor='black', linewidth=1.5)
            bars.append(bar)

            # Add percentage labels on bars
            for i, (val, bot) in enumerate(zip(values, bottom)):
                if val > 0.05:  # Only label if > 5%
                    ax1.text(x[i], bot + val/2, f'{val*100:.1f}%',
                            ha='center', va='center', fontweight='bold', fontsize=9)

            bottom += values

        ax1.set_ylabel('Proportion', fontsize=12, fontweight='bold')
        title_suffix = ' (Unknotted vs Knotted)' if two_categories else ''
        ax1.set_title(f'Knot Type Distribution: De Witt style curves vs Controls{title_suffix}',
                     fontsize=14, fontweight='bold', pad=15)
        ax1.set_xticks(x)
        ax1.set_xticklabels(groups, fontsize=11)
        ax1.set_ylim(0, 1)
        ax1.legend(bars, category_labels, loc='upper right', frameon=True,
                  fancybox=True, shadow=True, fontsize=10)
        ax1.grid(True, axis='y', alpha=0.2)
        ax1.set_axisbelow(True)

        print("    ✓ Plot 1: Knot distribution comparison created")

    except Exception as e:
        print(f"    ✗ Warning: Could not create knot distribution plot: {e}")
        ax1.text(0.5, 0.5, f'Knot distribution plot unavailable:\n{e}',
                ha='center', va='center', transform=ax1.transAxes, fontsize=11)

    # ========================================================================
    # Plot 2 (Bottom-left): ACN Distribution Summary
    # ========================================================================
    ax2 = fig.add_subplot(gs[1, 0])

    try:
        if walk_acn_data and control_acn_values:
            # Collect all experimental ACN values
            all_exp_acn = []
            for walk_id, data in walk_acn_data.items():
                sampled_acn_clean = [x for x in data['sampled_acn'] if not np.isnan(x)]
                all_exp_acn.extend(sampled_acn_clean)

            # Clean control ACN values
            control_acn_clean = [x for x in control_acn_values if not np.isnan(x)]

            if len(all_exp_acn) > 0 and len(control_acn_clean) > 0:
                # Create violin plot
                data_for_violin = [control_acn_clean, all_exp_acn]
                positions = [0, 1]

                parts = ax2.violinplot(data_for_violin, positions=positions,
                                      showmeans=True, showmedians=True, widths=0.6)

                # Color the violins
                colors_violin = ['steelblue', 'coral']
                for i, pc in enumerate(parts['bodies']):
                    pc.set_facecolor(colors_violin[i])
                    pc.set_alpha(0.6)
                    pc.set_edgecolor('black')
                    pc.set_linewidth(1.5)

                # Style the other components
                for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians', 'cmeans'):
                    if partname in parts:
                        vp = parts[partname]
                        vp.set_edgecolor('black')
                        vp.set_linewidth(1.5)

                # Calculate and display statistics
                ctrl_mean, ctrl_std = np.mean(control_acn_clean), np.std(control_acn_clean)
                exp_mean, exp_std = np.mean(all_exp_acn), np.std(all_exp_acn)

                # Add mean value labels
                ax2.text(0, ctrl_mean, f'μ={ctrl_mean:.1f}',
                        ha='right', va='center', fontsize=9,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
                ax2.text(1, exp_mean, f'μ={exp_mean:.1f}',
                        ha='left', va='center', fontsize=9,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

                # Perform statistical test
                from scipy import stats
                statistic, pvalue = stats.mannwhitneyu(control_acn_clean, all_exp_acn,
                                                       alternative='two-sided')

                # Add test result
                test_text = f'Mann-Whitney U test\np = {pvalue:.4f}'
                if pvalue < 0.001:
                    test_text += '\n***'
                elif pvalue < 0.01:
                    test_text += '\n**'
                elif pvalue < 0.05:
                    test_text += '\n*'

                ax2.text(0.98, 0.98, test_text, transform=ax2.transAxes,
                        fontsize=9, verticalalignment='top', horizontalalignment='right',
                        bbox=dict(boxstyle='round,pad=0.4', facecolor='yellow', alpha=0.7))

                ax2.set_ylabel('Average Crossing Number (ACN)', fontsize=11, fontweight='bold')
                ax2.set_title('ACN Distribution: Experimental vs Control',
                            fontsize=12, fontweight='bold')
                ax2.set_xticks([0, 1])
                ax2.set_xticklabels(['Control\n(Random)', 'All Walks\n(Experimental)'], fontsize=10)
                ax2.grid(True, axis='y', alpha=0.2)
                ax2.set_axisbelow(True)

                print("    ✓ Plot 2: ACN distribution created")
            else:
                ax2.text(0.5, 0.5, 'Insufficient ACN data', ha='center', va='center',
                        transform=ax2.transAxes, fontsize=11)
        else:
            ax2.text(0.5, 0.5, 'ACN data not computed\n(use --acn flag)',
                    ha='center', va='center', transform=ax2.transAxes, fontsize=11)

    except Exception as e:
        print(f"    ✗ Warning: Could not create ACN distribution plot: {e}")
        ax2.text(0.5, 0.5, f'ACN plot unavailable:\n{e}',
                ha='center', va='center', transform=ax2.transAxes, fontsize=10)

    # ========================================================================
    # Plot 3 (Bottom-right): Moran's I Distribution
    # ========================================================================
    ax3 = fig.add_subplot(gs[1, 1])

    try:
        if morans_i_dict:
            # Extract Moran's I values
            morans_values = []
            morans_pvalues = []
            for walk_id in sorted(walk_results.keys()):
                if walk_id in morans_i_dict:
                    morans_values.append(morans_i_dict[walk_id]['I_value'])
                    morans_pvalues.append(morans_i_dict[walk_id]['P_value'])

            if len(morans_values) > 0:
                morans_values = np.array(morans_values)
                morans_pvalues = np.array(morans_pvalues)

                # Separate significant and non-significant
                significant = morans_pvalues < 0.05

                # Create histogram with KDE overlay
                bins = 20

                # Plot all values
                n, bins_edges, patches = ax3.hist(morans_values, bins=bins,
                                                  alpha=0.5, color='steelblue',
                                                  edgecolor='black', linewidth=1)

                # Color bars by significance
                for i, patch in enumerate(patches):
                    # Find which values fall in this bin
                    in_bin = (morans_values >= bins_edges[i]) & (morans_values < bins_edges[i+1])
                    if i == len(patches) - 1:  # Last bin includes right edge
                        in_bin = (morans_values >= bins_edges[i]) & (morans_values <= bins_edges[i+1])

                    # Check if any significant values in this bin
                    if np.any(significant[in_bin]):
                        patch.set_facecolor('coral')
                        patch.set_alpha(0.7)

                # Add KDE overlay
                if len(morans_values) > 1:
                    from scipy.stats import gaussian_kde
                    kde = gaussian_kde(morans_values)
                    x_range = np.linspace(morans_values.min(), morans_values.max(), 200)
                    density = kde(x_range)
                    # Scale density to match histogram
                    density_scaled = density * len(morans_values) * (bins_edges[1] - bins_edges[0])
                    ax3.plot(x_range, density_scaled, color='darkblue', linewidth=2,
                            label='KDE', alpha=0.8)

                # Add mean line
                mean_val = np.mean(morans_values)
                ax3.axvline(mean_val, color='red', linestyle='--', linewidth=2.5,
                           label=f'Mean: {mean_val:.3f}', alpha=0.8)

                # Add zero line
                ax3.axvline(0, color='gray', linestyle=':', linewidth=2,
                           label='Zero (no correlation)', alpha=0.6)

                # Statistics text box
                std_val = np.std(morans_values)
                n_sig = np.sum(significant)
                pct_sig = (n_sig / len(morans_values)) * 100

                stats_text = f"N = {len(morans_values)}\n"
                stats_text += f"Mean: {mean_val:.3f}\n"
                stats_text += f"Std: {std_val:.3f}\n"
                stats_text += f"Significant (p<0.05): {n_sig} ({pct_sig:.1f}%)"

                ax3.text(0.02, 0.98, stats_text, transform=ax3.transAxes,
                        fontsize=9, verticalalignment='top',
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                                 alpha=0.8, edgecolor='gray'))

                ax3.set_xlabel("Moran's I", fontsize=11, fontweight='bold')
                ax3.set_ylabel('Frequency', fontsize=11, fontweight='bold')
                ax3.set_title("Moran's I Distribution Across Walks",
                            fontsize=12, fontweight='bold')
                ax3.legend(loc='upper right', frameon=True, fancybox=True,
                          shadow=True, fontsize=9)
                ax3.grid(True, alpha=0.2)
                ax3.set_axisbelow(True)

                print("    ✓ Plot 3: Moran's I distribution created")
            else:
                ax3.text(0.5, 0.5, "No Moran's I data available",
                        ha='center', va='center', transform=ax3.transAxes, fontsize=11)
        else:
            ax3.text(0.5, 0.5, "Moran's I data not loaded",
                    ha='center', va='center', transform=ax3.transAxes, fontsize=11)

    except Exception as e:
        print(f"    ✗ Warning: Could not create Moran's I distribution plot: {e}")
        ax3.text(0.5, 0.5, f"Moran's I plot unavailable:\n{e}",
                ha='center', va='center', transform=ax3.transAxes, fontsize=10)

    # Add overall title
    n_walks = len(walk_results)
    cat_suffix = ' - Unknotted vs Knotted' if two_categories else ''
    title = f'Dataset Summary Overview (N={n_walks}){cat_suffix}'
    if label:
        title = f'Dataset Summary Overview ({label}, N={n_walks}){cat_suffix}'
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)

    print("  ✓ First-page summary completed")
    return fig


def create_visualizations(walk_results, controls, output_prefix, metrics='all'):
    """
    Create visualization plots with stacked bar charts and KL vs EMD scatter plot.
    Supports multiple control distributions.

    Args:
        walk_results: Dictionary of walk results
        controls: Dictionary of control results {label: results}
        output_prefix: Prefix for output files
        metrics: Which metrics to show ('all', 'kl', 'wasserstein')
    """
    print("Creating visualizations...")

    # Set up the plot style
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['savefig.dpi'] = 300

    # Define colors for each knot type
    colors = {
        '0_1': '#2ecc71',      # Green for unknots
        '3_1': '#ff9800',      # Orange for trefoils
        'Complex': '#e74c3c'   # Red for complex knots
    }

    categories = ['0_1', '3_1', 'Complex']
    figures = []  # Collect all figures

    # Calculate control distributions
    control_probs = {}
    for label, control_results in controls.items():
        control_classified = [classify_name(name) for name, _ in control_results]
        control_counts = Counter(control_classified)
        control_probs[label] = {cat: control_counts.get(cat, 0) / len(control_classified) for cat in categories}

    # Calculate KL divergences and EMDs for all walks against all controls
    walk_stats = {}
    for walk_id, walk_data in walk_results.items():
        walk_classified = [classify_name(name) for name, _ in walk_data]
        walk_counts = Counter(walk_classified)
        walk_probs = {cat: walk_counts.get(cat, 0) / len(walk_classified) for cat in categories}

        walk_stats[walk_id] = {'probs': walk_probs}

        # Calculate metrics for each control
        for control_label, control_results in controls.items():
            if metrics in ['all', 'kl']:
                kl_div, _, _ = calculate_kl_divergence(walk_data, control_results)
                walk_stats[walk_id][f'kl_div_{control_label}'] = kl_div

            if metrics in ['all', 'wasserstein']:
                emd = calculate_emd(walk_data, control_results)
                walk_stats[walk_id][f'emd_{control_label}'] = emd

    # 1. Create stacked bar plots (6 per page)
    walk_ids = list(walk_results.keys())
    n_walks = len(walk_ids)
    plots_per_page = 6
    n_pages = (n_walks + plots_per_page - 1) // plots_per_page
    n_controls = len(controls)

    control_labels = list(controls.keys())

    for page in range(n_pages):
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        title = f'Knot Type Distributions: Walk-IDs vs {", ".join(control_labels)}'
        fig.suptitle(title, fontsize=16)

        start_idx = page * plots_per_page
        end_idx = min(start_idx + plots_per_page, n_walks)
        page_walk_ids = walk_ids[start_idx:end_idx]

        # Fill subplots
        for idx, ax in enumerate(axes.flat):
            if idx < len(page_walk_ids):
                walk_id = page_walk_ids[idx]
                stats = walk_stats[walk_id]

                # Create stacked bars for walk and all controls
                n_bars = 1 + n_controls  # 1 for walk, rest for controls
                x = np.arange(n_bars)
                width = 0.6
                bottoms = [0] * n_bars

                # Plot stacked bars
                for cat in categories:
                    heights = []

                    # Walk bar
                    heights.append(stats['probs'][cat])

                    # Control bars
                    for control_label in control_labels:
                        heights.append(control_probs[control_label][cat])

                    # Plot all bars for this category
                    for i, height in enumerate(heights):
                        ax.bar(x[i], height, width, bottom=bottoms[i],
                              color=colors[cat], edgecolor='black', linewidth=0.5)
                        bottoms[i] += height

                # Customize subplot
                ax.set_ylim([0, 1])
                ax.set_xlim([-0.5, n_bars - 0.5])
                ax.set_xticks(x)

                # Create x-tick labels with "Vs" prefix for controls
                xlabels = [f'Walk-{walk_id}'] + [f'{label}' for label in control_labels]
                ax.set_xticklabels(xlabels, rotation=45, ha='right')
                ax.set_ylabel('Proportion')
                ax.set_title(f'Walk-ID {walk_id}', fontweight='bold')

                # Add statistics text in lower right
                stats_text_lines = []
                for control_label in control_labels:
                    # stats_text_lines.append(f'Vs {control_label}:')

                    if metrics in ['all', 'kl']:
                        kl_key = f'kl_div_{control_label}'
                        if kl_key in stats:
                            stats_text_lines.append(f'KL vs {control_label}: {stats[kl_key]:.4f}')

                    if metrics in ['all', 'wasserstein']:
                        emd_key = f'emd_{control_label}'
                        if emd_key in stats:
                            stats_text_lines.append(f'Wasserstein vs {control_label}: {stats[emd_key]:.4f}')

                textstr = '\n'.join(stats_text_lines)
                ax.text(0.98, 0.02, textstr, transform=ax.transAxes,
                       fontsize=8, verticalalignment='bottom', horizontalalignment='right',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

                # Add legend to first subplot only
                if idx == 0:
                    legend_elements = [plt.Rectangle((0, 0), 1, 1, fc=colors[cat], edgecolor='black')
                                     for cat in categories]
                    legend_labels = ['Unknot (0_1)', 'Trefoil (3_1)', 'Complex']
                    ax.legend(legend_elements, legend_labels, loc='lower left', fontsize=9)

            else:
                # Hide empty subplots
                ax.axis('off')

        plt.tight_layout()

        # Append this page's figure to the list
        figures.append(fig)

    # 2. Scatter plot of KL vs EMD for each control (only if both metrics are computed)
    # Create separate plot for each control or combined plot for multiple controls
    if metrics == 'all' and n_controls == 1:
        # Single control: one scatter plot
        control_label = control_labels[0]
        kl_divergences = [stats[f'kl_div_{control_label}'] for stats in walk_stats.values()]
        emds = [stats[f'emd_{control_label}'] for stats in walk_stats.values()]

        fig, ax = plt.subplots(figsize=(8, 6))

        scatter = ax.scatter(kl_divergences, emds, alpha=0.6, s=100, c='steelblue', edgecolors='black', rasterized=True)

        # Add Walk-ID labels for outliers
        for i, (kl, emd, walk_id) in enumerate(zip(kl_divergences, emds, walk_stats.keys())):
            if kl > np.percentile(kl_divergences, 90) or emd > np.percentile(emds, 90):
                ax.annotate(f'W{walk_id}', (kl, emd), fontsize=8, alpha=0.7)

        ax.set_xlabel('KL Divergence', fontsize=12)
        ax.set_ylabel('Wasserstein Distance', fontsize=12)
        ax.set_title(f'KL Divergence vs Wasserstein Distance ({control_label})', fontsize=14)
        ax.grid(True, alpha=0.3)

        # Add correlation
        correlation = np.corrcoef(kl_divergences, emds)[0, 1]
        ax.text(0.05, 0.95, f'Correlation: {correlation:.3f}',
                transform=ax.transAxes, fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()
        figures.append(fig)

    elif metrics == 'all' and n_controls > 1:
        # Multiple controls: subplot for each
        fig, axes = plt.subplots(1, n_controls, figsize=(6*n_controls, 5))
        if n_controls == 2:
            axes = [axes[0], axes[1]]  # Ensure it's a list

        control_colors = ['steelblue', 'coral', 'mediumseagreen']

        for i, control_label in enumerate(control_labels):
            ax = axes[i] if n_controls > 1 else axes
            kl_divergences = [stats[f'kl_div_{control_label}'] for stats in walk_stats.values()]
            emds = [stats[f'emd_{control_label}'] for stats in walk_stats.values()]

            scatter = ax.scatter(kl_divergences, emds, alpha=0.6, s=100,
                                c=control_colors[i % len(control_colors)], edgecolors='black', rasterized=True)

            # Add Walk-ID labels for outliers
            for j, (kl, emd, walk_id) in enumerate(zip(kl_divergences, emds, walk_stats.keys())):
                if kl > np.percentile(kl_divergences, 90) or emd > np.percentile(emds, 90):
                    ax.annotate(f'W{walk_id}', (kl, emd), fontsize=8, alpha=0.7)

            ax.set_xlabel('KL Divergence', fontsize=11)
            ax.set_ylabel('Wasserstein Distance', fontsize=11)
            ax.set_title(f'{control_label}', fontsize=12)
            ax.grid(True, alpha=0.3)

            # Add correlation
            correlation = np.corrcoef(kl_divergences, emds)[0, 1]
            ax.text(0.05, 0.95, f'Corr: {correlation:.3f}',
                    transform=ax.transAxes, fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        fig.suptitle('KL Divergence vs Wasserstein Distance', fontsize=14)
        plt.tight_layout()
        figures.append(fig)

    print(f"Created {len(figures)} visualization(s)")
    return figures


def load_unknotted_globule_file(filepath):
    """
    Load a TSV file containing 10k-gon coordinates.
    Returns: numpy array of shape (10000, 3)
    """
    data = np.loadtxt(filepath, delimiter='\t')
    return data


def segment_and_sample_polygon_arc_length(polygon_10k, n_segments=20):
    """
    Divide a 10k-gon into segments of equal arc length and sample one point from each.
    This ensures each segment represents an equal physical length along the curve.

    Args:
        polygon_10k: numpy array of shape (N, 3)
        n_segments: number of segments to divide into (default: 20)

    Returns:
        numpy array of shape (n_segments, 3)
    """
    n_points = len(polygon_10k)

    # Calculate distances between consecutive points
    distances = np.sqrt(np.sum(np.diff(polygon_10k, axis=0)**2, axis=1))

    # Calculate cumulative arc length
    cumulative_length = np.zeros(n_points)
    cumulative_length[1:] = np.cumsum(distances)
    total_length = cumulative_length[-1]

    # Divide into equal arc-length segments
    segment_length = total_length / n_segments
    sampled_points = []

    for i in range(n_segments):
        # Find points within this arc-length segment
        start_length = i * segment_length
        end_length = (i + 1) * segment_length

        # Get indices of points in this segment
        # Handle last segment to include final point
        if i == n_segments - 1:
            segment_mask = (cumulative_length >= start_length) & (cumulative_length <= total_length)
        else:
            segment_mask = (cumulative_length >= start_length) & (cumulative_length < end_length)

        segment_indices = np.where(segment_mask)[0]

        if len(segment_indices) > 0:
            # Randomly select one point from this segment
            random_idx = np.random.choice(segment_indices)
            sampled_points.append(polygon_10k[random_idx])
        else:
            # Edge case: if no points in segment, use nearest point
            target_length = start_length + segment_length / 2
            nearest_idx = np.argmin(np.abs(cumulative_length - target_length))
            sampled_points.append(polygon_10k[nearest_idx])

    return np.array(sampled_points)


def segment_and_sample_polygon_point_count(polygon_10k, n_segments=20):
    """
    Divide a 10k-gon into segments by point count and sample one point from each.
    Original implementation that divides by number of points.

    Args:
        polygon_10k: numpy array of shape (N, 3)
        n_segments: number of segments to divide into (default: 20)

    Returns:
        numpy array of shape (n_segments, 3)
    """
    n_points = len(polygon_10k)
    segment_size = n_points // n_segments
    sampled_points = []

    for i in range(n_segments):
        start_idx = i * segment_size
        end_idx = start_idx + segment_size if i < n_segments - 1 else n_points
        segment = polygon_10k[start_idx:end_idx]

        # Randomly select one point from this segment
        random_idx = np.random.randint(0, len(segment))
        sampled_points.append(segment[random_idx])

    return np.array(sampled_points)


def segment_and_sample_polygon_uniform(polygon_10k, n_segments=20):
    """Pick n_segments random indices from the whole chain, sorted to preserve order.

    Mirrors `sample_random_polygons_uniform` from salmonella_random_20gon_classify.py.
    """
    n_points = len(polygon_10k)
    idx = np.sort(np.random.choice(n_points, size=n_segments, replace=False))
    return polygon_10k[idx]


def segment_and_sample_polygon_systematic(polygon_10k, n_segments=20):
    """Deterministic every-Nth-point downsample (stride = N // n_segments).

    Mirrors `downsample_systematic` from salmonella_random_20gon_classify.py.
    """
    n_points = len(polygon_10k)
    stride = max(1, n_points // n_segments)
    sampled = polygon_10k[::stride][:n_segments]
    if len(sampled) < n_segments:
        # Fallback: pad with evenly spaced indices
        idx = np.linspace(0, n_points - 1, n_segments).astype(int)
        sampled = polygon_10k[idx]
    return sampled


def segment_and_sample_polygon_centroid(polygon_10k, n_segments=20):
    """Replace each (n_points // n_segments) chunk with its centroid.

    Output vertices do NOT lie on the original curve. Mirrors
    `downsample_centroid` from salmonella_random_20gon_classify.py.
    """
    n_points = len(polygon_10k)
    chunk_size = n_points // n_segments
    centroids = np.empty((n_segments, polygon_10k.shape[1]))
    for i in range(n_segments):
        start = i * chunk_size
        end = start + chunk_size if i < n_segments - 1 else n_points
        centroids[i] = polygon_10k[start:end].mean(axis=0)
    return centroids


# Methods whose output is deterministic given the input polygon (no RNG).
_DETERMINISTIC_SEGMENT_METHODS = {'systematic', 'centroid'}


def segment_and_sample_polygon(polygon_10k, n_segments=20, method='arc_length'):
    """
    Unified function to segment and sample from a polygon.

    Args:
        polygon_10k: numpy array of shape (N, 3)
        n_segments: number of segments to divide into (default: 20)
        method: 'arc_length' (one random pt per equal-arc-length chunk),
                'point_count' / 'stride' (one random pt per equal-point-count chunk),
                'uniform' (N random points from the whole chain, sorted),
                'systematic' (deterministic every-Nth point),
                'centroid' (mean of each chunk; points off the curve).

    Returns:
        numpy array of shape (n_segments, 3)
    """
    if method == 'arc_length':
        return segment_and_sample_polygon_arc_length(polygon_10k, n_segments)
    elif method in ('point_count', 'stride'):
        return segment_and_sample_polygon_point_count(polygon_10k, n_segments)
    elif method == 'uniform':
        return segment_and_sample_polygon_uniform(polygon_10k, n_segments)
    elif method == 'systematic':
        return segment_and_sample_polygon_systematic(polygon_10k, n_segments)
    elif method == 'centroid':
        return segment_and_sample_polygon_centroid(polygon_10k, n_segments)
    else:
        raise ValueError(f"Unknown segmentation method: {method}")


def generate_globule_polygons(samples_per_file=5, globule_dir=None, segment_method='arc_length'):
    """
    Generate 20-gons from globule files (knotted or unknotted).

    Args:
        samples_per_file: number of 20-gons to generate from each 10k-gon file.
            Forced to 1 when ``segment_method`` is deterministic
            ('systematic' or 'centroid') since repeated samples are identical.
        globule_dir: directory containing the TSV files
        segment_method: one of 'arc_length', 'point_count' / 'stride',
            'uniform', 'systematic', 'centroid'.

    Returns:
        List of (knot_name, determinant) tuples
    """
    import glob

    if globule_dir is None:
        raise ValueError("globule_dir must be specified")

    if segment_method in _DETERMINISTIC_SEGMENT_METHODS and samples_per_file != 1:
        print(f"  Note: segment_method='{segment_method}' is deterministic; "
              f"forcing samples_per_file=1 (was {samples_per_file}).")
        samples_per_file = 1

    # Expand user home directory
    globule_dir = os.path.expanduser(globule_dir)

    # Get all TSV files in the directory
    tsv_files = sorted(glob.glob(os.path.join(globule_dir, '*.tsv')))

    globule_type = "knotted" if "knotted" in globule_dir else "unknotted"
    print(f"Found {len(tsv_files)} {globule_type} globule files")

    results = []

    for filepath in tqdm(tsv_files, desc=f"Processing {globule_type} globule files"):
        # Load the 10k-gon
        polygon_10k = load_unknotted_globule_file(filepath)

        # Scale coordinates if needed (assuming they need to be scaled similar to experimental data)
        # polygon_10k = polygon_10k * 1e-9  # Uncomment if scaling is needed

        # Generate multiple 20-gons from this file
        for sample_idx in range(samples_per_file):
            # Sample a 20-gon preserving order with the specified method
            polygon_20 = segment_and_sample_polygon(polygon_10k, n_segments=20, method=segment_method)

            # Calculate knot properties
            knot_name, determinant = calculate_knot_properties(polygon_20)
            results.append((knot_name, determinant))

    print(f"Generated {len(results)} 20-gons from {globule_type} globule files using {segment_method} segmentation")
    return results


def process_walk_parallel(args):
    """
    Helper function for parallel processing of walks.

    Parameters:
    -----------
    args : tuple
        (walk_id, walk_data, n_iterations) or
        (walk_id, walk_data, n_iterations, randomize_order)

    Returns:
    --------
    tuple : (walk_id, results_list)
        walk_id and list of (curve, knot_name, determinant) tuples
    """
    if len(args) == 4:
        walk_id, walk_data, n_iterations, randomize_order = args
    else:
        walk_id, walk_data, n_iterations = args
        randomize_order = False
    results = sample_trajectory_from_walk(walk_data, n_iterations,
                                          randomize_order=randomize_order)

    # Validate results before returning
    is_valid, error_msg = validate_knot_results(results, f"Walk-ID {walk_id} (parallel)")
    if not is_valid:
        print(f"Warning: Invalid results from parallel processing for Walk-ID {walk_id}: {error_msg}")
        print(f"  Results type: {type(results)}, Length: {len(results) if hasattr(results, '__len__') else 'N/A'}")
        if len(results) > 0:
            print(f"  First item: {results[0]}")
        # Return walk_id with empty list to avoid breaking the pipeline
        return walk_id, []

    return walk_id, results


def plot_acn_distributions(acn_values: list, control_acn_values: list,
                          walk_id: str, save_path: str, centroid_acn: float = None,
                          global_x_range: np.ndarray = None, global_y_max: float = None):
    """Create smooth density plot of ACN distributions using fill_between.
    Copied from acn.py to maintain consistency."""
    fig, ax = plt.subplots(figsize=(12, 7))

    # Remove NaN values
    acn_values = [x for x in acn_values if not np.isnan(x)]
    control_acn_values = [x for x in control_acn_values if not np.isnan(x)]

    # Determine range (use global range if provided)
    if global_x_range is not None:
        x_range = global_x_range
    else:
        all_values = acn_values + control_acn_values
        if centroid_acn is not None:
            all_values.append(centroid_acn)

        if not all_values:
            warnings.warn(f"No valid ACN values for walkid {walk_id}")
            return

        min_val, max_val = np.min(all_values), np.max(all_values)
        x_range = np.linspace(min_val - 0.1*(max_val-min_val), max_val + 0.1*(max_val-min_val), 200)

    # Define colors using seaborn color palette
    colors = sns.color_palette("deep")
    raw_color = colors[0]  # blue
    control_color = colors[7]  # gray
    centroid_color = colors[3]  # red

    # Compute kernel density estimates
    # Raw data KDE
    if len(acn_values) > 1:
        kde_raw = gaussian_kde(acn_values)
        density_raw = kde_raw(x_range)
        ax.fill_between(x_range, density_raw, alpha=0.6, color=raw_color, label='Raw')
        ax.plot(x_range, density_raw, color=raw_color, linewidth=2, alpha=0.8)

    # Control data KDE
    if len(control_acn_values) > 1:
        kde_control = gaussian_kde(control_acn_values)
        density_control = kde_control(x_range)
        ax.fill_between(x_range, density_control, alpha=0.4, color=control_color, label='Random Control')
        ax.plot(x_range, density_control, color=control_color, linewidth=2, alpha=0.8)

    # Add statistics
    bio_mean = np.mean(acn_values)
    bio_std = np.std(acn_values)
    ctrl_mean = np.mean(control_acn_values)
    ctrl_std = np.std(control_acn_values)

    # Add mean lines
    ax.axvline(bio_mean, color=raw_color, linestyle='--', linewidth=2.5, alpha=0.8)
    ax.axvline(ctrl_mean, color=control_color, linestyle='--', linewidth=2.5, alpha=0.8)

    # Add centroid ACN line if provided (callers pass centroid_acn only when
    # --show_centroids is set, so this is the documented gate)
    if centroid_acn is not None:
        ax.axvline(centroid_acn, color=centroid_color, linestyle='-', linewidth=3,
                   alpha=0.9, label=f'Centroid: {centroid_acn:.2f}')

    # Styling
    ax.set_xlabel('Average Crossing Number', fontsize=14, fontweight='bold')
    ax.set_ylabel('Probability Density', fontsize=14, fontweight='bold')
    ax.set_title(f'Average Crossing Number Distribution - Walk ID {walk_id}', fontsize=16, pad=15, fontweight='bold')

    # Custom legend
    handles, labels = ax.get_legend_handles_labels()
    legend = ax.legend(handles, labels, loc='upper right', frameon=True,
                      fancybox=True, shadow=True, fontsize=12)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.9)

    # Set y-axis limit if provided (starting at 0)
    if global_y_max is not None:
        ax.set_ylim(0, global_y_max)
    else:
        # Always start at 0
        ax.set_ylim(bottom=0)

    # Add text box with statistics
    stats_text = f'Raw: μ={bio_mean:.2f}, σ={bio_std:.2f}\n'
    stats_text += f'Control: μ={ctrl_mean:.2f}, σ={ctrl_std:.2f}'
    if centroid_acn is not None:
        stats_text += f'\nCentroid: {centroid_acn:.2f}'

    # Create text box with better styling
    props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='gray')
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='top', bbox=props)

    # Improve grid
    ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)

    # Add subtle spines
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
        spine.set_alpha(0.5)

    # Tight layout
    plt.tight_layout()

    # Return the figure (don't save or close it)
    return fig


def create_acn_visualizations(walk_acn_data, control_acn_values, output_prefix,
                              global_x_range=None, global_y_max=None,
                              show_centroids=False):
    """
    Create ACN distribution visualizations using plot_acn_distributions for each walk.

    Parameters:
    -----------
    walk_acn_data : dict
        Dictionary with walk_id as key and dict containing:
        - 'centroid_acn': ACN of centroid curve
        - 'sampled_acn': list of ACN values from N sampled curves
    control_acn_values : list
        List of ACN values from control distribution
    output_prefix : str
        Prefix for output files
    global_x_range : np.ndarray, optional
        Global x-axis range for consistent plotting
    global_y_max : float, optional
        Global y-axis maximum for consistent plotting
    """
    print("Creating ACN visualizations...")

    walk_ids = list(walk_acn_data.keys())
    n_walks = len(walk_ids)
    figures = []

    # Create individual plot for each walk using plot_acn_distributions
    for walk_id in walk_ids:
        data = walk_acn_data[walk_id]
        centroid_acn = data['centroid_acn']
        sampled_acn = data['sampled_acn']

        # Call plot_acn_distributions for this walk (returns figure)
        fig = plot_acn_distributions(
            acn_values=sampled_acn,
            control_acn_values=control_acn_values,
            walk_id=walk_id,
            save_path=None,  # Not needed anymore
            centroid_acn=centroid_acn if show_centroids else None,
            global_x_range=global_x_range,
            global_y_max=global_y_max
        )
        if fig is not None:
            figures.append(fig)

    print(f"  Created {len(figures)} ACN visualization(s)")
    return figures


def create_kendall_visualizations(walk_kendall_data, output_prefix):
    """
    Create Kendall isogon 3D visualizations with 6 plots per page.

    Parameters:
    -----------
    walk_kendall_data : dict
        Dictionary with walk_id as key and dict containing:
        - 'average_isogon': 20×3 array of average isogon
        - 'converged': boolean
        - 'iterations': number of iterations
        - 'mean_procrustes_dist': mean Procrustes distance
        - 'knot_type': knot type of average isogon
        - 'determinant': determinant
    output_prefix : str
        Prefix for output files
    """
    print("Creating Kendall shape analysis visualizations...")

    # Set up the plot style
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['savefig.dpi'] = 300

    walk_ids = list(walk_kendall_data.keys())
    n_walks = len(walk_ids)
    plots_per_page = 6
    n_pages = (n_walks + plots_per_page - 1) // plots_per_page
    figures = []

    for page in range(n_pages):
        fig = plt.figure(figsize=(15, 10))
        fig.suptitle('Kendall Average Icosagons', fontsize=16, fontweight='bold')

        start_idx = page * plots_per_page
        end_idx = min(start_idx + plots_per_page, n_walks)
        page_walk_ids = walk_ids[start_idx:end_idx]

        # Create subplots
        for idx, walk_id in enumerate(page_walk_ids):
            ax = fig.add_subplot(2, 3, idx + 1, projection='3d')
            data = walk_kendall_data[walk_id]

            average_isogon = data['average_isogon']

            # Plot the isogon
            xs = average_isogon[:, 0]
            ys = average_isogon[:, 1]
            zs = average_isogon[:, 2]

            # Plot vertices
            ax.scatter(xs, ys, zs, c=np.arange(20), cmap='viridis', s=50, alpha=0.8, rasterized=True)

            # Plot edges
            for i in range(20):
                j = (i + 1) % 20
                ax.plot([xs[i], xs[j]], [ys[i], ys[j]], [zs[i], zs[j]],
                       'k-', alpha=0.6, linewidth=1.5)

            # Labels
            ax.set_xlabel('X', fontsize=8)
            ax.set_ylabel('Y', fontsize=8)
            ax.set_zlabel('Z', fontsize=8)
            ax.set_title(f'Walk-ID {walk_id}', fontsize=12, fontweight='bold')

            # Statistics text
            stats_text = f"Knot: {data['knot_type']}\n"
            stats_text += f"Det: {data['determinant']:.1f}\n"
            stats_text += f"Mean Proc. Dist: {data['mean_procrustes_dist']:.3f}\n"
            stats_text += f"Converged: {data['converged']}"

            # Add text annotation
            ax.text2D(0.02, 0.98, stats_text, transform=ax.transAxes,
                     fontsize=7, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        # Hide empty subplots
        for idx in range(len(page_walk_ids), 6):
            ax = fig.add_subplot(2, 3, idx + 1)
            ax.axis('off')

        plt.tight_layout()

        # Add this page's figure to the list
        figures.append(fig)

    print(f"  Created {len(figures)} Kendall visualization page(s)")
    return figures


def create_morans_i_correlation_plot(results_df, output_prefix):
    """
    Create Moran's I correlation plots.

    Parameters:
    -----------
    results_df : pd.DataFrame
        Results dataframe with columns including:
        - Walk_ID, Morans_I, Distance_TP1_to_TP21, ACN_mean, knot fractions
    output_prefix : str
        Prefix for output files
    """
    print("Creating Moran's I correlation plot...")

    # Filter out walks without Moran's I data
    df_with_morans = results_df[results_df['Morans_I'].notna()].copy()

    if len(df_with_morans) == 0:
        print("  Warning: No Moran's I data available for correlation plot")
        return

    # Create knot type classification for coloring
    def classify_knot_type(row):
        if row['Unknot_fraction'] > 0.5:
            return 'Unknot-dominant'
        elif row['Trefoil_fraction'] > 0.3:
            return 'Trefoil-enriched'
        else:
            return 'Complex-enriched'

    df_with_morans['knot_class'] = df_with_morans.apply(classify_knot_type, axis=1)

    # Create 2×2 panel plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Moran's I Correlation Analysis", fontsize=16, fontweight='bold')

    colors_map = {
        'Unknot-dominant': '#2ecc71',
        'Trefoil-enriched': '#ff9800',
        'Complex-enriched': '#e74c3c'
    }

    # Panel 1: Moran's I vs Distance (TP1→TP21)
    ax = axes[0, 0]
    for knot_class, color in colors_map.items():
        mask = df_with_morans['knot_class'] == knot_class
        ax.scatter(df_with_morans[mask]['Distance_TP1_to_TP21'],
                  df_with_morans[mask]['Morans_I'],
                  c=color, label=knot_class, alpha=0.7, s=80, edgecolors='black', rasterized=True)

    # Regression line
    from scipy.stats import linregress
    x = df_with_morans['Distance_TP1_to_TP21'].values
    y = df_with_morans['Morans_I'].values
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    ax.plot(x, slope * x + intercept, 'k--', linewidth=2, alpha=0.5)

    ax.set_xlabel('Euclidean Distance TP1→TP21 (nm)', fontsize=11)
    ax.set_ylabel("Moran's I", fontsize=11)
    ax.set_title("Moran's I vs Distance", fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Add R² and p-value
    stats_text = f'R² = {r_value**2:.3f}\np = {p_value:.4f}'
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
           fontsize=9, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Panel 2: Moran's I vs Mean ACN
    ax = axes[0, 1]
    if 'ACN_mean' in df_with_morans.columns:
        df_with_acn = df_with_morans[df_with_morans['ACN_mean'].notna()]

        for knot_class, color in colors_map.items():
            mask = df_with_acn['knot_class'] == knot_class
            ax.scatter(df_with_acn[mask]['ACN_mean'],
                      df_with_acn[mask]['Morans_I'],
                      c=color, label=knot_class, alpha=0.7, s=80, edgecolors='black', rasterized=True)

        # Regression line
        if len(df_with_acn) > 1:
            x = df_with_acn['ACN_mean'].values
            y = df_with_acn['Morans_I'].values
            slope, intercept, r_value, p_value, std_err = linregress(x, y)
            ax.plot(x, slope * x + intercept, 'k--', linewidth=2, alpha=0.5)

            stats_text = f'R² = {r_value**2:.3f}\np = {p_value:.4f}'
            ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
                   fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax.set_xlabel('Mean ACN', fontsize=11)
        ax.set_ylabel("Moran's I", fontsize=11)
        ax.set_title("Moran's I vs Mean ACN", fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'ACN data not available', transform=ax.transAxes,
               ha='center', va='center', fontsize=12)

    # Panel 3: Distance vs Mean ACN
    ax = axes[1, 0]
    if 'ACN_mean' in df_with_morans.columns:
        df_with_acn = df_with_morans[df_with_morans['ACN_mean'].notna()]

        for knot_class, color in colors_map.items():
            mask = df_with_acn['knot_class'] == knot_class
            ax.scatter(df_with_acn[mask]['Distance_TP1_to_TP21'],
                      df_with_acn[mask]['ACN_mean'],
                      c=color, label=knot_class, alpha=0.7, s=80, edgecolors='black', rasterized=True)

        # Regression line
        if len(df_with_acn) > 1:
            x = df_with_acn['Distance_TP1_to_TP21'].values
            y = df_with_acn['ACN_mean'].values
            slope, intercept, r_value, p_value, std_err = linregress(x, y)
            ax.plot(x, slope * x + intercept, 'k--', linewidth=2, alpha=0.5)

            stats_text = f'R² = {r_value**2:.3f}\np = {p_value:.4f}'
            ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
                   fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax.set_xlabel('Euclidean Distance TP1→TP21 (nm)', fontsize=11)
        ax.set_ylabel('Mean ACN', fontsize=11)
        ax.set_title('Distance vs Mean ACN', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'ACN data not available', transform=ax.transAxes,
               ha='center', va='center', fontsize=12)

    # Panel 4: Summary statistics
    ax = axes[1, 1]
    ax.axis('off')

    summary_text = "Summary Statistics\n" + "="*30 + "\n\n"
    summary_text += f"Number of walks analyzed: {len(df_with_morans)}\n\n"
    summary_text += f"Moran's I range:\n"
    summary_text += f"  Min: {df_with_morans['Morans_I'].min():.3f}\n"
    summary_text += f"  Max: {df_with_morans['Morans_I'].max():.3f}\n"
    summary_text += f"  Mean: {df_with_morans['Morans_I'].mean():.3f}\n"
    summary_text += f"  Std: {df_with_morans['Morans_I'].std():.3f}\n\n"
    summary_text += f"Distance TP1→TP21 range:\n"
    summary_text += f"  Min: {df_with_morans['Distance_TP1_to_TP21'].min():.1f} nm\n"
    summary_text += f"  Max: {df_with_morans['Distance_TP1_to_TP21'].max():.1f} nm\n"
    summary_text += f"  Mean: {df_with_morans['Distance_TP1_to_TP21'].mean():.1f} nm\n"

    if 'ACN_mean' in df_with_morans.columns:
        df_with_acn = df_with_morans[df_with_morans['ACN_mean'].notna()]
        if len(df_with_acn) > 0:
            summary_text += f"\nMean ACN range:\n"
            summary_text += f"  Min: {df_with_acn['ACN_mean'].min():.2f}\n"
            summary_text += f"  Max: {df_with_acn['ACN_mean'].max():.2f}\n"
            summary_text += f"  Mean: {df_with_acn['ACN_mean'].mean():.2f}\n"

    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top', family='monospace',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()

    print(f"  Created Moran's I correlation plot")
    return fig


def create_comprehensive_pdf_report(all_figures, output_prefix):
    """
    Assemble all plots into a single multi-page PDF report.

    Parameters:
    -----------
    all_figures : list
        List of matplotlib figure objects
    output_prefix : str
        Output filename prefix
    """
    if not all_figures:
        print("Warning: No figures to save in PDF report")
        return

    pdf_path = f'{output_prefix}_comprehensive_report.pdf'

    print(f"\nAssembling comprehensive PDF report...")
    print(f"  Total figures: {len(all_figures)}")

    with PdfPages(pdf_path) as pdf:
        for i, fig in enumerate(all_figures, 1):
            if fig is not None:
                try:
                    pdf.savefig(fig, bbox_inches='tight')
                    plt.close(fig)
                except Exception as e:
                    print(f"  Warning: Could not save figure {i}: {e}")

    print(f"  Comprehensive report saved to: {pdf_path}")
    print(f"  Total pages: {len(all_figures)}")


# ============================================================================
# Population-level parabola & distance-matrix pages for De Witt curves
# (uses helpers from Salmonella_WF/qc_post_filter.py and the constrained PCA
# transform from Salmonella_WF/report_cards_single_curve.py)
# ============================================================================
SALMONELLA_WF_DIR = '/Users/dimos/Documents/GitHub/Salmonella_WF'


def _import_qc_helpers():
    """Try to import the qc_post_filter figure builders. Returns dict or None.

    The constrained-PCA transform is vendored inline below to avoid pulling
    in the rest of report_cards_single_curve.py (which has numba caching
    that breaks when imported from another package).
    """
    try:
        if SALMONELLA_WF_DIR not in sys.path:
            sys.path.insert(0, SALMONELLA_WF_DIR)
        from qc_post_filter import (
            build_mean_distance_matrices_figure,
            build_mean_parabola_xyz_figure,
            build_mean_parabola_3d_only_figure,
        )
        return {
            'distance_matrices': build_mean_distance_matrices_figure,
            'parabola_xyz': build_mean_parabola_xyz_figure,
            'parabola_3d': build_mean_parabola_3d_only_figure,
            'to_pc': walks_to_constrained_pc,
        }
    except Exception as e:
        print(f"  Warning: qc_post_filter helpers not importable from {SALMONELLA_WF_DIR}: {e}")
        return None


# ---- vendored constrained-PCA transform (from report_cards_single_curve.py) ----
def _constrained_axes_from_pc1(pc1):
    pc1 = np.asarray(pc1, dtype=float)
    pc1 = pc1 / np.linalg.norm(pc1)
    xy_norm = np.linalg.norm(pc1[:2])
    if xy_norm < 1e-9:
        pc2 = np.array([1.0, 0.0, 0.0])
        pc3 = np.array([0.0, 1.0, 0.0])
    else:
        u = pc1[:2] / xy_norm
        pc2 = np.array([-u[1], u[0], 0.0])
        pc3 = np.cross(pc1, pc2)
        pc3 = pc3 / np.linalg.norm(pc3)
    return pc1, pc2, pc3


def _canonical_orient_pc(pc1, pc2, pc3):
    if pc1[2] < 0:
        pc1 = -pc1
    if pc2[0] < 0:
        pc2 = -pc2
    expected_pc3 = np.cross(pc1, pc2)
    if np.dot(pc3, expected_pc3) < 0:
        pc3 = -pc3
    return pc1, pc2, pc3


def _walk_to_constrained_pc(curve):
    """Project one (N, 3) curve into canonical constrained PC1/PC2/PC3."""
    center = curve.mean(axis=0)
    centered = curve - center
    cov = np.cov(centered, rowvar=False)
    _, eigvecs = np.linalg.eigh(cov)
    pc1 = eigvecs[:, -1]
    pc1, pc2, pc3 = _constrained_axes_from_pc1(pc1)
    pc1, pc2, pc3 = _canonical_orient_pc(pc1, pc2, pc3)
    R = np.column_stack([pc1, pc2, pc3])
    return centered @ R


def walks_to_constrained_pc(walks):
    return {wid: _walk_to_constrained_pc(np.asarray(curve, dtype=float))
            for wid, curve in walks.items()}


def _build_dewitt_population(walk_results):
    """Flatten {(walk_id, sample_idx): (n, 3) curve} from walk_results."""
    pop = {}
    for walk_id, samples in walk_results.items():
        for i, item in enumerate(samples):
            curve = np.asarray(item[0], dtype=float)
            if curve.ndim != 2 or curve.shape[1] != 3:
                continue
            pop[(walk_id, i)] = curve
    return pop


def _tidy_parabola_fig(fig):
    """Post-process a parabola figure produced by qc_post_filter helpers:
      - Force major xticks every 5, kill minor ticks.
      - Strip "(n=...)" from the legend entry for the included-in-fit points.
    """
    import re
    from matplotlib.ticker import MultipleLocator
    for ax in fig.axes:
        try:
            ax.xaxis.set_major_locator(MultipleLocator(5))
            ax.minorticks_off()
        except Exception:
            pass
        legend = ax.get_legend()
        if legend is None:
            continue
        handles, labels = ax.get_legend_handles_labels()
        new_labels = []
        for lab in labels:
            new_labels.append(re.sub(r'\s*\(n=\d+\)', '', lab))
        if new_labels != labels:
            ax.legend(handles, new_labels, loc=legend._loc, fontsize=8)
    return fig


def build_population_parabola_pages(walk_results, label=''):
    """Generate parabola + distance-matrix figures from the De Witt curve population.

    Returns dict:
      'main_pages'        -> list of figures for the main PDF (XYZ parabola,
                              PC parabola, XYZ matrices, PC matrices)
      'standalone_3d_xyz' -> Figure of 3D-only parabola in XYZ space (or None)
      'standalone_3d_pc'  -> Figure of 3D-only parabola in PC space (or None)
    """
    out = {'main_pages': [], 'standalone_3d_xyz': None, 'standalone_3d_pc': None}
    helpers = _import_qc_helpers()
    if helpers is None:
        return out

    pop = _build_dewitt_population(walk_results)
    if not pop:
        print("  No De Witt curves available for population parabola pages.")
        return out

    n_vertices = next(iter(pop.values())).shape[0]
    tp_start = 1
    tp_end = n_vertices

    print(f"  Building population parabola/distance-matrix pages "
          f"({len(pop)} De Witt curves, {n_vertices} vertices each)...")

    try:
        pop_pc = helpers['to_pc'](pop)
    except Exception as e:
        print(f"  Warning: PCA transform failed: {e}")
        pop_pc = None

    suffix = f' ({label})' if label else ''

    # XYZ parabola
    try:
        fig, _, _ = helpers['parabola_xyz'](
            pop, tp_start, tp_end, n_tp_in_range=n_vertices,
            axis_names=('X', 'Y', 'Z'),
            space_label=f'XYZ - De Witt population{suffix}')
        out['main_pages'].append(_tidy_parabola_fig(fig))
    except Exception as e:
        print(f"  Warning: XYZ parabola page failed: {e}")

    # PC-space parabola
    if pop_pc is not None:
        try:
            fig, _, _ = helpers['parabola_xyz'](
                pop_pc, tp_start, tp_end, n_tp_in_range=n_vertices,
                axis_names=('PC1', 'PC2', 'PC3'),
                space_label=f'PC space - De Witt population{suffix}')
            out['main_pages'].append(_tidy_parabola_fig(fig))
        except Exception as e:
            print(f"  Warning: PC parabola page failed: {e}")

    # XYZ distance matrices
    try:
        fig, _ = helpers['distance_matrices'](
            pop, tp_start, tp_end,
            axis_names=('X', 'Y', 'Z'),
            space_label=f'XYZ - De Witt population{suffix}')
        out['main_pages'].append(fig)
    except Exception as e:
        print(f"  Warning: XYZ distance-matrix page failed: {e}")

    # PC-space distance matrices
    if pop_pc is not None:
        try:
            fig, _ = helpers['distance_matrices'](
                pop_pc, tp_start, tp_end,
                axis_names=('PC1', 'PC2', 'PC3'),
                space_label=f'PC space - De Witt population{suffix}')
            out['main_pages'].append(fig)
        except Exception as e:
            print(f"  Warning: PC distance-matrix page failed: {e}")

    # Standalone 3D-only parabolas (sidecar PDFs)
    try:
        fig, _, _ = helpers['parabola_3d'](
            pop, tp_start, tp_end, n_tp_in_range=n_vertices,
            space_label=f'XYZ - De Witt population{suffix}')
        out['standalone_3d_xyz'] = _tidy_parabola_fig(fig)
    except Exception as e:
        print(f"  Warning: XYZ 3D-only parabola failed: {e}")
    if pop_pc is not None:
        try:
            fig, _, _ = helpers['parabola_3d'](
                pop_pc, tp_start, tp_end, n_tp_in_range=n_vertices,
                space_label=f'PC space - De Witt population{suffix}')
            out['standalone_3d_pc'] = _tidy_parabola_fig(fig)
        except Exception as e:
            print(f"  Warning: PC 3D-only parabola failed: {e}")

    return out


def main():
    parser = argparse.ArgumentParser(description='Compare knot distributions between experimental data and random polygons')
    parser.add_argument('N', type=int, help='Number of iterations for sampling from Walk-IDs')
    parser.add_argument('M', type=int, nargs='?', default=1000,
                       help='Number of random 20-gons to generate (default: 1000)')
    parser.add_argument('--input', type=str,
                       default='/Users/dimos/knot_analysis_Nir/datasets/New_data_08_18_2025_24walks.csv',
                       help='Input CSV file path')
    parser.add_argument('--output', type=str, default='knot_distribution_comparison',
                       help='Output file prefix for results')
    parser.add_argument('--parallel', action='store_true',
                       help='Use parallel processing for Walk-IDs')
    parser.add_argument('--workers', type=int, default=None,
                       help='Number of parallel workers (default: number of CPUs)')
    parser.add_argument('--verbose', action='store_true',
                       help='Show KnotoID output (default: suppressed if wurlitzer is installed)')
    parser.add_argument('--unknotted_globule', action='store_true',
                       help='Use unknotted globule control')
    parser.add_argument('--knotted_globule', action='store_true',
                       help='Use knotted globule control')
    parser.add_argument('--unknotted_globule_dir', type=str,
                       default='/Users/dimos/Salmonella_sims_model1/rotated_tsv/unknotted',
                       help='Directory containing unknotted globule TSV files')
    parser.add_argument('--knotted_globule_dir', type=str,
                       default='/Users/dimos/Salmonella_sims_model1/rotated_tsv/knotted_run3',
                       help='Directory containing knotted globule TSV files')
    parser.add_argument('--samples_per_globule', type=int, default=5,
                       help='Number of 20-gons to sample from each 10k-gon file (default: 5)')
    parser.add_argument('--segment_by',
                       choices=['arc_length', 'point_count', 'stride',
                                'uniform', 'systematic', 'centroid'],
                       default='arc_length',
                       help='Downsampling method for 10k-gons (default: arc_length). '
                            "'arc_length' / 'point_count' / 'stride': one random point per "
                            "equal-arc-length or equal-count chunk. 'uniform': N random "
                            "points from the whole chain, sorted. 'systematic': deterministic "
                            "every-Nth point. 'centroid': mean of each chunk (off-curve). "
                            "Deterministic methods force samples_per_globule=1.")
    parser.add_argument('--show_random', action='store_true',
                       help='Include random polygon control when using globule controls')
    parser.add_argument('--metrics', choices=['all', 'kl', 'wasserstein'],
                       default='all',
                       help='Which metrics to compute (default: all)')
    parser.add_argument('--acn', action='store_true',
                       help='Enable ACN (Average Crossing Number) computation')
    parser.add_argument('--n_acn_projections', type=int, default=1000,
                       help='Number of projections for ACN computation (default: 1000)')
    parser.add_argument('--kendall', action='store_true',
                       help='Enable Kendall shape space analysis')
    parser.add_argument('--kendall_max_iter', type=int, default=100,
                       help='Maximum Kendall iterations (default: 100)')
    parser.add_argument('--kendall_tol', type=float, default=1e-6,
                       help='Kendall convergence tolerance (default: 1e-6)')
    parser.add_argument('--vassiliev', action='store_true',
                       help='Enable Vassiliev V2 invariant computation')
    parser.add_argument('--num_vassiliev_proj', type=int, default=100,
                       help='Number of projections for V2 computation (default: 100)')
    parser.add_argument('--xyCN', action='store_true',
                       help='Enable xy-plane crossing number computation (mutually exclusive with --vassiliev)')
    parser.add_argument('--xzCN', action='store_true',
                       help='Enable xz-plane crossing number computation (mutually exclusive with --vassiliev)')
    parser.add_argument('--yzCN', action='store_true',
                       help='Enable yz-plane crossing number computation (mutually exclusive with --vassiliev)')
    parser.add_argument('--rog', action='store_true',
                       help='Enable Radius of Gyration computation')
    parser.add_argument('--puknot', action='store_true',
                       help='Show P(unknot) convergence plot vs number of DeWitt curves')
    parser.add_argument('--label', type=str, default='',
                       help='Label for title (e.g. "d<100nm")')
    parser.add_argument('--randomize_order', action='store_true',
                       help='Connect sampled De Witt points in random order '
                            '(permute the 20 vertices) instead of genomic order. '
                            'Use as a control: genomic structure should be destroyed.')
    parser.add_argument('--no_stacked_barplots', action='store_true',
                       help='Per-walk pages: only Unknotted/Knotted bars (no controls). '
                            'Also adds a 2-category version of the first page after the original.')
    parser.add_argument('--show_centroids', action='store_true',
                       help='Re-enable centroid display: adds a "Centroid Curves" bar to '
                            'the first-page knot-distribution barplot, and draws solid red '
                            'centroid lines on the per-walk ACN and RMS-Writhe distribution '
                            'panels of every report card.')
    parser.add_argument('--morans_i_file', type=str,
                       default='../Knot_analysis_pipeline/BiggieExp3_09182025_97walks_morans_i_results.csv',
                       help="Path to Moran's I CSV file")

    args = parser.parse_args()

    # Check mutual exclusivity
    if args.vassiliev and (args.xyCN or args.xzCN or args.yzCN):
        parser.error("--vassiliev is mutually exclusive with planar CN flags (--xyCN, --xzCN, --yzCN)")

    # Handle verbosity settings
    global WURLITZER_AVAILABLE
    if args.verbose and WURLITZER_AVAILABLE:
        print("Verbose mode: KnotoID output will be shown")
        WURLITZER_AVAILABLE = False  # Temporarily disable wurlitzer

    print(f"=" * 60)
    print(f"Knot Distribution Comparison Analysis")
    print(f"=" * 60)
    print(f"Parameters:")
    print(f"  N (iterations per Walk-ID): {args.N}")

    # Print control information
    controls_to_use = []
    if not (args.unknotted_globule or args.knotted_globule) or args.show_random:
        controls_to_use.append("Random Polygons")
        print(f"  Control: Random Polygons")
        print(f"    M (random polygons): {args.M}")
    if args.unknotted_globule:
        controls_to_use.append("Unknotted Globule")
        print(f"  Control: Unknotted Globule")
        print(f"    Samples per globule file: {args.samples_per_globule}")
        print(f"    Segmentation method: {args.segment_by}")
    if args.knotted_globule:
        controls_to_use.append("Knotted Globule")
        print(f"  Control: Knotted Globule")
        print(f"    Samples per globule file: {args.samples_per_globule}")
        print(f"    Segmentation method: {args.segment_by}")

    print(f"  Metrics to compute: {args.metrics}")

    print(f"  Input file: {args.input}")
    print(f"  Output prefix: {args.output}")
    print(f"  Parallel processing: {args.parallel}")
    if WURLITZER_AVAILABLE:
        print(f"  KnotoID output: Suppressed (use --verbose to show)")
    else:
        print(f"  KnotoID output: Visible")
    print(f"=" * 60)

    # Load and preprocess data
    walk_groups = load_and_preprocess_data(args.input)

    # Sample trajectories from each Walk-ID
    print(f"\nSampling {args.N} trajectories from each Walk-ID...")
    walk_results = {}
    all_sampled_polygons = []  # Store all sampled polygons for convex hull computation

    if args.parallel:
        # Parallel processing
        n_workers = args.workers if args.workers else multiprocessing.cpu_count()
        print(f"Using {n_workers} parallel workers...")

        # Prepare arguments for parallel processing
        parallel_args = [(walk_id, walk_data, args.N, args.randomize_order)
                        for walk_id, walk_data in walk_groups.items()]

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            results = list(tqdm(
                executor.map(process_walk_parallel, parallel_args),
                total=len(parallel_args),
                desc="Processing Walk-IDs"
            ))

        walk_results = dict(results)
    else:
        # Sequential processing
        for walk_id, walk_data in tqdm(walk_groups.items(), desc="Processing Walk-IDs"):
            walk_results[walk_id] = sample_trajectory_from_walk(
                walk_data, args.N, randomize_order=args.randomize_order)

    # Validate all walk_results after collection
    print("\nValidating walk results...")
    invalid_walks = []
    for walk_id, walk_data in walk_results.items():
        is_valid, error_msg = validate_knot_results(walk_data, f"Walk-ID {walk_id}")
        if not is_valid:
            print(f"  ERROR: Walk-ID {walk_id} has invalid data structure: {error_msg}")
            print(f"    Type: {type(walk_data)}, Length: {len(walk_data) if hasattr(walk_data, '__len__') else 'N/A'}")
            if hasattr(walk_data, '__len__') and len(walk_data) > 0:
                print(f"    First item: {walk_data[0]}")
                print(f"    Item type: {type(walk_data[0])}")
                if hasattr(walk_data[0], '__len__'):
                    print(f"    Item length: {len(walk_data[0])}")
            invalid_walks.append(walk_id)

    if invalid_walks:
        print(f"\n  WARNING: Found {len(invalid_walks)} walk(s) with invalid data structures:")
        print(f"  {invalid_walks}")
        print(f"  These walks will be skipped during report card generation.")
        # Remove invalid walks from walk_results
        for walk_id in invalid_walks:
            del walk_results[walk_id]
    else:
        print(f"  All {len(walk_results)} walks validated successfully!")

    # Collect all sampled polygons for convex hull computation
    print("\nCollecting sampled polygons for convex hull computation...")
    for walk_id, walk_data_results in walk_results.items():
        # Extract curves directly from walk_results (no need to resample!)
        for curve, knot_name, determinant in walk_data_results:
            all_sampled_polygons.append(curve)

    # Compute average convex hull from sampled polygons
    print(f"Computing convex hull from {len(all_sampled_polygons)} sampled polygons...")
    avg_hull = compute_average_convex_hull(all_sampled_polygons)

    # Determine which controls to use
    use_random = not (args.unknotted_globule or args.knotted_globule) or args.show_random
    controls = {}

    # Generate control polygons based on flags
    if use_random:
        print("\nUsing random polygon control...")
        controls['Random'] = generate_random_polygons(args.M, convex_hull=avg_hull)

    if args.unknotted_globule:
        print("\nUsing unknotted globule control...")
        controls['Unknotted Globule'] = generate_globule_polygons(
            samples_per_file=args.samples_per_globule,
            globule_dir=args.unknotted_globule_dir,
            segment_method=args.segment_by
        )

    if args.knotted_globule:
        print("\nUsing knotted globule control...")
        controls['Knotted Globule'] = generate_globule_polygons(
            samples_per_file=args.samples_per_globule,
            globule_dir=args.knotted_globule_dir,
            segment_method=args.segment_by
        )

    # ========================================================================
    # ACN, Kendall, and Moran's I Analysis
    # ========================================================================

    walk_acn_data = {}
    walk_kendall_data = {}
    walk_v2_data = {}
    walk_xycn_data = {}
    walk_xzcn_data = {}
    walk_yzcn_data = {}
    walk_centroid_data = {}
    morans_i_dict = {}

    # Load Moran's I data if available
    if os.path.exists(args.morans_i_file):
        print(f"\nLoading Moran's I data from {args.morans_i_file}...")
        morans_i_dict = load_morans_i_data(args.morans_i_file)
    else:
        print(f"\nMoran's I file not found: {args.morans_i_file}")

    # Compute weighted centroids, ACN, Kendall, and V2 analysis
    if args.acn or args.kendall or args.vassiliev or args.puknot:
        print("\nComputing weighted centroids for all walks...")

        for walk_id in tqdm(walk_results.keys(), desc="Computing centroids"):
            walk_data = walk_groups[walk_id]

            # Ensure we have required precision columns
            if 'precisionx' not in walk_data.columns:
                print(f"Warning: Walk-ID {walk_id} missing precision columns. Using uniform weights.")
                walk_data['precisionx'] = 1.0
                walk_data['precisiony'] = 1.0
                walk_data['precisionz'] = 1.0

            # Compute weighted centroid curve
            try:
                centroid_curve = compute_weighted_centroids(walk_data)
                walk_centroid_data[walk_id] = centroid_curve
            except Exception as e:
                print(f"Error computing centroid for Walk-ID {walk_id}: {e}")
                continue

    # Analyze knot types of centroid curves
    centroid_knot_types = {}
    if walk_centroid_data:
        print("\nAnalyzing knot types of centroid curves...")
        for walk_id, centroid_curve in tqdm(walk_centroid_data.items(), desc="Centroid knot types"):
            try:
                knot_name, determinant = calculate_knot_properties(centroid_curve)
                centroid_knot_types[walk_id] = (knot_name, determinant)
            except Exception as e:
                print(f"  Error analyzing centroid for Walk-ID {walk_id}: {e}")
                centroid_knot_types[walk_id] = ('Unknown', 0.0)

    # Compute V2 for centroid curves
    if args.vassiliev and walk_centroid_data:
        print(f"\nComputing V2 invariant for centroid curves (n_projections={args.num_vassiliev_proj})...")
        for walk_id, centroid_curve in tqdm(walk_centroid_data.items(), desc="Centroid V2"):
            try:
                centroid_v2, _ = compute_v2_invariant(
                    centroid_curve,
                    closed=True,
                    n_resample=20,
                    n_projections=args.num_vassiliev_proj,
                    show_progress=False
                )
                walk_v2_data[walk_id] = {'centroid_v2': centroid_v2, 'sampled_v2': []}
            except Exception as e:
                print(f"  Error computing centroid V2 for Walk-ID {walk_id}: {e}")
                walk_v2_data[walk_id] = {'centroid_v2': np.nan, 'sampled_v2': []}

    # Skip TP1→TP21 distance computation (not used in plots, dataset-specific)
    walk_tp1_tp21_distances = {}

    # ACN and Writhe Computation
    control_acn_values = []  # Initialize empty list
    control_writhe_values = []  # Initialize empty list for writhe
    control_rms_writhe_values = []  # Initialize empty list for RMS writhe
    global_x_range = None
    global_y_max = None
    global_writhe_x_range = None
    global_writhe_y_max = None
    global_rms_writhe_x_range = None
    global_rms_writhe_y_max = None
    walk_writhe_data = {}  # Per-walk writhe data (includes RMS writhe)

    # Radius of Gyration Computation
    control_rog_values = []  # Initialize empty list for RoG
    global_rog_x_range = None
    global_rog_y_max = None
    walk_rog_data = {}  # Per-walk RoG data
    control_rog_mean = 0.0
    control_rog_std = 1.0

    if args.acn:
        print(f"\nComputing ACN with {args.n_acn_projections} projections...")

        # STEP 1: Compute SHARED control ACN and Writhe distribution FIRST
        print("  Computing SHARED control ACN/Writhe distribution for all walks...")
        n_control_samples = 200  # Use more samples for robust control

        if avg_hull is not None:
            for i in tqdm(range(n_control_samples), desc="  Control ACN/Writhe"):
                try:
                    control_curve = generate_random_polygon_in_hull(avg_hull, 20)
                    control_acn, control_writhe, control_rms_writhe = average_crossing_number(control_curve, args.n_acn_projections)
                    control_acn_values.append(control_acn)
                    control_writhe_values.append(control_writhe)
                    control_rms_writhe_values.append(control_rms_writhe)
                except Exception as e:
                    pass

        control_acn_mean = np.mean(control_acn_values) if control_acn_values else 0.0
        control_acn_std = np.std(control_acn_values) if control_acn_values else 1.0
        control_writhe_mean = np.mean(control_writhe_values) if control_writhe_values else 0.0
        control_writhe_std = np.std(control_writhe_values) if control_writhe_values else 1.0
        control_rms_writhe_mean = np.mean(control_rms_writhe_values) if control_rms_writhe_values else 0.0
        control_rms_writhe_std = np.std(control_rms_writhe_values) if control_rms_writhe_values else 1.0
        print(f"  Control ACN: {control_acn_mean:.2f} ± {control_acn_std:.2f} (N={len(control_acn_values)})")
        print(f"  Control Writhe: {control_writhe_mean:.2f} ± {control_writhe_std:.2f}")
        print(f"  Control RMS Writhe: {control_rms_writhe_mean:.2f} ± {control_rms_writhe_std:.2f}")

        # STEP 2: Compute ACN, Writhe, and RMS Writhe for centroid curves
        print("  Computing ACN/Writhe/RMS Writhe for centroid curves...")
        for walk_id, centroid_curve in tqdm(walk_centroid_data.items(), desc="Centroid ACN/Writhe"):
            try:
                centroid_acn, centroid_writhe, centroid_rms_writhe = average_crossing_number(centroid_curve, args.n_acn_projections)
                walk_acn_data[walk_id] = {'centroid_acn': centroid_acn, 'sampled_acn': []}
                walk_writhe_data[walk_id] = {'centroid_writhe': centroid_writhe, 'sampled_writhe': [],
                                             'centroid_rms_writhe': centroid_rms_writhe, 'sampled_rms_writhe': []}
            except Exception as e:
                print(f"  Error computing centroid ACN/Writhe for Walk-ID {walk_id}: {e}")
                walk_acn_data[walk_id] = {'centroid_acn': np.nan, 'sampled_acn': []}
                walk_writhe_data[walk_id] = {'centroid_writhe': np.nan, 'sampled_writhe': [],
                                             'centroid_rms_writhe': np.nan, 'sampled_rms_writhe': []}

        # Compute ACN/Writhe for sampled curves (this is computationally expensive)
        print(f"  Computing ACN/Writhe for sampled curves (N={args.N} per walk)...")

        # Collect all curves directly from walk_results (no need to resample!)
        all_curves_with_metadata = []
        for walk_id, walk_data_results in walk_results.items():
            for curve, knot_name, determinant in walk_data_results:
                all_curves_with_metadata.append((walk_id, curve))

        # Compute ACN/Writhe in parallel if possible
        print(f"  Total curves to process: {len(all_curves_with_metadata)}")

        if args.parallel and len(all_curves_with_metadata) > 0:
            n_workers = args.workers if args.workers else multiprocessing.cpu_count()
            print(f"  Using {n_workers} parallel workers...")

            # Process in chunks to avoid overwhelming the executor's internal queues
            chunk_size = 5000  # Submit and process 5000 tasks at a time
            total_curves = len(all_curves_with_metadata)

            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                # Process curves in chunks
                with tqdm(total=total_curves, desc="  ACN/Writhe computation") as pbar:
                    for chunk_start in range(0, total_curves, chunk_size):
                        chunk_end = min(chunk_start + chunk_size, total_curves)
                        chunk = all_curves_with_metadata[chunk_start:chunk_end]

                        # Submit this chunk
                        future_to_walk = {}
                        for walk_id, curve in chunk:
                            future = executor.submit(average_crossing_number, curve, args.n_acn_projections)
                            future_to_walk[future] = walk_id

                        # Process results for this chunk as they complete
                        for future in as_completed(future_to_walk):
                            walk_id = future_to_walk[future]
                            try:
                                acn_value, writhe_value, rms_writhe_value = future.result()
                                if walk_id in walk_acn_data:
                                    walk_acn_data[walk_id]['sampled_acn'].append(acn_value)
                                if walk_id in walk_writhe_data:
                                    walk_writhe_data[walk_id]['sampled_writhe'].append(writhe_value)
                                    walk_writhe_data[walk_id]['sampled_rms_writhe'].append(rms_writhe_value)
                            except Exception as e:
                                print(f"    Error computing ACN/Writhe for Walk-ID {walk_id}: {e}")
                            pbar.update(1)
        else:
            # Sequential processing
            for walk_id, curve in tqdm(all_curves_with_metadata, desc="  ACN/Writhe computation"):
                try:
                    acn_value, writhe_value, rms_writhe_value = average_crossing_number(curve, args.n_acn_projections)
                    if walk_id in walk_acn_data:
                        walk_acn_data[walk_id]['sampled_acn'].append(acn_value)
                    if walk_id in walk_writhe_data:
                        walk_writhe_data[walk_id]['sampled_writhe'].append(writhe_value)
                        walk_writhe_data[walk_id]['sampled_rms_writhe'].append(rms_writhe_value)
                except Exception as e:
                    print(f"    Error computing ACN/Writhe for Walk-ID {walk_id}: {e}")

        # STEP 3: Calculate global x-range and y-max for consistent axes
        print("  Calculating global axis ranges for consistent plotting...")

        # Collect all ACN values
        all_acn_values = []
        all_acn_values.extend(control_acn_values)

        for walk_id, data in walk_acn_data.items():
            if not np.isnan(data['centroid_acn']):
                all_acn_values.append(data['centroid_acn'])
            all_acn_values.extend([x for x in data['sampled_acn'] if not np.isnan(x)])

        if len(all_acn_values) > 0:
            # Calculate global x-range
            min_acn = np.min(all_acn_values)
            max_acn = np.max(all_acn_values)
            padding = 0.1 * (max_acn - min_acn)

            # Create tuple for histogram/xlim (needed by matplotlib)
            global_x_range = (min_acn - padding, max_acn + padding)

            # Create linspace for KDE density calculations
            x_linspace = np.linspace(min_acn - padding, max_acn + padding, 200)

            # Calculate global y-max by finding maximum density across all distributions
            max_density = 0.0

            # Check control density
            if len(control_acn_values) > 1:
                kde_control = gaussian_kde(control_acn_values)
                density_control = kde_control(x_linspace)
                max_density = max(max_density, np.max(density_control))

            # Check each walk density
            for walk_id, data in walk_acn_data.items():
                sampled_acn_clean = [x for x in data['sampled_acn'] if not np.isnan(x)]
                if len(sampled_acn_clean) > 1:
                    kde_walk = gaussian_kde(sampled_acn_clean)
                    density_walk = kde_walk(x_linspace)
                    max_density = max(max_density, np.max(density_walk))

            # Add 10% padding to y-max
            global_y_max = max_density * 1.1

            print(f"  Global ACN range: [{min_acn:.2f}, {max_acn:.2f}]")
            print(f"  Global x_range tuple: {global_x_range}")
            print(f"  Control mean: {control_acn_mean:.2f} ± {control_acn_std:.2f}")
            print(f"  Global y-max: {global_y_max:.4f}")
        else:
            print("  Warning: No valid ACN values found for global range computation")

        # STEP 4: Calculate global writhe x-range and y-max
        print("  Calculating global Writhe axis ranges...")

        # Collect all writhe values
        all_writhe_values = []
        all_writhe_values.extend([x for x in control_writhe_values if not np.isnan(x)])

        for walk_id, data in walk_writhe_data.items():
            if not np.isnan(data['centroid_writhe']):
                all_writhe_values.append(data['centroid_writhe'])
            all_writhe_values.extend([x for x in data['sampled_writhe'] if not np.isnan(x)])

        if len(all_writhe_values) > 0:
            # Calculate global writhe x-range (can be negative!)
            min_writhe = np.min(all_writhe_values)
            max_writhe = np.max(all_writhe_values)
            writhe_range = max_writhe - min_writhe
            padding = 0.1 * writhe_range if writhe_range > 0 else 1.0

            # Create tuple for histogram/xlim
            global_writhe_x_range = (min_writhe - padding, max_writhe + padding)

            # Create linspace for KDE density calculations
            writhe_linspace = np.linspace(min_writhe - padding, max_writhe + padding, 200)

            # Calculate global y-max for writhe
            max_writhe_density = 0.0

            # Check control density
            control_writhe_clean = [x for x in control_writhe_values if not np.isnan(x)]
            if len(control_writhe_clean) > 1:
                kde_control_writhe = gaussian_kde(control_writhe_clean)
                density_control_writhe = kde_control_writhe(writhe_linspace)
                max_writhe_density = max(max_writhe_density, np.max(density_control_writhe))

            # Check each walk density
            for walk_id, data in walk_writhe_data.items():
                sampled_writhe_clean = [x for x in data['sampled_writhe'] if not np.isnan(x)]
                if len(sampled_writhe_clean) > 1:
                    kde_walk_writhe = gaussian_kde(sampled_writhe_clean)
                    density_walk_writhe = kde_walk_writhe(writhe_linspace)
                    max_writhe_density = max(max_writhe_density, np.max(density_walk_writhe))

            # Add 10% padding to y-max
            global_writhe_y_max = max_writhe_density * 1.1

            print(f"  Global Writhe range: [{min_writhe:.2f}, {max_writhe:.2f}]")
            print(f"  Global writhe_x_range tuple: {global_writhe_x_range}")
            print(f"  Control Writhe mean: {control_writhe_mean:.2f} ± {control_writhe_std:.2f}")
            print(f"  Global writhe y-max: {global_writhe_y_max:.4f}")
        else:
            print("  Warning: No valid Writhe values found for global range computation")

        # STEP 5: Calculate global RMS writhe x-range and y-max
        print("  Calculating global RMS Writhe axis ranges...")

        # Collect all RMS writhe values
        all_rms_writhe_values = []
        all_rms_writhe_values.extend([x for x in control_rms_writhe_values if not np.isnan(x)])

        for walk_id, data in walk_writhe_data.items():
            if 'centroid_rms_writhe' in data and not np.isnan(data['centroid_rms_writhe']):
                all_rms_writhe_values.append(data['centroid_rms_writhe'])
            if 'sampled_rms_writhe' in data:
                all_rms_writhe_values.extend([x for x in data['sampled_rms_writhe'] if not np.isnan(x)])

        if len(all_rms_writhe_values) > 0:
            # Calculate global RMS writhe x-range (always >= 0)
            min_rms_writhe = max(0, np.min(all_rms_writhe_values))  # RMS is always >= 0
            max_rms_writhe = np.max(all_rms_writhe_values)
            rms_writhe_range = max_rms_writhe - min_rms_writhe
            padding = 0.1 * rms_writhe_range if rms_writhe_range > 0 else 1.0

            # Create tuple for histogram/xlim
            global_rms_writhe_x_range = (max(0, min_rms_writhe - padding), max_rms_writhe + padding)

            # Create linspace for KDE density calculations
            rms_writhe_linspace = np.linspace(max(0, min_rms_writhe - padding), max_rms_writhe + padding, 200)

            # Calculate global y-max for RMS writhe
            max_rms_writhe_density = 0.0

            # Check control density
            control_rms_writhe_clean = [x for x in control_rms_writhe_values if not np.isnan(x)]
            if len(control_rms_writhe_clean) > 1:
                kde_control_rms = gaussian_kde(control_rms_writhe_clean)
                density_control_rms = kde_control_rms(rms_writhe_linspace)
                max_rms_writhe_density = max(max_rms_writhe_density, np.max(density_control_rms))

            # Check each walk density
            for walk_id, data in walk_writhe_data.items():
                if 'sampled_rms_writhe' in data:
                    sampled_rms_clean = [x for x in data['sampled_rms_writhe'] if not np.isnan(x)]
                    if len(sampled_rms_clean) > 1:
                        kde_walk_rms = gaussian_kde(sampled_rms_clean)
                        density_walk_rms = kde_walk_rms(rms_writhe_linspace)
                        max_rms_writhe_density = max(max_rms_writhe_density, np.max(density_walk_rms))

            # Add 10% padding to y-max
            global_rms_writhe_y_max = max_rms_writhe_density * 1.1

            print(f"  Global RMS Writhe range: [{min_rms_writhe:.2f}, {max_rms_writhe:.2f}]")
            print(f"  Global rms_writhe_x_range tuple: {global_rms_writhe_x_range}")
            print(f"  Control RMS Writhe mean: {control_rms_writhe_mean:.2f} ± {control_rms_writhe_std:.2f}")
            print(f"  Global RMS writhe y-max: {global_rms_writhe_y_max:.4f}")
        else:
            print("  Warning: No valid RMS Writhe values found for global range computation")

    # Radius of Gyration Computation
    if args.rog:
        print(f"\nComputing Radius of Gyration...")

        # STEP 1: Compute SHARED control RoG distribution FIRST
        print("  Computing SHARED control RoG distribution for all walks...")
        n_control_samples = 200  # Use more samples for robust control

        if avg_hull is not None:
            for i in tqdm(range(n_control_samples), desc="  Control RoG"):
                try:
                    control_curve = generate_random_polygon_in_hull(avg_hull, 20)
                    control_rog = radius_of_gyration(control_curve)
                    control_rog_values.append(control_rog)
                except Exception as e:
                    pass

        control_rog_mean = np.mean(control_rog_values) if control_rog_values else 0.0
        control_rog_std = np.std(control_rog_values) if control_rog_values else 1.0
        print(f"  Control RoG: {control_rog_mean:.2f} +/- {control_rog_std:.2f} (N={len(control_rog_values)})")

        # STEP 2: Compute RoG for centroid curves
        print("  Computing RoG for centroid curves...")
        for walk_id, centroid_curve in tqdm(walk_centroid_data.items(), desc="Centroid RoG"):
            try:
                centroid_rog = radius_of_gyration(centroid_curve)
                walk_rog_data[walk_id] = {'centroid_rog': centroid_rog, 'sampled_rog': []}
            except Exception as e:
                print(f"  Error computing centroid RoG for Walk-ID {walk_id}: {e}")
                walk_rog_data[walk_id] = {'centroid_rog': np.nan, 'sampled_rog': []}

        # STEP 3: Compute RoG for sampled curves
        print(f"  Computing RoG for sampled curves...")
        for walk_id, walk_data_results in walk_results.items():
            for curve, knot_name, determinant in walk_data_results:
                try:
                    rog_value = radius_of_gyration(curve)
                    if walk_id in walk_rog_data:
                        walk_rog_data[walk_id]['sampled_rog'].append(rog_value)
                except Exception as e:
                    pass

        # STEP 4: Calculate global RoG x-range and y-max
        print("  Calculating global RoG axis ranges...")

        # Collect all RoG values
        all_rog_values = []
        all_rog_values.extend([x for x in control_rog_values if not np.isnan(x)])

        for walk_id, data in walk_rog_data.items():
            if 'centroid_rog' in data and not np.isnan(data['centroid_rog']):
                all_rog_values.append(data['centroid_rog'])
            if 'sampled_rog' in data:
                all_rog_values.extend([x for x in data['sampled_rog'] if not np.isnan(x)])

        if len(all_rog_values) > 0:
            # Calculate global RoG x-range (always >= 0)
            min_rog = max(0, np.min(all_rog_values))
            max_rog = np.max(all_rog_values)
            rog_range = max_rog - min_rog
            padding = 0.1 * rog_range if rog_range > 0 else 1.0

            # Create tuple for histogram/xlim
            global_rog_x_range = (max(0, min_rog - padding), max_rog + padding)

            # Create linspace for KDE density calculations
            rog_linspace = np.linspace(max(0, min_rog - padding), max_rog + padding, 200)

            # Calculate global y-max for RoG
            max_rog_density = 0.0

            # Check control density
            control_rog_clean = [x for x in control_rog_values if not np.isnan(x)]
            if len(control_rog_clean) > 1:
                kde_control_rog = gaussian_kde(control_rog_clean)
                density_control_rog = kde_control_rog(rog_linspace)
                max_rog_density = max(max_rog_density, np.max(density_control_rog))

            # Check each walk density
            for walk_id, data in walk_rog_data.items():
                if 'sampled_rog' in data:
                    sampled_rog_clean = [x for x in data['sampled_rog'] if not np.isnan(x)]
                    if len(sampled_rog_clean) > 1:
                        kde_walk_rog = gaussian_kde(sampled_rog_clean)
                        density_walk_rog = kde_walk_rog(rog_linspace)
                        max_rog_density = max(max_rog_density, np.max(density_walk_rog))

            # Add 10% padding to y-max
            global_rog_y_max = max_rog_density * 1.1

            print(f"  Global RoG range: [{min_rog:.2f}, {max_rog:.2f}]")
            print(f"  Global rog_x_range tuple: {global_rog_x_range}")
            print(f"  Control RoG mean: {control_rog_mean:.2f} +/- {control_rog_std:.2f}")
            print(f"  Global RoG y-max: {global_rog_y_max:.4f}")
        else:
            print("  Warning: No valid RoG values found for global range computation")

    # V2 Invariant Computation
    control_v2_values = []  # Initialize empty list
    global_v2_x_range = None
    global_v2_y_max = None

    if args.vassiliev:
        print(f"\nComputing Vassiliev V2 invariant with {args.num_vassiliev_proj} projections...")

        # STEP 1: Compute SHARED control V2 distribution FIRST
        print("  Computing SHARED control V2 distribution for all walks...")
        n_control_samples = 200  # Use more samples for robust control

        if avg_hull is not None:
            for i in tqdm(range(n_control_samples), desc="  Control V2"):
                try:
                    control_curve = generate_random_polygon_in_hull(avg_hull, 20)
                    control_v2, _ = compute_v2_invariant(
                        control_curve,
                        closed=True,
                        n_resample=20,
                        n_projections=args.num_vassiliev_proj,
                        show_progress=False
                    )
                    control_v2_values.append(control_v2)
                except Exception as e:
                    pass

        control_v2_mean = np.mean(control_v2_values) if control_v2_values else 0.0
        control_v2_std = np.std(control_v2_values) if control_v2_values else 1.0
        print(f"  Control V2: {control_v2_mean:.6f} ± {control_v2_std:.6f} (N={len(control_v2_values)})")

        # STEP 2: Compute V2 for sampled curves
        print(f"  Computing V2 for sampled curves (N={args.N} per walk)...")

        # Collect all curves directly from walk_results (no need to resample!)
        all_curves_with_metadata = []
        for walk_id, walk_data_results in walk_results.items():
            for curve, knot_name, determinant in walk_data_results:
                all_curves_with_metadata.append((walk_id, curve))

        print(f"  Total curves to process: {len(all_curves_with_metadata)}")

        # Compute V2 in parallel if possible
        if args.parallel and len(all_curves_with_metadata) > 0:
            n_workers = args.workers if args.workers else multiprocessing.cpu_count()
            print(f"  Using {n_workers} parallel workers...")

            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                # Submit all tasks and map future -> walk_id
                future_to_walk = {}
                for walk_id, curve in all_curves_with_metadata:
                    future = executor.submit(
                        compute_v2_invariant,
                        curve,
                        closed=True,
                        n_resample=20,
                        n_projections=args.num_vassiliev_proj,
                        show_progress=False
                    )
                    future_to_walk[future] = walk_id

                # Process results as they complete (prevents queue buildup and deadlock)
                for future in tqdm(as_completed(future_to_walk), total=len(future_to_walk), desc="  V2 computation"):
                    walk_id = future_to_walk[future]
                    try:
                        v2_value, _ = future.result()
                        if walk_id in walk_v2_data:
                            walk_v2_data[walk_id]['sampled_v2'].append(v2_value)
                    except Exception as e:
                        print(f"    Error computing V2 for Walk-ID {walk_id}: {e}")
        else:
            # Sequential processing
            for walk_id, curve in tqdm(all_curves_with_metadata, desc="  V2 computation"):
                try:
                    v2_value, _ = compute_v2_invariant(
                        curve,
                        closed=True,
                        n_resample=20,
                        n_projections=args.num_vassiliev_proj,
                        show_progress=False
                    )
                    if walk_id in walk_v2_data:
                        walk_v2_data[walk_id]['sampled_v2'].append(v2_value)
                except Exception as e:
                    print(f"    Error computing V2 for Walk-ID {walk_id}: {e}")

        # STEP 3: Calculate global x-range and y-max for consistent axes
        print("  Calculating global V2 axis ranges for consistent plotting...")

        # Collect all V2 values
        all_v2_values = []
        all_v2_values.extend(control_v2_values)

        for walk_id, data in walk_v2_data.items():
            if not np.isnan(data['centroid_v2']):
                all_v2_values.append(data['centroid_v2'])
            all_v2_values.extend([x for x in data['sampled_v2'] if not np.isnan(x)])

        if len(all_v2_values) > 0:
            # Calculate global x-range
            min_v2 = np.min(all_v2_values)
            max_v2 = np.max(all_v2_values)
            padding = 0.1 * (max_v2 - min_v2)

            # Create tuple for histogram/xlim
            global_v2_x_range = (min_v2 - padding, max_v2 + padding)

            # Create linspace for KDE density calculations
            v2_linspace = np.linspace(min_v2 - padding, max_v2 + padding, 200)

            # Calculate global y-max by finding maximum density across all distributions
            max_density = 0.0

            # Check control density
            if len(control_v2_values) > 1:
                kde_control = gaussian_kde(control_v2_values)
                density_control = kde_control(v2_linspace)
                max_density = max(max_density, np.max(density_control))

            # Check each walk density
            for walk_id, data in walk_v2_data.items():
                sampled_v2_clean = [x for x in data['sampled_v2'] if not np.isnan(x)]
                if len(sampled_v2_clean) > 1:
                    kde_walk = gaussian_kde(sampled_v2_clean)
                    density_walk = kde_walk(v2_linspace)
                    max_density = max(max_density, np.max(density_walk))

            # Add 10% padding to y-max
            global_v2_y_max = max_density * 1.1

            print(f"  Global V2 range: [{min_v2:.6f}, {max_v2:.6f}]")
            print(f"  Global V2 x_range tuple: {global_v2_x_range}")
            print(f"  Control mean: {control_v2_mean:.6f} ± {control_v2_std:.6f}")
            print(f"  Global V2 y-max: {global_v2_y_max:.4f}")
        else:
            print("  Warning: No valid V2 values found for global range computation")

    # xyCN (xy-plane Crossing Number) Computation
    control_xycn_values = []  # Initialize empty list
    global_xycn_x_range = None
    global_xycn_y_max = None

    if args.xyCN:
        print(f"\nComputing xy-plane Crossing Number...")

        # STEP 1: Compute SHARED control xyCN distribution FIRST
        print("  Computing SHARED control xyCN distribution for all walks...")
        n_control_samples = 200  # Use more samples for robust control

        if avg_hull is not None:
            for i in tqdm(range(n_control_samples), desc="  Control xyCN"):
                try:
                    control_curve = generate_random_polygon_in_hull(avg_hull, 20)
                    control_xycn = xy_crossing_number(control_curve)
                    if not np.isnan(control_xycn):
                        control_xycn_values.append(control_xycn)
                except Exception as e:
                    pass

        control_xycn_mean = np.mean(control_xycn_values) if control_xycn_values else 0.0
        control_xycn_std = np.std(control_xycn_values) if control_xycn_values else 1.0
        print(f"  Control xyCN: {control_xycn_mean:.2f} ± {control_xycn_std:.2f} (N={len(control_xycn_values)})")

        # STEP 2: Compute xyCN for centroid curves
        print("  Computing xyCN for centroid curves...")
        for walk_id, centroid_curve in tqdm(walk_centroid_data.items(), desc="Centroid xyCN"):
            try:
                centroid_xycn = xy_crossing_number(centroid_curve)
                walk_xycn_data[walk_id] = {'centroid_xycn': centroid_xycn, 'sampled_xycn': []}
            except Exception as e:
                print(f"  Error computing centroid xyCN for Walk-ID {walk_id}: {e}")
                walk_xycn_data[walk_id] = {'centroid_xycn': np.nan, 'sampled_xycn': []}

        # STEP 3: Compute xyCN for sampled curves (reuse stored curves)
        print(f"  Computing xyCN for sampled curves (N={args.N} per walk)...")

        # Collect all curves directly from walk_results (no need to resample!)
        all_curves_with_metadata = []
        for walk_id, walk_data_results in walk_results.items():
            for curve, knot_name, determinant in walk_data_results:
                all_curves_with_metadata.append((walk_id, curve))

        print(f"  Total curves to process: {len(all_curves_with_metadata)}")

        # xyCN is fast (single projection), so we can compute sequentially
        for walk_id, curve in tqdm(all_curves_with_metadata, desc="  xyCN computation"):
            try:
                xycn_value = xy_crossing_number(curve)
                if walk_id in walk_xycn_data:
                    walk_xycn_data[walk_id]['sampled_xycn'].append(xycn_value)
            except Exception as e:
                print(f"    Error computing xyCN for Walk-ID {walk_id}: {e}")

        # STEP 4: Calculate global x-range and y-max for consistent axes
        print("  Computing global x-range and y-max for xyCN plots...")

        # Gather all xyCN values
        all_xycn = control_xycn_values.copy()
        for walk_id, data in walk_xycn_data.items():
            all_xycn.extend([x for x in data['sampled_xycn'] if not np.isnan(x)])
            if not np.isnan(data['centroid_xycn']):
                all_xycn.append(data['centroid_xycn'])

        if all_xycn:
            min_xycn = np.min(all_xycn)
            max_xycn = np.max(all_xycn)

            # Add 10% padding for better visualization
            padding = (max_xycn - min_xycn) * 0.1
            global_xycn_x_range = (min_xycn - padding, max_xycn + padding)

            # Create linspace for KDE density calculations
            xycn_linspace = np.linspace(min_xycn - padding, max_xycn + padding, 200)

            # Calculate global y-max by finding maximum density across all distributions
            max_density = 0.0

            # Check control density
            if len(control_xycn_values) > 1:
                kde_control = gaussian_kde(control_xycn_values)
                density_control = kde_control(xycn_linspace)
                max_density = max(max_density, np.max(density_control))

            # Check each walk density
            for walk_id, data in walk_xycn_data.items():
                sampled_xycn_clean = [x for x in data['sampled_xycn'] if not np.isnan(x)]
                if len(sampled_xycn_clean) > 1:
                    kde_walk = gaussian_kde(sampled_xycn_clean)
                    density_walk = kde_walk(xycn_linspace)
                    max_density = max(max_density, np.max(density_walk))

            # Add 10% padding to y-max
            global_xycn_y_max = max_density * 1.1

            print(f"  Global xyCN range: [{min_xycn:.2f}, {max_xycn:.2f}]")
            print(f"  Global xyCN x_range tuple: {global_xycn_x_range}")
            print(f"  Control mean: {control_xycn_mean:.2f} ± {control_xycn_std:.2f}")
            print(f"  Global xyCN y-max: {global_xycn_y_max:.4f}")
        else:
            print("  Warning: No valid xyCN values found for global range computation")

    # xzCN (xz-plane Crossing Number) Computation
    control_xzcn_values = []  # Initialize empty list
    global_xzcn_x_range = None
    global_xzcn_y_max = None

    if args.xzCN:
        print(f"\nComputing xz-plane Crossing Number...")

        # STEP 1: Compute SHARED control xzCN distribution FIRST
        print("  Computing SHARED control xzCN distribution for all walks...")
        n_control_samples = 200  # Use more samples for robust control

        if avg_hull is not None:
            for i in tqdm(range(n_control_samples), desc="  Control xzCN"):
                try:
                    control_curve = generate_random_polygon_in_hull(avg_hull, 20)
                    control_xzcn = xz_crossing_number(control_curve)
                    if not np.isnan(control_xzcn):
                        control_xzcn_values.append(control_xzcn)
                except Exception as e:
                    pass

        control_xzcn_mean = np.mean(control_xzcn_values) if control_xzcn_values else 0.0
        control_xzcn_std = np.std(control_xzcn_values) if control_xzcn_values else 1.0
        print(f"  Control xzCN: {control_xzcn_mean:.2f} ± {control_xzcn_std:.2f} (N={len(control_xzcn_values)})")

        # STEP 2: Compute xzCN for centroid curves
        print("  Computing xzCN for centroid curves...")
        for walk_id, centroid_curve in tqdm(walk_centroid_data.items(), desc="Centroid xzCN"):
            try:
                centroid_xzcn = xz_crossing_number(centroid_curve)
                walk_xzcn_data[walk_id] = {'centroid_xzcn': centroid_xzcn, 'sampled_xzcn': []}
            except Exception as e:
                print(f"  Error computing centroid xzCN for Walk-ID {walk_id}: {e}")
                walk_xzcn_data[walk_id] = {'centroid_xzcn': np.nan, 'sampled_xzcn': []}

        # STEP 3: Compute xzCN for sampled curves (reuse stored curves)
        print(f"  Computing xzCN for sampled curves (N={args.N} per walk)...")

        # Collect all curves directly from walk_results (no need to resample!)
        all_curves_with_metadata = []
        for walk_id, walk_data_results in walk_results.items():
            for curve, knot_name, determinant in walk_data_results:
                all_curves_with_metadata.append((walk_id, curve))

        print(f"  Total curves to process: {len(all_curves_with_metadata)}")

        # xzCN is fast (single projection), so we can compute sequentially
        for walk_id, curve in tqdm(all_curves_with_metadata, desc="  xzCN computation"):
            try:
                xzcn_value = xz_crossing_number(curve)
                if walk_id in walk_xzcn_data:
                    walk_xzcn_data[walk_id]['sampled_xzcn'].append(xzcn_value)
            except Exception as e:
                print(f"    Error computing xzCN for Walk-ID {walk_id}: {e}")

        # STEP 4: Calculate global x-range and y-max for consistent axes
        print("  Computing global x-range and y-max for xzCN plots...")

        # Gather all xzCN values
        all_xzcn = control_xzcn_values.copy()
        for walk_id, data in walk_xzcn_data.items():
            all_xzcn.extend([x for x in data['sampled_xzcn'] if not np.isnan(x)])
            if not np.isnan(data['centroid_xzcn']):
                all_xzcn.append(data['centroid_xzcn'])

        if all_xzcn:
            min_xzcn = np.min(all_xzcn)
            max_xzcn = np.max(all_xzcn)

            # Add 10% padding for better visualization
            padding = (max_xzcn - min_xzcn) * 0.1
            global_xzcn_x_range = (min_xzcn - padding, max_xzcn + padding)

            # Create linspace for KDE density calculations
            xzcn_linspace = np.linspace(min_xzcn - padding, max_xzcn + padding, 200)

            # Calculate global y-max by finding maximum density across all distributions
            max_density = 0.0

            # Check control density
            if len(control_xzcn_values) > 1:
                kde_control = gaussian_kde(control_xzcn_values)
                density_control = kde_control(xzcn_linspace)
                max_density = max(max_density, np.max(density_control))

            # Check each walk density
            for walk_id, data in walk_xzcn_data.items():
                sampled_xzcn_clean = [x for x in data['sampled_xzcn'] if not np.isnan(x)]
                if len(sampled_xzcn_clean) > 1:
                    kde_walk = gaussian_kde(sampled_xzcn_clean)
                    density_walk = kde_walk(xzcn_linspace)
                    max_density = max(max_density, np.max(density_walk))

            # Add 10% padding to y-max
            global_xzcn_y_max = max_density * 1.1

            print(f"  Global xzCN range: [{min_xzcn:.2f}, {max_xzcn:.2f}]")
            print(f"  Global xzCN x_range tuple: {global_xzcn_x_range}")
            print(f"  Control mean: {control_xzcn_mean:.2f} ± {control_xzcn_std:.2f}")
            print(f"  Global xzCN y-max: {global_xzcn_y_max:.4f}")
        else:
            print("  Warning: No valid xzCN values found for global range computation")

    # yzCN (yz-plane Crossing Number) Computation
    control_yzcn_values = []  # Initialize empty list
    global_yzcn_x_range = None
    global_yzcn_y_max = None

    if args.yzCN:
        print(f"\nComputing yz-plane Crossing Number...")

        # STEP 1: Compute SHARED control yzCN distribution FIRST
        print("  Computing SHARED control yzCN distribution for all walks...")
        n_control_samples = 200  # Use more samples for robust control

        if avg_hull is not None:
            for i in tqdm(range(n_control_samples), desc="  Control yzCN"):
                try:
                    control_curve = generate_random_polygon_in_hull(avg_hull, 20)
                    control_yzcn = yz_crossing_number(control_curve)
                    if not np.isnan(control_yzcn):
                        control_yzcn_values.append(control_yzcn)
                except Exception as e:
                    pass

        control_yzcn_mean = np.mean(control_yzcn_values) if control_yzcn_values else 0.0
        control_yzcn_std = np.std(control_yzcn_values) if control_yzcn_values else 1.0
        print(f"  Control yzCN: {control_yzcn_mean:.2f} ± {control_yzcn_std:.2f} (N={len(control_yzcn_values)})")

        # STEP 2: Compute yzCN for centroid curves
        print("  Computing yzCN for centroid curves...")
        for walk_id, centroid_curve in tqdm(walk_centroid_data.items(), desc="Centroid yzCN"):
            try:
                centroid_yzcn = yz_crossing_number(centroid_curve)
                walk_yzcn_data[walk_id] = {'centroid_yzcn': centroid_yzcn, 'sampled_yzcn': []}
            except Exception as e:
                print(f"  Error computing centroid yzCN for Walk-ID {walk_id}: {e}")
                walk_yzcn_data[walk_id] = {'centroid_yzcn': np.nan, 'sampled_yzcn': []}

        # STEP 3: Compute yzCN for sampled curves (reuse stored curves)
        print(f"  Computing yzCN for sampled curves (N={args.N} per walk)...")

        # Collect all curves directly from walk_results (no need to resample!)
        all_curves_with_metadata = []
        for walk_id, walk_data_results in walk_results.items():
            for curve, knot_name, determinant in walk_data_results:
                all_curves_with_metadata.append((walk_id, curve))

        print(f"  Total curves to process: {len(all_curves_with_metadata)}")

        # yzCN is fast (single projection), so we can compute sequentially
        for walk_id, curve in tqdm(all_curves_with_metadata, desc="  yzCN computation"):
            try:
                yzcn_value = yz_crossing_number(curve)
                if walk_id in walk_yzcn_data:
                    walk_yzcn_data[walk_id]['sampled_yzcn'].append(yzcn_value)
            except Exception as e:
                print(f"    Error computing yzCN for Walk-ID {walk_id}: {e}")

        # STEP 4: Calculate global x-range and y-max for consistent axes
        print("  Computing global x-range and y-max for yzCN plots...")

        # Gather all yzCN values
        all_yzcn = control_yzcn_values.copy()
        for walk_id, data in walk_yzcn_data.items():
            all_yzcn.extend([x for x in data['sampled_yzcn'] if not np.isnan(x)])
            if not np.isnan(data['centroid_yzcn']):
                all_yzcn.append(data['centroid_yzcn'])

        if all_yzcn:
            min_yzcn = np.min(all_yzcn)
            max_yzcn = np.max(all_yzcn)

            # Add 10% padding for better visualization
            padding = (max_yzcn - min_yzcn) * 0.1
            global_yzcn_x_range = (min_yzcn - padding, max_yzcn + padding)

            # Create linspace for KDE density calculations
            yzcn_linspace = np.linspace(min_yzcn - padding, max_yzcn + padding, 200)

            # Calculate global y-max by finding maximum density across all distributions
            max_density = 0.0

            # Check control density
            if len(control_yzcn_values) > 1:
                kde_control = gaussian_kde(control_yzcn_values)
                density_control = kde_control(yzcn_linspace)
                max_density = max(max_density, np.max(density_control))

            # Check each walk density
            for walk_id, data in walk_yzcn_data.items():
                sampled_yzcn_clean = [x for x in data['sampled_yzcn'] if not np.isnan(x)]
                if len(sampled_yzcn_clean) > 1:
                    kde_walk = gaussian_kde(sampled_yzcn_clean)
                    density_walk = kde_walk(yzcn_linspace)
                    max_density = max(max_density, np.max(density_walk))

            # Add 10% padding to y-max
            global_yzcn_y_max = max_density * 1.1

            print(f"  Global yzCN range: [{min_yzcn:.2f}, {max_yzcn:.2f}]")
            print(f"  Global yzCN x_range tuple: {global_yzcn_x_range}")
            print(f"  Control mean: {control_yzcn_mean:.2f} ± {control_yzcn_std:.2f}")
            print(f"  Global yzCN y-max: {global_yzcn_y_max:.4f}")
        else:
            print("  Warning: No valid yzCN values found for global range computation")

    # Compute unified global range across all enabled planar CNs for combined plotting
    unified_planar_cn_x_range = None
    unified_planar_cn_y_max = None

    if args.xyCN or args.xzCN or args.yzCN:
        print("\nComputing unified global range for planar CN plots...")

        # Collect all CN values from all enabled projections
        all_planar_cn_values = []
        all_planar_control_values = []

        if args.xyCN:
            all_planar_cn_values.extend(control_xycn_values)
            all_planar_control_values.extend(control_xycn_values)
            for walk_id, data in walk_xycn_data.items():
                all_planar_cn_values.extend([x for x in data['sampled_xycn'] if not np.isnan(x)])
                if not np.isnan(data['centroid_xycn']):
                    all_planar_cn_values.append(data['centroid_xycn'])

        if args.xzCN:
            all_planar_cn_values.extend(control_xzcn_values)
            all_planar_control_values.extend(control_xzcn_values)
            for walk_id, data in walk_xzcn_data.items():
                all_planar_cn_values.extend([x for x in data['sampled_xzcn'] if not np.isnan(x)])
                if not np.isnan(data['centroid_xzcn']):
                    all_planar_cn_values.append(data['centroid_xzcn'])

        if args.yzCN:
            all_planar_cn_values.extend(control_yzcn_values)
            all_planar_control_values.extend(control_yzcn_values)
            for walk_id, data in walk_yzcn_data.items():
                all_planar_cn_values.extend([x for x in data['sampled_yzcn'] if not np.isnan(x)])
                if not np.isnan(data['centroid_yzcn']):
                    all_planar_cn_values.append(data['centroid_yzcn'])

        if all_planar_cn_values:
            min_cn = np.min(all_planar_cn_values)
            max_cn = np.max(all_planar_cn_values)

            # Add 10% padding for better visualization
            padding = (max_cn - min_cn) * 0.1
            unified_planar_cn_x_range = (min_cn - padding, max_cn + padding)

            # Create linspace for KDE density calculations
            cn_linspace = np.linspace(min_cn - padding, max_cn + padding, 200)

            # Calculate global y-max by finding maximum density across all enabled projections
            max_density = 0.0

            # Check control densities
            if args.xyCN and len(control_xycn_values) > 1:
                kde = gaussian_kde(control_xycn_values)
                density = kde(cn_linspace)
                max_density = max(max_density, np.max(density))

            if args.xzCN and len(control_xzcn_values) > 1:
                kde = gaussian_kde(control_xzcn_values)
                density = kde(cn_linspace)
                max_density = max(max_density, np.max(density))

            if args.yzCN and len(control_yzcn_values) > 1:
                kde = gaussian_kde(control_yzcn_values)
                density = kde(cn_linspace)
                max_density = max(max_density, np.max(density))

            # Check walk densities
            if args.xyCN:
                for walk_id, data in walk_xycn_data.items():
                    sampled_clean = [x for x in data['sampled_xycn'] if not np.isnan(x)]
                    if len(sampled_clean) > 1:
                        kde = gaussian_kde(sampled_clean)
                        density = kde(cn_linspace)
                        max_density = max(max_density, np.max(density))

            if args.xzCN:
                for walk_id, data in walk_xzcn_data.items():
                    sampled_clean = [x for x in data['sampled_xzcn'] if not np.isnan(x)]
                    if len(sampled_clean) > 1:
                        kde = gaussian_kde(sampled_clean)
                        density = kde(cn_linspace)
                        max_density = max(max_density, np.max(density))

            if args.yzCN:
                for walk_id, data in walk_yzcn_data.items():
                    sampled_clean = [x for x in data['sampled_yzcn'] if not np.isnan(x)]
                    if len(sampled_clean) > 1:
                        kde = gaussian_kde(sampled_clean)
                        density = kde(cn_linspace)
                        max_density = max(max_density, np.max(density))

            # Add 10% padding to y-max
            unified_planar_cn_y_max = max_density * 1.1

            print(f"  Unified planar CN range: [{min_cn:.2f}, {max_cn:.2f}]")
            print(f"  Unified x_range: {unified_planar_cn_x_range}")
            print(f"  Unified y-max: {unified_planar_cn_y_max:.4f}")
        else:
            print("  Warning: No valid planar CN values found for unified range computation")

    # Kendall Shape Space Analysis
    if args.kendall:
        print(f"\nPerforming Kendall shape space analysis...")

        for walk_id in tqdm(walk_results.keys(), desc="Kendall analysis"):
            try:
                walk_data_results = walk_results[walk_id]

                # Collect all sampled curves directly from walk_results (no need to resample!)
                sampled_curves = [curve for curve, knot_name, determinant in walk_data_results]

                if len(sampled_curves) > 0:
                    # Perform Kendall averaging
                    averager = KendallIsogonAverager(dim=3)
                    average_isogon, converged, iterations = averager.compute_average(
                        sampled_curves,
                        max_iter=args.kendall_max_iter,
                        tol=args.kendall_tol,
                        return_scale=True
                    )

                    # Compute Procrustes distances
                    procrustes_distances = []
                    for curve in sampled_curves:
                        try:
                            dist = averager.procrustes_distance(curve, average_isogon)
                            procrustes_distances.append(dist)
                        except:
                            pass

                    mean_procrustes = np.mean(procrustes_distances) if procrustes_distances else 0.0
                    std_procrustes = np.std(procrustes_distances) if procrustes_distances else 0.0

                    # Analyze knot type of average isogon
                    knot_type = 'Unknown'
                    determinant = 0.0
                    try:
                        knot_name, det = calculate_knot_properties(average_isogon)
                        knot_type = knot_name
                        determinant = det
                    except:
                        pass

                    walk_kendall_data[walk_id] = {
                        'average_isogon': average_isogon,
                        'converged': converged,
                        'iterations': iterations,
                        'mean_procrustes_dist': mean_procrustes,
                        'std_procrustes_dist': std_procrustes,
                        'knot_type': knot_type,
                        'determinant': determinant
                    }
            except Exception as e:
                print(f"  Error in Kendall analysis for Walk-ID {walk_id}: {e}")

    # Calculate statistics for each Walk-ID against each control
    print("\nCalculating statistics...")
    results_data = []

    for walk_id, walk_data in walk_results.items():
        # Extract knot properties (handles 3-tuple format with curves)
        walk_knot_props = extract_knot_properties(walk_data)

        # Count knot types
        walk_classified = [classify_name(name) for name, _ in walk_knot_props]
        walk_counts = Counter(walk_classified)

        row_data = {
            'Walk_ID': walk_id,
            'N_samples': len(walk_data),
            'Unknot_fraction': walk_counts.get('0_1', 0) / len(walk_data),
            'Trefoil_fraction': walk_counts.get('3_1', 0) / len(walk_data),
            'Complex_fraction': walk_counts.get('Complex', 0) / len(walk_data),
        }

        # Note: Distance TP1→TP21 computation skipped (dataset-specific, not used in analysis)

        # Add Moran's I
        if walk_id in morans_i_dict:
            row_data['Morans_I'] = morans_i_dict[walk_id]['I_value']
            row_data['Morans_P_value'] = morans_i_dict[walk_id]['P_value']
        else:
            row_data['Morans_I'] = np.nan
            row_data['Morans_P_value'] = np.nan

        # Add centroid knot type
        if walk_id in centroid_knot_types:
            knot_name, determinant = centroid_knot_types[walk_id]
            row_data['Centroid_knot_type'] = knot_name
            row_data['Centroid_knot_det'] = determinant
        else:
            row_data['Centroid_knot_type'] = ''
            row_data['Centroid_knot_det'] = np.nan

        # Add ACN data
        if walk_id in walk_acn_data:
            acn_data = walk_acn_data[walk_id]
            row_data['ACN_centroid'] = acn_data['centroid_acn']

            sampled_acn_clean = [x for x in acn_data['sampled_acn'] if not np.isnan(x)]
            if len(sampled_acn_clean) > 0:
                row_data['ACN_mean'] = np.mean(sampled_acn_clean)
                row_data['ACN_std'] = np.std(sampled_acn_clean)
                row_data['ACN_median'] = np.median(sampled_acn_clean)
                row_data['ACN_min'] = np.min(sampled_acn_clean)
                row_data['ACN_max'] = np.max(sampled_acn_clean)

                # Z-score relative to control
                if args.acn and control_acn_std > 0:
                    z_score = (row_data['ACN_mean'] - control_acn_mean) / control_acn_std
                    row_data['ACN_zscore_vs_control'] = z_score
                else:
                    row_data['ACN_zscore_vs_control'] = np.nan
            else:
                row_data['ACN_mean'] = np.nan
                row_data['ACN_std'] = np.nan
                row_data['ACN_median'] = np.nan
                row_data['ACN_min'] = np.nan
                row_data['ACN_max'] = np.nan
                row_data['ACN_zscore_vs_control'] = np.nan
        else:
            row_data['ACN_centroid'] = np.nan
            row_data['ACN_mean'] = np.nan
            row_data['ACN_std'] = np.nan
            row_data['ACN_median'] = np.nan
            row_data['ACN_min'] = np.nan
            row_data['ACN_max'] = np.nan
            row_data['ACN_zscore_vs_control'] = np.nan

        # Add Writhe and RMS Writhe data
        if walk_id in walk_writhe_data:
            writhe_data = walk_writhe_data[walk_id]
            row_data['Writhe_centroid'] = writhe_data.get('centroid_writhe', np.nan)
            row_data['RMS_Writhe_centroid'] = writhe_data.get('centroid_rms_writhe', np.nan)

            sampled_writhe_clean = [x for x in writhe_data.get('sampled_writhe', []) if not np.isnan(x)]
            sampled_rms_writhe_clean = [x for x in writhe_data.get('sampled_rms_writhe', []) if not np.isnan(x)]

            if len(sampled_writhe_clean) > 0:
                row_data['Writhe_mean'] = np.mean(sampled_writhe_clean)
                row_data['Writhe_std'] = np.std(sampled_writhe_clean)
                row_data['Writhe_median'] = np.median(sampled_writhe_clean)
                row_data['Writhe_min'] = np.min(sampled_writhe_clean)
                row_data['Writhe_max'] = np.max(sampled_writhe_clean)

                # Z-score relative to control
                if args.acn and control_writhe_std > 0:
                    z_score = (row_data['Writhe_mean'] - control_writhe_mean) / control_writhe_std
                    row_data['Writhe_zscore_vs_control'] = z_score
                else:
                    row_data['Writhe_zscore_vs_control'] = np.nan
            else:
                row_data['Writhe_mean'] = np.nan
                row_data['Writhe_std'] = np.nan
                row_data['Writhe_median'] = np.nan
                row_data['Writhe_min'] = np.nan
                row_data['Writhe_max'] = np.nan
                row_data['Writhe_zscore_vs_control'] = np.nan

            if len(sampled_rms_writhe_clean) > 0:
                row_data['RMS_Writhe_mean'] = np.mean(sampled_rms_writhe_clean)
                row_data['RMS_Writhe_std'] = np.std(sampled_rms_writhe_clean)
                row_data['RMS_Writhe_median'] = np.median(sampled_rms_writhe_clean)
                row_data['RMS_Writhe_min'] = np.min(sampled_rms_writhe_clean)
                row_data['RMS_Writhe_max'] = np.max(sampled_rms_writhe_clean)

                # Z-score relative to control
                if args.acn and control_rms_writhe_std > 0:
                    z_score = (row_data['RMS_Writhe_mean'] - control_rms_writhe_mean) / control_rms_writhe_std
                    row_data['RMS_Writhe_zscore_vs_control'] = z_score
                else:
                    row_data['RMS_Writhe_zscore_vs_control'] = np.nan
            else:
                row_data['RMS_Writhe_mean'] = np.nan
                row_data['RMS_Writhe_std'] = np.nan
                row_data['RMS_Writhe_median'] = np.nan
                row_data['RMS_Writhe_min'] = np.nan
                row_data['RMS_Writhe_max'] = np.nan
                row_data['RMS_Writhe_zscore_vs_control'] = np.nan
        else:
            row_data['Writhe_centroid'] = np.nan
            row_data['Writhe_mean'] = np.nan
            row_data['Writhe_std'] = np.nan
            row_data['Writhe_median'] = np.nan
            row_data['Writhe_min'] = np.nan
            row_data['Writhe_max'] = np.nan
            row_data['Writhe_zscore_vs_control'] = np.nan
            row_data['RMS_Writhe_centroid'] = np.nan
            row_data['RMS_Writhe_mean'] = np.nan
            row_data['RMS_Writhe_std'] = np.nan
            row_data['RMS_Writhe_median'] = np.nan
            row_data['RMS_Writhe_min'] = np.nan
            row_data['RMS_Writhe_max'] = np.nan
            row_data['RMS_Writhe_zscore_vs_control'] = np.nan

        # Add RoG data
        if walk_id in walk_rog_data:
            rog_data = walk_rog_data[walk_id]
            row_data['RoG_centroid'] = rog_data.get('centroid_rog', np.nan)

            sampled_rog_clean = [x for x in rog_data.get('sampled_rog', []) if not np.isnan(x)]

            if len(sampled_rog_clean) > 0:
                row_data['RoG_mean'] = np.mean(sampled_rog_clean)
                row_data['RoG_std'] = np.std(sampled_rog_clean)
                row_data['RoG_median'] = np.median(sampled_rog_clean)
                row_data['RoG_min'] = np.min(sampled_rog_clean)
                row_data['RoG_max'] = np.max(sampled_rog_clean)

                # Z-score relative to control
                if args.rog and control_rog_std > 0:
                    z_score = (row_data['RoG_mean'] - control_rog_mean) / control_rog_std
                    row_data['RoG_zscore_vs_control'] = z_score
                else:
                    row_data['RoG_zscore_vs_control'] = np.nan
            else:
                row_data['RoG_mean'] = np.nan
                row_data['RoG_std'] = np.nan
                row_data['RoG_median'] = np.nan
                row_data['RoG_min'] = np.nan
                row_data['RoG_max'] = np.nan
                row_data['RoG_zscore_vs_control'] = np.nan
        else:
            row_data['RoG_centroid'] = np.nan
            row_data['RoG_mean'] = np.nan
            row_data['RoG_std'] = np.nan
            row_data['RoG_median'] = np.nan
            row_data['RoG_min'] = np.nan
            row_data['RoG_max'] = np.nan
            row_data['RoG_zscore_vs_control'] = np.nan

        # Add V2 data (only if --vassiliev is enabled)
        if args.vassiliev:
            if walk_id in walk_v2_data:
                v2_data = walk_v2_data[walk_id]
                row_data['V2_centroid'] = v2_data['centroid_v2']

                sampled_v2_clean = [x for x in v2_data['sampled_v2'] if not np.isnan(x)]
                if len(sampled_v2_clean) > 0:
                    row_data['V2_mean'] = np.mean(sampled_v2_clean)
                    row_data['V2_std'] = np.std(sampled_v2_clean)
                    row_data['V2_median'] = np.median(sampled_v2_clean)
                    row_data['V2_min'] = np.min(sampled_v2_clean)
                    row_data['V2_max'] = np.max(sampled_v2_clean)

                    # Z-score relative to control
                    if control_v2_std > 0:
                        z_score = (row_data['V2_mean'] - control_v2_mean) / control_v2_std
                        row_data['V2_zscore_vs_control'] = z_score
                    else:
                        row_data['V2_zscore_vs_control'] = np.nan
                else:
                    row_data['V2_mean'] = np.nan
                    row_data['V2_std'] = np.nan
                    row_data['V2_median'] = np.nan
                    row_data['V2_min'] = np.nan
                    row_data['V2_max'] = np.nan
                    row_data['V2_zscore_vs_control'] = np.nan
            else:
                row_data['V2_centroid'] = np.nan
                row_data['V2_mean'] = np.nan
                row_data['V2_std'] = np.nan
                row_data['V2_median'] = np.nan
                row_data['V2_min'] = np.nan
                row_data['V2_max'] = np.nan
                row_data['V2_zscore_vs_control'] = np.nan

        # Add xyCN data
        if walk_id in walk_xycn_data:
            xycn_data = walk_xycn_data[walk_id]
            row_data['xyCN_centroid'] = xycn_data['centroid_xycn']

            sampled_xycn_clean = [x for x in xycn_data['sampled_xycn'] if not np.isnan(x)]
            if len(sampled_xycn_clean) > 0:
                row_data['xyCN_mean'] = np.mean(sampled_xycn_clean)
                row_data['xyCN_std'] = np.std(sampled_xycn_clean)
                row_data['xyCN_median'] = np.median(sampled_xycn_clean)
                row_data['xyCN_min'] = np.min(sampled_xycn_clean)
                row_data['xyCN_max'] = np.max(sampled_xycn_clean)

                # Z-score relative to control
                if args.xyCN and control_xycn_std > 0:
                    z_score = (row_data['xyCN_mean'] - control_xycn_mean) / control_xycn_std
                    row_data['xyCN_zscore_vs_control'] = z_score
                else:
                    row_data['xyCN_zscore_vs_control'] = np.nan
            else:
                row_data['xyCN_mean'] = np.nan
                row_data['xyCN_std'] = np.nan
                row_data['xyCN_median'] = np.nan
                row_data['xyCN_min'] = np.nan
                row_data['xyCN_max'] = np.nan
                row_data['xyCN_zscore_vs_control'] = np.nan
        else:
            row_data['xyCN_centroid'] = np.nan
            row_data['xyCN_mean'] = np.nan
            row_data['xyCN_std'] = np.nan
            row_data['xyCN_median'] = np.nan
            row_data['xyCN_min'] = np.nan
            row_data['xyCN_max'] = np.nan
            row_data['xyCN_zscore_vs_control'] = np.nan

        # Add xzCN data
        if walk_id in walk_xzcn_data:
            xzcn_data = walk_xzcn_data[walk_id]
            row_data['xzCN_centroid'] = xzcn_data['centroid_xzcn']

            sampled_xzcn_clean = [x for x in xzcn_data['sampled_xzcn'] if not np.isnan(x)]
            if len(sampled_xzcn_clean) > 0:
                row_data['xzCN_mean'] = np.mean(sampled_xzcn_clean)
                row_data['xzCN_std'] = np.std(sampled_xzcn_clean)
                row_data['xzCN_median'] = np.median(sampled_xzcn_clean)
                row_data['xzCN_min'] = np.min(sampled_xzcn_clean)
                row_data['xzCN_max'] = np.max(sampled_xzcn_clean)

                # Z-score relative to control
                if args.xzCN and control_xzcn_std > 0:
                    z_score = (row_data['xzCN_mean'] - control_xzcn_mean) / control_xzcn_std
                    row_data['xzCN_zscore_vs_control'] = z_score
                else:
                    row_data['xzCN_zscore_vs_control'] = np.nan
            else:
                row_data['xzCN_mean'] = np.nan
                row_data['xzCN_std'] = np.nan
                row_data['xzCN_median'] = np.nan
                row_data['xzCN_min'] = np.nan
                row_data['xzCN_max'] = np.nan
                row_data['xzCN_zscore_vs_control'] = np.nan
        else:
            row_data['xzCN_centroid'] = np.nan
            row_data['xzCN_mean'] = np.nan
            row_data['xzCN_std'] = np.nan
            row_data['xzCN_median'] = np.nan
            row_data['xzCN_min'] = np.nan
            row_data['xzCN_max'] = np.nan
            row_data['xzCN_zscore_vs_control'] = np.nan

        # Add yzCN data
        if walk_id in walk_yzcn_data:
            yzcn_data = walk_yzcn_data[walk_id]
            row_data['yzCN_centroid'] = yzcn_data['centroid_yzcn']

            sampled_yzcn_clean = [x for x in yzcn_data['sampled_yzcn'] if not np.isnan(x)]
            if len(sampled_yzcn_clean) > 0:
                row_data['yzCN_mean'] = np.mean(sampled_yzcn_clean)
                row_data['yzCN_std'] = np.std(sampled_yzcn_clean)
                row_data['yzCN_median'] = np.median(sampled_yzcn_clean)
                row_data['yzCN_min'] = np.min(sampled_yzcn_clean)
                row_data['yzCN_max'] = np.max(sampled_yzcn_clean)

                # Z-score relative to control
                if args.yzCN and control_yzcn_std > 0:
                    z_score = (row_data['yzCN_mean'] - control_yzcn_mean) / control_yzcn_std
                    row_data['yzCN_zscore_vs_control'] = z_score
                else:
                    row_data['yzCN_zscore_vs_control'] = np.nan
            else:
                row_data['yzCN_mean'] = np.nan
                row_data['yzCN_std'] = np.nan
                row_data['yzCN_median'] = np.nan
                row_data['yzCN_min'] = np.nan
                row_data['yzCN_max'] = np.nan
                row_data['yzCN_zscore_vs_control'] = np.nan
        else:
            row_data['yzCN_centroid'] = np.nan
            row_data['yzCN_mean'] = np.nan
            row_data['yzCN_std'] = np.nan
            row_data['yzCN_median'] = np.nan
            row_data['yzCN_min'] = np.nan
            row_data['yzCN_max'] = np.nan
            row_data['yzCN_zscore_vs_control'] = np.nan

        # Add Kendall data
        if walk_id in walk_kendall_data:
            kendall_data = walk_kendall_data[walk_id]
            row_data['Kendall_converged'] = kendall_data['converged']
            row_data['Kendall_iterations'] = kendall_data['iterations']
            row_data['Kendall_mean_procrustes_dist'] = kendall_data['mean_procrustes_dist']
            row_data['Kendall_std_procrustes_dist'] = kendall_data['std_procrustes_dist']
            row_data['Kendall_average_knot_type'] = kendall_data['knot_type']
            row_data['Kendall_average_knot_det'] = kendall_data['determinant']
        else:
            row_data['Kendall_converged'] = np.nan
            row_data['Kendall_iterations'] = np.nan
            row_data['Kendall_mean_procrustes_dist'] = np.nan
            row_data['Kendall_std_procrustes_dist'] = np.nan
            row_data['Kendall_average_knot_type'] = ''
            row_data['Kendall_average_knot_det'] = np.nan

        # Calculate metrics for each control
        for control_label, control_results in controls.items():
            # Extract knot properties from control (handles both 2-tuple and 3-tuple formats)
            control_knot_props = extract_knot_properties(control_results)

            # Calculate KL divergence if needed
            if args.metrics in ['all', 'kl']:
                kl_div, _, _ = calculate_kl_divergence(walk_knot_props, control_knot_props)
                row_data[f'KL_divergence_{control_label}'] = kl_div

            # Calculate EMD if needed
            if args.metrics in ['all', 'wasserstein']:
                emd = calculate_emd(walk_knot_props, control_knot_props)
                row_data[f'EMD_{control_label}'] = emd

        results_data.append(row_data)

    # Create DataFrame and save results
    results_df = pd.DataFrame(results_data)
    results_df = results_df.sort_values('Walk_ID')

    # Save to CSV
    output_file = f'{args.output}_results.csv'
    results_df.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")

    # Calculate pairwise comparisons between controls if there are multiple
    pairwise_results = {}
    if len(controls) > 1:
        print("\nCalculating pairwise control comparisons...")
        control_labels = list(controls.keys())

        for i, label1 in enumerate(control_labels):
            for j, label2 in enumerate(control_labels):
                if i < j:  # Only compute once for each pair
                    pair_key = f"{label1}_vs_{label2}"
                    pairwise_results[pair_key] = {}

                    # Calculate metrics
                    if args.metrics in ['all', 'kl']:
                        kl_div, _, _ = calculate_kl_divergence(controls[label1], controls[label2])
                        pairwise_results[pair_key]['KL_divergence'] = kl_div

                    if args.metrics in ['all', 'wasserstein']:
                        emd = calculate_emd(controls[label1], controls[label2])
                        pairwise_results[pair_key]['EMD'] = emd

    # Print summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)

    # Control distributions summary
    for control_label, control_results in controls.items():
        # Extract knot properties (controls are 2-tuples, but extract_knot_properties handles both)
        control_knot_props = extract_knot_properties(control_results)
        control_classified = [classify_name(name) for name, _ in control_knot_props]
        control_counts = Counter(control_classified)
        n_control = len(control_results)
        print(f"\n{control_label} Distribution (N={n_control}):")
        print(f"  Unknot (0_1): {control_counts.get('0_1', 0)/n_control:.3f}")
        print(f"  Trefoil (3_1): {control_counts.get('3_1', 0)/n_control:.3f}")
        print(f"  Complex: {control_counts.get('Complex', 0)/n_control:.3f}")

    # Experimental data summary
    print(f"\nExperimental Data Summary:")
    print(f"  Mean Unknot fraction: {results_df['Unknot_fraction'].mean():.3f} ± {results_df['Unknot_fraction'].std():.3f}")
    print(f"  Mean Trefoil fraction: {results_df['Trefoil_fraction'].mean():.3f} ± {results_df['Trefoil_fraction'].std():.3f}")
    print(f"  Mean Complex fraction: {results_df['Complex_fraction'].mean():.3f} ± {results_df['Complex_fraction'].std():.3f}")

    # Distance metrics for experimental vs controls
    print(f"\nDistance Metrics (Experimental Data):")
    for control_label in controls.keys():
        print(f"\n  Vs {control_label}:")

        if args.metrics in ['all', 'kl']:
            kl_col = f'KL_divergence_{control_label}'
            print(f"    Mean KL divergence: {results_df[kl_col].mean():.4f} ± {results_df[kl_col].std():.4f}")

            # Identify outliers for this control
            kl_threshold = results_df[kl_col].mean() + 2 * results_df[kl_col].std()
            outliers_kl = results_df[results_df[kl_col] > kl_threshold]

            if not outliers_kl.empty:
                print(f"    Outliers by KL divergence (>2σ):")
                for _, row in outliers_kl.iterrows():
                    print(f"      Walk-ID {row['Walk_ID']}: KL={row[kl_col]:.4f}")

        if args.metrics in ['all', 'wasserstein']:
            emd_col = f'EMD_{control_label}'
            print(f"    Mean EMD: {results_df[emd_col].mean():.4f} ± {results_df[emd_col].std():.4f}")

            # Identify outliers for this control
            emd_threshold = results_df[emd_col].mean() + 2 * results_df[emd_col].std()
            outliers_emd = results_df[results_df[emd_col] > emd_threshold]

            if not outliers_emd.empty:
                print(f"    Outliers by EMD (>2σ):")
                for _, row in outliers_emd.iterrows():
                    print(f"      Walk-ID {row['Walk_ID']}: EMD={row[emd_col]:.4f}")

    # Print pairwise control comparisons
    if pairwise_results:
        print(f"\nPairwise Control Comparisons:")
        for pair_key, metrics in pairwise_results.items():
            print(f"\n  {pair_key.replace('_', ' ')}:")
            if 'KL_divergence' in metrics:
                print(f"    KL divergence: {metrics['KL_divergence']:.4f}")
            if 'EMD' in metrics:
                print(f"    EMD: {metrics['EMD']:.4f}")

    # Generate comprehensive PDF report with report cards
    print("\nGenerating comprehensive PDF report...")
    all_figures = []

    # Part 0: Create first-page summary (always include as page 1)
    try:
        first_page_fig = create_first_page_summary(
            walk_results=walk_results,
            controls=controls,
            walk_acn_data=walk_acn_data if args.acn else {},
            control_acn_values=control_acn_values if args.acn else [],
            morans_i_dict=morans_i_dict,
            results_df=results_df,
            centroid_knot_types=centroid_knot_types if args.kendall else None,
            label=args.label,
            show_centroids=args.show_centroids
        )
        all_figures.append(first_page_fig)
        print(f"  ✓ First-page summary added")

        # Optional second page: 2-category (Unknotted vs Knotted) version
        if args.no_stacked_barplots:
            second_page_fig = create_first_page_summary(
                walk_results=walk_results,
                controls=controls,
                walk_acn_data=walk_acn_data if args.acn else {},
                control_acn_values=control_acn_values if args.acn else [],
                morans_i_dict=morans_i_dict,
                results_df=results_df,
                centroid_knot_types=centroid_knot_types if args.kendall else None,
                label=args.label,
                two_categories=True,
                show_centroids=args.show_centroids
            )
            all_figures.append(second_page_fig)
            print(f"  ✓ Second-page summary (Unknotted vs Knotted) added")
    except Exception as e:
        print(f"  Warning: Could not create first-page summary: {e}")
        import traceback
        traceback.print_exc()

    # Population-level parabola + distance-matrix pages (De Witt curves)
    parabola_bundle = build_population_parabola_pages(walk_results, label=args.label)
    for fig in parabola_bundle['main_pages']:
        all_figures.append(fig)

    # Standalone 3D-only parabola PDFs (sidecar files)
    if parabola_bundle['standalone_3d_xyz'] is not None:
        side_path = f"{args.output}_parabola_3d_xyz.pdf"
        parabola_bundle['standalone_3d_xyz'].savefig(side_path)
        plt.close(parabola_bundle['standalone_3d_xyz'])
        print(f"  ✓ Wrote {side_path}")
    if parabola_bundle['standalone_3d_pc'] is not None:
        side_path = f"{args.output}_parabola_3d_pc.pdf"
        parabola_bundle['standalone_3d_pc'].savefig(side_path)
        plt.close(parabola_bundle['standalone_3d_pc'])
        print(f"  ✓ Wrote {side_path}")

    # Part 1: Create report cards (one page per walk-id)
    walk_ids = sorted(walk_results.keys())
    print(f"  Creating report cards for {len(walk_ids)} walk-ids...")

    for walk_id in tqdm(walk_ids, desc="Generating report cards"):
        try:
            fig = create_walk_report_card(
                walk_id=walk_id,
                walk_results=walk_results,
                controls=controls,
                walk_acn_data=walk_acn_data if args.acn else {},
                walk_kendall_data=walk_kendall_data if args.kendall else {},
                morans_i_dict=morans_i_dict,
                walk_tp1_tp21_distances=walk_tp1_tp21_distances,
                control_acn_values=control_acn_values if args.acn else [],
                global_x_range=global_x_range if args.acn else (0, 10),
                global_y_max=global_y_max if args.acn else 1,
                walk_groups=walk_groups,
                centroid_knot_types=centroid_knot_types if (args.kendall or args.puknot) else {},
                walk_v2_data=walk_v2_data if args.vassiliev else {},
                control_v2_values=control_v2_values if args.vassiliev else [],
                global_v2_x_range=global_v2_x_range if args.vassiliev else (0, 1),
                global_v2_y_max=global_v2_y_max if args.vassiliev else 1,
                walk_xycn_data=walk_xycn_data if args.xyCN else {},
                control_xycn_values=control_xycn_values if args.xyCN else [],
                global_xycn_x_range=global_xycn_x_range if args.xyCN else (0, 10),
                global_xycn_y_max=global_xycn_y_max if args.xyCN else 1,
                walk_xzcn_data=walk_xzcn_data if args.xzCN else {},
                control_xzcn_values=control_xzcn_values if args.xzCN else [],
                walk_yzcn_data=walk_yzcn_data if args.yzCN else {},
                control_yzcn_values=control_yzcn_values if args.yzCN else [],
                unified_planar_cn_x_range=unified_planar_cn_x_range if (args.xyCN or args.xzCN or args.yzCN) else (0, 10),
                unified_planar_cn_y_max=unified_planar_cn_y_max if (args.xyCN or args.xzCN or args.yzCN) else 1,
                metrics=args.metrics,
                walk_writhe_data=walk_writhe_data if args.acn else {},
                control_writhe_values=control_writhe_values if args.acn else [],
                global_writhe_x_range=global_writhe_x_range if args.acn else (-5, 5),
                global_writhe_y_max=global_writhe_y_max if args.acn else 1,
                control_rms_writhe_values=control_rms_writhe_values if args.acn else [],
                global_rms_writhe_x_range=global_rms_writhe_x_range if args.acn else (0, 10),
                global_rms_writhe_y_max=global_rms_writhe_y_max if args.acn else 1,
                walk_rog_data=walk_rog_data if args.rog else {},
                control_rog_values=control_rog_values if args.rog else [],
                global_rog_x_range=global_rog_x_range if args.rog else (0, 100),
                global_rog_y_max=global_rog_y_max if args.rog else 1,
                puknot=args.puknot,
                no_stacked=args.no_stacked_barplots,
                show_centroids=args.show_centroids
            )
            all_figures.append(fig)
        except Exception as e:
            import traceback
            print(f"    Warning: Could not create report card for Walk-ID {walk_id}: {e}")
            print(f"    Error type: {type(e).__name__}")
            if walk_id in walk_results:
                walk_data = walk_results[walk_id]
                print(f"    Walk data type: {type(walk_data)}")
                print(f"    Walk data length: {len(walk_data) if hasattr(walk_data, '__len__') else 'N/A'}")
                if hasattr(walk_data, '__len__') and len(walk_data) > 0:
                    print(f"    First 3 items: {walk_data[:3]}")
            # Print first line of traceback to see where error occurred
            tb_lines = traceback.format_tb(e.__traceback__)
            if tb_lines:
                print(f"    Error location: {tb_lines[-1].strip()}")

    print(f"  Created {len(all_figures)} report cards")

    # Part 2: Create summary visualizations
    print("  Creating summary visualizations...")
    try:
        summary_figs = create_summary_visualizations(
            walk_results=walk_results,
            controls=controls,
            results_df=results_df,
            metrics=args.metrics
        )
        if summary_figs:
            all_figures.extend(summary_figs)
            print(f"  Added {len(summary_figs)} summary plots")
    except Exception as e:
        print(f"    Warning: Could not create summary visualizations: {e}")

    # Part 3: Assemble comprehensive PDF report
    if all_figures:
        create_comprehensive_pdf_report(all_figures, args.output)
    else:
        print("  Warning: No figures were created for the PDF report")

    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()