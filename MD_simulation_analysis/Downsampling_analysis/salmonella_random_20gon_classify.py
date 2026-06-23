#!/usr/bin/env python3
"""
Salmonella Simulations Random Sampling Knot Type Classification

For each chain and each valid stride, produce downsampled polygons using one
of four methods: stride (one random point per chunk), uniform (N random
points from the whole chain), systematic (deterministic every-Nth-point),
or centroid (mean of each chunk).

Supports .tsv (tab-delimited) and .dat (space-delimited with integer header
line) input formats. The format is auto-detected from the file extension.

Optional dominant closure: with --closure, each downsampled polygon is
independently classified by testing multiple sphere-based closures and
taking a majority vote to determine the dominant knot type.

Usage:
    python salmonella_random_20gon_classify.py <polygon_dir>
    python salmonella_random_20gon_classify.py <polygon_dir> --length 4000
    python salmonella_random_20gon_classify.py <polygon_dir> -t knotted
    python salmonella_random_20gon_classify.py <polygon_dir> -t knotted -n 200
    python salmonella_random_20gon_classify.py <polygon_dir> -t knotted --complex 10000 5000 2500
    python salmonella_random_20gon_classify.py <polygon_dir> --closure -t closure_test
    python salmonella_random_20gon_classify.py <polygon_dir> --closure --directions 50 -t closure_50
"""

import os
import sys
import argparse
import multiprocessing

# Threading config: sequential run, all threads to knoodle
# THREADS_PER_PROCESS = 5
# MAX_THREADS = multiprocessing.cpu_count()
# NUM_WORKERS = max(1, (MAX_THREADS - 1) // THREADS_PER_PROCESS)
THREADS_PER_PROCESS = multiprocessing.cpu_count()
NUM_WORKERS = 1

# Set internal threading BEFORE other imports
os.environ["OMP_NUM_THREADS"] = str(THREADS_PER_PROCESS)
os.environ["OPENBLAS_NUM_THREADS"] = str(THREADS_PER_PROCESS)
os.environ["MKL_NUM_THREADS"] = str(THREADS_PER_PROCESS)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(THREADS_PER_PROCESS)
os.environ["NUMEXPR_NUM_THREADS"] = str(THREADS_PER_PROCESS)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import pandas as pd
from multiprocessing import Pool
from tqdm import tqdm
import glob

# Configuration
MIN_LENGTH = 5
NUM_SAMPLES_DEFAULT = 100


def load_polygon(filepath):
    """Load polygon coordinates from .tsv or .dat files.

    .tsv files are tab-delimited with no header.
    .dat files have an integer header line (point count) followed by
    space-delimited coordinates.
    """
    try:
        ext = os.path.splitext(filepath)[1].lower()
        if ext == ".dat":
            with open(filepath, "r") as f:
                first_line = f.readline().strip()
            expected_count = int(first_line)
            points = np.loadtxt(filepath, skiprows=1)
            if points.size == 0:
                return None
            if len(points) != expected_count:
                print(f"  WARNING: {filepath} header says {expected_count} "
                      f"points but found {len(points)}")
            return points
        else:
            points = np.loadtxt(filepath, delimiter='\t')
            if points.size == 0:
                return None
            return points
    except Exception:
        return None


def flatten_coords(points):
    """Flatten 3D points to a single list of coordinates."""
    return [float(coord) for point in points for coord in point]


def classify_knot_type(determinant, crossing_count):
    """Classify knot type based on crossing count."""
    if crossing_count == 0:
        return "Unknotted"
    elif crossing_count == 3:
        if determinant is not None and determinant > 0:
            if abs(determinant - 3.0) < 1e-6:
                return "Trefoil"
            else:
                return "Complex"
        else:
            return "Trefoil"
    else:
        return "Complex"


def get_valid_strides(length, min_length):
    """Get all strides that give length >= min_length."""
    strides = []
    for stride in range(1, length + 1):
        if length % stride == 0:
            result_length = length // stride
            if result_length >= min_length:
                strides.append(stride)
    return sorted(strides)


def fibonacci_sphere(n=100):
    """Generate n uniformly distributed points on unit sphere."""
    points = []
    phi = np.pi * (3.0 - np.sqrt(5.0))

    for i in range(n):
        y = 1 - (i / float(n - 1)) * 2
        radius = np.sqrt(1 - y * y)
        theta = phi * i
        x = np.cos(theta) * radius
        z = np.sin(theta) * radius
        points.append([x, y, z])

    return np.array(points)


def compute_bounding_sphere(points):
    """Compute bounding sphere center and radius."""
    center = np.mean(points, axis=0)
    distances = np.linalg.norm(points - center, axis=1)
    radius = np.max(distances) * 1.1
    return center, radius


def close_curve_along_direction(points, direction, sphere_center, sphere_radius):
    """Close curve by extending both endpoints along same direction."""
    extension_distance = 5.0 * sphere_radius
    direction = direction / np.linalg.norm(direction)

    first_point = points[0]
    last_point = points[-1]

    extended_first = first_point + direction * extension_distance
    extended_last = last_point + direction * extension_distance

    closed_curve = np.vstack([extended_first, points, extended_last])
    return closed_curve


def analyze_closure(points, direction, sphere_center, sphere_radius):
    """Analyze a single closure and return knot type."""
    import knoodle

    try:
        closed_curve = close_curve_along_direction(points, direction, sphere_center, sphere_radius)
        coords = flatten_coords(closed_curve)
        analyzer = knoodle.KnotAnalyzer(coords, simplify=True, simplify_level=5)

        alexander_result = analyzer.alexander([-1.0])[0]
        determinant = abs(alexander_result.to_double())

        rounded_det = round(determinant)
        if abs(determinant - rounded_det) < 1e-6:
            determinant = float(rounded_det)

        knot_type = classify_knot_type(determinant, analyzer.crossing_count)

        return {
            'knot_type': knot_type,
            'determinant': determinant,
            'crossing_count': analyzer.crossing_count,
            'closed_curve': closed_curve
        }
    except Exception:
        return None


def find_dominant_closure(points, n_samples=100):
    """
    Find dominant knot type among n_samples closures via majority vote.

    Returns dict with dominant_type, representative determinant/crossing_count,
    and per-type fractions. Returns None on total failure.
    """
    import knoodle

    center, radius = compute_bounding_sphere(points)
    directions = fibonacci_sphere(n_samples)

    closure_results = []
    for i, direction in enumerate(directions):
        result = analyze_closure(points, direction, center, radius)
        if result:
            result['index'] = i
            closure_results.append(result)

    if len(closure_results) == 0:
        return None

    # Count knot types
    type_counts = {}
    for result in closure_results:
        knot_type = result['knot_type']
        type_counts[knot_type] = type_counts.get(knot_type, 0) + 1

    dominant_type = max(type_counts, key=type_counts.get)

    total = len(closure_results)
    type_fractions = {k: v/total for k, v in type_counts.items()}

    # Pick first representative of dominant type
    rep = None
    for result in closure_results:
        if result['knot_type'] == dominant_type:
            rep = result
            break

    return {
        "dominant_type": dominant_type,
        "crossing_count": rep["crossing_count"],
        "determinant": rep["determinant"],
        "unknot_fraction": type_fractions.get("Unknotted", 0.0),
        "trefoil_fraction": type_fractions.get("Trefoil", 0.0),
        "complex_fraction": type_fractions.get("Complex", 0.0),
    }


def downsample_systematic(points, stride):
    """Downsample by taking every Nth point (systematic/regular sampling)."""
    return points[::stride]


def downsample_centroid(points, stride):
    """Downsample by replacing each chunk with its centroid.

    Splits the chain into (len // stride) chunks of (stride) points each
    and returns the mean of each chunk. The output points do not necessarily
    lie on the original curve.
    """
    num_chunks = len(points) // stride
    centroids = np.empty((num_chunks, points.shape[1]))
    for i in range(num_chunks):
        centroids[i] = points[i * stride:(i + 1) * stride].mean(axis=0)
    return centroids


def sample_random_polygons(points, stride, num_samples, rng):
    """Sample random polygons by picking one point per chunk.

    For a given stride, splits the chain into (LENGTH // stride) chunks of
    (stride) points each. For each sample, picks one random index per chunk.

    Returns list of numpy arrays of shape (num_chunks, 3).
    """
    num_chunks = len(points) // stride
    chunk_size = stride

    samples = []
    for _ in range(num_samples):
        indices = []
        for chunk_idx in range(num_chunks):
            start = chunk_idx * chunk_size
            idx = start + rng.integers(0, chunk_size)
            indices.append(idx)
        samples.append(points[indices])
    return samples


def sample_random_polygons_uniform(points, stride, num_samples, rng):
    """Sample random polygons by picking N random points from the whole chain.

    Picks N = len(points) // stride random indices, sorts them to maintain
    chain order.
    """
    n_points = len(points) // stride
    samples = []
    for _ in range(num_samples):
        indices = sorted(rng.choice(len(points), size=n_points, replace=False))
        samples.append(points[indices])
    return samples


def analyze_stride_random_samples(args):
    """Analyze random samples at a specific stride for a single file.

    When use_closure is True, each downsampled polygon is independently
    classified via dominant closure (majority vote over n_directions
    sphere-based closures).  When False, the polygon is classified directly.
    """
    import knoodle

    filepath, filename, stride, file_index, num_samples, method, chain_length, \
        use_closure, n_directions = args
    target_length = chain_length // stride
    results = []

    try:
        points = load_polygon(filepath)
        if points is None:
            raise ValueError("Failed to load polygon")

        if method == "systematic":
            samples = [downsample_systematic(points, stride)]
        elif method == "centroid":
            samples = [downsample_centroid(points, stride)]
        else:
            rng = np.random.default_rng(42 + file_index * chain_length + stride)
            if method == "uniform":
                samples = sample_random_polygons_uniform(points, stride, num_samples, rng)
            else:
                samples = sample_random_polygons(points, stride, num_samples, rng)

        for sample_id, sample_points in enumerate(samples):
            try:
                if use_closure:
                    # Dominant closure: majority vote over n_directions closures
                    closure_result = find_dominant_closure(sample_points, n_directions)
                    if closure_result is None:
                        raise ValueError("All closures failed")

                    results.append({
                        "filename": filename,
                        "sample_id": sample_id,
                        "stride": stride,
                        "length": target_length,
                        "crossing_count": closure_result["crossing_count"],
                        "determinant": closure_result["determinant"],
                        "knot_type": closure_result["dominant_type"],
                        "dominant_type": closure_result["dominant_type"],
                        "unknot_fraction": closure_result["unknot_fraction"],
                        "trefoil_fraction": closure_result["trefoil_fraction"],
                        "complex_fraction": closure_result["complex_fraction"],
                        "error": None
                    })
                else:
                    # Direct classification (no closure)
                    coords = flatten_coords(sample_points)
                    analyzer = knoodle.KnotAnalyzer(coords, simplify=True, simplify_level=5)

                    alexander_result = analyzer.alexander([-1.0])[0]
                    determinant = abs(alexander_result.to_double())

                    rounded_det = round(determinant)
                    if abs(determinant - rounded_det) < 1e-6:
                        determinant = float(rounded_det)

                    knot_type = classify_knot_type(determinant, analyzer.crossing_count)

                    results.append({
                        "filename": filename,
                        "sample_id": sample_id,
                        "stride": stride,
                        "length": target_length,
                        "crossing_count": analyzer.crossing_count,
                        "determinant": determinant,
                        "knot_type": knot_type,
                        "error": None
                    })
            except Exception as e:
                row = {
                    "filename": filename,
                    "sample_id": sample_id,
                    "stride": stride,
                    "length": target_length,
                    "crossing_count": None,
                    "determinant": None,
                    "knot_type": None,
                    "error": str(e)
                }
                if use_closure:
                    row["dominant_type"] = None
                    row["unknot_fraction"] = 0.0
                    row["trefoil_fraction"] = 0.0
                    row["complex_fraction"] = 0.0
                results.append(row)

    except Exception as e:
        error_count = 1 if method in ("systematic", "centroid") else num_samples
        for sample_id in range(error_count):
            row = {
                "filename": filename,
                "sample_id": sample_id,
                "stride": stride,
                "length": target_length,
                "crossing_count": None,
                "determinant": None,
                "knot_type": None,
                "error": str(e)
            }
            if use_closure:
                row["dominant_type"] = None
                row["unknot_fraction"] = 0.0
                row["trefoil_fraction"] = 0.0
                row["complex_fraction"] = 0.0
            results.append(row)

    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Classify random polygon samples from salmonella structure chains"
    )
    parser.add_argument(
        "polygon_dir",
        help="Directory containing structure files (.tsv or .dat)"
    )
    parser.add_argument(
        "--length", "-l",
        type=int,
        default=10000,
        help="Number of points per input polygon (default: 10000)"
    )
    parser.add_argument(
        "--complex",
        nargs="+",
        type=int,
        default=[],
        help="Lengths to skip and mark as Complex (e.g., --complex 10000 5000 2500)"
    )
    parser.add_argument(
        "-t", "--tag",
        type=str,
        default="",
        help="Tag to add to output filename (e.g., -t knotted -> random_sample_analysis_knotted.csv)"
    )
    parser.add_argument(
        "-n", "--num-samples",
        type=int,
        default=NUM_SAMPLES_DEFAULT,
        help=f"Number of random samples per chain per stride (default: {NUM_SAMPLES_DEFAULT})"
    )
    parser.add_argument(
        "--method",
        choices=["stride", "uniform", "systematic", "centroid"],
        default="stride",
        help="Sampling method: stride (one random point per chunk), "
             "uniform (N random points from entire chain), "
             "systematic (deterministic every-Nth-point), "
             "or centroid (mean of each chunk)"
    )
    parser.add_argument(
        "--closure", action="store_true", default=False,
        help="Close curves using dominant closure method before sampling"
    )
    parser.add_argument(
        "--directions", type=int, default=100,
        help="Number of sphere directions for dominant closure (default: 100)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    polygon_dir = os.path.expanduser(args.polygon_dir)
    complex_lengths = set(args.complex)
    tag = args.tag
    num_samples = args.num_samples
    method = args.method
    chain_length = args.length
    use_closure = args.closure
    n_directions = args.directions

    print("=" * 60)
    print("Salmonella Random Sampling Knot Type Classification")
    print(f"Directory: {polygon_dir}")
    print(f"Chain length: {chain_length}")
    print(f"Sampling method: {method}")
    if method not in ("systematic", "centroid"):
        print(f"Samples per chain per stride: {num_samples}")
    if use_closure:
        print(f"Closure: dominant closure ({n_directions} directions)")
    if complex_lengths:
        print(f"Marking as Complex: {sorted(complex_lengths, reverse=True)}")
    if tag:
        print(f"Output tag: {tag}")
    print("=" * 60)

    # Discover input files by extension: prefer .tsv, fall back to .dat
    all_files = sorted(glob.glob(os.path.join(polygon_dir, "*.tsv")))
    if not all_files:
        all_files = sorted(glob.glob(os.path.join(polygon_dir, "*.dat")))
    print(f"\nFound {len(all_files)} structure files")

    if not all_files:
        print("ERROR: No structure files found!")
        sys.exit(1)

    all_filenames = [os.path.basename(f) for f in all_files]

    # Determine which strides to skip based on --complex lengths
    strides = get_valid_strides(chain_length, MIN_LENGTH)
    skip_strides = []
    for length in complex_lengths:
        if chain_length % length == 0:
            stride = chain_length // length
            skip_strides.append(stride)

    # Filter out strides that produce complex lengths
    compute_strides = [s for s in strides if s not in skip_strides]

    print("\n" + "=" * 60)
    print("Random sampling analysis...")
    print("=" * 60)
    print(f"  Using {len(compute_strides)} strides: {compute_strides}")
    print(f"  Resulting lengths: {[chain_length // s for s in compute_strides]}")
    if skip_strides:
        print(f"  Skipping strides (marked Complex): {skip_strides}")
        print(f"  Skipped lengths: {[chain_length // s for s in skip_strides]}")

    # Build tasks: (filepath, filename, stride, file_index, num_samples, method,
    #               chain_length, use_closure, n_directions)
    all_tasks = []
    print("\n  Creating tasks...")
    for file_index, filename in enumerate(all_filenames):
        filepath = os.path.join(polygon_dir, filename)
        for stride in compute_strides:
            all_tasks.append((
                filepath, filename, stride, file_index, num_samples, method,
                chain_length, use_closure, n_directions
            ))

    samples_per_task = 1 if method in ("systematic", "centroid") else num_samples
    total_classifications = len(all_tasks) * samples_per_task
    print(f"\n  Total analysis tasks: {len(all_tasks)}")
    print(f"  Total classifications: {total_classifications}")

    print(f"  Running analysis ({NUM_WORKERS} workers, {THREADS_PER_PROCESS} threads each)...")
    # Restart workers every 20 tasks to prevent memory leaks in knoodle
    with Pool(processes=NUM_WORKERS, maxtasksperchild=20) as pool:
        nested_results = list(tqdm(
            pool.imap(analyze_stride_random_samples, all_tasks),
            total=len(all_tasks),
            desc="Analyzing"
        ))

    # Flatten results
    analysis_results = []
    for task_results in nested_results:
        analysis_results.extend(task_results)

    # Add entries for skipped lengths as Complex
    complex_results = []
    for filename in all_filenames:
        for stride in skip_strides:
            length = chain_length // stride
            for sample_id in range(samples_per_task):
                row = {
                    "filename": filename,
                    "sample_id": sample_id,
                    "stride": stride,
                    "length": length,
                    "crossing_count": None,
                    "determinant": None,
                    "knot_type": "Complex",
                    "error": None
                }
                if use_closure:
                    row["dominant_type"] = "Complex"
                    row["unknot_fraction"] = None
                    row["trefoil_fraction"] = None
                    row["complex_fraction"] = None
                complex_results.append(row)

    df = pd.DataFrame(complex_results + analysis_results)

    if tag:
        output_file = os.path.join(polygon_dir, f"random_sample_analysis_{tag}.csv")
    else:
        output_file = os.path.join(polygon_dir, "random_sample_analysis.csv")
    df.to_csv(output_file, index=False)
    print(f"\n  Results saved to {output_file}")

    # Summary
    print("\n" + "=" * 60)
    print("Summary Statistics")
    print("=" * 60)

    if len(df) > 0 and "error" in df.columns:
        valid_df = df[df["error"].isna()]
    else:
        valid_df = df

    if len(valid_df) > 0:
        num_polygons = len(all_filenames)
        print(f"\nAll {num_polygons} structures:")
        lengths_to_show = sorted(valid_df["length"].unique(), reverse=True)[:5]
        for length in lengths_to_show:
            length_df = valid_df[valid_df["length"] == length]
            type_counts = length_df["knot_type"].value_counts()
            total = len(length_df)
            print(f"  Length {length}: ", end="")
            for kt in ["Unknotted", "Trefoil", "Complex"]:
                count = type_counts.get(kt, 0)
                pct = 100 * count / total if total > 0 else 0
                print(f"{kt}={pct:.1f}% ", end="")
            print()

    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
