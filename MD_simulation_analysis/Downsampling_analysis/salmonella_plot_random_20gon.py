#!/usr/bin/env python3
"""
Plot Salmonella Random Sampling Knot Type Analysis Results

Creates stacked area plots showing knot type percentages vs length with gradient
coloring, identical in style to salmonella_plot_knot_types.py. Supports single
or two-population comparison (side-by-side subplots).

Colors:
- Single determinant colormap (green -> yellow/orange -> red)
- Unknotted (det=0): anchored at green
- Trefoil (det=3): anchored at yellow/orange
- Complex (all other det): smoothly spread from trefoil color toward red

Usage:
    python salmonella_plot_random_20gon.py <csv_file>
    python salmonella_plot_random_20gon.py <csv_file> --x-start 2000
    python salmonella_plot_random_20gon.py <csv1> <csv2> --labels Knotted Unknotted
"""

import os
import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import seaborn as sns

# Arsuaga reference point
ARSUAGA_LENGTH = 20

# Determinant color mapping anchors
UNKNOT_DETERMINANT = 0
TREFOIL_DETERMINANT = 3
TREFOIL_COLOR_POSITION = 0.16
COMPLEX_COLOR_GAMMA = 0.80
MAX_COMPLEX_BANDS = 80
DEFAULT_SMOOTH_WINDOW = 5

# Single colormap for all knot classes by determinant
DETERMINANT_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "determinant_gradient", ["#2ca02c", "#f1c40f", "#d62728"]
)


def load_data(filepath):
    """Load analysis results."""
    print(f"[INFO] Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    print(f"[INFO] Loaded {len(df)} rows")
    return df


def compute_determinant_percentages(df):
    """Compute percentages by length and knot group.

    Complex determinants are binned into at most MAX_COMPLEX_BANDS groups to keep
    plotting tractable for ensembles with very high determinant diversity.
    """
    print("[INFO] Computing percentages by determinant...")

    # Filter out errors
    valid_df = df[df["error"].isna()].copy()

    grouped_parts = []

    # Unknotted
    unknotted_df = valid_df[valid_df["knot_type"] == "Unknotted"]
    if not unknotted_df.empty:
        unknotted_grouped = unknotted_df.groupby("length", as_index=False).size()
        unknotted_grouped = unknotted_grouped.rename(columns={"size": "count"})
        unknotted_grouped["group_key"] = "Unknotted"
        unknotted_grouped["group_value"] = float(UNKNOT_DETERMINANT)
        grouped_parts.append(unknotted_grouped)

    # Trefoil
    trefoil_df = valid_df[valid_df["knot_type"] == "Trefoil"]
    if not trefoil_df.empty:
        trefoil_grouped = trefoil_df.groupby("length", as_index=False).size()
        trefoil_grouped = trefoil_grouped.rename(columns={"size": "count"})
        trefoil_grouped["group_key"] = "Trefoil"
        trefoil_grouped["group_value"] = float(TREFOIL_DETERMINANT)
        grouped_parts.append(trefoil_grouped)

    # All non-unknotted/non-trefoil are treated as complex-like groups.
    is_complex_like = ~valid_df["knot_type"].isin(["Unknotted", "Trefoil"])

    # Complex-like rows with known determinants: bin in log-space.
    complex_known = valid_df[is_complex_like & valid_df["determinant"].notna()]
    if not complex_known.empty:
        det_values = np.clip(complex_known["determinant"].to_numpy(dtype=float), 1e-12, None)
        log_det = np.log10(det_values)

        unique_log_count = int(np.unique(log_det).size)
        n_bands = max(1, min(MAX_COMPLEX_BANDS, unique_log_count))

        if n_bands == 1:
            bin_idx = np.zeros_like(log_det, dtype=int)
            representative_values = np.array([float(np.power(10.0, log_det.mean()))], dtype=float)
        else:
            log_edges = np.linspace(log_det.min(), log_det.max(), n_bands + 1)
            bin_idx = np.searchsorted(log_edges[1:-1], log_det, side="right")
            representative_logs = 0.5 * (log_edges[:-1] + log_edges[1:])
            representative_values = np.power(10.0, representative_logs)

        complex_group_df = pd.DataFrame({
            "length": complex_known["length"].to_numpy(),
            "band_idx": bin_idx,
        })
        complex_grouped = complex_group_df.groupby(["length", "band_idx"], as_index=False).size()
        complex_grouped = complex_grouped.rename(columns={"size": "count"})
        complex_grouped["group_key"] = complex_grouped["band_idx"].map(lambda idx: f"Complex_{int(idx):03d}")
        complex_grouped["group_value"] = complex_grouped["band_idx"].map(
            lambda idx: float(representative_values[int(idx)])
        )
        complex_grouped = complex_grouped.drop(columns=["band_idx"])
        grouped_parts.append(complex_grouped)

    # Complex-like rows with missing determinant
    complex_unknown = valid_df[is_complex_like & valid_df["determinant"].isna()]
    if not complex_unknown.empty:
        complex_unknown_grouped = complex_unknown.groupby("length", as_index=False).size()
        complex_unknown_grouped = complex_unknown_grouped.rename(columns={"size": "count"})
        complex_unknown_grouped["group_key"] = "Complex_Unknown"
        complex_unknown_grouped["group_value"] = np.nan
        grouped_parts.append(complex_unknown_grouped)

    if not grouped_parts:
        empty = pd.DataFrame(columns=["length", "count", "group_key", "group_value", "total", "percentage"])
        return empty, valid_df

    grouped = pd.concat(grouped_parts, ignore_index=True)

    # Compute total per length
    total_per_length = grouped.groupby("length", as_index=False)["count"].sum()
    total_per_length = total_per_length.rename(columns={"count": "total"})

    # Merge and compute percentages
    merged = grouped.merge(total_per_length, on="length", how="left")
    merged["percentage"] = merged["count"] / merged["total"] * 100

    return merged, valid_df


def create_determinant_colors(complex_determinants, trefoil_pos=TREFOIL_COLOR_POSITION,
                              gamma=COMPLEX_COLOR_GAMMA):
    """Create a single determinant-based colormap for all knot classes.

    Anchors:
    - det=0 (unknot) -> green
    - det=3 (trefoil) -> yellow/orange
    - complex determinants -> smoothly distributed between trefoil and red

    Complex determinants are normalized with a blend of rank and log scaling:
    - rank term prevents extreme values from collapsing most colors near green
    - log term preserves determinant magnitude ordering
    """
    colors = {
        UNKNOT_DETERMINANT: DETERMINANT_CMAP(0.0),
        TREFOIL_DETERMINANT: DETERMINANT_CMAP(trefoil_pos),
    }

    unique_complex = sorted({float(d) for d in complex_determinants if pd.notna(d)})
    if not unique_complex:
        return colors

    if len(unique_complex) == 1:
        colors[unique_complex[0]] = DETERMINANT_CMAP(1.0)
        return colors

    complex_array = np.array(unique_complex, dtype=float)
    log_vals = np.log10(np.clip(complex_array, 1e-12, None))
    log_span = log_vals.max() - log_vals.min()
    if log_span == 0:
        log_norm = np.zeros_like(log_vals)
    else:
        log_norm = (log_vals - log_vals.min()) / log_span

    rank_norm = np.linspace(0.0, 1.0, len(unique_complex))
    blended_norm = 0.70 * rank_norm + 0.30 * log_norm
    blended_norm = np.clip(blended_norm, 0.0, 1.0) ** gamma

    color_positions = trefoil_pos + (1.0 - trefoil_pos) * blended_norm
    for det, pos in zip(unique_complex, color_positions):
        colors[det] = DETERMINANT_CMAP(float(np.clip(pos, 0.0, 1.0)))

    return colors


class AnchoredDeterminantNorm(mcolors.Normalize):
    """Piecewise normalization anchoring det=0 and det=3 at fixed color positions."""

    def __init__(self, vmin, vmax, trefoil_det=TREFOIL_DETERMINANT,
                 trefoil_pos=TREFOIL_COLOR_POSITION, clip=False):
        super().__init__(vmin=vmin, vmax=vmax, clip=clip)
        self.trefoil_det = trefoil_det
        self.trefoil_pos = trefoil_pos

    def __call__(self, value, clip=None):
        result, is_scalar = self.process_value(value)
        values = result.data.astype(float)
        normalized = np.zeros_like(values, dtype=float)

        low_mask = values <= self.trefoil_det
        low_denom = max(self.trefoil_det - self.vmin, 1e-12)
        normalized[low_mask] = ((values[low_mask] - self.vmin) / low_denom) * self.trefoil_pos

        high_mask = ~low_mask
        high_denom = max(self.vmax - self.trefoil_det, 1e-12)
        normalized[high_mask] = self.trefoil_pos + (
            ((values[high_mask] - self.trefoil_det) / high_denom) * (1.0 - self.trefoil_pos)
        )

        normalized = np.clip(normalized, 0.0, 1.0)
        masked = np.ma.array(normalized, mask=np.ma.getmask(result))
        if is_scalar:
            masked = masked[0]
        return masked

    def inverse(self, value):
        values = np.asarray(value, dtype=float)
        original = np.zeros_like(values, dtype=float)

        low_mask = values <= self.trefoil_pos
        low_denom = max(self.trefoil_pos, 1e-12)
        original[low_mask] = self.vmin + (values[low_mask] / low_denom) * (self.trefoil_det - self.vmin)

        high_mask = ~low_mask
        high_denom = max(1.0 - self.trefoil_pos, 1e-12)
        original[high_mask] = self.trefoil_det + (
            ((values[high_mask] - self.trefoil_pos) / high_denom) * (self.vmax - self.trefoil_det)
        )

        return original


def _add_determinant_colorbar(ax, max_det, has_complex):
    """Attach determinant colorbar to an axis with semantic tick labels."""
    display_max = float(max_det) if pd.notna(max_det) else float(TREFOIL_DETERMINANT)
    if not np.isfinite(display_max):
        display_max = float(TREFOIL_DETERMINANT)
    display_max = max(display_max, float(TREFOIL_DETERMINANT))
    norm = AnchoredDeterminantNorm(
        vmin=UNKNOT_DETERMINANT,
        vmax=display_max,
        trefoil_det=TREFOIL_DETERMINANT,
        trefoil_pos=TREFOIL_COLOR_POSITION
    )

    scalar_map = plt.cm.ScalarMappable(norm=norm, cmap=DETERMINANT_CMAP)
    scalar_map.set_array([])
    cbar = ax.figure.colorbar(scalar_map, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Determinant", fontsize=10)

    if has_complex and display_max > float(TREFOIL_DETERMINANT):
        complex_mid = TREFOIL_DETERMINANT + 0.5 * (display_max - TREFOIL_DETERMINANT)
        cbar.set_ticks([UNKNOT_DETERMINANT, TREFOIL_DETERMINANT, complex_mid])
        cbar.set_ticklabels(["0 Unknot", "3 Trefoil", "Complex"])
    else:
        cbar.set_ticks([UNKNOT_DETERMINANT, TREFOIL_DETERMINANT])
        cbar.set_ticklabels(["0 Unknot", "3 Trefoil"])
    cbar.ax.tick_params(labelsize=9)


def _smooth_percentage_matrix(pivot, smooth_window):
    """Smooth only the Unknotted boundary and preserve other group gradients."""
    if pivot.empty:
        return pivot

    if smooth_window is None or int(smooth_window) <= 1:
        return pivot

    if "Unknotted" not in pivot.columns:
        return pivot

    window = int(smooth_window)
    if window % 2 == 0:
        window += 1  # keep centered smoothing symmetric

    smoothed = pivot.copy()
    unknot_smoothed = pivot["Unknotted"].rolling(window=window, center=True, min_periods=1).mean()
    unknot_smoothed = unknot_smoothed.clip(lower=0.0, upper=100.0)
    smoothed["Unknotted"] = unknot_smoothed

    other_cols = [col for col in pivot.columns if col != "Unknotted"]
    if other_cols:
        other_total_orig = pivot[other_cols].sum(axis=1)
        other_total_target = 100.0 - unknot_smoothed
        scale = np.ones(len(pivot), dtype=float)
        valid = other_total_orig.to_numpy() > 1e-12
        scale[valid] = (other_total_target.to_numpy()[valid] / other_total_orig.to_numpy()[valid])
        smoothed.loc[:, other_cols] = pivot[other_cols].mul(scale, axis=0)
        if (~valid).any():
            smoothed.loc[~valid, other_cols] = 0.0
            smoothed.loc[~valid, "Unknotted"] = 100.0

    # Final numerical cleanup: force rows to sum to 100.
    row_sums = smoothed.sum(axis=1).replace(0.0, np.nan)
    smoothed = smoothed.div(row_sums, axis=0).fillna(0.0) * 100.0
    return smoothed


def _draw_stacked_area(ax, df_raw, percentages, title_str=None, x_start=None, smooth_window=DEFAULT_SMOOTH_WINDOW):
    """Draw a stacked area plot on the given axes.

    Shared logic for single and comparison plots.

    Parameters
    ----------
    x_start : int or None
        If given, only include lengths <= x_start.
    smooth_window : int
        Centered moving-average window in number of sampled lengths. Use <=1 to disable.
    """
    # Get unique lengths (sorted descending)
    all_lengths = sorted(percentages["length"].unique(), reverse=True)

    # Filter lengths by x_start, but keep one length above x_start so we
    # can interpolate the left edge of the plot to the requested value.
    x_left_override = None
    if x_start is not None:
        lengths = [l for l in all_lengths if l <= x_start]
        # Find the first length above x_start to include as context
        above = [l for l in all_lengths if l > x_start]
        if above:
            anchor = min(above)  # closest length above x_start
            lengths = [anchor] + lengths
            # Interpolate where x_start falls between anchor (pos 0) and
            # the next length (pos 1) using the descending length values.
            if len(lengths) >= 2:
                l_high = float(anchor)
                l_low = float(lengths[1])
                if l_high != l_low:
                    x_left_override = (l_high - x_start) / (l_high - l_low)
    else:
        lengths = all_lengths

    x = np.arange(len(lengths))

    # Map each group to representative determinant-like value
    group_meta = percentages[["group_key", "group_value"]].drop_duplicates(subset=["group_key"])
    group_value_map = dict(zip(group_meta["group_key"], group_meta["group_value"]))

    present_groups = set(percentages["group_key"].unique())
    complex_groups = sorted(
        [g for g in present_groups if g.startswith("Complex_")],
        key=lambda g: (
            np.inf if pd.isna(group_value_map.get(g, np.nan)) else float(group_value_map[g]),
            g
        )
    )

    complex_values = [
        float(group_value_map[g]) for g in complex_groups
        if pd.notna(group_value_map.get(g, np.nan))
    ]
    determinant_colors = create_determinant_colors(complex_values)

    plot_order = []
    if "Unknotted" in present_groups:
        plot_order.append(("Unknotted", determinant_colors[UNKNOT_DETERMINANT]))
    if "Trefoil" in present_groups:
        plot_order.append(("Trefoil", determinant_colors[TREFOIL_DETERMINANT]))
    for group_key in complex_groups:
        group_value = group_value_map.get(group_key, np.nan)
        if pd.isna(group_value):
            color = DETERMINANT_CMAP(1.0)
        else:
            color = determinant_colors.get(float(group_value), DETERMINANT_CMAP(1.0))
        plot_order.append((group_key, color))

    # Build a compact percentage matrix (length x group_key)
    pivot = (
        percentages[percentages["length"].isin(lengths)]
        .pivot_table(index="length", columns="group_key", values="percentage", aggfunc="first", fill_value=0.0)
        .reindex(lengths, fill_value=0.0)
    )
    pivot = _smooth_percentage_matrix(pivot, smooth_window)

    # Stack areas
    bottoms = np.zeros(len(x))

    has_complex = False

    for group_key, color in plot_order:
        if group_key in pivot.columns:
            y = pivot[group_key].to_numpy(dtype=float)
        else:
            y = np.zeros(len(x), dtype=float)

        if not np.all(y == 0):
            # Opaque fill pre-blended with the white background (same look as
            # alpha=0.9) plus a same-color edge, so adjacent bands leave no
            # anti-aliasing seams or grid lines showing through.
            color = tuple(0.9 * c + 0.1 for c in mcolors.to_rgb(color))
            if group_key.startswith("Complex_"):
                has_complex = True
                ax.fill_between(x, bottoms, bottoms + y,
                                facecolor=color, edgecolor=color, linewidth=0.5)
            else:
                ax.fill_between(x, bottoms, bottoms + y,
                                facecolor=color, edgecolor=color, linewidth=0.5)
            bottoms = bottoms + y

    # Add dotted lines at Arsuaga length (20) meeting at unknotted percentage
    if ARSUAGA_LENGTH in lengths:
        idx_arsuaga = lengths.index(ARSUAGA_LENGTH)
        if ("Unknotted" in pivot.columns) and (ARSUAGA_LENGTH in pivot.index):
            unknot_pct = float(pivot.loc[ARSUAGA_LENGTH, "Unknotted"])
            # Vertical line from bottom to unknotted level
            ax.plot([idx_arsuaga, idx_arsuaga], [0, unknot_pct],
                   color="black", linestyle=":", linewidth=1.5, zorder=10)
            # Horizontal line from left edge to the vertical line
            edge = x_left_override if x_left_override is not None else 0
            ax.plot([edge, idx_arsuaga], [unknot_pct, unknot_pct],
                   color="black", linestyle=":", linewidth=1.5, zorder=10)
            # Add percentage label at left edge
            ax.annotate(f"{int(round(unknot_pct))}%", xy=(edge - 0.5, unknot_pct),
                       fontsize=10, ha="right", va="center")

    ax.set_xlabel("Number of segments", fontsize=12)
    ax.set_ylabel("Knotting (%)", fontsize=12)

    if title_str:
        ax.set_title(title_str, fontsize=14, pad=10)

    # Set x-axis ticks
    xlim_left = x_left_override if x_left_override is not None else 0
    tick_spacing = max(1, len(x) // 15)
    tick_indices = list(range(0, len(x), tick_spacing))
    if len(x) - 1 not in tick_indices:
        tick_indices.append(len(x) - 1)

    # Drop ticks that fall outside the visible left edge
    tick_indices = [i for i in tick_indices if x[i] >= xlim_left]

    tick_positions = [x[i] for i in tick_indices]
    tick_labels = [str(lengths[i]) for i in tick_indices]

    # If x_start clipped the left edge, add a tick there labeled with x_start
    if x_left_override is not None:
        tick_positions.insert(0, x_left_override)
        tick_labels.insert(0, str(x_start))

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")
    ax.set_xlim(xlim_left, len(x) - 1)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.grid(False)

    # Replace legend with determinant colorbar
    complex_like_mask = (
        (~df_raw["knot_type"].isin(["Unknotted", "Trefoil"])) &
        (df_raw["error"].isna()) &
        (df_raw["determinant"].notna())
    )
    complex_max_series = df_raw.loc[complex_like_mask, "determinant"]
    max_det = float(complex_max_series.max()) if not complex_max_series.empty else float(TREFOIL_DETERMINANT)
    _add_determinant_colorbar(ax, max_det=max_det, has_complex=has_complex)


def plot_gradient_percentages(df_raw, percentages, output_dir, base_name, title=None, x_start=None,
                              smooth_window=DEFAULT_SMOOTH_WINDOW, rasterize=True):
    """Create stacked area plot with gradient coloring for complex knots."""
    print("[INFO] Generating gradient plot...")

    # Count number of trajectories
    num_trajectories = df_raw["filename"].nunique()

    # Set seaborn style
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.2)

    fig, ax = plt.subplots(figsize=(12, 6))

    title_str = f"{title} ({num_trajectories} trajectories)" if title else None
    _draw_stacked_area(
        ax, df_raw, percentages, title_str=title_str, x_start=x_start, smooth_window=smooth_window
    )

    plt.tight_layout()

    # Save plot
    output_file = os.path.join(output_dir, f"{base_name}_plot.png")
    plt.savefig(output_file, dpi=150)
    print(f"[INFO] Plot saved to {output_file}")

    output_pdf = os.path.join(output_dir, f"{base_name}_plot.pdf")
    if rasterize:
        ax.set_rasterized(True)
    plt.savefig(output_pdf, dpi=150)
    print(f"[INFO] PDF saved to {output_pdf}")

    plt.close()


def plot_gradient_percentages_comparison(df_raw1, percentages1, df_raw2, percentages2,
                                          pop_labels, output_dir, base_name, title=None, x_start=None,
                                          smooth_window=DEFAULT_SMOOTH_WINDOW, rasterize=True):
    """Create side-by-side stacked area plots comparing two populations."""
    print("[INFO] Generating comparison gradient plot...")

    num_traj1 = df_raw1["filename"].nunique()
    num_traj2 = df_raw2["filename"].nunique()

    # Set seaborn style
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6), sharey=True)

    _draw_stacked_area(
        ax1, df_raw1, percentages1,
        title_str=f"{pop_labels[0]} ({num_traj1} trajectories)", x_start=x_start, smooth_window=smooth_window
    )
    _draw_stacked_area(
        ax2, df_raw2, percentages2,
        title_str=f"{pop_labels[1]} ({num_traj2} trajectories)", x_start=x_start, smooth_window=smooth_window
    )

    # Only show y-label on left subplot
    ax2.set_ylabel("")

    if title:
        fig.suptitle(title, fontsize=16, y=1.02)

    plt.tight_layout()

    # Save plot
    output_file = os.path.join(output_dir, f"{base_name}_plot.png")
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"[INFO] Plot saved to {output_file}")

    output_pdf = os.path.join(output_dir, f"{base_name}_plot.pdf")
    if rasterize:
        ax1.set_rasterized(True)
        ax2.set_rasterized(True)
    plt.savefig(output_pdf, dpi=150, bbox_inches="tight")
    print(f"[INFO] PDF saved to {output_pdf}")

    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot random sampling knot type analysis with gradient")
    parser.add_argument("csv_file", help="Path to CSV file with analysis results")
    parser.add_argument("csv_file2", nargs="?", default=None,
                        help="Path to second CSV file for comparison (optional)")
    parser.add_argument("--labels", nargs=2, default=["Population 1", "Population 2"],
                        help="Labels for comparison subplots (e.g., --labels Knotted Unknotted)")
    parser.add_argument("--title", "-t", help="Title for the plot (e.g., 'Knotted Population')", default=None)
    parser.add_argument("--output-dir", "-o", help="Output directory (default: same as input)", default=None)
    parser.add_argument(
        "--x-start",
        type=int,
        default=None,
        help="Starting length for x-axis (default: max length in CSV). Plot shows lengths from x-start down to the minimum."
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=DEFAULT_SMOOTH_WINDOW,
        help="Centered moving-average window for the Unknotted boundary (default: 5, use 1 to disable)."
    )
    parser.add_argument(
        "--no_rasterize",
        dest="rasterize",
        action="store_false",
        help="Disable PDF rasterization (default: rasterized)."
    )

    args = parser.parse_args()

    input_file = os.path.expanduser(args.csv_file)
    input_file2 = os.path.expanduser(args.csv_file2) if args.csv_file2 else None
    output_dir = os.path.expanduser(args.output_dir) if args.output_dir else os.path.dirname(input_file)
    if not output_dir:
        output_dir = "."
    base_name = os.path.splitext(os.path.basename(input_file))[0]

    print("=" * 60)
    print("Random Sampling Knot Type Plotting (Gradient)")
    print(f"Input: {input_file}")
    if input_file2:
        print(f"Input 2: {input_file2}")
        print(f"Labels: {args.labels}")
    if args.title:
        print(f"Title: {args.title}")
    print(f"Smoothing window: {args.smooth_window}")
    print("=" * 60)

    # Load data
    df1 = load_data(input_file)

    # Compute percentages preserving determinant info
    percentages1, df_raw1 = compute_determinant_percentages(df1)

    if input_file2:
        df2 = load_data(input_file2)
        percentages2, df_raw2 = compute_determinant_percentages(df2)

        comp_base_name = "random_sample_comparison"
        plot_gradient_percentages_comparison(df_raw1, percentages1, df_raw2, percentages2,
                                             args.labels, output_dir, comp_base_name,
                                             title=args.title, x_start=args.x_start,
                                             smooth_window=args.smooth_window,
                                             rasterize=args.rasterize)
    else:
        # Generate gradient plot
        plot_gradient_percentages(df_raw1, percentages1, output_dir, base_name,
                                 title=args.title, x_start=args.x_start,
                                 smooth_window=args.smooth_window,
                                 rasterize=args.rasterize)

    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
