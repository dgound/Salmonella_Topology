#!/usr/bin/env python3
"""
Plot delta between two populations' knot-type distributions vs length.

The delta is computed as pop1 - pop2 (typically Knotted - Unknotted).

Two modes:
- stacked: signed stacked-area showing per-group delta, with the same
  determinant-gradient colorscale (and colorbar) used by
  salmonella_plot_random_20gon.py. Positive deltas (group enriched in pop1)
  are stacked above zero, negative deltas (group enriched in pop2) below.
- single: a single curve showing the difference in overall knotting
  probability, i.e. (100 - %Unknotted) between the two populations.

Usage:
    python salmonella_plot_delta.py <csv1> <csv2> --labels Knotted Unknotted
    python salmonella_plot_delta.py <csv1> <csv2> --mode single --labels Knotted Unknotted
"""

import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from salmonella_plot_random_20gon import (
    load_data,
    create_determinant_colors,
    DETERMINANT_CMAP,
    UNKNOT_DETERMINANT,
    TREFOIL_DETERMINANT,
    MAX_COMPLEX_BANDS,
    DEFAULT_SMOOTH_WINDOW,
    ARSUAGA_LENGTH,
    _add_determinant_colorbar,
    _smooth_percentage_matrix,
)


def _coarse_count_parts(valid):
    """Build the Unknotted and Trefoil count tables for one population."""
    parts = []
    for group_name, group_value in [
        ("Unknotted", float(UNKNOT_DETERMINANT)),
        ("Trefoil", float(TREFOIL_DETERMINANT)),
    ]:
        sub = valid[valid["knot_type"] == group_name]
        if not sub.empty:
            grouped = (
                sub.groupby("length", as_index=False).size()
                .rename(columns={"size": "count"})
            )
            grouped["group_key"] = group_name
            grouped["group_value"] = group_value
            parts.append(grouped)
    return parts


def compute_paired_determinant_percentages(df1, df2):
    """
    Compute percentages per population using unified complex-determinant bins.

    Complex determinants from BOTH populations are pooled to define log-space
    bin edges so the two pivots share the same group_keys and colors.
    """
    valid1 = df1[df1["error"].isna()].copy()
    valid2 = df2[df2["error"].isna()].copy()

    parts1 = _coarse_count_parts(valid1)
    parts2 = _coarse_count_parts(valid2)

    is_complex1 = ~valid1["knot_type"].isin(["Unknotted", "Trefoil"])
    is_complex2 = ~valid2["knot_type"].isin(["Unknotted", "Trefoil"])
    complex1 = valid1[is_complex1 & valid1["determinant"].notna()].copy()
    complex2 = valid2[is_complex2 & valid2["determinant"].notna()].copy()

    pooled = np.concatenate([
        complex1["determinant"].to_numpy(dtype=float),
        complex2["determinant"].to_numpy(dtype=float),
    ]) if (len(complex1) + len(complex2)) > 0 else np.array([], dtype=float)

    if pooled.size > 0:
        log_det = np.log10(np.clip(pooled, 1e-12, None))
        unique_log_count = int(np.unique(log_det).size)
        n_bands = max(1, min(MAX_COMPLEX_BANDS, unique_log_count))

        if n_bands == 1:
            log_edges = None
            representative_values = np.array(
                [float(np.power(10.0, log_det.mean()))], dtype=float
            )
        else:
            log_edges = np.linspace(log_det.min(), log_det.max(), n_bands + 1)
            representative_logs = 0.5 * (log_edges[:-1] + log_edges[1:])
            representative_values = np.power(10.0, representative_logs)

        def _bin_counts(cdf, parts):
            if cdf.empty:
                return
            det_c = np.clip(cdf["determinant"].to_numpy(dtype=float), 1e-12, None)
            log_c = np.log10(det_c)
            if log_edges is None:
                bin_idx = np.zeros_like(log_c, dtype=int)
            else:
                bin_idx = np.searchsorted(log_edges[1:-1], log_c, side="right")
            sub = pd.DataFrame({"length": cdf["length"].to_numpy(), "band_idx": bin_idx})
            grouped = (
                sub.groupby(["length", "band_idx"], as_index=False).size()
                .rename(columns={"size": "count"})
            )
            grouped["group_key"] = grouped["band_idx"].map(
                lambda idx: f"Complex_{int(idx):03d}"
            )
            grouped["group_value"] = grouped["band_idx"].map(
                lambda idx: float(representative_values[int(idx)])
            )
            grouped = grouped.drop(columns=["band_idx"])
            parts.append(grouped)

        _bin_counts(complex1, parts1)
        _bin_counts(complex2, parts2)

    def _finalize(parts):
        if not parts:
            return pd.DataFrame(
                columns=["length", "count", "group_key", "group_value", "total", "percentage"]
            )
        combined = pd.concat(parts, ignore_index=True)
        total_per_length = (
            combined.groupby("length", as_index=False)["count"].sum()
            .rename(columns={"count": "total"})
        )
        merged = combined.merge(total_per_length, on="length", how="left")
        merged["percentage"] = merged["count"] / merged["total"] * 100
        return merged

    return _finalize(parts1), _finalize(parts2)


def filter_lengths(all_lengths, x_start):
    """Apply optional x_start clipping, mirroring salmonella_plot_random_20gon."""
    if x_start is None:
        return all_lengths, None

    lengths = [l for l in all_lengths if l <= x_start]
    above = [l for l in all_lengths if l > x_start]
    x_left_override = None
    if above:
        anchor = min(above)
        lengths = [anchor] + lengths
        if len(lengths) >= 2:
            l_high = float(anchor)
            l_low = float(lengths[1])
            if l_high != l_low:
                x_left_override = (l_high - x_start) / (l_high - l_low)
    return lengths, x_left_override


def configure_x_axis(ax, x, lengths, x_left_override, x_start):
    """Set ticks/xlim consistent with the existing downsampling plot."""
    xlim_left = x_left_override if x_left_override is not None else 0
    tick_spacing = max(1, len(x) // 15)
    tick_indices = list(range(0, len(x), tick_spacing))
    if len(x) - 1 not in tick_indices:
        tick_indices.append(len(x) - 1)
    tick_indices = [i for i in tick_indices if x[i] >= xlim_left]

    tick_positions = [x[i] for i in tick_indices]
    tick_labels = [str(lengths[i]) for i in tick_indices]
    if x_left_override is not None:
        tick_positions.insert(0, x_left_override)
        tick_labels.insert(0, str(x_start))

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")
    ax.set_xlim(xlim_left, len(x) - 1)


def _build_aligned_pivots(pct1, pct2, lengths, smooth_window):
    """Build column-aligned pivots for both populations, optionally smoothed."""
    keys = sorted(set(pct1["group_key"]).union(pct2["group_key"]))

    def _pivot(pct):
        p = (
            pct[pct["length"].isin(lengths)]
            .pivot_table(
                index="length", columns="group_key", values="percentage",
                aggfunc="first", fill_value=0.0,
            )
            .reindex(lengths, fill_value=0.0)
        )
        for k in keys:
            if k not in p.columns:
                p[k] = 0.0
        return p[keys]

    pivot1 = _smooth_percentage_matrix(_pivot(pct1), smooth_window)
    pivot2 = _smooth_percentage_matrix(_pivot(pct2), smooth_window)
    return pivot1, pivot2, keys


def plot_stacked_delta(pct1, pct2, pop_labels, lengths, x_left_override, x_start,
                       output_dir, base_name, smooth_window, title=None, absolute=False):
    """Signed (or absolute) stacked-area delta, colored by determinant gradient."""
    pivot1, pivot2, keys = _build_aligned_pivots(pct1, pct2, lengths, smooth_window)

    group_meta = pd.concat([
        pct1[["group_key", "group_value"]],
        pct2[["group_key", "group_value"]],
    ], ignore_index=True).drop_duplicates(subset=["group_key"])
    group_value_map = dict(zip(group_meta["group_key"], group_meta["group_value"]))

    complex_groups = sorted(
        [g for g in keys if g.startswith("Complex_")],
        key=lambda g: (
            np.inf if pd.isna(group_value_map.get(g, np.nan)) else float(group_value_map[g]),
            g,
        ),
    )
    complex_values = [
        float(group_value_map[g]) for g in complex_groups
        if pd.notna(group_value_map.get(g, np.nan))
    ]
    determinant_colors = create_determinant_colors(complex_values)

    plot_order = []
    if "Unknotted" in keys:
        plot_order.append(("Unknotted", determinant_colors[UNKNOT_DETERMINANT]))
    if "Trefoil" in keys:
        plot_order.append(("Trefoil", determinant_colors[TREFOIL_DETERMINANT]))
    for g in complex_groups:
        gv = group_value_map.get(g, np.nan)
        if pd.isna(gv):
            color = DETERMINANT_CMAP(1.0)
        else:
            color = determinant_colors.get(float(gv), DETERMINANT_CMAP(1.0))
        plot_order.append((g, color))

    delta = pivot1 - pivot2

    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.2)
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(lengths))
    has_complex = False

    if absolute:
        stack = np.zeros(len(x))
        for group_key, color in plot_order:
            if group_key not in delta.columns:
                continue
            y = np.abs(delta[group_key].to_numpy(dtype=float))
            if group_key.startswith("Complex_"):
                has_complex = True
            if np.any(y > 0):
                ax.fill_between(x, stack, stack + y, color=color, alpha=0.95)
                stack = stack + y
        ax.axhline(0, color="black", linewidth=1.0, zorder=5)
        ax.set_ylim(0, 200)
        ax.set_yticks([0, 50, 100, 150, 200])
        ax.set_ylabel(f"|Δ Knotting| (%) [{pop_labels[0]} vs {pop_labels[1]}]", fontsize=12)
    else:
        pos_stack = np.zeros(len(x))
        neg_stack = np.zeros(len(x))
        for group_key, color in plot_order:
            if group_key not in delta.columns:
                continue
            y = delta[group_key].to_numpy(dtype=float)
            pos = np.where(y > 0, y, 0.0)
            neg = np.where(y < 0, y, 0.0)
            if group_key.startswith("Complex_"):
                has_complex = True
            if np.any(pos > 0):
                ax.fill_between(x, pos_stack, pos_stack + pos, color=color, alpha=0.95)
                pos_stack = pos_stack + pos
            if np.any(neg < 0):
                ax.fill_between(x, neg_stack, neg_stack + neg, color=color, alpha=0.95)
                neg_stack = neg_stack + neg
        ax.axhline(0, color="black", linewidth=1.0, zorder=5)
        ax.set_ylim(-100, 100)
        ax.set_yticks([-100, -50, 0, 50, 100])
        ax.set_ylabel(f"Δ Knotting (%) [{pop_labels[0]} − {pop_labels[1]}]", fontsize=12)

    ax.set_xlabel("Number of segments", fontsize=12)
    if title:
        ax.set_title(title, fontsize=14, pad=10)

    configure_x_axis(ax, x, lengths, x_left_override, x_start)
    ax.grid(False)

    complex_max_series = group_meta[group_meta["group_key"].str.startswith("Complex_")]["group_value"]
    max_det = float(complex_max_series.max()) if not complex_max_series.empty else float(TREFOIL_DETERMINANT)
    _add_determinant_colorbar(ax, max_det=max_det, has_complex=has_complex)

    plt.tight_layout()

    suffix = "_abs" if absolute else ""
    png_path = os.path.join(output_dir, f"{base_name}_delta_stacked{suffix}.png")
    pdf_path = os.path.join(output_dir, f"{base_name}_delta_stacked{suffix}.pdf")
    plt.savefig(png_path, dpi=150)
    ax.set_rasterized(True)
    plt.savefig(pdf_path, dpi=150)
    plt.close()
    print(f"[INFO] Stacked delta plot saved to {png_path}")
    print(f"[INFO] PDF saved to {pdf_path}")


def plot_single_curve(pct1, pct2, pop_labels, lengths, x_left_override, x_start,
                      output_dir, base_name, smooth_window, title=None, absolute=False):
    """Single curve: difference in overall knotting probability (100 − %Unknotted)."""
    pivot1, pivot2, _ = _build_aligned_pivots(pct1, pct2, lengths, smooth_window)

    unknot1 = pivot1["Unknotted"].to_numpy(dtype=float) if "Unknotted" in pivot1.columns else np.zeros(len(lengths))
    unknot2 = pivot2["Unknotted"].to_numpy(dtype=float) if "Unknotted" in pivot2.columns else np.zeros(len(lengths))
    delta = (100.0 - unknot1) - (100.0 - unknot2)
    if absolute:
        delta = np.abs(delta)

    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.2)
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(lengths))
    if absolute:
        ax.fill_between(x, 0, delta, color="#1f77b4", alpha=0.4)
        ax.plot(x, delta, color="black", linewidth=1.8)
        ax.axhline(0, color="black", linewidth=1.0, zorder=5)
        ax.set_ylim(0, 100)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_ylabel(f"|Δ P(knotted)| (%) [{pop_labels[0]} vs {pop_labels[1]}]", fontsize=12)
    else:
        ax.fill_between(x, 0, delta, where=delta >= 0, color="#1f77b4", alpha=0.4, interpolate=True)
        ax.fill_between(x, 0, delta, where=delta < 0, color="#d62728", alpha=0.4, interpolate=True)
        ax.plot(x, delta, color="black", linewidth=1.8)
        ax.axhline(0, color="black", linewidth=1.0, zorder=5)
        ax.set_ylim(-100, 100)
        ax.set_yticks([-100, -50, 0, 50, 100])
        ax.set_ylabel(f"Δ P(knotted) (%) [{pop_labels[0]} − {pop_labels[1]}]", fontsize=12)

    ax.set_xlabel("Number of segments", fontsize=12)
    if title:
        ax.set_title(title, fontsize=14, pad=10)

    configure_x_axis(ax, x, lengths, x_left_override, x_start)
    ax.grid(False)

    plt.tight_layout()

    suffix = "_abs" if absolute else ""
    png_path = os.path.join(output_dir, f"{base_name}_delta_single{suffix}.png")
    pdf_path = os.path.join(output_dir, f"{base_name}_delta_single{suffix}.pdf")
    plt.savefig(png_path, dpi=150)
    ax.set_rasterized(True)
    plt.savefig(pdf_path, dpi=150)
    plt.close()
    print(f"[INFO] Single-curve delta plot saved to {png_path}")
    print(f"[INFO] PDF saved to {pdf_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot delta between two knot-type distributions (pop1 − pop2)")
    parser.add_argument("csv1", help="First CSV (minuend, typically Knotted)")
    parser.add_argument("csv2", help="Second CSV (subtrahend, typically Unknotted)")
    parser.add_argument("--labels", nargs=2, default=["Population 1", "Population 2"],
                        help="Labels for the two populations")
    parser.add_argument("--mode", choices=["stacked", "single", "both"], default="both",
                        help="Plot mode (default: both)")
    parser.add_argument("--title", default=None, help="Optional title")
    parser.add_argument("--output-dir", "-o", default=None,
                        help="Output directory (default: same as csv1)")
    parser.add_argument("--base-name", default=None,
                        help="Output base filename (default: derived from csv1)")
    parser.add_argument("--x-start", type=int, default=None,
                        help="Starting length for x-axis")
    parser.add_argument("--smooth-window", type=int, default=DEFAULT_SMOOTH_WINDOW,
                        help=f"Centered moving-average window for the Unknotted boundary "
                             f"(default: {DEFAULT_SMOOTH_WINDOW}, use 1 to disable)")
    parser.add_argument("--abs", dest="absolute", action="store_true",
                        help="Plot the absolute value of the delta instead of the signed delta")
    args = parser.parse_args()

    csv1 = os.path.expanduser(args.csv1)
    csv2 = os.path.expanduser(args.csv2)
    output_dir = os.path.expanduser(args.output_dir) if args.output_dir else os.path.dirname(csv1)
    if not output_dir:
        output_dir = "."
    base_name = args.base_name or os.path.splitext(os.path.basename(csv1))[0]

    print("=" * 60)
    print("Knot-Type Delta Plotting")
    print(f"CSV 1 ({args.labels[0]}): {csv1}")
    print(f"CSV 2 ({args.labels[1]}): {csv2}")
    print(f"Mode: {args.mode}")
    print(f"Smoothing window: {args.smooth_window}")
    print("=" * 60)

    df1 = load_data(csv1)
    df2 = load_data(csv2)

    pct1, pct2 = compute_paired_determinant_percentages(df1, df2)

    all_lengths = sorted(set(pct1["length"]).union(pct2["length"]), reverse=True)
    lengths, x_left_override = filter_lengths(all_lengths, args.x_start)

    if args.mode in ("stacked", "both"):
        plot_stacked_delta(pct1, pct2, args.labels, lengths, x_left_override,
                           args.x_start, output_dir, base_name,
                           smooth_window=args.smooth_window, title=args.title,
                           absolute=args.absolute)
    if args.mode in ("single", "both"):
        plot_single_curve(pct1, pct2, args.labels, lengths, x_left_override,
                          args.x_start, output_dir, base_name,
                          smooth_window=args.smooth_window, title=args.title,
                          absolute=args.absolute)

    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
