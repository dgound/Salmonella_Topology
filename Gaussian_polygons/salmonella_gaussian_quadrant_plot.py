#!/usr/bin/env python3
"""
Threshold + length sweep quadrant plot for the Gaussian random polygon
experiment.

Reads the per-polygon stats CSV produced by salmonella_gaussian_quadrant_compute.py
(which has one mean-crossing column per downsampling length L) and, for each
length, runs a threshold sweep:

  - on the full-chain axis: keep top p% and bottom p% polygons.
  - on the L-gon-score axis: same.
  - restrict to polygons that are in an extreme group on BOTH axes ("kept set").
  - rows = full-chain top/bottom, cols = L-gon top/bottom.
  - cell = 2 * (count / kept) so both rows and columns sum to 100% in the
    balanced case.

Per length L we save:
  - <base>_L<L>_sweep.{png,pdf}  : grid of 2x2 panels, one per threshold
  - <base>_L<L>_sweep_summary.csv

Plus one overall summary CSV (<base>_summary.csv) and a convergence line
plot (<base>_convergence.{png,pdf}) showing the smallest p% at which the
matrix reaches 100/0/0/100 ("perfect classification"), for each L.

Usage:
    python salmonella_gaussian_quadrant_plot.py <stats_csv>
        [--thresholds 50,45,...,1]
        [--lengths 100,75,50,25,20,15,10,6]  (default: all in CSV)
        [-o OUTDIR]
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DIAG_COLOR = "#a6d49f"
OFFDIAG_COLOR = "#e8b8c4"
DEFAULT_THRESHOLDS = "50,45,40,35,30,25,20,15,10,5,2.5,1"


def load_stats(path):
    df = pd.read_csv(path)
    df = df[df["error"].isna()].copy()
    df["crossing_full"] = df["crossing_full"].astype(float)
    return df


def discover_lengths(df):
    """Find the lengths L for which mean_crossing_L<L> columns exist."""
    lengths = []
    prefix = "mean_crossing_L"
    for col in df.columns:
        if col.startswith(prefix):
            try:
                lengths.append(int(col[len(prefix):]))
            except ValueError:
                continue
    return sorted(lengths, reverse=True)


def threshold_matrix(values_y, values_x, p_percent, seed=42):
    """For threshold p% return (counts 2x2, kept_n, n_each).

    Rows = full-chain (0=bottom, 1=top), cols = L-gon (0=bottom, 1=top).
    """
    n = len(values_y)
    q = p_percent / 100.0
    if q <= 0 or q > 0.5:
        raise ValueError(f"threshold percent {p_percent} must be in (0, 50]")
    n_each = max(1, int(round(q * n)))

    rng = np.random.default_rng(seed)
    rank_y = np.argsort(np.argsort(values_y + rng.random(n) * 1e-12))
    rank_x = np.argsort(np.argsort(values_x + rng.random(n) * 1e-12))

    y_top = rank_y >= n - n_each
    y_bot = rank_y < n_each
    x_top = rank_x >= n - n_each
    x_bot = rank_x < n_each

    keep = (y_top | y_bot) & (x_top | x_bot)

    counts = np.zeros((2, 2), dtype=np.int64)
    counts[0, 0] = int(np.sum(keep & y_bot & x_bot))
    counts[0, 1] = int(np.sum(keep & y_bot & x_top))
    counts[1, 0] = int(np.sum(keep & y_top & x_bot))
    counts[1, 1] = int(np.sum(keep & y_top & x_top))
    return counts, int(np.sum(keep)), n_each


def draw_panel(ax, counts, kept_n, p_percent, n_each, show_xlabels, show_ylabels):
    matrix = 2.0 * 100.0 * counts / kept_n if kept_n > 0 else np.zeros_like(counts, dtype=float)
    cell_colors = np.array([[DIAG_COLOR, OFFDIAG_COLOR],
                            [OFFDIAG_COLOR, DIAG_COLOR]])
    for i in range(2):
        for j in range(2):
            ax.add_patch(plt.Rectangle((j - 0.5, 1 - i - 0.5), 1, 1,
                                        facecolor=cell_colors[i, j],
                                        edgecolor="black", linewidth=1.0))
            ax.text(j, 1 - i, f"{matrix[i, j]:.1f}%",
                    ha="center", va="center",
                    fontsize=14, fontweight="bold", color="black")
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)
    ax.set_aspect("equal")
    if show_xlabels:
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Under", "Over"], fontsize=9)
    else:
        ax.set_xticks([])
    if show_ylabels:
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Over", "Under"], fontsize=9, rotation=90, va="center")
    else:
        ax.set_yticks([])
    ax.tick_params(axis="x", length=0, pad=3)
    ax.tick_params(axis="y", length=0, pad=3)
    ax.set_title(f"p = {p_percent:g}%  (kept={kept_n})", fontsize=10, pad=4)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)


def plot_length_sweep(df, length, thresholds, output_dir, base_name, title):
    score_col = f"mean_crossing_L{length}"
    sub = df.dropna(subset=[score_col]).copy()
    y = sub["crossing_full"].to_numpy()
    x = sub[score_col].to_numpy()

    rows = []
    for p in thresholds:
        counts, kept, n_each = threshold_matrix(y, x, p)
        m = 2.0 * 100.0 * counts / kept if kept > 0 else np.zeros_like(counts, dtype=float)
        rows.append({
            "length": length,
            "threshold_percent": p,
            "n_each": n_each,
            "kept": kept,
            "bot_bot": int(counts[0, 0]),
            "bot_top": int(counts[0, 1]),
            "top_bot": int(counts[1, 0]),
            "top_top": int(counts[1, 1]),
            "pct_bot_bot": m[0, 0],
            "pct_bot_top": m[0, 1],
            "pct_top_bot": m[1, 0],
            "pct_top_top": m[1, 1],
        })
    summary_df = pd.DataFrame(rows)

    summary_path = os.path.join(output_dir, f"{base_name}_L{length}_sweep_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"  [L={length}] summary -> {summary_path}")

    n_panels = len(thresholds)
    n_cols = min(4, n_panels)
    n_rows = int(np.ceil(n_panels / n_cols))

    sns.set_style("white")
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(3.0 * n_cols, 3.0 * n_rows + 0.4),
                              squeeze=False)
    for idx, (p, row) in enumerate(zip(thresholds, rows)):
        r, c = divmod(idx, n_cols)
        ax = axes[r][c]
        counts = np.array([[row["bot_bot"], row["bot_top"]],
                           [row["top_bot"], row["top_top"]]], dtype=np.int64)
        draw_panel(ax, counts, row["kept"], p, row["n_each"],
                   show_xlabels=(r == n_rows - 1), show_ylabels=(c == 0))
    for idx in range(n_panels, n_rows * n_cols):
        r, c = divmod(idx, n_cols)
        axes[r][c].axis("off")

    fig.supxlabel(f"{length}-gon score class", fontsize=13, fontweight="bold")
    fig.supylabel("Full-chain crossing class", fontsize=13, fontweight="bold")
    panel_title = f"{title} — L = {length}" if title else f"L = {length}"
    fig.suptitle(panel_title, fontsize=14, y=0.995)

    plt.tight_layout(rect=(0.02, 0.02, 1, 0.97))
    png_path = os.path.join(output_dir, f"{base_name}_L{length}_sweep.png")
    pdf_path = os.path.join(output_dir, f"{base_name}_L{length}_sweep.pdf")
    plt.savefig(png_path, dpi=150)
    plt.savefig(pdf_path, dpi=150)
    plt.close()
    print(f"  [L={length}] figure  -> {png_path}")

    return summary_df


def smallest_perfect_threshold(summary_df):
    """Smallest p% where the matrix is exactly 100/0/0/100, else NaN."""
    eps = 1e-9
    perfect = summary_df[
        (summary_df["pct_bot_top"] < eps) &
        (summary_df["pct_top_bot"] < eps) &
        (summary_df["pct_bot_bot"] > 100 - eps) &
        (summary_df["pct_top_top"] > 100 - eps)
    ]
    if perfect.empty:
        return None
    return float(perfect["threshold_percent"].min())


def plot_convergence(combined, lengths, output_dir, base_name, title):
    points = []
    for L in lengths:
        sub = combined[combined["length"] == L]
        p = smallest_perfect_threshold(sub)
        points.append((L, p))

    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.2)
    fig, ax = plt.subplots(figsize=(8, 5))

    L_arr = np.array([L for L, _ in points])
    p_arr = np.array([p for _, p in points], dtype=float)
    finite = ~np.isnan(p_arr)
    ax.plot(L_arr[finite], p_arr[finite], marker="o", linewidth=2,
            color="#1f77b4")
    for L, p in points:
        if p is None:
            ax.scatter([L], [0.5], marker="x", s=60, color="red")
            ax.annotate("no perfect p\nin this sweep",
                        xy=(L, 0.5), xytext=(0, -18), textcoords="offset points",
                        ha="center", fontsize=8, color="red")
        else:
            ax.annotate(f"{p:g}%", xy=(L, p), xytext=(4, 4),
                        textcoords="offset points", fontsize=9)

    ax.set_xlabel("Downsampling length L", fontsize=12)
    ax.set_ylabel("Smallest threshold p% giving 100/0 classification", fontsize=12)
    ax.set_xscale("log")
    ax.set_xticks(L_arr)
    ax.set_xticklabels([str(L) for L in L_arr])
    ax.invert_xaxis()
    if title:
        ax.set_title(title, fontsize=13, pad=8)
    plt.tight_layout()
    png_path = os.path.join(output_dir, f"{base_name}_convergence.png")
    pdf_path = os.path.join(output_dir, f"{base_name}_convergence.pdf")
    plt.savefig(png_path, dpi=150)
    plt.savefig(pdf_path, dpi=150)
    plt.close()
    print(f"\n[INFO] Convergence summary saved to {png_path}")


def main():
    parser = argparse.ArgumentParser(description="Length + threshold sweep quadrant plot")
    parser.add_argument("stats_csv")
    parser.add_argument("--output-dir", "-o", default=None)
    parser.add_argument("--base-name", default=None)
    parser.add_argument("--thresholds", default=DEFAULT_THRESHOLDS,
                        help=f"Comma-separated threshold percents (default: {DEFAULT_THRESHOLDS})")
    parser.add_argument("--lengths", default=None,
                        help="Comma-separated lengths to plot (default: all in CSV)")
    parser.add_argument("--title", default=None)
    args = parser.parse_args()

    input_path = os.path.expanduser(args.stats_csv)
    output_dir = os.path.expanduser(args.output_dir) if args.output_dir else os.path.dirname(input_path)
    if not output_dir:
        output_dir = "."
    base_name = args.base_name or os.path.splitext(os.path.basename(input_path))[0]

    thresholds = [float(x) for x in args.thresholds.split(",") if x.strip()]
    for p in thresholds:
        if not (0 < p <= 50):
            raise SystemExit(f"Threshold {p} out of range (0, 50]")

    df = load_stats(input_path)
    print(f"[INFO] Loaded {len(df)} valid polygons from {input_path}")

    if args.lengths:
        lengths = sorted({int(s) for s in args.lengths.split(",") if s.strip()}, reverse=True)
    else:
        lengths = discover_lengths(df)
    if not lengths:
        raise SystemExit("No mean_crossing_L<L> columns found in CSV")
    print(f"[INFO] Lengths: {lengths}")
    print(f"[INFO] Thresholds: {thresholds}")

    combined = []
    for L in lengths:
        print(f"\n[length sweep] L = {L}")
        summary_df = plot_length_sweep(df, L, thresholds, output_dir, base_name, args.title)
        combined.append(summary_df)
    combined_df = pd.concat(combined, ignore_index=True)
    summary_path = os.path.join(output_dir, f"{base_name}_summary.csv")
    combined_df.to_csv(summary_path, index=False)
    print(f"\n[INFO] Combined summary saved to {summary_path}")

    plot_convergence(combined_df, lengths, output_dir, base_name, args.title)


if __name__ == "__main__":
    main()
