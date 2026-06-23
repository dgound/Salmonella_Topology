#!/usr/bin/env python3
"""
Plot the (un|ov) x (u|k) matrix and per-population crossing-count histogram
from the per-sample parquet file produced by salmonella_quadrant_compute.py.

For each object the 20-gon score is the arithmetic MEAN of its 10,000 deWitt
20-gon crossing counts. The 2000 scores are rank-split at their median with
random tie-breaking into "un" (underknotted) and "ov" (overknotted), 1000
each. The 2x2 matrix shows 2 * fraction of all objects per cell, so both
rows and columns sum to 100%.

The histogram shows raw per-sample crossing counts per source population
(log y-axis).

Usage:
    python salmonella_quadrant_plot.py <samples_parquet> [-o OUTDIR] [--title TITLE]
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

POP_COLORS = {"unknotted": "#2ca02c", "knotted": "#d62728"}
POP_LABELS = {"unknotted": "Unknotted", "knotted": "Knotted"}
ROW_ORDER = ["unknotted", "knotted"]
DIAG_COLOR = "#a6d49f"
OFFDIAG_COLOR = "#e8b8c4"


def load_samples(path):
    df = pd.read_parquet(path)
    df["population"] = df["population"].astype("category")
    df["crossing_count"] = df["crossing_count"].astype(np.int16)
    return df


def per_object_mean(df):
    """Return per-object dataframe with columns [filename, population, score]."""
    grouped = (
        df.groupby(["population", "filename"], observed=True)["crossing_count"]
          .mean()
          .reset_index(name="score")
    )
    return grouped


def split_at_median(scores, seed=42):
    """Rank-split scores into bottom half (un) and top half (ov) with random
    tie-breaking so |un| == |ov| == len(scores) // 2 exactly. Returns a
    boolean array where True == overknotted."""
    n = len(scores)
    half = n // 2
    rng = np.random.default_rng(seed)
    jitter = rng.random(n)
    order = np.lexsort([jitter, scores])
    is_ov = np.zeros(n, dtype=bool)
    is_ov[order[half:]] = True
    return is_ov


def plot_scatter(per_obj, output_dir, base_name, title):
    """Per-object scatter in raw units.

    Median lines for both axes split the plane into four quadrants.
    """
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.2)
    fig, ax = plt.subplots(figsize=(8, 7))

    x = per_obj["score"].to_numpy(dtype=float)
    y = per_obj["crossing_10k"].to_numpy(dtype=float)
    x_med = float(np.median(x))
    y_med = float(np.median(y))

    populations = per_obj["population"].to_numpy()
    for pop in ROW_ORDER:
        mask = (populations == pop)
        ax.scatter(x[mask], y[mask], s=14, alpha=0.55,
                   color=POP_COLORS[pop], label=POP_LABELS[pop], edgecolors="none")

    ax.axvline(x_med, color="black", linestyle="--", linewidth=1.4, zorder=5)
    ax.axhline(y_med, color="black", linestyle="--", linewidth=1.4, zorder=5)
    ax.annotate(f"median (20-gon) = {x_med:.3f}",
                xy=(x_med, 1.0), xycoords=("data", "axes fraction"),
                xytext=(4, -4), textcoords="offset points",
                fontsize=9, ha="left", va="top")
    ax.annotate(f"median (10k-gon) = {y_med:g}",
                xy=(0.0, y_med), xycoords=("axes fraction", "data"),
                xytext=(4, 4), textcoords="offset points",
                fontsize=9, ha="left", va="bottom")

    ax.set_xlabel("Per-chain mean 20-gon crossing number", fontsize=12)
    ax.set_ylabel("10k-gon crossing count", fontsize=12)
    if title:
        ax.set_title(title, fontsize=14, pad=10)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0),
              fontsize=10, framealpha=0.9, borderaxespad=0.0)
    ax.grid(False)
    plt.tight_layout()

    png_path = os.path.join(output_dir, f"{base_name}_scatter.png")
    pdf_path = os.path.join(output_dir, f"{base_name}_scatter.pdf")
    plt.savefig(png_path, dpi=150)
    ax.set_rasterized(True)
    plt.savefig(pdf_path, dpi=150)
    plt.close()
    print(f"[INFO] Scatter saved to {png_path}")


def plot_quadrants(per_obj, output_dir, base_name, title):
    n_total = len(per_obj)
    scores = per_obj["score"].to_numpy()
    populations = per_obj["population"].to_numpy()

    median_score = float(np.median(scores))
    print(f"[INFO] Median 20-gon score across {n_total} objects: {median_score:.4f}")

    is_ov = split_at_median(scores, seed=42)

    counts = np.zeros((2, 2), dtype=np.int64)
    for i, pop in enumerate(ROW_ORDER):
        pop_mask = (populations == pop)
        counts[i, 0] = int(np.sum(pop_mask & ~is_ov))
        counts[i, 1] = int(np.sum(pop_mask &  is_ov))
    matrix = 2.0 * 100.0 * counts / n_total

    cell_colors = np.array([[DIAG_COLOR, OFFDIAG_COLOR],
                            [OFFDIAG_COLOR, DIAG_COLOR]])

    sns.set_style("white")
    fig, ax = plt.subplots(figsize=(7, 7))
    for i in range(2):
        for j in range(2):
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                        facecolor=cell_colors[i, j],
                                        edgecolor="black", linewidth=1.5))
            ax.text(j, i, f"{matrix[i, j]:.1f}%",
                    ha="center", va="center",
                    fontsize=34, fontweight="bold", color="black")

    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(1.5, -0.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Under-knotted", "Over-knotted"], fontsize=14)
    ax.set_yticks([0, 1])
    ax.set_yticklabels([POP_LABELS[p] for p in ROW_ORDER],
                       fontsize=14, rotation=90, va="center")
    ax.tick_params(axis="x", length=0, pad=8)
    ax.tick_params(axis="y", length=0, pad=8)
    ax.set_xlabel("20-gon score class", fontsize=16, fontweight="bold", labelpad=12)
    ax.set_ylabel("Source population", fontsize=16, fontweight="bold", labelpad=12)
    if title:
        ax.set_title(title, fontsize=14, pad=10)
    for spine in ax.spines.values():
        spine.set_linewidth(2.0)
    ax.set_aspect("equal")
    plt.tight_layout()

    png_path = os.path.join(output_dir, f"{base_name}_quadrants.png")
    pdf_path = os.path.join(output_dir, f"{base_name}_quadrants.pdf")
    plt.savefig(png_path, dpi=150)
    plt.savefig(pdf_path, dpi=150)
    plt.close()
    print(f"[INFO] Quadrant matrix saved to {png_path}")

    print("\nMatrix (rows and columns each sum to 100%):")
    print(f"  {'':14s}  {'underknotted':>13s}  {'overknotted':>12s}")
    for i, pop in enumerate(ROW_ORDER):
        print(f"  {POP_LABELS[pop]:14s}  {matrix[i, 0]:>12.2f}%  {matrix[i, 1]:>11.2f}%")
    print(f"\nCounts (sum per row = {n_total // 2}):")
    print(f"  {'':14s}  {'underknotted':>13s}  {'overknotted':>12s}")
    for i, pop in enumerate(ROW_ORDER):
        print(f"  {POP_LABELS[pop]:14s}  {counts[i, 0]:>13,}  {counts[i, 1]:>12,}")


def plot_histogram(per_obj, output_dir, base_name, title):
    """Histogram of per-object 20-gon scores (raw units)."""
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.2)
    fig, ax = plt.subplots(figsize=(10, 5.5))

    scores = per_obj["score"].to_numpy(dtype=float)
    bins = np.linspace(0.0, max(scores.max(), 1e-6), 41)

    for pop in ROW_ORDER:
        mask = (per_obj["population"].to_numpy() == pop)
        ax.hist(scores[mask], bins=bins, color=POP_COLORS[pop],
                label=f"{POP_LABELS[pop]} (n={int(mask.sum())})",
                edgecolor="black", linewidth=0.4, alpha=0.75)

    median_val = float(np.median(scores))
    ax.axvline(median_val, color="black", linestyle="--", linewidth=1.4)
    ax.annotate(f"median = {median_val:.3f}",
                xy=(median_val, 1.0), xycoords=("data", "axes fraction"),
                xytext=(4, -4), textcoords="offset points",
                fontsize=10, ha="left", va="top")

    ax.set_xlabel("Per-chain mean 20-gon crossing number", fontsize=12)
    ax.set_ylabel("Number of objects", fontsize=12)
    if title:
        ax.set_title(title, fontsize=14, pad=10)
    ax.legend(loc="upper right", fontsize=10, framealpha=0.9)
    ax.grid(False, axis="x")
    plt.tight_layout()

    png_path = os.path.join(output_dir, f"{base_name}_histogram.png")
    pdf_path = os.path.join(output_dir, f"{base_name}_histogram.pdf")
    plt.savefig(png_path, dpi=150)
    plt.savefig(pdf_path, dpi=150)
    plt.close()
    print(f"[INFO] Histogram saved to {png_path}")


def main():
    parser = argparse.ArgumentParser(description="Quadrant matrix + histogram (per-object mean score)")
    parser.add_argument("samples_parquet",
                        help="Parquet from salmonella_quadrant_compute.py (per-sample crossings)")
    parser.add_argument("--output-dir", "-o", default=None,
                        help="Output directory (default: same as input)")
    parser.add_argument("--base-name", default=None,
                        help="Output base filename (default: derived from input)")
    parser.add_argument("--title", default=None)
    parser.add_argument("--summary-csv", default=None,
                        help="Per-object summary CSV with crossing_10k column "
                             "(default: <parquet>_samples.parquet → <parquet>.csv)")
    args = parser.parse_args()

    input_path = os.path.expanduser(args.samples_parquet)
    output_dir = os.path.expanduser(args.output_dir) if args.output_dir else os.path.dirname(input_path)
    if not output_dir:
        output_dir = "."
    base_name = args.base_name or os.path.splitext(os.path.basename(input_path))[0]

    df = load_samples(input_path)
    print(f"[INFO] Loaded {len(df):,} samples "
          f"({(df['population']=='unknotted').sum():,} u, "
          f"{(df['population']=='knotted').sum():,} k)")

    per_obj = per_object_mean(df)
    print(f"[INFO] Computed mean score for {len(per_obj)} objects")

    summary_csv = args.summary_csv
    if summary_csv is None:
        root = input_path[: -len("_samples.parquet")] if input_path.endswith("_samples.parquet") else os.path.splitext(input_path)[0]
        summary_csv = root + ".csv"
    if os.path.exists(summary_csv):
        summary = pd.read_csv(summary_csv)
        per_obj = per_obj.merge(
            summary[["filename", "population", "crossing_10k"]],
            on=["filename", "population"], how="left",
        )
        print(f"[INFO] Joined 10k-gon crossing counts from {summary_csv}")
        plot_scatter(per_obj, output_dir, base_name, args.title)
    else:
        print(f"[WARN] Summary CSV not found at {summary_csv}; skipping scatter")

    plot_quadrants(per_obj, output_dir, base_name, args.title)
    plot_histogram(per_obj, output_dir, base_name, args.title)


if __name__ == "__main__":
    main()
