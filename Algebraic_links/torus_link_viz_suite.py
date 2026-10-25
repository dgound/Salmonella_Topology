#!/usr/bin/env python3
"""
Four visualizations of torus-link topology preservation under De Witt
downsampling, all driven from the dense-L CSVs produced by
torus_link_downsample_analyze.py.

Message: as polygon length L decreases, "inseparability" (|lk| != 0)
survives far longer than the "correct link type" (|lk| = expected). The
gap between the two is the region of links that are still linked but no
longer identifiable as the specific torus link they came from.

Outputs (into --output-dir):
  A_survival_gap.{png,pdf}       — grid of survival-gap curves
  B_density_heatmap.{png,pdf}    — grid of |lk|-vs-L probability heatmaps
  C_streamgraph.{png,pdf}        — grid of centered |lk| streamgraphs
  D_universal_collapse.{png,pdf} — single overlay with L rescaled by |lk|
"""

import argparse
import os
from math import gcd

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

TORUS_LINK_NAMES = {
    (2, 2): "Hopf", (2, 4): "Solomon", (2, 6): "Star of David",
    (3, 3): "3-chain", (4, 4): "4-chain", (5, 5): "5-chain", (6, 6): "6-chain",
}

COLOR_LINKED = "#1f77b4"
COLOR_CORRECT = "#d62728"
COLOR_GAP = "#f7b267"


def torus_link_name(p, q):
    return TORUS_LINK_NAMES.get(tuple(sorted((p, q))), f"T({p},{q})")


def parse_entry(spec):
    pq, path = spec.split(":", 1)
    p, q = [int(x) for x in pq.split(",")]
    return p, q, path


def load_entry(p, q, path):
    d = gcd(p, q)
    expected = p * q // (d * d)
    df = pd.read_csv(path)
    return {"p": p, "q": q, "d": d, "expected": expected, "df": df}


# ---- shared aggregates -------------------------------------------------------

def survival_curves(df, expected):
    """L (desc) -> still_linked %, correct %."""
    df = df.copy()
    df["linked"] = (df["lk_abs_int"] != 0).astype(int)
    df["correct"] = (df["lk_abs_int"] == expected).astype(int)
    g = df.groupby("length").agg(still_linked=("linked", "mean"),
                                 correct=("correct", "mean")) * 100
    return g.sort_index(ascending=False)


def lk_distribution(df, k_max):
    """Return (lengths_desc, k_values, matrix[L,k] = fraction %)."""
    df = df.copy()
    df["k"] = df["lk_abs_int"].clip(upper=k_max).astype(int)
    grp = df.groupby(["length", "k"]).size().reset_index(name="n")
    tot = df.groupby("length").size().reset_index(name="tot")
    m = grp.merge(tot, on="length")
    m["pct"] = 100 * m["n"] / m["tot"]
    lengths = sorted(df["length"].unique(), reverse=True)
    ks = list(range(0, k_max + 1))
    mat = np.zeros((len(lengths), len(ks)))
    li = {L: i for i, L in enumerate(lengths)}
    ki = {k: i for i, k in enumerate(ks)}
    for _, r in m.iterrows():
        mat[li[r["length"]], ki[int(r["k"])]] = r["pct"]
    return np.array(lengths), np.array(ks), mat


def _log_xaxis(ax, xlim):
    ax.set_xscale("log")
    ax.set_xlim(max(xlim), min(xlim))
    ticks = [t for t in [500, 200, 100, 50, 20, 10, 5]
             if min(xlim) <= t <= max(xlim)]
    ax.xaxis.set_major_locator(mticker.FixedLocator(ticks))
    ax.xaxis.set_major_formatter(mticker.FixedFormatter([f"{t:g}" for t in ticks]))
    ax.xaxis.set_minor_locator(mticker.NullLocator())


# ---- (A) SURVIVAL-GAP CURVES -------------------------------------------------

def viz_A(entries, outdir, xlim):
    n = len(entries)
    ncol = 5
    nrow = int(np.ceil(n / ncol))
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.0)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 2.7 * nrow + 1.2),
                             squeeze=False)
    for idx, e in enumerate(entries):
        r, c = divmod(idx, ncol)
        ax = axes[r][c]
        g = survival_curves(e["df"], e["expected"])
        g = g[(g.index >= min(xlim)) & (g.index <= max(xlim))]
        L = g.index.to_numpy(dtype=float)
        linked = g["still_linked"].to_numpy()
        correct = g["correct"].to_numpy()
        ax.fill_between(L, correct, linked, color=COLOR_GAP, alpha=0.55,
                        linewidth=0, zorder=1)
        ax.plot(L, linked, color=COLOR_LINKED, lw=2.2, zorder=3)
        ax.plot(L, correct, color=COLOR_CORRECT, lw=2.2, ls="--", zorder=3)

        # Critical lengths: largest L where curve first < 99.5%
        def first_drop(y):
            below = L[y < 99.5]
            return below.max() if below.size else None
        Ll = first_drop(linked)
        Lt = first_drop(correct)
        for Lc, col in [(Lt, COLOR_CORRECT), (Ll, COLOR_LINKED)]:
            if Lc is not None:
                ax.axvline(Lc, color=col, ls=":", lw=1.0, alpha=0.6, zorder=2)

        _log_xaxis(ax, xlim)
        ax.set_ylim(-2, 102)
        ax.set_yticks([0, 20, 40, 60, 80, 100])
        if c != 0:
            ax.tick_params(labelleft=False)
        if not (r == nrow - 1 or idx + ncol >= n):
            ax.tick_params(labelbottom=False)
        ax.grid(True, alpha=0.3)
        ax.set_title(f"{torus_link_name(e['p'], e['q'])}  T({e['p']},{e['q']})  |lk|={e['expected']}",
                     fontsize=9.5, loc="left", pad=3)
    for idx in range(n, nrow * ncol):
        r, c = divmod(idx, ncol)
        axes[r][c].axis("off")

    handles = [
        plt.Line2D([0], [0], color=COLOR_LINKED, lw=2.2, label="still linked  (|lk| ≠ 0)"),
        plt.Line2D([0], [0], color=COLOR_CORRECT, lw=2.2, ls="--", label="correct type  (|lk| = expected)"),
        mpatches.Patch(facecolor=COLOR_GAP, alpha=0.55, label="linked but mistyped"),
        plt.Line2D([0], [0], color=COLOR_CORRECT, lw=1.0, ls=":", label="onset of type loss"),
        plt.Line2D([0], [0], color=COLOR_LINKED, lw=1.0, ls=":", label="onset of link loss"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=10.5,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Survival of link topology under downsampling", fontsize=14, y=1.0)
    fig.supxlabel("Downsampled length L per component (log, descending)", y=0.04, fontsize=11)
    fig.supylabel("Fraction of pair-instances (%)", x=0.005, fontsize=11)
    _save(fig, outdir, "A_survival_gap")


# ---- (B) DENSITY HEATMAP -----------------------------------------------------

def viz_B(entries, outdir, xlim):
    n = len(entries)
    ncol = 5
    nrow = int(np.ceil(n / ncol))
    k_max_global = max(e["expected"] + 2 for e in entries)
    sns.set_style("white")
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 2.9 * nrow + 1.2),
                             squeeze=False)
    im = None
    for idx, e in enumerate(entries):
        r, c = divmod(idx, ncol)
        ax = axes[r][c]
        k_max = min(k_max_global, e["expected"] + 3)
        lengths, ks, mat = lk_distribution(e["df"], k_max)
        mask = (lengths >= min(xlim)) & (lengths <= max(xlim))
        lengths = lengths[mask]
        mat = mat[mask]
        # x positions in log space, descending
        im = ax.pcolormesh(lengths, ks, mat.T, cmap="magma",
                           norm=mcolors.PowerNorm(0.5, vmin=0, vmax=100),
                           shading="nearest")
        ax.set_xscale("log")
        ax.set_xlim(max(xlim), min(xlim))
        ticks = [t for t in [500, 100, 20, 5] if min(xlim) <= t <= max(xlim)]
        ax.xaxis.set_major_locator(mticker.FixedLocator(ticks))
        ax.xaxis.set_major_formatter(mticker.FixedFormatter([f"{t:g}" for t in ticks]))
        ax.xaxis.set_minor_locator(mticker.NullLocator())
        ax.axhline(e["expected"], color="#00e5ff", lw=1.4, ls="-", alpha=0.9)
        ax.axhline(0, color="#39ff14", lw=1.4, ls="-", alpha=0.7)
        ax.set_yticks(ks)
        if c != 0:
            ax.tick_params(labelleft=False)
        if not (r == nrow - 1 or idx + ncol >= n):
            ax.tick_params(labelbottom=False)
        ax.set_title(f"{torus_link_name(e['p'], e['q'])}  T({e['p']},{e['q']})  |lk|={e['expected']}",
                     fontsize=9.5, loc="left", pad=3)
    for idx in range(n, nrow * ncol):
        r, c = divmod(idx, ncol)
        axes[r][c].axis("off")
    cbar = fig.colorbar(im, ax=axes, fraction=0.015, pad=0.02)
    cbar.set_label("fraction of samples (%)", fontsize=10)
    fig.suptitle("Where the linking number goes as L drops  "
                 "(cyan = expected, green = lost)", fontsize=13, y=1.0)
    fig.supxlabel("Downsampled length L per component (log, descending)", y=0.04, fontsize=11)
    fig.supylabel("linking number |lk|", x=0.005, fontsize=11)
    _save(fig, outdir, "B_density_heatmap")


# ---- (C) STREAMGRAPH ---------------------------------------------------------

def viz_C(entries, outdir, xlim):
    n = len(entries)
    ncol = 5
    nrow = int(np.ceil(n / ncol))
    sns.set_style("white")
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 2.7 * nrow + 1.2),
                             squeeze=False)
    for idx, e in enumerate(entries):
        r, c = divmod(idx, ncol)
        ax = axes[r][c]
        exp = e["expected"]
        k_max = exp + 3
        lengths, ks, mat = lk_distribution(e["df"], k_max)
        mask = (lengths >= min(xlim)) & (lengths <= max(xlim))
        lengths = lengths[mask]
        mat = mat[mask]  # [L, k]
        # Order bands so 'expected' is centered: expected, then +/-1, +/-2...
        order = sorted(ks, key=lambda k: (abs(k - exp), -(k - exp)))
        # streamgraph: center the stack (ThemeRiver) around 0
        cum = np.zeros(len(lengths))
        total = mat.sum(axis=1)
        baseline = -total / 2.0
        bottoms = baseline.copy()
        cmap = plt.cm.viridis
        norm = mcolors.Normalize(vmin=0, vmax=max(ks))
        for k in order:
            ki = list(ks).index(k)
            y = mat[:, ki]
            color = "#d62728" if k == 0 else cmap(norm(k))
            if k == exp:
                color = "#2ca02c"
            ax.fill_between(lengths, bottoms, bottoms + y, color=color,
                            alpha=0.9, linewidth=0)
            bottoms += y
        ax.set_xscale("log")
        ax.set_xlim(max(xlim), min(xlim))
        ticks = [t for t in [500, 100, 20, 5] if min(xlim) <= t <= max(xlim)]
        ax.xaxis.set_major_locator(mticker.FixedLocator(ticks))
        ax.xaxis.set_major_formatter(mticker.FixedFormatter([f"{t:g}" for t in ticks]))
        ax.xaxis.set_minor_locator(mticker.NullLocator())
        ax.set_yticks([])
        if not (r == nrow - 1 or idx + ncol >= n):
            ax.tick_params(labelbottom=False)
        ax.set_title(f"{torus_link_name(e['p'], e['q'])}  T({e['p']},{e['q']})  |lk|={e['expected']}",
                     fontsize=9.5, loc="left", pad=3)
    for idx in range(n, nrow * ncol):
        r, c = divmod(idx, ncol)
        axes[r][c].axis("off")
    handles = [
        mpatches.Patch(color="#2ca02c", label="correct type (|lk| = expected)"),
        mpatches.Patch(color="#d62728", label="lost (|lk| = 0)"),
        mpatches.Patch(color=plt.cm.viridis(0.6), label="other |lk| (mistyped)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=11,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Linking-number distribution flows as L drops  (river width = fraction)",
                 fontsize=13, y=1.0)
    fig.supxlabel("Downsampled length L per component (log, descending)", y=0.04, fontsize=11)
    _save(fig, outdir, "C_streamgraph")


# ---- (D) UNIVERSAL COLLAPSE --------------------------------------------------

def viz_D(entries, outdir):
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.2)
    fig, ax = plt.subplots(figsize=(10, 6))
    palette = sns.color_palette("tab10", n_colors=len(entries))
    for i, e in enumerate(sorted(entries, key=lambda x: (x["expected"], x["d"]))):
        exp = e["expected"]
        g = survival_curves(e["df"], exp)
        L = g.index.to_numpy(dtype=float)
        scaled = L / exp  # rescale length by pairwise linking number
        color = palette[i]
        ax.plot(scaled, g["still_linked"], color=color, lw=1.8, ls="-",
                label=f"{torus_link_name(e['p'], e['q'])} |lk|={exp}")
        ax.plot(scaled, g["correct"], color=color, lw=1.4, ls="--", alpha=0.7)
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("scaled length  L / |lk|   (log, descending)", fontsize=12)
    ax.set_ylabel("Fraction of pair-instances (%)", fontsize=12)
    ax.set_ylim(-2, 102)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.grid(True, alpha=0.3)
    ax.set_title("Universal collapse: solid = still linked, dashed = correct type\n"
                 "(length rescaled by pairwise linking number)", fontsize=12, pad=8)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9,
              title="link", title_fontsize=10)
    _save(fig, outdir, "D_universal_collapse")


def viz_E(entries, outdir, xlim):
    """Hero figure: 'still linked' bundle vs 'correct type' bundle, all links."""
    sns.set_style("whitegrid")
    sns.set_context("talk", font_scale=0.85)
    fig, ax = plt.subplots(figsize=(11, 6.5))

    linked_curves = []
    correct_curves = []
    common_L = None
    for e in entries:
        g = survival_curves(e["df"], e["expected"])
        g = g[(g.index >= min(xlim)) & (g.index <= max(xlim))]
        L = g.index.to_numpy(dtype=float)
        if common_L is None:
            common_L = L
        ax.plot(L, g["still_linked"], color=COLOR_LINKED, lw=1.4, alpha=0.55, zorder=3)
        ax.plot(L, g["correct"], color=COLOR_CORRECT, lw=1.4, alpha=0.55,
                ls="--", zorder=3)
        linked_curves.append(np.interp(common_L[::-1], L[::-1], g["still_linked"].to_numpy()[::-1])[::-1])
        correct_curves.append(np.interp(common_L[::-1], L[::-1], g["correct"].to_numpy()[::-1])[::-1])

    linked_mean = np.mean(linked_curves, axis=0)
    correct_mean = np.mean(correct_curves, axis=0)
    ax.fill_between(common_L, correct_mean, linked_mean, color=COLOR_GAP,
                    alpha=0.35, zorder=1,
                    label="still linked but wrong type\n(mean gap across links)")
    ax.plot(common_L, linked_mean, color=COLOR_LINKED, lw=4, zorder=4)
    ax.plot(common_L, correct_mean, color=COLOR_CORRECT, lw=4, ls="--", zorder=4)

    _log_xaxis(ax, xlim)
    ax.set_ylim(-2, 103)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("Downsampled length L per component (log, descending)")
    ax.set_ylabel("Fraction of pair-instances (%)")
    ax.set_title("Linking survives downsampling long after the exact link type is lost",
                 fontsize=15, pad=10)

    handles = [
        plt.Line2D([0], [0], color=COLOR_LINKED, lw=4, label="still linked  (|lk| ≠ 0)"),
        plt.Line2D([0], [0], color=COLOR_CORRECT, lw=4, ls="--", label="correct type  (|lk| = expected)"),
        mpatches.Patch(facecolor=COLOR_GAP, alpha=0.35, label="linked but mistyped"),
    ]
    ax.legend(handles=handles, loc="lower left", fontsize=11, framealpha=0.95)
    ax.annotate("thin lines: individual links\nthick lines: mean over all 10 links",
                xy=(0.985, 0.04), xycoords="axes fraction", ha="right", va="bottom",
                fontsize=9, color="#555")
    _save(fig, outdir, "E_hero_bundles")


def _save(fig, outdir, base):
    os.makedirs(outdir, exist_ok=True)
    png = os.path.join(outdir, f"{base}.png")
    pdf = os.path.join(outdir, f"{base}.pdf")
    fig.savefig(png, dpi=160, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved {png}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", action="append", required=True,
                        help="'p,q:dense_csv' per link")
    parser.add_argument("--output-dir", "-o", required=True)
    parser.add_argument("--xlim", nargs=2, type=float, default=(200, 5))
    args = parser.parse_args()

    entries = [load_entry(*parse_entry(s)) for s in args.csv]
    entries.sort(key=lambda x: (x["expected"], x["d"]))
    xlim = tuple(args.xlim)

    viz_A(entries, args.output_dir, xlim)
    viz_B(entries, args.output_dir, xlim)
    viz_C(entries, args.output_dir, xlim)
    viz_D(entries, args.output_dir)
    viz_E(entries, args.output_dir, xlim)


if __name__ == "__main__":
    main()
