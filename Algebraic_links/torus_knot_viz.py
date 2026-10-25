#!/usr/bin/env python3
"""
Knot-topology preservation plots for (p, q)-torus knots, mirroring the
torus-link viz (A survival-gap grid + E hero bundles).

    still knotted  <=>  crossing_count > 0        (analog of 'still linked')
    correct type   <=>  Alexander fingerprint match (analog of 'correct |lk|')
    gap            =    knotted but wrong type

Driven by CSVs from torus_knot_downsample_analyze.py.
"""

import argparse
import os
from math import gcd

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

COLOR_KNOTTED = "#1f77b4"
COLOR_CORRECT = "#d62728"
COLOR_GAP = "#f7b267"

# Broken x-axis: left = compressed plateau from the full 10k-gon down,
# right = expanded transition zone. Everything at L >= ~100 is a flat
# 100%/100% plateau (all transitions occur at L < 60).
LEFT_XLIM = (10000, 2000)
RIGHT_XLIM = (200, 5)
LEFT_TICKS = [10000, 2000]
RIGHT_TICKS = [100, 50, 20, 10, 5]

TORUS_KNOT_NAMES = {
    (2, 3): "trefoil", (2, 5): "5_1", (2, 7): "7_1",
    (2, 9): "9_1", (2, 11): "11_1", (2, 13): "13_1",
    (2, 15): "15_1", (2, 17): "17_1", (2, 19): "19_1", (2, 21): "21_1",
    (3, 4): "8_19", (3, 5): "10_124", (4, 5): "T(4,5)",
    (3, 7): "T(3,7)", (5, 6): "T(5,6)", (4, 7): "T(4,7)", (3, 8): "T(3,8)",
}


def knot_name(p, q):
    return TORUS_KNOT_NAMES.get(tuple(sorted((p, q))), f"T({p},{q})")


def parse_entry(spec):
    pq, path = spec.split(":", 1)
    p, q = [int(x) for x in pq.split(",")]
    return p, q, path


def load_entry(p, q, path):
    df = pd.read_csv(path)
    # Determinant label = mode of the determinant column at the largest L
    # (topologically faithful plateau).
    Lmax = df["length"].max()
    det_col = df[df["length"] == Lmax]["determinant"].dropna()
    expected = int(round(det_col.mode().iloc[0])) if not det_col.empty else None
    return {"p": p, "q": q, "expected": expected, "df": df}


def survival_curves(df):
    g = df.groupby("length").agg(knotted=("knotted", "mean"),
                                 correct=("correct", "mean")) * 100
    return g.sort_index(ascending=False)


def _cfg_log(ax, xlim, ticks, show_ticklabels=True):
    ax.set_xscale("log")
    ax.set_xlim(max(xlim), min(xlim))
    tk = [t for t in ticks if min(xlim) <= t <= max(xlim)]
    ax.xaxis.set_major_locator(mticker.FixedLocator(tk))
    ax.xaxis.set_major_formatter(mticker.FixedFormatter([f"{t:g}" for t in tk]))
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    if show_ticklabels:
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    else:
        ax.tick_params(labelbottom=False)


def _break_marks(axL, axR):
    """Diagonal break ticks at the shared boundary of two panels."""
    axL.spines["right"].set_visible(False)
    axR.spines["left"].set_visible(False)
    axR.tick_params(axis="y", which="both", left=False, labelleft=False)
    d = 0.03
    kw = dict(transform=axL.transAxes, color="k", clip_on=False, linewidth=1.0)
    axL.plot([1, 1], [-d, d], **kw)
    axL.plot([1 - d / 2, 1 + d / 2], [1 - d, 1 + d], **kw)
    axL.plot([1 - d / 2, 1 + d / 2], [-d, d], **kw)
    kw = dict(transform=axR.transAxes, color="k", clip_on=False, linewidth=1.0)
    axR.plot([0, 0], [-d, d], **kw)
    axR.plot([-d / 2, d / 2], [1 - d, 1 + d], **kw)
    axR.plot([-d / 2, d / 2], [-d, d], **kw)


def _panel_data(e, xlim):
    """Return (L, knotted%, correct%) restricted to xlim."""
    g = survival_curves(e["df"])
    g = g[(g.index >= min(xlim)) & (g.index <= max(xlim))]
    return (g.index.to_numpy(dtype=float),
            g["knotted"].to_numpy(), g["correct"].to_numpy())


def _draw_broken_panel(axL, axR, e, show_ylabel, show_xlabels):
    """Plateau (axL) + transition (axR) with a broken x-axis."""
    # Left: flat 100%/100% plateau from the full 10k-gon down.
    xp = np.array([max(LEFT_XLIM), min(LEFT_XLIM)], dtype=float)
    axL.plot(xp, [100, 100], color=COLOR_KNOTTED, lw=2.2, zorder=3)
    axL.plot(xp, [100, 100], color=COLOR_CORRECT, lw=2.2, ls="--", zorder=3)

    # Right: transition data.
    L, knotted, correct = _panel_data(e, RIGHT_XLIM)
    axR.fill_between(L, correct, knotted, color=COLOR_GAP, alpha=0.55, linewidth=0, zorder=1)
    axR.plot(L, knotted, color=COLOR_KNOTTED, lw=2.2, zorder=3)
    axR.plot(L, correct, color=COLOR_CORRECT, lw=2.2, ls="--", zorder=3)

    def first_drop(y):
        below = L[y < 99.5]
        return below.max() if below.size else None
    for Lc, col in [(first_drop(correct), COLOR_CORRECT), (first_drop(knotted), COLOR_KNOTTED)]:
        if Lc is not None:
            axR.axvline(Lc, color=col, ls=":", lw=1.0, alpha=0.6, zorder=2)

    for ax in (axL, axR):
        ax.set_ylim(-2, 102)
        ax.set_yticks([0, 20, 40, 60, 80, 100])
        ax.grid(True, alpha=0.3)
    _cfg_log(axL, LEFT_XLIM, LEFT_TICKS, show_ticklabels=show_xlabels)
    _cfg_log(axR, RIGHT_XLIM, RIGHT_TICKS, show_ticklabels=show_xlabels)
    if not show_ylabel:
        axL.tick_params(labelleft=False)
    _break_marks(axL, axR)
    axL.set_title(f"T({e['p']},{e['q']})", fontsize=10, loc="left", pad=3)


def viz_grid(entries, outdir, xlim):
    n = len(entries)
    ncol = 5
    nrow = int(np.ceil(n / ncol))
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.0)
    fig = plt.figure(figsize=(3.8 * ncol, 2.9 * nrow + 1.2))
    outer = fig.add_gridspec(nrow, ncol, hspace=0.5, wspace=0.30)

    for idx, e in enumerate(entries):
        r, c = divmod(idx, ncol)
        inner = outer[r, c].subgridspec(1, 2, width_ratios=[0.55, 3.0], wspace=0.06)
        axL = fig.add_subplot(inner[0])
        axR = fig.add_subplot(inner[1], sharey=axL)
        show_ylabel = (c == 0)
        show_xlabels = (r == nrow - 1) or (idx + ncol >= n)
        _draw_broken_panel(axL, axR, e, show_ylabel, show_xlabels)

    handles = [
        plt.Line2D([0], [0], color=COLOR_KNOTTED, lw=2.2, label="still knotted  (crossings > 0)"),
        plt.Line2D([0], [0], color=COLOR_CORRECT, lw=2.2, ls="--", label="correct type  (Alexander fingerprint)"),
        mpatches.Patch(facecolor=COLOR_GAP, alpha=0.55, label="knotted but wrong type"),
        plt.Line2D([0], [0], color=COLOR_CORRECT, lw=1.0, ls=":", label="onset of type loss"),
        plt.Line2D([0], [0], color=COLOR_KNOTTED, lw=1.0, ls=":", label="onset of knot loss"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=10,
               bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Survival of knot topology under downsampling", fontsize=14, y=1.0)
    fig.supxlabel("Downsampled length L (log, descending; broken axis)",
                  y=0.005, fontsize=11)
    fig.supylabel("Fraction of subsamples (%)", x=0.005, fontsize=11)
    _save(fig, outdir, "knot_A_survival_gap")


def viz_hero(entries, outdir, xlim):
    sns.set_style("whitegrid")
    sns.set_context("talk", font_scale=0.85)
    fig, (axL, axR) = plt.subplots(
        1, 2, sharey=True,
        gridspec_kw={"width_ratios": [0.5, 4], "wspace": 0.04},
        figsize=(12, 6.5))

    knotted_curves, correct_curves, common_L = [], [], None
    for e in entries:
        L, knotted, correct = _panel_data(e, RIGHT_XLIM)
        if common_L is None:
            common_L = L
        axR.plot(L, knotted, color=COLOR_KNOTTED, lw=1.4, alpha=0.5, zorder=3)
        axR.plot(L, correct, color=COLOR_CORRECT, lw=1.4, alpha=0.5, ls="--", zorder=3)
        knotted_curves.append(np.interp(common_L[::-1], L[::-1], knotted[::-1])[::-1])
        correct_curves.append(np.interp(common_L[::-1], L[::-1], correct[::-1])[::-1])
    km = np.mean(knotted_curves, axis=0)
    cm = np.mean(correct_curves, axis=0)
    axR.fill_between(common_L, cm, km, color=COLOR_GAP, alpha=0.35, zorder=1)
    axR.plot(common_L, km, color=COLOR_KNOTTED, lw=4, zorder=4)
    axR.plot(common_L, cm, color=COLOR_CORRECT, lw=4, ls="--", zorder=4)

    # Left: flat 100% plateau from the full 10k-gon down.
    xp = np.array([max(LEFT_XLIM), min(LEFT_XLIM)], dtype=float)
    axL.plot(xp, [100, 100], color=COLOR_KNOTTED, lw=4, zorder=4)
    axL.plot(xp, [100, 100], color=COLOR_CORRECT, lw=4, ls="--", zorder=4)

    for ax in (axL, axR):
        ax.set_ylim(-2, 103)
        ax.set_yticks([0, 20, 40, 60, 80, 100])
        ax.grid(True, alpha=0.3)
    _cfg_log(axL, LEFT_XLIM, LEFT_TICKS)
    _cfg_log(axR, RIGHT_XLIM, RIGHT_TICKS)
    _break_marks(axL, axR)

    axL.set_ylabel("Fraction of subsamples (%)")
    fig.supxlabel("Downsampled length L (log, descending; broken axis)", y=0.01, fontsize=13)
    fig.suptitle("Knottedness survives downsampling long after the exact knot type is lost",
                 fontsize=14, y=0.98)
    handles = [
        plt.Line2D([0], [0], color=COLOR_KNOTTED, lw=4, label="still knotted  (crossings > 0)"),
        plt.Line2D([0], [0], color=COLOR_CORRECT, lw=4, ls="--", label="correct type  (Alexander fingerprint)"),
        mpatches.Patch(facecolor=COLOR_GAP, alpha=0.35, label="knotted but wrong type"),
    ]
    axR.legend(handles=handles, loc="lower left", fontsize=11, framealpha=0.95)
    axR.annotate("thin lines: individual knots\nthick lines: mean over all knots",
                 xy=(0.985, 0.04), xycoords="axes fraction", ha="right", va="bottom",
                 fontsize=9, color="#555")
    _save(fig, outdir, "knot_E_hero_bundles")


def _save(fig, outdir, base):
    os.makedirs(outdir, exist_ok=True)
    fig.savefig(os.path.join(outdir, f"{base}.png"), dpi=160, bbox_inches="tight")
    fig.savefig(os.path.join(outdir, f"{base}.pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved {os.path.join(outdir, base)}.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", action="append", required=True,
                        help="'p,q:csv' per knot")
    parser.add_argument("--output-dir", "-o", required=True)
    parser.add_argument("--xlim", nargs=2, type=float, default=(200, 5))
    args = parser.parse_args()

    entries = []
    for spec in args.csv:
        p, q, path = parse_entry(spec)
        entries.append(load_entry(p, q, path))
    entries.sort(key=lambda x: (x["p"], x["q"]))
    xlim = tuple(args.xlim)
    viz_grid(entries, args.output_dir, xlim)
    viz_hero(entries, args.output_dir, xlim)


if __name__ == "__main__":
    main()
