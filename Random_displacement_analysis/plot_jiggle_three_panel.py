#!/usr/bin/env python3
"""
Three-panel plot (Experiment | Perturbed Trajectories | Random Control) of
jiggle-distance sweep results from `jiggle_control.py`.

Auto-detects the category scheme stored in the H5:
  2-cat : columns 'Unknotted' / 'Knotted'
            -> Unknotted = green, Knotted = red
  3-cat : columns '0_1' / '3_1' / 'Complex'
            -> 0_1 = green, 3_1 = orange, Complex = red

Layout matches the original plots_fig3.py three_panel_plot.

Usage:
  python plot_jiggle_2cat_three_panel.py \
      --input rez_df_<engine>_<C>cat.h5 \
      --output three_panel.pdf \
      --title 'Super-resolution (...)'
"""

import argparse
import numpy as np
import pandas as pd
from natsort import natsorted
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt

# Default distance array used by jiggle_control.py
DISTANCES = np.array([1, 5, 10, 20, 30, 40] +
                     list(range(50, 1050, 50)) +
                     [2000, 2500, 3000, 4000, 5000,
                      7500, 10000, 15000, 20000, 30000, 50000, 100000])

# Category-name -> color. Stacking order goes from bottom (unknot-like) to top
# (most-knotted), matching the original 3-cat plots_fig3.py.
SCHEMES = {
    2: {
        'order': ['Unknotted', 'Knotted'],
        'colors': {
            'Unknotted': '#2ecc71',  # green
            'Knotted':   '#e74c3c',  # red
        },
    },
    3: {
        'order': ['0_1', '3_1', 'Complex'],
        'colors': {
            '0_1':     '#2ecc71',    # green (unknot)
            '3_1':     '#ff9800',    # orange (trefoil)
            'Complex': '#e74c3c',    # red
        },
        'pretty': {
            '0_1': 'Unknotted ($0_1$)',
            '3_1': 'Trefoils ($3_1$)',
            'Complex': 'Complex',
        },
    },
}


def detect_categories(df):
    """Inspect a per-distance DataFrame's MultiIndex and return the category list
    that the H5 uses, in the order defined by SCHEMES."""
    present = set(df.index.get_level_values(1))
    for n, scheme in SCHEMES.items():
        if set(scheme['order']) <= present:
            return n, scheme['order']
    raise ValueError(f"Unknown H5 category scheme. Index level-1 values: {present}")


def load_h5(path):
    """Return (n_cat, ordered_cats, payload_dict).

    payload_dict[cat] = {'raw': float, 'random': float, 'jiggled': np.ndarray}
    plus payload_dict['distances'] = np.ndarray.
    Distances where every jiggled category is 0 (early-stop tail) are dropped.
    """
    with pd.HDFStore(path, 'r') as store:
        keys = natsorted(store.keys())
        dfs = [store[k] for k in keys]

    n_cat, cats = detect_categories(dfs[0])
    n = min(len(DISTANCES), len(dfs))
    distances = DISTANCES[:n]

    payload = {}
    for cat in cats:
        payload[cat] = {
            'raw':    float(dfs[0].loc[('Raw', cat)]),
            'random': float(dfs[0].loc[('Random Curves', cat)]),
            'jiggled': np.array([float(df.loc[('Jiggled', cat)]) for df in dfs[:n]]),
        }

    # Drop early-stop'd empty distances (all-zero across all cats)
    totals = np.sum([payload[cat]['jiggled'] for cat in cats], axis=0)
    mask = totals > 0
    for cat in cats:
        payload[cat]['jiggled'] = payload[cat]['jiggled'][mask]
    payload['distances'] = distances[mask]
    return n_cat, cats, payload


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True, help='H5 file from jiggle_control.py')
    p.add_argument('--output', required=True, help='Output PDF path')
    p.add_argument('--title', default='', help='Suptitle for the figure')
    p.add_argument('--reference_x', type=float, default=100.0,
                   help='Vertical reference line on the center panel (default 100 nm)')
    args = p.parse_args()

    n_cat, cats, d = load_h5(args.input)
    scheme = SCHEMES[n_cat]
    colors = scheme['colors']
    pretty = scheme.get('pretty', {c: c for c in cats})

    fig = plt.figure(figsize=(18, 6))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.05, 2, 0.05], wspace=0.01)

    ax_raw = fig.add_subplot(gs[0])
    ax_main = fig.add_subplot(gs[1], sharey=ax_raw)
    ax_ctrl = fig.add_subplot(gs[2], sharey=ax_raw)

    # Snap the right edge of the perturbed-trajectories area to the Random
    # Control (asymptote alignment): append a synthetic "d -> infinity" point
    # whose values equal the Random Control bar.
    plot_d = np.append(d['distances'], d['distances'][-1] * 2)
    plot_jig = {cat: np.append(d[cat]['jiggled'], d[cat]['random']) for cat in cats}

    # --- Center panel: stacked area Unknotted / ... / Knotted vs distance ---
    bottom = np.zeros_like(plot_d, dtype=float)
    for cat in cats:
        top = bottom + plot_jig[cat]
        ax_main.fill_between(plot_d, bottom, top,
                             color=colors[cat], alpha=0.85, label=pretty[cat])
        bottom = top
    ax_main.set_xscale('log')
    ax_main.set_xlabel('Jiggle distance in nm (log)')
    ax_main.set_title('Perturbed Trajectories')
    ax_main.set_ylim(0, 100)
    ax_main.set_xlim(1, plot_d[-1])
    ax_main.grid(alpha=0.3)
    ax_main.axvline(x=args.reference_x, color='black', linestyle=':',
                    linewidth=1.5, label=f'{args.reference_x:.0f} nm')
    ax_main.tick_params(axis='y', which='both', left=False, labelleft=False)

    # --- Left panel: experiment (Raw) stacked bar ---
    bottom = 0.0
    for cat in cats:
        v = d[cat]['raw']
        ax_raw.bar(0, v, bottom=bottom, color=colors[cat], alpha=0.85)
        bottom += v
    ax_raw.set_title('Experiment')
    ax_raw.set_ylim(0, 100)
    ax_raw.set_xticks([])
    ax_raw.set_ylabel('Percentage')
    ax_raw.grid(axis='y', alpha=0.3)

    # --- Right panel: random control stacked bar ---
    bottom = 0.0
    for cat in cats:
        v = d[cat]['random']
        ax_ctrl.bar(0, v, bottom=bottom, color=colors[cat], alpha=0.85)
        bottom += v
    ax_ctrl.set_title('Random Control')
    ax_ctrl.set_ylim(0, 100)
    ax_ctrl.set_xticks([])
    ax_ctrl.tick_params(axis='y', which='both', left=False, labelleft=False)
    ax_ctrl.grid(axis='y', alpha=0.3)

    # Spine cleanup (match original)
    for ax in (ax_raw, ax_main, ax_ctrl):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
    ax_main.spines['left'].set_visible(False)
    ax_ctrl.spines['left'].set_visible(False)

    # Single shared legend (reversed so the most-knotted category sits on top)
    handles, labels = ax_main.get_legend_handles_labels()
    handles = handles[::-1]
    labels = labels[::-1]
    fig.legend(handles, labels, loc='center left', bbox_to_anchor=(0.9, 0.5), ncol=1)

    if args.title:
        fig.suptitle(args.title, fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0.07, 1, 0.95])
    else:
        plt.tight_layout(rect=[0, 0.07, 1, 1])

    fig.savefig(args.output, dpi=300, bbox_inches='tight')
    png = args.output.replace('.pdf', '.png')
    fig.savefig(png, dpi=300, bbox_inches='tight')
    print(f'Wrote {args.output} and {png}  ({n_cat}-category schema)')
    print('  Raw    :', ', '.join(f'{cat} {d[cat]["raw"]:.1f}%' for cat in cats))
    print('  Random :', ', '.join(f'{cat} {d[cat]["random"]:.1f}%' for cat in cats))


if __name__ == '__main__':
    main()
