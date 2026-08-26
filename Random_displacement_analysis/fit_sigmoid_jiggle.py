#!/usr/bin/env python3
"""
Fit a 4-parameter logistic to each jiggle-distance H5 produced by
jiggle_control_det_2cat.py and compare datasets.

Model (in log10 distance):
    P(d) = A + (B - A) / (1 + exp(-k * (log10(d) - log10(d_mid))))

Parameters:
    A      : baseline %Knotted at small d (~raw)
    B      : saturation %Knotted at large d (~random)
    d_mid  : transition midpoint distance (nm)
    k      : slope in log10(d) units

Usage:
    python fit_sigmoid_jiggle.py \
        --inputs rez_df_det_2cat_Data200nm_443W_lt100nm_dewitt.h5 \
                 rez_df_det_2cat_widefield_tp3to22.h5 \
        --labels SMLM Widefield \
        --output sigmoid_jiggle_fit.pdf
"""

import argparse
import numpy as np
import pandas as pd
from natsort import natsorted
from scipy.optimize import curve_fit
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt


DISTANCES = np.array([1, 5, 10, 20, 30, 40] +
                     list(range(50, 1050, 50)) +
                     [2000, 2500, 3000, 4000, 5000,
                      7500, 10000, 15000, 20000, 30000, 50000, 100000])


def load_h5(path):
    with pd.HDFStore(path, 'r') as store:
        keys = natsorted(store.keys())
        dfs = [store[k] for k in keys]
    raw_k = float(dfs[0].loc[('Raw', 'Knotted')])
    rnd_k = float(dfs[0].loc[('Random Curves', 'Knotted')])
    jig_k = np.array([float(df.loc[('Jiggled', 'Knotted')]) for df in dfs])
    n = min(len(DISTANCES), len(jig_k))
    d = DISTANCES[:n]
    jig_k = jig_k[:n]
    # Drop early-stop'd zero entries
    jig_u = np.array([float(df.loc[('Jiggled', 'Unknotted')]) for df in dfs])[:n]
    mask = (jig_k + jig_u) > 0
    return d[mask], jig_k[mask], raw_k, rnd_k


def logistic_log10(d, A, B, log10_dmid, k):
    return A + (B - A) / (1.0 + np.exp(-k * (np.log10(d) - log10_dmid)))


def fit(d, y, raw, rnd):
    # Reasonable initial guesses
    p0 = [raw, rnd, np.log10(300.0), 4.0]
    bounds = ([0.0, 0.0, np.log10(1.0), 0.1],
              [100.0, 100.0, np.log10(1e6), 50.0])
    popt, pcov = curve_fit(logistic_log10, d, y, p0=p0, bounds=bounds, maxfev=20000)
    perr = np.sqrt(np.diag(pcov))
    return popt, perr


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--inputs', nargs='+', required=True)
    p.add_argument('--labels', nargs='+', default=None)
    p.add_argument('--output', default='sigmoid_jiggle_fit.pdf')
    args = p.parse_args()

    labels = args.labels or args.inputs
    if len(labels) != len(args.inputs):
        raise ValueError('--labels must match --inputs length')

    colors = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd']

    fig, ax = plt.subplots(figsize=(11, 6))
    fits = {}

    print(f'{"label":<12} {"A (raw)":>9} {"B (sat)":>9} {"d_mid (nm)":>11} {"k":>6} {"raw":>7} {"rnd":>7}')
    print('-' * 70)

    # Extrapolation range for the fitted curve (so plot extends past early-stop)
    x_min, x_max = 1.0, 1e5

    for i, (path, label) in enumerate(zip(args.inputs, labels)):
        d, knt, raw, rnd = load_h5(path)
        popt, perr = fit(d, knt, raw, rnd)
        A, B, lmid, k = popt
        d_mid = 10 ** lmid
        fits[label] = {'A': A, 'B': B, 'd_mid': d_mid, 'k': k,
                       'raw': raw, 'rnd': rnd, 'distances': d, 'knotted': knt}

        c = colors[i % len(colors)]
        d_smooth = np.logspace(np.log10(x_min), np.log10(x_max), 600)
        y_smooth = logistic_log10(d_smooth, *popt)
        ax.scatter(d, knt, color=c, s=30, edgecolors='black', linewidths=0.5,
                   zorder=3, label=f'{label} (data)')
        ax.plot(d_smooth, y_smooth, color=c, linewidth=2.0,
                label=(f'{label} fit: A={A:.1f}%, B={B:.1f}%, '
                       f'd_mid={d_mid:.0f} nm, k={k:.2f}'))
        # Reference lines for raw and random
        ax.axhline(y=raw, color=c, linestyle='--', linewidth=1.0, alpha=0.6,
                   label=f'{label} Raw ({raw:.1f}%)')
        ax.axhline(y=rnd, color=c, linestyle=':', linewidth=1.0, alpha=0.6,
                   label=f'{label} Random ({rnd:.1f}%)')

        print(f'{label:<12} {A:>9.2f} {B:>9.2f} {d_mid:>11.1f} {k:>6.2f} {raw:>7.2f} {rnd:>7.2f}')

    ax.set_xscale('log')
    ax.set_xlim(x_min, x_max)
    ax.set_xlabel('Jiggle distance (nm, log)', fontsize=12, fontweight='bold')
    ax.set_ylabel('% Knotted', fontsize=12, fontweight='bold')
    ax.set_title('Sigmoid fit: knotting % vs jiggle distance', fontsize=13, fontweight='bold')
    ax.set_ylim(0, max(60, max(f['B'] for f in fits.values()) * 1.1))
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(loc='best', fontsize=8, frameon=True, fancybox=True)

    fig.tight_layout()
    fig.savefig(args.output, dpi=200, bbox_inches='tight')
    png = args.output.replace('.pdf', '.png')
    fig.savefig(png, dpi=200, bbox_inches='tight')
    print(f'\nWrote {args.output} and {png}')

    # Pairwise comparison
    if len(fits) >= 2:
        print('\nPairwise comparison (param differences):')
        items = list(fits.items())
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                la, fa = items[i]
                lb, fb = items[j]
                print(f'  {la} vs {lb}:')
                print(f'    delta A      = {fa["A"] - fb["A"]:+.2f} %')
                print(f'    delta B      = {fa["B"] - fb["B"]:+.2f} %')
                print(f'    d_mid ratio  = {fa["d_mid"] / fb["d_mid"]:.2f}x  ({la}/{lb})')
                print(f'    delta d_mid  = {fa["d_mid"] - fb["d_mid"]:+.1f} nm')
                print(f'    delta k      = {fa["k"] - fb["k"]:+.2f}')


if __name__ == '__main__':
    main()
