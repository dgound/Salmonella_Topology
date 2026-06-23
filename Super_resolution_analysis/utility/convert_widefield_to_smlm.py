#!/usr/bin/env python3
"""
Convert the widefield drift-corrected CSV into the SMLM-schema CSV expected by
the existing jiggle_control / jiggle_control_det / jiggle_control_det_2cat
pipeline.

Widefield CSV (input) provides one centroid point per (WalkID, tp):
  columns include 'tp' (e.g. 'TP012_CYC000'), 'WalkID', 'x_nm', 'y_nm', 'z_nm'.

The output CSV mirrors the SMLM schema used elsewhere in this project:
  Walk-IDs, time-point (int), x, y, z, precisionx, precisiony, precisionz

Since the widefield centroids carry no per-axis precision estimate, the
precision columns are set to 1.0 (uniform weights -- the existing
`weighted_average_and_precision` step then trivially returns the input point).
"""

import argparse
import re
import pandas as pd

TP_RE = re.compile(r'TP(\d+)')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True,
                   help='Widefield CSV (drift-corrected, filtered)')
    p.add_argument('--output', required=True,
                   help='Output CSV in SMLM schema')
    args = p.parse_args()

    print(f'Reading {args.input} ...')
    df = pd.read_csv(args.input,
                     usecols=['tp', 'WalkID', 'x_nm', 'y_nm', 'z_nm'])

    # Parse integer TP from labels like 'TP012_CYC000' -> 12
    def parse_tp(s):
        m = TP_RE.match(str(s))
        if m is None:
            return -1
        return int(m.group(1))

    df['time-point'] = df['tp'].apply(parse_tp)
    bad = (df['time-point'] < 0).sum()
    if bad:
        print(f'  Dropping {bad} rows with unparsable TP labels')
        df = df[df['time-point'] >= 0]

    out = pd.DataFrame({
        'Walk-IDs': df['WalkID'].astype(int),
        'time-point': df['time-point'].astype(int),
        'x': df['x_nm'].astype(float),
        'y': df['y_nm'].astype(float),
        'z': df['z_nm'].astype(float),
        'precisionx': 1.0,
        'precisiony': 1.0,
        'precisionz': 1.0,
    })

    print(f'  Walks: {out["Walk-IDs"].nunique()}')
    print(f'  TPs:   {sorted(out["time-point"].unique())[:3]} ... {sorted(out["time-point"].unique())[-3:]}')
    print(f'  Rows:  {len(out)}')
    out.to_csv(args.output, index=False)
    print(f'Wrote {args.output}')


if __name__ == '__main__':
    main()
