#!/usr/bin/env python3
"""Convert GROMACS .gro files to rotated TSV coordinate files."""

import numpy as np
import argparse
import random
from pathlib import Path
from scipy.spatial.transform import Rotation


def extract_xyz_coordinates(gro_file_path):
    coordinates = []
    try:
        with open(gro_file_path, 'r') as file:
            lines = file.readlines()
        if len(lines) < 4:
            return None
        n_atoms = int(lines[1].strip())
        for line in lines[2:2 + n_atoms]:
            try:
                x = float(line[20:28].strip())
                y = float(line[28:36].strip())
                z = float(line[36:44].strip())
                coordinates.append([x, y, z])
            except (ValueError, IndexError):
                continue
    except Exception as e:
        print(f"Error reading {gro_file_path}: {e}")
        return None
    return np.array(coordinates) if coordinates else None


def process_gro_file(gro_file_path, output_file_path):
    coords = extract_xyz_coordinates(gro_file_path)
    if coords is None or len(coords) == 0:
        print(f"Warning: No coordinates in {gro_file_path.name}")
        return False
    rot = Rotation.random().as_matrix()
    rotated = coords @ rot.T
    np.savetxt(output_file_path, rotated, delimiter='\t', fmt='%.6f')
    print(f"OK: {gro_file_path.name} -> {output_file_path.name} ({len(coords)} pts)")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-folder', '-i', required=True)
    parser.add_argument('--output-folder', '-o', required=True)
    parser.add_argument('--num-samples', '-n', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    in_path = Path(args.input_folder)
    out_path = Path(args.output_folder)
    out_path.mkdir(parents=True, exist_ok=True)

    files = sorted(in_path.glob('*.gro'))
    if args.num_samples is not None and args.num_samples < len(files):
        random.seed(args.seed)
        files = sorted(random.sample(files, args.num_samples))

    for f in files:
        process_gro_file(f, out_path / f"{f.stem}.tsv")


if __name__ == "__main__":
    main()
