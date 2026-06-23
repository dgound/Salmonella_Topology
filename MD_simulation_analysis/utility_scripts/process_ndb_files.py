#!/usr/bin/env python3
"""
Process NDB files by extracting XYZ coordinates and applying random rotations.
Combines functionality from ndb2tsv.py and rotate_tsv.py into a single pipeline.
"""

import numpy as np
import argparse
from pathlib import Path
from scipy.spatial.transform import Rotation


def extract_xyz_coordinates(ndb_file_path):
    """
    Extract XYZ coordinates from NDB file after the MODEL line.

    Args:
        ndb_file_path: Path to the NDB file

    Returns:
        Nx3 numpy array of coordinates, or None if extraction failed
    """
    coordinates = []
    found_model = False

    try:
        with open(ndb_file_path, 'r') as file:
            for line in file:
                # Look for the MODEL line
                if "MODEL" in line:
                    found_model = True
                    continue

                # Once we found MODEL, start extracting coordinates
                if found_model and len(line) >= 66:
                    try:
                        # Extract coordinates from specific column positions
                        x = float(line[40:48].strip())
                        y = float(line[49:57].strip())
                        z = float(line[58:66].strip())
                        coordinates.append([x, y, z])
                    except (ValueError, IndexError):
                        # Skip lines that don't have valid coordinates
                        continue

    except FileNotFoundError:
        print(f"Error: File {ndb_file_path} not found")
        return None
    except Exception as e:
        print(f"Error reading {ndb_file_path}: {e}")
        return None

    if not coordinates:
        return None

    return np.array(coordinates)


def generate_random_rotation_matrix():
    """
    Generate a random 3D rotation matrix using scipy.

    Returns:
        3x3 numpy array representing the rotation matrix
    """
    rotation = Rotation.random()
    return rotation.as_matrix()


def apply_rotation_to_coordinates(coordinates, rotation_matrix):
    """
    Apply rotation matrix to a set of 3D coordinates.

    Args:
        coordinates: Nx3 numpy array of coordinates
        rotation_matrix: 3x3 rotation matrix

    Returns:
        Nx3 numpy array of rotated coordinates
    """
    return coordinates @ rotation_matrix.T


def process_ndb_file(ndb_file_path, output_file_path):
    """
    Process a single NDB file: extract coordinates, rotate, and save as TSV.

    Args:
        ndb_file_path: Path to input NDB file
        output_file_path: Path to output TSV file

    Returns:
        True if successful, False otherwise
    """
    try:
        # Extract coordinates from NDB file
        coordinates = extract_xyz_coordinates(ndb_file_path)

        if coordinates is None or len(coordinates) == 0:
            print(f"Warning: No coordinates found in {ndb_file_path.name}")
            return False

        # Generate random rotation matrix
        rotation_matrix = generate_random_rotation_matrix()

        # Apply rotation
        rotated_coordinates = apply_rotation_to_coordinates(coordinates, rotation_matrix)

        # Save rotated coordinates to TSV file
        np.savetxt(output_file_path, rotated_coordinates, delimiter='\t', fmt='%.6f')

        print(f"✓ Processed: {ndb_file_path.name} -> {output_file_path.name} ({len(coordinates)} coordinates)")
        return True

    except Exception as e:
        print(f"✗ Error processing {ndb_file_path.name}: {e}")
        return False


def process_ndb_folder(input_folder, output_folder):
    """
    Process all NDB files in a folder: extract coordinates, rotate, and save as TSV.

    Args:
        input_folder: Path to folder containing NDB files
        output_folder: Path to folder for output TSV files
    """
    input_path = Path(input_folder)
    output_path = Path(output_folder)

    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)

    # Find all NDB files
    ndb_files = sorted(list(input_path.glob('*.ndb')))

    if not ndb_files:
        print(f"No NDB files found in {input_folder}")
        return

    print(f"Found {len(ndb_files)} NDB files in {input_folder}")
    print(f"Output will be saved to {output_folder}")
    print()

    success_count = 0
    for ndb_file in ndb_files:
        # Create output filename (same name but .tsv extension)
        output_file = output_path / f"{ndb_file.stem}.tsv"

        if process_ndb_file(ndb_file, output_file):
            success_count += 1

    print()
    print(f"{'='*60}")
    print(f"Processing complete!")
    print(f"Successfully processed: {success_count}/{len(ndb_files)} files")

    if success_count < len(ndb_files):
        print(f"Failed: {len(ndb_files) - success_count} files")
    else:
        print("All files processed successfully!")


def main():
    parser = argparse.ArgumentParser(
        description='Process NDB files by extracting and rotating XYZ coordinates',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python process_ndb_files.py --input-folder ./ndb_data --output-folder ./rotated_tsv
  python process_ndb_files.py -i ./raw_data -o ./processed_data
        """
    )

    parser.add_argument(
        '--input-folder', '-i',
        required=True,
        help='Folder containing NDB files to process'
    )
    parser.add_argument(
        '--output-folder', '-o',
        required=True,
        help='Folder where rotated TSV files will be saved'
    )

    args = parser.parse_args()

    # Process all NDB files in the input folder
    process_ndb_folder(args.input_folder, args.output_folder)


if __name__ == "__main__":
    main()
