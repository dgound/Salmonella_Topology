import numpy as np
import glob
import os


def tsv_to_gro(tsv_file, gro_file=None,
               title=None,
               residue_name="ASP",
               atom_name="CA",
               scale=1):
    """
    Convert TSV xyz coordinates to GROMACS .gro format.

    Parameters
    ----------
    tsv_file : str
        Input TSV file with x y z coordinates.
    gro_file : str, optional
        Output .gro filename. If None, replaces .tsv with .gro.
    title : str
        First line of GRO file.
    residue_name : str
        Residue label (default: ASP).
    atom_name : str
        Atom label (default: CA).
    scale : float
        Coordinate scaling factor.
        Use 0.1 if TSV is in Å and GRO should be in nm.
    """
    
    if title is None:
        title = os.path.splitext(os.path.basename(tsv_file))[0]

    # -------------------------
    # Load coordinates
    # -------------------------
    coords = np.loadtxt(tsv_file)

    if coords.shape[1] != 3:
        raise ValueError("TSV file must have exactly 3 columns: x y z")

    n_atoms = len(coords)

    # Convert units if needed
    coords *= scale

    # Output filename
    if gro_file is None:
        gro_file = os.path.splitext(tsv_file)[0] + ".gro"

    # -------------------------
    # Write GRO
    # -------------------------
    with open(gro_file, "w") as f:
        # line 1: title
        f.write(f"{title}\n")

        # line 2: atom count
        f.write(f"{n_atoms}\n")

        # atom lines
        for i, (x, y, z) in enumerate(coords, start=1):

            # GRO fixed-width formatting
            line = (
                f"{i:5d}"
                f"{residue_name:<5}"
                f"{atom_name:>5}"
                f"{i:5d}"
                f"{x:8.3f}"
                f"{y:8.3f}"
                f"{z:8.3f}\n"
            )
            f.write(line)

        # final box line
        f.write("   0.00000   0.00000   0.00000\n")

    print(f"Saved: {gro_file}")


# Inserted: npy_to_gro function
def npy_to_gro(npy_file, gro_file=None,
               title=None,
               residue_name="ASP",
               atom_name="CA",
               scale=1):
    """
    Convert a NumPy (.npy) array of xyz coordinates to GROMACS .gro format.

    Parameters
    ----------
    npy_file : str
        Input .npy file containing an (N,3) coordinate array.
    gro_file : str, optional
        Output .gro filename. If None, replaces .npy with .gro.
    title : str
        First line of GRO file.
    residue_name : str
        Residue label (default: ASP).
    atom_name : str
        Atom label (default: CA).
    scale : float
        Coordinate scaling factor.
    """

    if title is None:
        title = os.path.splitext(os.path.basename(npy_file))[0]

    coords = np.load(npy_file)
    coords = np.asarray(coords, dtype=float)

    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("NPY file must contain an (N,3) coordinate array")

    n_atoms = len(coords)

    coords *= scale

    if gro_file is None:
        gro_file = os.path.splitext(npy_file)[0] + ".gro"

    with open(gro_file, "w") as f:
        f.write(f"{title}\n")
        f.write(f"{n_atoms}\n")

        for i, (x, y, z) in enumerate(coords, start=1):
            line = (
                f"{i:5d}"
                f"{residue_name:<5}"
                f"{atom_name:>5}"
                f"{i:5d}"
                f"{x:8.3f}"
                f"{y:8.3f}"
                f"{z:8.3f}\n"
            )
            f.write(line)

        f.write("   0.00000   0.00000   0.00000\n")

    print(f"Saved: {gro_file}")


# Coarse-grain .gro file: keep every `step` bead, total `n_keep` beads.
def coarse_grain_gro(input_gro, output_gro=None, step=500, n_keep=20):
    """
    Create a reduced .gro file by keeping every `step` bead.

    For a 10k-bead polymer and step=500, this keeps beads:
    1, 501, 1001, ..., 9501 (20 beads total).
    """
    if output_gro is None:
        output_gro = os.path.splitext(input_gro)[0] + "_20beads.gro"

    with open(input_gro, "r") as f:
        lines = f.readlines()

    title = lines[0].strip()
    atom_lines = lines[2:-1]
    box_line = lines[-1]

    selected = atom_lines[::step][:n_keep]
    # Replicate the first bead as the final bead to close the cycle
    if len(selected) > 0:
        selected = selected + [selected[0]]

    with open(output_gro, "w") as f:
        f.write(title + "_20beads\n")
        f.write(f"{len(selected)}\n")

        for new_i, line in enumerate(selected, start=1):
            x = float(line[20:28])
            y = float(line[28:36])
            z = float(line[36:44])

            new_line = (
                f"{new_i:5d}"
                f"{'ASP':<5}"
                f"{'CA':>5}"
                f"{new_i:5d}"
                f"{x:8.3f}"
                f"{y:8.3f}"
                f"{z:8.3f}\n"
            )
            f.write(new_line)

        f.write(box_line)

    print(f"Saved: {output_gro}")


# Convert all TSV files to GRO
for file in glob.glob("*.tsv"):
    tsv_to_gro(file)

# Convert all NPY files to GRO
for file in glob.glob("*.npy"):
    npy_to_gro(file)

# Create 20-bead coarse-grained GRO files from existing 10k-bead GRO files
# for file in ["knotted.gro", "unknotted.gro", "polygon-10k.gro"]:
for file in glob.glob("*.gro"):
    if os.path.exists(file):
        coarse_grain_gro(file)