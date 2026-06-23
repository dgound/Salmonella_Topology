# Salmonella Topology

Knot-topology analysis of the *Salmonella* chromosome from fluorescence microscopy,
with simulation-based and synthetic null-model controls.

## Overview

A tracked chromosomal locus is imaged across a series of timepoints (TPs). Each
timepoint corresponds to a genomic position along the (circular) bacterial
chromosome, so connecting the per-timepoint 3D positions **in genomic order**
produces a closed space curve, a polygon. The knot type of that polygon is then
computed and compared against random and simulated controls. The central
questions are:

1. Is the chromosome trajectory knotted, and if so, what knot types appear?
2. Does spatial proximity track genomic proximity (genomic Moran's I)?
3. Are the observed topologies distinguishable from random null models?

The repository has **two main experimental paths** that differ only in the
microscopy modality and therefore in how many points each timepoint contributes:

| Path | Modality | Points per timepoint | Per-walk object |
|------|----------|----------------------|-----------------|
| `Super_resolution_analysis/` | SMLM (single-molecule localization) | many localizations (a cloud) | a **distribution** of sampled curves per walk |
| `Widefield_analysis/` | Widefield | one drift-corrected point | a **single ordered curve** per walk |

and **two control paths** that supply null models:

| Path | Role |
|------|------|
| `MD_simulation_analysis/` | Molecular-dynamics chromosome models (knotted vs unknotted), crossing-number classification and downsampling sweeps |
| `Gaussian_polygons/` | Synthetic Gaussian random closed polygons, an analytic null model |

Plus auxiliary tooling: `Polygon_plotting/` (knot-diagram rendering) and the
per-path `utility/` converters.

## Repository layout

```
Salmonella_Topology/
├── Super_resolution_analysis/          # MAIN PATH 1 (SMLM)
│   ├── comprehensive_report_cards.py   #   main pipeline: knot distributions + metrics + report cards
│   ├── jiggle_control/
│   │   ├── jiggle_control.py           #   perturb vertices over a distance sweep, track knot stability
│   │   └── plot_jiggle_three_panel.py  #   three-panel plot of the jiggle sweep
│   ├── Morans_I/                       #   C++ genomic Moran's I tool (spatial-genomic autocorrelation)
│   │   ├── main_genomic.cpp
│   │   ├── morans_i_genomic.{cpp,h}
│   │   └── Makefile
│   └── utility/
│       └── convert_widefield_to_smlm.py# convert widefield centroid CSV into SMLM schema
│
├── Widefield_analysis/                 # MAIN PATH 2 (widefield, single curve per walk)
│   └── comprehensive_walk_report/
│       ├── report_cards_single_curve.py# main pipeline (CLI)
│       ├── qc_post_filter.py           #   imported module: distance matrices + parabola fits
│       └── morans_i.py                 #   imported module: genomic Moran's I (Python port of the C++)
│
├── MD_simulation_analysis/             # CONTROL: MD chromosome models
│   ├── utility_scripts/
│   │   ├── gro_to_rotated_tsv.py       #   GROMACS .gro -> randomly rotated TSV
│   │   ├── process_ndb_files.py        #   .ndb structure -> randomly rotated TSV
│   │   └── tsv_to_gro.py               #   TSV/NPY <-> .gro, plus coarse-graining helper
│   ├── Classifier_analysis/            #   2x2 quadrant classification (full chain vs 20-gon)
│   │   ├── salmonella_quadrant_compute.py
│   │   └── salmonella_quadrant_plot.py
│   └── Downsampling_analysis/          #   knot type vs chain length sweeps
│       ├── salmonella_random_20gon_classify.py
│       ├── salmonella_plot_random_20gon.py
│       └── salmonella_plot_delta.py
│
├── Gaussian_polygons/                  # CONTROL: synthetic random polygons (null model)
│   ├── generate_gaussian_polygons.py
│   ├── salmonella_gaussian_quadrant_compute.py
│   ├── salmonella_gaussian_quadrant_plot.py
│   └── run_gaussian_polygon_pipeline.sh# orchestrates generate -> compute -> plot
│
└── Polygon_plotting/                   # AUXILIARY: 2D knot diagrams with over/under crossings
    └── plot_polygon_over_under.py
```

## Main path 1: super-resolution (SMLM)

SMLM gives many localizations per timepoint. The pipeline samples one
localization per timepoint, connects them in genomic order into a closed 20-gon
(a "De Witt curve"), and repeats `N` times per walk to build a per-walk
**distribution** of knot types and geometric metrics.

### `comprehensive_report_cards.py`

The main analysis. It loads SMLM data, samples trajectories per Walk-ID,
classifies each sampled curve's knot type, generates controls (random polygons
in the convex hull, or downsampled simulated "globule" curves), computes
geometric topology metrics, and writes a multi-page PDF of per-walk report cards
plus summary CSVs.

**Input CSV columns:** `Walk-IDs`, `time-point` (0-19), `x`, `y`, `z` (nm),
`precisionx`, `precisiony`, `precisionz` (nm). Only walks with all timepoints are
kept.

**Key arguments** (see `--help` for the full list of ~40):

| Argument | Meaning |
|----------|---------|
| `N` (positional) | number of sampled trajectories per Walk-ID |
| `M` (positional, default 1000) | number of random 20-gon controls |
| `--input PATH` | input SMLM CSV (default is a hardcoded local path, see Caveats) |
| `--output PREFIX` | output filename prefix |
| `--parallel` / `--workers N` | multiprocessing across walks |
| `--metrics {all,kl,wasserstein}` | distribution-distance metric(s) |
| `--acn` / `--n_acn_projections N` | average crossing number and writhe (1000 projections) |
| `--rog` | radius of gyration |
| `--kendall` | Kendall shape-space analysis |
| `--vassiliev` / `--num_vassiliev_proj N` | Vassiliev V2 invariant |
| `--xyCN` / `--xzCN` / `--yzCN` | per-plane crossing numbers |
| `--unknotted_globule` / `--knotted_globule` (+ `_dir`) | use simulated globule downsamples as controls |
| `--segment_by` | how 10k-gon globules are downsampled to 20-gons |
| `--randomize_order` | connect sampled points in random (not genomic) order (a control) |
| `--morans_i_file PATH` | overlay precomputed Moran's I results on report cards |

**Outputs:** `<prefix>_comprehensive_report.pdf` (per-walk report cards),
`<prefix>_summary.csv` (aggregate knot distribution), and metric CSVs.

**Knot classification** uses `KnotoID` (projection-based identifier). Names are
bucketed by `classify_name()` into `0_1` (unknot), `3_1` (trefoil, merging the
mirror `3_1m`), and `Complex`.

```bash
python comprehensive_report_cards.py 100 1000 \
  --input data.csv --output exp1 \
  --parallel --acn --rog --kendall --metrics all
```

### `jiggle_control/` (robustness control)

`jiggle_control.py` tests whether the observed topology survives positional
perturbation. It builds per-walk weighted-centroid curves, then sweeps a jiggle
distance from 1 nm to 100 um, adding Gaussian noise (sigma = distance) to every
vertex and reclassifying the knot at each step. It compares the jiggled
distribution against Raw, Permuted, and Random-Curve controls and writes an HDF5
file (`rez_df_<engine>_<C>cat.h5`).

- `--engine {knoodle,knotoid}`: `knoodle` classifies via the Alexander
  determinant at t = -1 (det 1 -> unknot, det 3 -> trefoil, else complex);
  `knotoid` uses KnotoID.
- `--categories {2,3}`: 2 = Unknotted/Knotted, 3 = `0_1`/`3_1`/Complex.
- `--dewitt` / `--sample_size N`: sample De Witt curves instead of jiggling centroids.

`plot_jiggle_three_panel.py` renders that HDF5 as a three-panel figure: Raw
distribution (left), the jiggle sweep on a log-distance axis (center), and the
Random-Curve control (right). It auto-detects the 2- vs 3-category scheme.

```bash
python jiggle_control.py data.txt --engine knoodle --categories 2 --workers 8
python plot_jiggle_three_panel.py --input rez_df_knoodle_2cat.h5 --output three_panel.pdf
```

### `Morans_I/` (genomic spatial autocorrelation, C++)

A standalone C++ tool computing genomic Moran's I: are spatially close points
also genomically close? It builds inverse-squared-distance spatial weights, maps
each timepoint to an angle on the circular genome, and computes the circular
Moran's I with a permutation-test p-value.

```bash
cd Super_resolution_analysis/Morans_I
make                                  # builds ./morans_i_genomic
./morans_i_genomic data.csv [max_sample_points]
# -> data_morans_i_results.csv  (Filename, Walk-ID, Time-points, I-value, P-value)
```

The resulting CSV can be fed back into `comprehensive_report_cards.py` via
`--morans_i_file`. The Python port (`Widefield_analysis/.../morans_i.py`)
implements the same algorithm for the widefield path.

### `utility/convert_widefield_to_smlm.py`

Bridges the two modalities: rewrites a widefield centroid CSV (`tp`, `WalkID`,
`x_nm`, `y_nm`, `z_nm`) into the SMLM schema expected by
`comprehensive_report_cards.py`, with `precision*` columns set to 1.0 (uniform
weights).

## Main path 2: widefield (single curve per walk)

Widefield data is already one drift-corrected point per timepoint, so each
Walk-ID is a single ordered curve rather than a sampled distribution. This path
is adapted from the SMLM pipeline with the per-sample distribution machinery
removed.

> **Upstream stage:** the input CSV is the output of a `drift_correct.py`
> step (drift correction + full-coverage filter + movement filter). The input
> schema is `WalkID`, `tp` (e.g. `TP007` or `TP007_CYC000`), `x_nm`, `y_nm`,
> `z_nm`.

### `report_cards_single_curve.py`

The main widefield pipeline (CLI). It loads one ordered curve per walk, computes
per-walk knot type and metrics, builds convex-hull and optional globule
controls, computes genomic Moran's I, and writes a report-card PDF plus CSVs.

**Notable arguments:**

| Argument | Meaning |
|----------|---------|
| `--input` (required) | filtered walks CSV |
| `--outdir` / `--output_prefix` | output location and naming |
| `--tp_start_1based` / `--tp_end_1based` (default 3, 22) | analyzed timepoint range |
| `--acn`, `--rog`, `--xyCN`/`--xzCN`/`--yzCN`, `--kendall` | metric toggles |
| `--n_random_controls` (default 1000) | random n-gons in the convex hull |
| `--unknotted_globule` / `--knotted_globule` / `--show_random` | control sources |
| `--morans_permutations`, `--genome_cycle_bins`, `--morans_absolute_tp`, `--morans_seed`, `--morans_alpha`, `--morans_exclude_circular_offsets_leq` | Moran's I configuration |
| `--morans_cache_csv` / `--morans_i_file` | reuse precomputed Moran's I instead of recomputing |
| `--compact` | shorter 5-page summary report instead of full per-walk cards |
| `--parallel` / `--workers` | multiprocessing |

**Outputs:** `<prefix>_comprehensive_report.pdf` (or `_compact_report.pdf`),
`<prefix>_results.csv` (per-walk knot type, determinant, ACN, writhe, Rg, planar
CNs, Moran's I, KL/EMD vs controls), `<prefix>_morans_i_results.csv`, and
side-car parabola PDFs.

```bash
python report_cards_single_curve.py \
  --input filtered_walks.csv --outdir out --output_prefix wf1 \
  --tp_start_1based 3 --tp_end_1based 22 \
  --acn --rog --xyCN --xzCN --yzCN --parallel
```

### Imported modules (no CLI)

- **`qc_post_filter.py`** builds the population-level QC figures used by the main
  pipeline: 4-panel pairwise-distance heatmaps (3D / X / Y / Z) and mean-parabola
  fits of distance vs genomic offset (weighted quadratic least squares, R^2
  reported), in both XYZ and constrained PC space.
- **`morans_i.py`** is the Python port of the C++ Moran's I, adapted for the
  one-point-per-timepoint model. It exposes `compute_for_walks`,
  `write_morans_results_csv`, and `load_morans_cache_csv`.

The widefield and SMLM paths share the same metric vocabulary (ACN, writhe, Rg,
planar crossing numbers, knot type + determinant, KL / Wasserstein); the
relevant functions are vendored inline in each main script.

## Control 1: MD simulations (`MD_simulation_analysis/`)

Analyzes molecular-dynamics chromosome models, comparing **knotted** and
**unknotted** simulated ensembles. Each model is a ~10,000-bead chain. The path
asks whether downsampled knottedness (a 20-gon-scale crossing measure)
recapitulates the full-chain ground truth.

### Utility scripts (preprocessing)

| Script | Converts |
|--------|----------|
| `gro_to_rotated_tsv.py` | GROMACS `.gro` -> randomly rotated TSV (`x y z`) |
| `process_ndb_files.py` | `.ndb` structure -> randomly rotated TSV |
| `tsv_to_gro.py` | TSV/NPY <-> `.gro`, plus `coarse_grain_gro()` to downsample 10k beads to 20 |

Random rotation (`scipy.spatial.transform.Rotation.random()`) removes any
projection bias before crossing-number analysis.

### `Classifier_analysis/` (2x2 quadrant)

`salmonella_quadrant_compute.py` scores each chain two ways: the full 10k-gon
crossing number, and the **median** crossing number over 10,000 De Witt 20-gons
(stride 500, one random point per chunk). `salmonella_quadrant_plot.py` rank-splits
objects at the median into under/over classes and cross-tabulates source
population (knotted/unknotted) against 20-gon class into a normalized 2x2 matrix,
plus a scatter (20-gon score vs 10k crossing) and a per-sample histogram.

```bash
python salmonella_quadrant_compute.py \
  --knotted-dir rotated_tsv/knotted --unknotted-dir rotated_tsv/unknotted \
  --output quadrant_stats.csv
python salmonella_quadrant_plot.py quadrant_stats_samples.parquet -o plots
```

### `Downsampling_analysis/` (knot type vs chain length)

`salmonella_random_20gon_classify.py` classifies knot type (Unknot / Trefoil /
Complex, via crossing count and determinant) across a sweep of downsampling
lengths and sampling methods (`stride`, `uniform`, `systematic`, `centroid`),
optionally using majority-vote sphere-direction closure (`--closure`).
`salmonella_plot_random_20gon.py` renders stacked-area plots of knot-type
percentage vs length with determinant-gradient coloring; `salmonella_plot_delta.py`
plots the signed (or absolute) difference between two populations.

```bash
python salmonella_random_20gon_classify.py rotated_tsv/knotted -l 10000 -t knotted -n 100
python salmonella_plot_random_20gon.py knotted.csv unknotted.csv --labels Knotted Unknotted
python salmonella_plot_delta.py knotted.csv unknotted.csv --labels Knotted Unknotted --mode both
```

## Control 2: Gaussian polygons (`Gaussian_polygons/`)

A pure synthetic null model: closed 3D polygons whose increments are zero-mean
Gaussian (a port of Mathematica's `GaussianRandomPolygon`). The
`run_gaussian_polygon_pipeline.sh` orchestrator chains the three stages in order:

1. `generate_gaussian_polygons.py` -> 25,000 polygon TSV files
2. `salmonella_gaussian_quadrant_compute.py` -> `quadrant_stats.csv`
   (full-chain crossing + mean crossing at each downsampling length)
3. `salmonella_gaussian_quadrant_plot.py` -> threshold-sweep quadrant figures,
   per-length summary CSVs, and a convergence plot

```bash
cd Gaussian_polygons
./run_gaussian_polygon_pipeline.sh            # full run with default parameters
./run_gaussian_polygon_pipeline.sh --skip-generate   # reuse existing polygons
```

Stages 2 and 3 require the `knoodle` conda environment (the script runs
`conda activate knoodle`).

## Auxiliary: polygon over/under diagrams (`Polygon_plotting/`)

`plot_polygon_over_under.py` renders a single polygon as a 2D knot diagram with
**explicit indexed over/under crossing callouts** rather than the traditional
broken-underpass convention (which is ambiguous for dense curves). It can
auto-select an optimal projection (searching directions on a sphere to minimize
crossings or maximize crossing spacing), emit an authoritative crossing-table
CSV, produce a 3-page PDF report, and save the rotated polygon as `.npy`.

It accepts the pipeline CSV (`--walk_id` / `--walk_ids`, with
`--tp_start_1based`/`--tp_end_1based`), an XYZ text file, or an `.npy` array.

```bash
python plot_polygon_over_under.py \
  --input pipeline.csv --walk_id 135 --output walk135.png \
  --projection auto --projection_objective min_crossings \
  --report_pdf walk135_report.pdf --crossing_csv walk135_crossings.csv
```

## Glossary

- **Walk / Walk-ID**: one tracked locus, its ordered set of per-timepoint 3D
  positions.
- **Timepoint (TP)**: an imaging frame, mapped to a genomic position on the
  circular chromosome.
- **De Witt curve**: a closed polygon built by sampling one localization per
  timepoint and connecting them in genomic order (used to turn an SMLM cloud into
  a discrete curve).
- **Knot type**: `0_1` (unknot), `3_1` (trefoil), or `Complex`.
- **ACN / writhe**: average crossing number and signed crossing number over many
  random 2D projections.
- **Crossing number (CN)**: number of self-crossings in a planar projection
  (`xy`/`xz`/`yz` variants, or a simplified 3D minimum).
- **Determinant**: knot determinant (Alexander polynomial at t = -1); 1 for the
  unknot, 3 for the trefoil.
- **Rg**: radius of gyration of the curve.
- **Genomic Moran's I**: circular spatial-autocorrelation statistic testing
  whether spatial neighbors are genomic neighbors.
- **Globule control**: 20-gons downsampled from simulated ~10k-bead knotted or
  unknotted chromosome models, used as a structured (non-random) null.
- **Quadrant matrix**: a normalized 2x2 contingency table relating source
  population to a downsampled-score class.

## Dependencies

No `requirements.txt` or environment file is committed. The scripts assume a
scientific Python stack plus two knot-analysis libraries.

**Python (standard scientific):** `numpy`, `scipy`, `pandas`, `matplotlib`,
`seaborn`, `tqdm`.

**Knot analysis (required for classification):**
- `knoodle` - crossing number, Alexander determinant. The Gaussian and MD
  pipelines expect it inside a conda environment named `knoodle`.
- `KnotoID` - projection-based knot identifier.

**Optional / situational:**
- `numba` - JIT acceleration of ACN/writhe (falls back to pure Python).
- `wurlitzer` - suppresses C++ stdout from KnotoID.
- `ot` (POT, Python Optimal Transport) - Wasserstein / EMD distances.
- `networkx` - used by the jiggle control.
- `natsort` - used by the jiggle three-panel plot.
- `vassiliev_v2_projection` - custom module for the V2 invariant (external,
  imported by the SMLM main script).

**C++ (Moran's I):** a C++17 compiler (`g++`) and `make`. Build with `make` in
`Super_resolution_analysis/Morans_I/`.

## End-to-end examples

**Super-resolution path**

```bash
# (optional) compute Moran's I first
cd Super_resolution_analysis/Morans_I && make && ./morans_i_genomic data.csv

# main analysis
cd .. && python comprehensive_report_cards.py 100 1000 \
  --input data.csv --output exp1 --parallel \
  --acn --rog --kendall --morans_i_file Morans_I/data_morans_i_results.csv

# robustness sweep
cd jiggle_control
python jiggle_control.py data.txt --engine knoodle --categories 2 --workers 8
python plot_jiggle_three_panel.py --input rez_df_knoodle_2cat.h5 --output sweep.pdf
```

**Widefield path**

```bash
# input must already be drift-corrected/filtered (drift_correct.py, upstream)
cd Widefield_analysis/comprehensive_walk_report
python report_cards_single_curve.py \
  --input filtered_walks.csv --outdir out --output_prefix wf1 \
  --tp_start_1based 3 --tp_end_1based 22 --acn --rog --parallel
```

**Controls**

```bash
# Gaussian null model
cd Gaussian_polygons && ./run_gaussian_polygon_pipeline.sh

# MD simulations
cd MD_simulation_analysis
python utility_scripts/gro_to_rotated_tsv.py -i raw_gro/knotted -o rotated_tsv/knotted
python Classifier_analysis/salmonella_quadrant_compute.py \
  --knotted-dir rotated_tsv/knotted --unknotted-dir rotated_tsv/unknotted \
  --output quadrant_stats.csv
python Classifier_analysis/salmonella_quadrant_plot.py quadrant_stats_samples.parquet -o plots
```

## Data formats

| Format | Columns / structure | Used by |
|--------|---------------------|---------|
| SMLM CSV | `Walk-IDs`, `time-point`, `x`, `y`, `z`, `precisionx/y/z` | SMLM main, Moran's I C++ |
| Widefield CSV | `WalkID`, `tp`, `x_nm`, `y_nm`, `z_nm` | widefield main, polygon plotting |
| Jiggle input TXT | whitespace `time-point Walk-ID x y z` | jiggle control |
| Polygon TSV | tab-delimited `x y z`, one row per vertex | MD + Gaussian controls |
| GROMACS `.gro` / `.ndb` | fixed-width MD structure files | MD utility scripts |
| Moran's I CSV | `Filename`, `Walk-ID`, `Time-points`, `I-value`, `P-value` | overlay onto report cards |

## Caveats

- **Hardcoded default paths.** Several `--input` (and globule directory) defaults
  point to the original author's machine (for example
  `/Users/dimos/knot_analysis_Nir/datasets/...`). Always pass explicit
  `--input`/`--output` paths.
- **Upstream stage.** The widefield path consumes the output of
  `drift_correct.py`, an upstream drift-correction and filtering step.
- **External libraries not vendored.** `knoodle`, `KnotoID`, and
  `vassiliev_v2_projection` must be installed/available separately; classification
  and the V2 invariant are skipped or error out without them.
- **Timepoint conventions differ.** The SMLM tools use 0-based `time-point` 0-19;
  the widefield and plotting tools use 1-based inclusive ranges (default 3-22).
  Mind the offset when comparing across paths.
