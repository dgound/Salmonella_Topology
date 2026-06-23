#!/usr/bin/env bash
#
# End-to-end pipeline for the Gaussian random polygon quadrant experiment:
#   1. Generate Gaussian random polygons (Python port of Mathematica's
#      GaussianRandomPolygon[n_]).
#   2. Compute per-polygon stats: full-chain crossing + per-length deWitt
#      mean crossing across a length sweep.
#   3. Plot a threshold sweep for each length plus a convergence summary.
#
# Steps 2 and 3 require the `knoodle` conda environment.
#
# Usage:
#   ./run_gaussian_polygon_pipeline.sh [-n NUM_POLYGONS] [-v N_VERTICES]
#                                      [-s NUM_SAMPLES] [-o OUTPUT_DIR]
#                                      [-L LENGTHS] [-P THRESHOLDS]
#                                      [--skip-generate] [--skip-compute] [--skip-plot]
#
# Defaults reproduce the original experiment: 25000 polygons of 25000 vertices,
# 10000 deWitt samples per length, lengths 100,75,50,25,20,15,10,6, thresholds
# 50,45,40,35,30,25,20,15,10,5,2.5,1.

set -euo pipefail

NUM_POLYGONS=25000
N_VERTICES=25000
NUM_SAMPLES=10000
OUTPUT_DIR="Gaussian_polygons"
LENGTHS="100,75,50,25,20,15,10,6"
THRESHOLDS="50,45,40,35,30,25,20,15,10,5,2.5,1"
SEED=42
CONDA_ENV="knoodle"

SKIP_GENERATE=0
SKIP_COMPUTE=0
SKIP_PLOT=0

usage() {
    sed -n '2,18p' "$0"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -n) NUM_POLYGONS="$2"; shift 2 ;;
        -v) N_VERTICES="$2"; shift 2 ;;
        -s) NUM_SAMPLES="$2"; shift 2 ;;
        -o) OUTPUT_DIR="$2"; shift 2 ;;
        -L) LENGTHS="$2"; shift 2 ;;
        -P) THRESHOLDS="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --conda-env) CONDA_ENV="$2"; shift 2 ;;
        --skip-generate) SKIP_GENERATE=1; shift ;;
        --skip-compute)  SKIP_COMPUTE=1; shift ;;
        --skip-plot)     SKIP_PLOT=1; shift ;;
        -h|--help) usage ;;
        *) echo "Unknown option: $1" >&2; usage ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

STATS_CSV="${OUTPUT_DIR}/quadrant_stats.csv"

echo "============================================================"
echo "Gaussian random polygon pipeline"
echo "  Output dir:                 ${OUTPUT_DIR}"
echo "  Polygons:                   ${NUM_POLYGONS}"
echo "  Vertices per polygon:       ${N_VERTICES}"
echo "  deWitt samples per length:  ${NUM_SAMPLES}"
echo "  Lengths:                    ${LENGTHS}"
echo "  Thresholds (%):             ${THRESHOLDS}"
echo "  Conda env (compute/plot):   ${CONDA_ENV}"
echo "============================================================"

# Step 1: generate
if [[ "$SKIP_GENERATE" -eq 0 ]]; then
    echo
    echo "[1/3] Generating ${NUM_POLYGONS} polygons of ${N_VERTICES} vertices ..."
    python generate_gaussian_polygons.py \
        --num-polygons "${NUM_POLYGONS}" \
        --n-vertices   "${N_VERTICES}"   \
        --output-folder "${OUTPUT_DIR}"  \
        --seed "${SEED}"
else
    echo
    echo "[1/3] SKIPPED (--skip-generate)"
fi

# Activate conda for steps that need knoodle
activate_conda() {
    if ! command -v conda >/dev/null 2>&1; then
        echo "ERROR: conda not found on PATH" >&2
        exit 1
    fi
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV}"
}

# Step 2: compute
if [[ "$SKIP_COMPUTE" -eq 0 ]]; then
    echo
    echo "[2/3] Computing per-polygon stats (env: ${CONDA_ENV}) ..."
    (
        activate_conda
        python salmonella_gaussian_quadrant_compute.py \
            --input-folder "${OUTPUT_DIR}" \
            --output       "${STATS_CSV}" \
            --lengths      "${LENGTHS}"   \
            --num-samples  "${NUM_SAMPLES}"
    )
else
    echo
    echo "[2/3] SKIPPED (--skip-compute)"
fi

# Step 3: plot
if [[ "$SKIP_PLOT" -eq 0 ]]; then
    echo
    echo "[3/3] Plotting threshold + length sweeps ..."
    (
        activate_conda
        python salmonella_gaussian_quadrant_plot.py \
            "${STATS_CSV}" \
            -o          "${OUTPUT_DIR}" \
            --thresholds "${THRESHOLDS}" \
            --lengths    "${LENGTHS}"
    )
else
    echo
    echo "[3/3] SKIPPED (--skip-plot)"
fi

echo
echo "Pipeline complete. Outputs in: ${OUTPUT_DIR}"
