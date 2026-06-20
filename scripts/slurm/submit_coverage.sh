#!/usr/bin/env bash
# =============================================================================
# submit_coverage.sh
# =============================================================================
# Orchestrates the full coverage-matrix pipeline on a SLURM cluster.
#
# Three-stage pipeline with automatic job dependencies:
#   1. prepare  -- serial; builds coverage_job.json manifest
#   2. workers  -- SLURM array; one task per unidentifiable query
#   3. reduce   -- serial; merges shards -> coverage_matrix.csv
#
# Usage (from repo root):
#   bash scripts/slurm/submit_coverage.sh
#
# Or with custom paths:
#   GRAPHML=/path/to/network.graphml \
#   SUPPTABLE=/path/to/supptable1.csv \
#   OUTDIR=/path/to/output \
#   bash scripts/slurm/submit_coverage.sh
#
# Environment variables (all have defaults):
#   GRAPHML      path to .graphml network file
#   SUPPTABLE    path to supptable1.csv
#   OUTDIR       directory for manifest, shards, and final CSV
#   REPO_ROOT    repo root (default: directory containing this script's parent)
#   MAX_ARRAY_CONCURRENT  max simultaneous array tasks (default: 20)
#
# =============================================================================
# SLURM RESOURCE PLACEHOLDERS
# Edit the #SBATCH lines below to match your cluster's partition/queue names
# and resource limits.  The values shown are conservative starting points.
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"

GRAPHML="${GRAPHML:-${REPO_ROOT}/notebooks/Ecoli_Analysis_Notebooks/ecoli_full_network_no_small_rna.graphml}"
SUPPTABLE="${SUPPTABLE:-${REPO_ROOT}/notebooks/Ecoli_Analysis_Notebooks/supptable1.csv}"
OUTDIR="${OUTDIR:-${REPO_ROOT}/notebooks/Ecoli_Analysis_Notebooks}"

MANIFEST="${OUTDIR}/coverage_job.json"
SHARDS_DIR="${OUTDIR}/shards"
OUTPUT_CSV="${OUTDIR}/coverage_matrix.csv"

MAX_ARRAY_CONCURRENT="${MAX_ARRAY_CONCURRENT:-20}"

echo "=== Coverage Matrix Pipeline ==="
echo "  REPO_ROOT:  ${REPO_ROOT}"
echo "  GRAPHML:    ${GRAPHML}"
echo "  SUPPTABLE:  ${SUPPTABLE}"
echo "  OUTDIR:     ${OUTDIR}"
echo "  MANIFEST:   ${MANIFEST}"
echo "  SHARDS_DIR: ${SHARDS_DIR}"
echo "  OUTPUT_CSV: ${OUTPUT_CSV}"
echo ""

# ---------------------------------------------------------------------------
# Stage 1: Prepare (serial, runs on login node or a short job)
# ---------------------------------------------------------------------------
# This step is fast (~minutes) and can run on the login node.
# If your cluster policy forbids heavy work on login nodes, uncomment the
# sbatch block below and comment out the direct `uv run` call.

echo "--- Stage 1: Prepare ---"

# Option A: run directly (fast, login-node safe):
uv run python "${REPO_ROOT}/scripts/coverage_prepare.py" \
    --graphml   "${GRAPHML}" \
    --supptable "${SUPPTABLE}" \
    --manifest  "${MANIFEST}"

# Option B: submit as a short job (uncomment if needed):
# PREPARE_JOB=$(sbatch --parsable \
#     --job-name=cov_prepare \
#     --account=crispr_carb \
#     --partition=slurm \
#     --time=00:30:00 \
#     --mem=8G \
#     --cpus-per-task=1 \
#     --output="${OUTDIR}/logs/prepare_%j.out" \
#     --wrap="cd ${REPO_ROOT} && uv run python scripts/coverage_prepare.py \
#         --graphml ${GRAPHML} --supptable ${SUPPTABLE} --manifest ${MANIFEST}")
# echo "  Prepare job: ${PREPARE_JOB}"
# PREPARE_DEP="--dependency=afterok:${PREPARE_JOB}"

# ---------------------------------------------------------------------------
# Read array size from manifest
# ---------------------------------------------------------------------------
if [[ ! -f "${MANIFEST}" ]]; then
    echo "ERROR: Manifest not found after prepare step: ${MANIFEST}"
    exit 1
fi

N_QUERIES=$(python3 -c "import json; d=json.load(open('${MANIFEST}')); print(d['n_queries'])")
ARRAY_MAX=$((N_QUERIES - 1))
echo "  Array size: ${N_QUERIES} (indices 0-${ARRAY_MAX})"
echo ""

# ---------------------------------------------------------------------------
# Stage 2: Worker array
# ---------------------------------------------------------------------------
echo "--- Stage 2: Worker array (${N_QUERIES} tasks, max ${MAX_ARRAY_CONCURRENT} concurrent) ---"

mkdir -p "${OUTDIR}/logs"
mkdir -p "${SHARDS_DIR}"

WORKER_JOB=$(sbatch --parsable \
    ${PREPARE_DEP:-} \
    --job-name=cov_worker \
    --account=crispr_carb \
    --partition=slurm \
    --time=08:00:00 \
    --mem=12G \
    --cpus-per-task=1 \
    --array="0-${ARRAY_MAX}%${MAX_ARRAY_CONCURRENT}" \
    --output="${OUTDIR}/logs/worker_%A_%a.out" \
    --error="${OUTDIR}/logs/worker_%A_%a.err" \
    --wrap="cd ${REPO_ROOT} && uv run python scripts/coverage_worker.py \
        --manifest  ${MANIFEST} \
        --shards-dir ${SHARDS_DIR} \
        --n-tasks   ${N_QUERIES}")

echo "  Worker array job ID: ${WORKER_JOB}"
echo ""

# ---------------------------------------------------------------------------
# Stage 3: Reduce (runs after all workers succeed)
# ---------------------------------------------------------------------------
echo "--- Stage 3: Reduce ---"

REDUCE_JOB=$(sbatch --parsable \
    --dependency=afterok:${WORKER_JOB} \
    --job-name=cov_reduce \
    --account=crispr_carb \
    --partition=slurm \
    --time=00:15:00 \
    --mem=4G \
    --cpus-per-task=1 \
    --output="${OUTDIR}/logs/reduce_%j.out" \
    --error="${OUTDIR}/logs/reduce_%j.err" \
    --wrap="cd ${REPO_ROOT} && uv run python scripts/coverage_reduce.py \
        --manifest   ${MANIFEST} \
        --shards-dir ${SHARDS_DIR} \
        --output     ${OUTPUT_CSV}")

echo "  Reduce job ID: ${REDUCE_JOB}"
echo ""
echo "=== Pipeline submitted ==="
echo "  Prepare:  (completed inline)"
echo "  Workers:  job ${WORKER_JOB}  (array 0-${ARRAY_MAX})"
echo "  Reduce:   job ${REDUCE_JOB}  (runs after workers)"
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo "  tail -f ${OUTDIR}/logs/worker_${WORKER_JOB}_0.out"
echo ""
echo "Final output: ${OUTPUT_CSV}"
