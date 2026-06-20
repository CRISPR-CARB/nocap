#!/usr/bin/env bash
# =============================================================================
# submit_scc_perturb.sh
# =============================================================================
# Orchestrates the full SCC-perturbation pipeline on a SLURM cluster.
#
# Three-stage pipeline with automatic job dependencies:
#   1. prepare  -- serial; computes B(t) for each TF, writes scc_perturb_job.json
#   2. workers  -- SLURM array; one task per TF (one joint cyclic_id call each)
#   3. reduce   -- serial; merges shards -> scc_perturbation_results.csv
#
# Usage (from repo root):
#   bash scripts/slurm/submit_scc_perturb.sh
#
# Or with custom paths:
#   GRAPHML=/path/to/network.graphml \
#   SUPPTABLE=/path/to/supptable1.csv \
#   OUTDIR=/path/to/output \
#   bash scripts/slurm/submit_scc_perturb.sh
#
# Environment variables (all have defaults):
#   GRAPHML      path to .graphml network file
#   SUPPTABLE    path to supptable1.csv
#   OUTDIR       directory for manifest, shards, and final CSV
#   REPO_ROOT    repo root (default: directory containing this script's parent)
#   MAX_ARRAY_CONCURRENT  max simultaneous array tasks (default: 20)
#   PER_GENE_ON_FAILURE   set to "1" to enable per-gene fallback in workers
#
# =============================================================================
# SLURM RESOURCE PLACEHOLDERS
# Edit the #SBATCH lines below to match your cluster's partition/queue names
# and resource limits.  Each worker runs one joint cyclic_id call — typically
# <1 min for small SCCs, up to ~30 min for a TF in a large SCC with many
# descendants.  4G RAM per worker is conservative.
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

MANIFEST="${OUTDIR}/scc_perturb_job.json"
SHARDS_DIR="${OUTDIR}/scc_perturb_shards"
OUTPUT_CSV="${OUTDIR}/scc_perturbation_results.csv"

MAX_ARRAY_CONCURRENT="${MAX_ARRAY_CONCURRENT:-20}"
PER_GENE_ON_FAILURE="${PER_GENE_ON_FAILURE:-0}"

echo "=== SCC Perturbation Pipeline ==="
echo "  REPO_ROOT:  ${REPO_ROOT}"
echo "  GRAPHML:    ${GRAPHML}"
echo "  SUPPTABLE:  ${SUPPTABLE}"
echo "  OUTDIR:     ${OUTDIR}"
echo "  MANIFEST:   ${MANIFEST}"
echo "  SHARDS_DIR: ${SHARDS_DIR}"
echo "  OUTPUT_CSV: ${OUTPUT_CSV}"
echo "  PER_GENE_ON_FAILURE: ${PER_GENE_ON_FAILURE}"
echo ""

# ---------------------------------------------------------------------------
# Stage 1: Prepare (serial, runs on login node)
# ---------------------------------------------------------------------------
# Computes B(t) for every TF via local SCC min-cut (fast, networkx only).
# Idempotent: skip if manifest already exists unless FORCE_PREPARE=1.

echo "--- Stage 1: Prepare ---"

if [[ -f "${MANIFEST}" && "${FORCE_PREPARE:-0}" != "1" ]]; then
    echo "  Manifest already exists — skipping prepare (set FORCE_PREPARE=1 to override)."
    echo "  ${MANIFEST}"
else
    echo "  Running prepare..."
    uv run python "${REPO_ROOT}/scripts/scc_perturb_prepare.py" \
        --graphml   "${GRAPHML}" \
        --supptable "${SUPPTABLE}" \
        --manifest  "${MANIFEST}"
fi

# ---------------------------------------------------------------------------
# Read array size from manifest
# ---------------------------------------------------------------------------
if [[ ! -f "${MANIFEST}" ]]; then
    echo "ERROR: Manifest not found after prepare step: ${MANIFEST}"
    exit 1
fi

N_TASKS=$(python3 -c "import json; d=json.load(open('${MANIFEST}')); print(d['n_tasks'])")

if [[ "${N_TASKS}" -eq 0 ]]; then
    echo "No TFs require SCC perturbation — nothing to submit."
    exit 0
fi

ARRAY_MAX=$((N_TASKS - 1))
echo "  Array size: ${N_TASKS} (indices 0-${ARRAY_MAX})"
echo ""

# ---------------------------------------------------------------------------
# Stage 2: Worker array (one task per TF)
# ---------------------------------------------------------------------------
echo "--- Stage 2: Worker array (${N_TASKS} tasks, max ${MAX_ARRAY_CONCURRENT} concurrent) ---"

mkdir -p "${OUTDIR}/logs"
mkdir -p "${SHARDS_DIR}"

# Build optional per-gene flag
PER_GENE_FLAG=""
if [[ "${PER_GENE_ON_FAILURE}" == "1" ]]; then
    PER_GENE_FLAG="--per-gene-on-failure"
fi

WORKER_JOB=$(sbatch --parsable \
    --job-name=scc_worker \
    --account=crispr_carb \
    --partition=slurm \
    --time=02:00:00 \
    --mem=8G \
    --cpus-per-task=1 \
    --array="0-${ARRAY_MAX}%${MAX_ARRAY_CONCURRENT}" \
    --output="${OUTDIR}/logs/scc_worker_%A_%a.out" \
    --error="${OUTDIR}/logs/scc_worker_%A_%a.err" \
    --wrap="cd ${REPO_ROOT} && uv run python scripts/scc_perturb_worker.py \
        --manifest   ${MANIFEST} \
        --shards-dir ${SHARDS_DIR} \
        --n-tasks    ${N_TASKS} \
        ${PER_GENE_FLAG}")

echo "  Worker array job ID: ${WORKER_JOB}"
echo ""

# ---------------------------------------------------------------------------
# Stage 3: Reduce (runs after all workers succeed)
# ---------------------------------------------------------------------------
echo "--- Stage 3: Reduce ---"

REDUCE_JOB=$(sbatch --parsable \
    --dependency=afterok:${WORKER_JOB} \
    --job-name=scc_reduce \
    --account=crispr_carb \
    --partition=slurm \
    --time=00:10:00 \
    --mem=2G \
    --cpus-per-task=1 \
    --output="${OUTDIR}/logs/scc_reduce_%j.out" \
    --error="${OUTDIR}/logs/scc_reduce_%j.err" \
    --wrap="cd ${REPO_ROOT} && uv run python scripts/scc_perturb_reduce.py \
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
echo "  tail -f ${OUTDIR}/logs/scc_worker_${WORKER_JOB}_0.out"
echo ""
echo "Final output: ${OUTPUT_CSV}"
