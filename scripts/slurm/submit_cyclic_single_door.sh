#!/usr/bin/env bash
# =============================================================================
# submit_cyclic_single_door.sh
# =============================================================================
# Submits the full cyclic-single-door σ-separation classification workflow as
# a self-contained SLURM job that runs Snakemake as the workflow controller.
#
# The Snakemake controller runs on a compute node (so nothing hangs in your
# terminal) and itself submits the per-shard classify jobs via the SLURM
# executor plugin.  When every shard is classified the controller runs gather
# and exits cleanly.
#
# Pipeline stages (managed by Snakemake):
#   prepare   → shard the edge list into JSON shards
#   classify  → scatter: one 12-hour job per shard (n_shards = idle nodes at submit time)
#   gather    → merge all classified shards into CSV + summary JSON
#
# Usage (from repo root):
#   bash scripts/slurm/submit_cyclic_single_door.sh
#
# Override environment variables if needed:
#   REPO_ROOT    path to repo root (default: two levels up from this script)
#   OUTDIR       output directory  (default: results/cyclic_single_door)
#   SNAKEFILE    path to Snakefile (default: workflow/Snakefile)
#   CONFIG       path to config    (default: workflow/config.yaml)
#
# Monitor with:
#   squeue -u $USER
#   tail -f results/cyclic_single_door/logs/controller_<JOBID>.out
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
OUTDIR="${OUTDIR:-${REPO_ROOT}/results/cyclic_single_door}"
CONFIG="${CONFIG:-${REPO_ROOT}/workflow/config.yaml}"

LOG_DIR="${OUTDIR}/logs"
mkdir -p "${LOG_DIR}"

echo "=== Cyclic Single-Door Classification Workflow ==="
echo "  REPO_ROOT:  ${REPO_ROOT}"
echo "  OUTDIR:     ${OUTDIR}"
echo "  CONFIG:     ${CONFIG}"
echo "  LOG_DIR:    ${LOG_DIR}"
echo ""

# ---------------------------------------------------------------------------
# Submit the Snakemake controller as a SLURM job
# Controller walltime must exceed the longest classify child job (720 min).
# We use 16 h = 960 min to be safe.
# ---------------------------------------------------------------------------
CONTROLLER_JOB=$(sbatch --parsable \
    --job-name=csd_ctl \
    --account=crispr_carb \
    --partition=slurm \
    --time=16:00:00 \
    --mem=4G \
    --cpus-per-task=1 \
    --output="${LOG_DIR}/controller_%j.out" \
    --error="${LOG_DIR}/controller_%j.err" \
    --wrap="
set -euo pipefail
export UV_CACHE_DIR=${REPO_ROOT}/.uv-cache
cd ${REPO_ROOT}
echo \"[controller] started on \$(hostname) at \$(date)\"
uv run snakemake \
    --profile workflow/profiles/slurm \
    --configfile ${CONFIG}
echo \"[controller] finished at \$(date)\"
")

echo "=== Workflow submitted ==="
echo "  Controller job ID: ${CONTROLLER_JOB}"
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo "  tail -f ${LOG_DIR}/controller_${CONTROLLER_JOB}.out"
echo ""
echo "Final outputs (once complete):"
echo "  ${OUTDIR}/classification_results.csv"
echo "  ${OUTDIR}/classification_summary.json"
