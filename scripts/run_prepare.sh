#!/usr/bin/env bash
# =============================================================================
# run_prepare.sh
# =============================================================================
# Serial preparation step for the coverage-matrix pipeline.
# Runs coverage_prepare.py with Phase 1 parallelised across all allocated CPUs.
#
# Submit as a SLURM job:
#   sbatch scripts/run_prepare.sh
#
# Or run directly on the login node (uses all local CPUs):
#   bash scripts/run_prepare.sh
#
# Override paths via environment variables:
#   GRAPHML=/path/to/network.graphml bash scripts/run_prepare.sh
# =============================================================================
#SBATCH --job-name=cov_prepare
#SBATCH --account=crispr_carb
#SBATCH --partition=slurm
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=16
#SBATCH --output=notebooks/Ecoli_Analysis_Notebooks/logs/prepare_%j.out
#SBATCH --error=notebooks/Ecoli_Analysis_Notebooks/logs/prepare_%j.err

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

GRAPHML="${GRAPHML:-notebooks/Ecoli_Analysis_Notebooks/ecoli_full_network_no_small_rna.graphml}"
SUPPTABLE="${SUPPTABLE:-notebooks/Ecoli_Analysis_Notebooks/supptable1.csv}"
MANIFEST="${MANIFEST:-notebooks/Ecoli_Analysis_Notebooks/coverage_job.json}"

# When running under SLURM use the allocated CPU count; fall back to all local CPUs.
N_WORKERS="${SLURM_CPUS_PER_TASK:-$(python3 -c 'import os; print(os.cpu_count())')}"

mkdir -p notebooks/Ecoli_Analysis_Notebooks/logs

echo "=== Coverage Prepare ==="
echo "  REPO_ROOT:  ${REPO_ROOT}"
echo "  GRAPHML:    ${GRAPHML}"
echo "  SUPPTABLE:  ${SUPPTABLE}"
echo "  MANIFEST:   ${MANIFEST}"
echo "  N_WORKERS:  ${N_WORKERS}"
echo ""

uv run python scripts/coverage_prepare.py \
    --graphml   "${GRAPHML}" \
    --supptable "${SUPPTABLE}" \
    --manifest  "${MANIFEST}" \
    --n-workers "${N_WORKERS}"
