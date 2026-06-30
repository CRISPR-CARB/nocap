#!/usr/bin/env bash
# =============================================================================
# submit_csd_recovery.sh
# =============================================================================
# Run the CSD recovery bank analysis on a single compute node.
#
# The greedy search is inherently sequential (each gene pick depends on the
# previous), so this runs as a single srun task — off the login node but not
# distributed.  Two runs are submitted as sequential job-array steps:
#   1. n=10, k=3 (10 experiments of 3 simultaneous knockouts)
#   2. n=5,  k=6 (5 experiments of 6 simultaneous knockouts)
#
# Both runs share the same break CSV and graphml; the second run starts after
# the first completes (--dependency=afterok).
#
# Prerequisites:
#   - submit_csd_break_full.sh has completed and csd_break_gather.py has run,
#     producing notebooks/Ecoli_Analysis_Notebooks/csd_break_results_full.csv
#
# Usage (from repo root):
#   bash scripts/slurm/submit_csd_recovery.sh
#
# Outputs (in notebooks/Ecoli_Analysis_Notebooks/):
#   csd_recovery_n10_k3.csv
#   csd_recovery_edges_n10_k3.csv
#   csd_recovery_n5_k6.csv
#   csd_recovery_edges_n5_k6.csv
#   csd_recovery_summary.json
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
OUTDIR="${REPO_ROOT}/notebooks/Ecoli_Analysis_Notebooks"
LOG_DIR="${REPO_ROOT}/results/cyclic_single_door/recovery_logs"
GRAPHML="${REPO_ROOT}/notebooks/Ecoli_Analysis_Notebooks/ecoli_full_network_no_small_rna.graphml"
CLASSIFICATION="${REPO_ROOT}/results/cyclic_single_door/classification_results.csv"
BREAK_CSV="${REPO_ROOT}/notebooks/Ecoli_Analysis_Notebooks/csd_break_results_full.csv"

mkdir -p "${LOG_DIR}"

# ---------------------------------------------------------------------------
# Validate prerequisites
# ---------------------------------------------------------------------------
if [[ ! -f "${BREAK_CSV}" ]]; then
    echo "ERROR: break CSV not found: ${BREAK_CSV}"
    echo "Run submit_csd_break_full.sh and then csd_break_gather.py first:"
    echo "  uv run python scripts/csd_break_gather.py \\"
    echo "      --input-dir results/cyclic_single_door/break_classified_full \\"
    echo "      --output-csv notebooks/Ecoli_Analysis_Notebooks/csd_break_results_full.csv \\"
    echo "      --output-summary notebooks/Ecoli_Analysis_Notebooks/csd_break_summary_full.json"
    exit 1
fi

if [[ ! -f "${CLASSIFICATION}" ]]; then
    echo "ERROR: classification results not found: ${CLASSIFICATION}"
    exit 1
fi

BREAK_ROWS=$(python3 -c "
import csv
n = sum(1 for _ in open('${BREAK_CSV}').readlines()) - 1
print(n)
" 2>/dev/null || echo 0)

echo "=== CSD Recovery Bank Submit ==="
echo "  Break CSV rows: ${BREAK_ROWS}"
echo "  Classification: ${CLASSIFICATION}"
echo "  Output dir:     ${OUTDIR}"
echo ""

# ---------------------------------------------------------------------------
# Job 1: n=10, k=3
# ---------------------------------------------------------------------------
JOB1=$(sbatch --parsable \
    --job-name="csd_recovery_n10_k3" \
    --account=crispr_carb \
    --partition=slurm \
    --time=4:00:00 \
    --mem=32G \
    --cpus-per-task=4 \
    --output="${LOG_DIR}/recovery_n10_k3_%j.out" \
    --error="${LOG_DIR}/recovery_n10_k3_%j.err" \
    --wrap="
set -euo pipefail
export UV_CACHE_DIR=${REPO_ROOT}/.uv-cache
cd ${REPO_ROOT}
echo \"[recovery n=10 k=3] started on \$(hostname) at \$(date)\"
uv run python ${REPO_ROOT}/scripts/csd_recovery_bank.py \\
    --graphml ${GRAPHML} \\
    --classification ${CLASSIFICATION} \\
    --break-csv ${BREAK_CSV} \\
    --n 10 --k 3 \\
    --output-dir ${OUTDIR}
echo \"[recovery n=10 k=3] finished at \$(date)\"
")

echo "  Submitted n=10 k=3: job ${JOB1}"

# ---------------------------------------------------------------------------
# Job 2: n=5, k=6 (after job 1 completes)
# ---------------------------------------------------------------------------
JOB2=$(sbatch --parsable \
    --job-name="csd_recovery_n5_k6" \
    --account=crispr_carb \
    --partition=slurm \
    --time=4:00:00 \
    --mem=32G \
    --cpus-per-task=4 \
    --dependency="afterok:${JOB1}" \
    --output="${LOG_DIR}/recovery_n5_k6_%j.out" \
    --error="${LOG_DIR}/recovery_n5_k6_%j.err" \
    --wrap="
set -euo pipefail
export UV_CACHE_DIR=${REPO_ROOT}/.uv-cache
cd ${REPO_ROOT}
echo \"[recovery n=5 k=6] started on \$(hostname) at \$(date)\"
uv run python ${REPO_ROOT}/scripts/csd_recovery_bank.py \\
    --graphml ${GRAPHML} \\
    --classification ${CLASSIFICATION} \\
    --break-csv ${BREAK_CSV} \\
    --n 5 --k 6 \\
    --output-dir ${OUTDIR}
echo \"[recovery n=5 k=6] finished at \$(date)\"
")

echo "  Submitted n=5  k=6: job ${JOB2} (depends on ${JOB1})"
echo ""
echo "=== Both recovery jobs submitted ==="
echo "  Job IDs: ${JOB1} (n=10,k=3)  ${JOB2} (n=5,k=6)"
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo "  cat ${LOG_DIR}/recovery_n10_k3_${JOB1}.out | tail -20"
echo ""
echo "Once both complete, build the notebook:"
echo "  uv run python scripts/build_csd_recovery_notebook.py"
echo "  uv run python scripts/smoke_csd_recovery_notebook.py"
