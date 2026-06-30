#!/usr/bin/env bash
# =============================================================================
# check_csd_progress.sh
# =============================================================================
# Quick status check for the cyclic-single-door classify sweep.
#
# Sections
# --------
#   1. Shard progress bar  (done / total / %)
#   2. Live SLURM job status
#   3. FAIL markers in packed-job stdout logs
#   4. Python tracebacks in .err logs (last 20 lines each)
#   5. Live edge tally from completed classified shards
#   6. Tail of the most recent packed job log
#   7. Gather command (once all shards are done)
#
# Usage:
#   bash scripts/check_csd_progress.sh
# =============================================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUTDIR="${REPO_ROOT}/results/cyclic_single_door"
CLASSIFIED_DIR="${OUTDIR}/classified"
SHARDS_DIR="${OUTDIR}/shards"
LOG_DIR="${OUTDIR}/logs"

# ---------------------------------------------------------------------------
# 1. Shard progress
# ---------------------------------------------------------------------------
N_TOTAL=$(ls "${SHARDS_DIR}"/shard_*.json 2>/dev/null | wc -l | tr -d ' ')
N_DONE=$(ls "${CLASSIFIED_DIR}"/shard_*.json 2>/dev/null | wc -l | tr -d ' ')
N_PENDING=$(( N_TOTAL - N_DONE ))

echo "=== CSD Classify Progress ==="
if [[ "${N_TOTAL}" -gt 0 ]]; then
    printf "  Classified : %d / %d  (%d%%)\n" "${N_DONE}" "${N_TOTAL}" "$(( N_DONE * 100 / N_TOTAL ))"
    printf "  Remaining  : %d\n" "${N_PENDING}"
else
    echo "  (no shards found in ${SHARDS_DIR})"
fi
echo ""

# ---------------------------------------------------------------------------
# 2. SLURM job status
# ---------------------------------------------------------------------------
echo "=== SLURM job status ==="
JOB_LINES=$(squeue -u "$USER" -o "%.12i %.12j %.8T %.10M %.12l %R" 2>/dev/null | grep -E "(csd_pack|JOBID)" || true)
if [[ -n "${JOB_LINES}" ]]; then
    echo "${JOB_LINES}"
else
    echo "  (no csd_pack jobs in queue)"
fi
echo ""

# ---------------------------------------------------------------------------
# 3. FAIL markers in packed-job stdout logs
# ---------------------------------------------------------------------------
echo "=== Failed shards (from packed job logs) ==="
FAIL_LINES=$(grep -h "\[FAIL \]" "${LOG_DIR}"/packed_*.out 2>/dev/null || true)
if [[ -n "${FAIL_LINES}" ]]; then
    echo "  FAILED shards detected:"
    echo "${FAIL_LINES}" | sed 's/^/    /'
    N_FAIL=$(echo "${FAIL_LINES}" | wc -l | tr -d ' ')
    echo "  Total FAIL markers: ${N_FAIL}"
else
    echo "  No [FAIL] markers found in logs so far — good."
fi
echo ""

# ---------------------------------------------------------------------------
# 4. Python tracebacks in .err logs
# ---------------------------------------------------------------------------
echo "=== Tracebacks in .err logs ==="
ERR_FILES_WITH_TB=$(grep -l "Traceback" "${LOG_DIR}"/packed_*.err 2>/dev/null || true)
if [[ -n "${ERR_FILES_WITH_TB}" ]]; then
    for f in ${ERR_FILES_WITH_TB}; do
        echo "  --- ${f} (last 20 lines) ---"
        tail -20 "${f}" | sed 's/^/    /'
        echo ""
    done
else
    echo "  No Traceback found in any .err log so far."
fi
echo ""

# ---------------------------------------------------------------------------
# 5. Live edge tally from completed classified shards (uses csd_identified_edges.py)
# ---------------------------------------------------------------------------
echo "=== Live edge tally (completed shards so far) ==="
if [[ "${N_DONE}" -gt 0 ]]; then
    uv run python "${REPO_ROOT}/scripts/csd_identified_edges.py" \
        --classified-dir "${CLASSIFIED_DIR}" \
        2>/dev/null || echo "  (csd_identified_edges.py not yet available or error)"
else
    echo "  (no classified shards yet)"
fi
echo ""

# ---------------------------------------------------------------------------
# 6. Tail of the most recent packed job log
# ---------------------------------------------------------------------------
echo "=== Recent log tail (most recent packed job stdout) ==="
LATEST_LOG=$(ls -t "${LOG_DIR}"/packed_*.out 2>/dev/null | head -1 || true)
if [[ -n "${LATEST_LOG}" ]]; then
    echo "  ${LATEST_LOG}"
    tail -8 "${LATEST_LOG}" 2>/dev/null | sed 's/^/  /' || true
else
    echo "  (no packed job logs yet)"
fi
echo ""

# ---------------------------------------------------------------------------
# 7. Gather command (only once all shards are done)
# ---------------------------------------------------------------------------
if [[ "${N_PENDING}" -eq 0 && "${N_TOTAL}" -gt 0 ]]; then
    echo "=== ALL SHARDS DONE — ready to gather ==="
    echo ""
    echo "Run:"
    echo "  uv run python scripts/cyclic_single_door_gather.py \\"
    echo "      --classified-dir results/cyclic_single_door/classified \\"
    echo "      --manifest results/cyclic_single_door/shard_manifest.json \\"
    echo "      --output results/cyclic_single_door/classification_results.csv"
    echo ""
    echo "Or check identified edges now:"
    echo "  uv run python scripts/csd_identified_edges.py --classified-dir results/cyclic_single_door/classified --list | head -40"
fi
