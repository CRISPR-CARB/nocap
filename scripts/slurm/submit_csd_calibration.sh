#!/usr/bin/env bash
# =============================================================================
# submit_csd_calibration.sh
# =============================================================================
# Calibration run: classify 128 sampled timeout edges at --timeout 1200 (20 min)
# to measure what fraction resolves with a longer budget.
#
# Topology: 2 exclusive 64-CPU nodes, 64 single-edge shards per node,
# each node runs xargs -P 64 so all 64 shards run in parallel.
# Wall time per node: 1 wave x 20 min max + headroom = 1:30:00.
#
# Usage (from repo root):
#   uv run python scripts/csd_calibrate_prepare.py
#   bash scripts/slurm/submit_csd_calibration.sh
#
# After both jobs complete, check results with:
#   uv run python scripts/csd_calibrate_report.py
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
OUTDIR="${REPO_ROOT}/results/cyclic_single_door"
CALIB_SHARDS_DIR="${OUTDIR}/calib_shards"
CALIB_CLASSIFIED_DIR="${OUTDIR}/calib_classified"
LOG_DIR="${OUTDIR}/logs"
GRAPHML="${REPO_ROOT}/notebooks/Ecoli_Analysis_Notebooks/ecoli_full_network_no_small_rna.graphml"

mkdir -p "${CALIB_CLASSIFIED_DIR}" "${LOG_DIR}"

# ---------------------------------------------------------------------------
# Build list of calibration shards that still need classifying
# ---------------------------------------------------------------------------
PENDING_LIST="${OUTDIR}/calib_pending.txt"
rm -f "${PENDING_LIST}"

for shard_file in "${CALIB_SHARDS_DIR}"/calib_*.json; do
    shard_id=$(basename "${shard_file}" .json)
    out_file="${CALIB_CLASSIFIED_DIR}/${shard_id}.json"
    if [[ ! -f "${out_file}" ]]; then
        echo "${shard_id}" >> "${PENDING_LIST}"
    fi
done

N_PENDING=$(wc -l < "${PENDING_LIST}" | tr -d ' ')
N_TOTAL=$(ls "${CALIB_SHARDS_DIR}"/calib_*.json 2>/dev/null | wc -l | tr -d ' ')
N_DONE=$(( N_TOTAL - N_PENDING ))

echo "=== CSD Calibration Submit (2 nodes, --timeout 1200) ==="
echo "  Total calib shards : ${N_TOTAL}"
echo "  Already done       : ${N_DONE}"
echo "  Pending            : ${N_PENDING}"
echo ""

if [[ "${N_PENDING}" -eq 0 ]]; then
    echo "All calibration shards already classified."
    echo "Run: uv run python scripts/csd_calibrate_report.py"
    exit 0
fi

# ---------------------------------------------------------------------------
# Split into exactly 2 batches (~64 shards each)
# ---------------------------------------------------------------------------
BATCH_SIZE=$(( (N_PENDING + 1) / 2 ))
BATCH_DIR="${OUTDIR}/calib_batches"
mkdir -p "${BATCH_DIR}"
rm -f "${BATCH_DIR}"/calib_batch_*.txt

split -l "${BATCH_SIZE}" --numeric-suffixes=0 --suffix-length=1 \
    "${PENDING_LIST}" "${BATCH_DIR}/calib_batch_"
for f in "${BATCH_DIR}"/calib_batch_[0-9]*; do
    mv "${f}" "${f}.txt"
done

SUBMITTED=()

for batch_file in "${BATCH_DIR}"/calib_batch_*.txt; do
    batch_id=$(basename "${batch_file}" .txt | sed 's/calib_batch_//')
    N_BATCH=$(wc -l < "${batch_file}" | tr -d ' ')
    job_log="${LOG_DIR}/calib_${batch_id}_%j"

    JOB_ID=$(sbatch --parsable \
        --job-name="csd_calib_${batch_id}" \
        --account=crispr_carb \
        --partition=slurm \
        --time=1:30:00 \
        --mem=0 \
        --exclusive \
        --cpus-per-task=1 \
        --output="${job_log}.out" \
        --error="${job_log}.err" \
        --wrap="
set -euo pipefail
export UV_CACHE_DIR=${REPO_ROOT}/.uv-cache
cd ${REPO_ROOT}
echo \"[calib ${batch_id}] started on \$(hostname) at \$(date)\"
echo \"[calib ${batch_id}] classifying \$(wc -l < ${batch_file}) shards at --timeout 1200\"

classify_one() {
    local sid=\"\$1\"
    local shard_file=\"${CALIB_SHARDS_DIR}/\${sid}.json\"
    local out_file=\"${CALIB_CLASSIFIED_DIR}/\${sid}.json\"
    if [[ -f \"\${out_file}\" ]]; then
        echo \"[skip] \${sid} already done\"
        return 0
    fi
    echo \"[start] \${sid}\"
    uv run python ${REPO_ROOT}/scripts/cyclic_single_door_classify.py classify \
        --graphml ${GRAPHML} \
        --shard \"\${shard_file}\" \
        --output \"\${out_file}\" \
        --timeout 1200 \
        && echo \"[done ] \${sid}\" \
        || echo \"[FAIL ] \${sid}\"
}

export -f classify_one
xargs -a ${batch_file} -P 64 -I{} bash -c 'classify_one \"\$@\"' _ {}

echo \"[calib ${batch_id}] finished at \$(date)\"
")
    echo "  Submitted batch ${batch_id}: job ${JOB_ID}  (${N_BATCH} shards)"
    SUBMITTED+=("${JOB_ID}")
done

echo ""
echo "=== Calibration jobs submitted ==="
echo "  Job IDs: ${SUBMITTED[*]}"
echo ""
echo "Monitor with:  squeue -u \$USER"
echo "When done run: uv run python scripts/csd_calibrate_report.py"
