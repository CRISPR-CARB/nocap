#!/usr/bin/env bash
# =============================================================================
# submit_csd_packed.sh
# =============================================================================
# Node-packing resubmit for the cyclic-single-door classify sweep.
#
# Problem with the previous approach
# -----------------------------------
# The slurm partition allocates WHOLE NODES exclusively.  Each old job
# requested 1 CPU but Slurm allocated all 64 (AllocTRES=cpu=64, billing=64).
# Because the classify task is single-threaded Python (serial loop, GIL-bound),
# 63 of those CPUs sat idle while the full node was billed against
# GrpTRESMins — creating a 64× budget multiplier that throttled the queue.
#
# Solution: node-packing
# ----------------------
# Since we pay for a whole 64-CPU node anyway, run up to 64 shards IN PARALLEL
# on that one node using `xargs -P 64`.  Each shard still uses 1 CPU, but now
# we extract full value from every billed node.  179 shards → 3 node-allocations
# instead of 179, finishing in roughly one shard's wall-time.
#
# Budget maths
# ------------
# Previous: 179 jobs × 64 billed CPUs × 12 h walltime = 137,088 CPU·h
# Packed:     3 jobs × 64 billed CPUs ×  6 h walltime =   1,152 CPU·h
# (actual runtime ≈ 1 h/shard; we request 6 h as comfortable headroom)
#
# Usage (from repo root):
#   bash scripts/slurm/submit_csd_packed.sh
#
# The script is idempotent: shards whose output already exists in
# results/cyclic_single_door/classified/ are skipped automatically.
#
# Monitor with:
#   bash scripts/check_csd_progress.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
OUTDIR="${REPO_ROOT}/results/cyclic_single_door"
CLASSIFIED_DIR="${OUTDIR}/classified"
SHARDS_DIR="${OUTDIR}/shards"
LOG_DIR="${OUTDIR}/logs"
GRAPHML="${REPO_ROOT}/notebooks/Ecoli_Analysis_Notebooks/ecoli_full_network_no_small_rna.graphml"

mkdir -p "${LOG_DIR}"

# ---------------------------------------------------------------------------
# Build list of shards that still need classifying (skip completed ones)
# ---------------------------------------------------------------------------
PENDING_LIST="${OUTDIR}/pending_shards.txt"
rm -f "${PENDING_LIST}"

for shard_file in "${SHARDS_DIR}"/shard_*.json; do
    shard_id=$(basename "${shard_file}" .json | sed 's/shard_//')
    out_file="${CLASSIFIED_DIR}/shard_${shard_id}.json"
    if [[ ! -f "${out_file}" ]]; then
        echo "${shard_id}" >> "${PENDING_LIST}"
    fi
done

N_PENDING=$(wc -l < "${PENDING_LIST}" | tr -d ' ')
N_TOTAL=$(ls "${SHARDS_DIR}"/shard_*.json | wc -l | tr -d ' ')
N_DONE=$(( N_TOTAL - N_PENDING ))

echo "=== Cyclic Single-Door Node-Packing Submit ==="
echo "  Total shards   : ${N_TOTAL}"
echo "  Already done   : ${N_DONE}"
echo "  Pending        : ${N_PENDING}"
echo ""

if [[ "${N_PENDING}" -eq 0 ]]; then
    echo "All shards already classified.  Nothing to submit."
    echo "Run cyclic_single_door_gather.py to produce the final CSV."
    exit 0
fi

# ---------------------------------------------------------------------------
# Split pending shards into batches of 64 (one batch per node, ~32 nodes)
# ---------------------------------------------------------------------------
BATCH_SIZE=64
N_NODES=$(( (N_PENDING + BATCH_SIZE - 1) / BATCH_SIZE ))

echo "  Batches (nodes): ${N_NODES}  (${BATCH_SIZE} shards/node)"
echo ""

# Split into per-node batch files
BATCH_DIR="${OUTDIR}/batches"
mkdir -p "${BATCH_DIR}"
rm -f "${BATCH_DIR}"/batch_*.txt
split -l "${BATCH_SIZE}" --numeric-suffixes=0 --suffix-length=3 \
    "${PENDING_LIST}" "${BATCH_DIR}/batch_"
# rename to .txt
for f in "${BATCH_DIR}"/batch_[0-9]*; do
    mv "${f}" "${f}.txt"
done

SUBMITTED=()

for batch_file in "${BATCH_DIR}"/batch_*.txt; do
    batch_id=$(basename "${batch_file}" .txt | sed 's/batch_//')
    job_log="${LOG_DIR}/packed_${batch_id}_%j"

    JOB_ID=$(sbatch --parsable \
        --job-name="csd_pack_${batch_id}" \
        --account=crispr_carb \
        --partition=slurm \
        --time=6:00:00 \
        --mem=0 \
        --exclusive \
        --cpus-per-task=1 \
        --output="${job_log}.out" \
        --error="${job_log}.err" \
        --wrap="
set -euo pipefail
export UV_CACHE_DIR=${REPO_ROOT}/.uv-cache
cd ${REPO_ROOT}
echo \"[pack ${batch_id}] started on \$(hostname) at \$(date)\"
echo \"[pack ${batch_id}] classifying \$(wc -l < ${batch_file}) shards in parallel...\"

classify_one() {
    local sid=\"\$1\"
    local shard_file=\"${SHARDS_DIR}/shard_\${sid}.json\"
    local out_file=\"${CLASSIFIED_DIR}/shard_\${sid}.json\"
    if [[ -f \"\${out_file}\" ]]; then
        echo \"[skip] shard \${sid} already done\"
        return 0
    fi
    echo \"[start] shard \${sid}\"
    uv run python ${REPO_ROOT}/scripts/cyclic_single_door_classify.py classify \
        --graphml ${GRAPHML} \
        --shard \"\${shard_file}\" \
        --output \"\${out_file}\" \
        --timeout 120 \
        && echo \"[done ] shard \${sid}\" \
        || echo \"[FAIL ] shard \${sid}\"
}

export -f classify_one
xargs -a ${batch_file} -P 64 -I{} bash -c 'classify_one \"\$@\"' _ {}

echo \"[pack ${batch_id}] all shards finished at \$(date)\"
")
    echo "  Submitted batch ${batch_id}: job ${JOB_ID}  ($(wc -l < "${batch_file}") shards)"
    SUBMITTED+=("${JOB_ID}")
done

echo ""
echo "=== All batches submitted ==="
echo "  Job IDs: ${SUBMITTED[*]}"
echo ""
echo "Monitor with:"
echo "  bash scripts/check_csd_progress.sh"
echo "  squeue -u \$USER"
echo ""
echo "Once all jobs complete, gather results with:"
echo "  uv run python scripts/cyclic_single_door_gather.py \\"
echo "      --classified-dir results/cyclic_single_door/classified \\"
echo "      --manifest results/cyclic_single_door/shard_manifest.json \\"
echo "      --output results/cyclic_single_door/classification_results.csv"
