#!/usr/bin/env bash
# =============================================================================
# submit_csd_rescue.sh
# =============================================================================
# Node-packing sweep for the CSD rescue analysis (diagnose non-identifiability
# cause + compute single-node do() rescue interventions for all unidentifiable
# edges in the E. coli cyclic causal graph).
#
# Workflow
# --------
#  1. Run csd_rescue_prepare.py  -- shards unidentifiable edges
#  2. Batch shards into node-packing groups (64 workers/node, same as CSD classify)
#  3. Each sbatch job runs csd_rescue_worker.py for its shard batch via xargs -P 64
#  4. After all jobs complete, run csd_rescue_gather.py to produce the final CSV
#
# Usage (from repo root):
#   bash scripts/slurm/submit_csd_rescue.sh
#
# Idempotent: shards whose output already exists in
# results/cyclic_single_door/rescue_classified/ are skipped automatically.
#
# Monitor with:
#   squeue -u $USER
#   ls results/cyclic_single_door/rescue_classified/ | wc -l
#
# After completion, gather with:
#   uv run python scripts/csd_rescue_gather.py \
#       --input-dir results/cyclic_single_door/rescue_classified \
#       --output-csv notebooks/Ecoli_Analysis_Notebooks/csd_rescue_results.csv \
#       --output-summary notebooks/Ecoli_Analysis_Notebooks/csd_rescue_summary.json
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
OUTDIR="${REPO_ROOT}/results/cyclic_single_door"
RESCUE_SHARDS_DIR="${OUTDIR}/rescue_shards"
RESCUE_CLASSIFIED_DIR="${OUTDIR}/rescue_classified"
LOG_DIR="${OUTDIR}/rescue_logs"
GRAPHML="${REPO_ROOT}/notebooks/Ecoli_Analysis_Notebooks/ecoli_full_network_no_small_rna.graphml"
RESULTS_CSV="${REPO_ROOT}/notebooks/Ecoli_Analysis_Notebooks/csd_results.csv"
MANIFEST="${OUTDIR}/rescue_manifest.json"

mkdir -p "${LOG_DIR}" "${RESCUE_SHARDS_DIR}" "${RESCUE_CLASSIFIED_DIR}"

# ---------------------------------------------------------------------------
# Step 1: Prepare shards (idempotent — skips if manifest already present)
# ---------------------------------------------------------------------------
if [[ ! -f "${MANIFEST}" ]]; then
    echo "=== Step 1: Preparing rescue shards ==="
    uv run python "${REPO_ROOT}/scripts/csd_rescue_prepare.py" \
        --results-csv "${RESULTS_CSV}" \
        --shard-dir "${RESCUE_SHARDS_DIR}" \
        --manifest "${MANIFEST}" \
        --shard-size 50
else
    echo "=== Step 1: Manifest already exists at ${MANIFEST}, skipping prepare ==="
fi

N_SHARDS=$(python3 -c "import json; d=json.load(open('${MANIFEST}')); print(d['n_shards'])")
echo "  Total rescue shards: ${N_SHARDS}"

# ---------------------------------------------------------------------------
# Step 2: Build pending list
# ---------------------------------------------------------------------------
PENDING_LIST="${OUTDIR}/rescue_pending.txt"
rm -f "${PENDING_LIST}"

for shard_file in "${RESCUE_SHARDS_DIR}"/shard_*.json; do
    shard_id=$(basename "${shard_file}" .json | sed 's/shard_//')
    out_file="${RESCUE_CLASSIFIED_DIR}/shard_${shard_id}.json"
    if [[ ! -f "${out_file}" ]]; then
        echo "${shard_id}" >> "${PENDING_LIST}"
    fi
done

N_PENDING=$(wc -l < "${PENDING_LIST}" | tr -d ' ')
N_DONE=$(( N_SHARDS - N_PENDING ))

echo ""
echo "=== Rescue Sweep Submit ==="
echo "  Total shards   : ${N_SHARDS}"
echo "  Already done   : ${N_DONE}"
echo "  Pending        : ${N_PENDING}"
echo ""

if [[ "${N_PENDING}" -eq 0 ]]; then
    echo "All rescue shards already done.  Nothing to submit."
    echo "Run csd_rescue_gather.py to produce the final CSV."
    exit 0
fi

# ---------------------------------------------------------------------------
# Step 3: Split into per-node batches (64 shards / node)
# ---------------------------------------------------------------------------
BATCH_SIZE=64
N_NODES=$(( (N_PENDING + BATCH_SIZE - 1) / BATCH_SIZE ))

echo "  Batches (nodes): ${N_NODES}  (${BATCH_SIZE} shards/node)"
echo ""

BATCH_DIR="${OUTDIR}/rescue_batches"
mkdir -p "${BATCH_DIR}"
rm -f "${BATCH_DIR}"/batch_*.txt
split -l "${BATCH_SIZE}" --numeric-suffixes=0 --suffix-length=3 \
    "${PENDING_LIST}" "${BATCH_DIR}/batch_"
for f in "${BATCH_DIR}"/batch_[0-9]*; do
    mv "${f}" "${f}.txt"
done

SUBMITTED=()

for batch_file in "${BATCH_DIR}"/batch_*.txt; do
    batch_id=$(basename "${batch_file}" .txt | sed 's/batch_//')
    job_log="${LOG_DIR}/rescue_pack_${batch_id}_%j"

    JOB_ID=$(sbatch --parsable \
        --job-name="csd_rescue_${batch_id}" \
        --account=crispr_carb \
        --partition=slurm \
        --time=8:00:00 \
        --mem=0 \
        --exclusive \
        --cpus-per-task=1 \
        --output="${job_log}.out" \
        --error="${job_log}.err" \
        --wrap="
set -euo pipefail
export UV_CACHE_DIR=${REPO_ROOT}/.uv-cache
cd ${REPO_ROOT}
echo \"[rescue pack ${batch_id}] started on \$(hostname) at \$(date)\"
echo \"[rescue pack ${batch_id}] processing \$(wc -l < ${batch_file}) shards in parallel...\"

rescue_one() {
    local sid=\"\$1\"
    local shard_file=\"${RESCUE_SHARDS_DIR}/shard_\${sid}.json\"
    local out_file=\"${RESCUE_CLASSIFIED_DIR}/shard_\${sid}.json\"
    if [[ -f \"\${out_file}\" ]]; then
        echo \"[skip] rescue shard \${sid} already done\"
        return 0
    fi
    echo \"[start] rescue shard \${sid}\"
    uv run python ${REPO_ROOT}/scripts/csd_rescue_worker.py \
        --graphml ${GRAPHML} \
        --shard \"\${shard_file}\" \
        --output \"\${out_file}\" \
        && echo \"[done ] rescue shard \${sid}\" \
        || echo \"[FAIL ] rescue shard \${sid}\"
}

export -f rescue_one
xargs -a ${batch_file} -P 64 -I{} bash -c 'rescue_one \"\$@\"' _ {}

echo \"[rescue pack ${batch_id}] all shards finished at \$(date)\"
")
    echo "  Submitted batch ${batch_id}: job ${JOB_ID}  ($(wc -l < "${batch_file}") shards)"
    SUBMITTED+=("${JOB_ID}")
done

echo ""
echo "=== All rescue batches submitted ==="
echo "  Job IDs: ${SUBMITTED[*]}"
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo "  ls ${RESCUE_CLASSIFIED_DIR}/ | wc -l   # completed shards"
echo ""
echo "Once all jobs complete, gather results with:"
echo "  uv run python scripts/csd_rescue_gather.py \\"
echo "      --input-dir ${RESCUE_CLASSIFIED_DIR} \\"
echo "      --output-csv notebooks/Ecoli_Analysis_Notebooks/csd_rescue_results.csv \\"
echo "      --output-summary notebooks/Ecoli_Analysis_Notebooks/csd_rescue_summary.json"
