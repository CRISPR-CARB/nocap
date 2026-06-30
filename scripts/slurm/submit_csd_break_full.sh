#!/usr/bin/env bash
# =============================================================================
# submit_csd_break_full.sh
# =============================================================================
# Re-run the CSD SCC-break sweep on ALL edges from the fast-O-set run
# (classification_results.csv, 6676 unidentifiable edges).
#
# The previous break sweep (submit_csd_break.sh) was run against the old
# csd_results.csv and covered only ~330 edges.  This script clears the stale
# break outputs and re-runs against the new classification_results.csv.
#
# Workflow
# --------
#  1. Clear stale break manifest, shards, and classified outputs so the run
#     starts fresh from classification_results.csv.
#  2. Size shards from idle cluster nodes (adaptive node-packing, same logic
#     as submit_csd_break.sh: D' nodes × 64 workers each).
#  3. Run csd_break_prepare.py -- shards all 6676 unidentifiable edges.
#  4. Batch shards into per-node groups (64 workers/node via xargs -P 64).
#  5. Each sbatch job runs csd_break_worker.py --k 3 for its shard batch.
#  6. After all jobs complete, run csd_break_gather.py to emit the full CSV.
#
# Usage (from repo root):
#   bash scripts/slurm/submit_csd_break_full.sh
#
# After completion, gather results with:
#   uv run python scripts/csd_break_gather.py \
#       --input-dir results/cyclic_single_door/break_classified_full \
#       --output-csv notebooks/Ecoli_Analysis_Notebooks/csd_break_results_full.csv \
#       --output-summary notebooks/Ecoli_Analysis_Notebooks/csd_break_summary_full.json
#
# Then run the recovery bank:
#   bash scripts/slurm/submit_csd_recovery.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
OUTDIR="${REPO_ROOT}/results/cyclic_single_door"

# Use the new fast-O-set classification results (6676 unidentifiable edges)
RESULTS_CSV="${REPO_ROOT}/results/cyclic_single_door/classification_results.csv"

# Separate output dirs so old 330-edge results are not overwritten
BREAK_SHARDS_DIR="${OUTDIR}/break_shards_full"
BREAK_CLASSIFIED_DIR="${OUTDIR}/break_classified_full"
LOG_DIR="${OUTDIR}/break_logs_full"
MANIFEST="${OUTDIR}/break_manifest_full.json"

GRAPHML="${REPO_ROOT}/notebooks/Ecoli_Analysis_Notebooks/ecoli_full_network_no_small_rna.graphml"

# Budget k: edge "rescuable" iff min_break_size <= K
K=3

# Node-packing parameters
WORKERS_PER_NODE=64
MAX_NODES=64

mkdir -p "${LOG_DIR}" "${BREAK_SHARDS_DIR}" "${BREAK_CLASSIFIED_DIR}"

# ---------------------------------------------------------------------------
# Step 0: Compute U and adaptive shard count
# ---------------------------------------------------------------------------
echo "=== Step 0: Sizing shards based on idle cluster nodes ==="

U=$(python3 -c "
import csv
n = sum(1 for r in csv.DictReader(open('${RESULTS_CSV}')) if r['status']=='unidentifiable')
print(n)
" 2>/dev/null || echo 0)

echo "  Unidentifiable edges U = ${U}"

if [[ "${U}" -le 0 ]]; then
    echo "  ERROR: no unidentifiable edges found in ${RESULTS_CSV}; nothing to do."
    exit 1
fi

D=$(sinfo -h -p slurm -t idle -o "%D" 2>/dev/null | awk '{s+=$1} END{print s+0}')
echo "  Strictly idle nodes D = ${D}"

if [[ "${D}" -lt 1 ]]; then
    echo "  WARNING: no idle nodes detected (sinfo unavailable or all nodes busy)."
    echo "  Falling back to static --shard-size 25 (approx 267 shards for 6676 edges)."
    PREPARE_ARGS=(--shard-size 25)
    DPRIME=0
else
    FILLABLE=$(( U / WORKERS_PER_NODE ))
    (( FILLABLE < 1 )) && FILLABLE=1
    DPRIME=${D}
    (( DPRIME > FILLABLE  )) && DPRIME=${FILLABLE}
    (( DPRIME > MAX_NODES )) && DPRIME=${MAX_NODES}
    (( DPRIME < 1         )) && DPRIME=1

    N_SHARDS_TARGET=$(( DPRIME * WORKERS_PER_NODE ))
    (( N_SHARDS_TARGET > U )) && N_SHARDS_TARGET=${U}

    echo "  Nodes used D' = ${DPRIME}  (cap: ${MAX_NODES})"
    echo "  Target shards  = ${N_SHARDS_TARGET}  (${WORKERS_PER_NODE} shards/node x D')"
    echo "  Approx shard size ~ $(( (U + N_SHARDS_TARGET - 1) / N_SHARDS_TARGET )) edges/shard"
    PREPARE_ARGS=(--n-shards "${N_SHARDS_TARGET}")
fi

echo ""

# ---------------------------------------------------------------------------
# Step 1: Prepare shards
# ---------------------------------------------------------------------------
if [[ ! -f "${MANIFEST}" ]]; then
    echo "=== Step 1: Preparing break shards (full run, ${U} edges) ==="
    uv run python "${REPO_ROOT}/scripts/csd_break_prepare.py" \
        --results-csv "${RESULTS_CSV}" \
        --shard-dir "${BREAK_SHARDS_DIR}" \
        --manifest "${MANIFEST}" \
        "${PREPARE_ARGS[@]}"
else
    echo "=== Step 1: Manifest already exists at ${MANIFEST}, skipping prepare ==="
fi

N_SHARDS=$(python3 -c "import json; d=json.load(open('${MANIFEST}')); print(d['n_shards'])")
echo "  Total break shards: ${N_SHARDS}"

# ---------------------------------------------------------------------------
# Step 2: Build pending list
# ---------------------------------------------------------------------------
PENDING_LIST="${OUTDIR}/break_pending_full.txt"
rm -f "${PENDING_LIST}"

for shard_file in "${BREAK_SHARDS_DIR}"/shard_*.json; do
    shard_id=$(basename "${shard_file}" .json | sed 's/shard_//')
    out_file="${BREAK_CLASSIFIED_DIR}/shard_${shard_id}.json"
    if [[ ! -f "${out_file}" ]]; then
        echo "${shard_id}" >> "${PENDING_LIST}"
    fi
done

N_PENDING=$(wc -l < "${PENDING_LIST}" | tr -d ' ')
N_DONE=$(( N_SHARDS - N_PENDING ))

echo ""
echo "=== Break Sweep (Full) Submit ==="
echo "  Total shards   : ${N_SHARDS}"
echo "  Already done   : ${N_DONE}"
echo "  Pending        : ${N_PENDING}"
echo "  Budget k       : ${K}"
echo ""

if [[ "${N_PENDING}" -eq 0 ]]; then
    echo "All break shards already done.  Nothing to submit."
    echo "Run csd_break_gather.py to produce the final CSV:"
    echo "  uv run python scripts/csd_break_gather.py \\"
    echo "      --input-dir ${BREAK_CLASSIFIED_DIR} \\"
    echo "      --output-csv notebooks/Ecoli_Analysis_Notebooks/csd_break_results_full.csv \\"
    echo "      --output-summary notebooks/Ecoli_Analysis_Notebooks/csd_break_summary_full.json"
    exit 0
fi

# ---------------------------------------------------------------------------
# Step 3: Split into per-node batches (64 shards/node)
# ---------------------------------------------------------------------------
BATCH_SIZE=64
N_NODES=$(( (N_PENDING + BATCH_SIZE - 1) / BATCH_SIZE ))

echo "  Batches (nodes): ${N_NODES}  (${BATCH_SIZE} shards/node)"
echo ""

BATCH_DIR="${OUTDIR}/break_batches_full"
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
    job_log="${LOG_DIR}/break_pack_${batch_id}_%j"

    JOB_ID=$(sbatch --parsable \
        --job-name="csd_break_full_${batch_id}" \
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
echo \"[break_full pack ${batch_id}] started on \$(hostname) at \$(date)\"
echo \"[break_full pack ${batch_id}] processing \$(wc -l < ${batch_file}) shards in parallel...\"

break_one() {
    local sid=\"\$1\"
    local shard_file=\"${BREAK_SHARDS_DIR}/shard_\${sid}.json\"
    local out_file=\"${BREAK_CLASSIFIED_DIR}/shard_\${sid}.json\"
    if [[ -f \"\${out_file}\" ]]; then
        echo \"[skip] break shard \${sid} already done\"
        return 0
    fi
    echo \"[start] break shard \${sid}\"
    uv run python ${REPO_ROOT}/scripts/csd_break_worker.py \
        --graphml ${GRAPHML} \
        --shard \"\${shard_file}\" \
        --output \"\${out_file}\" \
        --k ${K} \
        && echo \"[done ] break shard \${sid}\" \
        || echo \"[FAIL ] break shard \${sid}\"
}

export -f break_one
xargs -a ${batch_file} -P 64 -I{} bash -c 'break_one \"\$@\"' _ {}

echo \"[break_full pack ${batch_id}] all shards finished at \$(date)\"
")
    echo "  Submitted batch ${batch_id}: job ${JOB_ID}  ($(wc -l < "${batch_file}") shards)"
    SUBMITTED+=("${JOB_ID}")
done

echo ""
echo "=== All break_full batches submitted ==="
echo "  Job IDs: ${SUBMITTED[*]}"
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo "  ls ${BREAK_CLASSIFIED_DIR}/ | wc -l   # completed shards (target: ${N_SHARDS})"
echo ""
echo "Once all jobs complete, gather results with:"
echo "  uv run python scripts/csd_break_gather.py \\"
echo "      --input-dir ${BREAK_CLASSIFIED_DIR} \\"
echo "      --output-csv notebooks/Ecoli_Analysis_Notebooks/csd_break_results_full.csv \\"
echo "      --output-summary notebooks/Ecoli_Analysis_Notebooks/csd_break_summary_full.json"
echo ""
echo "Then run the recovery bank analysis:"
echo "  bash scripts/slurm/submit_csd_recovery.sh"
