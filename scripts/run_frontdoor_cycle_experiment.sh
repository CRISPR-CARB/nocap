#!/usr/bin/env bash
# Run a local CSD experiment on the built-in frontdoor_cycle graph.
#
# The setup script creates one task per parameter condition and replicate. Each
# task is then executed locally with csd_estimate.py; no SLURM is required.
#
# Examples:
#   bash scripts/run_frontdoor_cycle_experiment.sh
#   N_SCM_REPLICATES=3 N_DATA_REPLICATES_PER_SCM=5 \
#     bash scripts/run_frontdoor_cycle_experiment.sh
#   NO_LATENT_EXPRESSION_HAT=1 bash scripts/run_frontdoor_cycle_experiment.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/results/frontdoor_cycle/${TIMESTAMP}}"
N_SAMPLES_LIST="${N_SAMPLES_LIST:-500,1000,10000,50000}"
MISSING_EDGE_RATES="${MISSING_EDGE_RATES:-0}"
MISSING_DATA_RATES="${MISSING_DATA_RATES:-0,0.3,0.9,0.99}"
MISSING_DATA_MECHANISM="${MISSING_DATA_MECHANISM:-biological_error}"
SEED="${SEED:-0}"
DESIGN_MODE="${DESIGN_MODE:-paired_hierarchical}"
EXPERIMENT_ID="${EXPERIMENT_ID:-frontdoor-cycle-local}"
N_SCM_REPLICATES="${N_SCM_REPLICATES:-1}"
N_DATA_REPLICATES_PER_SCM="${N_DATA_REPLICATES_PER_SCM:-1}"
NO_LATENT_EXPRESSION_HAT="${NO_LATENT_EXPRESSION_HAT:-0}"
MANIFEST="${MANIFEST:-${OUTPUT_DIR}/experiment.json}"
TASKS_DIR="${TASKS_DIR:-${OUTPUT_DIR}/tasks}"

setup_args=(
    --output "${MANIFEST}"
    --tasks-dir "${TASKS_DIR}"
    --output-dir "${OUTPUT_DIR}/csv"
    --experiment-id "${EXPERIMENT_ID}"
    --design-mode "${DESIGN_MODE}"
    --base-seed "${SEED}"
    --scm-replicates "${N_SCM_REPLICATES}"
    --data-replicates "${N_DATA_REPLICATES_PER_SCM}"
    --n-samples-list "${N_SAMPLES_LIST}"
    --missing-edge-rates "${MISSING_EDGE_RATES}"
    --missing-data-rates "${MISSING_DATA_RATES}"
    --mechanisms "${MISSING_DATA_MECHANISM}"
)

if [[ "${NO_LATENT_EXPRESSION_HAT}" == "1" ]]; then
    setup_args+=(--no-latent-expression-hat)
fi

mkdir -p "${OUTPUT_DIR}" "${TASKS_DIR}"
cd "${REPO_ROOT}"

uv run python scripts/setup_csd_experiment.py "${setup_args[@]}"

shopt -s nullglob
task_files=("${TASKS_DIR}"/*.json)
if (( ${#task_files[@]} == 0 )); then
    printf 'No task files were generated in %s\n' "${TASKS_DIR}" >&2
    exit 1
fi

for task_json in "${task_files[@]}"; do
    output_csv="$(uv run python -c 'import json, sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["output_csv"])' "${task_json}")"
    if [[ -s "${output_csv}" ]]; then
        printf '[skip] %s\n' "${output_csv}"
        continue
    fi
    printf '[run] %s\n' "${task_json}"
    uv run python scripts/csd_estimate.py \
        --task-json "${task_json}" \
        --demo frontdoor_cycle
done
