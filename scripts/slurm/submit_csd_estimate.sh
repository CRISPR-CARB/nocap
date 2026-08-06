#!/bin/bash
# =============================================================================
# submit_csd_estimate.sh
# =============================================================================
# Run the CSD estimation experiments (scripts/csd_estimate.py) on SLURM.
#
# This script builds a small parameter grid over synthetic observational
# generation settings (sample size, missing-edge rate, missing-data rate,
# measurement-error mechanism) and submits node-packed sbatch jobs.
#
# Idempotency: if the target output CSV already exists and is non-empty, the
# corresponding job is skipped.
#
# Usage (from repo root):
#   bash scripts/slurm/submit_csd_estimate.sh
#
# Common environment overrides:
#   GRAPHML=/path/to/graph.graphml
#   OUTDIR=/path/to/output
#   ADJUSTMENTS_CSV=/path/to/csd_identifiable_edges.csv
#   INTERVENTION_CSV=/path/to/csd_recovery.csv
#   INTERVENTION_GRAPH_DIR=/path/to/intervention-graphs
#   INCLUDE_OBSERVATIONAL=1
#   USE_LATENT_EXPRESSION_HAT=0  # use true latent expression instead of count-derived estimates
#   DRY_RUN=1     # only print sbatch commands
# Paired mode creates task JSON records with setup_csd_experiment.py and passes
# each record directly to csd_estimate.py.
#
# Seed contract:
#   scm_seed controls structural betas and true missing edges.
#   data_seed controls exogenous noise, size factors, q0, counts, and missingness.
# Full mode assigns a deterministic unique seed pair to every parameter cell
# and replicate. fixed_scm assigns one SCM seed per edge-rate condition and
# varies data seeds across mechanisms, data rates, sample sizes, and replicates.
# The legacy default remains one process per mechanism/rate cell with --seed.
# For example:
#   bash scripts/slurm/submit_csd_estimate.sh
# Statistical bootstrap intervals are computed downstream from completed
# replicate outputs, not by generating additional simulation tasks here.

set -euo pipefail


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# ---------------------------------------------------------------------------
# Paths / config
# ---------------------------------------------------------------------------
GRAPHML="${GRAPHML:-${REPO_ROOT}/notebooks/Ecoli_Analysis_Notebooks/ecoli_full_network_no_small_rna.graphml}"
ADJUSTMENTS_CSV="${ADJUSTMENTS_CSV:-${REPO_ROOT}/notebooks/Ecoli_Analysis_Notebooks/csd_identifiable_edges.csv}"
INTERVENTION_CSV="${INTERVENTION_CSV:-}"

OUTDIR_WAS_SET="${OUTDIR+x}"
if [[ -z "${OUTDIR_WAS_SET}" ]]; then
    TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
    OUTDIR="${REPO_ROOT}/notebooks/Ecoli_Analysis_Notebooks/estimation/${TIMESTAMP}"
fi

LOG_DIR="${OUTDIR}/logs"
mkdir -p "${OUTDIR}" "${LOG_DIR}"

mkdir -p "${OUTDIR}/csv"

ACCOUNT="${ACCOUNT:-crispr_carb}"
PARTITION="${PARTITION:-slurm}"
TIME="${TIME:-12:00:00}"
MEM="${MEM:-0}"  # "0" lets slurm use partition default
CPUS_PER_TASK="${CPUS_PER_TASK:-1}"
BATCH_SIZE="${BATCH_SIZE:-64}"

DRY_RUN="${DRY_RUN:-0}"
DESIGN_MODE="${DESIGN_MODE:-paired_hierarchical}"
EXPERIMENT_ID="${EXPERIMENT_ID:-csd-${TIMESTAMP}}"
N_SCM_REPLICATES="${N_SCM_REPLICATES:-1}"
N_DATA_REPLICATES_PER_SCM="${N_DATA_REPLICATES_PER_SCM:-1}"
INCLUDE_OBSERVATIONAL="${INCLUDE_OBSERVATIONAL:-0}"

# ---------------------------------------------------------------------------
# Parameter grid
# ---------------------------------------------------------------------------

# Measurement-error/missing-data mechanisms supported by scripts/csd_estimate.py.
# instrument_error: low expression is too low for the instrument to detect.
# biological_error: a gene is not expressed at the time of measurement.
# biological_error+instrument_error: both independent mechanisms are active.
MECHANISMS=(
    "instrument_error"
    "biological_error"
    "biological_error+instrument_error"
)

# Synthetic regression parameters (linear-Gaussian SCM)
SEED_BASE="${SEED_BASE:-0}"
N_SAMPLES_LIST="${N_SAMPLES_LIST:-100,500,1000,2000,5000,10000}"
N_SAMPLES_LIST_SANITIZED="$(echo "${N_SAMPLES_LIST}" | tr ',' '-')"

N_MISSING_EDGE_RATES_LIST="${MISSING_EDGE_RATES_LIST:-0.0,0.2,0.4}"
N_MISSING_DATA_RATES_LIST="${MISSING_DATA_RATES_LIST:-0.0,0.3}"
[[ -n "${N_SAMPLES_LIST}" && -n "${N_MISSING_EDGE_RATES_LIST}" && -n "${N_MISSING_DATA_RATES_LIST}" ]] || { echo "Parameter lists must be nonempty" >&2; exit 2; }

printf '{"design_mode":"%s","experiment_id":"%s","seed_base":%s,"n_samples_list":"%s","missing_edge_rates":"%s","missing_data_rates":"%s","mechanisms":"%s"}\n' \
    "${DESIGN_MODE}" "${EXPERIMENT_ID}" "${SEED_BASE}" \
    "${N_SAMPLES_LIST}" "${N_MISSING_EDGE_RATES_LIST}" "${N_MISSING_DATA_RATES_LIST}" \
    "$(IFS=,; echo "${MECHANISMS[*]}")" \
    > "${OUTDIR}/run_metadata.json"

# If you need stronger confounding or different beta sampling, you can
# override these env vars:
SCC_CONFOUNDING_STRENGTH="${SCC_CONFOUNDING_STRENGTH:-0.0}"
BETA_MED="${BETA_MED:-0.5}"
BETA_LOG_SD="${BETA_LOG_SD:-0.5}"
BETA_ABS_MAX="${BETA_ABS_MAX:-5}"
BETA_P="${BETA_P:-0.5}"

DISPERSION="${DISPERSION:-0.1}"
SIZE_FACTOR_LOG_SD="${SIZE_FACTOR_LOG_SD:-0.4}"
BASELINE_EXPRESSION_LOG_MEAN="${BASELINE_EXPRESSION_LOG_MEAN:-0.0}"
BASELINE_EXPRESSION_LOG_SD="${BASELINE_EXPRESSION_LOG_SD:-2.0}"
UMI_PSEUDOCOUNT="${UMI_PSEUDOCOUNT:-1.0}"
USE_LATENT_EXPRESSION_HAT="${USE_LATENT_EXPRESSION_HAT:-1}"
if [[ "${USE_LATENT_EXPRESSION_HAT}" == "0" ]]; then
    LATENT_EXPRESSION_HAT_ARG="--no-latent-expression-hat"
else
    LATENT_EXPRESSION_HAT_ARG=""
fi

SELF_MASK_QUANTILE="${SELF_MASK_QUANTILE:-0.25}"
SELF_MASK_K="${SELF_MASK_K:-8.0}"
SELF_MASK_DIRECTION="${SELF_MASK_DIRECTION:-low}"

MIN_ROWS_AFTER_DROPNA="${MIN_ROWS_AFTER_DROPNA:-30}"

submit_batch() {
    local batch_file="$1"
    local batch_id="$2"
    local job_name="csd_est_pack_${batch_id}"
    local log_prefix="${LOG_DIR}/${job_name}_%j"

    local wrap_str="set -euo pipefail
ml python
source /share/apps/python/miniconda25.5.1/etc/profile.d/conda.sh
ml uv
cd ${REPO_ROOT}
export UV_CACHE_DIR=/tmp/\$USER/uv-cache-\$\$
mkdir -p \"\$UV_CACHE_DIR\"
uv sync --locked
run_one() {
  task_json=\"\$1\"
  out_csv=\"\$(uv run python -c 'import json,sys; print(json.load(open(sys.argv[1]))[\"output_csv\"])' \"\$task_json\")\"
  [[ -s \"\$out_csv\" ]] && { echo \"[skip] \$out_csv\"; return; }
  mkdir -p \"\$(dirname \"\$out_csv\")\"
   uv run python ${REPO_ROOT}/scripts/csd_estimate.py --task-json \"\$task_json\" --graphml ${GRAPHML} --output-csv \"\$out_csv\" --adjustments-csv ${ADJUSTMENTS_CSV} --assume-adjustments-csv-complete --design-mode ${DESIGN_MODE} --self-mask-quantile ${SELF_MASK_QUANTILE} --self-mask-k ${SELF_MASK_K} --self-mask-direction ${SELF_MASK_DIRECTION} --beta-med ${BETA_MED} --beta-log-sd ${BETA_LOG_SD} --beta-abs-max ${BETA_ABS_MAX} --beta-p ${BETA_P} --dispersion ${DISPERSION} --size-factor-log-sd ${SIZE_FACTOR_LOG_SD} --baseline-expression-log-mean ${BASELINE_EXPRESSION_LOG_MEAN} --baseline-expression-log-sd ${BASELINE_EXPRESSION_LOG_SD} --umi-pseudocount ${UMI_PSEUDOCOUNT} --scc-confounding-strength ${SCC_CONFOUNDING_STRENGTH} --min-rows-after-dropna ${MIN_ROWS_AFTER_DROPNA} ${LATENT_EXPRESSION_HAT_ARG}
}
export -f run_one
xargs -a ${batch_file} -P ${BATCH_SIZE} -I{} bash -c 'run_one "\$@"' _ {}
"

    local sbatch_args=(
        --parsable
        --job-name="${job_name}"
        --account="${ACCOUNT}"
        --partition="${PARTITION}"
        --time="${TIME}"
        --mem="${MEM}"
        --exclusive
        --cpus-per-task="${CPUS_PER_TASK}"
        --output="${log_prefix}.out"
        --error="${log_prefix}.err"
        --wrap="${wrap_str}"
    )

    echo "[submit] ${job_name} ($(wc -l < "${batch_file}") tasks)"
    if [[ "${DRY_RUN}" == "1" ]]; then
        echo "  DRY_RUN=1: not calling sbatch"
        return 0
    fi

    sbatch "${sbatch_args[@]}" >/dev/null
}

# ---------------------------------------------------------------------------
# Main: iterate the parameter grid
# ---------------------------------------------------------------------------

function main {
    # If no argument is provided, run the full parameter grid.
    # If an argument is provided, run a small smoke-test job.
    if [[ "$#" -eq 0 ]]; then
        IFS=',' read -r -a EDGE_RATES <<< "${N_MISSING_EDGE_RATES_LIST}"
        IFS=',' read -r -a DATA_RATES <<< "${N_MISSING_DATA_RATES_LIST}"

        echo "=== CSD Estimation Submit ==="
        echo "  REPO_ROOT: ${REPO_ROOT}"
        echo "  GRAPHML:   ${GRAPHML}"
        echo "  OUTDIR:    ${OUTDIR}"
        echo "  LOG_DIR:   ${LOG_DIR}"
        echo "  Mechanisms: ${MECHANISMS[*]}"
        echo "  Edge rates: ${EDGE_RATES[*]}"
        echo "  Data rates: ${DATA_RATES[*]}"
        echo "  n_samples_list: ${N_SAMPLES_LIST}"
        echo "  DRY_RUN: ${DRY_RUN}"
        echo "  Design mode: ${DESIGN_MODE}"
        echo "  Use latent expression hat: ${USE_LATENT_EXPRESSION_HAT}"
        echo "  Intervention CSV: ${INTERVENTION_CSV:-none}"
        echo ""

        pending_file="${OUTDIR}/pending_tasks.txt"
        batch_dir="${OUTDIR}/batches"
        tasks_dir="${OUTDIR}/tasks"
        mkdir -p "${tasks_dir}"
        mkdir -p "${batch_dir}"
        setup_args=(
            --output "${OUTDIR}/experiment.json" --tasks-dir "${tasks_dir}"
            --output-dir "${OUTDIR}/csv" --experiment-id "${EXPERIMENT_ID}"
            --design-mode "${DESIGN_MODE}" --base-seed "${SEED_BASE}"
            --scm-replicates "${N_SCM_REPLICATES}"
            --data-replicates "${N_DATA_REPLICATES_PER_SCM}"
            --n-samples-list "${N_SAMPLES_LIST}"
            --missing-edge-rates "${N_MISSING_EDGE_RATES_LIST}"
            --missing-data-rates "${N_MISSING_DATA_RATES_LIST}"
            --mechanisms "$(IFS=,; echo "${MECHANISMS[*]}")"
            --graphml "${GRAPHML}"
        )
        if [[ "${USE_LATENT_EXPRESSION_HAT}" == "0" ]]; then
            setup_args+=(--no-latent-expression-hat)
        else
            setup_args+=(--use-latent-expression-hat)
        fi
        if [[ -n "${INTERVENTION_CSV}" ]]; then
            INTERVENTION_GRAPH_DIR="${INTERVENTION_GRAPH_DIR:-${OUTDIR}/intervention-graphs}"
            setup_args+=(--intervention-csv "${INTERVENTION_CSV}" --intervention-graph-dir "${INTERVENTION_GRAPH_DIR}")
            [[ "${INCLUDE_OBSERVATIONAL}" == "1" ]] && setup_args+=(--include-observational)
        fi
        if [[ -z "${OUTDIR_WAS_SET}" || ! -s "${OUTDIR}/experiment.json" || ! -d "${tasks_dir}" ]]; then
            uv run python "${REPO_ROOT}/scripts/setup_csd_experiment.py" "${setup_args[@]}"
        else
            echo "Using existing experiment manifest and tasks in ${OUTDIR}"
        fi
        : > "${pending_file}"
        for task_json in "${tasks_dir}"/*.json; do
            out_csv="$(uv run python -c 'import json,sys; print(json.load(open(sys.argv[1]))["output_csv"])' "${task_json}")"
            [[ -s "${out_csv}" ]] || printf '%s\n' "${task_json}" >> "${pending_file}"
        done
        rm -f "${batch_dir}"/batch_*.txt
        if [[ -s "${pending_file}" ]]; then
            split -l "${BATCH_SIZE}" --numeric-suffixes=0 --suffix-length=3 "${pending_file}" "${batch_dir}/batch_"
            for batch_file in "${batch_dir}"/batch_*; do
                submit_batch "${batch_file}" "$(basename "${batch_file}")"
            done
        else
            echo "All CSD estimation outputs already exist. Nothing to submit."
        fi
        echo "=== Done submitting CSD estimation jobs ==="
        return 0
    else
        echo "Smoke-test mode is not supported by node packing; narrow the grid with environment variables."
        exit 2
    fi
}

# Invoke main when executed as a script.
main "$@"
