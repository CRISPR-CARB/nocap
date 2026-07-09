#!/bin/bash
# =============================================================================
# submit_csd_estimate.sh
# =============================================================================
# Run the CSD estimation experiments (scripts/csd_estimate.py) on SLURM.
#
# This script builds a small parameter grid over synthetic observational
# generation settings (sample size, missing-edge rate, missing-data rate,
# missing-data mechanism) and submits one sbatch job per grid point.
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
#   DRY_RUN=1     # only print sbatch commands

set -euo pipefail


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# ---------------------------------------------------------------------------
# Paths / config
# ---------------------------------------------------------------------------
GRAPHML="${GRAPHML:-${REPO_ROOT}/notebooks/Ecoli_Analysis_Notebooks/ecoli_full_network_no_small_rna.graphml}"
ADJUSTMENTS_CSV="${ADJUSTMENTS_CSV:-${REPO_ROOT}/notebooks/Ecoli_Analysis_Notebooks/csd_identifiable_edges.csv}"

OUTDIR="${OUTDIR:-${REPO_ROOT}/notebooks/Ecoli_Analysis_Notebooks/estimation}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTDIR="${OUTDIR}/${TIMESTAMP}"

LOG_DIR="${OUTDIR}/logs"
mkdir -p "${OUTDIR}" "${LOG_DIR}"

mkdir -p "${OUTDIR}/csv"

ACCOUNT="${ACCOUNT:-crispr_carb}"
PARTITION="${PARTITION:-slurm}"
TIME="${TIME:-12:00:00}"
MEM="${MEM:-0}"  # "0" lets slurm use partition default
CPUS_PER_TASK="${CPUS_PER_TASK:-1}"

DRY_RUN="${DRY_RUN:-0}"

# ---------------------------------------------------------------------------
# Parameter grid
# ---------------------------------------------------------------------------

# Missingness mechanisms supported by scripts/csd_estimate.py
MECHANISMS=(
    "MCAR"
    "MNAR_self_mask"
)

# Synthetic regression parameters (linear-Gaussian SCM)
SEED_BASE="${SEED_BASE:-0}"
N_SAMPLES_LIST="${N_SAMPLES_LIST:-100,500,1000,2000,5000,10000}"
N_SAMPLES_LIST_SANITIZED="$(echo "${N_SAMPLES_LIST}" | tr ',' '-')"

N_MISSING_EDGE_RATES_LIST="${MISSING_EDGE_RATE_LIST:-0.0,0.2,0.4}"
N_MISSING_DATA_RATES_LIST="${MISSING_DATA_RATE_LIST:-0.0,0.3}"

# If you need stronger confounding or different beta sampling, you can
# override these env vars:
SCC_CONFOUNDING_STRENGTH="${SCC_CONFOUNDING_STRENGTH:-0.0}"
BETA_MEAN="${BETA_MEAN:-1.0}"
BETA_STD="${BETA_STD:-0.2}"
BETA_ABS_MAX="${BETA_ABS_MAX:-0.9}"

SELF_MASK_QUANTILE="${SELF_MASK_QUANTILE:-0.25}"
SELF_MASK_K="${SELF_MASK_K:-8.0}"
SELF_MASK_DIRECTION="${SELF_MASK_DIRECTION:-low}"

MIN_ROWS_AFTER_DROPNA="${MIN_ROWS_AFTER_DROPNA:-30}"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

sanitize_num() {
    # 0.2 -> 0p2, -0.1 -> m0p1
    local x="$1"
    local sign=""
    if [[ "${x}" == -* ]]; then
        sign="m"
        x="${x#-}"
    fi
    x="${x/./p}"
    echo "${sign}${x}"
}

csv_out_for_job() {
    # Arguments: mech, missing_edge_rate, missing_data_rate
    local mech="$1"
    local megr="$2"
    local mdr="$3"

    local mech_tag="${mech}"
    mech_tag="${mech_tag//./_}"
    mech_tag="${mech_tag//\//_}"
    mech_tag="${mech_tag//_/_}"

    local megr_tag
    megr_tag="$(sanitize_num "${megr}")"

    local mdr_tag
    mdr_tag="$(sanitize_num "${mdr}")"

    echo "${OUTDIR}/csv/csd_estimate_mech_${mech_tag}_edge_${megr_tag}_data_${mdr_tag}_n_${N_SAMPLES_LIST_SANITIZED}.csv"
}

submit_one() {
    local mech="$1"
    local missing_edge_rate="$2"
    local missing_data_rate="$3"

    local out_csv
    out_csv="$(csv_out_for_job "${mech}" "${missing_edge_rate}" "${missing_data_rate}")"
    local log_prefix
    log_prefix="${LOG_DIR}/csd_estimate_$(basename "${out_csv%.*}")"

    # Skip if already exists and non-empty.
    if [[ -s "${out_csv}" ]]; then
        echo "[skip] already exists: ${out_csv}"
        return 0
    fi

    local job_name
    job_name="csd_est_${mech}_e$(sanitize_num "${missing_edge_rate}")_d$(sanitize_num "${missing_data_rate}")"
    job_name="${job_name//_/-}"
    job_name="${job_name//__/-}"

    # Build command pieces so optional args are easy.
    local adjustments_arg=()
    if [[ -n "${ADJUSTMENTS_CSV}" ]]; then
        adjustments_arg=(--adjustments-csv "${ADJUSTMENTS_CSV}")
        # If the CSV doesn't contain every edge, the script will fall back to the
        # sigma-extension oracle. To enforce "only use what you have", export
        # ASSUME_ADJUSTMENTS_CSV_COMPLETE=1.
        if [[ "${ASSUME_ADJUSTMENTS_CSV_COMPLETE:-1}" == "1" ]]; then
            adjustments_arg+=(--assume-adjustments-csv-complete)
        fi
    fi

    # Build the python invocation as an argv array so flags like
    # --adjustments-csv don't get mangled across sbatch/--wrap newlines.
    local python_cmd=(
        "uv" "run" "python" "${REPO_ROOT}/scripts/csd_estimate.py"
        --graphml "${GRAPHML}"
        --output-csv "${out_csv}"
    )
    if [[ "${#adjustments_arg[@]}" -gt 0 ]]; then
        python_cmd+=("${adjustments_arg[@]}")
    fi
    python_cmd+=(
        --seed "${SEED_BASE}"
        --n-samples-list "${N_SAMPLES_LIST}"
        --missing-edge-rate "${missing_edge_rate}"
        --missing-data-rate "${missing_data_rate}"
        --missing-data-mechanism "${mech}"
        --self-mask-quantile "${SELF_MASK_QUANTILE}"
        --self-mask-k "${SELF_MASK_K}"
        --self-mask-direction "${SELF_MASK_DIRECTION}"
        --beta-mean "${BETA_MEAN}"
        --beta-std "${BETA_STD}"
        --beta-abs-max "${BETA_ABS_MAX}"
        --scc-confounding-strength "${SCC_CONFOUNDING_STRENGTH}"
        --min-rows-after-dropna "${MIN_ROWS_AFTER_DROPNA}"
    )

    # Convert the argv array to a single shell-escaped command string.
    local python_cmd_str=""
    local arg
    for arg in "${python_cmd[@]}"; do
        python_cmd_str+="$(printf '%q' "${arg}") "
    done
    python_cmd_str="${python_cmd_str%% }"  # trim trailing space

    echo "Executing: ${python_cmd_str}"

    local wrap_cmd
    wrap_cmd=(
        "set -euo pipefail"
        # Load the modules
        "ml python"
        "source /share/apps/python/miniconda25.5.1/etc/profile.d/conda.sh"
        "ml uv"
        "cd ${REPO_ROOT}"
        # Prevent race condition with uv cache across jobs
        "export UV_CACHE_DIR=/tmp/$USER/uv-cache-$$"
        "mkdir -p \"\$UV_CACHE_DIR\""
        "uv sync"
        "echo \"[csd_estimate] job: ${job_name}\""
        "echo \"  graphml: ${GRAPHML}\""
        "echo \"  out: ${out_csv}\""
        "echo \"  mech: ${mech} edge: ${missing_edge_rate} data: ${missing_data_rate}\""
        "${python_cmd_str}"
    )

    # Convert wrap_cmd array to a single string for sbatch --wrap.
    # shellcheck disable=SC2145
    local wrap_str=""
    local part
    for part in "${wrap_cmd[@]}"; do
        if [[ -z "${wrap_str}" ]]; then
            wrap_str="${part}"
        else
            wrap_str+=$'\n'"${part}"
        fi
    done

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

    echo "[submit] ${job_name} -> ${out_csv}"
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
        echo ""

        for mech in "${MECHANISMS[@]}"; do
            for edge_rate in "${EDGE_RATES[@]}"; do
                for data_rate in "${DATA_RATES[@]}"; do
                    submit_one "${mech}" "${edge_rate}" "${data_rate}"
                done
            done
        done

        echo "=== Done submitting CSD estimation jobs ==="
    else
        submit_one MCAR 0.0 0.0
        echo "=== Done submitting CSD estimation test job ==="
    fi
}

# Invoke main when executed as a script.
main "$@"
