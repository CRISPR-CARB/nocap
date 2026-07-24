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
#   DRY_RUN=1     # only print sbatch commands
#   BOOTSTRAP=1 BOOTSTRAP_MODE=full BOOTSTRAP_R=3
#   BOOTSTRAP=1 BOOTSTRAP_MODE=fixed_scm BOOTSTRAP_R=3
# Replicate mode runs one sample size per csd_estimate.py process.
#
# Seed contract:
#   scm_seed controls structural betas and true missing edges.
#   data_seed controls exogenous noise, counts, library sizes, and missingness.
# Full mode assigns a deterministic unique seed pair to every parameter cell
# and replicate. fixed_scm assigns one SCM seed per edge-rate condition and
# varies data seeds across mechanisms, data rates, sample sizes, and replicates.
# The legacy default remains one process per mechanism/rate cell with --seed.
# For example:
#   bash scripts/slurm/submit_csd_estimate.sh
#   BOOTSTRAP=1 BOOTSTRAP_MODE=full BOOTSTRAP_R=3 \
#     bash scripts/slurm/submit_csd_estimate.sh
#   BOOTSTRAP=1 BOOTSTRAP_MODE=fixed_scm BOOTSTRAP_R=3 \
#     bash scripts/slurm/submit_csd_estimate.sh
# These are simulation-replicate performance intervals, not row-bootstrap
# confidence intervals or formal uncertainty for a real biological edge.

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
BATCH_SIZE="${BATCH_SIZE:-64}"

DRY_RUN="${DRY_RUN:-0}"
BOOTSTRAP="${BOOTSTRAP:-0}"
BOOTSTRAP_MODE="${BOOTSTRAP_MODE:-full}"
BOOTSTRAP_R="${BOOTSTRAP_R:-1}"
if [[ "${BOOTSTRAP}" != "0" && "${BOOTSTRAP}" != "1" ]]; then echo "BOOTSTRAP must be 0 or 1" >&2; exit 2; fi
if [[ "${BOOTSTRAP_MODE}" != "full" && "${BOOTSTRAP_MODE}" != "fixed_scm" ]]; then echo "BOOTSTRAP_MODE must be full or fixed_scm" >&2; exit 2; fi
if ! [[ "${BOOTSTRAP_R}" =~ ^[1-9][0-9]*$ ]]; then echo "BOOTSTRAP_R must be a positive integer" >&2; exit 2; fi

# ---------------------------------------------------------------------------
# Parameter grid
# ---------------------------------------------------------------------------

# Measurement-error/misisng-data mechanisms supported by scripts/csd_estimate.py.
# instrument_error: low expression is too low for the instrument to detect.
# biological_error: a gene is not expressed at the time of measurement.
MECHANISMS=(
    "instrument_error"
    "biological_error"
)

# Synthetic regression parameters (linear-Gaussian SCM)
SEED_BASE="${SEED_BASE:-0}"
N_SAMPLES_LIST="${N_SAMPLES_LIST:-100,500,1000,2000,5000,10000}"
N_SAMPLES_LIST_SANITIZED="$(echo "${N_SAMPLES_LIST}" | tr ',' '-')"

N_MISSING_EDGE_RATES_LIST="${MISSING_EDGE_RATES_LIST:-0.0,0.2,0.4}"
N_MISSING_DATA_RATES_LIST="${MISSING_DATA_RATES_LIST:-0.0,0.3}"
[[ -n "${N_SAMPLES_LIST}" && -n "${N_MISSING_EDGE_RATES_LIST}" && -n "${N_MISSING_DATA_RATES_LIST}" ]] || { echo "Parameter lists must be nonempty" >&2; exit 2; }

printf '{"bootstrap":%s,"bootstrap_mode":"%s","bootstrap_r":%s,"seed_base":%s,"n_samples_list":"%s","missing_edge_rates":"%s","missing_data_rates":"%s"}\n' \
    "${BOOTSTRAP}" "${BOOTSTRAP_MODE}" "${BOOTSTRAP_R}" "${SEED_BASE}" \
    "${N_SAMPLES_LIST}" "${N_MISSING_EDGE_RATES_LIST}" "${N_MISSING_DATA_RATES_LIST}" \
    > "${OUTDIR}/run_metadata.json"

# If you need stronger confounding or different beta sampling, you can
# override these env vars:
SCC_CONFOUNDING_STRENGTH="${SCC_CONFOUNDING_STRENGTH:-0.0}"
BETA_MED="${BETA_MED:-2.0}"
BETA_LOG_SD="${BETA_LOG_SD:-0.5}"
BETA_ABS_MAX="${BETA_ABS_MAX:-5}"
BETA_P="${BETA_P:-0.5}"

UMI_DISP="${UMI_DISP:-0.1}"
LIB_SIZE_MEAN="${LIB_SIZE_MEAN:-13.2877}"
LIB_SIZE_SD="${LIB_SIZE_SD:-0.4}"
UMI_COUNT="${UMI_COUNT:-1.0}"

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

seed_hash() {
    printf '%s' "$1" | cksum | cut -d' ' -f1
}

csv_out_for_job() {
    # Arguments: mech, missing_edge_rate, missing_data_rate, n, replicate, scm, data
    local mech="$1"
    local megr="$2"
    local mdr="$3"
    local n_samples="$4"
    local replicate="$5"
    local scm_seed="$6"
    local data_seed="$7"

    local mech_tag="${mech}"
    mech_tag="${mech_tag//./_}"
    mech_tag="${mech_tag//\//_}"
    mech_tag="${mech_tag//_/_}"

    local megr_tag
    megr_tag="$(sanitize_num "${megr}")"

    local mdr_tag
    mdr_tag="$(sanitize_num "${mdr}")"

    if [[ "${BOOTSTRAP}" == 1 ]]; then
        echo "${OUTDIR}/csv/csd_estimate_${BOOTSTRAP_MODE}_mech_${mech_tag}_edge_${megr_tag}_data_${mdr_tag}_n_${n_samples}_rep_${replicate}_scm_${scm_seed}_data_${data_seed}.csv"
    else
        echo "${OUTDIR}/csv/csd_estimate_mech_${mech_tag}_edge_${megr_tag}_data_${mdr_tag}_n_${N_SAMPLES_LIST_SANITIZED}.csv"
    fi
}

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
  IFS='|' read -r mech edge_rate data_rate n_samples replicate scm_seed data_seed out_csv <<< \"\$1\"
  [[ -s \"\$out_csv\" ]] && { echo \"[skip] \$out_csv\"; return; }
  if [[ \"${BOOTSTRAP}\" == 1 ]]; then
    seed_args=(--scm-seed \"\$scm_seed\" --data-seed \"\$data_seed\" --n-samples-list \"\$n_samples\")
  else
    seed_args=(--seed ${SEED_BASE} --n-samples-list ${N_SAMPLES_LIST})
  fi
  uv run python ${REPO_ROOT}/scripts/csd_estimate.py --graphml ${GRAPHML} --output-csv \"\$out_csv\" --adjustments-csv ${ADJUSTMENTS_CSV} --assume-adjustments-csv-complete \"\${seed_args[@]}\" --save-config ${OUTDIR}/config.json --missing-edge-rate \"\$edge_rate\" --missing-data-rate \"\$data_rate\" --missing-data-mechanism \"\$mech\" --self-mask-quantile ${SELF_MASK_QUANTILE} --self-mask-k ${SELF_MASK_K} --self-mask-direction ${SELF_MASK_DIRECTION} --beta-med ${BETA_MED} --beta-log-sd ${BETA_LOG_SD} --beta-abs-max ${BETA_ABS_MAX} --beta-p ${BETA_P} --umi-dispersion ${UMI_DISP} --library-size-log-mean ${LIB_SIZE_MEAN} --library-size-log-sd ${LIB_SIZE_SD} --umi-pseudocount ${UMI_COUNT} --scc-confounding-strength ${SCC_CONFOUNDING_STRENGTH} --min-rows-after-dropna ${MIN_ROWS_AFTER_DROPNA}
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
        echo "  Bootstrap: ${BOOTSTRAP} mode=${BOOTSTRAP_MODE} R=${BOOTSTRAP_R}"
        echo ""

        pending_file="${OUTDIR}/pending_tasks.txt"
        batch_dir="${OUTDIR}/batches"
        declare -A scm_seed_by_edge=()
        declare -A used_seed_pairs=()
        : > "${pending_file}"
        mkdir -p "${batch_dir}"
        for mech in "${MECHANISMS[@]}"; do
            for edge_rate in "${EDGE_RATES[@]}"; do
                for data_rate in "${DATA_RATES[@]}"; do
                    if [[ "${BOOTSTRAP}" == 1 ]]; then
                        for n in $(printf '%s' "${N_SAMPLES_LIST}" | tr ',' ' '); do
                            for ((rep=0; rep<BOOTSTRAP_R; rep++)); do
                                if [[ "${BOOTSTRAP_MODE}" == fixed_scm ]]; then
                                    edge_key="${edge_rate}"
                                    if [[ -z "${scm_seed_by_edge[${edge_key}]+x}" ]]; then
                                        scm_seed=$((SEED_BASE + $(seed_hash "scm:${edge_key}")))
                                        scm_seed_by_edge["${edge_key}"]="${scm_seed}"
                                    else
                                        scm_seed="${scm_seed_by_edge[${edge_key}]}"
                                    fi
                                    data_key="${mech}:${edge_rate}:${data_rate}:${n}:${rep}"
                                else
                                    key="${mech}:${edge_rate}:${data_rate}:${n}:${rep}"
                                    scm_seed=$((SEED_BASE + $(seed_hash "scm:${key}")))
                                    data_key="${key}"
                                fi
                                data_seed=$((SEED_BASE + $(seed_hash "data:${data_key}")))
                                while [[ -n "${used_seed_pairs[${scm_seed}:${data_seed}]+x}" ]]; do
                                    data_seed=$((data_seed + 1))
                                done
                                used_seed_pairs["${scm_seed}:${data_seed}"]=1
                                out_csv="$(csv_out_for_job "${mech}" "${edge_rate}" "${data_rate}" "${n}" "${rep}" "${scm_seed}" "${data_seed}")"
                                [[ -s "${out_csv}" ]] || printf '%s|%s|%s|%s|%s|%s|%s|%s\n' "${mech}" "${edge_rate}" "${data_rate}" "${n}" "${rep}" "${scm_seed}" "${data_seed}" "${out_csv}" >> "${pending_file}"
                            done
                        done
                    else
                        out_csv="$(csv_out_for_job "${mech}" "${edge_rate}" "${data_rate}" "" "" "" "")"
                        [[ -s "${out_csv}" ]] || printf '%s|%s|%s|||||%s\n' "${mech}" "${edge_rate}" "${data_rate}" "${out_csv}" >> "${pending_file}"
                    fi
                done
            done
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
    else
        echo "Smoke-test mode is not supported by node packing; narrow the grid with environment variables."
        exit 2
    fi
}

# Invoke main when executed as a script.
main "$@"
