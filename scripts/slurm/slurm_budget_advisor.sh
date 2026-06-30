#!/usr/bin/env bash
# =============================================================================
# slurm_budget_advisor.sh
# =============================================================================
# Displays current SLURM CPU-minute budget status for the crispr_carb account
# and, given a planned sweep's parameters, recommends:
#   - how many CPU-minutes the sweep will cost
#   - whether the budget can absorb it
#   - a suggested shard size and node count
#
# Usage (no arguments — interactive):
#   bash scripts/slurm/slurm_budget_advisor.sh
#
# Usage (non-interactive, all params supplied):
#   bash scripts/slurm/slurm_budget_advisor.sh \
#       --tasks 6676 \
#       --workers-per-node 64 \
#       --time-per-task-sec 5 \
#       --cores-per-node 64 \
#       --exclusive
#
## Non-interactive (before submitting a sweep):
# bash scripts/slurm/slurm_budget_advisor.sh \
#     --tasks 6676 --workers-per-node 64 \
#     --time-per-task-sec 5 --exclusive

## Just check budget and recent history (then get prompted):
# bash scripts/slurm/slurm_budget_advisor.sh

## Check budget impact without --exclusive billing:
# bash scripts/slurm/slurm_budget_advisor.sh \
#     --tasks 6676 --time-per-task-sec 5 --no-exclusive
#
# # Options:
#   --account ACCT          Slurm account to check  [default: crispr_carb]
#   --tasks N               Total number of tasks (shards/edges) to process
#   --workers-per-node W    Parallel workers per node (xargs -P W)  [default: 64]
#   --time-per-task-sec T   Expected wall-clock seconds for one task  [default: prompt]
#   --cores-per-node C      Physical CPUs per node  [default: 64]
#   --exclusive             Use --exclusive node reservation (3× billing multiplier)
#   --no-exclusive          Use per-CPU reservation (1× billing multiplier)
#   --budget-fraction F     Max fraction of remaining budget to consume  [default: 0.50]
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
ACCOUNT="crispr_carb"
N_TASKS=""
WORKERS_PER_NODE=64
TIME_PER_TASK_SEC=""
CORES_PER_NODE=64
EXCLUSIVE=1             # assume --exclusive by default (conservative)
BUDGET_FRACTION=0.50    # warn if sweep > 50% of remaining budget

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --account)          ACCOUNT="$2"; shift 2 ;;
        --tasks)            N_TASKS="$2"; shift 2 ;;
        --workers-per-node) WORKERS_PER_NODE="$2"; shift 2 ;;
        --time-per-task-sec) TIME_PER_TASK_SEC="$2"; shift 2 ;;
        --cores-per-node)   CORES_PER_NODE="$2"; shift 2 ;;
        --exclusive)        EXCLUSIVE=1; shift ;;
        --no-exclusive)     EXCLUSIVE=0; shift ;;
        --budget-fraction)  BUDGET_FRACTION="$2"; shift 2 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Helper: pretty-print large numbers
# ---------------------------------------------------------------------------
pp() { python3 -c "print(f'{int($1):,}')"; }

# ---------------------------------------------------------------------------
# Step 1: Current budget status
# ---------------------------------------------------------------------------
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║          SLURM Budget Advisor — $(date '+%Y-%m-%d %H:%M')          ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "── Account: ${ACCOUNT} ─────────────────────────────────────────"

# GrpCPUMins is in minutes; sshare RawUsage is in seconds
LIMIT_MIN=$(sacctmgr show account "${ACCOUNT}" withassoc \
    format=GrpCPUMins --noheader 2>/dev/null | awk 'NR==1 {gsub(/[^0-9]/,"",$1); if($1+0>0) print $1+0}' | head -1)
# sshare -u USER returns two rows: account summary + user row; grep for the user line
USED_SEC=$(sshare -A "${ACCOUNT}" -u "$USER" -h -o "User,RawUsage" 2>/dev/null | \
    awk -v u="$USER" '$1==u {gsub(/[^0-9]/,"",$2); print $2+0}' | head -1)

if [[ -z "${LIMIT_MIN}" || "${LIMIT_MIN}" -eq 0 ]]; then
    echo "  ⚠  Could not read GrpCPUMins limit for ${ACCOUNT}."
    echo "     (sacctmgr may require admin view — using last known value)"
    LIMIT_MIN=11996700   # last known value
fi
USED_SEC="${USED_SEC:-0}"

USED_MIN=$(python3 -c "print(int(${USED_SEC}) // 60)")
REMAINING_MIN=$(python3 -c "print(max(0, int(${LIMIT_MIN}) - int(${USED_MIN})))")
PCT_USED=$(python3 -c "print(f'{int(${USED_MIN})/int(${LIMIT_MIN})*100:.1f}')")
PCT_FREE=$(python3 -c "print(f'{int(${REMAINING_MIN})/int(${LIMIT_MIN})*100:.1f}')")

echo "  Limit       : $(pp $LIMIT_MIN) CPU-min  ($(pp $(( LIMIT_MIN / 60 ))) CPU-hrs)"
echo "  Used        : $(pp $USED_MIN) CPU-min  (${PCT_USED}% of limit)"
echo "  Remaining   : $(pp $REMAINING_MIN) CPU-min  (${PCT_FREE}% free)"
echo ""

if (( REMAINING_MIN < 500000 )); then
    echo "  ⚠  WARNING: Less than 500K CPU-min remaining — budget is nearly exhausted."
    echo "     Large sweeps will be throttled (AssocGrpCPUMinutesLimit)."
elif (( REMAINING_MIN < 1000000 )); then
    echo "  ⚠  CAUTION: Less than 1M CPU-min remaining — plan carefully."
else
    echo "  ✓  Budget looks healthy."
fi
echo ""

# ---------------------------------------------------------------------------
# Step 2: Current idle nodes
# ---------------------------------------------------------------------------
echo "── Cluster availability ────────────────────────────────────────"
N_IDLE=$(sinfo -p slurm -t idle -o "%n" -h 2>/dev/null | wc -l | tr -d ' ')
N_ALLOC=$(sinfo -p slurm -t alloc -o "%n" -h 2>/dev/null | wc -l | tr -d ' ')
N_DOWN=$(sinfo -p slurm -t down,drain -o "%n" -h 2>/dev/null | wc -l | tr -d ' ')
echo "  Idle nodes  : ${N_IDLE}"
echo "  Busy nodes  : ${N_ALLOC}"
echo "  Down/drain  : ${N_DOWN}"
echo ""

# ---------------------------------------------------------------------------
# Step 3: Prompt for missing sweep parameters (interactive mode)
# ---------------------------------------------------------------------------
echo "── Sweep parameters ────────────────────────────────────────────"

if [[ -z "${N_TASKS}" ]]; then
    read -rp "  Total tasks (shards/edges) to process: " N_TASKS
fi

if [[ -z "${TIME_PER_TASK_SEC}" ]]; then
    read -rp "  Expected wall time per task (seconds) [e.g. 5]: " TIME_PER_TASK_SEC
fi

# Ask about exclusive if not set via flag
if [[ -z "${EXCLUSIVE+x}" ]] || ! [[ "${EXCLUSIVE}" =~ ^[01]$ ]]; then
    read -rp "  Use --exclusive node reservation? (y/n) [y]: " excl_ans
    excl_ans="${excl_ans:-y}"
    [[ "${excl_ans}" =~ ^[yY] ]] && EXCLUSIVE=1 || EXCLUSIVE=0
fi

echo ""
echo "  Tasks               : $(pp $N_TASKS)"
echo "  Workers/node        : ${WORKERS_PER_NODE}"
echo "  Time/task           : ${TIME_PER_TASK_SEC}s"
echo "  Cores/node          : ${CORES_PER_NODE}"
echo "  Exclusive billing   : $([ "${EXCLUSIVE}" -eq 1 ] && echo 'YES (3× multiplier)' || echo 'NO (1× multiplier)')"

# ---------------------------------------------------------------------------
# Step 4: Cost calculation
# ---------------------------------------------------------------------------
echo ""
echo "── Cost projection ─────────────────────────────────────────────"

python3 - <<PYEOF
import math

n_tasks           = int(${N_TASKS})
workers_per_node  = int(${WORKERS_PER_NODE})
time_per_task_sec = float(${TIME_PER_TASK_SEC})
cores_per_node    = int(${CORES_PER_NODE})
exclusive         = bool(${EXCLUSIVE})
remaining_min     = int(${REMAINING_MIN})
budget_fraction   = float(${BUDGET_FRACTION})
n_idle            = int(${N_IDLE})

# --- Node/shard sizing ---
# Wall time per node = time to run one batch of workers_per_node tasks serially
# (assuming tasks are homogeneous and all workers finish at roughly the same time)
wall_sec_per_node = time_per_task_sec   # with -P workers all run in parallel

# CPU-minutes billed per node per second (exclusive vs per-cpu)
billing_multiplier = 3 if exclusive else 1  # exclusive: main + extern + batch all charge full node
cpumin_per_node_per_sec = cores_per_node * billing_multiplier / 60

# --- Option A: use all idle nodes (fastest) ---
nodes_fast = max(1, min(n_idle, math.ceil(n_tasks / workers_per_node)))
shard_size_fast = max(1, math.ceil(n_tasks / (nodes_fast * workers_per_node)))
# actual tasks processed = nodes * workers_per_node * (1 shard of shard_size)
# wall time = shard_size * time_per_task_sec (shards run sequentially within a node slot)
wall_sec_fast = shard_size_fast * wall_sec_per_node
total_cpumin_fast = nodes_fast * cpumin_per_node_per_sec * wall_sec_fast
pct_fast = total_cpumin_fast / remaining_min * 100 if remaining_min > 0 else float('inf')

# --- Option B: stay within budget_fraction of remaining ---
budget_cpumin = remaining_min * budget_fraction
# budget_cpumin = nodes * cpumin_per_node_per_sec * wall_sec_per_node
# wall_sec = shard_size * time_per_task_sec
# nodes = ceil(n_tasks / (workers_per_node * shard_size))
# Solve: budget = ceil(n/(w*s)) * rate * s * t
#               ≈ (n/w) * rate * t   (independent of s — shards just shift work between nodes)
# So total cost is ~fixed regardless of shard size! The variable is wall time.
# Minimum cost = n_tasks * time_per_task_sec * billing_mult * cores / 60 / workers_per_node
min_cpumin = n_tasks * time_per_task_sec / 60 * cores_per_node * billing_multiplier / workers_per_node
pct_min = min_cpumin / remaining_min * 100 if remaining_min > 0 else float('inf')

# Max safe nodes = budget_cpumin / (cpumin_per_node_per_sec * wall_sec_per_node)
# At shard_size=1: wall_sec = time_per_task_sec
max_nodes_budget = int(budget_cpumin / (cpumin_per_node_per_sec * time_per_task_sec))
max_nodes_budget = max(1, min(max_nodes_budget, n_idle))
shard_size_budget = max(1, math.ceil(n_tasks / (max_nodes_budget * workers_per_node)))
wall_sec_budget = shard_size_budget * time_per_task_sec
total_cpumin_budget = max_nodes_budget * cpumin_per_node_per_sec * wall_sec_budget
pct_budget = total_cpumin_budget / remaining_min * 100 if remaining_min > 0 else float('inf')

# --- Option C: conservative (20% of remaining) ---
budget_cons = remaining_min * 0.20
max_nodes_cons = int(budget_cons / (cpumin_per_node_per_sec * time_per_task_sec))
max_nodes_cons = max(1, min(max_nodes_cons, n_idle))
shard_size_cons = max(1, math.ceil(n_tasks / (max_nodes_cons * workers_per_node)))
wall_sec_cons = shard_size_cons * time_per_task_sec
total_cpumin_cons = max_nodes_cons * cpumin_per_node_per_sec * wall_sec_cons
pct_cons = total_cpumin_cons / remaining_min * 100 if remaining_min > 0 else float('inf')

def fmt_time(s):
    if s < 60: return f"{s:.0f}s"
    if s < 3600: return f"{s/60:.1f}m"
    return f"{s/3600:.1f}h"

def bar(pct, width=30):
    filled = min(width, int(pct / 100 * width))
    color = '\033[92m' if pct < 50 else ('\033[93m' if pct < 80 else '\033[91m')
    return color + '█' * filled + '\033[0m' + '░' * (width - filled) + f' {pct:.1f}%'

print(f"  Minimum possible cost (any node count):")
print(f"    {min_cpumin:>12,.0f} CPU-min  |  {bar(pct_min)}")
print()
print(f"  ┌─ Option A: FASTEST  (use all {n_idle} idle nodes)")
print(f"  │  Nodes       : {nodes_fast}")
print(f"  │  Shard size  : {shard_size_fast} task(s)/shard")
print(f"  │  Wall time   : ~{fmt_time(wall_sec_fast)}")
print(f"  │  Cost        : {total_cpumin_fast:>10,.0f} CPU-min  |  {bar(pct_fast)}")
print(f"  │  {'⚠  Exceeds budget fraction!' if pct_fast > 100 else ('⚠  High spend — check budget' if pct_fast > budget_fraction*100 else '✓  Within budget')}")
print(f"  └──────────────────────────────────────────────────────────")
print()
print(f"  ┌─ Option B: BALANCED  (use ≤{budget_fraction*100:.0f}% of remaining budget)")
print(f"  │  Nodes       : {max_nodes_budget}")
print(f"  │  Shard size  : {shard_size_budget} task(s)/shard")
print(f"  │  Wall time   : ~{fmt_time(wall_sec_budget)}")
print(f"  │  Cost        : {total_cpumin_budget:>10,.0f} CPU-min  |  {bar(pct_budget)}")
print(f"  └──────────────────────────────────────────────────────────")
print()
print(f"  ┌─ Option C: CONSERVATIVE  (use ≤20% of remaining budget)")
print(f"  │  Nodes       : {max_nodes_cons}")
print(f"  │  Shard size  : {shard_size_cons} task(s)/shard")
print(f"  │  Wall time   : ~{fmt_time(wall_sec_cons)}")
print(f"  │  Cost        : {total_cpumin_cons:>10,.0f} CPU-min  |  {bar(pct_cons)}")
print(f"  └──────────────────────────────────────────────────────────")
print()

# Recommendation
print("  ── Recommendation ───────────────────────────────────────────")
if pct_min > 100:
    print("  ✗  Even the cheapest run exceeds remaining budget.")
    print("     Wait for the billing window to roll over before submitting.")
elif pct_fast <= budget_fraction * 100:
    print(f"  ★  Option A (all {n_idle} idle nodes) is safe — go for speed.")
    print(f"     sbatch args: --ntasks={nodes_fast * workers_per_node} + shard-size={shard_size_fast}")
elif pct_budget <= 80:
    print(f"  ★  Option B (balanced) recommended.")
    print(f"     sbatch args: {max_nodes_budget} nodes + shard-size={shard_size_budget}")
else:
    print(f"  ★  Option C (conservative) to protect remaining budget.")
    print(f"     sbatch args: {max_nodes_cons} nodes + shard-size={shard_size_cons}")

if exclusive:
    print()
    print("  💡 Using --no-exclusive would reduce cost by ~3× (remove extern/batch overhead).")
    print("     Switch to --ntasks=N --cpus-per-task=1 (without --exclusive) for short jobs.")
PYEOF

echo ""
echo "── Recent usage by job type (last 30 days) ─────────────────────"
sacct -u "$USER" -A "${ACCOUNT}" \
    --starttime="$(date -d '30 days ago' +%Y-%m-%d)" \
    --format=JobID%10,JobName%30,Elapsed,NCPUS,CPUTimeRAW,State \
    --noheader 2>&1 \
    | grep -v '\.\(batch\|extern\|[0-9]\+\)' \
    | grep -v '^\s*$' \
    | awk '{
        name=$2; cpumin=$5/60;
        # Normalize UUID job names
        if (length(name)==36 || name ~ /^[0-9a-f]{8}-/) name="(uuid-workflow)";
        # Normalize numbered job names: strip trailing _NNN suffix for grouping
        else gsub(/_[0-9]+$/, "", name);
        total[name]+=cpumin; count[name]++;
    }
    END {
        for (k in total)
            printf "  %-32s jobs=%4d  cpu_hrs=%7.0f\n", k, count[k], total[k]/60
    }' | sort -k4 -rn | head -15
echo ""
