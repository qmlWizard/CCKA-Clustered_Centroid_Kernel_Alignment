#!/bin/bash
#SBATCH --job-name=ccka-suite
#SBATCH --partition=terai
#SBATCH --nodes=25
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=/home/cdacB/digvijaysinha.cdacb/CCKA-Clustered_Centroid_Kernel_Alignment/output/%j.out
#SBATCH --error=/home/cdacB/digvijaysinha.cdacb/CCKA-Clustered_Centroid_Kernel_Alignment/output/%j.err

# Runs the WHOLE experiment suite (all configs/exp*.yaml plus the kernel,
# ansatz, budget and scalability analyses) as one Ray pool over every node.
#
#   sbatch jobs/run_all.slurm
#   sbatch --nodes=40 jobs/run_all.slurm
#   sbatch --export=ALL,SUITE_ARGS="--only exp06_k_sweep kernel" jobs/run_all.slurm
#   sbatch --export=ALL,TIMING_NODES=2 jobs/run_all.slurm
#
# TIMING_NODES (default 1): the last N nodes are started with the custom Ray
# resource "timing" and run ONLY the scalability measurements, so timings are
# not disturbed by other work. They take no part in the other experiments.
# Set TIMING_NODES=0 to disable (scalability then runs alone at the end).
# Results go to results/suite-<jobid>/ (override with OUTPUT_ROOT).

# ============================================================
# Environment
# ============================================================

source ~/.bashrc

. /home/apps/spack/share/spack/setup-env.sh

spack load /oxmullt
spack load /gaily2jw
spack load /2wdhn7r

# ============================================================
# OpenBLAS
# ============================================================

export OPENBLAS_ROOT=$(spack location -i /geypa2)
export LD_LIBRARY_PATH=$OPENBLAS_ROOT/lib:$LD_LIBRARY_PATH

export MPI_LIB=/home/apps/spack/opt/spack/linux-cascadelake/openmpi-4.1.8-gaily2jwc3klcubp55unnlgiaglqu7mf/lib
export LD_LIBRARY_PATH=$MPI_LIB:$LD_LIBRARY_PATH

conda activate py310
export RAY_AUTH_MODE=disabled
ulimit -n 65535

# ============================================================
# Get allocated nodes
# ============================================================

nodes=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
HEAD_NODE="${nodes[0]}"
TIMING_NODES="${TIMING_NODES:-1}"
if (( TIMING_NODES >= ${#nodes[@]} )); then
    TIMING_NODES=0
fi

echo "=========================================="
echo "SLURM JOB ID      : $SLURM_JOB_ID"
echo "NUMBER OF NODES   : $SLURM_JOB_NUM_NODES (timing nodes: $TIMING_NODES)"
echo "CPUS PER NODE     : $SLURM_CPUS_PER_TASK"
echo "=========================================="

HEAD_IP=$(srun --nodes=1 --ntasks=1 --nodelist="$HEAD_NODE" hostname -I | awk '{print $1}')
if [[ -z "$HEAD_IP" ]]; then
    echo "ERROR: Could not determine the Ray head-node IP."
    exit 1
fi

RAY_PORT=6379
RAY_ADDRESS="${HEAD_IP}:${RAY_PORT}"
RAY_CPUS="${SLURM_CPUS_ON_NODE:-${SLURM_CPUS_PER_TASK:-1}}"

cleanup() {
    srun --overlap --nodes="${#nodes[@]}" --ntasks="${#nodes[@]}" --ntasks-per-node=1 \
        ray stop --force >/dev/null 2>&1 || true
}
trap cleanup EXIT

# The head only coordinates and runs ordinary tasks like every other node.
if ! srun --nodes=1 --ntasks=1 --nodelist="$HEAD_NODE" \
    ray start --head --node-ip-address="$HEAD_IP" --port="$RAY_PORT" \
    --num-cpus="$RAY_CPUS" --disable-usage-stats; then
    echo "ERROR: Could not start the Ray head on $HEAD_NODE."
    exit 1
fi

split=$(( ${#nodes[@]} - TIMING_NODES ))
compute_workers=("${nodes[@]:1:$((split - 1))}")
timing_workers=("${nodes[@]:$split}")

start_workers() {  # $1 = cpus advertised to Ray, $2 = extra ray args, rest = nodes
    local cpus="$1" extra="$2"; shift 2
    (( $# > 0 )) || return 0
    local list
    list=$(IFS=,; echo "$*")
    srun --nodes="$#" --ntasks="$#" --ntasks-per-node=1 --nodelist="$list" \
        ray start --address="$RAY_ADDRESS" --num-cpus="$cpus" \
        --disable-usage-stats $extra
}

start_workers "$RAY_CPUS" "" "${compute_workers[@]}" || { echo "ERROR: compute workers failed"; exit 1; }
# 0 schedulable CPUs: only tasks asking for the "timing" resource land here,
# and each gets the whole (otherwise idle) node.
start_workers 0 "--resources={\"timing\":1}" "${timing_workers[@]}" || { echo "ERROR: timing workers failed"; exit 1; }

# Wait until every node has joined.
for _ in $(seq 1 60); do
    joined=$(python - <<'PY'
import ray
ray.init(address="auto", logging_level="ERROR")
print(sum(1 for n in ray.nodes() if n["Alive"]))
PY
)
    (( joined >= ${#nodes[@]} )) && break
    sleep 5
done
echo "Ray nodes joined: ${joined:-0}/${#nodes[@]}"

# ============================================================
# Run everything
# ============================================================

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/suite-$SLURM_JOB_ID}"
echo "Output: $OUTPUT_ROOT"

python ray_suite.py --address "$RAY_ADDRESS" --output-root "$OUTPUT_ROOT" ${SUITE_ARGS:-}
STATUS=$?

echo "=========================================="
echo "Suite finished (exit $STATUS)"
echo "=========================================="
exit $STATUS
