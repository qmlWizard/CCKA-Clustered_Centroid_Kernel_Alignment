#!/bin/bash
#SBATCH --job-name=ccka-experiments
#SBATCH --partition=terai
#SBATCH --nodes=25
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/home/cdacB/digvijaysinha.cdacb/CCKA-Clustered_Centroid_Kernel_Alignment/output/%j.out
#SBATCH --error=/home/cdacB/digvijaysinha.cdacb/CCKA-Clustered_Centroid_Kernel_Alignment/output/%j.err

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

echo "=========================================="
echo "SLURM JOB ID      : $SLURM_JOB_ID"
echo "SLURM JOB NAME    : $SLURM_JOB_NAME"
echo "NUMBER OF NODES   : $SLURM_JOB_NUM_NODES"
echo "CPUS PER NODE     : $SLURM_CPUS_PER_TASK"
echo "=========================================="

# ============================================================
# Start Ray head
# ============================================================

# Determine the IP address from the first allocated node.
HEAD_IP=$(srun \
    --nodes=1 \
    --ntasks=1 \
    --nodelist="$HEAD_NODE" \
    hostname -I | awk '{print $1}')

if [[ -z "$HEAD_IP" ]]; then
    echo "ERROR: Could not determine the Ray head-node IP."
    exit 1
fi

RAY_PORT=6379
RAY_ADDRESS="${HEAD_IP}:${RAY_PORT}"
RAY_CPUS="${SLURM_CPUS_ON_NODE:-${SLURM_CPUS_PER_TASK:-1}}"

if ! srun \
    --nodes=1 \
    --ntasks=1 \
    --nodelist="$HEAD_NODE" \
    ray start \
    --head \
    --node-ip-address="$HEAD_IP" \
    --port="$RAY_PORT" \
    --num-cpus="$RAY_CPUS" \
    --disable-usage-stats; then
    echo "ERROR: Could not start the Ray head on $HEAD_NODE."
    exit 1
fi

worker_nodes=("${nodes[@]:1}")
if ((${#worker_nodes[@]} > 0)); then
    WORKER_NODELIST=$(IFS=,; echo "${worker_nodes[*]}")
    if ! srun \
        --nodes="${#worker_nodes[@]}" \
        --ntasks="${#worker_nodes[@]}" \
        --ntasks-per-node=1 \
        --nodelist="$WORKER_NODELIST" \
        ray start \
        --address="$RAY_ADDRESS" \
        --num-cpus="$RAY_CPUS" \
        --disable-usage-stats; then
        echo "ERROR: Could not start Ray workers on the allocated nodes."
        exit 1
    fi
fi

# Select the experiment with: sbatch --export=ALL,CONFIG=configs/exp06_k_sweep.yaml \
#   --nodes=8 jobs/run_experiments.slurm
CONFIG="${CONFIG:-configs/default.yaml}"
RUN_NAME="$(basename "$CONFIG" .yaml)"
OUTPUT_DIR="${OUTPUT_DIR:-results/$RUN_NAME}"
echo "CONFIG: $CONFIG -> $OUTPUT_DIR"

python main.py --config "$CONFIG" --address="$RAY_ADDRESS" --output-dir "$OUTPUT_DIR"
STATUS=$?

ray stop --force || true


echo "=========================================="
echo "Execution Complete (exit $STATUS)"
echo "=========================================="
exit $STATUS
