#!/bin/bash
#SBATCH --job-name=ccka-experiment
#SBATCH --partition=gpu-large
#SBATCH --nodes=26
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:2
#SBATCH --time=12:00:00
#SBATCH --output=/home/cdacB/digvijaysinha.cdacb/CCKA-Clustered_Centroid_Kernel_Alignment/output/%j.out
#SBATCH --error=/home/cdacB/digvijaysinha.cdacb/CCKA-Clustered_Centroid_Kernel_Alignment/output/%j.err

# ============================================================
# Environment
# ============================================================
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
export RAY_raylet_start_wait_time_s=120
export RAY_BACKEND_LOG_LEVEL=debug
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

# Select the experiment with: sbatch --export=ALL,CONFIG=configs/exp06_k_sweep.yaml \
#   --nodes=8 jobs/run_experiments.slurm
CONFIG="${CONFIG:-configs/default.yaml}"
RUN_NAME="$(basename "$CONFIG" .yaml)"
OUTPUT_DIR="${OUTPUT_DIR:-results/$RUN_NAME}"
echo "CONFIG: $CONFIG -> $OUTPUT_DIR"


python main.py --config "$CONFIG" --address auto --output-dir "$OUTPUT_DIR"
STATUS=$?


echo "=========================================="
echo "Execution Complete (exit $STATUS)"
echo "=========================================="
exit $STATUS
