#!/bin/bash
#SBATCH --job-name=compare_ansatz
#SBATCH --partition=shiwalik
#SBATCH --nodes=128
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --time=24:00:00
#SBATCH --output=/home/cdacB/digvijaysinha.cdacb/CCKA-Clustered_Centroid_Kernel_Alignment/output/%j.out
#SBATCH --error=/home/cdacB/digvijaysinha.cdacb/CCKA-Clustered_Centroid_Kernel_Alignment/error/%j.err

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

# ============================================================
# Get allocated nodes
# ============================================================

nodes=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))

echo "=========================================="
echo "SLURM JOB ID      : $SLURM_JOB_ID"
echo "SLURM JOB NAME    : $SLURM_JOB_NAME"
echo "NUMBER OF NODES   : $SLURM_JOB_NUM_NODES"
echo "CPUS PER NODE     : $SLURM_CPUS_PER_TASK"
echo "=========================================="

# ============================================================
# Run Compare Ansatz Comparision benchmark
# ============================================================

echo "=========================================="
echo "Starting Ansatz Comparision benchmark"
echo "=========================================="


python main.py --config configs/exp13_ansatz_depth.yaml

echo "=========================================="
echo "Execution Complete!!!"
echo "=========================================="