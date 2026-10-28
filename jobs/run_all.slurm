#!/bin/bash
#SBATCH --job-name=ccka-suite
#SBATCH --partition=himadri
#SBATCH --nodes=513
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --time=06:00:00
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
export RAY_raylet_start_wait_time_s=120
export RAY_BACKEND_LOG_LEVEL=debug

echo "Host: $(hostname)"
which python
python -c "import ray; print('Ray:', ray.__version__)"
python -c "import grpc; print('gRPC:', grpc.__version__)"


ulimit -n 65535

# ============================================================
# Get allocated nodes
# ============================================================

nodes=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))

HEAD_NODE=${nodes[0]}

echo "=========================================="
echo "SLURM JOB ID      : $SLURM_JOB_ID"
echo "SLURM JOB NAME    : $SLURM_JOB_NAME"
echo "NUMBER OF NODES   : $SLURM_JOB_NUM_NODES"
echo "CPUS PER NODE     : $SLURM_CPUS_PER_TASK"
echo "HEAD NODE         : $HEAD_NODE"
echo "=========================================="

# ============================================================
# Run suite with Python-managed Ray initialization
# ============================================================

python ray_suite.py --address auto --output-root "$OUTPUT_ROOT" ${SUITE_ARGS:-}
STATUS=$?

echo "=========================================="
echo "Suite finished (exit $STATUS)"
echo "=========================================="
exit $STATUS

