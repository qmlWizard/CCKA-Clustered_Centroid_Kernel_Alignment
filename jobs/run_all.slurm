#!/bin/bash
#SBATCH --job-name=ccka-experiment
#SBATCH --partition=terai
#SBATCH --nodes=40
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --time=24:00:00
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
# OpenBLAS and MPI
# ============================================================

export OPENBLAS_ROOT="$(spack location -i /geypa2)"
export LD_LIBRARY_PATH="${OPENBLAS_ROOT}/lib:${LD_LIBRARY_PATH:-}"

export MPI_LIB="/home/apps/spack/opt/spack/linux-cascadelake/openmpi-4.1.8-gaily2jwc3klcubp55unnlgiaglqu7mf/lib"
export LD_LIBRARY_PATH="${MPI_LIB}:${LD_LIBRARY_PATH:-}"

# ============================================================
# Conda and Ray
# ============================================================

source "$(conda info --base)/etc/profile.d/conda.sh"
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

echo "=================================================="
echo "SLURM JOB ID       : $SLURM_JOB_ID"
echo "SLURM JOB NAME     : $SLURM_JOB_NAME"
echo "NUMBER OF NODES    : $SLURM_JOB_NUM_NODES"
echo "CPUS PER TASK      : $SLURM_CPUS_PER_TASK"
echo "HEAD NODE          : $HEAD_NODE"
echo "=================================================="

# ============================================================
# Discover configurations
# ============================================================

# Discover all YAML files under configs/.
# Exclude the optional manifest file if you create one.
mapfile -t CONFIGS < <(
    find configs -type f \
        \( -name '*.yaml' -o -name '*.yml' \) \
        | sort
)

TOTAL=${#CONFIGS[@]}

if [[ "$TOTAL" -eq 0 ]]; then
    echo "ERROR: No YAML configurations found under configs/"
    exit 1
fi

echo "Total configurations: $TOTAL"

# ============================================================
# Run configurations sequentially
# ============================================================

FAILED=0
COMPLETED=0
FAILED_CONFIGS=()

for CONFIG in "${CONFIGS[@]}"; do

    INDEX=$((COMPLETED + FAILED + 1))
    CONFIG_NAME="$(basename "$CONFIG")"
    CONFIG_NAME="${CONFIG_NAME%.*}"

    OUTPUT_DIR="results/${CONFIG_NAME}"

    mkdir -p "$OUTPUT_DIR"

    echo
    echo "=================================================="
    echo "CONFIGURATION $INDEX / $TOTAL"
    echo "CONFIG        : $CONFIG"
    echo "OUTPUT DIR    : $OUTPUT_DIR"
    echo "START TIME    : $(date)"
    echo "=================================================="

    # Run the current configuration.
    if python main.py \
        --config "$CONFIG" \
        --address auto \
        --output-dir "$OUTPUT_DIR"
    then
        COMPLETED=$((COMPLETED + 1))
        echo "STATUS: SUCCESS"
    else
        STATUS=$?
        FAILED=$((FAILED + 1))
        FAILED_CONFIGS+=("$CONFIG")

        echo "STATUS: FAILED (exit code $STATUS)"
        echo "Continuing with the next configuration..."
    fi

    echo "FINISH TIME: $(date)"

done

# ============================================================
# Final summary
# ============================================================

echo
echo "=================================================="
echo "ALL CONFIGURATIONS PROCESSED"
echo "=================================================="
echo "TOTAL CONFIGURATIONS : $TOTAL"
echo "SUCCESSFUL           : $COMPLETED"
echo "FAILED               : $FAILED"
echo "RESULTS DIRECTORY    : results"
echo "=================================================="

if [[ "$FAILED" -gt 0 ]]; then
    echo "Failed configurations:"
    printf '  - %s\n' "${FAILED_CONFIGS[@]}"
    exit 1
fi

exit 0