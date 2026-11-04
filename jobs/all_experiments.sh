#!/bin/bash
#SBATCH --job-name=ccka_all_experiments
#SBATCH --partition=shiwalik
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --time=7-00:00:00
#SBATCH --output=/home/cdacB/digvijaysinha.cdacb/CCKA-Clustered_Centroid_Kernel_Alignment/output/%j.out
#SBATCH --error=/home/cdacB/digvijaysinha.cdacb/CCKA-Clustered_Centroid_Kernel_Alignment/error/%j.err

cd "$(dirname "${BASH_SOURCE[0]}")/.."

. /home/apps/spack/share/spack/setup-env.sh

spack load /oxmullt
spack load /gaily2jw
spack load /2wdhn7r

export OPENBLAS_ROOT=$(spack location -i /geypa2)
export LD_LIBRARY_PATH="$OPENBLAS_ROOT/lib:$LD_LIBRARY_PATH"

export MPI_LIB=/home/apps/spack/opt/spack/linux-cascadelake/openmpi-4.1.8-gaily2jwc3klcubp55unnlgiaglqu7mf/lib
export LD_LIBRARY_PATH="$MPI_LIB:$LD_LIBRARY_PATH"

conda activate py310

echo "=========================================="
echo "SLURM JOB ID      : $SLURM_JOB_ID"
echo "SLURM JOB NAME    : $SLURM_JOB_NAME"
echo "NUMBER OF NODES   : $SLURM_JOB_NUM_NODES"
echo "CPUS PER NODE     : $SLURM_CPUS_PER_TASK"
echo "TOTAL CPUs        : $((SLURM_JOB_NUM_NODES * SLURM_CPUS_PER_TASK))"
echo "=========================================="

cleanup_ray() {
    local nodes="${SLURM_JOB_NUM_NODES:-1}"
    if command -v srun >/dev/null 2>&1 && (( nodes > 1 )); then
        srun --nodes="$nodes" --ntasks="$nodes" --ntasks-per-node=1 --overlap \
            bash -lc 'ray stop --force >/dev/null 2>&1 || true' \
            >/dev/null 2>&1 || true
    elif command -v ray >/dev/null 2>&1; then
        ray stop --force >/dev/null 2>&1 || true
    fi
}

trap 'exit 130' INT
trap 'exit 143' TERM
trap cleanup_ray EXIT

configs=(
    configs/exp01_main_benchmark.yaml
    configs/exp02_multiclass.yaml
    configs/exp03_ablation_structure.yaml
    configs/exp04_ablation_lambda_grid.yaml
    configs/exp05_ablation_lr_grid.yaml
    configs/exp06_k_sweep.yaml
    configs/exp07_ntrain.yaml
    configs/exp08_label_noise.yaml
    configs/exp09_imbalance.yaml
    configs/exp10_feature_dim.yaml
    configs/exp11_noise.yaml
    configs/exp12_shots.yaml
    configs/exp13_ansatz_depth.yaml
    configs/exp14_optimizer_init.yaml
    configs/exp15_synthetic_claims.yaml
    configs/exp16_decision_boundaries.yaml
)

failures=0
for config in "${configs[@]}"; do
    echo "=========================================="
    echo "Starting experiment: $config"
    echo "=========================================="

    cleanup_ray
    if python main.py --config "$config"; then
        echo "Completed experiment: $config"
    else
        status=$?
        failures=$((failures + 1))
        echo "Experiment failed (status $status): $config" >&2
    fi
    cleanup_ray
done

echo "Completed all configured experiments; failures: $failures"
exit "$failures"
