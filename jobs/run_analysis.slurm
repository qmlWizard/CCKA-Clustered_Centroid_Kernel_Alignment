#!/bin/bash
#SBATCH --job-name=ccka-analysis
#SBATCH --partition=terai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/home/cdacB/digvijaysinha.cdacb/CCKA-Clustered_Centroid_Kernel_Alignment/output/%j.out
#SBATCH --error=/home/cdacB/digvijaysinha.cdacb/CCKA-Clustered_Centroid_Kernel_Alignment/output/%j.err

# Single-node analysis jobs (no Ray). Select with:
#   sbatch --export=ALL,ANALYSIS=scalability jobs/run_analysis.slurm
# ANALYSIS in: budget | scalability | kernel | ansatz | stats | plots

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


export PYTHONPATH="$PWD:$PYTHONPATH"
ANALYSIS="${ANALYSIS:?set ANALYSIS to budget|scalability|kernel|ansatz|stats|plots}"

case "$ANALYSIS" in
    budget)
        # Section 2: accuracy vs circuit evaluations on corners/checkerboard
        # (plus spirals, whose claim is about the budget). 10 seeds.
        for dataset in corners checkerboard spirals; do
            for seed in 0 1 2 3 4 5 6 7 8 9; do
                python -m experiments.accuracy_vs_executions \
                    --config configs/base.yaml --dataset "$dataset" --seed "$seed" \
                    --methods fullKTA quackKTA ccka --centroids 4 \
                    --target-accuracy 0.9 --run-full-iterations \
                    --output-prefix "results/budget/${dataset}_seed${seed}"
            done
        done
        ;;
    scalability)
        python -m experiments.scalability --output-dir results/scalability
        ;;
    kernel)
        python -m experiments.kernel_analysis --output-dir results/kernel_analysis
        ;;
    ansatz)
        python -m experiments.ansatz_analysis --output-dir results/ansatz_analysis \
            --accuracy-csv results/exp13_ansatz_depth/summary.csv
        ;;
    stats)
        python -m experiments.stats_report results/exp01_main_benchmark \
            --output-dir results/stats/main_benchmark
        python -m experiments.stats_report results/exp15_synthetic_claims \
            --output-dir results/stats/synthetic_claims
        ;;
    plots)
        python -m experiments.suite_plots --results-dir results --output-dir results/plots
        ;;
    *)
        echo "Unknown ANALYSIS=$ANALYSIS"
        exit 2
        ;;
esac
