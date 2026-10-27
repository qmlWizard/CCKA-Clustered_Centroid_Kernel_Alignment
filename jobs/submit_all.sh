#!/bin/bash
# Submit the whole suite as ONE Ray job (all experiments + analyses in parallel).
# Usage: bash jobs/submit_all.sh [extra sbatch args]
#   e.g. bash jobs/submit_all.sh --nodes=40
# The older per-config path is still available:
#   sbatch --export=ALL,CONFIG=configs/exp06_k_sweep.yaml jobs/run_experiments.slurm
set -euo pipefail
cd "$(dirname "$0")/.."
sbatch "$@" jobs/run_all.slurm
