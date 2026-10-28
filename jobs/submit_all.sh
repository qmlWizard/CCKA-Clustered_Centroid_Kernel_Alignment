#!/bin/bash
# Submit the default configured experiment as one Python-managed Ray job.
# Usage: bash jobs/submit_all.sh [extra sbatch args]
#   e.g. bash jobs/submit_all.sh --nodes=40
# Select another configuration with:
#   sbatch --export=ALL,CONFIG=configs/exp06_k_sweep.yaml jobs/run_experiments.slurm
set -euo pipefail
cd "$(dirname "$0")/.."
sbatch "$@" jobs/run_all.slurm
