#!/usr/bin/env bash
#SBATCH --job-name=ccka-experiments
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

# Add the cluster's partition/account directives above if your site requires them.
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

if [[ -f .venv/bin/activate ]]; then
    source .venv/bin/activate
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG="${CONFIG:-experiments/configs/default.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-results/slurm-${SLURM_JOB_ID:-manual}}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "Python executable not found: $PYTHON_BIN" >&2
    echo "Set PYTHON_BIN or activate the experiment environment before submitting." >&2
    exit 1
fi

if [[ ! -f "$CONFIG" ]]; then
    echo "Experiment config not found: $CONFIG" >&2
    exit 1
fi

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MPLBACKEND=Agg

mkdir -p "$OUTPUT_DIR"

echo "Project: $PROJECT_ROOT"
echo "Config: $CONFIG"
echo "Output: $OUTPUT_DIR"
echo "CPUs: ${SLURM_CPUS_PER_TASK:-unknown}"

"$PYTHON_BIN" main.py \
    --config "$CONFIG" \
    --address local \
    --output-dir "$OUTPUT_DIR"

"$PYTHON_BIN" experiments/plot_results.py \
    --input "$OUTPUT_DIR" \
    --output "$OUTPUT_DIR/plots"

echo "Experiment run and plotting completed: $OUTPUT_DIR"
