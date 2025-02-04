#!/bin/bash
SBATCH --job-name=checkerboard_efficient_hardware  # Job name
SBATCH --output=output_%j.log        # Output log file (%j expands to job ID)
SBATCH --error=error_%j.log          # Error log file
#SBATCH --time=02:00:00               # Job time limit (hh:mm:ss)
#SBATCH --ntasks=1                    # Number of tasks (1 for sequential execution)
SBATCH --cpus-per-task=24             # Number of CPU cores per task
SBATCH --mem=30G                      # Memory per node (adjust as needed)
#SBATCH --partition=standard          # Partition (adjust as per cluster)
#SBATCH --mail-type=END,FAIL          # Notifications for job completion/failure
#SBATCH --mail-user=your_email@example.com  # Email to receive notifications

# Load the Python module if required (uncomment if needed)
# module load python/3.9

echo "Starting job on $(hostname) at $(date)"

source .venv/bin/activate
# Run Python scripts one after another
python3 cross_validate.py --config configs/cross_validation/checkerboard.yaml

echo "Job completed at $(date)"
