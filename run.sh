#!/bin/bash
set -e  # Exit immediately if a command fails

# Activate virtual environment
source .venv/bin/activate

# Array of configs
configs=(
    "checkerboard"
    "checkerboard_quack"
    "corners"
    "corners_quack"
    "double_cake"
    "double_cake_quack"
    "moons"
    "moons_quack"
    "donuts"
    "donuts_quack"
)

# Loop through and run
for cfg in "${configs[@]}"; do
    echo "Running config: $cfg"
    python3 main.py --backend pennylane --config "configs/extended/${cfg}.yaml"
done

echo "Thank You!"

