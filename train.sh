#!/bin/bash

# Array of strategies and arguments
strategies=("random" "greedy" "probabilistic")
args=(4 6 8 10 12 18 24)

# Run the Python script in parallel with different strategies and arguments
parallel python3 main.py checkerboard {1} {2} ::: "${strategies[@]}" ::: "${args[@]}"

