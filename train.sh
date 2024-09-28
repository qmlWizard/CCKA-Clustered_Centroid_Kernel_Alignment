#!/bin/bash

source ./kgreedy/bin/activate

# Array of strategies and arguments
dataset=("microgrid" "checkerboard" "linear" "powerline")
strategies=("approx_greedy" "random")
args=(1 2 4 6 8 10 12)

# Run the Python script in parallel with different strategies and arguments
parallel python3 main.py {1} {2} {3} ::: "${dataset[@]}"  ::: "${strategies[@]}" ::: "${args[@]}"

