#!/bin/bash

source ./kgreedy/bin/activate

# Array of strategies and arguments
dataset=("microgrid" "checkerboard" "linear" "powerline")
strategies=("approx_greedy" "random" "approx_greedy_prob")
args=(1 2 4 8 12)
ansatz=("strong_entangled" "tutorial_ansatz" "basic_entangled")

# Run the Python script in parallel with different strategies and arguments
parallel --jobs 30 --bar python3 main.py {1} {2} {3} {4} ::: "${dataset[@]}"  ::: "${strategies[@]}" ::: "${ansatz[@]}" ::: "${args[@]}"

