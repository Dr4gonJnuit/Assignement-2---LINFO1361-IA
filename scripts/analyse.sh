#!/bin/bash

# Constants
PYTHON_FILES="../python"

agents=(random alphabeta agent)

for agent in "${agents[@]}"; do
    echo "Analyzing $agent"
    python3 $PYTHON_FILES/main.py -w $agent -b random | awk 'NR==3'
done