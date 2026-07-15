#!/bin/bash

# Put this in your home folder alongside environment.yaml and it should set everything up!
# Super simple, this is just very slightly more convenient than a google doc since it get shared alongside the environment.yaml file
# This should be safe to run multiple times, the commands have safeties built in, and their default behavior is to just delete and reinitialize 

CONDA_NAME="modern_env"
KERNEL_NAME="Gigalens_2.1_Fixed"
ENVIRONMENT_FILE="./modern-env.yaml"

# 1. Create the environment
conda env create -f $ENVIRONMENT_FILE -n $CONDA_NAME
echo "Created conda environment."

# 2. Register the Jupyter kernel using 'conda run' to target the correct Python instance
conda run -n $CONDA_NAME python -m ipykernel install --user --name "$CONDA_NAME" --display-name "$KERNEL_NAME"
echo "Created jupyter kernel."

# And that's it! It doesn't have its own version of CUDA, but ideally we should be using the NERSC default