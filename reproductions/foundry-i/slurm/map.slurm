#!/bin/bash
#SBATCH -J foundry-map
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -G 4
#SBATCH -q regular
#SBATCH -t 02:00:00
#SBATCH -A deepsrch_g
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH -o %x-%j.out

set -x
source $HOME/foundry-i/venv/bin/activate
cd $HOME/foundry-i

# args via --export: N_SAMPLES, NUM_STEPS, TAG, EXTRA
srun -n 1 python 41_map_paper_scale.py \
    --n-samples ${N_SAMPLES:-1000} --num-steps ${NUM_STEPS:-2000} \
    --tag ${TAG:-prod} ${EXTRA:-}
