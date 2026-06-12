#!/bin/bash
#SBATCH -J foundry-svi
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -G 4
#SBATCH -q regular
#SBATCH -t 03:00:00
#SBATCH -A deepsrch_g
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH -o %x-%j.out

set -x
source $HOME/foundry-i/venv/bin/activate
cd $HOME/foundry-i

srun -n 1 python 42_svi_paper_scale.py \
    --start ${START:-data/map_v11_prod.npz} \
    --n-vi ${N_VI:-500} --num-steps ${NUM_STEPS:-1500} \
    --tag ${TAG:-prod} ${EXTRA:-}
