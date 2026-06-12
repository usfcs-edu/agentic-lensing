#!/bin/bash
#SBATCH -J foundry-hmc
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -G 4
#SBATCH -q regular
#SBATCH -t 01:30:00
#SBATCH -A deepsrch_g
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH -o %x-%j.out

set -x
source $HOME/foundry-i/venv/bin/activate
cd $HOME/foundry-i

srun -n 1 python 43_hmc_paper_scale.py \
    --qz ${QZ:-data/svi_v12_prod.npz} \
    --n-hmc ${N_HMC:-48} --burn ${BURN:-250} --results ${RESULTS:-750} \
    --tag ${TAG:-prod} ${EXTRA:-}

# diagnostics in the same job so results come back complete
srun -n 1 python 44_diagnostics.py --run data/hmc_v13_${TAG:-prod}.npz
