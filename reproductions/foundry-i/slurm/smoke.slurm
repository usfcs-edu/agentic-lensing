#!/bin/bash
#SBATCH -J foundry-smoke
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -G 4
#SBATCH -q debug
#SBATCH -t 00:20:00
#SBATCH -A deepsrch_g
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH -o %x-%j.out

set -x
source $HOME/foundry-i/venv/bin/activate
cd $HOME/foundry-i

nvidia-smi --query-gpu=index,name,memory.total --format=csv
python -c "import jax; print('jax', jax.__version__, jax.devices())"

# tiny end-to-end: MAP smoke through the sean ModellingSequence
srun -n 1 python 41_map_paper_scale.py --n-samples 64 --num-steps 100 --tag pm_smoke
