#!/bin/bash
# DR11-south full re-sweep: score the 3 DR11 fine-tuned members on the existing parent cutouts.
# 32-task array, one part each (all shards), 3 members via 112 --only-extra. effnet_B + zoobot_N
# already scored in the original stage-1 (reused at combine). cosmo_g GPU.
#   sbatch nersc/dr11_resweep_score.slurm
#SBATCH -A cosmo_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 60
#SBATCH -N 1
#SBATCH --gpus 1
#SBATCH -J dr11resweep
#SBATCH --array=0-31
#SBATCH -o /global/homes/g/gdbenson/claudenet/logs/dr11resweep_%a.out
module load pytorch/2.8.0
export HF_HOME=$SCRATCH/claudenet/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd ~/claudenet
NN=$(printf "%02d" "$SLURM_ARRAY_TASK_ID")
OUT=$SCRATCH/claudenet/scores/dr11s_resweep
mkdir -p "$OUT"
echo "[resweep] part $NN start $(date)"
python -u 112_score_pool.py --only-extra --extra-ckpt-dir data/v2/ckpt_dr11 \
   --cutout-root "$SCRATCH/claudenet/cutouts/sweep_dr11_south/part$NN" \
   --out "$OUT/stage1_part${NN}_dr11.parquet"
echo "[resweep] part $NN done $(date) rc=$?"
