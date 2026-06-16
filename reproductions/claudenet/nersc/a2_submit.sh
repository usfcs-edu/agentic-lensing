#!/bin/bash
# ClaudeNet v3 A2 pipeline DAG: stage (CPU) -> retrain x5 (GPU) -> eval+verdict (GPU).
# Run from ~/claudenet on a Perlmutter login node:  bash nersc/a2_submit.sh
set -euo pipefail
cd "$HOME/claudenet"
LOG="$SCRATCH/claudenet/logs"; mkdir -p "$LOG"

# 1. STAGE (CPU, cosmo): gather train-pool FITS from DR10 shards + build blend manifests
STAGE=$(sbatch --parsable -J cn-a2-stage -t 01:00:00 \
  -o "$LOG/a2-stage-%j.out" -e "$LOG/a2-stage-%j.err" \
  --export=ALL,CMD='python 313_stage_mimic_fits.py --mode train-pool --n 8000 && python 314_build_blends.py' \
  nersc/shared_cpu.slurm)
echo "STAGE=$STAGE"

# 2. RETRAIN (GPU, cosmo_g) x5, each after the stage job
TR=()
submit_train () {  # member variant blendtag
  local jid
  jid=$(sbatch --parsable -J "cn-a2-$1-$2" -t 03:00:00 \
    --dependency=afterok:$STAGE \
    -o "$LOG/a2-train-$1-$2-%j.out" -e "$LOG/a2-train-$1-$2-%j.err" \
    --export=ALL,CMD="python 121_retrain_mined_members.py --member $1 --variant $2 --mined-manifest data/v2/blend_$3.parquet --n-mine 10000" \
    nersc/shared_gpu.slurm)
  echo "TRAIN $1 $2 = $jid"; TR+=("$jid")
}
submit_train effnet_S2  b30 b30
submit_train effnet_S2  b50 b50
submit_train effnet_S2  b70 b70
submit_train effnet_B3  b50 b50
submit_train resnet46_C b50 b50
DEP=$(IFS=:; echo "${TR[*]}")

# 3. EVAL (GPU, cosmo_g): score 10 members on held-out mimic-eval + v2-vs-v3 verdict
EVAL=$(sbatch --parsable -J cn-a2-eval -t 02:00:00 \
  --dependency=afterok:$DEP \
  -o "$LOG/a2-eval-%j.out" -e "$LOG/a2-eval-%j.err" \
  --export=ALL,CMD='python 315_score_mimic_eval.py --scorelist data/v3/a2_scorelist.json && python 316_v3_verdict.py' \
  nersc/shared_gpu.slurm)
echo "EVAL=$EVAL"
echo "A2_JOBS stage=$STAGE train=${TR[*]} eval=$EVAL"
