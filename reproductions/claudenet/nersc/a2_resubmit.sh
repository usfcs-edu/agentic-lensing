#!/bin/bash
# A2 fix: efficientnet retrains exceeded the 3h shared-GPU limit (25 ep effnetv2_s is
# I/O-bound on per-FITS Lustre reads ~3.1-3.3h). resnet46_C_b50 already COMPLETED. Cancel
# the doomed runs + the orphaned eval, resubmit the 4 efficientnet trains at 6h (unbuffered),
# re-chain eval on them (resnet ckpt already on disk). Stage FITS/blends persist.
set -euo pipefail
cd "$HOME/claudenet"
LOG="$SCRATCH/claudenet/logs"; mkdir -p "$LOG"
scancel 54524184 54524187 54524192 2>/dev/null || true   # running effnets + orphan eval
sleep 3
TR=()
submit_train () {  # member variant blendtag
  local jid
  jid=$(sbatch --parsable -J "cn-a2-$1-$2" -t 06:00:00 \
    -o "$LOG/a2-train-$1-$2-%j.out" -e "$LOG/a2-train-$1-$2-%j.err" \
    --export=ALL,CMD="PYTHONUNBUFFERED=1 python -u 121_retrain_mined_members.py --member $1 --variant $2 --mined-manifest data/v2/blend_$3.parquet --n-mine 10000" \
    nersc/shared_gpu.slurm)
  echo "TRAIN $1 $2 = $jid"; TR+=("$jid")
}
submit_train effnet_S2 b30 b30
submit_train effnet_S2 b50 b50
submit_train effnet_S2 b70 b70
submit_train effnet_B3 b50 b50
DEP=$(IFS=:; echo "${TR[*]}")
EVAL=$(sbatch --parsable -J cn-a2-eval -t 02:00:00 \
  --dependency=afterok:$DEP \
  -o "$LOG/a2-eval-%j.out" -e "$LOG/a2-eval-%j.err" \
  --export=ALL,CMD='PYTHONUNBUFFERED=1 python -u 315_score_mimic_eval.py --scorelist data/v3/a2_scorelist.json && python -u 316_v3_verdict.py' \
  nersc/shared_gpu.slurm)
echo "EVAL=$EVAL"
echo "A2_RESUBMIT train=${TR[*]} eval=$EVAL"
