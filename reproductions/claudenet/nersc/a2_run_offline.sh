#!/bin/bash
# A2 retry with the hang fixed: build_model -> EfficientNetV2Lens(pretrained=True) was
# downloading ImageNet weights from HF Hub and hanging on the internet-less compute nodes
# (GPU 0%, 2:40 CPU over 4.5h). The weight cache is now populated in ~/.cache/huggingface
# (login-node download), so run with HF_HUB_OFFLINE=1 -> instant cache build, then real
# GPU training (~30-90 min/25-epoch effnet). resnet46_C_b50 already COMPLETED; stage
# FITS/blends persist. -t 04:00:00 (ample now that training actually runs).
set -euo pipefail
cd "$HOME/claudenet"
LOG="$SCRATCH/claudenet/logs"; mkdir -p "$LOG"
OFF="HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1"
TR=()
submit_train () {  # member variant blendtag
  local jid
  jid=$(sbatch --parsable -J "cn-a2-$1-$2" -t 04:00:00 \
    -o "$LOG/a2-train-$1-$2-%j.out" -e "$LOG/a2-train-$1-$2-%j.err" \
    --export=ALL,CMD="$OFF python -u 121_retrain_mined_members.py --member $1 --variant $2 --mined-manifest data/v2/blend_$3.parquet --n-mine 10000" \
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
  --export=ALL,CMD="$OFF python -u 315_score_mimic_eval.py --scorelist data/v3/a2_scorelist.json && python -u 316_v3_verdict.py" \
  nersc/shared_gpu.slurm)
echo "EVAL=$EVAL"
echo "A2_RUN_OFFLINE train=${TR[*]} eval=$EVAL"
