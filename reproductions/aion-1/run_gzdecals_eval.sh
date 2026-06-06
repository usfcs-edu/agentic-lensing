#!/bin/bash
# Watcher: run the faithful GZ-DECaLS retrieval (tasks 7/8) at a corpus
# milestone and again at campaign completion (the multi-day cutout fetch).
cd /home2/benson/git/agentic-lensing/reproductions/aion-1
export HF_HOME=/home2/benson/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home2/benson/.venvs/aion/bin/python
run(){ echo "##### $1 @ $(date) #####"; shift; "$@" || echo "STEP_FAILED: $*"; }
waitfor(){ local m=$1 f=$2 maxs=${3:-400000}; local d=$((SECONDS+maxs));
  until grep -q "$m" "$f" 2>/dev/null || [ $SECONDS -ge $d ]; do sleep 120; done; }

# intermediate: ~all positives + ~16k distractors cached
waitfor "campaign 40000" data/results/_gzdecals_campaign.log 400000
run "GZ-DECaLS retrieval (intermediate corpus)" $PY -u 44_retrieve_gzdecals.py --gpus 0,2,3,4,5,6
run "tables" $PY -u 60_make_tables.py

# final: full 63k corpus
waitfor "GZDECALS_CAMPAIGN_OK" data/results/_gzdecals_campaign.log 400000
run "GZ-DECaLS retrieval (full corpus)" $PY -u 44_retrieve_gzdecals.py --gpus 0,2,3,4,5,6
run "tables+figs" bash -c "$PY -u 60_make_tables.py && $PY -u 61_make_figures.py"
echo "GZDECALS_EVAL_DONE @ $(date)"
