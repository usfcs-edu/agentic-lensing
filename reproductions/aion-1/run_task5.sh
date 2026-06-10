#!/bin/bash
cd /home2/benson/git/agentic-lensing/reproductions/aion-1
export HF_HOME=/home2/benson/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home2/benson/.venvs/aion/bin/python
run(){ echo "##### $1 @ $(date) #####"; shift; "$@" || echo "STEP_FAILED: $*"; }
run "gz3d fetch (threaded)" $PY -u 30_fetch_gz3d.py --n 2000 --workers 12
run "task5 segmentation"    $PY -u 31_seg_gz3d.py --gpus 0,2,3,4,5,6
run "tables+figs"           bash -c "$PY -u 60_make_tables.py && $PY -u 61_make_figures.py"
echo "TASK5_DONE @ $(date)"
