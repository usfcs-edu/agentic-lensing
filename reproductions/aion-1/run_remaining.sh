#!/bin/bash
# Master driver for the remaining tasks. Waits for the fixes to free GPUs 0/2,
# then runs each task as its (slow, rate-limited cutout) data becomes ready.
cd /home2/benson/git/agentic-lensing/reproductions/aion-1
export HF_HOME=/home2/benson/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home2/benson/.venvs/aion/bin/python
G=0,2,3,4,5,6   # healthy GPUs (exclude GPU1 thermal throttler)
run(){ echo "##### $1 @ $(date) #####"; shift; "$@" || echo "STEP_FAILED: $*"; }
waitfor(){ local marker=$1 logf=$2 maxs=${3:-7200}; local d=$((SECONDS+maxs));
  until grep -q "$marker" "$logf" 2>/dev/null || [ $SECONDS -ge $d ]; do sleep 30; done; }

waitfor FIXES_DONE data/results/_fixes.log 9000
echo "WATCH: fixes done @ $(date)"

# TASK 9 -- strong-lens retrieval (needs SuGOHI lens cutouts)
waitfor SUGOHI_OK data/results/_sugohi.log 7200
run "TASK 9 lens retrieval" $PY -u 42_retrieve_lenses.py --gpus $G

# TASK 5 -- GZ3D segmentation (needs gz3d image/mask pairs; 100-min deadline)
waitfor GZ3D_OK data/results/_gz3d.log 6000
run "TASK 5 segmentation" $PY -u 31_seg_gz3d.py --gpus $G

# TASK 4/7/8 -- re-run on the grown Galaxy10 corpus (100-min deadline on growth)
waitfor "LS_IMAGES_OK\|cutouts 17" data/results/_gz10_grow.log 6000
rm -f data/emb/gz10_*.npy
run "GZ10 refinalize (larger)" $PY -u 06b_finalize_gz10.py --subsample 14000
run "TASK 4 morphology"        bash -c "$PY -u 12_embed_gz10.py --gpus $G && $PY -u 22_probe_gz10.py"
run "TASK 7/8 retrieval"       $PY -u 40_retrieve_gz10.py

run "TABLES+FIGS" bash -c "$PY -u 60_make_tables.py && $PY -u 61_make_figures.py"
echo "REMAINING_DONE @ $(date)"
