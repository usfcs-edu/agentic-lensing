#!/bin/bash
cd /home2/benson/git/agentic-lensing/reproductions/aion-1
export HF_HOME=/home2/benson/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home2/benson/.venvs/aion/bin/python
run(){ echo "##### $1 @ $(date) #####"; shift; "$@" || echo "STEP_FAILED: $*"; }
# task 10 FIRST (fast + important: the redshift-posterior figure)
run "task10 redshift posterior"    bash -c "CUDA_VISIBLE_DEVICES=2 $PY -u 50_redshift_posterior.py --variant base --n 2000"
# xlarge image-config probes (skip base/large -> fresh process; larger batch now eval is fixed)
run "probe phot_image xlarge"      bash -c "CUDA_VISIBLE_DEVICES=0 $PY -u 20_probe_provabgs.py --config phot_image"
run "probe phot_image_spec xlarge" bash -c "CUDA_VISIBLE_DEVICES=0 $PY -u 20_probe_provabgs.py --config phot_image_spec"
echo "FIXES_DONE @ $(date)"
