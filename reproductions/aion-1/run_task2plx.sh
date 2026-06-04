#!/bin/bash
cd /home2/benson/git/agentic-lensing/reproductions/aion-1
export HF_HOME=/home2/benson/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home2/benson/.venvs/aion/bin/python
echo "## task2 desi_plx embed @ $(date)"
$PY -u 13_embed_ddpayne.py --config desi_plx --gpus 3,4,5,6
echo "## task2 desi_plx probe @ $(date)"
CUDA_VISIBLE_DEVICES=3 $PY -u 23_probe_ddpayne.py --config desi_plx
echo "TASK2_PLX_DONE @ $(date)"
