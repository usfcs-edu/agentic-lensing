#!/bin/bash
# M1 GPU batch: task 3 (gaia/apogee), task 4 (gz10 morphology), tasks 7/8
# (retrieval), plus task 10 (redshift posterior) and task 6 (low-data).
# Assumes the flagship image-config embed has freed the GPUs.
set -e
cd /home2/benson/git/agentic-lensing/reproductions/aion-1
export HF_HOME=/home2/benson/.cache/huggingface
PY=/home2/benson/.venvs/aion/bin/python

echo "########## TASK 3 gaia x apogee @ $(date) ##########"
$PY -u 11_embed_gaia_xp.py
$PY -u 21_probe_gaia_apogee.py

echo "########## TASK 4/7/8 gz10 @ $(date) ##########"
$PY -u 12_embed_gz10.py
$PY -u 22_probe_gz10.py
$PY -u 40_retrieve_gz10.py

echo "########## TASK 10 redshift posterior @ $(date) ##########"
$PY -u 50_redshift_posterior.py --variant base --n 2000

echo "########## TASK 6 low-data @ $(date) ##########"
$PY -u 25_lowdata.py --variant base

echo "M1_GPU_DONE"
