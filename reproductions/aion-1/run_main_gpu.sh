#!/bin/bash
# Non-gz10 GPU tasks: 3 (gaia/apogee), 2 (ddpayne/desi), 10 (redshift),
# 6 (low-data), 11 (super-res). Runs right after the flagship frees the GPUs;
# does NOT depend on the slow gz10 cutout fetch. 6 healthy GPUs (GPU1 excluded).
cd /home2/benson/git/agentic-lensing/reproductions/aion-1
export HF_HOME=/home2/benson/.cache/huggingface
PY=/home2/benson/.venvs/aion/bin/python
run() { echo "########## $1 @ $(date) ##########"; shift; "$@" || echo "STEP_FAILED: $*"; }

run "TASK 3 gaia x apogee"  bash -c "$PY -u 11_embed_gaia_xp.py && $PY -u 21_probe_gaia_apogee.py"
run "TASK 2 ddpayne x desi" bash -c "$PY -u 13_embed_ddpayne.py && $PY -u 23_probe_ddpayne.py"
run "TASK 10 redshift post" bash -c "$PY -u 50_redshift_posterior.py --variant base --n 2000"
run "TASK 6 low-data"       bash -c "$PY -u 25_lowdata.py --variant base"
run "TASK 11 super-res"     bash -c "$PY -u 09_xmatch_gaia_desi.py && $PY -u 51_spectral_superres.py --variant base --n 300"
run "TABLES"                bash -c "$PY -u 60_make_tables.py && $PY -u 61_make_figures.py"
echo "MAIN_GPU_DONE"
