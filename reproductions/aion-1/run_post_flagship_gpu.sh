#!/bin/bash
# All remaining GPU work, run AFTER the flagship frees the GPUs.
# Uses 6 healthy GPUs (GPU1 excluded as a thermal throttler).
# Each task is isolated so one failure does not abort the rest.
cd /home2/benson/git/agentic-lensing/reproductions/aion-1
export HF_HOME=/home2/benson/.cache/huggingface
PY=/home2/benson/.venvs/aion/bin/python
run() { echo "########## $1 @ $(date) ##########"; shift; "$@" || echo "STEP_FAILED: $*"; }

run "TASK 3 gaia x apogee"  bash -c "$PY -u 11_embed_gaia_xp.py && $PY -u 21_probe_gaia_apogee.py"
run "TASK 2 ddpayne x desi" bash -c "$PY -u 13_embed_ddpayne.py && $PY -u 23_probe_ddpayne.py"
run "TASK 4/7/8 gz10"       bash -c "$PY -u 12_embed_gz10.py && $PY -u 22_probe_gz10.py && $PY -u 40_retrieve_gz10.py"
run "TASK 10 redshift post" bash -c "$PY -u 50_redshift_posterior.py --variant base --n 2000"
run "TASK 6 low-data"       bash -c "$PY -u 25_lowdata.py --variant base"
run "TASK 11 super-res"     bash -c "$PY -u 09_xmatch_gaia_desi.py && $PY -u 51_spectral_superres.py --variant base --n 300"
run "TABLES"                bash -c "$PY -u 60_make_tables.py"
echo "POST_FLAGSHIP_DONE"
