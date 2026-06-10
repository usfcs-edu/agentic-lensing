#!/bin/bash
# GZ10 GPU tasks: 4 (morphology), 7/8 (retrieval). Builds image arrays from the
# currently-cached cutouts (06b), then embeds/probes/retrieves on that subset.
# 6 healthy GPUs (GPU1 excluded).
cd /home2/benson/git/agentic-lensing/reproductions/aion-1
export HF_HOME=/home2/benson/.cache/huggingface
PY=/home2/benson/.venvs/aion/bin/python
run() { echo "########## $1 @ $(date) ##########"; shift; "$@" || echo "STEP_FAILED: $*"; }

run "GZ10 finalize-from-cache" bash -c "$PY -u 06b_finalize_gz10.py --subsample 8000"
run "TASK 4/7/8 gz10" bash -c "$PY -u 12_embed_gz10.py && $PY -u 22_probe_gz10.py && $PY -u 40_retrieve_gz10.py"
run "TABLES"          bash -c "$PY -u 60_make_tables.py && $PY -u 61_make_figures.py"
echo "GZ10_GPU_DONE"
