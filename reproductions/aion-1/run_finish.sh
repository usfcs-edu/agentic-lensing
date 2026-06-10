#!/bin/bash
# Resume driver after the _watch.sh stall (xlarge phot_image probe OOM blocked
# the IMAGE_CONFIGS_DONE marker, freezing the main + gz10 batches).
#  1) finish flagship: phot_image_spec embed (safe per-variant batch) + re-probe
#     phot_image (fixes xlarge OOM via token-aware batch) + probe phot_image_spec
#  2) main batch (tasks 3,2,10,6,11)  3) gz10 batch (tasks 4,7,8)  4) tables/figs
cd /home2/benson/git/agentic-lensing/reproductions/aion-1
export HF_HOME=/home2/benson/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home2/benson/.venvs/aion/bin/python
G=0,2,3,4,5,6   # exclude GPU1 (thermal throttler)
run(){ echo "##### $1 @ $(date) #####"; shift; "$@" || echo "STEP_FAILED: $*"; }

run "embed phot_image_spec base"   $PY -u 10_embed_provabgs.py --config phot_image_spec --variant base   --gpus $G --batch 80
run "embed phot_image_spec large"  $PY -u 10_embed_provabgs.py --config phot_image_spec --variant large  --gpus $G --batch 56
run "embed phot_image_spec xlarge" $PY -u 10_embed_provabgs.py --config phot_image_spec --variant xlarge --gpus $G --batch 12
run "probe phot_image"      $PY -u 20_probe_provabgs.py --config phot_image
run "probe phot_image_spec" $PY -u 20_probe_provabgs.py --config phot_image_spec
echo "IMAGE_CONFIGS_DONE @ $(date)"

run "MAIN batch (3,2,10,6,11)" bash run_main_gpu.sh
run "GZ10 batch (4,7,8)"       bash run_gz10_gpu.sh
run "tables+figures" bash -c "$PY -u 60_make_tables.py && $PY -u 61_make_figures.py"
echo "FINISH_ALL_DONE @ $(date)"
