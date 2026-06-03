#!/usr/bin/env bash
# run_infer_fidelity.sh — Layer (a): same-checkpoint MPS-vs-CUDA forward-pass fidelity.
#
# Runs the xcheck on the Mac (MPS, fp32) and on phoenix (CUDA, fp32) over the SAME 4
# byte-identical DESI pixels + the frozen tokenizer + ignition transformer (step 9500),
# then scores the comparison. No training; minutes. This isolates MPS-vs-CUDA numerical
# drift of the ConvNeXt tokenizer + transformer + cross-attention readout.
set -uo pipefail
cd "$(dirname "$0")"
export PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONUNBUFFERED=1
PY=.venv/bin/python

REMOTE=phoenix
RDATA=/raid/benson/data/desi_dr1_medium
RREPO=/raid/benson/git/agentic-lensing/lensing-repos/redshifty

PIXELS="spectro/redux/iron/healpix/main/bright/101/10191,\
spectro/redux/fuji/healpix/sv3/bright/104/10408,\
spectro/redux/fuji/healpix/sv3/dark/119/11936,\
spectro/redux/iron/healpix/main/dark/100/10048"

# ensure the 4 xcheck pixels are present locally (idempotent)
DL=_raid/benson/data/desi_dr1_medium
for rel in ${PIXELS//,/ }; do
  if [ -z "$(ls "$DL/$rel/"coadd-*.fits 2>/dev/null)" ]; then
    mkdir -p "$DL/$rel"
    /opt/homebrew/bin/rsync -a -e ssh "$REMOTE:$RDATA/$rel/" "$DL/$rel/"
  fi
done

echo "[a] Mac MPS fp32"
$PY xcheck_mps_inference.py --device mps --pixels "$PIXELS" --out data/xcheck_mps.json

echo "[a] phoenix CUDA fp32 (reference)"
scp -q xcheck_mps_inference.py "$REMOTE:/tmp/xcheck_redshifty.py"
ssh "$REMOTE" "/raid/benson/.venvs/redshifty/bin/python /tmp/xcheck_redshifty.py \
  --repo $RREPO --ckpt-dir $RDATA/checkpoints/checkpoints/approach_a_phase10_mix \
  --tokenizer-ckpt $RDATA/checkpoints/tokenizer_v1_large/best.pt \
  --data-root $RDATA --pixels '$PIXELS' --device cuda --out /tmp/xcheck_cuda.json"
scp -q "$REMOTE:/tmp/xcheck_cuda.json" data/xcheck_cuda.json

echo "[a] compare"
$PY compare_xcheck.py
