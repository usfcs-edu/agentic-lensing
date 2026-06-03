#!/usr/bin/env bash
# run_tier2.sh — Layer (c): full-mix redshift-ignition reproduction on MPS, driven
# through the UNMODIFIED tools/spectrumfm/exp_run.py harness (which also validates the
# Track-2 tooling on Apple Silicon). ~15-30 h, ONE MPS process, resumable via the
# spec's --save-every 2500. Reference (phoenix L4): val_z_acc 14.86% peak @ step 9500,
# val_loss min 190.67, val_loss_redshift drop 1.19, AR/TF ~0.73.
set -uo pipefail
cd "$(dirname "$0")"
export PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONUNBUFFERED=1
AS="$(pwd)"
REPO_ROOT="$(cd ../../.. && pwd)"
RAID_LOCAL="$AS/_raid"
SPEC="$REPO_ROOT/experiments/specs/redshifty_approach_a_phase10_mix_mps.yaml"

# --- preconditions ---
if ! { [ -L /raid ] && [ "$(readlink /raid)" = "$RAID_LOCAL" ]; }; then
  echo "[tier2] MISSING /raid symlink (one-time, sudo). In this session run:"
  echo "    ! sudo ln -s \"$RAID_LOCAL\" /raid"
  echo "    (reversible: sudo rm /raid)"
  exit 3
fi
./sync_from_phoenix.sh links >/dev/null 2>&1
test -f /raid/benson/data/desi_dr1_medium/manifest_mix.jsonl \
  || { echo "[tier2] manifest_mix missing — run ./sync_from_phoenix.sh mix first"; exit 3; }

echo "[tier2] dry-run preflight (resolve command + venv):"
.venv/bin/python "$REPO_ROOT/tools/spectrumfm/exp_run.py" "$SPEC" --dry-run || exit 4

echo "[tier2] launching ignition via the harness — the ~15-30h capstone run"
.venv/bin/python "$REPO_ROOT/tools/spectrumfm/exp_run.py" "$SPEC" --runs-dir "$AS/data/runs"
