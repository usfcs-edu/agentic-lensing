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
LRD="$RAID_LOCAL/benson/data/desi_dr1_medium"
# optional spec name (default the 10k ignition spec; e.g. redshifty_approach_a_phase10_mix_mps_20k)
SPEC_NAME="${1:-redshifty_approach_a_phase10_mix_mps}"
SPEC="$REPO_ROOT/experiments/specs/${SPEC_NAME}.yaml"
[ -f "$SPEC" ] || { echo "[tier2] spec not found: $SPEC"; exit 2; }

./sync_from_phoenix.sh links >/dev/null 2>&1

# --- resolve data paths: prefer the verbatim /raid symlink; else a no-sudo local spec ---
if [ -L /raid ] && [ "$(readlink /raid)" = "$RAID_LOCAL" ]; then
  echo "[tier2] /raid symlink present — running the verbatim spec"
  test -f /raid/benson/data/desi_dr1_medium/manifest_mix.jsonl \
    || { echo "[tier2] manifest_mix missing — run ./sync_from_phoenix.sh mix first"; exit 3; }
else
  echo "[tier2] no /raid symlink — building a no-sudo local-path spec (harness still unmodified)"
  test -f "$LRD/manifest_mix.jsonl" \
    || { echo "[tier2] $LRD/manifest_mix.jsonl missing — run ./sync_from_phoenix.sh mix first"; exit 3; }
  # rewrite the manifest's absolute /raid coadd/redrock paths to the Mac-local data root
  sed "s#/raid/benson/data/desi_dr1_medium#$LRD#g" "$LRD/manifest_mix.jsonl" > "$LRD/manifest_mix_local.jsonl"
  # derive a local spec from the committed one (only the absolute paths change)
  SPEC_LOCAL="$AS/data/${SPEC_NAME}_local.yaml"
  mkdir -p "$AS/data"
  sed -e "s#/raid/benson/data/desi_dr1_medium/manifest_mix.jsonl#$LRD/manifest_mix_local.jsonl#g" \
      -e "s#/raid/benson/data/desi_dr1_medium#$LRD#g" "$SPEC" > "$SPEC_LOCAL"
  SPEC="$SPEC_LOCAL"
fi

echo "[tier2] dry-run preflight (resolve command + venv):"
.venv/bin/python "$REPO_ROOT/tools/spectrumfm/exp_run.py" "$SPEC" --dry-run || exit 4

echo "[tier2] launching ignition via the harness — the ~15-30h capstone run"
.venv/bin/python "$REPO_ROOT/tools/spectrumfm/exp_run.py" "$SPEC" --runs-dir "$AS/data/runs"
