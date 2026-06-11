#!/usr/bin/env bash
# sync_from_phoenix.sh — stage the redshifty SpectrumFM port from phoenix onto the Mac.
#
# Stages:
#   code      rsync the redshifty package source -> src-redshifty/ (originals on phoenix untouched)
#   links     wire harness symlinks (~/.venvs/redshifty, lensing-repos/redshifty) + print the /raid step
#   priority  frozen V1 tokenizer + ignition checkpoints + manifests + ref metrics + xcheck spectra (~5 GB)
#   medium    the ~10 sv3-bright pixel dirs chosen by make_tier1_submanifest.py (~9-27 GB)  [needs $TIER1_MANIFEST]
#   mix       the full 4-way mix data tree (~760 GiB) for the Tier-2 ignition capstone
#
# All DESI data + checkpoints land under _raid/ mirroring phoenix's absolute layout, and are
# reached at /raid via a one-time `sudo ln -s _raid /raid` (see `links`). That lets the
# unmodified manifest_mix.jsonl + spec + exp_run.py harness resolve paths verbatim on the Mac.
set -uo pipefail

RSYNC="${RSYNC:-/opt/homebrew/bin/rsync}"
[ -x "$RSYNC" ] || RSYNC="$(command -v rsync)"
FLAGS=(-avh --partial --partial-dir=.rsync-partial --info=progress2 -e ssh)
REMOTE=phoenix
SRC=/raid/benson/git/agentic-lensing

AS="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$AS/../../.." && pwd)"
RAID_LOCAL="$AS/_raid"
DATA_REMOTE=/raid/benson/data/desi_dr1_medium
DATA_LOCAL="$RAID_LOCAL/benson/data/desi_dr1_medium"
CKPT_REMOTE="$DATA_REMOTE/checkpoints"
CKPT_LOCAL="$DATA_LOCAL/checkpoints"

rs() { echo "[rsync] $*"; "$RSYNC" "${FLAGS[@]}" "$@"; }

sync_code() {
  echo "[code] redshifty source -> $AS/src-redshifty/"
  mkdir -p "$AS/src-redshifty"
  rs --exclude '.git' --exclude '__pycache__' --exclude '*.egg-info' \
     --exclude 'checkpoints' --exclude 'data' --exclude '*.pdf' --exclude 'notebooks' \
     "$REMOTE:$SRC/lensing-repos/redshifty/" "$AS/src-redshifty/"
}

setup_links() {
  echo "[links] harness symlinks so the unmodified exp_run.py resolves redshifty on the Mac"
  mkdir -p "$REPO_ROOT/lensing-repos" "$HOME/.venvs"
  ln -sfn "$AS/src-redshifty" "$REPO_ROOT/lensing-repos/redshifty"
  ln -sfn "$AS/.venv"          "$HOME/.venvs/redshifty"
  echo "  lensing-repos/redshifty -> $AS/src-redshifty"
  echo "  ~/.venvs/redshifty      -> $AS/.venv"
  echo
  if [ -L /raid ] && [ "$(readlink /raid)" = "$RAID_LOCAL" ]; then
    echo "[links] /raid -> $RAID_LOCAL  (OK)"
  else
    echo "[links] ACTION REQUIRED (one-time, sudo): expose the data root at /raid so the"
    echo "        unmodified manifest/spec paths resolve. In this session run:"
    echo
    echo "    ! sudo ln -s \"$RAID_LOCAL\" /raid"
    echo
    echo "        (reversible: sudo rm /raid). No-sudo fallback: rewrite manifest prefixes."
  fi
}

sync_priority() {
  echo "[priority] frozen tokenizer + ignition checkpoints + manifests (~5 GB)"
  mkdir -p "$CKPT_LOCAL/tokenizer_v1_large" \
           "$CKPT_LOCAL/checkpoints/approach_a_phase10_mix" "$AS/data/ref"
  # frozen V1 tokenizer (val_recon 1.38) — the ignition spec's --tokenizer-ckpt
  rs "$REMOTE:$CKPT_REMOTE/tokenizer_v1_large/best.pt" \
     "$REMOTE:$CKPT_REMOTE/tokenizer_v1_large/config.json" \
     "$REMOTE:$CKPT_REMOTE/tokenizer_v1_large/metrics.jsonl" \
     "$CKPT_LOCAL/tokenizer_v1_large/"
  # ignition transformer checkpoints (best + a mid-run step) for the same-ckpt fidelity xcheck
  rs "$REMOTE:$CKPT_REMOTE/checkpoints/approach_a_phase10_mix/best.pt" \
     "$REMOTE:$CKPT_REMOTE/checkpoints/approach_a_phase10_mix/step_00007500.pt" \
     "$REMOTE:$CKPT_REMOTE/checkpoints/approach_a_phase10_mix/config.json" \
     "$REMOTE:$CKPT_REMOTE/checkpoints/approach_a_phase10_mix/metrics.jsonl" \
     "$CKPT_LOCAL/checkpoints/approach_a_phase10_mix/"
  # manifests (absolute /raid paths inside resolve via the /raid symlink)
  rs "$REMOTE:$DATA_REMOTE/manifest_mix.jsonl" \
     "$REMOTE:$DATA_REMOTE/manifest.jsonl" "$DATA_LOCAL/"
  # phoenix reference metrics for the verifier
  cp -f "$CKPT_LOCAL/checkpoints/approach_a_phase10_mix/metrics.jsonl" "$AS/data/ref/metrics_mix.jsonl" 2>/dev/null || true
  echo "[priority] NOTE: xcheck spectra (a few mix pixels) are pulled by run_infer_fidelity.sh"
}

# medium: pull only the pixel dirs referenced by the Tier-1 sub-manifest (paths in $TIER1_MANIFEST).
sync_medium() {
  local mf="${TIER1_MANIFEST:-$AS/data/tier1_submanifest.jsonl}"
  [ -f "$mf" ] || { echo "[medium] missing $mf — run make_tier1_submanifest.py first"; return 2; }
  echo "[medium] pulling pixel dirs from $mf"
  # extract unique parent dirs of coadd paths (relative to desi_dr1_medium/), rsync each
  python3 - "$mf" > "$AS/data/.tier1_pixels.txt" <<'PY'
import json, sys, os
seen = set()
for line in open(sys.argv[1]):
    line = line.strip()
    if not line:
        continue
    rel = os.path.dirname(json.loads(line)["coadd"]).split("desi_dr1_medium/", 1)[-1]
    if rel not in seen:
        seen.add(rel)
        print(rel)
PY
  while read -r rel; do
    mkdir -p "$DATA_LOCAL/$rel"
    rs "$REMOTE:$DATA_REMOTE/$rel/" "$DATA_LOCAL/$rel/"
  done < "$AS/data/.tier1_pixels.txt"
}

sync_mix() {
  echo "[mix] full 4-way mix (~760 GiB) — resumable"
  for tree in spectro/redux/fuji/healpix/sv3/bright spectro/redux/fuji/healpix/sv3/dark \
              spectro/redux/iron/healpix/main/bright spectro/redux/iron/healpix/main/dark; do
    mkdir -p "$DATA_LOCAL/$tree"
    rs "$REMOTE:$DATA_REMOTE/$tree/" "$DATA_LOCAL/$tree/"
  done
}

for stage in "${@:-all}"; do
  case "$stage" in
    code)     sync_code ;;
    links)    setup_links ;;
    priority) sync_priority ;;
    medium)   sync_medium ;;
    mix)      sync_mix ;;
    all)      sync_code; setup_links; sync_priority ;;
    *) echo "usage: $0 [code|links|priority|medium|mix|all]..."; exit 2 ;;
  esac
done
echo "[sync] done: ${*:-all}"
