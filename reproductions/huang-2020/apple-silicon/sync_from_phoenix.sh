#!/usr/bin/env bash
# sync_from_phoenix.sh — pull base survey data + reference artifacts from phoenix
# into the canonical shared dirs (huang-2020/data, hsu-2025/data) and the port's
# ref/papers dirs. The Apple Silicon port reaches the bulk inputs via symlinks.
#
# Usage:
#   ./sync_from_phoenix.sh priority   # small, needed-first: training cutouts, zcat, PDFs, refs (~22 GB)
#   ./sync_from_phoenix.sh bulk       # dr7_sweep (476 GB) + cutouts_fits_dr7 (28 GB)
#   ./sync_from_phoenix.sh all        # priority then bulk  (default)
set -euo pipefail

RSYNC="${RSYNC:-/opt/homebrew/bin/rsync}"
[ -x "$RSYNC" ] || RSYNC="$(command -v rsync)"
# -P = --partial --progress; --partial-dir keeps interrupted files for resume.
# (--append-verify conflicts with --partial-dir on rsync 3.4.x, so we omit it.)
FLAGS=(-avh --partial --partial-dir=.rsync-partial --info=progress2 -e ssh)
REMOTE=phoenix
SRC=/raid/benson/git/agentic-lensing

ROOT=/Users/benson/sync-git/sync-lens/agentic-lensing/reproductions
H20="$ROOT/huang-2020"
HSU="$ROOT/hsu-2025"
AS="$H20/apple-silicon"

mkdir -p "$H20/data" "$HSU/data" "$AS/data/ref" "$AS/data/papers"

echo "[rsync] $($RSYNC --version | head -1)  ($RSYNC)"

sync_priority() {
  echo "[priority] training cutouts (DR9 + DR7-train)"
  "$RSYNC" "${FLAGS[@]}" "$REMOTE:$SRC/reproductions/huang-2020/data/cutouts_fits_dr9/"       "$H20/data/cutouts_fits_dr9/"
  "$RSYNC" "${FLAGS[@]}" "$REMOTE:$SRC/reproductions/huang-2020/data/cutouts_fits_dr7_train/" "$H20/data/cutouts_fits_dr7_train/"
  echo "[priority] zcat for negatives (21 GB)"
  "$RSYNC" "${FLAGS[@]}" "$REMOTE:$SRC/reproductions/hsu-2025/data/zall-pix-iron.fits"        "$HSU/data/"
  echo "[priority] paper PDFs"
  "$RSYNC" "${FLAGS[@]}" "$REMOTE:$SRC/papers/Huang_2020_DECaLS_lenses.pdf" \
                          "$REMOTE:$SRC/papers/Huang_2021_DESI_legacy_lenses.pdf" "$AS/data/papers/"
  echo "[priority] reference checkpoints + full inference scores"
  "$RSYNC" "${FLAGS[@]}" "$REMOTE:$SRC/reproductions/huang-2020/data/checkpoint_best.pt" \
                          "$REMOTE:$SRC/reproductions/huang-2020/data/checkpoint_best_dr7.pt" "$AS/data/ref/"
  "$RSYNC" "${FLAGS[@]}" "$REMOTE:$SRC/reproductions/huang-2020/data/inference_scores_dr9trained.parquet" \
                          "$REMOTE:$SRC/reproductions/huang-2020/data/inference_scores_dr7trained.parquet" "$AS/data/ref/"
}

sync_bulk() {
  # Only the DR7 sweeps are a pure input (read by 10/15). The kept inference
  # cutouts (cutouts_fits_dr7/) are PRODUCED locally by 11b in this from-scratch
  # run and feed the viewer, so they are NOT pulled from phoenix.
  echo "[bulk] DR7 sweeps (476 GB)"
  "$RSYNC" "${FLAGS[@]}" "$REMOTE:$SRC/reproductions/huang-2020/data/dr7_sweep/" "$H20/data/dr7_sweep/"
}

case "${1:-all}" in
  priority) sync_priority ;;
  bulk)     sync_bulk ;;
  all)      sync_priority; sync_bulk ;;
  *) echo "usage: $0 [priority|bulk|all]"; exit 2 ;;
esac
echo "[rsync] done: ${1:-all}"
