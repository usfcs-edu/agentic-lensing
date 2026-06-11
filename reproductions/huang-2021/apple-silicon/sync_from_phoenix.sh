#!/usr/bin/env bash
# sync_from_phoenix.sh — pull Huang-2021 reference artifacts from phoenix and wire
# up the shared-input symlinks. Tier 1 needs only the `priority` stage (~2 GB);
# `bulk` (724 GB dr8_sweep) is for the deferred Tier-2 full deployment sweep.
#
# Usage:
#   ./sync_from_phoenix.sh links      # just (re)create the data/ input symlinks
#   ./sync_from_phoenix.sh priority   # northaug ckpts + full score parquets + parent_dr8 (~2 GB)  [+links]
#   ./sync_from_phoenix.sh north      # cutouts_fits_north fallback (368 MB; else script 18 builds it)
#   ./sync_from_phoenix.sh bulk       # Tier 2: dr8_sweep 724 GB
#   ./sync_from_phoenix.sh all        # links + priority   (default)
set -uo pipefail

RSYNC="${RSYNC:-/opt/homebrew/bin/rsync}"
[ -x "$RSYNC" ] || RSYNC="$(command -v rsync)"
FLAGS=(-avh --partial --partial-dir=.rsync-partial --info=progress2 -e ssh)
REMOTE=phoenix
SRC=/raid/benson/git/agentic-lensing
RH21="$SRC/reproductions/huang-2021/data"

AS="$(cd "$(dirname "$0")" && pwd)"
D="$AS/data"
H20="../../../huang-2020/data"                  # relative to $D
H20AS="../../../huang-2020/apple-silicon/data"  # Mac-trained L18 checkpoints

mkdir -p "$D/ref"

link() {  # link <target-relative-to-$D> <name>
  local tgt="$1" name="$2"
  if [ -e "$D/$tgt" ]; then
    ln -sfn "$tgt" "$D/$name"
  else
    echo "[links] WARN target missing: $D/$tgt  (skipped $name)"
  fi
}

setup_links() {
  echo "[links] wiring shared inputs into $D"
  link "$H20/cutouts_fits_dr9"               cutouts_fits_dr9
  link "$H20/cutouts_fits_dr7_train"         cutouts_fits_dr7_train
  link "$H20/positives_huang2020.parquet"    positives_huang2020.parquet
  link "$H20/positives_all.parquet"          positives_all.parquet
  link "$H20/negatives.parquet"              negatives.parquet
  link "$H20/neuralens_catalog.csv"          neuralens_catalog.csv
  # Mac-trained L18 checkpoints (pre-northaug); fall back to canonical huang-2020/data.
  if [ -e "$D/$H20AS/checkpoint_best.pt" ]; then
    link "$H20AS/checkpoint_best.pt"     checkpoint_best.pt
    link "$H20AS/checkpoint_best_dr7.pt" checkpoint_best_dr7.pt
  else
    link "$H20/checkpoint_best.pt"       checkpoint_best.pt
    link "$H20/checkpoint_best_dr7.pt"   checkpoint_best_dr7.pt
  fi
  # huang-2020 published catalog (15 primary read); fall back handled in-script.
  if [ -e "$D/$H20AS/huang2020_published_catalog.csv" ]; then
    link "$H20AS/huang2020_published_catalog.csv" huang2020_published_catalog.csv
  elif [ -e "$D/$H20/huang2020_published_catalog.csv" ]; then
    link "$H20/huang2020_published_catalog.csv" huang2020_published_catalog.csv
  fi
  echo "[links] done"; ls -la "$D" | grep -E '\->'
}

sync_priority() {
  echo "[rsync] $($RSYNC --version | head -1)  ($RSYNC)"
  echo "[priority] phoenix northaug checkpoints (deployment weights, for the xcheck)"
  "$RSYNC" "${FLAGS[@]}" \
    "$REMOTE:$RH21/checkpoint_best_l18_northaug.pt" \
    "$REMOTE:$RH21/checkpoint_best_shielded_northaug.pt" "$D/ref/"
  echo "[priority] full DR8 two-model score parquets (xcheck + analysis refs)"
  "$RSYNC" "${FLAGS[@]}" \
    "$REMOTE:$RH21/inference_scores_l18_dr8.parquet" \
    "$REMOTE:$RH21/inference_scores_shielded_dr8.parquet" "$D/ref/"
  echo "[priority] parent_dr8.parquet (so 18 can sample north negatives)"
  "$RSYNC" "${FLAGS[@]}" "$REMOTE:$RH21/parent_dr8.parquet" "$D/"
}

sync_north() {
  echo "[north] cutouts_fits_north fallback"
  "$RSYNC" "${FLAGS[@]}" "$REMOTE:$RH21/cutouts_fits_north/" "$D/ref/cutouts_fits_north/"
}

sync_bulk() {
  echo "[bulk] DR8 sweeps (724 GB south+north)"
  "$RSYNC" "${FLAGS[@]}" "$REMOTE:$RH21/dr8_sweep/" "$D/dr8_sweep/"
}

case "${1:-all}" in
  links)    setup_links ;;
  priority) sync_priority; setup_links ;;
  north)    sync_north ;;
  bulk)     sync_bulk ;;
  all)      sync_priority; setup_links ;;
  *) echo "usage: $0 [links|priority|north|bulk|all]"; exit 2 ;;
esac
echo "[rsync] done: ${1:-all}"
