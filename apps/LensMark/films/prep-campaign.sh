#!/usr/bin/env bash
# Prepare films/campaign-film for the LensMark demo film. Re-running resets the campaign.
#
# DISK-STATE LEDGER (what each scene assumes / how to restore for a retake):
#   a2-tour   deck-01 & deck-03 have NO .lensmark.json (blank rows). Restore: rm their json/annot/log.
#   a3-arrow  deck-01 blank at take start; the act saves an arrow+mask. Retake: rm deck-01.lensmark.json
#             deck-01.annot.png deck-01.lensmark.log.jsonl, then re-record a3,a4,a5.
#   a4-ring   deck-01 has a3's save. After a good a4 take: snapshot deck-01.lensmark.json into
#             .film-snapshots/ for a10/a11 retakes.
#   a7/a8     deck-03 blank -> a7 fires a real sonnet/low propose; a8 polls the merged ghosts.
#             Retake: re-run `lensmark propose films/campaign-film --id deck-03 --model sonnet
#             --effort low` (re-propose prunes stale ghosts) or rm deck-03.* to reshoot from blank.
#   a10-voice adds one galaxy mask per take to deck-01. Retake: restore the a4 snapshot first.
#   a11       idempotent (exports overwrite).
set -euo pipefail
cd "$(dirname "$0")/.."                      # apps/LensMark
export PATH="$HOME/.venvs/lensmark/bin:$PATH"
C=films/campaign-film

rm -rf "$C"; cp -R examples/nine "$C"
rm -f "$C"/*.annot.png "$C"/*.lensmark.log.jsonl "$C"/lensmark.manifest.json
rm -f "$C"/deck-01.lensmark.json "$C"/deck-03.lensmark.json     # the by-hand and by-model targets
mkdir -p "$C"/.film-snapshots

python -m lensmark.cli render "$C"                               # pre-render the 7 annotated images

echo "== real proposals for the eval table (opus/xhigh ~75s, sonnet/low ~12s)"
python -m lensmark.cli propose "$C" --id deck-05 --model opus   --effort xhigh --budget 0.50
python -m lensmark.cli propose "$C" --id deck-06 --model sonnet --effort low   --budget 0.25
python -m lensmark.cli render "$C" --id deck-05
python -m lensmark.cli render "$C" --id deck-06   # proposals change the JSON -> re-pin the renders

python - "$C" <<'PY'
import json, sys
p = f"{sys.argv[1]}/lensmark.config.json"; d = json.load(open(p))
d["default_model"] = "sonnet"; d["default_effort"] = "low"   # fast real patch calls in the voice scene
json.dump(d, open(p, "w"), indent=2)
PY

echo "== gates"
python -m lensmark.cli render "$C" --check
test ! -f "$C"/deck-01.lensmark.json && test ! -f "$C"/deck-03.lensmark.json
ls "$C"/proposals/deck-05.*.json "$C"/proposals/deck-06.*.json >/dev/null
echo "CAMPAIGN READY: $C  (next: submit critiques for deck-05/deck-06 via the Review UI)"
