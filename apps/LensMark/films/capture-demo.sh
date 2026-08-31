#!/usr/bin/env bash
# Real terminal capture for the film's a6-files and a9-eval scenes. Every byte is a real run.
# Leaves the campaign exactly as found (the stale demo edits deck-02 and restores it).
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.venvs/lensmark/bin:$PATH"
C=films/campaign-film
echo '## FILES'
ls -1 "$C"/deck-02* "$C"/deck-03* | sed "s|$C/||"
echo '## STALE'
cp "$C/deck-02.lensmark.json" /tmp/deck-02.film-bak.json
python - "$C" <<'PY'
import json, sys
p = f"{sys.argv[1]}/deck-02.lensmark.json"
d = json.load(open(p))
d["system"]["description"] += " (edited behind the app's back)"
json.dump(d, open(p, "w"), indent=2)
PY
python -m lensmark.cli render "$C" --check || true
cp /tmp/deck-02.film-bak.json "$C/deck-02.lensmark.json"
python -m lensmark.cli render "$C" --check
echo '## EVAL'
python -m lensmark.cli eval "$C" --by model,effort > /dev/null
python films/eval_view.py "$C"
