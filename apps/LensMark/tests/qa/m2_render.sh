#!/usr/bin/env bash
# M2: deterministic annotated PNG: render CLI, --check semantics, "Rendered" toggle in the UI.
cd "$(dirname "$0")/../.." && source tests/qa/_lib.sh
start_server fixture
python -m lensmark.cli render "$CAMPAIGN" && ls "$CAMPAIGN"/*.annot.png | wc -l | xargs -I{} echo "rendered {} files"
python -m lensmark.cli render "$CAMPAIGN" --check && echo "ok: --check clean"
python3 - "$CAMPAIGN/deck-01.lensmark.json" <<'PY'
import json,sys; p=sys.argv[1]; d=json.load(open(p)); d["system"]["description"]+=" (edited by QA)"; json.dump(d,open(p,"w"),indent=2)
PY
if python -m lensmark.cli render "$CAMPAIGN" --check; then echo "ASSERT FAILED: --check should be stale"; exit 1; else echo "ok: --check detects stale"; fi
python -m lensmark.cli render "$CAMPAIGN" --id deck-01 && python -m lensmark.cli render "$CAMPAIGN" --check && echo "ok: re-render clears stale"
open_app; EVAL "window.__lensmark.load('deck-01')" >/dev/null; sleep 1
snap m2_preview
PW click "[data-testid=render-toggle]" >/dev/null 2>&1 || EVAL "document.querySelector('[data-testid=render-toggle]').click()" >/dev/null
sleep 1.5
src=$(EVAL "document.querySelector('#base').getAttribute('src')")
echo "$src" | grep -q "/annot" && echo "ok: base shows annot ($src)" || { echo "ASSERT FAILED: render toggle src=$src"; exit 1; }
assert_eq "$(EVAL "document.querySelector('#base').naturalWidth")" "403" "annot natural width"
snap m2_rendered; console_clean
echo "M2 QA PASSED"
