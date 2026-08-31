#!/usr/bin/env bash
# M6: voice/NL patch through the text path (mic-free): transcript -> ops -> ghost -> apply.
cd "$(dirname "$0")/../.." && source tests/qa/_lib.sh
start_server fixture; open_app
EVAL "window.__lensmark.load('deck-01')" >/dev/null; sleep 1
n0=$(EVAL "document.querySelectorAll('#items .annot.mask.galaxy').length")
EVAL "(()=>{const e=document.querySelector('[data-testid=voice-text]');e.value='put a dashed circle around the galaxy at upper left';e.dispatchEvent(new Event('input',{bubbles:true}));return 1})()" >/dev/null
EVAL "document.querySelector('[data-testid=voice-send]').click()" >/dev/null; sleep 3
assert_ge "$(EVAL "document.querySelectorAll('[data-testid=apply-op]').length")" "1" "ops shown"
EVAL "document.querySelector('[data-testid=apply-all]').click()" >/dev/null; sleep 2
assert_eq "$(EVAL "document.querySelectorAll('#items .annot.mask.galaxy').length")" "$((n0+1))" "galaxy mask added by voice"
tail -1 "$CAMPAIGN/deck-01.lensmark.log.jsonl" | grep -q '"source": "voice"' && echo "ok: log source voice"
snap m6_voice; console_clean
echo "M6 QA PASSED"
