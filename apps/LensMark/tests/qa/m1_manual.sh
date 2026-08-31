#!/usr/bin/env bash
# M1: load dir, view, draw an arrow / galaxy mask / star mask / ring, save, reload -> JSON persisted.
cd "$(dirname "$0")/../.." && source tests/qa/_lib.sh
start_server fixture; open_app
assert_eq "$(EVAL "document.querySelectorAll('[data-testid=image-row]').length")" "9" "image rows"
EVAL "window.__lensmark.load('deck-01')" >/dev/null; sleep 1
assert_eq "$(EVAL "document.querySelector('#overlay').getAttribute('viewBox')")" "0 0 403 403" "viewBox"
n0=$(EVAL "document.querySelectorAll('#items .annot.arrow').length")
PW press a >/dev/null; drag_uv 0.41 0.70 0.446 0.586
assert_eq "$(EVAL "document.querySelectorAll('#items .annot.arrow').length")" "$((n0+1))" "arrow drawn"
PW snapshot >/dev/null
EVAL "(()=>{const e=document.querySelector('[data-testid=label-input]');e.value='tight arc';e.dispatchEvent(new Event('input',{bubbles:true}));e.dispatchEvent(new Event('change',{bubbles:true}));return 1})()" >/dev/null
EVAL "(()=>{const e=document.querySelector('[data-testid=color-select]');e.value='cyan';e.dispatchEvent(new Event('change',{bubbles:true}));return 1})()" >/dev/null
PW press g >/dev/null; drag_uv 0.10 0.30 0.16 0.30
PW press s >/dev/null; drag_uv 0.80 0.15 0.83 0.15
PW press Escape >/dev/null
assert_ge "$(EVAL "document.querySelectorAll('#items .annot.mask.galaxy').length")" "1" "galaxy mask drawn"
assert_ge "$(EVAL "document.querySelectorAll('#items .annot.mask.star').length")" "1" "star mask drawn"
gd=$(EVAL "getComputedStyle(document.querySelector('#items .annot.mask.galaxy circle.mask-circle')).strokeDasharray")
sd=$(EVAL "getComputedStyle(document.querySelector('#items .annot.mask.star circle.mask-circle')).strokeDasharray")
[ "$gd" != "$sd" ] && echo "ok: dash arrays differ ($gd vs $sd)" || { echo "ASSERT FAILED: galaxy/star dasharray equal"; exit 1; }
EVAL "window.__lensmark.save()" >/dev/null; sleep 1
PW requests | grep -E "PUT.*api/ann/deck-01" | tail -1
python3 - "$CAMPAIGN/deck-01.lensmark.json" <<'PY'
import json,sys
d=json.load(open(sys.argv[1])); items=d["items"]
arrows=[i for i in items if i["type"]=="arrow" and i.get("label")=="tight arc"]
assert arrows, "arrow with label not saved: %r" % [(i["type"], i.get("label")) for i in items]
a=arrows[-1]; assert abs(a["head"][1]-0.586)<0.02 and a["color"]=="cyan", a
kinds=[i["kind"] for i in items if i["type"]=="mask_circle"]
assert "galaxy" in kinds and "star" in kinds, kinds
g=[i for i in items if i["type"]=="mask_circle" and i["kind"]=="galaxy"][-1]; assert 0.5 < g["radius_arcsec"] < 2.0, g
print("ok: JSON persisted", len(items), "items")
PY
PW reload >/dev/null; sleep 1
assert_ge "$(EVAL "document.querySelectorAll('#items .annot.arrow').length")" "$((n0+1))" "arrow present after reload"
snap m1_after_reload; console_clean
echo "M1 QA PASSED"
