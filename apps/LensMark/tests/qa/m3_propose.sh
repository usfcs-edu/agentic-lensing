#!/usr/bin/env bash
# M3: Claude proposal with model x effort through the fixture engine (and once through the fake CLI via SdkEngine).
cd "$(dirname "$0")/../.." && source tests/qa/_lib.sh
ENGINE=${ENGINE:-fixture}
if [ "$ENGINE" = "sdk" ]; then export LENSMARK_CLAUDE_BIN="$PWD/tests/fixtures/fake_claude" LENSMARK_FAKE_ARGV="$QA_DIR/argv.txt"; rm -f "$LENSMARK_FAKE_ARGV"; fi
start_server "$ENGINE"; open_app
EVAL "window.__lensmark.load('deck-02')" >/dev/null; sleep 1
EVAL "(()=>{const e=document.querySelector('[data-testid=model-select]');e.value='opus';e.dispatchEvent(new Event('change',{bubbles:true}));return e.value})()"
EVAL "(()=>{const e=document.querySelector('[data-testid=effort-select]');e.value='xhigh';e.dispatchEvent(new Event('change',{bubbles:true}));return e.value})()"
EVAL "document.querySelector('[data-testid=propose]').click()" >/dev/null
for i in $(seq 1 60); do
  log=$(EVAL "document.querySelector('[data-testid=propose-log]').textContent"); echo "$log" | grep -qE "done|error" && break; sleep 1
done
echo "$log" | tail -c 400
echo "$log" | grep -q "done" || { echo "ASSERT FAILED: proposal did not finish"; exit 1; }
assert_ge "$(EVAL "document.querySelectorAll('#ghost .annot').length")" "3" "ghost items"
PW requests | grep -E "POST.*api/propose/deck-02" | head -2
ls "$CAMPAIGN"/proposals/deck-02.*.json
python3 - "$CAMPAIGN" <<'PY'
import json,sys,glob
p=sorted(glob.glob(sys.argv[1]+"/proposals/deck-02.*.json"))[-1]; d=json.load(open(p))
print("proposal:", d.get("model"), d.get("effort"), "items", len(d.get("items",[])))
assert d.get("model")=="claude-opus-5" and d.get("effort")=="xhigh", (d.get("model"), d.get("effort"))
f=json.load(open(sys.argv[1]+"/deck-02.lensmark.json"))
assert any(i["status"]=="proposed" for i in f["items"]), "no proposed items merged"
assert f["provenance"]["proposal_runs"][-1]["model"]=="claude-opus-5"
print("ok: proposal merged with", sum(1 for i in f["items"] if i["status"]=="proposed"), "proposed items")
PY
if [ "$ENGINE" = "sdk" ]; then grep -q -- "--effort xhigh" "$LENSMARK_FAKE_ARGV" && grep -q -- "--model claude-opus-5" "$LENSMARK_FAKE_ARGV" && grep -q -- "--setting-sources" "$LENSMARK_FAKE_ARGV" && echo "ok: fake CLI argv has model/effort/setting-sources"; fi
snap m3_ghost; console_clean
echo "M3 QA PASSED ($ENGINE)"
