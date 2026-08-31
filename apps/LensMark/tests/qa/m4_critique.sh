#!/usr/bin/env bash
# M4: critique a fixture proposal: accept / reject / verdict / human-added item -> critique file + eval row.
cd "$(dirname "$0")/../.." && source tests/qa/_lib.sh
start_server fixture; open_app
EVAL "window.__lensmark.load('deck-02')" >/dev/null; sleep 1
EVAL "document.querySelector('[data-testid=propose]').click()" >/dev/null
for i in $(seq 1 60); do log=$(EVAL "document.querySelector('[data-testid=propose-log]').textContent"); echo "$log" | grep -qE "done|error" && break; sleep 1; done
ghost=$(EVAL "JSON.stringify([...document.querySelectorAll('#ghost .annot')].map(g=>g.dataset.id))"); echo "ghost ids: $ghost"
first=$(EVAL "document.querySelector('#ghost .annot').dataset.id"); second=$(EVAL "document.querySelectorAll('#ghost .annot')[1].dataset.id")
EVAL "window.__lensmark.selectItem('$first')" >/dev/null; PW press 1 >/dev/null; PW press Enter >/dev/null
EVAL "window.__lensmark.selectItem('$second')" >/dev/null; PW press 6 >/dev/null; PW press x >/dev/null
PW press a >/dev/null; drag_uv 0.20 0.20 0.30 0.30      # human-added arrow during review
EVAL "document.querySelector('[data-testid=submit-critique]').click()" >/dev/null; sleep 2
ls "$CAMPAIGN"/critiques/ && python3 - "$CAMPAIGN" "$first" "$second" <<'PY'
import json,sys,glob
c=json.load(open(sorted(glob.glob(sys.argv[1]+"/critiques/deck-02.*.json"))[-1]))
v={i["item_id"]:i["verdict"] for i in c["items"]}
assert v.get(sys.argv[2])=="correct" and v.get(sys.argv[3])=="spurious", v
f=json.load(open(sys.argv[1]+"/deck-02.lensmark.json")); st={i["id"]:i["status"] for i in f["items"]}
assert st[sys.argv[2]]=="accepted" and st[sys.argv[3]]=="rejected", st
added=[i for i in f["items"] if i["created_by"]["kind"]=="human" and (i.get("review") or {}).get("verdict")=="missed_by_model"]
assert added, "human-added item lacks missed_by_model"
print("ok: critique", c["counts"])
PY
python -m lensmark.cli eval "$CAMPAIGN" --by model,effort | tail -3
snap m4_review; console_clean
echo "M4 QA PASSED"
