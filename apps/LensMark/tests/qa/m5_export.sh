#!/usr/bin/env bash
# M5: exports (coco / ds9 / masks / fewshot) via CLI + UI buttons.
cd "$(dirname "$0")/../.." && source tests/qa/_lib.sh
start_server fixture
for f in coco ds9 masks fewshot; do python -m lensmark.cli export "$CAMPAIGN" $f --k 3 | tail -1; done
python3 - "$CAMPAIGN" <<'PY'
import json,sys,glob,os
root=sys.argv[1]
c=json.load(open(root+"/exports/coco/instances.json")); assert c["images"] and c["annotations"] and c["categories"]; print("ok: coco", len(c["annotations"]), "annotations")
reg=open(sorted(glob.glob(root+"/exports/ds9/*.reg"))[0]).read(); assert reg.startswith("# Region file format"); print("ok: ds9")
assert glob.glob(root+"/exports/masks/*.mask.png"), "no masks"; print("ok: masks")
m=json.load(open(root+"/exports/fewshot/manifest.json")); assert m["k"]==len(m["examples"])==3 and open(root+"/exports/fewshot/prompt.sha256").read().strip()==m["prompt_sha256"]; print("ok: fewshot", [e["id"] for e in m["examples"]])
PY
open_app; EVAL "window.__lensmark.load('deck-01')" >/dev/null; sleep 1
EVAL "document.querySelector('[data-testid=export-coco]').click()" >/dev/null; sleep 2
PW requests | grep -E "POST.*api/export/coco" | tail -1 | grep -q "\[200\]" && echo "ok: UI export 200"
snap m5_export; console_clean
echo "M5 QA PASSED"
