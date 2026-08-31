# shared helpers for the playwright-cli QA scripts (source me)
set -euo pipefail
PORT=${PORT:-8765}
BASE="http://127.0.0.1:${PORT}"
QA_DIR=${QA_DIR:-$PWD/.qa}
SCRATCH=${SCRATCH:-/tmp}
S=${S:-lm}                                   # playwright-cli session name
PW() { playwright-cli -s="$S" "$@"; }
PWR() { playwright-cli -s="$S" --raw "$@"; }  # raw result value only
EVAL() { PWR eval "$1" | python3 -c 'import sys,json
v=sys.stdin.read().strip()
try: print(json.loads(v))
except Exception: print(v)'; }
export PATH="$HOME/.venvs/lensmark/bin:$HOME/.nvm/versions/node/v22.22.3/bin:$PATH"
CAMPAIGN="$QA_DIR/campaign-$S"
mkdir -p "$SCRATCH"

start_server() {   # start_server [engine]
  local engine=${1:-fixture}
  rm -rf "$CAMPAIGN"; mkdir -p "$QA_DIR"; cp -R examples/nine "$CAMPAIGN"
  rm -f "$CAMPAIGN"/*.annot.png "$CAMPAIGN"/*.log.jsonl "$CAMPAIGN"/lensmark.manifest.json
  LENSMARK_ENGINE=$engine LENSMARK_FIXTURE_DELAY=0.05 python -m lensmark.cli serve "$CAMPAIGN" --port "$PORT" --no-open \
      > "$QA_DIR/server.log" 2>&1 &
  SERVER_PID=$!
  for i in $(seq 1 40); do curl -sf "$BASE/api/health" >/dev/null && break; sleep 0.25; done
  curl -sf "$BASE/api/health" >/dev/null || { echo "server did not start"; cat "$QA_DIR/server.log"; exit 1; }
  echo "server up (pid $SERVER_PID) campaign=$CAMPAIGN"
}
stop_server() { PW close >/dev/null 2>&1 || true; kill "${SERVER_PID:-0}" >/dev/null 2>&1 || true; }
trap stop_server EXIT

open_app() { PW open "$BASE/" >/dev/null; PW resize 1920 1080 >/dev/null; sleep 1; }
# drag in IMAGE pixel coordinates (u,v fractions) using the overlay's screen rect
drag_uv() {   # drag_uv u0 v0 u1 v1
  local rect; rect=$(EVAL "JSON.stringify(document.querySelector('#overlay').getBoundingClientRect())")
  local x0 y0 x1 y1
  read -r x0 y0 x1 y1 < <(python3 - "$rect" "$1" "$2" "$3" "$4" <<'PY'
import json,sys
r=json.loads(sys.argv[1]); u0,v0,u1,v1=map(float,sys.argv[2:6])
print(r['x']+u0*r['width'], r['y']+v0*r['height'], r['x']+u1*r['width'], r['y']+v1*r['height'])
PY
)
  PW mousemove "$x0" "$y0" >/dev/null; PW mousedown >/dev/null; PW mousemove "$x1" "$y1" >/dev/null; PW mouseup >/dev/null
}
click_uv() {  # click_uv u v
  local rect; rect=$(EVAL "JSON.stringify(document.querySelector('#overlay').getBoundingClientRect())")
  local x y
  read -r x y < <(python3 - "$rect" "$1" "$2" <<'PY'
import json,sys
r=json.loads(sys.argv[1]); u,v=map(float,sys.argv[2:4])
print(r['x']+u*r['width'], r['y']+v*r['height'])
PY
)
  PW mousemove "$x" "$y" >/dev/null; PW mousedown >/dev/null; PW mouseup >/dev/null
}
assert_eq() { if [ "$1" != "$2" ]; then echo "ASSERT FAILED: $3: got '$1' expected '$2'"; exit 1; else echo "ok: $3 = $1"; fi; }
assert_ge() { if [ "$1" -lt "$2" ]; then echo "ASSERT FAILED: $3: got $1 < $2"; exit 1; else echo "ok: $3 = $1"; fi; }
snap() { PW screenshot --filename="$SCRATCH/$1.png" >/dev/null && echo "screenshot $SCRATCH/$1.png"; }
console_clean() { local c n; c=$(PW console 2>/dev/null || true); n=$(echo "$c" | sed -n 's/.*Errors: \([0-9]*\).*/\1/p' | head -1); n=${n:-0}; if [ "$n" != "0" ]; then echo "CONSOLE ERRORS ($n):"; echo "$c" | tail -20; exit 1; else echo "ok: console clean"; fi; }
