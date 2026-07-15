#!/usr/bin/env bash
# Preview the docs in a browser over SSH.
#   docs/serve.sh            serve the last build on a free localhost port
#   docs/serve.sh --build    rebuild first, then serve
# The chosen port is printed; VS Code Remote will offer to forward it.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HTML="$ROOT/docs/_build/html"

# Prefer the docs venv (has sphinx); fall back to python3 on PATH.
if [[ -x "$ROOT/.venv-docs/bin/python" ]]; then
  PY="$ROOT/.venv-docs/bin/python"
else
  PY="${PYTHON:-python3}"
fi

if [[ "${1:-}" == "-b" || "${1:-}" == "--build" ]]; then
  echo "Building docs with $PY ..."
  "$PY" -m sphinx -b html "$ROOT/docs" "$HTML"
fi

if [[ ! -f "$HTML/index.html" ]]; then
  echo "No build found at $HTML — run: $0 --build" >&2
  exit 1
fi

# Pick a free port ourselves and print it (so it's always visible, even when
# output is captured); bind to localhost so it isn't exposed to other users on a
# shared login node. Override with PORT=NNNN if you want a fixed one.
PORT="${PORT:-$("$PY" -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1]); s.close()')}"
echo "Serving $HTML"
echo ">>> open  http://127.0.0.1:${PORT}/  (Ctrl-C to stop; VS Code will offer to forward the port)"
exec "$PY" -m http.server -d "$HTML" -b 127.0.0.1 "$PORT"
