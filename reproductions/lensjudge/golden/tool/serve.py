#!/usr/bin/env python3
"""serve.py — optional local server for a golden grading kit (stdlib only; no installs).

grade.html works from file:// on its own. Serving it over http adds a second copy of every
commit on disk *as it happens*: the page POSTs each event to /api/event and this server
appends it to records/events_<kit_id>_<session_id>.jsonl in the kit directory, so a lost
browser profile or a forgotten export cannot lose grades. It is deliberately tiny and
dependency-free because it ships inside the kit to the grader's machine — it must NOT import
anything from the lensjudge package.

    cd <kit dir>
    python3 serve.py --port 8765        # then open http://localhost:8765/grade.html

Endpoints
  GET  /<static>                 files in the kit directory (grade.html, items/NNN.jpg, ...)
  POST /api/event                one JSON event -> appended to records/events_<kit>_<session>.jsonl
  GET  /api/events[?grader=XH]   JSON array of every stored event (optionally one grader's)

An event is accepted only if it is a JSON object with exactly the contract keys below — the
server, like the collector, refuses anything extra so no free text can ride along.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

EVENT_KEYS = (
    "kit_id", "manifest_sha", "item_id", "presentation_index", "session_id", "grader_id",
    "score_1_4", "confidence_lmh", "seconds", "revision", "flag", "timestamp", "ua",
    "tool_version",
)
_SAFE = re.compile(r"[^A-Za-z0-9_.+-]")


def _safe(name: str) -> str:
    return _SAFE.sub("_", str(name))[:120]


class Handler(SimpleHTTPRequestHandler):
    kit_dir = os.getcwd()

    def log_message(self, fmt, *args):  # quieter: one line per request, no timestamps
        sys.stderr.write("%s %s\n" % (self.command, fmt % args))

    # ---------------------------------------------------------------- helpers
    def _json(self, code: int, obj) -> None:
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _records_dir(self) -> str:
        d = os.path.join(self.kit_dir, "records")
        os.makedirs(d, exist_ok=True)
        return d

    # ------------------------------------------------------------------- GET
    def do_GET(self):
        u = urlsplit(self.path)
        if u.path == "/api/events":
            grader = parse_qs(u.query).get("grader", [None])[0]
            out = []
            d = self._records_dir()
            for fn in sorted(os.listdir(d)):
                if not (fn.startswith("events_") and fn.endswith(".jsonl")):
                    continue
                with open(os.path.join(d, fn)) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            ev = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if grader is None or ev.get("grader_id") == grader:
                            out.append(ev)
            self._json(200, out)
            return
        if u.path == "/":
            self.path = "/grade.html"
        return super().do_GET()

    def end_headers(self):
        # never let the browser cache grade.html/items across a kit rebuild (add-repeats)
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    # ------------------------------------------------------------------ POST
    def do_POST(self):
        u = urlsplit(self.path)
        if u.path != "/api/event":
            self._json(404, {"error": "unknown endpoint"})
            return
        try:
            n = int(self.headers.get("Content-Length", "0"))
            ev = json.loads(self.rfile.read(n).decode())
        except Exception as e:  # noqa: BLE001
            self._json(400, {"error": f"bad JSON: {e}"})
            return
        if not isinstance(ev, dict) or set(ev) != set(EVENT_KEYS):
            extra = sorted(set(ev) - set(EVENT_KEYS)) if isinstance(ev, dict) else "?"
            missing = sorted(set(EVENT_KEYS) - set(ev)) if isinstance(ev, dict) else "?"
            self._json(400, {"error": "event keys must be exactly the contract keys",
                             "extra": extra, "missing": missing})
            return
        fn = f"events_{_safe(ev['kit_id'])}_{_safe(ev['session_id'])}.jsonl"
        with open(os.path.join(self._records_dir(), fn), "a") as f:
            f.write(json.dumps(ev, sort_keys=False) + "\n")
        self._json(200, {"ok": True, "file": f"records/{fn}"})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--bind", default="127.0.0.1", help="interface (default: localhost only)")
    ap.add_argument("--dir", default=os.path.dirname(os.path.abspath(__file__)),
                    help="kit directory to serve (default: where serve.py lives)")
    a = ap.parse_args(argv)
    Handler.kit_dir = os.path.abspath(a.dir)
    os.chdir(Handler.kit_dir)
    srv = ThreadingHTTPServer((a.bind, a.port), Handler)
    print(f"serving {Handler.kit_dir} at http://{a.bind}:{a.port}/grade.html  "
          f"(events -> records/; Ctrl-C to stop)", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
