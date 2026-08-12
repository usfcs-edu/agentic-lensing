#!/usr/bin/env python3
"""Stream the per-agent JSONL transcripts and produce the paper's accounting tables.

Outputs (to assets/analysis/out/):
  agent_stats.csv   one row per agent transcript: session, workflow, role, killed,
                    n_records, n_assistant_turns, tool-use counts (Read/Bash/Write/
                    WebSearch/WebFetch/other), image reads, token usage
                    (input/output/cache_read/cache_creation), first/last timestamp,
                    wall seconds, model.
  role_summary.md   aggregation by pipeline role (markdown table for the paper).

Roles are assigned per workflow ID (identified from workflow system prompts).
"""
import csv
import json
import os
import sys
from collections import defaultdict
from datetime import datetime

ASSETS = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
OUT = os.path.join(ASSETS, "analysis", "out")
os.makedirs(OUT, exist_ok=True)

# Workflow -> role map (from workflow system prompts; see paper §5).
ROLE = {
    # S5 vision inspection
    "wf_159c2096-f41": "inspect", "wf_3efd2cf6-6e1": "inspect",
    "wf_7cd3c942-6db": "inspect", "wf_3eee50ad-0ed": "inspect",
    "wf_4d006e8d-029": "inspect", "wf_f0b28c52-3d6": "inspect",
    "wf_fee387bc-c12": "inspect", "wf_129a5e5a-bc8": "inspect",
    "wf_92353a9b-629": "inspect", "wf_c7d35643-f8a": "inspect",
    "wf_d0f3f8e1-2ff": "inspect", "wf_87b923e5-6e3": "inspect",
    "wf_c02479fb-7d0": "inspect",
    # S6 adversarial verification
    "wf_a6eddb51-7d3": "verify", "wf_f1accf8e-02a": "verify",
    "wf_fa3b9002-d04": "verify", "wf_1b524235-0f4": "verify",
    "wf_a529d4ad-5be": "verify", "wf_cec30479-369": "verify",
    "wf_45d78719-238": "verify", "wf_f61affa8-59b": "verify",
    "wf_864202f5-82c": "verify", "wf_70b18f44-c8f": "verify",
    "wf_8d4b9f61-52d": "verify", "wf_f3dccd64-237": "verify",
    # literature crossmatch (per object)
    "wf_5bf73fb9-15c": "literature", "wf_5ca4f5b2-500": "literature",
    "wf_888688f5-223": "literature", "wf_6c945d1c-873": "literature",
    "wf_b2eb649d-628": "literature", "wf_56fc856e-503": "literature",
    "wf_812fd3c7-2f2": "literature", "wf_037410c8-5da": "literature",
    # catalogue acquisition
    "wf_d82eefc9-f3a": "catalogue", "wf_90e80b1e-2bc": "catalogue",
    # notes / annotation
    "wf_6364ef1e-685": "notes", "wf_5482beff-eb4": "notes",
    "wf_3c1a2b04-a22": "notes",
}

SESSIONS = {
    "2b7e6f8b-04b2-4b85-8d10-8a2bac92eddf": "design-2",
    "727a50d7-0697-4223-8237-70d14459ed43": "design-1",
    "dab6143e-76d1-4b4b-bcbd-e201c60a630c": "discovery",
}

IMAGE_EXT = (".png", ".jpg", ".jpeg")


def iter_agent_files():
    for sess in SESSIONS:
        root = os.path.join(ASSETS, sess)
        for dirpath, _dirnames, filenames in os.walk(root):
            for fn in filenames:
                if fn.startswith("agent-") and fn.endswith(".jsonl"):
                    yield sess, os.path.join(dirpath, fn)


def parse_ts(s):
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None


def main():
    rows = []
    for sess, path in iter_agent_files():
        wf = "none"
        parts = path.split(os.sep)
        if "workflows" in parts:
            wf = parts[parts.index("workflows") + 1]
        role = ROLE.get(wf, "design" if sess != "dab6143e-76d1-4b4b-bcbd-e201c60a630c" else "plan")
        st = {
            "session": SESSIONS[sess], "workflow": wf, "role": role,
            "file": os.path.relpath(path, ASSETS), "killed": 0,
            "n_records": 0, "n_assistant": 0,
            "read": 0, "read_img": 0, "bash": 0, "write": 0,
            "websearch": 0, "webfetch": 0, "other_tools": 0,
            "in_tok": 0, "out_tok": 0, "cache_read": 0, "cache_create": 0,
            "model": "", "t0": None, "t1": None,
        }
        first_text_seen = False
        with open(path, "r", errors="replace") as f:
            for line in f:
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                st["n_records"] += 1
                ts = parse_ts(d.get("timestamp", "") or "")
                if ts:
                    if st["t0"] is None or ts < st["t0"]:
                        st["t0"] = ts
                    if st["t1"] is None or ts > st["t1"]:
                        st["t1"] = ts
                m = d.get("message") or {}
                if d.get("type") == "user" and not first_text_seen:
                    c = m.get("content")
                    txt = c if isinstance(c, str) else ""
                    if isinstance(c, list):
                        for p in c:
                            if isinstance(p, dict) and p.get("type") == "text":
                                txt = p.get("text", "")
                                break
                    if txt:
                        first_text_seen = True
                        if txt.startswith("You've hit your session limit"):
                            st["killed"] = 1
                if m.get("role") != "assistant":
                    continue
                st["n_assistant"] += 1
                st["model"] = m.get("model") or st["model"]
                u = m.get("usage") or {}
                st["in_tok"] += u.get("input_tokens", 0) or 0
                st["out_tok"] += u.get("output_tokens", 0) or 0
                st["cache_read"] += u.get("cache_read_input_tokens", 0) or 0
                st["cache_create"] += u.get("cache_creation_input_tokens", 0) or 0
                c = m.get("content")
                if not isinstance(c, list):
                    continue
                for p in c:
                    if not (isinstance(p, dict) and p.get("type") == "tool_use"):
                        continue
                    name = p.get("name", "")
                    inp = p.get("input") or {}
                    if name == "Read":
                        st["read"] += 1
                        fp = str(inp.get("file_path", "")).lower()
                        if fp.endswith(IMAGE_EXT):
                            st["read_img"] += 1
                    elif name == "Bash":
                        st["bash"] += 1
                    elif name == "Write":
                        st["write"] += 1
                    elif name == "WebSearch":
                        st["websearch"] += 1
                    elif name == "WebFetch":
                        st["webfetch"] += 1
                    else:
                        st["other_tools"] += 1
        st["wall_s"] = (
            (st["t1"] - st["t0"]).total_seconds() if st["t0"] and st["t1"] else 0.0
        )
        st["t0"] = st["t0"].isoformat() if st["t0"] else ""
        st["t1"] = st["t1"].isoformat() if st["t1"] else ""
        rows.append(st)
        if len(rows) % 100 == 0:
            print(f"  {len(rows)} agents...", file=sys.stderr)

    cols = ["session", "workflow", "role", "file", "killed", "n_records",
            "n_assistant", "read", "read_img", "bash", "write", "websearch",
            "webfetch", "other_tools", "in_tok", "out_tok", "cache_read",
            "cache_create", "wall_s", "t0", "t1", "model"]
    with open(os.path.join(OUT, "agent_stats.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in cols})

    # Role aggregation
    agg = defaultdict(lambda: defaultdict(float))
    for r in rows:
        a = agg[r["role"]]
        a["agents"] += 1
        a["killed"] += r["killed"]
        for k in ("read", "read_img", "bash", "write", "websearch", "webfetch",
                  "in_tok", "out_tok", "cache_read", "cache_create"):
            a[k] += r[k]
    order = ["plan", "inspect", "verify", "literature", "catalogue", "notes", "design"]
    lines = ["| Role | Agents | Killed | Image reads | Bash | Web | Output Mtok | Billable Mtok* |",
             "|---|---|---|---|---|---|---|---|"]
    tot = defaultdict(float)
    for role in order:
        if role not in agg:
            continue
        a = agg[role]
        billable = a["in_tok"] + a["out_tok"] + a["cache_create"]
        lines.append(
            f"| {role} | {int(a['agents'])} | {int(a['killed'])} | "
            f"{int(a['read_img'])} | {int(a['bash'])} | "
            f"{int(a['websearch'] + a['webfetch'])} | "
            f"{a['out_tok'] / 1e6:.2f} | {billable / 1e6:.1f} |")
        for k, v in a.items():
            tot[k] += v
    billable = tot["in_tok"] + tot["out_tok"] + tot["cache_create"]
    lines.append(
        f"| **total** | {int(tot['agents'])} | {int(tot['killed'])} | "
        f"{int(tot['read_img'])} | {int(tot['bash'])} | "
        f"{int(tot['websearch'] + tot['webfetch'])} | "
        f"{tot['out_tok'] / 1e6:.2f} | {billable / 1e6:.1f} |")
    lines.append("")
    lines.append("*billable = input + output + cache-creation tokens "
                 "(cache reads excluded).")
    with open(os.path.join(OUT, "role_summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
