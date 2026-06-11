#!/usr/bin/env python3
"""Re-record full grading episodes for the report's worked-examples appendix.

The benchmark traces (hooks.Trace) log tool calls only; the appendix needs the
agent's interleaved reasoning too. This runner grades a small shortlist with the
EXACT lean Table-1 configuration (rubric, sonnet, fetch_cutout+get_photometry,
6 turns, $0.50 cap) but captures the entire SDK message stream — text blocks,
tool_use blocks, tool_result blocks — and dumps each episode as JSON + readable
markdown under outputs/appendix/, plus the rendered views the agent saw as PNGs.

  python lensjudge/eval/run_appendix_examples.py --tier all
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

# bootstrap: put reproductions/ on the path so `import lensjudge` works when run directly
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd  # noqa: E402
from claude_agent_sdk import (AssistantMessage, ClaudeAgentOptions,  # noqa: E402
                              ResultMessage, SystemMessage, TextBlock, ThinkingBlock,
                              ToolResultBlock, ToolUseBlock, UserMessage, query)

from lensjudge import config  # noqa: E402
from lensjudge.common import fetch, hooks, parse, render  # noqa: E402
from lensjudge.common.schemas import ImageGrade  # noqa: E402
from lensjudge.imaging import grader_lean  # noqa: E402
from lensjudge.tools import server  # noqa: E402

OUT = config.OUT / "appendix"
FIG = OUT / "figs"
TOOLS = ("fetch_cutout", "get_photometry")  # the Table-1 lean pair

# The three tiers of the appendix: positive / neutral / negative examples.
# Names come from outputs/lensbench_manifest.csv (so every example is a
# LensBench-VI member with a benchmark-run grade to cross-reference).
SHORTLIST = {
    "A": ["DESI-091.7214-58.9787",   # the report's 'obvious arc' (bench: B, p_lens 0.75)
          "DESI-008.6210-45.8601",
          "DESI-0016.9290-76.1465",
          "DESI-0037.4073-66.4542"],
    "C": ["DESI-014.6921-22.5859",   # bench: B, photometry call in benchmark trace
          "DESI-030.9742+12.4683",
          "DESI-072.4328-20.0637",   # bench: C, photometry call in benchmark trace
          "DESI-061.3703-23.8666"],
    "D": ["DESI-082.1798-60.5502",   # bench: D, photometry call in benchmark trace
          "DESI-125.3337+36.3640",
          "DESI-055.5346-21.3181",
          "DESI-124.2632+17.1177",
          "DESI-342.9236-57.4053"],
}


def _block_content(content) -> list:
    """Serialize a ToolResultBlock's content; image payloads become placeholders."""
    if content is None:
        return []
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    out = []
    for item in content:
        if isinstance(item, dict):
            if item.get("type") == "image":
                out.append({"type": "image",
                            "note": "[image: PNG returned to agent — not logged]"})
            else:
                out.append({k: v for k, v in item.items() if k != "data"})
        else:
            out.append({"type": "text", "text": str(item)})
    return out


async def _collect_episode(prompt: str, opts: ClaudeAgentOptions) -> dict:
    """Run one query(); record every event in order (the full episode)."""
    events, texts = [], []
    ep = {"model": None, "cost_usd": 0.0, "turns": 0, "events": events, "final": ""}
    async for msg in query(prompt=prompt, options=opts):
        if isinstance(msg, SystemMessage):
            if msg.subtype == "init":
                ep["model"] = msg.data.get("model")
        elif isinstance(msg, AssistantMessage):
            for b in msg.content:
                if isinstance(b, ThinkingBlock):
                    events.append({"type": "thinking", "text": b.thinking})
                elif isinstance(b, TextBlock):
                    events.append({"type": "text", "text": b.text})
                    texts.append(b.text)
                elif isinstance(b, ToolUseBlock):
                    events.append({"type": "tool_use", "id": b.id,
                                   "name": b.name, "input": b.input})
        elif isinstance(msg, UserMessage):
            content = msg.content if isinstance(msg.content, list) else []
            for b in content:
                if isinstance(b, ToolResultBlock):
                    events.append({"type": "tool_result", "tool_use_id": b.tool_use_id,
                                   "is_error": bool(b.is_error),
                                   "content": _block_content(b.content)})
        elif isinstance(msg, ResultMessage):
            ep["cost_usd"] = msg.total_cost_usd or 0.0
            ep["turns"] = msg.num_turns or 0
            if msg.result:
                texts.append(msg.result)
    ep["final"] = texts[-1] if texts else ""
    return ep


async def run_episode(cand: dict, model: str | None = None) -> dict:
    """Grade one candidate with the lean Table-1 config, full stream capture."""
    mcp_servers, allowed = server.build(list(TOOLS))
    tr = hooks.Trace(OUT / f"{cand['name']}.trace.jsonl")
    opts = ClaudeAgentOptions(
        model=model or config.MODELS["grader"],
        system_prompt=grader_lean._RUBRIC,
        mcp_servers=mcp_servers,
        allowed_tools=allowed,
        permission_mode="bypassPermissions",
        max_turns=config.MAX_TURNS,
        max_budget_usd=config.MAX_BUDGET_USD,
        setting_sources=None,
        hooks=tr.hooks(),
        **config.thinking_options(),
    )
    prompt = grader_lean._user_message(cand)
    t0 = time.time()
    ep = await _collect_episode(prompt, opts)
    ep.update(name=cand["name"], tier=cand.get("tier"), prompt=prompt,
              grade_truth=cand.get("grade"), p_meta=cand.get("p_meta"),
              bench_grade=cand.get("bench_grade"), bench_p_lens=cand.get("bench_p_lens"),
              wall_s=round(time.time() - t0, 1),
              thinking=os.environ.get("LENSJUDGE_THINKING", "off"),
              thinking_chars=sum(len(e["text"]) for e in ep["events"]
                                 if e["type"] == "thinking"))
    g = parse.parse_model(ep["final"], ImageGrade)
    ep["parse_ok"] = g is not None
    ep["grade"] = g.model_dump() if g is not None else None
    return ep


def save_views(cand: dict):
    """Re-render the exact views the agent saw (same cube, same render params)."""
    cube = fetch.get_cube(name=cand["name"], ra=cand.get("ra"), dec=cand.get("dec"),
                          survey=cand.get("survey_key", "storfer"))
    if cube is None:
        print(f"  [warn] no cube for {cand['name']} — figure strip unavailable")
        return
    safe = cand["name"].replace(".", "_")
    for view, img in render.render_views(cube, views=["full", "zoom", "residual"]).items():
        img.save(FIG / f"{safe}__{view}.png")


def render_markdown(ep: dict) -> str:
    """Readable episode dump — the editing source of truth for the appendix."""
    g = ep.get("grade") or {}
    lines = [f"# Episode: {ep['name']} (tier {ep['tier']})",
             "",
             f"- truth grade: {ep.get('grade_truth')}  |  p_meta: {ep.get('p_meta')}",
             f"- benchmark lean run: grade {ep.get('bench_grade')}, "
             f"p_lens {ep.get('bench_p_lens')}",
             f"- this episode: grade {g.get('grade')}, p_lens {g.get('p_lens')}, "
             f"confidence {g.get('confidence')}, contaminant {g.get('contaminant')}, "
             f"escalate {g.get('escalate_to_human')}",
             f"- model: {ep.get('model')}  |  turns: {ep['turns']}  |  "
             f"cost: ${ep['cost_usd']:.3f}  |  wall: {ep['wall_s']}s  |  "
             f"thinking: {ep.get('thinking', 'off')} "
             f"({ep.get('thinking_chars', 0)} chars)",
             "",
             "## Task message", "", "```", ep["prompt"], "```", ""]
    for ev in ep["events"]:
        if ev["type"] == "thinking":
            lines += ["## Thinking (summarized)", "", ev["text"], ""]
        elif ev["type"] == "text":
            lines += ["## Agent", "", ev["text"], ""]
        elif ev["type"] == "tool_use":
            lines += [f"## Tool call: {ev['name']}", "", "```json",
                      json.dumps(ev["input"], indent=2), "```", ""]
        elif ev["type"] == "tool_result":
            tag = " (ERROR)" if ev["is_error"] else ""
            lines += [f"## Tool result{tag}", ""]
            for item in ev["content"]:
                if item.get("type") == "image":
                    lines += [item["note"], ""]
                else:
                    lines += ["```", item.get("text", json.dumps(item)), "```", ""]
    lines += ["## Final JSON", "", "```json", ep["final"], "```", ""]
    return "\n".join(lines)


def load_shortlist(tiers: list[str]) -> list[dict]:
    m = pd.read_csv(config.OUT / "lensbench_manifest.csv")
    p = pd.read_parquet(config.OUT / "preds_lensbench_lean.parquet")
    p = p.set_index("name")[["grade_pred", "p_lens"]]
    cands = []
    for tier in tiers:
        for name in SHORTLIST[tier]:
            row = m[m["name"] == name]
            if row.empty:
                print(f"  [warn] {name} not in manifest — skipped")
                continue
            cand = row.iloc[0].to_dict()
            cand["grade"] = cand.pop("grade_truth")
            cand["catalog"] = cand.get("survey_key", "storfer")
            cand["tier"] = tier
            if name in p.index:
                cand["bench_grade"] = p.loc[name, "grade_pred"]
                cand["bench_p_lens"] = float(p.loc[name, "p_lens"])
            cands.append(cand)
    return cands


async def main_async(args):
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    tiers = ["A", "C", "D"] if args.tier == "all" else [args.tier]
    cands = load_shortlist(tiers)
    if args.only:
        cands = [c for c in cands if c["name"] == args.only]
    sfx = args.suffix
    todo = [c for c in cands
            if args.force or not (OUT / f"{c['name']}{sfx}.json").exists()]
    print(f"[appendix] {len(todo)} episodes to record ({len(cands) - len(todo)} cached)")

    for cand in cands:           # views are cheap and idempotent — always ensure
        save_views(cand)

    sem = asyncio.Semaphore(args.concurrency)

    async def one(cand):
        async with sem:
            ep = await run_episode(cand, model=args.model)
        (OUT / f"{cand['name']}{sfx}.json").write_text(json.dumps(ep, indent=2, default=str))
        (OUT / f"{cand['name']}{sfx}.md").write_text(render_markdown(ep))
        g = ep.get("grade") or {}
        print(f"  {cand['name']:26s} tier={cand['tier']} truth={cand['grade']} "
              f"-> {g.get('grade')} p_lens={g.get('p_lens')} turns={ep['turns']} "
              f"${ep['cost_usd']:.3f}")
        return ep

    await asyncio.gather(*(one(c) for c in todo))

    # selection table over everything on disk
    print(f"\n{'name':26s} {'tier':4s} {'truth':5s} {'bench':5s} "
          f"{'grade':5s} {'p_lens':6s} {'esc':3s} {'turns':5s} {'cost':6s}")
    for c in cands:
        f = OUT / f"{c['name']}{sfx}.json"
        if not f.exists():
            continue
        ep = json.loads(f.read_text())
        g = ep.get("grade") or {}
        print(f"{c['name']:26s} {c['tier']:4s} {str(c['grade']):5s} "
              f"{str(c.get('bench_grade')):5s} {str(g.get('grade')):5s} "
              f"{str(g.get('p_lens')):6s} {str(g.get('escalate_to_human'))[:3]:3s} "
              f"{ep['turns']:<5d} ${ep['cost_usd']:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", choices=("A", "C", "D", "all"), default="all")
    ap.add_argument("--concurrency", type=int, default=3)
    ap.add_argument("--force", action="store_true", help="re-run even if episode exists")
    ap.add_argument("--only", default=None, help="record just this candidate name")
    ap.add_argument("--suffix", default="", help="output filename suffix (episode variants)")
    ap.add_argument("--model", default=None, help="model override (default: config grader)")
    ap.add_argument("--thinking", choices=("off", "adaptive"), default=None)
    ap.add_argument("--effort", choices=("low", "medium", "high", "xhigh", "max"), default=None)
    args = ap.parse_args()
    if args.thinking:
        os.environ["LENSJUDGE_THINKING"] = args.thinking
    if args.effort:
        os.environ["LENSJUDGE_EFFORT"] = args.effort
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
