#!/usr/bin/env python3
"""Record v3 CASCADE episodes for the report's v3 worked-examples appendix.

The v3 deploy path is a cascade with two kinds of step, and the appendix must show both:
  stage-1  DETERMINISTIC  — grader_direct: render views + compute aperture photometry IN PYTHON,
                            then ONE base Messages-API call (1 turn, no tools) -> grade.
  stage-2  AGENTIC (DESI) — grader_lean (v2 rubric): the multi-turn Claude Agent SDK loop
                            (ToolSearch -> fetch_cutout -> get_photometry -> grade).
  stage-3  AGENTIC (HSC)  — run_hsc.grade_hsc: another SDK loop (fetch_hsc_cutout -> grade), the
                            tier-2 high-res re-grade that flips the DESI verdict.

Captures the full ordered event stream of each stage (reusing run_appendix_examples._collect_episode
for the agentic stages), dumps JSON + readable markdown under outputs/appendix_v3/, and renders the
DESI + HSC figure strips to papers/figures/appendix_v3_*.png. Adaptive thinking ON so the reasoning
is visible (as in the v1 appendix).

  python lensjudge/eval/run_appendix_v3.py --max-usd 1.5
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd  # noqa: E402
from claude_agent_sdk import ClaudeAgentOptions  # noqa: E402

from lensjudge import config  # noqa: E402
from lensjudge.common import fetch, hsc, parse, render  # noqa: E402
from lensjudge.common.schemas import ImageGrade  # noqa: E402
from lensjudge.imaging import grader_direct, grader_lean  # noqa: E402
from lensjudge.eval import run_hsc  # noqa: E402
from lensjudge.eval.run_appendix_examples import _collect_episode  # noqa: E402
from lensjudge.tools import server  # noqa: E402
from lensjudge.tools.photometry import _aperture_colors  # noqa: E402

OUT = config.OUT / "appendix_v3"
FIG = config.HERE / "papers" / "figures"
RUBRIC_INLINE = config.HERE / "prompts" / "rubric_imaging_v2_inline.md"   # stage-1 (deterministic)
RUBRIC_V2 = config.HERE / "prompts" / "rubric_imaging_v2.md"              # stage-2 (agentic)

B1 = {"name": "DESI-029.0755-01.1297", "survey_key": "storfer", "catalog": "storfer",
      "ra": 29.075515, "dec": -1.129749, "region": "DECaLS"}


def _lookup_pmeta(name, survey):
    csv = config.SCORE_CSV.get(survey)
    if csv and Path(csv).exists():
        d = pd.read_csv(csv)
        col = next((c for c in d.columns if c.lower() in ("name",)), None)
        row = d[d[col] == name] if col else d[0:0]
        if len(row):
            for sc in ("our_p_meta", "p_meta", "probability"):
                if sc in row:
                    return float(row.iloc[0][sc])
    return None


async def stage1_deterministic(cand) -> dict:
    """grader_direct internals, captured: deterministic evidence (views + photometry) + 1 call."""
    cube = fetch.get_cube(name=cand.get("name"), ra=cand.get("ra"), dec=cand.get("dec"),
                          survey=cand.get("survey_key", "storfer"))
    if cube is None:
        return {"stage": "1_direct_deterministic", "error": "no cube"}
    views = list(grader_direct._DEFAULT_VIEWS)
    phot = _aperture_colors(cube)
    content = grader_direct._build_content(cand, views)
    client = grader_direct._get_client()
    model_id = grader_direct._resolve_model(None)
    topt = config.thinking_options()
    kw: dict = {"max_tokens": 2048}
    if topt.get("thinking"):
        kw["max_tokens"] = 8192
        kw["thinking"] = {"type": "adaptive", "display": "summarized"}
        if topt.get("effort"):
            kw["output_config"] = {"effort": topt["effort"]}
    resp = await client.messages.create(
        model=model_id, system=RUBRIC_INLINE.read_text(),
        messages=[{"role": "user", "content": content}], **kw)
    raw = "".join(b.text for b in resp.content if b.type == "text")
    think = "\n".join(getattr(b, "thinking", "") for b in resp.content if b.type == "thinking")
    g = parse.parse_model(raw, ImageGrade)
    return {"stage": "1_direct_deterministic", "model": model_id, "turns": 1,
            "candidate_text": content[0]["text"], "views": views, "photometry": phot,
            "thinking": think, "final": raw, "grade": g.model_dump() if g else None,
            "cost_usd": grader_direct._cost(model_id, resp.usage)}


async def _agentic(prompt, system_prompt, tools, label) -> dict:
    mcp_servers, allowed = server.build(list(tools))
    opts = ClaudeAgentOptions(
        model=config.MODELS["grader"], system_prompt=system_prompt,
        mcp_servers=mcp_servers, allowed_tools=allowed, permission_mode="bypassPermissions",
        max_turns=config.MAX_TURNS, max_budget_usd=config.MAX_BUDGET_USD,
        setting_sources=None, **config.thinking_options())
    ep = await _collect_episode(prompt, opts)
    g = parse.parse_model(ep["final"], ImageGrade)
    ep["stage"] = label
    ep["grade"] = g.model_dump() if g else None
    return ep


async def stage2_desi(cand):
    return await _agentic(grader_lean._user_message(cand), RUBRIC_V2.read_text(),
                          ("fetch_cutout", "get_photometry"), "2_desi_agentic")


async def stage3_hsc(obj):
    return await _agentic(run_hsc._user_message(obj), run_hsc.HSC_SYS,
                          ("fetch_hsc_cutout",), "3_hsc_agentic")


def save_desi_strip(cand, prefix):
    cube = fetch.get_cube(name=cand.get("name"), ra=cand.get("ra"), dec=cand.get("dec"),
                          survey=cand.get("survey_key", "storfer"))
    if cube is None:
        print(f"  [warn] no cube for {cand['name']}"); return
    for v, img in render.render_views(cube, views=["full", "zoom", "residual"]).items():
        img.save(FIG / f"{prefix}__{v}.png")


def save_hsc_strip(ra, dec, prefix):
    bands = hsc.load_hsc(float(ra), float(dec))
    if not bands:
        print(f"  [warn] no HSC bands for {ra},{dec}"); return
    for v, img in hsc.render_hsc_views(bands, views=["full", "lum", "zoom", "lum_sub"]).items():
        img.save(FIG / f"{prefix}__{v}.png")


def _g(ep):
    return ep.get("grade") or {}


def render_md_stage(ep) -> str:
    g = _g(ep)
    head = (f"### {ep['stage']}  —  grade {g.get('grade')} p_lens {g.get('p_lens')} "
            f"conf {g.get('confidence')} contam {g.get('contaminant')}  "
            f"(turns {ep.get('turns')}, ${ep.get('cost_usd', 0):.3f})\n")
    lines = [head]
    if ep["stage"].startswith("1"):
        lines += [f"[deterministic] candidate text:\n```\n{ep.get('candidate_text','')}\n```",
                  f"[deterministic] views rendered: {ep.get('views')}",
                  f"[deterministic] aperture photometry: ```{json.dumps(ep.get('photometry'))}```"]
        if ep.get("thinking"):
            lines += [f"thinking (summarized):\n{ep['thinking']}"]
        lines += [f"final JSON:\n```json\n{ep.get('final','')}\n```"]
        return "\n\n".join(lines) + "\n"
    for ev in ep.get("events", []):
        t = ev["type"]
        if t == "thinking":
            lines.append(f"thinking (summarized):\n{ev['text']}")
        elif t == "text":
            lines.append(f"agent text:\n{ev['text']}")
        elif t == "tool_use":
            lines.append(f"tool call: {ev['name']}\n```json\n{json.dumps(ev['input'])}\n```")
        elif t == "tool_result":
            body = "\n".join(i.get("text", i.get("note", "")) for i in ev.get("content", []))
            lines.append(f"tool result{' (ERROR)' if ev['is_error'] else ''}:\n```\n{body}\n```")
    lines.append(f"final JSON:\n```json\n{ep.get('final','')}\n```")
    return "\n\n".join(lines) + "\n"


async def build_b1():
    cand = dict(B1); cand["p_meta"] = _lookup_pmeta(cand["name"], "storfer")
    print(f"[B.1] {cand['name']} (p_meta={cand['p_meta']})")
    save_desi_strip(cand, "appendix_v3_desi")
    save_hsc_strip(cand["ra"], cand["dec"], "appendix_v3_hsc")
    s1 = await stage1_deterministic(cand)
    print(f"  stage-1 deterministic -> {_g(s1).get('grade')} p_lens={_g(s1).get('p_lens')}")
    s2 = await stage2_desi(cand)
    print(f"  stage-2 agentic DESI  -> {_g(s2).get('grade')} p_lens={_g(s2).get('p_lens')} turns={s2.get('turns')}")
    s3 = await stage3_hsc({"ra": cand["ra"], "dec": cand["dec"], "name": cand["name"]})
    print(f"  stage-3 agentic HSC   -> {_g(s3).get('grade')} p_lens={_g(s3).get('p_lens')} turns={s3.get('turns')}")
    (OUT / "B1.json").write_text(json.dumps({"cand": cand, "stages": [s1, s2, s3]}, indent=2, default=str))
    (OUT / "B1.md").write_text(f"# B.1 cascade: {cand['name']}\n\n"
                               + "\n".join(render_md_stage(s) for s in [s1, s2, s3]))


async def build_b2(n_try):
    m = pd.read_csv(config.OUT / "lensbench_evidence.csv")
    mim = m[m["source"] == "mimic_neg"].head(n_try)
    chosen = None
    for _, r in mim.iterrows():
        cand = {"name": r["name"], "survey_key": "claudenet", "catalog": "claudenet",
                "ra": r.get("ra"), "dec": r.get("dec"), "region": r.get("region"),
                "p_meta": r.get("p_meta")}
        s1 = await stage1_deterministic(cand)
        g = _g(s1)
        print(f"  [B.2 try] {cand['name'][:24]:24s} -> {g.get('grade')} "
              f"contam={g.get('contaminant')} conf={g.get('confidence')}")
        if g.get("grade") == "D" and g.get("contaminant") and (g.get("confidence") or 0) >= 0.6:
            chosen = (cand, s1)
            if g.get("contaminant") == "lrg_companion":
                break   # prefer the dominant class; otherwise keep first qualifying
    if chosen is None:
        print("  [B.2] no clean confident-mimic rejection found"); return
    cand, s1 = chosen
    print(f"[B.2] chosen {cand['name']} -> D/{_g(s1).get('contaminant')}/{_g(s1).get('confidence')}")
    save_desi_strip(cand, "appendix_v3_mimic")
    (OUT / "B2.json").write_text(json.dumps({"cand": cand, "stages": [s1]}, indent=2, default=str))
    (OUT / "B2.md").write_text(f"# B.2 deterministic mimic reject: {cand['name']}\n\n"
                               + render_md_stage(s1))


async def main_async(args):
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("LENSJUDGE_THINKING", "adaptive")   # show reasoning, as in the v1 appendix
    if not args.skip_b1:
        await build_b1()
    if not args.skip_b2:
        await build_b2(args.b2_tries)
    print(f"\nsaved episodes -> {OUT}  ; figure strips -> {FIG}/appendix_v3_*")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-usd", type=float, default=1.5)
    ap.add_argument("--b2-tries", type=int, default=5)
    ap.add_argument("--skip-b1", action="store_true")
    ap.add_argument("--skip-b2", action="store_true")
    args = ap.parse_args()
    # rough gate: B.1 ~3 stages*$0.1 + B.2 up to b2_tries*$0.015
    est = 0.35 + args.b2_tries * 0.02
    print(f"[appendix-v3] est ~${est:.2f} (cap ${args.max_usd})")
    if est > args.max_usd:
        raise SystemExit(f"ABORT: est ${est:.2f} > ${args.max_usd}")
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
