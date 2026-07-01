#!/usr/bin/env python3
"""Grade strong-lens candidates at HSC-SSP PDR3 resolution (the tier-2 analog of run_euclid).

`grade_hsc(obj)` points the lean grader at HSC pixels via the fetch_hsc_cutout tool and the
contaminant-aware v2 rubric, returning the same dict the escalate grader consumes. HSC (0.168",
~0.6" seeing) is one rung coarser than Euclid (0.1") but covers far more equatorial sky, so it
widens the tier-2 escalation set well beyond the tiny Euclid Q1 overlap.

Standalone smoke (needs HSC_USER/HSC_PASSWORD):
  python lensjudge/eval/run_hsc.py --ra 0.078839 --dec 0.271578
"""
from __future__ import annotations

import argparse
import asyncio
import time
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

try:  # the open-weight backend doesn't need the Claude Agent SDK (offline/Perlmutter deploys omit it)
    from claude_agent_sdk import ClaudeAgentOptions  # noqa: E402
except ModuleNotFoundError:
    ClaudeAgentOptions = None

from lensjudge import config  # noqa: E402
from lensjudge.common import parse, llm_client  # noqa: E402
from lensjudge.common.schemas import ImageGrade  # noqa: E402
from lensjudge.imaging.grader_lean import _collect  # noqa: E402
from lensjudge.tools import server  # noqa: E402

# the contaminant-aware v2 rubric (failure-mode aware) is what tier-2 mimic adjudication wants
RUBRIC_V2 = (config.HERE / "prompts" / "rubric_imaging_v2.md").read_text()

HSC_NOTE = """

# IMPORTANT — these are HSC-SSP PDR3 cutouts, not DESI grz
Call **fetch_hsc_cutout(ra=..., dec=...)** (NOT fetch_cutout) to see the images.
- Resolution is 0.168"/px (~0.6" seeing), ~8x sharper than DESI's ~1.3" ground seeing: tangential
  arcs and Einstein rings blurred into the lens-galaxy light at ground resolution are resolved here
  (one rung coarser than Euclid 0.1"). Judge the morphology you actually see.
- Bands: grizy. Color views: R=i, G=r, B=g — the old red lens galaxy is red/orange, a lensed
  background source is blue. The 'lum' (sharp i-band) and 'lum_sub' (lens-light-subtracted) views
  are where thin arcs/rings show best.
- Apply the SAME A/B/C/D scale and the v2 rubric (rule out LRG+companion / ring / spiral / blend /
  merger via colour + radial geometry). There are no CNN scores here.
"""
HSC_SYS = RUBRIC_V2 + HSC_NOTE


def _user_message(obj: dict) -> str:
    return (f"Grade this strong-lens candidate at HSC resolution. "
            f"ra={float(obj['ra']):.6f}, dec={float(obj['dec']):.6f}"
            + (f" (name={obj.get('name')!r})" if obj.get("name") else "") + ".\n"
            f"Call fetch_hsc_cutout with that ra/dec, inspect the views, then respond with ONLY "
            f"the ImageGrade JSON object.")


async def _grade_hsc_openai(obj: dict, model: str | None = None) -> dict:
    """Open-weight tier-2 (LENSJUDGE_BACKEND=openai): drive the SAME fetch_hsc_cutout tool through
    the OpenAI tool-calling loop (tools/openai_tools), no Claude SDK. Returns the same dict shape
    grade_hsc/grader_escalate consume."""
    from lensjudge.tools import openai_tools
    model_id = model or config.MODELS["grader"]
    t0 = time.time()
    try:
        res = await llm_client.run_tool_loop(
            system=HSC_SYS, user_content=[{"type": "text", "text": _user_message(obj)}],
            tools=openai_tools.tool_schemas(["fetch_hsc_cutout"]),
            execute_tool=openai_tools.execute_tool, model=model_id, max_turns=config.MAX_TURNS)
    except Exception as e:
        return {**obj, "error": f"{type(e).__name__}: {e}", "p_lens": np.nan,
                "agent_grade": None, "cost_usd": 0.0}
    g = parse.parse_model(res.text, ImageGrade)
    return {**obj, "p_lens": (g.p_lens if g else np.nan),
            "agent_grade": (g.grade if g else None),
            "confidence": (g.confidence if g else np.nan),
            "contaminant": (g.contaminant if g else None),
            "cost_usd": res.cost_usd, "wall_s": round(time.time() - t0, 1),
            "rationale": (g.rationale[:300] if g else res.text[:200]),
            "backend": "openai", "model_id": model_id, "tool_calls": res.tool_calls}


async def grade_hsc(obj: dict, model: str | None = None) -> dict:
    if llm_client.is_open():
        return await _grade_hsc_openai(obj, model)
    mcp_servers, allowed = server.build(["fetch_hsc_cutout"])
    opts = ClaudeAgentOptions(
        model=model or config.MODELS["grader"],
        system_prompt=HSC_SYS,
        mcp_servers=mcp_servers, allowed_tools=allowed,
        permission_mode="bypassPermissions",
        max_turns=config.MAX_TURNS, max_budget_usd=config.MAX_BUDGET_USD,
        setting_sources=None,
    )
    t0 = time.time()
    try:
        raw, cost, turns, _ = await _collect(_user_message(obj), opts)
    except Exception as e:
        return {**obj, "error": f"{type(e).__name__}: {e}", "p_lens": np.nan,
                "agent_grade": None, "cost_usd": 0.0}
    g = parse.parse_model(raw, ImageGrade)
    return {**obj, "p_lens": (g.p_lens if g else np.nan),
            "agent_grade": (g.grade if g else None),
            "confidence": (g.confidence if g else np.nan),
            "contaminant": (g.contaminant if g else None),
            "cost_usd": cost, "wall_s": round(time.time() - t0, 1),
            "rationale": (g.rationale[:300] if g else raw[:200])}


async def _bounded(coro_fns, concurrency):
    sem = asyncio.Semaphore(concurrency)

    async def run(fn):
        async with sem:
            return await fn()
    return await asyncio.gather(*[run(f) for f in coro_fns])


async def run_manifest(manifest: str, out: str | None, model: str | None, concurrency: int):
    """Batch-grade a CSV of ra,dec[,name] at HSC resolution (the tier-2 known-lens gate runner).

    Offline-friendly: with LENSJUDGE_HSC_CACHE pointed at a warm cache, each position renders from
    staged FITS credential-free (see common/hsc_fetch). Reports recovery = fraction graded A/B."""
    import pandas as pd
    from lensjudge.common import llm_client
    df = pd.read_csv(manifest)
    cols = {c.lower(): c for c in df.columns}
    if "ra" not in cols or "dec" not in cols:
        raise SystemExit(f"manifest needs ra,dec columns; got {list(df.columns)}")
    rows = df.to_dict("records")

    def obj_of(r):
        o = {"ra": float(r[cols["ra"]]), "dec": float(r[cols["dec"]])}
        if "name" in cols:
            o["name"] = str(r[cols["name"]])
        for k in ("source", "sugohi_grade", "desi_grade", "grade"):  # carry labels through for scoring
            if k in cols:
                o[k] = r[cols[k]]
        return o

    print(f"[hsc-batch] {len(rows)} objects | backend={llm_client.get_backend()} | "
          f"model={model or config.MODELS['grader']}")
    recs = await _bounded([lambda o=obj_of(r): grade_hsc(o, model) for r in rows], concurrency)
    out_df = pd.DataFrame(recs)
    outp = out or str(config.OUT / "hsc_batch_preds.parquet")
    out_df.to_parquet(outp, index=False)
    graded = out_df[out_df.agent_grade.notna()]
    ab = graded.agent_grade.isin(["A", "B"]).mean() if len(graded) else float("nan")
    print(f"[hsc-batch] saved {outp} | parsed {len(graded)}/{len(out_df)} | "
          f"grades {graded.agent_grade.value_counts().to_dict() if len(graded) else {}}")
    print(f"[hsc-batch] recovery(A/B)={ab:.0%} | mean p_lens={graded.p_lens.mean():.3f} | "
          f"total ${out_df.cost_usd.sum():.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ra", type=float)
    ap.add_argument("--dec", type=float)
    ap.add_argument("--name", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--manifest", default=None, help="CSV ra,dec[,name] -> batch HSC grade")
    ap.add_argument("--out", default=None, help="batch: output parquet path")
    ap.add_argument("--concurrency", type=int, default=4)
    args = ap.parse_args()
    if args.manifest:
        asyncio.run(run_manifest(args.manifest, args.out, args.model, args.concurrency))
        return
    if args.ra is None or args.dec is None:
        ap.error("need --ra and --dec (single object), or --manifest (batch)")
    obj = {"ra": args.ra, "dec": args.dec, "name": args.name}
    res = asyncio.run(grade_hsc(obj, args.model))
    print({k: res.get(k) for k in ("name", "ra", "dec", "agent_grade", "p_lens",
                                   "confidence", "contaminant", "cost_usd", "error")})
    print("rationale:", res.get("rationale"))


if __name__ == "__main__":
    main()
