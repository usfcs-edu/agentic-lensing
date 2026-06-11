"""§9.1 multi-agent factor-decomposition grader (the ablation arm).

The onboarding-plan §9.1 design: instead of one integrated judgment (lean) or a panel
of full-grade judges (Option C), specialist agents each assess ONE factor of the
cutout and a GradeArbitrator fuses their factor reports into A/B/C/D.

  MorphologyAgent  — curvature / counter-images / arc geometry (criteria 3,4,5)
  ColorAgent       — lens-red / source-blue contrast + low surface brightness (1,2)
  ContaminantAgent — false-positive morphologies; the veto channel
  (CrossmatchAgent — prior-catalog overlap; OFF for the consensus eval: a graded
   candidate trivially matches its own published catalog, which would leak the label)
  GradeArbitrator  — fuses the factor reports under the 2-of-N A-rule + contaminant veto

Orchestration is programmatic (Python fans out the specialist query() calls, then calls
the arbitrator) rather than LLM-dispatched AgentDefinition subagents — more reliable and
controllable, and avoids the SDK caveats (subagents can't nest, JSON-in-text fragility).
Exposes grade_candidate(...) -> GradeResult, drop-in with grader_lean / judges.
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import Optional

from claude_agent_sdk import ClaudeAgentOptions

from lensjudge import config
from lensjudge.common import hooks, parse
from lensjudge.common.schemas import ImageGrade
from lensjudge.imaging.grader_lean import GradeResult, _collect, _user_message
from lensjudge.tools import server

_FOV = "~26\" grz cutout, 0.26\"/px (z=R,r=G,g=B); lens galaxies red, sources blue."

# Each specialist returns a tiny JSON: {"score":0-10, "assessment": "...", "contaminant": null|str}
_FACTORS = {
    "morphology": dict(
        tools=("fetch_cutout",),
        prompt="You are a MORPHOLOGY specialist grading one factor of a strong-lens "
        "candidate. Call fetch_cutout(name, survey) and study the views (weight "
        "`residual`). Score 0-10 how strongly a blue feature curves TANGENTIALLY around "
        "the central red galaxy with a counter-image/ring (true lensing geometry), vs a "
        "radial spiral arm / edge-on disk / straight tidal feature. " + _FOV),
    "color": dict(
        tools=("fetch_cutout", "get_photometry"),
        prompt="You are a COLOR specialist. Call fetch_cutout and get_photometry. Score "
        "0-10 how well the candidate shows the lens-red / source-blue signature: a red "
        "central elliptical with a distinct BLUER, low-surface-brightness feature 1-5\" "
        "away (annulus bluer than core supports this). " + _FOV),
    "contaminant": dict(
        tools=("fetch_cutout",),
        prompt="You are a CONTAMINANT specialist (the veto channel). Call fetch_cutout. "
        "Decide if this is a known false positive: spiral arms, ring/polar-ring galaxy, "
        "merger/tidal tail, bright-star halo/diffraction spikes, cosmic ray/satellite "
        "trail, or noise. Score 0-10 = probability it IS a contaminant (10 = certainly "
        "not a lens). Put the named cause in `contaminant` if score>=6. " + _FOV),
    "crossmatch": dict(
        tools=("crossmatch_local",),
        prompt="You are a CROSSMATCH specialist. Call crossmatch_local(ra, dec). Score "
        "0-10 how strongly a prior published/confirmed lens corroborates this position."),
}
_FACTOR_OUT = ('\n\nRespond with ONLY this JSON: '
               '{"score": 0-10, "assessment": "one sentence", "contaminant": null | "cause"}')

_ARBITER = """You are the GRADE ARBITRATOR. You receive per-factor reports from specialist
agents who each examined one aspect of a strong-lens candidate, plus the CNN ML scores.
Fuse them into a final grade.

Grades: A almost-certain lens (clear tangential arc/counter-image/ring around a red
galaxy); B probable; C possible/ambiguous; D not a lens.
Rules: an **A requires strong morphology AND at least two of {morphology, color,
crossmatch} corroborating**; if the contaminant factor names a decisive contaminant
(score>=7) cap at C/D unless morphology is overwhelming; when between A and B choose B
and set escalate_to_human. Be conservative.

Respond with ONLY this JSON:
{"grade":"A|B|C|D","criteria":{"blue_source":0-10,"low_surface_brightness":0-10,
"curvature":0-10,"counter_images":0-10,"arc_morphology":0-10},"p_lens":0-1,
"confidence":0-1,"contaminant":null|"cause","escalate_to_human":bool,"rationale":"..."}"""

DEFAULT_FACTORS = ("morphology", "color", "contaminant")


async def _factor(cand, name, model, trace_dir):
    spec = _FACTORS[name]
    mcp_servers, allowed = server.build(list(spec["tools"]))
    tr = hooks.Trace(trace_dir / f"{cand['name']}.{name}.jsonl") if trace_dir else None
    opts = ClaudeAgentOptions(
        model=model or config.MODELS["worker"], system_prompt=spec["prompt"] + _FACTOR_OUT,
        mcp_servers=mcp_servers, allowed_tools=allowed, permission_mode="bypassPermissions",
        max_turns=config.MAX_TURNS, max_budget_usd=config.MAX_BUDGET_USD,
        setting_sources=None, hooks=tr.hooks() if tr else None)
    try:
        raw, cost, turns, _ = await _collect(_user_message(cand), opts)
    except Exception:
        return name, {"score": None, "assessment": "error"}, 0.0, 0
    obj = parse.extract_json_block(raw) or {"score": None, "assessment": raw[:120]}
    return name, obj, cost, turns


async def grade_candidate(cand: dict, *, model: Optional[str] = None,
                          factors=DEFAULT_FACTORS, trace_path: Optional[str] = None) -> GradeResult:
    from pathlib import Path
    trace_dir = Path(trace_path).parent if trace_path else config.OUT / "traces_multiagent"
    trace_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    reports = await asyncio.gather(*[_factor(cand, f, model, trace_dir) for f in factors])
    factor_json = {name: obj for name, obj, _, _ in reports}
    cost = sum(c for _, _, c, _ in reports)
    turns = sum(t for _, _, _, t in reports)

    # arbitrator: pure reasoning over the factor reports (no tools)
    ml = {k: cand.get(k) for k in ("p_resnet", "p_effnet", "p_meta") if cand.get(k) is not None}
    arb_user = (f"Candidate {cand['name']}. CNN scores: {ml}.\n"
                f"Specialist factor reports:\n{json.dumps(factor_json, indent=2)}\n\n"
                "Fuse into the final grade JSON.")
    arb_opts = ClaudeAgentOptions(model=model or config.MODELS["arbitrator"],
                                  system_prompt=_ARBITER, permission_mode="bypassPermissions",
                                  max_turns=1, setting_sources=None)
    try:
        raw, c2, t2, _ = await _collect(arb_user, arb_opts)
        cost += c2; turns += t2
    except Exception as e:
        return GradeResult(None, "", cost_usd=cost, num_turns=turns, error=str(e),
                           meta={"name": cand.get("name")})
    grade = parse.parse_model(raw, ImageGrade)
    return GradeResult(
        grade=grade, raw=json.dumps({"factors": factor_json, "arbiter": raw}),
        cost_usd=cost, num_turns=turns, parse_ok=grade is not None,
        meta={"name": cand.get("name"), "wall_s": round(time.time() - t0, 2),
              "factors": list(factor_json)})
