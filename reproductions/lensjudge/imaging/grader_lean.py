"""Phase-1 lean single-agent imaging grader (the baseline).

One Claude call per candidate: the agent calls fetch_cutout (multi-view PNG) + the
codified 5-criterion rubric + the ML scores, and returns one ImageGrade JSON. The
five criteria are scored *fields* of a single integrated judgment, not separate
agents — matching how a human grades at a glance. This is the cheap baseline the
robust (Option C) and §9.1 multi-agent designs are measured against in LensBench.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from claude_agent_sdk import (AssistantMessage, ClaudeAgentOptions, ResultMessage,
                              TextBlock, query)

from lensjudge import config
from lensjudge.common import hooks, parse
from lensjudge.common.schemas import ImageGrade
from lensjudge.tools import server

_RUBRIC = (Path(__file__).resolve().parents[1] / "prompts" / "rubric_imaging.md").read_text()

_REPAIR = ("Your previous reply was not valid JSON for the required schema. Re-emit "
           "EXACTLY ONE JSON object with keys grade, criteria{blue_source,"
           "low_surface_brightness,curvature,counter_images,arc_morphology}, p_lens, "
           "confidence, contaminant, escalate_to_human, rationale — and nothing else. "
           "Here is your previous reply to fix:\n\n")


@dataclass
class GradeResult:
    grade: Optional[ImageGrade]
    raw: str
    cost_usd: float = 0.0
    num_turns: int = 0
    parse_ok: bool = False
    error: Optional[str] = None
    meta: dict = field(default_factory=dict)


def _user_message(cand: dict) -> str:
    parts = [f"{k}={cand[k]:.3f}" for k in ("p_resnet", "p_effnet", "p_meta")
             if cand.get(k) is not None]
    ml = ("CNN ensemble scores (prior, not ground truth): " + ", ".join(parts) + "."
          if parts else "")
    loc = f"name={cand['name']!r}, survey={cand.get('survey_key', cand.get('catalog', 'storfer'))!r}"
    radec = ""
    if cand.get("ra") is not None and cand.get("dec") is not None:
        radec = f" RA={cand['ra']:.6f}, Dec={cand['dec']:.6f}"
    return (f"Grade this strong-lens candidate. {loc}.{radec}\n"
            f"Tractor type: {cand.get('tractor_type', '?')}, region: {cand.get('region', '?')}.\n"
            f"{ml}\n\nCall fetch_cutout with that name and survey, inspect the views, "
            f"then respond with ONLY the JSON object.")


async def _collect(prompt: str, opts: ClaudeAgentOptions):
    """Run one query(); return (final_text, cost_usd, num_turns)."""
    texts, cost, turns = [], 0.0, 0
    async for msg in query(prompt=prompt, options=opts):
        if isinstance(msg, AssistantMessage):
            for b in msg.content:
                if isinstance(b, TextBlock):
                    texts.append(b.text)
        elif isinstance(msg, ResultMessage):
            cost = msg.total_cost_usd or 0.0
            turns = msg.num_turns or 0
            if msg.result:
                texts.append(msg.result)
    return (texts[-1] if texts else ""), cost, turns


async def grade_candidate(cand: dict, *, model: Optional[str] = None,
                          tools=("fetch_cutout", "get_photometry"),
                          system_prompt: Optional[str] = None,
                          trace_path: Optional[str] = None) -> GradeResult:
    mcp_servers, allowed = server.build(list(tools))
    tr = hooks.Trace(trace_path) if trace_path else None
    opts = ClaudeAgentOptions(
        model=model or config.MODELS["grader"],
        system_prompt=system_prompt or _RUBRIC,
        mcp_servers=mcp_servers,
        allowed_tools=allowed,
        permission_mode="bypassPermissions",
        max_turns=config.MAX_TURNS,
        max_budget_usd=config.MAX_BUDGET_USD,
        setting_sources=None,          # don't load the repo's .claude settings
        hooks=tr.hooks() if tr else None,
    )
    t0 = time.time()
    try:
        raw, cost, turns = await _collect(_user_message(cand), opts)
    except Exception as e:
        return GradeResult(None, "", error=f"{type(e).__name__}: {e}")
    grade = parse.parse_model(raw, ImageGrade)
    total_cost, total_turns = cost, turns

    if grade is None and raw:
        # one repair retry — no tools, just reformat the prior text
        repair_opts = ClaudeAgentOptions(
            model=model or config.MODELS["grader"],
            system_prompt=_RUBRIC, permission_mode="bypassPermissions",
            max_turns=1, setting_sources=None)
        try:
            raw2, cost2, turns2 = await _collect(_REPAIR + raw, repair_opts)
            total_cost += cost2; total_turns += turns2
            g2 = parse.parse_model(raw2, ImageGrade)
            if g2 is not None:
                grade, raw = g2, raw2
        except Exception:
            pass

    return GradeResult(
        grade=grade, raw=raw, cost_usd=total_cost, num_turns=total_turns,
        parse_ok=grade is not None,
        meta={"name": cand.get("name"), "wall_s": round(time.time() - t0, 2),
              "trace": str(tr.path) if tr else None},
    )
