"""Option-C robust grader: a parallel panel of perspective-diverse judges.

Four judges (Advocate / Skeptic / Morphologist / Contaminant-hunter) each grade the
SAME candidate independently — separate query() calls, each with a role-biased system
prompt over the shared rubric, all rendering the identical (cached) cutout. Their
JudgeVotes are fused by aggregate.aggregate() with a skeptic-veto and a 2-of-N A-rule.
Exposes grade_candidate(...) -> GradeResult, drop-in compatible with grader_lean.
"""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Optional

from claude_agent_sdk import ClaudeAgentOptions

from lensjudge import config
from lensjudge.common import hooks, parse
from lensjudge.common.schemas import JudgeVote
from lensjudge.imaging.aggregate import aggregate
from lensjudge.imaging.grader_lean import GradeResult, _RUBRIC, _collect, _user_message
from lensjudge.tools import server

ROLES = ("advocate", "skeptic", "morphology", "contaminant")
_PROMPT_DIR = Path(__file__).resolve().parents[1] / "prompts"
_ROLE_PROMPTS = {r: (_PROMPT_DIR / f"{r}.md").read_text() for r in ROLES}


async def _one_judge(cand, role, model, tools, trace_dir) -> tuple[Optional[JudgeVote], float, int]:
    mcp_servers, allowed = server.build(list(tools))
    tr = hooks.Trace(trace_dir / f"{cand['name']}.{role}.jsonl") if trace_dir else None
    opts = ClaudeAgentOptions(
        model=model or config.MODELS["judge"],
        system_prompt=_ROLE_PROMPTS[role] + "\n\n" + _RUBRIC,
        mcp_servers=mcp_servers, allowed_tools=allowed,
        permission_mode="bypassPermissions", max_turns=config.MAX_TURNS,
        max_budget_usd=config.MAX_BUDGET_USD, setting_sources=None,
        hooks=tr.hooks() if tr else None,
    )
    try:
        raw, cost, turns = await _collect(_user_message(cand), opts)
    except Exception:
        return None, 0.0, 0
    vote = parse.parse_model(raw, JudgeVote)
    if vote is not None:
        vote.role = role
    return vote, cost, turns


async def grade_candidate(cand: dict, *, model: Optional[str] = None,
                          roles=ROLES, tools=("fetch_cutout", "get_photometry"),
                          trace_path: Optional[str] = None) -> GradeResult:
    trace_dir = Path(trace_path).parent if trace_path else config.OUT / "traces_panel"
    trace_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    results = await asyncio.gather(*[
        _one_judge(cand, r, model, tools, trace_dir) for r in roles])
    votes = [v for v, _, _ in results if v is not None]
    cost = sum(c for _, c, _ in results)
    turns = sum(t for _, _, t in results)
    if not votes:
        return GradeResult(None, "", cost_usd=cost, num_turns=turns, parse_ok=False,
                           error="no parseable judge votes",
                           meta={"name": cand.get("name"), "wall_s": round(time.time() - t0, 2)})
    final = aggregate(votes)
    raw = json.dumps({"votes": [v.model_dump() for v in votes], "final": final.model_dump()})
    return GradeResult(
        grade=final, raw=raw, cost_usd=cost, num_turns=turns, parse_ok=True,
        meta={"name": cand.get("name"), "wall_s": round(time.time() - t0, 2),
              "n_votes": len(votes), "roles": [v.role for v in votes]},
    )
