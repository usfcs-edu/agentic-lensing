"""Spectroscopic VI pre-grader: classify a discordant-redshift pair lens/dimple/not_lens.

Reasons over the Hsu catalog features (z_lens, z_src, separation, sigma_v, theta_E,
logmstar, class_algo) plus the DR10 imaging cutout (fetch_cutout) and the SIS
theta_E/separation consistency (get_specfit). The raw DESI fiber flux is the documented
enhancement (see tools/spectrum.fetch_spectrum); this grader is the catalog+imaging
version that is testable today.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from claude_agent_sdk import ClaudeAgentOptions

from lensjudge import config
from lensjudge.common import hooks, parse
from lensjudge.common.schemas import SpecGrade
from lensjudge.imaging.grader_lean import _collect
from lensjudge.tools import server

_RUBRIC = (Path(__file__).resolve().parents[1] / "prompts" / "rubric_spectro.md").read_text()

_REPAIR = ("Your previous reply was not valid JSON for the required schema. Re-emit "
           "EXACTLY ONE JSON object with keys cls (lens|dimple|not_lens), plausible, "
           "confidence, z_fg, z_bg, sigma_v, rationale — and nothing else. "
           "Here is your previous reply to fix:\n\n")


@dataclass
class SpecResult:
    grade: Optional[SpecGrade]
    raw: str
    cost_usd: float = 0.0
    num_turns: int = 0
    parse_ok: bool = False
    error: Optional[str] = None
    meta: dict = field(default_factory=dict)


def _user_message(pair: dict) -> str:
    feats = {k: pair.get(k) for k in
             ("z_lens", "z_src", "sep_arcsec", "sigma_v_lens", "theta_E_arcsec",
              "logmstar_lens", "class_algo")}
    name = pair.get("name", "pair")
    loc = ""
    if pair.get("ra") is not None and pair.get("dec") is not None:
        loc = f" lens RA={pair['ra']:.6f}, Dec={pair['dec']:.6f}"
    return (f"Grade this Hsu-style discordant-redshift spectroscopic lens candidate.\n"
            f"Lens name={name!r}, survey='inchausti'.{loc}\n"
            f"Catalog features: {json.dumps(feats)}\n\n"
            f"Call get_specfit (sigma_v={feats['sigma_v_lens']}, z_lens={feats['z_lens']}, "
            f"z_src={feats['z_src']}, sep_arcsec={feats['sep_arcsec']}) for the Einstein-"
            f"radius/separation check, and fetch_cutout for the imaging. Then respond "
            f"with ONLY the JSON object.")


async def grade(pair: dict, *, model: Optional[str] = None,
                tools=("get_specfit", "fetch_cutout"),
                trace_path: Optional[str] = None) -> SpecResult:
    mcp_servers, allowed = server.build(list(tools))
    tr = hooks.Trace(trace_path) if trace_path else None
    opts = ClaudeAgentOptions(
        model=model or config.MODELS["spectro"], system_prompt=_RUBRIC,
        mcp_servers=mcp_servers, allowed_tools=allowed, permission_mode="bypassPermissions",
        max_turns=config.MAX_TURNS, max_budget_usd=config.MAX_BUDGET_USD,
        setting_sources=None, hooks=tr.hooks() if tr else None,
        **config.thinking_options())
    t0 = time.time()
    try:
        raw, cost, turns, tstats = await _collect(_user_message(pair), opts, tr)
    except Exception as e:
        return SpecResult(None, "", error=f"{type(e).__name__}: {e}")
    g = parse.parse_model(raw, SpecGrade)

    if g is None and raw:
        # one repair retry — no tools, just reformat the prior text (mirrors grader_lean)
        repair_opts = ClaudeAgentOptions(
            model=model or config.MODELS["spectro"], system_prompt=_RUBRIC,
            permission_mode="bypassPermissions", max_turns=1, setting_sources=None)
        try:
            raw2, cost2, turns2, _ = await _collect(_REPAIR + raw, repair_opts)
            cost += cost2; turns += turns2
            g2 = parse.parse_model(raw2, SpecGrade)
            if g2 is not None:
                g, raw = g2, raw2
        except Exception:
            pass

    return SpecResult(grade=g, raw=raw, cost_usd=cost, num_turns=turns, parse_ok=g is not None,
                      meta={"name": pair.get("name"), "wall_s": round(time.time() - t0, 2),
                            "n_thinking_blocks": tstats["n_thinking_blocks"],
                            "thinking_chars": tstats["thinking_chars"]})
