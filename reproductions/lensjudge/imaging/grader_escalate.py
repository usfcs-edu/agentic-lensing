"""imaging/grader_escalate.py — two-tier DESI -> high-res escalation grader (LensJudge v2 B1).

Tier-1: the lean DESI grz grade (optionally with the v2 rubric). If the candidate is
**ambiguous** (grade in {B,C} or escalate_to_human) AND higher-resolution imagery covers it
(common/highres.resolve_highres), Tier-2 re-grades the SAME sky position at high resolution
(Euclid 0.1") and SUPERSEDES the tier-1 grade — the README shows grade-C blobs resolve into
clear arcs/non-arcs there. On the DECaLS-south footprint (no Euclid Q1 overlap, and when the
Euclid cutouts are not staged) this is a SAFE NO-OP: every row stays tier-1, escalated=False.

Provenance (tier, escalated, highres_survey, p_lens_tier1, p_lens_tier2, grade_tier1) is
recorded in GradeResult.meta and surfaced as columns by run_batch._row_dict.
"""
from __future__ import annotations

from typing import Optional

from lensjudge.common import highres
from lensjudge.imaging import grader_lean
from lensjudge.imaging.grader_lean import GradeResult

ESCALATE_GRADES = ("B", "C")


async def grade_candidate(cand: dict, *, model: Optional[str] = None,
                          tools=("fetch_cutout", "get_photometry"),
                          system_prompt: Optional[str] = None,
                          trace_path: Optional[str] = None) -> GradeResult:
    g1 = await grader_lean.grade_candidate(
        cand, model=model, tools=tools, system_prompt=system_prompt, trace_path=trace_path)
    g1.meta.update({"tier": 1, "escalated": False, "highres_survey": None,
                    "grade_tier1": (g1.grade.grade if g1.grade else None),
                    "p_lens_tier1": (g1.grade.p_lens if g1.grade else None),
                    "p_lens_tier2": None})

    trigger = g1.parse_ok and g1.grade is not None and (
        g1.grade.grade in ESCALATE_GRADES or g1.grade.escalate_to_human)
    if not trigger:
        return g1
    hit = highres.resolve_highres(cand.get("name"), cand.get("ra"), cand.get("dec"))
    if not hit:
        return g1

    # Tier-2: re-grade the same object at high resolution (reuse the Euclid grader).
    from lensjudge.eval import run_euclid
    try:
        res = await run_euclid.grade_euclid({"id_str": hit["id_str"]}, model=model)
    except Exception as e:  # missing data / network -> keep tier-1, record why
        g1.meta["escalate_error"] = f"{type(e).__name__}: {e}"
        return g1

    if not res or res.get("agent_grade") is None:
        g1.meta["escalate_error"] = res.get("error", "tier-2 parse failed") if res else "no result"
        return g1

    g2 = g1.grade.model_copy(update={
        "grade": res["agent_grade"], "p_lens": float(res.get("p_lens") or 0.0),
        "confidence": float(res.get("confidence") or g1.grade.confidence),
        "contaminant": res.get("contaminant"),
        "rationale": f"[tier2 {hit['survey']} 0.1\"] " + str(res.get("rationale", ""))[:280]})
    g1.meta.update({"tier": 2, "escalated": True, "highres_survey": hit["survey"],
                    "highres_id": hit["id_str"], "p_lens_tier2": g2.p_lens})
    return GradeResult(grade=g2, raw=g1.raw, parse_ok=True,
                       cost_usd=g1.cost_usd + float(res.get("cost_usd") or 0.0),
                       num_turns=g1.num_turns, meta=g1.meta)
