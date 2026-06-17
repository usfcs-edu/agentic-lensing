#!/usr/bin/env python3
"""eval/run_cascade.py — the B7 stage-2 deploy cascade (the natural "v3" milestone).

The agency ablation (AGENCY_ABLATION.md) prescribes vetting survivors as a cascade, not one mode:
  1. **Detection triage (bulk):** the loop-free `direct` grader (v2-inline rubric) on EVERY survivor
     — matches the agentic detection AUC at ~$0.012/cand (1 turn), ~6x cheaper than lean.
  2. **Mimic adjudication (survivors that pass):** the agentic `escalate` grader (v2 rubric) — the
     only config that reaches the 0.71 lens-vs-mimic AUC — plus tier-2 high-res (Euclid/HSC) by
     coverage. Confident non-lenses from stage 1 are dropped cheaply.
Output: one preds parquet with a `stage` column + per-stage cost.

Cost (AGENCY_ABLATION, per 5k survivors): ~$60 all-direct / ~$170 cascade / ~$355 all-agentic-v2 /
~$1,575 all-multiagent. A --max-usd gate estimates spend up front and aborts if over.

Auth: stage-1 `direct` needs ANTHROPIC_API_KEY (base Messages API); stage-2 `escalate`/`lean` ride
the Claude CLI login (Agent SDK). HSC tier-2 needs HSC_USER/HSC_PASSWORD.

  python lensjudge/eval/run_cascade.py --manifest outputs/lensbench_evidence.csv --limit 8 --max-usd 2
"""
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd  # noqa: E402

from lensjudge import config  # noqa: E402
from lensjudge.imaging import grader_direct, grader_escalate  # noqa: E402

RUBRIC_DIRECT = config.HERE / "prompts" / "rubric_imaging_v2_inline.md"   # stage-1 (no loop)
RUBRIC_AGENTIC = config.HERE / "prompts" / "rubric_imaging_v2.md"         # stage-2 (agentic)


def _cands(manifest: str, limit: int) -> list[dict]:
    df = pd.read_csv(manifest).rename(columns={"grade_truth": "grade"})
    if "survey_key" not in df.columns:
        df["survey_key"] = df.get("catalog", "storfer")
    if "catalog" not in df.columns:
        df["catalog"] = df["survey_key"]
    if limit:
        df = df.head(limit)
    return df.to_dict("records")


def top_frac_indices(scores, pass_frac: float) -> set:
    """Indices of the top ``pass_frac`` of ``scores`` (ties broken by original order). Pure /
    unit-tested: this is the cascade's rank-routing gate. >=1 always escalates when there is data."""
    n = len(scores)
    if n == 0:
        return set()
    n_pass = max(1, int(round(pass_frac * n)))
    order = sorted(range(n), key=lambda i: scores[i], reverse=True)
    return set(order[:n_pass])


def _s1_row(c, g1) -> dict:
    return {"name": c["name"], "grade_truth": c.get("grade"),
            "stage1_grade": (g1.grade.grade if g1.grade else None),
            "stage1_p_lens": (g1.grade.p_lens if g1.grade else None),
            "stage1_contaminant": (g1.grade.contaminant if g1.grade else None),
            "stage1_cost": round(g1.cost_usd, 5)}


async def run(cands, out: Path, model, pass_frac: float, concurrency: int) -> pd.DataFrame:
    """Two-phase cascade. The direct grader is over-skeptical (B7): its absolute grade/contaminant
    over-rejects real lenses, but it *ranks* lens-vs-random at AUC ~0.66. So we route by RANK —
    direct-triage all survivors cheaply, then escalate the top `pass_frac` by the stage-1 p_lens to
    the agentic v2 grader (mimic adjudication + tier-2 high-res by coverage). The bottom
    (1-pass_frac), which the cheap stage scored lowest, is dropped without paying for the loop."""
    sysd, sysa = RUBRIC_DIRECT.read_text(), RUBRIC_AGENTIC.read_text()
    sem = asyncio.Semaphore(concurrency)

    # --- phase 1: cheap direct triage on every survivor ---
    async def triage(c):
        async with sem:
            return c, await grader_direct.grade_candidate(c, model=model, system_prompt=sysd)
    s1 = await asyncio.gather(*(triage(c) for c in cands))

    # rank by stage-1 p_lens (parse failures score -1, sort last) and escalate the top pass_frac
    scores = [(g.grade.p_lens if (g.parse_ok and g.grade) else -1.0) for _, g in s1]
    passers = top_frac_indices(scores, pass_frac)

    # --- phase 2: agentic v2 mimic adjudication (+ tier-2) on the passers ---
    async def adjudicate(i):
        c, g1 = s1[i]
        async with sem:
            g2 = await grader_escalate.grade_candidate(c, model=model, system_prompt=sysa)
        return i, g2

    s2 = dict(await asyncio.gather(*(adjudicate(i) for i in passers)))

    rows = []
    for i, (c, g1) in enumerate(s1):
        row = _s1_row(c, g1)
        if i in passers:
            g2 = s2[i]
            row.update({"stage": 2, "grade_pred": (g2.grade.grade if g2.grade else None),
                        "p_lens": (g2.grade.p_lens if g2.grade else None),
                        "contaminant": (g2.grade.contaminant if g2.grade else None),
                        "tier": g2.meta.get("tier"), "escalated": g2.meta.get("escalated"),
                        "highres_survey": g2.meta.get("highres_survey"),
                        "p_lens_tier1": g2.meta.get("p_lens_tier1"),
                        "p_lens_tier2": g2.meta.get("p_lens_tier2"),
                        "stage2_cost": round(g2.cost_usd, 5),
                        "cost_usd": round(g1.cost_usd + g2.cost_usd, 5)})
        else:
            row.update({"stage": 1, "grade_pred": (g1.grade.grade if g1.grade else None),
                        "p_lens": (g1.grade.p_lens if g1.grade else None),
                        "contaminant": row["stage1_contaminant"], "escalated": False,
                        "stage2_cost": 0.0, "cost_usd": round(g1.cost_usd, 5)})
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_parquet(out, index=False)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(config.OUT / "lensbench_evidence.csv"))
    ap.add_argument("--out", default=str(config.OUT / "preds_cascade.parquet"))
    ap.add_argument("--pass-frac", type=float, default=0.5,
                    help="fraction of survivors (top-ranked by the cheap stage-1 score) escalated "
                         "to the agentic stage-2 (the cost/recall knob; B7 deploy ~0.3)")
    ap.add_argument("--limit", type=int, default=0, help="grade only the first N (smoke)")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--model", default=None)
    ap.add_argument("--max-usd", type=float, default=2.0)
    args = ap.parse_args()

    cands = _cands(args.manifest, args.limit)
    est = len(cands) * 0.012 + args.pass_frac * len(cands) * 0.07   # stage-1 all + stage-2 on passers
    print(f"[cascade] {len(cands)} candidates; pass_frac={args.pass_frac}; "
          f"est ~${est:.2f} (cap ${args.max_usd})")
    if est > args.max_usd:
        raise SystemExit(f"ABORT: estimate ${est:.2f} > --max-usd ${args.max_usd}. "
                         f"Lower --limit/--pass-frac or raise --max-usd.")
    df = asyncio.run(run(cands, Path(args.out), args.model, args.pass_frac, args.concurrency))
    n2 = int((df["stage"] == 2).sum())
    n_t2 = int(df.get("escalated", pd.Series(dtype=bool)).fillna(False).sum())
    print(f"[cascade] {len(df)} graded | {n2} escalated to stage-2 | {n_t2} reached tier-2 high-res "
          f"| total ${df['cost_usd'].sum():.3f} (stage-1 ${df['stage1_cost'].sum():.3f})")
    print(f"  saved {args.out}")
    cols = [c for c in ["name", "grade_truth", "stage1_grade", "stage", "grade_pred", "p_lens",
                        "highres_survey", "cost_usd"] if c in df.columns]
    print(df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
