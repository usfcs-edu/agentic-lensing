#!/usr/bin/env python3
"""eval/run_hsc_validation.py — validate HSC PDR3 tier-2 escalation on known lenses (Euclid-B1 analog).

Grade the SuGOHI-matched DESI candidates (known HSC lenses with committee grades) through the
escalate grader and measure: (i) HSC coverage, (ii) the DESI 1.3" -> HSC 0.168" p_lens flip,
(iii) recovery = fraction graded A/B at HSC, (iv) agreement with the SuGOHI committee grade. These
are real lenses, so the over-skeptical DESI grader should under-grade them and HSC tier-2 recover.

  python lensjudge/eval/run_hsc_validation.py --concurrency 4 --max-usd 4
"""
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd  # noqa: E402

from lensjudge import config  # noqa: E402
from lensjudge.imaging import grader_escalate  # noqa: E402

SURV = {"storfer": "storfer", "inchausti": "inchausti"}


async def _one(r, sysa, model, sem):
    cand = {"name": r["name"], "survey_key": SURV.get(r["source"], "storfer"),
            "catalog": r["source"], "ra": float(r["ra"]), "dec": float(r["dec"]),
            "region": "DECaLS"}
    async with sem:
        g = await grader_escalate.grade_candidate(cand, model=model, system_prompt=sysa)
    return {"name": r["name"], "source": r["source"], "desi_grade": r["grade"],
            "sugohi_grade": r["sugohi_grade"], "tier1_grade": g.meta.get("grade_tier1"),
            "p_lens_tier1": g.meta.get("p_lens_tier1"), "escalated": bool(g.meta.get("escalated")),
            "highres_survey": g.meta.get("highres_survey"), "p_lens_tier2": g.meta.get("p_lens_tier2"),
            "final_grade": (g.grade.grade if g.grade else None),
            "final_p_lens": (g.grade.p_lens if g.grade else None),
            "cost_usd": g.cost_usd, "err": g.meta.get("escalate_error")}


async def run(df, model, conc):
    sysa = (config.HERE / "prompts" / "rubric_imaging_v2.md").read_text()
    sem = asyncio.Semaphore(conc)
    recs = await asyncio.gather(*(_one(r, sysa, model, sem) for _, r in df.iterrows()))
    return pd.DataFrame(recs)


def summarize(df):
    n, nesc = len(df), int(df.escalated.sum())
    print(f"\n=== HSC validation: {n} SuGOHI-matched known lenses ===")
    print(f"HSC coverage (escalated to tier-2): {nesc}/{n}  "
          f"| highres_survey {df.highres_survey.value_counts(dropna=False).to_dict()}")
    e = df[df.escalated].copy()
    if len(e):
        print(f"DESI tier-1 grade dist: {e.tier1_grade.value_counts().to_dict()}")
        print(f"HSC  tier-2 grade dist: {e.final_grade.value_counts().to_dict()}")
        t1 = pd.to_numeric(e.p_lens_tier1, errors="coerce")
        t2 = pd.to_numeric(e.p_lens_tier2, errors="coerce")
        print(f"median p_lens: DESI 1.3\" {t1.median():.3f} -> HSC 0.168\" {t2.median():.3f}  "
              f"(rose on {(t2 > t1).sum()}/{len(e)})")
        ab = e.final_grade.isin(["A", "B"])
        print(f"recovery (graded A/B at HSC): {int(ab.sum())}/{len(e)} = {ab.mean():.0%}")
        agree = (ab.values == e.sugohi_grade.isin(["A", "B"]).values).mean()
        print(f"agreement with SuGOHI committee (lens=A/B): {agree:.0%}")
    if df.err.notna().any():
        print("\nescalate errors:")
        print(df[df.err.notna()][["name", "err"]].to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--model", default=None)
    ap.add_argument("--max-usd", type=float, default=4.0)
    args = ap.parse_args()
    xm = pd.read_csv(config.OUT / "xmatch_sugohi.csv")
    df = xm[xm.source.isin(["storfer", "inchausti"])].reset_index(drop=True)
    if args.limit:
        df = df.head(args.limit)
    est = len(df) * 0.09   # ~tier-1 + tier-2 per object
    print(f"[hsc-val] {len(df)} candidates; est ~${est:.2f} (cap ${args.max_usd})")
    if est > args.max_usd:
        raise SystemExit(f"ABORT: est ${est:.2f} > --max-usd ${args.max_usd}")
    out = asyncio.run(run(df, args.model, args.concurrency))
    out.to_parquet(config.OUT / "hsc_validation.parquet", index=False)
    print(f"saved {config.OUT/'hsc_validation.parquet'}  (total ${out.cost_usd.sum():.2f})")
    summarize(out)
    cols = ["name", "desi_grade", "sugohi_grade", "tier1_grade", "p_lens_tier1",
            "final_grade", "p_lens_tier2", "highres_survey"]
    print("\n" + out[[c for c in cols if c in out]].to_string(index=False))


if __name__ == "__main__":
    main()
