#!/usr/bin/env python3
"""eval/run_cascade_recall.py — does the cascade's rank-routing keep the real lenses?

The cascade escalates the top --pass-frac of survivors BY the cheap stage-1 `direct` score. The
recall question — are the real lenses in that top fraction, or does rank-routing drop them? — depends
only on the stage-1 ranking. So we run stage-1 ONCE on a labeled set (lens A/B/C + mimic + random) and
sweep pass_frac analytically: lens recall (good, want high) vs mimic/random escalation (want lower).
~1/6 the cost of running the full stage-1->stage-2 cascade at each threshold.

  python lensjudge/eval/run_cascade_recall.py --manifest outputs/lensbench_evidence.csv --max-usd 2
"""
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from lensjudge import config  # noqa: E402
from lensjudge.imaging import grader_direct  # noqa: E402
from lensjudge.eval.run_cascade import top_frac_indices  # noqa: E402

RUBRIC_DIRECT = config.HERE / "prompts" / "rubric_imaging_v2_inline.md"


def _cands(manifest: str, limit: int) -> pd.DataFrame:
    df = pd.read_csv(manifest).rename(columns={"grade_truth": "grade"})
    if "survey_key" not in df.columns:
        df["survey_key"] = df.get("catalog", "storfer")
    if "catalog" not in df.columns:
        df["catalog"] = df["survey_key"]
    return df.head(limit) if limit else df


async def run_direct(cands: pd.DataFrame, model, conc: int) -> pd.DataFrame:
    sysd = RUBRIC_DIRECT.read_text()
    sem = asyncio.Semaphore(conc)

    async def one(c):
        async with sem:
            g = await grader_direct.grade_candidate(c, model=model, system_prompt=sysd)
        return {"name": c["name"], "grade_truth": c.get("grade"), "source": c.get("source"),
                "p_lens": (g.grade.p_lens if (g.parse_ok and g.grade) else -1.0),
                "grade_pred": (g.grade.grade if g.grade else None), "cost_usd": g.cost_usd}

    recs = await asyncio.gather(*(one(c) for _, c in cands.iterrows()))
    return pd.DataFrame(recs)


def sweep(df: pd.DataFrame):
    df = df.reset_index(drop=True).copy()
    df["is_lens"] = df.grade_truth.isin(["A", "B", "C"])
    scores = df.p_lens.tolist()
    n_lens = int(df.is_lens.sum())
    mim, ran = df.source.eq("mimic_neg"), df.source.eq("random_neg")
    print(f"\n=== cascade rank-routing recall sweep (n={len(df)} | {n_lens} lenses, "
          f"{int(mim.sum())} mimics, {int(ran.sum())} randoms) ===")
    # detection AUC of the cheap stage-1 score (lens A/B/C vs random)
    try:
        from sklearn.metrics import roc_auc_score
        m = df.source.isin(["graded"]) | ran
        auc = roc_auc_score(df.is_lens[m].astype(int), df.p_lens[m]) if m.sum() else float("nan")
        print(f"stage-1 detection AUC (lens vs random): {auc:.3f}")
    except Exception:
        pass
    print(f"\n{'pass_frac':>9} {'escalated':>9} {'lens_recall':>11} {'mimic_esc':>9} {'rand_esc':>8}")
    for pf in [0.2, 0.3, 0.5, 0.7, 1.0]:
        passers = top_frac_indices(scores, pf)
        idx = df.index.isin(passers)
        lens_rec = (df.is_lens & idx).sum() / max(1, n_lens)
        mim_esc = (mim & idx).sum() / max(1, int(mim.sum()))
        ran_esc = (ran & idx).sum() / max(1, int(ran.sum()))
        print(f"{pf:>9.2f} {len(passers):>9} {lens_rec:>11.0%} {mim_esc:>9.0%} {ran_esc:>8.0%}")
    print("\n(want: lens_recall high while mimic_esc/rand_esc lower than pass_frac == the cheap "
          "stage-1 enriches lenses into the escalated set, saving stage-2 spend on junk.)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(config.OUT / "lensbench_evidence.csv"))
    ap.add_argument("--out", default=str(config.OUT / "cascade_recall_stage1.parquet"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=5)
    ap.add_argument("--model", default=None)
    ap.add_argument("--max-usd", type=float, default=2.0)
    args = ap.parse_args()
    cands = _cands(args.manifest, args.limit)
    est = len(cands) * 0.013
    print(f"[recall] stage-1 direct on {len(cands)} labeled rows; est ~${est:.2f} (cap ${args.max_usd})")
    if est > args.max_usd:
        raise SystemExit(f"ABORT: est ${est:.2f} > --max-usd ${args.max_usd}")
    df = asyncio.run(run_direct(cands, args.model, args.concurrency))
    df.to_parquet(args.out, index=False)
    print(f"saved {args.out}  (total ${df.cost_usd.sum():.2f})")
    sweep(df)


if __name__ == "__main__":
    main()
