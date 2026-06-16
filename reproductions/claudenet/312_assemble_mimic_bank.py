#!/usr/bin/env python3
"""312_assemble_mimic_bank.py — ClaudeNet v3 A1: assemble the v3 lens-mimic bank.

Combines the DR9 seed (300, 601 confirmed-non-lens campaign rejects) with the DR10-native pool
(311, 148k CNN-high non-lenses, morphology-typed; G1 yield gate = 98.3% non-lens), excludes the
real-lens candidates (the 5 qualified + any agentic-graded A/B), and carves a frozen,
stratified-by-type held-out **mimic-eval** that NEVER enters training (the negative denominator
for the recovery@matched-mimic-FPR metric, kept honest). Output:
  data/v3/mimic_bank.parquet       — training bank (typed, DR9+DR10)
  data/v3/mimic_bank_eval.parquet  — frozen 20% held-out mimic-eval

    /home2/benson/.venvs/claudenet/bin/python 312_assemble_mimic_bank.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
V3 = ROOT / "data" / "v3"
COLS = ["row_id", "RA", "DEC", "mimic_type", "p_final", "source"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-frac", type=float, default=0.20)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    dr10 = pd.read_parquet(V3 / "mimic_pool_dr10.parquet")
    dr10["source"] = "dr10_sweep"
    dr9 = pd.read_parquet(V3 / "mimic_bank_seed.parquet")
    dr9["source"] = "dr9_seed"
    for d in (dr10, dr9):
        d["row_id"] = d["row_id"].astype(str)

    # exclusions: real-lens candidates must never be negatives
    excl = set()
    qp = V3 / "dr10_qualified_candidates.csv"
    if qp.exists():
        excl |= set(pd.read_csv(qp)["name"].astype(str))
    g1 = ROOT.parent / "lensjudge" / "outputs" / "g1_sample.parquet"
    if g1.exists():
        g = pd.read_parquet(g1)
        excl |= set(g[g["grade_pred"].isin(["A", "B"])]["name"].astype(str))
    # any agentic A/B already in the pool (grade_pred col from 311 fold-in)
    if "grade_pred" in dr10.columns:
        excl |= set(dr10[dr10["grade_pred"].isin(["A", "B"])]["row_id"].astype(str))
    n0 = len(dr10)
    dr10 = dr10[~dr10["row_id"].isin(excl)]
    print(f"[312] excluded {n0 - len(dr10)} real-lens candidates from the DR10 pool")

    bank = pd.concat([dr10[[c for c in COLS if c in dr10.columns]],
                      dr9[[c for c in COLS if c in dr9.columns]]], ignore_index=True)
    bank = bank.drop_duplicates("row_id").reset_index(drop=True)

    # frozen stratified-by-type held-out mimic-eval
    rng = np.random.default_rng(args.seed)
    bank["is_eval"] = False
    for t, g in bank.groupby("mimic_type"):
        k = int(round(len(g) * args.eval_frac))
        if k:
            bank.loc[g.sample(k, random_state=args.seed).index, "is_eval"] = True
    train = bank[~bank.is_eval].drop(columns="is_eval")
    ev = bank[bank.is_eval].drop(columns="is_eval")
    train.to_parquet(V3 / "mimic_bank.parquet", index=False)
    ev.to_parquet(V3 / "mimic_bank_eval.parquet", index=False)

    print(f"[312] mimic bank: {len(train):,} train + {len(ev):,} held-out eval "
          f"(DR9 seed {int((bank.source=='dr9_seed').sum())}, DR10 {int((bank.source=='dr10_sweep').sum())})")
    print("[312] train mimic_type composition:")
    for t, n in train.mimic_type.value_counts().items():
        print(f"        {t:18s} {n:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
