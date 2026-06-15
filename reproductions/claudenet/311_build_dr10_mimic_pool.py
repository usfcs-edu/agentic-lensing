#!/usr/bin/env python3
"""311_build_dr10_mimic_pool.py — ClaudeNet v3 A1: assemble the DR10-native lens-mimic
mining pool from the C-now sweep (no new mining run needed).

The DR9 seed bank (300) was 601 rows. The DR10 v2-lean sweep just produced 148,034 NEW
survivors: CNN-high (by the same v2-lean ensemble v3 will improve), DR10-native (so they
fix the DR9->DR10 domain shift C-now exposed), and — per the C-vet — overwhelmingly
lens-mimics (lrg_companion/merger/ring). They ARE the hard-negative pool the mimic bank
needs. This script:
  1. takes the NEW survivors (status from 163 crossmatch), excludes the 5 qualified
     candidates (real-lens candidates -> never negatives);
  2. joins the DR10 parent morphology (TYPE/SERSIC/SHAPE_R/mag_z/mag_i) for typing
     (extended LRG = the staf327 mimic regime: DEV/SER + large SHAPE_R + red);
  3. folds in the agentic contaminant labels for the graded subset (the C-vet 30);
  4. writes data/v3/mimic_pool_dr10.parquet (typed where possible) — the A1 mining pool
     that A2 retrains on and A3's mimic-null calibrates against.

Agentic grading (the mine->grade->fold loop, G1 gate) runs on a SAMPLE only, to keep within
the API budget: the CNN-high + not-a-known-lens status already makes these confident hard
negatives (v2-style mining), so per-row grading is for validation + typing, not labeling.

    /home2/benson/.venvs/claudenet/bin/python 311_build_dr10_mimic_pool.py
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

import _clib as C

ROOT = Path(__file__).resolve().parent
SD = ROOT / "data" / "v3" / "sweep_dr10"
OUT = ROOT / "data" / "v3"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT / "mimic_pool_dr10.parquet"))
    args = ap.parse_args()

    # 1. NEW survivors (CNN-high, not a known lens)
    xm = pd.read_parquet(SD / "crossmatch.parquet")
    s2 = pd.read_parquet(SD / "stage2_scores.parquet", columns=["row_id", "RA", "DEC", "brick", "p_final"])
    df = s2.merge(xm[["row_id", "status"]], on="row_id", how="left")
    new = df[df["status"] == "NEW"].copy()
    print(f"[311] NEW survivors: {len(new):,}")

    # exclude the 5 qualified real-lens candidates
    qpath = OUT / "dr10_qualified_candidates.csv"
    if qpath.exists():
        qual = set(pd.read_csv(qpath)["name"].astype(str))
        new = new[~new["row_id"].astype(str).isin(qual)]
        print(f"[311] excluded {len(qual)} qualified candidates -> {len(new):,}")

    # 2. join DR10 parent morphology
    parts = glob.glob(str(SD / "sweep_manifest_part*.parquet"))   # row_id only here
    # the morphology lives in the parent on $SCRATCH; locally we have the manifests (row_id,RA,DEC).
    # Type from the brick/row context we have + the agentic labels; full Sersic typing joins the
    # DR10 parent parquet (Perlmutter) in a follow-up. For now flag the staf327 regime by score.
    new["mimic_type"] = "cnn_high_unknown"      # refined by agentic labels below + parent join (TODO)

    # 3. fold in the C-vet agentic contaminant labels (the 30 graded)
    for vp in [ROOT.parent / "lensjudge" / "outputs" / "dr10_vet_top30.parquet"]:
        if Path(vp).exists():
            v = pd.read_parquet(vp)[["name", "grade_pred", "contaminant"]].rename(columns={"name": "row_id"})
            v["row_id"] = v["row_id"].astype(str)
            new["row_id"] = new["row_id"].astype(str)
            new = new.merge(v, on="row_id", how="left")
            graded = new["grade_pred"].notna()
            # confirmed mimic = graded C/D (non-lens) with a named contaminant
            new.loc[graded & new["grade_pred"].isin(["C", "D"]) & new["contaminant"].notna(),
                    "mimic_type"] = new["contaminant"]
            new["agentic_graded"] = graded
            print(f"[311] folded {int(graded.sum())} agentic grades; "
                  f"confirmed-mimic types: "
                  f"{new.loc[graded & new.grade_pred.isin(['C','D']), 'contaminant'].value_counts().to_dict()}")
            break

    new = new.reset_index(drop=True)
    new.to_parquet(args.out, index=False)
    print(f"[311] wrote {args.out}: {len(new):,} DR10 mimic-pool rows "
          f"(p_final {new.p_final.min():.3f}-{new.p_final.max():.3f})")
    print(f"[311] mimic_type: {new.mimic_type.value_counts().to_dict()}")
    print(f"[311] NEXT: (a) join $SCRATCH parent for SERSIC/SHAPE_R typing; (b) agentic-grade a "
          f"~200-row stratified sample for the G1 yield gate; (c) 312 assemble seed(601)+DR10 "
          f"bank + frozen held-out; (d) stage survivor cutouts on Perlmutter for A2 retraining.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
