#!/usr/bin/env python3
"""240_consensus.py — combine the expanded-crossmatch status, my visual grade
(skeptic-verified), and the lensjudge grade into the qualified-candidate list.

QUALIFIED = still NEW after the expanded crossmatch (not in any local/external
lens catalog, no SIMBAD lens-type) AND graded A or B by BOTH independent passes.
Tiers: gold (A&A), silver (>=B & >=1 A), bronze (B&B). Single-grader->=B rows go
to the escalation set (listed, not qualified). Everything is recorded in
consensus_full_737 with the disqualifying reason.

lensjudge_grade = the multiagent re-grade where it exists (the A/B subset), else
the lean grade.

    /home2/benson/.venvs/claudenet/bin/python campaign/240_consensus.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "v2" / "campaign"
LJ = ROOT.parent / "lensjudge" / "outputs"
RANK = {"A": 3, "B": 2, "C": 1, "D": 0}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xmatched", default=str(OUT / "manifest_737_xmatched.parquet"))
    ap.add_argument("--visual", default=str(OUT / "visual_grades_verified.parquet"))
    ap.add_argument("--lj-lean", default=str(LJ / "claudenet_737_lean.parquet"))
    ap.add_argument("--lj-multi", default=str(LJ / "claudenet_737_multiagent.parquet"))
    args = ap.parse_args()

    man = pd.read_parquet(args.xmatched)
    man["row_id"] = man["row_id"].astype(str)
    vis = pd.read_parquet(args.visual)[
        ["row_id", "first_grade", "my_grade", "my_p_lens", "my_confidence",
         "skeptic_ran", "skeptic_confirms", "contaminant", "rationale_visual"]].copy()
    vis["row_id"] = vis["row_id"].astype(str)

    lean = pd.read_parquet(args.lj_lean)[["name", "grade_pred", "p_lens", "contaminant",
                                          "rationale"]].rename(columns={"name": "row_id"})
    lean["row_id"] = lean["row_id"].astype(str)
    lean = lean.rename(columns={"grade_pred": "lj_lean_grade", "p_lens": "lj_lean_p_lens",
                                "contaminant": "lj_lean_contaminant",
                                "rationale": "lj_lean_rationale"})
    df = man.merge(vis, on="row_id", how="left").merge(lean, on="row_id", how="left")

    multi_path = Path(args.lj_multi)
    if multi_path.exists():
        multi = pd.read_parquet(multi_path)[["name", "grade_pred", "p_lens", "rationale"]]
        multi = multi.rename(columns={"name": "row_id", "grade_pred": "lj_multi_grade",
                                      "p_lens": "lj_multi_p_lens",
                                      "rationale": "lj_multi_rationale"})
        multi["row_id"] = multi["row_id"].astype(str)
        df = df.merge(multi, on="row_id", how="left")
    else:
        df["lj_multi_grade"] = pd.NA
        df["lj_multi_p_lens"] = np.nan
        df["lj_multi_rationale"] = pd.NA

    # lensjudge final grade = multiagent if present else lean
    df["lensjudge_grade"] = df["lj_multi_grade"].where(df["lj_multi_grade"].notna(),
                                                       df["lj_lean_grade"])
    df["lensjudge_p_lens"] = df["lj_multi_p_lens"].where(df["lj_multi_p_lens"].notna(),
                                                         df["lj_lean_p_lens"])
    df["lensjudge_rationale"] = df["lj_multi_rationale"].where(
        df["lj_multi_rationale"].notna(), df["lj_lean_rationale"])

    def ab(g):
        return isinstance(g, str) and g in ("A", "B")

    df["my_ab"] = df["my_grade"].apply(ab)
    df["lj_ab"] = df["lensjudge_grade"].apply(ab)
    df["still_new"] = (df["status"] == "NEW")
    df["qualified"] = df["still_new"] & df["my_ab"] & df["lj_ab"]

    def tier(r):
        if not r["qualified"]:
            return "none"
        a = (r["my_grade"] == "A") + (r["lensjudge_grade"] == "A")
        if a == 2:
            return "gold"
        if a >= 1:
            return "silver"
        return "bronze"

    df["tier"] = df.apply(tier, axis=1)
    df["agree"] = df["my_grade"] == df["lensjudge_grade"]
    # escalation: exactly one grader says >=B, still new, not qualified
    df["escalation"] = df["still_new"] & (df["my_ab"] ^ df["lj_ab"])

    def disq(r):
        if r["qualified"]:
            return ""
        if not r["still_new"]:
            return f"not-new ({r['status']})"
        if not r["my_ab"] and not r["lj_ab"]:
            return "both<B"
        return "single-grader (escalation)"

    df["disqualify_reason"] = df.apply(disq, axis=1)

    rankcol = (df["my_p_lens"].fillna(0) + df["lensjudge_p_lens"].fillna(0)) / 2
    df["consensus_p"] = rankcol

    full_cols = ["row_id", "RA", "DEC", "p_final", "q_group", "brick", "status",
                 "nearest_catalog", "nearest_name", "nearest_sep_arcsec",
                 "my_grade", "first_grade", "skeptic_confirms", "my_p_lens",
                 "lensjudge_grade", "lj_lean_grade", "lj_multi_grade", "lensjudge_p_lens",
                 "consensus_p", "qualified", "tier", "agree", "escalation",
                 "disqualify_reason", "rationale_visual", "lensjudge_rationale"]
    full_cols = [c for c in full_cols if c in df.columns]
    full = df[full_cols].sort_values(["qualified", "tier", "consensus_p"],
                                     ascending=[False, True, False])
    full.to_parquet(OUT / "consensus_full_737.parquet", index=False)

    qual = full[full.qualified].copy()
    tier_order = {"gold": 0, "silver": 1, "bronze": 2}
    qual["__t"] = qual.tier.map(tier_order)
    qual = qual.sort_values(["__t", "consensus_p"], ascending=[True, False]).drop(columns="__t")
    qual.insert(0, "rank", range(1, len(qual) + 1))
    qual.to_parquet(OUT / "candidates_qualified.parquet", index=False)
    qual.to_csv(OUT / "candidates_qualified.csv", index=False)

    print(f"[240] {len(df)} candidates; still-NEW after expanded xmatch: "
          f"{int(df.still_new.sum())}; KNOWN now: {int((~df.still_new).sum())}")
    print(f"[240] my A/B: {int(df.my_ab.sum())} | lensjudge A/B: {int(df.lj_ab.sum())} | "
          f"agree(grade): {int(df.agree.sum())}")
    print(f"[240] QUALIFIED (NEW & both A/B): {int(df.qualified.sum())} "
          f"-> gold {int((df.tier=='gold').sum())}, silver {int((df.tier=='silver').sum())}, "
          f"bronze {int((df.tier=='bronze').sum())}")
    print(f"[240] escalation (single-grader >=B, still NEW): {int(df.escalation.sum())}")
    print(f"[240] wrote candidates_qualified.parquet/csv ({len(qual)}) + consensus_full_737.parquet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
