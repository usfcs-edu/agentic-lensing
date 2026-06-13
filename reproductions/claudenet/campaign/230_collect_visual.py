#!/usr/bin/env python3
"""230_collect_visual.py — turn the visual-judging Workflow's JSON result into
visual_grades_verified.parquet. Applies the skeptic gate: a candidate's final
my_grade holds at its first-pass value only if the skeptic confirmed it stays
>=B; otherwise it takes the skeptic's (lower) grade.

    /home2/benson/.venvs/claudenet/bin/python campaign/230_collect_visual.py \
        --result data/v2/campaign/visual_workflow_result.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "v2" / "campaign"
RANK = {"A": 3, "B": 2, "C": 1, "D": 0}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", default=str(OUT / "visual_workflow_result.json"))
    args = ap.parse_args()
    res = json.load(open(args.result))
    grades = pd.DataFrame(res["grades"])
    grades["row_id"] = grades["row_id"].astype(str)
    # explode criteria dict into columns
    crit = pd.json_normalize(grades["criteria"]).add_prefix("crit_")
    g = pd.concat([grades.drop(columns=["criteria"]), crit], axis=1)
    g = g.rename(columns={"grade": "first_grade", "p_lens": "my_p_lens",
                          "confidence": "my_confidence", "rationale": "rationale_visual"})

    skept = pd.DataFrame(res.get("skeptic", []))
    if len(skept):
        skept["row_id"] = skept["row_id"].astype(str)
        g = g.merge(skept, on="row_id", how="left")
    else:
        for c in ("skeptic_grade", "skeptic_confirms", "skeptic_contaminant",
                  "skeptic_rationale"):
            g[c] = pd.NA

    def final_grade(r):
        fg = r["first_grade"]
        if fg in ("A", "B"):
            if r.get("skeptic_confirms") is True:
                # confirmed: keep the MIN of first and skeptic grade (skeptic can downgrade A->B)
                sg = r.get("skeptic_grade")
                if isinstance(sg, str) and RANK.get(sg, 3) < RANK[fg]:
                    return sg
                return fg
            # not confirmed (or skeptic missing): take skeptic grade if present, else demote to C
            sg = r.get("skeptic_grade")
            return sg if isinstance(sg, str) else "C"
        return fg

    g["my_grade"] = g.apply(final_grade, axis=1)
    g["skeptic_ran"] = g["first_grade"].isin(["A", "B"])
    n = len(g)
    out = OUT / "visual_grades_verified.parquet"
    g.to_parquet(out, index=False)
    fa = (g.first_grade.value_counts().to_dict())
    fb = (g.my_grade.value_counts().to_dict())
    print(f"[230] {n} graded. first-pass {fa}; post-skeptic {fb}")
    print(f"[230] A/B after skeptic: {int(g.my_grade.isin(['A','B']).sum())} "
          f"(first-pass A/B was {int(g.first_grade.isin(['A','B']).sum())})")
    print(f"[230] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
