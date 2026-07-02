#!/usr/bin/env python3
"""396_export_v4_new_candidates.py — export the v4 full-re-sweep candidate ranking to CSV.

The v4 fine-tuned re-sweep (395) writes survivors_dr11s_v4.parquet = the top-150k DR11-south
galaxies by the v4 ensemble mean (mean5), each with a rank (1..150000), RA/DEC, footprint, brick.
Of those, 15,922 overlap the old union-95k survivors (survivors_dr10_recal.parquet) and
**134,078 are NEW** (in the v4 top-150k but not in the union-95k set) — the number quoted in the
v4 report. This script emits those new galaxies as a CSV, carrying the rank.

  # on Perlmutter (where the parquets live), claudenet venv or plain python+pandas:
  python 396_export_v4_new_candidates.py                 # -> <BASE>/dr11s_v4_new_candidates.csv
  python 396_export_v4_new_candidates.py --full          # all 150k with an is_new flag
  python 396_export_v4_new_candidates.py --base /some/local/dir   # if you rsync'd the parquets

Provenance note: `mean5` is the RANKING score the top-150k was selected by (arithmetic mean of the
5 v4 members). It is NOT a calibrated P(lens) at the survey base rate — it orders candidates for
inspection; graded p_lens comes only from the LensJudge vetting stage (not run on this set yet).
"""
from __future__ import annotations
import argparse
import pandas as pd

BASE = "/pscratch/sd/g/gdbenson/claudenet/sweep_dr11/south"
COLS = ["rank", "rank_within_new", "row_id", "RA", "DEC", "mean5", "footprint", "brick"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=BASE, help="dir holding survivors_dr11s_v4.parquet + baselines")
    ap.add_argument("--v4", default=None, help="override path to survivors_dr11s_v4.parquet")
    ap.add_argument("--union95k", default=None,
                    help="override path to the union-95k set (default survivors_dr10_recal.parquet)")
    ap.add_argument("--out", default=None, help="output CSV path")
    ap.add_argument("--full", action="store_true",
                    help="export all 150k with an is_new column instead of only the new subset")
    a = ap.parse_args()

    v4_path = a.v4 or f"{a.base}/survivors_dr11s_v4.parquet"
    union_path = a.union95k or f"{a.base}/survivors_dr10_recal.parquet"

    v4 = pd.read_parquet(v4_path)
    v4["row_id"] = v4["row_id"].astype(str)
    if "rank" not in v4.columns:  # be robust if an older parquet lacks it
        v4 = v4.sort_values("mean5", ascending=False).reset_index(drop=True)
        v4["rank"] = range(1, len(v4) + 1)
    v4 = v4.sort_values("rank").reset_index(drop=True)

    union_ids = set(pd.read_parquet(union_path, columns=["row_id"]).row_id.astype(str))
    is_new = ~v4["row_id"].isin(union_ids)
    print(f"[396] v4 top-{len(v4):,}: {int(is_new.sum()):,} new, {int((~is_new).sum()):,} retained "
          f"(report says 134,078 new / 15,922 retained vs union-95k)")

    if a.full:
        out = v4.copy()
        out["is_new"] = is_new.astype(int)
        out["rank_within_new"] = out["is_new"].cumsum().where(is_new)  # rank among new only
        out_path = a.out or f"{a.base}/dr11s_v4_candidates_150k.csv"
        cols = ["rank", "is_new", "rank_within_new", "row_id", "RA", "DEC", "mean5", "footprint", "brick"]
    else:
        out = v4[is_new].copy().reset_index(drop=True)
        out["rank_within_new"] = range(1, len(out) + 1)  # 1..134078 among the new subset
        out_path = a.out or f"{a.base}/dr11s_v4_new_candidates.csv"
        cols = COLS

    out[[c for c in cols if c in out.columns]].to_csv(out_path, index=False)
    print(f"[396] wrote {out_path}: {len(out):,} rows, cols={[c for c in cols if c in out.columns]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
