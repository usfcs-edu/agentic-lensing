#!/usr/bin/env python3
"""make_cascade_manifest.py — build a LensJudge cascade manifest from a DR11 candidate CSV.

The cascade (lensjudge/eval/run_cascade.py) expects rows with: name, ra, dec, catalog
(=survey_key -> CUTOUT_DIRS lookup) and optionally grade_truth + p_meta (carried for the
active-learning hard-negative export). We map the 365/select CSV columns:
  row_id -> name, RA -> ra, DEC -> dec, v3blend8 -> p_meta; catalog := --survey.

    python make_cascade_manifest.py --in <cand.csv> --survey dr11 \
        --out <manifest.csv> [--limit N] [--ra-col RA --dec-col DEC --id-col row_id --pmeta-col v3blend8]
"""
import argparse
import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--survey", default="dr11", help="catalog/survey_key -> CUTOUT_DIRS key")
    ap.add_argument("--id-col", default="row_id")
    ap.add_argument("--ra-col", default="RA")
    ap.add_argument("--dec-col", default="DEC")
    ap.add_argument("--pmeta-col", default="v3blend8")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    df = pd.read_csv(args.inp)
    out = pd.DataFrame({
        "name": df[args.id_col].astype(str),
        "ra": df[args.ra_col].astype(float),
        "dec": df[args.dec_col].astype(float),
        "catalog": args.survey,
        "survey_key": args.survey,
    })
    if args.pmeta_col in df.columns:
        out["p_meta"] = df[args.pmeta_col].astype(float)
    if args.limit:
        out = out.head(args.limit)
    out.to_csv(args.out, index=False)
    print(f"[manifest] {len(out)} rows -> {args.out} (survey={args.survey})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
