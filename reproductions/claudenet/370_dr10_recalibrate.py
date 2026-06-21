#!/usr/bin/env python3
"""370_dr10_recalibrate.py — ClaudeNet v3 C-now: recalibrate the v2-lean operating point
into DR10 score-space, using the ALREADY-SCORED full sweep (no new extraction).

The head-to-head showed v2-lean *finds* ~88% of Inchausti's DR10 grade-A at high score, but the
DR9-calibrated per-member 1e-4 thresholds, transferred to DR10, are too strict — ~17 high-scoring
grade-A fall below the union threshold (recall 69% at the operating point). Because stage-1 scored
ALL 43.7M parent galaxies, the DR10 random-galaxy null is just a random sample of those scores.
This script:
  1. applies the persisted v2-lean isotonic calibrators to the full population (raw -> calibrated);
  2. draws a ~N-row random NegEval, computes per-member (1-1e-4) thresholds in DR10 calibrated
     space, writes operating_points_dr10.csv (162 schema) + negeval_dr10_combined.parquet (the
     v2lean_average conformal calibration for 165, ~N rows -> better full-m power than the 1M DR9);
  3. re-thresholds the full population (union rule, DR10 thresholds) -> recalibrated survivors;
  4. re-measures recall of the published 811 by grade (DR9 vs DR10 thresholds).

    /home2/benson/.venvs/claudenet/bin/python 370_dr10_recalibrate.py --negeval 6000000
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u

import _clib as C
import importlib.util


def _load_162():
    spec = importlib.util.spec_from_file_location("cn162", C.ROOT / "162_stage2_rescore.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-dir", default="data/v3/sweep_dr10")
    ap.add_argument("--fits", default="data/v2/ensemble_v2lean_fits")
    ap.add_argument("--dr9-op", default="data/v2/ensemble_v2lean_operating_points.csv")
    ap.add_argument("--negeval", type=int, default=6_000_000, help="NegEval sample size")
    ap.add_argument("--fpr", type=float, default=1e-4)
    ap.add_argument("--budget", type=int, default=150_000)
    ap.add_argument("--combiner", choices=("mean", "union"), default="mean",
                    help="survivor selector: 'mean' (DEFAULT) = top --budget by the "
                         "calibrated v2lean mean (381 recall fix, no threshold "
                         "recalibration needed); 'union' = legacy per-member 1e-4 union")
    ap.add_argument("--seed", type=int, default=C.SEED)
    args = ap.parse_args()
    sd = Path(args.sweep_dir)
    M162 = _load_162()
    cals = M162.load_calibrators(args.fits)               # {member_col: isotonic calibrator}
    members = ["member_effnet_B", "member_effnet_B3_hard", "member_effnet_S2_hard",
               "member_resnet46_C_hard", "member_zoobot_N"]

    # 1. load full scored population (raw members) + apply calibrators -> calibrated + v2lean_average
    print("[370] loading full stage-1 scores (43.7M) ...")
    df = pd.concat([pd.read_parquet(f, columns=["row_id"] + members)
                    for f in sorted(glob.glob(f"{sd}/stage1/stage1_*.parquet"))],
                   ignore_index=True)
    print(f"[370] {len(df):,} rows; applying calibrators")
    cal = np.empty((len(df), len(members)), np.float64)
    for j, m in enumerate(members):
        raw = np.nan_to_num(df[m].to_numpy(np.float64), nan=0.0)  # 8 failed cutouts -> 0
        cal[:, j] = cals[m].transform(raw)
    df_avg = cal.mean(axis=1)                              # v2lean_average over ALL 43.7M

    # 2. NegEval sample -> per-member 1e-4 thresholds (DR10 calibrated space)
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(df), size=min(args.negeval, len(df)), replace=False)
    thr_dr10 = {m: float(np.quantile(cal[idx, j], 1.0 - args.fpr)) for j, m in enumerate(members)}
    dr9 = pd.read_csv(args.dr9_op)
    dr9 = dr9[(dr9.fpr == args.fpr)].set_index("scorer")["thr"]
    print(f"\n[370] per-member 1e-4 thresholds (DR9 -> DR10 recalibrated):")
    rows_op = []
    for m in members:
        s = m.replace("member_", "")
        old = float(dr9.get(s, dr9.get(m, np.nan)))
        print(f"   {s:18s} DR9={old:.4f}  DR10={thr_dr10[m]:.4f}  ({'looser' if thr_dr10[m]<old else 'tighter'})")
        rows_op.append({"scorer": s, "kind": "member", "roster": "v2lean", "fpr": args.fpr,
                        "thr": thr_dr10[m], "thr_lo": np.nan, "thr_hi": np.nan})
    pd.DataFrame(rows_op).to_csv(sd / "operating_points_dr10.csv", index=False)

    # 3. select survivors. --combiner 'mean' (DEFAULT, the 381 recall fix) ranks by the
    #    calibrated v2lean MEAN (df_avg) and keeps the top --budget — the per-member union
    #    loses ~20pt grade-A recall on the deeper DR11 (a member saturates) and the mean
    #    needs no per-release threshold recalibration at all. 'union' = legacy rule.
    margin = np.full(len(df), -np.inf)
    for j, m in enumerate(members):
        margin = np.maximum(margin, cal[:, j] - thr_dr10[m])
    union_pass = (cal >= np.array([thr_dr10[m] for m in members])).any(axis=1)
    n_pass = int(union_pass.sum())              # union pass count (informational under 'mean')
    if args.combiner == "union":
        sel = np.flatnonzero(union_pass)
        print(f"\n[370] DR10-recalibrated stage-1 pass (union): "
              f"{n_pass:,}/{len(df):,} = {n_pass/len(df):.3e}")
        if len(sel) > args.budget:
            sel = sel[np.argsort(-margin[sel], kind="stable")[:args.budget]]
    else:                                       # mean: top-budget by calibrated mean
        sel = np.argsort(-df_avg, kind="stable")[:args.budget]
        print(f"\n[370] mean selector: top-{min(args.budget, len(df)):,} by calibrated "
              f"mean (cutoff {df_avg[sel[-1]]:.6f}); the per-member union would pass "
              f"{int(union_pass.sum()):,}")
    surv = df.iloc[sel][["row_id"]].copy()
    surv["p_final"] = df_avg[sel]
    surv["stage1_margin"] = margin[sel]
    surv["stage1_mean"] = df_avg[sel]
    print(f"[370] survivors ({args.combiner}, budget {args.budget:,}): {len(surv):,}")

    # join RA/DEC from manifests for recall + downstream
    man = pd.concat([pd.read_parquet(p, columns=["row_id", "RA", "DEC", "footprint", "brick"])
                     for p in glob.glob(f"{sd}/sweep_manifest_part*.parquet")], ignore_index=True)
    surv = surv.merge(man, on="row_id", how="left")
    surv.to_parquet(sd / "survivors_dr10_recal.parquet", index=False)

    # NegEval combined (conformal calibration for 165): row_id + v2lean_average + footprint
    neg = df.iloc[idx][["row_id"]].copy(); neg["v2lean_average"] = df_avg[idx]
    neg = neg.merge(man[["row_id", "footprint"]], on="row_id", how="left")
    neg.to_parquet(sd / "negeval_dr10_combined.parquet", index=False)

    # 4. recall of the 811 by grade (DR9 survivors vs DR10-recal survivors)
    cat = pd.read_csv(C.DATA / "inchausti2025_published_catalog.csv")
    c811 = SkyCoord(cat.RA.values * u.deg, cat.DEC.values * u.deg)
    csurv = SkyCoord(surv.RA.values * u.deg, surv.DEC.values * u.deg)
    _, d, _ = c811.match_to_catalog_sky(csurv)
    cat["rec"] = d.arcsec < 5
    print("\n[370] recall of the published 811 by grade (DR10-recalibrated survivors):")
    rec = {}
    for g in ["A", "B", "C"]:
        s = cat[cat.grade == g]
        rec[g] = [int(s.rec.sum()), len(s)]
        print(f"   grade {g}: {int(s.rec.sum())}/{len(s)} ({s.rec.mean():.2f})")
    print(f"   ALL: {int(cat.rec.sum())}/{len(cat)} ({cat.rec.mean():.2f})")

    json.dump({"combiner": args.combiner, "n_union_pass": n_pass,
               "n_survivors": int(len(surv)),
               "thr_dr10": {m.replace('member_', ''): thr_dr10[m] for m in members},
               "recall_811_by_grade": rec, "recall_811_all": [int(cat.rec.sum()), len(cat)]},
              open(sd / "recalibrate_dr10_summary.json", "w"), indent=2)
    print(f"\n[370] wrote operating_points_dr10.csv, survivors_dr10_recal.parquet, "
          f"negeval_dr10_combined.parquet, recalibrate_dr10_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
