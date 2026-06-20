#!/usr/bin/env python3
"""382_dr11s_reselect_mean.py — re-select DR11-south stage-1 survivors by the calibrated
5-lean MEAN combiner, replacing the per-member 1e-4 UNION (the recall fix established by 381).

Streams the 32 full-parent stage-1 parts on Perlmutter, applies the persisted v2-lean isotonic
calibrators, ranks the whole 53.8M parent by the 5-member mean, takes the top --budget, attaches
RA/DEC, and DIRECTLY measures recovered known-lens recall (end-to-end confirmation of 381's
NegEval-estimated 56%->82% Inchausti grade-A jump). Also reports retained/added/dropped vs the
old union survivor set. CPU only — the parent scores already exist; no retrain, no re-extraction.

Run on Perlmutter (module load python):
  cd ~/claudenet
  srun -A cosmo -C cpu -q debug -t 25 -N1 -n1 -c32 --mem=96G \
       python -u 382_dr11s_reselect_mean.py --budget 150000
Outputs -> <BASE>/survivors_dr11s_mean.parquet + reselect_mean_summary.json
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

import _clib as C

BASE = "/pscratch/sd/g/gdbenson/claudenet/sweep_dr11/south"
CATDIR = "/global/homes/g/gdbenson/claudenet/data"
FITS = "data/v2/ensemble_v2lean_fits"
MEMBERS = ["member_effnet_B", "member_effnet_B3_hard", "member_effnet_S2_hard",
           "member_resnet46_C_hard", "member_zoobot_N"]
OP_BUDGET = 95_104


def load_cals():
    M162 = C._load("cn382_162", C.ROOT / "162_stage2_rescore.py")
    return M162.load_calibrators(FITS)


def calmean(df, cals):
    cal = np.column_stack([cals[m].transform(np.nan_to_num(df[m].to_numpy(float), nan=0.0))
                           for m in MEMBERS])
    return cal.mean(axis=1)


def xyz(ra, dec):
    ra = np.radians(np.asarray(ra, float)); dec = np.radians(np.asarray(dec, float))
    return np.column_stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=150_000)
    ap.add_argument("--out", default=f"{BASE}/survivors_dr11s_mean.parquet")
    args = ap.parse_args()
    cals = load_cals()
    stage1 = sorted(glob.glob(f"{BASE}/stage1/stage1_part*_lean_*.parquet"))
    mani = sorted(glob.glob(f"{BASE}/sweep_manifest_part*of32.parquet"))
    assert len(stage1) == len(mani) == 32

    # ---- stream: per-part top-budget by calibrated mean ----
    keep = []
    total = 0
    for i, f in enumerate(stage1):
        s1 = pd.read_parquet(f, columns=["row_id", "ok"] + MEMBERS)
        m5 = calmean(s1, cals)
        sub = pd.DataFrame({"row_id": s1.row_id.values, "mean5": m5, "ok": s1.ok.values})
        keep.append(sub.nlargest(args.budget, "mean5"))
        total += len(s1)
        print(f"[382] part {i:02d}/32 rows={len(s1):,} cum={total:,}", flush=True)
    allk = pd.concat(keep, ignore_index=True)
    sel = allk.nlargest(args.budget, "mean5").reset_index(drop=True)
    sel["rank"] = np.arange(1, len(sel) + 1)
    thr_mean = float(sel.mean5.min())
    thr_95k = float(sel.mean5.nlargest(OP_BUDGET).min())
    print(f"[382] parent={total:,}  selected top-{args.budget:,}  thr_mean={thr_mean:.6f}  "
          f"thr@95k={thr_95k:.6f}")

    # ---- attach RA/DEC (filter manifests to the selected row_ids) ----
    selset = set(sel.row_id)
    mrows = []
    for p in mani:
        m = pd.read_parquet(p, columns=["row_id", "RA", "DEC", "footprint", "brick"])
        mrows.append(m[m.row_id.isin(selset)])
    man_sel = pd.concat(mrows, ignore_index=True)
    sel = sel.merge(man_sel, on="row_id", how="left")
    sel.to_parquet(args.out, index=False)

    # ---- direct recovered-recall on known lenses (reuse 380 extract) ----
    k = pd.read_parquet(f"{BASE}/diag/dr11s_diag_knownlens.parquet")
    k["mean5"] = calmean(k, cals)
    cur = k[k.src == "curated"]
    tr = cKDTree(xyz(cur.ra, cur.dec))
    d, _ = tr.query(xyz(k.ra, k.dec), k=1)
    k["leak"] = (np.degrees(2 * np.arcsin(np.clip(d / 2, 0, 1))) * 3600) < 5.0

    rep = {"budget": args.budget, "parent": total, "thr_mean": thr_mean, "thr_95k": thr_95k,
           "recall": {}}

    def recall_block(mask, label):
        sub = k[mask]
        inp = int(sub.in_parent.sum())
        rec150 = int(((sub.mean5 >= thr_mean) & sub.in_parent).sum())
        rec95 = int(((sub.mean5 >= thr_95k) & sub.in_parent).sum())
        held = sub[~sub.leak]
        inp_h = int(held.in_parent.sum())
        rec150_h = int(((held.mean5 >= thr_mean) & held.in_parent).sum())
        d = dict(n=len(sub), in_parent=inp,
                 recovered_mean_budget=rec150, recall_inparent_budget=rec150 / inp if inp else None,
                 recovered_mean_95k=rec95, recall_inparent_95k=rec95 / inp if inp else None,
                 heldout_in_parent=inp_h, heldout_recovered_budget=rec150_h,
                 heldout_recall_budget=rec150_h / inp_h if inp_h else None)
        rep["recall"][label] = d
        print(f"[382] {label:14s} in-parent={inp:4d}  recovered@95k={rec95:4d} "
              f"({(rec95/inp if inp else 0):.3f})  recovered@{args.budget//1000}k={rec150:4d} "
              f"({(rec150/inp if inp else 0):.3f})  heldout@{args.budget//1000}k={rec150_h}/{inp_h}")
        return d

    recall_block((k.src == "inchausti") & (k.grade == "A"), "inchausti_A")
    recall_block((k.src == "inchausti") & (k.grade == "B"), "inchausti_B")
    recall_block((k.src == "storfer") & (k.grade == "A"), "storfer_A")

    # ---- vs the old union survivor set ----
    old = set(pd.read_parquet(f"{BASE}/survivors_dr10_recal.parquet").row_id.astype(str))
    new = set(sel.row_id.astype(str))
    rep["vs_union"] = dict(n_union=len(old), n_mean=len(new),
                           retained=len(old & new), added=len(new - old), dropped=len(old - new))
    print(f"[382] vs union: union={len(old):,} mean={len(new):,} "
          f"retained={len(old & new):,} added={len(new - old):,} dropped={len(old - new):,}")

    json.dump(rep, open(f"{BASE}/reselect_mean_summary.json", "w"), indent=2, default=float)
    print(f"[382] wrote {args.out} + reselect_mean_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
