#!/usr/bin/env python3
"""380_dr11s_recall_diag_extract.py — recall-recoverability diagnosis, Perlmutter side.

The DR11-south sweep reports grade-A recall-of-811 = 41% (37/90) at the 1e-4 operating
point, vs 62% in-domain on DR10 — a genuine DR9->DR11 domain shift (the members were only
SCORED on DR11, never trained on DR11 pixels). But 370/163 measure recall by positionally
crossmatching the FULL published catalogs to the *survivors*, with no coverage restriction,
so a lost lens has no recorded score. To decide how much of the gap is recoverable WITHOUT
retraining (re-threshold / drop the saturated resnet member / better combiner) vs needs a
DR11-native fine-tune, we need the per-member scores of the lenses that were CUT.

This streams the 32 full-parent stage-1 parts (low memory) and, for every known lens, keeps
the nearest parent galaxy's RAW 5-member scores + match separation + the parent ok flag.
It also dumps a random per-member NegEval sample (the DR11 random-galaxy null) for the
threshold-free AUC. All calibration + analysis happens locally (381) with the persisted
isotonic fits; this side only needs raw scores.

Run on Perlmutter (module load python):
  srun -A cosmo -C cpu -q shared -t 20 -n1 -c16 --mem=32G \
       python 380_dr11s_recall_diag_extract.py --neg 2000000
Outputs (small) -> <outdir>/dr11s_diag_knownlens.parquet + dr11s_diag_negeval.parquet
"""
from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

BASE = "/pscratch/sd/g/gdbenson/claudenet/sweep_dr11/south"
CATDIR = os.path.expanduser("~/claudenet/data")
MEMBERS = ["member_effnet_B", "member_effnet_B3_hard", "member_effnet_S2_hard",
           "member_resnet46_C_hard", "member_zoobot_N"]


def radec_to_xyz(ra, dec):
    r = np.radians(np.asarray(ra, float)); d = np.radians(np.asarray(dec, float))
    return np.column_stack([np.cos(d) * np.cos(r), np.cos(d) * np.sin(r), np.sin(d)])


def chord_to_arcsec(chord):
    return np.degrees(2.0 * np.arcsin(np.clip(chord / 2.0, 0.0, 1.0))) * 3600.0


def load_known():
    frames = []
    inc = pd.read_csv(f"{CATDIR}/inchausti2025_published_catalog.csv")
    frames.append(pd.DataFrame(dict(name=inc.name.astype(str), ra=inc.RA, dec=inc.DEC,
                                    grade=inc.grade.astype(str), src="inchausti",
                                    tractor_type=inc.get("tractor_type", "").astype(str))))
    sto = pd.read_csv(f"{CATDIR}/storfer2024_published_catalog.csv")
    frames.append(pd.DataFrame(dict(name=sto.name.astype(str), ra=sto.RA, dec=sto.DEC,
                                    grade=sto.grade.astype(str), src="storfer",
                                    tractor_type=sto.get("tractor_type", "").astype(str))))
    cur = pd.read_parquet(f"{CATDIR}/positives_curated.parquet")
    frames.append(pd.DataFrame(dict(name=cur.name.astype(str), ra=cur.RA, dec=cur.DEC,
                                    grade="train:" + cur.get("grade", "?").astype(str),
                                    src="curated", tractor_type="")))
    k = pd.concat(frames, ignore_index=True).dropna(subset=["ra", "dec"]).reset_index(drop=True)
    print(f"[380] known catalogs: {len(k)} entries "
          f"({k.src.value_counts().to_dict()})")
    return k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--neg", type=int, default=2_000_000, help="total NegEval sample size")
    ap.add_argument("--outdir", default=f"{BASE}/diag")
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    stage1 = sorted(glob.glob(f"{BASE}/stage1/stage1_part*_lean_*.parquet"))
    mani = sorted(glob.glob(f"{BASE}/sweep_manifest_part*of32.parquet"))
    assert len(stage1) == len(mani) == 32, f"parts: {len(stage1)} stage1, {len(mani)} manifests"

    k = load_known()
    kxyz = radec_to_xyz(k.ra.values, k.dec.values)
    n = len(k)
    best_sep = np.full(n, np.inf)
    best_rowid = np.array([""] * n, dtype=object)
    best_ok = np.zeros(n, bool)
    best_ra = np.full(n, np.nan); best_dec = np.full(n, np.nan)
    best_fp = np.array([""] * n, dtype=object)
    best_m = {c: np.full(n, np.nan) for c in MEMBERS}

    rng = np.random.default_rng(args.seed)
    neg_per = int(np.ceil(args.neg / len(stage1)))
    neg_chunks = []
    total = 0

    for i, (s1f, mf) in enumerate(zip(stage1, mani)):
        s1 = pd.read_parquet(s1f, columns=["row_id", "ok"] + MEMBERS)
        mn = pd.read_parquet(mf, columns=["row_id", "RA", "DEC", "footprint"])
        df = mn.merge(s1, on="row_id", how="inner")
        total += len(df)
        pxyz = radec_to_xyz(df.RA.values, df.DEC.values)
        tree = cKDTree(pxyz)
        dist, idx = tree.query(kxyz, k=1)
        sep = chord_to_arcsec(dist)
        upd = sep < best_sep
        if upd.any():
            ui = np.flatnonzero(upd); pi = idx[ui]
            best_sep[ui] = sep[ui]
            best_rowid[ui] = df.row_id.values[pi]
            best_ok[ui] = df.ok.values[pi]
            best_ra[ui] = df.RA.values[pi]; best_dec[ui] = df.DEC.values[pi]
            best_fp[ui] = df.footprint.values[pi]
            for c in MEMBERS:
                best_m[c][ui] = df[c].values[pi]
        ns = min(neg_per, len(df))
        si = rng.choice(len(df), size=ns, replace=False)
        neg_chunks.append(df.iloc[si][["row_id", "footprint", "ok"] + MEMBERS].copy())
        print(f"[380] part {i:02d}/32  rows={len(df):,}  cum={total:,}  "
              f"known<5\"={int((best_sep < 5).sum())}/{n}", flush=True)

    known = k.copy()
    known["match_sep_arcsec"] = best_sep
    known["match_rowid"] = best_rowid.astype(str)
    known["match_ok"] = best_ok
    known["match_ra"] = best_ra; known["match_dec"] = best_dec
    known["match_footprint"] = best_fp.astype(str)
    known["in_parent"] = best_sep < 5.0
    for c in MEMBERS:
        known[c] = best_m[c]
    known.to_parquet(outdir / "dr11s_diag_knownlens.parquet", index=False)

    neg = pd.concat(neg_chunks, ignore_index=True)
    if len(neg) > args.neg:
        neg = neg.sample(n=args.neg, random_state=args.seed).reset_index(drop=True)
    neg.to_parquet(outdir / "dr11s_diag_negeval.parquet", index=False)

    print(f"\n[380] parent rows scanned: {total:,}")
    print(f"[380] known in-parent (<5\"): {int(known.in_parent.sum())}/{n}")
    for s in ["inchausti", "storfer", "curated"]:
        sub = known[known.src == s]
        print(f"   {s:10s} in-parent {int(sub.in_parent.sum())}/{len(sub)}")
    print(f"[380] NegEval rows: {len(neg):,}")
    print(f"[380] wrote {outdir}/dr11s_diag_knownlens.parquet + dr11s_diag_negeval.parquet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
