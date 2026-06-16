#!/usr/bin/env python3
"""365_d4_edf_select.py — ClaudeNet v3 D4: region-local v3 selection over the Euclid EDF-F/S sky.

D1 escalated only the v3 SURVIVORS (global 150k cap) that overlap Euclid. D4 re-scores the FULL
EDF-F/S parent region (127k galaxies, no global cap; 8 members via 315 --tag edf) and ranks it
region-locally with v3blend8 + the A3 head — so it recovers Euclid lenses that fell below the
global stage-1 threshold. Then:
  * v3 recall of Euclid Q1 lenses in the region BY grade (region-local top-K vs the global 6% of D1);
  * precision@K of v3's top candidates against the Euclid expert truth set;
  * the escalate-gradeable shortlist: v3 top candidates with a staged Euclid cutout (+NEW status)
    -> manifest for run_batch --mode escalate (tier-2 Euclid 0.1" confirmation).

Ranks by v3blend8 (tail-robust; A3 showed the head overfits the strict tail) and reports the head too.

    /home2/benson/.venvs/claudenet/bin/python 365_d4_edf_select.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u

import _clib as C
h321 = C._load("cn_365_321", C.ROOT / "321_lensvmimic_head.py")

V3 = C.DATA / "v3"
REPRO = C.ROOT.parent
EUCLID_CAT = REPRO / "euclid-q1" / "data" / "raw" / "q1_discovery_engine_lens_catalog.csv"
FEATS = h321.FEATS


def _lg(x):
    x = np.clip(np.asarray(x, float), 1e-6, 1 - 1e-6)
    return np.log(x / (1 - x))


def main() -> int:
    sys.path.insert(0, str(REPRO))
    from lensjudge.common import euclid
    from sklearn.linear_model import LogisticRegression
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--topn", type=int, default=60, help="escalate shortlist size")
    ap.add_argument("--out", default=str(V3 / "cv3_edf_candidates.csv"))
    args = ap.parse_args()

    # --- fit the A3 head (lenses vs DR10 mimic bank) ---
    Xpos = np.vstack([h321.lens_feats(sp) for sp in h321.POS]); Xm = h321.dr10_feats()
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0).fit(
        _lg(np.vstack([Xpos, Xm])), np.r_[np.ones(len(Xpos)), np.zeros(len(Xm))])

    # --- 8-member edf scores -> v3blend8 + head ---
    base = pd.read_parquet(V3 / "cv3_edf_parent_ids.parquet").astype({"row_id": str})
    df = base[["row_id", "RA", "DEC"]].copy()
    for m in FEATS:
        s = pd.read_parquet(V3 / f"scores_member_{m}_edf.parquet")[["row_id", "pc"]]
        df = df.merge(s.rename(columns={"pc": m}).astype({"row_id": str}), on="row_id", how="inner")
    X = df[FEATS].to_numpy(float)
    df["v3blend8"] = X.mean(axis=1)
    df["head"] = clf.decision_function(_lg(X))
    print(f"[365] scored EDF region: {len(df):,} galaxies (8 members)")

    # --- cross-match to Euclid Q1 (grade truth) + to known catalogs ---
    q = pd.read_csv(EUCLID_CAT).astype({"id_str": str})
    cq = SkyCoord(q.right_ascension.values * u.deg, q.declination.values * u.deg)
    cdf = SkyCoord(df.RA.values * u.deg, df.DEC.values * u.deg)
    iq, sepq, _ = cdf.match_to_catalog_sky(cq)
    df["euclid_grade"] = np.where(sepq.arcsec < 3, q.grade.values[iq], None)
    df["euclid_id"] = np.where(sepq.arcsec < 2, q.id_str.values[iq], None)
    known = np.zeros(len(df), bool)
    for cat, r in [("storfer2024", 5), ("inchausti2025", 5), ("huang2021", 5)]:
        kc = pd.read_csv(C.DATA / f"{cat}_published_catalog.csv")
        ck = SkyCoord(kc.RA.values * u.deg, kc.DEC.values * u.deg)
        _, sk, _ = cdf.match_to_catalog_sky(ck); known |= sk.arcsec < r
    df["known"] = known

    # --- region-local recall of Euclid A/B lenses (vs D1's global-survivor 6.2%) ---
    rep = {"n_region": len(df), "recall": {}}
    df["v3b_pct"] = df.v3blend8.rank(pct=True)
    print("\n[365] region-local v3 recall of Euclid Q1 lenses (rank by v3blend8):")
    for g in ["A", "B", "C"]:
        sub = df[df.euclid_grade == g]
        if not len(sub):
            continue
        topk = {k: int((sub.v3blend8 >= df.v3blend8.nlargest(k).min()).sum()) for k in (100, 500, 1000)}
        rep["recall"][g] = {"n_in_region": len(sub), "top100": topk[100], "top500": topk[500],
                            "top1000": topk[1000], "median_pct": float(sub.v3b_pct.median())}
        print(f"  grade {g}: n={len(sub):4d} in region | in v3 top-100/500/1000 = "
              f"{topk[100]}/{topk[500]}/{topk[1000]} | median percentile {sub.v3b_pct.median():.3f}")

    # --- precision@K of v3 top candidates vs Euclid truth ---
    print("\n[365] precision@K of v3blend8-ranked candidates (Euclid A/B among top-K):")
    for k in (30, 60, 100):
        top = df.nlargest(k, "v3blend8")
        ab = int(top.euclid_grade.isin(["A", "B"]).sum())
        rep.setdefault("precision", {})[str(k)] = {"euclid_AB": ab, "euclid_any": int(top.euclid_grade.notna().sum())}
        print(f"  top-{k}: Euclid-A/B {ab}  (any Euclid grade {int(top.euclid_grade.notna().sum())})")

    # --- escalate shortlist: top v3 candidates with a staged Euclid cutout ---
    df["euclid_fits"] = [euclid.obj_dir(str(e)) is not None if e else False for e in df.euclid_id]
    cand = df.sort_values("v3blend8", ascending=False)
    short = cand[cand.euclid_fits].head(args.topn).copy()
    qg = q.set_index("id_str")["grade"]
    short["grade_truth"] = short.euclid_id.map(qg)
    man = short.rename(columns={"RA": "ra", "DEC": "dec"})[["row_id", "ra", "dec", "grade_truth", "euclid_id", "known"]]
    man.insert(0, "name", man.pop("row_id")); man["catalog"] = "d4_edf"; man["region"] = "edf_fs"
    man.to_csv(V3 / "manifests_d4_edf.csv", index=False)
    cand.head(300)[["row_id", "RA", "DEC", "v3blend8", "head", "euclid_grade", "euclid_fits", "known"]].to_csv(args.out, index=False)
    rep["escalate_shortlist"] = {"n": len(man), "euclid_AB": int(short.grade_truth.isin(["A", "B"]).sum()),
                                 "new": int((~short.known).sum())}
    print(f"\n[365] escalate shortlist: {len(man)} v3-top candidates with Euclid cutouts "
          f"({int(short.grade_truth.isin(['A','B']).sum())} Euclid-A/B, {int((~short.known).sum())} NEW) "
          f"-> manifests_d4_edf.csv")
    (V3 / "cv3_edf_select_summary.json").write_text(json.dumps(rep, indent=2, default=str))
    print(f"[365] wrote {args.out} + cv3_edf_select_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
