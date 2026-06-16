#!/usr/bin/env python3
"""364_euclid_overlap.py — ClaudeNet v3 D1: Euclid Q1 × DR10 overlap + v3 recall of Euclid lenses.

The C-vet conclusion: DR10 discovery is resolution-limited, and the lever is higher-res vetting
where Euclid overlaps. This quantifies that overlap against the Euclid Q1 "Strong Lensing Discovery
Engine" expert-graded catalog (reproductions/euclid-q1/data/raw/), and uses Euclid as an INDEPENDENT
benchmark for v3 (stronger than the held-out Storfer/Inchausti):

  1. Euclid Q1 lens candidates in DR10-south (EDF-F + EDF-S; EDF-N is north), by expert grade A/B/C.
  2. v3 recall by grade: fraction in the DESI parent (43.7M) and in the v3 survivor pool (150k) —
     reveals grade-selectivity (does v3 preferentially recover the REAL/grade-A lenses?) and the
     DESI depth/resolution ceiling (how many Euclid lenses DESI can't even detect).
  3. The cross-validated set: v3 survivors ∩ Euclid lens candidates (NEW vs known; Euclid cutouts
     staged) → the escalate-gradeable candidates (315/run_batch --mode escalate, tier-2 Euclid 0.1").

Writes data/v3/euclid_overlap_summary.json + data/v3/cv3_euclid_xmatch.csv (the cross-matches).

    /home2/benson/.venvs/claudenet/bin/python 364_euclid_overlap.py
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

V3 = C.DATA / "v3"
SD = V3 / "sweep_dr10"
REPRO = C.ROOT.parent
EUCLID_CAT = REPRO / "euclid-q1" / "data" / "raw" / "q1_discovery_engine_lens_catalog.csv"


def _match(cat_coord, ref_ra, ref_dec, radius=3.0):
    ref = SkyCoord(ref_ra * u.deg, ref_dec * u.deg)
    idx, sep, _ = cat_coord.match_to_catalog_sky(ref)
    return idx, sep.arcsec < radius, sep.arcsec


def main() -> int:
    sys.path.insert(0, str(REPRO))
    from lensjudge.common import euclid
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dec-max", type=float, default=-20.0, help="DR10-south EDF-F/S Dec cut")
    ap.add_argument("--radius", type=float, default=3.0, help="match radius arcsec")
    args = ap.parse_args()

    q = pd.read_csv(EUCLID_CAT).astype({"id_str": str})
    qS = q[q["declination"] < args.dec_max].reset_index(drop=True)
    cq = SkyCoord(qS["right_ascension"].values * u.deg, qS["declination"].values * u.deg)

    par = pd.read_parquet(SD / "parent_dr10_south.parquet", columns=["RA", "DEC"])
    s2 = pd.read_parquet(SD / "stage2_scores.parquet", columns=["row_id", "RA", "DEC"]).astype({"row_id": str})
    xm = pd.read_parquet(SD / "crossmatch.parquet",
                         columns=["row_id", "match_storfer2024", "match_inchausti2025", "match_huang2021"]).astype({"row_id": str})
    s2 = s2.merge(xm, on="row_id", how="left")

    _, in_par, _ = _match(cq, par["RA"].values, par["DEC"].values, args.radius)
    sidx, in_surv, ssep = _match(cq, s2["RA"].values, s2["DEC"].values, args.radius)
    qS["in_par"], qS["in_surv"] = in_par, in_surv

    summary = {"n_q1_dr10south": len(qS), "by_grade": {}}
    print(f"[364] Euclid Q1 lens cand in DR10-south (Dec<{args.dec_max}): {len(qS)}")
    print(f"  {'grade':6s} {'n':>5s} {'in_parent':>14s} {'in_v3_survivors':>16s}")
    for g in ["A", "B", "C"]:
        sub = qS[qS["grade"] == g]
        e = {"n": len(sub), "in_parent": int(sub.in_par.sum()), "in_parent_frac": float(sub.in_par.mean()),
             "in_survivors": int(sub.in_surv.sum()), "in_survivors_frac": float(sub.in_surv.mean())}
        summary["by_grade"][g] = e
        print(f"  {g:6s} {len(sub):5d} {e['in_parent']:6d} ({e['in_parent_frac']*100:4.1f}%) "
              f"{e['in_survivors']:7d} ({e['in_survivors_frac']*100:4.1f}%)")
    ab = qS[qS["grade"].isin(["A", "B"])]
    summary["AB"] = {"n": len(ab), "in_parent_frac": float(ab.in_par.mean()),
                     "in_survivors_frac": float(ab.in_surv.mean()),
                     "A_over_C_enrichment": float(summary["by_grade"]["A"]["in_survivors_frac"]
                                                  / max(summary["by_grade"]["C"]["in_survivors_frac"], 1e-9))}
    print(f"  A+B recall into survivors {ab.in_surv.mean()*100:.1f}% | grade-A/C survivor-recall "
          f"enrichment {summary['AB']['A_over_C_enrichment']:.1f}x")

    # the cross-validated set: Euclid lens cand that ARE v3 survivors
    hit = qS[qS.in_surv].copy()
    hit["row_id"] = s2["row_id"].values[sidx[in_surv]]
    hr = s2.iloc[sidx[in_surv]]
    hit["v3_known"] = hr[["match_storfer2024", "match_inchausti2025", "match_huang2021"]].fillna(False).any(axis=1).values
    hit["euclid_fits"] = [euclid.obj_dir(str(e)) is not None for e in hit["id_str"]]
    cols = ["row_id", "right_ascension", "declination", "id_str", "grade", "expert_score",
            "v3_known", "euclid_fits"]
    hit[cols].rename(columns={"right_ascension": "RA", "declination": "DEC", "id_str": "euclid_id",
                              "grade": "euclid_grade"}).to_csv(V3 / "cv3_euclid_xmatch.csv", index=False)
    summary["xmatch"] = {"n": int(len(hit)), "grade": hit.grade.value_counts().to_dict(),
                         "new": int((~hit.v3_known).sum()), "with_euclid_fits": int(hit.euclid_fits.sum()),
                         "AB_new_with_fits": int(((hit.grade.isin(["A", "B"])) & (~hit.v3_known) & hit.euclid_fits).sum())}
    print(f"\n[364] v3-survivor ∩ Euclid-Q1-lens: {len(hit)} (grade {summary['xmatch']['grade']}); "
          f"NEW {summary['xmatch']['new']}; with Euclid cutouts {summary['xmatch']['with_euclid_fits']}; "
          f"NEW A/B with cutouts {summary['xmatch']['AB_new_with_fits']}")
    (V3 / "euclid_overlap_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[364] wrote {V3/'euclid_overlap_summary.json'} + {V3/'cv3_euclid_xmatch.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
