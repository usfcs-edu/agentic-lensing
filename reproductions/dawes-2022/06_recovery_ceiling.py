#!/usr/bin/env python3
"""
06_recovery_ceiling.py

Honest, proxy-aware recovery analysis.

The paper's headline is 100% recovery of the 94 "discoverable known" systems --
those known lenses that have >=2 objects *in the DESI Quasar Sample*. The
analog in our proxy is: of the published Dawes candidate systems that have >=2
DR1 spectroscopic QSOs near them (i.e. are "discoverable" in OUR sample), how
many does our FoF algorithm group together?

We compute, per published candidate position:
  n_qso_within_5     = # DR1 spectroscopic QSOs within 5"
  in_fof_group       = nearest 5" FoF group centroid is within 3"
and report:
  - raw recovery over all 436 (limited by the proxy sample size)
  - conditional recovery over the subset with n_qso_within_5 >= 2
    ("discoverable in our proxy") -- this is the apples-to-apples analog of the
    paper's 94/94 = 100%.

Output: data/recovery_ceiling.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from astropy.io import fits
from astropy.coordinates import SkyCoord, search_around_sky
import astropy.units as u


HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
PUB = DATA / "dawes2023_vizier_table2.csv"
GROUPS = DATA / "qso_groups_5arcsec.parquet"
ZCAT = HERE.parent / "hsu-2025" / "data" / "zall-pix-iron.fits"
OUT = DATA / "recovery_ceiling.json"

DEDUP = 0.2   # arcsec; collapse spectroscopic duplicates of the same object


def load_qso_positions() -> SkyCoord:
    with fits.open(ZCAT, memmap=True) as h:
        zh = next(x for x in h if x.name == "ZCATALOG")
        d = zh.data
        st = np.char.strip(np.asarray(d["SPECTYPE"], dtype="U20"))
        m = (st == "QSO") & (d["ZWARN"] == 0) & (d["ZCAT_PRIMARY"].astype(bool))
        ra = np.ascontiguousarray(d["TARGET_RA"][m]).astype(float)
        dec = np.ascontiguousarray(d["TARGET_DEC"][m]).astype(float)
    return SkyCoord(ra=ra * u.deg, dec=dec * u.deg)


def main() -> None:
    pub = pd.read_csv(PUB)
    prim = pub[pub["Grade"].astype(str).str.strip().isin(["A", "B", "C"])]
    prim = prim.drop_duplicates("Name").reset_index(drop=True)
    pub_sc = SkyCoord(ra=prim["_RA"].to_numpy() * u.deg,
                      dec=prim["_DE"].to_numpy() * u.deg)
    print(f"[pub ] {len(prim)} published candidates")

    qso_sc = load_qso_positions()
    print(f"[qso ] {len(qso_sc):,} DR1 QSOs")

    # count DISTINCT DR1 QSO images within 5" of each published position
    idx_pub, idx_qso, sep2d, _ = search_around_sky(pub_sc, qso_sc, 5 * u.arcsec)
    n_within = np.zeros(len(prim), dtype=int)
    n_distinct = np.zeros(len(prim), dtype=int)
    for p in range(len(prim)):
        sel = idx_qso[idx_pub == p]
        if len(sel) == 0:
            continue
        n_within[p] = len(sel)
        sub = qso_sc[sel]
        keep = []
        for k in range(len(sub)):
            if all(sub[k].separation(sub[m]).to_value(u.arcsec) > DEDUP
                   for m in keep):
                keep.append(k)
        n_distinct[p] = len(keep)
    prim["n_qso_within_5"] = n_within
    prim["n_distinct_qso_within_5"] = n_distinct

    # FoF group recovery (nearest 5" group centroid within 3")
    g = pq.read_table(GROUPS).to_pandas()
    cen = g.groupby("group_id")[["RA", "DEC"]].mean().reset_index()
    grp_sc = SkyCoord(ra=cen["RA"].to_numpy() * u.deg,
                      dec=cen["DEC"].to_numpy() * u.deg)
    _, sep, _ = pub_sc.match_to_catalog_sky(grp_sc)
    prim["fof_offset_arcsec"] = sep.to_value(u.arcsec)
    prim["in_fof_group"] = prim["fof_offset_arcsec"] < 3.0

    n = len(prim)
    n_raw = int(prim["in_fof_group"].sum())
    disc = prim["n_distinct_qso_within_5"] >= 2
    n_disc = int(disc.sum())
    n_disc_rec = int(prim.loc[disc, "in_fof_group"].sum())

    has_any = prim["n_qso_within_5"] >= 1
    print()
    print(f"[ceiling] published candidates with >=1 DR1 QSO within 5\": "
          f"{int(has_any.sum())}/{n}")
    print(f"[ceiling] published candidates 'discoverable in proxy' "
          f"(>=2 distinct DR1 QSOs within 5\"): {n_disc}/{n}")
    print(f"[ceiling] RAW recovery (all 436): {n_raw}/{n} = "
          f"{100*n_raw/n:.1f}%  (proxy-sample-limited)")
    print(f"[ceiling] CONDITIONAL recovery among 'discoverable in proxy': "
          f"{n_disc_rec}/{n_disc} = "
          f"{100*n_disc_rec/n_disc if n_disc else 0:.1f}%  "
          f"(analog of paper's 94/94 = 100%)")

    stats = {
        "dedup_arcsec": DEDUP,
        "n_published": n,
        "n_with_any_dr1_qso_within_5": int(has_any.sum()),
        "n_discoverable_in_proxy_>=2_distinct": n_disc,
        "n_raw_recovered": n_raw,
        "raw_recovery_frac": n_raw / n,
        "n_conditional_recovered": n_disc_rec,
        "conditional_recovery_frac": (n_disc_rec / n_disc) if n_disc else None,
        "paper_target": "100% recovery of 94 discoverable-known systems",
        "proxy_note": (
            "Our DR1 spectroscopic QSO sample (~1.6M) is ~1/3 the size of "
            "Dawes' ~5M photometric DESI Quasar Sample and selected "
            "differently. Most published candidate IMAGES have no DR1 "
            "spectroscopic QSO counterpart (faint, blended lensed images), so "
            "raw recovery is capped near 138/436. The conditional recovery -- "
            "among candidates that ARE discoverable in our proxy -- is the "
            "apples-to-apples analog of the paper's 94/94."
        ),
    }
    OUT.write_text(json.dumps(stats, indent=2))
    print(f"[save] wrote {OUT}")


if __name__ == "__main__":
    main()
