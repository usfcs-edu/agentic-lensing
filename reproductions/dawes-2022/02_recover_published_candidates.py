#!/usr/bin/env python3
"""
02_recover_published_candidates.py

Measure recovery of the published Dawes+2023 candidate systems by our FoF
candidate groups.

PUBLISHED CATALOG: data/dawes2023_vizier_table2.csv
  VizieR J/ApJS/269/61 table2, downloaded via astroquery in
  00_fetch_published_catalog.py. 436 unique candidate systems (102 A, 118 B,
  216 C; 432 Double + 4 Quad in the published Type column), each with (RA, Dec)
  in the DESI-{RA}{+-DEC} naming convention and decoded into the _RA/_DE cols.

RECOVERY DEFINITION (mirrors hsu-2025/06_xmatch_published_catalog.py):
  For each published candidate, find the nearest FoF group centroid in
  data/qso_groups_{5,10}arcsec.parquet and record the great-circle offset.
  A candidate is "recovered" if the nearest group centroid is within MATCH_TOL.
  We use 3" (the published systems all have image separation <5", so the
  centroid of a recovered pair should sit within a few arcsec of the published
  centroid).

The paper's headline target is 100% recovery of the 94 "discoverable known"
systems. We do not have the 94-system list separately, but the published 436
candidate positions are the algorithm's own output positions, so recovering
them measures whether our FoF (proxy QSO sample) re-identifies the same close
quasar pairs. We report recovery vs the full 436 and note the proxy sample is
~1/3 the size of Dawes' photometric target catalog (so misses are expected
where a Dawes image lacks a DR1 spectroscopic QSO counterpart).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from astropy.coordinates import SkyCoord
import astropy.units as u


HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
PUB = DATA / "dawes2023_vizier_table2.csv"
OUT = DATA / "recovery.json"

MATCH_TOL_ARCSEC = 3.0


def load_published() -> pd.DataFrame:
    df = pd.read_csv(PUB)
    # one row per system: the row that carries a non-null Grade is the primary
    prim = df[df["Grade"].astype(str).str.strip().isin(["A", "B", "C"])].copy()
    prim = prim.drop_duplicates(subset="Name")
    prim["ra"] = pd.to_numeric(prim["_RA"], errors="coerce")
    prim["dec"] = pd.to_numeric(prim["_DE"], errors="coerce")
    prim = prim.dropna(subset=["ra", "dec"]).reset_index(drop=True)
    return prim


def recover(pub: pd.DataFrame, groups_path: Path, label: str) -> dict:
    groups = pq.read_table(groups_path).to_pandas()
    centroids = groups.groupby("group_id")[["RA", "DEC"]].mean().reset_index()

    pub_sc = SkyCoord(ra=pub["ra"].to_numpy() * u.deg,
                      dec=pub["dec"].to_numpy() * u.deg)
    grp_sc = SkyCoord(ra=centroids["RA"].to_numpy() * u.deg,
                      dec=centroids["DEC"].to_numpy() * u.deg)
    idx, sep, _ = pub_sc.match_to_catalog_sky(grp_sc)
    off = sep.to_value(u.arcsec)

    pub = pub.copy()
    pub["offset_arcsec"] = off
    pub["nearest_group_id"] = centroids["group_id"].to_numpy()[idx]
    pub["recovered"] = off < MATCH_TOL_ARCSEC

    n = len(pub)
    n_rec = int(pub["recovered"].sum())
    by_grade = (
        pub.groupby("Grade")["recovered"].agg(["sum", "count"]).astype(int).to_dict("index")
    )
    by_grade = {g: {"recovered": v["sum"], "total": v["count"]} for g, v in by_grade.items()}

    print(f"\n[{label}] recovery of {n} published candidates "
          f"(centroid within {MATCH_TOL_ARCSEC}\"):")
    print(f"  recovered: {n_rec}/{n} = {100*n_rec/n:.1f}%")
    print(f"  median offset (recovered): "
          f"{pub.loc[pub['recovered'],'offset_arcsec'].median():.3f}\"")
    for g in ["A", "B", "C"]:
        if g in by_grade:
            r = by_grade[g]
            print(f"    Grade {g}: {r['recovered']}/{r['total']} = "
                  f"{100*r['recovered']/r['total']:.1f}%")

    return {
        "groups_file": groups_path.name,
        "match_tol_arcsec": MATCH_TOL_ARCSEC,
        "n_published": n,
        "n_recovered": n_rec,
        "recovery_frac": n_rec / n,
        "median_offset_recovered_arcsec": float(
            pub.loc[pub["recovered"], "offset_arcsec"].median()),
        "by_grade": by_grade,
        "misses": [
            {"name": r["Name"], "ra": float(r["ra"]), "dec": float(r["dec"]),
             "grade": str(r["Grade"]).strip(),
             "nearest_offset_arcsec": float(r["offset_arcsec"])}
            for _, r in pub[~pub["recovered"]].iterrows()
        ],
    }


def main() -> None:
    pub = load_published()
    print(f"[pub ] {len(pub)} published candidate systems with positions")

    out = {
        "note": (
            "Recovery of the published Dawes+2023 436-candidate positions by "
            "our FoF groups built on the DR1 spectroscopic QSO proxy sample "
            "(~1.6M vs Dawes ~5M photometric targets). Misses are expected "
            "where a published image lacks a DR1 spectroscopic QSO counterpart."
        ),
        "recovery_5arcsec": recover(pub, DATA / "qso_groups_5arcsec.parquet", "5arcsec"),
        "recovery_10arcsec": recover(pub, DATA / "qso_groups_10arcsec.parquet", "10arcsec"),
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(f"\n[save] wrote {OUT}")


if __name__ == "__main__":
    main()
