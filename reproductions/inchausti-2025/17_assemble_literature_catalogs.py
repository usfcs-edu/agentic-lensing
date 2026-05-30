#!/usr/bin/env python3
"""
17_assemble_literature_catalogs.py — Phase-5 Stage B.

Harvest the published literature strong-lens catalogues that Storfer+2024 and
Inchausti+2025 used as positives, from VizieR's HTTP API (no astroquery in this
venv), to enlarge our training positive set toward the papers' 1,961 / 1,372.
Deduplicate internally and against our existing 949 Huang-2020/2021 positives,
and report how many genuinely NEW positions we gain.

VizieR catalogue IDs were verified by live query (return real coordinate rows).
Sources NOT on VizieR (More 2016 SpaceWarps, Jacobs 2017 CFHTLS, Talbot 2021
eBOSS-SILO, Stein 2022 ssl-legacysurvey) are noted but not fetched here.

Output:
  data/positives_literature.parquet   Name, RA, DEC, source, is_new
  (Name = DESI-RRR.rrrr+DD.dddd, matching our training-positive naming so the
   Stage-B trainer can reuse build_split.)
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from astropy.coordinates import SkyCoord
from astropy import units as u

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
VIZIER = "https://vizier.cds.unistra.fr/viz-bin/asu-tsv"
DEDUP_RADIUS = 5.0  # arcsec

# (label, vizier_id, ra_col, dec_col, sexagesimal?)  — all verified to return rows.
CATALOGS = [
    ("Sonnenfeld2020_SuGOHI_VI", "J/A+A/642/A148/sugohi",    "RAJ2000", "DEJ2000", False),
    ("Petrillo2019_KiDS_LinKS",  "J/MNRAS/484/3879/tablea1", "RAJ2000", "DEJ2000", False),
    ("Diehl2017_DES_components", "J/ApJS/232/15/table3",     "RAJ2000", "DEJ2000", False),
    ("Jacobs2019b_DES_t1",       "J/ApJS/243/17/table1",     "RAJ2000", "DEJ2000", False),
    ("Jacobs2019b_DES_t5",       "J/ApJS/243/17/table5",     "RAJ2000", "DEJ2000", True),
    ("Canameras2020_HOLISMOKES", "J/A+A/644/A163/table1",    "RAJ2000", "DEJ2000", True),
    ("Jaelani2020_SuGOHI_V",     "J/MNRAS/495/1291/table3",  "RAJ2000", "DEJ2000", False),
    ("More2012_SL2S_SARCS",      "J/ApJ/749/38/table2",      "RAJ2000", "DEJ2000", True),
    ("Pourrahmani2018_LensFlow", "J/ApJ/856/68/table2",      "RAJ2000", "DEJ2000", False),
    ("Jacobs2019a_DES_highz",    "J/MNRAS/484/5330/table4",  "RAJ2000", "DEJ2000", False),
    ("Wong2018_SuGOHI_II",       "J/ApJ/867/107/table1",     "RAJ2000", "DEJ2000", False),
]


def fetch_vizier(catid: str, ra_col: str, dec_col: str) -> pd.DataFrame:
    params = {"-source": catid, "-out": f"{ra_col},{dec_col}", "-out.max": "unlimited"}
    r = requests.get(VIZIER, params=params, timeout=120)
    r.raise_for_status()
    # asu-tsv: comment lines (#...), a column-name row, a units row, a '---'
    # separator, then tab-separated data. Take rows after the dashes separator.
    lines = r.text.splitlines()
    sep = [i for i, ln in enumerate(lines) if ln.startswith("---") or set(ln) <= {"-", "\t", " "}
           and "-" in ln]
    if not sep:
        return pd.DataFrame(columns=["ra_raw", "dec_raw"])
    data = []
    for ln in lines[sep[-1] + 1:]:
        if not ln.strip() or ln.startswith("#"):
            continue
        parts = ln.split("\t")
        if len(parts) >= 2 and parts[0].strip() and parts[1].strip():
            data.append((parts[0].strip(), parts[1].strip()))
    return pd.DataFrame(data, columns=["ra_raw", "dec_raw"])


def to_deg(df: pd.DataFrame, sexagesimal: bool) -> tuple[np.ndarray, np.ndarray]:
    if sexagesimal:
        sc = SkyCoord(ra=df["ra_raw"].values, dec=df["dec_raw"].values,
                      unit=(u.hourangle, u.deg))
        return sc.ra.deg, sc.dec.deg
    return (pd.to_numeric(df["ra_raw"], errors="coerce").values,
            pd.to_numeric(df["dec_raw"], errors="coerce").values)


def desi_name(ra: float, dec: float) -> str:
    return f"DESI-{ra:08.4f}{dec:+08.4f}"


def main() -> None:
    frames = []
    for label, catid, ra_col, dec_col, sex in CATALOGS:
        try:
            raw = fetch_vizier(catid, ra_col, dec_col)
            ra, dec = to_deg(raw, sex)
            d = pd.DataFrame({"RA": ra, "DEC": dec, "source": label}).dropna()
            d = d[(d.RA >= 0) & (d.RA <= 360) & (d.DEC >= -90) & (d.DEC <= 90)]
            frames.append(d)
            print(f"  [{label:28s}] {catid:26s} -> {len(d):5d} rows")
        except Exception as e:
            print(f"  [{label:28s}] FAILED: {e}")
    lit = pd.concat(frames, ignore_index=True)
    print(f"\n[harvest] {len(lit)} raw positions from {lit['source'].nunique()} catalogues")

    # Internal dedup (keep first occurrence within 5"), greedy via a coarse grid + SkyCoord.
    sky = SkyCoord(ra=lit.RA.values * u.deg, dec=lit.DEC.values * u.deg)
    keep = np.ones(len(lit), dtype=bool)
    idx, sep, _ = sky.match_to_catalog_sky(sky, nthneighbor=2)
    # mark the higher-index member of each close pair as duplicate
    for i in range(len(lit)):
        if keep[i] and sep[i].to(u.arcsec).value < DEDUP_RADIUS and idx[i] < i:
            keep[i] = False
    lit = lit[keep].reset_index(drop=True)
    print(f"[dedup] {len(lit)} unique positions after internal 5\" dedup")

    # Tag NEW vs existing training positives (949 Huang-2020/2021).
    train = pd.read_parquet(DATA / "positives_huang2020.parquet")
    tsky = SkyCoord(ra=train.RA.values * u.deg, dec=train.DEC.values * u.deg)
    lsky = SkyCoord(ra=lit.RA.values * u.deg, dec=lit.DEC.values * u.deg)
    _, sep_tr, _ = lsky.match_to_catalog_sky(tsky)
    lit["is_new"] = sep_tr.to(u.arcsec).value >= DEDUP_RADIUS
    lit["Name"] = [desi_name(r.RA, r.DEC) for r in lit.itertuples()]
    lit = lit.drop_duplicates(subset="Name").reset_index(drop=True)

    lit.to_parquet(DATA / "positives_literature.parquet", index=False)
    n_new = int(lit["is_new"].sum())
    print(f"\n[done] wrote positives_literature.parquet")
    print(f"  unique literature positions : {len(lit)}")
    print(f"  already in our training set  : {len(lit) - n_new}")
    print(f"  NEW (not in our 949)         : {n_new}")
    print(f"  -> enlarged positive pool    : {len(train) + n_new} "
          f"(paper targets: Storfer 1961 / Inchausti 1372)")
    print("\n[by source — new only]")
    print(lit[lit.is_new].groupby("source").size().sort_values(ascending=False).to_string())


if __name__ == "__main__":
    main()
