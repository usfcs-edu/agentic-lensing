#!/usr/bin/env python3
"""
23_build_curated_positives.py — Phase-5 Stage D.

Build the closest-achievable reconstruction of the papers' CURATED training
positive set: the recoverable union of "known lenses + high-quality candidates
from Papers I-III + literature", confidence-tiered to mimic their visual-grade
curation, capped to the Storfer training scale (1,961), with DR9 cutouts.

Sources (all on disk / re-fetchable):
  - NEW (retrieved beyond the 11 VizieR catalogues, this stage):
      Stein 2022   GitHub new_lenses.tsv (1192, grade A/B) + training_lenses.tsv
                   (1615 known lenses) -- DR9-native.
      Talbot 2021  SILO eBOSS VAC (1551, TOTAL_GRADE) -- /raid/.../data/silo/.
      More 2016    SpaceWarps II (59) -- arXiv e-print extract.
  - On disk: Huang2020/2021 + Storfer + Inchausti published catalogues
    (graded) and positives_literature.parquet (10 VizieR catalogues).

Confidence tiers (reconstruction of "high-quality"; documented in the report):
  5 spectroscopically confirmed (SILO grade-A family)
  4 known lens (Stein training set) or grade-A published/literature candidate
  3 grade-B candidate
  2 grade-C from a *vetted published* catalogue (Huang/Storfer/Inchausti) or a
    graded literature catalogue (SuGOHI / HOLISMOKES / DES / SL2S)
  0 raw low-purity CNN candidate lists (KiDS-Petrillo) and ungraded COSMOS -> dropped

Dedup 5"; keep max-confidence per cluster; take the top 1,961 by confidence
(seeded tiebreak). Download DR9 cutouts (101px) for ~headroom and keep the first
1,961 that are in-footprint (non-blank).

Output:
  data/cutouts_fits_curated_dr9/<name>.fits
  data/positives_curated.parquet  name, RA, DEC, source, grade, conf
"""
from __future__ import annotations

import io
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.io import fits
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
SILO_CSV = Path("/raid/benson/git/agentic-lensing/data/silo/silo_eboss_radec.csv")
OUT = DATA / "cutouts_fits_curated_dr9"
TARGET = 1961
HEADROOM = 2600
DEDUP = 5.0
ENDPOINT = "https://www.legacysurvey.org/viewer/fits-cutout"


def _grade_letter(g):
    s = str(g).strip().upper()
    return s[0] if s and s[0] in "ABCD" else ""


def load_sources() -> pd.DataFrame:
    frames = []

    def add(ra, dec, grade, source):
        frames.append(pd.DataFrame({"RA": np.asarray(ra, float), "DEC": np.asarray(dec, float),
                                    "grade": list(grade), "source": source}))

    # --- Stein 2022 (re-fetch GitHub raw if /tmp copy gone) ---
    base = "https://raw.githubusercontent.com/georgestein/ssl-legacysurvey/main/strong_lensing_paper/data"
    for fn, src, has_grade in (("new_lenses.tsv", "Stein2022_new", True),
                               ("training_lenses.tsv", "Stein2022_known", False)):
        p = Path("/tmp") / fn
        try:
            if not p.exists():
                r = requests.get(f"{base}/{fn}", timeout=60); r.raise_for_status()
                p.write_bytes(r.content)
            d = pd.read_csv(p, sep="\t")
            g = d["grade"].map(_grade_letter) if has_grade and "grade" in d else ["K"] * len(d)
            add(d["ra"], d["dec"], g, src)
            print(f"  [{src:18s}] {len(d)} rows")
        except Exception as e:
            print(f"  [{src:18s}] FAILED: {e}")

    # --- Talbot 2021 SILO ---
    if SILO_CSV.exists():
        d = pd.read_csv(SILO_CSV)
        add(d["RA"], d["DEC"], d["TOTAL_GRADE"].map(_grade_letter), "Talbot2021_SILO")
        print(f"  [Talbot2021_SILO  ] {len(d)} rows")

    # --- More 2016 SpaceWarps II ---
    mp = Path("/tmp/sw2_eprint/spacewarps2_cfhtls_candidates.csv")
    if mp.exists():
        d = pd.read_csv(mp)
        rc = [c for c in d.columns if c.lower().startswith("ra")][0]
        dc = [c for c in d.columns if c.lower().startswith("dec")][0]
        add(d[rc], d[dc], ["B"] * len(d), "More2016_SpaceWarps")
        print(f"  [More2016         ] {len(d)} rows")

    # --- on-disk published catalogues (graded, vetted) ---
    for fn, src in (("huang2021_published_catalog.csv", "Huang2021"),
                    ("storfer2024_published_catalog.csv", "Storfer2024"),
                    ("inchausti2025_published_catalog.csv", "Inchausti2025")):
        d = pd.read_csv(DATA / fn)
        add(d["RA"], d["DEC"], d["grade"].map(_grade_letter), src)
        print(f"  [{src:18s}] {len(d)} rows")
    h20 = HERE.parent / "huang-2020" / "data" / "huang2020_published_catalog.csv"
    if h20.exists():
        d = pd.read_csv(h20)
        g = d["grade"].map(_grade_letter) if "grade" in d else ["A"] * len(d)
        add(d["RA"], d["DEC"], g, "Huang2020")
        print(f"  [Huang2020        ] {len(d)} rows")

    # --- literature (10 VizieR catalogues) ---
    lit = pd.read_parquet(DATA / "positives_literature.parquet")
    add(lit["RA"], lit["DEC"], [""] * len(lit), "lit:" + lit["source"].astype(str))
    print(f"  [literature        ] {len(lit)} rows ({lit['source'].nunique()} catalogues)")

    return pd.concat(frames, ignore_index=True).dropna(subset=["RA", "DEC"])


GRADED_LIT = ("Sonnenfeld", "Canameras", "Diehl", "Jacobs2019", "More2012", "Jaelani", "Wong")
RAW_LOWPURITY = ("Petrillo", "Pourrahmani")  # raw CNN candidates / out-of-footprint COSMOS


def confidence(row) -> int:
    src, g = row["source"], row["grade"]
    if src == "Talbot2021_SILO" and g == "A":
        return 5
    if src == "Stein2022_known":
        return 4
    if g == "A":
        return 4
    if g == "B":
        return 3
    if src in ("Huang2020", "Huang2021", "Storfer2024", "Inchausti2025") and g == "C":
        return 2
    if src.startswith("lit:"):
        cat = src.split("lit:")[1]
        if any(k in cat for k in RAW_LOWPURITY):
            return 0
        if any(k in cat for k in GRADED_LIT):
            return 2
        return 1
    if src in ("Talbot2021_SILO",):   # B/C-family spectroscopic
        return 2
    if src == "More2016_SpaceWarps":
        return 3
    return 1


def dedup_maxconf(df) -> pd.DataFrame:
    df = df.sort_values("conf", ascending=False).reset_index(drop=True)
    sky = SkyCoord(ra=df.RA.values * u.deg, dec=df.DEC.values * u.deg)
    idx1, idx2, sep, _ = sky.search_around_sky(sky, DEDUP * u.arcsec)
    keep = np.ones(len(df), dtype=bool)
    for i, j in zip(idx1, idx2):
        if i < j and keep[i] and keep[j]:
            keep[j] = False  # i has >= conf (sorted), drop the later/lower
    return df[keep].reset_index(drop=True)


def desi_name(ra, dec):
    return f"DESI-{ra:08.4f}{dec:+08.4f}"


def fetch_cutout(row):
    name = row["name"]
    out = OUT / f"{name}.fits"
    if out.exists() and out.stat().st_size > 0:
        with fits.open(out) as h:
            return (name, float(np.nanstd(np.asarray(h[0].data, np.float32))))
    url = (f"{ENDPOINT}?ra={row['RA']:.6f}&dec={row['DEC']:.6f}&size=101&layer=ls-dr9"
           f"&pixscale=0.262&bands=grz")
    for attempt in range(1, 5):
        try:
            r = requests.get(url, timeout=60, stream=True)
            if r.status_code == 429:
                time.sleep(30); continue
            r.raise_for_status()
            with open(out, "wb") as f:
                for ch in r.iter_content(65536):
                    if ch:
                        f.write(ch)
            if out.stat().st_size < 256:
                out.unlink(); raise RuntimeError("small")
            with fits.open(out) as h:
                return (name, float(np.nanstd(np.asarray(h[0].data, np.float32))))
        except Exception:
            if attempt < 4:
                time.sleep(4 * attempt)
    return (name, 0.0)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("[load] assembling sources")
    df = load_sources()
    df["conf"] = df.apply(confidence, axis=1)
    df = df[df["conf"] > 0].reset_index(drop=True)
    print(f"[pool] {len(df)} graded/vetted positions (dropped raw low-purity)")
    df = dedup_maxconf(df)
    print(f"[dedup] {len(df)} unique; conf histogram: "
          f"{df['conf'].value_counts().sort_index(ascending=False).to_dict()}")

    rng = np.random.default_rng(2026)
    df["_tie"] = rng.random(len(df))
    df = df.sort_values(["conf", "_tie"], ascending=[False, True]).reset_index(drop=True)
    df["name"] = [desi_name(r.RA, r.DEC) for r in df.itertuples()]
    df = df.drop_duplicates("name")
    cand = df.head(HEADROOM).copy()
    print(f"[cutouts] downloading DR9 cutouts for top {len(cand)} (headroom for footprint cut)")

    rows = cand.to_dict("records")
    fstd = {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(fetch_cutout, r) for r in rows]
        for fut in tqdm(as_completed(futs), total=len(futs), unit="cut"):
            n, s = fut.result(); fstd[n] = s
    cand["flux_std"] = cand["name"].map(fstd)
    good = cand[cand["flux_std"] > 1e-3].head(TARGET).reset_index(drop=True)
    keep = good[["name", "RA", "DEC", "source", "grade", "conf"]]
    keep.to_parquet(DATA / "positives_curated.parquet", index=False)

    print(f"\n[done] curated positives: {len(keep)} (target {TARGET}); "
          f"in-footprint of top-{HEADROOM} = {int((cand['flux_std']>1e-3).sum())}")
    print("[conf in curated]\n" + keep["conf"].value_counts().sort_index(ascending=False).to_string())
    print("[top sources]\n" + keep["source"].str.replace(r"^lit:", "lit:", regex=True)
          .value_counts().head(12).to_string())


if __name__ == "__main__":
    main()
