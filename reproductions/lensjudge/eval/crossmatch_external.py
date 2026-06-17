#!/usr/bin/env python3
"""Crossmatch the LensJudge / Huang-group lens candidates against external high-res
and multi-grader archives, to find the subset re-gradable at better resolution today.

Build the master candidate coordinate table (union of Storfer, Inchausti, Huang-2020/2021,
dedup by position) and crossmatch it against external high-res / multi-grader archives:
  - Euclid Q1 (local catalog; 0.1" VIS, ~10 expert votes) — always.
  - SuGOHI/HOLISMOKES (local aion-1 HSC catalog; 0.6", committee grades + spec-z) — always.
  - HST/MAST, AGEL (VizieR) — best-effort under --online (graceful skip if astroquery/network absent).
A merged `external_overlap.csv` flags, per candidate, which archives cover it (the B3 deliverable +
the known-lens anchor set for HSC tier-2 validation).

  python lensjudge/eval/crossmatch_external.py --match-radius 5            # local (Euclid+SuGOHI)
  python lensjudge/eval/crossmatch_external.py --match-radius 5 --online   # + HST/MAST, AGEL
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u

REPRO = Path(__file__).resolve().parents[2]
OUT = REPRO / "lensjudge" / "outputs"
EUCLID_CAT = REPRO / "euclid-q1" / "data" / "raw" / "q1_discovery_engine_lens_catalog.csv"
# SuGOHI/HOLISMOKES HSC lens catalog (staged for the AION-1 reproduction): grade/type/spec-z/theta_E.
SUGOHI_CAT = REPRO / "aion-1" / "data" / "raw" / "sugohi" / "lens_radec.parquet"

# (path, ra_col, dec_col, name_col, grade_col, score_col, source_tag)
SOURCES = [
    ("inchausti-2025/data/candidate_scores_storfer.csv", "RA", "DEC", "name", "grade", "our_p_meta", "storfer"),
    ("inchausti-2025/data/candidate_scores_inchausti.csv", "RA", "DEC", "name", "grade", "our_p_meta", "inchausti"),
    ("huang-2021/data/huang2021_published_catalog.csv", "RA", "DEC", "name", "grade", "probability", "huang2021"),
    ("huang-2020/data/huang2020_published_catalog.csv", "RA", "DEC", "Name", "Grade", "Probability", "huang2020"),
]


def build_master() -> pd.DataFrame:
    frames = []
    for rel, rac, dec, namec, gradec, scorec, tag in SOURCES:
        p = REPRO / rel
        if not p.exists():
            print(f"  [skip] {rel} (missing)")
            continue
        df = pd.read_csv(p)
        cols = {c.lower(): c for c in df.columns}
        def pick(want, default=None):
            return cols.get(want.lower(), default)
        rac_, dec_ = pick(rac), pick(dec)
        if rac_ is None or dec_ is None:
            print(f"  [skip] {rel} (no ra/dec: {list(df.columns)[:6]})")
            continue
        out = pd.DataFrame({
            "ra": pd.to_numeric(df[rac_], errors="coerce"),
            "dec": pd.to_numeric(df[dec_], errors="coerce"),
            "name": df[pick(namec)] if pick(namec) in df else np.arange(len(df)),
            "grade": df[pick(gradec)] if pick(gradec) in df else np.nan,
            "score": pd.to_numeric(df[pick(scorec)], errors="coerce") if pick(scorec) in df else np.nan,
            "source": tag,
        })
        out = out.dropna(subset=["ra", "dec"])
        frames.append(out)
        print(f"  [{tag:10s}] {len(out):5d} rows with coords")
    master = pd.concat(frames, ignore_index=True)

    # dedup by position: keep one row per 1.5" cluster, preferring the best grade
    sc = SkyCoord(master.ra.values * u.deg, master.dec.values * u.deg)
    idx, sep2d, _ = sc.match_to_catalog_sky(sc, nthneighbor=2)
    # union-find by 1.5" links
    parent = list(range(len(master)))
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    for i, (j, s) in enumerate(zip(idx, sep2d)):
        if s.arcsec < 1.5:
            ri, rj = find(i), find(int(j))
            if ri != rj:
                parent[max(ri, rj)] = min(ri, rj)
    master["cluster"] = [find(i) for i in range(len(master))]
    grade_rank = {"A": 0, "B": 1, "C": 2, "D": 3}
    master["_gr"] = master.grade.map(lambda g: grade_rank.get(str(g).strip().upper(), 9))
    master = master.sort_values(["cluster", "_gr"]).groupby("cluster", as_index=False).first()
    master = master.drop(columns=["_gr"])
    print(f"  -> {len(master)} unique candidates after 1.5\" dedup")
    return master


def crossmatch(master: pd.DataFrame, cat: pd.DataFrame, ra2: str, dec2: str,
               radius_arcsec: float, tag: str) -> pd.DataFrame:
    a = SkyCoord(master.ra.values * u.deg, master.dec.values * u.deg)
    b = SkyCoord(cat[ra2].values * u.deg, cat[dec2].values * u.deg)
    idx, sep2d, _ = a.match_to_catalog_sky(b)
    hit = sep2d.arcsec < radius_arcsec
    m = master[hit].copy()
    m[f"{tag}_sep_arcsec"] = sep2d.arcsec[hit]
    m[f"{tag}_idx"] = idx[hit]
    return m, hit.sum()


def _xmatch_sugohi(master, radius, overlap):
    """SuGOHI/HOLISMOKES (local HSC catalog): committee grade + spec-z + theta_E."""
    if not SUGOHI_CAT.exists():
        print("  [skip] SuGOHI catalog not staged"); return
    su = pd.read_parquet(SUGOHI_CAT)
    su = su.dropna(subset=["ra", "dec"])
    print(f"  SuGOHI: {len(su)} HSC lenses, "
          f"RA [{su.ra.min():.1f},{su.ra.max():.1f}]  Dec [{su.dec.min():.1f},{su.dec.max():.1f}]")
    m, n = crossmatch(master, su, "ra", "dec", radius, "sugohi")
    print(f"  r<{radius:.1f}\": {n} candidate(s) coincide with a SuGOHI HSC lens")
    if n:
        si = m["sugohi_idx"].values
        m = m.assign(sugohi_name=su.name.values[si], sugohi_grade=su.grade.values[si],
                     sugohi_type=su["type"].values[si], sugohi_reference=su.reference.values[si],
                     sugohi_theta_E=su.theta_E.values[si])
        m.to_csv(OUT / "xmatch_sugohi.csv", index=False)
        print(f"  saved {OUT/'xmatch_sugohi.csv'}")
        overlap["sugohi"] = set(m["name"].astype(str))


def _xmatch_online(master, radius, overlap):
    """Best-effort VizieR/MAST crossmatches (HST archival, AGEL). Graceful skip if astroquery /
    network are unavailable — SuGOHI + Euclid are the guaranteed-local core."""
    try:
        from astroquery.vizier import Vizier
        from astropy.coordinates import SkyCoord
        import astropy.units as u
    except Exception as e:
        print(f"  [skip] --online: astroquery unavailable ({type(e).__name__}); "
              f"install with `pip install astroquery`")
        return
    # AGEL DR2 Keck-confirmed lenses (VizieR table). Best-effort: any failure -> skip cleanly.
    for tag, viz_id, racol, deccol in [("agel", "J/AJ/167/236/table2", "RAJ2000", "DEJ2000")]:
        try:
            Vizier.ROW_LIMIT = -1
            tabs = Vizier.get_catalogs(viz_id)
            if not len(tabs):
                print(f"  [skip] {tag}: VizieR returned no table"); continue
            cat = tabs[0].to_pandas()
            cat = cat.rename(columns={racol: "ra", deccol: "dec"}).dropna(subset=["ra", "dec"])
            m, n = crossmatch(master, cat, "ra", "dec", radius, tag)
            print(f"  {tag}: r<{radius:.1f}\" -> {n} candidate(s)")
            if n:
                m.to_csv(OUT / f"xmatch_{tag}.csv", index=False)
                overlap[tag] = set(m["name"].astype(str))
        except Exception as e:
            print(f"  [skip] {tag}: {type(e).__name__}: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--match-radius", type=float, default=5.0,
                    help="crossmatch radius in arcsec (Euclid astrometry is sub-arcsec; "
                         "DESI candidate centroids ~1\")")
    ap.add_argument("--online", action="store_true",
                    help="also attempt VizieR/MAST sources (HST, AGEL); best-effort, graceful skip")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    overlap: dict[str, set] = {}   # survey_tag -> set of matched candidate names

    print("=== building master candidate table ===")
    master = build_master()
    master.to_csv(OUT / "master_candidates.csv", index=False)
    print(f"  saved {OUT/'master_candidates.csv'}")
    print(f"  footprint: RA [{master.ra.min():.1f},{master.ra.max():.1f}]  "
          f"Dec [{master.dec.min():.1f},{master.dec.max():.1f}]")
    print(f"  grade mix: {master.grade.value_counts(dropna=False).to_dict()}")

    print("\n=== crossmatch vs Euclid Q1 (local) ===")
    euc = pd.read_csv(EUCLID_CAT)
    print(f"  Euclid Q1: {len(euc)} candidates, "
          f"RA [{euc.right_ascension.min():.1f},{euc.right_ascension.max():.1f}]  "
          f"Dec [{euc.declination.min():.1f},{euc.declination.max():.1f}]")
    for radius in (3.0, 5.0, 10.0, 30.0):
        _, n = crossmatch(master, euc, "right_ascension", "declination", radius, "euclid")
        print(f"  r<{radius:5.1f}\": {n:3d} candidate(s) fall in Euclid Q1")
    m, n = crossmatch(master, euc, "right_ascension", "declination", args.match_radius, "euclid")
    if n:
        ei = m["euclid_idx"].values
        m = m.assign(euclid_grade=euc.grade.values[ei],
                     euclid_score=euc.expert_score.values[ei],
                     euclid_votes=euc.expert_total_votes.values[ei],
                     euclid_id=euc.id_str.values[ei])
        m.to_csv(OUT / "xmatch_euclid_q1.csv", index=False)
        print(f"  saved {OUT/'xmatch_euclid_q1.csv'}")
        overlap["euclid"] = set(m["name"].astype(str))
    else:
        print("  no candidates fall in the Euclid Q1 footprint at this radius.")

    print("\n=== crossmatch vs SuGOHI/HOLISMOKES (local HSC) ===")
    _xmatch_sugohi(master, args.match_radius, overlap)

    if args.online:
        print("\n=== best-effort online crossmatches (VizieR/MAST) ===")
        _xmatch_online(master, args.match_radius, overlap)

    # merged overlap table: one row per candidate covered by >=1 external archive
    print("\n=== merged external overlap ===")
    covered = sorted(set().union(*overlap.values())) if overlap else []
    if covered:
        mt = master[master["name"].astype(str).isin(covered)].copy()
        for tag, names in overlap.items():
            mt[f"in_{tag}"] = mt["name"].astype(str).isin(names)
        mt.to_csv(OUT / "external_overlap.csv", index=False)
        per = {t: len(s) for t, s in overlap.items()}
        print(f"  {len(covered)} unique candidates with external coverage {per}")
        print(f"  saved {OUT/'external_overlap.csv'}")
    else:
        print("  no external overlap found at this radius.")


if __name__ == "__main__":
    main()
