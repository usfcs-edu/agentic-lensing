#!/usr/bin/env python
"""Harvest gravitationally-lensed QUASAR catalogs from VizieR.

Catalogs (real, verified VizieR IDs):
  - J/ApJ/921/42/table2   Gaia GraL VI (Stern+ 2021): CONFIRMED quad lenses
  - J/MNRAS/520/3305/table1 Lemon+ 2023: ~100 confirmed lensed quasars (spectroscopic
                          follow-up; keep only lens-positive Class)
  - J/A+A/622/A165/catalog Gaia GraL III (Delchambre+ 2019): full 6.46M clustering
                          catalog -> we extract ONLY the named `Known` lenses (curated)
  - J/AJ/132/999/table2   SQLS DR3 (Oguri+ 2006): known/confirmed lenses
  - J/AJ/143/119/table5   SQLS V final (Inada+ 2012): DR7 statistical sample (confirmed)
  - J/AJ/143/119/table6   SQLS V final (Inada+ 2012): additional confirmed lensed quasars

Normalized columns: name, ra(deg float), dec(deg float), grade, source, ref
source tag = "lensed_quasars". Union, dedup at 5 arcsec, keep dec <= 32.375.
"""
import warnings, re, sys
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from astroquery.vizier import Vizier
import pyvo
from astropy.coordinates import SkyCoord
import astropy.units as u

OUT = "/home2/benson/git/agentic-lensing/reproductions/claudenet/data/harvest/lensed_quasars.parquet"
DEC_CUT = 32.375

frames = []  # list of dicts->DataFrame

def parse_jname(name):
    """Parse Jhhmmss[.s]+ddmmss[.s] -> (ra_deg, dec_deg). Returns (nan,nan) if no match."""
    if name is None:
        return np.nan, np.nan
    s = str(name).strip()
    # strip leading prefixes like SDSS, GraL, WISE, etc., keep the J... token
    m = re.search(r'J?\s*(\d{2})(\d{2})(\d{2}(?:\.\d+)?)\s*([+\-])\s*(\d{2})(\d{2})(\d{2}(?:\.\d+)?)', s)
    if not m:
        # short form Jhhmm+ddmm
        m2 = re.search(r'J?\s*(\d{2})(\d{2})(?:\.\d+)?\s*([+\-])\s*(\d{2})(\d{2})(?:\.\d+)?', s)
        if not m2:
            return np.nan, np.nan
        hh, mm, sgn, dd, dm = m2.groups()
        ra = (int(hh) + int(mm)/60.0) * 15.0
        dec = (int(dd) + int(dm)/60.0)
        if sgn == '-':
            dec = -dec
        return ra, dec
    hh, mm, ss, sgn, dd, dm, ds = m.groups()
    ra = (int(hh) + int(mm)/60.0 + float(ss)/3600.0) * 15.0
    dec = abs(int(dd)) + int(dm)/60.0 + float(ds)/3600.0
    if sgn == '-':
        dec = -dec
    return ra, dec

# ---------------------------------------------------------------------------
# 1) Gaia GraL VI (Stern+ 2021) confirmed quads -- coords from GraL name
# ---------------------------------------------------------------------------
try:
    v = Vizier(row_limit=-1)
    cats = v.get_catalogs("J/ApJ/921/42")
    t2 = next(t for t in cats if t.meta.get("name") == "J/ApJ/921/42/table2")
    n = 0
    for r in t2:
        nm = str(r["GraL"])
        ra, dec = parse_jname(nm)
        frames.append(dict(name="GraLJ" + nm.lstrip("J"), ra=ra, dec=dec,
                           grade="confirmed_quad",
                           source="lensed_quasars",
                           ref="Stern+2021 (Gaia GraL VI), J/ApJ/921/42"))
        n += 1
    print(f"[GraL VI Stern+2021 table2] fetched rows={n}", flush=True)
except Exception as e:
    print("[GraL VI] ERR", type(e).__name__, e, flush=True)

# ---------------------------------------------------------------------------
# 2) Lemon+ 2023 master compilation -- keep only lens-positive Class
# ---------------------------------------------------------------------------
LENS_POS = {"lens", "quad", "lensed gal."}
LENS_MAYBE = {"lens (?)"}
try:
    v = Vizier(row_limit=-1)
    cats = v.get_catalogs("J/MNRAS/520/3305")
    t1 = next(t for t in cats if t.meta.get("name") == "J/MNRAS/520/3305/table1")
    n = 0
    for r in t1:
        cls = str(r["Class"]).strip()
        if cls in LENS_POS:
            grade = "confirmed:" + cls
        elif cls in LENS_MAYBE:
            grade = "candidate:" + cls
        else:
            continue  # non-lens (UQP, proj. QSOs, QSO+star, stars, etc.) -> skip
        ra = float(r["RAJ2000"]); dec = float(r["DEJ2000"])
        frames.append(dict(name="Lemon" + str(r["Name"]), ra=ra, dec=dec,
                           grade=grade, source="lensed_quasars",
                           ref="Lemon+2023, J/MNRAS/520/3305"))
        n += 1
    print(f"[Lemon+2023 table1] lens-positive rows kept={n}", flush=True)
except Exception as e:
    print("[Lemon+2023] ERR", type(e).__name__, e, flush=True)

# ---------------------------------------------------------------------------
# 3) Gaia GraL III (Delchambre+ 2019): extract ONLY named Known lenses via TAP
#    (full catalog = 6.46M clustering rows; we do NOT ingest raw candidate clusters)
# ---------------------------------------------------------------------------
try:
    tap = pyvo.dal.TAPService("https://tapvizier.cds.unistra.fr/TAPVizieR/tap")
    q = ('SELECT "Known", AVG("RAICRS") ra, AVG("DEICRS") de, MAX("Nimg") nimg, MAX("P") p '
         'FROM "J/A+A/622/A165/catalog" '
         "WHERE \"Known\" IS NOT NULL AND \"Known\" <> '' "
         'GROUP BY "Known"')
    res = tap.search(q).to_table()
    n = 0
    for r in res:
        known = str(r["Known"]).strip()
        if not known:
            continue
        ra = float(r["ra"]); dec = float(r["de"])
        cand = known.startswith("?")
        grade = ("candidate_known" if cand else "known_lens")
        frames.append(dict(name="GraLknown_" + known.lstrip("?"), ra=ra, dec=dec,
                           grade=grade, source="lensed_quasars",
                           ref="Delchambre+2019 (Gaia GraL III), J/A+A/622/A165"))
        n += 1
    print(f"[GraL III Delchambre+2019] named-Known lenses={n} "
          f"(NOTE: 6.46M raw clusters intentionally excluded)", flush=True)
except Exception as e:
    print("[GraL III] ERR", type(e).__name__, e, flush=True)

# ---------------------------------------------------------------------------
# 4) SQLS DR3 (Oguri+ 2006) table2 -- known/confirmed lenses (has _RA/_DE)
# ---------------------------------------------------------------------------
try:
    v = Vizier(row_limit=-1)
    cats = v.get_catalogs("J/AJ/132/999")
    t2 = next(t for t in cats if t.meta.get("name") == "J/AJ/132/999/table2")
    n = 0
    for r in t2:
        ra = float(r["_RA"]); dec = float(r["_DE"])
        morph = str(r["Type"]).strip()
        frames.append(dict(name="SQLS" + str(r["Name"]), ra=ra, dec=dec,
                           grade="confirmed" + (f":{morph}" if morph and morph != "--" else ""),
                           source="lensed_quasars",
                           ref="Oguri+2006 (SQLS DR3), J/AJ/132/999"))
        n += 1
    print(f"[SQLS Oguri+2006 table2] fetched rows={n}", flush=True)
except Exception as e:
    print("[SQLS DR3] ERR", type(e).__name__, e, flush=True)

# ---------------------------------------------------------------------------
# 5) SQLS V final (Inada+ 2012): table5 statistical (confirmed; coords from name)
#    + table6 additional confirmed lensed quasars (has _RA/_DE)
# ---------------------------------------------------------------------------
try:
    v = Vizier(row_limit=-1)
    cats = v.get_catalogs("J/AJ/143/119")
    t5 = next(t for t in cats if t.meta.get("name") == "J/AJ/143/119/table5")
    n5 = 0
    for r in t5:
        nm = str(r["SDSS"])
        ra, dec = parse_jname(nm)
        frames.append(dict(name="SDSSJ" + nm.lstrip("J"), ra=ra, dec=dec,
                           grade="confirmed_statistical",
                           source="lensed_quasars",
                           ref="Inada+2012 (SQLS V), J/AJ/143/119"))
        n5 += 1
    t6 = next(t for t in cats if t.meta.get("name") == "J/AJ/143/119/table6")
    n6 = 0
    for r in t6:
        # table6 = additional lensed quasars; some rows are companion galaxies.
        # Use _RA/_DE; the 'S' flag marks the lensed-QSO (selected) row.
        ra = float(r["_RA"]); dec = float(r["_DE"])
        frames.append(dict(name="SDSSJ" + str(r["SDSS"]).lstrip("J"), ra=ra, dec=dec,
                           grade="confirmed_additional",
                           source="lensed_quasars",
                           ref="Inada+2012 (SQLS V), J/AJ/143/119"))
        n6 += 1
    print(f"[SQLS V Inada+2012] table5={n5} table6={n6}", flush=True)
except Exception as e:
    print("[SQLS V] ERR", type(e).__name__, e, flush=True)

# ===========================================================================
# Build DataFrame
# ===========================================================================
df = pd.DataFrame(frames, columns=["name", "ra", "dec", "grade", "source", "ref"])
df["name"] = df["name"].astype(str)
df["ra"] = pd.to_numeric(df["ra"], errors="coerce")
df["dec"] = pd.to_numeric(df["dec"], errors="coerce")
df["grade"] = df["grade"].astype(str)
df["source"] = df["source"].astype(str)
df["ref"] = df["ref"].astype(str)

n_raw = len(df)
print(f"\nRAW rows (all catalogs, pre-clean): {n_raw}", flush=True)

# Drop null / non-finite ra/dec
mask = np.isfinite(df["ra"].to_numpy()) & np.isfinite(df["dec"].to_numpy())
df = df[mask].copy()
# sanity range
mask2 = (df["ra"] >= 0) & (df["ra"] <= 360) & (df["dec"] >= -90) & (df["dec"] <= 90)
df = df[mask2].copy()
print(f"After dropping null/non-finite/out-of-range ra/dec: {len(df)}", flush=True)

# ---------------------------------------------------------------------------
# Dedup at 5 arcsec (union across catalogs). Prefer "confirmed" over candidate.
# ---------------------------------------------------------------------------
def grade_rank(g):
    g = g.lower()
    if g.startswith("candidate"):
        return 0
    return 1  # confirmed / known / statistical / etc.

df = df.reset_index(drop=True)
sc = SkyCoord(ra=df["ra"].to_numpy()*u.deg, dec=df["dec"].to_numpy()*u.deg)
# self-match
idx, sep2d, _ = sc.match_to_catalog_sky(sc, nthneighbor=2)
# Build groups via union-find on pairs within 5"
parent = list(range(len(df)))
def find(a):
    while parent[a] != a:
        parent[a] = parent[parent[a]]; a = parent[a]
    return a
def union(a, b):
    ra_, rb_ = find(a), find(b)
    if ra_ != rb_:
        parent[rb_] = ra_

# all pairs within 5": use search_around_sky
i1, i2, sep, _ = sc.search_around_sky(sc, 5*u.arcsec)
for a, b in zip(i1, i2):
    if a < b:
        union(int(a), int(b))

groups = {}
for i in range(len(df)):
    groups.setdefault(find(i), []).append(i)

keep_idx = []
for g, members in groups.items():
    if len(members) == 1:
        keep_idx.append(members[0])
    else:
        # pick highest grade_rank, then shortest name (cleanest)
        best = sorted(members, key=lambda i: (-grade_rank(df.at[i, "grade"]),
                                              len(df.at[i, "name"])))[0]
        keep_idx.append(best)

dedup = df.iloc[sorted(keep_idx)].reset_index(drop=True)
print(f"After 5-arcsec dedup (union): {len(dedup)}  (removed {len(df)-len(dedup)} dups)", flush=True)

# ---------------------------------------------------------------------------
# Dec cut
# ---------------------------------------------------------------------------
n_before_cut = len(dedup)
south = dedup[dedup["dec"] <= DEC_CUT].reset_index(drop=True)
n_after_cut = len(south)
print(f"\nROW COUNT before dec<= {DEC_CUT} cut: {n_before_cut}", flush=True)
print(f"ROW COUNT after  dec<= {DEC_CUT} cut: {n_after_cut}", flush=True)

# grade breakdown
print("\nGrade breakdown (post dec-cut):", flush=True)
print(south["grade"].value_counts().to_string(), flush=True)
print("\nRef breakdown (post dec-cut):", flush=True)
print(south["ref"].value_counts().to_string(), flush=True)

south.to_parquet(OUT, index=False)
print(f"\nWROTE {OUT}  rows={len(south)}", flush=True)

# echo dtypes + head for verification
print("\ndtypes:\n", south.dtypes, flush=True)
print(south.head(8).to_string(), flush=True)
