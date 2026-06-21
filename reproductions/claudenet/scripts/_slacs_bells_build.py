#!/usr/bin/env python
"""
Harvest SDSS spectroscopically-confirmed galaxy-scale lens families from VizieR.

Catalogs actually present in VizieR (verified via Vizier.find_catalogs + TAP METAcat):
  - SLACS V    Bolton+2008    J/ApJ/682/964  (table4: 131 candidates, grade col 'Lens' A/B/X)
  - SLACS IX   Auger+2009     J/ApJ/705/1099 (lenses: 85 SLACS lens sample; table1: 12 new, 'Grade' A/B/X)
  - SLACS XIII Shu+2017       J/ApJ/851/48   (table1: 118 S4TM / "SLACS for the Masses", grade col 'Class')
  - BELLS GAL  Shu+2016       J/ApJ/824/86   (table1: 21 confirmed BELLS GALLERY lenses)

NOT in VizieR as standalone catalogs (verified: TAP METAcat has no entry):
  - BELLS I    Brownstein+2012 ApJ 744/41   -> no VizieR catalog
  - SLACS XII / S4TM Shu+2015  ApJ 803/71   -> no standalone catalog (S4TM sample folded into Shu+2017 J/ApJ/851/48)
  - SLACS X    Auger+2010      ApJ 724/511  -> no VizieR catalog

Normalize to: name, ra(deg), dec(deg), grade, source, ref. UNION, dedup @5", dec<=32.375.
"""
import warnings
import numpy as np
import pandas as pd
from astroquery.vizier import Vizier
from astropy.coordinates import SkyCoord
import astropy.units as u

warnings.filterwarnings("ignore")

DEC_CUT = 32.375
OUT = "/home2/benson/git/agentic-lensing/reproductions/claudenet/data/harvest/slacs_bells.parquet"

v = Vizier(columns=["**"], row_limit=-1)


def sexa_to_deg(ra_str, dec_str):
    """Parse 'HH MM SS.s' / '+DD MM SS.s' to decimal degrees."""
    try:
        c = SkyCoord(str(ra_str), str(dec_str), unit=(u.hourangle, u.deg))
        return float(c.ra.deg), float(c.dec.deg)
    except Exception:
        return np.nan, np.nan


rows = []  # dicts: name, ra, dec, grade, source, ref

# ---- 1) SLACS V Bolton+2008  J/ApJ/682/964 table4 (131) ----
t = v.get_catalogs("J/ApJ/682/964")[0]  # table4
ref_bolton = "Bolton+2008, ApJ 682, 964 (SLACS V)"
n_bolton = 0
for r in t:
    ra = float(r["_RA"]); dec = float(r["_DE"])
    name = str(r["Name"]).strip()
    grade = str(r["Lens"]).strip() or "?"  # A/B/X
    rows.append(dict(name=f"SLACS {name}", ra=ra, dec=dec,
                     grade=f"SLACS:{grade}", source="SLACS_BELLS", ref=ref_bolton))
    n_bolton += 1
print(f"[Bolton+2008  J/ApJ/682/964 table4] fetched {n_bolton} rows (published 131)")

# ---- 2) SLACS IX Auger+2009  J/ApJ/705/1099 ----
cat = v.get_catalogs("J/ApJ/705/1099")
tl = [x for x in cat if x.meta.get("name").endswith("lenses")][0]   # 85 SLACS lens sample
t1 = [x for x in cat if x.meta.get("name").endswith("table1")][0]   # 12 new, with Grade
ref_auger = "Auger+2009, ApJ 705, 1099 (SLACS IX)"
n_auger = 0
for r in tl:  # main SLACS lens sample (confirmed lenses, no per-row grade column)
    ra, dec = sexa_to_deg(r["RAJ2000"], r["DEJ2000"])
    name = str(r["SDSS"]).strip()
    rows.append(dict(name=f"SLACS {name}", ra=ra, dec=dec,
                     grade="A_spec", source="SLACS_BELLS", ref=ref_auger))
    n_auger += 1
for r in t1:  # 12 newly graded systems
    ra, dec = sexa_to_deg(r["RAJ2000"], r["DEJ2000"])
    name = str(r["SDSS"]).strip()
    grade = str(r["Grade"]).strip() or "?"
    rows.append(dict(name=f"SLACS {name}", ra=ra, dec=dec,
                     grade=f"SLACS:{grade}", source="SLACS_BELLS", ref=ref_auger))
    n_auger += 1
print(f"[Auger+2009   J/ApJ/705/1099 lenses+table1] fetched {n_auger} rows (85 lenses + 12 new = 97)")

# ---- 3) SLACS XIII Shu+2017 (S4TM "SLACS for the Masses")  J/ApJ/851/48 table1 (118) ----
t = v.get_catalogs("J/ApJ/851/48")[0]  # table1
ref_shu17 = "Shu+2017, ApJ 851, 48 (SLACS XIII / S4TM 'SLACS for the Masses')"
n_shu17 = 0
for r in t:
    ra = float(r["_RA"]); dec = float(r["_DE"])
    name = str(r["Target"]).strip()
    cls = str(r["Class"]).strip() or "?"  # e.g. E-S-A ; last letter is the lens grade
    rows.append(dict(name=f"S4TM {name}", ra=ra, dec=dec,
                     grade=f"S4TM:{cls}", source="SLACS_BELLS", ref=ref_shu17))
    n_shu17 += 1
print(f"[Shu+2017     J/ApJ/851/48 table1] fetched {n_shu17} rows (published 118)")

# ---- 4) BELLS GALLERY Shu+2016  J/ApJ/824/86 table1 (21 confirmed) ----
cat = v.get_catalogs("J/ApJ/824/86")
t1 = cat[0]  # table1 = 21 confirmed BELLS GALLERY lenses
ref_shu16 = "Shu+2016, ApJ 824, 86 (BELLS GALLERY III)"
n_bellsg = 0
for r in t1:
    ra, dec = sexa_to_deg(r["RAJ2000"], r["DEJ2000"])
    name = str(r["SDSS"]).strip()
    rows.append(dict(name=f"BELLS-GALLERY J{name}", ra=ra, dec=dec,
                     grade="A_spec", source="SLACS_BELLS", ref=ref_shu16))
    n_bellsg += 1
print(f"[Shu+2016     J/ApJ/824/86 table1] fetched {n_bellsg} rows (published 21 confirmed)")

# ---- UNION ----
df = pd.DataFrame(rows, columns=["name", "ra", "dec", "grade", "source", "ref"])
df["ra"] = pd.to_numeric(df["ra"], errors="coerce")
df["dec"] = pd.to_numeric(df["dec"], errors="coerce")
n_union_raw = len(df)
print(f"\nUNION raw rows: {n_union_raw}")

# Drop null / non-finite ra/dec
df = df[np.isfinite(df["ra"]) & np.isfinite(df["dec"])].reset_index(drop=True)
print(f"After dropping null/non-finite ra/dec: {len(df)}")

# ---- Dedup at 5 arcsec ----
# Greedy: keep first occurrence; drop any later row within 5" of an already-kept row.
coords = SkyCoord(df["ra"].values * u.deg, df["dec"].values * u.deg)
keep = np.ones(len(df), dtype=bool)
idx1, idx2, sep2d, _ = coords.search_around_sky(coords, 5 * u.arcsec)
for a, b in zip(idx1, idx2):
    if a < b and keep[a]:   # b is a later duplicate of an earlier kept row a
        keep[b] = False
df_dedup = df[keep].reset_index(drop=True)
n_before_cut = len(df_dedup)
print(f"After dedup @5 arcsec: {n_before_cut}  (removed {len(df) - n_before_cut})")

# ---- dec <= 32.375 cut ----
df_final = df_dedup[df_dedup["dec"] <= DEC_CUT].reset_index(drop=True)
print(f"\nROW COUNT before dec cut: {n_before_cut}")
print(f"ROW COUNT after dec<= {DEC_CUT}: {len(df_final)}")

# enforce dtypes exactly
df_final["name"] = df_final["name"].astype(str)
df_final["ra"] = df_final["ra"].astype(float)
df_final["dec"] = df_final["dec"].astype(float)
df_final["grade"] = df_final["grade"].astype(str)
df_final["source"] = df_final["source"].astype(str)
df_final["ref"] = df_final["ref"].astype(str)

df_final.to_parquet(OUT, index=False)
print(f"\nWROTE {OUT}  ({len(df_final)} rows)")

# grade breakdown
print("\nGrade breakdown (final):")
print(df_final["grade"].value_counts().to_dict())
print("\nRef breakdown (final):")
print(df_final["ref"].value_counts().to_dict())
print("\ndec range:", float(df_final["dec"].min()), "->", float(df_final["dec"].max()))
print("ra range:", float(df_final["ra"].min()), "->", float(df_final["ra"].max()))
print("\nSample rows:")
print(df_final.head(8).to_string())
