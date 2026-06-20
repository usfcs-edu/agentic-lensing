#!/usr/bin/env python
"""Harvest SuGOHI (HSC) + DESI Legacy Imaging strong-lens catalogs via astroquery.vizier.

Normalize to EXACT columns: name, ra, dec, grade, source, ref
Drop null/non-finite ra/dec; keep dec <= +32.375 (DR11-south footprint).
source = "DESI_SuGOHI_vizier".

Held-out recall catalogs (Storfer2024 J/ApJS/274/16, Stein2022 J/ApJ/932/107,
Inchausti2025) are NOT present in VizieR (verified empty) so they are not fetched.
If any held-out catalog were fetched its grade would be prefixed "heldout:".
"""
import numpy as np
import pandas as pd
from astroquery.vizier import Vizier

OUT = "/home2/benson/git/agentic-lensing/reproductions/claudenet/data/harvest/desi_sugohi.parquet"
DEC_MAX = 32.375
SOURCE = "DESI_SuGOHI_vizier"

# Held-out catalogs to tag if ever encountered (excluded from training downstream)
HELDOUT_IDS = {"J/ApJS/274/16", "J/ApJ/932/107"}  # Storfer2024, Stein2022

v = Vizier(row_limit=-1, columns=["**"])

# (vizier_id, table_short_name, ra_col, dec_col, name_col, grade_col_or_None, ref_citation)
JOBS = [
    # ---- SuGOHI series (HSC SSP) ----
    ("J/ApJ/867/107", "table1", "RAJ2000", "DEJ2000", "Name", "Grade",
     "SuGOHI II (Wong+2018, ApJ 867, 107)"),
    ("J/ApJ/867/107", "sample", "RAJ2000", "DEJ2000", "Name", None,
     "SuGOHI II (Wong+2018, ApJ 867, 107)"),
    ("J/ApJ/867/107", "added", "RAJ2000", "DEJ2000", "Name", None,
     "SuGOHI II (Wong+2018, ApJ 867, 107)"),
    ("J/MNRAS/495/1291", "table3", "RAJ2000", "DEJ2000", "Name", "Grade",
     "SuGOHI V (Jaelani+2020, MNRAS 495, 1291)"),
    ("J/MNRAS/495/1291", "tablea2", "RAJ2000", "DEJ2000", "Name", "Grade",
     "SuGOHI V (Jaelani+2020, MNRAS 495, 1291)"),
    ("J/A+A/642/A148", "sugohi", "RAJ2000", "DEJ2000", "Name", "Grade",
     "SuGOHI VI (Sonnenfeld+2020, A&A 642, A148)"),
    # ---- DESI Legacy Imaging strong-lens catalogs ----
    ("J/ApJ/894/78", "lens", "_RA", "_DE", "Name", "Q",
     "DECaLS lenses I (Huang+2020, ApJ 894, 78)"),
    ("J/ApJ/909/27", "cand", "_RA", "_DE", "Name", "Q",
     "DESI Legacy lenses II (Huang+2021, ApJ 909, 27)"),
    ("J/ApJS/269/61", "table2", "_RA", "_DE", "Name", "Grade",
     "DESI multiply-lensed quasars (Dawes+2023, ApJS 269, 61)"),
]


def fetch_table(cid, short):
    cats = v.get_catalogs(cid)
    if len(cats) == 0:
        raise RuntimeError(f"{cid}: returned no tables (not in VizieR)")
    byname = {t.meta.get("name", "").split("/")[-1]: t for t in cats}
    if short not in byname:
        raise RuntimeError(f"{cid}: table '{short}' not found; have {list(byname)}")
    return byname[short]


frames = []
report = []
for cid, short, racol, deccol, namecol, gradecol, ref in JOBS:
    t = fetch_table(cid, short)
    df = t[[c for c in (racol, deccol, namecol) + ((gradecol,) if gradecol else ())]].to_pandas()
    n_raw = len(df)
    ra = pd.to_numeric(df[racol], errors="coerce")
    dec = pd.to_numeric(df[deccol], errors="coerce")
    name = df[namecol].astype(str).str.strip()
    if gradecol:
        grade = df[gradecol].astype(str).str.strip().replace({"": "?", "nan": "?", "--": "?"})
    else:
        grade = pd.Series(["?"] * n_raw, index=df.index)
    # Tag held-out catalogs explicitly so the merge can exclude them from training
    if cid in HELDOUT_IDS:
        grade = "heldout:" + grade.astype(str)
    out = pd.DataFrame({
        "name": name,
        "ra": ra.astype(float),
        "dec": dec.astype(float),
        "grade": grade.astype(str),
        "source": SOURCE,
        "ref": ref,
        "_vizier_id": cid,
    })
    frames.append(out)
    report.append((f"{cid}/{short}", n_raw, ref))
    print(f"  fetched {cid}/{short}: {n_raw} rows  [{ref}]")

allrows = pd.concat(frames, ignore_index=True)

# Drop null/non-finite ra/dec
finite = np.isfinite(allrows["ra"].to_numpy()) & np.isfinite(allrows["dec"].to_numpy())
allrows = allrows[finite].copy()

# Dedupe: same object can appear across catalogs (and Dawes lists one row per image).
# Dedupe by rounded coordinate, preferring a non-"?" / non-empty grade.
allrows["_key"] = (allrows["ra"].round(4).astype(str) + "_" + allrows["dec"].round(4).astype(str))
def _grade_rank(g):
    s = str(g)
    if s in ("?", "", "heldout:?"):
        return 0
    return 1
allrows["_grank"] = allrows["grade"].map(_grade_rank)
allrows = (allrows.sort_values(["_key", "_grank"], ascending=[True, False])
                  .drop_duplicates("_key", keep="first"))

n_before_cut = len(allrows)
print(f"\nROW COUNT before dec cut (finite + deduped): {n_before_cut}")

# Keep only DR11-south footprint
south = allrows[allrows["dec"] <= DEC_MAX].copy()
n_after_cut = len(south)
print(f"ROW COUNT after dec<= {DEC_MAX} cut: {n_after_cut}")

# Final exact column order
final = south[["name", "ra", "dec", "grade", "source", "ref"]].reset_index(drop=True)

# sanity dtypes
final["ra"] = final["ra"].astype(float)
final["dec"] = final["dec"].astype(float)
for c in ("name", "grade", "source", "ref"):
    final[c] = final[c].astype(str)

final.to_parquet(OUT, index=False)
print(f"\nWrote {OUT}  ({len(final)} rows)")

print("\nGrade breakdown (post-cut):")
print(final["grade"].value_counts().to_string())
print("\nPer-catalog (pre-dedup) fetched:")
for nm, n, ref in report:
    print(f"  {nm}: {n}")
print(f"\nDEC range: [{final['dec'].min():.3f}, {final['dec'].max():.3f}]")
