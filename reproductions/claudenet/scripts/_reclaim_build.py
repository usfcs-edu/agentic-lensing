import numpy as np
import pandas as pd
import os

DEC_CUT = 32.375
COLS = ["name", "ra", "dec", "grade", "source", "ref"]

frames = []
stats = {}

def finalize(df, tag):
    """Coerce ra/dec to float, drop null/non-finite, return cleaned df + counts."""
    df = df.copy()
    df["ra"] = pd.to_numeric(df["ra"], errors="coerce")
    df["dec"] = pd.to_numeric(df["dec"], errors="coerce")
    n_raw = len(df)
    df = df[np.isfinite(df["ra"]) & np.isfinite(df["dec"])]
    n_finite = len(df)
    df["name"] = df["name"].astype("string").fillna("").astype(str)
    df["grade"] = df["grade"].astype("string").fillna("?").replace("", "?").astype(str)
    df["source"] = df["source"].astype(str)
    df["ref"] = df["ref"].astype("string").fillna("").astype(str)
    df = df[COLS]
    n_south = int((df["dec"] <= DEC_CUT).sum())
    stats[tag] = {"n_fetched": n_raw, "n_finite": n_finite, "n_south": n_south}
    return df

# 1. inchausti positives_literature
f1 = "/home2/benson/git/agentic-lensing/reproductions/inchausti-2025/data/positives_literature.parquet"
d1 = pd.read_parquet(f1)
df1 = pd.DataFrame({
    "name": d1["Name"],
    "ra": d1["RA"],
    "dec": d1["DEC"],
    "grade": "?",                      # no grade/confidence column present
    "source": d1["source"].astype(str),
    "ref": d1["source"].astype(str),   # VizieR source tag is the citation
})
frames.append(finalize(df1, "inchausti_positives_literature"))

# 2. Euclid Q1 discovery engine
f2 = "/home2/benson/git/agentic-lensing/reproductions/euclid-q1/data/raw/q1_discovery_engine_lens_catalog.csv"
d2 = pd.read_csv(f2)
ref2 = d2["references"].astype("string").fillna("Euclid Q1 Discovery Engine")
df2 = pd.DataFrame({
    "name": d2["id_str"],
    "ra": d2["right_ascension"],
    "dec": d2["declination"],
    "grade": d2["grade"],
    "source": "euclid_q1",
    "ref": ref2,
})
frames.append(finalize(df2, "euclid_q1"))
# A/B breakdown for euclid (pre-dec-cut and south)
e_clean = frames[-1]
stats["euclid_q1"]["AB_all"] = int(d2["grade"].isin(["A", "B"]).sum())
stats["euclid_q1"]["AB_south"] = int(e_clean[(e_clean["dec"] <= DEC_CUT) & e_clean["grade"].isin(["A", "B"])].shape[0])

# 3. SuGOHI / HSC
f3 = "/home2/benson/git/agentic-lensing/reproductions/aion-1/data/raw/sugohi/lens_radec.parquet"
d3 = pd.read_parquet(f3)
df3 = pd.DataFrame({
    "name": d3["name"],
    "ra": d3["ra"],
    "dec": d3["dec"],
    "grade": d3["grade"],
    "source": "sugohi_hsc",
    "ref": d3["reference"].astype("string").fillna("SuGOHI"),
})
frames.append(finalize(df3, "sugohi_hsc"))

# 4. external SIMBAD/VizieR set
f4 = "/home2/benson/git/agentic-lensing/reproductions/claudenet/v3/external_lens_catalog.csv"
d4 = pd.read_csv(f4)
df4 = pd.DataFrame({
    "name": d4["name"],
    "ra": d4["RA"],
    "dec": d4["DEC"],
    "grade": "?",                      # no grade column
    "source": "external_simbad",
    "ref": d4["ref"].astype("string").fillna(""),
})
frames.append(finalize(df4, "external_simbad"))

# UNION
combined = pd.concat(frames, ignore_index=True)
n_before = len(combined)
combined_south = combined[combined["dec"] <= DEC_CUT].reset_index(drop=True)
n_after = len(combined_south)

print("=== row count before dec cut:", n_before)
print("=== row count after  dec cut (dec<=%.3f): %d" % (DEC_CUT, n_after))
print()
print("=== per-input-source stats ===")
for k, v in stats.items():
    print(k, v)
print()
print("=== combined grade_breakdown (south) ===")
print(combined_south["grade"].value_counts(dropna=False).to_dict())
print()
print("=== combined source_breakdown (south) ===")
print(combined_south["source"].value_counts(dropna=False).to_dict())

out = "/home2/benson/git/agentic-lensing/reproductions/claudenet/data/harvest/local_reclaim.parquet"
os.makedirs(os.path.dirname(out), exist_ok=True)
combined_south.to_parquet(out, index=False)
print()
print("WROTE", out, "rows=", len(combined_south))
# sanity re-read
rb = pd.read_parquet(out)
print("REREAD shape", rb.shape, "cols", list(rb.columns))
print("dec max after cut:", rb["dec"].max())
