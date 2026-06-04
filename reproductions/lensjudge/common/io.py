"""Load graded candidates, gold-label sets, and the imaging metadata LensJudge needs.

Primary source: the inchausti-2025 reproduction's per-candidate score CSVs, which
join the published consensus grade (A/B/C) to the frozen ensemble probabilities
(our_p_resnet/effnet/meta) and Tractor metadata, with FITS cutouts on disk.

Negatives:
  * Grade-D human-rejects  (the best hard negatives; cutouts not on disk -> fetch)
  * random-galaxy negatives (cutouts_fits_neg_dr9, already on disk)
Gold (confirmed, blind): Foundry-II 20 confirmed + 4 confirmed NON-lenses.
"""
from __future__ import annotations

import re

import pandas as pd

from lensjudge import config

# columns guaranteed present in candidate_scores_*.csv
_SCORE_COLS = ["name", "RA", "DEC", "grade", "probability", "region",
               "tractor_type", "our_p_resnet", "our_p_effnet", "our_p_meta", "our_p_avg"]


def load_candidates(which: str = "both", graded_only: bool = True) -> pd.DataFrame:
    """Graded A/B/C candidates with ML scores + on-disk cutout paths.

    Returns columns: name, ra, dec, grade, region, tractor_type,
    p_resnet, p_effnet, p_meta, p_avg, catalog, survey, fits_path.
    """
    keys = ("storfer", "inchausti") if which == "both" else (which,)
    frames = []
    for k in keys:
        raw = pd.read_csv(config.SCORE_CSV[k])
        # The two CSVs differ (Inchausti also carries the paper's own p_* columns and
        # lacks `region`); select explicitly off the always-present `our_p_*` columns
        # — the reproduction's frozen ensemble, consistent across both catalogs.
        df = pd.DataFrame({
            "name": raw["name"].astype(str),
            "ra": pd.to_numeric(raw["RA"], errors="coerce"),
            "dec": pd.to_numeric(raw["DEC"], errors="coerce"),
            "grade": raw["grade"].astype(str).str.upper().str[:1],
            "region": raw["region"] if "region" in raw.columns else config.SURVEY_LAYER[k],
            "tractor_type": raw.get("tractor_type", pd.Series(["?"] * len(raw))),
            "p_resnet": pd.to_numeric(raw["our_p_resnet"], errors="coerce"),
            "p_effnet": pd.to_numeric(raw["our_p_effnet"], errors="coerce"),
            "p_meta": pd.to_numeric(raw["our_p_meta"], errors="coerce"),
            "p_avg": pd.to_numeric(raw["our_p_avg"], errors="coerce"),
        })
        df["catalog"] = k
        df["survey"] = config.SURVEY_LAYER[k]
        df["fits_path"] = df["name"].map(lambda n: str(config.CUTOUT_DIRS[k] / f"{n}.fits"))
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    if graded_only:
        out = out[out["grade"].astype(str).str.upper().isin(list("ABC"))].reset_index(drop=True)
    keep = ["name", "ra", "dec", "grade", "region", "tractor_type",
            "p_resnet", "p_effnet", "p_meta", "p_avg", "catalog", "survey", "fits_path"]
    return out[[c for c in keep if c in out.columns]].copy()


_DESI_NAME = re.compile(r"DESI[-_ ]?(\d{1,3}\.\d+)\s*([+-]\d{1,2}\.\d+)")


def _radec_from_name(name: str):
    m = _DESI_NAME.search(str(name))
    if not m:
        return (None, None)
    return (float(m.group(1)), float(m.group(2)))


def load_grade_d(which: str = "both") -> pd.DataFrame:
    """Grade-D human-rejected hard negatives (cutouts fetched on demand).

    The raw files are messy Google-Sheets exports with two different layouts:
    Storfer has lowercase ``ra``/``dec`` columns; Inchausti encodes RA/Dec only in
    the DESI name (``DESI-<ra><±dec>``). We extract name + RA/Dec robustly (column
    first, then parse from the name) and capture the meta-probability where present.
    Returns: name, ra, dec, grade='D', p_meta, catalog, survey.
    """
    keys = ("storfer", "inchausti") if which == "both" else (which,)
    frames = []
    for k in keys:
        path = config.GRADE_D_RAW[k]
        if not path.exists():
            continue
        df = pd.read_csv(path, engine="python", on_bad_lines="skip")
        low = {str(c).lower().strip().split("\n")[0]: c for c in df.columns}
        ra_c = next((low[c] for c in low if c in ("ra", "ra_deg", "raj2000")), None)
        dec_c = next((low[c] for c in low if c in ("dec", "dec_deg", "dej2000", "de")), None)
        name_c = next((low[c] for c in low if c in ("name", "id", "object")), None)
        meta_c = next((low[c] for c in low
                       if c in ("meta-learner probability", "probability")), None)

        names = df[name_c].astype(str) if name_c else pd.Series([""] * len(df))
        if ra_c is not None and dec_c is not None:
            ra = pd.to_numeric(df[ra_c], errors="coerce")
            dec = pd.to_numeric(df[dec_c], errors="coerce")
        else:
            rd = names.map(_radec_from_name)
            ra = rd.map(lambda t: t[0]); dec = rd.map(lambda t: t[1])
        # fill any name gaps from RA/Dec, and any RA/Dec gaps from the name
        parsed = names.map(_radec_from_name)
        ra = ra.where(ra.notna(), parsed.map(lambda t: t[0]))
        dec = dec.where(dec.notna(), parsed.map(lambda t: t[1]))
        sub = pd.DataFrame({"name": names, "ra": pd.to_numeric(ra, errors="coerce"),
                            "dec": pd.to_numeric(dec, errors="coerce")})
        sub["p_meta"] = pd.to_numeric(df[meta_c], errors="coerce") if meta_c else float("nan")
        sub = sub[sub["name"].str.contains("DESI", na=False) | sub["ra"].notna()]
        sub = sub.dropna(subset=["ra", "dec"]).reset_index(drop=True)
        sub["grade"] = "D"
        sub["catalog"] = k
        sub["survey"] = config.SURVEY_LAYER[k]
        frames.append(sub)
    if not frames:
        return pd.DataFrame(columns=["name", "ra", "dec", "grade", "p_meta", "catalog", "survey"])
    return pd.concat(frames, ignore_index=True)


def load_foundry_ii_gold() -> pd.DataFrame:
    """Foundry-II confirmed lenses + the 4 confirmed NON-lenses (blind gold)."""
    p = config.FOUNDRY_II_DATA / "foundry_ii_master_comparison.csv"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)
