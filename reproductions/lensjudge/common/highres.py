"""common/highres.py — higher-resolution coverage resolver for the escalate grader (B1).

The README's strongest finding: ambiguous DESI grade-C candidates flip to A/B at Euclid
0.1" (mean p_lens 0.05 -> 0.90) — the wall is resolution, not the algorithm. The escalate
grader (imaging/grader_escalate.py) re-grades ambiguous tier-1 candidates at higher
resolution WHEN coverage exists. This resolver maps a DESI candidate (name / ra / dec) to an
available high-res object.

Wired sources, tried in priority order (default ``("euclid", "hsc")``):
  1. **Euclid Q1** (local "Strong Lensing Discovery Engine" cutouts, 0.1") — a local catalog lookup.
  2. **HSC-SSP PDR3** (0.168", ~0.6" seeing) — the public das_cutout service; a successful cutout
     fetch IS the coverage check (no local catalog), gated on HSC_USER/HSC_PASSWORD being set.

**Degrades gracefully**: returns None when neither source covers the position (Euclid cutouts not
staged, no HSC credentials, or out of footprint) — so escalate mode is a safe no-op (e.g. on the
DECaLS-south footprint, which has ~no Euclid Q1 overlap and partial HSC coverage).
"""
from __future__ import annotations

from functools import lru_cache
from typing import Optional

import numpy as np
import pandas as pd

from lensjudge.common import euclid


@lru_cache(maxsize=1)
def _euclid_catalog() -> Optional[pd.DataFrame]:
    cat = euclid.EUCLID_ROOT / "raw" / "q1_discovery_engine_lens_catalog.csv"
    if not cat.exists():
        return None
    df = pd.read_csv(cat)
    if not {"id_str", "right_ascension", "declination"} <= set(df.columns):
        return None
    return df


def _resolve_euclid(name, ra, dec, radius_arcsec: float) -> Optional[dict]:
    if name and euclid.obj_dir(str(name)) is not None:
        return {"survey": "euclid", "id_str": str(name), "sep_arcsec": 0.0}
    if ra is None or dec is None:
        return None
    df = _euclid_catalog()
    if df is None:
        return None
    dra = (df["right_ascension"].to_numpy(float) - float(ra)) * np.cos(np.radians(float(dec)))
    ddec = df["declination"].to_numpy(float) - float(dec)
    sep = np.hypot(dra, ddec) * 3600.0
    i = int(np.argmin(sep))
    if sep[i] <= radius_arcsec:
        idd = str(df.iloc[i]["id_str"])
        if euclid.obj_dir(idd) is not None:
            return {"survey": "euclid", "id_str": idd, "sep_arcsec": float(sep[i])}
    return None


def _resolve_hsc(name, ra, dec) -> Optional[dict]:
    """HSC PDR3 coverage probe: gated on credentials; a successful (cached) cutout fetch == coverage.
    Carries ra/dec so the HSC tier-2 grader (position-based) can re-render the same object."""
    if ra is None or dec is None:
        return None
    from lensjudge.common import hsc_fetch
    if not hsc_fetch.have_credentials():
        return None
    try:
        bands = hsc_fetch.fetch_hsc_cutout(float(ra), float(dec))
    except Exception:
        return None
    if not bands:
        return None
    return {"survey": "hsc", "id_str": f"{float(ra):.6f}_{float(dec):+.6f}",
            "ra": float(ra), "dec": float(dec), "sep_arcsec": 0.0}


def resolve_highres(name, ra, dec, radius_arcsec: float = 2.0,
                    surveys=("euclid", "hsc")) -> Optional[dict]:
    """Return {survey, id_str[, ra, dec, sep_arcsec]} for an available high-res cutout, or None.

    Tries each source in ``surveys`` order and returns the first hit: Euclid Q1 (local catalog +
    FITS on disk) then HSC PDR3 (live coverage probe via the cutout service, creds-gated). Anything
    else -> None (no coverage), keeping escalate mode a safe no-op."""
    for s in surveys:
        hit = _resolve_euclid(name, ra, dec, radius_arcsec) if s == "euclid" \
            else _resolve_hsc(name, ra, dec) if s == "hsc" else None
        if hit:
            return hit
    return None
