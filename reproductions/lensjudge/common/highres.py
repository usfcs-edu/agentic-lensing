"""common/highres.py — higher-resolution coverage resolver for the escalate grader (B1).

The README's strongest finding: ambiguous DESI grade-C candidates flip to A/B at Euclid
0.1" (mean p_lens 0.05 -> 0.90) — the wall is resolution, not the algorithm. The escalate
grader (imaging/grader_escalate.py) re-grades ambiguous tier-1 candidates at higher
resolution WHEN coverage exists. This resolver maps a DESI candidate (name / ra / dec) to an
available high-res object.

Currently wired: Euclid Q1 (local "Strong Lensing Discovery Engine" cutouts). HSC PDR3 and
the Euclid SAS network cutout service are future sources (plan B1). **Degrades gracefully**:
returns None when the Euclid catalog/cutouts are not staged (they are not, in this checkout)
or the position has no coverage — so escalate mode is a safe no-op on the DECaLS-south
footprint, which has ~no Euclid Q1 overlap.
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


def resolve_highres(name, ra, dec, radius_arcsec: float = 2.0) -> Optional[dict]:
    """Return {survey, id_str[, sep_arcsec]} for an available high-res cutout, or None.

    Order: (1) the candidate name IS a local Euclid id; (2) positional match to the Euclid Q1
    catalog AND the matched object's FITS are on disk. Anything else -> None (no coverage)."""
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
