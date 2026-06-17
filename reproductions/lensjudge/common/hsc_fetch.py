"""common/hsc_fetch.py — fetch HSC-SSP PDR3 coadd cutouts from the public das_cutout service.

The README/v2 result is that the imaging wall is *resolution*: DESI grade-C blobs resolve into
clear arcs/non-arcs at higher resolution. Euclid Q1 (0.1") is the sharpest source but its DESI
overlap is tiny; HSC-SSP Wide (~0.6" seeing, 0.168"/px) covers far more equatorial sky, giving an
intermediate tier-2 rung. This client fetches grizy coadd cutouts for an (ra, dec) from the public
PDR3 image-cutout CGI.

Verified end-to-end (2026-06): HTTPS GET + HTTP Basic Auth (HSC_USER/HSC_PASSWORD) returns one FITS
per filter, the image in HDU[1], 0.168"/px; an out-of-footprint position returns HTTP 404 (clean
graceful-None). Cutouts are cached under config.CACHE/hsc/ so each position is fetched once.

**Credentials are env-only** (HSC_USER / HSC_PASSWORD); they are never written to disk or committed.
Missing credentials -> returns None (escalate mode stays a safe no-op).
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np
import requests
from astropy.io import fits
from requests.auth import HTTPBasicAuth

from lensjudge import config

BASE = "https://hsc-release.mtk.nao.ac.jp/das_cutout/pdr3/cgi-bin/cutout"
FILTER = {"g": "HSC-G", "r": "HSC-R", "i": "HSC-I", "z": "HSC-Z", "y": "HSC-Y"}
PRIMARY = ("i", "r")          # if neither primary band is covered -> treat as no coverage
PIXSCALE = 0.168              # arcsec/px (confirmed from the returned WCS)
HSC_CACHE = config.CACHE / "hsc"


def have_credentials() -> bool:
    return bool(os.environ.get("HSC_USER") and os.environ.get("HSC_PASSWORD"))


def _auth() -> Optional[HTTPBasicAuth]:
    u, p = os.environ.get("HSC_USER"), os.environ.get("HSC_PASSWORD")
    return HTTPBasicAuth(u, p) if u and p else None


def _key(ra: float, dec: float) -> str:
    return f"{float(ra):.6f}_{float(dec):+.6f}"


def _read_image(path) -> Optional[np.ndarray]:
    """First 2D image HDU as a finite float32 array (HSC coadd image is in HDU[1])."""
    try:
        with fits.open(path) as h:
            for hd in h:
                if hd.data is not None and getattr(hd.data, "ndim", 0) == 2:
                    return np.nan_to_num(np.asarray(hd.data, np.float32))
    except Exception:
        return None
    return None


def fetch_hsc_cutout(ra, dec, bands: str = "grizy", sw_arcsec: float = 14.0,
                     rerun: str = "pdr3_wide", timeout: int = 40) -> Optional[dict]:
    """Return ``{band_letter: 2D float array}`` for the covered bands, or ``None``.

    None means: no credentials, or the position is outside the HSC PDR3 footprint (a 404 on the
    primary i/r band), or a network error. Per-band FITS are cached under
    ``cache/hsc/{rerun}/{ra}_{dec}/{band}.fits`` so a position is fetched at most once per band.
    """
    auth = _auth()
    if auth is None or ra is None or dec is None:
        return None
    d = HSC_CACHE / rerun / _key(ra, dec)
    d.mkdir(parents=True, exist_ok=True)
    sw = f"{float(sw_arcsec) / 60.0:.4f}amin"
    out: dict = {}
    for b in bands:
        if b not in FILTER:
            continue
        fp = d / f"{b}.fits"
        if fp.exists():
            arr = _read_image(fp)
            if arr is not None:
                out[b] = arr
            continue
        params = dict(ra=f"{float(ra):.6f}", dec=f"{float(dec):.6f}", sw=sw, sh=sw,
                      type="coadd", image="on", mask="off", variance="off",
                      filter=FILTER[b], rerun=rerun)
        try:
            r = requests.get(BASE, params=params, auth=auth, timeout=timeout)
        except Exception:
            # network error: return whatever bands we already have, else no coverage
            return out or None
        if r.status_code != 200 or r.content[:9] != b"SIMPLE  =":
            # 404 / non-FITS = no coverage for this band. If a PRIMARY band is uncovered the
            # position is effectively out of footprint -> no usable cutout.
            if b in PRIMARY and not out:
                return None
            continue
        fp.write_bytes(r.content)
        arr = _read_image(fp)
        if arr is not None:
            out[b] = arr
    # require at least one primary band for a usable luminance/color render
    if not any(b in out for b in PRIMARY):
        return None
    return out or None
