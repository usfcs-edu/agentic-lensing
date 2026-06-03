"""
Legacy Survey image-cutout fetcher.

MMU's `legacysurvey` dataset has no RA/Dec join key, so for samples defined by
sky position (GZ10 morphology, GZ-DECaLS retrieval, HSC lenses, PROVABGS image
config) we pull g,r,i,z FITS cutouts straight from the Legacy Survey cutout
service -- the same source the AION Tutorial uses
(`legacysurvey.org/viewer/cutout.fits?ra=..&dec=..&layer=ls-dr10`). Results are
cached on disk per (ra,dec,layer,size) and fetched with a thread pool. Returns a
(4, H, W) float32 array in g,r,i,z order, ready for
``aion.modalities.LegacySurveyImage(bands=['DES-G','DES-R','DES-I','DES-Z'])``.
"""

from __future__ import annotations

import hashlib
import io
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

import _config as C

CACHE = C.RAW / "ls_cutouts"
CACHE.mkdir(parents=True, exist_ok=True)
_BASE = "https://www.legacysurvey.org/viewer/cutout.fits"


def _key(ra, dec, layer, size, pixscale):
    s = f"{ra:.6f}_{dec:.6f}_{layer}_{size}_{pixscale}"
    return hashlib.md5(s.encode()).hexdigest()


def fetch_one(ra, dec, layer="ls-dr10", size=160, pixscale=0.262, bands="griz",
              retries=3, timeout=30):
    """Return (4,H,W) float32 g,r,i,z cutout, or None on failure. Cached."""
    import requests
    from astropy.io import fits

    cpath = CACHE / (_key(ra, dec, layer, size, pixscale) + ".npy")
    if cpath.exists():
        try:
            return np.load(cpath)
        except Exception:
            cpath.unlink(missing_ok=True)
    url = (f"{_BASE}?ra={ra}&dec={dec}&layer={layer}&pixscale={pixscale}"
           f"&size={size}&bands={bands}")
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=timeout)
            if r.status_code == 200 and r.content[:6] == b"SIMPLE":
                with fits.open(io.BytesIO(r.content)) as hdul:
                    arr = np.asarray(hdul[0].data, dtype=np.float32)  # (4,H,W)
                if arr.ndim == 3 and arr.shape[0] == 4:
                    np.save(cpath, arr)
                    return arr
                return None
            time.sleep(1.0 + attempt)
        except Exception:
            time.sleep(1.0 + attempt)
    return None


def fetch_many(coords, layer="ls-dr10", size=160, pixscale=0.262, workers=16,
               progress=True):
    """coords: iterable of (ra,dec). Returns (list_of_arrays_or_None, ok_mask)."""
    coords = list(coords)
    results = [None] * len(coords)

    def _job(i):
        ra, dec = coords[i]
        results[i] = fetch_one(ra, dec, layer=layer, size=size, pixscale=pixscale)
        return i

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for _ in ex.map(_job, range(len(coords))):
            done += 1
            if progress and done % 500 == 0:
                ok = sum(r is not None for r in results)
                print(f"  cutouts {done}/{len(coords)} ok={ok}", flush=True)
    ok_mask = np.array([r is not None for r in results])
    return results, ok_mask
