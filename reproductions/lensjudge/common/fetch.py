"""Acquire a (3,101,101) grz cube for a candidate: on-disk first, else legacysurvey.

On-disk lookup covers the ~2,706 graded Storfer/Inchausti cutouts and the ~65K
random-galaxy negatives already materialized by the inchausti-2025 reproduction.
For anything else (e.g. the 2,769 Grade-D human-rejects, whose cutouts were never
downloaded) we hit the legacysurvey fits-cutout endpoint at the exact geometry the
training cutouts use (size=101, pixscale=0.262, bands=grz, layer ls-dr9/ls-dr10) —
the same call as reproductions/inchausti-2025/12_download_candidate_cutouts.py —
and cache the result under cache/cubes/.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import requests
from astropy.io import fits

from lensjudge import config

BASE = "https://www.legacysurvey.org/viewer"
TIMEOUT, RETRIES, RETRY_BACKOFF, RATELIMIT_BACKOFF = 60, 5, 4.0, 30.0
MAX_FETCH_WALL = 120.0   # total wall-clock cap per candidate across retries (graceful give-up)
_CUBE_CACHE = config.CACHE / "cubes"


def _read_fits_cube(path: Path) -> np.ndarray | None:
    try:
        with fits.open(path) as h:
            cube = np.asarray(h[0].data, dtype=np.float32)
    except Exception:
        return None
    if cube is None or cube.shape != config.CUTOUT_SHAPE:
        return None
    return np.nan_to_num(cube, nan=0.0, posinf=0.0, neginf=0.0)


def on_disk_path(name: str, survey: str | None = None) -> Path | None:
    """Return an existing on-disk FITS path for this candidate name, if any."""
    cands = []
    if survey in config.CUTOUT_DIRS:
        cands.append(config.CUTOUT_DIRS[survey] / f"{name}.fits")
    cands += [d / f"{name}.fits" for d in config.CUTOUT_DIRS.values()]
    cands.append(config.NEG_RANDOM_DIR / f"{name}.fits")
    for p in cands:
        if p.exists() and p.stat().st_size > 256:
            return p
    return None


def _endpoint_url(ra: float, dec: float, layer: str) -> str:
    return (f"{BASE}/fits-cutout?ra={ra:.6f}&dec={dec:.6f}"
            f"&size={config.SIZE_PIX}&layer={layer}&pixscale={config.PIXSCALE}&bands=grz")


def fetch_endpoint(ra: float, dec: float, layer: str = "ls-dr10",
                   cache_key: str | None = None) -> np.ndarray | None:
    """Download a grz cutout from legacysurvey; cache by key. Returns the cube."""
    _CUBE_CACHE.mkdir(parents=True, exist_ok=True)
    key = cache_key or f"{layer}_{ra:.5f}_{dec:+.5f}"
    cached = _CUBE_CACHE / f"{key}.fits"
    if cached.exists() and cached.stat().st_size > 256:
        cube = _read_fits_cube(cached)
        if cube is not None:
            return cube
    url = _endpoint_url(ra, dec, layer)
    t0 = time.time()
    for attempt in range(1, RETRIES + 1):
        if time.time() - t0 > MAX_FETCH_WALL:   # graceful give-up instead of an unbounded slow tail
            break
        try:
            r = requests.get(url, timeout=TIMEOUT, stream=True)
            if r.status_code == 429:
                if time.time() - t0 > MAX_FETCH_WALL - RATELIMIT_BACKOFF:
                    break
                time.sleep(RATELIMIT_BACKOFF)
                continue
            r.raise_for_status()
            tmp = cached.with_suffix(".tmp")
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=64 * 1024):
                    if chunk:
                        f.write(chunk)
            if tmp.stat().st_size < 256:   # HTML error page
                tmp.unlink(missing_ok=True)
                raise RuntimeError("cutout too small (off-footprint?)")
            tmp.rename(cached)
            return _read_fits_cube(cached)
        except Exception:
            if attempt < RETRIES and time.time() - t0 < MAX_FETCH_WALL:
                time.sleep(RETRY_BACKOFF * attempt)
    return None


def get_cube(name: str | None = None, ra: float | None = None, dec: float | None = None,
             survey: str = "storfer") -> np.ndarray | None:
    """Resolve a candidate to a cube: on-disk by name, else endpoint by RA/Dec.

    ``survey`` is the catalog key ('storfer'->ls-dr9, 'inchausti'->ls-dr10) or a
    raw layer string ('ls-dr9'/'ls-dr10').
    """
    if name:
        p = on_disk_path(name, survey if survey in config.CUTOUT_DIRS else None)
        if p is not None:
            cube = _read_fits_cube(p)
            if cube is not None:
                return cube
    if ra is not None and dec is not None:
        layer = config.SURVEY_LAYER.get(survey, survey if survey.startswith("ls-") else "ls-dr10")
        return fetch_endpoint(float(ra), float(dec), layer=layer, cache_key=name)
    return None
