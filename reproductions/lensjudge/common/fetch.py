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

import os
import re
import time
from pathlib import Path

import numpy as np
import requests
from astropy.io import fits

from lensjudge import config

BASE = "https://www.legacysurvey.org/viewer"

# DESI candidate names encode the coordinates: DESI-<RA>{+|-}<Dec> in degrees
# (e.g. DESI-029.0755-01.1297 -> RA=29.0755, Dec=-1.1297). When a candidate is not on
# local disk and no RA/Dec were supplied, we recover them from the name so the cutout can
# still be fetched (a fallback for callers/tools that omit the optional coordinates).
_DESI_RADEC_RE = re.compile(r"(\d{1,3}\.\d+)\s*([+-]\d{1,2}\.\d+)")


def parse_radec_from_name(name: str | None) -> tuple[float | None, float | None]:
    if not name:
        return None, None
    m = _DESI_RADEC_RE.search(str(name))
    if not m:
        return None, None
    try:
        ra, dec = float(m.group(1)), float(m.group(2))
    except ValueError:
        return None, None
    if 0.0 <= ra <= 360.0 and -90.0 <= dec <= 90.0:
        return ra, dec
    return None, None
TIMEOUT, RETRIES, RETRY_BACKOFF, RATELIMIT_BACKOFF = 60, 5, 4.0, 30.0
MAX_FETCH_WALL = 120.0   # total wall-clock cap per candidate across retries (graceful give-up)
_CUBE_CACHE = config.CACHE / "cubes"


def _read_fits_cube(path: Path, shape: tuple = config.CUTOUT_SHAPE) -> np.ndarray | None:
    try:
        with fits.open(path) as h:
            cube = np.asarray(h[0].data, dtype=np.float32)
    except Exception:
        return None
    if cube is None or cube.shape != shape:
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


def _endpoint_url(ra: float, dec: float, layer: str, size: int = config.SIZE_PIX) -> str:
    return (f"{BASE}/fits-cutout?ra={ra:.6f}&dec={dec:.6f}"
            f"&size={size}&layer={layer}&pixscale={config.PIXSCALE}&bands=grz")


def fetch_endpoint(ra: float, dec: float, layer: str = "ls-dr10",
                   cache_key: str | None = None,
                   size: int = config.SIZE_PIX) -> np.ndarray | None:
    """Download a grz cutout from legacysurvey; cache by key. Returns the cube.

    ``size`` (px, same 0.262"/px scale) defaults to the standard 101px geometry so all
    existing callers/caches are untouched; non-default sizes get an ``_s{size}`` suffix
    on the auto cache key (callers passing an explicit cache_key own its uniqueness).
    """
    _CUBE_CACHE.mkdir(parents=True, exist_ok=True)
    key = cache_key or (f"{layer}_{ra:.5f}_{dec:+.5f}"
                        + (f"_s{size}" if size != config.SIZE_PIX else ""))
    shape = (3, size, size)
    cached = _CUBE_CACHE / f"{key}.fits"
    if cached.exists() and cached.stat().st_size > 256:
        cube = _read_fits_cube(cached, shape)
        if cube is not None:
            return cube
    url = _endpoint_url(ra, dec, layer, size)
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
            return _read_fits_cube(cached, shape)
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
        # deployment override: a single staged dir of {name}.fits (e.g. Perlmutter offline serving,
        # where cutouts are pre-staged and CUTOUT_DIRS' repo-relative paths don't apply).
        staged = os.environ.get("LENSJUDGE_CUTOUT_DIR")
        if staged:
            sp = Path(staged) / f"{name}.fits"
            if sp.exists():
                cube = _read_fits_cube(sp)
                if cube is not None:
                    return cube
        p = on_disk_path(name, survey if survey in config.CUTOUT_DIRS else None)
        if p is not None:
            cube = _read_fits_cube(p)
            if cube is not None:
                return cube
    # fallback: recover RA/Dec from a DESI-coordinate name when not supplied (e.g. a tool call
    # that omitted the optional ra/dec for an off-disk candidate).
    if ra is None or dec is None:
        pra, pdec = parse_radec_from_name(name)
        ra = ra if ra is not None else pra
        dec = dec if dec is not None else pdec
    if ra is not None and dec is not None:
        layer = config.SURVEY_LAYER.get(survey, survey if survey.startswith("ls-") else "ls-dr10")
        return fetch_endpoint(float(ra), float(dec), layer=layer, cache_key=name)
    return None


# --- WIDE context cutout (matched-inputs grader, parity Phase C3) -------------
# Human graders had a Legacy Surveys sky-viewer link and could zoom OUT for context;
# the matched grader stands that in with one wider cutout at the SAME pixel scale.

def wide_size(factor: int = 4) -> int:
    """Wide-cutout size in px: same center pixel, ``factor`` x the standard half-width
    (factor=4 -> 401 px = ~105" at 0.262"/px, comfortably under the endpoint's cap)."""
    return factor * (config.SIZE_PIX - 1) + 1


def get_wide_cube(name: str | None = None, ra: float | None = None, dec: float | None = None,
                  survey: str = "storfer", factor: int = 4) -> np.ndarray | None:
    """Fetch a WIDE context grz cube (~4x the standard field, same 0.262"/px scale).

    Wide cutouts are never staged on disk, so this always resolves RA/Dec (from the args
    or a DESI-coordinate name) and hits the fits-cutout endpoint. Cached under a distinct
    ``*_wide{size}`` key so the standard 101px cubes/caches are untouched. Callers that
    batch this must stay polite (<=3 workers against legacysurvey.org)."""
    if ra is None or dec is None:
        pra, pdec = parse_radec_from_name(name)
        ra = ra if ra is not None else pra
        dec = dec if dec is not None else pdec
    if ra is None or dec is None:
        return None
    layer = config.SURVEY_LAYER.get(
        survey, survey if str(survey).startswith("ls-") else "ls-dr10")
    size = wide_size(factor)
    key = (f"{name}_wide{size}" if name
           else f"{layer}_{float(ra):.5f}_{float(dec):+.5f}_wide{size}")
    return fetch_endpoint(float(ra), float(dec), layer=layer, cache_key=key, size=size)
