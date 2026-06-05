"""
30 -- Fetch Galaxy Zoo 3D masks + Legacy Survey cutouts (task 5 segmentation).

GZ3D VAC (Masters+2021, SDSS DR17) provides per-galaxy crowd-vote masks for
spiral arms (HDU3) and bars (HDU4) on a 525x525 / 0.099"-pix MaNGA grid, plus
ra/dec (HDU5). We keep galaxies with non-empty spiral or bar votes, pull the
matching g,r,i,z LS cutout, and resample each mask onto the AION image grid
(the codec center-crops 160px@0.262" -> 96px = central ~25"; GZ3D central
~25" = 254px -> resize to 96). Binarize masks at vote>0.

Threaded (GZ3D FITS from data.sdss.org is not rate-limited) with on-disk FITS
caching and incremental saves so a partial run is usable.

Outputs (data/raw/gz3d/): image_flux.npy (M,4,160,160),
spiral_mask.npy / bar_mask.npy (M,96,96), radec.npy.

Run: HF_HOME=... python 30_fetch_gz3d.py [--n 2000] [--workers 12]
"""

import argparse
import gzip
import io
import os
import re
import threading

import numpy as np

import _config as C
import _ls_cutout as LS

OUT = C.RAW / "gz3d"
FITS_CACHE = OUT / "fits_cache"
OUT.mkdir(parents=True, exist_ok=True)
FITS_CACHE.mkdir(parents=True, exist_ok=True)
BASE = "https://data.sdss.org/sas/dr17/manga/morphology/galaxyzoo3d/v4_0_0/"
HDR = {"User-Agent": "Mozilla/5.0"}
GZ3D_PIXSCALE = 0.099
LS_CROP_ARCSEC = 96 * 0.262


def _resample_mask(mask):
    from scipy.ndimage import zoom
    n = mask.shape[0]
    crop_px = int(round(LS_CROP_ARCSEC / GZ3D_PIXSCALE))
    c = n // 2
    h = crop_px // 2
    sub = mask[c - h:c + h, c - h:c + h]
    z = zoom((sub > 0).astype(np.float32), 96.0 / sub.shape[0], order=1)
    z = z[:96, :96]
    if z.shape != (96, 96):
        pad = np.zeros((96, 96), np.float32)
        pad[:z.shape[0], :z.shape[1]] = z
        z = pad
    return (z > 0.5).astype(np.uint8)


def _process(fn):
    """Return (img(4,160,160), spiral(96,96), bar(96,96), ra, dec) or None."""
    import requests
    from astropy.io import fits
    cpath = FITS_CACHE / fn
    try:
        if cpath.exists():
            raw = cpath.read_bytes()
        else:
            raw = requests.get(BASE + fn, headers=HDR, timeout=60).content
            cpath.write_bytes(raw)
        with fits.open(io.BytesIO(gzip.decompress(raw))) as h:
            spiral = np.asarray(h[3].data, float)
            bar = np.asarray(h[4].data, float)
            if np.count_nonzero(spiral) == 0 and np.count_nonzero(bar) == 0:
                return None
            meta = h[5].data
            ra = float(meta["ra"][0]); dec = float(meta["dec"][0])
    except Exception:
        return None
    img = LS.fetch_one(ra, dec, layer="ls-dr10", size=160)
    if img is None:
        return None
    return (img.astype(np.float32), _resample_mask(spiral), _resample_mask(bar), ra, dec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    import requests
    from concurrent.futures import ThreadPoolExecutor

    html = requests.get(BASE, headers=HDR, timeout=60).text
    files = sorted(set(re.findall(r"gz3d_1-\d+_\d+_\d+\.fits\.gz", html)))
    rng = np.random.default_rng(C.SEED)
    rng.shuffle(files)
    print(f"GZ3D: {len(files)} files; targeting {args.n} image/mask pairs", flush=True)

    imgs, sp, bar, radec = [], [], [], []
    lock = threading.Lock()
    stop = threading.Event()

    def save():
        np.save(OUT / "image_flux.npy", np.stack(imgs))
        np.save(OUT / "spiral_mask.npy", np.stack(sp))
        np.save(OUT / "bar_mask.npy", np.stack(bar))
        np.save(OUT / "radec.npy", np.array(radec))

    def job(fn):
        if stop.is_set():
            return
        r = _process(fn)
        if r is None:
            return
        with lock:
            if len(imgs) >= args.n:
                stop.set(); return
            imgs.append(r[0]); sp.append(r[1]); bar.append(r[2]); radec.append((r[3], r[4]))
            n = len(imgs)
            if n % 200 == 0:
                save()
                print(f"  pairs {n}/{args.n} saved", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(job, files))
    if imgs:
        save()
    print(f"GZ3D_OK saved {len(imgs)} image/mask pairs", flush=True)


if __name__ == "__main__":
    main()
