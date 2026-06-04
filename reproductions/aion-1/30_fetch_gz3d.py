"""
30 -- Fetch Galaxy Zoo 3D masks + Legacy Survey cutouts (task 5 segmentation).

GZ3D VAC (Masters+2021, SDSS DR17) provides per-galaxy crowd-vote masks for
spiral arms (HDU3) and bars (HDU4) on a 525x525 / 0.099"-pix MaNGA grid, plus
ra/dec (HDU5). We keep galaxies with non-empty spiral or bar votes, pull the
matching g,r,i,z LS cutout, and resample each mask onto the AION image grid
(the codec center-crops 160px@0.262" -> 96px = central ~25"; GZ3D central
~25" = 254px -> resize to 96). Binarize masks at vote>0.

Outputs (data/raw/gz3d/): image_flux.npy (M,4,160,160),
spiral_mask.npy / bar_mask.npy (M,96,96), radec.npy.

Run: HF_HOME=... python 30_fetch_gz3d.py [--n 2800] [--workers 3]
"""

import argparse
import gzip
import io
import re

import numpy as np

import _config as C
import _ls_cutout as LS

OUT = C.RAW / "gz3d"
OUT.mkdir(parents=True, exist_ok=True)
BASE = "https://data.sdss.org/sas/dr17/manga/morphology/galaxyzoo3d/v4_0_0/"
HDR = {"User-Agent": "Mozilla/5.0"}
GZ3D_PIXSCALE = 0.099
LS_CROP_ARCSEC = 96 * 0.262  # AION sees the central 96px @ 0.262" of the cutout


def _resample_mask(mask):
    """Central-crop the GZ3D 525x525 mask to the AION FOV, resize to 96, binarize."""
    from scipy.ndimage import zoom
    n = mask.shape[0]
    crop_px = int(round(LS_CROP_ARCSEC / GZ3D_PIXSCALE))  # ~254
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2800)
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()

    import requests
    from astropy.io import fits

    html = requests.get(BASE, headers=HDR, timeout=60).text
    files = sorted(set(re.findall(r"gz3d_1-\d+_\d+_\d+\.fits\.gz", html)))
    print(f"GZ3D: {len(files)} files in VAC")

    rng = np.random.default_rng(C.SEED)
    rng.shuffle(files)
    metas, spirals, bars = [], [], []
    radec = []
    for fn in files:
        if len(metas) >= args.n:
            break
        try:
            raw = requests.get(BASE + fn, headers=HDR, timeout=60).content
            with fits.open(io.BytesIO(gzip.decompress(raw))) as h:
                spiral = np.asarray(h[3].data, float)
                bar = np.asarray(h[4].data, float)
                if np.count_nonzero(spiral) == 0 and np.count_nonzero(bar) == 0:
                    continue
                meta = h[5].data
                ra = float(meta["ra"][0]); dec = float(meta["dec"][0])
        except Exception:
            continue
        img = LS.fetch_one(ra, dec, layer="ls-dr10", size=160)
        if img is None:
            continue
        metas.append(img.astype(np.float32))
        spirals.append(_resample_mask(spiral))
        bars.append(_resample_mask(bar))
        radec.append((ra, dec))
        if len(metas) % 100 == 0:
            print(f"  pairs {len(metas)}/{args.n}", flush=True)

    np.save(OUT / "image_flux.npy", np.stack(metas))
    np.save(OUT / "spiral_mask.npy", np.stack(spirals))
    np.save(OUT / "bar_mask.npy", np.stack(bars))
    np.save(OUT / "radec.npy", np.array(radec))
    print(f"GZ3D_OK saved {len(metas)} image/mask pairs; "
          f"spiral coverage {np.mean([m.mean() for m in spirals]):.3f}, "
          f"bar {np.mean([m.mean() for m in bars]):.3f}")


if __name__ == "__main__":
    main()
