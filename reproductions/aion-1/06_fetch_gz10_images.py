"""
06 -- Fetch Legacy Survey g,r,i,z cutouts for Galaxy10 DECaLS (tasks 4, 7, 8).

The astroNN Galaxy10_DECals.h5 carries ra/dec/ans (10-class morphology label)
for 17,736 galaxies but only RGB images; AION ingests 4-band g,r,i,z FITS, so we
re-fetch cutouts from the Legacy Survey service by position (same path as the
PROVABGS image config, via _ls_cutout). Galaxy10 DECaLS cutouts are 256x256 at
0.262"/pix in the source; we request size 160 to match our PROVABGS image config
and AION's LegacySurveyImage expectations.

Outputs (data/raw/gz10/): labels.npy (all), ra.npy, dec.npy,
image_flux.npy (M,4,160,160) + image_index.npy (rows with a successful cutout).

Run: python 06_fetch_gz10_images.py [--subsample N] [--workers 6]
"""

import argparse

import h5py
import numpy as np

import _config as C
import _ls_cutout as LS

RAW = C.RAW / "gz10"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subsample", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--size", type=int, default=160)
    args = ap.parse_args()
    C.seed_everything()

    with h5py.File(RAW / "Galaxy10_DECals.h5", "r") as f:
        ra = f["ra"][:].astype(np.float64)
        dec = f["dec"][:].astype(np.float64)
        ans = f["ans"][:].astype(np.int64)
    np.save(RAW / "labels.npy", ans)
    np.save(RAW / "ra.npy", ra)
    np.save(RAW / "dec.npy", dec)

    idx = np.arange(len(ra))
    if args.subsample and args.subsample < len(idx):
        rng = np.random.default_rng(C.SEED)
        # stratified across the 10 morphology classes so every class survives
        per = args.subsample // 10
        chosen = []
        for c in range(10):
            ci = idx[ans == c]
            take = min(per, len(ci))
            chosen.append(rng.choice(ci, take, replace=False))
        idx = np.sort(np.concatenate(chosen))

    coords = list(zip(ra[idx], dec[idx]))
    print(f"fetching {len(coords)} GZ10 LS DR10 cutouts (size {args.size}) ...")
    arrs, ok = LS.fetch_many(coords, layer="ls-dr10", size=args.size,
                             workers=args.workers)
    good_idx = idx[ok]
    imgs = np.stack([arrs[i] for i in range(len(arrs)) if ok[i]]).astype(np.float32)
    np.save(RAW / "image_flux.npy", imgs)
    np.save(RAW / "image_index.npy", good_idx)
    print(f"GZ10_IMAGES_OK saved {imgs.shape} for {len(good_idx)}/{len(idx)} "
          f"({100*ok.mean():.1f}% success)")


if __name__ == "__main__":
    main()
