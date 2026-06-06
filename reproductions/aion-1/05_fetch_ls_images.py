"""
05 -- Fetch Legacy Survey g,r,i,z image cutouts for PROVABGS galaxies (task 1, +image).

MMU legacysurvey has no RA/Dec join key, so we pull DR10 cutouts directly from
the Legacy Survey service by position (the AION-tutorial path), via _ls_cutout.
By default fetches the spectrum-overlap subset (so the phot+image+spec headline
config is available on the same galaxies); pass --index all (optionally
--subsample N) for a larger image-only set.

Outputs (data/raw/provabgs/): image_flux.npy (M,4,160,160) + image_index.npy
(the PROVABGS rows with a successful cutout).

Run: HF_HOME=... python 05_fetch_ls_images.py --index spec [--subsample 20000]
"""

import argparse

import numpy as np

import _config as C
import _ls_cutout as LS

RAW = C.RAW / "provabgs"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="spec", choices=["spec", "all"])
    ap.add_argument("--subsample", type=int, default=0)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--size", type=int, default=160)
    args = ap.parse_args()
    C.seed_everything()

    ra = np.load(RAW / "ra.npy")
    dec = np.load(RAW / "dec.npy")
    if args.index == "spec":
        idx = np.load(RAW / "spec_index.npy")
    else:
        idx = np.arange(len(ra))
    if args.subsample and args.subsample < len(idx):
        rng = np.random.default_rng(C.SEED)
        idx = np.sort(rng.choice(idx, args.subsample, replace=False))

    coords = list(zip(ra[idx], dec[idx]))
    print(f"fetching {len(coords)} LS DR10 cutouts (size {args.size}) ...")
    arrs, ok = LS.fetch_many(coords, layer="ls-dr10", size=args.size, workers=args.workers)
    good_idx = idx[ok]
    imgs = np.stack([arrs[i] for i in range(len(arrs)) if ok[i]]).astype(np.float32)
    np.save(RAW / "image_flux.npy", imgs)
    np.save(RAW / "image_index.npy", good_idx)
    print(f"LS_IMAGES_OK saved {imgs.shape} for {len(good_idx)}/{len(idx)} galaxies "
          f"({100*ok.mean():.1f}% success)")


if __name__ == "__main__":
    main()
