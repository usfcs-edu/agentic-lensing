#!/usr/bin/env python3
"""
02_reproject_and_reference.py  --  Sheu+2023 §3.1 reprojection + reference build

For each band we:
  1. define a common 801x801 @ 0.262"/pix tangent-plane WCS centred on the lens
     (the paper's 216.27" cutout grid);
  2. reproject every exposure cutout onto that grid with `reproject`
     (interp / exact flux-conserving) -- the role Sheu+2023 give to Montage;
  3. build the per-band REFERENCE as the *median* coadd across epochs, which
     suppresses any transient and most cosmic rays / CCD artifacts (paper §3.1);
  4. also record a per-pixel MAD noise map of the reference.

We additionally rescale every exposure to a common photometric zeropoint using
its MAGZERO header (InstCal frames are already in nanomaggies w/ MAGZERO, so the
rescale is typically ~1; we apply it for safety against mixed reductions).

Outputs
-------
  data/grid_wcs.txt                         the common WCS (header text)
  data/reproj/<band>/<stem>.fits            reprojected science frame (+ footprint)
  data/reference_<band>.fits                median-coadd reference image
  data/reference_<band>_noise.fits          MAD noise map of the reference
  data/reproj_manifest.csv                  reprojected-frame bookkeeping
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from reproject import reproject_interp

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
EXP = DATA / "exposures"
REPROJ = DATA / "reproj"
REPROJ.mkdir(parents=True, exist_ok=True)

CAND_RA = 38.0655
CAND_DEC = -24.4942
PIXSCALE = 0.262 / 3600.0  # deg/pix
NPIX = 801


def common_wcs(ra: float, dec: float) -> WCS:
    w = WCS(naxis=2)
    w.wcs.crpix = [(NPIX + 1) / 2, (NPIX + 1) / 2]
    w.wcs.crval = [ra, dec]
    w.wcs.cdelt = [-PIXSCALE, PIXSCALE]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    return w


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ra", type=float, default=CAND_RA)
    ap.add_argument("--dec", type=float, default=CAND_DEC)
    ap.add_argument("--bands", nargs="+", default=["g", "r", "z"])
    args = ap.parse_args()

    meta = pd.read_csv(DATA / "exposure_manifest.csv")
    wgrid = common_wcs(args.ra, args.dec)
    hdr = wgrid.to_header()
    (DATA / "grid_wcs.txt").write_text(hdr.tostring(sep="\n"))

    rows = []
    for band in args.bands:
        sub = meta[meta.band == band]
        if len(sub) == 0:
            print(f"[{band}] no exposures; skip")
            continue
        bdir = REPROJ / band
        bdir.mkdir(exist_ok=True)
        stack = []
        for _, r in sub.iterrows():
            img = Path(r["img"])
            if not img.exists():
                continue
            with fits.open(img) as hd:
                data = hd[0].data.astype("float32")
                src_wcs = WCS(hd[0].header)
                magzero = hd[0].header.get("MAGZERO", np.nan)
            # photometric rescale to a fixed zeropoint of 30 (nanomaggie convention)
            if np.isfinite(magzero) and magzero > 0:
                data = data * 10 ** (-0.4 * (magzero - 30.0))
            out, foot = reproject_interp((data, src_wcs), wgrid,
                                         shape_out=(NPIX, NPIX))
            out = np.where(foot > 0, out, np.nan).astype("float32")
            outp = bdir / f"{r['stem']}.fits"
            fits.PrimaryHDU(out, hdr).writeto(outp, overwrite=True)
            stack.append(out)
            rows.append(dict(band=band, stem=r["stem"], mjd=r["mjd"],
                             reproj=str(outp)))
        if not stack:
            continue
        cube = np.stack(stack, axis=0)
        # median coadd reference: suppresses transients & CRs (paper §3.1)
        ref = np.nanmedian(cube, axis=0).astype("float32")
        # MAD-based per-pixel noise of the reference stack
        mad = np.nanmedian(np.abs(cube - ref[None]), axis=0) * 1.4826
        fits.PrimaryHDU(ref, hdr).writeto(DATA / f"reference_{band}.fits",
                                          overwrite=True)
        fits.PrimaryHDU(mad.astype("float32"), hdr).writeto(
            DATA / f"reference_{band}_noise.fits", overwrite=True)
        print(f"[{band}] reprojected {len(stack)} frames -> "
              f"median reference (median sky={np.nanmedian(ref):.4f})")

    pd.DataFrame(rows).to_csv(DATA / "reproj_manifest.csv", index=False)
    print(f"[done] reproj manifest -> {DATA/'reproj_manifest.csv'}")


if __name__ == "__main__":
    main()
