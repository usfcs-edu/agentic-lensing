#!/usr/bin/env python3
"""
03_difference_imaging_b08.py  --  Sheu+2023 §3.2 image subtraction (Bramich 2008)

Implements the Bramich (2008) NON-PARAMETRIC difference-imaging kernel from
scratch (the paper's primary B08 path; SFFT is the optional GPU alternative).

B08 model
---------
Find a discrete convolution kernel K (delta-function basis -- one free parameter
per kernel pixel) and a spatially-flat differential background b such that

      science(x,y)  ~=  sum_{u,v} K(u,v) * ref(x-u, y-v)  +  b

is satisfied in a least-squares sense.  Because the model is LINEAR in the
kernel coefficients and background, the normal equations are linear:

      M c = d ,   c = [K.flatten(), b]

where column n of the design matrix is ref shifted by the (u,v) of kernel pixel
n (and a column of ones for b).  We solve with numpy.linalg.lstsq (float64),
optionally inverse-variance weighted by the science-frame weight map.  This is
exactly Bramich (2008) eq. 2-4 for a spatially-constant kernel; we tile the
image into a small grid of stamps to capture spatial variation of the PSF
(the "spatially varying kernel" of the paper) -- each stamp gets its own K, b.

The difference image is  D = science - (ref (x) K + b).

Outputs (per band, per epoch)
-----------------------------
  data/diff/<band>/<stem>_diff.fits     difference image (science - convolved ref)
  data/diff/<band>/<stem>_conv.fits     convolved+offset reference (model)
  data/diff_manifest.csv                bookkeeping (kernel sum ~ flux scaling, rms)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from numpy.lib.stride_tricks import sliding_window_view

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
REPROJ = DATA / "reproj"
DIFF = DATA / "diff"
DIFF.mkdir(parents=True, exist_ok=True)


def b08_kernel_solve(ref: np.ndarray, sci: np.ndarray, hw: int,
                     weight: np.ndarray | None = None):
    """Solve Bramich-2008 delta-basis kernel + flat bg over a single stamp.

    Returns (kernel (2hw+1)^2, background b, model = ref(x)K + b).
    Operates on the valid (interior) region where the full kernel footprint
    fits; pads the model back to ref shape.
    """
    ks = 2 * hw + 1
    ny, nx = ref.shape
    # sliding windows of ref: shape (ny-2hw, nx-2hw, ks, ks)
    win = sliding_window_view(ref, (ks, ks))
    vy, vx = win.shape[:2]
    # design matrix columns: each kernel tap = ref shifted; flip so it's a true
    # convolution (correlation of flipped kernel). Build A (Npix, ks*ks + 1).
    A = win.reshape(vy * vx, ks * ks)
    A = np.hstack([A, np.ones((A.shape[0], 1))])  # + background column
    b = sci[hw:hw + vy, hw:hw + vx].reshape(-1)
    good = np.isfinite(b) & np.all(np.isfinite(A), axis=1)
    if weight is not None:
        w = weight[hw:hw + vy, hw:hw + vx].reshape(-1)
        good &= np.isfinite(w) & (w > 0)
    A_ = A[good]
    b_ = b[good]
    if weight is not None:
        sw = np.sqrt(w[good])
        A_ = A_ * sw[:, None]
        b_ = b_ * sw
    if A_.shape[0] < A_.shape[1] + 5:
        return None
    coef, *_ = np.linalg.lstsq(A_.astype(np.float64), b_.astype(np.float64),
                               rcond=None)
    kern = coef[:ks * ks].reshape(ks, ks)
    bg = float(coef[-1])
    # full model over valid region, then pad
    model_valid = (win.reshape(vy * vx, ks * ks) @ coef[:ks * ks]).reshape(vy, vx) + bg
    model = np.full_like(ref, np.nan, dtype=np.float64)
    model[hw:hw + vy, hw:hw + vx] = model_valid
    return kern, bg, model


def difference_image(ref: np.ndarray, sci: np.ndarray, hw: int = 4,
                     ntile: int = 2, weight: np.ndarray | None = None):
    """Spatially-varying B08: tile the frame, solve a kernel per tile.

    ntile x ntile tiles; each tile solved on a margin-padded stamp so the kernel
    footprint is fully sampled. Returns (diff, model, mean_kernel_sum).
    """
    ny, nx = ref.shape
    model = np.full_like(ref, np.nan, dtype=np.float64)
    ksums = []
    ys = np.linspace(0, ny, ntile + 1).astype(int)
    xs = np.linspace(0, nx, ntile + 1).astype(int)
    for ti in range(ntile):
        for tj in range(ntile):
            y0, y1 = ys[ti], ys[ti + 1]
            x0, x1 = xs[tj], xs[tj + 1]
            # pad by hw so the kernel footprint of the tile interior is covered
            py0, py1 = max(0, y0 - hw), min(ny, y1 + hw)
            px0, px1 = max(0, x0 - hw), min(nx, x1 + hw)
            rstamp = ref[py0:py1, px0:px1]
            sstamp = sci[py0:py1, px0:px1]
            wstamp = weight[py0:py1, px0:px1] if weight is not None else None
            out = b08_kernel_solve(rstamp, sstamp, hw, wstamp)
            if out is None:
                continue
            kern, bg, mstamp = out
            ksums.append(float(np.nansum(kern)))
            # write back only the tile interior (offset within the padded stamp)
            oy0 = y0 - py0
            ox0 = x0 - px0
            model[y0:y1, x0:x1] = mstamp[oy0:oy0 + (y1 - y0),
                                         ox0:ox0 + (x1 - x0)]
    diff = sci - model
    return diff.astype("float32"), model.astype("float32"), float(np.nanmean(ksums))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bands", nargs="+", default=["g", "r", "z"])
    ap.add_argument("--hw", type=int, default=4, help="kernel half-width (px)")
    ap.add_argument("--ntile", type=int, default=2, help="spatial tiles per axis")
    args = ap.parse_args()

    rows = []
    for band in args.bands:
        ref_p = DATA / f"reference_{band}.fits"
        if not ref_p.exists():
            print(f"[{band}] no reference; run 02 first")
            continue
        ref = fits.getdata(ref_p).astype("float64")
        hdr = fits.getheader(ref_p)
        odir = DIFF / band
        odir.mkdir(exist_ok=True)
        frames = sorted((REPROJ / band).glob("*.fits"))
        for fp in frames:
            sci = fits.getdata(fp).astype("float64")
            diff, model, ksum = difference_image(ref, sci, args.hw, args.ntile)
            stem = fp.stem
            fits.PrimaryHDU(diff, hdr).writeto(odir / f"{stem}_diff.fits",
                                               overwrite=True)
            fits.PrimaryHDU(model, hdr).writeto(odir / f"{stem}_conv.fits",
                                                overwrite=True)
            rms = float(np.nanstd(diff))
            rows.append(dict(band=band, stem=stem, kernel_sum=ksum,
                             diff_rms=rms, diff=str(odir / f"{stem}_diff.fits")))
            print(f"[{band}] {stem}: kernel_sum={ksum:.3f} diff_rms={rms:.4f}")

    pd.DataFrame(rows).to_csv(DATA / "diff_manifest.csv", index=False)
    print(f"[done] diff manifest -> {DATA/'diff_manifest.csv'}")


if __name__ == "__main__":
    main()
