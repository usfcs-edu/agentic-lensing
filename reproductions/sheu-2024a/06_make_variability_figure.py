#!/usr/bin/env python3
"""
06_make_variability_figure.py  --  Sheu+2024a-style evidence panel

Builds the paper's Figure-4/8-style evidence for the target lensed quasar:
  * top row: the median-coadd REFERENCE (g/r/z) with the two posited lensed
    images A and B marked;
  * bottom row: a montage of B08 DIFFERENCE images at the epochs of strongest
    residual at the image positions, showing the over/under-subtraction
    (positive AND negative residuals) that is the signature of quasar
    variability -- exactly the "select single-epoch exposures and their
    difference images that exhibit variability" panels of the paper.

The over/under-subtraction at the SAME location across epochs (some epochs
brighter than the median coadd -> positive diff, others fainter -> negative
diff) is the difference-imaging evidence that complements the light-curve sigma.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import fits
from matplotlib.patches import Circle

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
FIGS = HERE / "figs"
DIFF = DATA / "diff"
PIXSCALE = 0.262
CAND_NAME = "DESI-038.0655-24.4942"


def zscale(a, lo=2, hi=98):
    a = a[np.isfinite(a)]
    if a.size == 0:
        return 0, 1
    return np.percentile(a, lo), np.percentile(a, hi)


def diff_residual_at(stem, band, x, y, r=3):
    dp = DIFF / band / f"{stem}_diff.fits"
    if not dp.exists():
        return np.nan
    d = fits.getdata(dp)
    ny, nx = d.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    ap = (np.hypot(xx - x, yy - y) <= r) & np.isfinite(d)
    return float(np.nansum(d[ap])) if ap.sum() else np.nan


def main():
    pos = pd.read_csv(DATA / "image_positions.csv")
    meta = pd.read_csv(DATA / "exposure_manifest.csv").set_index("stem")
    A = pos[pos.image == "A"].iloc[0]
    B = pos[pos.image == "B"].iloc[0]

    # find, per band, the epochs with the LARGEST |residual| at image A -> these
    # are the strongest over/under-subtraction epochs (quasar variability).
    picks = []
    for band in ["g", "r", "z"]:
        cands = []
        for fp in sorted((DIFF / band).glob("*_diff.fits")):
            stem = fp.stem.replace("_diff", "")
            res = diff_residual_at(stem, band, A.x, A.y, r=3)
            if np.isfinite(res):
                cands.append((stem, band, res))
        if cands:
            cands.sort(key=lambda t: -abs(t[2]))
            picks.append(cands[0])

    fig, axes = plt.subplots(2, 3, figsize=(11, 7.5))
    for j, band in enumerate(["g", "r", "z"]):
        ax = axes[0, j]
        refp = DATA / f"reference_{band}.fits"
        if refp.exists():
            ref = fits.getdata(refp)
            v1, v2 = zscale(ref)
            ax.imshow(ref, origin="lower", cmap="gray", vmin=v1, vmax=v2)
            ax.add_patch(Circle((A.x, A.y), 4, fill=False, ec="cyan", lw=1.3))
            ax.add_patch(Circle((B.x, B.y), 4, fill=False, ec="orange", lw=1.3))
            ax.text(A.x + 5, A.y + 5, "A", color="cyan", fontsize=11, weight="bold")
            ax.text(B.x + 5, B.y + 5, "B", color="orange", fontsize=11, weight="bold")
        ax.set_title(f"{band} median reference", fontsize=10)
        ax.set_xlim(360, 440); ax.set_ylim(360, 440)
        ax.set_xticks([]); ax.set_yticks([])

    for j in range(3):
        ax = axes[1, j]
        if j < len(picks):
            stem, band, res = picks[j]
            dp = DIFF / band / f"{stem}_diff.fits"
            d = fits.getdata(dp)
            v = np.nanpercentile(np.abs(d[np.isfinite(d)]), 99)
            ax.imshow(d, origin="lower", cmap="RdBu_r", vmin=-v, vmax=v)
            ax.add_patch(Circle((A.x, A.y), 4, fill=False, ec="k", lw=1.3))
            ax.add_patch(Circle((B.x, B.y), 4, fill=False, ec="k", lw=1.3, ls="--"))
            mjd = float(meta.loc[stem, "mjd"]) if stem in meta.index else np.nan
            sign = "over" if res > 0 else "under"
            ax.set_title(f"{band} diff  MJD {mjd:.0f}\n{sign}-subtraction at A "
                         f"(res={res:.0f})", fontsize=9)
            ax.set_xlim(360, 440); ax.set_ylim(360, 440)
        ax.set_xticks([]); ax.set_yticks([])

    # annotate with the measured sigma if available
    sub = ""
    sp = DATA / "sigma_real.csv"
    if sp.exists():
        sdf = pd.read_csv(sp)
        if len(sdf):
            avg = sdf.sigma_stddev.mean()
            sub = f"   <sigma> = {avg:.3f} mag (paper Table 2: 0.25)"
    fig.suptitle(f"{CAND_NAME} -- lensed-quasar variability evidence\n"
                 f"over/under-subtraction at the lensed-image positions across "
                 f"epochs{sub}", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(FIGS / "variability_panel.png", dpi=130)
    print(f"[fig] {FIGS/'variability_panel.png'}")


if __name__ == "__main__":
    main()
