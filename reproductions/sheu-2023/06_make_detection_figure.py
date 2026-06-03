#!/usr/bin/env python3
"""
06_make_detection_figure.py  --  visual re-detection of the Grade-A L-SN candidate

Builds a triage panel for DESI-344.6252-48.8977:
  * the median-coadd REFERENCE (g/r/z) with the lens centre + counter-image marked
  * a small montage of B08 DIFFERENCE images at the epochs where the >=3-subdetection
    group near the lens centre lights up, with the detected transient circled.

This is the "Figure 10"-style evidence: a transient sitting on the counter-image,
appearing across multiple epochs/bands in the difference images.
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


def zscale(a, lo=2, hi=98):
    a = a[np.isfinite(a)]
    if a.size == 0:
        return 0, 1
    return np.percentile(a, lo), np.percentile(a, hi)


def main():
    groups = pd.read_csv(DATA / "groups.csv")
    meta = pd.read_csv(DATA / "exposure_manifest.csv").set_index("stem")
    cen = groups[groups.offset_arcsec < 3.0].sort_values("n_subdet", ascending=False)
    gx, gy = (float(cen.iloc[0].x), float(cen.iloc[0].y)) if len(cen) else (400, 400)

    # pick epochs across bands that detect near (gx,gy): use the highest-flux diffs
    subs = pd.read_csv(DATA / "subdetections.csv")
    near = subs[np.hypot(subs.x - gx, subs.y - gy) < 4]
    # one representative diff per band, the one with the strongest peak
    picks = (near.sort_values("peak", ascending=False)
                 .drop_duplicates("band").head(3))

    nrow = 2
    ncol = 3
    fig, axes = plt.subplots(nrow, ncol, figsize=(11, 7.5))

    # top row: references
    for j, band in enumerate(["g", "r", "z"]):
        ax = axes[0, j]
        ref = fits.getdata(DATA / f"reference_{band}.fits")
        v1, v2 = zscale(ref)
        ax.imshow(ref, origin="lower", cmap="gray", vmin=v1, vmax=v2)
        ax.add_patch(Circle((400, 400), 6, fill=False, ec="lime", lw=1.2))
        ax.add_patch(Circle((gx, gy), 5, fill=False, ec="red", lw=1.5, ls="--"))
        ax.set_title(f"{band} median reference", fontsize=10)
        ax.set_xlim(330, 470); ax.set_ylim(330, 470)
        ax.set_xticks([]); ax.set_yticks([])

    # bottom row: difference images at detection epochs
    for j in range(3):
        ax = axes[1, j]
        if j < len(picks):
            row = picks.iloc[j]
            band = row.band
            dp = DIFF / band / f"{row.stem}_diff.fits"
            if dp.exists():
                d = fits.getdata(dp)
                v1, v2 = zscale(d)
                ax.imshow(d, origin="lower", cmap="gray", vmin=v1, vmax=v2)
                ax.add_patch(Circle((gx, gy), 5, fill=False, ec="red", lw=1.5))
                mjd = float(meta.loc[row.stem, "mjd"]) if row.stem in meta.index else np.nan
                ax.set_title(f"{band} diff  MJD {mjd:.0f}  peak={row.peak:.0f}",
                             fontsize=9)
                ax.set_xlim(330, 470); ax.set_ylim(330, 470)
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle("DESI-344.6252-48.8977 -- re-detection of the Grade-A L-SN candidate\n"
                 f"(transient {np.hypot(gx-400,gy-400)*PIXSCALE:.2f}\" from lens centre, "
                 f"on the counter-image; {int(cen.iloc[0].n_subdet)} sub-detections, "
                 f"{int(cen.iloc[0].n_epochs)} epochs, bands {cen.iloc[0].bands})",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(FIGS / "redetection_panel.png", dpi=130)
    print(f"[fig] {FIGS/'redetection_panel.png'}")


if __name__ == "__main__":
    main()
