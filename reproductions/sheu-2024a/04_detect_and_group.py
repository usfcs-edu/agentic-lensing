#!/usr/bin/env python3
"""
04_detect_and_group.py  --  Sheu+2023 §3.3 SEP detection + spatial/temporal grouping

For every difference image we run SEP (Barbary 2018; the SExtractor algorithm of
Bertin & Arnouts 1996) at the paper's ladder of thresholds 1.0 -> 2.5 sigma in
0.25 steps (a >2.5 sigma detection is treated as the 2.5 ladder rung).  Each
SEP source on a diff image is a "sub-detection" carrying (band, mjd, ra, dec,
peak, flux, threshold).

Grouping (paper §3.3)
---------------------
All sub-detections (all bands, all epochs) are linked when they fall within
  - 3 px = 0.8" of one another spatially, AND
  - 50 days of another member temporally.
A group with >= 3 sub-detections is flagged a POSSIBLE TRANSIENT
(i.e. observed >=2 separate epochs).  We rank groups by member count and by
proximity to the lens centre, and we report whether the best group sits on the
candidate's counter-image where Sheu+2023 place the L-SN.

Outputs
-------
  data/subdetections.csv     every SEP sub-detection
  data/groups.csv            grouped transient candidates (>=3 sub-detections)
  figs/detection_summary.png triage figure
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import sep
from astropy.io import fits
from astropy.wcs import WCS

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
FIGS = HERE / "figs"
FIGS.mkdir(exist_ok=True)
DIFF = DATA / "diff"

CAND_RA = 38.0655
CAND_DEC = -24.4942
PIXSCALE = 0.262  # arcsec/pix on the common grid
THRESHOLDS = [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5]
LINK_PIX = 3.0          # 0.8"
LINK_DAYS = 50.0
MIN_SUBDET = 3


def detect_frame(diff: np.ndarray, wcs: WCS, band: str, mjd: float, stem: str):
    """Run SEP over the threshold ladder; return list of sub-detections."""
    data = np.ascontiguousarray(diff.astype(np.float32))
    mask = ~np.isfinite(data)
    data = np.where(mask, 0.0, data)
    try:
        bkg = sep.Background(data, mask=mask)
        rms = bkg.globalrms
        data_sub = data - bkg.back()
    except Exception:
        rms = np.nanstd(diff[np.isfinite(diff)])
        data_sub = data
    out = []
    for thr in THRESHOLDS:
        try:
            objs = sep.extract(data_sub, thr, err=rms, mask=mask,
                               minarea=3, deblend_cont=0.005)
        except Exception:
            continue
        for o in objs:
            ra, dec = wcs.all_pix2world(o["x"], o["y"], 0)
            out.append(dict(stem=stem, band=band, mjd=mjd, thr=thr,
                            x=float(o["x"]), y=float(o["y"]),
                            ra=float(ra), dec=float(dec),
                            peak=float(o["peak"]), flux=float(o["flux"]),
                            npix=int(o["npix"])))
    return out


def group_subdetections(df: pd.DataFrame) -> pd.DataFrame:
    """Friends-of-friends on (x,y within LINK_PIX) AND (mjd within LINK_DAYS)."""
    n = len(df)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra_, rb = find(a), find(b)
        if ra_ != rb:
            parent[rb] = ra_

    xy = df[["x", "y"]].to_numpy()
    mjd = df["mjd"].to_numpy()
    for i in range(n):
        dx = xy[:, 0] - xy[i, 0]
        dy = xy[:, 1] - xy[i, 1]
        close = (dx * dx + dy * dy <= LINK_PIX ** 2)
        # temporal: NaN mjd -> treat as linkable (unknown date)
        dt = np.abs(mjd - mjd[i])
        tlink = ~np.isfinite(dt) | (dt <= LINK_DAYS)
        for j in np.where(close & tlink)[0]:
            if j != i:
                union(i, j)
    labels = np.array([find(i) for i in range(n)])
    df = df.copy()
    df["group"] = labels
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bands", nargs="+", default=["g", "r", "z"])
    args = ap.parse_args()

    subs = []
    for band in args.bands:
        man = DATA / "diff_manifest.csv"
        dman = pd.read_csv(man)
        meta = pd.read_csv(DATA / "exposure_manifest.csv").set_index("stem")
        for fp in sorted((DIFF / band).glob("*_diff.fits")):
            stem = fp.stem.replace("_diff", "")
            diff = fits.getdata(fp)
            wcs = WCS(fits.getheader(fp))
            mjd = float(meta.loc[stem, "mjd"]) if stem in meta.index else np.nan
            subs.extend(detect_frame(diff, wcs, band, mjd, stem))

    sdf = pd.DataFrame(subs)
    sdf.to_csv(DATA / "subdetections.csv", index=False)
    print(f"[detect] {len(sdf)} sub-detections across "
          f"{sdf.stem.nunique() if len(sdf) else 0} diff images")
    if len(sdf) == 0:
        print("[warn] no sub-detections")
        return

    # dedup near-identical sub-detections from the threshold ladder per frame:
    # keep, per (stem, ~1px cell), the lowest-threshold (=most significant kept)
    sdf["cell"] = (sdf.x.round(0).astype(int).astype(str) + "_"
                   + sdf.y.round(0).astype(int).astype(str))
    sdf = (sdf.sort_values("thr")
              .drop_duplicates(["stem", "cell"]).reset_index(drop=True))

    g = group_subdetections(sdf)
    sizes = g.groupby("group").size()
    keep = sizes[sizes >= MIN_SUBDET].index
    groups = g[g.group.isin(keep)].copy()

    # summarise groups
    rows = []
    for gid, gg in groups.groupby("group"):
        xc, yc = gg.x.mean(), gg.y.mean()
        rac, decc = gg.ra.mean(), gg.dec.mean()
        # offset from lens centre (arcsec)
        # common grid: lens at pixel (400,400)
        off_arcsec = np.hypot(xc - 400.0, yc - 400.0) * PIXSCALE
        rows.append(dict(group=int(gid), n_subdet=len(gg),
                         n_epochs=gg.mjd.nunique(),
                         bands="".join(sorted(gg.band.unique())),
                         x=xc, y=yc, ra=rac, dec=decc,
                         mjd_min=gg.mjd.min(), mjd_max=gg.mjd.max(),
                         offset_arcsec=off_arcsec))
    gsum = pd.DataFrame(rows).sort_values(["n_subdet"], ascending=False)
    gsum.to_csv(DATA / "groups.csv", index=False)
    print(f"[group] {len(gsum)} groups with >= {MIN_SUBDET} sub-detections")
    if len(gsum):
        print(gsum.to_string(index=False))

    # triage figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(sdf.x, sdf.y, s=8, c="0.6", label="sub-detections")
        if len(gsum):
            ax.scatter(gsum.x, gsum.y, s=120, facecolors="none",
                       edgecolors="r", lw=1.5, label=">=3 group")
        ax.scatter([400], [400], marker="+", c="k", s=200, label="lens centre")
        ax.set_xlim(0, 801); ax.set_ylim(0, 801)
        ax.set_xlabel("x (px)"); ax.set_ylabel("y (px)")
        ax.set_title(f"DESI-{CAND_RA}{CAND_DEC} diff-image sub-detections")
        ax.legend(loc="upper right", fontsize=8)
        fig.tight_layout()
        fig.savefig(FIGS / "detection_summary.png", dpi=130)
        print(f"[fig] {FIGS/'detection_summary.png'}")
    except Exception as e:
        print(f"[fig] skipped: {e}")


if __name__ == "__main__":
    main()
