#!/usr/bin/env python3
"""
04_make_figures.py

Diagnostic figures for the dawes-2022 reproduction.

  figs/separation_hist.png   image-pair separation distribution of our 5" FoF
                             candidate groups (pair-level), overlaid with the
                             published Dawes+2023 catalog Sep column
                             (analog of Dawes Figure 2 / Section 3 discussion:
                             >95% of discoverable-known systems have images <5").
  figs/recovery_offsets.png  great-circle offset between published candidate
                             positions and the nearest FoF group centroid
                             (recovered vs miss).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
import astropy.units as u


HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
FIGS = HERE / "figs"
FIGS.mkdir(exist_ok=True)


def pair_separations(groups_path: Path) -> np.ndarray:
    g = pq.read_table(groups_path).to_pandas()
    seps = []
    for _, sub in g.groupby("group_id"):
        if len(sub) < 2:
            continue
        sc = SkyCoord(ra=sub["RA"].to_numpy() * u.deg,
                      dec=sub["DEC"].to_numpy() * u.deg)
        # all pairwise separations within the group; take max (image sep)
        idx1, idx2 = np.triu_indices(len(sc), k=1)
        d = sc[idx1].separation(sc[idx2]).to_value(u.arcsec)
        seps.append(d.max())
    return np.array(seps)


def fig_separation_hist() -> None:
    ours = pair_separations(DATA / "qso_groups_5arcsec.parquet")
    pub = pd.read_csv(DATA / "dawes2023_vizier_table2.csv")
    pub_sep = pd.to_numeric(pub["Sep"], errors="coerce").dropna().to_numpy()

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bins = np.linspace(0, 5, 26)
    ax.hist(ours, bins=bins, density=True, alpha=0.55,
            label=f"our 5\" FoF groups (N={len(ours)})", color="C0")
    ax.hist(pub_sep, bins=bins, density=True, histtype="step", lw=2.0,
            label=f"Dawes+2023 published Sep (N={len(pub_sep)})", color="C3")
    ax.axvline(5.0, ls="--", color="k", lw=1, alpha=0.6)
    ax.set_xlabel("max image separation within group [arcsec]")
    ax.set_ylabel("normalized counts")
    ax.set_title("Dawes-2022 reproduction: candidate image separations")
    ax.legend()
    fig.tight_layout()
    out = FIGS / "separation_hist.png"
    fig.savefig(out, dpi=130)
    print(f"[fig] wrote {out}")


def fig_recovery_offsets() -> None:
    pub = pd.read_csv(DATA / "dawes2023_vizier_table2.csv")
    prim = pub[pub["Grade"].astype(str).str.strip().isin(["A", "B", "C"])]
    prim = prim.drop_duplicates("Name")
    pub_sc = SkyCoord(ra=prim["_RA"].to_numpy() * u.deg,
                      dec=prim["_DE"].to_numpy() * u.deg)

    g = pq.read_table(DATA / "qso_groups_5arcsec.parquet").to_pandas()
    cen = g.groupby("group_id")[["RA", "DEC"]].mean().reset_index()
    grp_sc = SkyCoord(ra=cen["RA"].to_numpy() * u.deg,
                      dec=cen["DEC"].to_numpy() * u.deg)
    _, sep, _ = pub_sc.match_to_catalog_sky(grp_sc)
    off = sep.to_value(u.arcsec)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(np.clip(off, 0, 30), bins=np.linspace(0, 30, 61), color="C2")
    ax.axvline(3.0, ls="--", color="k", lw=1)
    n_rec = int((off < 3).sum())
    ax.set_xlabel("offset to nearest FoF group centroid [arcsec]")
    ax.set_ylabel("# published candidates")
    ax.set_title(f"Recovery of 436 Dawes candidates: {n_rec}/436 within 3\" "
                 f"(proxy QSO sample)")
    fig.tight_layout()
    out = FIGS / "recovery_offsets.png"
    fig.savefig(out, dpi=130)
    print(f"[fig] wrote {out}")


def main() -> None:
    fig_separation_hist()
    fig_recovery_offsets()


if __name__ == "__main__":
    main()
