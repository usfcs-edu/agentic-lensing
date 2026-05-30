#!/usr/bin/env python3
"""
15_extended_crossmatch.py — Phase-5 provenance.

Provenance of the top-N DR8 *ensemble* candidates (track-i re-score, 11_): which
are training-set leakage (NeuraLens-L18), which are previously published
(NeuraLens-shielded, Huang+2020, Huang+2021, Storfer+2024, Inchausti+2025,
Hsu+2025), and which are unmatched ("potentially new"). 10" match radius.

Usage:
  ./15_extended_crossmatch.py --score-col p_meta --top-n 2000
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy import units as u

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
MATCH_RADIUS = 10.0
HSU2025 = HERE.parent / "hsu-2025" / "data" / "classified_pairs.parquet"
H2020_CAT = HERE.parent / "huang-2020" / "data" / "huang2020_published_catalog.csv"


def xmatch(top, ra, dec, label):
    top_sky = SkyCoord(ra=top["ra"].values * u.deg, dec=top["dec"].values * u.deg)
    cat_sky = SkyCoord(ra=np.asarray(ra) * u.deg, dec=np.asarray(dec) * u.deg)
    _, sep2d, _ = top_sky.match_to_catalog_sky(cat_sky)
    m = sep2d.to(u.arcsec).value < MATCH_RADIUS
    print(f"  [{label:24s}] {int(m.sum()):>6,}/{len(top)} within {MATCH_RADIUS:.0f}\"")
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default=str(DATA / "inference_scores_ensemble_dr8.parquet"))
    ap.add_argument("--score-col", default="p_meta", dest="score_col")
    ap.add_argument("--top-n", type=int, default=2000)
    args = ap.parse_args()

    scr = pd.read_parquet(args.scores).dropna(subset=[args.score_col, "ra", "dec"])
    top = scr.sort_values(args.score_col, ascending=False).head(args.top_n).reset_index(drop=True)
    print(f"[init] top {len(top):,} ensemble candidates by {args.score_col} "
          f"({top[args.score_col].min():.4f}–{top[args.score_col].max():.4f})")

    print("\n[catalogs]")
    nl = pd.read_parquet(DATA / "positives_all.parquet")
    is_l18 = nl["resnet_model"].str.upper() == "L18"
    m = {}
    m["NeuraLens-L18 (leak)"] = xmatch(top, nl.loc[is_l18, "RA"], nl.loc[is_l18, "DEC"], "NeuraLens-L18 (leak)")
    m["NeuraLens-shielded"] = xmatch(top, nl.loc[~is_l18, "RA"], nl.loc[~is_l18, "DEC"], "NeuraLens-shielded")
    if H2020_CAT.exists():
        h20 = pd.read_csv(H2020_CAT)
        m["Huang+2020"] = xmatch(top, h20["RA"], h20["DEC"], "Huang+2020")
    h21 = pd.read_csv(DATA / "huang2021_published_catalog.csv")
    m["Huang+2021"] = xmatch(top, h21["RA"], h21["DEC"], "Huang+2021")
    st = pd.read_csv(DATA / "storfer2024_published_catalog.csv")
    m["Storfer+2024"] = xmatch(top, st["RA"], st["DEC"], "Storfer+2024")
    inc = pd.read_csv(DATA / "inchausti2025_published_catalog.csv")
    m["Inchausti+2025"] = xmatch(top, inc["RA"], inc["DEC"], "Inchausti+2025")
    if HSU2025.exists():
        hsu = pd.read_parquet(HSU2025)
        racol = "RA_lens" if "RA_lens" in hsu.columns else "RA"
        deccol = "DEC_lens" if "DEC_lens" in hsu.columns else "DEC"
        m["Hsu+2025"] = xmatch(top, hsu[racol], hsu[deccol], "Hsu+2025")

    any_known = np.zeros(len(top), dtype=bool)
    for label, mask in m.items():
        top[f"in_{label.split()[0].lower().replace('+','').replace('-','_')}"] = mask
        any_known |= mask
    top["in_any"] = any_known

    n = len(top)
    print(f"\n[summary] provenance of top {n:,} ensemble candidates ({args.score_col}):")
    for label, mask in m.items():
        print(f"  {label:24s} {int(mask.sum()):>6,}  ({mask.mean():.1%})")
    n_any = int(any_known.sum())
    print(f"  {'ANY known':24s} {n_any:>6,}  ({n_any/n:.1%})")
    print(f"  {'UNMATCHED (new?)':24s} {n-n_any:>6,}  ({(n-n_any)/n:.1%})")

    out = DATA / f"extended_crossmatch_ensemble_top{n}.csv"
    top.to_csv(out, index=False)
    print(f"\n[done] wrote {out.name}")


if __name__ == "__main__":
    main()
