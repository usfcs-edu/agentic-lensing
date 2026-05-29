#!/usr/bin/env python3
"""
15_extended_crossmatch.py — Phase 4c.

Provenance of the top-N DR8 candidates (one model): which are training-set
leakage (NeuraLens-L18), which are previously published (NeuraLens-shielded,
Huang+2020, Huang+2021, Hsu+2025), and which are unmatched ("potentially new").
Clone of huang-2020/17_extended_crossmatch.py pointed at the DR8 outputs.

10″ match radius (looser than 5″ — published positions drift vs DR8 Tractor).

Usage:
  ./15_extended_crossmatch.py --model shielded --top-n 2000
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


def xmatch(top: pd.DataFrame, cat_sky: SkyCoord, label: str) -> np.ndarray:
    top_sky = SkyCoord(ra=top["ra"].values * u.deg, dec=top["dec"].values * u.deg)
    _, sep2d, _ = top_sky.match_to_catalog_sky(cat_sky)
    matched = sep2d.to(u.arcsec).value < MATCH_RADIUS
    print(f"  [{label}] {int(matched.sum())}/{len(top)} within {MATCH_RADIUS:.0f}″")
    return matched


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=("l18", "shielded", "combined"), default="shielded")
    ap.add_argument("--scores", default=None)
    ap.add_argument("--top-n", type=int, default=2000)
    args = ap.parse_args()

    scores = args.scores or str(DATA / f"inference_scores_{args.model}_dr8.parquet")
    scr = pd.read_parquet(scores).sort_values("score", ascending=False).head(args.top_n)
    top = scr.reset_index(drop=True)
    print(f"[init] top {len(top):,} {args.model} candidates "
          f"(p {top['score'].min():.4f}–{top['score'].max():.4f})")

    print("\n[catalogs]")
    nl = pd.read_parquet(DATA / "positives_all.parquet")
    is_l18 = nl["resnet_model"].str.upper() == "L18"
    m_l18 = xmatch(top, SkyCoord(ra=nl.loc[is_l18, "RA"].values * u.deg,
                                 dec=nl.loc[is_l18, "DEC"].values * u.deg), "NeuraLens-L18 (training)")
    m_sh = xmatch(top, SkyCoord(ra=nl.loc[~is_l18, "RA"].values * u.deg,
                                dec=nl.loc[~is_l18, "DEC"].values * u.deg), "NeuraLens-shielded")

    h20 = pd.read_csv(DATA / "huang2020_published_catalog.csv") \
        if (DATA / "huang2020_published_catalog.csv").exists() else None
    if h20 is None:
        h20 = pd.read_csv(HERE.parent / "huang-2020" / "data" / "huang2020_published_catalog.csv")
    m_h20 = xmatch(top, SkyCoord(ra=h20["RA"].values * u.deg,
                                 dec=h20["DEC"].values * u.deg), "Huang+2020")

    h21 = pd.read_csv(DATA / "huang2021_published_catalog.csv")
    m_h21 = xmatch(top, SkyCoord(ra=h21["RA"].values * u.deg,
                                 dec=h21["DEC"].values * u.deg), "Huang+2021")

    if HSU2025.exists():
        hsu = pd.read_parquet(HSU2025)
        m_hsu = xmatch(top, SkyCoord(ra=hsu["RA_lens"].values * u.deg,
                                     dec=hsu["DEC_lens"].values * u.deg), "Hsu+2025")
    else:
        print(f"  [Hsu+2025] {HSU2025} not found — skipping")
        m_hsu = np.zeros(len(top), dtype=bool)

    top["in_neuralens_l18"] = m_l18
    top["in_neuralens_shielded"] = m_sh
    top["in_huang2020"] = m_h20
    top["in_huang2021"] = m_h21
    top["in_hsu2025"] = m_hsu
    top["in_any"] = m_l18 | m_sh | m_h20 | m_h21 | m_hsu

    n = len(top)
    print(f"\n[summary] provenance of top {n:,} ({args.model}):")
    for lbl, m in [("NeuraLens-L18 (leak)", m_l18), ("NeuraLens-shielded", m_sh),
                   ("Huang+2020", m_h20), ("Huang+2021", m_h21), ("Hsu+2025", m_hsu)]:
        print(f"  {lbl:24s} {int(m.sum()):>6,d}  ({m.mean():.1%})")
    n_any = int(top["in_any"].sum())
    print(f"  {'ANY known':24s} {n_any:>6,d}  ({n_any/n:.1%})")
    print(f"  {'UNMATCHED (new?)':24s} {n-n_any:>6,d}  ({(n-n_any)/n:.1%})")

    out = DATA / f"extended_crossmatch_{args.model}_top{n}.csv"
    top.to_csv(out, index=False)
    print(f"\n[done] wrote {out}")


if __name__ == "__main__":
    main()
