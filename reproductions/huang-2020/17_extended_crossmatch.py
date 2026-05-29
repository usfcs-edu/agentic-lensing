#!/usr/bin/env python3
"""
17_extended_crossmatch.py — cross-match the top-N DR7-trained candidates
against every publicly available lens catalog we have on disk, so we can
distinguish:

  (a) memorization hits (top-scorers that overlap our Phase 3a training set)
  (b) candidates already published by Huang or others
  (c) candidates not present in any catalog we know about → potentially new

Catalogs (in priority order):
  - NeuraLens-full       1,312 = L18 (949) + shielded (363); positives_all.parquet
  - Huang+2020 published   342 = 60 A + 106 B + 176 C; huang2020_published_catalog.csv
  - Huang+2021 published   ~89 names extracted from PDF Tables on pp 29-30
  - Hsu+2025 candidates 13,530 = DESI DR1 group lens pairs;
                              ../hsu-2025/data/classified_pairs.parquet

Each match uses a 10″ radius (looser than the 5″ used elsewhere — published
positions can shift up to ~8″ between paper-time and DR7 Tractor reprocessing,
as observed for the missing-7).

Output:
  data/extended_crossmatch_top<N>.csv
  printed summary table of provenance for the top-N
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pypdf
from astropy.coordinates import SkyCoord
from astropy import units as u

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

MATCH_RADIUS = 10.0  # arcsec
HUANG2021_PDF = Path("/raid/benson/git/agentic-lensing/papers/Huang_2021_DESI_legacy_lenses.pdf")
HSU2025_CATALOG = HERE.parent / "hsu-2025" / "data" / "classified_pairs.parquet"
NAME_RE = re.compile(r"DESI-(\d{3}\.\d{4})([+\-]\d{2}\.\d{4})")


def extract_huang2021() -> pd.DataFrame:
    """Parse DESI-RA±DEC names from Huang+2021 PDF Tables (pp 17, 29-31 in v1)."""
    pdf = pypdf.PdfReader(str(HUANG2021_PDF))
    rows = []
    seen = set()
    for page_idx in range(len(pdf.pages)):
        text = pdf.pages[page_idx].extract_text()
        for m in NAME_RE.finditer(text):
            ra_s, dec_s = m.group(1), m.group(2)
            name = f"DESI-{ra_s}{dec_s}"
            if name in seen:
                continue
            seen.add(name)
            rows.append({"name": name, "RA": float(ra_s), "DEC": float(dec_s)})
    return pd.DataFrame(rows)


def crossmatch(top: pd.DataFrame, cat_sky: SkyCoord, label: str,
               radius_arcsec: float = MATCH_RADIUS) -> tuple[np.ndarray, np.ndarray]:
    """For each row in `top`, find nearest neighbour in `cat_sky`.
    Returns (matched_bool, sep_arcsec)."""
    top_sky = SkyCoord(ra=top["ra"].values * u.deg, dec=top["dec"].values * u.deg)
    idx, sep2d, _ = top_sky.match_to_catalog_sky(cat_sky)
    sep = sep2d.to(u.arcsec).value
    matched = sep < radius_arcsec
    print(f"  [{label}] {int(matched.sum())} / {len(top)} matched within {radius_arcsec:.0f}″")
    return matched, sep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores",
                    default=str(DATA / "inference_scores_dr7trained.parquet"))
    ap.add_argument("--top-n", type=int, default=2000)
    args = ap.parse_args()

    print(f"[init] loading {args.scores}")
    scr = pd.read_parquet(args.scores)
    scr = scr.sort_values("score", ascending=False).reset_index(drop=True)
    top = scr.head(args.top_n).reset_index(drop=True)
    print(f"[init] top {len(top):,} candidates (score {top['score'].min():.4f} – {top['score'].max():.4f})")

    print("\n[catalogs] loading & matching:")

    # 1. NeuraLens full (1312)
    nl = pd.read_parquet(DATA / "positives_all.parquet")
    nl_sky = SkyCoord(ra=nl["RA"].values * u.deg, dec=nl["DEC"].values * u.deg)
    m_nl, sep_nl = crossmatch(top, nl_sky, "NeuraLens-1312")
    # Split into L18 (in training) vs shielded (not in training)
    is_l18 = (nl["resnet_model"].str.upper() == "L18")
    nl_l18_sky = SkyCoord(ra=nl.loc[is_l18, "RA"].values * u.deg,
                          dec=nl.loc[is_l18, "DEC"].values * u.deg)
    m_l18, _ = crossmatch(top, nl_l18_sky, "  └ L18 (training, 949)")
    nl_sh_sky = SkyCoord(ra=nl.loc[~is_l18, "RA"].values * u.deg,
                         dec=nl.loc[~is_l18, "DEC"].values * u.deg)
    m_sh, _ = crossmatch(top, nl_sh_sky, "  └ shielded (post-2020, 363)")

    # 2. Huang+2020 published (342)
    h2020 = pd.read_csv(DATA / "huang2020_published_catalog.csv")
    h2020_sky = SkyCoord(ra=h2020["RA"].values * u.deg, dec=h2020["DEC"].values * u.deg)
    m_h20, _ = crossmatch(top, h2020_sky, "Huang+2020 (342)")

    # 3. Huang+2021 published (~89)
    h2021 = extract_huang2021()
    print(f"  [Huang+2021] extracted {len(h2021)} names from PDF")
    h2021_sky = SkyCoord(ra=h2021["RA"].values * u.deg, dec=h2021["DEC"].values * u.deg)
    m_h21, _ = crossmatch(top, h2021_sky, "Huang+2021 (PDF)")
    h2021.to_csv(DATA / "huang2021_extracted_names.csv", index=False)

    # 4. Hsu+2025 (13,530)
    hsu = pd.read_parquet(HSU2025_CATALOG)
    hsu_sky = SkyCoord(ra=hsu["RA_lens"].values * u.deg,
                       dec=hsu["DEC_lens"].values * u.deg)
    m_hsu, _ = crossmatch(top, hsu_sky, "Hsu+2025 (13,530)")

    # Build provenance summary on the top-N
    top["in_neuralens_l18"] = m_l18
    top["in_neuralens_shielded"] = m_sh
    top["in_huang2020"] = m_h20
    top["in_huang2021"] = m_h21
    top["in_hsu2025"] = m_hsu
    top["in_any_published"] = m_l18 | m_sh | m_h20 | m_h21 | m_hsu

    n = len(top)
    n_any = int(top["in_any_published"].sum())
    n_l18_only = int((m_l18 & ~(m_sh | m_h20 | m_h21 | m_hsu)).sum())
    n_unmatched = n - n_any

    print("\n[summary] provenance of top {:,} DR7-trained candidates:".format(n))
    print(f"  in NeuraLens-L18 (training-set leakage):  {int(m_l18.sum()):>6,d}  ({m_l18.mean():.1%})")
    print(f"    └ only NeuraLens-L18 (no other catalog): {n_l18_only:>6,d}  ({n_l18_only/n:.1%})")
    print(f"  in NeuraLens-shielded:                     {int(m_sh.sum()):>6,d}  ({m_sh.mean():.1%})")
    print(f"  in Huang+2020:                              {int(m_h20.sum()):>6,d}  ({m_h20.mean():.1%})")
    print(f"  in Huang+2021:                              {int(m_h21.sum()):>6,d}  ({m_h21.mean():.1%})")
    print(f"  in Hsu+2025:                                {int(m_hsu.sum()):>6,d}  ({m_hsu.mean():.1%})")
    print(f"  --")
    print(f"  in ANY known catalog:                       {n_any:>6,d}  ({n_any/n:.1%})")
    print(f"  UNMATCHED (potentially new):                {n_unmatched:>6,d}  ({n_unmatched/n:.1%})")

    out_path = DATA / f"extended_crossmatch_top{n}.csv"
    top.to_csv(out_path, index=False)
    print(f"\n[done] wrote {out_path}")


if __name__ == "__main__":
    main()
