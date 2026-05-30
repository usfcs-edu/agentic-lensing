#!/usr/bin/env python3
"""
16_build_inspection_viewer.py — Phase-5 visual inspection.

Paginated static HTML viewer of the top-N DR8 *ensemble* candidates (track-i
re-score, 11_), with Lupton-stretched grz thumbnails from the on-disk DR8
cutouts. Tiles matching any published catalogue (Huang+2021 / Storfer+2024 /
Inchausti+2025, within 5") are flagged with a star.

Usage:
  ./16_build_inspection_viewer.py --score-col p_meta --top-n 2000 --per-page 50
"""
from __future__ import annotations

import argparse
import html
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.io import fits
from astropy.visualization import make_lupton_rgb
from PIL import Image
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
FITS_DIR = DATA / "cutouts_fits_dr8"
THUMB_PX = 200
LUPTON_Q = 8.0
LUPTON_STRETCH = 0.5


def fits_to_thumb(path: Path):
    try:
        with fits.open(path) as hdul:
            cube = hdul[0].data.astype(np.float32)
    except Exception:
        return None
    if cube is None or cube.shape != (3, 101, 101):
        return None
    rgb = make_lupton_rgb(cube[2], cube[1], cube[0], Q=LUPTON_Q, stretch=LUPTON_STRETCH)
    return Image.fromarray(rgb[::-1, :, :]).resize((THUMB_PX, THUMB_PX), Image.NEAREST)


def tile_html(row, published: set, score_col: str) -> str:
    rid = html.escape(str(row["row_id"]))
    badge = ' <span style="color:#d7191c">&#9733;</span>' if rid in published else ""
    return (f'<div class="tile"><img src="thumbs/{rid}.png" alt="{rid}">'
            f'<div class="meta"><div><b>{rid}</b>{badge}</div>'
            f'<div>{row["ra"]:.4f}, {row["dec"]:+.4f}</div>'
            f'<div>p<sub>meta</sub> = {row[score_col]:.4f}</div></div></div>')


def render_page(pi, n_pages, rows, published, score_col) -> str:
    nav = []
    if pi > 1:
        nav.append(f'<a href="page_{pi-1:03d}.html">prev</a>')
    if pi < n_pages:
        nav.append(f'<a href="page_{pi+1:03d}.html">next</a>')
    head = (f'<a href="index.html">index</a> &nbsp;|&nbsp; page {pi} of {n_pages}'
            + (' &nbsp;|&nbsp; ' + " &nbsp; ".join(nav) if nav else ''))
    tiles = "\n".join(tile_html(r, published, score_col) for _, r in rows.iterrows())
    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>Phase 5 DR8 ensemble — page {pi}</title><style>
body{{font-family:system-ui,sans-serif;margin:14px;background:#1a1a1a;color:#ddd}}
a{{color:#6cf}} nav{{margin-bottom:12px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:10px}}
.tile{{background:#222;border:1px solid #333;padding:6px;border-radius:4px;text-align:center}}
.tile img{{display:block;width:200px;height:200px;margin:0 auto;image-rendering:pixelated}}
.meta{{font-size:12px;margin-top:4px;line-height:1.4}} .meta b{{color:#fff;font-family:monospace}}
</style></head><body><nav>{head}</nav><div class="grid">{tiles}</div></body></html>"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default=str(DATA / "inference_scores_ensemble_dr8.parquet"))
    ap.add_argument("--score-col", default="p_meta", dest="score_col")
    ap.add_argument("--top-n", type=int, default=2000)
    ap.add_argument("--per-page", type=int, default=50)
    args = ap.parse_args()

    out_dir = HERE / "papers" / "figures" / "inspection_ensemble"
    thumb_dir = out_dir / "thumbs"
    thumb_dir.mkdir(parents=True, exist_ok=True)

    scr = pd.read_parquet(args.scores).dropna(subset=[args.score_col, "ra", "dec"])
    scr = scr.sort_values(args.score_col, ascending=False).head(args.top_n).reset_index(drop=True)
    print(f"[init] top {len(scr):,} ensemble candidates "
          f"(p {scr[args.score_col].min():.4f}-{scr[args.score_col].max():.4f})")

    # Published star = within 5" of Huang+2021 / Storfer / Inchausti.
    pubs = []
    for f in ("huang2021_published_catalog.csv", "storfer2024_published_catalog.csv",
              "inchausti2025_published_catalog.csv"):
        p = DATA / f
        if p.exists():
            pubs.append(pd.read_csv(p)[["RA", "DEC"]])
    pub = pd.concat(pubs, ignore_index=True)
    pub_sky = SkyCoord(ra=pub["RA"].values * u.deg, dec=pub["DEC"].values * u.deg)
    top_sky = SkyCoord(ra=scr["ra"].values * u.deg, dec=scr["dec"].values * u.deg)
    _, sep, _ = top_sky.match_to_catalog_sky(pub_sky)
    pub_ids = set(scr.loc[sep.to(u.arcsec).value < 5.0, "row_id"].astype(str))
    print(f"[init] {len(pub_ids)} of top-{len(scr)} match a published candidate")

    keep = []
    for i, row in tqdm(scr.iterrows(), total=len(scr), desc="thumbs"):
        rid = str(row["row_id"])
        out = thumb_dir / f"{rid}.png"
        if out.exists() and out.stat().st_size > 0:
            keep.append(i); continue
        fp = FITS_DIR / f"{rid}.fits"
        img = fits_to_thumb(fp) if fp.exists() else None
        if img is None:
            continue
        img.save(out); keep.append(i)
    scr = scr.loc[keep].reset_index(drop=True)

    per = args.per_page
    n_pages = (len(scr) + per - 1) // per
    for p in range(1, n_pages + 1):
        rows = scr.iloc[(p - 1) * per: p * per]
        (out_dir / f"page_{p:03d}.html").write_text(
            render_page(p, n_pages, rows, pub_ids, args.score_col))
    idx = "".join(f'<li><a href="page_{i:03d}.html">page {i}</a></li>'
                  for i in range(1, n_pages + 1))
    (out_dir / "index.html").write_text(
        f"""<!doctype html><html><head><meta charset="utf-8"><title>Phase 5 DR8 ensemble</title>
<style>body{{font-family:system-ui,sans-serif;margin:20px;background:#1a1a1a;color:#ddd;max-width:720px}}
a{{color:#6cf}} ul{{columns:4;list-style:none;padding-left:0}}</style></head><body>
<h2>Phase 5 DR8 ensemble inspection ({args.score_col})</h2>
<p>{len(scr):,} cutouts sorted by ensemble score. &#9733; = published
Huang+2021 / Storfer+2024 / Inchausti+2025 candidate (5").</p>
<ul>{idx}</ul></body></html>""")
    print(f"[done] {n_pages} pages -> {out_dir}/index.html")


if __name__ == "__main__":
    main()
