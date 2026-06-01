#!/usr/bin/env python3
"""
16_build_inspection_viewer.py — Phase 3b M5.

Generate a paginated static HTML viewer for visually inspecting the top-N
DR7-trained candidates. Each tile shows a Lupton-stretched RGB thumbnail
of the local cutout FITS, the row_id, RA, Dec, and ResNet score.

Inputs:
  data/inference_scores_dr7trained.parquet
  data/cutouts_fits_dr7/<row_id>.fits   (only those with score >= 0.5)
  data/huang2020_published_catalog.csv  (for "★ published" badges)

Outputs:
  papers/figures/inspection/index.html
  papers/figures/inspection/page_NNN.html  (50 candidates each)
  papers/figures/inspection/thumbs/<row_id>.png

Usage:
  ./16_build_inspection_viewer.py [--top-n 2000] [--per-page 50]
"""
from __future__ import annotations

import argparse
import html
import os
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.visualization import make_lupton_rgb
from PIL import Image
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
FITS_DIR = DATA / "cutouts_fits_dr7"
OUT_DIR = HERE / "papers" / "figures" / "inspection"
THUMB_DIR = OUT_DIR / "thumbs"

THUMB_PX = 200  # upscale 101×101 → 200×200 for visibility
LUPTON_Q = 8.0
LUPTON_STRETCH = 0.5


def fits_to_thumb(path: Path) -> Image.Image | None:
    try:
        with fits.open(path) as hdul:
            cube = hdul[0].data.astype(np.float32)  # (3, H, W)  g, r, z
    except Exception:
        return None
    if cube is None or cube.shape != (3, 101, 101):
        return None
    # Lupton wants (R, G, B). For grz: z is reddest, g is bluest.
    rgb = make_lupton_rgb(cube[2], cube[1], cube[0],
                          Q=LUPTON_Q, stretch=LUPTON_STRETCH)
    # rgb is uint8, shape (101, 101, 3) — upscale + flip vertically (FITS y-axis)
    img = Image.fromarray(rgb[::-1, :, :])
    img = img.resize((THUMB_PX, THUMB_PX), Image.NEAREST)
    return img


def tile_html(row: pd.Series, published: set[str]) -> str:
    rid = html.escape(str(row["row_id"]))
    is_pub = rid in published
    badge = ' <span style="color:#d7191c">★</span>' if is_pub else ""
    return f'''<div class="tile">
  <img src="thumbs/{rid}.png" alt="{rid}">
  <div class="meta">
    <div><b>{rid}</b>{badge}</div>
    <div>{row["ra"]:.4f}, {row["dec"]:+.4f}</div>
    <div>p = {row["score"]:.4f}</div>
  </div>
</div>'''


def render_page(page_idx: int, n_pages: int, rows: pd.DataFrame,
                published: set[str]) -> str:
    nav = []
    if page_idx > 1:
        nav.append(f'<a href="page_{page_idx-1:03d}.html">‹ prev</a>')
    if page_idx < n_pages:
        nav.append(f'<a href="page_{page_idx+1:03d}.html">next ›</a>')
    navbar = " &nbsp;|&nbsp; ".join(nav)
    nav_with_idx = ('<a href="index.html">index</a> &nbsp;|&nbsp; '
                    f'page {page_idx} of {n_pages}'
                    + (' &nbsp;|&nbsp; ' + navbar if navbar else ''))
    tiles = "\n".join(tile_html(r, published) for _, r in rows.iterrows())
    return f"""<!doctype html>
<html><head><meta charset="utf-8">
<title>Phase 3b inspection — page {page_idx}</title>
<style>
body {{ font-family: -apple-system, system-ui, sans-serif; margin: 14px;
       background: #1a1a1a; color: #ddd; }}
a {{ color: #6cf; }}
nav {{ margin-bottom: 12px; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
        gap: 10px; }}
.tile {{ background: #222; border: 1px solid #333; padding: 6px; border-radius: 4px;
        text-align: center; }}
.tile img {{ display: block; width: 200px; height: 200px; margin: 0 auto;
            image-rendering: pixelated; }}
.meta {{ font-size: 12px; margin-top: 4px; line-height: 1.4; }}
.meta b {{ color: #fff; font-family: monospace; }}
footer {{ margin-top: 16px; color: #777; font-size: 12px; }}
</style></head>
<body>
<nav>{nav_with_idx}</nav>
<div class="grid">
{tiles}
</div>
<footer>
Top-{len(rows)} Phase-3b DR7-trained candidates at p &geq; 0.5. Tiles marked ★
appear in Huang+2020's published A/B/C catalog (5″ match).
</footer>
</body></html>
"""


def render_index(n_pages: int, total: int, n_pub_recovered: int) -> str:
    rows = []
    for i in range(1, n_pages + 1):
        rows.append(f'<li><a href="page_{i:03d}.html">page {i}</a></li>')
    return f"""<!doctype html>
<html><head><meta charset="utf-8">
<title>Phase 3b inspection viewer (DR7-trained)</title>
<style>
body {{ font-family: -apple-system, system-ui, sans-serif; margin: 20px;
       background: #1a1a1a; color: #ddd; max-width: 720px; }}
a {{ color: #6cf; }}
ul {{ columns: 4; column-gap: 16px; list-style: none; padding-left: 0; }}
</style></head>
<body>
<h2>Phase 3b inspection viewer (DR7-trained checkpoint)</h2>
<p>{total:,} cutouts sorted by ResNet sigmoid score (descending).
The top {n_pub_recovered:,} marked ★ are Huang+2020 published candidates
(5″ cross-match).</p>
<ul>
{"".join(rows)}
</ul>
</body></html>
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default=str(DATA / "inference_scores_dr7trained.parquet"))
    ap.add_argument("--published", default=str(DATA / "huang2020_published_catalog.csv"))
    ap.add_argument("--top-n", type=int, default=2000)
    ap.add_argument("--per-page", type=int, default=50)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    THUMB_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[init] loading scores from {args.scores}")
    scr = pd.read_parquet(args.scores)
    scr = scr.sort_values("score", ascending=False).reset_index(drop=True)
    scr = scr.head(args.top_n)
    print(f"[init] selected top {len(scr):,} candidates (score range "
          f"{scr['score'].min():.4f} – {scr['score'].max():.4f})")

    # Mark which row_ids correspond to published Huang+2020 candidates
    # (by spatial cross-match within 5″)
    pub_df = pd.read_csv(args.published)
    from astropy.coordinates import SkyCoord
    from astropy import units as u
    pub_sky = SkyCoord(ra=pub_df["RA"].values * u.deg,
                       dec=pub_df["DEC"].values * u.deg)
    top_sky = SkyCoord(ra=scr["ra"].values * u.deg,
                       dec=scr["dec"].values * u.deg)
    idx, sep2d, _ = top_sky.match_to_catalog_sky(pub_sky)
    is_pub = sep2d.to(u.arcsec).value < 5.0
    pub_row_ids = set(scr.loc[is_pub, "row_id"].tolist())
    print(f"[init] {len(pub_row_ids)} of the top-{len(scr)} are published candidates")

    # Generate thumbnails for any rows missing a FITS file
    n_thumbs = n_missing_fits = 0
    rows_with_thumb: list[int] = []
    for i, row in tqdm(scr.iterrows(), total=len(scr), desc="thumbs"):
        rid = str(row["row_id"])
        out = THUMB_DIR / f"{rid}.png"
        if out.exists() and out.stat().st_size > 0:
            rows_with_thumb.append(i)
            n_thumbs += 1
            continue
        fits_path = FITS_DIR / f"{rid}.fits"
        if not fits_path.exists():
            n_missing_fits += 1
            continue
        img = fits_to_thumb(fits_path)
        if img is None:
            n_missing_fits += 1
            continue
        img.save(out)
        rows_with_thumb.append(i)
        n_thumbs += 1
    print(f"[thumbs] generated/loaded {n_thumbs:,}  missing-fits {n_missing_fits:,}")

    # Keep only candidates that have a thumb
    scr = scr.loc[rows_with_thumb].reset_index(drop=True)

    # Render paginated HTML
    per = args.per_page
    n_pages = (len(scr) + per - 1) // per
    for p in range(1, n_pages + 1):
        rows = scr.iloc[(p - 1) * per: p * per]
        (OUT_DIR / f"page_{p:03d}.html").write_text(
            render_page(p, n_pages, rows, pub_row_ids))
    (OUT_DIR / "index.html").write_text(
        render_index(n_pages, len(scr), len(pub_row_ids)))
    print(f"[done] {n_pages} pages of {per} tiles → {OUT_DIR}/index.html")


if __name__ == "__main__":
    main()
