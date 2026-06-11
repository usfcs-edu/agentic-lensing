#!/usr/bin/env python3
"""
16_build_inspection_viewer.py — Phase 4c.

Paginated static HTML viewer of the top-N DR8 candidates for one model, with
Lupton-stretched grz thumbnails. Tiles matching a published Huang+2021
candidate (5″) are flagged ★. Clone of huang-2020/16 pointed at the DR8 outputs
and a selectable model.

Usage:
  ./16_build_inspection_viewer.py --model shielded --top-n 2000 --per-page 50
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


def fits_to_thumb(path: Path) -> Image.Image | None:
    try:
        with fits.open(path) as hdul:
            cube = hdul[0].data.astype(np.float32)
    except Exception:
        return None
    if cube is None or cube.shape != (3, 101, 101):
        return None
    rgb = make_lupton_rgb(cube[2], cube[1], cube[0], Q=LUPTON_Q, stretch=LUPTON_STRETCH)
    return Image.fromarray(rgb[::-1, :, :]).resize((THUMB_PX, THUMB_PX), Image.NEAREST)


def tile_html(row, published: set[str]) -> str:
    rid = html.escape(str(row["row_id"]))
    badge = ' <span style="color:#d7191c">★</span>' if rid in published else ""
    return (f'<div class="tile"><img src="thumbs/{rid}.png" alt="{rid}">'
            f'<div class="meta"><div><b>{rid}</b>{badge}</div>'
            f'<div>{row["ra"]:.4f}, {row["dec"]:+.4f}</div>'
            f'<div>p = {row["score"]:.4f}</div></div></div>')


def render_page(page_idx, n_pages, rows, published, model) -> str:
    nav = []
    if page_idx > 1:
        nav.append(f'<a href="page_{page_idx-1:03d}.html">‹ prev</a>')
    if page_idx < n_pages:
        nav.append(f'<a href="page_{page_idx+1:03d}.html">next ›</a>')
    navbar = " &nbsp;|&nbsp; ".join(nav)
    head = (f'<a href="index.html">index</a> &nbsp;|&nbsp; page {page_idx} of {n_pages}'
            + (' &nbsp;|&nbsp; ' + navbar if navbar else ''))
    tiles = "\n".join(tile_html(r, published) for _, r in rows.iterrows())
    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>Phase 4 DR8 {model} — page {page_idx}</title><style>
body{{font-family:system-ui,sans-serif;margin:14px;background:#1a1a1a;color:#ddd}}
a{{color:#6cf}} nav{{margin-bottom:12px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:10px}}
.tile{{background:#222;border:1px solid #333;padding:6px;border-radius:4px;text-align:center}}
.tile img{{display:block;width:200px;height:200px;margin:0 auto;image-rendering:pixelated}}
.meta{{font-size:12px;margin-top:4px;line-height:1.4}} .meta b{{color:#fff;font-family:monospace}}
</style></head><body><nav>{head}</nav><div class="grid">{tiles}</div></body></html>"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=("l18", "shielded", "combined"), default="shielded")
    ap.add_argument("--scores", default=None)
    ap.add_argument("--published", default=str(DATA / "huang2021_published_catalog.csv"))
    ap.add_argument("--top-n", type=int, default=2000)
    ap.add_argument("--per-page", type=int, default=50)
    args = ap.parse_args()

    scores = args.scores or str(DATA / f"inference_scores_{args.model}_dr8.parquet")
    out_dir = HERE / "papers" / "figures" / f"inspection_{args.model}"
    thumb_dir = out_dir / "thumbs"
    out_dir.mkdir(parents=True, exist_ok=True)
    thumb_dir.mkdir(parents=True, exist_ok=True)

    scr = pd.read_parquet(scores).sort_values("score", ascending=False).head(args.top_n)
    scr = scr.reset_index(drop=True)
    print(f"[init] top {len(scr):,} {args.model} candidates "
          f"(p {scr['score'].min():.4f}–{scr['score'].max():.4f})")

    pub = pd.read_csv(args.published)
    pub_sky = SkyCoord(ra=pub["RA"].values * u.deg, dec=pub["DEC"].values * u.deg)
    top_sky = SkyCoord(ra=scr["ra"].values * u.deg, dec=scr["dec"].values * u.deg)
    _, sep, _ = top_sky.match_to_catalog_sky(pub_sky)
    pub_ids = set(scr.loc[sep.to(u.arcsec).value < 5.0, "row_id"].astype(str))
    print(f"[init] {len(pub_ids)} of top-{len(scr)} match a published Huang+2021 candidate")

    keep = []
    for i, row in tqdm(scr.iterrows(), total=len(scr), desc="thumbs"):
        rid = str(row["row_id"])
        out = thumb_dir / f"{rid}.png"
        if out.exists() and out.stat().st_size > 0:
            keep.append(i); continue
        fp = FITS_DIR / f"{rid}.fits"
        if not fp.exists():
            continue
        img = fits_to_thumb(fp)
        if img is None:
            continue
        img.save(out); keep.append(i)
    scr = scr.loc[keep].reset_index(drop=True)

    per = args.per_page
    n_pages = (len(scr) + per - 1) // per
    for p in range(1, n_pages + 1):
        rows = scr.iloc[(p - 1) * per: p * per]
        (out_dir / f"page_{p:03d}.html").write_text(
            render_page(p, n_pages, rows, pub_ids, args.model))
    idx = "".join(f'<li><a href="page_{i:03d}.html">page {i}</a></li>'
                  for i in range(1, n_pages + 1))
    (out_dir / "index.html").write_text(
        f"""<!doctype html><html><head><meta charset="utf-8"><title>Phase 4 DR8 {args.model}</title>
<style>body{{font-family:system-ui,sans-serif;margin:20px;background:#1a1a1a;color:#ddd;max-width:720px}}
a{{color:#6cf}} ul{{columns:4;list-style:none;padding-left:0}}</style></head><body>
<h2>Phase 4 DR8 inspection — {args.model} model</h2>
<p>{len(scr):,} cutouts sorted by score. ★ = published Huang+2021 candidate (5″).</p>
<ul>{idx}</ul></body></html>""")
    print(f"[done] {n_pages} pages → {out_dir}/index.html")


if __name__ == "__main__":
    main()
