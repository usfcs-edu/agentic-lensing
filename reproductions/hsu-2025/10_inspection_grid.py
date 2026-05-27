#!/usr/bin/env python3
"""
10_inspection_grid.py

Lay out DR10 JPEG cutouts (from script 09) into inspection grids for visual
sanity-checking our reproduced pair list. NOT a substitute for Hsu's §4 visual
inspection — this is for our own QA / debugging of the algorithmic pipeline.

Modes (mutually exclusive):
  --table2       : grid of the 20 Hsu+2025 Table 2 Grade A new candidates we
                   matched (script 06). 4×5 layout. Annotates with the
                   published name + (z_lens, z_src).
  --top-thetae N : N pairs with the largest estimated θ_E (most likely real
                   strong lenses). Annotates with θ_E and σ_v.
  --random N     : N random pairs (uniformly sampled across the 13,530 set).

Output: figs/inspect_<mode>.jpg  (a single large composite PNG/JPG).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from PIL import Image, ImageDraw, ImageFont


HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
FIGS = HERE / "figs"
CUTOUT_DIR = FIGS / "cutouts_jpeg_dr10"
CLASSIFIED = DATA / "classified_pairs.parquet"
XMATCH = DATA / "xmatch_table2.json"

TILE = 200      # per-cutout panel pixel size
PAD = 24        # padding for label strip
FONT_SIZE = 11


def font():
    try:
        return ImageFont.truetype("DejaVuSans.ttf", FONT_SIZE)
    except OSError:
        return ImageFont.load_default()


def load_cutout(group_id: int) -> Image.Image | None:
    p = CUTOUT_DIR / f"group_{group_id:08d}.jpg"
    if not p.exists() or p.stat().st_size < 256:
        return None
    return Image.open(p).convert("RGB")


def build_grid(panels: list[tuple[int, str]], cols: int, title: str) -> Image.Image:
    rows = (len(panels) + cols - 1) // cols
    W = cols * TILE
    H = rows * (TILE + PAD) + PAD  # extra row at top for title
    canvas = Image.new("RGB", (W, H + PAD), (0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    f = font()
    draw.text((10, 4), title, font=f, fill=(255, 255, 255))
    for i, (gid, label) in enumerate(panels):
        r, c = divmod(i, cols)
        x0 = c * TILE
        y0 = PAD + r * (TILE + PAD)
        img = load_cutout(gid)
        if img is None:
            draw.rectangle((x0, y0, x0 + TILE, y0 + TILE), fill=(40, 40, 40))
            draw.text((x0 + 4, y0 + TILE // 2), f"missing\n{gid}", font=f, fill=(200, 80, 80))
        else:
            img = img.resize((TILE, TILE), Image.LANCZOS)
            canvas.paste(img, (x0, y0))
        # label strip below
        draw.rectangle((x0, y0 + TILE, x0 + TILE, y0 + TILE + PAD),
                       fill=(0, 0, 0))
        # word-wrap label across two short lines if it has a comma
        if "\n" in label:
            for k, line in enumerate(label.split("\n")):
                draw.text((x0 + 4, y0 + TILE + 2 + k * (FONT_SIZE + 1)),
                          line, font=f, fill=(220, 220, 220))
        else:
            draw.text((x0 + 4, y0 + TILE + 4), label, font=f, fill=(220, 220, 220))
    return canvas


def mode_table2() -> tuple[list[tuple[int, str]], int, str]:
    xm = json.loads(XMATCH.read_text())
    panels = []
    for m in xm["matches"]:
        gid = int(m["nearest_group_id"])
        zs = sorted(m["member_z"])
        label = f"{m['name']}\nz=({zs[0]:.2f}, {zs[-1]:.2f})"
        panels.append((gid, label))
    return panels, 5, "Hsu+2025 Table 2 — 20 Grade A new candidates (algorithmic match)"


def mode_top_thetae(n: int) -> tuple[list[tuple[int, str]], int, str]:
    df = pq.read_table(CLASSIFIED).to_pandas()
    df = df[np.isfinite(df["theta_E_arcsec"]) & (df["theta_E_arcsec"] > 0)]
    df = df.nlargest(n, "theta_E_arcsec")
    panels = []
    for _, r in df.iterrows():
        gid = int(r["group_id"])
        label = (f"θ_E={r['theta_E_arcsec']:.2f}″\n"
                 f"σ={r['sigma_v_lens']:.0f}km/s "
                 f"z=({r['Z_lens']:.2f},{r['Z_src']:.2f})")
        panels.append((gid, label))
    cols = int(np.ceil(np.sqrt(n)))
    return panels, cols, f"Top {n} pairs by estimated θ_E"


def mode_random(n: int, seed: int = 0) -> tuple[list[tuple[int, str]], int, str]:
    df = pq.read_table(CLASSIFIED).to_pandas()
    rng = np.random.default_rng(seed)
    pick = df.iloc[rng.choice(len(df), size=n, replace=False)]
    panels = []
    for _, r in pick.iterrows():
        gid = int(r["group_id"])
        sigma = r["sigma_v_lens"]
        sigma_str = f"{sigma:.0f}" if np.isfinite(sigma) and sigma > 0 else "—"
        label = (f"g{gid}\n"
                 f"z=({r['Z_lens']:.2f},{r['Z_src']:.2f}) σ={sigma_str}")
        panels.append((gid, label))
    cols = int(np.ceil(np.sqrt(n)))
    return panels, cols, f"Random sample of {n} pairs"


def main() -> None:
    ap = argparse.ArgumentParser()
    mx = ap.add_mutually_exclusive_group(required=True)
    mx.add_argument("--table2", action="store_true")
    mx.add_argument("--top-thetae", type=int, metavar="N")
    mx.add_argument("--random", type=int, metavar="N")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.table2:
        panels, cols, title = mode_table2()
        out = FIGS / "inspect_table2.jpg"
    elif args.top_thetae:
        panels, cols, title = mode_top_thetae(args.top_thetae)
        out = FIGS / f"inspect_top_thetae_{args.top_thetae}.jpg"
    else:
        panels, cols, title = mode_random(args.random, seed=args.seed)
        out = FIGS / f"inspect_random_{args.random}.jpg"

    img = build_grid(panels, cols, title)
    img.save(out, quality=92)
    print(f"[done] wrote {out}  ({img.size[0]}×{img.size[1]} px)")


if __name__ == "__main__":
    main()
