#!/usr/bin/env python3
"""250_build_gallery.py — per-candidate composite gallery PNGs for the report.

For each qualified candidate (and, optionally, the escalation set), stitch its
full|zoom|residual views side-by-side with a caption strip (row_id, RA/DEC,
p_final, q_group, both grades) into one PNG the Markdown/LaTeX report embeds.

    /home2/benson/.venvs/claudenet/bin/python campaign/250_build_gallery.py [--also-escalation]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "v2" / "campaign"
PNG = OUT / "png"
GAL = OUT / "gallery"
VIEWS = ("full", "zoom", "residual")
TILE = 300


def _font(sz):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
              "/usr/share/fonts/dejavu/DejaVuSans.ttf"):
        if Path(p).exists():
            return ImageFont.truetype(p, sz)
    return ImageFont.load_default()


def composite(row, out_path: Path):
    imgs = []
    for v in VIEWS:
        p = PNG / row["row_id"] / f"{v}.png"
        im = Image.open(p).convert("RGB").resize((TILE, TILE), Image.NEAREST)
        d = ImageDraw.Draw(im)
        d.text((5, 5), v, fill=(255, 255, 0), font=_font(16))
        imgs.append(im)
    strip = 64
    W = TILE * len(VIEWS)
    canvas = Image.new("RGB", (W, TILE + strip), (0, 0, 0))
    for i, im in enumerate(imgs):
        canvas.paste(im, (i * TILE, 0))
    d = ImageDraw.Draw(canvas)
    f = _font(15); fb = _font(17)
    cap1 = (f"{row['row_id']}   RA={row['RA']:.5f}  DEC={row['DEC']:.5f}   "
            f"p_final={row['p_final']:.3f}  q_group={row.get('q_group', float('nan')):.2e}")
    cap2 = (f"tier={row.get('tier','-')}   visual={row.get('my_grade','-')}   "
            f"lensjudge={row.get('lensjudge_grade','-')}   status={row.get('status','-')}")
    d.text((6, TILE + 6), cap1, fill=(230, 230, 230), font=f)
    d.text((6, TILE + 34), cap2, fill=(140, 220, 140), font=fb)
    canvas.save(out_path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--topnew", type=int, default=12,
                    help="also render the top-N still-NEW candidates by consensus_p")
    args = ap.parse_args()
    GAL.mkdir(parents=True, exist_ok=True)
    full = pd.read_parquet(OUT / "consensus_full_737.parquet")
    full["row_id"] = full["row_id"].astype(str)

    # 1. qualified (still-NEW & both >=B) — may be empty
    qual = pd.read_parquet(OUT / "candidates_qualified.parquet")
    for _, r in qual.iterrows():
        composite(r, GAL / f"{r['tier']}_{int(r['rank']):03d}_{r['row_id']}.png")
    print(f"[250] {len(qual)} qualified gallery composites")

    # 2. recovered: candidates either grader rates >=B (the validation set —
    #    these turn out to be already-known lenses)
    rec = full[(full.my_grade.isin(["A", "B"])) | (full.lensjudge_grade.isin(["A", "B"]))].copy()
    rec = rec.sort_values("consensus_p", ascending=False)
    for i, (_, r) in enumerate(rec.iterrows(), 1):
        composite(r, GAL / f"recovered_{i:02d}_{r['row_id']}.png")
    print(f"[250] {len(rec)} 'recovered' (>=B by either grader) composites")

    # 3. top still-NEW by consensus probability (the best of the genuinely-new)
    new = full[full.status == "NEW"].sort_values("consensus_p", ascending=False).head(args.topnew)
    for i, (_, r) in enumerate(new.iterrows(), 1):
        composite(r, GAL / f"topnew_{i:02d}_{r['row_id']}.png")
    print(f"[250] {len(new)} top still-NEW composites -> {GAL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
