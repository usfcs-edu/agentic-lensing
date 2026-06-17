#!/usr/bin/env python3
"""93_make_gallery.py — cutout atlas for the Table 11 NEW v3 grade-A candidates.

Builds one composite PNG (full | zoom | residual) per object behind Table 11 of
papers/main.pdf (the four panel-qualified NEW v3 grade-A candidates from C-vet),
for the standalone report papers/new_candidates.tex.

The ids come from the exact Table-11 source (v3/cv3_new_gradeA.csv); RA/Dec are
joined from v3/manifests_d2_newA.csv (whose p_meta equals Table 11's C-vet
p_lens). Cutouts are grz ls-dr10 at the training geometry (101 px, 0.262"/px),
fetched + cached and rendered with the team's inspection-viewer renderer
(reused from lensjudge), so these are exactly the pixels the v3 model and the
qualification panel saw.

    python 93_make_gallery.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

# reproductions/claudenet ; put reproductions/ on sys.path so `import lensjudge` works
# (same pattern as campaign/201_render_galleries.py). Paths computed locally to keep
# this a pure fetch/render script with no dependency on the _clib training hub.
ROOT = Path(__file__).resolve().parent
REPRO = ROOT.parent
sys.path.insert(0, str(REPRO))
from lensjudge.common import render, fetch  # noqa: E402

V3 = ROOT / "v3"
FIG = ROOT / "papers" / "figures"
VIEWS = ("full", "zoom", "residual")
TILE = render.config.RENDER_PX if hasattr(render, "config") else 400  # 400 px upsample
LAYER = "ls-dr10"  # all four Table-11 candidates are dr10_south


def _font(sz):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
              "/usr/share/fonts/dejavu/DejaVuSans.ttf",
              "/Library/Fonts/Arial.ttf",
              "/System/Library/Fonts/Supplemental/Arial.ttf"):
        if Path(p).exists():
            return ImageFont.truetype(p, sz)
    return ImageFont.load_default()


def composite(views: dict, out_path: Path) -> None:
    """Stitch full|zoom|residual side-by-side; only a small view label baked in
    (id/coords/scores live in the LaTeX caption). Mirrors campaign/250_build_gallery."""
    imgs = []
    for v in VIEWS:
        im = views[v].convert("RGB").resize((TILE, TILE), Image.NEAREST)
        ImageDraw.Draw(im).text((8, 8), v, fill=(255, 255, 0), font=_font(22))
        imgs.append(im)
    canvas = Image.new("RGB", (TILE * len(imgs), TILE), (0, 0, 0))
    for i, im in enumerate(imgs):
        canvas.paste(im, (i * TILE, 0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def main() -> int:
    ids = pd.read_csv(V3 / "cv3_new_gradeA.csv")        # Table-11 source: name, grade_pred, p_lens, list
    coords = pd.read_csv(V3 / "manifests_d2_newA.csv")  # name, ra, dec, ... (coords; p_meta == C-vet p_lens)
    df = ids.merge(coords[["name", "ra", "dec"]], on="name", how="left")

    missing = df[df["ra"].isna()]
    if len(missing):
        print(f"[93] WARNING: no coordinates for {list(missing['name'])}", file=sys.stderr)

    n_ok = 0
    for _, r in df.dropna(subset=["ra", "dec"]).iterrows():
        name = str(r["name"])
        cube = fetch.get_cube(name=name, ra=float(r["ra"]), dec=float(r["dec"]), survey=LAYER)
        if cube is None:
            print(f"[93] WARNING: cutout fetch failed (off-footprint/network?): {name}", file=sys.stderr)
            continue
        views = render.render_views(cube, views=VIEWS)
        if any(v not in views for v in VIEWS):
            print(f"[93] WARNING: render incomplete for {name}: {sorted(views)}", file=sys.stderr)
            continue
        out = FIG / f"newcand_{name}.png"
        composite(views, out)
        print(f"[93] {name}  RA={float(r['ra']):.5f} DEC={float(r['dec']):+.5f}  -> {out.relative_to(ROOT)}")
        n_ok += 1

    print(f"[93] wrote {n_ok}/{len(df)} composites to {FIG}")
    return 0 if n_ok == len(df) else 1


if __name__ == "__main__":
    raise SystemExit(main())
