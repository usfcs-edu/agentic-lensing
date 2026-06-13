#!/usr/bin/env python3
"""201_render_galleries.py — render per-candidate multi-view RGB PNGs for the
visual-judging workflow (220) and the report gallery (250/260).

Reuses lensjudge/common/render.py (the same Lupton Q=8/stretch=0.5 renderer,
400px upsample, 2.5x zoom, lens-light residual, high-contrast) so the images the
human/agent judges see are exactly the lensjudge views — the legibility levers
that surface 4-8px arcs at DECaLS 0.262"/px.

Writes: data/v2/campaign/png/{row_id}/{full,zoom,residual,highcontrast}.png
and back-fills the PNG paths into manifest_737.parquet.

    /home2/benson/.venvs/claudenet/bin/python campaign/201_render_galleries.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]          # reproductions/claudenet
REPRO = ROOT.parent                                 # reproductions/
sys.path.insert(0, str(REPRO))                      # so `import lensjudge` works
from lensjudge.common import render                 # noqa: E402

DATA = ROOT / "data"
SWEEP = DATA / "v2" / "sweep"
OUT = DATA / "v2" / "campaign"
PNG = OUT / "png"
VIEWS = render.VIEWS                                 # full, zoom, residual, highcontrast


def main() -> int:
    PNG.mkdir(parents=True, exist_ok=True)
    man = pd.read_parquet(OUT / "manifest_737.parquet")
    man["row_id"] = man["row_id"].astype(str)

    z = np.load(SWEEP / "vet_topnew.npz")
    loc = {r: i for i, r in enumerate(z["row_ids"].astype(str))}
    cubes = z["cutouts"]

    view_paths = {v: [] for v in VIEWS}
    n = 0
    for r in man.row_id:
        cube = np.asarray(cubes[loc[r]], dtype=np.float32)
        d = PNG / r
        d.mkdir(exist_ok=True)
        imgs = render.render_views(cube, views=VIEWS)
        for v in VIEWS:
            p = d / f"{v}.png"
            render.save_png(imgs[v], p)
            view_paths[v].append(str(p))
        n += 1
        if n % 100 == 0:
            print(f"[201] rendered {n}/{len(man)}", flush=True)

    for v in VIEWS:
        man[f"png_{v}"] = view_paths[v]
    man["png_dir"] = [str(PNG / r) for r in man.row_id]
    man.to_parquet(OUT / "manifest_737.parquet", index=False)
    print(f"[201] rendered {n} candidates x {len(VIEWS)} views -> {PNG}; "
          f"manifest updated with png_* paths")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
