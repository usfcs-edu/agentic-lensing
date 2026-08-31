"""Build ``examples/nine/`` from the deck: split the un-annotated 3x3 grid (slide 1, ``ppt/media/image6.jpg``)
into nine tiles and write a campaign config with the ranks / reference theta_E from slides 10-18.

The deck is read from the pptx zip directly (never modified). The tiles are lossy ~410 px JPEG crops
of unknown survey pixel scale, so ``cutout_arcsec`` is recorded as *assumed* (16", deck PROMPT 3): fine
as UI/test fixtures, not as few-shot exemplars.
"""
from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from . import config

DECK = Path(__file__).resolve().parents[1] / "examples" / "Xiaosheng-Claude-Fable5-Lens-Annotations.pptx"
GRID_MEMBER = "ppt/media/image6.jpg"
# slides 10-18, deck order (row-major on the 3x3 grid): rank, theta_E (arcsec), theta_E alt / method note
DECK_SYSTEMS = [
    (91, 1.50, "geometric"), (89, 2.00, "geometric"), (68, 2.13, None),
    (74, 2.92, "alt 2.09"), (60, 2.21, None), (41, 2.40, "geometric"),
    (30, 2.77, "alt 2.04"), (24, 1.51, None), (40, 2.47, None),
]


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) runs of True in a 1-D boolean array."""
    runs, start = [], None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        elif not m and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(mask)))
    return runs


def split_grid(im: Image.Image, n: int = 3, flat_std: float = 4.0, min_tile: int = 100) -> list[tuple[int, int, int, int]]:
    """Find the flat gutters separating the n x n tiles; return n*n (left, upper, right, lower) boxes.

    Sky is dark but noisy (column std >= 7 in the deck grid); gutters and the outer border are a flat
    grey (std < 3.5 even after JPEG ringing). A column/row is a gutter when its std is below
    ``flat_std``; tiles are the wide (> ``min_tile``) runs between gutters.
    """
    a = np.asarray(im.convert("L"), dtype=np.float32)

    def bright_runs(axis: int) -> list[tuple[int, int]]:
        dark_mask = a.std(axis=axis) < flat_std
        runs = [r for r in _runs(~dark_mask) if r[1] - r[0] > min_tile]
        if len(runs) != n:
            raise RuntimeError(f"expected {n} tiles along axis {axis}, found {len(runs)}: {runs}")
        return runs

    cols, rows = bright_runs(0), bright_runs(1)
    return [(c0, r0, c1, r1) for (r0, r1) in rows for (c0, c1) in cols]


def build(out_dir: Path, deck: Path = DECK, force: bool = False) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(deck) as z:
        grid = Image.open(io.BytesIO(z.read(GRID_MEMBER))).convert("RGB")
    boxes = split_grid(grid)
    written: list[Path] = []
    overrides = {}
    for i, (box, (rank, te, note)) in enumerate(zip(boxes, DECK_SYSTEMS), start=1):
        image_id = f"deck-{i:02d}"
        p = out_dir / f"{image_id}.png"
        if force or not p.exists():
            tile = grid.crop(box)
            side = min(tile.size)
            tile = tile.crop((0, 0, side, side))       # square, from the top-left
            tile.save(p, format="PNG", optimize=False)
        written.append(p)
        overrides[image_id] = {"rank": rank, "theta_e_ref_arcsec": te, "theta_e_ref_note": note,
                               "deck_slide": 9 + i}
    cfg = dict(config.CAMPAIGN_DEFAULTS)
    cfg.update({
        "campaign": "deck-nine",
        "note": "Nine strong-lens candidates from Xiaosheng's deck (slide 1 grid). Survey/pixel scale not "
                "recorded in the deck; 16 arcsec per PROMPT 3 is ASSUMED.",
        "cutout_arcsec": 16.0, "cutout_arcsec_source": "assumed",
        "overrides": overrides,
    })
    with open(out_dir / config.CAMPAIGN_CONFIG_NAME, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return written
