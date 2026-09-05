#!/usr/bin/env python3
"""Measure the AstroMark notation arms, and build the comparison sheets.

    python measure.py metrics            # the numeric table
    python measure.py sheets             # contact sheets, CVD sheets, polarity triad, A/B pages
    python measure.py all

WHAT IS MEASURED, AND THE ONE TRAP

  ink %            overlay pixels differing from the null arm by >10/255, over panel pixels.
  occlusion %      the same, restricted to INFORMATIVE base pixels (luminance > sky + 3 sigma).
                   This is "let me see more of what's behind it" as a number.
  ink/statement    ink % divided by the number of marks the arm actually drew.

  The trap: raw ink is NOT comparable across arms that express different amounts. The two reference
  arms cannot draw bound rings or segmentation polygons at all, so they look thinner for free. An
  arm is penalised for saying more, which is the opposite of what the comparison is for. Both raw
  and normalised figures are therefore reported, and the statement count is printed beside them, so
  nobody reads the raw column alone.

  contrast p5      5th-percentile |L_ink - L_local| over stroke pixels, local = 5-px annulus median.
                   This is the number that says whether the casing works over a bright core, where
                   a thin unhaloed stroke fails and a heavy black halo passes but looks blocky.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import cvd
from render_proposals import ARMS, UPSCALE

HERE = Path(__file__).parent
REF = HERE / "reference"
PANEL = None          # set from the null arm


def load(arm: str, panel_only: bool = True) -> np.ndarray:
    im = Image.open(REF / f"{arm}.png").convert("RGB")
    if panel_only:
        n = im.width
        im = im.crop((0, 0, n, n))
    return np.asarray(im, dtype=np.float64)


def informative_mask(base: np.ndarray) -> np.ndarray:
    lum = base @ [0.2126, 0.7152, 0.0722]
    sky = float(np.median(lum))
    sig = float(np.std(lum[lum < np.percentile(lum, 80)]))
    return lum > sky + 3.0 * sig


def local_median(lum: np.ndarray, r: int = 5) -> np.ndarray:
    """Median of a (2r+1) box, computed by shifted stacking — no scipy."""
    stack = []
    for dy in range(-r, r + 1, 2):
        for dx in range(-r, r + 1, 2):
            stack.append(np.roll(np.roll(lum, dy, 0), dx, 1))
    return np.median(np.stack(stack), axis=0)


def statement_count(content: dict, arm: str) -> int:
    """How many marks this arm actually renders — the denominator for a fair ink figure."""
    marks = content["marks"]
    if arm == "NULL":
        return 0
    if arm == "R-CURRENT":
        # cannot express bound rings, polygons, treatment, polarity, or source grouping
        return sum(1 for m in marks
                   if not (m["role"] == "einstein_ring" and m.get("bound") != "nominal")
                   and m["geometry"]["kind"] != "polygon")
    if arm == "R-CAMPAIGN":
        return sum(1 for m in marks
                   if not (m["role"] == "einstein_ring" and m.get("bound") != "nominal")
                   and m["geometry"]["kind"] != "polygon")
    return len(marks)


def metrics(arms: list[str]) -> list[dict]:
    base = load("null")
    inf = informative_mask(base)
    blum = base @ [0.2126, 0.7152, 0.0722]
    loc = local_median(blum)
    content = json.loads((HERE / "content.json").read_text())
    rows = []
    for arm in arms:
        if arm == "NULL":
            continue
        a = load(arm.lower())
        d = np.abs(a - base).max(axis=2) > 10
        # Contrast is measured on stroke CORES, not on the anti-aliased boundary: every stroke has
        # a fading edge, and including it drags the 5th percentile toward zero for any arm, which
        # would make the metric measure supersampling rather than legibility.
        core = np.abs(a - base).max(axis=2) > 60
        alum = a @ [0.2126, 0.7152, 0.0722]
        contrast = np.abs(alum[core] - loc[core]) / 255.0 if core.any() else np.array([0.0])
        n = statement_count(content, arm)
        ink = 100.0 * d.mean()
        rows.append({
            "arm": arm,
            "title": ARMS[arm][0],
            "statements": n,
            "ink_pct": round(ink, 2),
            "ink_per_statement": round(ink / n, 3) if n else None,
            "occlusion_pct": round(100.0 * (d & inf).sum() / inf.sum(), 2),
            "contrast_p5": round(float(np.percentile(contrast, 5)), 3),
            "contrast_median": round(float(np.median(contrast)), 3),
        })
    return rows


# --- sheets -------------------------------------------------------------------------------------

def font(sz: int, bold=False):
    root = HERE.resolve().parents[3] / "apps/LensMark/lensmark/render/fonts"
    return ImageFont.truetype(str(root / ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")), sz)


def label_strip(img: Image.Image, text: str, sub: str = "", h: int = 40) -> Image.Image:
    out = Image.new("RGB", (img.width, img.height + h), (12, 12, 14))
    out.paste(img, (0, 0))
    d = ImageDraw.Draw(out)
    d.text((8, img.height + 5), text, font=font(15, True), fill=(255, 255, 255))
    if sub:
        d.text((8, img.height + 20), sub, font=font(12), fill=(165, 165, 172))
    return out


def grid(tiles: list[Image.Image], cols: int, pad: int = 10, bg=(8, 8, 10)) -> Image.Image:
    if not tiles:
        raise ValueError("no tiles")
    w, h = tiles[0].size
    rows = (len(tiles) + cols - 1) // cols
    out = Image.new("RGB", (cols * w + pad * (cols + 1), rows * h + pad * (rows + 1)), bg)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        out.paste(t, (pad + c * (w + pad), pad + r * (h + pad)))
    return out


def contact_sheet(arms: list[str], size: int, out: Path):
    """The thumbnail test. Panels only — the caption band is not available at this size."""
    tiles = []
    for arm in arms:
        im = Image.open(REF / f"{arm.lower()}.png").convert("RGB")
        im = im.crop((0, 0, im.width, im.width)).resize((size, size), Image.LANCZOS)
        tiles.append(label_strip(im, arm, ARMS[arm][0], h=38))
    grid(tiles, cols=min(4, len(tiles))).save(out, optimize=False)
    print(f"wrote {out}  ({size} px/panel)")


def cvd_sheet(arms: list[str], out: Path, size: int = 330):
    """One row per arm, one column per vision condition. The accessibility evidence, visually."""
    conds = [(None, "normal"), ("deuteranopia", "deuteranopia"), ("protanopia", "protanopia"),
             ("tritanopia", "tritanopia"), ("greyscale", "greyscale")]
    tiles = []
    for arm in arms:
        im = Image.open(REF / f"{arm.lower()}.png").convert("RGB")
        im = im.crop((0, 0, im.width, im.width))
        for kind, name in conds:
            a = np.asarray(im, dtype=np.float64) / 255.0
            sim = cvd.simulate(a, kind) if kind else a
            t = Image.fromarray((sim * 255 + 0.5).astype(np.uint8)).resize((size, size), Image.LANCZOS)
            tiles.append(label_strip(t, f"{arm}", name, h=38))
    grid(tiles, cols=len(conds)).save(out, optimize=False)
    print(f"wrote {out}")


def polarity_triad(out: Path):
    """One document rendered three times, all marks positive / negative / ambiguous, labels off.
    If the three are indistinguishable without reading, the polarity channel has failed."""
    import render_proposals as rp
    content = json.loads((HERE / "content.json").read_text())
    base = Image.open(REF / "astromark-ref.png").convert("RGB")
    tiles = []
    for pol in ("positive", "negative", "ambiguous"):
        c = json.loads(json.dumps(content))
        for m in c["marks"]:
            if m["role"] in rp.DESIGNATING:
                m["polarity"] = pol
                m["label"] = ""
                if pol != "positive":
                    m["alternative"] = ""
                m.pop("emphasis", None)
        img = rp.render_arm("N1", c, base, caption=False)
        tiles.append(label_strip(img.resize((380, 380), Image.LANCZOS), f"all {pol}",
                                 "labels removed", h=30))
    grid(tiles, cols=3).save(out, optimize=False)
    print(f"wrote {out}")


def ab_pages(arms: list[str], out: Path):
    tiles = []
    for arm in arms:
        im = Image.open(REF / f"{arm.lower()}.png").convert("RGB")
        tiles.append(label_strip(im.resize((im.width // 2, im.height // 2), Image.LANCZOS),
                                 arm, ARMS[arm][0]))
    grid(tiles, cols=4).save(out, optimize=False)
    print(f"wrote {out}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["metrics", "sheets", "all"])
    args = ap.parse_args()
    arms = [a for a in ARMS if a != "NULL"]

    if args.cmd in ("metrics", "all"):
        rows = metrics(arms)
        hdr = f'{"arm":<12}{"stmts":>6}{"ink %":>8}{"ink/stmt":>10}{"occl %":>9}{"contrast p5":>13}'
        print(hdr)
        print("-" * len(hdr))
        for r in rows:
            print(f'{r["arm"]:<12}{r["statements"]:>6}{r["ink_pct"]:>8.2f}'
                  f'{r["ink_per_statement"]:>10.3f}{r["occlusion_pct"]:>9.2f}{r["contrast_p5"]:>13.3f}')
        (REF / "metrics.json").write_text(json.dumps(rows, indent=2) + "\n")
        print(f"\nwrote {REF / 'metrics.json'}")

    if args.cmd in ("sheets", "all"):
        for size, name in ((528, "528"), (328, "328"), (260, "260")):
            contact_sheet(arms, size, REF / f"contact-{name}.png")
        cvd_sheet(["N1", "N2", "N3", "N4", "R-CURRENT"], REF / "cvd-sheet.png")
        polarity_triad(REF / "polarity-triad.png")
        ab_pages(arms, REF / "ab-page.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
