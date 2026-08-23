#!/usr/bin/env python3
"""golden/render_v2.py — the "jwst_v2r" composite: panel (f) replaced by a signed-chi SW | LW montage.

Why: panel (f) of the run's composite is `jwst_fetch.radial_profile_subtract` — a CIRCULAR
median radial profile about the stamp centre, subtracted from a 3.5" zoom. On any elliptical,
barred or inclined deflector it leaves a four-lobed butterfly / bowtie plus concentric rings,
and the incumbent personas attacked that artefact instead of the sky (diag_forensics §C3: 88 %
of known-lens rejections invoke an over-subtraction residual; the COWLS AAAAAB lens at
theta_E 1.0" was rejected by all three as a "Sérsic bowtie"; the PI's reading of the annotated
top-100 reached the same verdict on the residual's quality). This module keeps the other five
panels BYTE-IDENTICAL (it pastes
into the v1 JPEG) and replaces only slot (f) with the honest residual LensJudge already uses at
DESI (`common/render.py`: sigma-clipped median in ELLIPTICAL annuli carrying the galaxy's own
eps/PA, chi = residual / robust sigma, fixed +/-5 sigma RdBu diverging scale, one tile per band
so cross-band coherence is a visible test). It is an ARM (A3 = advocate-only on this render,
the gated panel-level arm R), never the default: at DESI the honest residual ran hotter with no
AUC gain and its description was load-bearing (R9/R10), so the image and `render_v2_desc.md`
ship as one unit and the arm is FPR-matched, never compared raw.

What it does (compose_v2):
  1. read `<stamp_dir>/<id>_{SW,LW}_10as.fits` (320 px at 0.03125"/px, row 0 = North); a
     channel counts as present when its file exists and its header FINITE >= MIN_FINITE — the
     same gate that decided the v1 layout, so a gray-layout stamp gets a single tile;
  2. galaxy shape: `render._estimate_shape` on a STACKED pseudo-cube [SW, (SW+LW)/2, LW]
     (it reads cube[1]+cube[2]; a single band is stacked three times) restricted to the central
     64-px box (2") — on the full frame the second moments latch onto neighbours
     (diag_assets §4). The box centroid is offset back to stamp coordinates and REPORTED; the
     model centre itself is FIXED at the catalogued galaxy (159.5, 159.5);
  3. per band: `_elliptical_median_model(band, 159.5, 159.5, eps, theta)` on the full 10"
     stamp, `_chi_band` (residual / `_robust_sigma`), central 112-px (3.5") crop,
     `_diverging_rgb(lim=5)` -> a 112x112 RGB tile, NOT vertically flipped (JWST stamps are
     already North-up with origin='upper', unlike DESI cubes);
  4. montage: the tiles NEAREST-upscaled to 119 px with a 2-px white divider = 240 px wide,
     vertically centred in the 240-px slot (a single tile fills the slot); the yellow centre
     ticks of the other panels are redrawn on every tile; the slot label becomes
     'signed chi SW | LW 3.5"';
  5. paste at the measured slot (x=504, y=292, 240 px; label strip y=274..292) of the v1
     composite and write a 752x562 JPEG (quality 95 like _v1.jpg). Returns
     (render_sha16 of the written bytes, desc_sha16 of render_v2_desc.md).

Measured on real stamps (2026-08-23): the elliptical model cuts the 0.4-1.5" annulus RMS of
the COWLS AAAAAB lens from 27 sigma (circular) to 5.9 sigma, but JWST S/N is high enough that
a few-percent model mismatch inside ~1" still saturates the +/-5 sigma display on bright or
elongated deflectors (the description says to ignore symmetric lobes there). `--engine
isophote` (photutils, radius-dependent eps/PA, ~3-4 s/band; 3.2 sigma on the same lens) and
`--lim` are integrator options that produce a DIFFERENT, separately-tagged render
(`render_tag`); the contract's defaults are what this module registers as "jwst_v2r".

CLI: one v2r composite per manifest row, then the footer-cropped q92 kit JPEG the model is
served (752x540, like golden/kits_truth/), plus a small CSV the integrator merges into the
pinned manifest as `image_path_v2r` / `render_sha_v2r` (this script never rewrites the manifest):

  python lensjudge/golden/render_v2.py --manifest golden/truth_manifest.csv \\
      --out-dir golden/kits_truth_v2r/ [--stamps-dir golden/stamps] [--limit N]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from lensjudge.common import jwst_fetch, render  # noqa: E402
from lensjudge.golden import _util  # noqa: E402

RENDER_VERSION = "jwst_v2r"
DESC_PATH = _util.HERE / "render_v2_desc.md"
STAMPS_DIR = _util.HERE / "stamps"
OUT_DIR = _util.HERE / "kits_truth_v2r"
# --- the slot, measured from jwst_fetch.render_cutout (px=240, gap=8, th=18, foot_h=22):
#     row r: y = gap + r*(px+th+gap) + th ; col c: x = gap + c*(px+gap)
PX, GAP, TH = 240, 8, 18
SLOT_F = (GAP + 2 * (PX + GAP), GAP + 1 * (PX + TH + GAP) + TH)        # (504, 292)
LABEL_XY = (SLOT_F[0] + 1, SLOT_F[1] - TH + 3)                         # (505, 277) as the run draws it
BG = (12, 12, 16)                 # canvas colour of render_cutout
LABEL_FILL = (205, 205, 215)
OUTLINE = (70, 70, 80)
TICK = (255, 215, 90)
DIVIDER = 2                       # px of white between the two tiles
ZOOM_ARCSEC = jwst_fetch.ZOOM_ARCSEC                                   # 3.5
CROP_PX = int(round(jwst_fetch.OUT_PX * ZOOM_ARCSEC / jwst_fetch.CUT_ARCSEC))   # 112 native px
SHAPE_BOX = 64                    # central box (2") the eps/PA are measured in
CENTRE = ((jwst_fetch.OUT_PX - 1) / 2.0, (jwst_fetch.OUT_PX - 1) / 2.0)         # (159.5, 159.5)
CHI_LIM = 5.0                     # the contract's fixed +/-5 sigma display (render.RESID_SIGMA_LIM)
ENGINE = "median"                 # the contract's model; "isophote" = render._model_band's opt-in photutils
                                  # engine (radius-dependent eps/PA, ~3-4 s/band, median fallback) — an
                                  # integrator option, NOT the registered render (see render_tag)
KIT_JPEG_KW = dict(quality=92, optimize=True)   # build_kit.JPEG_KW — the served-kit encoding


def render_tag(engine: str = ENGINE, lim: float = CHI_LIM) -> str:
    """The render_version string: the contract's defaults are plain "jwst_v2r"; any other
    engine/limit is a DIFFERENT render and says so (its image sha changes with it)."""
    tag = RENDER_VERSION
    if engine != "median":
        tag += f"-{engine}"
    if float(lim) != CHI_LIM:
        tag += f"-lim{lim:g}"
    return tag


def desc_text() -> str:
    return DESC_PATH.read_text()


def desc_sha16() -> str:
    return _util.sha_text(desc_text())


# ------------------------------------------------------------------ the residual
def estimate_shape_central(sw: Optional[np.ndarray], lw: Optional[np.ndarray],
                           box: int = SHAPE_BOX) -> dict:
    """eps / theta of the deflector from `render._estimate_shape` on the stacked pseudo-cube
    [SW, (SW+LW)/2, LW] (one band stacked thrice when the other is absent) restricted to the
    central `box` px. Returns {eps, theta, x0, y0 (box centroid offset back to stamp
    coordinates), x0_box, y0_box}."""
    bands = [b for b in (sw, lw) if b is not None]
    if not bands:
        raise ValueError("at least one band is required")
    a = np.nan_to_num(np.asarray(bands[0], float))
    b = np.nan_to_num(np.asarray(bands[-1], float))
    cube = np.stack([a, (a + b) / 2.0, b])
    n = cube.shape[1]
    lo = n // 2 - box // 2
    sub = cube[:, lo:lo + box, lo:lo + box]
    x0b, y0b, eps, theta = render._estimate_shape(sub)
    return {"eps": float(eps), "theta": float(theta), "x0": float(x0b + lo), "y0": float(y0b + lo),
            "x0_box": float(x0b), "y0_box": float(y0b), "box_origin": int(lo)}


def chi_band(band: np.ndarray, eps: float, theta: float, centre=CENTRE, engine: str = ENGINE) -> np.ndarray:
    """Signed chi of one 10" band against the elliptical median model centred on the
    catalogued galaxy (NaNs -> 0 before modelling; the model carries the stamp's sky)."""
    b = np.nan_to_num(np.asarray(band, float))
    return render._chi_band(b, (centre[0], centre[1], eps, theta), engine)


def central_crop(a: np.ndarray, k: int = CROP_PX) -> np.ndarray:
    """The run's `_zoom` crop: k px about n//2 (for 320 -> 112 the box is rows 104..215)."""
    n = a.shape[0]
    c, h = n // 2, k // 2
    return a[c - h:c - h + k, c - h:c - h + k]


def chi_tile(band: np.ndarray, eps: float, theta: float, lim: float = CHI_LIM,
             engine: str = ENGINE) -> np.ndarray:
    """112x112x3 uint8 RdBu tile of the central 3.5" of chi (NaN -> 0 = white). Row 0 stays
    North: no vertical flip (render._img_from_rgb flips because DESI cubes are South-up)."""
    chi = central_crop(chi_band(band, eps, theta, engine=engine))
    return render._diverging_rgb(np.nan_to_num(chi, nan=0.0), lim)


def _draw_ticks(d: ImageDraw.ImageDraw, x: int, y: int, size: int, fov: float = ZOOM_ARCSEC) -> None:
    """The run's four yellow centre ticks, scaled to a tile of `size` px covering `fov` arcsec."""
    cxp, cyp = x + size / 2.0, y + size / 2.0
    off = size * 0.9 / fov
    ln = size * 0.5 / fov
    for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
        d.line([cxp + dx * off, cyp + dy * off, cxp + dx * (off + ln), cyp + dy * (off + ln)],
               fill=TICK, width=1)


def chi_panel(sw: Optional[np.ndarray], lw: Optional[np.ndarray], labels=("SW", "LW"),
              px: int = PX, lim: float = CHI_LIM, engine: str = ENGINE) -> tuple[Image.Image, dict]:
    """The 240x240 replacement for slot (f): chi_SW | chi_LW tiles (or one tile), centre ticks,
    band labels. Returns (panel, info) with info = shape params + n_tiles + per-band sigma."""
    if sw is None and lw is None:
        raise ValueError("both channels absent: nothing to render")
    shape = estimate_shape_central(sw, lw)
    bands = [(lab, b) for lab, b in zip(labels, (sw, lw)) if b is not None]
    tiles = [chi_tile(b, shape["eps"], shape["theta"], lim, engine) for _, b in bands]
    panel = Image.new("RGB", (px, px), BG)
    d = ImageDraw.Draw(panel)
    f9 = jwst_fetch._font(11)
    if len(tiles) == 1:
        im = Image.fromarray(tiles[0]).resize((px, px), Image.NEAREST)
        panel.paste(im, (0, 0))
        _draw_ticks(d, 0, 0, px)
        d.text((4, 3), f"{bands[0][0]} chi", font=f9, fill=LABEL_FILL)
    else:
        size = (px - DIVIDER) // 2                      # 119
        y0 = (px - size) // 2                           # 60: vertically centred
        for i, (lab, _) in enumerate(bands):
            x0 = i * (size + DIVIDER)
            im = Image.fromarray(tiles[i]).resize((size, size), Image.NEAREST)
            panel.paste(im, (x0, y0))
            _draw_ticks(d, x0, y0, size)
            d.text((x0 + 2, y0 - 15), f"{lab} chi", font=f9, fill=LABEL_FILL)
        d.rectangle([size, y0, size + DIVIDER - 1, y0 + size - 1], fill=(255, 255, 255))
    d.rectangle([0, 0, px - 1, px - 1], outline=OUTLINE)
    info = dict(shape, n_tiles=len(tiles), bands=[lab for lab, _ in bands], engine=engine, lim=float(lim))
    return panel, info


def paste_panel_f(composite: Image.Image, panel: Image.Image, label: str) -> Image.Image:
    """Slot (f) of a 752x562 v1 composite replaced by `panel`; the label strip above it
    repainted with `label` exactly where render_cutout draws its own."""
    if composite.size != jwst_fetch.COMPOSITE_SIZE:
        raise ValueError(f"composite is {composite.size}, expected {jwst_fetch.COMPOSITE_SIZE}")
    out = composite.convert("RGB").copy()
    x, y = SLOT_F
    d = ImageDraw.Draw(out)
    d.rectangle([x, y - TH, x + PX - 1, y - 1], fill=BG)            # the old label
    out.paste(panel, (x, y))
    d.text(LABEL_XY, label, font=jwst_fetch._font(11), fill=LABEL_FILL)
    return out


# ------------------------------------------------------------------ stamps in, composite out
def _present(path: Path, min_finite: float = jwst_fetch.MIN_FINITE) -> Optional[np.ndarray]:
    """The 10" band array when the stamp exists and passes the run's finite gate, else None."""
    if not path.exists():
        return None
    arr, hdr = jwst_fetch.read_stamp_fits(path)
    finite = float(hdr.get("FINITE", np.isfinite(arr).mean()))
    if finite < min_finite:
        return None
    return arr


def load_bands(stamp_dir: Path, cid: Optional[str] = None) -> tuple[Optional[np.ndarray], Optional[np.ndarray], dict]:
    """(SW, LW, {filters}) for <stamp_dir>/<id>_{SW,LW}_10as.fits; absent/gated -> None."""
    stamp_dir = Path(stamp_dir)
    cid = cid or stamp_dir.name
    out, filt = [], {}
    for ch in jwst_fetch.CHANNELS:
        p = stamp_dir / f"{cid}_{ch}_10as.fits"
        arr = _present(p)
        out.append(arr)
        if arr is not None:
            try:
                filt[ch] = str(jwst_fetch.read_stamp_fits(p)[1].get("FILTER", "")) or ch
            except Exception:  # noqa: BLE001
                filt[ch] = ch
    return out[0], out[1], filt


def compose_v2(stamp_dir: Path, out_jpg: Path, v1_jpg: Optional[Path] = None,
               quality: int = jwst_fetch.JPEG_QUALITY, lim: float = CHI_LIM,
               engine: str = ENGINE) -> tuple[str, str]:
    """The jwst_v2r composite for one stamp dir -> out_jpg (752x562). Returns
    (render_sha16 of the written JPEG bytes, desc_sha16 of render_v2_desc.md). Raises when
    the v1 composite or both bands are missing (the caller records the failure). `lim` /
    `engine` other than the defaults make a different render (render_tag)."""
    stamp_dir = Path(stamp_dir)
    cid = stamp_dir.name
    v1 = Path(v1_jpg) if v1_jpg else stamp_dir / f"{cid}_v1.jpg"
    if not v1.exists():
        raise FileNotFoundError(f"{v1}: v1 composite missing (build_stamps.py writes it)")
    sw, lw, filt = load_bands(stamp_dir, cid)
    if sw is None and lw is None:
        raise FileNotFoundError(f"{stamp_dir}: no 10\" band passes the finite gate")
    panel, info = chi_panel(sw, lw, lim=lim, engine=engine)
    label = (f'signed chi {" | ".join(info["bands"])} {ZOOM_ARCSEC}"' if info["n_tiles"] == 2
             else f'signed chi {info["bands"][0]} {ZOOM_ARCSEC}"')
    comp = paste_panel_f(Image.open(v1), panel, label)
    out_jpg = Path(out_jpg)
    out_jpg.parent.mkdir(parents=True, exist_ok=True)
    comp.save(str(out_jpg), quality=quality)
    return _util.sha_file(out_jpg), desc_sha16()


def compose_v2_info(stamp_dir: Path, out_jpg: Path, lim: float = CHI_LIM, engine: str = ENGINE,
                    **kw) -> tuple[str, str, dict]:
    """compose_v2 plus the shape/tiles info (for the CSV and tests). The shape is computed
    once more here (cheap: a 64-px moment) so compose_v2's signature stays the contract's."""
    stamp_dir = Path(stamp_dir)
    sw, lw, _ = load_bands(stamp_dir)
    if sw is None and lw is None:
        raise FileNotFoundError(f"{stamp_dir}: no 10\" band passes the finite gate")
    info = dict(estimate_shape_central(sw, lw), n_tiles=int(sw is not None) + int(lw is not None),
                bands=[lab for lab, b in (("SW", sw), ("LW", lw)) if b is not None],
                engine=engine, lim=float(lim))
    rsha, dsha = compose_v2(stamp_dir, out_jpg, lim=lim, engine=engine, **kw)
    return rsha, dsha, info


def crop_to_kit(src: Path, dst: Path) -> str:
    """The served form: footer cropped at y=540 (752x540), quality 92 like build_kit. Returns
    sha16 of the bytes as served (= the manifest's render_sha_v2r)."""
    im = jwst_fetch.crop_footer(Image.open(src).convert("RGB"))
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    im.save(str(dst), **KIT_JPEG_KW)
    return _util.sha_file(dst)


# ------------------------------------------------------------------ CLI
def _ids(manifest: Path) -> list[str]:
    df = _util.read_pinned(manifest, dtype=str) if manifest.with_suffix(".csv.sha").exists() \
        else pd.read_csv(manifest, dtype=str)
    col = "name" if "name" in df.columns else "candidate_id"
    return df[col].astype(str).tolist()


def run(manifest: Path, out_dir: Path, stamps_dir: Path = STAMPS_DIR, limit: Optional[int] = None,
        csv_path: Optional[Path] = None, quiet: bool = False, lim: float = CHI_LIM,
        engine: str = ENGINE) -> pd.DataFrame:
    ids = _ids(Path(manifest))
    if limit:
        ids = ids[:limit]
    out_dir = Path(out_dir).resolve()          # the CSV's image_path_v2r is absolute whatever the cwd
    full_dir = out_dir / "full"
    rows = []
    dsha = desc_sha16()
    tag = render_tag(engine, lim)
    for cid in ids:
        sd = Path(stamps_dir) / cid
        row = {"id": cid, "image_path_v2r": "", "render_sha_v2r": "", "render_version": tag,
               "desc_sha16": dsha, "n_tiles": 0, "eps": np.nan, "theta_deg": np.nan,
               "centroid_dx_px": np.nan, "centroid_dy_px": np.nan, "status": ""}
        try:
            full = full_dir / f"{cid}_v2r.jpg"
            _, _, info = compose_v2_info(sd, full, lim=lim, engine=engine)
            kit = out_dir / f"{cid}.jpg"
            row["render_sha_v2r"] = crop_to_kit(full, kit)
            row["image_path_v2r"] = str(kit)
            row.update(n_tiles=info["n_tiles"], eps=round(info["eps"], 4),
                       theta_deg=round(float(np.degrees(info["theta"])), 2),
                       centroid_dx_px=round(info["x0"] - CENTRE[0], 2),
                       centroid_dy_px=round(info["y0"] - CENTRE[1], 2), status="ok")
        except Exception as e:  # noqa: BLE001
            row["status"] = f"{type(e).__name__}: {e}"
        rows.append(row)
        if not quiet:
            print(f"  {cid:22s} {row['status'][:70]}")
    df = pd.DataFrame(rows)
    csv_path = Path(csv_path) if csv_path else out_dir / "render_v2r.csv"
    sha = _util.pin(df, csv_path)
    if not quiet:
        ok = int((df["status"] == "ok").sum())
        print(f"{ok}/{len(df)} rendered as {tag}; desc sha {dsha}; wrote {csv_path} (sha {sha})")
    return df


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, required=True, help="golden/truth_manifest.csv (read only)")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--stamps-dir", type=Path, default=STAMPS_DIR)
    ap.add_argument("--csv", type=Path, help="default <out-dir>/render_v2r.csv (pinned)")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--lim", type=float, default=CHI_LIM, help="display limit in sigma (contract: 5)")
    ap.add_argument("--engine", choices=("median", "isophote"), default=ENGINE,
                    help="smooth-light model (contract: median; isophote = photutils, slower, a different render)")
    a = ap.parse_args(argv)
    run(a.manifest, a.out_dir, a.stamps_dir, a.limit, a.csv, lim=a.lim, engine=a.engine)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
