"""common/jwst_fetch.py — JWST NIRCam streaming cutouts + the run's 6-panel renderer, vendored.

VENDORED from jwst-strong-lens-search/scripts/util.py @ commit 4f81493 (2026-08-21,
"Add two review sets: clean rank-ordered, and scrambled for blind assessment"), copied
2026-08-22 for the golden single-grader campaign (plan: ~/.claude/plans/
i-want-to-explore-golden-seal.md). The JWST run repo is READ-ONLY for LensJudge; this
copy exists so the golden package can re-fetch FITS stamps and re-render the *exact*
composite the discovery agents saw, without importing across repos or writing there.

What is verbatim (the section marked VENDORED): S3/MAST roots, SW/LW filter priorities,
the geometry constants (CUT_ARCSEC=10, OUT_PX=320, PIX=0.03125"/px, ZOOM_ARCSEC=3.5),
`split_filters`, `channel_of`, `product_urls`, `RemoteImage` (fsspec range reads +
linearised tangent-plane resampling), `sky`, `radial_profile_subtract`, `arc_score`,
`asinh_stretch`, the font/panel helpers and `render_cutout`. These are byte-for-byte the
run's code (only `from __future__ import annotations` and this docstring were added) so a
re-render of a top-100 system reproduces `J/top100_clean/<id>.jpg`; `golden/build_stamps.py
--check-against` measures that.

What was dropped: footprint helpers (`parse_s_region`, `poly_bbox`, `points_in_polys`),
`BASE`/`disk_free_gib`/`append_rows` (run bookkeeping) — nothing the golden set needs.

What was added (section marked ADDED, below the vendored block): the run's operational
constants that lived in the stage scripts (`MIN_FINITE` from 05_fetch_cutouts.py,
`JPEG_QUALITY=95`, `FOOTER_Y=540` from 22_clean_sets.py), `open_image` with polite retries,
`cutout_channels`/`fetch_channels` (SW+LW on the same grid, any scale), `gate_min_finite`
(the run's "drop a channel below 0.55 finite" rule, applied before rendering),
`render_v1_composite` (render_cutout + size assertion), `write_stamp_fits`/`read_stamp_fits`
(float32 PrimaryHDU with the contract's provenance keys + a minimal TAN WCS) and
`crop_footer` (the blind-kit crop).

Grid convention (from the run): row 0 = North, column 0 = East, centre at pixel
((n-1)/2, (n-1)/2), i.e. North-up East-left when displayed with origin='upper' (PIL,
matplotlib default). The FITS stamps keep that array untouched; their WCS carries
CDELT1 < 0 and CDELT2 < 0 so a WCS-aware viewer orients them correctly.
"""
from __future__ import annotations

# ============================================================================ VENDORED
# jwst-strong-lens-search/scripts/util.py @ 4f81493 — do not edit; fix upstream and re-vendor.
"""Shared helpers: footprints, product URLs, streaming cutouts, rendering.

Design notes
------------
* Cutouts are built directly on a north-up tangent-plane grid centred on the
  target, so SW and LW land on an identical grid and RGB composition is free.
* The sky->pixel mapping is linearised from the WCS around the target (a 10"
  box is far inside the regime where the tangent-plane Jacobian is exact to
  well under a pixel), which avoids 10^5 per-cutout WCS evaluations.
* Only the pixel bounding box actually needed is pulled through
  ``ImageHDU.section``, which issues true HTTP range reads.
"""
import io
import os
import shutil
import threading
import warnings

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from scipy.ndimage import map_coordinates

warnings.filterwarnings("ignore", module="astropy.wcs")

S3_ROOT = "https://stpubdata.s3.amazonaws.com/jwst/public"
MAST_ROOT = "https://mast.stsci.edu/api/v0.1/Download/file?uri="

# NIRCam short/long wavelength channels, best-first for lens searching.
SW_PRIORITY = ["F150W", "F200W", "F115W", "F090W", "F210M", "F182M",
               "F150W2", "F140M", "F162M", "F070W", "F187N", "F212N", "F164N"]
LW_PRIORITY = ["F277W", "F356W", "F444W", "F410M", "F335M", "F300M",
               "F360M", "F322W2", "F250M", "F430M", "F460M", "F480M",
               "F323N", "F405N", "F466N", "F470N"]
SW_SET, LW_SET = set(SW_PRIORITY), set(LW_PRIORITY)
# Coronagraphic masks occult the target; weak-lens pupil is defocused.
BAD_ELEMENTS = {"MASKRND", "MASKBAR", "MASKA210R", "MASKA335R", "MASKA430R",
                "MASKALWB", "MASKASWB", "WLP4", "WLM8", "WLP8"}

CUT_ARCSEC = 10.0
OUT_PX = 320
PIX = CUT_ARCSEC / OUT_PX  # 0.03125 "/px


def split_filters(s):
    """'F444W;F405N' -> ['F444W', 'F405N']; drops coronagraph/WLP elements."""
    if not isinstance(s, str):
        return []
    parts = [p.strip().upper() for p in s.replace(",", ";").split(";") if p.strip()]
    return [p for p in parts if p not in BAD_ELEMENTS and p != "CLEAR"]


def channel_of(filters):
    """Return ('SW'|'LW'|None, filter_name) for the best usable element."""
    parts = split_filters(filters)
    for f in SW_PRIORITY:
        if f in parts:
            return "SW", f
    for f in LW_PRIORITY:
        if f in parts:
            return "LW", f
    return None, None


def product_urls(obs_id):
    """(primary stpubdata URL, MAST range-read fallback URL) for an i2d file."""
    prog = obs_id.split("-")[0]
    group = obs_id.split("-")[1].split("_")[0]
    name = f"{obs_id}_i2d.fits"
    return (f"{S3_ROOT}/{prog}/L3/t/{group}/{name}",
            f"{MAST_ROOT}mast:JWST/product/{name}")


# ---------------------------------------------------------------- streaming

class RemoteImage:
    """An open remote i2d file with a cached SCI header/WCS.

    Opening costs seconds; each subsequent ``cutout`` is a sub-second range
    read, so callers should group work by file.
    """

    def __init__(self, urls, timeout=120):
        self.hdul = None
        self.err = None
        for url in [u for u in urls if u]:
            try:
                self.hdul = fits.open(url, use_fsspec=True,
                                      fsspec_kwargs={"block_size": 1 << 20})
                self.sci = self.hdul["SCI"]
                self.hdr = self.sci.header
                self.wcs = WCS(self.hdr)
                self.ny, self.nx = self.sci.shape
                self.url = url
                return
            except Exception as e:  # try the next transport
                self.err = f"{type(e).__name__}: {e}"
                try:
                    if self.hdul:
                        self.hdul.close()
                except Exception:
                    pass
                self.hdul = None
        if self.hdul is None:
            raise OSError(self.err or "no url")

    def close(self):
        try:
            if self.hdul:
                self.hdul.close()
        except Exception:
            pass
        self.hdul = None

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    def _jacobian(self, ra, dec):
        """Linearise sky->pixel at (ra, dec).

        Returns (x0, y0, dpix/dEast, dpix/dNorth) with offsets in arcsec.
        """
        cosd = max(np.cos(np.radians(dec)), 1e-6)
        d = 1.0 / 3600.0  # 1 arcsec probe
        pts = np.array([[ra, dec],
                        [ra + d / cosd, dec],   # +1" East
                        [ra, dec + d]])         # +1" North
        xy = self.wcs.wcs_world2pix(pts, 0)
        x0, y0 = xy[0]
        de = xy[1] - xy[0]
        dn = xy[2] - xy[0]
        return x0, y0, de, dn

    def cutout(self, ra, dec, out_px=OUT_PX, arcsec=CUT_ARCSEC):
        """North-up, East-left cutout on a fixed tangent-plane grid.

        Returns (image, finite_fraction) or (None, 0.0) if off-image.
        """
        x0, y0, de, dn = self._jacobian(ra, dec)
        scale = arcsec / out_px
        # row 0 = North, col 0 = East  (standard astronomical display)
        off = -(np.arange(out_px) - (out_px - 1) / 2.0) * scale
        east = off[None, :]
        north = off[:, None]
        xs = x0 + east * de[0] + north * dn[0]
        ys = y0 + east * de[1] + north * dn[1]

        ix0 = int(np.floor(np.nanmin(xs))) - 2
        ix1 = int(np.ceil(np.nanmax(xs))) + 3
        iy0 = int(np.floor(np.nanmin(ys))) - 2
        iy1 = int(np.ceil(np.nanmax(ys))) + 3
        if ix1 <= 0 or iy1 <= 0 or ix0 >= self.nx or iy0 >= self.ny:
            return None, 0.0
        cx0, cy0 = max(ix0, 0), max(iy0, 0)
        cx1, cy1 = min(ix1, self.nx), min(iy1, self.ny)
        if cx1 - cx0 < 3 or cy1 - cy0 < 3:
            return None, 0.0

        sub = np.asarray(self.sci.section[cy0:cy1, cx0:cx1], dtype=float)
        out = map_coordinates(np.nan_to_num(sub, nan=0.0),
                              [ys - cy0, xs - cx0], order=1,
                              mode="constant", cval=np.nan)
        valid = map_coordinates(np.isfinite(sub).astype(float),
                                [ys - cy0, xs - cx0], order=1,
                                mode="constant", cval=0.0)
        out = np.where(valid > 0.5, out, np.nan)
        # data-free padding in a mosaic reads as exactly 0.0
        good = np.isfinite(out) & (out != 0.0)
        return out, float(good.mean())


# ---------------------------------------------------------------- photometry

def sky(im):
    """Robust (median, sigma) via the MAD. From the prior COSMOS-Web search."""
    v = im[np.isfinite(im)]
    if v.size == 0:
        return 0.0, 1.0
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med)))
    return med, 1.4826 * mad + 1e-12


def radial_profile_subtract(im, cy=None, cx=None):
    """Subtract the median radial profile about the centre: removes a smooth
    deflector, leaving arcs. From the prior COSMOS-Web search."""
    n, m = im.shape
    cy = (n - 1) / 2.0 if cy is None else cy
    cx = (m - 1) / 2.0 if cx is None else cx
    yy, xx = np.ogrid[:n, :m]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).astype(int)
    rmax = int(r.max())
    prof = np.zeros(rmax + 1)
    flat_r, flat_v = r.ravel(), im.ravel()
    order = np.argsort(flat_r)
    fr, fv = flat_r[order], flat_v[order]
    edges = np.searchsorted(fr, np.arange(rmax + 2))
    for ri in range(rmax + 1):
        seg = fv[edges[ri]:edges[ri + 1]]
        seg = seg[np.isfinite(seg)]
        prof[ri] = np.median(seg) if seg.size else 0.0
    return im - prof[r]


def arc_score(im, pix=PIX, r_in=0.35, r_out=1.8, thresh=2.5):
    """Summed significance of residual flux in the arc annulus after removing
    the smooth deflector. Ranking tie-breaker only -- never a filter."""
    if im is None:
        return 0.0
    n = im.shape[0]
    c = (n - 1) / 2.0
    med, s = sky(im)
    res = radial_profile_subtract(im - med)
    yy, xx = np.ogrid[:n, :n]
    rr = np.sqrt((yy - c) ** 2 + (xx - c) ** 2) * pix
    ann = (rr >= r_in) & (rr <= r_out)
    sig = res / s
    m = ann & np.isfinite(sig) & (sig > thresh)
    return float(np.nansum(sig[m]))


# ---------------------------------------------------------------- rendering

def asinh_stretch(im, soft=2.0, cap=400.0, ref_sigma=None):
    """Sky-subtracted asinh scaling to [0,1]; the prior search's stretch."""
    if im is None:
        return None
    med, s = sky(im)
    s = ref_sigma if ref_sigma else s
    x = (im - med) / s
    y = np.arcsinh(np.clip(x, -2.0, cap) / soft) / np.arcsinh(cap / soft)
    return np.clip(np.nan_to_num(y, nan=0.0), 0.0, 1.0)


# ---------------------------------------------------------------- panels

_FONT_CACHE = {}
_FONT_PATHS = ["/System/Library/Fonts/Supplemental/Arial.ttf",
               "/System/Library/Fonts/Helvetica.ttc",
               "/System/Library/Fonts/Supplemental/Courier New.ttf"]


def _font(size):
    from PIL import ImageFont
    if size in _FONT_CACHE:
        return _FONT_CACHE[size]
    f = None
    for p in _FONT_PATHS:
        if os.path.exists(p):
            try:
                f = ImageFont.truetype(p, size)
                break
            except Exception:
                pass
    if f is None:
        f = ImageFont.load_default()
    _FONT_CACHE[size] = f
    return f


def _gray_panel(y, px):
    """[0,1] array -> uint8 RGB panel at px resolution."""
    from PIL import Image
    a = (np.clip(np.nan_to_num(y, nan=0.0), 0, 1) * 255).astype(np.uint8)
    img = Image.fromarray(a, mode="L").convert("RGB")
    return img.resize((px, px), Image.LANCZOS) if img.size[0] != px else img


def _rgb_panel(r, g, b, px):
    from PIL import Image
    a = (np.clip(np.dstack([np.nan_to_num(r, nan=0.0),
                            np.nan_to_num(g, nan=0.0),
                            np.nan_to_num(b, nan=0.0)]), 0, 1) * 255).astype(np.uint8)
    img = Image.fromarray(a, mode="RGB")
    return img.resize((px, px), Image.LANCZOS) if img.size[0] != px else img


def _zoom(im, arcsec_out, arcsec_in=CUT_ARCSEC):
    n = im.shape[0]
    k = max(int(round(n * arcsec_out / arcsec_in)), 8)
    c, h = n // 2, k // 2
    return im[c - h:c - h + k, c - h:c - h + k]


ZOOM_ARCSEC = 3.5


def render_cutout(sw, lw, meta, px=240, arcsec=CUT_ARCSEC):
    """Inspection image: full field for quadrant calls, zoom for the centre.

    Row 1 (10" field, quadrant assessment): normal | deep | colour
    Row 2 (3.5" zoom, centre assessment):   deep   | colour | deflector-subtracted

    North up, East left, so NE=top-left, NW=top-right, SE=bottom-left,
    SW=bottom-right. Labelled in-image so the convention is never inferred.
    Typical galaxy-scale lenses have theta_E ~ 0.5-2", which spans only a few
    percent of the 10" field; the zoom row is what makes them legible.
    """
    from PIL import Image, ImageDraw

    base = sw if sw is not None else lw
    if base is None:
        return None
    bname = meta.get("sw_filter") if sw is not None else meta.get("lw_filter")

    def color(a, b):
        # Normalise each band by its own noise so the sky stays neutral; a
        # genuine colour difference still shows as a flux-over-noise ratio.
        # SW is natively finer than LW, so its per-pixel speckle would tint the
        # sky blue -- a matched smoothing suppresses that without touching the
        # resolved structure that matters.
        from scipy.ndimage import gaussian_filter
        sm = lambda z: gaussian_filter(np.nan_to_num(z, nan=0.0), 1.1)
        rr = asinh_stretch(sm(a), soft=1.2, cap=120.0)
        bb = asinh_stretch(sm(b), soft=1.2, cap=120.0)
        return rr, (rr + bb) / 2.0, bb

    row1 = [(_gray_panel(asinh_stretch(base, soft=2.0, cap=400.0), px), f"{bname} normal 10\""),
            (_gray_panel(asinh_stretch(base, soft=0.7, cap=30.0), px), f"{bname} deep 10\"")]
    if lw is not None and sw is not None:
        row1.append((_rgb_panel(*color(lw, sw), px),
                     f"{meta.get('lw_filter')}/{meta.get('sw_filter')} colour 10\""))
    else:
        med, _ = sky(base)
        row1.append((_gray_panel(asinh_stretch(radial_profile_subtract(base - med),
                                               soft=1.0, cap=40.0), px),
                     "deflector subtracted 10\""))

    zb = _zoom(base, ZOOM_ARCSEC)
    med, _ = sky(zb)
    row2 = [(_gray_panel(asinh_stretch(zb, soft=0.7, cap=30.0), px), f'{bname} deep {ZOOM_ARCSEC}"')]
    if lw is not None and sw is not None:
        row2.append((_rgb_panel(*color(_zoom(lw, ZOOM_ARCSEC), _zoom(sw, ZOOM_ARCSEC)), px),
                     f'colour {ZOOM_ARCSEC}"'))
    else:
        row2.append((_gray_panel(asinh_stretch(zb, soft=2.0, cap=400.0), px),
                     f'{bname} normal {ZOOM_ARCSEC}"'))
    row2.append((_gray_panel(asinh_stretch(radial_profile_subtract(zb - med),
                                           soft=0.9, cap=35.0), px),
                 f'deflector subtracted {ZOOM_ARCSEC}"'))

    gap, th, foot_h = 8, 18, 22
    ncol = 3
    W = ncol * px + (ncol + 1) * gap
    H = 2 * (px + th) + 3 * gap + foot_h
    canvas = Image.new("RGB", (W, H), (12, 12, 16))
    d = ImageDraw.Draw(canvas)
    f9, f11 = _font(11), _font(12)

    for r, row in enumerate((row1, row2)):
        y = gap + r * (px + th + gap) + th
        for cidx, (p, t) in enumerate(row):
            x = gap + cidx * (px + gap)
            canvas.paste(p, (x, y))
            d.text((x + 1, y - th + 3), t, font=f9, fill=(205, 205, 215))
            d.rectangle([x, y, x + px - 1, y + px - 1], outline=(70, 70, 80))
            if r == 0:      # quadrant labels only on the full-field row
                m = 4
                for lbl, pos in (("NE", (x + m, y + m)),
                                 ("NW", (x + px - m - 17, y + m)),
                                 ("SE", (x + m, y + px - m - 12)),
                                 ("SW", (x + px - m - 17, y + px - m - 12))):
                    d.text(pos, lbl, font=f9, fill=(120, 190, 255))
            fov = arcsec if r == 0 else ZOOM_ARCSEC
            cxp, cyp = x + px / 2.0, y + px / 2.0
            off = px * (0.9 if r else 1.2) / fov
            ln = px * (0.5 if r else 0.6) / fov
            for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                d.line([cxp + dx * off, cyp + dy * off,
                        cxp + dx * (off + ln), cyp + dy * (off + ln)],
                       fill=(255, 215, 90), width=1)
            if cidx == 0:   # 1" scale bar per row (the two rows differ in scale)
                bar = px / fov
                d.line([x + 6, y + px - 8, x + 6 + bar, y + px - 8],
                       fill=(255, 255, 255), width=2)
                d.text((x + 8 + bar, y + px - 17), '1"', font=f9, fill=(255, 255, 255))
    # compass on the full-field row
    cx2, cy2 = gap + px - 24, gap + th + 28
    d.line([cx2, cy2, cx2, cy2 - 15], fill=(120, 255, 160), width=2)
    d.text((cx2 - 3, cy2 - 29), "N", font=f9, fill=(120, 255, 160))
    d.line([cx2, cy2, cx2 - 15, cy2], fill=(120, 255, 160), width=2)
    d.text((cx2 - 28, cy2 - 6), "E", font=f9, fill=(120, 255, 160))

    mr = meta.get("mag_r", float("nan"))
    foot = (f"{meta.get('id','')}   RA {meta.get('ra',0):.5f}  Dec {meta.get('dec',0):+.5f}"
            f"   r={mr:.2f}  {meta.get('type','')}   prog {meta.get('proposal','')}")
    d.text((gap, H - foot_h + 4), foot, font=f11, fill=(180, 180, 195))
    return canvas
# ======================================================================== END VENDORED


# ============================================================================ ADDED
# LensJudge-side glue. Nothing above this line may change without re-vendoring.
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Operational constants the run kept in its stage scripts, not in util.py.
MIN_FINITE = 0.55     # 05_fetch_cutouts.py: a channel below this finite fraction is dropped
JPEG_QUALITY = 95     # 05_fetch_cutouts.py: img.save(png, quality=95) -> PIL defaults otherwise
FOOTER_Y = 540        # 22_clean_sets.py: the footer strip sits below the second panel row
COMPOSITE_SIZE = (752, 562)   # render_cutout(px=240): W = 3*240+4*8, H = 2*(240+18)+3*8+22
RENDER_VERSION = "jwst_v1"    # the kit/registry tag for this composite (contract)
CHANNELS = ("SW", "LW")
CREATOR = "lensjudge.common.jwst_fetch (util.py@4f81493)"


def open_image(obs_id, retries: int = 3, backoff: float = 3.0) -> Optional[RemoteImage]:
    """Open one remote i2d mosaic by observation id, or None if the id is blank.

    RemoteImage already falls back S3 -> MAST per attempt; this adds polite retries with
    growing sleeps for transient HTTP/SSL failures. Raises after the last attempt so the
    caller can record the failure (the run silently returned None; we want the reason)."""
    if not isinstance(obs_id, str) or not obs_id:
        return None
    last = None
    for attempt in range(retries):
        try:
            return RemoteImage(product_urls(obs_id))
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(backoff * (attempt + 1))
    raise OSError(f"open_image({obs_id}) failed after {retries} attempts: {last}")


def _cutout_retry(img: RemoteImage, ra, dec, out_px, arcsec, retries=3, backoff=2.0):
    last = None
    for attempt in range(retries):
        try:
            return img.cutout(ra, dec, out_px=out_px, arcsec=arcsec)
        except Exception as e:  # noqa: BLE001   (range read hiccups are transient)
            last = e
            time.sleep(backoff * (attempt + 1))
    raise OSError(f"cutout({ra:.5f},{dec:+.5f}) failed after {retries} attempts: {last}")


def cutout_channels(images: dict, ra: float, dec: float,
                    arcsec: float = CUT_ARCSEC, out_px: int = OUT_PX) -> dict:
    """SW+LW cutouts at one position from already-open mosaics.

    `images` maps channel -> RemoteImage or None. Returns {channel: (array|None, finite)}
    for every channel in CHANNELS (None when the mosaic is missing or the target is
    off-image). Same grid for both channels, exactly as the run built them."""
    out = {}
    for ch in CHANNELS:
        img = images.get(ch)
        if img is None:
            out[ch] = (None, 0.0)
            continue
        out[ch] = _cutout_retry(img, ra, dec, out_px, arcsec)
    return out


def fetch_channels(ra: float, dec: float, sw_obs, lw_obs,
                   arcsec: float = CUT_ARCSEC, out_px: int = OUT_PX) -> dict:
    """One-shot convenience: open both mosaics, cut, close.

    Returns {"SW": (arr, finite), "LW": (arr, finite)}. For many targets on the same
    mosaics use open_image + cutout_channels (opening costs seconds, cutting does not)."""
    images = {"SW": None, "LW": None}
    try:
        images["SW"] = open_image(sw_obs)
        images["LW"] = open_image(lw_obs)
        return cutout_channels(images, ra, dec, arcsec=arcsec, out_px=out_px)
    finally:
        for im in images.values():
            if im is not None:
                im.close()


def gate_min_finite(channels: dict, min_finite: float = MIN_FINITE) -> dict:
    """The run's render gate (05_fetch_cutouts.py): a channel with finite fraction below
    MIN_FINITE is treated as absent, which is what flips a system to the gray layout."""
    return {ch: (None if (arr is None or ff < min_finite) else arr)
            for ch, (arr, ff) in channels.items()}


def render_v1_composite(sw, lw, meta: dict):
    """The v1 composite the discovery agents saw: render_cutout at the run's defaults.

    `sw`/`lw` are the gated 10" arrays (None => that channel absent); `meta` carries
    id, ra, dec, mag_r, type, proposal, sw_filter, lw_filter (targets.parquet fields —
    the footer prints them, so byte-faithfulness needs the run's own values). Returns a
    752x562 RGB PIL image, or None when both channels are absent."""
    img = render_cutout(sw, lw, meta)
    if img is not None and img.size != COMPOSITE_SIZE:
        raise RuntimeError(f"composite is {img.size}, expected {COMPOSITE_SIZE}")
    return img


def save_composite(img, path) -> Path:
    """Write the composite exactly as the run did (JPEG, quality 95, PIL defaults)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(path), quality=JPEG_QUALITY)
    return path


def crop_footer(img, y: int = FOOTER_Y):
    """Drop the footer strip (id, RA/Dec, r mag, programme) so the identity cannot be read
    off the image — 22_clean_sets.py's crop for the scrambled/blind set. 752x562 -> 752x540."""
    return img.crop((0, 0, img.width, min(y, img.height)))


def stamp_header(candidate_id: str, ra: float, dec: float, channel: str, filt, obs_id,
                 url, arcsec: float, out_px: int, finite: float, fetched_at=None) -> dict:
    """The contract's provenance keys for one stamp (all FITS-legal: <=8 chars)."""
    return {
        "OBJECT": str(candidate_id),
        "RA_DEG": float(ra),
        "DEC_DEG": float(dec),
        "FILTER": "" if filt is None or filt != filt else str(filt),
        "CHANNEL": str(channel),
        "OBS_ID": "" if obs_id is None or obs_id != obs_id else str(obs_id),
        "SRC_URL": str(url or ""),
        "PIXSCALE": float(arcsec) / int(out_px),          # arcsec/px (0.03125 at 10"/320)
        "CUTARC": float(arcsec),
        "FINITE": float(finite),
        "ORIENT": "row0=N col0=E (N-up E-left, origin upper)",
        "CREATOR": CREATOR,
        "FETCHED": fetched_at or datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def write_stamp_fits(path, arr, header_dict: dict) -> Path:
    """float32 PrimaryHDU + header_dict as cards (+ a TAN WCS when RA_DEG/DEC_DEG/PIXSCALE
    are present). The array is written as-is (row 0 = North); the WCS's negative CDELT2
    is what tells a FITS viewer the rows run southward."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.asarray(arr, dtype=np.float32)
    hdu = fits.PrimaryHDU(data)
    h = hdu.header
    for k, v in header_dict.items():
        key = str(k).upper()
        if len(key) > 8:
            raise ValueError(f"FITS key too long: {key}")
        h[key] = v
    if all(k in header_dict for k in ("RA_DEG", "DEC_DEG", "PIXSCALE")):
        ny, nx = data.shape
        pix_deg = float(header_dict["PIXSCALE"]) / 3600.0
        h["WCSAXES"] = 2
        h["CTYPE1"], h["CTYPE2"] = "RA---TAN", "DEC--TAN"
        h["CUNIT1"] = h["CUNIT2"] = "deg"
        h["CRVAL1"], h["CRVAL2"] = float(header_dict["RA_DEG"]), float(header_dict["DEC_DEG"])
        h["CRPIX1"], h["CRPIX2"] = (nx + 1) / 2.0, (ny + 1) / 2.0   # 1-based centre
        h["CDELT1"], h["CDELT2"] = -pix_deg, -pix_deg                # col->West, row->South
        h["RADESYS"], h["EQUINOX"] = "ICRS", 2000.0
    hdu.writeto(str(path), overwrite=True)
    return path


def read_stamp_fits(path):
    """(float32 array, header) for a stamp written by write_stamp_fits."""
    with fits.open(str(path)) as hdul:
        return np.asarray(hdul[0].data, dtype=np.float32), hdul[0].header.copy()
# ======================================================================== END ADDED
