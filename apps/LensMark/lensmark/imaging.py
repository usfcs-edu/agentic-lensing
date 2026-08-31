"""Image helpers shared by the propose pipeline, voice patch and the server (thumbnails)."""
from __future__ import annotations

import base64
import io
from pathlib import Path

from PIL import Image, ImageDraw

from . import config


def load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as im:
        return im.convert("RGB")


def upsample(im: Image.Image, min_px: int = 400) -> Image.Image:
    """Upsample small cutouts so the model sees legible pixels (like lensjudge RENDER_PX=400)."""
    if min(im.size) >= min_px:
        return im
    f = -(-min_px // min(im.size))          # ceil division -> integer factor
    return im.resize((im.width * f, im.height * f), Image.NEAREST)


def png_bytes(im: Image.Image) -> bytes:
    buf = io.BytesIO()
    im.save(buf, format="PNG", optimize=False)
    return buf.getvalue()


def png_b64(im: Image.Image) -> str:
    return base64.b64encode(png_bytes(im)).decode("ascii")


def image_block(im: Image.Image) -> dict:
    """Anthropic Messages-API image content block (base64 PNG)."""
    return {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": png_b64(im)}}


def grid_overlay(im: Image.Image, step: float = 0.1) -> Image.Image:
    """Copy of the image with a labelled u/v grid (fractions of width/height) so the model can name
    normalized coordinates. Lines every ``step``; labels along the top (u) and left (v) edges."""
    out = im.convert("RGB").copy()
    d = ImageDraw.Draw(out, "RGBA")
    W, H = out.size
    try:
        from PIL import ImageFont
        font = ImageFont.truetype(str(config.FONT_DIR / "DejaVuSans.ttf"), max(10, W // 40))
    except Exception:  # pragma: no cover
        font = None
    n = int(round(1 / step))
    for k in range(n + 1):
        x = round(k * step * W)
        y = round(k * step * H)
        d.line([(x, 0), (x, H)], fill=(255, 255, 0, 110), width=1)
        d.line([(0, y), (W, y)], fill=(255, 255, 0, 110), width=1)
        lab = f"{k * step:.1f}"
        if 0 < k < n:
            d.text((x + 2, 1), lab, fill=(255, 255, 0, 230), font=font)
            d.text((1, y + 1), lab, fill=(255, 255, 0, 230), font=font)
    return out


def thumbnail(path: Path, px: int = 160) -> bytes:
    im = load_rgb(path)
    im.thumbnail((px, px))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=85)
    return buf.getvalue()
