"""Mask PNG export: ``<id>.mask.png`` = uint8 image, 255 inside the union of the accepted|edited mask
circles, 0 elsewhere. Rendered at the survey's NATIVE pixel scale when ``image.native_pixel_scale_arcsec``
is set (size = cutout / native, ``r_px = r_arcsec / native``), else at display scale. Pure PIL + numpy;
independent of ``lensmark.render``."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from ..model import LensMarkFile, MaskCircle
from ..store import Campaign, atomic_write_bytes
from . import export_dir, exportable_items, load_files


def mask_size(file: LensMarkFile) -> tuple[int, int, float]:
    """(width_px, height_px, arcsec_per_px) of the mask raster."""
    native = file.image.native_pixel_scale_arcsec
    if native:
        W = max(1, int(round(file.image.cutout_arcsec / native)))
        H = max(1, int(round(file.image.cutout_arcsec * file.image.height / file.image.width / native)))
        return W, H, float(native)
    return file.image.width, file.image.height, file.image.pixel_scale_arcsec


def mask_array(file: LensMarkFile) -> np.ndarray:
    W, H, scale = mask_size(file)
    out = np.zeros((H, W), dtype=bool)
    ys, xs = np.mgrid[0:H, 0:W]
    xs = xs + 0.5           # pixel centres
    ys = ys + 0.5
    for it in exportable_items(file):
        if not isinstance(it, MaskCircle):
            continue
        cx, cy, r = it.center[0] * W, it.center[1] * H, it.radius_arcsec / scale
        out |= (xs - cx) ** 2 + (ys - cy) ** 2 <= r * r
    return out


def mask_image(file: LensMarkFile) -> Image.Image:
    return Image.fromarray((mask_array(file) * 255).astype(np.uint8), mode="L")


def export_masks(campaign: Campaign, *, out: str | Path | None = None, ids: Optional[list[str]] = None) -> list[Path]:
    import io
    paths: list[Path] = []
    for f in load_files(campaign, ids):
        if not any(isinstance(it, MaskCircle) for it in exportable_items(f)):
            continue
        buf = io.BytesIO()
        mask_image(f).save(buf, format="PNG", optimize=False)
        path = export_dir(campaign, "masks", out) / f"{f.id}.mask.png"
        atomic_write_bytes(path, buf.getvalue())
        paths.append(path)
    return paths
