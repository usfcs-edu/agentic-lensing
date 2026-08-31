"""COCO export (``exports/coco/instances.json``): accepted arrows as 2-keypoint skeletons, mask circles
as 64-gon segmentations, Einstein rings and text notes as boxed annotations, all in display pixels."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Optional

from ..model import Arrow, EinsteinRing, LensMarkFile, MaskCircle, TextNote, now_iso
from ..store import Campaign, atomic_write_text
from . import export_dir, exportable_items, load_files, ring_center

CATEGORIES: list[dict[str, Any]] = [
    {"id": 1, "name": "arrow", "supercategory": "pointer", "keypoints": ["tail", "head"], "skeleton": [[1, 2]]},
    {"id": 2, "name": "field_galaxy_mask", "supercategory": "mask"},
    {"id": 3, "name": "star_mask", "supercategory": "mask"},
    {"id": 4, "name": "artifact_mask", "supercategory": "mask"},
    {"id": 5, "name": "einstein_ring", "supercategory": "lens"},
    {"id": 6, "name": "text", "supercategory": "note"},
]
CATEGORY_ID = {c["name"]: c["id"] for c in CATEGORIES}
MASK_CATEGORY = {"galaxy": "field_galaxy_mask", "star": "star_mask", "artifact": "artifact_mask"}
POLYGON_N = 64


def _r(x: float) -> float:
    return round(float(x), 2)


def _clip_box(x0: float, y0: float, x1: float, y1: float, W: int, H: int) -> list[float]:
    """[x, y, w, h] clipped to the image (COCO boxes must lie inside the image)."""
    x0c, y0c = min(max(x0, 0.0), W), min(max(y0, 0.0), H)
    x1c, y1c = min(max(x1, 0.0), W), min(max(y1, 0.0), H)
    return [_r(x0c), _r(y0c), _r(max(x1c - x0c, 0.0)), _r(max(y1c - y0c, 0.0))]


def _polygon(cx: float, cy: float, r: float, W: int, H: int, n: int = POLYGON_N) -> list[float]:
    pts: list[float] = []
    for k in range(n):
        a = 2 * math.pi * k / n
        pts.append(_r(min(max(cx + r * math.cos(a), 0.0), W)))
        pts.append(_r(min(max(cy + r * math.sin(a), 0.0), H)))
    return pts


def _common_attrs(it) -> dict[str, Any]:
    return {"lensmark_item_id": it.id, "color": it.color, "label": it.label, "status": it.status,
            "created_by": it.created_by.model_dump(mode="json", exclude_none=True)}


def to_coco(files: list[LensMarkFile]) -> dict[str, Any]:
    """Build the COCO document for the accepted|edited items of ``files`` (image ids 1..N in order)."""
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    ann_id = 0
    for img_id, f in enumerate(files, start=1):
        W, H = f.image.width, f.image.height
        ps = f.image.pixel_scale_arcsec
        images.append({
            "id": img_id, "file_name": f.image.file, "width": W, "height": H, "lensmark_id": f.id,
            "attributes": {"cutout_arcsec": f.image.cutout_arcsec, "pixel_scale_arcsec": ps,
                           "grade": f.system.grade, "verdict": f.system.verdict,
                           "theta_e_arcsec": f.system.theta_e.value_arcsec, "description": f.system.description,
                           "rank": f.system.rank, "sha256": f.image.sha256},
        })
        for it in exportable_items(f):
            ann_id += 1
            ann: dict[str, Any] = {"id": ann_id, "image_id": img_id, "iscrowd": 0, "attributes": _common_attrs(it)}
            if isinstance(it, Arrow):
                tx, ty = it.tail[0] * W, it.tail[1] * H
                hx, hy = it.head[0] * W, it.head[1] * H
                box = _clip_box(min(tx, hx), min(ty, hy), max(tx, hx), max(ty, hy), W, H)
                ann.update({"category_id": CATEGORY_ID["arrow"],
                            "keypoints": [_r(tx), _r(ty), 2, _r(hx), _r(hy), 2], "num_keypoints": 2,
                            "bbox": box, "area": _r(box[2] * box[3])})
                ann["attributes"].update({"tail_uv": list(it.tail), "head_uv": list(it.head), "label_anchor": it.label_anchor})
            elif isinstance(it, MaskCircle):
                cx, cy, r = it.center[0] * W, it.center[1] * H, it.radius_arcsec / ps
                ann.update({"category_id": CATEGORY_ID[MASK_CATEGORY[it.kind]],
                            "bbox": _clip_box(cx - r, cy - r, cx + r, cy + r, W, H),
                            "segmentation": [_polygon(cx, cy, r, W, H)], "area": _r(math.pi * r * r)})
                ann["attributes"].update({"radius_arcsec": it.radius_arcsec, "kind": it.kind,
                                          "center_uv": list(it.center), "center_px": [_r(cx), _r(cy)], "radius_px": _r(r)})
            elif isinstance(it, EinsteinRing):
                c = ring_center(f, it)
                cx, cy, r = c[0] * W, c[1] * H, it.theta_e_arcsec / ps
                ann.update({"category_id": CATEGORY_ID["einstein_ring"],
                            "bbox": _clip_box(cx - r, cy - r, cx + r, cy + r, W, H),
                            "segmentation": [_polygon(cx, cy, r, W, H)], "area": _r(math.pi * r * r)})
                ann["attributes"].update({"theta_e_arcsec": it.theta_e_arcsec, "center_uv": c, "center_ref": it.center_ref,
                                          "center_px": [_r(cx), _r(cy)], "radius_px": _r(r)})
            elif isinstance(it, TextNote):
                x, y = it.pos[0] * W, it.pos[1] * H
                box = _clip_box(x, y, x + 1, y + 1, W, H)
                ann.update({"category_id": CATEGORY_ID["text"], "bbox": box, "area": _r(box[2] * box[3])})
                ann["attributes"].update({"text": it.text, "pos_uv": list(it.pos)})
            else:  # pragma: no cover - the union is closed
                continue
            annotations.append(ann)
    return {
        "info": {"description": "LensMark accepted annotations", "version": "lensmark/1.0", "date_created": now_iso(),
                 "coordinates": "display pixels of file_name (origin top-left); sizes in attributes are arcsec"},
        "licenses": [],
        "images": images, "categories": CATEGORIES, "annotations": annotations,
    }


def export_coco(campaign: Campaign, *, out: str | Path | None = None, ids: Optional[list[str]] = None) -> list[Path]:
    files = [f for f in load_files(campaign, ids) if exportable_items(f)]
    if not files:
        return []
    doc = to_coco(files)
    path = export_dir(campaign, "coco", out) / "instances.json"
    atomic_write_text(path, json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
    return [path]
