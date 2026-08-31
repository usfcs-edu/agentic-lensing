"""Exports of ACCEPTED annotations: few-shot bundles, COCO, DS9 regions, mask PNGs.

Only items with status ``accepted`` or ``edited`` are exported - never ``proposed`` (unreviewed),
``rejected`` or ``invalid``. Each format lives in its own module; this package holds the shared
selection helpers and the ``run_export`` / ``cli_export`` dispatch used by the CLI and the server.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

from ..model import Arrow, EinsteinRing, ItemBase, LensMarkFile
from ..store import Campaign

EXPORT_STATUSES: tuple[str, ...] = ("accepted", "edited")
FORMATS: tuple[str, ...] = ("coco", "ds9", "masks", "fewshot")


def exportable_items(file: LensMarkFile) -> list[ItemBase]:
    return [it for it in file.items if it.status in EXPORT_STATUSES]


def load_files(campaign: Campaign, ids: Optional[Iterable[str]] = None) -> list[LensMarkFile]:
    """The saved LensMark files for ``ids`` (default: every image), in id order; images without a JSON are skipped."""
    wanted = list(ids) if ids is not None else campaign.list_ids()
    out: list[LensMarkFile] = []
    for image_id in wanted:
        f = campaign.load(image_id)
        if f is not None:
            out.append(f)
    return out


def ring_center(file: LensMarkFile, ring: EinsteinRing) -> list[float]:
    """The ring centre, following ``center_ref`` to the deflector arrow's head when it resolves (CONTRACT.md)."""
    if ring.center_ref:
        ref = file.item(ring.center_ref)
        if isinstance(ref, Arrow):
            return list(ref.head)
    return list(ring.center)


def export_dir(campaign: Campaign, fmt: str, out: str | Path | None = None) -> Path:
    d = Path(out).expanduser() if out else campaign.exports_dir / fmt
    d.mkdir(parents=True, exist_ok=True)
    return d


def run_export(campaign: Campaign, fmt: str, *, out: str | Path | None = None, k: int = 6,
               require_flag: bool = False, ids: Optional[list[str]] = None) -> list[Path]:
    """Dispatch one export format; returns the written paths."""
    if fmt == "coco":
        from .coco import export_coco
        return export_coco(campaign, out=out, ids=ids)
    if fmt == "ds9":
        from .ds9 import export_ds9
        return export_ds9(campaign, out=out, ids=ids)
    if fmt == "masks":
        from .masks import export_masks
        return export_masks(campaign, out=out, ids=ids)
    if fmt == "fewshot":
        from .fewshot import export_fewshot
        return export_fewshot(campaign, out=out, k=k, require_flag=require_flag, ids=ids)
    raise ValueError(f"unknown export format {fmt!r}; choose from {FORMATS}")


def cli_export(dir: str, fmt: str, *, out: Optional[str] = None, k: int = 6, require_flag: bool = False,
               ids: Optional[list[str]] = None) -> int:
    paths = run_export(Campaign(dir), fmt, out=out, k=k, require_flag=require_flag, ids=ids)
    for p in paths:
        print(p)
    if not paths:
        print(f"nothing to export as {fmt} (no accepted/edited items" + (", or no image passes the fewshot filter)" if fmt == "fewshot" else ")"))
        return 1
    print(f"wrote {len(paths)} file(s)")
    return 0
