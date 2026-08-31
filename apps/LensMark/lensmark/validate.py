"""Turn a model-emitted ``Proposal`` (or a voice-patch ``add`` item) into strict ``Item``s.

Lenient in the spirit of reproductions/lensjudge/common/schemas.py: clamp / default / snap rather
than discard, and record every repair; only geometry that cannot be drawn becomes ``status:"invalid"``
(kept in the file so the human sees the failure). The repair and invalid counters are dataset columns.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

from . import config
from .model import (Arrow, COLOR_NAMES, CreatedBy, EinsteinRing, ItemBase, LensMarkFile, MaskCircle,
                    Proposal, ProposalItem, TextNote)

CLAMP_SLACK = 0.05           # coordinates within [-0.05, 1.05] are clamped; further out -> invalid
DEGENERATE_ARROW = 0.01      # |tail - head| below this (uv units) -> invalid
MAX_LABEL = 40
_NAMED_RGB = {
    "red": (255, 0, 0), "blue": (0, 0, 255), "purple": (160, 32, 240), "pink": (255, 105, 180),
    "lime": (0, 255, 0), "gold": (255, 215, 0), "grey": (191, 191, 191), "black": (0, 0, 0),
    "teal": (0, 128, 128), "violet": (238, 130, 238), "salmon": (250, 128, 114),
}


@dataclass
class ValidationResult:
    items: list[ItemBase] = field(default_factory=list)
    repairs: list[dict[str, Any]] = field(default_factory=list)      # {item_id, field, from, to, why}
    invalid: list[dict[str, Any]] = field(default_factory=list)      # {item_id, reason}

    @property
    def n_repaired(self) -> int:
        return len({r["item_id"] for r in self.repairs})

    @property
    def n_invalid(self) -> int:
        return len(self.invalid)


# ----------------------------------------------------------------------------- colours
def _hex_to_rgb(h: str) -> Optional[tuple[int, int, int]]:
    h = h.strip().lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    if len(h) != 6:
        return None
    try:
        return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    except ValueError:
        return None


def nearest_palette(name_or_hex: str, *, allowed: tuple[str, ...]) -> str:
    """Snap any colour word / hex to the nearest allowed palette name (RGB distance)."""
    key = name_or_hex.strip().lower()
    if key == "grey":
        key = "gray"
    if key in allowed:
        return key
    rgb = _hex_to_rgb(key) or _NAMED_RGB.get(key)
    if rgb is None:
        return allowed[0]
    best, best_d = allowed[0], float("inf")
    for n in allowed:
        pr = config.palette_rgb(n)
        d = sum((a - b) ** 2 for a, b in zip(rgb, pr))
        if d < best_d:
            best, best_d = n, d
    return best


def _is_deflector(label: Optional[str]) -> bool:
    return bool(label) and "deflector" in label.lower()


# ----------------------------------------------------------------------------- geometry helpers
def _clamp_uv(p: list[float], item_id: str, fld: str, res: ValidationResult) -> tuple[Optional[list[float]], Optional[str]]:
    if not isinstance(p, (list, tuple)) or len(p) != 2:
        return None, "missing_or_malformed_" + fld
    try:
        u, v = float(p[0]), float(p[1])
    except (TypeError, ValueError):
        return None, "missing_or_malformed_" + fld
    if not (math.isfinite(u) and math.isfinite(v)):
        return None, "non_finite_" + fld
    if u < -CLAMP_SLACK or u > 1 + CLAMP_SLACK or v < -CLAMP_SLACK or v > 1 + CLAMP_SLACK:
        return [u, v], "out_of_bounds"
    cu, cv = min(max(u, 0.0), 1.0), min(max(v, 0.0), 1.0)
    if (cu, cv) != (u, v):
        res.repairs.append({"item_id": item_id, "field": fld, "from": [u, v], "to": [cu, cv], "why": "clamped_to_image"})
    return [cu, cv], None


def _synth_tail(head: list[float], length: float) -> list[float]:
    """Tail placed 'outside' the feature: away from the image centre (21_annotate.py approach_angle)."""
    dx, dy = head[0] - 0.5, head[1] - 0.5
    n = math.hypot(dx, dy)
    if n < 1e-6:
        dx, dy = -0.7071, 0.7071            # degenerate: come in from lower-left
    else:
        dx, dy = dx / n, dy / n
    t = [head[0] + dx * length, head[1] + dy * length]
    # keep inside the image; if that pushes the tail onto the head, flip direction
    t = [min(max(t[0], 0.02), 0.98), min(max(t[1], 0.02), 0.98)]
    if math.hypot(t[0] - head[0], t[1] - head[1]) < length / 2:
        t = [head[0] - dx * length, head[1] - dy * length]
        t = [min(max(t[0], 0.02), 0.98), min(max(t[1], 0.02), 0.98)]
    return t


# ----------------------------------------------------------------------------- items
def item_from_proposal(d: ProposalItem | dict[str, Any], file: LensMarkFile, created_by: CreatedBy, *,
                       used_ids: set[str], res: ValidationResult, status: str = "proposed",
                       arrow_colors_used: Optional[set[str]] = None) -> Optional[ItemBase]:
    """Convert one proposed item. Returns None only when the type is unknown (recorded in ``res.invalid``)."""
    p = d if isinstance(d, ProposalItem) else ProposalItem.model_validate(d)
    t = (p.type or "").strip()
    if t not in ("arrow", "mask_circle", "einstein_ring", "text"):
        res.invalid.append({"item_id": None, "reason": f"unknown_type:{t}"})
        return None
    item_id = _mint(file, t, used_ids)
    label = (p.label or "").strip()[:MAX_LABEL] or None
    style_len = float(file.style_defaults.get("arrow", {}).get("default_len", 0.14))
    half_cut = file.image.cutout_arcsec / 2
    invalid_reason: Optional[str] = None
    common = dict(id=item_id, created_by=created_by, status=status)
    item: ItemBase

    if t == "arrow":
        head, err = _clamp_uv(p.head, item_id, "head", res)
        if head is None:
            res.invalid.append({"item_id": item_id, "reason": err})
            return None
        invalid_reason = invalid_reason or err
        tail: Optional[list[float]] = None
        if p.tail is not None:
            tail, err2 = _clamp_uv(p.tail, item_id, "tail", res)
            invalid_reason = invalid_reason or err2
        if tail is None:
            tail = _synth_tail(head, style_len)
            res.repairs.append({"item_id": item_id, "field": "tail", "from": None, "to": tail, "why": "synthesized_tail"})
        if math.hypot(tail[0] - head[0], tail[1] - head[1]) < DEGENERATE_ARROW:
            invalid_reason = invalid_reason or "degenerate_arrow"
        color = _arrow_color(p.color, label, item_id, res, arrow_colors_used)
        anchor = p.label_anchor if p.label_anchor in ("tail", "head", "auto") else "auto"
        item = Arrow(tail=tail, head=head, color=color, label=label, label_anchor=anchor, **common)

    elif t == "mask_circle":
        center, err = _clamp_uv(p.center, item_id, "center", res)
        if center is None:
            res.invalid.append({"item_id": item_id, "reason": err})
            return None
        invalid_reason = invalid_reason or err
        kind = p.kind if p.kind in ("galaxy", "star", "artifact") else None
        if kind is None:
            kind = "galaxy"
            res.repairs.append({"item_id": item_id, "field": "kind", "from": p.kind, "to": kind, "why": "default_kind"})
        r = p.radius_arcsec
        if r is None or not math.isfinite(r) or r <= 0:
            invalid_reason = invalid_reason or "bad_radius"
            r = max(0.3, 0.02 * file.image.cutout_arcsec)
        elif r > half_cut:
            invalid_reason = invalid_reason or "bad_radius"
        color = "mask_red"
        if p.color and p.color != "mask_red":
            res.repairs.append({"item_id": item_id, "field": "color", "from": p.color, "to": color, "why": "masks_are_mask_red"})
        item = MaskCircle(center=center, radius_arcsec=float(r), kind=kind, color=color, label=label, **common)

    elif t == "einstein_ring":
        center: Optional[list[float]] = None
        if p.center is not None:
            center, err = _clamp_uv(p.center, item_id, "center", res)
            invalid_reason = invalid_reason or err
        if center is None:
            center = _deflector_center(file) or [0.5, 0.5]
            res.repairs.append({"item_id": item_id, "field": "center", "from": None, "to": center, "why": "default_center"})
        te = p.theta_e_arcsec
        if te is None or not math.isfinite(te) or te <= 0:
            invalid_reason = invalid_reason or "theta_e_missing"
            te = max(0.5, 0.1 * file.image.cutout_arcsec)
        elif te > half_cut:
            invalid_reason = invalid_reason or "theta_e_implausible"
        item = EinsteinRing(center=center, theta_e_arcsec=float(te), color="ring_white", label=label, **common)

    else:  # text
        pos, err = _clamp_uv(p.pos, item_id, "pos", res)
        if pos is None:
            res.invalid.append({"item_id": item_id, "reason": err})
            return None
        invalid_reason = invalid_reason or err
        text = (p.text or label or "").strip()
        if not text:
            invalid_reason = invalid_reason or "empty_text"
            text = "(empty)"
        color = nearest_palette(p.color or "white", allowed=tuple(c for c in COLOR_NAMES if c not in config.RESERVED_COLORS))
        item = TextNote(pos=pos, text=text[:120], color=color, label=None, **common)

    if invalid_reason:
        item.status = "invalid"
        item.invalid_reason = invalid_reason
        res.invalid.append({"item_id": item_id, "reason": invalid_reason})
    if p.rationale:
        item.notes = p.rationale[:300]
    return item


def _mint(file: LensMarkFile, item_type: str, used_ids: set[str]) -> str:
    prefix = {"arrow": "ann-arrow-", "mask_circle": "ann-mask-", "einstein_ring": "ann-ring-", "text": "ann-text-"}[item_type]
    existing = {it.id for it in file.items} | used_ids
    n = 1
    while f"{prefix}{n:03d}" in existing:
        n += 1
    new_id = f"{prefix}{n:03d}"
    used_ids.add(new_id)
    return new_id


def _arrow_color(raw: Optional[str], label: Optional[str], item_id: str, res: ValidationResult,
                 used: Optional[set[str]]) -> str:
    allowed = tuple(config.ARROW_ORDER)
    if _is_deflector(label):
        color = config.DEFLECTOR_COLOR
    elif raw:
        color = nearest_palette(raw, allowed=allowed)
    else:
        color = next((c for c in allowed if c != config.DEFLECTOR_COLOR and (used is None or c not in used)),
                     allowed[1])
        res.repairs.append({"item_id": item_id, "field": "color", "from": None, "to": color, "why": "auto_palette"})
    if raw and color != raw.strip().lower():
        res.repairs.append({"item_id": item_id, "field": "color", "from": raw, "to": color,
                            "why": "deflector_is_green" if _is_deflector(label) else "color_snapped"})
    if used is not None:
        used.add(color)
    return color


def _deflector_center(file: LensMarkFile) -> Optional[list[float]]:
    for it in file.items:
        if isinstance(it, Arrow) and _is_deflector(it.label) and it.status != "rejected":
            return list(it.head)
    return None


def validate_proposal(proposal: Proposal, file: LensMarkFile, created_by: CreatedBy, *,
                      status: str = "proposed", mask_cap: int = config.MASK_CAP) -> ValidationResult:
    """All items of a proposal -> strict items (ids minted against ``file``), with repair/invalid records.
    Masks beyond ``mask_cap`` are kept but flagged ``over_cap`` (status invalid) so nothing is silently dropped."""
    res = ValidationResult()
    used: set[str] = set()
    colors_used: set[str] = set()
    n_masks = 0
    # deflector arrows first so their colour/centre are known to the rest
    ordered = sorted(proposal.items, key=lambda p: 0 if (p.type == "arrow" and _is_deflector(p.label)) else 1)
    for p in ordered:
        it = item_from_proposal(p, file, created_by, used_ids=used, res=res, status=status, arrow_colors_used=colors_used)
        if it is None:
            continue
        if isinstance(it, MaskCircle):
            n_masks += 1
            if n_masks > mask_cap and it.status != "invalid":
                it.status = "invalid"
                it.invalid_reason = "over_cap"
                res.invalid.append({"item_id": it.id, "reason": "over_cap"})
        res.items.append(it)
    return res
