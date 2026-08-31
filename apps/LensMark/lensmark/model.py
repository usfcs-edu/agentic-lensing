"""LensMark data model (pydantic v2). ``lensmark/1.0`` storage schema + the model-emitted proposal,
critique and patch documents.

Storage models are strict (``extra="forbid"``, like reproductions/lensjudge/golden/schemas_panel.py) so
a typo in a hand-edited file is an error, not silent data loss. Model-emitted documents (``Proposal``,
``Patch``) are lenient (``extra="ignore"``) and go through ``validate.py`` before becoming items.
"""
from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from datetime import datetime, timezone
from typing import Annotated, Any, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

from . import config

# ----------------------------------------------------------------------------- shared aliases
UV = Annotated[list[float], Field(min_length=2, max_length=2)]
ColorName = Literal["magenta", "cyan", "green", "yellow", "white", "orange", "gray", "mask_red", "ring_white"]
COLOR_NAMES: tuple[str, ...] = ("magenta", "cyan", "green", "yellow", "white", "orange", "gray", "mask_red", "ring_white")
Effort = Literal["low", "medium", "high", "xhigh", "max"]
Status = Literal["proposed", "accepted", "edited", "rejected", "invalid"]
Verdict = Literal["correct", "wrong_position", "wrong_label", "wrong_type", "wrong_size",
                  "spurious", "redundant", "missed_by_model"]
VERDICTS: tuple[str, ...] = ("correct", "wrong_position", "wrong_label", "wrong_type", "wrong_size",
                             "spurious", "redundant", "missed_by_model")
ItemType = Literal["arrow", "mask_circle", "einstein_ring", "text"]
ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Lenient(BaseModel):
    model_config = ConfigDict(extra="ignore")


# ----------------------------------------------------------------------------- provenance / review
class CreatedBy(Strict):
    kind: Literal["human", "claude", "import", "voice"] = "human"
    model: Optional[str] = None
    effort: Optional[Effort] = None
    run_id: Optional[str] = None
    reviewer: Optional[str] = None


class Review(Strict):
    verdict: Verdict
    severity: Optional[Literal["minor", "major"]] = None
    comment: str = ""
    reviewer: Optional[str] = None
    reviewed_at: Optional[str] = None
    delta_arcsec: Optional[float] = None


# ----------------------------------------------------------------------------- items
class ItemBase(Strict):
    id: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")
    label: Optional[str] = None
    color: ColorName = "white"
    show_in_legend: bool = True
    style: Optional[dict[str, Any]] = None          # per-item overrides of style_defaults[<kind>]
    created_by: CreatedBy = Field(default_factory=CreatedBy)
    created_at: str = Field(default_factory=now_iso)
    status: Status = "accepted"
    invalid_reason: Optional[str] = None
    edit_of: Optional[dict[str, Any]] = None          # pre-edit geometry when status == "edited"
    review: Optional[Review] = None
    notes: Optional[str] = None


class Arrow(ItemBase):
    type: Literal["arrow"] = "arrow"
    tail: UV
    head: UV                                          # the pointed-at feature
    label_anchor: Literal["tail", "head", "auto"] = "auto"
    label_offset: Optional[UV] = None                 # (du, dv) nudge applied to the label position
    color: ColorName = "cyan"


class MaskCircle(ItemBase):
    type: Literal["mask_circle"] = "mask_circle"
    center: UV
    radius_arcsec: float = Field(gt=0)                # = the mask radius to apply during lens modelling
    kind: Literal["galaxy", "star", "artifact"]       # galaxy -> dashed, star -> dotted (deck PROMPT 2)
    color: ColorName = "mask_red"
    show_in_legend: bool = False


class EinsteinRing(ItemBase):
    type: Literal["einstein_ring"] = "einstein_ring"
    center: UV                                        # on the deflector
    theta_e_arcsec: float = Field(gt=0)
    center_ref: Optional[str] = None                  # id of the deflector arrow whose head the centre tracks
    label_pos: Optional[UV] = None                    # where the "theta_E ~ 1.5"" text goes (auto if None)
    color: ColorName = "ring_white"
    show_in_legend: bool = False


class TextNote(ItemBase):
    type: Literal["text"] = "text"
    pos: UV
    text: str
    color: ColorName = "white"
    show_in_legend: bool = False


Item = Annotated[Union[Arrow, MaskCircle, EinsteinRing, TextNote], Field(discriminator="type")]
ITEM_CLASSES: dict[str, type[ItemBase]] = {"arrow": Arrow, "mask_circle": MaskCircle,
                                           "einstein_ring": EinsteinRing, "text": TextNote}


def parse_item(d: dict[str, Any]) -> ItemBase:
    cls = ITEM_CLASSES.get(str(d.get("type")))
    if cls is None:
        raise ValueError(f"unknown item type {d.get('type')!r}")
    return cls.model_validate(d)


# ----------------------------------------------------------------------------- image / system
class Wcs(Strict):
    ra_deg: float
    dec_deg: float
    rot_deg: float = 0.0


class ImageMeta(Strict):
    file: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    width: int = Field(ge=1)
    height: int = Field(ge=1)
    cutout_arcsec: float = Field(gt=0)                # angular size of the image WIDTH
    pixel_scale_arcsec: float = Field(gt=0)           # = cutout_arcsec / width (asserted)
    native_pixel_scale_arcsec: Optional[float] = None
    array_origin: Literal["upper", "lower"] = "upper"
    north_up: bool = True
    east_left: bool = True
    survey: Optional[str] = None
    instrument: Optional[str] = None
    filters: Optional[list[str]] = None
    wcs: Optional[Wcs] = None
    render_recipe: Optional[dict[str, Any]] = None
    scale_source: Optional[Literal["config", "override", "header", "assumed"]] = None

    @model_validator(mode="after")
    def _scale_consistent(self) -> "ImageMeta":
        expect = self.cutout_arcsec / self.width
        if abs(self.pixel_scale_arcsec - expect) > 1e-5 * expect:
            raise ValueError(f"pixel_scale_arcsec {self.pixel_scale_arcsec} != cutout_arcsec/width {expect}")
        return self


class ThetaE(Strict):
    value_arcsec: Optional[float] = None
    method: Optional[str] = None                      # geometric | arc_radius | model | human | alt | ...
    alt_arcsec: Optional[float] = None
    uncertainty_arcsec: Optional[float] = None


class SystemBlock(Strict):
    object_id: Optional[str] = None
    rank: Optional[int] = None
    grade: Optional[Literal["A", "B", "C", "D"]] = None       # lensjudge alphabet (common/schemas.py)
    score_1_4: Optional[int] = Field(default=None, ge=1, le=4)  # golden-campaign scale
    confidence_lmh: Optional[Literal["L", "M", "H"]] = None
    p_lens: Optional[float] = Field(default=None, ge=0, le=1)
    confidence: Optional[float] = Field(default=None, ge=0, le=1)
    theta_e: ThetaE = Field(default_factory=ThetaE)
    verdict: Optional[Literal["likely_lens", "possible", "not_lens", "unclear"]] = None
    description: str = ""                             # free text; refers to arrows by colour
    description_refs: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)


class CoordinatesBlock(Strict):
    geometry_units: Literal["normalized"] = "normalized"
    origin: Literal["top_left"] = "top_left"
    x_axis: Literal["right"] = "right"
    y_axis: Literal["down"] = "down"
    size_units: Literal["arcsec"] = "arcsec"
    note: str = ("px=(u*W, v*H); pixel_scale=cutout_arcsec/W; dE=(0.5-u)*W*ps if east_left; "
                 "dN=(0.5-v)*H*ps; PA from N through E; x_fits=u*W+0.5, y_fits=(1-v)*H+0.5 (array_origin=upper)")


class Legend(Strict):
    show: bool = True
    position: Literal["auto", "top_left", "top_right", "bottom_left", "bottom_right"] = "auto"
    order: Optional[list[str]] = None                 # item ids; default = items order


class ProposalRun(Strict):
    run_id: str
    model: str
    effort: Optional[str] = None
    engine: str = "sdk"
    prompt_sha256: Optional[str] = None
    fewshot_sha256: Optional[str] = None
    started_at: Optional[str] = None
    duration_s: Optional[float] = None
    usage: Optional[dict[str, Any]] = None
    cost_usd: Optional[float] = None
    num_turns: Optional[int] = None
    n_items_proposed: int = 0
    n_invalid: int = 0
    n_repaired: int = 0
    parse_ok: bool = True
    proposal_file: Optional[str] = None
    error: Optional[str] = None
    proposed_system: Optional[dict[str, Any]] = None   # the model's system block (verdict, description, theta_e...) - never auto-applied


class Provenance(Strict):
    proposal_runs: list[ProposalRun] = Field(default_factory=list)
    critiques: list[str] = Field(default_factory=list)
    log: Optional[str] = None


class RenderInfo(Strict):
    renderer: str
    output: str
    of_json_sha256: str
    rendered_at: Optional[str] = None


# ----------------------------------------------------------------------------- the file
_COLOR_WORD_RE = re.compile(r"\b(magenta|cyan|green|yellow|white|orange|gr[ae]y)\b", re.I)


class LensMarkFile(Strict):
    schema_version: Literal["lensmark/1.0"] = "lensmark/1.0"
    id: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")
    created: str = Field(default_factory=now_iso)
    modified: str = Field(default_factory=now_iso)
    image: ImageMeta
    coordinates: CoordinatesBlock = Field(default_factory=CoordinatesBlock)
    system: SystemBlock = Field(default_factory=SystemBlock)
    palette: str = config.PALETTE_VERSION
    style_defaults: dict[str, Any] = Field(default_factory=lambda: deepcopy(config.STYLE_DEFAULTS))
    legend: Legend = Field(default_factory=Legend)
    items: list[Item] = Field(default_factory=list)
    provenance: Provenance = Field(default_factory=Provenance)
    render: Optional[RenderInfo] = None

    @model_validator(mode="after")
    def _unique_ids(self) -> "LensMarkFile":
        seen: set[str] = set()
        for it in self.items:
            if it.id in seen:
                raise ValueError(f"duplicate item id {it.id!r}")
            seen.add(it.id)
        return self

    # ---- helpers
    def item(self, item_id: str) -> Optional[ItemBase]:
        for it in self.items:
            if it.id == item_id:
                return it
        return None

    def next_id(self, item_type: str) -> str:
        prefix = {"arrow": "ann-arrow-", "mask_circle": "ann-mask-", "einstein_ring": "ann-ring-",
                  "text": "ann-text-"}[item_type]
        n = 1
        ids = {it.id for it in self.items}
        while f"{prefix}{n:03d}" in ids:
            n += 1
        return f"{prefix}{n:03d}"

    def lint(self) -> list[str]:
        """Non-fatal consistency warnings (shown in the UI, never block a save)."""
        warns: list[str] = []
        ids = {it.id for it in self.items}
        colors = {it.color for it in self.items if it.status in ("accepted", "edited", "proposed")}
        for word in {w.lower() for w in _COLOR_WORD_RE.findall(self.system.description)}:
            word = "gray" if word == "grey" else word
            if word not in colors:
                warns.append(f"description mentions '{word}' but no item has that colour")
        for ref in self.system.description_refs:
            if ref not in ids:
                warns.append(f"description_refs: unknown item {ref!r}")
        for it in self.items:
            if isinstance(it, EinsteinRing) and it.center_ref and it.center_ref not in ids:
                warns.append(f"{it.id}: center_ref {it.center_ref!r} does not exist")
            if it.color in config.RESERVED_COLORS and config.RESERVED_COLORS[it.color] != it.type:
                warns.append(f"{it.id}: colour {it.color!r} is reserved for {config.RESERVED_COLORS[it.color]}")
        return warns

    def to_dict(self, *, without_render: bool = False) -> dict[str, Any]:
        d = self.model_dump(mode="json", exclude_none=True)
        if without_render:
            d.pop("render", None)
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n"

    def content_sha256(self) -> str:
        """sha256 of the canonical JSON minus the ``render`` block - what ``render.of_json_sha256`` pins."""
        payload = json.dumps(self.to_dict(without_render=True), sort_keys=True, separators=(",", ":"),
                             ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @classmethod
    def from_json(cls, text: str) -> "LensMarkFile":
        return cls.model_validate(json.loads(text))


# ----------------------------------------------------------------------------- model-emitted: proposal
class ProposalThetaE(Lenient):
    value_arcsec: Optional[float] = None
    method: Optional[str] = None
    alt_arcsec: Optional[float] = None
    uncertainty_arcsec: Optional[float] = None


class ProposalSystem(Lenient):
    grade: Optional[str] = None
    p_lens: Optional[float] = None
    confidence: Optional[float] = None
    verdict: Optional[str] = None
    theta_e: Optional[ProposalThetaE] = None
    description: str = ""
    tags: list[str] = Field(default_factory=list)


class ProposalItem(Lenient):
    """One item as the model emits it - flat so the JSON Schema has no $ref cycles."""
    type: str
    label: Optional[str] = None
    color: Optional[str] = None
    tail: Optional[list[float]] = None
    head: Optional[list[float]] = None
    center: Optional[list[float]] = None
    radius_arcsec: Optional[float] = None
    kind: Optional[str] = None
    theta_e_arcsec: Optional[float] = None
    text: Optional[str] = None
    pos: Optional[list[float]] = None
    label_anchor: Optional[str] = None
    confidence: Optional[float] = None
    rationale: Optional[str] = None


class Proposal(Lenient):
    system: ProposalSystem = Field(default_factory=ProposalSystem)
    items: list[ProposalItem] = Field(default_factory=list)
    notes: str = ""


# ----------------------------------------------------------------------------- critique
class CritiqueItem(Strict):
    item_id: str
    verdict: Verdict
    severity: Optional[Literal["minor", "major"]] = None
    comment: str = ""
    delta_arcsec: Optional[float] = None


class CritiquePanel(Strict):
    completeness: Optional[int] = Field(default=None, ge=1, le=5)
    geometric_accuracy: Optional[int] = Field(default=None, ge=1, le=5)
    label_quality: Optional[int] = Field(default=None, ge=1, le=5)
    description_quality: Optional[int] = Field(default=None, ge=1, le=5)
    theta_e_verdict: Optional[Literal["correct", "too_small", "too_large", "missing", "spurious"]] = None
    theta_e_human_arcsec: Optional[float] = None
    free_text: str = ""
    would_use_as_fewshot: Optional[bool] = None


class Critique(Strict):
    schema_version: Literal["lensmark-critique/1.0"] = "lensmark-critique/1.0"
    image_id: str
    run_id: str
    model: Optional[str] = None
    effort: Optional[str] = None
    reviewer: str
    reviewed_at: str = Field(default_factory=now_iso)
    lead_time_s: Optional[float] = None
    items: list[CritiqueItem] = Field(default_factory=list)
    panel: CritiquePanel = Field(default_factory=CritiquePanel)
    counts: dict[str, int] = Field(default_factory=dict)   # proposed / accepted / edited / rejected / added_by_human


# ----------------------------------------------------------------------------- voice / NL patch
class PatchOp(Lenient):
    op: Literal["add", "update", "delete"]
    id: Optional[str] = None                          # target item id ("$system" = the system block)
    item: Optional[dict[str, Any]] = None             # for add: a ProposalItem-shaped dict
    set: Optional[dict[str, Any]] = None              # for update/delete: fields to set (delete may set review)
    confidence: Optional[float] = None
    rationale: Optional[str] = None


class Patch(Lenient):
    schema_version: str = "lensmark-patch/1.0"
    transcript: str = ""
    ops: list[PatchOp] = Field(default_factory=list)
    clarification: Optional[str] = None
