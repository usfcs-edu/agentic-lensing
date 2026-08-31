"""LensMark configuration: palette, style constants, model/effort matrix, defaults, env overrides.

Everything that both the Python renderer and the browser preview must agree on lives in
``lensmark/schema/*.json`` and is served verbatim at ``GET /api/style`` so there is exactly one copy.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

PKG = Path(__file__).resolve().parent
SCHEMA_DIR = PKG / "schema"
FONT_DIR = PKG / "render" / "fonts"
STATIC_DIR = PKG / "static"
PROMPT_DIR = PKG / "claude" / "prompts"

SCHEMA_VERSION = "lensmark/1.0"
PROPOSAL_SCHEMA_VERSION = "lensmark-proposal/1.0"
CRITIQUE_SCHEMA_VERSION = "lensmark-critique/1.0"
PATCH_SCHEMA_VERSION = "lensmark-patch/1.0"
PALETTE_VERSION = "lensmark/v1"
RENDERER_VERSION = "lensmark-render/0.1.0"


def _load_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


_PALETTE_DOC = _load_json(SCHEMA_DIR / "palette.json")
PALETTE: dict[str, str] = dict(_PALETTE_DOC["colors"])           # name -> "#RRGGBB"
ARROW_ORDER: tuple[str, ...] = tuple(_PALETTE_DOC["arrow_order"])   # auto-assignment order for arrows
DEFLECTOR_COLOR: str = _PALETTE_DOC["deflector"]
RESERVED_COLORS: dict[str, str] = dict(_PALETTE_DOC["reserved"])   # colour -> the only item type allowed
STYLE_DEFAULTS: dict[str, Any] = _load_json(SCHEMA_DIR / "style_defaults.json")


def palette_rgb(name: str) -> tuple[int, int, int]:
    h = PALETTE[name].lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


# ----------------------------------------------------------------------------- models / effort
# Full model ids only are sent to Claude (aliases like "opus" resolve differently across CLI
# versions - the SDK's bundled CLI may predate the Claude-5 ids).
MODELS: list[dict[str, Any]] = [
    {"alias": "fable", "id": "claude-fable-5", "label": "Fable 5", "supports_effort": True,
     "price_in": 10.0, "price_out": 50.0},
    {"alias": "opus", "id": "claude-opus-5", "label": "Opus 5", "supports_effort": True,
     "price_in": 5.0, "price_out": 25.0},
    {"alias": "sonnet", "id": "claude-sonnet-5", "label": "Sonnet 5", "supports_effort": True,
     "price_in": 2.0, "price_out": 10.0},
    {"alias": "haiku", "id": "claude-haiku-4-5", "label": "Haiku 4.5", "supports_effort": False,
     "price_in": 1.0, "price_out": 5.0},
]
FULL_ID: dict[str, str] = {m["alias"]: m["id"] for m in MODELS}
_BY_ID: dict[str, dict[str, Any]] = {m["id"]: m for m in MODELS}
EFFORTS: tuple[str, ...] = ("low", "medium", "high", "xhigh", "max")
DEFAULT_MODEL = "opus"
DEFAULT_EFFORT = "xhigh"


def resolve_model(name: str) -> str:
    """Alias or full id -> full id. Unknown full ids (``claude-…``) pass through unchanged."""
    if name in FULL_ID:
        return FULL_ID[name]
    if name.startswith("claude-"):
        return name
    raise ValueError(f"unknown model {name!r}; use one of {sorted(FULL_ID)} or a full claude-* id")


def model_supports_effort(model_id: str) -> bool:
    m = _BY_ID.get(model_id)
    return True if m is None else bool(m["supports_effort"])


# ----------------------------------------------------------------------------- runtime / env
DEFAULT_PORT = 8765
DEFAULT_BIND = "127.0.0.1"
IMAGE_EXTS = (".png", ".jpg", ".jpeg")
DERIVED_SUFFIXES = (".annot", ".mask")     # <id>.annot.png / <id>.mask.png are never treated as originals
MAX_TURNS = 2
MASK_CAP = 12                               # the propose prompt caps mask circles here (deck: "prominent only")


def engine_name() -> str:
    """``sdk`` (real Claude via the Agent SDK) or ``fixture`` (canned proposals; used by all UI QA)."""
    return os.environ.get("LENSMARK_ENGINE", "sdk")


def claude_bin() -> str | None:
    """The ``claude`` executable the SDK must spawn. Pinned to PATH by default because the SDK's
    ``_find_cli`` prefers its *bundled* CLI, which can lag the installed one by months."""
    return os.environ.get("LENSMARK_CLAUDE_BIN") or shutil.which("claude")


def max_budget_usd() -> float:
    return float(os.environ.get("LENSMARK_MAX_BUDGET_USD", "0.50"))


# ----------------------------------------------------------------------------- campaign defaults
CAMPAIGN_CONFIG_NAME = "lensmark.config.json"
CAMPAIGN_MANIFEST_NAME = "lensmark.manifest.json"
CAMPAIGN_DEFAULTS: dict[str, Any] = {
    "schema_version": "lensmark-config/1.0",
    "cutout_arcsec": 16.0,               # angular size of the full image width
    "cutout_arcsec_source": "assumed",   # config | override | header | assumed  (shown as a badge in the UI)
    "native_pixel_scale_arcsec": None,   # survey scale, used for mask export at native resolution
    "survey": None,
    "north_up": True,
    "east_left": True,
    "array_origin": "upper",
    "default_model": DEFAULT_MODEL,
    "default_effort": DEFAULT_EFFORT,
    "reviewer": "xhuang",
    "overrides": {},                     # {id: {cutout_arcsec, rank, object_id, ra_deg, dec_deg, theta_e_ref_arcsec, ...}}
}
