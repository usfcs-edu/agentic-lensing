"""LensMark HTTP server (FastAPI) - the routes in ``API.md``.

The route modules import the sibling core modules (``render``, ``claude.propose``, ``critique``,
``evaluate``, ``exports``, ``voice``) **lazily inside the handlers** so the server starts - and the
manual-annotation routes work - while those modules are still missing; their routes answer
``501 {"error": "module not available: <name>"}`` until they land.
"""
from __future__ import annotations

import importlib
import math
import re
from pathlib import Path
from types import ModuleType
from typing import Any, Optional

from fastapi import HTTPException, Request

from ..store import Campaign

ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


class ModuleUnavailable(HTTPException):
    """A sibling module is not importable (yet) -> 501."""

    def __init__(self, name: str, exc: Optional[BaseException] = None):
        super().__init__(status_code=501, detail=f"module not available: {name}")
        self.module_name = name
        self.cause_text = f"{type(exc).__name__}: {exc}" if exc is not None else None


def lazy_module(name: str, *attrs: str) -> ModuleType:
    """Import ``name`` now; a missing module - or one that does not (yet) define ``attrs`` because a
    sibling agent is still writing it - becomes a 501 instead of a crash."""
    try:
        mod = importlib.import_module(name)
    except ImportError as e:
        raise ModuleUnavailable(name, e) from e
    for a in attrs:
        if not hasattr(mod, a):
            raise ModuleUnavailable(f"{name}.{a}", AttributeError(f"{name} has no attribute {a!r}"))
    return mod


def optional_module(name: str, *attrs: str) -> Optional[ModuleType]:
    try:
        return lazy_module(name, *attrs)
    except ModuleUnavailable:
        return None


def campaign_of(request: Request) -> Campaign:
    return request.app.state.campaign


def check_id(image_id: str) -> str:
    if not ID_RE.match(image_id):
        raise HTTPException(status_code=400, detail=f"invalid image id {image_id!r}")
    return image_id


def require_image(campaign: Campaign, image_id: str) -> Path:
    """The original image path; unknown id -> 404 (FileNotFoundError is mapped by the app)."""
    check_id(image_id)
    return campaign.image_path(image_id)


def rel_path(campaign: Campaign, p: Path | str) -> str:
    """Path relative to the campaign root when inside it, else absolute."""
    p = Path(p)
    try:
        return str(p.resolve().relative_to(campaign.root))
    except ValueError:
        return str(p)


def engine_or_none(name: str) -> Any:
    """``lensmark.claude.engine.get_engine(name)`` if that module exists, else None (the callee then
    resolves the engine itself from ``$LENSMARK_ENGINE``)."""
    mod = optional_module("lensmark.claude.engine", "get_engine")
    if mod is None:
        return None
    return mod.get_engine(name)


def clean_json(obj: Any) -> Any:
    """Recursively make a value JSON-safe: NaN/inf -> None, pydantic models -> dicts, Paths -> str."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {str(k): clean_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [clean_json(v) for v in obj]
    if hasattr(obj, "model_dump"):
        return clean_json(obj.model_dump(mode="json", exclude_none=True))
    if isinstance(obj, Path):
        return str(obj)
    try:  # numpy scalars
        import numpy as np  # noqa: WPS433
        if isinstance(obj, np.generic):
            return clean_json(obj.item())
    except ImportError:  # pragma: no cover
        pass
    return obj


__all__ = ["ID_RE", "ModuleUnavailable", "campaign_of", "check_id", "clean_json", "engine_or_none",
           "lazy_module", "optional_module", "rel_path", "require_image"]
