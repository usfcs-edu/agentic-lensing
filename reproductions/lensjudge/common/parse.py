"""Robustly extract a JSON object from an LLM's free-text final message.

The SDK returns the agent's final answer as text (``ResultMessage.result``). Graders
instruct the model to emit one JSON object; this module finds and validates it. A
single repair retry is the caller's job (see ``imaging.grader_lean``) — here we just
parse, returning ``None`` on failure so parse failures can be counted as a metric.
"""
from __future__ import annotations

import json
from typing import Optional, Type, TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


def extract_json_block(text: str) -> Optional[dict]:
    """Find the last balanced ``{...}`` object in text and json-load it."""
    if not text:
        return None
    # Prefer a ```json fenced block if present.
    fenced = text
    if "```" in text:
        parts = text.split("```")
        for seg in parts:
            seg = seg.lstrip()
            if seg.startswith("json"):
                seg = seg[4:]
            seg = seg.strip()
            if seg.startswith("{"):
                obj = _try_balanced(seg)
                if obj is not None:
                    return obj
    return _try_balanced(fenced)


def _try_balanced(text: str) -> Optional[dict]:
    """Scan for the outermost balanced brace span and parse it (last one wins)."""
    best = None
    depth = 0
    start = -1
    in_str = False
    esc = False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    chunk = text[start:i + 1]
                    try:
                        best = json.loads(chunk)
                    except Exception:
                        pass
    return best


def parse_model(text: str, model: Type[T]) -> Optional[T]:
    """Extract JSON and validate into ``model``; None on any failure."""
    obj = extract_json_block(text)
    if obj is None:
        return None
    try:
        return model.model_validate(obj)
    except ValidationError:
        return None
    except Exception:
        return None
