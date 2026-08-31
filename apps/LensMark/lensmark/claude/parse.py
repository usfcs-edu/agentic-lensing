"""Robustly extract a JSON object from a model's free-text reply.

Vendored from reproductions/lensjudge/common/parse.py:18-90. The structured-output path
(``ResultMessage.structured_output``) normally makes this unnecessary; it is the fallback when the
model answers in prose. ``_try_balanced`` tracks string state only INSIDE a brace span because
prose before the JSON routinely carries an odd number of double quotes - arcsecond marks such as
``1.5"`` in labels and descriptions - which would otherwise put the opening brace "inside a string".
"""
from __future__ import annotations

import json
from typing import Optional, Type, TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


# from reproductions/lensjudge/common/parse.py:18
def extract_json_block(text: str) -> Optional[dict]:
    """Find the last balanced ``{...}`` object in text and json-load it (prefers a ```json fence)."""
    if not text:
        return None
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


# from reproductions/lensjudge/common/parse.py:38
def _try_balanced(text: str) -> Optional[dict]:
    """Scan for the outermost balanced brace span and parse it (last parseable one wins).

    String state is tracked only INSIDE a brace span: prose before the JSON routinely carries an odd
    number of double quotes (arcsecond marks such as 3.5", a quoted word), and tracking quotes from
    the first character would put the record's opening brace "inside a string" and return its last
    nested object instead of the record."""
    best = None
    depth = 0
    start = -1
    in_str = False
    esc = False
    for i, ch in enumerate(text):
        if depth == 0:
            if ch == "{":
                start, depth, in_str, esc = i, 1, False, False
            continue
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
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                chunk = text[start:i + 1]
                try:
                    obj = json.loads(chunk)
                except Exception:
                    obj = None
                if isinstance(obj, dict):
                    best = obj
    return best


# from reproductions/lensjudge/common/parse.py:80
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
