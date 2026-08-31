"""Append-only JSONL trace of one engine call: SDK tool hooks + the reasoning/text the hooks never see.

Vendored from reproductions/lensjudge/common/hooks.py:21-39 (``_field``/``_trim``) and :42-88
(``Trace``, ``Trace.hooks``). Image payloads are never logged - only a byte count - so traces stay
small. Tracing is on when ``LENSMARK_TRACE_DIR`` is set; ``propose.py`` copies the engine's trace next
to the immutable proposal file as ``proposals/<id>.<run_id>.trace.jsonl``.
"""
from __future__ import annotations

import json
import os
import secrets
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

try:  # HookMatcher only matters for the SDK engine; the fixture engine never builds hooks
    from claude_agent_sdk import HookMatcher
except ImportError:  # pragma: no cover
    HookMatcher = None


def trace_dir() -> Optional[Path]:
    """``LENSMARK_TRACE_DIR`` as a Path, or None (tracing off)."""
    v = os.environ.get("LENSMARK_TRACE_DIR")
    return Path(v).expanduser() if v else None


# from reproductions/lensjudge/common/hooks.py:21
def _field(obj: Any, name: str, default=None):
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


# from reproductions/lensjudge/common/hooks.py:27
def _trim(value: Any, limit: int = 600) -> Any:
    """Trim large/base64 fields out of a tool-input dict before logging."""
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            if isinstance(v, str) and len(v) > limit:
                out[k] = f"<{len(v)} chars elided>"
            else:
                out[k] = _trim(v, limit)
        return out
    if isinstance(value, list):
        return [_trim(v, limit) for v in value]
    if isinstance(value, str) and len(value) > limit:
        return f"<{len(value)} chars elided>"
    return value


def elide_images(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Content blocks with base64 image data replaced by a byte count (for logging the request)."""
    out = []
    for b in blocks:
        if isinstance(b, dict) and b.get("type") == "image":
            src = dict(b.get("source") or {})
            data = src.get("data")
            if isinstance(data, str):
                src["data"] = f"<{len(data)} b64 chars elided>"
            out.append({**b, "source": src})
        else:
            out.append(_trim(b))
    return out


# from reproductions/lensjudge/common/hooks.py:42
class Trace:
    """Append-only JSONL event log for one engine call."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.n = 0

    def write(self, event: str, **fields: Any) -> None:
        rec = {"t": time.time(), "event": event, **fields}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=str, ensure_ascii=False) + "\n")
        self.n += 1

    def hooks(self) -> dict:
        """Build the ``ClaudeAgentOptions(hooks=...)`` dict bound to this trace (observe-only)."""
        if HookMatcher is None:  # pragma: no cover
            raise RuntimeError("Trace.hooks() requires claude_agent_sdk")

        async def pre(input_data, tool_use_id, context):
            self.write("pre_tool", tool=_field(input_data, "tool_name"), tool_use_id=tool_use_id,
                       input=_trim(_field(input_data, "tool_input", {})))
            return {}

        async def post(input_data, tool_use_id, context):
            resp = _field(input_data, "tool_response", _field(input_data, "tool_result"))
            is_err = _field(input_data, "is_error", False)
            self.write("post_tool", tool=_field(input_data, "tool_name"), tool_use_id=tool_use_id,
                       is_error=bool(is_err), response=_trim(resp))
            return {}

        async def subagent_stop(input_data, tool_use_id, context):
            self.write("subagent_stop", tool_use_id=tool_use_id)
            return {}

        return {
            "PreToolUse": [HookMatcher(hooks=[pre])],
            "PostToolUse": [HookMatcher(hooks=[post])],
            "SubagentStop": [HookMatcher(hooks=[subagent_stop])],
        }


def for_request(purpose: str, key: Optional[str]) -> Optional[Trace]:
    """A fresh Trace under ``LENSMARK_TRACE_DIR`` for one engine call, or None when tracing is off."""
    d = trace_dir()
    if d is None:
        return None
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    safe = (key or "na").replace("/", "_")
    return Trace(d / f"{purpose}.{safe}.{ts}.{secrets.token_hex(2)}.jsonl")
