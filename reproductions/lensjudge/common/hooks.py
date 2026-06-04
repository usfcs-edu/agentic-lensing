"""SDK hooks that log every tool call to a JSONL trace — the eval-harness substrate.

PreToolUse/PostToolUse/SubagentStop callbacks append one record per event to a
trace file (tool name, trimmed input, latency, errors), so a grading run is fully
replayable and auditable. Callbacks return ``{}`` (no-op allow) — they observe, they
do not gate. Image payloads are never logged (only a byte count), to keep traces small.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from claude_agent_sdk import HookMatcher


def _field(obj: Any, name: str, default=None):
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


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
    if isinstance(value, str) and len(value) > limit:
        return f"<{len(value)} chars elided>"
    return value


class Trace:
    """Append-only JSONL event log for one grading run."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.n = 0

    def write(self, event: str, **fields):
        rec = {"t": time.time(), "event": event, **fields}
        with open(self.path, "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")
        self.n += 1

    def hooks(self) -> dict:
        """Build the ClaudeAgentOptions(hooks=...) dict bound to this trace."""

        async def pre(input_data, tool_use_id, context):
            self.write("pre_tool",
                       tool=_field(input_data, "tool_name"),
                       tool_use_id=tool_use_id,
                       input=_trim(_field(input_data, "tool_input", {})))
            return {}

        async def post(input_data, tool_use_id, context):
            resp = _field(input_data, "tool_response", _field(input_data, "tool_result"))
            is_err = _field(input_data, "is_error", False)
            self.write("post_tool",
                       tool=_field(input_data, "tool_name"),
                       tool_use_id=tool_use_id,
                       is_error=bool(is_err),
                       response=_trim(resp))
            return {}

        async def subagent_stop(input_data, tool_use_id, context):
            self.write("subagent_stop", tool_use_id=tool_use_id)
            return {}

        return {
            "PreToolUse": [HookMatcher(hooks=[pre])],
            "PostToolUse": [HookMatcher(hooks=[post])],
            "SubagentStop": [HookMatcher(hooks=[subagent_stop])],
        }
