"""The engine contract every Claude caller in LensMark uses (propose, voice patch, doctor).

An engine takes ONE user turn made of Anthropic content blocks (text + base64 images), a system
prompt and a JSON Schema, and returns the model's structured output plus cost/usage. Implementations:
``SdkEngine`` (claude_agent_sdk, rides the logged-in ``claude`` CLI) and ``FixtureEngine`` (canned
JSON from tests/fixtures; used by all UI QA). ``get_engine()`` lives in ``lensmark.claude.engine``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Protocol

EventCallback = Callable[[dict[str, Any]], None]
# on_event dicts: {"phase": "started"|"thinking"|"partial"|"tool"|"validated"|"done"|"error",
#                  "detail": str, "text"?: str, "cost_usd"?: float}


@dataclass
class EngineRequest:
    system: str                               # system prompt (frozen text; sha recorded by the caller)
    content: list[dict[str, Any]]             # Anthropic content blocks: {"type":"text",...} / {"type":"image","source":{...}}
    schema: dict[str, Any]                    # JSON Schema for structured output (flat, additionalProperties:false)
    model: str                                # FULL model id (config.resolve_model)
    effort: Optional[str] = None              # low|medium|high|xhigh|max or None (haiku)
    max_budget_usd: float = 0.50
    max_turns: int = 2
    cwd: Optional[Path] = None
    fixture_key: Optional[str] = None         # FixtureEngine: which canned file (e.g. the image id)
    purpose: str = "propose"                  # "propose" | "patch" | "doctor" - FixtureEngine picks the fixture dir by this


@dataclass
class EngineResult:
    structured: Optional[dict[str, Any]]      # parsed structured output (None if the model failed to comply)
    text: str = ""                            # final assistant text (fallback JSON extraction happens here)
    thinking: str = ""
    cost_usd: Optional[float] = None
    usage: dict[str, Any] = field(default_factory=dict)
    num_turns: Optional[int] = None
    duration_s: Optional[float] = None
    model: Optional[str] = None
    effort: Optional[str] = None
    error: Optional[str] = None
    raw: dict[str, Any] = field(default_factory=dict)   # engine-specific extras (session id, stop_reason, argv...)


class Engine(Protocol):
    name: str

    async def run(self, req: EngineRequest, on_event: Optional[EventCallback] = None) -> EngineResult: ...
