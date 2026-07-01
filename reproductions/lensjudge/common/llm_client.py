"""common/llm_client.py — backend-selectable LLM transport for LensJudge v4.

LensJudge historically called Anthropic Claude only (``grader_direct`` via the raw
Messages API; the agentic graders via the Claude Agent SDK). v4 makes the backend
**selectable** so the same graders can run on OPEN-WEIGHT models served behind any
OpenAI-compatible endpoint (vLLM / SGLang on GPUs, MLX-VLM / LM Studio / llama.cpp /
Ollama on the Mac), fully offline.

Design: the open-weight path is **purely additive**. The Anthropic path is untouched
and remains the default; this module is only reached when ``LENSJUDGE_BACKEND=openai``.
The schema (``common.schemas.ImageGrade``) and validator (``common.parse.parse_model``)
are provider-agnostic and reused verbatim — only the transport differs.

Env switches (read at call time so run scripts can set os.environ from CLI flags):
  LENSJUDGE_BACKEND      anthropic (default) | openai
  LENSJUDGE_BASE_URL     OpenAI-compatible base, e.g. http://localhost:8000/v1
  LENSJUDGE_API_KEY      key for the server (local servers accept anything; default "EMPTY")
  LENSJUDGE_TEMPERATURE  sampling temperature for the open backend (default 0.0 — grading is deterministic)
  LENSJUDGE_JSON_MODE    "1" -> request response_format={"type":"json_object"} (portable JSON nudge)
  LENSJUDGE_GUIDED_JSON  "1" -> send extra_body={"guided_json": <schema>} (vLLM/SGLang constrained decoding)

The served MODEL NAME flows through the EXISTING per-role seam: set e.g.
``LENSJUDGE_MODEL_GRADER=Qwen/Qwen3-VL-8B-Instruct`` (config._m already reads this) and
``config.MODELS["grader"]`` carries the open model id straight through to the request.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

# NOTE: `openai` is imported LAZILY inside _get_client so the default Anthropic path
# never requires it (it is not a hard dependency of the anthropic backend).

ANTHROPIC = "anthropic"
OPENAI = "openai"


def get_backend() -> str:
    """Return the active backend: 'anthropic' (default) or 'openai'."""
    b = os.environ.get("LENSJUDGE_BACKEND", ANTHROPIC).strip().lower()
    if b not in (ANTHROPIC, OPENAI):
        raise ValueError(f"LENSJUDGE_BACKEND must be {ANTHROPIC!r} or {OPENAI!r}, got {b!r}")
    return b


def is_open() -> bool:
    return get_backend() == OPENAI


def _temperature() -> float:
    return float(os.environ.get("LENSJUDGE_TEMPERATURE", "0.0"))


def _max_tokens(default: int = 2048) -> int:
    """Output-token budget. LENSJUDGE_MAX_TOKENS lets reasoning VLMs (e.g. GLM-4.6V, which emit long
    <think> blocks before the final JSON) have room to finish; too small a budget truncates the JSON
    and parsing fails. Default 2048 keeps existing (non-thinking) behavior unchanged."""
    return int(os.environ.get("LENSJUDGE_MAX_TOKENS", str(default)))


# --- open-weight price table (local inference is free; record tokens regardless) ---
# $/Mtok (input, output). Add hosted-endpoint prices here if a paid OpenAI-compatible
# provider is ever used; default 0.0 for self-hosted vLLM/SGLang/MLX.
_OPEN_PRICE: dict[str, tuple[float, float]] = {}


def _open_cost(model_id: str, usage: Any) -> float:
    pin, pout = _OPEN_PRICE.get(model_id, (0.0, 0.0))
    pin_t = getattr(usage, "prompt_tokens", 0) or 0
    pout_t = getattr(usage, "completion_tokens", 0) or 0
    return (pin_t * pin + pout_t * pout) / 1e6


# --- cached async client per (base_url) -------------------------------------
_clients: dict[str, Any] = {}


def _get_client():
    """Lazily build (and cache) an AsyncOpenAI client pointed at LENSJUDGE_BASE_URL."""
    base_url = os.environ.get("LENSJUDGE_BASE_URL")
    if not base_url:
        raise RuntimeError(
            "LENSJUDGE_BACKEND=openai requires LENSJUDGE_BASE_URL (e.g. "
            "http://localhost:8000/v1 for a local vLLM/SGLang/MLX server).")
    if base_url not in _clients:
        try:
            from openai import AsyncOpenAI  # lazy: only needed for the openai backend
        except ImportError as e:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "The 'openai' package is required for LENSJUDGE_BACKEND=openai. "
                "Install it in the lensjudge venv: pip install 'openai>=1.40'.") from e
        api_key = os.environ.get("LENSJUDGE_API_KEY", "EMPTY")
        _clients[base_url] = AsyncOpenAI(base_url=base_url, api_key=api_key)
    return _clients[base_url]


# --- content conversion: Anthropic blocks -> OpenAI parts -------------------
def anthropic_content_to_openai(content: list[dict]) -> list[dict]:
    """Convert the Anthropic Messages content-block list used by the graders into
    OpenAI chat ``content`` parts.

    Anthropic text  {"type":"text","text":...}                       -> {"type":"text","text":...}
    Anthropic image {"type":"image","source":{"type":"base64",       -> {"type":"image_url",
                     "media_type":"image/png","data":B64}}               "image_url":{"url":"data:image/png;base64,B64"}}
    """
    out: list[dict] = []
    for blk in content:
        t = blk.get("type")
        if t == "text":
            out.append({"type": "text", "text": blk.get("text", "")})
        elif t == "image":
            src = blk.get("source", {})
            if src.get("type") == "base64":
                media = src.get("media_type", "image/png")
                data = src.get("data", "")
                out.append({"type": "image_url",
                            "image_url": {"url": f"data:{media};base64,{data}"}})
            elif src.get("type") == "url":
                out.append({"type": "image_url", "image_url": {"url": src.get("url", "")}})
            else:  # pragma: no cover - defensive
                raise ValueError(f"unsupported image source: {src!r}")
        else:  # pragma: no cover - defensive
            raise ValueError(f"unsupported content block type: {t!r}")
    return out


@dataclass
class ChatResult:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    finish_reason: Optional[str] = None
    model: Optional[str] = None
    raw: Any = None


def _maybe_json_kwargs(json_schema: Optional[dict]) -> dict:
    """Optional, server-dependent JSON-constraint kwargs, gated by env (default off)."""
    kw: dict = {}
    if os.environ.get("LENSJUDGE_JSON_MODE") == "1":
        kw["response_format"] = {"type": "json_object"}
    if os.environ.get("LENSJUDGE_GUIDED_JSON") == "1" and json_schema is not None:
        # vLLM / SGLang constrained decoding; ignored by servers that don't support it
        kw["extra_body"] = {"guided_json": json_schema}
    return kw


async def chat_with_images(*, system: str, content: list[dict], model: str,
                           max_tokens: Optional[int] = None,
                           json_schema: Optional[dict] = None,
                           temperature: Optional[float] = None) -> ChatResult:
    """One multimodal chat completion (system + a single user turn with text+images).

    ``content`` is the SAME Anthropic-style block list the graders already build, so
    callers do not change how they render evidence — only the transport.
    """
    client = _get_client()
    parts = anthropic_content_to_openai(content)
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": parts}]
    kw = _maybe_json_kwargs(json_schema)
    resp = await client.chat.completions.create(
        model=model, messages=messages, max_tokens=max_tokens or _max_tokens(),
        temperature=_temperature() if temperature is None else temperature, **kw)
    return _chat_result(resp, model)


async def chat_text(*, system: str, text: str, model: str, max_tokens: Optional[int] = None,
                    temperature: Optional[float] = None) -> ChatResult:
    """One text-only chat completion (used for the JSON repair retry)."""
    client = _get_client()
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": text}]
    resp = await client.chat.completions.create(
        model=model, messages=messages, max_tokens=max_tokens or _max_tokens(),
        temperature=_temperature() if temperature is None else temperature)
    return _chat_result(resp, model)


def _chat_result(resp: Any, model: str) -> ChatResult:
    choice = resp.choices[0]
    msg = choice.message
    return ChatResult(
        text=msg.content or "",
        input_tokens=getattr(resp.usage, "prompt_tokens", 0) or 0,
        output_tokens=getattr(resp.usage, "completion_tokens", 0) or 0,
        cost_usd=_open_cost(model, resp.usage),
        finish_reason=getattr(choice, "finish_reason", None),
        model=model, raw=resp)


# --- agentic tool-calling loop (wired into the graders in Phase 3) ----------
@dataclass
class LoopResult:
    text: str
    cost_usd: float = 0.0
    num_turns: int = 0
    tool_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    messages: list = field(default_factory=list)


# An executor returns the tool's result split into (a) a short text summary placed in
# the OpenAI `tool` message (OpenAI tool results are text-only) and (b) optional image
# blocks (Anthropic-style) that get injected as a follow-up user turn so the VLM can
# actually SEE fetched cutouts. Signature: (name, args) -> (text, [anthropic_image_blocks]).
ToolExecutor = Callable[[str, dict], Awaitable[tuple[str, list[dict]]]]


async def run_tool_loop(*, system: str, user_content: list[dict], tools: list[dict],
                        execute_tool: ToolExecutor, model: str, max_turns: int = 6,
                        max_tokens: Optional[int] = None,
                        temperature: Optional[float] = None) -> LoopResult:
    """Manual OpenAI tool-calling loop mirroring the Claude Agent SDK behavior.

    The SAME repo tool functions are exposed as OpenAI tool schemas (``tools``) and run
    via ``execute_tool``. Image-returning tools (e.g. fetch_cutout) put their pixels in
    the second tuple element; those are injected as a user turn (``image_url`` parts)
    after the tool result, because OpenAI tool messages are text-only.
    """
    client = _get_client()
    temp = _temperature() if temperature is None else temperature
    max_tokens = max_tokens if max_tokens is not None else _max_tokens()
    messages: list[dict] = [
        {"role": "system", "content": system},
        {"role": "user", "content": anthropic_content_to_openai(user_content)},
    ]
    cost = 0.0
    n_tool = 0
    in_tok = out_tok = 0
    for turn in range(max_turns):
        resp = await client.chat.completions.create(
            model=model, messages=messages, tools=tools, tool_choice="auto",
            max_tokens=max_tokens, temperature=temp)
        cost += _open_cost(model, resp.usage)
        in_tok += getattr(resp.usage, "prompt_tokens", 0) or 0
        out_tok += getattr(resp.usage, "completion_tokens", 0) or 0
        msg = resp.choices[0].message
        calls = getattr(msg, "tool_calls", None) or []
        # record the assistant turn (with any tool_calls) before appending results
        messages.append({
            "role": "assistant",
            "content": msg.content or "",
            **({"tool_calls": [tc.model_dump() if hasattr(tc, "model_dump") else tc
                               for tc in calls]} if calls else {}),
        })
        if not calls:
            return LoopResult(text=msg.content or "", cost_usd=cost, num_turns=turn + 1,
                              tool_calls=n_tool, input_tokens=in_tok,
                              output_tokens=out_tok, messages=messages)
        pending_images: list[dict] = []
        for tc in calls:
            n_tool += 1
            fn = tc.function
            try:
                args = json.loads(fn.arguments) if fn.arguments else {}
            except Exception:
                args = {}
            try:
                text, images = await execute_tool(fn.name, args)
            except Exception as e:  # a tool crash must not abort the loop — feed it back as an observation
                text, images = (f"ERROR: tool {fn.name} failed: {type(e).__name__}: {e}", [])
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": text})
            pending_images.extend(images)
        if pending_images:
            messages.append({"role": "user",
                             "content": anthropic_content_to_openai(pending_images)})
    # max_turns exhausted: ask once for the final JSON with no tools
    resp = await client.chat.completions.create(
        model=model, messages=messages, max_tokens=max_tokens, temperature=temp)
    cost += _open_cost(model, resp.usage)
    final = resp.choices[0].message.content or ""
    messages.append({"role": "assistant", "content": final})
    return LoopResult(text=final, cost_usd=cost, num_turns=max_turns, tool_calls=n_tool,
                      input_tokens=in_tok, output_tokens=out_tok, messages=messages)
