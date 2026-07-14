"""common/llm_client.py — backend-selectable LLM transport for LensJudge v4.

LensJudge historically called Anthropic Claude only (``grader_direct`` via the raw
Messages API; the agentic graders via the Claude Agent SDK). v4 makes the backend
**selectable** so the same graders can run on OPEN-WEIGHT models served behind any
OpenAI-compatible endpoint (vLLM / SGLang on GPUs, MLX-VLM / LM Studio / llama.cpp /
Ollama on the Mac), fully offline.

Design (Phase F, v5): the OPEN-WEIGHT backend is the DEFAULT; the Claude engine is the
retained, explicitly-selected option (``LENSJUDGE_BACKEND=anthropic`` or its alias
``claude``). The schema (``common.schemas.ImageGrade``) and validator
(``common.parse.parse_model``) are provider-agnostic and reused verbatim — only the
transport differs.

Env switches (read at call time so run scripts can set os.environ from CLI flags):
  LENSJUDGE_BACKEND      openai (default) | anthropic (alias: claude)
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
import math
import os
import re
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

# NOTE: both provider clients are imported LAZILY inside their client paths, so
# neither `openai` nor `anthropic` is a hard dependency of the other backend.

ANTHROPIC = "anthropic"
OPENAI = "openai"


def get_backend() -> str:
    """Return the active backend: 'openai' (default) or 'anthropic' ('claude' is an alias)."""
    b = os.environ.get("LENSJUDGE_BACKEND", OPENAI).strip().lower()
    if b == "claude":
        b = ANTHROPIC
    if b not in (ANTHROPIC, OPENAI):
        raise ValueError(f"LENSJUDGE_BACKEND must be {OPENAI!r}, {ANTHROPIC!r} or 'claude', got {b!r}")
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


# --- structured-outputs shim (vLLM 0.24 deprecated `guided_json` -> `structured_outputs`) ----
_STRUCTURED_RESOLVED: Optional[str] = None   # cached auto-probe result for this process


def _server_version(base_url: str) -> Optional[tuple]:
    """Best-effort vLLM version probe: GET <root>/version (the endpoint lives at the server root,
    not under /v1). Returns (major, minor) or None (non-vLLM server / unreachable / parse fail)."""
    try:
        import requests
        root = (base_url or "").rstrip("/")
        if root.endswith("/v1"):
            root = root[:-3].rstrip("/")
        if not root:
            return None
        r = requests.get(root + "/version", timeout=3)
        parts = tuple(int(x) for x in str(r.json().get("version", "")).split(".")[:2])
        return parts if len(parts) == 2 else None
    except Exception:
        return None


def _structured_mode() -> str:
    """Which structured-output request key to send: 'legacy' (guided_json, vLLM <= 0.23) or 'new'
    (structured_outputs, vLLM >= 0.24 where guided_json is deprecated).

    LENSJUDGE_STRUCTURED = auto (default) | new | legacy. auto probes the server /version ONCE per
    process and falls back to legacy when the probe fails (older vLLM accepts legacy; 0.24 still
    accepts it too, only deprecated — so legacy is the safe default)."""
    global _STRUCTURED_RESOLVED
    mode = os.environ.get("LENSJUDGE_STRUCTURED", "auto").strip().lower()
    if mode in ("new", "legacy"):
        return mode
    if _STRUCTURED_RESOLVED is None:
        v = _server_version(os.environ.get("LENSJUDGE_BASE_URL", ""))
        _STRUCTURED_RESOLVED = "new" if (v is not None and v >= (0, 24)) else "legacy"
    return _STRUCTURED_RESOLVED


# --- grade-token logprob scoring (v5: generated p_lens floats are miscalibrated on faint rare
# objects; score from the token distribution at the `"grade":"X"` position instead) -------------
def _want_logprobs() -> bool:
    """Request per-token logprobs on open-backend completions (default ON; LENSJUDGE_LOGPROBS=0 off)."""
    return os.environ.get("LENSJUDGE_LOGPROBS", "1") == "1"


def _lp_kwargs() -> dict:
    return {"logprobs": True, "top_logprobs": 10} if _want_logprobs() else {}


def _f(obj: Any, name: str, default=None):
    """Field access across SDK objects and plain dicts (tests / raw JSON)."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


_GRADE_KEY_RE = re.compile(r'"grade"\s*:\s*"')


def extract_grade_probs(logprobs_content) -> Optional[dict]:
    """P(letter) for A/B/C/D at the ImageGrade ``"grade":"X"`` position.

    Walks the chat-completion logprobs content, finds the token holding the grade letter (the char
    right after the first `"grade":"` in the concatenated token text), and sums exp(logprob) over
    the top_logprobs entries whose token starts with each letter (handles `A` vs `"A` tokenizations).
    Returns e.g. {"A": 0.71, "B": 0.22, "D": 0.01} (unnormalized true probabilities, each capped at
    1.0), or None when no grade position / no letter mass is found."""
    try:
        items = list(logprobs_content or [])
    except TypeError:
        return None
    if not items:
        return None
    texts = [str(_f(t, "token") or "") for t in items]
    m = _GRADE_KEY_RE.search("".join(texts))
    if not m:
        return None
    pos, off, idx = m.end(), 0, None
    for i, s in enumerate(texts):
        if off <= pos < off + len(s):
            idx = i
            break
        off += len(s)
    if idx is None:
        return None
    probs: dict = {}
    for cand in (_f(items[idx], "top_logprobs") or []):
        ttxt = str(_f(cand, "token") or "").lstrip().lstrip('"')
        lp = _f(cand, "logprob")
        if ttxt and ttxt[0] in "ABCD" and lp is not None:
            letter = ttxt[0]
            probs[letter] = min(1.0, probs.get(letter, 0.0) + math.exp(float(lp)))
    if not probs:   # top_logprobs absent -> at least use the sampled token's own logprob
        ttxt = texts[idx].lstrip().lstrip('"')
        lp = _f(items[idx], "logprob")
        if ttxt and ttxt[0] in "ABCD" and lp is not None:
            probs[ttxt[0]] = min(1.0, math.exp(float(lp)))
    return {k: round(v, 6) for k, v in probs.items()} or None


def _grade_probs_from_choice(choice: Any) -> Optional[dict]:
    lp = getattr(choice, "logprobs", None)
    if lp is None:
        return None
    return extract_grade_probs(_f(lp, "content"))


def logprob_p_lens(grade_probs: Optional[dict]) -> Optional[float]:
    """Uncalibrated detection score from the grade-token distribution: P(A) + P(B).
    Calibration (isotonic/Platt per backend+survey) happens downstream in eval/calibrate."""
    if not grade_probs:
        return None
    return round(min(1.0, grade_probs.get("A", 0.0) + grade_probs.get("B", 0.0)), 4)


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
    grade_probs: Optional[dict] = None   # P(A/B/C/D) at the grade token (open backend, logprobs on)
    raw: Any = None


def _maybe_json_kwargs(json_schema: Optional[dict] = None) -> dict:
    """Optional, server-dependent request kwargs, gated by env (all default off).

    LENSJUDGE_JSON_MODE   -> response_format={"type":"json_object"}
    LENSJUDGE_GUIDED_JSON -> extra_body.guided_json (vLLM/SGLang constrained decoding)
    LENSJUDGE_NOTHINK     -> extra_body.chat_template_kwargs.enable_thinking=False. Reasoning VLMs
        (GLM-4.6V) emit long <think> blocks that, in an agentic loop, never terminate with the final
        JSON (more output tokens => more rambling, WORSE parse rate). Disabling thinking makes them
        answer directly. No-op on servers/models that ignore the flag.
    """
    kw: dict = {}
    eb: dict = {}
    if os.environ.get("LENSJUDGE_JSON_MODE") == "1":
        kw["response_format"] = {"type": "json_object"}
    if os.environ.get("LENSJUDGE_GUIDED_JSON") == "1" and json_schema is not None:
        # vLLM 0.24 deprecated guided_json in favor of structured_outputs (see _structured_mode)
        if _structured_mode() == "new":
            eb["structured_outputs"] = {"json": json_schema}
        else:
            eb["guided_json"] = json_schema
    if os.environ.get("LENSJUDGE_NOTHINK") == "1":
        eb["chat_template_kwargs"] = {"enable_thinking": False}
    if eb:
        kw["extra_body"] = eb
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
    kw = {**_maybe_json_kwargs(json_schema), **_lp_kwargs()}
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
        temperature=_temperature() if temperature is None else temperature, **_lp_kwargs())
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
        model=model, grade_probs=_grade_probs_from_choice(choice), raw=resp)


# --- agentic tool-calling loop (wired into the graders in Phase 3) ----------
@dataclass
class LoopResult:
    text: str
    cost_usd: float = 0.0
    num_turns: int = 0
    tool_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    grade_probs: Optional[dict] = None   # P(A/B/C/D) at the grade token of the FINAL reply
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
            max_tokens=max_tokens, temperature=temp,
            **_maybe_json_kwargs(), **_lp_kwargs())
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
                              output_tokens=out_tok,
                              grade_probs=_grade_probs_from_choice(resp.choices[0]),
                              messages=messages)
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
        model=model, messages=messages, max_tokens=max_tokens, temperature=temp,
        **_maybe_json_kwargs(), **_lp_kwargs())
    cost += _open_cost(model, resp.usage)
    final = resp.choices[0].message.content or ""
    messages.append({"role": "assistant", "content": final})
    return LoopResult(text=final, cost_usd=cost, num_turns=max_turns, tool_calls=n_tool,
                      input_tokens=in_tok, output_tokens=out_tok,
                      grade_probs=_grade_probs_from_choice(resp.choices[0]), messages=messages)
