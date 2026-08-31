"""Claude engines.

``SdkEngine`` drives ``claude_agent_sdk.query()`` over the logged-in ``claude`` CLI (no API key);
``FixtureEngine`` replays canned JSON from ``tests/fixtures`` (all UI QA runs on it). Both satisfy
``engine_base.Engine``. Port of reproductions/lensjudge/imaging/grader_lean.py:64-90 (``_collect``)
and :139-151 (options) with the plan's fixes: full model ids, ``setting_sources=[]`` (isolation from
~/.claude/settings.json, whose ``effortLevel: xhigh`` otherwise leaks), ``cli_path`` pinned to the
PATH binary, JSON-schema structured output, streaming-input content blocks (images).

claude-agent-sdk 0.2.148 facts relied on (paths under site-packages/claude_agent_sdk/):
* ``ClaudeAgentOptions`` - types.py:1941 (class), :2035 model, :2294 effort, :2281 thinking,
  :2309 output_format, :2216 setting_sources ("When None, all sources are loaded"), :2063 cli_path,
  :2135 include_partial_messages, :2116 hooks, :2060 cwd, :2015 max_turns, :2021 max_budget_usd,
  :1944 tools ("[] disables all built-in tools"), :1967 system_prompt.
* CLI flags - _internal/transport/subprocess_cli.py:562-785 ``_build_command``: --model :613,
  --effort :768, --json-schema :779, --setting-sources= :720, --thinking / --thinking-display
  :752/:761, --include-partial-messages :685, --tools "" :586, --permission-mode :627,
  --max-turns :601, --max-budget-usd :604, --input-format stream-json :783.
* ``_find_cli`` prefers the *bundled* binary - subprocess_cli.py:247-252 and :333-344 - which is why
  ``cli_path=config.claude_bin()`` is always passed.
* Version probe ``<cli> -v`` before spawning - subprocess_cli.py:799-800, :1144-1161.
* Streaming user-message shape - query.py:47-53 and _internal/client.py:176-186.
* Message shapes - _internal/message_parser.py: system :226-302, assistant :151-220 (blocks
  :168-208), result :308-341 (``structured_output`` :322), stream_event :343-354.
* An ``is_error`` result is followed by a ``ResultError`` raised from the iterator once the CLI exits
  non-zero - _internal/query.py:384-387 and :405-436; the ResultMessage is yielded first, so the
  error text is taken from the result and the trailing exception is folded into ``raw``.
* Initialize handshake - _internal/query.py:231-283 (request), :598-627 (wire format),
  :318-330 (control_response routing); hooks keep stdin open until the result (:819-861).
"""
from __future__ import annotations

import asyncio
import functools
import json
import os
import platform
import shutil
import subprocess
import time
from collections import deque
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional

from .. import config
from . import parse, trace
from .engine_base import Engine, EngineRequest, EngineResult, EventCallback

FIXTURE_SUBDIRS = {"propose": "proposals", "patch": "patches", "doctor": "doctor"}
DEFAULT_FIXTURE_DIR = config.PKG.parent / "tests" / "fixtures"
THINKING: dict[str, str] = {"type": "adaptive", "display": "summarized"}
DOCTOR_SCHEMA: dict[str, Any] = {"type": "object", "properties": {"ok": {"type": "boolean"}},
                                 "required": ["ok"], "additionalProperties": False}
FIXTURE_COST_USD = 0.0123


def _emit(cb: Optional[EventCallback], phase: str, detail: str = "", **extra: Any) -> None:
    if cb is not None:
        cb({"phase": phase, "detail": detail, **extra})


# ----------------------------------------------------------------------------- SDK engine
class SdkEngine:
    """One ``query()`` per call: a single user turn of content blocks, structured JSON back."""

    name = "sdk"

    def __init__(self, transport: Any = None):
        self._transport = transport          # test injection: query(transport=...)

    def build_options(self, req: EngineRequest, hooks: Optional[dict] = None, stderr: Any = None):
        """The ``ClaudeAgentOptions`` for ``req`` (see the module docstring for what each becomes)."""
        from claude_agent_sdk import ClaudeAgentOptions
        effort = req.effort if (req.effort and config.model_supports_effort(req.model)) else None
        return ClaudeAgentOptions(
            stderr=stderr,                   # capture the CLI's stderr tail: the real cause of a ProcessError
            model=req.model,
            effort=effort,
            thinking=dict(THINKING),
            output_format={"type": "json_schema", "schema": req.schema},
            system_prompt=req.system,
            setting_sources=[],              # NEVER None: None loads ~/.claude settings (effortLevel etc.)
            tools=[],
            permission_mode="bypassPermissions",
            max_turns=req.max_turns,
            max_budget_usd=req.max_budget_usd,
            include_partial_messages=True,
            cli_path=config.claude_bin(),
            cwd=str(req.cwd) if req.cwd else None,
            hooks=hooks,
        )

    @staticmethod
    def _result_error(msg: Any) -> str:
        """Most informative text of an ``is_error`` result (mirrors _internal/query.py:55-80)."""
        errs = [str(e) for e in (msg.errors or []) if e]
        if errs:
            return "; ".join(errs)
        if isinstance(msg.result, str) and msg.result.strip():
            return msg.result.strip()
        if msg.subtype and msg.subtype != "success":
            return str(msg.subtype)
        if msg.api_error_status is not None:
            return f"API error (HTTP {msg.api_error_status})"
        return "unknown error"

    @staticmethod
    def _stream_event(ev: dict[str, Any], cb: Optional[EventCallback]) -> None:
        t = ev.get("type")
        if t == "content_block_delta":
            d = ev.get("delta") or {}
            if d.get("type") == "thinking_delta":
                _emit(cb, "thinking", "", text=d.get("thinking", ""))
            elif d.get("type") == "text_delta":
                _emit(cb, "partial", "", text=d.get("text", ""))
        elif t == "content_block_start":
            blk = ev.get("content_block") or {}
            if blk.get("type") == "tool_use":
                _emit(cb, "tool", str(blk.get("name", "")))

    async def run(self, req: EngineRequest, on_event: Optional[EventCallback] = None, *,
                  transport: Any = None) -> EngineResult:
        from claude_agent_sdk import (AssistantMessage, ResultMessage, StreamEvent, SystemMessage,
                                      TextBlock, ThinkingBlock, ToolUseBlock, query)
        tr = trace.for_request(req.purpose, req.fixture_key)
        stderr_tail: deque[str] = deque(maxlen=30)
        opts = self.build_options(req, hooks=tr.hooks() if tr else None, stderr=stderr_tail.append)
        res = EngineResult(structured=None, model=req.model, effort=opts.effort, raw={})
        if tr:
            res.raw["trace_path"] = str(tr.path)
            tr.write("request", model=req.model, effort=opts.effort, purpose=req.purpose, key=req.fixture_key,
                     cwd=str(req.cwd) if req.cwd else None, max_turns=req.max_turns,
                     max_budget_usd=req.max_budget_usd, content=trace.elide_images(req.content))

        async def prompt():
            yield {"type": "user", "message": {"role": "user", "content": req.content},
                   "parent_tool_use_id": None, "session_id": ""}

        texts: list[str] = []
        thinking: list[str] = []
        result_text: Optional[str] = None
        saw_stream = False
        t0 = time.monotonic()
        try:
            async for msg in query(prompt=prompt(), options=opts, transport=transport or self._transport):
                if isinstance(msg, SystemMessage):
                    if msg.subtype == "init":
                        res.raw["session_id"] = msg.data.get("session_id")
                        res.raw["model_reported"] = msg.data.get("model")
                        _emit(on_event, "started", f"session {msg.data.get('session_id')} model {msg.data.get('model')}")
                elif isinstance(msg, StreamEvent):
                    saw_stream = True
                    self._stream_event(msg.event or {}, on_event)
                elif isinstance(msg, AssistantMessage):
                    res.raw["model_reported"] = msg.model
                    for b in msg.content:
                        if isinstance(b, ThinkingBlock):
                            thinking.append(b.thinking)
                            if tr:
                                tr.write("thinking", text=b.thinking)
                            if not saw_stream:
                                _emit(on_event, "thinking", "", text=b.thinking)
                        elif isinstance(b, TextBlock):
                            texts.append(b.text)
                            if tr:
                                tr.write("assistant_text", text=b.text)
                            if not saw_stream:
                                _emit(on_event, "partial", "", text=b.text)
                        elif isinstance(b, ToolUseBlock):      # e.g. the CLI's StructuredOutput tool
                            if tr:
                                tr.write("tool_use", tool=b.name, tool_use_id=b.id, input=trace._trim(b.input))
                            if not saw_stream:
                                _emit(on_event, "tool", b.name)
                elif isinstance(msg, ResultMessage):
                    res.cost_usd = msg.total_cost_usd
                    res.usage = dict(msg.usage or {})
                    res.num_turns = msg.num_turns
                    res.raw.update(session_id=msg.session_id, stop_reason=msg.stop_reason, subtype=msg.subtype,
                                   terminal_reason=msg.terminal_reason, model_usage=msg.model_usage,
                                   api_error_status=msg.api_error_status, duration_api_ms=msg.duration_api_ms)
                    if isinstance(msg.structured_output, dict):
                        res.structured = deepcopy(msg.structured_output)
                    if isinstance(msg.result, str) and msg.result:
                        result_text = msg.result
                    if msg.is_error:
                        res.error = self._result_error(msg)
                        _emit(on_event, "error", res.error, cost_usd=res.cost_usd)
                    else:
                        _emit(on_event, "done", f"result {msg.subtype}", cost_usd=res.cost_usd)
        except asyncio.CancelledError:
            if tr:
                tr.write("cancelled")
            raise                                   # query()'s finally terminates the subprocess
        except Exception as e:                       # model/CLI failures never raise
            err = f"{type(e).__name__}: {e}"
            tail = [l.strip() for l in stderr_tail if l.strip() and not l.startswith("Fatal error in message reader")]
            if tail and res.error is None:
                err = f"{err} | stderr: {' | '.join(tail[-5:])}"
            if res.error is None:
                res.error = err
                _emit(on_event, "error", err)
            else:
                res.raw["exception"] = err          # the trailing ResultError after an is_error result
        res.duration_s = round(time.monotonic() - t0, 3)
        if stderr_tail:
            res.raw["stderr_tail"] = list(stderr_tail)
        res.text = result_text if result_text else (texts[-1] if texts else "")
        res.thinking = "\n\n".join(thinking)
        if res.structured is None and res.error is None:
            res.structured = parse.extract_json_block(res.text)
            if res.structured is None and texts and texts[-1] != res.text:
                res.structured = parse.extract_json_block(texts[-1])
        if tr:
            tr.write("result", cost_usd=res.cost_usd, num_turns=res.num_turns, duration_s=res.duration_s,
                     structured=res.structured is not None, error=res.error, usage=res.usage)
        return res


# ----------------------------------------------------------------------------- fixture engine
class FixtureEngine:
    """Canned proposals from ``tests/fixtures/<purpose>s/<key>.json`` (else ``default.json``)."""

    name = "fixture"

    def __init__(self, fixture_dir: str | Path | None = None, delay: Optional[float] = None):
        self._dir = fixture_dir
        self._delay = delay

    @property
    def fixture_dir(self) -> Path:
        return Path(self._dir or os.environ.get("LENSMARK_FIXTURE_DIR") or DEFAULT_FIXTURE_DIR).expanduser()

    def resolve(self, req: EngineRequest) -> Optional[Path]:
        sub = self.fixture_dir / FIXTURE_SUBDIRS.get(req.purpose, req.purpose + "s")
        cands = ([sub / f"{req.fixture_key}.json"] if req.fixture_key else []) + [sub / "default.json"]
        return next((p for p in cands if p.is_file()), None)

    async def run(self, req: EngineRequest, on_event: Optional[EventCallback] = None) -> EngineResult:
        delay = self._delay if self._delay is not None else float(os.environ.get("LENSMARK_FIXTURE_DELAY", "0.2"))
        t0 = time.monotonic()
        path = self.resolve(req)
        if path is None:
            err = f"no fixture for purpose {req.purpose!r} key {req.fixture_key!r} under {self.fixture_dir}"
            _emit(on_event, "error", err)
            return EngineResult(structured=None, error=err, model=req.model, effort=req.effort,
                                duration_s=round(time.monotonic() - t0, 3), raw={"fixture": None})
        doc = json.loads(path.read_text(encoding="utf-8"))
        text = json.dumps(doc, ensure_ascii=False)
        think = ("Locating the deflector, arcs and field objects on the grid image and reading u/v for each "
                 f"item. (fixture engine: {path.name})")
        _emit(on_event, "started", f"fixture {path.name} model {req.model}")
        await asyncio.sleep(delay)
        _emit(on_event, "thinking", "reading the grid image (canned)", text=think)
        await asyncio.sleep(delay)
        _emit(on_event, "partial", "streaming (canned)", text=text[:160])
        await asyncio.sleep(delay)
        _emit(on_event, "done", f"result success (fixture {path.name})", cost_usd=FIXTURE_COST_USD)
        return EngineResult(
            structured=deepcopy(doc), text=text, thinking=think, cost_usd=FIXTURE_COST_USD,
            usage={"input_tokens": 3210, "output_tokens": 640, "cache_creation_input_tokens": 0,
                   "cache_read_input_tokens": 0},
            num_turns=1, duration_s=round(time.monotonic() - t0, 3), model=req.model, effort=req.effort,
            raw={"fixture": str(path), "session_id": f"fixture-{path.stem}", "subtype": "success",
                 "stop_reason": "end_turn"})


# ----------------------------------------------------------------------------- factory / doctor
def get_engine(name: Optional[str] = None) -> Engine:
    name = name or config.engine_name()
    if name == "sdk":
        return SdkEngine()
    if name == "fixture":
        return FixtureEngine()
    raise ValueError(f"unknown engine {name!r} (sdk|fixture)")


@functools.lru_cache(maxsize=16)
def _version_of(binary: str) -> Optional[str]:
    try:
        out = subprocess.run([binary, "--version"], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    lines = (out.stdout or out.stderr or "").strip().splitlines()
    return lines[0].strip() if lines else None


def claude_version(binary: Optional[str] = None) -> Optional[str]:
    """``<claude_bin> --version`` (first line), cached per binary; None when unavailable."""
    b = binary or config.claude_bin()
    return _version_of(b) if b else None


def bundled_cli_path() -> Optional[str]:
    """The Agent SDK's bundled ``claude`` binary, if the wheel ships one."""
    try:
        import claude_agent_sdk
    except ImportError:  # pragma: no cover
        return None
    name = "claude.exe" if platform.system() == "Windows" else "claude"
    p = Path(claude_agent_sdk.__file__).resolve().parent / "_bundled" / name
    return str(p) if p.is_file() else None


def cli_doctor(dir: Optional[str] = None, *, call: bool = True, budget: float = 0.10) -> int:
    """Print how the engine resolves (SDK, binaries, versions, env) and run one cheap real turn."""
    try:
        import claude_agent_sdk
        sdk_ver, sdk_where = claude_agent_sdk.__version__, str(Path(claude_agent_sdk.__file__).parent)
    except ImportError:  # pragma: no cover
        sdk_ver, sdk_where = "(not installed)", ""
    path_bin = shutil.which("claude")
    bundled = bundled_cli_path()
    used = config.claude_bin()
    print(f"claude-agent-sdk       {sdk_ver}  {sdk_where}")
    print(f"PATH claude            {path_bin or '(not found)'}  {claude_version(path_bin) or '' if path_bin else ''}")
    print(f"bundled cli            {bundled or '(none)'}  {claude_version(bundled) or '' if bundled else ''}")
    print(f"engine binary          {used or '(none)'}  (LENSMARK_CLAUDE_BIN={os.environ.get('LENSMARK_CLAUDE_BIN') or '-'})")
    print(f"LENSMARK_ENGINE        {config.engine_name()}")
    print(f"LENSMARK_TRACE_DIR     {os.environ.get('LENSMARK_TRACE_DIR') or '-'}")
    print(f"LENSMARK_MAX_BUDGET_USD {config.max_budget_usd()}")
    if not call:
        return 0
    if not used:
        print("no claude binary resolved; cannot run the smoke turn")
        return 1
    model = config.resolve_model("sonnet")
    req = EngineRequest(
        system='You are a connectivity check for an annotation tool. Reply with the JSON object {"ok": true}.',
        content=[{"type": "text", "text": 'Health check: respond with {"ok": true} and nothing else.'}],
        schema=DOCTOR_SCHEMA, model=model, effort="low", max_budget_usd=budget, max_turns=1,
        cwd=Path(dir).expanduser() if dir else None, purpose="doctor")
    print(f"smoke turn             model={model} effort=low max_budget_usd={budget}")

    def _ev(e: dict[str, Any]) -> None:
        if e.get("phase") in ("started", "done", "error", "tool"):
            print(f"  [{e['phase']}] {e.get('detail', '')}")

    res = asyncio.run(SdkEngine().run(req, on_event=_ev))
    print(f"  structured           {json.dumps(res.structured)}")
    print(f"  cost_usd             {res.cost_usd}")
    print(f"  usage                {json.dumps(res.usage)}")
    print(f"  model reported       {res.raw.get('model_reported')}")
    print(f"  num_turns/duration_s {res.num_turns} / {res.duration_s}")
    if res.error:
        print(f"  error                {res.error}")
        return 1
    return 0 if isinstance(res.structured, dict) and res.structured.get("ok") is True else 1
