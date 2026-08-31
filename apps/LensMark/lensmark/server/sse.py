"""Server-Sent Events: ``event: message\\ndata: <json>\\n\\n`` framing with an idle keepalive comment.

Proposal runs on Fable can take minutes between events, so an idle stream emits ``: keepalive`` every
``KEEPALIVE_S`` seconds - a comment line browsers ignore but which keeps proxies/EventSource alive.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncIterable, AsyncIterator, Optional

from fastapi.responses import StreamingResponse

KEEPALIVE_S = 15.0
SSE_HEADERS = {"Cache-Control": "no-store", "X-Accel-Buffering": "no"}


def format_event(data: Any, event: str = "message") -> bytes:
    """One SSE frame. ``data`` is JSON-encoded on a single line (JSON never contains raw newlines)."""
    payload = json.dumps(data, ensure_ascii=False, default=str)
    return f"event: {event}\ndata: {payload}\n\n".encode("utf-8")


def parse_events(text: str) -> list[dict[str, Any]]:
    """Inverse of :func:`format_event` for tests/CLI clients: every ``data:`` line -> parsed JSON."""
    out: list[dict[str, Any]] = []
    for line in text.splitlines():
        if line.startswith("data:"):
            out.append(json.loads(line[5:].strip()))
    return out


async def with_keepalive(events: AsyncIterable[Any], keepalive_s: float = KEEPALIVE_S) -> AsyncIterator[bytes]:
    """Frame ``events`` as SSE bytes; when no event arrives within ``keepalive_s`` emit a comment.

    The pending ``__anext__`` is kept across timeouts (never cancelled by the timer) so a slow
    producer is not interrupted mid-step.
    """
    it = events.__aiter__()
    pending: Optional[asyncio.Future] = None
    try:
        while True:
            if pending is None:
                pending = asyncio.ensure_future(it.__anext__())
            done, _ = await asyncio.wait({pending}, timeout=keepalive_s)
            if not done:
                yield b": keepalive\n\n"
                continue
            fut, pending = pending, None
            try:
                ev = fut.result()
            except StopAsyncIteration:
                return
            yield format_event(ev)
    finally:
        if pending is not None and not pending.done():
            pending.cancel()
        try:
            if pending is not None:
                await asyncio.wait({pending}, timeout=1.0)
            aclose = getattr(it, "aclose", None)
            if aclose is not None:
                await aclose()
        except BaseException:  # best-effort cleanup on client disconnect
            pass


def sse_response(events: AsyncIterable[Any], *, keepalive_s: float = KEEPALIVE_S) -> StreamingResponse:
    """``text/event-stream`` StreamingResponse over an async iterator of JSON-able event dicts."""
    return StreamingResponse(with_keepalive(events, keepalive_s), media_type="text/event-stream",
                             headers=dict(SSE_HEADERS))
