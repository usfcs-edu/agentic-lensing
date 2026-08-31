"""In-memory registry of background proposal runs: one asyncio task each, with a replayable event log.

``POST /api/propose/{id}`` must answer 202 with a ``run_id`` synchronously, so the registry mints the
id, starts the task and returns; the SSE route replays the recorded events and then follows the run
live until a terminal ``done``/``error`` event. ``run_propose`` is offered the registry id via an
optional ``run_id`` kwarg; if it does not accept one, the ``ProposalRun`` it returns is kept alongside
(``RunState.result`` / ``propose_run_id``) so both ids resolve.
"""
from __future__ import annotations

import asyncio
import importlib
import inspect
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Callable, Optional

from ..model import now_iso

TERMINAL_PHASES = ("done", "error")
PHASES = ("queued", "started", "thinking", "partial", "tool", "validated", "done", "error")


def new_run_id() -> str:
    return "run-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]


@dataclass
class RunState:
    run_id: str
    image_id: str
    engine: str
    request: Any = None
    created_at: str = field(default_factory=now_iso)
    events: list[dict[str, Any]] = field(default_factory=list)
    task: Optional[asyncio.Task] = None
    result: Any = None                                   # the ProposalRun returned by run_propose
    finished: bool = False
    finished_at: Optional[str] = None
    pending_final: Optional[dict[str, Any]] = None       # done/error emitted by run_propose, published at task end
    wakeup: asyncio.Event = field(default_factory=asyncio.Event)
    loop: Optional[asyncio.AbstractEventLoop] = None
    thread_id: Optional[int] = None

    @property
    def phase(self) -> str:
        return str(self.events[-1].get("phase")) if self.events else "queued"

    @property
    def propose_run_id(self) -> Optional[str]:
        r = self.result
        if r is None:
            return None
        return getattr(r, "run_id", None) if not isinstance(r, dict) else r.get("run_id")

    def snapshot(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id, "image_id": self.image_id, "engine": self.engine,
            "created_at": self.created_at, "finished": self.finished, "finished_at": self.finished_at,
            "phase": self.phase, "n_events": len(self.events), "propose_run_id": self.propose_run_id,
            "run": _dump(self.result),
        }


def _dump(run: Any) -> Optional[dict[str, Any]]:
    if run is None:
        return None
    if hasattr(run, "model_dump"):
        return run.model_dump(mode="json", exclude_none=True)
    return dict(run) if isinstance(run, dict) else {"value": str(run)}


class RunRegistry:
    def __init__(self) -> None:
        self._runs: dict[str, RunState] = {}

    # ------------------------------------------------------------------ lookup
    def get(self, run_id: str) -> Optional[RunState]:
        st = self._runs.get(run_id)
        if st is None:  # also accept the id the propose module minted
            st = next((s for s in self._runs.values() if s.propose_run_id == run_id), None)
        return st

    def __contains__(self, run_id: str) -> bool:
        return self.get(run_id) is not None

    def __len__(self) -> int:
        return len(self._runs)

    def list(self, image_id: Optional[str] = None) -> list[dict[str, Any]]:
        return [s.snapshot() for s in self._runs.values() if image_id is None or s.image_id == image_id]

    # ------------------------------------------------------------------ start / cancel
    def start(self, campaign: Any, image_id: str, req: Any, engine_name: str) -> str:
        """Create the task on the running loop and return the registry run id immediately."""
        loop = asyncio.get_running_loop()
        st = RunState(run_id=new_run_id(), image_id=image_id, engine=engine_name, request=req,
                      loop=loop, thread_id=threading.get_ident())
        self._runs[st.run_id] = st
        self._push(st, {"phase": "queued", "detail": f"queued {image_id} on engine {engine_name}"})
        st.task = loop.create_task(self._run(st, campaign, image_id, req, engine_name), name=f"propose:{st.run_id}")
        return st.run_id

    def cancel(self, run_id: str) -> bool:
        """Cancel a running task and record ``error: cancelled``. False if it had already finished."""
        st = self.get(run_id)
        if st is None:
            raise KeyError(run_id)
        if st.finished:
            return False
        if st.task is not None and not st.task.done():
            st.task.cancel()
        self._finish(st, phase="error", detail="cancelled")
        return True

    async def shutdown(self) -> None:
        for st in list(self._runs.values()):
            if not st.finished:
                self.cancel(st.run_id)
        tasks = [s.task for s in self._runs.values() if s.task is not None and not s.task.done()]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    # ------------------------------------------------------------------ events
    async def events(self, run_id: str) -> AsyncIterator[dict[str, Any]]:
        """Replay recorded events, then follow live; stops after the terminal event."""
        st = self.get(run_id)
        if st is None:
            raise KeyError(run_id)
        i = 0
        while True:
            while i < len(st.events):
                ev = st.events[i]
                i += 1
                yield ev
                if ev.get("phase") in TERMINAL_PHASES:
                    return
            if st.finished:  # defensive: finished without a terminal event
                return
            await st.wakeup.wait()

    # ------------------------------------------------------------------ internals
    def _push(self, st: RunState, ev: dict[str, Any]) -> None:
        if st.loop is not None and st.thread_id is not None and threading.get_ident() != st.thread_id:
            st.loop.call_soon_threadsafe(self._push, st, ev)   # engine callbacks from a worker thread
            return
        ev = dict(ev)
        ev.setdefault("phase", "partial")
        ev.setdefault("detail", "")
        st.events.append(ev)
        old, st.wakeup = st.wakeup, asyncio.Event()
        old.set()

    def _callback(self, st: RunState) -> Callable[[dict[str, Any]], None]:
        def on_event(ev: Any) -> None:
            if not isinstance(ev, dict):
                ev = {"phase": "partial", "detail": str(ev)}
            if ev.get("phase") in TERMINAL_PHASES:
                st.pending_final = dict(ev)      # merged with the ProposalRun when the task returns
                return
            self._push(st, ev)
        return on_event

    def _finish(self, st: RunState, *, run: Any = None, phase: Optional[str] = None,
                detail: Optional[str] = None) -> None:
        if run is not None and st.result is None:
            st.result = run
        if st.finished:
            return
        ev = dict(st.pending_final or {})
        if run is not None:
            err = getattr(run, "error", None) if not isinstance(run, dict) else run.get("error")
            n_items = getattr(run, "n_items_proposed", None) if not isinstance(run, dict) else run.get("n_items_proposed")
            cost = getattr(run, "cost_usd", None) if not isinstance(run, dict) else run.get("cost_usd")
            ev.setdefault("phase", "error" if err else "done")
            ev.setdefault("detail", err or f"{n_items or 0} items proposed")
            ev["run"] = _dump(run)
            if cost is not None:
                ev.setdefault("cost_usd", cost)
            if n_items is not None:
                ev.setdefault("n_items", n_items)
        if phase:
            ev["phase"] = phase
        if detail:
            ev["detail"] = detail
        ev.setdefault("phase", "done")
        if ev["phase"] not in TERMINAL_PHASES:
            ev["phase"] = "done"
        st.finished = True
        st.finished_at = now_iso()
        self._push(st, ev)

    async def _run(self, st: RunState, campaign: Any, image_id: str, req: Any, engine_name: str) -> None:
        try:
            try:
                propose = importlib.import_module("lensmark.claude.propose")
            except ImportError as e:
                self._finish(st, phase="error", detail=f"module not available: lensmark.claude.propose ({e})")
                return
            engine = None
            try:
                from ..claude.engine import get_engine
                engine = get_engine(engine_name)
            except ImportError:
                engine = None                 # run_propose resolves the engine from $LENSMARK_ENGINE
            kwargs: dict[str, Any] = {}
            try:
                if "run_id" in inspect.signature(propose.run_propose).parameters:
                    kwargs["run_id"] = st.run_id
            except (TypeError, ValueError):
                pass
            run = await propose.run_propose(campaign, image_id, req, engine=engine,
                                            on_event=self._callback(st), **kwargs)
            self._finish(st, run=run)
        except asyncio.CancelledError:
            self._finish(st, phase="error", detail="cancelled")
            raise
        except Exception as e:  # noqa: BLE001 - every failure becomes an SSE error event
            self._finish(st, phase="error", detail=f"{type(e).__name__}: {e}")
