"""FastAPI application factory (``create_app``) and ``serve()`` (uvicorn + browser).

Cross-cutting behaviour: every response carries ``Cache-Control: no-store`` (a pure-ASGI middleware,
after reproductions/lensjudge/golden/tool/serve.py:84-91), every error is JSON ``{"error", "detail"}``,
pydantic ``ValidationError`` -> 422 with the error list, and the static front end (``lensmark/static``)
is mounted last so the ``/api`` routes always win.
"""
from __future__ import annotations

import json
import os
import threading
import webbrowser
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Optional

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError
from starlette.datastructures import MutableHeaders
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from .. import __version__, config
from ..model import Critique, LensMarkFile, Patch
from ..store import Campaign
from . import ModuleUnavailable, api_annotations, api_critique, api_export, api_images, api_propose, api_voice, \
    campaign_of, optional_module
from .runs import RunRegistry

_UNSET = object()
SCHEMA_FILES = {
    "lensmark": "lensmark-1.0.schema.json",
    "proposal": "lensmark-proposal-1.0.schema.json",
    "critique": "lensmark-critique-1.0.schema.json",
    "patch": "lensmark-patch-1.0.schema.json",
}

PLACEHOLDER_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>LensMark</title>
<style>body{font-family:system-ui,sans-serif;max-width:48rem;margin:3rem auto;padding:0 1rem;color:#ddd;background:#14161a}
a{color:#7fd3ff}code{background:#22262c;padding:.1em .3em;border-radius:3px}</style></head>
<body><h1>LensMark</h1>
<p>The front end is not built yet: run <code>cd frontend &amp;&amp; npm install &amp;&amp; npm run build</code>
(output goes to <code>lensmark/static/</code>) and reload this page.</p>
<p>The API is up: <a href="/api/health">/api/health</a> &middot; <a href="/api/images">/api/images</a> &middot;
<a href="/api/models">/api/models</a> &middot; <a href="/api/style">/api/style</a> &middot;
<a href="/api/config">/api/config</a> &middot; <a href="/api/docs">/api/docs</a></p>
</body></html>
"""


# ----------------------------------------------------------------------------- middleware
class NoStoreMiddleware:
    """Pure ASGI: add ``Cache-Control: no-store`` to every HTTP response (static files + SSE included)."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_no_store(message: Message) -> None:
            if message["type"] == "http.response.start":
                message.setdefault("headers", [])
                MutableHeaders(scope=message)["Cache-Control"] = "no-store"
            await send(message)

        await self.app(scope, receive, send_no_store)


# ----------------------------------------------------------------------------- errors
def _error(status: int, error: str, detail: Any = None) -> JSONResponse:
    return JSONResponse({"error": error, "detail": jsonable_encoder(detail)}, status_code=status)


def _install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ModuleUnavailable)
    async def _module_unavailable(request: Request, exc: ModuleUnavailable) -> JSONResponse:
        return _error(501, f"module not available: {exc.module_name}", exc.cause_text)

    @app.exception_handler(StarletteHTTPException)
    async def _http(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        detail = exc.detail
        error = detail if isinstance(detail, str) else (detail.get("error") if isinstance(detail, dict) else str(detail))
        return _error(exc.status_code, str(error), detail)

    @app.exception_handler(RequestValidationError)
    async def _request_validation(request: Request, exc: RequestValidationError) -> JSONResponse:
        return _error(422, "request validation error", exc.errors())

    @app.exception_handler(ValidationError)
    async def _pydantic_validation(request: Request, exc: ValidationError) -> JSONResponse:
        return _error(422, f"validation error: {exc.error_count()} error(s) in {exc.title}",
                      json.loads(exc.json(include_url=False)))

    @app.exception_handler(FileNotFoundError)
    async def _not_found(request: Request, exc: FileNotFoundError) -> JSONResponse:
        return _error(404, "not found", str(exc))

    @app.exception_handler(ImportError)
    async def _import_error(request: Request, exc: ImportError) -> JSONResponse:
        return _error(501, f"module not available: {exc.name or exc}", str(exc))

    @app.exception_handler(NotImplementedError)
    async def _not_implemented(request: Request, exc: NotImplementedError) -> JSONResponse:
        return _error(501, "not implemented", str(exc))

    @app.exception_handler(ValueError)
    async def _value_error(request: Request, exc: ValueError) -> JSONResponse:
        return _error(400, "bad request", str(exc))

    @app.exception_handler(Exception)
    async def _internal(request: Request, exc: Exception) -> JSONResponse:
        return _error(500, "internal error", f"{type(exc).__name__}: {exc}")


# ----------------------------------------------------------------------------- meta routes
def _install_meta_routes(app: FastAPI) -> None:
    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def index() -> HTMLResponse:
        page = config.STATIC_DIR / "index.html"
        if page.exists():
            return HTMLResponse(page.read_text(encoding="utf-8"))
        return HTMLResponse(PLACEHOLDER_HTML)

    @app.get("/api/health")
    def health(request: Request) -> dict[str, Any]:
        campaign = campaign_of(request)
        state = request.app.state
        if state.claude_version is _UNSET:
            ver: Optional[str] = None
            mod = optional_module("lensmark.claude.engine", "claude_version")
            if mod is not None:
                try:
                    ver = mod.claude_version()
                except Exception:  # noqa: BLE001 - health must never fail
                    ver = None
            state.claude_version = ver
        return {"version": __version__, "campaign_dir": str(campaign.root), "engine": state.engine_name,
                "claude_bin": config.claude_bin(), "claude_version": state.claude_version,
                "n_images": len(campaign.list_ids())}

    @app.get("/api/models")
    def models(request: Request) -> dict[str, Any]:
        cfg = campaign_of(request).config
        return {"models": config.MODELS, "efforts": list(config.EFFORTS),
                "default": {"model": cfg.get("default_model", config.DEFAULT_MODEL),
                            "effort": cfg.get("default_effort", config.DEFAULT_EFFORT)}}

    @app.get("/api/style")
    def style() -> dict[str, Any]:
        return {"palette": _load(config.SCHEMA_DIR / "palette.json"),
                "style_defaults": _load(config.SCHEMA_DIR / "style_defaults.json")}

    @app.get("/api/schema/{name}")
    def schema(name: str) -> dict[str, Any]:
        path = config.SCHEMA_DIR / SCHEMA_FILES.get(name, "-")
        if name == "lensmark":
            return LensMarkFile.model_json_schema()
        if name == "proposal":
            return _load(path)
        if name == "critique":
            return _load(path) if path.exists() else Critique.model_json_schema()
        if name == "patch":
            return _load(path) if path.exists() else Patch.model_json_schema()
        raise StarletteHTTPException(status_code=404, detail=f"unknown schema {name!r}; one of {sorted(SCHEMA_FILES)}")

    @app.get("/api/config")
    def campaign_config(request: Request) -> dict[str, Any]:
        return campaign_of(request).config


def _load(path: Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ----------------------------------------------------------------------------- factory
def create_app(campaign_dir: str | Path, engine: Optional[str] = None) -> FastAPI:
    campaign = Campaign(campaign_dir)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        yield
        await app.state.runs.shutdown()

    app = FastAPI(title="LensMark", version=__version__, lifespan=lifespan,
                  docs_url="/api/docs", openapi_url="/api/openapi.json", redoc_url=None)
    app.state.campaign = campaign
    app.state.engine_name = engine or config.engine_name()
    app.state.runs = RunRegistry()
    app.state.claude_version = _UNSET
    app.add_middleware(NoStoreMiddleware)
    _install_error_handlers(app)
    _install_meta_routes(app)
    for mod in (api_images, api_annotations, api_propose, api_critique, api_export, api_voice):
        app.include_router(mod.router)
    # static front end last: /assets/*, /favicon.ico ... ; API routes above take precedence
    app.mount("/", StaticFiles(directory=str(config.STATIC_DIR), html=True, check_dir=False), name="static")
    return app


def app_from_env() -> FastAPI:
    """uvicorn factory for ``--reload`` (``LENSMARK_CAMPAIGN_DIR`` set by :func:`serve`)."""
    return create_app(os.environ["LENSMARK_CAMPAIGN_DIR"], engine=os.environ.get("LENSMARK_ENGINE"))


def serve(campaign_dir: str | Path, *, port: int = config.DEFAULT_PORT, bind: str = config.DEFAULT_BIND,
          engine: Optional[str] = None, open_browser: bool = True, reload: bool = False) -> int:
    """Run the app under uvicorn; open the browser from a delayed thread unless ``open_browser`` is False."""
    import uvicorn

    if engine:
        os.environ["LENSMARK_ENGINE"] = engine
    root = Path(campaign_dir).expanduser().resolve()
    if not root.is_dir():
        print(f"lensmark serve: campaign directory not found: {root}", flush=True)
        return 2
    host_for_url = "127.0.0.1" if bind in ("0.0.0.0", "::", "") else bind
    url = f"http://{host_for_url}:{port}/"
    print(f"LensMark {__version__}: serving {root} at {url} (engine: {engine or config.engine_name()})", flush=True)
    if open_browser:
        t = threading.Timer(1.0, webbrowser.open, args=[url])
        t.daemon = True
        t.start()
    if reload:
        os.environ["LENSMARK_CAMPAIGN_DIR"] = str(root)
        uvicorn.run("lensmark.server.app:app_from_env", factory=True, host=bind, port=port,
                    log_level="info", reload=True)
    else:
        uvicorn.run(create_app(root, engine=engine), host=bind, port=port, log_level="info")
    return 0
