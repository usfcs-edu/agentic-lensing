"""tools/openai_tools.py — OpenAI-compatible tool schemas + executor for the open-weight agentic graders.

Phase 3 of LensJudge v4: the agentic graders (grader_lean/escalate/judges) historically drove the MCP
tools (tools/server.py) via the Claude Agent SDK. For LENSJUDGE_BACKEND=openai we expose the SAME tool
logic as OpenAI function schemas + a plain-Python executor, so common.llm_client.run_tool_loop can drive
them with no Claude SDK. The actual work reuses the exact building blocks the MCP tools use
(fetch.get_cube, render.*, _aperture_colors), so the open agent sees byte-identical evidence.

The executor returns (text_summary, [anthropic-style image blocks]); run_tool_loop puts the text in the
OpenAI `tool` message and injects the images as a follow-up user turn (OpenAI tool results are text-only).
"""
from __future__ import annotations

import json

from lensjudge.common import fetch, render
from lensjudge.tools.photometry import _aperture_colors   # reused; same as get_photometry's core
# tier-2 (high-res) loaders/renderers + the SAME view descriptions the SDK tools use, so the open
# agent sees byte-identical HSC/Euclid evidence. hsc_cutout/euclid_cutout are SDK-guarded (no-op
# @tool without claude_agent_sdk), so importing their VIEW_DESC stays SDK-free.
from lensjudge.tools.hsc_cutout import VIEW_DESC as _HSC_VIEW_DESC
from lensjudge.tools.euclid_cutout import VIEW_DESC as _EUCLID_VIEW_DESC

_FOV = render.config.SIZE_PIX * render.config.PIXSCALE

# short view descriptions (the residual one is taken live from render so it tracks the active version)
_VIEW_DESC = {
    "full": f"whole cutout, {_FOV:.1f}\" across (~0.26\"/px), Lupton-RGB (z=R, r=G, g=B); "
            "lens galaxies red/orange, lensed sources typically blue.",
    "zoom": "2.5x center crop — the 1-5\" region where arcs/counter-images appear.",
    "highcontrast": "hard stretch (Q=20) to boost faint features.",
}

_FETCH_PARAMS = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "description": "DESI candidate name, e.g. DESI-000.7487-62.6672"},
        "survey": {"type": "string", "description": "catalog key (storfer/inchausti/claudenet/dr11)"},
        "ra": {"type": "number", "description": "deg; only if not on disk"},
        "dec": {"type": "number", "description": "deg; only if not on disk"},
        "views": {"type": "array", "items": {"type": "string",
                  "enum": ["full", "zoom", "residual", "highcontrast"]},
                  "description": "subset (default full, zoom, residual)"},
    },
    "required": ["name"],
}
_PHOT_PARAMS = {
    "type": "object",
    "properties": {"name": {"type": "string"}, "survey": {"type": "string"},
                   "ra": {"type": "number"}, "dec": {"type": "number"}},
    "required": ["name"],
}
# tier-2 high-res tools (HSC by ra/dec, Euclid by id_str) — mirror the SDK @tool schemas
_HSC_PARAMS = {
    "type": "object",
    "properties": {
        "ra": {"type": "number", "description": "ICRS right ascension in degrees"},
        "dec": {"type": "number", "description": "ICRS declination in degrees"},
        "views": {"type": "array", "items": {"type": "string",
                  "enum": ["full", "lum", "zoom", "lum_zoom", "lum_sub"]},
                  "description": "subset of views (default: full, lum, zoom, lum_sub)"},
    },
    "required": ["ra", "dec"],
}
_EUCLID_PARAMS = {
    "type": "object",
    "properties": {
        "id_str": {"type": "string",
                   "description": "Euclid Q1 object id, e.g. 102044821_NEG509449559274572380"},
        "views": {"type": "array", "items": {"type": "string",
                  "enum": ["full", "vis", "zoom", "vis_zoom", "vis_sub"]},
                  "description": "subset of views (default: full, vis, zoom, vis_sub)"},
    },
    "required": ["id_str"],
}

_TOOLS = {
    "fetch_cutout": {"type": "function", "function": {
        "name": "fetch_cutout",
        "description": "Render a strong-lens candidate's grz cutout (full / center-zoom / lens-light "
                       "residual) and return the images to inspect.",
        "parameters": _FETCH_PARAMS}},
    "get_photometry": {"type": "function", "function": {
        "name": "get_photometry",
        "description": "Measure relative aperture colors (g-r, r-z) of the central galaxy vs the "
                       "surrounding annulus, to test the lens-red / source-blue signature numerically.",
        "parameters": _PHOT_PARAMS}},
    "fetch_hsc_cutout": {"type": "function", "function": {
        "name": "fetch_hsc_cutout",
        "description": "Render an HSC-SSP PDR3 strong-lens candidate's 0.168\"/px grizy cutout at the "
                       "given ra/dec to several views (color full/zoom, sharp i-band, i hard-zoom, "
                       "lens-subtracted residual) and return them as images to inspect.",
        "parameters": _HSC_PARAMS}},
    "fetch_euclid_cutout": {"type": "function", "function": {
        "name": "fetch_euclid_cutout",
        "description": "Render a Euclid Q1 strong-lens candidate's 0.1\"/px VIS+NIR cutout to several "
                       "views (color full/zoom, sharp VIS, VIS hard-zoom, lens-subtracted residual) "
                       "and return them as images to inspect.",
        "parameters": _EUCLID_PARAMS}},
}


def tool_schemas(names) -> list:
    """OpenAI tool schemas for the requested tool names (unknown names skipped)."""
    return [_TOOLS[n] for n in (names or ["fetch_cutout", "get_photometry"]) if n in _TOOLS]


def _survey(args: dict) -> str:
    return args.get("survey") or args.get("survey_key") or args.get("catalog") or "storfer"


async def execute_tool(name: str, args: dict):
    """Run one tool. Returns (text_summary, [anthropic image blocks])."""
    if name == "fetch_cutout":
        cube = fetch.get_cube(name=args.get("name"), ra=args.get("ra"), dec=args.get("dec"),
                              survey=_survey(args))
        if cube is None:
            return (f"ERROR: could not load a cutout for {args.get('name')}.", [])
        views = args.get("views") or ["full", "zoom", "residual"]
        imgs = render.render_views(cube, views=[v for v in views if v in render.VIEWS])
        parts = [f"Candidate {args.get('name')} — grz cutout, {_FOV:.1f}\" field. Views (images below):"]
        images = []
        for v, img in imgs.items():
            desc = render.residual_view_desc() if v == "residual" else _VIEW_DESC.get(v, "")
            parts.append(f"[{v}] {desc}")
            images.append({"type": "image", "source": {
                "type": "base64", "media_type": "image/png", "data": render.png_b64(img)}})
        return (" ".join(parts), images)

    if name == "get_photometry":
        cube = fetch.get_cube(name=args.get("name"), ra=args.get("ra"), dec=args.get("dec"),
                              survey=_survey(args))
        if cube is None:
            return ("ERROR: no cutout for photometry.", [])
        res = _aperture_colors(cube)
        res["note"] = "relative instrumental colors from aperture flux ratios; negative g-r = bluer."
        return (json.dumps(res), [])

    if name == "fetch_hsc_cutout":
        from lensjudge.common import hsc            # lazy: pulls requests/astropy only when used
        ra, dec = args.get("ra"), args.get("dec")
        bands = hsc.load_hsc(ra, dec)
        if bands is None:
            return (f"ERROR: no HSC PDR3 coverage (or no credentials) for ra={ra}, dec={dec}.", [])
        views = args.get("views") or ["full", "lum", "zoom", "lum_sub"]
        imgs = hsc.render_hsc_views(bands, views=[v for v in views if v in hsc.VIEWS])
        parts = [f"HSC-SSP PDR3 candidate ra={ra}, dec={dec} — 0.168\"/px grizy "
                 f"(bands present: {''.join(sorted(bands))}). Views (images below):"]
        images = []
        for v, img in imgs.items():
            parts.append(f"[{v}] {_HSC_VIEW_DESC.get(v, '')}")
            images.append({"type": "image", "source": {
                "type": "base64", "media_type": "image/png", "data": render.png_b64(img)}})
        return (" ".join(parts), images)

    if name == "fetch_euclid_cutout":
        from lensjudge.common import euclid         # lazy: pulls astropy only when used
        id_str = args.get("id_str") or args.get("name")
        if not id_str:   # a small VLM may omit the arg — return a recoverable observation, not a crash
            return ("ERROR: fetch_euclid_cutout requires a non-empty 'id_str'.", [])
        bands = euclid.load_euclid(id_str)
        if bands is None:
            return (f"ERROR: no Euclid cutout on disk for id_str={id_str!r}.", [])
        views = args.get("views") or ["full", "vis", "zoom", "vis_sub"]
        imgs = euclid.render_euclid_views(bands, views=[v for v in views if v in euclid.VIEWS])
        parts = [f"Euclid Q1 candidate {id_str} — 0.1\"/px VIS+NIR. Views (images below):"]
        images = []
        for v, img in imgs.items():
            parts.append(f"[{v}] {_EUCLID_VIEW_DESC.get(v, '')}")
            images.append({"type": "image", "source": {
                "type": "base64", "media_type": "image/png", "data": render.png_b64(img)}})
        return (" ".join(parts), images)

    return (f"ERROR: unknown tool {name}.", [])
