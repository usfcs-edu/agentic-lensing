"""fetch_cutout — the vision tool. Renders a candidate's grz cutout to image blocks.

The Agent SDK has no image-in-prompt, so this is the ONLY way the model sees pixels.
Given a candidate name (and optionally RA/Dec for off-disk candidates like Grade-D
rejects), it loads the (3,101,101) cube (on-disk first, else legacysurvey endpoint),
renders the requested Lupton-RGB views, and returns interleaved text+image content.
"""
from __future__ import annotations

try:  # optional: @tool (MCP) is only used by the anthropic path; no-op keeps core logic SDK-free
    from claude_agent_sdk import tool
except ModuleNotFoundError:
    def tool(*_a, **_k):
        def _deco(fn):
            return fn
        return _deco

from lensjudge.common import fetch, render

_FOV = render.config.SIZE_PIX * render.config.PIXSCALE  # arcsec across the cutout

VIEW_DESC = {
    "full": f"whole cutout, {_FOV:.1f}\" across (~0.26\"/px), Lupton-RGB (z=R, r=G, g=B). "
            "Lens galaxies are red/orange; lensed sources are typically blue.",
    "zoom": "2.5x center crop — the 1-5\" region where arcs/counter-images appear.",
    "residual": "SIGNED lens-light residual, shown as a g | r | z montage. Each panel is "
                "chi=(data - smooth elliptical-galaxy model)/noise on a diverging scale: "
                "RED = unmodelled EXCESS flux (a candidate arc/ring/counter-image), BLUE = "
                "over-subtraction, white = consistent with the model + noise (fixed +/-5 sigma). "
                "A REAL lensed arc is RED and spatially coherent across BOTH g and r (the source "
                "is blue, so it is strong in g/r, weak in z) — cross-band coherence is the key "
                "real-vs-artifact test. Known FALSE positives to discount: a BLUE core or a "
                "central blue/red dipole (over-subtraction of the bright galaxy nucleus); a thin "
                "red/blue ring tightly concentric with the galaxy at one radius (lens color "
                "gradient or model mismatch); and features in only ONE band.",
    "highcontrast": "hard stretch (Q=20, stretch=0.1) to boost faint features.",
}

_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string",
                 "description": "DESI candidate name, e.g. DESI-000.7487-62.6672"},
        "survey": {"type": "string", "enum": ["storfer", "inchausti", "ls-dr9", "ls-dr10"],
                   "description": "catalog key or imaging layer (default: storfer/ls-dr9)"},
        "ra": {"type": "number", "description": "deg; only needed if not on disk"},
        "dec": {"type": "number", "description": "deg; only needed if not on disk"},
        "views": {"type": "array", "items": {"type": "string",
                  "enum": ["full", "zoom", "residual", "highcontrast"]},
                  "description": "subset of views (default: full, zoom, residual)"},
    },
    "required": ["name"],
}


@tool("fetch_cutout", "Render a strong-lens candidate's grz image cutout to one or more "
      "Lupton-RGB views (full / center-zoom / lens-light-residual / high-contrast) "
      "and return them as images to inspect.", _SCHEMA)
async def fetch_cutout(args: dict) -> dict:
    name = args.get("name")
    survey = args.get("survey") or "storfer"
    views = args.get("views") or ["full", "zoom", "residual"]
    cube = fetch.get_cube(name=name, ra=args.get("ra"), dec=args.get("dec"), survey=survey)
    if cube is None:
        return {"content": [{"type": "text",
                "text": f"ERROR: could not load a cutout for {name} (survey={survey})."}],
                "is_error": True}
    imgs = render.render_views(cube, views=[v for v in views if v in VIEW_DESC])
    content = [{"type": "text",
                "text": f"Candidate {name} — grz cutout, {_FOV:.1f}\" field. Views follow:"}]
    for v, img in imgs.items():
        # residual description tracks the active residual image (LENSJUDGE_RESIDUAL_VERSION)
        desc = render.residual_view_desc() if v == "residual" else VIEW_DESC[v]
        content.append({"type": "text", "text": f"[{v}] {desc}"})
        content.append({"type": "image", "data": render.png_b64(img), "mimeType": "image/png"})
    return {"content": content}
