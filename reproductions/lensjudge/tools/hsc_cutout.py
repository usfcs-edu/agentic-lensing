"""fetch_hsc_cutout — the HSC-SSP PDR3 vision tool (0.168" grizy high-res analog of fetch_cutout).

Given an (ra, dec), fetches the HSC PDR3 coadd cutout (common/hsc) and returns rendered image
blocks (color context + sharp i-band luminance + tight zoom + lens-subtracted residual). Same
contract as fetch_cutout / fetch_euclid_cutout, so the grader can be pointed at HSC by swapping the
tool name. Returns an error block (no coverage / no credentials) which the grader treats as a
tier-1-only result.
"""
from __future__ import annotations

try:  # optional: @tool (MCP) is only used by the anthropic path; no-op keeps core logic SDK-free
    from claude_agent_sdk import tool
except ModuleNotFoundError:
    def tool(*_a, **_k):
        def _deco(fn):
            return fn
        return _deco

from lensjudge.common import hsc, render

VIEW_DESC = {
    "full": "16\" field, HSC color (R=i, G=r, B=g) at 0.168\"/px (~8x sharper than DESI grz). "
            "Old red lens galaxies look red/orange; lensed background sources look blue.",
    "zoom": "6\" center crop, HSC color — the arc / counter-image region.",
    "lum": "i-band only (the sharp 0.168\" luminance band), 10\" field, asinh stretch — thin "
           "tangential arcs and Einstein rings are clearest here.",
    "lum_zoom": "i-band only, 5\" hard stretch — faint thin arcs / counter-images.",
    "lum_sub": "SIGNED i-band lens-light residual: chi=(data - smooth model)/noise on a "
               "diverging scale (RED = unmodelled excess / arc / counter-image, BLUE = "
               "over-subtraction, fixed +/-5 sigma). A blue core or a thin ring concentric with "
               "the galaxy is a known artifact (over-subtraction / color gradient), not a lens.",
}

_SCHEMA = {
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


@tool("fetch_hsc_cutout", "Render an HSC-SSP PDR3 strong-lens candidate's 0.168\"/px grizy cutout "
      "at the given ra/dec to several views (color full/zoom, sharp i-band, i hard-zoom, "
      "lens-subtracted residual) and return them as images to inspect.", _SCHEMA)
async def fetch_hsc_cutout(args: dict) -> dict:
    ra, dec = args.get("ra"), args.get("dec")
    views = args.get("views") or ["full", "lum", "zoom", "lum_sub"]
    bands = hsc.load_hsc(ra, dec)
    if bands is None:
        return {"content": [{"type": "text",
                "text": f"ERROR: no HSC PDR3 coverage (or no credentials) for ra={ra}, dec={dec}."}],
                "is_error": True}
    imgs = hsc.render_hsc_views(bands, views=[v for v in views if v in VIEW_DESC])
    content = [{"type": "text",
                "text": f"HSC-SSP PDR3 candidate ra={ra}, dec={dec} — 0.168\"/px grizy "
                        f"(bands present: {''.join(sorted(bands))}). Views follow:"}]
    for v, img in imgs.items():
        content.append({"type": "text", "text": f"[{v}] {VIEW_DESC[v]}"})
        content.append({"type": "image", "data": render.png_b64(img), "mimeType": "image/png"})
    return {"content": content}
