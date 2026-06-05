"""fetch_euclid_cutout — the Euclid vision tool (0.1" VIS+NIR high-res analog of fetch_cutout).

Given a Euclid Q1 id_str, loads the multi-band cutout from euclid-q1/data/ and returns
rendered image blocks (color context + sharp VIS luminance + tight zoom + lens-subtracted
residual). Same contract as fetch_cutout, so the lean grader can be pointed at Euclid by
swapping the tool name and the id.
"""
from __future__ import annotations

from claude_agent_sdk import tool

from lensjudge.common import euclid, render

VIEW_DESC = {
    "full": "16\" field, Euclid color (B=VIS optical, G=NIR-J, R=NIR-H) at 0.1\"/px "
            "(~13x sharper than DESI grz). Old red lens galaxies look red/orange; "
            "lensed background sources look blue.",
    "zoom": "6\" center crop, Euclid color — the arc / counter-image region.",
    "vis": "VIS band only (the sharp 0.1\" broad-optical luminance band), 10\" field, "
           "asinh stretch — thin tangential arcs and Einstein rings are clearest here.",
    "vis_zoom": "VIS band only, 5\" hard stretch — faint thin arcs / counter-images.",
    "vis_sub": "VIS with the smooth lens-galaxy light subtracted (azimuthal-median model) "
               "— a ring/arc residual stands out against a flat background.",
}

_SCHEMA = {
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


@tool("fetch_euclid_cutout", "Render a Euclid Q1 strong-lens candidate's 0.1\"/px VIS+NIR "
      "cutout to several views (color full/zoom, sharp VIS, VIS hard-zoom, lens-subtracted "
      "residual) and return them as images to inspect.", _SCHEMA)
async def fetch_euclid_cutout(args: dict) -> dict:
    id_str = args.get("id_str") or args.get("name")
    views = args.get("views") or ["full", "vis", "zoom", "vis_sub"]
    bands = euclid.load_euclid(id_str)
    if bands is None:
        return {"content": [{"type": "text",
                "text": f"ERROR: no Euclid cutout on disk for id_str={id_str!r}."}],
                "is_error": True}
    imgs = euclid.render_euclid_views(bands, views=[v for v in views if v in VIEW_DESC])
    content = [{"type": "text",
                "text": f"Euclid Q1 candidate {id_str} — 0.1\"/px VIS+NIR. Views follow:"}]
    for v, img in imgs.items():
        content.append({"type": "text", "text": f"[{v}] {VIEW_DESC[v]}"})
        content.append({"type": "image", "data": render.png_b64(img), "mimeType": "image/png"})
    return {"content": content}
