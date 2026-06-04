"""lens_representations — engineered views + scalar lensing-features for the grader.

Returns the contract scalars-FIRST (one JSON feature block so the model reads numbers
before pixels) then the requested representation images (lens-light subtraction, polar/
tangential, 180-degree symmetry, blue-excess color isolation, arcness). All Tier-1
(scipy/numpy, in-process; no subprocess, ~50-100 ms). Tier-2 (photutils isophote /
skimage frangi / SEP) is added later via tools/representations_proto.py (.venvs/lens).

MEASURED efficacy (eval/run_representations.py, do not over-claim): the features
separate lenses from ORDINARY galaxies (combo AUC ~0.70) and work on spectroscopically
CONFIRMED labels (tangential/parity AUC ~0.80) and clean sims (arcness ~0.80), but DO
NOT separate hard human-rejected candidates (AUC ~0.51) — the same wall vision and a
GIGA-Lens fit hit. Supporting evidence; never decisive on the hard pool.
"""
from __future__ import annotations

import json

from claude_agent_sdk import tool

from lensjudge import config
from lensjudge.common import fetch, render
from lensjudge.common import representations as R

_CACHE = config.CACHE / "representations"

_DEFAULT_VIEWS = ["lenssub", "polar", "symmetry", "color_iso", "arcness"]

_READING_GUIDE = (
    "\n\nReading guide — map scalars to the Huang criteria:\n"
    "- tangential arc / curvature: high tangential_extent_deg with tangentiality>1 and "
    "high arcness_score; on the [polar] view a real arc is a HORIZONTAL bar.\n"
    "- counter-image / multiplicity: high counterimage_parity (flux at theta and "
    "theta+180); on [symmetry] a partner pair lights up.\n"
    "- blue source: high blue_excess_at_thetaE / ring_blue_fraction; on [color_iso] a "
    "blue arc/ring around the centre.\n"
    "- lens host: low annulus_core_ratio_* = a concentrated red elliptical.\n"
    "CAVEAT (measured): these reliably separate lenses from ORDINARY galaxies (AUC~0.70) "
    "and work on CONFIRMED labels (AUC~0.80), but DO NOT separate hard human-rejected "
    "candidates (AUC~0.51). Use as supporting evidence; never override clear contaminant "
    "morphology with a single high scalar."
)

_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "survey": {"type": "string"},
        "ra": {"type": "number"}, "dec": {"type": "number"},
        "views": {"type": "array", "items": {"type": "string", "enum": list(R.REPR_RENDERERS)},
                  "description": "subset of engineered views (default: lenssub, polar, "
                                 "symmetry, color_iso, arcness)"},
    },
    "required": ["name"],
}


@tool("lens_representations",
      "Compute engineered representations of a candidate's grz cutout that make the "
      "lensing signal explicit: a scalar lensing-FEATURE vector (tangential_extent, "
      "tangentiality, counterimage_parity, arcness_score, blue_excess_at_thetaE, "
      "ring_blue_fraction, annulus_core_ratio) PLUS images (lens-light subtraction, "
      "polar/tangential transform, 180-deg symmetry residual, blue-excess color "
      "isolation, arcness ridge map). Reliable vs ordinary galaxies, weak on hard "
      "human-rejects — supporting evidence only.", _SCHEMA)
async def lens_representations(args: dict) -> dict:
    name = args.get("name")
    survey = args.get("survey") or "storfer"
    views = [v for v in (args.get("views") or _DEFAULT_VIEWS) if v in R.REPR_RENDERERS]
    cube = fetch.get_cube(name=name, ra=args.get("ra"), dec=args.get("dec"), survey=survey)
    if cube is None:
        return {"content": [{"type": "text", "text": "ERROR: no cutout for representations."}],
                "is_error": True}

    _CACHE.mkdir(parents=True, exist_ok=True)
    cache_file = _CACHE / f"{name}.json" if name else None
    if cache_file is not None and cache_file.exists():
        feats = json.loads(cache_file.read_text())
    else:
        feats = R.compute_features(cube)
        if cache_file is not None:
            cache_file.write_text(json.dumps(feats))

    content = [{"type": "text",
                "text": "Lensing-feature vector (quantitative evidence):\n"
                        + json.dumps(feats) + _READING_GUIDE}]
    for v in views:
        img = R.render_view(cube, v)
        if img is None:
            continue
        content.append({"type": "text", "text": f"[{v}] {R.REPR_VIEW_DESC.get(v, '')}"})
        content.append({"type": "image", "data": render.png_b64(img), "mimeType": "image/png"})
    return {"content": content}
