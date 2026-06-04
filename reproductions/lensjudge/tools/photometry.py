"""get_photometry — quantitative color/morphology evidence, measured from the cube.

The candidate score CSVs don't carry magnitudes, so rather than an external catalog
call this measures relative aperture colors directly from the grz pixels: a central
aperture (the lens galaxy) and an annulus (where arcs/sources sit). It reports the
g-r and r-z colors of each region so the agent can check the lens-red / source-blue
signature (Huang-2020 criterion 1) numerically, not just by eye.
"""
from __future__ import annotations

import numpy as np
from claude_agent_sdk import tool

from lensjudge.common import fetch

_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "survey": {"type": "string"},
        "ra": {"type": "number"},
        "dec": {"type": "number"},
    },
    "required": ["name"],
}


def _aperture_colors(cube: np.ndarray):
    n = cube.shape[1]
    yy, xx = np.mgrid[0:n, 0:n]
    r = np.hypot(xx - n // 2, yy - n // 2)
    core = r <= 6                      # ~1.6"  central (lens galaxy)
    annulus = (r > 6) & (r <= 18)      # ~1.6-4.7"  (arc/source region)
    out = {}
    for region, mask in (("core", core), ("annulus", annulus)):
        flux = {b: float(np.clip(cube[i][mask].sum(), 1e-6, None))
                for i, b in enumerate("grz")}
        gr = -2.5 * np.log10(flux["g"] / flux["r"])
        rz = -2.5 * np.log10(flux["r"] / flux["z"])
        out[region] = {"g_minus_r": round(gr, 3), "r_minus_z": round(rz, 3)}
    # bluer annulus than core (more negative g-r) is the lens/source color contrast
    out["annulus_bluer_than_core"] = bool(
        out["annulus"]["g_minus_r"] < out["core"]["g_minus_r"] - 0.1)
    return out


@tool("get_photometry", "Measure relative aperture colors (g-r, r-z) of the central "
      "galaxy vs the surrounding annulus from the candidate's grz pixels, to test the "
      "lens-red / source-blue signature numerically.", _SCHEMA)
async def get_photometry(args: dict) -> dict:
    import json
    cube = fetch.get_cube(name=args.get("name"), ra=args.get("ra"),
                          dec=args.get("dec"), survey=args.get("survey") or "storfer")
    if cube is None:
        return {"content": [{"type": "text", "text": "ERROR: no cutout for photometry."}],
                "is_error": True}
    res = _aperture_colors(cube)
    res["note"] = ("relative instrumental colors from aperture flux ratios; "
                   "negative g-r = bluer. Compare annulus vs core.")
    return {"content": [{"type": "text", "text": json.dumps(res)}]}
