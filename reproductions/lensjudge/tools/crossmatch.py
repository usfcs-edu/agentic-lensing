"""crossmatch_local — flag overlap with prior-known published / confirmed lenses.

Matches a candidate's RA/Dec against the group's published catalogs (Huang 2021,
Storfer 2024, Inchausti 2025) and the Foundry-II confirmed gold within a few arcsec,
returning the nearest prior system, its grade, and separation — the §9.1 Crossmatch
factor. ``min_sep_arcsec`` excludes the candidate matching its own catalog row, so
this reports independent prior knowledge, not a tautology. Off by default during the
consensus eval (it would let the agent read the grade off the catalog); on for
production grading of genuinely new candidates.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from claude_agent_sdk import tool

from lensjudge import config

_CAT = None  # cached (DataFrame, SkyCoord)


def _load_catalogs():
    global _CAT
    if _CAT is not None:
        return _CAT
    frames = []
    for f, label in (("huang2021_published_catalog.csv", "Huang2021"),
                     ("storfer2024_published_catalog.csv", "Storfer2024"),
                     ("inchausti2025_published_catalog.csv", "Inchausti2025")):
        p = config.INCH_DATA / f
        if p.exists():
            df = pd.read_csv(p)
            if {"RA", "DEC"}.issubset(df.columns):
                df = df.rename(columns={"RA": "ra", "DEC": "dec"})
                df["catalog"] = label
                df["grade"] = df["grade"] if "grade" in df.columns else "?"
                frames.append(df[["ra", "dec", "grade", "catalog"]])
    g = config.FOUNDRY_II_DATA / "foundry_ii_master_comparison.csv"
    if g.exists():
        gd = pd.read_csv(g)
        racol = next((c for c in gd.columns if c.lower() in ("ra", "ra_deg")), None)
        deccol = next((c for c in gd.columns if c.lower() in ("dec", "dec_deg")), None)
        if racol and deccol:
            frames.append(pd.DataFrame({"ra": pd.to_numeric(gd[racol], errors="coerce"),
                                        "dec": pd.to_numeric(gd[deccol], errors="coerce"),
                                        "grade": "confirmed", "catalog": "FoundryII"}))
    if not frames:
        _CAT = (pd.DataFrame(columns=["ra", "dec", "grade", "catalog"]), None)
        return _CAT
    cat = pd.concat(frames, ignore_index=True).dropna(subset=["ra", "dec"]).reset_index(drop=True)
    sky = SkyCoord(ra=cat["ra"].values * u.deg, dec=cat["dec"].values * u.deg)
    _CAT = (cat, sky)
    return _CAT


_SCHEMA = {
    "type": "object",
    "properties": {
        "ra": {"type": "number"}, "dec": {"type": "number"},
        "radius_arcsec": {"type": "number", "description": "search radius (default 5)"},
        "min_sep_arcsec": {"type": "number",
                           "description": "ignore matches closer than this (self; default 1.0)"},
    },
    "required": ["ra", "dec"],
}


@tool("crossmatch_local", "Find the nearest prior-known published/confirmed lens to a "
      "candidate position (Huang2021/Storfer/Inchausti/FoundryII) and report its grade "
      "and separation in arcsec.", _SCHEMA)
async def crossmatch_local(args: dict) -> dict:
    cat, sky = _load_catalogs()
    if sky is None:
        return {"content": [{"type": "text", "text": json.dumps({"match": None,
                "note": "no catalogs available"})}]}
    radius = float(args.get("radius_arcsec", 5.0))
    min_sep = float(args.get("min_sep_arcsec", 1.0))
    c = SkyCoord(ra=float(args["ra"]) * u.deg, dec=float(args["dec"]) * u.deg)
    seps = c.separation(sky).to(u.arcsec).value
    order = np.argsort(seps)
    for i in order:
        s = float(seps[i])
        if s < min_sep:
            continue
        if s <= radius:
            row = cat.iloc[int(i)]
            return {"content": [{"type": "text", "text": json.dumps({
                "match": {"catalog": row["catalog"], "grade": str(row["grade"]),
                          "sep_arcsec": round(s, 2)}})}]}
        break
    return {"content": [{"type": "text", "text": json.dumps(
        {"match": None, "note": f"no prior lens within {radius}\""})}]}
