"""imaging/grader_matched.py — MATCHED-INPUTS grader (parity program, Phase C3).

LensJudge v1's direct grader gave the model only 101-px renders; the Phase-A human-
baseline report established that the Paper II/DR10 graders were never pixels-only:
their inspection pages had multi-band composites, the per-channel g/r/z images, a
Legacy Surveys sky-viewer link (zoom-out context), and photo-z/spec-z/tractor-type
metadata — and they graded as RELATIVE triage of a CNN-selected stream. This grader
upgrades direct mode to that information set, removing the input asymmetry any fair
human-vs-machine comparison must remove:

  (a) standard Lupton composite [full] + [zoom]
  (b) per-channel g|r|z grayscale montage [channels]   (render.band_montage)
  (c) a WIDE context view [wide] — ~4x the standard field at the SAME pixel scale,
      clearly labeled, fetched+cached under a distinct key (fetch.get_wide_cube);
      the stand-in for the sky-viewer link the humans used
  (d) a metadata block: photo-z ± err / spec-z (huang-2021 published catalog, joined
      by name then <2" coordinates), tractor type + region + the CNN recommendation
      score (candidate score CSVs; the humans knew the stream was CNN-selected), and
      grz aperture mags measured from the cutout itself (Legacy Surveys cutouts are
      nanomaggies -> AB; none of the local catalogs publish magnitudes)
  (e) prompts/rubric_matched.md — the five-criterion rubric reframed as relative
      triage of a CNN-selected stream (the site report's finding), same JSON contract

Deliberate DEVIATIONS from grader_direct's evidence set: the [residual] view and the
aperture-color JSON are dropped — machine-only affordances the human graders did not
have (the color information reaches the model as [channels] + the mags instead) —
and the few-shot manifest prefix is not wired (humans calibrated on the stream, not
on labeled exemplars). Transport is grader_direct.grade_candidate itself via its
pre-rendered ``content`` seam, so the ImageGrade JSON contract, single-call flow,
repair retry, thinking options, cost accounting, and dual (anthropic/openai) backend
plumbing are shared by construction; only meta["mode"]="matched" and the
matched_request trace event differ.

LEAK GUARD: the metadata joins touch catalogs that carry human grades/scores; only
tractor_type / region / CNN probability / photo-z / spec-z columns are ever read into
the prompt. Never add grade or (Paper II) score columns here.
"""
from __future__ import annotations

import math
import time
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np

from lensjudge import config
from lensjudge.common import fetch, hooks, render
from lensjudge.imaging import grader_direct
from lensjudge.imaging.grader_lean import GradeResult

_RUBRIC_MATCHED = (Path(__file__).resolve().parents[1]
                   / "prompts" / "rubric_matched.md").read_text()
_H21_CSV = config.REPRO / "huang-2021" / "data" / "huang2021_published_catalog.csv"
WIDE_FACTOR = 4                                    # 401 px = ~105" at 0.262"/px
MATCHED_VIEWS = ("full", "zoom", "channels", "wide")

_TRACTOR_GLOSS = {
    "DEV": "de Vaucouleurs profile (elliptical)",
    "SER": "Sersic profile (elliptical-like)",
    "EXP": "exponential disk",
    "REX": "round exponential (compact)",
    "PSF": "point source",
    "COMP": "composite profile",
    "DC": "de Vaucouleurs (compact)",
}


# --- metadata joins (cached table loads; per-candidate lookups are trivial) ---
@lru_cache(maxsize=None)
def _scores_table(survey: str):
    """Candidate score CSV for a catalog key, indexed by name (None if absent)."""
    path = config.SCORE_CSV.get(survey)
    if path is None or not Path(path).exists():
        return None
    import pandas as pd
    df = pd.read_csv(path)
    return df.set_index("name", drop=False)


@lru_cache(maxsize=1)
def _h21_table():
    """huang-2021 published catalog: the only local source of spec-z / photo-z±err."""
    if not _H21_CSV.exists():
        return None
    import pandas as pd
    return pd.read_csv(_H21_CSV)


def _clean(v) -> Optional[str]:
    s = str(v).strip()
    return None if (not s or s.lower() in ("nan", "none", "?")) else s


def _num(v) -> Optional[float]:
    try:
        f = float(v)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def _row_dict(row) -> dict:
    """One pandas row (Series or single-row frame) -> plain dict."""
    if hasattr(row, "iloc") and getattr(row, "ndim", 1) == 2:
        row = row.iloc[0]
    return dict(row)


def _scores_row(name: str, survey: str) -> Optional[dict]:
    tab = _scores_table(survey)
    if tab is None or name not in tab.index:
        return None
    return _row_dict(tab.loc[name])


def _h21_row(name: str, ra, dec) -> Optional[dict]:
    """Join to huang-2021 by exact name, else nearest coordinate match <2"."""
    df = _h21_table()
    if df is None:
        return None
    hit = df[df["name"] == name]
    if len(hit) == 0 and ra is not None and dec is not None:
        cosd = math.cos(math.radians(float(dec)))
        sep = np.hypot((df["RA"].to_numpy(float) - float(ra)) * cosd,
                       df["DEC"].to_numpy(float) - float(dec)) * 3600.0
        i = int(np.argmin(sep))
        if sep[i] < 2.0:
            hit = df.iloc[[i]]
    return _row_dict(hit) if len(hit) else None


def _aperture_mags(cube: np.ndarray, r_ap: int = 19, r_sky: int = 40) -> dict:
    """grz AB aperture mags measured from the cutout. Legacy Surveys cutouts are in
    nanomaggies, so m = 22.5 - 2.5 log10(flux). 19 px ~ 5" aperture on the central
    galaxy; sky = per-band median outside 40 px (~10.5", frame border)."""
    n = cube.shape[1]
    yy, xx = np.mgrid[0:n, 0:n]
    r = np.hypot(xx - n // 2, yy - n // 2)
    ap, sky = r <= r_ap, r >= r_sky
    out = {}
    for i, b in enumerate(config.BANDS):
        flux = float((cube[i][ap] - np.median(cube[i][sky])).sum())
        out[b] = round(22.5 - 2.5 * math.log10(flux), 2) if flux > 0 else None
    return out


def candidate_metadata(cand: dict, cube: np.ndarray) -> tuple[str, list]:
    """The metadata block (d): everything the human graders' page showed BESIDE the
    pixels. Returns (text_block, provenance_source_list). Reads only non-label
    columns — see the LEAK GUARD note in the module docstring."""
    name = cand.get("name")
    survey = cand.get("survey_key") or cand.get("catalog") or "storfer"
    ra, dec = _num(cand.get("ra")), _num(cand.get("dec"))
    sc = _scores_row(name, survey)
    h21 = _h21_row(name, ra, dec)
    prov = ([f"scores:{survey}"] if sc else []) + (["huang2021"] if h21 else [])

    def pick(key):
        for src in (sc, h21):
            if src is not None and _clean(src.get(key)) is not None:
                return _clean(src.get(key))
        return _clean(cand.get(key))

    tractor = pick("tractor_type")
    region = pick("region")
    # CNN recommendation score: published columns first, candidate-row fallback
    # (p_pub = the pool CSVs' published-probability column; displayed as `probability`).
    cnn_parts = []
    for src in (sc, h21, cand):
        if src is None:
            continue
        for k, shown in (("probability", "probability"), ("p_pub", "probability"),
                         ("p_meta", "p_meta"), ("p_resnet", "p_resnet"),
                         ("p_effnet", "p_effnet")):
            v = _num(src.get(k))
            if v is not None and not any(p.startswith(f"{shown}=") for p in cnn_parts):
                cnn_parts.append(f"{shown}={v:.3f}")
        if cnn_parts:
            break
    spec_z = _num(h21.get("spec_z")) if h21 else None
    photo_z = _clean(h21.get("photo_z")).replace("+/-", " ± ") if (
        h21 and _clean(h21.get("photo_z"))) else None
    mags = _aperture_mags(cube)
    mtxt = "  ".join(f"{b}={mags[b]:.2f}" if mags[b] is not None else f"{b}=n/a"
                     for b in config.BANDS)
    colors = []
    if mags["g"] is not None and mags["r"] is not None:
        colors.append(f"g-r={mags['g'] - mags['r']:+.2f}")
    if mags["r"] is not None and mags["z"] is not None:
        colors.append(f"r-z={mags['r'] - mags['z']:+.2f}")
    lines = [
        "CANDIDATE METADATA (the same catalog information the human graders had):",
        f"- name={name!r}, survey={survey!r}"
        + (f", RA={ra:.6f}, Dec={dec:.6f}" if ra is not None and dec is not None else ""),
        "- Tractor type: " + (f"{tractor} — {_TRACTOR_GLOSS.get(tractor, 'unglossed type')}"
                              if tractor else "not available")
        + (f" | imaging region: {region}" if region else ""),
        "- CNN recommendation score: "
        + ((", ".join(cnn_parts) + " (this candidate was CNN-selected; the score is a "
            "prior, not ground truth)") if cnn_parts else "not available"),
        f"- photo-z (central galaxy): {photo_z or 'not available'}",
        f"- spec-z (central galaxy): {spec_z if spec_z is not None else 'none'}",
        f"- grz aperture mags (5\" radius, measured from the cutout): {mtxt}"
        + (f"; colors {', '.join(colors)}" if colors else ""),
    ]
    return "\n".join(lines), prov


def _wide_image(wide: np.ndarray):
    """Render the wide cube with the SAME base stretch as full/zoom (A/B lever aware)."""
    if render.render_stretch() == "arcsinh":
        return render.arcsinh_rgb(wide, px=config.RENDER_PX)
    return render.lupton(wide, px=config.RENDER_PX)


def _build_content(cand: dict) -> tuple[Optional[list], dict]:
    """Render the matched evidence set into Messages content blocks.

    Returns (content, info); content is None when the standard cutout can't be
    loaded (mirrors grader_direct's error path). A missing WIDE cutout degrades to a
    labeled text note rather than failing the candidate."""
    name = cand.get("name")
    survey = cand.get("survey_key") or cand.get("catalog") or "storfer"
    cube = fetch.get_cube(name=name, ra=cand.get("ra"), dec=cand.get("dec"), survey=survey)
    if cube is None:
        return None, {}
    imgs = render.render_views(cube, views=("full", "zoom"))
    imgs["channels"] = render.band_montage(cube)
    wide = fetch.get_wide_cube(name=name, ra=_num(cand.get("ra")), dec=_num(cand.get("dec")),
                               survey=survey, factor=WIDE_FACTOR)
    meta_text, prov = candidate_metadata(cand, cube)
    fov = config.SIZE_PIX * config.PIXSCALE
    wfov = fetch.wide_size(WIDE_FACTOR) * config.PIXSCALE
    labels = {
        "full": (f"[full] Lupton-grz composite of the standard {fov:.1f}\" field "
                 "(~0.26\"/px; z=R/r=G/g=B; lens galaxies red, sources blue)"),
        "zoom": "[zoom] 2.5x center crop — the 1-5\" lens/source region",
        "channels": ("[channels] per-band grayscale montage, g | r | z left to right, "
                     "each band independently arcsinh-stretched — check that a candidate "
                     "feature appears in MORE THAN ONE band and compare relative "
                     "brightness (blue source = brighter in g than z)"),
        "wide": (f"[wide] CONTEXT view: {wfov:.0f}\" field ({WIDE_FACTOR}x the standard "
                 "field) at the SAME pixel scale, candidate exactly at center — check "
                 "environment (group/cluster of red galaxies), nearby bright stars, and "
                 "morphology extending beyond the small cutout"),
    }
    content: list = [{"type": "text", "text":
                      f"Grade this strong-lens candidate.\n\n{meta_text}\n\n"
                      "The views follow (all evidence is inline; there are no tools):"}]
    for v in ("full", "zoom", "channels"):
        if v in imgs:
            content.append({"type": "text", "text": labels[v]})
            content.append({"type": "image", "source": {
                "type": "base64", "media_type": "image/png",
                "data": render.png_b64(imgs[v])}})
    if wide is not None:
        content.append({"type": "text", "text": labels["wide"]})
        content.append({"type": "image", "source": {
            "type": "base64", "media_type": "image/png",
            "data": render.png_b64(_wide_image(wide))}})
    else:
        content.append({"type": "text", "text":
                        "[wide] context view unavailable (off-footprint or fetch "
                        "failure) — grade from the remaining views."})
    content.append({"type": "text", "text":
                    "Respond with ONLY the JSON object for the required schema."})
    info = {"views": [v for v in MATCHED_VIEWS if v in imgs or (v == "wide" and wide is not None)],
            "wide_ok": wide is not None, "metadata_sources": prov,
            "n_images": sum(1 for b in content if b.get("type") == "image")}
    return content, info


async def grade_candidate(cand: dict, *, model: Optional[str] = None,
                          tools=("fetch_cutout", "get_photometry"),  # ignored (no loop)
                          system_prompt: Optional[str] = None,
                          trace_path: Optional[str] = None) -> GradeResult:
    """Drop-in with grader_direct/grader_lean (same GradeResult, same signature), so
    run_batch.py --mode matched reuses the whole harness."""
    t0 = time.time()
    content, info = _build_content(cand)
    if content is None:
        return GradeResult(None, "", error="no cutout", parse_ok=False,
                           meta={"name": cand.get("name"), "mode": "matched"})
    if trace_path:
        hooks.Trace(trace_path).write(
            "matched_request", views=info["views"], wide_ok=info["wide_ok"],
            metadata_sources=info["metadata_sources"], n_images=info["n_images"])
    res = await grader_direct.grade_candidate(
        cand, model=model, system_prompt=system_prompt or _RUBRIC_MATCHED,
        content=content, trace_path=trace_path)
    res.meta.update({"mode": "matched", "wide_ok": info["wide_ok"],
                     "metadata_sources": ",".join(info["metadata_sources"]) or None,
                     "wall_s": round(time.time() - t0, 2)})  # include render+wide fetch
    return res
