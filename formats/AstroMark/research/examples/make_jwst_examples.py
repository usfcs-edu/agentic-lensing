#!/usr/bin/env python3
"""Build AstroMark examples from the public JWST NIRCam strong-lens search.

    python make_jwst_examples.py [--out jwst/] [--src <top100_clean dir>]

Why these, and why not invented: the three candidates below are real, they are public archive data
(no embargo), and — the point — **every mark below is DERIVED from a recorded measurement** in
`top100_clean.csv`, not placed by eye. The arc position comes from `blind_arc_radius_arcsec` and
`blind_arc_pa_deg`; the ring radius from `blind_theta_E_arcsec` and its stated method; the
counter-image from the coordinates written into the verifier's own note. So the examples demonstrate
the notation carrying measurements that already exist, which is the actual claim being made.

They were chosen because each one strains a different part of the format:

  rank 1  J3440482-522486   a published lens (SL2S J02176-0513). Arc AND counter-image, both
                            present, theta_E by half-separation. The clean positive case.
  rank 2  J15199556+2122210 a single tangential arc with NO counter-image found after searching.
                            Exercises counter_image = not_found (distinct from not_searched) and
                            the single-giant-arc exception.
  rank 3  J34707505-219476  a chain of knots, plus a blue blob antipodal to it whose radius EXCEEDS
                            the arc radius. The verifier explicitly declined half-separation for
                            that reason. Exercises: ambiguous polarity, the counter-image-outside-arc
                            hard case, and a coded estimation method that records WHICH rule was used
                            and which was rejected.

PANEL GEOMETRY (detected, not assumed): each figure is 752x562 holding six 240x240 panels at
columns (8,248) (256,496) (504,744) and rows (26,266) (292,532).

  row 0: F150W normal 10"  |  F150W deep 10"  |  colour 10"
  row 1: F150W deep 3.5"   |  colour 3.5"     |  deflector-subtracted 3.5"

The six panels are SIX RENDERINGS OF ONE SKY, sharing one geometry frame — which is exactly the
`variants` requirement (R33) that the current format cannot express. `--variants` writes that case.

COORDINATES: north up, east left. For a feature at position angle PA (north through east) and
radius r arcsec in a panel of width FOV:  u = 0.5 - r*sin(PA)/FOV,  v = 0.5 - r*cos(PA)/FOV.
The deflector sits at the panel centre by construction (the yellow ticks mark it).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from PIL import Image

SRC = Path("/Users/benson/sync/research/jwst-strong-lens-search/top100_clean")
COLS = [(8, 248), (256, 496), (504, 744)]
ROWS = [(26, 266), (292, 532)]
PANELS = {
    "normal_10": (0, 0, 10.0), "deep_10": (0, 1, 10.0), "colour_10": (0, 2, 10.0),
    "deep_3.5": (1, 0, 3.5), "colour_3.5": (1, 1, 3.5), "subtracted_3.5": (1, 2, 3.5),
}


def panel(img: Image.Image, name: str) -> Image.Image:
    r, c, _ = PANELS[name]
    x0, x1 = COLS[c]
    y0, y1 = ROWS[r]
    return img.crop((x0, y0, x1, y1))


def uv(r_arcsec: float, pa_deg: float, fov: float) -> list[float]:
    """Position angle (N through E) + radius -> normalized [u, v]. North up, east left."""
    a = math.radians(pa_deg)
    dN, dE = r_arcsec * math.cos(a), r_arcsec * math.sin(a)
    return [round(0.5 - dE / fov, 4), round(0.5 - dN / fov, 4)]


def tail_from(head: list[float], L: float = 0.17) -> list[float]:
    """Tail placed away from the panel centre so the shaft never crosses the system."""
    dx, dy = head[0] - 0.5, head[1] - 0.5
    n = math.hypot(dx, dy) or 1.0
    return [round(head[0] + L * dx / n, 4), round(head[1] + L * dy / n, 4)]


def mark(mid, kind, role, **kw):
    m = {"id": mid, "geometry": kind, "role": role, "polarity": kw.pop("polarity", "positive")}
    m.update(kw)
    return m


def vector(mid, role, r, pa, fov, **kw):
    h = uv(r, pa, fov)
    return mark(mid, {"kind": "vector", "tail": tail_from(h), "head": h}, role, **kw)


def circle(mid, role, r_arcsec, **kw):
    return mark(mid, {"kind": "circle", "center_ref": "m-defl", "radius_arcsec": r_arcsec},
                role, **kw)


# --- the three scenes, each built from the CSV's own numbers -------------------------------------

def scene_rank1(fov: float) -> tuple[dict, list[dict]]:
    """SL2S J02176-0513. Arc ridge measured at PA 90 (r=1.40), 34 (1.51), 150 (1.42); compact blue
    counter-image at r=0.87, PA 281. theta_E = (1.40+0.87)/2 = 1.14 -> 1.15, half-separation."""
    marks = [
        mark("m-defl", {"kind": "vector", "tail": [0.5, 0.5 - 0.20], "head": [0.5, 0.5]},
             "deflector", label="deflector"),
        vector("m-arc", "arc", 1.43, 65.0, fov, source="S1", label="arc",
               note="ridge traced at PA 34-150, r 1.40-1.51"),
        vector("m-arc-n", "knot", 1.51, 34.0, fov, source="S1"),
        vector("m-arc-s", "knot", 1.42, 150.0, fov, source="S1"),
        vector("m-ci", "counter_image", 0.87, 281.0, fov, source="S1", emphasis="key",
               label="counter-image", note="nearly antipodal to the arc apex"),
        circle("m-ring", "einstein_ring", 1.15, bound="nominal", source="S1",
               method="half_separation", lower_arcsec=0.95, upper_arcsec=1.35),
        circle("m-ring-lo", "einstein_ring", 0.95, bound="lower", source="S1"),
        circle("m-ring-hi", "einstein_ring", 1.35, bound="upper", source="S1"),
        # the verifier's own correction: the NE streak is a field galaxy, not a second arc
        mark("m-streak", {"kind": "vector", "tail": tail_from(uv(3.4, 40.0, fov)),
                          "head": uv(3.4, 40.0, fov)},
             "arc", polarity="negative", alternative="edge_on_disk", label="arc",
             note="runs NW-SE, radially rather than tangentially: a field galaxy, not a second arc"),
    ]
    system = {
        "verdict": "likely_lens", "grade": "A", "p_lens": 0.95,
        "counter_image": "found", "n_images": 2,
        "hard_case": [],
        "sources": [{"tag": "S1", "n_images": 2, "config": "fold", "theta_e_arcsec": 1.15}],
        "theta_e": {"value_arcsec": 1.15, "lower_arcsec": 0.95, "upper_arcsec": 1.35,
                    "method": "half_separation"},
        "description": ("Published lens SL2S J02176-0513. Blue tangential crescent east of a "
                        "red elliptical, concave toward it, with a compact blue counter-image "
                        "nearly opposite. Verifier grade A, 3 of 3."),
    }
    return system, marks


def scene_rank2(fov: float) -> tuple[dict, list[dict]]:
    """Single tangential arc, ridge 1.44-1.52, mean 1.48. No counter-image visible at any stretch,
    so theta_E rests on the arc radius alone."""
    marks = [
        mark("m-defl", {"kind": "vector", "tail": [0.5, 0.5 - 0.20], "head": [0.5, 0.5]},
             "deflector", label="deflector"),
        vector("m-arc", "arc", 1.44, 131.0, fov, source="S1", label="arc", emphasis="key",
               note="ridge PA 100-175, r 1.44-1.52"),
        vector("m-arc-e", "knot", 1.45, 105.0, fov, source="S1"),
        vector("m-arc-s", "knot", 1.52, 165.0, fov, source="S1"),
        circle("m-ring", "einstein_ring", 1.48, bound="nominal", source="S1",
               method="arc_midline", lower_arcsec=1.23, upper_arcsec=1.73),
        circle("m-ring-lo", "einstein_ring", 1.23, bound="lower", source="S1"),
        circle("m-ring-hi", "einstein_ring", 1.73, bound="upper", source="S1"),
    ]
    system = {
        "verdict": "likely_lens", "grade": "A", "p_lens": 0.82,
        "counter_image": "not_found", "n_images": 1,
        "hard_case": ["single_giant_arc"],
        "sources": [{"tag": "S1", "n_images": 1, "config": "partial_ring",
                     "theta_e_arcsec": 1.48}],
        "theta_e": {"value_arcsec": 1.48, "lower_arcsec": 1.23, "upper_arcsec": 1.73,
                    "method": "arc_midline"},
        "description": ("Thin tangential arc east through south-southeast of a red elliptical, "
                        "bluer than the host, surviving in the deflector-subtracted panel. "
                        "Searched for a counter-image at every stretch; none visible. theta_E "
                        "therefore rests on the arc radius alone and could be high if the source "
                        "is offset."),
    }
    return system, marks


def scene_rank3(fov: float) -> tuple[dict, list[dict]]:
    """Five knots at r 1.24-1.49 over PA 95-162. A blue blob at r=1.88, PA 302 is antipodal and the
    same colour as the knots — but its radius EXCEEDS the arc radius, so half-separation was
    explicitly declined and theta_E rests on the knot-chain radius."""
    knots = [(1.48, 95.0), (1.49, 105.0), (1.24, 120.0), (1.24, 142.0), (1.30, 162.0)]
    marks = [
        mark("m-defl", {"kind": "vector", "tail": [0.5, 0.5 - 0.20], "head": [0.5, 0.5]},
             "deflector", label="deflector"),
    ]
    for i, (r, pa) in enumerate(knots, 1):
        marks.append(vector(f"m-knot-{i}", "knot", r, pa, fov, source="S1",
                            label="knot" if i == 1 else None))
    marks += [
        # the honest one: plausible counter-image, but its radius exceeds the arc's
        vector("m-ci", "counter_image", 1.88, 302.0, fov, polarity="ambiguous",
               alternative="companion_projection", source="S1", emphasis="key",
               label="counter-image",
               note="antipodal and the same colour as the knots, but at a GREATER radius than the "
                    "chain, which is the reverse of the usual configuration"),
        circle("m-ring", "einstein_ring", 1.40, bound="nominal", source="S1",
               method="arc_midline", lower_arcsec=1.24, upper_arcsec=1.49),
        circle("m-ring-lo", "einstein_ring", 1.24, bound="lower", source="S1"),
        circle("m-ring-hi", "einstein_ring", 1.49, bound="upper", source="S1"),
    ]
    system = {
        "verdict": "likely_lens", "grade": "A", "p_lens": 0.78,
        "counter_image": "found", "n_images": 2,
        "hard_case": ["counter_image_outside_arc", "faint_counter_image"],
        "sources": [{"tag": "S1", "n_images": 2, "config": "partial_ring",
                     "theta_e_arcsec": 1.40}],
        "theta_e": {"value_arcsec": 1.40, "lower_arcsec": 1.24, "upper_arcsec": 1.49,
                    "method": "arc_midline"},
        "description": ("Five blue compact knots forming a chain from PA 95 to 162 at r 1.24-1.49, "
                        "concave toward an orange elliptical. A blue blob antipodal to the chain at "
                        "r=1.88 is a plausible counter-image, but it lies FARTHER out than the "
                        "chain, so half-separation was declined and theta_E rests on the chain "
                        "radius alone. If the blob is the counter-image, theta_E is nearer 1.6."),
    }
    return system, marks


SCENES = {
    "J3440482-522486": ("rank 1 — SL2S J02176-0513, a published lens", scene_rank1),
    "J15199556+2122210": ("rank 2 — single arc, counter-image searched for and not found", scene_rank2),
    "J34707505-219476": ("rank 3 — knot chain, counter-image outside the arc", scene_rank3),
}
PANEL_FOR_MARKS = "colour_3.5"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    here = Path(__file__).parent
    ap.add_argument("--src", type=Path, default=SRC)
    ap.add_argument("--out", type=Path, default=here / "jwst")
    ap.add_argument("--variants", action="store_true", help="also write the six-panel variants case")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    _, _, fov = PANELS[PANEL_FOR_MARKS]
    for cid, (title, builder) in SCENES.items():
        src = args.src / f"{cid}.jpg"
        if not src.is_file():
            print(f"  missing {src}")
            continue
        full = Image.open(src).convert("RGB")
        base = panel(full, PANEL_FOR_MARKS)
        dest = args.out / f"{cid}.png"
        Image.frombytes("RGB", base.size, base.tobytes()).save(dest, optimize=False)

        system, marks = builder(fov)
        doc = {
            "content_version": "astromark-content/0.1",
            "note": (f"{title}. Public JWST NIRCam archive; no embargo. Every mark is derived from a "
                     f"measurement recorded in top100_clean.csv, not placed by eye. Panel: "
                     f"{PANEL_FOR_MARKS} ({fov} arcsec across), north up, east left."),
            "image": {"file": dest.name,
                      "sha256": hashlib.sha256(dest.read_bytes()).hexdigest(),
                      "width": base.width, "height": base.height,
                      "cutout_arcsec": fov,
                      "pixel_scale_arcsec": round(fov / base.width, 8),
                      "north_up": True, "east_left": True,
                      "survey": "JWST NIRCam", "candidate_id": cid},
            "system": system,
            "marks": [m for m in marks],
        }
        (args.out / f"content-{cid}.json").write_text(json.dumps(doc, indent=2) + "\n")
        print(f"{cid}: {base.width}x{base.height} panel, {len(marks)} marks  — {title}")

        if args.variants:
            vdir = args.out / f"{cid}-panels"
            vdir.mkdir(exist_ok=True)
            variants = []
            for name, (_, _, f) in PANELS.items():
                p = panel(full, name)
                pd = vdir / f"{name}.png"
                Image.frombytes("RGB", p.size, p.tobytes()).save(pd, optimize=False)
                variants.append({"name": name, "file": f"{cid}-panels/{name}.png",
                                 "width": p.width, "height": p.height, "cutout_arcsec": f,
                                 "sha256": hashlib.sha256(pd.read_bytes()).hexdigest()[:16]})
            (args.out / f"variants-{cid}.json").write_text(json.dumps({
                "note": ("SIX RENDERINGS OF ONE SKY. The two field-of-view groups do NOT share a "
                         "geometry frame — a mark at [u,v] in a 3.5 arcsec panel is elsewhere in a "
                         "10 arcsec panel — so `variants` must carry each panel's own scale and the "
                         "record must say which panel the marks were placed on. This is requirement "
                         "R33, and the current format cannot express it: it has one image per file."),
                "annotated_on": PANEL_FOR_MARKS,
                "variants": variants}, indent=2) + "\n")
            print(f"  + {len(variants)} panels -> {vdir.name}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
