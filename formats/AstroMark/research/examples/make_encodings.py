#!/usr/bin/env python3
"""Encode the ONE reference annotation in every candidate metadata format, and measure the cost.

    python make_encodings.py [--out encodings/]

Every file written here says exactly the same thing about the same image. That is the point: the
formats can then be compared on size, shape and expressiveness rather than on how much their author
chose to record.

ON THE NUMBERS. Character counts are exact. Token counts are NOT measured — no Claude tokenizer is
available offline, and a different tokenizer family would give a misleading absolute. What is
reported is the exact character count plus a token estimate at a stated ratio, and the RATIO
BETWEEN ENCODINGS, which is what the decision actually turns on and is nearly tokenizer-independent.
Anywhere a token figure appears in the reports it is marked as derived, not measured.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

CHARS_PER_TOKEN = 3.5      # JSON is punctuation-dense; prose runs nearer 4.
VOCAB = "astromark-vocab/lens-1.0"


def load(here: Path) -> dict:
    return json.loads((here / "content.json").read_text(encoding="utf-8"))


def j(o) -> str:
    return json.dumps(o, indent=2, ensure_ascii=False) + "\n"


# --- P1: evolved bespoke -------------------------------------------------------------------------

def p1(c: dict) -> str:
    """Typed items, orthogonal semantic fields, portrayal and vocabulary hoisted out by reference."""
    out = {
        "schema": "astromark/1.0",
        "profile": "astromark/lens/1.0",
        "id": "astromark-ref",
        "style_ref": {"id": "astromark-style/thin-1.0", "sha256": "0" * 64},
        "vocab_ref": {"id": VOCAB, "sha256": "0" * 64},
        "image": c["image"],
        "system": c["system"],
        "items": [],
    }
    for m in c["marks"]:
        it = {"id": m["id"], "role": "lens:" + m["role"], "polarity": m.get("polarity", "positive")}
        g = m["geometry"]
        it["type"] = g["kind"]
        for k in ("tail", "head", "at", "center", "center_ref", "radius_arcsec", "points"):
            if k in g:
                it[k] = g[k]
        for k in ("source", "alternative", "treatment", "bound", "method",
                  "lower_arcsec", "upper_arcsec", "label", "emphasis", "status", "note"):
            if k in m:
                it[k] = ("lens:" + m[k]) if k == "alternative" else m[k]
        if "provenance" in m:
            it["prov"] = m["provenance"]
        if "review" in m:
            it["review"] = m["review"]
        out["items"].append(it)
    return j(out)


# --- P2: GeoJSON-shaped --------------------------------------------------------------------------

def p2(c: dict) -> str:
    """FeatureCollection core, profile-defined properties.

    Note the structural defect this makes visible: GeoJSON has no circle. Nine of twenty marks are
    circles whose RADIUS IS THE DATUM, so they degrade to a Point plus a properties field, and the
    geometry member no longer holds the geometry.
    """
    feats = []
    for m in c["marks"]:
        g = m["geometry"]
        props = {"mark": g["kind"], "role": m["role"], "polarity": m.get("polarity", "positive")}
        for k in ("source", "alternative", "treatment", "bound", "method",
                  "lower_arcsec", "upper_arcsec", "label", "emphasis", "status", "note"):
            if k in m:
                props[k] = m[k]
        if "provenance" in m:
            props["prov"] = m["provenance"]
        if "review" in m:
            props["review"] = m["review"]
        if g["kind"] == "vector":
            geom = {"type": "LineString", "coordinates": [g["tail"], g["head"]]}
        elif g["kind"] == "point":
            geom = {"type": "Point", "coordinates": g["at"]}
        elif g["kind"] == "circle":
            geom = {"type": "Point", "coordinates": g.get("center", [0, 0])}
            props["radius_arcsec"] = g["radius_arcsec"]        # the datum, exiled to properties
            if "center_ref" in g:
                props["center_ref"] = g["center_ref"]
            props["_geojson_note"] = "radius is not representable in GeoJSON geometry"
        else:
            geom = {"type": "Polygon", "coordinates": [g["points"] + [g["points"][0]]]}
        feats.append({"type": "Feature", "id": m["id"], "geometry": geom, "properties": props})
    return j({
        "type": "FeatureCollection",
        "astromark": {"core": "astromark/1.0", "profile": "astromark/lens/1.0",
                      "frame": {"units": "normalized", "origin": "top_left",
                                "x": "right", "y": "down", "size_units": "arcsec",
                                **{k: c["image"][k] for k in
                                   ("width", "height", "cutout_arcsec", "north_up", "east_left")}},
                      "system": c["system"]},
        "features": feats,
    })


# --- P3: W3C Web Annotation ------------------------------------------------------------------------

def p3(c: dict) -> str:
    """JSON-LD body/target/selector. Review is native — a critique is an Annotation whose target is
    another Annotation. The cost is that every semantic byte becomes an absolute IRI."""
    V = "https://astromark.io/vocab/lens/1.0#"
    C = "https://astromark.io/vocab/core/1.0#"
    items = []
    for m in c["marks"]:
        g = m["geometry"]
        if g["kind"] == "vector":
            svg = (f'<svg><line x1="{g["tail"][0]}" y1="{g["tail"][1]}" '
                   f'x2="{g["head"][0]}" y2="{g["head"][1]}"/></svg>')
        elif g["kind"] == "circle":
            ctr = g.get("center", [0, 0])
            svg = f'<svg><circle cx="{ctr[0]}" cy="{ctr[1]}" r="{g["radius_arcsec"]}"/></svg>'
        elif g["kind"] == "point":
            svg = f'<svg><circle cx="{g["at"][0]}" cy="{g["at"][1]}" r="0"/></svg>'
        else:
            pts = " ".join(f"{p[0]},{p[1]}" for p in g["points"])
            svg = f'<svg><polygon points="{pts}"/></svg>'
        body = [{"type": "SpecificResource", "purpose": "classifying", "source": V + m["role"]},
                {"type": "SpecificResource", "purpose": "assessing",
                 "source": C + "polarity-" + m.get("polarity", "positive")}]
        if m.get("alternative"):
            body.append({"type": "SpecificResource", "purpose": "describing",
                         "source": V + m["alternative"]})
        ann = {"id": "urn:astromark:ref/" + m["id"], "type": "Annotation",
               "motivation": "classifying", "body": body,
               "target": {"source": "urn:astromark:image/astromark-ref",
                          "selector": {"type": "SvgSelector", "value": svg}}}
        if "provenance" in m:
            ann["creator"] = {"id": "urn:agent:" + str(m["provenance"].get("agent", "unknown")),
                              "type": "Software"}
        items.append(ann)
        if "review" in m:
            items.append({"id": "urn:astromark:ref/" + m["id"] + "/review", "type": "Annotation",
                          "motivation": "assessing",
                          "creator": {"id": "urn:person:" + str(m["review"].get("by", "?"))},
                          "body": {"type": "SpecificResource",
                                   "source": C + "verdict-" + m["review"]["verdict"]},
                          "target": "urn:astromark:ref/" + m["id"]})
    return j({"@context": "http://www.w3.org/ns/anno.jsonld", "type": "AnnotationPage",
              "items": items})


# --- P4: markup / annotation split with coded concepts ---------------------------------------------

def p4(c: dict) -> str:
    """The DICOM SR / NCI AIM lineage: geometry carries NO meaning; meaning is a coded concept that
    references geometry. 'Green means deflector' is structurally impossible here."""
    markup, obs = [], []
    for i, m in enumerate(c["marks"], 1):
        g = m["geometry"]
        mid = f"k{i}"
        if g["kind"] == "vector":
            markup.append({"id": mid, "shape": "vector", "points": [g["tail"], g["head"]]})
        elif g["kind"] == "point":
            markup.append({"id": mid, "shape": "point", "points": [g["at"]]})
        elif g["kind"] == "circle":
            e = {"id": mid, "shape": "circle", "radius": {"value": g["radius_arcsec"], "unit": "arcsec"}}
            if "center_ref" in g:
                e["center_of"] = g["center_ref"]
            else:
                e["points"] = [g["center"]]
            markup.append(e)
        else:
            markup.append({"id": mid, "shape": "region", "points": g["points"]})
        o = {"id": "o-" + m["id"], "about": [mid],
             "concept": {"scheme": "AM-LENS", "code": m["role"].upper(), "meaning": m["role"].replace("_", " ")},
             "polarity": {"scheme": "AM-CORE", "code": m.get("polarity", "positive")[:3].upper(),
                          "meaning": m.get("polarity", "positive")}}
        if m.get("alternative"):
            o["because"] = {"scheme": "AM-LENS", "code": m["alternative"].upper(),
                            "meaning": m["alternative"].replace("_", " ")}
        if m.get("source"):
            o["member_of"] = m["source"]
        if m.get("treatment"):
            o["treatment"] = {"scheme": "AM-CORE", "code": m["treatment"].upper(),
                              "meaning": m["treatment"]}
        if m["role"] == "einstein_ring" and m.get("bound") == "nominal":
            o["value"] = {"value": g["radius_arcsec"], "unit": "arcsec",
                          "lower": m.get("lower_arcsec"), "upper": m.get("upper_arcsec"),
                          "method": {"scheme": "AM-LENS", "code": "ARCMID", "meaning": "arc midline"}}
        if "provenance" in m:
            o["provenance"] = m["provenance"]
        if "review" in m:
            o["review"] = {"verdict": {"scheme": "AM-CORE", "code": m["review"]["verdict"].upper(),
                                       "meaning": m["review"]["verdict"].replace("_", " ")},
                           "delta": {"value": m["review"].get("delta_arcsec"), "unit": "arcsec"},
                           "by": m["review"].get("by")}
        obs.append(o)
    return j({"schema": "astromark/1.0", "profile": "astromark/lens/1.0",
              "image": c["image"], "system": c["system"],
              "markup": markup, "observations": obs})


# --- P5: the compact line notation -----------------------------------------------------------------

ROLE_TAG = {
    "deflector": "defl", "second_deflector": "defl2", "satellite": "sat", "arc": "arc",
    "knot": "knot", "counter_image": "cimg", "lensed_image": "limg", "dust_lane": "dust",
    "field_galaxy": "gal", "star": "star", "einstein_ring": "ring",
    "lens_light": "seg.lens", "lensed_light": "seg.src",
}
POL = {"positive": "+", "negative": "-", "ambiguous": "?"}


def p5(c: dict) -> str:
    """One line per mark. Same information, a fifteenth of the bytes. This is the surface a model
    READS; it still writes JSON under a schema."""
    im, sy = c["image"], c["system"]
    th = sy["theta_e"]
    L = [
        f'#astromark lens/1.0  frame=norm,tl,+x-right,+y-down  size=arcsec',
        f'#image {im["width"]}x{im["height"]} cut={im["cutout_arcsec"]}" N=up E=left sha={im["sha256"][:12]}',
        f'#system {sy["verdict"]} grade={sy["grade"]} p={sy["p_lens"]} '
        f'cimg={sy["counter_image"]} n={sy["n_images"]} hard={",".join(sy["hard_case"])}',
    ]
    for s in sy["sources"]:
        L.append(f'#source {s["tag"]} images={s["n_images"]} config={s["config"]} thE={s["theta_e_arcsec"]}"')
    for m in c["marks"]:
        g, tag = m["geometry"], ROLE_TAG.get(m["role"], m["role"])
        pol = POL[m.get("polarity", "positive")]
        bits = []
        if g["kind"] == "vector":
            bits.append(f'{g["tail"][0]:.3f},{g["tail"][1]:.3f} -> {g["head"][0]:.3f},{g["head"][1]:.3f}')
        elif g["kind"] == "point":
            bits.append(f'{g["at"][0]:.3f},{g["at"][1]:.3f}')
        elif g["kind"] == "circle":
            ctr = f'@{g["center_ref"]}' if "center_ref" in g else \
                  f'{g["center"][0]:.3f},{g["center"][1]:.3f}'
            bits.append(f'{ctr} r={g["radius_arcsec"]}"')
        else:
            bits.append("poly:" + " ".join(f'{p[0]:.3f},{p[1]:.3f}' for p in g["points"]))
        if m.get("source"):
            bits.append(m["source"])
        if m.get("bound") and m["bound"] != "nominal":
            bits.append(f'bound={m["bound"]}')
        if m.get("lower_arcsec"):
            bits.append(f'[{m["lower_arcsec"]},{m["upper_arcsec"]}] method={m.get("method","")}')
        if m.get("treatment"):
            bits.append(f'treat={m["treatment"]}')
        if m.get("alternative"):
            bits.append(f'alt={m["alternative"]}')
        if m.get("emphasis"):
            bits.append(f'emph={m["emphasis"]}')
        line = f'{tag:<9}{pol} ' + "  ".join(bits)
        if m.get("review"):
            line += f'  ; rev={m["review"]["verdict"]} {m["review"].get("delta_arcsec","")}"'
        L.append(line)
    return "\n".join(L) + "\n"


def p5_read(c: dict) -> str:
    """The model-READ surface: the line form with provenance, review and free text dropped."""
    keep = [l for l in p5(c).splitlines() if not l.startswith("#image")]
    return "\n".join(l.split("  ; rev=")[0] for l in keep) + "\n"


# --- P6: relational ----------------------------------------------------------------------------------

def p6(c: dict) -> str:
    """Rows, the shape astronomy archives actually ingest. Best corpus-query story; worst
    single-document editing story."""
    marks, asserts, reviews = [], [], []
    for m in c["marks"]:
        g = m["geometry"]
        row = {"id": m["id"], "image": "astromark-ref", "kind": g["kind"]}
        if g["kind"] == "vector":
            row.update(u1=g["tail"][0], v1=g["tail"][1], u2=g["head"][0], v2=g["head"][1])
        elif g["kind"] == "point":
            row.update(u1=g["at"][0], v1=g["at"][1])
        elif g["kind"] == "circle":
            row.update(u1=(g.get("center") or [None, None])[0],
                       v1=(g.get("center") or [None, None])[1],
                       center_ref=g.get("center_ref"), r_arcsec=g["radius_arcsec"])
        else:
            row.update(n_points=len(g["points"]))
        marks.append(row)
        asserts.append({k: v for k, v in {
            "mark_id": m["id"], "role": m["role"], "polarity": m.get("polarity", "positive"),
            "alternative": m.get("alternative"), "source": m.get("source"),
            "bound": m.get("bound"), "method": m.get("method"),
            "lower": m.get("lower_arcsec"), "upper": m.get("upper_arcsec"),
            "treatment": m.get("treatment"), "emphasis": m.get("emphasis"),
        }.items() if v is not None})
        if "review" in m:
            reviews.append({"mark_id": m["id"], **m["review"]})
    return j({"tables": {"images": [c["image"]], "systems": [c["system"]],
                         "marks": marks, "assertions": asserts, "reviews": reviews},
              "note": "shipped as FITS BINTABLE / VOTable / Parquet in practice; JSON here for comparison"})


ENCODERS = [
    ("P1-evolved.json", "P1 evolved bespoke", p1),
    ("P2-geojson.json", "P2 GeoJSON-shaped", p2),
    ("P3-webannotation.jsonld", "P3 W3C Web Annotation", p3),
    ("P4-coded-concepts.json", "P4 markup/annotation split", p4),
    ("P5-lines.amk", "P5 line notation", p5),
    ("P5-lines-read.amk", "P5 model-READ surface", p5_read),
    ("P6-relational.json", "P6 relational", p6),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    here = Path(__file__).parent
    ap.add_argument("--out", type=Path, default=here / "encodings")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    c = load(here)

    base = len((here / "content.json").read_text(encoding="utf-8"))
    rows = []
    for fn, title, enc in ENCODERS:
        text = enc(c)
        (args.out / fn).write_text(text, encoding="utf-8")
        n = len(text)
        rows.append({"file": fn, "title": title, "chars": n,
                     "est_tokens": round(n / CHARS_PER_TOKEN),
                     "x_smallest": None, "lines": text.count("\n")})
    small = min(r["chars"] for r in rows)
    for r in rows:
        r["x_smallest"] = round(r["chars"] / small, 1)

    hdr = f'{"encoding":<28}{"chars":>8}{"est tok":>9}{"x smallest":>12}{"50-shot est tok":>17}'
    print(hdr); print("-" * len(hdr))
    for r in rows:
        print(f'{r["title"]:<28}{r["chars"]:>8}{r["est_tokens"]:>9}{r["x_smallest"]:>12.1f}'
              f'{r["est_tokens"] * 50:>17,}')
    print(f'\n(neutral fixture content.json = {base} chars, for reference)')
    print("Character counts are exact; token figures are DERIVED at "
          f"{CHARS_PER_TOKEN} chars/token, not measured.")
    (args.out / "sizes.json").write_text(j({"chars_per_token_assumed": CHARS_PER_TOKEN,
                                            "note": "char counts exact; token figures derived",
                                            "rows": rows}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
