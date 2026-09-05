#!/usr/bin/env python3
"""Generate every downstream artifact from the vocabulary file.

    python generate.py            # write the artifacts
    python generate.py --check    # fail if any artifact is stale (this is the CI gate)

This script IS the argument of `04-metadata-proposals.md`. The audit found the current format's
vocabulary defined in six uncoordinated places — pydantic Literals, three hand-written JSON Schemas,
a TypeScript union, a prompt, a palette file and a speech-recognition word list — with no generator,
so adding a role means editing six files and forgetting one is invisible until a release.

Here, `astromark-vocab-lens-1.0.json` is the only place a term is written down. Everything else is
derived. `--check` is what makes that true over time rather than on the day it was built.

Generated:
  astromark-core-1.0.schema.json   flat, closed JSON Schema for structured output (gate G1)
  generated/vocab_literals.py      pydantic Literal aliases
  generated/vocab.ts               TypeScript unions
  generated/vocabulary.md          the human-readable term tables
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
VOCAB = HERE / "astromark-vocab-lens-1.0.json"
GEN = HERE / "generated"


def load() -> dict:
    return json.loads(VOCAB.read_text(encoding="utf-8"))


def terms(v: dict, scheme: str, include_deprecated=False) -> list[dict]:
    return [t for t in v["schemes"][scheme]["terms"]
            if include_deprecated or t.get("status") != "deprecated"]


def ids(v: dict, scheme: str, **kw) -> list[str]:
    return [t["id"] for t in terms(v, scheme, **kw)]


# --- the JSON Schema -----------------------------------------------------------------------------

def build_schema(v: dict) -> dict:
    """Flat and closed: no $ref anywhere, additionalProperties false everywhere.

    Flatness is not stylistic. A structured-output schema with $ref cycles cannot be used to
    constrain generation, and the existing LensMark proposal schema is already flat for exactly this
    reason (`tests/test_schema.py` asserts it). Keeping that property is gate G1.
    """
    uv = {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2,
          "description": "normalized [u, v]; u right, v down, origin top-left"}
    item = {
        "type": "object", "additionalProperties": False,
        "required": ["id", "type", "role"],
        "properties": {
            "id": {"type": "string", "pattern": "^[A-Za-z0-9_.-]+$"},
            "type": {"type": "string", "enum": ["vector", "point", "circle", "polygon"],
                     "description": "vector: tail->head pointer. point: a location. "
                                    "circle: centre plus a MEASURED radius. polygon: a region."},
            "role": {"type": "string", "enum": ids(v, "role"),
                     "description": "what the feature is; see the vocabulary"},
            "polarity": {"type": ["string", "null"], "enum": ids(v, "polarity") + [None],
                         "description": "what the mark does to the lensing interpretation. REQUIRED "
                                        "on roles whose takes_polarity is true; MUST BE ABSENT on "
                                        "inventory and measurement roles (a field galaxy, a star, "
                                        "an artefact, the Einstein ring) which cannot bear on it"},
            "alternative": {"type": ["string", "null"], "enum": ids(v, "alternative") + [None],
                            "description": "REQUIRED when polarity is negative or ambiguous: "
                                           "the non-lens reading of this feature"},
            "tail": uv, "head": uv, "at": uv, "center": uv,
            "center_ref": {"type": ["string", "null"],
                           "description": "id of the mark whose position this circle tracks, so the "
                                          "ring and the deflector cannot drift apart"},
            "radius_arcsec": {"type": ["number", "null"], "exclusiveMinimum": 0,
                              "description": "MEASURED. For a mask, the radius to apply; for a ring, "
                                             "theta_E. Never scaled for emphasis."},
            "points": {"type": ["array", "null"], "items": uv, "minItems": 3},
            "source": {"type": ["string", "null"], "pattern": "^S[1-9]$",
                       "description": "which physical source this is an image of"},
            "bound": {"type": ["string", "null"], "enum": ["nominal", "lower", "upper", None]},
            "method": {"type": ["string", "null"], "enum": ids(v, "theta_e_method") + [None]},
            "lower_arcsec": {"type": ["number", "null"]},
            "upper_arcsec": {"type": ["number", "null"]},
            "treatment": {"type": ["string", "null"], "enum": ids(v, "treatment") + [None]},
            "emphasis": {"type": ["string", "null"], "enum": ids(v, "emphasis") + [None]},
            "label": {"type": ["string", "null"], "maxLength": 40,
                      "description": "display only; the role carries the meaning"},
            "rationale": {"type": ["string", "null"], "maxLength": 300},
        },
    }
    system = {
        "type": "object", "additionalProperties": False,
        "required": ["verdict", "counter_image", "description"],
        "properties": {
            "verdict": {"type": "string",
                        "enum": ["likely_lens", "possible", "not_lens", "unclear"]},
            "grade": {"type": ["string", "null"], "enum": ["A", "B", "C", "D", None]},
            "p_lens": {"type": ["number", "null"], "minimum": 0, "maximum": 1},
            "counter_image": {"type": "string", "enum": ids(v, "counter_image_search"),
                              "description": "'not found' must mean you searched, not that you "
                                             "did not look"},
            "n_images": {"type": ["integer", "null"], "minimum": 0,
                         "description": "a merged pair counts as two"},
            "hard_case": {"type": "array", "items": {"type": "string", "enum": ids(v, "hard_case")}},
            "theta_e": {
                "type": ["object", "null"], "additionalProperties": False,
                "properties": {
                    "value_arcsec": {"type": ["number", "null"], "exclusiveMinimum": 0},
                    "lower_arcsec": {"type": ["number", "null"]},
                    "upper_arcsec": {"type": ["number", "null"]},
                    "method": {"type": ["string", "null"], "enum": ids(v, "theta_e_method") + [None]},
                },
                "description": "theta_E is a RADIUS. It lives here and nowhere else; a ring mark "
                               "carries geometry, not a second copy of the measurement.",
            },
            "sources": {
                "type": "array",
                "items": {"type": "object", "additionalProperties": False,
                          "required": ["tag"],
                          "properties": {
                              "tag": {"type": "string", "pattern": "^S[1-9]$"},
                              "n_images": {"type": ["integer", "null"], "minimum": 1},
                              "config": {"type": ["string", "null"],
                                         "enum": ids(v, "source_config") + [None]},
                          }},
            },
            "description": {"type": "string",
                            "description": "free text FOR PEOPLE. Stripped before any model-facing "
                                           "use; see the embargo note in the spec."},
        },
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "AstroMarkLensAnnotation",
        "description": ("One annotated astronomical image, lens profile. GENERATED from "
                        "astromark-vocab/lens-1.0 — do not edit by hand. Flat (no $ref) and closed "
                        "(additionalProperties false) so it can be used as a structured-output "
                        "schema."),
        "x-generated-from": v["vocabulary"],
        "type": "object", "additionalProperties": False,
        "required": ["schema", "profile", "system", "marks"],
        "properties": {
            "schema": {"const": "astromark/1.0"},
            "profile": {"const": "astromark/lens/1.0"},
            "system": system,
            "marks": {"type": "array", "items": item},
        },
    }


# --- the other surfaces ---------------------------------------------------------------------------

def polarity_map(v: dict) -> str:
    """The roles that may carry polarity, as data — so the rule is enforced from the vocabulary."""
    ev = sorted(t["id"] for t in v["schemes"]["role"]["terms"] if t.get("takes_polarity"))
    inv = sorted(t["id"] for t in v["schemes"]["role"]["terms"] if not t.get("takes_polarity"))
    return json.dumps({"note": v["schemes"]["role"]["polarity_rule"],
                       "requires_polarity": ev, "forbids_polarity": inv}, indent=2) + "\n"


def py_literals(v: dict) -> str:
    L = ['"""GENERATED from astromark-vocab/lens-1.0. Do not edit."""',
         "from typing import Literal", ""]
    for scheme in v["schemes"]:
        name = "".join(p.capitalize() for p in scheme.split("_"))
        vals = ", ".join(f'"{i}"' for i in ids(v, scheme))
        L.append(f"{name} = Literal[{vals}]")
    return "\n".join(L) + "\n"


def ts_unions(v: dict) -> str:
    L = ["// GENERATED from astromark-vocab/lens-1.0. Do not edit.", ""]
    for scheme in v["schemes"]:
        name = "".join(p.capitalize() for p in scheme.split("_"))
        vals = " | ".join(f'"{i}"' for i in ids(v, scheme))
        L.append(f"export type {name} = {vals};")
    return "\n".join(L) + "\n"


def vocab_md(v: dict) -> str:
    L = [f"# {v['vocabulary']} — term tables", "",
         "GENERATED from `astromark-vocab-lens-1.0.json`. Do not edit.", ""]
    for scheme, s in v["schemes"].items():
        L += [f"## {scheme}", "", s["definition"], "",
              "| term | label | definition | status |", "|---|---|---|---|"]
        for t in s["terms"]:
            st = t.get("status", "")
            if t.get("replaced_by"):
                st += f" → `{t['replaced_by']}`"
            L.append(f"| `{t['id']}` | {t['label']} | "
                     f"{t.get('definition','').replace('|','\\|')} | {st} |")
        if s.get("open_question"):
            L += ["", f"> **Open question.** {s['open_question']}"]
        L.append("")
    return "\n".join(L)


ARTIFACTS = [
    (lambda v: json.dumps(build_schema(v), indent=2) + "\n", HERE / "astromark-core-1.0.schema.json"),
    (polarity_map, GEN / "polarity_rule.json"),
    (py_literals, GEN / "vocab_literals.py"),
    (ts_unions, GEN / "vocab.ts"),
    (vocab_md, GEN / "vocabulary.md"),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="fail if any artifact is stale")
    args = ap.parse_args()
    v = load()
    GEN.mkdir(exist_ok=True)

    stale = []
    for fn, dest in ARTIFACTS:
        text = fn(v)
        if args.check:
            cur = dest.read_text(encoding="utf-8") if dest.is_file() else None
            if cur != text:
                stale.append(dest.name)
        else:
            dest.write_text(text, encoding="utf-8")
            print(f"wrote {dest.relative_to(HERE.parent)}")

    if args.check:
        if stale:
            print(f"STALE: {', '.join(stale)} — run `python generate.py`", file=sys.stderr)
            return 1
        print(f"all {len(ARTIFACTS)} generated artifacts are current")
        return 0

    schema = build_schema(v)
    n_ref = json.dumps(schema).count('"$ref"')
    def closed(o):
        if isinstance(o, dict):
            bad = 0
            if o.get("type") == "object" and o.get("additionalProperties") is not False:
                bad += 1
            return bad + sum(closed(x) for x in o.values())
        if isinstance(o, list):
            return sum(closed(x) for x in o)
        return 0
    print(f"\ngate G1: $ref count {n_ref} (must be 0); "
          f"open objects {closed(schema)} (must be 0); "
          f"{sum(len(s['terms']) for s in v['schemes'].values())} terms across "
          f"{len(v['schemes'])} schemes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
