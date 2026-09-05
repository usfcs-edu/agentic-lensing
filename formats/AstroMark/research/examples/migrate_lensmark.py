#!/usr/bin/env python3
"""Convert a `lensmark/1.0` deck to the neutral content form, and report what the conversion costs.

    python migrate_lensmark.py DECK.lensmark.json [--out content-<id>.json]

This is the migration table of `07-recommendation.md` executed rather than asserted. It converts a
real deck, and — more usefully — it prints exactly what could NOT be carried across, which is the
honest measure of what the current format was unable to say in the first place.

Role inference is lexical, over the label vocabulary the current prompt actually uses. That is the
same weak mechanism the current code relies on, and reproducing it here makes its limits visible:
every label it cannot map is reported, and in a real migration those are the rows a human resolves.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

# label substring -> role. Ordered: the first match wins, so specific patterns precede general ones.
ROLE_RULES: list[tuple[str, str]] = [
    (r"second deflector", "second_deflector"),
    (r"deflector nucleus|main deflector|\bdeflector\b|lens galaxy", "deflector"),
    (r"satellite", "satellite"),
    (r"dust lane", "dust_lane"),
    (r"secondary ring", "secondary_ring"),
    (r"counter-?arc", "counter_arc"),
    (r"counter-?image", "counter_image"),
    (r"knot", "knot"),
    (r"\barc\b|arclet|tight arc|giant arc", "arc"),
    (r"lensed image", "lensed_image"),
    (r"nearby galaxy", "field_galaxy"),
    (r"companion galaxy", "field_galaxy"),
    (r"host spiral|host shell", "field_galaxy"),
    (r"diffuse", "diffuse_candidate"),
]
NEGATIVE = re.compile(r"\bnot\b", re.I)
AMBIGUOUS = re.compile(r"\bambiguous\b|\?", re.I)


def infer_role(label: str | None) -> tuple[str | None, str]:
    if not label:
        return None, "no label"
    low = label.lower()
    for pat, role in ROLE_RULES:
        if re.search(pat, low):
            return role, "matched " + pat
    return None, "no rule matched"


def infer_polarity(label: str | None) -> tuple[str, str | None]:
    if not label:
        return "positive", None
    if NEGATIVE.search(label):
        return "negative", None          # the alternative is in prose; a human must supply the term
    if AMBIGUOUS.search(label):
        return "ambiguous", None
    return "positive", None


def convert(deck: dict) -> tuple[dict, list[str]]:
    losses: list[str] = []
    marks = []
    for it in deck.get("items", []):
        if it.get("status") in ("rejected", "invalid"):
            continue
        t = it["type"]
        m: dict = {"id": it["id"]}
        label = it.get("label")

        if t == "arrow":
            role, why = infer_role(label)
            if role is None:
                role = "ambiguous_structure"
                losses.append(f'{it["id"]}: could not infer a role from label {label!r} ({why})')
            pol, alt = infer_polarity(label)
            if pol != "positive":
                losses.append(f'{it["id"]}: polarity {pol} inferred from the label; '
                              f'the ALTERNATIVE term is in prose and needs a human')
            m.update(geometry={"kind": "vector", "tail": it["tail"], "head": it["head"]},
                     role=role, polarity=pol, label=label)
        elif t == "mask_circle":
            kind = it.get("kind", "galaxy")
            m.update(geometry={"kind": "circle", "center": it["center"],
                               "radius_arcsec": it["radius_arcsec"]},
                     role="star" if kind == "star" else "field_galaxy",
                     polarity="positive", treatment="mask")
            if kind == "artifact":
                m["role"] = "artifact"
            losses.append(f'{it["id"]}: treatment defaulted to mask — '
                          f'mask-vs-model was not expressible in lensmark/1.0')
        elif t == "einstein_ring":
            g = {"kind": "circle", "radius_arcsec": it["theta_e_arcsec"]}
            if it.get("center_ref"):
                g["center_ref"] = it["center_ref"]
            else:
                g["center"] = it["center"]
            m.update(geometry=g, role="einstein_ring", polarity="positive", bound="nominal")
        elif t == "text":
            losses.append(f'{it["id"]}: free-text note dropped — it is prose, not a mark')
            continue
        else:
            losses.append(f'{it["id"]}: unknown type {t!r}, skipped')
            continue

        if it.get("color"):
            # intentional: colour is portrayal and is re-derived from the role by the style document
            pass
        if it.get("created_by"):
            m["provenance"] = {k: v for k, v in it["created_by"].items() if v is not None}
        if it.get("review"):
            m["review"] = {k: v for k, v in it["review"].items() if v is not None}
        marks.append(m)

    sysd = deck.get("system", {})
    th = dict(sysd.get("theta_e") or {})
    out_th = {"value_arcsec": th.get("value_arcsec"), "method": th.get("method")}
    if th.get("alt_arcsec") is not None and th.get("value_arcsec") is not None:
        # the deprecated 'alt' is mapped by comparison, which is what makes it a BOUND rather than
        # an unlabelled second number
        side = "lower_arcsec" if th["alt_arcsec"] < th["value_arcsec"] else "upper_arcsec"
        out_th[side] = th["alt_arcsec"]
        losses.append(f'system.theta_e.alt_arcsec {th["alt_arcsec"]} mapped to {side} by comparison')

    system = {
        "verdict": sysd.get("verdict") or "unclear",
        "grade": sysd.get("grade"),
        "p_lens": sysd.get("p_lens"),
        "counter_image": "not_searched",
        "n_images": None,
        "hard_case": [],
        "sources": [],
        "theta_e": out_th,
        "description": sysd.get("description", ""),
    }
    for field, note in (("counter_image", "search status was never recorded"),
                        ("n_images", "image count was never recorded"),
                        ("hard_case", "hard-case tags did not exist"),
                        ("sources", "source grouping did not exist")):
        losses.append(f"system.{field}: {note} — needs a human pass")

    return {"content_version": "astromark-content/0.1",
            "image": deck["image"], "system": system, "marks": marks}, losses


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("deck", type=Path)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    deck = json.loads(args.deck.read_text(encoding="utf-8"))
    content, losses = convert(deck)
    dest = args.out or args.deck.with_suffix("").with_suffix(".content.json")
    dest.write_text(json.dumps(content, indent=2) + "\n", encoding="utf-8")

    print(f"{args.deck.name} -> {dest.name}: {len(content['marks'])} marks")
    infer = [l for l in losses if "could not infer" in l or "needs a human" in l or "ALTERNATIVE" in l]
    auto = [l for l in losses if l not in infer]
    print(f"\n{len(auto)} automatic conversions worth knowing about:")
    for l in auto[:4]:
        print(f"  · {l}")
    if len(auto) > 4:
        print(f"  · … and {len(auto)-4} more of the same kind")
    print(f"\n{len(infer)} items REQUIRING A HUMAN — these are what lensmark/1.0 could not say:")
    for l in infer:
        print(f"  · {l}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
