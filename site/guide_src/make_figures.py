#!/usr/bin/env python
"""Generate every figure in the guide, from code, deterministically.

Run with the lensjudge venv (the only one on this host with the full stack)::

    ~/.venvs/lensjudge/bin/python site/guide_src/make_figures.py
    ~/.venvs/lensjudge/bin/python site/guide_src/make_figures.py --only ch18-sie-caustics
    ~/.venvs/lensjudge/bin/python site/guide_src/make_figures.py --check
    ~/.venvs/lensjudge/bin/python site/guide_src/make_figures.py --list

``--check`` re-renders every figure to memory and compares against the committed
bytes. It is the regression gate, and it only works because ``_style`` pins
``svg.hashsalt`` and we drop the SVG date stamp — see that module's docstring.
A clean ``--check`` plus an empty ``git diff`` after a full run is the contract.

Figures are COMMITTED, not built in CI: the deploy workflow installs only
mkdocs + mkdocs-material + pandoc, and adding numpy/scipy/matplotlib/astropy
would add ~250 MB to every docs build to recompute figures that change ~never.

WHY THE FIGURES ARE COMPUTED, NOT DRAWN
---------------------------------------
Wherever a figure can be derived, it is: the caustics come from numerically
differentiating a deflection field and contouring det A = 0, not from an artist
placing a curve. The figure IS the worked example, so it cannot disagree with
the prose, and the reader can change q and re-render.

Every function registered with @figure returns a dict of named scalars it
wants recorded in the manifest (``worked_values``); those are the numbers the
prose is allowed to quote, and 20_verify_numbers.py checks them.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from pathlib import Path

import _style
import matplotlib
import matplotlib.pyplot as plt
from registry import REGISTRY as _REGISTRY

HERE = Path(__file__).resolve().parent
FIGDIR = HERE.parent / "docs" / "guide" / "figures"
MANIFEST = HERE / "figures.json"

# An SVG above this is almost always a contourf/hexbin/large-scatter blowup
# (measured: a 30-level contourf is 576 KB vs 33 KB for a contour). Demote to
# PNG rather than ship it. imshow is fine in SVG — the raster embeds as base64.
MAX_SVG_BYTES = 250_000

def _render(slug: str, scheme: str) -> tuple[bytes, str, dict]:
    """Render one (slug, scheme) to bytes. Returns (data, fmt, values)."""
    spec = _REGISTRY[slug]
    with matplotlib.rc_context(_style.rc(scheme)):
        fig, values = spec["fn"](scheme)
        buf = io.BytesIO()
        # metadata={"Date": None} strips the <dc:date> stamp -> reproducible.
        fig.savefig(buf, format="svg", metadata={"Date": None})
        data, fmt = buf.getvalue(), "svg"
        if len(data) > MAX_SVG_BYTES:
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=200)
            data, fmt = buf.getvalue(), "png"
        plt.close(fig)
    return data, fmt, values


def _render_pdf(slug: str) -> bytes:
    """Light-variant PDF, for the print target only.

    pdflatex cannot embed SVG, and site/filters/guide.lua rewrites .svg -> .pdf
    for LaTeX output. Only the light variant is needed: paper has one scheme.
    """
    with matplotlib.rc_context(_style.rc("light")):
        fig, _ = _REGISTRY[slug]["fn"]("light")
        buf = io.BytesIO()
        fig.savefig(buf, format="pdf", metadata={"CreationDate": None})
        plt.close(fig)
    return buf.getvalue()


def build(only: str | None = None, check: bool = False, pdf: bool = False) -> int:
    slugs = [only] if only else sorted(_REGISTRY)
    if only and only not in _REGISTRY:
        print(f"unknown figure: {only}", file=sys.stderr)
        return 2

    FIGDIR.mkdir(parents=True, exist_ok=True)
    manifest, failures = {}, []

    for slug in slugs:
        spec = _REGISTRY[slug]
        entry = dict(
            chapter=spec["chapter"],
            caption_hint=spec["caption_hint"],
            width=spec["width"],
            source_fn=spec["fn"].__name__,
            variants={},
        )
        for scheme in _style.SCHEMES:
            data, fmt, values = _render(slug, scheme)
            path = FIGDIR / f"{slug}-{scheme}.{fmt}"
            digest = hashlib.sha256(data).hexdigest()

            if check:
                if not path.exists():
                    failures.append(f"{path.name}: missing (run without --check)")
                elif hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                    failures.append(f"{path.name}: bytes differ from committed")
            else:
                # Drop a stale variant if the format flipped svg<->png.
                other = FIGDIR / f"{slug}-{scheme}.{'png' if fmt == 'svg' else 'svg'}"
                if other.exists():
                    other.unlink()
                path.write_bytes(data)

            entry["variants"][scheme] = dict(
                file=path.name, fmt=fmt, bytes=len(data), sha256=digest
            )
            entry["worked_values"] = {k: v for k, v in values.items()}

        if pdf and not check:
            (FIGDIR / f"{slug}-light.pdf").write_bytes(_render_pdf(slug))

        manifest[slug] = entry
        if not check:
            v = entry["variants"]
            sizes = "/".join(f"{v[s]['bytes'] // 1024}K" for s in _style.SCHEMES)
            print(f"  {slug:38s} {v['light']['fmt']:3s} {sizes}")

    if check:
        if failures:
            print("FAIL — figures are not reproducible:", file=sys.stderr)
            for f in failures:
                print(f"  {f}", file=sys.stderr)
            return 1
        print(f"OK — {len(slugs)} figures byte-identical to committed")
        return 0

    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"\n{len(slugs)} figures -> {FIGDIR}")
    print(f"manifest -> {MANIFEST}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--only", metavar="SLUG", help="build one figure")
    ap.add_argument("--check", action="store_true",
                    help="assert committed figures are byte-identical")
    ap.add_argument("--list", action="store_true", help="list registered figures")
    ap.add_argument("--pdf", action="store_true",
                    help="also emit <slug>-light.pdf for the print target")
    args = ap.parse_args()

    import figures  # noqa: F401  (registers everything via @figure)

    if args.list:
        for slug in sorted(_REGISTRY):
            s = _REGISTRY[slug]
            print(f"ch{s['chapter']:02d}  {slug:38s} {s['caption_hint']}")
        return 0
    return build(only=args.only, check=args.check, pdf=args.pdf)


if __name__ == "__main__":
    raise SystemExit(main())
