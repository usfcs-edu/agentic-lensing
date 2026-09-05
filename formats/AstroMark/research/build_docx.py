#!/usr/bin/env python3
"""Build the AstroMark research report as a single Word document.

    ~/.venvs/workshop/bin/python build_docx.py [--out AstroMark-Research.docx]

Every Markdown file in this directory, in reading order, assembled into one document with the
figures from the presentation placed inline.

Two things this has to do that a plain `pandoc *.md` would not:

1. **Inject the figures.** The Markdown references images by PATH, in prose and tables — it never
   embeds them, because on disk the reader can open the file. In a single Word document that is
   useless, so figures are inserted at anchor points defined in FIGURES below, sized for the page.

2. **Neutralise cross-document links.** The sources link to each other (`[00-executive-summary.md]
   (00-executive-summary.md)`). Inside one document those become links to files that are not there,
   so they are flattened to plain text naming the section instead.

Post-processing reuses the workshop's `docx_post.py`: it sets `updateFields` so the table of
contents fills on first open, widens tables to the text column, and asserts that the inline-shape
count matches the number of image references — which is what catches a figure that failed to resolve.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
WORKSHOP_BUILD = HERE.resolve().parents[2] / "workshops/2026-08-31-LensMark/build"

# (source file, part heading to open with or None)
ORDER: list[tuple[str, str | None]] = [
    ("README.md", None),
    ("00-executive-summary.md", "Part I — The recommendation"),
    ("02-requirements.md", "Part II — What the format must do"),
    ("01-prior-art/README.md", "Part III — How other fields solved this"),
    ("01-prior-art/medical-imaging.md", None),
    ("01-prior-art/vocabularies-and-grading.md", None),
    ("01-prior-art/astronomy-overlays.md", None),
    ("01-prior-art/astronomy-metadata.md", None),
    ("01-prior-art/computer-vision-and-web.md", None),
    ("01-prior-art/notation-design.md", None),
    ("01-prior-art/accessibility.md", None),
    ("03-notation-proposals.md", "Part IV — The symbolic notation"),
    ("05-accessibility-design.md", None),
    ("04-metadata-proposals.md", "Part V — The metadata representation"),
    ("06-llm-ergonomics.md", None),
    ("09-jwst-examples.md", "Part VI — Worked examples on real JWST data"),
    ("07-recommendation.md", "Part VII — Recommendation"),
    ("08-draft-spec/astromark-core-1.0.md", "Part VIII — Draft specification"),
    ("08-draft-spec/astromark-lens-1.0.md", None),
    ("08-draft-spec/render-rules.md", None),
    ("naming-study.md", "Appendix A — The naming study"),
    ("qa/verification.md", "Appendix B — Verification and provenance"),
    ("qa/embargo-adjudication.md", None),
]

# figure -> (source file, anchor regex matched against a whole line, path, caption, width in)
# The figure is inserted after the paragraph the anchor line belongs to.
FIGURES: list[tuple[str, str, str, str, float]] = [
    ("00-executive-summary.md", r"^### 2\. ",
     "examples/reference/ab-page.png",
     "Every notation arm on the reference scene, with the two reference arms and the null arm. "
     "Normalised for how much each says, today's notation costs 3.2x more ink per unit of meaning "
     "than the best candidate.", 6.4),
    ("02-requirements.md", r"^## A\. What the notation must be able to SAY",
     "examples/reference/astromark-ref.png",
     "The synthetic reference scene. Built to contain every case the notation must express: a dust "
     "lane across the deflector, a second deflector, a bound satellite, a counter-image farther out "
     "than the arc, and a spiral arm that mimics an arc. Reproducible from a seed; no embargo.", 3.6),
    ("03-notation-proposals.md", r"^### N1 — Terminator alphabet",
     "examples/reference/n1.png",
     "N1, Terminator alphabet. Semantics live in the shape at the business end; polarity is the "
     "shaft texture, reinforced by a strike on negatives.", 4.6),
    ("03-notation-proposals.md", r"^### N2 — Bertin ledger",
     "examples/reference/n2.png",
     "N2, Bertin ledger. Tick orientation is derived from the radius vector: tangential means lensed "
     "light, which really is tangentially stretched.", 4.6),
    ("03-notation-proposals.md", r"^### N3 — Station model",
     "examples/reference/n3.png",
     "N3, Station model. One compact badge per object, carrying role, polarity, source index and "
     "treatment in slots.", 4.6),
    ("03-notation-proposals.md", r"^### N4 — Evidence graph",
     "examples/reference/n4.png",
     "N4, Evidence graph. Anchors, a source chord tying the images of one source, and stems to a "
     "mass dot. A struck chord means refuted.", 4.6),
    ("03-notation-proposals.md", r"^### Thumbnail thresholds",
     "examples/reference/contact-260.png",
     "The thumbnail test at 260 px per panel. Text is unreadable in every arm at this size, so only "
     "the visual channel remains.", 6.4),
    ("03-notation-proposals.md", r"^\| R-CURRENT \(LensMark today\)",
     "examples/reference/r-current.png",
     "The reference arm: today's LensMark notation at its own constants. The on-image legend plate "
     "is the single largest consumer of ink, and five of the twenty statements cannot be made at "
     "all.", 4.4),
    ("05-accessibility-design.md", r"^## 5\. Visual evidence",
     "examples/reference/polarity-triad.png",
     "The polarity triad: one document rendered three times, all marks positive, all negative, all "
     "ambiguous, with every label removed. Solid, dashed and dotted are unmistakable with no text.",
     6.4),
    ("05-accessibility-design.md", r"^- `examples/reference/cvd-sheet\.png`",
     "examples/reference/cvd-sheet.png",
     "Every arm under normal vision, deuteranopia, protanopia, tritanopia and greyscale.", 6.4),
    ("09-jwst-examples.md", r"^### rank 1 — ",
     "examples/jwst/J3440482-522486-n1.png",
     "Rank 1, J3440482-522486 — the published lens SL2S J02176-0513. The counter-image carries "
     "emphasis=key, hence the corner brackets; its label demoted to the index 1 because it sits too "
     "near the panel edge. The mark at upper right is negative polarity: the verifier's own "
     "correction that the NE streak is a field galaxy, not a second arc.", 4.2),
    ("09-jwst-examples.md", r"^### rank 2 — ",
     "examples/jwst/J15199556+2122210-n1.png",
     "Rank 2, J15199556+2122210 — a single tangential arc. The caption records counter-image NOT "
     "FOUND, which is a different statement from not searched.", 4.2),
    ("09-jwst-examples.md", r"^### rank 3 — ",
     "examples/jwst/J34707505-219476-wide-n1.png",
     "Rank 3, J34707505-219476, on the 10 arcsec field. The blue blob antipodal to the knot chain "
     "lies FARTHER out than the arc, so it is marked ambiguous with a named alternative, and the "
     "ring records that half-separation was declined.", 4.2),
    ("09-jwst-examples.md", r"^## The finding this produced",
     "examples/jwst/J3440482-522486-panels/colour_10.png",
     "One of the six panels the verifiers actually saw. The rank-3 counter-image at r = 1.88 arcsec "
     "falls outside the 3.5 arcsec zoom entirely, so the same annotation is unrenderable there — "
     "which is why a mark's coordinates are meaningless without a declared frame.", 3.6),
    ("07-recommendation.md", r"^### The measured case",
     "examples/reference/contact-528.png",
     "All seven arms side by side at 528 px per panel, including the null arm — the one that asks "
     "which marks earn their ink.", 6.4),
]

LINK = re.compile(r"(?<!!)\[([^\]]+)\]\((?!https?:)[^)]+\)")
FENCE = re.compile(r"^```.*?^```", re.S | re.M)
INLINE_CODE = re.compile(r"`[^`\n]*`")


def count_figures(text: str) -> int:
    """Image references, ignoring anything inside code.

    The report's own prose documents an image-syntax bug and therefore contains a literal
    `![caption](path)` inside a code span. A naive count of "![" sees that as a sixteenth figure and
    fails the assertion for the wrong reason — so code is stripped before counting.
    """
    stripped = INLINE_CODE.sub("", FENCE.sub("", text))
    return len(re.findall(r"!\[", stripped))


def flatten_links(text: str) -> str:
    """Cross-document links become plain text: inside one document they point at nothing."""
    return LINK.sub(lambda m: m.group(1), text)


def demote(text: str, by: int = 1) -> str:
    """Push every heading down so the part headings sit above them."""
    out = []
    fence = False
    for line in text.split("\n"):
        if line.startswith("```"):
            fence = not fence
        if not fence and re.match(r"^#{1,5} ", line):
            line = "#" * by + line
        out.append(line)
    return "\n".join(out)


def inject(src: str, text: str) -> tuple[str, int]:
    """Insert each figure for this source after the paragraph its anchor line belongs to."""
    n = 0
    for f_src, anchor, path, caption, width in FIGURES:
        if f_src != src:
            continue
        if not (HERE / path).is_file():
            print(f"  ! missing figure {path} — skipped", file=sys.stderr)
            continue
        rx = re.compile(anchor)
        lines = text.split("\n")
        for i, line in enumerate(lines):
            if not rx.search(line):
                continue
            j = i + 1                       # walk to the end of this block
            while j < len(lines) and lines[j].strip() and not lines[j].startswith("#"):
                j += 1
            fig = f"\n![{caption}]({path}){{width={width}in}}\n"
            lines.insert(j, fig)
            text = "\n".join(lines)
            n += 1
            break
        else:
            print(f"  ! anchor not found in {src}: {anchor}", file=sys.stderr)
    return text, n


TITLE = """---
title: "AstroMark — Research, Proposals and Recommendations"
subtitle: "A symbolic notation and metadata format for annotating astronomical images"
author: "Greg Benson, with Claude Code"
date: "4 September 2026"
---

> **Status: nothing in this document has been adopted.** Every recommendation is a recommendation
> for the team to accept, modify or reject. Where the evidence is thin, or the record is silent, the
> text says so rather than papering over it.
>
> Companion artifacts: `AstroMark-Research.pptx` presents the same material as a 59-slide
> walkthrough; `examples/` holds the code that generates every figure and number here, deterministic
> from a seed.

"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=HERE / "AstroMark-Research.docx")
    ap.add_argument("--keep-md", action="store_true", help="keep the assembled markdown")
    args = ap.parse_args()

    ref = WORKSHOP_BUILD / "reference.docx"
    post = WORKSHOP_BUILD / "docx_post.py"
    if not ref.is_file():
        print(f"build_docx: reference doc not found at {ref}", file=sys.stderr)
        return 1

    parts, n_fig, n_src = [TITLE], 0, 0
    for src, part in ORDER:
        p = HERE / src
        if not p.is_file():
            print(f"  ! missing source {src} — skipped", file=sys.stderr)
            continue
        text = p.read_text(encoding="utf-8")
        text, k = inject(src, text)
        n_fig += k
        text = flatten_links(demote(text))
        if part:
            parts.append(f"\n\n# {part}\n\n")
        parts.append(text.rstrip() + "\n")
        n_src += 1
        print(f"  {src:<44} {len(text):>7} chars  {k} figure(s)")

    md = HERE / ".astromark-report.md"
    md.write_text("\n".join(parts), encoding="utf-8")

    cmd = [
        "pandoc", str(md), "-o", str(args.out),
        f"--reference-doc={ref}",
        "--toc", "--toc-depth=2",
        f"--resource-path={HERE}",
        "--from", "markdown+pipe_tables+implicit_figures+link_attributes+yaml_metadata_block",
        "--standalone",
    ]
    print(f"\n==> pandoc {n_src} sources, {n_fig} figures -> {args.out.name}")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        print(r.stderr, file=sys.stderr)
        return 1
    if r.stderr.strip():
        print(r.stderr.strip(), file=sys.stderr)

    # Reuse the workshop's tested post-processing, but not its leftover-marker check: that scans
    # for "[[", and this report legitimately contains it inside JSON code blocks
    # ("points": [[0.851,0.358], ...]). Import the functions rather than shelling out, so the
    # table-widening and updateFields logic stays single-sourced.
    sys.path.insert(0, str(WORKSHOP_BUILD))
    try:
        import docx_post
    except ImportError as e:
        print(f"build_docx: could not import docx_post from {WORKSHOP_BUILD}: {e}", file=sys.stderr)
        return 1
    from docx import Document

    doc = Document(str(args.out))
    docx_post.set_update_fields(doc)
    n_tables = docx_post.widen_tables(doc)
    doc.save(str(args.out))

    doc = Document(str(args.out))
    paragraphs = list(docx_post.iter_paragraphs(doc))
    n_words = sum(len(p.text.split()) for p in paragraphs)
    n_shapes = len(doc.inline_shapes)
    n_refs = count_figures(md.read_text(encoding='utf-8'))
    leftovers = [p.text[:80] for p in paragraphs
                 if re.search(r"\bTODO\b|\bTBD\b", p.text)]
    problems = []
    if not (n_shapes == n_refs == n_fig):
        problems.append(f"figure count mismatch: {n_fig} injected, {n_refs} references in the "
                        f"markdown, {n_shapes} shapes in the document — they must all agree, "
                        f"or a figure was eaten before pandoc saw it or failed to resolve after")
    problems += [f"leftover marker: {t}" for t in leftovers]
    if problems:
        for x in problems:
            print("  - " + x, file=sys.stderr)
        return 1
    print(f"docx OK: {args.out.name} — {n_words:,} words, {len(paragraphs):,} paragraphs, "
          f"{n_shapes} figures, {n_tables} tables widened, TOC fills on open")
    if not args.keep_md:
        md.unlink()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
