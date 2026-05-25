#!/usr/bin/env python3
"""Build the agentic-lensing onboarding Word document.

Pipeline:
1. Run pandoc to convert the Markdown source to .docx (handles headings, lists,
   tables, hyperlinks, code blocks robustly).
2. Open the .docx with python-docx and apply a few post-processing tweaks
   (set sensible margins, adjust paragraph spacing, ensure top-of-doc title block).
3. Write the final .docx in place.

Usage:
    python scripts/build_onboarding_docx.py

Requires: pandoc on PATH, python-docx in the active Python.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from docx import Document
from docx.shared import Inches, Pt

ROOT = Path(__file__).resolve().parent.parent
MD_SRC = ROOT / "plans" / "AGENTIC_LENSING_ONBOARDING_PLAN.md"
DOCX_OUT = ROOT / "plans" / "agentic_lensing_onboarding_plan.docx"


def run_pandoc(md_path: Path, docx_path: Path) -> None:
    if shutil.which("pandoc") is None:
        raise SystemExit("pandoc not found on PATH; install pandoc or convert manually")

    cmd = [
        "pandoc",
        str(md_path),
        "-o",
        str(docx_path),
        "--from=gfm",
        "--toc",
        "--toc-depth=2",
        "--standalone",
        "--metadata=title:Agentic Lensing Onboarding Report",
        "--metadata=author:Greg Benson",
        "--metadata=date:2026-05-25",
    ]
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def post_process(docx_path: Path) -> None:
    doc = Document(str(docx_path))

    for section in doc.sections:
        section.top_margin = Inches(0.9)
        section.bottom_margin = Inches(0.9)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)

    styles = doc.styles
    normal = styles["Normal"]
    normal.font.name = "Calibri"
    normal.font.size = Pt(11)
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.15

    for heading_name, size_pt in [
        ("Heading 1", 20),
        ("Heading 2", 16),
        ("Heading 3", 13),
        ("Heading 4", 11),
    ]:
        try:
            style = styles[heading_name]
            style.font.name = "Calibri"
            style.font.size = Pt(size_pt)
            style.paragraph_format.space_before = Pt(14 if size_pt >= 16 else 10)
            style.paragraph_format.space_after = Pt(6)
            style.paragraph_format.keep_with_next = True
        except KeyError:
            continue

    doc.save(str(docx_path))


def main() -> int:
    if not MD_SRC.exists():
        print(f"ERROR: Markdown source not found at {MD_SRC}", file=sys.stderr)
        return 1

    DOCX_OUT.parent.mkdir(parents=True, exist_ok=True)
    run_pandoc(MD_SRC, DOCX_OUT)
    post_process(DOCX_OUT)

    size_kb = DOCX_OUT.stat().st_size / 1024
    print(f"Wrote {DOCX_OUT} ({size_kb:.1f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
