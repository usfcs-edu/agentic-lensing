#!/usr/bin/env python3
"""Build the standalone PDF and offline HTML of the guide from the SAME markdown
the mkdocs site serves.

    python3 site/build_guide.py --html --pdf

One source, three targets:

    site/docs/guide/*.md  --mkdocs-->  the public site        (no transform)
                          --this-->    guide/guide.html       (offline, MathML)
                          --this-->    guide/guide.pdf        (pandoc -> LaTeX)

WHY A PREPROCESSOR IS NEEDED (all verified by running it, not assumed)
----------------------------------------------------------------------
* pandoc does NOT understand mkdocs-material's `!!! note` / `??? question`
  syntax. It passes the marker through as literal text AND mangles the math
  inside (``|`` becomes ``\\textbar{}``). So the blocks must be rewritten to
  pandoc's native fenced divs before pandoc sees them.
* pandoc does NOT parse ``\\(...\\)`` without ``tex_math_single_backslash``.
  ``$...$`` parses correctly in pandoc AND is what arithmatex wants for the
  site AND converts to native MathML. Hence the contract's math rule.
* Material's ``#only-light`` / ``#only-dark`` trick is CSS keyed on ``src$=``.
  Outside Material, browsers ignore the fragment and render BOTH images,
  stacked. So for print/offline we drop the dark variant and de-suffix the light.

Stdlib only, mirroring site/build_reports.py. Requires pandoc >= 3.1.7 and, for
--pdf, a TeX Live install (pdflatex).
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
GUIDE = REPO / "site" / "docs" / "guide"
FIGDIR = GUIDE / "figures"
OUTDIR = REPO / "guide"
FILTER = REPO / "site" / "filters" / "guide.lua"
MIN_PANDOC = (3, 1, 7)

# `!!! type "Title"` / `??? type "Title"` / `???+ type "Title"`
BLOCK = re.compile(r'^(?P<ind>\s*)(?P<mark>!!!|\?\?\?\+?)\s+(?P<type>[\w-]+)'
                   r'(?:\s+"(?P<title>[^"]*)")?\s*$')

TITLES = {"note": "Note", "tip": "You already know this", "abstract": "Summary",
          "question": "Exercise", "success": "Solution", "warning": "Warning",
          "danger": "Danger", "info": "Info", "example": "Example",
          "quote": "Quote", "bug": "Bug", "failure": "Failure"}


def check_pandoc() -> None:
    try:
        out = subprocess.run(["pandoc", "--version"], capture_output=True,
                             text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        sys.exit("pandoc not found on PATH")
    ver = tuple(int(x) for x in out.split()[1].split(".")[:3])
    if ver < MIN_PANDOC:
        sys.exit(f"pandoc {'.'.join(map(str, ver))} < required "
                 f"{'.'.join(map(str, MIN_PANDOC))}")


def blocks_to_divs(text: str) -> str:
    """Rewrite material admonition/details blocks into pandoc fenced divs.

    Indentation-based and recursive: a `??? success` nested inside a
    `??? question` becomes a nested div, which is exactly what the Lua filter
    needs to render an exercise with its solution.
    """
    lines = text.split("\n")
    out: list[str] = []
    i = 0
    while i < len(lines):
        m = BLOCK.match(lines[i])
        if not m:
            out.append(lines[i])
            i += 1
            continue

        ind = m.group("ind")
        typ = m.group("type")
        title = m.group("title") or TITLES.get(typ, typ.title())
        base = len(ind)

        # Collect the indented body (blank lines belong to the block).
        i += 1
        body: list[str] = []
        while i < len(lines):
            ln = lines[i]
            if not ln.strip():
                body.append("")
                i += 1
                continue
            if len(ln) - len(ln.lstrip()) <= base:
                break
            body.append(ln[base + 4:] if len(ln) > base + 4 else "")
            i += 1
        while body and not body[-1].strip():
            body.pop()

        esc = title.replace('"', "'")
        out.append(f'{ind}::: {{.admonition .{typ} title="{esc}"}}')
        out.extend(ind + b if b else "" for b in blocks_to_divs("\n".join(body)).split("\n"))
        out.append(f"{ind}:::")
        out.append("")
    return "\n".join(out)


def single_variant_figures(text: str) -> str:
    """Drop the #only-dark image and de-suffix the #only-light one.

    Outside mkdocs-material the fragment is inert, so leaving both would stack
    two copies of every figure in the PDF and the offline HTML.
    """
    text = re.sub(r'^\s*!\[[^\]]*\]\([^)]*#only-dark\)(\{[^}]*\})?\s*$\n?',
                  "", text, flags=re.M)
    text = re.sub(r'(!\[[^\]]*\]\([^)]*?)#only-light(\))', r"\1\2", text)
    return text


FIGURE_HTML = re.compile(
    r'<figure[^>]*>\s*'
    r'!\[(?P<alt>[^\]]*)\]\((?P<src>[^)]+)\)(?P<attr>\{[^}]*\})?\s*'
    r'<figcaption[^>]*>(?P<cap>.*?)</figcaption>\s*'
    r'</figure>', re.S)


def html_figure_to_markdown(text: str) -> str:
    """Collapse the site's <figure> wrapper into a plain image with the caption
    as alt text.

    mkdocs-material needs `<figure markdown="span">` + `<figcaption>`. pandoc
    does not: given a paragraph containing only an image, it builds a proper
    figure and uses the ALT TEXT as the caption. Left alone it therefore emits
    TWO captions — its own from the alt text, plus the raw <figcaption> as an
    orphaned paragraph that page-breaks away from the image. Fold them into one.

    Run AFTER single_variant_figures(), so only the light image remains.
    """
    def sub(m):
        cap = " ".join(m.group("cap").split())
        # The site has no figure autonumbering, so chapters hand-write a
        # "**Figure 18.1.**" prefix. pandoc DOES autonumber, so keeping it
        # yields "Figure 1: **Figure 18.1.** ...". Drop ours; let pandoc count.
        cap = re.sub(r"^\*\*Figure\s+[\d.]+\.?\*\*\s*", "", cap)
        attr = m.group("attr") or ""
        return f'![{cap}]({m.group("src")}){attr}'
    return FIGURE_HTML.sub(sub, text)


def strip_site_only(text: str) -> str:
    """Remove idioms that mean nothing off the site."""
    text = re.sub(r'\{\s*\.md-button[^}]*\}', "", text)      # button styling
    # NB: the `<!-- check: ... -->` number tags are deliberately NOT stripped.
    # pandoc already does the right thing — it drops a real comment for LaTeX,
    # keeps it invisible in HTML, and PRESERVES one quoted inside backticks.
    # Regex-stripping them broke exactly that last case: Ch. 1 documents the tag
    # syntax in a code span, and deleting the tag from inside the span unpaired
    # the backticks and mangled the rest of the paragraph.
    # Chapter-to-chapter links collapse into one document. The target anchor must
    # carry the TARGET chapter's namespace (see namespace_anchors), not the
    # citing chapter's — so resolve it here, while we still know the target.
    #   [x](17-lens-equation.md#the-lens-equation) -> [x](#17-lens-equation-the-lens-equation)
    #   [x](17-lens-equation.md)                   -> [x](#17-lens-equation)
    text = re.sub(
        r'\]\((\d\d-[\w-]+)\.md(?:#([\w:-]+))?\)',
        lambda m: f"](#{m.group(1)}-{m.group(2)})" if m.group(2)
        else f"](#{m.group(1)})",
        text)
    # Report links become absolute URLs — those pages are not in this document.
    text = re.sub(r'\]\(\.\./current/([\w-]+)/index\.md(#[\w:-]+)?\)',
                  r"](https://usfcs-edu.github.io/agentic-lensing/current/\1/\2)", text)
    return text


def chapter_files() -> list[Path]:
    return sorted(p for p in GUIDE.glob("*.md") if re.match(r"\d\d-", p.name))


def namespace_anchors(text: str, slug: str) -> str:
    """Prefix every heading anchor and internal link with the chapter slug.

    On the site each chapter is its own page, so `{ #connect }` in 29 chapters
    is 29 distinct URLs. Concatenated into ONE document they collide — pandoc
    warns "Duplicate identifier 'connect'" and every cross-link to #connect
    silently resolves to chapter 1's. So `#connect` becomes `#01-orientation-connect`.
    """
    text = re.sub(r"^(#{2,6}) (.*?)\s*\{\s*#([\w-]+)\s*\}\s*$",
                  lambda m: f"{m.group(1)} {m.group(2)} {{#{slug}-{m.group(3)}}}",
                  text, flags=re.M)

    # Same-chapter links are bare `#anchor` in the source. Cross-chapter links
    # were already resolved to `#<target-slug>-<anchor>` by strip_site_only, so
    # skip anything that already starts with a chapter slug.
    known = {p.stem for p in chapter_files()}

    def fix(m):
        a = m.group(1)
        if any(a == k or a.startswith(k + "-") for k in known):
            return m.group(0)          # already namespaced
        return f"](#{slug}-{a})"

    return re.sub(r"\]\(#([\w-]+)\)", fix, text)


def assemble() -> str:
    parts = []
    for p in chapter_files():
        slug = p.stem
        t = p.read_text()
        t = strip_site_only(t)          # cross-chapter links -> #anchor
        t = single_variant_figures(t)   # drop the dark twin first...
        t = html_figure_to_markdown(t)  # ...then fold <figure> into one image
        t = namespace_anchors(t, slug)  # ...then de-collide the anchors
        t = blocks_to_divs(t)
        # Demote H1 -> H2 so the whole guide is one document with one title.
        t = re.sub(r"^# ", "## ", t, flags=re.M)
        parts.append(t)
    return "\n\n\\newpage\n\n".join(parts)


def build(fmt: str, body: str) -> Path:
    OUTDIR.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "guide.md"
        src.write_text(body)
        common = [
            "pandoc", str(src),
            # fenced_divs / link_attributes / tex_math_dollars / header_attributes
            # are all ON by default in pandoc's `markdown`; named explicitly so a
            # future reader can see what the pipeline actually depends on.
            "--from", "markdown+fenced_divs+link_attributes+tex_math_dollars"
                      "+header_attributes",
            "--standalone", "--toc", "--toc-depth=2",
            "--resource-path", f"{GUIDE}:{FIGDIR}",
            "--metadata", "title=From Calculus to the Money Number",
            "--metadata", "subtitle=The astrophysics, cosmology and mathematics behind this repository",
            "--metadata", "author=Agentic Lensing — USF Computer Science",
        ]
        if FILTER.exists():
            common += ["--lua-filter", str(FILTER)]

        if fmt == "html":
            out = OUTDIR / "guide.html"
            # --mathml, not --mathjax: no CDN, so the file works offline.
            cmd = common + ["--to", "html5", "--mathml", "--embed-resources",
                            "--css", str(REPO / "site" / "guide_src" / "guide_print.css"),
                            "-o", str(out)]
        else:
            out = OUTDIR / "guide.pdf"
            # xelatex, not pdflatex: the prose is full of Unicode that pdflatex
            # rejects outright — U+2212 MINUS in object names (DESI−165.4754),
            # M⊙, ×, é, en/em dashes. xelatex handles them natively instead of
            # needing a \DeclareUnicodeCharacter line per glyph, forever.
            cmd = common + ["--to", "pdf", "--pdf-engine=xelatex",
                            "-V", "geometry:margin=1in", "-V", "fontsize=11pt",
                            "-V", "colorlinks=true", "-V", "linkcolor=teal",
                            "-V", "urlcolor=teal",
                            "-o", str(out)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stderr[-3000:], file=sys.stderr)
            sys.exit(f"pandoc failed for {fmt}")
        if r.stderr.strip():
            print(f"  (pandoc warnings for {fmt}: "
                  f"{len(r.stderr.strip().splitlines())} lines)")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--html", action="store_true", help="build guide/guide.html")
    ap.add_argument("--pdf", action="store_true", help="build guide/guide.pdf")
    ap.add_argument("--dump", metavar="PATH", help="write the preprocessed markdown")
    args = ap.parse_args()
    if not (args.html or args.pdf or args.dump):
        ap.print_help()
        return 0

    check_pandoc()
    files = chapter_files()
    if not files:
        sys.exit(f"no chapters found in {GUIDE}")
    print(f"assembling {len(files)} chapters")
    body = assemble()

    if args.dump:
        Path(args.dump).write_text(body)
        print(f"-> {args.dump}")
    if args.html:
        print(f"-> {build('html', body)}")
    if args.pdf:
        if not shutil.which("xelatex"):
            sys.exit("xelatex not found; skip --pdf or install TeX Live")
        print(f"-> {build('pdf', body)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
