#!/usr/bin/env python3
"""Build mkdocs pages from the LaTeX tech reports and Apple Silicon READMEs.

For each report slug, converts reproductions/<slug>/papers/main.tex to
site/docs/<section>/<slug>/index.md via pandoc (gfm + MathJax math +
citeproc bibliography), copies the figures and the tracked main.pdf, and
prepends a header block with PDF-download and view-on-GitHub buttons.

Apple Silicon pages are Markdown-native: README_APPLE_SILICON.md is copied
to site/docs/other/apple-silicon/<slug>.md with a GitHub button prepended.

Stdlib only. Requires pandoc >= 3.1.7 on PATH (validated against 3.8.3).
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
REPRO = REPO / "reproductions"
DOCS = REPO / "site" / "docs"
LUA = REPO / "site" / "filters" / "mkdocs.lua"
TPL = REPO / "site" / "templates" / "page.tpl"
GH = "https://github.com/usfcs-edu/agentic-lensing/tree/main/reproductions"
TARGET = "gfm+tex_math_dollars-tex_math_gfm"
MIN_PANDOC = (3, 1, 7)  # first release with the tex_math_gfm extension

SLUGS = {
    "current": ["lensjudge", "claudenet", "redshifty"],
    "reproductions": [
        "aion-1", "cikota-2023", "dawes-2022", "foundry-i", "foundry-ii",
        "foundry-iii", "foundry-iv", "gu-2022", "hsu-2025", "huang-2020",
        "huang-2021", "inchausti-2025", "sheu-2023", "sheu-2024a",
        "sheu-2024b", "silver-2025",
    ],
}

APPLE_SILICON_SLUGS = ["huang-2020", "huang-2021", "redshifty"]

MARKERS = ("TITLE", "AUTHORS", "DATE", "ABSTRACT", "BODY")

# Pandoc-readable stand-ins for the tech-report.sty macros (pandoc must not
# read the real .sty: its \mbox-based \farcs leaks \mbox into math strings).
SHIM = r"""
\newcommand{\subtitle}[1]{}
\newcommand{\thanks}[1]{}
\newcommand{\rm}{\mathrm}
\newcommand{\AUC}{\ensuremath{\mathrm{AUC}}}
\newcommand{\dd}{\mathrm{d}}
\newcommand{\code}[1]{\texttt{#1}}
\newcommand{\addbibresource}[1]{}
\newcommand{\keywords}[1]{}
\newcommand{\software}[1]{\par\textit{Software:} #1\par}
\newcommand{\facility}[1]{\par\textit{Facility:} #1\par}
\newcommand{\facilities}[1]{\par\textit{Facilities:} #1\par}
\newcommand{\farcs}{\ensuremath{.\!\!^{\prime\prime}}}
\newcommand{\fdg}{\ensuremath{.\!\!^{\circ}}}
\newcommand{\farcm}{\ensuremath{.\!\!^{\prime}}}
\newcommand{\degr}{\ensuremath{^{\circ}}}
\newcommand{\arcsec}{\ensuremath{^{\prime\prime}}}
\newcommand{\arcmin}{\ensuremath{^{\prime}}}
\newenvironment{acknowledgments}{\section*{Acknowledgments}}{}
"""

# lensjudge transcript boxes: \begin{toolio}[opts]{title} body \end{toolio}
# (a \newtcblisting environment pandoc would otherwise LaTeX-mangle).
TOOLIO = re.compile(
    r"\\begin\{toolio\}(\[[^\]]*\])?\{((?:[^{}]|\{[^{}]*\})*)\}\n?"
    r"(.*?)\\end\{toolio\}",
    re.S,
)

USEPACKAGE_STY = re.compile(r"\\usepackage\{\.\./\.\./tech-report\}")
SUBTITLE = re.compile(r"\\subtitle\{([^}]*)\}")
WIKILINK = re.compile(r"\[\[([^\]]+)\]\]")

# \resizebox{\linewidth}{!}{ <tabular> } hides the tabular from pandoc's
# table parser, so the table loses its caption/label association.
RESIZEBOX = re.compile(
    r"\\resizebox\{[^{}]*\}\{[^{}]*\}\{%?\s*(.*?\\end\{tabular\})\s*\}", re.S
)

# pandoc doesn't understand booktabs partial rules and leaks their arguments
# into the adjacent header cell (e.g. "2-3(lr)4-5 sys").
CMIDRULE = re.compile(r"\\cmidrule(?:\([^)]*\))?\{[^}]*\}")

# Pandoc resolves \ref against body targets but not inside the abstract
# (metadata). Harvest the resolved numbers from the body instead.
REF_LINK = re.compile(
    r'<a href="#([^"]+)"[^>]*data-reference-type="(?:ref|eqref)"[^>]*>([^<]*)</a>'
)


def check_pandoc():
    try:
        out = subprocess.run(
            ["pandoc", "--version"], capture_output=True, text=True, check=True
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        sys.exit("error: pandoc not found on PATH (need >= %s)"
                 % ".".join(map(str, MIN_PANDOC)))
    m = re.match(r"pandoc(?:\.exe)?\s+(\d+)\.(\d+)(?:\.(\d+))?", out)
    version = tuple(int(g or 0) for g in m.groups()) if m else (0, 0, 0)
    if version < MIN_PANDOC:
        sys.exit(
            "error: pandoc %s is too old (need >= %s for --citeproc and "
            "the tex_math_gfm extension)"
            % (".".join(map(str, version)), ".".join(map(str, MIN_PANDOC)))
        )


def preprocess(tex: str) -> str:
    tex = USEPACKAGE_STY.sub("", tex)
    tex = RESIZEBOX.sub(r"\1", tex)
    tex = CMIDRULE.sub("", tex)
    tex = TOOLIO.sub(
        lambda m: "\\paragraph{%s}\n\\begin{verbatim}\n%s\n\\end{verbatim}"
        % (m.group(2), m.group(3).strip()),
        tex,
    )
    return tex


def run_pandoc(src: str, papers: Path) -> str:
    cmd = [
        "pandoc", "-f", "latex", "-t", TARGET,
        "--citeproc", "--bibliography", str(papers / "references.bib"),
        "--metadata", "link-citations=true",
        "--mathjax",
        "--shift-heading-level-by=1",
        "--lua-filter", str(LUA),
        "--template", str(TPL),
        "--wrap=none",
    ]
    proc = subprocess.run(
        cmd, input=src, cwd=papers, capture_output=True, text=True
    )
    if proc.returncode != 0:
        raise RuntimeError("pandoc failed in %s:\n%s" % (papers, proc.stderr))
    if proc.stderr.strip():
        print("  pandoc warnings:\n%s" % proc.stderr.rstrip(), file=sys.stderr)
    return proc.stdout


def parse_markers(out: str) -> dict:
    parts, current = {}, None
    for line in out.split("\n"):
        stripped = line.strip()
        if stripped.startswith("<<<") and stripped.endswith(">>>") and \
                stripped[3:-3] in MARKERS:
            current = stripped[3:-3]
            parts[current] = []
        elif current is not None:
            parts[current].append(line)
    return {k: "\n".join(v).strip() for k, v in parts.items()}


def one_line(s: str) -> str:
    return re.sub(r"\s*\n\s*", " ", s).strip()


def plain_text(s: str) -> str:
    s = re.sub(r"<[^>]+>", "", s)          # raw HTML tags
    s = s.replace("\\", "")                # stray escapes (\&, \$, ...)
    s = s.replace("$", "")                 # inline math delimiters (e.g. $-$)
    return one_line(s)


def buttons(gh_url: str, pdf: bool) -> str:
    rows = []
    if pdf:
        rows.append("[:material-file-download: Download PDF](main.pdf)"
                    "{ .md-button .md-button--primary }")
    rows.append("[:material-github: View on GitHub](%s){ .md-button }" % gh_url)
    return "\n".join(rows)


def resolve_abstract_refs(abstract: str, body: str) -> str:
    resolved = {
        target: text
        for target, text in REF_LINK.findall(body)
        if text and not text.startswith("[")
    }

    def fix(m):
        target, text = m.group(1), m.group(2)
        if text.startswith("[") and target in resolved:
            return '<a href="#%s">%s</a>' % (target, resolved[target])
        return m.group(0)

    return REF_LINK.sub(fix, abstract)


def assemble(meta: dict, subtitle: str, slug: str) -> str:
    if meta.get("ABSTRACT") and meta.get("BODY"):
        meta["ABSTRACT"] = resolve_abstract_refs(meta["ABSTRACT"], meta["BODY"])
    title_md = one_line(meta.get("TITLE", slug))
    title_plain = plain_text(title_md)
    byline_bits = [b for b in
                   ("**%s**" % meta["AUTHORS"] if meta.get("AUTHORS") else "",
                    one_line(meta.get("DATE", "")))
                   if b]
    parts = [
        "---",
        "title: %s" % json.dumps(title_plain),
        "---",
        "",
        "# %s" % title_md,
        "",
    ]
    if subtitle:
        parts += ["*%s*" % subtitle, ""]
    if byline_bits:
        parts += [" · ".join(byline_bits), ""]
    parts += [buttons("%s/%s" % (GH, slug), pdf=True), ""]
    if meta.get("ABSTRACT"):
        parts += ["## Abstract", "", meta["ABSTRACT"], ""]
    parts += [meta.get("BODY", ""), ""]
    return "\n".join(parts)


def write_if_changed(path: Path, content: str) -> None:
    if not path.exists() or path.read_text() != content:
        path.write_text(content)


def copy_assets(papers: Path, out: Path) -> None:
    figures = papers / "figures"
    if figures.is_dir():
        shutil.copytree(figures, out / "figures", dirs_exist_ok=True)
    for pattern in ("*.png", "*.jpg", "*.jpeg"):
        for img in papers.glob(pattern):
            shutil.copy2(img, out / img.name)
    shutil.copy2(papers / "main.pdf", out / "main.pdf")


def build_report(slug: str, section: str) -> None:
    papers = REPRO / slug / "papers"
    out = DOCS / section / slug
    out.mkdir(parents=True, exist_ok=True)

    tex = (papers / "main.tex").read_text()
    subtitle_m = SUBTITLE.search(tex)
    md = run_pandoc(SHIM + preprocess(tex), papers)
    meta = parse_markers(md)
    page = assemble(meta, subtitle_m.group(1) if subtitle_m else "", slug)

    write_if_changed(out / "index.md", page)
    copy_assets(papers, out)


def build_apple_silicon(slug: str) -> None:
    src = REPRO / slug / "apple-silicon" / "README_APPLE_SILICON.md"
    out = DOCS / "other" / "apple-silicon"
    out.mkdir(parents=True, exist_ok=True)

    text = WIKILINK.sub(r"\1", src.read_text())
    gh_url = "%s/%s/apple-silicon" % (GH, slug)
    lines = text.split("\n")
    # Insert the GitHub button right after the README's own H1 title.
    for i, line in enumerate(lines):
        if line.startswith("# "):
            lines[i + 1:i + 1] = ["", buttons(gh_url, pdf=False)]
            break
    else:
        lines[0:0] = [buttons(gh_url, pdf=False), ""]
    write_if_changed(out / ("%s.md" % slug), "\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", action="append", metavar="SLUG",
                        help="build only this slug (repeatable)")
    args = parser.parse_args()

    check_pandoc()

    failures = []
    for section, slugs in SLUGS.items():
        for slug in slugs:
            if args.only and slug not in args.only:
                continue
            print("[%s] %s" % (section, slug))
            try:
                build_report(slug, section)
            except Exception as exc:  # keep going; report all failures at end
                failures.append((slug, exc))
                print("  FAILED: %s" % exc, file=sys.stderr)

    for slug in APPLE_SILICON_SLUGS:
        if args.only and slug not in args.only:
            continue
        print("[other/apple-silicon] %s" % slug)
        try:
            build_apple_silicon(slug)
        except Exception as exc:
            failures.append(("apple-silicon/%s" % slug, exc))
            print("  FAILED: %s" % exc, file=sys.stderr)

    if failures:
        print("\n%d page(s) failed: %s"
              % (len(failures), ", ".join(s for s, _ in failures)),
              file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
