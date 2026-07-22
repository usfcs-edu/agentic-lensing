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
    "current": ["lensjudge", "claudenet", "redshifty", "dr11-campaign",
                "claude-giga-lens", "claude-giga-lens-linus"],
    "reproductions": [
        "aion-1", "cikota-2023", "dawes-2022", "foundry-i", "foundry-ii",
        "foundry-iii", "foundry-iv", "gu-2022", "hsu-2025", "huang-2020",
        "huang-2021", "inchausti-2025", "sheu-2023", "sheu-2024a",
        "sheu-2024b", "silver-2025",
    ],
}

APPLE_SILICON_SLUGS = ["huang-2020", "huang-2021", "redshifty"]

# Additional documents beyond a slug's main.tex, published as their own pages:
# (section, source slug, tex basename, output slug).
EXTRA_REPORTS = [
    ("reproductions", "foundry-i", "evolution", "foundry-i-evolution"),
    ("current", "claudenet", "new_candidates", "claudenet-new-candidates"),
    ("current", "lensjudge", "residual", "lensjudge-residual"),
    ("current", "lensjudge", "human_baseline", "lensjudge-human-baseline"),
    ("current", "lensjudge", "parity", "lensjudge-parity"),
]

MARKERS = ("TITLE", "AUTHORS", "DATE", "ABSTRACT", "BODY")

# Pandoc-readable stand-ins for the tech-report.sty macros (pandoc must not
# read the real .sty: its \mbox-based \farcs leaks \mbox into math strings).
#
# NOTE: \rm is deliberately NOT shimmed to \mathrm. \rm is a font *declaration*
# scoped to its group ({\rm eff} -> all of "eff" upright), whereas \mathrm takes
# a single argument, so the shim turned {\rm eff} into \mathrm{e} + italic "ff".
# Left alone, \rm passes through to MathJax, which supports it natively.
SHIM = r"""
\newcommand{\subtitle}[1]{}
\newcommand{\thanks}[1]{}
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
# The \subtitle argument may itself carry one level of braced macros
# (\texttt{...}, \citet{...}), so match balanced-at-depth-1 rather than
# stopping at the first closing brace; subtitle_md() then converts the
# light LaTeX for direct markdown injection.
SUBTITLE = re.compile(r"\\subtitle\{((?:[^{}]|\{[^{}]*\})*)\}")
WIKILINK = re.compile(r"\[\[([^\]]+)\]\]")

# Most reports keep figures in papers/figures/, but some (claude-giga-lens)
# reference a sibling dir: \includegraphics{../figs/x.png}. That path would
# resolve ABOVE the page's output dir and 404, so the leading ../ is stripped
# and the referenced dir is copied in alongside the page (see copy_assets).
INCLUDEGRAPHICS_UP = re.compile(
    r"(\\includegraphics(?:\[[^\]]*\])?\{)\.\./(?!\.\.)([^}]+)\}"
)

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


# pandoc resolves \input{...}/\include{...} from disk at conversion time —
# AFTER this script's source-string fixes have run — so none of the regex
# passes in preprocess() (figure-path rewrites, resizebox, cmidrule, toolio)
# ever applied to \input'ed section files, and copy_assets() never saw their
# ../ figure dirs (this is how the 5 ClaudeNet campaign_section figures
# broke). inline_inputs() therefore splices sub-files into the source string
# (recursively, resolved relative to papers/) before anything else looks at
# it; build_report feeds the inlined string to preprocess() and copy_assets()
# alike.
INPUT_TEX = re.compile(r"\\(?:input|include)\{([^}]+)\}")


def inline_inputs(tex: str, papers: Path) -> str:
    def splice(m):
        line_start = tex.rfind("\n", 0, m.start()) + 1
        if re.search(r"(?<!\\)%", tex[line_start:m.start()]):
            return m.group(0)  # commented out (e.g. usage notes in headers)
        name = m.group(1)
        path = papers / (name if name.endswith(".tex") else name + ".tex")
        if not path.is_file():
            path = papers / name
        if not path.is_file():
            return m.group(0)  # not a papers/ sub-file; leave for pandoc
        return "\n%s\n" % inline_inputs(path.read_text(), papers)

    return INPUT_TEX.sub(splice, tex)


def preprocess(tex: str) -> str:
    tex = USEPACKAGE_STY.sub("", tex)
    tex = INCLUDEGRAPHICS_UP.sub(r"\1\2}", tex)
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


def subtitle_md(s: str) -> str:
    r"""Convert the \subtitle argument for markdown injection. It is
    regex-harvested and never goes through pandoc, so translate the light
    LaTeX it may carry: \texttt/\code -> code spans, \citet/\citep ->
    bracketed keys, ties and thin spaces -> spaces."""
    s = re.sub(r"\\(?:texttt|code)\{([^{}]*)\}", r"`\1`", s)
    s = re.sub(r"\\cite[tp]\*?\{([^{}]*)\}", r"[\1]", s)
    s = re.sub(r"\\[,;]", " ", s)
    s = s.replace("~", " ")
    return one_line(s)


def plain_text(s: str) -> str:
    s = re.sub(r"<[^>]+>", "", s)          # raw HTML tags
    s = s.replace("\\", "")                # stray escapes (\&, \$, ...)
    s = s.replace("$", "")                 # inline math delimiters (e.g. $-$)
    return one_line(s)


def buttons(gh_url: str, pdf_name: str = None) -> str:
    rows = []
    if pdf_name:
        rows.append("[:material-file-download: Download PDF](%s)"
                    "{ .md-button .md-button--primary }" % pdf_name)
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


def assemble(meta: dict, subtitle: str, slug: str,
             pdf_name: str = "main.pdf") -> str:
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
    parts += [buttons("%s/%s" % (GH, slug), pdf_name=pdf_name), ""]
    if meta.get("ABSTRACT"):
        parts += ["## Abstract", "", meta["ABSTRACT"], ""]
    parts += [meta.get("BODY", ""), ""]
    return "\n".join(parts)


def write_if_changed(path: Path, content: str) -> None:
    if not path.exists() or path.read_text() != content:
        path.write_text(content)


def copy_assets(papers: Path, out: Path, pdf_name: str = "main.pdf",
                tex: str = "") -> None:
    figures = papers / "figures"
    if figures.is_dir():
        shutil.copytree(figures, out / "figures", dirs_exist_ok=True)
    for pattern in ("*.png", "*.jpg", "*.jpeg"):
        for img in papers.glob(pattern):
            shutil.copy2(img, out / img.name)
    # Sibling figure dirs whose ../ prefix preprocess() stripped, e.g.
    # ../figs/x.png -> figs/x.png, published as <page>/figs/.
    for name in {m.group(2).split("/")[0]
                 for m in INCLUDEGRAPHICS_UP.finditer(tex)}:
        src = papers.parent / name
        if src.is_dir():
            shutil.copytree(src, out / name, dirs_exist_ok=True)
    shutil.copy2(papers / pdf_name, out / pdf_name)


def build_report(slug: str, section: str, texname: str = "main",
                 out_slug: str = None) -> None:
    papers = REPRO / slug / "papers"
    out = DOCS / section / (out_slug or slug)
    out.mkdir(parents=True, exist_ok=True)
    pdf_name = "%s.pdf" % texname

    tex = inline_inputs((papers / ("%s.tex" % texname)).read_text(), papers)
    subtitle_m = SUBTITLE.search(tex)
    md = run_pandoc(SHIM + preprocess(tex), papers)
    meta = parse_markers(md)
    page = assemble(meta,
                    subtitle_md(subtitle_m.group(1)) if subtitle_m else "",
                    slug, pdf_name=pdf_name)

    write_if_changed(out / "index.md", page)
    copy_assets(papers, out, pdf_name, tex=tex)


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
            lines[i + 1:i + 1] = ["", buttons(gh_url, pdf_name=None)]
            break
    else:
        lines[0:0] = [buttons(gh_url, pdf_name=None), ""]
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

    for section, slug, texname, out_slug in EXTRA_REPORTS:
        if args.only and slug not in args.only and out_slug not in args.only:
            continue
        print("[%s] %s" % (section, out_slug))
        try:
            build_report(slug, section, texname=texname, out_slug=out_slug)
        except Exception as exc:
            failures.append((out_slug, exc))
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
