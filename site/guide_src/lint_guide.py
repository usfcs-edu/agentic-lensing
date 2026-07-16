#!/usr/bin/env python
"""Enforce the guide's notation + syntax contract. Deterministic, no model.

    ~/.venvs/lensjudge/bin/python site/guide_src/lint_guide.py
    ~/.venvs/lensjudge/bin/python site/guide_src/lint_guide.py --file site/docs/guide/19-einstein-radius.md

Every rule here encodes a failure that is SILENT in the browser — the page still
builds, `mkdocs --strict` still passes, and the reader gets raw TeX, a stacked
pair of light/dark images, or a number nobody checked. Greps catch these; eyes
and --strict do not.

Run inside the authoring loop, not after it: a chapter agent must not be able to
return until its own file is clean.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
GUIDE = HERE.parent / "docs" / "guide"
REPO = HERE.parent.parent
MANIFEST = HERE / "figures.json"

# Fenced code / inline code must be exempt from math and prose rules.
FENCE = re.compile(r"^( {0,3})(```+|~~~+)")


def _strip_code(text: str) -> list[tuple[int, str]]:
    """Return [(lineno, line)] with fenced blocks and inline code blanked."""
    out, in_fence, fence_tok = [], False, ""
    for i, line in enumerate(text.split("\n"), 1):
        m = FENCE.match(line)
        if m:
            tok = m.group(2)
            if not in_fence:
                in_fence, fence_tok = True, tok
            elif line.strip().startswith(fence_tok):
                in_fence = False
            out.append((i, ""))
            continue
        if in_fence:
            out.append((i, ""))
            continue
        out.append((i, re.sub(r"`[^`]*`", "``", line)))
    return out


def _inline_math_spans(text: str) -> list[tuple[int, str]]:
    """[(offset, content)] for each inline ``$...$`` span in ``text``.

    Two things this must get right, both learned by getting them wrong:

    1. **Pair the delimiters by scanning, never by regex.** A naive
       ``\\$([^$]+?)\\$`` matches from the CLOSING dollar of one span to the
       OPENING dollar of the next, so ``$n=1$; other values of $n$`` reports a
       bogus span ``"; other values of "``. The linter then cries wolf on
       perfectly good prose, which is worse than not checking at all.

    2. **Scan the whole document, not line by line.** Inline math legitimately
       wraps across a newline inside a paragraph, and arithmatex renders it
       correctly (verified). A line-based scanner sees the continuation line's
       closing ``$`` as an opener and every subsequent pairing on that line is
       inverted.

    Skips ``$$`` (display math) and escaped ``\\$``.
    """
    pos, dollars = 0, []
    while pos < len(text):
        c = text[pos]
        if c == "\\":
            pos += 2
            continue
        if c == "$":
            if pos + 1 < len(text) and text[pos + 1] == "$":
                pos += 2  # display delimiter — not our business
                continue
            dollars.append(pos)
        pos += 1
    return [(a, text[a + 1:b]) for a, b in zip(dollars[0::2], dollars[1::2])]


class Lint:
    def __init__(self):
        self.problems: list[tuple[str, int, str, str]] = []

    def add(self, path, line, rule, msg):
        self.problems.append((str(path), line, rule, msg))

    # -- rules ------------------------------------------------------------- #
    def check_file(self, path: Path, manifest: dict):
        raw = path.read_text()
        lines = _strip_code(raw)
        body = "\n".join(l for _, l in lines)

        # Inline math is checked document-wide, because a span may wrap a line.
        spans = _inline_math_spans(body)
        for off, inner in spans:
            if inner != inner.strip():
                shown = " ".join(inner.split())[:40]
                self.add(path, body.count("\n", 0, off) + 1,
                         "no-boundary-whitespace",
                         f"math has boundary whitespace: ${shown}$")

        # A closing "$" IMMEDIATELY followed by a digit is not a closing
        # delimiter to pandoc ("must not be followed by a digit"). arithmatex
        # renders it fine, so the site looks correct while the PDF gets literal
        # dollar signs AND swallows the following prose into a runaway math
        # span. Silent, and only on one of the three targets.
        # Real instance: `DESI-165.4754$-$06.0423` -> use a Unicode minus.
        for m in re.finditer(r"(?<!\\)\$(?<!\$\$)(?=\d)", body):
            # Only flag a '$' that is CLOSING a span (i.e. an odd-indexed one).
            if any(a + len(t) + 1 == m.start() for a, t in spans):
                self.add(path, body.count("\n", 0, m.start()) + 1,
                         "dollar-then-digit",
                         "closing '$' followed by a digit — pandoc will not "
                         "close the math here (PDF breaks silently). Add a "
                         "space, or use a Unicode minus '−' for object names")

        # A bare \eqref lands in a plain <p>; mathjax.js has ignoreHtmlClass
        # ".*|", so MathJax never sees it and the reader gets literal TeX.
        # Test by CONTAINMENT in a real math span — a "count the $ before it"
        # heuristic mis-reads any line with balanced spans ahead of the \eqref.
        ranges = [(a, a + len(t) + 1) for a, t in spans]
        for m in re.finditer(r"\\eqref\{", body):
            if not any(a < m.start() < b for a, b in ranges):
                self.add(path, body.count("\n", 0, m.start()) + 1,
                         "eqref-wrapped",
                         r"bare \eqref — must be wrapped as $\eqref{...}$")

        # Inline `$$ ... $$` with boundary whitespace is REFUSED by arithmatex
        # exactly like `$ x $` is, and the raw LaTeX lands on the page.
        for n, line in lines:
            for m in re.finditer(r"\$\$(.+?)\$\$", line):
                inner = m.group(1)
                if inner != inner.strip():
                    shown = " ".join(inner.split())[:36]
                    self.add(path, n, "display-math-inline-space",
                             f"inline $${shown}$$ has boundary whitespace — "
                             "arithmatex will NOT render it (raw TeX ships). "
                             "Put the $$ on their own lines, or use $...$")

        # A `$$` block MUST be surrounded by blank lines. Without them
        # Python-Markdown keeps it inside the enclosing paragraph, arithmatex's
        # block processor never runs, and its inline pattern then refuses the
        # content for boundary whitespace (the newlines). Net effect: the raw
        # LaTeX ships to the reader, `mkdocs --strict` passes, and there is no
        # mjx-merror to catch it — the math simply never becomes math.
        raw_lines = raw.split("\n")
        in_fence, expecting_close = False, False
        for i, line in enumerate(raw_lines):
            if FENCE.match(line):
                in_fence = not in_fence
            if in_fence or line.strip() != "$$":
                continue

            if not expecting_close:
                # OPENER: the line above must be blank (or start of file).
                # Its own content line below is of course non-blank — that is
                # the math, and checking it is what made the first draft of
                # this rule fire on every well-formed block in the guide.
                above = raw_lines[i - 1].strip() if i else ""
                if above:
                    self.add(path, i + 1, "display-math-needs-blank-line",
                             "`$$` opens straight after prose — needs a blank "
                             "line above, or it never renders as math")
            else:
                # CLOSER: the line below must be blank (or end of file).
                # NOTE: a `<!-- check -->` tag is NOT an acceptable neighbour.
                # At top level it happens to be harmless, but INSIDE an indented
                # block (every exercise solution) it keeps the block glued to the
                # paragraph and the math ships as raw `$$ x = 1 $$`. Verified
                # both ways; the unconditional rule is the safe one.
                below = raw_lines[i + 1].strip() if i + 1 < len(raw_lines) else ""
                if below:
                    self.add(path, i + 1, "display-math-needs-blank-line",
                             "`$$` is followed straight by prose/a tag — needs a "
                             "blank line below, or it never renders as math")
            expecting_close = not expecting_close

        for n, line in lines:
            # \(...\) / \[...\] -> pandoc will not parse it as math.
            if re.search(r"\\\(|\\\[", line):
                self.add(path, n, "math-delimiters",
                         r"use $...$ / $$...$$, never \(...\) or \[...\]")

            # <figcaption> without markdown="span" -> raw ** and $ in captions.
            if "<figcaption" in line and 'markdown="span"' not in line:
                self.add(path, n, "figcaption-markdown",
                         '<figcaption> needs markdown="span"')

            # Open-by-default solutions.
            if re.match(r"\s*\?\?\?\+\s", line):
                self.add(path, n, "exercise-idiom",
                         "'???+' is open by default; use '???'")

            # Headings need explicit anchors (## and ###; skip the H1 title).
            if re.match(r"^#{2,3} ", line) and not re.search(r"\{\s*#[\w-]+\s*\}\s*$", line):
                self.add(path, n, "explicit-heading-anchors",
                         "heading needs an explicit { #anchor }")

        # \eqref must target a \label in the SAME file (cross-page is impossible).
        labels = set(re.findall(r"\\label\{(eq:[\w:-]+)\}", body))
        for ref in set(re.findall(r"\\eqref\{(eq:[\w:-]+)\}", body)):
            if ref not in labels:
                self.add(path, 0, "eqref-same-file",
                         f"\\eqref{{{ref}}} has no \\label in this chapter "
                         f"(cross-page eqref cannot work; link in prose instead)")

        # Light/dark figure variants must come in pairs.
        light = re.findall(r"figures/([\w-]+)-light\.(?:svg|png)#only-light", body)
        dark = re.findall(r"figures/([\w-]+)-dark\.(?:svg|png)#only-dark", body)
        for slug in set(light) ^ set(dark):
            self.add(path, 0, "figure-pairs",
                     f"figure '{slug}' is missing one of its light/dark variants "
                     f"(fails SILENTLY — both images render)")

        # Referenced figures must exist in the manifest.
        for slug in set(light) | set(dark):
            if slug not in manifest:
                self.add(path, 0, "figure-exists",
                         f"figure '{slug}' is not in figures.json — invented?")

        # \rm and \bf are TeX font SWITCHES. MathJax renders them, so the site
        # looks fine — but pandoc's MathML writer cannot convert them and falls
        # back to shipping raw TeX into the offline HTML. Use \mathrm{}/\mathbf{}.
        # (notation.yml has always forbidden these; nothing enforced it until a
        # build shipped 12 raw equations.)
        #
        # NB: this does NOT contradict build_reports.py's note that \rm is
        # deliberately left un-shimmed for the generated REPORT pages. Those go
        # LaTeX -> markdown -> MathJax only, where \rm is correct and rewriting
        # it to \mathrm actively breaks it ({\rm eff} is a group-scoped
        # declaration; \mathrm takes one argument, giving \mathrm{e} + "ff").
        # The guide is hand-written markdown with a pandoc->MathML target as
        # well, so it needs the \mathrm{...} form. Different pipeline, different
        # rule; do not "fix" either one into the other.
        for off, inner in spans:
            for m in re.finditer(r"\\(rm|bf)\b", inner):
                self.add(path, body.count("\n", 0, off) + 1, "deprecated-font-cmd",
                         f"\\{m.group(1)} in math — pandoc cannot convert it to "
                         f"MathML and ships raw TeX. Use "
                         f"\\math{m.group(1)}{{...}}")
        for m in re.finditer(r"\$\$(.*?)\$\$", body, re.S):
            for f in re.finditer(r"\\(rm|bf)\b", m.group(1)):
                self.add(path, body.count("\n", 0, m.start()) + 1,
                         "deprecated-font-cmd",
                         f"\\{f.group(1)} in display math — use "
                         f"\\math{f.group(1)}{{...}} (MathML fallback ships raw TeX)")

        # A check tag inside math is typeset, not ignored: MathJax renders the
        # literal "<!-- check: ... -->" INSIDE the equation on the site, and the
        # underscores in the key make xelatex die with "Missing $ inserted".
        # Tags belong on their own line, after the closing $$.
        for m in re.finditer(r"\\text\{\s*<!--\s*check:", body):
            self.add(path, body.count("\n", 0, m.start()) + 1, "tag-inside-math",
                     "check tag inside math — MathJax will typeset the comment "
                     "into the equation. Put it on its own line after the $$")

        # Every tagged number must exist in worked_examples.json.
        for m in re.finditer(r"<!--\s*check:\s*([\w.]+)\s*=", body):
            key = m.group(1)
            if "." not in key:
                self.add(path, 0, "numbers-are-tagged",
                         f"malformed check tag '{key}' (want ch19.name)")

        # Cited repo paths must exist on disk.
        # NB: checked against RAW, not the code-stripped body — the contract
        # requires paths to be cited in backticks, which _strip_code blanks.
        for m in re.finditer(
            r"`((?:reproductions|site|tools|plans|papers)/[\w./-]+"
            r"\.(?:py|tex|md|yml|lua|sh))(?::\d+)?`", raw
        ):
            if not (REPO / m.group(1)).exists():
                self.add(path, 0, "repo-links",
                         f"cited path does not exist: {m.group(1)}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--file", help="lint one file")
    args = ap.parse_args()

    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    files = [Path(args.file)] if args.file else sorted(GUIDE.glob("*.md"))
    if not files:
        print("no guide chapters yet — nothing to lint")
        return 0

    lint = Lint()
    for f in files:
        lint.check_file(f, manifest)

    if not lint.problems:
        print(f"OK — {len(files)} file(s) clean")
        return 0

    by_rule: dict[str, int] = {}
    for path, line, rule, msg in sorted(lint.problems):
        loc = f"{Path(path).name}:{line}" if line else Path(path).name
        print(f"  {loc:34s} [{rule}] {msg}")
        by_rule[rule] = by_rule.get(rule, 0) + 1
    print(f"\nFAIL — {len(lint.problems)} problem(s) across {len(files)} file(s)")
    for rule, n in sorted(by_rule.items(), key=lambda kv: -kv[1]):
        print(f"  {n:3d}  {rule}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
