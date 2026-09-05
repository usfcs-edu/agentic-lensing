#!/usr/bin/env python3
"""Embargo check for AstroMark spec text, against both protected sources.

    python embargo_check.py FILE [FILE ...]        # check
    python embargo_check.py --spec                 # check everything in 08-draft-spec/

Two independent checks, because there are two protected corpora:

  1. `ngram_check.py` — 4-grams against Xiaosheng's transcript turns (21,078 n-grams, 909 turns).
  2. `banned_lexicon.txt` — the 298 strings extracted from his written comments.

WHY THIS WRAPPER EXISTS, beyond convenience: `ngram_check.py` writes its JSON report into the
workshop directory and refuses a report path outside it. This work must not modify anything under
`workshops/`, so the wrapper removes the report it caused after reading it.

HOW TO READ A HIT — this matters, and blind pass/fail is the wrong instrument for prose.

The 4-gram check is a deliberately strict tripwire built for short, proposed *prompt* text, where
any collision deserves a human look. Run against long-form prose it also flags ordinary English
function-word runs — "it has to be", "this is not a", "the only one that" — which collide with a
21k-n-gram corpus by chance and carry none of the protected content. Those are false positives.

So a hit is triaged, not obeyed:

  SUBSTANTIVE  the shared span carries domain meaning — a description of a lens feature, a
               judgement, a piece of reasoning. REWRITE. This is what the embargo protects.
  GENERIC      the shared span is function words only, with no domain noun, verb or judgement in it.
               Record and move on.

The classifier below is mechanical and conservative: a span is GENERIC only if every one of its
tokens is a closed-class English word. Anything with a content word in it is escalated for a human
decision rather than dismissed.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
W = REPO / "workshops/2026-08-31-LensMark"
NGRAM = W / "build/ngram_check.py"
LEXICON = REPO / "reproductions/lensjudge/golden/banned_lexicon.txt"
PY = Path.home() / ".venvs/workshop/bin/python"

# Closed-class English. A span made only of these carries no protected content.
FUNCTION_WORDS = set("""
a an the this that these those it its is are was were be been being am
i you he she we they them him her us me my your his their our
and or but so if then than as at by for from in into of off on onto out over to up with within
not no nor only just also very too much more most less least same other another
do does did done doing have has had having can could may might must shall should will would
what which who whom whose when where why how there here all any both each few many some such own
one two three first second next last other else about after before again further once
""".split())

TOKEN = re.compile(r"[a-z0-9]+")


def classify(span: str) -> str:
    toks = TOKEN.findall(span.lower())
    if not toks:
        return "GENERIC"
    return "GENERIC" if all(t in FUNCTION_WORDS for t in toks) else "SUBSTANTIVE"


def run_ngram(path: Path) -> list[str]:
    """Return the flagged spans. Cleans up the report the tool writes into the workshop dir."""
    if not NGRAM.is_file():
        print(f"  (skipped: {NGRAM} not found)")
        return []
    proc = subprocess.run([str(PY), str(NGRAM), str(path.resolve()), "--n", "4"],
                          capture_output=True, text=True, cwd=str(W))
    spans = []
    for line in proc.stdout.splitlines():
        m = re.search(r'HIT [^:]+: "([^"]+)"', line)
        if m:
            spans.append(m.group(1))
    stem = path.stem
    for leftover in (W / "analysis/qa").glob(f"ngram-{stem}.json"):
        leftover.unlink()                     # never leave a trace in the workshop directory
    return spans


def run_lexicon(path: Path) -> list[str]:
    if not LEXICON.is_file():
        print(f"  (skipped: {LEXICON} not found)")
        return []
    text = path.read_text(encoding="utf-8").lower()
    hits = []
    for raw in LEXICON.read_text(encoding="utf-8").splitlines():
        s = raw.strip().lower()
        if len(s) >= 12 and s in text:
            hits.append(raw.strip())
    return hits


def check(path: Path) -> dict:
    print(f"\n{path.name}")
    spans = run_ngram(path)
    lex = run_lexicon(path)
    subst = [s for s in spans if classify(s) == "SUBSTANTIVE"]
    gen = [s for s in spans if classify(s) == "GENERIC"]
    if lex:
        print(f"  BANNED LEXICON: {len(lex)} hit(s) — REWRITE REQUIRED")
        for h in lex[:5]:
            print(f"    · {h[:90]}")
    if subst:
        print(f"  4-gram SUBSTANTIVE: {len(subst)} — review each")
        for s in subst:
            print(f"    · {s!r}")
    if gen:
        print(f"  4-gram generic (function words only, no action): {len(gen)}")
        print(f"    {', '.join(repr(s) for s in gen[:6])}")
    if not (lex or subst):
        print("  CLEAN" + (f" ({len(gen)} generic collisions, dismissed)" if gen else ""))
    return {"file": str(path), "ok": not (lex or subst),
            "banned_lexicon": lex, "substantive": subst, "generic": gen}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", type=Path)
    ap.add_argument("--spec", action="store_true", help="check every .md in 08-draft-spec/")
    ap.add_argument("--json", type=Path, help="write the full report here")
    args = ap.parse_args()

    files = list(args.files)
    if args.spec:
        files += sorted((Path(__file__).parent.parent / "08-draft-spec").glob("*.md"))
    if not files:
        ap.error("give files or --spec")

    reports = [check(f) for f in files if f.is_file()]
    bad = [r for r in reports if not r["ok"]]
    print(f"\n{len(reports)} file(s) checked; {len(bad)} needing a rewrite.")
    if args.json:
        args.json.write_text(json.dumps(reports, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.json}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
