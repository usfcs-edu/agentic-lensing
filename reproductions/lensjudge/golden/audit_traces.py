#!/usr/bin/env python3
"""golden/audit_traces.py — prove from the traces that no embargoed text or validate pixel
reached the model.

Every golden model call writes a `golden_content_audit` event (golden/grader_jwst.py) with
the sha16 + 200-char head of each text block, the sha16 of each image payload and the
exemplar unit ids. This script replays those events and asserts, per event:

  1. no text block contains a banned string — case-folded, whitespace-normalised, matched
     as any 4-word window (>= 12 chars) of a lexicon entry, or the whole entry when it has
     fewer than 4 words (candidate ids, grade strings); the lexicon holds the validate-half
     candidate ids, the PI's 16 document comments and his validate-half grade values, so a
     verbatim copy of any of them is caught even when embedded in other text. (A raw
     12-character sliding window was tried first and fires on ordinary rubric phrases such
     as " the lens li" / "subtraction " — word windows keep the intent without that.)
  2. exemplar image shas are a subset of the ALIGN half's render_shas (kit key x splits);
  3. no VALIDATE-half render_sha appears as an exemplar;
  4. n_images == 1 + n_exemplars + n_extra_views (exactly one candidate composite, the
     declared extra views of a panel role — golden/panel.py's per-panel crops and the 20"
     context pair — and nothing else);
  5. every text block is fully verifiable: either its whole text fits the 200-char head (or
     the event was written with full_text, as the panel's item-specific blocks are), or its
     sha16 equals one of the known template constants (FEWSHOT_LEAD/TRAIL, the [composite]
     tag, grader_jwst.PANEL_GLOSS, the view-gloss strings of golden/views.py) — a long block
     the audit cannot read is a violation, not a pass.

The system prompt reaches the trace as a sha16 only, so pass the rubric files themselves
with --check-text (repeatable) to run the lexicon over them; do this before registering an
E3 rubric. Exit 1 on any violation; the JSON report goes to outputs/golden_audit.json.

  python lensjudge/golden/audit_traces.py --traces-dir outputs/traces_golden_e2_validate_r1 \\
      --banned golden/banned_lexicon.txt --splits golden/splits.csv --key golden/keys/<kit>_key.csv

`--build-lexicon` writes golden/banned_lexicon.txt (gitignored) from the validate-half
candidate ids (splits x frame/labels), the PI's 16 document comments and, when labels exist,
the PI's grade values keyed by validate unit. The comment strings live ONLY in the gitignored
golden/pi_comments.txt (one per line; the .docx itself is never opened by any code) — this
module carries their count and sha16 so a wrong or edited file is refused, never their text:
embargo rule 1 says the PI's free text is off limits to every tracked, model-adjacent file,
and a tracked module read by coding agents is one.

  python lensjudge/golden/audit_traces.py --build-lexicon --splits golden/splits.csv \\
      --frame golden/frame.csv [--labels golden/golden_labels.csv] --banned golden/banned_lexicon.txt

`--pi-only` builds a lexicon that needs NO splits file: the PI comments plus any ids from
`--extra-ids FILE` (a CSV with a candidate_id/name column or one id per line — the truth
holdout ids, so a persona prompt or a trace can be checked before golden/splits.csv exists;
the truth-eval's own firewall is split-independent, see plan PART 2). The same `extra_ids`
argument is accepted by build_lexicon() for the full build.

  python lensjudge/golden/audit_traces.py --build-lexicon --pi-only \\
      --extra-ids golden/truth_splits.csv --banned golden/banned_lexicon_truth.txt
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd  # noqa: E402

from lensjudge.golden import _util  # noqa: E402

NGRAM_WORDS = 4       # consecutive words of a lexicon entry that count as a verbatim copy
NGRAM_CHARS = 12      # ... provided the window is at least this long
HEAD = 200            # chars of each text block the audit event keeps
EVENT = "golden_content_audit"
DEFAULT_BANNED = _util.HERE / "banned_lexicon.txt"
DEFAULT_OUT = _util.LENSJUDGE / "outputs" / "golden_audit.json"

# The PI's 16 authored comments on the annotated top-100 document (2026-08-12) live in the
# gitignored golden/pi_comments.txt, one per line. Only their count and sha16 are recorded
# here (embargo rule 1: never the text in a tracked file). load_pi_comments() refuses a file
# that does not match, so the lexicon can neither silently shrink nor drift.
PI_COMMENTS_PATH = _util.HERE / "pi_comments.txt"
PI_COMMENTS_N = 16
PI_COMMENTS_SHA16 = "15dc9fa585f446a0"


def load_pi_comments(path: Path = PI_COMMENTS_PATH, required: bool = True) -> list[str]:
    """The PI comment strings from the gitignored file, verified against the pinned count and
    sha16. `required=False` returns [] when the file is absent (tests; a machine without it)."""
    path = Path(path)
    if not path.exists():
        if required:
            raise FileNotFoundError(
                f"{path} not found: the PI comment strings are gitignored and must be restored "
                f"by hand (16 lines, sha16 {PI_COMMENTS_SHA16}); pass --allow-missing-pi-comments "
                f"to build a lexicon without them (it will NOT catch a quoted comment)")
        return []
    text = path.read_text()
    lines = [ln.rstrip("\n") for ln in text.splitlines() if ln.strip()]
    if len(lines) != PI_COMMENTS_N or _util.sha_text("\n".join(lines) + "\n") != PI_COMMENTS_SHA16:
        raise ValueError(f"{path}: expected {PI_COMMENTS_N} lines with sha16 {PI_COMMENTS_SHA16}, "
                         f"got {len(lines)} lines, sha16 {_util.sha_text(chr(10).join(lines) + chr(10))}")
    return lines


def _read(path: Path) -> pd.DataFrame:
    path = Path(path)
    if path.with_suffix(path.suffix + ".sha").exists():
        return _util.read_pinned(path)
    return pd.read_csv(path)


# ------------------------------------------------------------------ lexicon
def _fold(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip().casefold()


EMPTY_SPLITS = pd.DataFrame({"unit_id": pd.Series(dtype=str), "split": pd.Series(dtype=str)})


def read_ids_file(path: Path) -> list[str]:
    """Candidate ids from a CSV (first of candidate_id / name / unit_id present) or a plain
    text file (one id per line; '#' comments ignored)."""
    path = Path(path)
    if path.suffix.lower() == ".csv":
        df = _read(path)
        for col in ("candidate_id", "name", "unit_id"):
            if col in df.columns:
                return [str(x) for x in df[col].dropna().astype(str) if str(x).strip()]
        raise ValueError(f"{path}: no candidate_id / name / unit_id column")
    return [ln.strip() for ln in path.read_text().splitlines()
            if ln.strip() and not ln.strip().startswith("#")]


def build_lexicon(splits: pd.DataFrame, frame: Optional[pd.DataFrame],
                  labels: Optional[pd.DataFrame], pi_comments=(), extra_ids=()) -> list[str]:
    """Validate-half candidate ids (+ aliases), the PI comments (`pi_comments`, from
    load_pi_comments), the PI's validate grade values as '<unit> score_1_4=<s>' /
    '<candidate_id> grade <L>' strings, and `extra_ids` (ids banned regardless of any split —
    the truth-eval holdout). `splits` may be EMPTY_SPLITS (the --pi-only build)."""
    sp = splits.copy()
    sp["unit_id"] = sp["unit_id"].astype(str)
    val = set(sp.loc[sp["split"] == "validate", "unit_id"])
    entries: list[str] = []
    if frame is not None:
        fr = frame.copy()
        fr["unit_id"] = fr["unit_id"].astype(str)
        fv = fr[fr["unit_id"].isin(val)]
        entries += fv["candidate_id"].astype(str).tolist()
        if "alias_ids" in fv:
            for a in fv["alias_ids"].dropna().astype(str):
                entries += [x for x in a.split("|") if x.strip()]
    entries += [str(c) for c in pi_comments]
    if labels is not None:
        lb = labels.copy()
        lb["unit_id"] = lb["unit_id"].astype(str)
        lv = lb[lb["unit_id"].isin(val)]
        if frame is None:
            entries += lv["candidate_id"].astype(str).tolist()
        for r in lv.itertuples():
            entries.append(f"{r.unit_id} score_1_4={int(r.score_1_4)} confidence_lmh={r.confidence_lmh}")
            entries.append(f"{r.candidate_id} grade {r.grade_letter}")
            entries.append(f"{r.candidate_id} score {int(r.score_1_4)}/4")
    entries += [str(x) for x in extra_ids]
    seen, out = set(), []
    for e in entries:
        e = str(e).strip()
        if e and e not in seen:
            seen.add(e)
            out.append(e)
    return out


def load_lexicon(path: Path) -> list[str]:
    return [ln.rstrip("\n") for ln in Path(path).read_text().splitlines() if ln.strip()]


_WORD_RE = re.compile(r"[^\W_]+(?:[-'_./:=+][^\W_]+)*", re.UNICODE)


def _words(s: str) -> list[str]:
    """Case-folded word tokens; intra-word punctuation (ids like J3440482-522486, spec-z,
    score_1_4=3) stays attached so an id is one token."""
    return _WORD_RE.findall(_fold(s))


def banned_hit(text: str, lexicon: list[str], n_words: int = NGRAM_WORDS,
               n_chars: int = NGRAM_CHARS) -> Optional[tuple[str, str]]:
    """(entry, matched window) for the first lexicon entry found in `text`, else None.
    Matching is on word tokens: any n_words-long window of the entry (joined by single
    spaces, >= n_chars long) found in the tokenised text, or the whole entry when it is
    shorter than n_words words."""
    tw = _words(text)
    if not tw:
        return None
    tj = " " + " ".join(tw) + " "
    for e in lexicon:
        ew = _words(e)
        if not ew:
            continue
        if len(ew) < n_words:
            g = " ".join(ew)
            if f" {g} " in tj:
                return e, g
            continue
        for i in range(0, len(ew) - n_words + 1):
            g = " ".join(ew[i:i + n_words])
            if len(g) >= n_chars and f" {g} " in tj:
                return e, g
    return None


# ------------------------------------------------------------------ events
def read_events(traces_dir: Path) -> list[dict]:
    evs = []
    for p in sorted(Path(traces_dir).rglob("*.jsonl")):
        for ln in p.read_text().splitlines():
            if not ln.strip():
                continue
            try:
                rec = json.loads(ln)
            except json.JSONDecodeError:
                continue
            if rec.get("event") == EVENT:
                rec["_trace"] = str(p)
                evs.append(rec)
    return evs


def split_shas(key: pd.DataFrame, splits: pd.DataFrame) -> tuple[set, set]:
    """(align render_shas, validate render_shas) from the kit key joined to the splits."""
    k = key.copy()
    k["unit_id"] = k["unit_id"].astype(str)
    sp = splits.copy()
    sp["unit_id"] = sp["unit_id"].astype(str)
    j = k.merge(sp[["unit_id", "split"]], on="unit_id", how="left")
    align = set(j.loc[j["split"] == "align", "render_sha"].astype(str))
    validate = set(j.loc[j["split"] == "validate", "render_sha"].astype(str))
    return align, validate


def known_template_shas() -> dict[str, str]:
    """sha16 -> label for the fixed text blocks the golden path may emit (long ones the
    200-char head cannot fully show): the few-shot wrapper, the single-call panel gloss and
    every view-gloss string of the panel roles (golden/views.py; the file-or-built-in gloss
    in use), including the v2r composite VIEW texts with golden/render_v2_desc.md appended
    exactly as views.view_text sends them. Imported lazily: grader_jwst reads the prompt
    files."""
    from lensjudge.golden import fewshot, grader_jwst, views
    known = {_util.sha_text(fewshot.FEWSHOT_LEAD): "FEWSHOT_LEAD",
             _util.sha_text(fewshot.FEWSHOT_TRAIL): "FEWSHOT_TRAIL",
             _util.sha_text(fewshot.COMPOSITE_TAG): "COMPOSITE_TAG",
             _util.sha_text(grader_jwst.PANEL_GLOSS): "PANEL_GLOSS"}
    descs = {}
    desc_path = _util.HERE / "render_v2_desc.md"
    if desc_path.exists():
        descs["v2r"] = desc_path.read_text()
    for i, t in enumerate(views.gloss_strings(render_descs=descs)):
        known.setdefault(_util.sha_text(t), f"VIEW_GLOSS_{i}")
    return known


def audit(events: list[dict], lexicon: list[str], align_shas: set, validate_shas: set,
          known: Optional[dict] = None) -> dict:
    known = known_template_shas() if known is None else known
    violations = []
    for ev in events:
        where = {"trace": ev.get("_trace"), "name": ev.get("name")}
        for i, tb in enumerate(ev.get("text_blocks", [])):
            head = tb.get("head", "")
            hit = banned_hit(head, lexicon)
            if hit:
                violations.append({**where, "check": "banned_text", "block": i,
                                   "entry": hit[0], "ngram": hit[1]})
            n_chars = int(tb.get("n_chars", len(head)))
            if n_chars > len(head) and tb.get("sha16") not in known:
                violations.append({**where, "check": "unverifiable_text_block", "block": i,
                                   "n_chars": n_chars, "head": head[:80]})
        ex = [str(s) for s in ev.get("exemplar_image_shas", [])]
        n_ex = int(ev.get("n_exemplars", 0))
        for s in ex:
            if s not in align_shas:
                violations.append({**where, "check": "exemplar_not_align", "sha": s})
            if s in validate_shas:
                violations.append({**where, "check": "validate_as_exemplar", "sha": s})
        n_xv = int(ev.get("n_extra_views", 0))
        if int(ev.get("n_images", -1)) != 1 + n_ex + n_xv:
            violations.append({**where, "check": "n_images", "n_images": ev.get("n_images"),
                               "n_exemplars": n_ex, "n_extra_views": n_xv})
        if len(ev.get("extra_view_shas", [])) != n_xv:
            violations.append({**where, "check": "extra_view_count",
                               "listed": len(ev.get("extra_view_shas", [])), "n_extra_views": n_xv})
        if len(ex) != n_ex:
            violations.append({**where, "check": "exemplar_count", "listed": len(ex), "n_exemplars": n_ex})
    return {"n_events": len(events), "n_violations": len(violations), "violations": violations,
            "n_lexicon": len(lexicon), "n_align_shas": len(align_shas),
            "n_validate_shas": len(validate_shas),
            "system_sha16s": sorted({str(e.get("system_sha16")) for e in events}),
            "exemplar_unit_ids": sorted({u for e in events for u in e.get("exemplar_unit_ids", [])}),
            "passed": not violations}


# ------------------------------------------------------------------ main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces-dir", default=None)
    ap.add_argument("--banned", default=str(DEFAULT_BANNED))
    ap.add_argument("--splits", default=str(_util.HERE / "splits.csv"))
    ap.add_argument("--key", action="append", default=None,
                    help="kit key CSV (repeatable); default: every golden/keys/*_key.csv")
    ap.add_argument("--frame", default=str(_util.HERE / "frame.csv"))
    ap.add_argument("--labels", default=str(_util.HERE / "golden_labels.csv"))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--build-lexicon", action="store_true",
                    help="write --banned from validate ids + PI comments (+ labels) and exit")
    ap.add_argument("--pi-comments", default=str(PI_COMMENTS_PATH),
                    help="gitignored file with the PI's comment strings, one per line")
    ap.add_argument("--allow-missing-pi-comments", action="store_true",
                    help="build the lexicon without the PI comments when the file is absent")
    ap.add_argument("--check-text", action="append", default=None,
                    help="also run the lexicon over this file (rubric / note); repeatable")
    ap.add_argument("--pi-only", action="store_true",
                    help="--build-lexicon from the PI comments (+ --extra-ids) only; no splits needed")
    ap.add_argument("--extra-ids", action="append", default=None,
                    help="file of candidate ids to ban regardless of split (CSV or one per line); repeatable")
    args = ap.parse_args(argv)

    if args.pi_only and not args.build_lexicon:
        ap.error("--pi-only only makes sense with --build-lexicon")
    if args.pi_only:
        splits = EMPTY_SPLITS.copy()
    elif Path(args.splits).exists():
        splits = _read(Path(args.splits))
    elif args.build_lexicon:
        raise SystemExit(f"--build-lexicon needs the splits file ({args.splits} not found; "
                         f"use --pi-only for a split-free lexicon)")
    else:
        # pre-split audit (the --smoke plumbing run happens before split_halves.py): no unit
        # is align or validate yet, so ANY exemplar is a violation — correct for zero-shot.
        print(f"[audit] WARNING: {args.splits} not found; auditing with empty align/validate sets")
        splits = EMPTY_SPLITS.copy()
    extra_ids: list[str] = []
    for f in (args.extra_ids or []):
        extra_ids += read_ids_file(Path(f))
    if args.build_lexicon:
        if args.pi_only:
            frame = labels = None
        else:
            frame = _read(Path(args.frame)) if Path(args.frame).exists() else None
            labels = _read(Path(args.labels)) if Path(args.labels).exists() else None
        pi = load_pi_comments(Path(args.pi_comments), required=not args.allow_missing_pi_comments)
        if not pi:
            print(f"[lexicon] WARNING: no PI comments ({args.pi_comments} absent) — a quoted comment "
                  f"would NOT be caught by this lexicon")
        lex = build_lexicon(splits, frame, labels, pi_comments=pi, extra_ids=extra_ids)
        Path(args.banned).parent.mkdir(parents=True, exist_ok=True)
        Path(args.banned).write_text("\n".join(lex) + "\n")
        print(f"[lexicon] {len(lex)} entries ({len(pi)} PI comments, {len(extra_ids)} extra ids"
              f"{', pi-only' if args.pi_only else ''}) -> {args.banned} (gitignored; never model-facing)")
        return 0

    if not args.traces_dir:
        ap.error("--traces-dir is required (or --build-lexicon)")
    key_paths = [Path(k) for k in args.key] if args.key else sorted((_util.HERE / "keys").glob("*_key.csv"))
    if not key_paths:
        raise SystemExit("no kit key CSV (pass --key)")
    key = pd.concat([_read(p) for p in key_paths], ignore_index=True)
    align_shas, validate_shas = split_shas(key, splits)
    lexicon = load_lexicon(Path(args.banned))
    events = read_events(Path(args.traces_dir))
    rep = audit(events, lexicon, align_shas, validate_shas)
    rep["traces_dir"] = str(args.traces_dir)
    rep["checked_files"] = {}
    for f in (args.check_text or []):   # rubrics / the note: not in the trace, checked directly
        hit = banned_hit(Path(f).read_text(), lexicon)
        rep["checked_files"][f] = None if hit is None else {"entry": hit[0], "ngram": hit[1]}
        if hit:
            rep["violations"].append({"check": "banned_text_file", "file": f, "entry": hit[0], "ngram": hit[1]})
    rep["n_violations"] = len(rep["violations"])
    rep["passed"] = not rep["violations"]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rep, indent=2))
    print(f"[audit] {rep['n_events']} events, {rep['n_violations']} violations -> {args.out}")
    for v in rep["violations"][:20]:
        print("  VIOLATION", json.dumps(v))
    if not events:
        print("[audit] WARNING: no golden_content_audit events found")
    return 0 if rep["passed"] and events else 1


if __name__ == "__main__":
    sys.exit(main())
