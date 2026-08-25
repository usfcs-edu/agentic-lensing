#!/usr/bin/env python3
"""Build the mkdocs pages for the JWST top-100 blind regrade.

Reads one run directory produced by reproductions/lensjudge/golden/regrade_scrambled.py
(+ explain.py + annotate.py) and writes a hand-authored-style page set:

    <out>/index.md                   overview, agreement cross-tabs, the all-100 table
    <out>/ranks-001-025.md ...       one H2 per candidate: grade strip, original + annotated
                                     composite, the explain markdown in a details block
    <out>/img/rank-NNN-{orig,annot}.jpg   JPEGs copied as-is
    <out>/data/comparison.csv        the comparison CSV without the blind kit ids (scrambled_item dropped, nate_* -> original_*)
    <out>/.build_top100              sentinel: this directory is owned by the builder

and adds a "JWST Top-100 Regrade" entry under the "Current" nav section of site/mkdocs.yml.
Deterministic and idempotent: the files it owns (index.md, ranks-*.md, img/, data/, the
sentinel) are deleted and recreated on every run; nothing else under site/ is touched, and
the builder refuses an --out directory it does not recognise as its own (non-empty, no
sentinel, index.md not starting with its H1).

The deployed letter rule (R1 / R2, REGISTRY.md "Deployment rule v2-deploy") is read from the
comparison CSV's `our_rule` column and every rule-dependent sentence follows it. The
pre-registered selection and the holdout transfer numbers are quoted from
golden/transfer_check.py's selected_rule.json (+ transfer_check.csv beside it) when given
or found. Every number on the pages comes from the comparison CSV, the run's *.meta.json or
those two transfer files, except the labelled "program holdout result (opus5-xhigh)"
constants in HOLDOUT below (REGISTRY.md v2-deploy preamble; outputs/truth_summary_claude5.md).

Stdlib only (csv / json / shutil / re) — runs under any python3 >= 3.9.

    python3 site/build_top100.py --run-dir reproductions/lensjudge/outputs/scrambled100_opus5 \\
        --transfer-json reproductions/lensjudge/outputs/transfer_opus5/selected_rule.json
    python3 site/build_top100.py --run-dir reproductions/lensjudge/outputs/scrambled100_dev_fix \\
        --explain-dir reproductions/lensjudge/outputs/scrambled100_dev_fix/explain_R1 \\
        --annot-dir   reproductions/lensjudge/outputs/scrambled100_dev_fix/annot
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import sys
from collections import Counter, OrderedDict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SITE = REPO / "site"
DOCS = SITE / "docs"
DEFAULT_OUT = DOCS / "jwst-top100"
DEFAULT_MKDOCS = SITE / "mkdocs.yml"
NAV_TITLE = "JWST Top-100 Regrade"
SENTINEL = ".build_top100"
SENTINEL_TEXT = ("written by site/build_top100.py — every file in this directory is regenerated "
                 "by that script; do not edit by hand\n")
INDEX_H1 = "# JWST Top-100: original grades vs the evidence-first judge"

LETTERS = ("A", "B", "C", "D")
LETTER_ORDER = {"A": 4, "B": 3, "C": 2, "D": 1, "U": 0}
DEPLOY_RULES = ("R1", "R2")
# Design anchors: the five top-100 items whose outcome, known during design, shaped the
# scheme's mechanisms (golden/REGISTRY.md "Design anchors"). Not an independent test.
ANCHOR_RANKS = (7, 13, 14, 15, 16)
ROLE_ABBREV = {"artifact": "Art", "geometry": "Geo", "morphology": "Mor"}
# Run-meta model keys -> Messages API ids (reproductions/lensjudge/imaging/grader_direct.py
# _MODEL_IDS). Display only; an unknown key is shown as itself.
MODEL_IDS = {
    "sonnet": "claude-sonnet-4-6",
    "opus": "claude-opus-4-8",
    "haiku": "claude-haiku-4-5",
    "opus5": "claude-opus-5",
    "sonnet5": "claude-sonnet-5",
}
DAGGER = "†"
FLAG = "⚑"

# Program holdout result (opus5-xhigh) — REGISTRY.md "Deployment rule v2-deploy" preamble and
# outputs/truth_summary_claude5.md. Quoted on the page under exactly that label.
HOLDOUT_LABEL = "program holdout result (opus5-xhigh)"
HOLDOUT = {
    "advocate_recall_5fpr": "0.333",
    "advocate_auc": "0.764",
    "stack_auc": "0.468",
    "stack_n_neg_scored": "149",
    "stack_stressD_rate": "6/20 = 0.30",
    "advocate_only_stressD_rate": "0/20",
}
# Threshold provenance by thresholds key (meta.thresholds_resolved.thresholds_key).
# Format fields: t_A, t_B, letter_source.
THRESHOLD_PROVENANCE = {
    "opus5_api": ("t_A = **{t_A}** is the smallest advocate score with ≤ 1 % false positives and "
                  "t_B = **{t_B}** the smallest with ≤ 5 % on the 200 clean random negatives of the "
                  "truth set's design half (fit on p_evidence by `golden/calibrate_thresholds.py`; "
                  "letters `{letter_source}`)."),
    "sonnet_api": ("t_A = **{t_A}** and t_B = **{t_B}** were frozen on the full-stack score S of the "
                   "sonnet design run, not on p_evidence, so rank letters under this key are **not** "
                   "FPR-controlled (letters `{letter_source}`)."),
}

# Material gives every <th> a 5rem min-width, so an 8-column table overflows the content
# column and scrolls sideways; scope a relaxed min-width to this page set's tables only.
STYLE = ('<style>\n'
         '.md-typeset .t100 table:not([class]) th { min-width: 0; }\n'
         '.md-typeset .t100 table:not([class]) td, .md-typeset .t100 table:not([class]) th { padding: .7em .8em; }\n'
         '</style>')
TBL_OPEN = '<div class="t100" markdown>'
TBL_CLOSE = '</div>'

CODE_SPAN = re.compile(r"`[^`]*`")
TAG_LIKE = re.compile(r"<(?=[A-Za-z/!?])")
HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")


# ----------------------------------------------------------------------------- helpers
def die(msg: str) -> None:
    sys.exit(f"build_top100: {msg}")


def fnum(x, nd: int, dash: str = "—") -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return dash
    return dash if math.isnan(v) else f"{v:.{nd}f}"


def fint(x, dash: str = "—") -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return dash
    return dash if math.isnan(v) else (str(int(v)) if v == int(v) else f"{v:g}")


def fpct(x, nd: int = 1, dash: str = "—") -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return dash
    return dash if math.isnan(v) else f"{100 * v:.{nd}f} %"


def to_float(x) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float("nan")
    return v


def letter(x: str) -> str:
    x = (x or "").strip()
    return x if x else "—"


def is_true(x: str) -> bool:
    return str(x).strip().lower() in ("true", "1", "yes", "y")


def fmt_veto(v: str) -> str:
    """'artifact:merger;geometry:merger;morphology:spiral_arm' -> 'merger (Art, Geo); spiral_arm (Mor)'."""
    v = (v or "").strip()
    if not v:
        return "—"
    by_alt: "OrderedDict[str, list[str]]" = OrderedDict()
    for part in v.split(";"):
        part = part.strip()
        if not part:
            continue
        role, _, alt = part.partition(":")
        by_alt.setdefault(alt or role, []).append(ROLE_ABBREV.get(role, role))
    return "; ".join(f"{alt} ({', '.join(roles)})" for alt, roles in by_alt.items())


def rank_page_name(start: int, end: int) -> str:
    return f"ranks-{start:03d}-{end:03d}.md"


def page_title(start: int, end: int) -> str:
    return f"Ranks {start}–{end}"


def cand_label(row: dict) -> str:
    cid = row["candidate_id"].strip() or row["scrambled_item"]
    return cid + (f" {DAGGER}" if int(row["rank"]) in ANCHOR_RANKS else "")


# ----------------------------------------------------------------------------- inputs
def load_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    need = {"scrambled_item", "rank", "candidate_id", "nate_grade", "nate_n_pass", "nate_inspector_conf",
            "blind_theta_E_arcsec", "our_S", "our_S_arb", "our_letter_llm", "our_p_evidence",
            "our_needs_human", "our_letter_rank", "our_letter_final", "our_veto", "our_rule"}
    missing = need - set(rows[0].keys() if rows else [])
    if missing:
        die(f"comparison CSV lacks columns: {sorted(missing)}")
    rows.sort(key=lambda r: int(r["rank"]))
    ranks = [int(r["rank"]) for r in rows]
    if len(set(ranks)) != len(ranks):
        die("duplicate ranks in the comparison CSV")
    return rows


def deployed_rule(rows: list[dict], meta: dict) -> str:
    """The deployment rule every row was lettered under (`our_rule`); one value or error."""
    seen = sorted({(r.get("our_rule") or "").strip() for r in rows})
    if len(seen) != 1:
        die(f"comparison CSV mixes deployment rules {seen}: refusing to describe one rule")
    rule = seen[0]
    if rule not in DEPLOY_RULES:
        die(f"unknown deployment rule {rule!r} in the comparison CSV (expected one of {DEPLOY_RULES})")
    if meta.get("rule") and meta["rule"] != rule:
        die(f"meta json says rule {meta['rule']!r} but the comparison CSV was lettered under {rule!r}")
    return rule


def load_meta(path: Path) -> dict:
    m = json.loads(path.read_text(encoding="utf-8"))
    t = m.get("tuple") or {}
    thr = m.get("thresholds_resolved") or {}
    return {
        "model": m.get("model") or t.get("model") or "?",
        "backend": m.get("backend") or "?",
        "thinking": t.get("thinking", "?"),
        "effort": t.get("effort", "?"),
        "k": t.get("k", "?"),
        "arm": t.get("arm", "?"),
        "letter_source": m.get("letter_source") or thr.get("letter_source") or "?",
        "thresholds_key": thr.get("thresholds_key") or thr.get("model_key") or "?",
        "tau0": thr.get("tau0"),
        "t_A": thr.get("t_A"),
        "t_B": thr.get("t_B"),
        "rule": m.get("rule"),
        "cost": m.get("cost_usd_total", m.get("summary", {}).get("total_cost_usd")),
        "n": m.get("n"),
        "n_parse_ok": m.get("n_parse_ok"),
        "n_gray": m.get("n_gray"),
        "scored_at": (m.get("scored_at") or "")[:10],
        "relettered_at": (m.get("relettered_at_utc") or "")[:10],
        "thresholds_sha16": m.get("thresholds_sha16") or t.get("thresholds_sha16"),
        "persona_set_sha16": t.get("persona_set_sha16"),
    }


def find_meta(run_dir: Path) -> Path:
    metas = sorted(run_dir.glob("*.meta.json"))
    if len(metas) != 1:
        die(f"expected exactly one *.meta.json in {run_dir}, found {len(metas)} — pass --meta")
    return metas[0]


def load_annot_index(annot_dir: Path) -> dict:
    idx = annot_dir / "annot_index.csv"
    if not idx.exists():
        return {}
    with idx.open(newline="", encoding="utf-8") as fh:
        return {r["name"]: r for r in csv.DictReader(fh)}


def load_transfer(path: Path) -> dict:
    """golden/transfer_check.py's selected_rule.json (+ transfer_check.csv beside it, when
    present) -> {"json": path, "rule", "reason", "numbers", "csv": path|None,
    "stats": {(statistic, rule): (value, ci_lo, ci_hi, n)}}. Nothing is derived here."""
    j = json.loads(path.read_text(encoding="utf-8"))
    rule = str(j.get("rule") or "")
    if rule not in DEPLOY_RULES:
        die(f"{path}: 'rule' is {rule!r}, expected one of {DEPLOY_RULES}")
    out = {"json": path, "rule": rule, "reason": str(j.get("reason") or ""),
           "numbers": j.get("numbers") or {}, "provisional": bool(j.get("provisional", False)),
           "csv": None, "stats": {}}
    csv_path = path.parent / "transfer_check.csv"
    if csv_path.exists():
        with csv_path.open(newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                out["stats"][(r["statistic"], r["rule"])] = (r.get("value"), r.get("ci_lo"), r.get("ci_hi"), r.get("n"))
        out["csv"] = csv_path
    return out


# ----------------------------------------------------------------------------- explain md
def escape_tags(line: str) -> str:
    """Escape tag-like '<' outside inline code spans (entities are literal inside code)."""
    out, pos = [], 0
    for m in CODE_SPAN.finditer(line):
        out.append(TAG_LIKE.sub("&lt;", line[pos:m.start()]))
        out.append(m.group(0))
        pos = m.end()
    out.append(TAG_LIKE.sub("&lt;", line[pos:]))
    return "".join(out)


def demote_heading(body: str, level: int) -> str:
    if level >= 3 and ": “" in body:
        head, rest = body.split(": “", 1)
        return f"**{head}:** “{rest}"
    return f"**{body}**"


def explain_block(md_text: str) -> list[str]:
    """The explain markdown re-indented 4 spaces for a pymdownx.details block: the H1
    dropped, other headings demoted to bold lines (with blank lines around them so a
    following list still parses), raw HTML escaped, 2-space sub-lists deepened to 4."""
    out: list[str] = []
    for raw in md_text.splitlines():
        line = raw.rstrip()
        if not line.strip():
            out.append("")
            continue
        m = HEADING.match(line)
        if m:
            level = len(m.group(1))
            if level == 1:
                continue
            out.extend(["", "    " + escape_tags(demote_heading(m.group(2), level)), ""])
            continue
        n = len(line) - len(line.lstrip(" "))
        out.append(" " * (4 + 2 * n) + escape_tags(line.lstrip(" ")))
    # collapse blank runs, trim leading/trailing blanks
    res: list[str] = []
    for l in out:
        if l == "" and (not res or res[-1] == ""):
            continue
        res.append(l)
    while res and res[-1] == "":
        res.pop()
    return res


# ----------------------------------------------------------------------------- stats
def crosstab(rows: list[dict], col: str) -> tuple[list[str], dict]:
    grades = sorted({r["nate_grade"] for r in rows}, key=lambda g: -LETTER_ORDER.get(g, -1))
    tab = {g: Counter() for g in grades}
    for r in rows:
        tab[r["nate_grade"]][letter(r[col])] += 1
    return grades, tab


def crosstab_md(rows: list[dict], col: str, label: str) -> list[str]:
    grades, tab = crosstab(rows, col)
    cols = list(LETTERS) + sorted({letter(r[col]) for r in rows} - set(LETTERS))
    head = f"| Original \\ {label} | " + " | ".join(cols) + " | Total |"
    sep = "|:--|" + "--:|" * (len(cols) + 1)
    lines = [head, sep]
    for g in grades:
        n = sum(tab[g].values())
        lines.append(f"| **{g}** ({n}) | " + " | ".join(str(tab[g][c]) for c in cols) + f" | {n} |")
    tot = Counter()
    for g in grades:
        tot.update(tab[g])
    lines.append("| **Total** | " + " | ".join(str(tot[c]) for c in cols) + f" | {sum(tot.values())} |")
    return lines


def summary_counts(rows: list[dict], tau0=None) -> dict:
    """Letter movement and A/B counts over the SCORED rows only (final letter present);
    unscored rows (final letter '—') are counted in n_unscored and excluded from every
    same/up/down, A/B and demotion count. `orig_dist` stays over all rows (it describes the
    original campaign's grades, not ours)."""
    o = LETTER_ORDER
    ab = ("A", "B")
    fin = [letter(r["our_letter_final"]) for r in rows]
    rk = [letter(r["our_letter_rank"]) for r in rows]
    og = [r["nate_grade"] for r in rows]
    scored = [i for i, f in enumerate(fin) if f != "—"]
    graded = [i for i in scored if og[i] != "U"]
    unex = [i for i in scored if og[i] == "U"]
    t0 = to_float(tau0)
    c = {
        "n": len(rows),
        "n_scored": len(scored),
        "n_unscored": len(rows) - len(scored),
        "same": sum(1 for i in scored if fin[i] == og[i]),
        "up": sum(1 for i in scored if o.get(fin[i], -1) > o.get(og[i], -1)),
        "down": sum(1 for i in scored if o.get(fin[i], -1) < o.get(og[i], -1)),
        "n_graded": len(graded),
        "g_same": sum(1 for i in graded if fin[i] == og[i]),
        "g_up": sum(1 for i in graded if o.get(fin[i], -1) > o[og[i]]),
        "g_down": sum(1 for i in graded if o.get(fin[i], -1) < o[og[i]]),
        "n_U": len(unex),
        "U_dist": Counter(fin[i] for i in unex),
        "U_AB": sum(1 for i in unex if fin[i] in ab),
        "orig_AB": sum(1 for i in scored if og[i] in ab),
        "final_AB": sum(1 for i in scored if fin[i] in ab),
        "rank_AB": sum(1 for i in scored if rk[i] in ab),
        "orig_dist": Counter(og),
        "final_dist": Counter(fin[i] for i in scored),
        "rank_dist": Counter(rk[i] for i in scored),
        "n_demoted": sum(1 for i in scored if o.get(fin[i], -1) < o.get(rk[i], -1)),
        "n_veto": sum(1 for r in rows if r["our_veto"].strip()),
        "n_needs_human": sum(1 for r in rows if is_true(r["our_needs_human"])),
        "n_needs_human_AB": sum(1 for r in rows if is_true(r["our_needs_human"])
                                and letter(r["our_letter_final"]) in ab),
        # critics are called when p_evidence >= tau0; the arbitrator ran when it left a letter
        "n_called": (None if math.isnan(t0) else
                     sum(1 for r in rows if to_float(r["our_p_evidence"]) >= t0)),
        "n_arb": sum(1 for r in rows if letter(r["our_letter_llm"]) != "—"),
        "alt_dist": Counter(a for r in rows for a in
                            {p.partition(":")[2] for p in r["our_veto"].split(";") if p.strip()}),
    }
    return c


# ----------------------------------------------------------------------------- rule text
def final_letter_row(rule: str) -> str:
    """The 'Final letter' row of the two-letter table, for the deployed rule."""
    if rule == "R1":
        return (f"| **Final letter** (rule R1) | S_arb through the same thresholds | The evidence that "
                "*survives arbitration* (upheld and partial refutations applied, overruled ones ignored) must "
                "clear the same bars. Because S_arb ≤ p_evidence the critic stack can only **demote**, and a "
                "demotion names the upheld or partial alternative(s) whose term entered S_arb (\"Demoted by\" "
                "below). |")
    return (f"| **Final letter** (rule R2) | the rank letter, unless the D-rule fires | Equals the advocate's "
            "rank letter unless an **upheld** critic's located alternative covers every evidence item at "
            "refutation strength r ≥ 0.8 — then **D**. \"Demoted by\" below is populated only for such "
            "D-rule vetoes; S_arb is shown for reference and does not set the letter. |")


def letter_rules_md(rule: str, meta: dict) -> list[str]:
    """Item 3: the complete assign_letter rules, then how each deployed letter feeds them."""
    L = []
    L.append("**Letter rules** (`aggregate_v2.assign_letter`). **A**: S ≥ t_A *and* at least two of {curvature,")
    L.append("counter-image, arc morphology} scored ≥ 6 *and* no included critic with r·a ≥ 0.8. **B**: S ≥ t_B")
    L.append("(and not A). **D**: S < t_B and either the advocate located no evidence and stated a *nothing_because*,")
    L.append("or an upheld critic covers every evidence item at r ≥ 0.8. **C**: otherwise. The rank letter reads")
    L.append("S = p_evidence with no critics. ")
    if rule == "R1":
        L[-1] += ("The final letter (R1) reads S = S_arb with the arbitrated critics: upheld and partial refutations "
                  "count, overruled ones do not.")
    else:
        L[-1] += ("Under R2 the final letter is the rank letter itself; only the D-rule (an upheld critic covering "
                  "every item at r ≥ 0.8) can override it, to D.")
    L.append(f"Critics are only called when p_evidence ≥ τ0 = **{fnum(meta['tau0'], 2)}**; below that the final")
    L.append("letter equals the rank letter.")
    return L


def calibration_md(meta: dict) -> list[str]:
    """Item 2: threshold provenance keyed on thresholds_key + letter_source. Provisional
    letters get no paragraph here (the warning admonition at the top already says so)."""
    key = str(meta["thresholds_key"])
    src = str(meta["letter_source"])
    calibrated = src.endswith("_calibrated")
    fields = {"t_A": fnum(meta["t_A"], 4), "t_B": fnum(meta["t_B"], 4), "letter_source": src}
    if key == "provisional" or not calibrated:
        return []
    if key == "opus5_api":
        return ["**Calibration.** " + THRESHOLD_PROVENANCE["opus5_api"].format(**fields)]
    if key == "sonnet_api":
        return ["**Calibration.** " + THRESHOLD_PROVENANCE["sonnet_api"].format(**fields)]
    return [f"**Calibration.** Thresholds key `{key}` (letters `{src}`): t_A = **{fields['t_A']}**, "
            f"t_B = **{fields['t_B']}**; the provenance of this key is not recorded by this builder."]


def why_rule_md(rule: str, transfer) -> list[str]:
    """Item 1: the pre-registered selection rule and, when a transfer json was given, the
    selection it produced — quoted, never derived."""
    L = []
    L.append("**Why this rule.** The deployment rule was pre-registered before any calibrated letter was read")
    L.append("(REGISTRY \"Deployment rule v2-deploy\", item 6): **R1** is deployed unless, on the already-scored")
    L.append("holdout, R1's recall at A∪B on the holdout positives is below one half of the rank letter's recall at")
    L.append("A∪B — then the pre-stated fallback **R2**.")
    if transfer is None:
        L.append(f"No transfer check was supplied to this build (`--transfer-json`), so the selection outcome is not")
        L.append(f"quoted here; this run's letters were assigned under rule **{rule}**.")
        return L
    nums = transfer["numbers"]
    parts = []
    if "recall_AB_letter_rank" in nums:
        parts.append(f"recall@A∪B rank letter {fnum(nums['recall_AB_letter_rank'], 3)}")
    if "recall_AB_R1" in nums:
        parts.append(f"R1 {fnum(nums['recall_AB_R1'], 3)}")
    if "bar" in nums:
        parts.append(f"bar {fnum(nums['fraction'], 1) if 'fraction' in nums else '0.5'} × rank = {fnum(nums['bar'], 3)}")
    if "recall_AB_R2" in nums:
        parts.append(f"R2 {fnum(nums['recall_AB_R2'], 3)}")
    if "n_pos_R1" in nums:
        parts.append(f"{fint(nums['n_pos_R1'])} holdout positives")
    thr = []
    if nums.get("letter_source"):
        thr.append(f"letters `{nums['letter_source']}`")
    if "t_A" in nums and "t_B" in nums:
        thr.append(f"t_A {fnum(nums['t_A'], 2)} / t_B {fnum(nums['t_B'], 2)}"
                   + (f" / τ0 {fnum(nums['tau0'], 2)}" if "tau0" in nums else ""))
    L.append(f"Transfer check — {HOLDOUT_LABEL}, from `{transfer['json'].name}`: selected **{transfer['rule']}** —")
    L.append("“" + escape_tags(transfer["reason"]) + "”")
    tail = "; ".join(parts + thr)
    if tail:
        L[-1] += f" ({tail})."
    else:
        L[-1] += "."
    if transfer["provisional"]:
        L.append("The transfer json is marked *provisional*.")
    if transfer["rule"] == rule:
        L.append(f"This run's letters are assigned under the selected rule **{rule}**.")
    else:
        L.append(f"This run's letters were assigned under rule **{rule}**, not the selected **{transfer['rule']}**.")
    return L


def transfer_table_md(transfer) -> list[str]:
    """Item 1: the holdout transfer endpoints from transfer_check.csv, one row per rule."""
    st = transfer["stats"]
    rules = [("letter_rank", "Rank letter (advocate only)"), ("R1", "R1"), ("R2", "R2")]
    rules = [(k, lab) for k, lab in rules if ("fpr_A", k) in st]
    if not rules:
        return []

    def pct_ci(stat, rule):
        v, lo, hi, _ = st.get((stat, rule), (None, None, None, None))
        s = fpct(v)
        if s != "—" and lo not in (None, "") and hi not in (None, ""):
            s += f" [{100 * to_float(lo):.1f}, {100 * to_float(hi):.1f}]"
        return s

    def n_of(stat, rule):
        return fint(st.get((stat, rule), (None, None, None, None))[3])

    def count_of(stat, rule):
        v, _, _, n = st.get((stat, rule), (None, None, None, None))
        return "—" if v in (None, "") else f"{fint(v)}/{fint(n)}"

    n_neg, n_pos, n_sd = n_of("fpr_A", rules[0][0]), n_of("recall_AB", rules[0][0]), n_of("stress_D_AB_count", rules[0][0])
    L = []
    L.append(f"Holdout transfer endpoints — {HOLDOUT_LABEL}, from `{transfer['csv'].name}` (95 % Clopper–Pearson")
    L.append(f"CIs; {n_neg} negatives, {n_pos} positives, {n_sd} stress_D panels):")
    L.append("")
    L.append(TBL_OPEN)
    L.append("")
    L.append("| Rule | FPR@A | FPR@A∪B | recall@A∪B | stress_D at A∪B |")
    L.append("|:--|--:|--:|--:|--:|")
    for k, lab in rules:
        mark = " — selected" if k == transfer["rule"] else ""
        L.append(f"| **{lab}**{mark} | {pct_ci('fpr_A', k)} | {pct_ci('fpr_AB', k)} | {pct_ci('recall_AB', k)} | "
                 f"{count_of('stress_D_AB_count', k)} ({pct_ci('stress_D_AB_rate', k)}) |")
    L.append("")
    L.append(TBL_CLOSE)
    return L


def holdout_caveat_md(meta: dict, transfer) -> list[str]:
    """Item 5: the holdout caveat, labelled as the program's opus5-xhigh result and keyed on
    this run's model."""
    H = HOLDOUT
    L = []
    L.append(f"- **Recall on hard lenses is limited** — {HOLDOUT_LABEL}. t_A / t_B were fit on clean random")
    L.append("  negatives, so the letters control false positives, not completeness. On the truth holdout (COWLS")
    L.append(f"  and literature lenses vs catalogue-purged negatives) the opus5-xhigh advocate reaches recall")
    L.append(f"  {H['advocate_recall_5fpr']} at 5 % FPR with AUC {H['advocate_auc']}; the full critic stack ranks at AUC")
    L.append(f"  {H['stack_auc']} on {H['stack_n_neg_scored']} scored negatives — it under-ranks the advocate — but it is the")
    L.append(f"  layer that demotes refuted panels: its D-rate on the stress_D panels was {H['stack_stressD_rate']}")
    L.append(f"  against {H['advocate_only_stressD_rate']} for every advocate-only arm (provisional letters).")
    if transfer is not None and transfer["stats"]:
        st = transfer["stats"]
        bits = []
        for k, lab in (("letter_rank", "rank letter"), ("R1", "R1"), ("R2", "R2")):
            if ("fpr_AB", k) not in st:
                continue
            v_fpr = st[("fpr_AB", k)][0]
            v_rec = st.get(("recall_AB", k), (None,) * 4)[0]
            v_sd, _, _, n_sd = st.get(("stress_D_AB_count", k), (None,) * 4)
            bits.append(f"{lab}: FPR@A∪B {fpct(v_fpr)}, recall@A∪B {fpct(v_rec)}, stress_D at A∪B "
                        f"{fint(v_sd)}/{fint(n_sd)}")
        if bits:
            L.append("  After calibration the transfer check gives — " + "; ".join(bits) + " (table under")
            L.append("  \"Why this rule\").")
    L.append("  Hence *advocate ranks, critic stack certifies*.")
    if meta["model"] == "opus5" and str(meta["effort"]) == "xhigh":
        L.append(f"  This run uses that opus5-xhigh configuration (thinking `{meta['thinking']}`, effort")
        L.append(f"  `{meta['effort']}`).")
    else:
        L.append(f"  This run used `{meta['model']}` (thinking `{meta['thinking']}`, effort `{meta['effort']}`); the")
        L.append("  holdout numbers quoted are the program's result for the opus5-xhigh configuration, not a")
        L.append("  measurement of this run's configuration.")
    L.append("  For scale: the original pass-count verifier gave 0 of 3 passes to 23 of the 24 known lenses it")
    L.append("  examined, and none of the 31 COWLS lenses in the truth set came out of it at A/B.")
    return L


# ----------------------------------------------------------------------------- pages
def index_page(rows, meta, rule, transfer, pages, top_n, csv_name) -> str:
    c = summary_counts(rows, meta["tau0"])
    model_id = MODEL_IDS.get(meta["model"], meta["model"])
    model_disp = f"`{meta['model']}`" + (f" ({model_id})" if model_id != meta["model"] else "")
    calibrated = str(meta["letter_source"]).endswith("_calibrated")
    cost = f"${float(meta['cost']):,.2f}" if meta["cost"] is not None else "—"
    first_page = pages[0][0]
    excl = f" ({c['n_unscored']} unscored rows excluded)" if c["n_unscored"] else ""

    def page_for(rank: int) -> str:
        for name, s, e in pages:
            if s <= rank <= e:
                return name
        raise KeyError(rank)

    def item_link(r) -> str:
        return f"[{cand_label(r)}]({page_for(int(r['rank']))}#rank-{int(r['rank'])})"

    def final_cell(r) -> str:
        f = letter(r["our_letter_final"])
        return f"**{f}**" + (f" {FLAG}" if is_true(r["our_needs_human"]) else "")

    demote_verb = "demoted" if rule == "R1" else "vetoed to D"

    L: list[str] = []
    L.append(INDEX_H1)
    L.append("")
    L.append(STYLE)
    L.append("")
    L.append("The 100 top-ranked candidates of the JWST NIRCam strong-lens search, re-graded **blind** by the")
    L.append("evidence-first [LensJudge](../current/lensjudge/index.md) panel — images shuffled, footer stripped,")
    L.append("no candidate id, coordinates, filters, rank or original grade in front of the model — and set")
    L.append("beside the grades the original campaign assigned. Every candidate's original and annotated cutout")
    L.append("and its full advocate → critics → arbitrator record is on the per-rank pages.")
    L.append("")
    L.append(f"[:material-download: Download the comparison table (CSV)](data/{csv_name}){{ .md-button .md-button--primary }}")
    L.append(f"[:material-image-multiple: Browse the cutouts, ranks {pages[0][1]}–{pages[0][2]}]({first_page}){{ .md-button }}")
    L.append("")
    L.append('!!! abstract "The result in one line"')
    L.append(f"    Under the deployed letter (rule {rule}), **{c['same']}** of the {c['n_scored']} scored candidates{excl}")
    L.append(f"    keep their original letter, **{c['up']}** move up and **{c['down']}** move down (U, never verified,")
    L.append(f"    counted below D). The original campaign verified only **{c['n_graded']}** of them; on those the")
    L.append(f"    two instruments agree on **{c['g_same']}**, the new judge is higher on **{c['g_up']}** and lower on")
    L.append(f"    **{c['g_down']}**. At A/B: original **{c['orig_AB']}**, new judge **{c['final_AB']}** (final letter)")
    L.append(f"    / **{c['rank_AB']}** (advocate-only rank letter); **{c['U_AB']}** of the **{c['n_U']}** never-verified")
    L.append(f"    (U) candidates reach A/B under the final letter, and **{c['n_needs_human']}** candidates carry the")
    L.append(f"    arbitrator's *needs human* flag. Model grades are a ranked reading of the pixels, **not** human vetting.")
    L.append("")
    if c["n_unscored"]:
        L.append('!!! warning "Unscored items"')
        L.append(f"    {c['n_unscored']} of the {c['n']} candidates have no final letter (parse failure or unscored row);")
        L.append("    they are shown as — in the tables and excluded from every same/up/down, A/B and demotion count")
        L.append("    on this page.")
        L.append("")
    if not calibrated:
        L.append('!!! warning "Provisional letters"')
        L.append(f"    This run's letters are stamped `{meta['letter_source']}` — the thresholds are **not** calibrated")
        L.append("    for this model. Read the p_evidence ranking; treat the letters as provisional.")
        L.append("")
    L.append("---")
    L.append("")
    # ---------------------------------------------------------------- what was graded
    L.append("## What was graded")
    L.append("")
    L.append("The original campaign is an agentic JWST NIRCam strong-lens search whose pipeline flags candidate")
    L.append("deflectors on six-panel cutout composites, has an *inspector* agent score each flag with a confidence")
    L.append("(0–100), and sends the most confident flags to a verification queue. There, three adversarial")
    L.append("*refuter* personas — artifact, morphology, geometry — each try to refute the candidate, and the grade")
    L.append("is the **pass count**: A = 3 passes, B = 2, C = 1, D = 0; U marks a flag that was never verified.")
    L.append("Of the run's top 100 by pipeline rank, "
             + ", ".join(f"**{c['orig_dist'][g]}** are {g}" for g in ("A", "B", "C", "D") if c["orig_dist"][g])
             + f" and **{c['orig_dist']['U']}** are U.")
    L.append("")
    L.append("**Blind protocol.** The JWST repository ships the same 100 composites shuffled out of rank order,")
    L.append("renamed 001–100 and with the footer strip (candidate id, coordinates, magnitude) removed, precisely")
    L.append("so a reviewer scores each field on the imaging alone. The judge ran over those files with frozen,")
    L.append("item-agnostic prompts; the only answer-key field read before scoring was the *layout* (two-band")
    L.append(f"colour vs single-band grey composite{'' if meta['n_gray'] is None else f': {meta['n'] - meta['n_gray']} colour / {meta['n_gray']} grey'}),")
    L.append("which selects the per-role views. Ranks, ids and the original grades were joined on afterwards.")
    L.append("")
    # ---------------------------------------------------------------- how the judge grades
    L.append("## How the new judge grades")
    L.append("")
    L.append("The pass-count verifier is replaced by an evidence-first panel in which the grade is *computed from*")
    L.append("the explanation:")
    L.append("")
    L.append("1. **Advocate** — lists *located* evidence items (panel, radius, position-angle span), scores five")
    L.append("   NIRCam-adapted criteria (source contrast, low surface brightness, curvature, counter-image, arc")
    L.append("   morphology) and returns **p_evidence**, the probability that the located evidence is lensing.")
    L.append("2. **Three competence-bounded critics** — artifact, geometry, morphology. Each must either abstain or")
    L.append("   *name* an alternative from a fixed vocabulary (merger, companion projection, spiral arm, edge-on")
    L.append("   disk, subtraction residual, PSF wing, diffraction spike, shell/tidal, ring galaxy, …), locate it, say")
    L.append("   which advocate items it accounts for and grade its refutation strength *r*. Forbidden grounds: the")
    L.append("   size of the Einstein radius, colour alone, and the symmetric residual of the circular-subtraction panel.")
    L.append("3. **Arbitrator** — sees the image and every text and rules each critic **upheld**, **partial** or")
    L.append("   **overruled**, writes the rationale paragraph, gives its own free letter and may flag *needs human*.")
    L.append("")
    L.append("Scores: **S** = p_evidence × Π(1 − r·a) over every critic (a = the fraction of the evidence the")
    L.append("alternative covers); **S_arb** = the same product over upheld/partial refutations only, with the")
    L.append("arbitrated coverage. Two letters are deployed:")
    L.append("")
    L.append(TBL_OPEN)
    L.append("")
    L.append("| Letter | From | Meaning |")
    L.append("|:--|:--|:--|")
    L.append("| **Rank letter** | p_evidence, advocate only | The letter whose false-positive rate the calibration controls; the ranking score. |")
    L.append(final_letter_row(rule))
    L.append("")
    L.append(TBL_CLOSE)
    L.append("")
    L.extend(why_rule_md(rule, transfer))
    L.append("")
    if transfer is not None and transfer["csv"] is not None:
        tt = transfer_table_md(transfer)
        if tt:
            L.extend(tt)
            L.append("")
    L.extend(letter_rules_md(rule, meta))
    L.append("")
    cal = calibration_md(meta)
    if cal:
        L.extend(cal)
        L.append("")
    n_called = "—" if c["n_called"] is None else f"**{c['n_called']}**"
    L.append(f"**Critics engaged.** In this run the critics were called on {n_called} of the {c['n']} candidates")
    L.append(f"(p_evidence ≥ τ0) and the arbitrator ruled on **{c['n_arb']}**; the stack {demote_verb} **{c['n_demoted']}**.")
    L.append("")
    L.append(f"**This run.** Model {model_disp} on the `{meta['backend']}` backend, thinking `{meta['thinking']}`,")
    L.append(f"effort `{meta['effort']}`, k = {meta['k']} (arm `{meta['arm']}`, full stack), scored {meta['scored_at'] or '—'}"
             + (f", letters re-assigned {meta['relettered_at']}" if meta["relettered_at"] else "")
             + f"; thresholds key `{meta['thresholds_key']}` (`{meta['thresholds_sha16']}`); API cost {cost}.")
    L.append("")
    # ---------------------------------------------------------------- legend
    L.append("## How to read the annotated cutouts")
    L.append("")
    L.append("Each candidate page shows the composite the judge saw and the same composite with the panel's")
    L.append("stored records painted on it. Panels in the two-band **colour** layout: **a** normal 10″, **b** deep 10″,")
    L.append("**c** two-band colour 10″, **d** deep 3.5″ zoom, **e** colour 3.5″ zoom, **f** circular-subtraction")
    L.append("residual 3.5″. In the single-band **grey** layouts there is no colour information: **c** is the 10″")
    L.append("circular-subtraction residual and **e** the normal-stretch 3.5″ zoom. North up, East left; the ticked")
    L.append("galaxy is the flagged deflector at every panel's centre.")
    L.append("")
    L.append("- **Cyan arcs** (k1, k2, …) are the advocate's located evidence items, drawn at their stated radius and")
    L.append("  position-angle span in the cited panel, in panel **a**, and in the 3.5″ zoom **d** when they fit")
    L.append("  (r ≤ 1.7″). A zero-length span is a small circle, a full 360° span a ring, a counter-image a cyan cross.")
    L.append("- **Dashed sectors** are the critics' location boxes (radius range × PA range), coloured by the")
    L.append("  arbitrator's ruling: <span style=\"color:#ff5050\">**red = upheld**</span>,")
    L.append("  <span style=\"color:#ffaa28\">**orange = partial**</span>, <span style=\"color:#aaaaaa\">**grey = overruled**</span>,")
    L.append("  <span style=\"color:#c8b400\">**yellow = no ruling**</span> — no arbitrator ruling for that critic (the")
    L.append("  arbitrator was absent or gave no ruling on it). The label is the role (Art / Geo / Mor) and the")
    L.append("  named alternative; abstaining critics draw nothing.")
    L.append("- The **legend strip** below the composite gives the rank → final letter, the veto if any, p_ev / S / S_arb,")
    L.append("  the arbitrator's letter and *needs_human* flag, scale class and layout; then one cyan line per")
    L.append("  evidence item (`*` = the arbitrator kept it; `[not drawn]` / `[r > panel X]` when it could not be")
    L.append("  placed) and one line per critic in its ruling colour with its refutation strength r.")
    L.append("")
    # ---------------------------------------------------------------- agreement
    L.append("## Agreement")
    L.append("")
    L.append(f"Original pass-count grade (rows) against the new judge's **final letter** (rule {rule}):")
    L.append("")
    L.append(TBL_OPEN)
    L.append("")
    L.extend(crosstab_md(rows, "our_letter_final", "final"))
    L.append("")
    L.append(TBL_CLOSE)
    L.append("")
    L.append("…and against the advocate-only **rank letter** (before the critic stack):")
    L.append("")
    L.append(TBL_OPEN)
    L.append("")
    L.extend(crosstab_md(rows, "our_letter_rank", "rank letter"))
    L.append("")
    L.append(TBL_CLOSE)
    L.append("")
    if c["alt_dist"]:
        alts = ", ".join(f"{a} ({n})" for a, n in c["alt_dist"].most_common())
        if rule == "R1":
            L.append(f"The {c['n_demoted']} demotions name these surviving alternatives (a candidate can carry more than")
            L.append(f"one): {alts}.")
        else:
            L.append(f"The {c['n_demoted']} D-rule vetoes name these upheld, full-coverage alternatives (a candidate can")
            L.append(f"carry more than one): {alts}.")
        L.append("")
    # ---------------------------------------------------------------- all 100
    L.append("## All 100")
    L.append("")
    L.append("Sorted by original pipeline rank. Click a candidate for its cutouts and the full record. Design")
    L.append(f"anchors are marked {DAGGER}[^anchor]; {FLAG} marks the arbitrator's *needs human* flag. p_ev is the")
    if rule == "R1":
        L.append("advocate's p_evidence (the ranking score), S_arb the arbitrated score the final letter is read from.")
    else:
        L.append("advocate's p_evidence (the ranking score and, under R2, the score the letter is read from); S_arb is")
        L.append("the arbitrated score, shown for reference.")
    L.append("")
    L.append(TBL_OPEN)
    L.append("")
    L.append("| Rank | Candidate | Original | Ours (final) | Rank letter | p_ev | S_arb | Demoted by |")
    L.append("|--:|:--|:--:|:--:|:--:|--:|--:|:--|")
    for r in rows:
        L.append(f"| {int(r['rank'])} | {item_link(r)} | {r['nate_grade']} | {final_cell(r)} | "
                 f"{letter(r['our_letter_rank'])} | {fnum(r['our_p_evidence'], 2)} | {fnum(r['our_S_arb'], 3)} | "
                 f"{fmt_veto(r['our_veto'])} |")
    L.append("")
    L.append(TBL_CLOSE)
    L.append("")
    L.append(f"[^anchor]: Ranks {', '.join(str(k) for k in ANCHOR_RANKS)} are the scheme's **design anchors**: their")
    L.append("    outcomes known during design shaped the scheme's mechanisms (forbidden grounds, the coverage")
    L.append("    rule). They are not an independent test of the judge, whatever letter they receive here.")
    L.append("")
    # ---------------------------------------------------------------- ranked by the judge
    L.append("## Ranked by the new judge")
    L.append("")
    L.append(f"The top {top_n} by the advocate's **p_evidence** (ties broken by S). This is the ranking the program")
    L.append(f"deploys — {HOLDOUT_LABEL}: the advocate ranks far better than the full critic product, whose job")
    L.append("is to certify and demote, not to order.")
    L.append("")
    L.append(TBL_OPEN)
    L.append("")
    L.append("| # | Orig. rank | Candidate | Original | Ours (final) | Rank letter | p_ev | S_arb | Demoted by |")
    L.append("|--:|--:|:--|:--:|:--:|:--:|--:|--:|:--|")

    def keyf(r):
        def f(x):
            try:
                v = float(x)
                return -1.0 if math.isnan(v) else v
            except (TypeError, ValueError):
                return -1.0
        return (-f(r["our_p_evidence"]), -f(r["our_S"]), int(r["rank"]))

    for i, r in enumerate(sorted(rows, key=keyf)[:top_n], 1):
        L.append(f"| {i} | {int(r['rank'])} | {item_link(r)} | {r['nate_grade']} | {final_cell(r)} | "
                 f"{letter(r['our_letter_rank'])} | {fnum(r['our_p_evidence'], 2)} | {fnum(r['our_S_arb'], 3)} | "
                 f"{fmt_veto(r['our_veto'])} |")
    L.append("")
    L.append(TBL_CLOSE)
    L.append("")
    # ---------------------------------------------------------------- caveats
    L.append("## Caveats")
    L.append("")
    L.append("- **Model grades are not human vetting.** Every letter here is a language model's reading of six")
    L.append("  panels. The letters are calibrated for false-positive rate on random negatives, not validated")
    L.append("  against expert inspection of these 100 fields; a human grader's blind pass on the same kit is the")
    L.append("  test that matters.")
    L.append("- **Blind to metadata.** The judge saw pixels only — no filters, redshifts, coordinates, Einstein-radius")
    L.append("  estimate, inspector confidence or original grade. A cluster-scale system with its arcs outside the")
    L.append("  10″ field, or a system whose case rests on the footer, is at a disadvantage by construction.")
    L.append(f"- **Design anchors.** Ranks {', '.join(str(k) for k in ANCHOR_RANKS)} ({DAGGER}) shaped the scheme's")
    L.append("  mechanisms during design; their letters are consistency checks, not evidence.")
    L.extend(holdout_caveat_md(meta, transfer))
    L.append(f"- **needs_human flags.** The arbitrator flagged **{c['n_needs_human']}** candidates ({c['n_needs_human_AB']} of")
    L.append("  them at A/B) as contested — the escalation set a human should look at first.")
    L.append("- **One draw.** k = " + str(meta["k"]) + " replicate per item; a re-run moves individual scores.")
    L.append("")
    L.append("---")
    L.append("")
    L.append("Pages: " + " · ".join(f"[{page_title(s, e)}]({name})" for name, s, e in pages) + ".")
    L.append("")
    return "\n".join(L)


def ranks_page(rows_page, start, end, pages, idx, meta, explain_dir, annot_dir, annot_idx, out, warnings) -> tuple[str, int]:
    name, _, _ = pages[idx]
    prev_ = pages[idx - 1] if idx > 0 else None
    next_ = pages[idx + 1] if idx + 1 < len(pages) else None
    img_bytes = 0
    L: list[str] = []
    L.append(f"# JWST Top-100 regrade: ranks {start}–{end}")
    L.append("")
    L.append(STYLE)
    L.append("")
    L.append(f"Candidates {start}–{end} by original pipeline rank, each with the composite the judge saw, the")
    L.append("annotated composite and the full grading record. Legend, thresholds and the summary tables are on")
    L.append("the [overview](index.md). Click an image to open it full size.")
    L.append("")
    L.append("Pages: " + " · ".join((f"**{page_title(s, e)}**" if n == name else f"[{page_title(s, e)}]({n})")
                                     for n, s, e in pages) + ".")
    L.append("")
    for r in rows_page:
        rank = int(r["rank"])
        scr = r["scrambled_item"].strip()
        anchor = rank in ANCHOR_RANKS
        L.append("---")
        L.append("")
        L.append(f"## Rank {rank}: {cand_label(r)} {{#rank-{rank}}}")
        L.append("")
        if anchor:
            L.append(f"*{DAGGER} Design anchor — this item shaped the scheme's mechanisms during design; it is not an")
            L.append("independent test.*")
            L.append("")
        npass = "—" if r["nate_grade"] == "U" else fint(r["nate_n_pass"])   # U: never verified
        theta = fnum(r["blind_theta_E_arcsec"], 2)
        L.append(TBL_OPEN)
        L.append("")
        L.append("| Original grade | Passes | Inspector conf. | Pipeline θ_E |")
        L.append("|:--:|:--:|:--:|:--:|")
        L.append(f"| **{r['nate_grade']}** | {npass}{'/3' if npass != '—' else ''} | {fint(r['nate_inspector_conf'])} | "
                 f"{theta + ('″' if theta != '—' else '')} |")
        L.append("")
        L.append("| New (final) | Rank letter | p_ev | S_arb | Demoted by | Arb. letter | Needs human |")
        L.append("|:--:|:--:|--:|--:|:--|:--:|:--:|")
        L.append(f"| **{letter(r['our_letter_final'])}** | {letter(r['our_letter_rank'])} | {fnum(r['our_p_evidence'], 2)} | "
                 f"{fnum(r['our_S_arb'], 3)} | {fmt_veto(r['our_veto'])} | {letter(r['our_letter_llm'])} | "
                 f"{'yes ' + FLAG if is_true(r['our_needs_human']) else 'no'} |")
        L.append("")
        L.append(TBL_CLOSE)
        L.append("")
        extra = []
        if r.get("discovery_status", "").strip():
            extra.append(f"pipeline discovery status `{r['discovery_status'].strip()}`")
        if r.get("our_scale_class", "").strip():
            extra.append(f"scale class `{r['our_scale_class'].strip()}`")
        if extra:
            L.append("<small>" + " · ".join(extra) + "</small>")
            L.append("")
        # images
        ai = annot_idx.get(scr, {})
        orig_src = annot_dir / (ai.get("orig_file") or f"{scr}_orig.jpg")
        annot_src = annot_dir / (ai.get("annot_file") or f"{scr}_annot.jpg")
        for kind, src, caption in (("orig", orig_src, "What the judge saw (blind, footer removed)"),
                                   ("annot", annot_src, "Advocate evidence (cyan) and critic alternatives coloured by the arbitrator ruling")):
            dst_name = f"rank-{rank:03d}-{kind}.jpg"
            if src.exists():
                shutil.copyfile(src, out / "img" / dst_name)
                img_bytes += src.stat().st_size
                L.append('<figure markdown="span">')
                L.append(f"  [![Rank {rank} {kind} composite](img/{dst_name}){{ width=\"100%\" }}](img/{dst_name}){{ target=_blank }}")
                L.append(f"  <figcaption>{caption}</figcaption>")
                L.append("</figure>")
            else:
                warnings.append(f"rank {rank}: missing image {src}")
                L.append(f"*({kind} composite not available: `{src.name}` missing from the annot dir)*")
            L.append("")
        # explain
        exp = explain_dir / f"{scr}.md"
        L.append('???+ note "Why this grade — advocate, critics, arbitrator"')
        if exp.exists():
            L.extend(explain_block(exp.read_text(encoding="utf-8")))
        else:
            warnings.append(f"rank {rank}: missing explanation {exp}")
            L.append(f"    *No explanation file (`{exp.name}`) in the explain dir.*")
        L.append("")
    L.append("---")
    L.append("")
    nav = []
    if prev_:
        nav.append(f"[:material-arrow-left: {page_title(prev_[1], prev_[2])}]({prev_[0]}){{ .md-button }}")
    nav.append("[:material-view-list: Overview](index.md){ .md-button }")
    if next_:
        nav.append(f"[{page_title(next_[1], next_[2])} :material-arrow-right:]({next_[0]}){{ .md-button }}")
    L.append("\n".join(nav))
    L.append("")
    return "\n".join(L), img_bytes


# ----------------------------------------------------------------------------- nav
def update_nav(mkdocs: Path, docs_rel: str, pages) -> str:
    """Insert/replace the NAV_TITLE block as the last child of the 'Current' section.
    Returns a short description of what changed."""
    text = mkdocs.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    i_cur = next((i for i, l in enumerate(lines) if re.match(r"^  - Current:\s*$", l)), None)
    if i_cur is None:
        die(f"no '  - Current:' section in {mkdocs}")
    j = i_cur + 1
    while j < len(lines) and (not lines[j].strip() or lines[j].startswith("      ")):
        j += 1
    block = lines[i_cur + 1:j]
    # drop an existing entry
    kept, k, removed = [], 0, False
    while k < len(block):
        if re.match(rf"^      - {re.escape(NAV_TITLE)}:\s*$", block[k]):
            removed = True
            k += 1
            while k < len(block) and (block[k].startswith("          ") or not block[k].strip()):
                k += 1
            continue
        kept.append(block[k])
        k += 1
    # trailing blank lines belong after the block
    tail = []
    while kept and not kept[-1].strip():
        tail.insert(0, kept.pop())
    new = [f"      - {NAV_TITLE}:\n", f"          - Overview: {docs_rel}/index.md\n"]
    new += [f'          - "{page_title(s, e)}": {docs_rel}/{name}\n' for name, s, e in pages]
    out = lines[:i_cur + 1] + kept + new + tail + lines[j:]
    new_text = "".join(out)
    if new_text == text:
        return "nav unchanged"
    mkdocs.write_text(new_text, encoding="utf-8")
    return f"nav: {'replaced' if removed else 'added'} '{NAV_TITLE}' ({1 + len(pages)} children) under Current"


# ----------------------------------------------------------------------------- output guard
def check_out_dir(out: Path) -> None:
    """Refuse to delete or overwrite anything in an --out directory the builder does not own:
    it must be absent, empty, carry the SENTINEL file, or hold an index.md whose first line
    is the builder's own H1."""
    if not out.exists():
        return
    if not out.is_dir():
        die(f"--out {out} exists and is not a directory")
    if not any(out.iterdir()):
        return
    if (out / SENTINEL).is_file():
        return
    idx = out / "index.md"
    if idx.is_file():
        with idx.open(encoding="utf-8") as fh:
            first = fh.readline().rstrip("\r\n")
        if first == INDEX_H1:
            return
    die(f"--out {out} is not empty and was not written by this builder (no {SENTINEL} sentinel and "
        f"index.md does not start with {INDEX_H1!r}); refusing to delete or overwrite anything in it")


# ----------------------------------------------------------------------------- main

def write_public_csv(src: Path, dst: Path) -> None:
    """Public copy of the comparison CSV: drop the blind kit id (a kit-to-candidate key) and
    rename the original-campaign columns nate_* -> original_* (no person names in the data file)."""
    with open(src, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise SystemExit(f"empty comparison CSV: {src}")
    cols = [c for c in rows[0].keys() if c != "scrambled_item"]
    ren = {c: ("original_" + c[len("nate_"):] if c.startswith("nate_") else c) for c in cols}
    with open(dst, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow([ren[c] for c in cols])
        for r in rows:
            w.writerow([r[c] for c in cols])

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--comparison", type=Path, help="default RUN_DIR/scrambled100_comparison.csv")
    ap.add_argument("--explain-dir", type=Path, help="default RUN_DIR/explain")
    ap.add_argument("--annot-dir", type=Path, help="default RUN_DIR/annot")
    ap.add_argument("--meta", type=Path, help="default: the single *.meta.json in RUN_DIR")
    ap.add_argument("--transfer-json", type=Path,
                    help="golden/transfer_check.py selected_rule.json (default: RUN_DIR/../transfer_opus5/"
                         "selected_rule.json when it exists, else omitted); a transfer_check.csv beside it is read too")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--per-page", type=int, default=25)
    ap.add_argument("--top", type=int, default=25, help="rows in 'Ranked by the new judge'")
    ap.add_argument("--mkdocs", type=Path, default=DEFAULT_MKDOCS)
    ap.add_argument("--no-nav", action="store_true", help="do not touch mkdocs.yml")
    a = ap.parse_args(argv)

    run = a.run_dir.resolve()
    if not run.is_dir():
        die(f"run dir not found: {run}")
    comparison = (a.comparison or run / "scrambled100_comparison.csv").resolve()
    explain_dir = (a.explain_dir or run / "explain").resolve()
    annot_dir = (a.annot_dir or run / "annot").resolve()
    meta_path = (a.meta or find_meta(run)).resolve()
    out = a.out.resolve()
    for p, what in ((comparison, "comparison CSV"), (explain_dir, "explain dir"), (annot_dir, "annot dir"), (meta_path, "meta json")):
        if not p.exists():
            die(f"{what} not found: {p}")
    if a.per_page < 1:
        die("--per-page must be >= 1")
    if a.transfer_json is not None:
        transfer_path = a.transfer_json.resolve()
        if not transfer_path.is_file():
            die(f"transfer json not found: {transfer_path}")
    else:
        cand = (run.parent / "transfer_opus5" / "selected_rule.json").resolve()
        transfer_path = cand if cand.is_file() else None

    rows = load_rows(comparison)
    meta = load_meta(meta_path)
    rule = deployed_rule(rows, meta)
    transfer = load_transfer(transfer_path) if transfer_path else None
    annot_idx = load_annot_index(annot_dir)

    # pages by original rank
    n = len(rows)
    pages = []
    for s in range(1, n + 1, a.per_page):
        e = min(s + a.per_page - 1, n)
        pages.append((rank_page_name(s, e), s, e))

    # recreate owned outputs only, and only in a directory the builder owns
    check_out_dir(out)
    out.mkdir(parents=True, exist_ok=True)
    for sub in ("img", "data"):
        shutil.rmtree(out / sub, ignore_errors=True)
        (out / sub).mkdir()
    for old in list(out.glob("ranks-*.md")) + [out / "index.md"]:
        if old.exists():
            old.unlink()
    (out / SENTINEL).write_text(SENTINEL_TEXT, encoding="utf-8")

    csv_name = "comparison.csv"
    write_public_csv(comparison, out / "data" / csv_name)

    warnings: list[str] = []
    written = []
    img_bytes = 0
    for idx, (name, s, e) in enumerate(pages):
        rows_page = [r for r in rows if s <= int(r["rank"]) <= e]
        text, nb = ranks_page(rows_page, s, e, pages, idx, meta, explain_dir, annot_dir, annot_idx, out, warnings)
        (out / name).write_text(text, encoding="utf-8")
        written.append(out / name)
        img_bytes += nb
    (out / "index.md").write_text(index_page(rows, meta, rule, transfer, pages, a.top, csv_name), encoding="utf-8")
    written.insert(0, out / "index.md")

    nav_msg = "nav: skipped (--no-nav)"
    if not a.no_nav:
        try:
            docs_rel = out.relative_to(DOCS).as_posix()
        except ValueError:
            docs_rel = None
        if docs_rel is None:
            warnings.append(f"--out {out} is not under {DOCS}: nav not updated")
        else:
            nav_msg = update_nav(a.mkdocs.resolve(), docs_rel, pages)

    c = summary_counts(rows, meta["tau0"])
    n_img = len(list((out / "img").glob("*.jpg")))
    print(f"run dir      : {run}")
    print(f"comparison   : {comparison}  ({n} rows, rule {rule})")
    print(f"explain dir  : {explain_dir}")
    print(f"annot dir    : {annot_dir}")
    print(f"meta         : {meta_path}  (model {meta['model']}, {meta['thinking']}/{meta['effort']}, {meta['letter_source']}, "
          f"key {meta['thresholds_key']}, tau0 {meta['tau0']} t_A {meta['t_A']} t_B {meta['t_B']})")
    if transfer:
        print(f"transfer     : {transfer['json']}  (selected {transfer['rule']}"
              + (f"; endpoints from {transfer['csv'].name}" if transfer["csv"] else "; no transfer_check.csv") + ")")
    else:
        print("transfer     : none (no --transfer-json and no RUN_DIR/../transfer_opus5/selected_rule.json)")
    print(f"out          : {out}")
    print(f"pages        : index.md + {len(pages)} ranks pages ({a.per_page}/page)")
    print(f"images       : {n_img} JPEGs, {img_bytes / 1e6:.1f} MB")
    print(f"letters      : same {c['same']} / up {c['up']} / down {c['down']} of {c['n_scored']} scored "
          f"(unscored {c['n_unscored']}); graded {c['n_graded']}: same {c['g_same']} up {c['g_up']} down {c['g_down']}; "
          f"A/B orig {c['orig_AB']} final {c['final_AB']} rank {c['rank_AB']}; U->A/B {c['U_AB']}/{c['n_U']}; "
          f"demoted {c['n_demoted']} (veto {c['n_veto']}); critics called {c['n_called']} arbitrator {c['n_arb']}; "
          f"needs_human {c['n_needs_human']}")
    print(nav_msg)
    for w in warnings:
        print("WARNING:", w)
    return 0


if __name__ == "__main__":
    sys.exit(main())
