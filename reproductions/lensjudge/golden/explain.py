#!/usr/bin/env python3
"""golden/explain.py — the rationale renderer: stored per-role records → Markdown + one paragraph.

The v2 scheme computes the grade FROM the explanation: every letter is `aggregate_v2` arithmetic
on the advocate's located items, the critics' located alternatives and the arbitrator's rulings
(REGISTRY.md "Deployment rule v2-deploy"). This module renders those stored records — as
`golden/records.py` rebuilds them from a run's votes parquet — into (a) a Markdown block and
(b) a one-paragraph plain-text summary per item, plus a JSON-able `facts` dict, so a reader can
check each number against the record it came from. Nothing here re-scores: S / S_arb are
recomputed only to SHOW the product term by term, next to the stored values.

Traceability rule: every sentence is a stored field or an `aggregate_v2` predicate spelled out
(the threshold comparison that decided the letter, the A guards, the D rule). Model prose —
item `what`, `nothing_because`, `notes`, `alternative_desc`, ruling `why`, `rationale` — is
always wrapped in “…” quotes and never paraphrased; the arbitrator's rationale is verbatim in
the Markdown and an ellipsis-marked excerpt in the paragraph when it does not fit.

Markdown order: (1) headline — letter, p_evidence, S_arb, scale, the mechanical threshold
sentence; (2) Advocate — criteria, items, nothing_because, scale/neighbour flags, notes;
(3) Critics — per role: abstention | alternative, location, accounts_for / leaves_standing,
strength, coverage, the arbitrator's ruling for that critic; (4) Arbitrator — surviving items,
letter_llm, needs_human, rationale; (5) Score — S and S_arb decomposed from the a_/r_/ruling
terms, the stored letters, and (with `deploy=`) letter_rank / letter_final / veto.

Handles advocate-only runs (no critic keys: not called), parse failures (None records, named
as such), votes rows whose raw was lost (record not rebuilt while the stored row is parse_ok),
missing thresholds (the row's tau0 / t_A / t_B, else "thresholds unavailable") and NaN S.

API:
    explain_item(name, records, row, thresholds=None, deploy=None) -> {markdown, paragraph, facts}
    explain_run(preds, records, thresholds=None, rule=None)        -> list of those, preds order
    write_outputs(results, out_dir, formats)                      -> {format: path}

CLI (zero API, read-only on the run; writes only into --out-dir):
    python lensjudge/golden/explain.py --preds outputs/<run>.parquet [--votes …]
        [--thresholds golden/thresholds_v2.json --model-key sonnet_api] [--rule R1|R2]
        --out-dir DIR [--format md,csv,json]
md: one <safe_name>.md per item + index.md; csv: explain.csv (name, letter, paragraph, pinned);
json: facts.json ({name: facts}, sha sidecar). Prints count tables only (no rationale text).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd  # noqa: E402

from lensjudge.golden import _util, aggregate_v2  # noqa: E402
from lensjudge.golden import records as R  # noqa: E402
from lensjudge.golden import schemas_panel as sp  # noqa: E402
from lensjudge.golden import views  # noqa: E402

CRITIC_ROLES = aggregate_v2.CRITIC_ROLES            # ("artifact", "geometry", "morphology")
ROLES = aggregate_v2.ROLES                          # advocate, critics, arbitrator
CRITERIA_V2 = sp.CRITERIA_V2
# the composite panels' fields of view (views.BUILTIN_GLOSS; layout-independent): a b c 10", d e f 3.5"
PANEL_FOV = {str(k): float(v) for k, v in views.BUILTIN_GLOSS["layouts"]["color"]["fov_arcsec"].items()}
CRITERIA_BY_INDEX = {i + 1: c for i, c in enumerate(CRITERIA_V2)}   # item.criteria 1..5 → names
LETTER_CRITERIA = aggregate_v2.LETTER_CRITERIA
LETTER_CRITERIA_MIN = aggregate_v2.LETTER_CRITERIA_MIN
STRONG_R = aggregate_v2.STRONG_R
DEPLOY_RULES = aggregate_v2.DEPLOY_RULES
FORMATS = ("md", "csv", "json")
PARAGRAPH_MAX = 600                                 # hard cap on the plain-text paragraph
PARAGRAPH_ITEMS = 3                                 # items listed in the paragraph (the rest counted)
PARAGRAPH_WHAT = 40                                 # chars of an item's `what` quoted in the paragraph
PARAGRAPH_RATIONALE_MIN = 24                        # smallest rationale excerpt worth appending
PARAGRAPH_RATIONALE_MAX = 200                       # longest rationale excerpt in the paragraph
CSV_COLS = ("name", "letter", "paragraph")
INDEX_NAME, CSV_NAME, FACTS_NAME = "index.md", "explain.csv", "facts.json"
# what a thresholds `letter_source` label means (REGISTRY.md v2-deploy item 1: t_A / t_B fit at
# design-negative FPR <= 1% / <= 5%; aggregate_v2.resolve_thresholds labels)
CALIBRATED_NOTE = {"t_A": "calibrated at 1% FPR on clean negatives",
                   "t_B": "calibrated at 5% FPR on clean negatives"}
PROVISIONAL_NOTE = "provisional, not calibrated"
_EPS = 1e-9
_ATOL = 1e-6                                         # recomputed vs stored S agreement


# ------------------------------------------------------------------ small helpers
def _num(x: Any) -> Optional[float]:
    if x is None or isinstance(x, bool):
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) else v


def _val(row: Any, col: str, default: Any = None) -> Any:
    """A row field (dict or pandas Series), None for missing / NaN."""
    if row is None:
        return default
    get = getattr(row, "get", None)
    v = get(col, default) if callable(get) else getattr(row, col, default)
    if v is None:
        return default
    if isinstance(v, float) and math.isnan(v):
        return default
    if not isinstance(v, (str, bytes, list, tuple, dict)):
        try:
            if bool(pd.isna(v)):
                return default
        except (TypeError, ValueError):
            pass
    return v


def _f(x: Any, nd: int = 3) -> str:
    """A score / threshold / strength: up to `nd` decimals, at least 2, "NaN" when missing."""
    v = _num(x)
    if v is None:
        return "NaN"
    s = f"{v:.{nd}f}".rstrip("0")
    if "." not in s or len(s.split(".")[1]) < 2:
        s = f"{v:.2f}"
    return s


def _g(x: Any) -> str:
    """A geometric quantity (arcsec, degrees): integer when whole, else up to 2 decimals."""
    v = _num(x)
    if v is None:
        return "NaN"
    if abs(v - round(v)) < 1e-9:
        return str(int(round(v)))
    return f"{v:.2f}".rstrip("0").rstrip(".")


def _q(text: Any, collapse: bool = False) -> str:
    """Model prose, quoted (“…”) and never paraphrased; `collapse` folds whitespace."""
    s = "" if text is None else str(text)
    if collapse:
        s = " ".join(s.split())
    return f"“{s}”"


def _yn(x: Any) -> str:
    return "yes" if bool(x) else "no"


def _ks(ks: Any) -> str:
    ks = [int(k) for k in (ks or [])]
    return ", ".join(f"k{k}" for k in ks) if ks else "none"


def _ordered(roles) -> list:
    return sorted(roles, key=lambda r: (CRITIC_ROLES.index(r) if r in CRITIC_ROLES else len(CRITIC_ROLES), r))


def _py(x: Any) -> Any:
    """JSON-safe copy of a facts value (numpy scalars → python, NaN → None)."""
    if isinstance(x, dict):
        return {str(k): _py(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_py(v) for v in x]
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, str)) or x is None:
        return x
    if isinstance(x, float):
        return None if math.isnan(x) else x
    if hasattr(x, "item"):                              # numpy scalar
        return _py(x.item())
    return str(x)


# ------------------------------------------------------------------ thresholds
def item_thresholds(row: Any, thresholds: Optional[dict] = None) -> Optional[dict]:
    """The resolved {tau0, t_A, t_B, letter_source} for one item: `thresholds` when given, else
    the row's stored tau0 / t_A / t_B / letter_source; None when neither holds t_A and t_B."""
    if thresholds is not None and _num(thresholds.get("t_A")) is not None and _num(thresholds.get("t_B")) is not None:
        tau0 = _num(thresholds.get("tau0"))
        return {"tau0": aggregate_v2.PROVISIONAL["tau0"] if tau0 is None else tau0,
                "t_A": float(thresholds["t_A"]), "t_B": float(thresholds["t_B"]),
                "letter_source": str(thresholds.get("letter_source", "provisional"))}
    try:
        thr = R.thresholds_from_row(row if row is not None else {})
    except ValueError:
        return None
    return {k: thr[k] for k in ("tau0", "t_A", "t_B", "letter_source")}


def threshold_note(thresholds: Optional[dict], which: str) -> str:
    """The parenthetical after a threshold: what its letter_source label says it is."""
    src = str((thresholds or {}).get("letter_source", "provisional") or "provisional")
    if src.endswith("_calibrated"):
        return CALIBRATED_NOTE[which]
    if src == "provisional":
        return PROVISIONAL_NOTE
    return f"letter_source {src}"


# ------------------------------------------------------------------ the letter, spelled out
def letter_reason(score: Any, advocate: Any, critics: Any, arbitrator: Any,
                  thresholds: Optional[dict], label: str) -> dict:
    """`aggregate_v2.assign_letter(score, advocate, critics, thresholds, arbitrator)` with the
    deciding comparison written out: {letter, sentence, sentence_short, comparison, n_strong,
    blockers, d_rule_roles, nothing_located}. `label` names the score ("S", "S_arb",
    "p_evidence"). `sentence_short` is the same without the threshold-provenance
    parentheticals (for the paragraph). Neither carries a trailing period."""
    s = _num(score)
    out = {"letter": None, "score": s, "label": label, "sentence": "", "sentence_short": "", "n_strong": None,
           "blockers": [], "d_rule_roles": [], "nothing_located": None}
    if thresholds is None:
        out["sentence"] = f"{label} {_f(s)}: thresholds unavailable (no t_A / t_B), letter not derived"
        out["sentence_short"] = out["sentence"]
        return out
    if advocate is None:
        out["sentence"] = f"{label} {_f(s)}: no advocate record, letter not derived"
        out["sentence_short"] = out["sentence"]
        return out
    letter, _ = aggregate_v2.assign_letter(s, advocate, critics, thresholds, arbitrator)
    out["letter"] = letter
    if s is None:
        out["sentence"] = f"{label} is NaN (parse failure): no letter"
        out["sentence_short"] = out["sentence"]
        return out
    t_a, t_b = float(thresholds["t_A"]), float(thresholds["t_B"])
    terms = aggregate_v2.critic_terms(advocate, critics, arbitrator)
    blockers = [role for role in _ordered(terms)
                if terms[role]["included"] and terms[role]["r"] * terms[role]["a"] >= STRONG_R - _EPS]
    crit = advocate.criteria
    strong = [c for c in LETTER_CRITERIA if (_num(getattr(crit, c, None)) or 0.0) >= LETTER_CRITERIA_MIN]
    items = list(advocate.items or [])
    nothing_because = str(advocate.nothing_because or "").strip()
    nothing_located = not items and nothing_because != ""
    d_roles = aggregate_v2.d_rule_roles(terms, arbitrator)
    out.update({"n_strong": len(strong), "blockers": blockers, "d_rule_roles": d_roles,
                "nothing_located": nothing_located})
    def _cmp(which: str, t: float, notes: bool) -> str:
        note = f" ({threshold_note(thresholds, which)})" if notes else ""
        return f"{label} {_f(s)} {'>=' if s >= t else '<'} {which} {_f(t)}{note}"

    def _alt(role):
        return terms[role]["alternative"]

    def _sentence(notes: bool) -> str:
        cmp_a, cmp_b = _cmp("t_A", t_a, notes), _cmp("t_B", t_b, notes)
        if letter == "A":
            return (f"{cmp_a}; {len(strong)} of 3 configuration criteria >= {LETTER_CRITERIA_MIN} "
                    f"({', '.join(strong)}); no critic with r x a >= {_f(STRONG_R, 1)}")
        if letter == "B":
            if s >= t_a:
                blocked = []
                if len(strong) < 2:
                    blocked.append(f"only {len(strong)} of 3 configuration criteria >= {LETTER_CRITERIA_MIN}")
                for role in blockers:
                    blocked.append(f"{role} ({_alt(role)}) r x a = {_f(terms[role]['r'] * terms[role]['a'])} >= {_f(STRONG_R, 1)}")
                return f"{cmp_a} but A is blocked: {'; '.join(blocked)}; {cmp_b}"
            return f"{cmp_b}; {cmp_a}"
        if letter == "D":
            if nothing_located:
                return f"{cmp_b} and nothing located: nothing_because {_q(nothing_because, True)}"
            parts = [f"{role} ({_alt(role)}) covers every item at r {_f(terms[role]['r'])} >= {_f(STRONG_R, 1)}"
                     + (" (upheld)" if arbitrator is not None else "") for role in d_roles]
            return f"{cmp_b} and {'; '.join(parts)}"
        not_d = []                                          # C
        if items:
            not_d.append(f"{len(items)} item(s) located")
        elif not nothing_located:
            not_d.append("no item located but nothing_because is empty")
        who = "no upheld critic" if arbitrator is not None else "no named critic"
        not_d.append(f"{who} covers every item at r >= {_f(STRONG_R, 1)}")
        return f"{cmp_b}; not D: {'; '.join(not_d)}"

    out["comparison"] = _cmp("t_A", t_a, True) if letter == "A" else _cmp("t_B", t_b, True)
    out["sentence"] = _sentence(True)
    out["sentence_short"] = _sentence(False)
    return out


# ------------------------------------------------------------------ the score, term by term
def _terms_from_row(row: Any) -> dict:
    """critic_terms-shaped dict from the stored a_/r_/alt_/no_opinion_/ruling_ columns (the
    un-arbitrated coverage; a partial ruling's arbitrated a' is not stored → None)."""
    terms = {}
    for role in CRITIC_ROLES:
        a, r = _num(_val(row, f"a_{role}")), _num(_val(row, f"r_{role}"))
        alt = _val(row, f"alt_{role}")
        no_op = _val(row, f"no_opinion_{role}")
        ruling = _val(row, f"ruling_{role}")
        if a is None and r is None and alt is None and no_op is None:
            continue                                        # role not called
        named = alt is not None and str(alt).strip() != "" and not bool(no_op)
        terms[role] = {"parsed": a is not None, "named": named, "included": named and a is not None,
                       "r": r, "a": a, "a_geom": a, "ruling": None if ruling is None else str(ruling),
                       "no_opinion": None if no_op is None else bool(no_op),
                       "alternative": None if alt is None else str(alt)}
    return terms


def _arbitrated_from_row(terms: dict) -> dict:
    out = {}
    for role, t in terms.items():
        u = dict(t)
        if t["named"] and t["ruling"] == "overruled":
            u["a"], u["included"] = 0.0, False
        elif t["named"] and t["ruling"] == "partial":
            u["a"] = None                                   # |covers ∩ items| / |items| lives in the records
        out[role] = u
    return out


def _product_terms(p_ev: Optional[float], terms: dict) -> dict:
    """{value, terms:[{role, r, a, alternative, ruling}], complete} for p_ev · Π_included (1 − r·a)."""
    used = []
    complete = p_ev is not None
    for role in _ordered(terms):
        t = terms[role]
        if not t["included"]:
            continue
        used.append({"role": role, "r": t["r"], "a": t["a"], "alternative": t["alternative"],
                     "ruling": t.get("ruling")})
        if _num(t["r"]) is None or _num(t["a"]) is None:
            complete = False
    value = None
    if complete:
        value = min(1.0, max(0.0, p_ev))
        for u in used:
            value *= 1.0 - min(1.0, max(0.0, u["r"] * u["a"]))
    return {"value": value, "terms": used, "complete": complete}


def decomposition(advocate: Any, critics: dict, arbitrator: Any, row: Any, parse_fail=()) -> dict:
    """S and S_arb as products: {p_evidence, source ("records" | "row"), S: {...}, S_arb: {...},
    S_stored, S_arb_stored, S_matches, S_arb_matches}. From the records when the advocate and
    every called critic parsed, else from the stored a_/r_/ruling_ columns. `parse_fail` lists
    the roles the run could not parse: any such role voids both values (the registered
    parse-failure policy — S NaN), whatever the other terms say."""
    critics = dict(critics or {})
    parse_fail = list(parse_fail or [])
    p_ev = _num(advocate.p_evidence) if advocate is not None else _num(_val(row, "p_evidence"))
    if advocate is not None and all(c is not None for c in critics.values()):
        source = "records"
        t0 = aggregate_v2.critic_terms(advocate, critics, None)
        t1 = aggregate_v2.critic_terms(advocate, critics, arbitrator)
    else:
        source = "row"
        t0 = _terms_from_row(row)
        t1 = _arbitrated_from_row(t0)
    dec_s, dec_arb = _product_terms(p_ev, t0), _product_terms(p_ev, t1)
    if parse_fail:
        for d in (dec_s, dec_arb):
            d["value"], d["complete"], d["parse_fail"] = None, False, parse_fail
    s_st, sa_st = _num(_val(row, "S")), _num(_val(row, "S_arb"))

    def _match(calc, stored):
        if calc is None or stored is None:
            return None
        return bool(abs(calc - stored) <= _ATOL)

    return {"p_evidence": p_ev, "source": source, "S": dec_s, "S_arb": dec_arb,
            "S_stored": s_st, "S_arb_stored": sa_st,
            "S_matches": _match(dec_s["value"], s_st), "S_arb_matches": _match(dec_arb["value"], sa_st)}


def _product_line(name: str, dec: dict, p_ev: Optional[float], stored: Optional[float], match: Optional[bool],
                  arbitrated: bool) -> tuple:
    """("S_arb = p_ev x prod(1 - r_i a_i) = 0.86 x (1-0.9x1.0) = 0.086", legend)."""
    prod_label = "prod_{upheld/partial}(1 - r_i a_i')" if arbitrated else "prod(1 - r_i a_i)"
    if p_ev is None:
        return f"{name} = NaN (no p_evidence)", ""
    if dec.get("parse_fail"):
        return f"{name} = NaN (parse failure: {', '.join(dec['parse_fail'])})", ""
    terms = dec["terms"]
    if not terms:
        line = f"{name} = p_ev = {_f(p_ev)} (no critic term entered the product)"
    else:
        factors = " x ".join(f"(1 - {_f(t['r'])}x{_f(t['a'])})" if _num(t["a"]) is not None
                             else f"(1 - {_f(t['r'])}xa')" for t in terms)
        value = _f(dec["value"]) if dec["complete"] else "not recomputable from the row (a' not stored)"
        line = f"{name} = p_ev x {prod_label} = {_f(p_ev)} x {factors} = {value}"
    if stored is not None and match is False:
        line += f" (stored {name} {_f(stored)} differs)"
    elif stored is not None and not dec["complete"]:
        line += f"; stored {name} {_f(stored)}"
    def a_term(t: dict) -> str:
        # a partial ruling's coverage is the ARBITRATED a' = |covers ∩ items| / |items| (the
        # header's a_i'), not the critic's own accounts_for coverage: label it as such
        label = "a'" if arbitrated and t.get("ruling") == "partial" else "a"
        return f"{label} {_f(t['a'])}" if _num(t["a"]) is not None else f"{label} ?"

    legend = "; ".join(f"{t['role']} {t['alternative']} r {_f(t['r'])} {a_term(t)}"
                       + (f" ({t['ruling']})" if t.get("ruling") else "") for t in terms)
    return line, legend


# ------------------------------------------------------------------ per-role facts
def _advocate_facts(adv: Any) -> Optional[dict]:
    if adv is None:
        return None
    return {
        "criteria": {c: int(getattr(adv.criteria, c)) for c in CRITERIA_V2},
        "items": [{"k": it.k, "what": it.what, "panel": it.panel, "r_arcsec": it.r_arcsec,
                   "pa_deg_from": it.pa_deg_from, "pa_deg_to": it.pa_deg_to,
                   "visible_in_direct": bool(it.visible_in_direct),
                   "criteria": [CRITERIA_BY_INDEX.get(int(c), str(c)) for c in it.criteria]} for it in adv.items],
        "nothing_because": adv.nothing_because, "scale_class": adv.scale_class,
        "n_red_neighbours_10as": adv.n_red_neighbours_10as, "bcg_like_halo": bool(adv.bcg_like_halo),
        "deflector_is_centre": bool(adv.deflector_is_centre), "arc_radius_arcsec": adv.arc_radius_arcsec,
        "arc_pa_span_deg": list(adv.arc_pa_span_deg) if adv.arc_pa_span_deg is not None else None,
        "counter_image_pos": adv.counter_image_pos.model_dump() if adv.counter_image_pos is not None else None,
        "centre_of_curvature_offset_arcsec": adv.centre_of_curvature_offset_arcsec,
        "p_evidence": adv.p_evidence, "notes": adv.notes,
    }


def _critic_facts(role: str, c: Any, ruling: Any, terms0: dict, terms1: dict) -> dict:
    t0, t1 = terms0.get(role), terms1.get(role)
    out = {"role": role, "record": c is not None}
    if c is None:
        return out
    out.update({
        "no_opinion": bool(c.no_opinion), "no_opinion_reason": c.no_opinion_reason,
        "alternative": c.alternative, "alternative_desc": c.alternative_desc,
        "location": c.location.model_dump() if c.location is not None else None,
        "accounts_for": [int(k) for k in c.accounts_for], "leaves_standing": [int(k) for k in c.leaves_standing],
        "refutation_strength": c.refutation_strength, "measured": c.measured, "scale_class": c.scale_class,
        "notes": c.notes,
        "a_geom": (t0["a_geom"] if t0 else None), "a_arbitrated": (t1["a"] if t1 else None),
        "included": bool(t1["included"]) if t1 else None,
        "ruling": None if ruling is None else {"ruling": ruling.ruling, "covers": [int(k) for k in ruling.covers],
                                                "why": ruling.why},
    })
    return out


def _arbitrator_facts(arb: Any) -> Optional[dict]:
    if arb is None:
        return None
    return {"surviving_items": [int(k) for k in arb.surviving_items], "letter_llm": arb.letter_llm,
            "scale_class_final": arb.scale_class_final, "needs_human": bool(arb.needs_human),
            "rationale": arb.rationale,
            "rulings": [{"persona": ru.persona, "ruling": ru.ruling, "covers": [int(k) for k in ru.covers],
                         "why": ru.why} for ru in arb.rulings]}


# ------------------------------------------------------------------ renderers
def panel_marker(panel: Any, r_arcsec: Any) -> str:
    """"" or " [r exceeds panel X: its field is Y arcsec]" when the cited composite panel is
    too small for the item's radius on-axis (a 3.5" zoom panel cited at r > 1.75", a 10"
    panel at r > 5") — the record is printed verbatim, the marker says the model's scale
    reading of that panel cannot be right (`annotate.r_exceeds_panel`)."""
    p = str(panel) if panel is not None else ""
    r = _num(r_arcsec)
    if r is None or p not in PANEL_FOV or r <= PANEL_FOV[p] / 2.0:
        return ""
    return f" [r exceeds panel {p}: its field is {_g(PANEL_FOV[p])} arcsec]"


def _item_line(it: dict) -> str:
    crit = ", ".join(it["criteria"]) if it["criteria"] else "none"
    return (f"k{it['k']} — {_q(it['what'], True)} (panel {it['panel']}, r {_g(it['r_arcsec'])} arcsec, "
            f"PA {_g(it['pa_deg_from'])}->{_g(it['pa_deg_to'])}, visible in direct: {_yn(it['visible_in_direct'])}; "
            f"criteria: {crit})" + panel_marker(it["panel"], it["r_arcsec"]))


def _location_str(loc: Optional[dict]) -> str:
    if loc is None:
        return "none"
    return (f"r {_g(loc['r_arcsec_from'])}-{_g(loc['r_arcsec_to'])} arcsec, "
            f"PA {_g(loc['pa_deg_from'])}->{_g(loc['pa_deg_to'])}")


def _critic_head(cf: dict, status: str) -> str:
    """The one-line status of a critic: parse state | no opinion | alternative | nothing named."""
    if not cf["record"]:
        return status
    if cf["no_opinion"]:
        return f"no opinion ({cf['no_opinion_reason'] or 'no reason given'})"
    if cf["alternative"] is None:
        return "no alternative named (nothing in its competence fits)"
    if str(cf["alternative_desc"] or "").strip() == "":
        return f"{cf['alternative']} (no description)"
    return f"{cf['alternative']}: {_q(cf['alternative_desc'], True)}"


def _ruling_str(cf: dict, has_arbitrator: bool, collapse: bool) -> str:
    ru = cf.get("ruling")
    if ru is None:
        return "no arbitrator" if not has_arbitrator else "no ruling for this critic"
    s = ru["ruling"]
    if ru["covers"]:
        s += f", covers {_ks(ru['covers'])}"
    if ru["why"]:
        s += f" — {_q(ru['why'], collapse)}"
    return s


def render_markdown(facts: dict) -> str:
    L = []
    name, letter = facts["name"], facts["letter"]
    L.append(f"# {name}")
    L.append("")
    head = (f"**Letter {letter or 'none'}** ({facts['letter_basis']}) · p_evidence {_f(facts['p_evidence'])} · "
            f"S_arb {_f(facts['S_arb'])} · S {_f(facts['S'])} · scale {facts['scale_class_final'] or facts['scale_class'] or 'n/a'}")
    L.append(head)
    L.append("")
    L.append(f"Letter: {facts['letter_reason']}.")
    if facts["parse_fail_roles"]:
        L.append(f"Parse failure (run): {', '.join(facts['parse_fail_roles'])}.")
    if facts["records_missing"]:
        L.append(f"Record not rebuilt from the votes (raw missing; the stored row is parse_ok): "
                 f"{', '.join(facts['records_missing'])}.")
    # ---- advocate
    L += ["", "## Advocate", ""]
    adv = facts["advocate"]
    if adv is None:
        L.append(f"- no advocate record ({facts['role_status']['advocate']})")
    else:
        L.append("- criteria: " + ", ".join(f"{c} {adv['criteria'][c]}" for c in CRITERIA_V2))
        if adv["items"]:
            L.append(f"- items ({len(adv['items'])}):")
            for it in adv["items"]:
                L.append(f"  - {_item_line(it)}")
        else:
            L.append("- items: none located")
            L.append(f"- nothing_because: {_q(adv['nothing_because'])}")
        flags = [f"scale {adv['scale_class']}", f"red neighbours within 10 arcsec {adv['n_red_neighbours_10as']}",
                 f"BCG-like halo {_yn(adv['bcg_like_halo'])}", f"deflector is centre {_yn(adv['deflector_is_centre'])}"]
        if adv["arc_radius_arcsec"] is not None:
            flags.append(f"arc radius {_g(adv['arc_radius_arcsec'])} arcsec")
        if adv["arc_pa_span_deg"] is not None:
            flags.append(f"arc PA span {_g(adv['arc_pa_span_deg'][0])}->{_g(adv['arc_pa_span_deg'][1])}")
        if adv["counter_image_pos"] is not None:
            cp = adv["counter_image_pos"]
            flags.append(f"counter-image at r {_g(cp['r_arcsec'])} arcsec PA {_g(cp['pa_deg'])}")
        if adv["centre_of_curvature_offset_arcsec"] is not None:
            flags.append(f"centre-of-curvature offset {_g(adv['centre_of_curvature_offset_arcsec'])} arcsec")
        L.append("- " + "; ".join(flags))
        L.append(f"- p_evidence: {_f(adv['p_evidence'])}")
        if adv["notes"]:
            L.append(f"- notes: {_q(adv['notes'])}")
    # ---- critics
    L += ["", "## Critics", ""]
    if not facts["critics"]:
        L.append(f"- not called ({facts['critics_not_called']})")
    has_arb = facts["arbitrator"] is not None
    for role in _ordered(facts["critics"]):
        cf = facts["critics"][role]
        L.append(f"### {role} — {_critic_head(cf, facts['role_status'][role])}")
        if not cf["record"]:
            continue
        if cf["alternative"] is not None:
            L.append(f"- location: {_location_str(cf['location'])}")
            cov = f"coverage a {_f(cf['a_geom'])} (geometric)" if cf["a_geom"] is not None else "coverage a n/a"
            ruling_kind = (cf["ruling"] or {}).get("ruling")
            if ruling_kind == "overruled":
                cov += ", excluded from S_arb (overruled)"
            elif cf["a_arbitrated"] is not None and cf["a_geom"] is not None and abs(cf["a_arbitrated"] - cf["a_geom"]) > _EPS:
                cov += f", arbitrated a' {_f(cf['a_arbitrated'])}"
            L.append(f"- accounts for: {_ks(cf['accounts_for'])}; leaves standing: {_ks(cf['leaves_standing'])}; "
                     f"strength r {_f(cf['refutation_strength'])}; {cov}")
        elif cf["leaves_standing"]:
            L.append(f"- leaves standing: {_ks(cf['leaves_standing'])}")
        if cf["scale_class"]:
            L.append(f"- scale_class: {cf['scale_class']}")
        if cf["measured"]:
            L.append(f"- measured: `{json.dumps(_py(cf['measured']), sort_keys=True)}`")
        if cf["notes"]:
            L.append(f"- notes: {_q(cf['notes'])}")
        L.append(f"- ruling: {_ruling_str(cf, has_arb, False)}")
    # ---- arbitrator
    L += ["", "## Arbitrator", ""]
    arb = facts["arbitrator"]
    if arb is None:
        L.append(f"- no arbitrator record ({facts['arbitrator_status']})")
    else:
        n_items = len(adv["items"]) if adv is not None else None
        of = f" ({len(arb['surviving_items'])} of {n_items})" if n_items is not None else ""
        L.append(f"- surviving items: {_ks(arb['surviving_items'])}{of}")
        L.append(f"- letter_llm: {arb['letter_llm']}; scale_class_final: {arb['scale_class_final'] or 'n/a'}; "
                 f"needs_human: {_yn(arb['needs_human'])}")
        L.append(f"- rationale: {_q(arb['rationale'])}")
    # ---- score
    L += ["", "## Score", ""]
    dec = facts["score"]
    line, legend = _product_line("S", dec["S"], dec["p_evidence"], dec["S_stored"], dec["S_matches"], False)
    L.append(f"- {line}")
    if legend:
        L.append(f"  - terms: {legend}")
    line, legend = _product_line("S_arb", dec["S_arb"], dec["p_evidence"], dec["S_arb_stored"], dec["S_arb_matches"], True)
    L.append(f"- {line}")
    if legend:
        L.append(f"  - terms: {legend}")
    st = facts["stored"]
    L.append(f"- stored grade_pred (on S): {st['grade_pred'] or 'none'} — {facts['stored_letter_reason']}")
    L.append(f"- stored letter_arb (on S_arb): {st['letter_arb'] or 'none'}; letter_source: {st['letter_source'] or 'n/a'}")
    dep = facts["deploy"]
    if dep is not None:
        L.append(f"- deploy rule {dep['rule']}: letter_rank (advocate only, on p_evidence) = {dep['letter_rank'] or 'none'} — "
                 f"{dep['letter_rank_reason']}")
        L.append(f"- deploy rule {dep['rule']}: letter_final = {dep['letter_final'] or 'none'} — {dep['letter_final_reason']}")
        L.append(f"- veto: {dep['veto'] or 'none (letter_final == letter_rank)'}")
    return "\n".join(L) + "\n"


def _excerpt(text: Any, n: int) -> str:
    """The first `n` characters of a prose field (whitespace folded), ellipsis-marked when cut."""
    s = " ".join(str(text or "").split())
    return s if len(s) <= n else s[:max(1, n - 1)].rstrip() + "…"


def render_paragraph(facts: dict) -> tuple:
    """(paragraph, truncated): the same facts as plain text, at most PARAGRAPH_MAX characters.
    The head sentence always leads; the others are added in priority order (letter, items,
    critics, arbitrator), each only if it still fits — a skipped sentence sets `truncated`.
    Item `what` quotes are ellipsis-marked excerpts (PARAGRAPH_WHAT chars, at most
    PARAGRAPH_ITEMS items listed); the arbitrator's rationale then fills up to
    PARAGRAPH_RATIONALE_MAX chars as an ellipsis-marked excerpt (its cut is reported in
    facts["rationale_excerpted"], not in `truncated`); the five criteria scores come last."""
    name, letter, dep = facts["name"], facts["letter"], facts["deploy"]
    basis = facts["letter_basis"]
    if dep is not None and letter is not None:
        basis += f"; letter_rank {dep['letter_rank'] or 'none'}, veto {dep['veto'] or 'none'}"
    parts = [f"{name}: letter {letter or 'none'} ({basis}); p_evidence {_f(facts['p_evidence'])}, "
             f"S_arb {_f(facts['S_arb'])}; scale {facts['scale_class_final'] or facts['scale_class'] or 'n/a'}.",
             f"Letter: {facts['letter_reason_short']}."]
    if facts["parse_fail_roles"]:
        parts.append(f"Parse failure: {', '.join(facts['parse_fail_roles'])}.")
    adv = facts["advocate"]
    if adv is None:
        parts.append(f"No advocate record ({facts['role_status']['advocate']}).")
    elif adv["items"]:
        shown = adv["items"][:PARAGRAPH_ITEMS]
        its = "; ".join(f"k{it['k']} {_q(_excerpt(it['what'], PARAGRAPH_WHAT))} (panel {it['panel']}, "
                        f"r {_g(it['r_arcsec'])} arcsec)" for it in shown)
        more = f"; +{len(adv['items']) - len(shown)} more" if len(adv["items"]) > len(shown) else ""
        parts.append(f"Advocate located {len(adv['items'])} item(s): {its}{more}.")
    else:
        parts.append(f"Advocate located nothing: {_q(adv['nothing_because'], True)}.")
    if not facts["critics"]:
        parts.append(f"Critics not called ({facts['critics_not_called']}).")
    else:
        has_arb = facts["arbitrator"] is not None
        bits = []
        for role in _ordered(facts["critics"]):
            cf = facts["critics"][role]
            if not cf["record"]:
                bits.append(f"{role} {facts['role_status'][role]}")
            elif cf["no_opinion"]:
                bits.append(f"{role} no opinion ({cf['no_opinion_reason'] or 'no reason'})")
            elif cf["alternative"] is None:
                bits.append(f"{role} no alternative named")
            else:
                s = f"{role} {cf['alternative']} r {_f(cf['refutation_strength'])} covering {_ks(cf['accounts_for'])}"
                if cf["ruling"] is not None:
                    s += f" ({cf['ruling']['ruling']})"
                elif has_arb:
                    s += " (no ruling)"
                bits.append(s)
        parts.append("Critics: " + "; ".join(bits) + ".")
    arb = facts["arbitrator"]
    if arb is not None:
        parts.append(f"Arbitrator: surviving {_ks(arb['surviving_items'])}; letter_llm {arb['letter_llm']}; "
                     f"needs_human {_yn(arb['needs_human'])}.")
    # the head always leads (hard-capped if it alone overflows); the rest is greedy in order
    out, truncated = parts[0], False
    if len(out) > PARAGRAPH_MAX:
        out, truncated = _excerpt(out, PARAGRAPH_MAX), True
    for p in parts[1:]:
        cand = f"{out} {p}"
        if len(cand) <= PARAGRAPH_MAX:
            out = cand
        else:
            truncated = True
    excerpted = False
    if arb is not None and arb["rationale"]:
        prefix = " Arbitrator rationale: "
        room = min(PARAGRAPH_RATIONALE_MAX, PARAGRAPH_MAX - len(out) - len(prefix) - 2)
        if room >= PARAGRAPH_RATIONALE_MIN:
            text = _excerpt(arb["rationale"], room)
            excerpted = text != " ".join(str(arb["rationale"]).split())
            out += prefix + _q(text)
        else:
            excerpted = True
    if adv is not None:                                     # the five scores: last-priority filler
        crit = " Criteria " + ", ".join(f"{c} {adv['criteria'][c]}" for c in CRITERIA_V2) + "."
        if len(out) + len(crit) <= PARAGRAPH_MAX:
            out += crit
        else:
            truncated = True
    facts["rationale_excerpted"] = excerpted
    return out, truncated


# ------------------------------------------------------------------ one item
def _split_roles(s: Any) -> list:
    return [r for r in str(s or "").split("+") if r]


def explain_item(name: str, records: Mapping, row: Any, thresholds: Optional[dict] = None,
                 deploy: Optional[dict] = None) -> dict:
    """Render one item: {markdown, paragraph, facts}. `records` = {role: record | None} for the
    roles the run CALLED (absent = not called; None = no record); `row` its preds row (the
    `schemas_panel.to_row` columns + tau0 / t_A / t_B / letter_source; may be {}); `thresholds`
    a resolved {tau0, t_A, t_B, letter_source} (default: the row's); `deploy` an
    `aggregate_v2.deploy_letters` dict — with it the headline letter is letter_final, without it
    the stored grade_pred."""
    records = dict(records or {})
    row = {} if row is None else row
    thr = item_thresholds(row, thresholds)
    adv = records.get("advocate")
    critics = {r: records[r] for r in CRITIC_ROLES if r in records}
    arb = records.get("arbitrator")
    parse_fail = _split_roles(_val(row, "parse_fail_roles"))
    # a None record is the run's parse failure when the row says so, else a raw the votes lost
    status, missing = {}, []
    for role in ROLES:
        if role not in records:
            status[role] = "not called"
        elif records[role] is not None:
            status[role] = "record"
        elif role in parse_fail:
            status[role] = "parse failure"
        else:
            status[role] = "record not rebuilt (raw missing from votes)"
            missing.append(role)
    known_critics = {r: c for r, c in critics.items() if c is not None or r in parse_fail}
    terms0 = aggregate_v2.critic_terms(adv, known_critics, None) if adv is not None else {}
    terms1 = aggregate_v2.critic_terms(adv, known_critics, arb) if adv is not None else {}
    rulings = {ru.persona: ru for ru in (arb.rulings if arb is not None else [])}

    p_ev = _num(adv.p_evidence) if adv is not None else _num(_val(row, "p_evidence"))
    s_row, sa_row = _num(_val(row, "S")), _num(_val(row, "S_arb"))
    dec = decomposition(adv, critics, arb, row, parse_fail)
    S = dec["S"]["value"] if dec["S"]["value"] is not None else s_row
    S_arb = dec["S_arb"]["value"] if dec["S_arb"]["value"] is not None else sa_row
    stored = {"grade_pred": _val(row, "grade_pred"), "letter_arb": _val(row, "letter_arb"),
              "letter_llm": _val(row, "letter_llm"), "letter_source": _val(row, "letter_source"),
              "S": s_row, "S_arb": sa_row, "p_evidence": _num(_val(row, "p_evidence"))}
    stored_reason = letter_reason(S, adv, known_critics, None, thr, "S")
    suffix = ""
    if stored["grade_pred"] is not None and stored_reason["letter"] not in (None, stored["grade_pred"]):
        suffix += f" [recomputed {stored_reason['letter']} differs from stored {stored['grade_pred']}]"
    if missing:
        suffix += f" [from the stored row; records not rebuilt: {', '.join(missing)}]"
    stored_reason["sentence"] += suffix
    stored_reason["sentence_short"] += suffix

    dep_facts = None
    if deploy is not None:
        rank = letter_reason(deploy.get("p_evidence"), adv, {}, None, thr, "p_evidence")
        if deploy.get("rule") == "R2":
            d_roles = aggregate_v2.d_rule_roles(terms1, arb) if adv is not None else []
            if deploy.get("letter_final") == "D" and d_roles:
                fin_sentence = "D rule: " + "; ".join(
                    f"{role} ({terms1[role]['alternative']}) covers every item (a = 1) at r {_f(terms1[role]['r'])} >= {_f(STRONG_R, 1)}"
                    + (" (upheld)" if arb is not None else "") for role in d_roles)
            elif deploy.get("letter_final") is None:
                fin_sentence = "S_arb is NaN (parse failure): no letter"
            else:
                fin_sentence = f"letter_rank stands (D rule not met); {rank['sentence']}"
            fin = {"letter": deploy.get("letter_final"), "sentence": fin_sentence, "sentence_short": fin_sentence}
        else:
            fin = letter_reason(deploy.get("S_arb"), adv, known_critics, arb, thr, "S_arb")
        dep_facts = {"rule": deploy.get("rule"), "letter_rank": deploy.get("letter_rank"),
                     "letter_final": deploy.get("letter_final"), "veto": deploy.get("veto") or "",
                     "S": _num(deploy.get("S")), "S_arb": _num(deploy.get("S_arb")),
                     "p_evidence": _num(deploy.get("p_evidence")),
                     "letter_rank_reason": rank["sentence"], "letter_final_reason": fin["sentence"]}
        letter, basis = dep_facts["letter_final"], f"deployed, rule {dep_facts['rule']}"
        reason, reason_short = fin["sentence"], fin["sentence_short"]
    else:
        letter, basis = stored["grade_pred"], "stored grade_pred, on S"
        reason, reason_short = stored_reason["sentence"], stored_reason["sentence_short"]
    if letter is None:
        basis = "no letter: " + (", ".join(f"{r} {status[r]}" for r in ROLES if status[r] not in ("record", "not called"))
                                 or "score NaN")

    scale = adv.scale_class if adv is not None else _val(row, "scale_class")
    scale_final = _val(row, "scale_class_final")
    if arb is not None and arb.scale_class_final:
        scale_final = arb.scale_class_final
    tau0 = thr["tau0"] if thr else _num(_val(row, "tau0"))
    if critics:
        not_called = ""
    elif p_ev is not None and tau0 is not None and p_ev < tau0:
        not_called = f"p_evidence {_f(p_ev)} < tau0 {_f(tau0)}"
    else:
        not_called = "no critic role in the records"
    arb_status = status["arbitrator"]
    if arb_status == "not called" and critics and not any(t["named"] for t in terms0.values()):
        arb_status = "not called (no critic named an alternative)"

    facts = {
        "name": str(name), "letter": letter, "letter_basis": basis, "letter_reason": reason,
        "letter_reason_short": reason_short,
        "p_evidence": p_ev, "S": S, "S_arb": S_arb, "scale_class": scale, "scale_class_final": scale_final,
        "thresholds": thr, "roles_called": [r for r in ROLES if r in records],
        "parse_fail_roles": parse_fail, "records_missing": missing, "role_status": status,
        "critics_not_called": not_called, "arbitrator_status": arb_status,
        "advocate": _advocate_facts(adv),
        "critics": {r: _critic_facts(r, c, rulings.get(r), terms0, terms1) for r, c in critics.items()},
        "arbitrator": _arbitrator_facts(arb),
        "score": dec, "stored": stored, "stored_letter_reason": stored_reason["sentence"],
        "deploy": dep_facts,
        "run": {c: _val(row, c) for c in ("layout", "arm", "model", "thinking", "effort", "k", "thresholds_sha16")
                if _val(row, c) is not None},
    }
    paragraph, truncated = render_paragraph(facts)      # also sets facts["rationale_excerpted"]
    facts["paragraph_truncated"] = truncated
    facts = _py(facts)
    return {"markdown": render_markdown(facts), "paragraph": paragraph, "facts": facts}


# ------------------------------------------------------------------ a run
def deploy_for(records: Mapping, thresholds: Optional[dict], rule: str) -> Optional[dict]:
    """`aggregate_v2.deploy_letters` on one item's records; None without thresholds."""
    if thresholds is None:
        return None
    return R.deploy_from_roles(dict(records or {}), thresholds, rule)   # voids a called-but-failed arbitrator


def explain_run(preds: pd.DataFrame, records: Mapping, thresholds: Optional[dict] = None,
                rule: Optional[str] = None) -> list:
    """One `explain_item` result per preds row, in preds order. `records` = {name: {role:
    record | None}} (`records.load_run`); items without votes get {}. With `rule` the
    deployment letters are computed per item (thresholds: the given dict or the row's)."""
    if rule is not None and rule not in DEPLOY_RULES:
        raise ValueError(f"rule must be one of {DEPLOY_RULES}, got {rule!r}")
    out = []
    for _, row in preds.iterrows():
        name = str(row["name"])
        recs = records.get(name, {})
        thr = item_thresholds(row, thresholds)
        dep = deploy_for(recs, thr, rule) if rule is not None else None
        out.append(explain_item(name, recs, row, thr, dep))
    return out


def summary_counts(results: list) -> dict:
    """Count table for the CLI (no ids, no prose)."""
    letters: dict = {}
    n_fail = n_adv_only = n_arb = n_veto = n_trunc = n_exc = n_missing = 0
    for res in results:
        f = res["facts"]
        letters[f["letter"] or "none"] = letters.get(f["letter"] or "none", 0) + 1
        n_fail += bool(f["parse_fail_roles"])
        n_adv_only += not f["critics"]
        n_arb += f["arbitrator"] is not None
        n_veto += bool(f["deploy"] and f["deploy"]["veto"])
        n_trunc += bool(f["paragraph_truncated"])
        n_exc += bool(f.get("rationale_excerpted"))
        n_missing += bool(f["records_missing"])
    return {"n_items": len(results), "letters": dict(sorted(letters.items())), "n_parse_fail": n_fail,
            "n_advocate_only": n_adv_only, "n_with_arbitrator": n_arb, "n_veto": n_veto,
            "n_paragraph_truncated": n_trunc, "n_rationale_excerpted": n_exc, "n_records_missing": n_missing}


def _unique_stem(name: str, taken: set) -> str:
    stem = _util.safe_name(name)
    cand, i = stem, 1
    while cand in taken:
        i += 1
        cand = f"{stem}_{i}"
    taken.add(cand)
    return cand


def write_outputs(results: list, out_dir, formats=FORMATS) -> dict:
    """Write the requested formats into `out_dir` (created): md → <safe_name>.md per item +
    index.md; csv → explain.csv (name, letter, paragraph; pinned with a .sha); json →
    facts.json ({name: facts}) + .sha. Returns {format: path}."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    formats = tuple(formats)
    bad = [f for f in formats if f not in FORMATS]
    if bad:
        raise ValueError(f"unknown format(s) {bad}; choose from {FORMATS}")
    written: dict = {}
    if "md" in formats:
        taken: set = set()
        idx = ["# Explanations", "", f"{len(results)} item(s).", "",
               "| name | letter | p_evidence | S_arb | veto | file |", "|---|---|---|---|---|---|"]
        for res in results:
            f = res["facts"]
            stem = _unique_stem(f["name"], taken)
            (out_dir / f"{stem}.md").write_text(res["markdown"])
            veto = (f["deploy"] or {}).get("veto") or ""
            idx.append(f"| {f['name']} | {f['letter'] or 'none'} | {_f(f['p_evidence'])} | {_f(f['S_arb'])} | "
                       f"{veto} | [{stem}.md]({stem}.md) |")
        (out_dir / INDEX_NAME).write_text("\n".join(idx) + "\n")
        written["md"] = out_dir / INDEX_NAME
    if "csv" in formats:
        df = pd.DataFrame([{"name": r["facts"]["name"], "letter": r["facts"]["letter"], "paragraph": r["paragraph"]}
                           for r in results], columns=list(CSV_COLS))
        _util.pin(df, out_dir / CSV_NAME)
        written["csv"] = out_dir / CSV_NAME
    if "json" in formats:
        text = json.dumps({r["facts"]["name"]: r["facts"] for r in results}, indent=1, sort_keys=True)
        (out_dir / FACTS_NAME).write_text(text)
        (out_dir / (FACTS_NAME + ".sha")).write_text(_util.sha_text(text) + "\n")
        written["json"] = out_dir / FACTS_NAME
    return written


# ------------------------------------------------------------------ CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--preds", required=True, help="preds_*.parquet of a finished run")
    ap.add_argument("--votes", default=None, help="its votes parquet (default: the _votes.parquet sibling)")
    ap.add_argument("--thresholds", default=None, help="thresholds_v2.json (default: each row's stored tau0/t_A/t_B)")
    ap.add_argument("--model-key", default=None, help="key into --thresholds (e.g. sonnet_api, opus5_api)")
    ap.add_argument("--rule", choices=list(DEPLOY_RULES), default=None,
                    help="also compute the v2-deploy letters (letter_rank / letter_final / veto)")
    ap.add_argument("--trace-dir", default=None, help="per-role traces: recovers a votes row whose raw is missing")
    ap.add_argument("--out-dir", required=True, help="NEW directory for the rendered explanations")
    ap.add_argument("--format", default="md,csv,json", help="comma-separated subset of md,csv,json")
    args = ap.parse_args(argv)

    formats = tuple(f.strip() for f in args.format.split(",") if f.strip())
    thresholds = None
    if args.thresholds:
        if not args.model_key:
            ap.error("--thresholds needs --model-key")
        thresholds = aggregate_v2.resolve_thresholds(aggregate_v2.load_thresholds(args.thresholds), args.model_key)
    preds, records = R.load_run(Path(args.preds), args.votes, trace_dir=args.trace_dir)
    results = explain_run(preds, records, thresholds, args.rule)
    written = write_outputs(results, args.out_dir, formats)
    counts = summary_counts(results)
    src = f"{thresholds['letter_source']}" if thresholds else "per-row"
    print(f"[explain] {Path(args.preds).name}: {counts['n_items']} items, thresholds {src}, rule {args.rule or 'none'}")
    for k, v in counts.items():
        if k != "n_items":
            print(f"[explain] {k}: {v}")
    for fmt, path in written.items():
        print(f"[explain] wrote {fmt}: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
