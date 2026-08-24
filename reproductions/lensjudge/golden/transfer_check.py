#!/usr/bin/env python3
"""golden/transfer_check.py — REGISTRY "Deployment rule v2-deploy" items 6–7: the derived
holdout transfer endpoints of the three letter rules and the pre-stated R1/R2 selection.

Zero API, no new scoring. Every letter here is a deterministic function of the STORED per-role
records (rebuilt from the runs' votes parquets by `golden/records.py`, the run-time parse path)
plus ONE thresholds file, exactly as `aggregate_v2.deploy_letters` states items 3–5:

    letter_rank  a2 holdout parquet (advocate only; S == p_evidence there). The A-criteria
                 guard (≥ 2 of curvature / counter_image / arc_morphology ≥ 6) applies from the
                 rebuilt AdvocateRecord. Only when the a2 votes parquet is absent do the stored
                 `crit_*` / `p_evidence` / `n_items` columns stand in (`advocate_from_row`;
                 exact for A/B membership — every endpoint — and the C/D split copied from the
                 stored letter where it is threshold-independent); the output SAYS so.
    R1 (primary) a1 holdout parquet + votes: assign_letter(S_arb, advocate, critics, thresholds,
                 arbitrator) — the arbitrated evidence must clear the same bars.
    R2 (fallback) letter_rank demoted to D only by the D rule (an upheld critic with a_geom = 1
                 at r ≥ 0.8).

Rows whose stored S is NaN (a1: 49 advocate transport + 2 artifact parse failures until the
registered top-up lands) are EXCLUDED and counted (`n_excluded_nan`); a row that would come
out unlettered despite a finite S is counted separately (`n_unlettered`, expected 0).
Anchors (is_anchor) leave the negative and positive pools as in `analyze_truth` (the holdout
carries none).

Endpoints per rule, each with a 95 % Clopper–Pearson CI: FPR at A and at A∪B on holdout
`truth_class == negative` (P2 holds iff the A upper CI ≤ 2.5 % AND the A∪B upper CI ≤ 7.5 %);
recall at A and at A∪B on `is_positive`; the stress_D count (and rate) at A∪B; the letter
distribution per truth_class. Beside them the rebuild parity of each parquet under its own
stored thresholds (`records.compare_rebuild` on S / S_arb / grade_pred / letter_arb — the
mismatch counts must be 0 for the derived letters to mean what the stored ones mean).

Selection (item 6, stated before this check runs): R1 is deployed unless recall_AB(R1) <
0.5 · recall_AB(letter_rank), then R2. Point estimates, no CI.

Thresholds: `aggregate_v2.resolve_thresholds(table, --model-key)`; when the key is null /
absent the provisional numbers are used and EVERY output is labelled PROVISIONAL (stdout, the
markdown header, a csv row, `selected_rule.json: provisional`). `--rule-select` marks the
registered selection run and REFUSES provisional thresholds (item 1: the fitted key is written
before any holdout letter is computed).

Outputs (a NEW --out-dir; existing files refuse without --overwrite):
    transfer_check.md            the tables
    transfer_check.csv (+ .sha)  statistic, rule, value, ci_lo, ci_hi, n
    selected_rule.json           {rule, reason, numbers, thresholds_sha16, provisional}

    cd reproductions && ~/.venvs/lensjudge/bin/python lensjudge/golden/transfer_check.py \\
        --thresholds lensjudge/golden/thresholds_v2.json --model-key opus5_api \\
        --a2 lensjudge/outputs/preds_truth_a2_opus5_holdout_k1_r1.parquet \\
        --a1 lensjudge/outputs/preds_truth_a1_opus5_holdout_k1_r1.parquet \\
        --out-dir lensjudge/outputs/transfer_opus5 [--rule-select] [--overwrite]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from pydantic import BaseModel, ConfigDict  # noqa: E402
from scipy.stats import beta  # noqa: E402

from lensjudge.golden import _util, aggregate_v2  # noqa: E402
from lensjudge.golden import records as R  # noqa: E402
from lensjudge.golden import run_truth_eval as rte  # noqa: E402  (thresholds_sha only)
from lensjudge.golden import schemas_panel as sp  # noqa: E402

OUT = _util.LENSJUDGE / "outputs"
THRESHOLDS_DEFAULT = _util.HERE / "thresholds_v2.json"
A2_DEFAULT = OUT / "preds_truth_a2_opus5_holdout_k1_r1.parquet"
A1_DEFAULT = OUT / "preds_truth_a1_opus5_holdout_k1_r1.parquet"
MODEL_KEY_DEFAULT = "opus5_api"

RULES = ("letter_rank", "R1", "R2")              # letter_rank from a2; R1 / R2 from a1
DEPLOY_RULES = aggregate_v2.DEPLOY_RULES        # ("R1", "R2")
NEG_CLASS = "negative"
STRESS_D = "stress_D"
LENS_LETTERS = ("A", "B")
LETTER_COLS = ("A", "B", "C", "D", "None")
P2_UPPER = {"A": 0.025, "AB": 0.075}            # item 7: t_A upper CI ≤ 2.5 %, t_B upper CI ≤ 7.5 %
SELECT_FRACTION = 0.5                           # item 6: R2 iff recall_AB(R1) < 0.5 · recall_AB(letter_rank)
CSV_COLS = ("statistic", "rule", "value", "ci_lo", "ci_hi", "n")
LETTER_TABLE_COLS = ("name", "truth_class", "is_positive", "is_anchor", "S_stored", "p_evidence",
                     "letter_rank", "letter", "veto", "S_arb", "excluded", "exclude_reason")
REBUILD_COLS = ("S", "S_arb", "grade_pred", "letter_arb", "p_evidence")
SOURCE_VOTES = "votes"
SOURCE_CRIT_COLUMNS = "crit_columns_fallback"
MD_NAME, CSV_NAME, JSON_NAME = "transfer_check.md", "transfer_check.csv", "selected_rule.json"


# ------------------------------------------------------------------ small statistics
def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact binomial CI (the `analyze_truth.clopper_pearson` formula); (nan, nan) when n == 0."""
    if n <= 0:
        return float("nan"), float("nan")
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lo, hi


def _rate(k: int, n: int) -> float:
    return k / n if n else float("nan")


def _letter_str(x) -> str:
    """A letter cell as text: "None" for a missing letter (pandas stores None as NaN)."""
    return "None" if R.is_missing(x) else str(x)


# ------------------------------------------------------------------ thresholds
def load_thresholds(path, model_key: str) -> dict:
    """The RESOLVED thresholds for `model_key` (`aggregate_v2.resolve_thresholds`; a null /
    absent key ⇒ the provisional numbers) plus `model_key`, `thresholds_sha16` (the tuple sha
    `run_truth_eval.thresholds_sha` stamps on a run), `provisional` (the model's own key was
    NOT used) and `table_sha16` (of the file's text)."""
    p = Path(path)
    text = p.read_text()
    table = json.loads(text)
    thr = dict(aggregate_v2.resolve_thresholds(table, model_key))
    thr["model_key"] = model_key
    thr["thresholds_sha16"] = rte.thresholds_sha(thr)
    thr["provisional"] = str(thr.get("thresholds_key")) != str(model_key)
    thr["table_sha16"] = _util.sha_text(text)
    return thr


def _bare(thr: dict) -> dict:
    """The keys `deploy_letters` / `assign_letter` read (tau0, t_A, t_B, letter_source)."""
    return {k: thr[k] for k in ("tau0", "t_A", "t_B", "letter_source") if k in thr}


# ------------------------------------------------------------------ letters per row
def _get(row, col, default=None):
    if hasattr(row, "get"):
        v = row.get(col, default)
    else:
        v = getattr(row, col, default)
    return default if R.is_missing(v) else v


def advocate_from_row(row) -> Optional[dict]:
    """FALLBACK ONLY (a2 votes parquet absent): an advocate-shaped dict from the stored row —
    criteria from `crit_*`, `p_evidence`, `n_items` placeholder items. `nothing_because` is
    recovered from the stored letter where the D rule is threshold-independent (n_items == 0
    and stored grade_pred == D); otherwise "". A/B membership (every endpoint) is exact; the
    C/D split is exact whenever the stored letter was C or D. None when p_evidence is
    missing (the row is then unlettered)."""
    p_ev = _get(row, "p_evidence")
    if p_ev is None:
        return None
    crit = {c: _get(row, f"crit_{c}") for c in sp.CRITERIA_V2}
    n_items = _get(row, "n_items")
    n_items = 0 if n_items is None else int(n_items)
    nothing = "stored grade_pred D" if (n_items == 0 and str(_get(row, "grade_pred", "")) == "D") else ""
    return {"p_evidence": float(p_ev), "criteria": crit, "items": list(range(1, n_items + 1)),
            "nothing_because": nothing}


def _deploy(roles: Optional[dict], thresholds: dict, rule: str) -> dict:
    """`records.deploy_from_roles` on one item's {role: record | None} (a called-but-failed
    arbitrator voids letter_final, as at run time)."""
    return R.deploy_from_roles(roles, thresholds, rule)


def letter_table(preds: pd.DataFrame, records: Optional[dict], thresholds: dict, rule: str) -> pd.DataFrame:
    """One row per preds row (LETTER_TABLE_COLS): the rule's letter from the rebuilt records
    (`records` = {name: {role: record | None}}, `records.load_run`'s shape) under
    `thresholds`. rule "letter_rank" ⇒ `letter` is the advocate-only letter (critics
    ignored); "R1" / "R2" ⇒ `letter_final` of that rule. `records=None` is the crit-column
    fallback (`advocate_from_row`; letter_rank only). A row is `excluded` when its stored S
    is NaN (exclude_reason "nan_S") or its letter is None despite a finite S
    ("unlettered")."""
    if rule not in RULES:
        raise ValueError(f"rule must be one of {RULES}, got {rule!r}")
    if records is None and rule != "letter_rank":
        raise ValueError(f"rule {rule} needs the rebuilt records (critics); the crit-column fallback "
                         f"serves letter_rank only")
    R.check_cols(preds, ("name", "S", "truth_class", "is_positive"), "preds")
    thr = _bare(thresholds)
    rows = []
    for _, r in preds.iterrows():
        name = str(r["name"])
        if records is None:
            adv = advocate_from_row(r)
            d = aggregate_v2.deploy_letters(adv, {}, None, thr, "R1")
        else:
            d = _deploy(records.get(name), thr, rule if rule in DEPLOY_RULES else "R1")
        letter = d["letter_rank"] if rule == "letter_rank" else d["letter_final"]
        s_stored = _get(r, "S")
        s_stored = float("nan") if s_stored is None else float(s_stored)
        if math.isnan(s_stored):
            excluded, why = True, "nan_S"
        elif R.is_missing(letter):
            excluded, why = True, "unlettered"
        else:
            excluded, why = False, ""
        rows.append({
            "name": name, "truth_class": str(r["truth_class"]),
            "is_positive": bool(r["is_positive"]), "is_anchor": bool(_get(r, "is_anchor", False) or False),
            "S_stored": s_stored, "p_evidence": d["p_evidence"], "letter_rank": d["letter_rank"],
            "letter": letter, "veto": d["veto"] if rule != "letter_rank" else "",
            "S_arb": d["S_arb"], "excluded": excluded, "exclude_reason": why})
    return pd.DataFrame(rows, columns=list(LETTER_TABLE_COLS))


# ------------------------------------------------------------------ endpoints
def _row(statistic: str, rule: str, value, lo=float("nan"), hi=float("nan"), n=None) -> dict:
    return {"statistic": statistic, "rule": rule, "value": value, "ci_lo": lo, "ci_hi": hi, "n": n}


def _prop(statistic: str, rule: str, k: int, n: int) -> dict:
    lo, hi = clopper_pearson(k, n)
    return _row(statistic, rule, _rate(k, n), lo, hi, n)


def endpoint_rows(tab: pd.DataFrame, rule: str) -> list[dict]:
    """The item-7 endpoints of one letter table as CSV rows (statistic, rule, value, ci_lo,
    ci_hi, n): counts (n_rows, n_excluded_nan, n_unlettered, n_scored, n_anchor_excluded,
    n_neg, n_pos, n_stress_D), fpr_A / fpr_AB on the negatives, P2_A / P2_AB / P2 (1.0 pass,
    0.0 fail; nan without negatives), recall_A / recall_AB on the positives, stress_D_AB_count
    / stress_D_AB_rate, and letter_dist_<truth_class>_<letter> counts (n = the class's scored
    rows)."""
    scored = tab[~tab["excluded"]]
    neg = scored[(scored["truth_class"] == NEG_CLASS) & ~scored["is_anchor"]]
    pos = scored[scored["is_positive"] & ~scored["is_anchor"]]
    stress = scored[scored["truth_class"] == STRESS_D]
    n_neg, n_pos, n_stress = len(neg), len(pos), len(stress)
    n_anchor = int((scored["is_anchor"] & ((scored["truth_class"] == NEG_CLASS) | scored["is_positive"])).sum())
    rows = [
        _row("n_rows", rule, len(tab)),
        _row("n_excluded_nan", rule, int((tab["exclude_reason"] == "nan_S").sum())),
        _row("n_unlettered", rule, int((tab["exclude_reason"] == "unlettered").sum())),
        _row("n_scored", rule, len(scored)),
        _row("n_anchor_excluded", rule, n_anchor),
        _row("n_neg", rule, n_neg), _row("n_pos", rule, n_pos), _row("n_stress_D", rule, n_stress),
    ]
    k_a = int(neg["letter"].eq("A").sum())
    k_ab = int(neg["letter"].isin(LENS_LETTERS).sum())
    fpr_a, fpr_ab = _prop("fpr_A", rule, k_a, n_neg), _prop("fpr_AB", rule, k_ab, n_neg)
    rows += [fpr_a, fpr_ab]
    if n_neg:
        p2_a = float(fpr_a["ci_hi"] <= P2_UPPER["A"])
        p2_ab = float(fpr_ab["ci_hi"] <= P2_UPPER["AB"])
        rows += [_row("P2_A", rule, p2_a, n=n_neg), _row("P2_AB", rule, p2_ab, n=n_neg),
                 _row("P2", rule, float(bool(p2_a) and bool(p2_ab)), n=n_neg)]
    else:
        rows += [_row(s, rule, float("nan"), n=0) for s in ("P2_A", "P2_AB", "P2")]
    rows += [_prop("recall_A", rule, int(pos["letter"].eq("A").sum()), n_pos),
             _prop("recall_AB", rule, int(pos["letter"].isin(LENS_LETTERS).sum()), n_pos)]
    k_s = int(stress["letter"].isin(LENS_LETTERS).sum())
    rows += [_row("stress_D_AB_count", rule, k_s, n=n_stress), _prop("stress_D_AB_rate", rule, k_s, n_stress)]
    for cls, sub in scored.groupby("truth_class", sort=True):
        letters = sub["letter"].map(_letter_str)
        for L in LETTER_COLS:
            rows.append(_row(f"letter_dist_{cls}_{L}", rule, int(letters.eq(L).sum()), n=len(sub)))
    return rows


def stat(rows: list[dict], statistic: str, rule: str) -> dict:
    """The one CSV row (statistic, rule); KeyError when absent."""
    for r in rows:
        if r["statistic"] == statistic and r["rule"] == rule:
            return r
    raise KeyError(f"no row ({statistic!r}, {rule!r})")


def letter_distribution(tab: pd.DataFrame) -> pd.DataFrame:
    """truth_class × letter counts of the scored rows, plus an `excluded` column."""
    scored = tab[~tab["excluded"]].copy()
    scored["L"] = scored["letter"].map(_letter_str)
    classes = sorted(tab["truth_class"].unique())
    out = pd.DataFrame(index=classes, columns=list(LETTER_COLS) + ["excluded", "n"]).fillna(0).astype(int)
    for cls in classes:
        sub = scored[scored["truth_class"] == cls]
        for L in LETTER_COLS:
            out.at[cls, L] = int(sub["L"].eq(L).sum())
        out.at[cls, "excluded"] = int((tab["truth_class"].eq(cls) & tab["excluded"]).sum())
        out.at[cls, "n"] = int(tab["truth_class"].eq(cls).sum())
    out.index.name = "truth_class"
    return out


# ------------------------------------------------------------------ selection (item 6)
def select_rule(recall_ab_rank: float, recall_ab_r1: float, fraction: float = SELECT_FRACTION) -> dict:
    """R1 unless recall_AB(R1) < fraction · recall_AB(letter_rank), then R2 (point estimates).
    A NaN recall (no positives) keeps R1 and says so."""
    numbers = {"recall_AB_letter_rank": recall_ab_rank, "recall_AB_R1": recall_ab_r1, "fraction": fraction,
               "bar": fraction * recall_ab_rank if not math.isnan(recall_ab_rank) else float("nan")}
    if math.isnan(recall_ab_rank) or math.isnan(recall_ab_r1):
        return {"rule": "R1", "reason": "recall undefined (no scored positives): the primary rule R1 stands",
                "numbers": numbers}
    bar = fraction * recall_ab_rank
    # item 6 reads "on the already-scored holdout a1 parquet"; letter_rank here is the advocate-only
    # letter of the a2 holdout parquet (item 7's assignment) — the sentence states both sources
    src = "recall_AB(R1; a1 holdout parquet, scored rows)"
    ref = "recall_AB(letter_rank; a2 holdout parquet)"
    if recall_ab_r1 < bar:
        return {"rule": "R2", "reason": f"{src} = {recall_ab_r1:.4f} < {fraction} x {ref} "
                                        f"= {bar:.4f}: the pre-stated fallback R2 is deployed", "numbers": numbers}
    return {"rule": "R1", "reason": f"{src} = {recall_ab_r1:.4f} >= {fraction} x {ref} "
                                    f"= {bar:.4f}: the primary rule R1 is deployed", "numbers": numbers}


# ------------------------------------------------------------------ rebuild parity
def rebuild_parity(preds: pd.DataFrame, records: dict, cols: tuple = REBUILD_COLS) -> pd.DataFrame:
    """`records.compare_rebuild` of the parquet against its own stored thresholds, on `cols`."""
    rebuilt = R.rebuild_rows(preds, records)
    return R.compare_rebuild(preds, rebuilt, cols)


# ------------------------------------------------------------------ the check
class SelectedRule(BaseModel):
    model_config = ConfigDict(extra="forbid")
    rule: str
    reason: str
    numbers: dict
    thresholds_sha16: str
    provisional: bool


class TransferResult(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)
    thresholds: dict
    letter_rank_source: str
    tables: dict                       # rule -> letter table (DataFrame)
    rows: list                         # CSV rows
    parity: dict                       # "a2" / "a1" -> compare_rebuild DataFrame
    selected: SelectedRule
    inputs: dict


def _input_meta(path: Path) -> dict:
    return {"file": path.name, "sha16": _util.sha_file(path)}


def run(a2_path, a1_path, thresholds: dict) -> TransferResult:
    """Everything the CLI writes, computed read-only: the three letter tables, the endpoint
    rows, the rebuild parity of both parquets, and the selection."""
    a2_path, a1_path = Path(a2_path), Path(a1_path)
    inputs = {"a2": _input_meta(a2_path), "a1": _input_meta(a1_path)}
    # a2 → letter_rank (records from votes; crit-column fallback when the votes are absent)
    a2_votes = R.votes_path_for(a2_path)
    parity: dict = {}
    if a2_votes.exists():
        a2, rec2 = R.load_run(a2_path, a2_votes)
        source = SOURCE_VOTES
        inputs["a2_votes"] = _input_meta(a2_votes)
        parity["a2"] = rebuild_parity(a2, rec2)
    else:
        a2, rec2, source = pd.read_parquet(a2_path), None, SOURCE_CRIT_COLUMNS
    # a1 → R1 / R2 (records required)
    a1_votes = R.votes_path_for(a1_path)
    if not a1_votes.exists():
        raise FileNotFoundError(f"{a1_votes.name}: R1 / R2 need the a1 votes parquet (critic + arbitrator records)")
    a1, rec1 = R.load_run(a1_path, a1_votes)
    inputs["a1_votes"] = _input_meta(a1_votes)
    parity["a1"] = rebuild_parity(a1, rec1)
    tables = {"letter_rank": letter_table(a2, rec2, thresholds, "letter_rank")}
    for rule in DEPLOY_RULES:
        tables[rule] = letter_table(a1, rec1, thresholds, rule)
    rows: list = []
    for rule in RULES:
        rows += endpoint_rows(tables[rule], rule)
    for key, tab in parity.items():
        for_rules = ("letter_rank",) if key == "a2" else DEPLOY_RULES
        for _, pr in tab.iterrows():
            for rule in for_rules:
                rows.append(_row(f"rebuild_mismatch_{pr['col']}", rule, int(pr["n_mismatch"]), n=int(pr["n_compared"])))
    sel = select_rule(stat(rows, "recall_AB", "letter_rank")["value"], stat(rows, "recall_AB", "R1")["value"])
    rows += [_row("selected_rule_is_R2", "selection", float(sel["rule"] == "R2")),
             _row("selection_bar", "selection", sel["numbers"]["bar"]),
             _row("t_A", "thresholds", float(thresholds["t_A"])), _row("t_B", "thresholds", float(thresholds["t_B"])),
             _row("tau0", "thresholds", float(thresholds["tau0"])),
             _row("provisional", "thresholds", float(bool(thresholds.get("provisional")))),
             _row("letter_rank_from_votes", "letter_rank", float(source == SOURCE_VOTES))]
    numbers = dict(sel["numbers"])
    numbers.update({
        "n_pos_letter_rank": stat(rows, "n_pos", "letter_rank")["value"],
        "n_pos_R1": stat(rows, "n_pos", "R1")["value"],
        "n_excluded_nan_a2": stat(rows, "n_excluded_nan", "letter_rank")["value"],
        "n_excluded_nan_a1": stat(rows, "n_excluded_nan", "R1")["value"],
        "n_unlettered_a1": stat(rows, "n_unlettered", "R1")["value"],
        "model_key": thresholds.get("model_key"), "letter_source": thresholds.get("letter_source"),
        "t_A": float(thresholds["t_A"]), "t_B": float(thresholds["t_B"]), "tau0": float(thresholds["tau0"]),
        "letter_rank_source": source, "inputs": inputs,
    })
    for rule in RULES:
        for s in ("fpr_A", "fpr_AB", "recall_A", "recall_AB", "stress_D_AB_count", "P2"):
            r = stat(rows, s, rule)
            numbers[f"{s}_{rule}"] = r["value"]
            if not R.is_missing(r["ci_hi"]):
                numbers[f"{s}_{rule}_ci"] = [r["ci_lo"], r["ci_hi"]]
    selected = SelectedRule(rule=sel["rule"], reason=sel["reason"], numbers=numbers,
                            thresholds_sha16=str(thresholds["thresholds_sha16"]),
                            provisional=bool(thresholds.get("provisional")))
    return TransferResult(thresholds=dict(thresholds), letter_rank_source=source, tables=tables, rows=rows,
                          parity=parity, selected=selected, inputs=inputs)


# ------------------------------------------------------------------ rendering
def _pct(v, lo=None, hi=None) -> str:
    if R.is_missing(v):
        return "n/a"
    s = f"{100 * float(v):.1f} %"
    if lo is not None and hi is not None and not R.is_missing(lo) and not R.is_missing(hi):
        s += f" [{100 * float(lo):.1f}, {100 * float(hi):.1f}]"
    return s


def _md_table(header: list, rows: list) -> str:
    lines = ["| " + " | ".join(str(h) for h in header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def render_md(res: TransferResult) -> str:
    thr, rows = res.thresholds, res.rows
    prov = bool(thr.get("provisional"))
    out = ["# Transfer check — v2-deploy items 6–7 (holdout, derived, zero API)", ""]
    if prov:
        out += ["> **PROVISIONAL THRESHOLDS.** `thresholds_v2.json` holds no calibrated entry for "
                f"`{thr.get('model_key')}`; the provisional numbers (t_A {thr['t_A']}, t_B {thr['t_B']}) were used. "
                "Nothing below is the registered transfer result until item 1 (calibration) has landed "
                "and this check is re-run with `--rule-select`.", ""]
    out += [f"Thresholds: key `{thr.get('thresholds_key')}` (`{thr.get('letter_source')}`), "
            f"t_A = {thr['t_A']}, t_B = {thr['t_B']}, tau0 = {thr['tau0']}, thresholds_sha16 `{thr['thresholds_sha16']}`, "
            f"table sha16 `{thr.get('table_sha16')}`.", ""]
    out += ["Inputs: " + "; ".join(f"{k} `{v['file']}` ({v['sha16']})" for k, v in res.inputs.items()) + ".", ""]
    if res.letter_rank_source != SOURCE_VOTES:
        out += ["> **letter_rank was built from the stored `crit_*` / `p_evidence` / `n_items` columns** — the a2 "
                "votes parquet is absent, so the AdvocateRecord could not be rebuilt. A/B membership is exact; "
                "the C/D split is copied from the stored letter where threshold-independent.", ""]
    else:
        out += ["letter_rank: AdvocateRecords rebuilt from the a2 votes parquet (run-time parse path).", ""]
    # exclusions
    out += ["## Rows", ""]
    out.append(_md_table(["rule", "parquet", "rows", "excluded (S NaN)", "unlettered", "scored", "negatives",
                          "positives", "stress_D"],
                         [[rule, "a2" if rule == "letter_rank" else "a1",
                           stat(rows, "n_rows", rule)["value"], stat(rows, "n_excluded_nan", rule)["value"],
                           stat(rows, "n_unlettered", rule)["value"], stat(rows, "n_scored", rule)["value"],
                           stat(rows, "n_neg", rule)["value"], stat(rows, "n_pos", rule)["value"],
                           stat(rows, "n_stress_D", rule)["value"]] for rule in RULES]))
    out.append("")
    # endpoints
    out += ["## Endpoints (95 % Clopper–Pearson)", ""]
    ep = []
    for rule in RULES:
        fa, fab = stat(rows, "fpr_A", rule), stat(rows, "fpr_AB", rule)
        ra, rab = stat(rows, "recall_A", rule), stat(rows, "recall_AB", rule)
        sd, sdr = stat(rows, "stress_D_AB_count", rule), stat(rows, "stress_D_AB_rate", rule)
        p2 = stat(rows, "P2", rule)["value"]
        ep.append([rule, _pct(fa["value"], fa["ci_lo"], fa["ci_hi"]), _pct(fab["value"], fab["ci_lo"], fab["ci_hi"]),
                   "n/a" if R.is_missing(p2) else ("PASS" if p2 else "FAIL"),
                   _pct(ra["value"], ra["ci_lo"], ra["ci_hi"]), _pct(rab["value"], rab["ci_lo"], rab["ci_hi"]),
                   f"{sd['value']}/{sd['n']} ({_pct(sdr['value'], sdr['ci_lo'], sdr['ci_hi'])})"])
    out.append(_md_table(["rule", "FPR@A (neg)", "FPR@A∪B (neg)", "P2 (A ≤ 2.5 %, A∪B ≤ 7.5 % upper CI)",
                          "recall@A (pos)", "recall@A∪B (pos)", "stress_D @A∪B"], ep))
    out.append("")
    # selection
    sel = res.selected
    out += ["## Selection (item 6)", "",
            f"**Selected rule: {sel.rule}**{' (PROVISIONAL — not binding)' if prov else ''}. {sel.reason}", ""]
    # letter distributions
    out += ["## Letter distribution per truth_class", ""]
    for rule in RULES:
        dist = letter_distribution(res.tables[rule])
        out += [f"### {rule}", "",
                _md_table(["truth_class"] + list(dist.columns),
                          [[cls] + [int(v) for v in dist.loc[cls]] for cls in dist.index]), ""]
    # vetoes
    for rule in DEPLOY_RULES:
        tab = res.tables[rule]
        demoted = tab[~tab["excluded"] & (tab["veto"] != "")]
        if len(demoted):
            counts = demoted.groupby(["truth_class", "veto"]).size().reset_index(name="n")
            out += [f"### {rule}: demotions by veto (role:alternative)", "",
                    _md_table(["truth_class", "veto", "n"], counts.values.tolist()), ""]
    # parity
    out += ["## Rebuild parity (each parquet under its own stored thresholds)", ""]
    for key, tab in res.parity.items():
        out += [f"### {key}", "", _md_table(list(tab.columns), tab.values.tolist()), ""]
    return "\n".join(out).rstrip() + "\n"


def rows_frame(rows: list) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=list(CSV_COLS))


def write_outputs(res: TransferResult, out_dir, overwrite: bool = False) -> dict:
    """transfer_check.md, transfer_check.csv (+ .sha), selected_rule.json in `out_dir`
    (created); refuses existing files unless `overwrite`. Returns {name: path}."""
    out = Path(out_dir)
    paths = {n: out / n for n in (MD_NAME, CSV_NAME, JSON_NAME)}
    clash = [p.name for p in paths.values() if p.exists()]
    if clash and not overwrite:
        raise FileExistsError(f"{out}: {clash} exist; pass --overwrite to replace them")
    out.mkdir(parents=True, exist_ok=True)
    paths[MD_NAME].write_text(render_md(res))
    _util.pin(rows_frame(res.rows), paths[CSV_NAME])
    paths[JSON_NAME].write_text(json.dumps(res.selected.model_dump(mode="json"), indent=2, sort_keys=False) + "\n")
    return paths


# ------------------------------------------------------------------ CLI
def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--thresholds", type=Path, default=THRESHOLDS_DEFAULT)
    ap.add_argument("--model-key", default=MODEL_KEY_DEFAULT)
    ap.add_argument("--a2", type=Path, default=A2_DEFAULT, help="a2 (advocate-only) holdout preds parquet")
    ap.add_argument("--a1", type=Path, default=A1_DEFAULT, help="a1 (full stack) holdout preds parquet")
    ap.add_argument("--out-dir", type=Path, required=True, help="NEW directory for the three outputs")
    ap.add_argument("--rule-select", action="store_true",
                    help="the registered selection run: refuses provisional thresholds")
    ap.add_argument("--overwrite", action="store_true", help="replace existing outputs in --out-dir")
    args = ap.parse_args(argv)

    thr = load_thresholds(args.thresholds, args.model_key)
    if thr["provisional"]:
        print(f"[transfer_check] PROVISIONAL THRESHOLDS: no calibrated {args.model_key!r} entry in "
              f"{args.thresholds} — using {thr['thresholds_key']!r} (t_A {thr['t_A']}, t_B {thr['t_B']})")
        if args.rule_select:
            print("[transfer_check] --rule-select refuses provisional thresholds (item 1: calibrate first)")
            return 2
    res = run(args.a2, args.a1, thr)
    paths = write_outputs(res, args.out_dir, overwrite=args.overwrite)
    rows = res.rows
    for rule in RULES:
        fa, fab, rab = stat(rows, "fpr_A", rule), stat(rows, "fpr_AB", rule), stat(rows, "recall_AB", rule)
        p2 = stat(rows, "P2", rule)["value"]
        print(f"[transfer_check] {rule:11s} FPR@A {_pct(fa['value'], fa['ci_lo'], fa['ci_hi'])}  "
              f"FPR@AB {_pct(fab['value'], fab['ci_lo'], fab['ci_hi'])}  P2 {'PASS' if p2 else 'FAIL'}  "
              f"recall@AB {_pct(rab['value'], rab['ci_lo'], rab['ci_hi'])}  "
              f"stress_D@AB {stat(rows, 'stress_D_AB_count', rule)['value']}  "
              f"(excluded NaN {stat(rows, 'n_excluded_nan', rule)['value']})")
    for key, tab in res.parity.items():
        bad = int(tab["n_mismatch"].sum())
        print(f"[transfer_check] rebuild parity {key}: {bad} mismatches over {len(tab)} columns")
    tag = " [PROVISIONAL — not binding]" if thr["provisional"] else ""
    print(f"[transfer_check] selected rule: {res.selected.rule}{tag} — {res.selected.reason}")
    print(f"[transfer_check] wrote " + ", ".join(str(p) for p in paths.values()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
