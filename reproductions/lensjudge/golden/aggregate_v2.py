#!/usr/bin/env python3
"""golden/aggregate_v2.py — the evidence-first aggregator: S, S_arb, letters, ranking, pass-count.

The incumbent JWST verifier graded a candidate by COUNTING how many of three adversarial
personas said "pass" (A=3 … D=0; `J/scripts/09_rank_report.py:grade()`). That is a
conjunction of three correlated vetoes: 23/24 known lenses the verifier saw scored 0/3, and
nothing in the output ranks (AUC ≈ 0.5). The v2 scheme replaces the count with a score:

    S     = p_ev · Π_{i named}          (1 − r_i · a_i)      PRIMARY ranking score
    S_arb = p_ev · Π_{i upheld/partial} (1 − r_i · a_i')     SECONDARY (arbitrated) arm

where p_ev is the ADVOCATE's probability that the evidence it LOCATED is lensing, r_i is a
CRITIC's graded refutation_strength (0–1) for a NAMED alternative, and a_i is the fraction of
the advocate's items that the critic's alternative accounts for — GEOMETRICALLY GUARDED: an
item only counts as covered when the critic's stated location box actually overlaps the
item's (r ± 0.3″, PA ± 20°) box, so a self-reported `accounts_for` cannot refute evidence the
critic did not point at. A critic with `no_opinion` or no named alternative is EXCLUDED
from the product (abstention is neither a pass nor a fail). `scale_tension` (the only
admissible θ_E critique) is capped at r ≤ 0.4 upstream, so scale alone can never make a D.

Letters are thresholds on S (t_A / t_B frozen on the design half at design-negative FPR
≤ 1 % / ≤ 5 %; `thresholds_v2.json` is keyed by model, a null entry falls back to the
provisional numbers and stamps `letter_source`), with two qualitative guards: A additionally
needs ≥ 2 of {curvature, counter_image, arc_morphology} ≥ 6 and no critic with r·a ≥ 0.8;
D needs S < t_B AND either nothing located (items == [] with `nothing_because` naming what
the centre is — the Huang-VI D definition) or one named critic covering EVERY item at
r ≥ 0.8 (in the arbitrated arm that critic must be upheld). Everything else is C.
U (never examined) ranks STRICTLY below every examined item (`rank_key`), so the top-N is an
examined list and the U tail is the next verification queue — the incumbent put U above D.

Pure python, stdlib + math ONLY, and records are read through `_get` so the same code
accepts pydantic records (lensjudge, `golden/schemas_panel.py`) and plain dicts (Nate's
`scripts/09_rank_report_v2.py` reads jsonl). A byte-identical copy of this file is shipped in
`golden/verifier_patch/scripts/aggregate_v2.py` and sha-parity-tested there — keep it
dependency-free. `self_test()` runs the pre-registered rank-15-like vs rank-13-like
scenarios (the poster-child lens the old count sent to C vs the PI-confirmed spiral FP).

    python lensjudge/golden/aggregate_v2.py      # runs self_test()
"""
from __future__ import annotations

import json
import math
from typing import Any, Iterable, Optional

__version__ = "2.1.0"

CRITIC_ROLES = ("artifact", "geometry", "morphology")
ROLES = ("advocate",) + CRITIC_ROLES + ("arbitrator",)
# the three "configuration" criteria an A must carry (≥ 2 of them scored ≥ 6 by the advocate)
LETTER_CRITERIA = ("curvature", "counter_image", "arc_morphology")
LETTER_CRITERIA_MIN = 6
# r·a at/above which a named critic blocks an A; r at/above which a critic that covers
# EVERY item makes a D
STRONG_R = 0.8
# the item box a critic's location must overlap to count as covering the item
DR_ARCSEC = 0.3
DPA_DEG = 20.0
# fallback thresholds until the design half freezes t_A/t_B (thresholds_v2.json "provisional")
PROVISIONAL = {"tau0": 0.15, "t_A": 0.80, "t_B": 0.50}
# the ONE map from a lensjudge --model alias to its thresholds_v2.json key (the runner, the
# analysis and the meta all go through it; Nate's Claude-Code run is keyed "opus_claude_code"
# by 09_rank_report_v2.py --model-key). An Opus run through the API is a different backbone
# setting from Opus inside Claude Code, hence its own key.
MODEL_KEYS = {"sonnet": "sonnet_api", "opus": "opus_api",
              "opus5": "opus5_api", "sonnet5": "sonnet5_api"}
VERDICTS = ("pass", "fail", "uncertain")
RULINGS = ("upheld", "partial", "overruled")
_EPS = 1e-9


# ------------------------------------------------------------------ record access
def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Field access that works for dicts (Nate's jsonl rows), pandas rows (any mapping with
    .get — a Series' own `.name` attribute must not shadow a `name` column) and
    pydantic/dataclass records."""
    if obj is None:
        return default
    get = getattr(obj, "get", None)
    if callable(get):
        try:
            return get(key, default)
        except Exception:
            return default
    return getattr(obj, key, default)


def _num(x: Any) -> Optional[float]:
    """float(x) or None when x is missing / non-numeric / NaN."""
    if x is None or isinstance(x, bool):
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) else v


def _int_set(xs: Any) -> set:
    out = set()
    for x in xs or []:
        try:
            out.add(int(x))
        except (TypeError, ValueError):
            continue
    return out


def is_nan(x: Any) -> bool:
    v = _num(x)
    return v is None


# ------------------------------------------------------------------ geometry
def _arc(from_deg: float, to_deg: float) -> tuple:
    """A position-angle arc as (start in [0, 360), length in [0, 360]) going from `from_deg`
    to `to_deg` in the positive direction: 350 → 10 is the 20° arc through north, 10 → 350
    the 340° arc the other way, from == to a point, to − from ≥ 360 the full circle."""
    span = to_deg - from_deg
    if span >= 360.0:
        span = 360.0
    elif span < 0.0:
        span = span % 360.0          # python's modulo: −340 → 20
    return (from_deg % 360.0, span)


def _in_arc(x_deg: float, arc: tuple) -> bool:
    start, length = arc
    return length >= 360.0 or ((x_deg - start) % 360.0) <= length + _EPS


def _arcs_overlap(a: tuple, b: tuple) -> bool:
    # two arcs on a circle intersect iff one contains the other's start point
    return _in_arc(b[0], a) or _in_arc(a[0], b)


def covers(item: Any, location: Any, dr: float = DR_ARCSEC, dpa: float = DPA_DEG) -> bool:
    """Does a critic's location box {r_arcsec_from, r_arcsec_to, pa_deg_from, pa_deg_to}
    overlap the advocate item's box (r_arcsec ± dr, [pa_deg_from − dpa, pa_deg_to + dpa])?
    Position angles wrap at 360 (a 350→10 box covers PA 5). `location` None ⇒ False: a
    critic that does not say WHERE its alternative is covers nothing."""
    if location is None:
        return False
    r = _num(_get(item, "r_arcsec"))
    pf, pt = _num(_get(item, "pa_deg_from")), _num(_get(item, "pa_deg_to"))
    lf, lt = _num(_get(location, "r_arcsec_from")), _num(_get(location, "r_arcsec_to"))
    qf, qt = _num(_get(location, "pa_deg_from")), _num(_get(location, "pa_deg_to"))
    if None in (r, pf, pt, lf, lt, qf, qt):
        return False
    # radius: closed intervals
    item_lo, item_hi = max(0.0, r - dr), r + dr
    loc_lo, loc_hi = min(lf, lt), max(lf, lt)
    if item_lo > loc_hi + _EPS or loc_lo > item_hi + _EPS:
        return False
    # position angle: arcs with wrap-around; the item arc is padded by dpa on both sides AFTER
    # it is resolved (padding the endpoints first would turn a 340° wrap-form arc such as
    # 100 → 80 into the 20° arc 80 → 100 — the padded arc must always be the longer one)
    start, length = _arc(pf, pt)
    item_arc = ((start - dpa) % 360.0, min(360.0, length + 2.0 * dpa))
    return _arcs_overlap(item_arc, _arc(qf, qt))


def coverage_fraction(items: Iterable[Any], critic: Any, dr: float = DR_ARCSEC,
                      dpa: float = DPA_DEG) -> float:
    """a_i ∈ [0, 1]: the fraction of the advocate's items that the critic BOTH lists in
    `accounts_for` AND geometrically covers with its `location`. No items ⇒ 0."""
    items = list(items or [])
    if not items:
        return 0.0
    claimed = _int_set(_get(critic, "accounts_for"))
    loc = _get(critic, "location")
    n = 0
    for it in items:
        try:
            k = int(_get(it, "k"))
        except (TypeError, ValueError):
            continue
        if k in claimed and covers(it, loc, dr, dpa):
            n += 1
    return n / len(items)


# ------------------------------------------------------------------ terms of the product
def critic_named(critic: Any) -> bool:
    """A critic enters the product only with a NAMED alternative and no abstention."""
    if critic is None or bool(_get(critic, "no_opinion", False)):
        return False
    alt = _get(critic, "alternative")
    return alt is not None and str(alt).strip() != ""


def _critic_map(critics: Any) -> dict:
    """role → record. Accepts {role: record|None} (None = called but failed to parse) or a
    list of records keyed by their `persona`."""
    if critics is None:
        return {}
    if isinstance(critics, dict):
        return dict(critics)
    out = {}
    for c in critics:
        if c is not None:
            out[str(_get(c, "persona"))] = c
    return out


def _ruling_map(arbitrator: Any) -> dict:
    return {str(_get(r, "persona")): r for r in (_get(arbitrator, "rulings") or [])}


def critic_terms(advocate: Any, critics: Any, arbitrator: Any = None,
                 dr: float = DR_ARCSEC, dpa: float = DPA_DEG) -> dict:
    """Per critic role: {r, a, a_geom, named, included, parsed, ruling, no_opinion,
    alternative}. `a_geom` is the geometric coverage; `a` is what enters the product —
    equal to a_geom without an arbitrator or when the critic is upheld (or not ruled on),
    |covers ∩ items| / |items| when the ruling is partial, and the critic is excluded when
    overruled. `parsed=False` marks a role that was called but returned no record."""
    items = list(_get(advocate, "items") or [])
    item_ks = _int_set(_get(it, "k") for it in items)
    rulings = _ruling_map(arbitrator) if arbitrator is not None else {}
    terms = {}
    for role, c in _critic_map(critics).items():
        if c is None:
            terms[role] = {"parsed": False, "named": False, "included": False,
                           "r": float("nan"), "a": float("nan"), "a_geom": float("nan"),
                           "ruling": None, "no_opinion": None, "alternative": None}
            continue
        named = critic_named(c)
        r = _num(_get(c, "refutation_strength"))
        r = 0.0 if r is None else min(1.0, max(0.0, r))
        a_geom = coverage_fraction(items, c, dr, dpa) if named else 0.0
        ruling = rulings.get(role)
        kind = str(_get(ruling, "ruling", "") or "") if ruling is not None else None
        a, included = a_geom, named
        if named and kind == "partial":
            cov = _int_set(_get(ruling, "covers")) & item_ks
            a = len(cov) / len(item_ks) if item_ks else 0.0
        elif named and kind == "overruled":
            a, included = 0.0, False
        terms[role] = {"parsed": True, "named": named, "included": included,
                       "r": r, "a": a, "a_geom": a_geom, "ruling": kind,
                       "no_opinion": bool(_get(c, "no_opinion", False)),
                       "alternative": _get(c, "alternative")}
    return terms


def _product(advocate: Any, terms: dict) -> float:
    """p_ev · Π_included (1 − r·a); NaN when the advocate or any called critic is missing."""
    p_ev = _num(_get(advocate, "p_evidence"))
    if p_ev is None or any(not t["parsed"] for t in terms.values()):
        return float("nan")
    s = min(1.0, max(0.0, p_ev))
    for t in terms.values():
        if t["included"]:
            s *= 1.0 - min(1.0, max(0.0, t["r"] * t["a"]))
    return s


def score_S(advocate: Any, critics: Any, dr: float = DR_ARCSEC, dpa: float = DPA_DEG) -> float:
    """PRIMARY: S = p_ev · Π_{named critics} (1 − r_i·a_i). Critics with no_opinion or no
    alternative are excluded; no critics at all (p_ev < τ0, critics never called) ⇒ S = p_ev.
    NaN when the advocate is missing or a called critic failed to parse (dict value None)."""
    return _product(advocate, critic_terms(advocate, critics, None, dr, dpa))


def score_S_arb(advocate: Any, critics: Any, arbitrator: Any,
                dr: float = DR_ARCSEC, dpa: float = DPA_DEG) -> float:
    """SECONDARY: the same product over the critics the arbitrator upheld (a_i as computed)
    or ruled partial (a_i' = |covers ∩ items| / |items|); overruled critics drop out. With no
    arbitrator (none was needed: no critic named an alternative) S_arb == S."""
    return _product(advocate, critic_terms(advocate, critics, arbitrator, dr, dpa))


# ------------------------------------------------------------------ thresholds + letters
def load_thresholds(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _usable(entry: Any) -> bool:
    return isinstance(entry, dict) and _num(entry.get("t_A")) is not None \
        and _num(entry.get("t_B")) is not None


def resolve_thresholds(table: dict, model_key: str, fallback_keys: tuple = ("provisional",),
                       fallback_source: Optional[str] = None) -> dict:
    """Pick the thresholds for `model_key` from a thresholds_v2.json table:
    {"tau0", "t_A", "t_B", "letter_source", "thresholds_key"}. A null / incomplete entry
    falls back to the first usable `fallback_keys` entry (default the "provisional" numbers)
    and is labelled: "<model_key>_calibrated" when the model's own entry is used,
    "provisional" for the provisional numbers, "<key>_uncalibrated" for another model's
    numbers (or `fallback_source` verbatim when given, e.g. Nate's
    "sonnet_thresholds_uncalibrated")."""
    prov = table.get("provisional") if isinstance(table, dict) else None
    tau_default = _num((prov or {}).get("tau0")) if isinstance(prov, dict) else None
    if tau_default is None:
        tau_default = PROVISIONAL["tau0"]
    chain = [(model_key, f"{model_key}_calibrated")]
    for k in fallback_keys:
        chain.append((k, "provisional" if k == "provisional" else f"{k}_uncalibrated"))
    for key, label in chain:
        entry = table.get(key) if isinstance(table, dict) else None
        if not _usable(entry):
            continue
        if key != model_key and fallback_source:
            label = fallback_source
        tau0 = _num(entry.get("tau0"))
        return {"tau0": tau_default if tau0 is None else tau0,
                "t_A": float(entry["t_A"]), "t_B": float(entry["t_B"]),
                "letter_source": label, "thresholds_key": key}
    # nothing usable in the table at all: the module's own provisional numbers
    return {**PROVISIONAL, "letter_source": "provisional", "thresholds_key": "provisional"}


def assign_letter(S: Any, advocate: Any, critics: Any, thresholds: dict,
                  arbitrator: Any = None, dr: float = DR_ARCSEC, dpa: float = DPA_DEG) -> tuple:
    """(letter, letter_source) for a score S (pass S for the primary letter, S_arb plus the
    arbitrator for the arbitrated one):
      A  iff S ≥ t_A AND ≥ 2 of {curvature, counter_image, arc_morphology} ≥ 6 AND no
         (included) critic with r·a ≥ 0.8
      B  iff S ≥ t_B (and not A)
      D  iff S < t_B AND (items == [] with a non-empty nothing_because, OR a named critic
         with a_i == 1 and r_i ≥ 0.8 — under an arbitrator that critic must be upheld)
      C  otherwise.  S NaN (parse failure) ⇒ letter None.
    `thresholds` is a `resolve_thresholds` dict (t_A, t_B, letter_source)."""
    source = str(thresholds.get("letter_source", "provisional"))
    s = _num(S)
    if s is None:
        return None, source
    t_a, t_b = float(thresholds["t_A"]), float(thresholds["t_B"])
    terms = critic_terms(advocate, critics, arbitrator, dr, dpa)
    blocker = any(t["included"] and t["r"] * t["a"] >= STRONG_R - _EPS for t in terms.values())
    crit = _get(advocate, "criteria") or {}
    n_strong = sum(1 for c in LETTER_CRITERIA
                   if (_num(_get(crit, c)) or 0.0) >= LETTER_CRITERIA_MIN)
    if s >= t_a and n_strong >= 2 and not blocker:
        return "A", source
    if s >= t_b:
        return "B", source
    items = list(_get(advocate, "items") or [])
    nothing_located = not items and str(_get(advocate, "nothing_because", "") or "").strip() != ""
    fully_refuted = any(
        t["included"] and t["a_geom"] >= 1.0 - _EPS and t["r"] >= STRONG_R - _EPS
        and (arbitrator is None or t["ruling"] == "upheld")
        for t in terms.values())
    if nothing_located or fully_refuted:
        return "D", source
    return "C", source


# ------------------------------------------------------------------ ranking
def _examined_score(row: Any) -> Optional[float]:
    s = _num(_get(row, "S"))
    return _num(_get(row, "p_lens")) if s is None else s


def _u_score(row: Any) -> float:
    """Within the U tail: the inspector's 0–100 confidence (Nate's `confidence`, the frame's /
    truth manifest's `pipe_inspector_conf` — also on the run's 0–100 scale, 15..95 — or
    `inspector_score`), mapped to [0, 1) so a U row can never reach an examined row's S."""
    for key in ("inspector_score", "confidence", "pipe_inspector_conf"):
        v = _num(_get(row, key))
        if v is not None:
            return min(0.999, max(0.0, v / 100.0))
    return 0.0


def rank_score(row: Any) -> float:
    """One number for a `score` column: S for examined rows, −1 + conf/100 for U rows, so
    every U sits strictly below every examined item."""
    s = _examined_score(row)
    return s if s is not None else -1.0 + _u_score(row)


def rank_key(row: Any) -> tuple:
    """Sort key (ascending ⇒ best first): examined rows (a finite S) before every U row,
    then S desc, then p_evidence desc, then the id for a deterministic order. U rows are
    ordered among themselves by the inspector's confidence."""
    ident = str(_get(row, "id", None) or _get(row, "name", None) or "")
    s = _examined_score(row)
    if s is None:
        return (1, -_u_score(row), 0.0, ident)
    p_ev = _num(_get(row, "p_evidence"))
    return (0, -s, -(p_ev if p_ev is not None else 0.0), ident)


# ------------------------------------------------------------------ the incumbent pass-count
def passcount_incumbent(verdicts: Iterable[Any], strict: bool = False) -> tuple:
    """(n_pass, n_fail, n_uncertain, letter) table-identical to
    `J/scripts/09_rank_report.py:grade()`: no votes ⇒ U; 3 passes A, 2 B, 1 C, else D
    (`uncertain` is a vote that is not a pass, so pass/pass/uncertain is B). Verdicts may
    be strings or records with a `verdict` field; like the incumbent loader an unknown
    verdict string is read as `uncertain` (strict=True raises instead)."""
    n = {"pass": 0, "fail": 0, "uncertain": 0}
    for v in verdicts or []:
        s = v if isinstance(v, str) else _get(v, "verdict", "")
        s = str(s or "").lower().strip()
        if s not in VERDICTS:
            if strict:
                raise ValueError(f"unknown verdict {s!r}")
            s = "uncertain"
        n[s] += 1
    n_votes = n["pass"] + n["fail"] + n["uncertain"]
    letter = "U" if n_votes == 0 else {3: "A", 2: "B", 1: "C"}.get(n["pass"], "D")
    return n["pass"], n["fail"], n["uncertain"], letter


# ------------------------------------------------------------------ self-test
def _scenarios() -> dict:
    """The two pre-registered synthetic cases (plain dicts, as Nate's jsonl rows arrive)."""
    rank15 = {
        "advocate": {"id": "rank15like", "persona": "advocate",
                     "criteria": {"source_contrast": 7, "low_surface_brightness": 6, "curvature": 8,
                                  "counter_image": 6, "arc_morphology": 7},
                     "items": [{"k": 1, "what": "arc", "panel": "d", "r_arcsec": 3.6, "pa_deg_from": 40,
                                "pa_deg_to": 120, "visible_in_direct": True, "criteria": [3, 5]},
                               {"k": 2, "what": "counter", "panel": "d", "r_arcsec": 3.2, "pa_deg_from": 230,
                                "pa_deg_to": 250, "visible_in_direct": True, "criteria": [4]},
                               {"k": 3, "what": "knot", "panel": "e", "r_arcsec": 3.7, "pa_deg_from": 150,
                                "pa_deg_to": 170, "visible_in_direct": True, "criteria": [2]}],
                     "scale_class": "group", "n_red_neighbours_10as": 2, "bcg_like_halo": False,
                     "deflector_is_centre": True, "p_evidence": 0.8, "nothing_because": "", "notes": ""},
        "critics": {
            "artifact": {"id": "rank15like", "persona": "artifact", "no_opinion": True,
                         "no_opinion_reason": "feature_not_in_my_views", "alternative": None,
                         "accounts_for": [], "leaves_standing": [1, 2, 3], "refutation_strength": 0.0},
            # scale_tension over the whole stamp: covers all three items at the capped r = 0.4
            "geometry": {"id": "rank15like", "persona": "geometry", "no_opinion": False,
                         "alternative": "scale_tension",
                         "location": {"r_arcsec_from": 2.5, "r_arcsec_to": 5.0, "pa_deg_from": 0, "pa_deg_to": 360},
                         "accounts_for": [1, 2, 3], "leaves_standing": [], "refutation_strength": 0.4},
            # a spiral arm located only where item 1 is: claims all three, covers one
            "morphology": {"id": "rank15like", "persona": "morphology", "no_opinion": False,
                           "alternative": "spiral_arm",
                           "location": {"r_arcsec_from": 3.0, "r_arcsec_to": 4.0, "pa_deg_from": 30, "pa_deg_to": 100},
                           "accounts_for": [1, 2, 3], "leaves_standing": [], "refutation_strength": 0.6}},
        "arbitrator": {"id": "rank15like", "persona": "arbitrator",
                       "rulings": [{"persona": "geometry", "ruling": "overruled", "covers": [], "why": "group halo visible"},
                                   {"persona": "morphology", "ruling": "partial", "covers": [1], "why": "arm reaches item 1 only"}],
                       "surviving_items": [2, 3], "letter_llm": "B", "scale_class_final": "group",
                       "needs_human": False, "rationale": "two items survive"}}
    rank13 = {
        "advocate": {"id": "rank13like", "persona": "advocate",
                     "criteria": {"source_contrast": 3, "low_surface_brightness": 5, "curvature": 5,
                                  "counter_image": 2, "arc_morphology": 4},
                     "items": [{"k": 1, "what": "arc-like arm", "panel": "d", "r_arcsec": 1.2, "pa_deg_from": 300,
                                "pa_deg_to": 30, "visible_in_direct": True, "criteria": [3]},
                               {"k": 2, "what": "second arm", "panel": "d", "r_arcsec": 1.4, "pa_deg_from": 120,
                                "pa_deg_to": 200, "visible_in_direct": True, "criteria": [5]}],
                     "scale_class": "galaxy", "n_red_neighbours_10as": 0, "bcg_like_halo": False,
                     "deflector_is_centre": True, "p_evidence": 0.7, "nothing_because": "", "notes": ""},
        "critics": {
            "artifact": {"id": "rank13like", "persona": "artifact", "no_opinion": False, "alternative": None,
                         "accounts_for": [], "leaves_standing": [1, 2], "refutation_strength": 0.0},
            "geometry": {"id": "rank13like", "persona": "geometry", "no_opinion": True,
                         "no_opinion_reason": "outside_competence", "alternative": None,
                         "accounts_for": [], "leaves_standing": [1, 2], "refutation_strength": 0.0},
            # spiral arms with a stellar bridge, located over BOTH items (a 300→30 box wraps north)
            "morphology": {"id": "rank13like", "persona": "morphology", "no_opinion": False,
                           "alternative": "spiral_arm",
                           "location": {"r_arcsec_from": 0.8, "r_arcsec_to": 1.8, "pa_deg_from": 0, "pa_deg_to": 360},
                           "accounts_for": [1, 2], "leaves_standing": [], "refutation_strength": 0.9}},
        "arbitrator": {"id": "rank13like", "persona": "arbitrator",
                       "rulings": [{"persona": "morphology", "ruling": "upheld", "covers": [1, 2], "why": "bridge to nucleus"}],
                       "surviving_items": [], "letter_llm": "D", "scale_class_final": "galaxy",
                       "needs_human": False, "rationale": "arms"}}
    return {"rank15": rank15, "rank13": rank13}


def self_test(verbose: bool = False) -> dict:
    """Canned rank-15-like vs rank-13-like scenarios under the provisional thresholds;
    raises AssertionError on any departure from the pre-registered behaviour."""
    thr = {**PROVISIONAL, "letter_source": "provisional"}
    sc = _scenarios()
    out = {}
    for name, case in sc.items():
        adv, crit, arb = case["advocate"], case["critics"], case["arbitrator"]
        s = score_S(adv, crit)
        s_arb = score_S_arb(adv, crit, arb)
        letter, src = assign_letter(s, adv, crit, thr)
        letter_arb, _ = assign_letter(s_arb, adv, crit, thr, arbitrator=arb)
        out[name] = {"S": s, "S_arb": s_arb, "letter": letter, "letter_arb": letter_arb, "source": src}
    # rank 15: p_ev 0.8; geometry covers 3/3 at r 0.4; morphology covers 1/3 at r 0.6
    want15 = 0.8 * (1 - 0.4 * 1.0) * (1 - 0.6 * (1 / 3))
    assert abs(out["rank15"]["S"] - want15) < 1e-12, out["rank15"]
    assert out["rank15"]["letter"] == "C", out["rank15"]         # < t_B, no critic at a=1 & r≥0.8
    # arbitrated: geometry overruled, morphology partial on item 1 → 0.8·(1 − 0.6/3) = 0.64 → B
    assert abs(out["rank15"]["S_arb"] - 0.8 * (1 - 0.6 / 3)) < 1e-12, out["rank15"]
    assert out["rank15"]["letter_arb"] == "B", out["rank15"]
    # rank 13: spiral_arm at r 0.9 covering both items → S = 0.07, D by the full-coverage rule
    assert abs(out["rank13"]["S"] - 0.7 * (1 - 0.9)) < 1e-12, out["rank13"]
    assert out["rank13"]["letter"] == "D" and out["rank13"]["letter_arb"] == "D", out["rank13"]
    assert out["rank15"]["S"] > out["rank13"]["S"]
    assert out["rank15"]["source"] == "provisional"
    # ranking: the unexamined row sorts below both examined rows whatever its confidence
    rows = [{"id": "u", "S": None, "confidence": 99.0},
            {"id": "r13", "S": out["rank13"]["S"], "p_evidence": 0.7},
            {"id": "r15", "S": out["rank15"]["S"], "p_evidence": 0.8}]
    assert [r["id"] for r in sorted(rows, key=rank_key)] == ["r15", "r13", "u"]
    # the incumbent table
    assert passcount_incumbent([])[3] == "U"
    assert passcount_incumbent(["pass", "pass", "uncertain"]) == (2, 0, 1, "B")
    assert passcount_incumbent(["fail", "fail", "fail"]) == (0, 3, 0, "D")
    # PA wrap-around in the guard
    assert covers({"r_arcsec": 1.0, "pa_deg_from": 5, "pa_deg_to": 5},
                  {"r_arcsec_from": 0.5, "r_arcsec_to": 2.0, "pa_deg_from": 350, "pa_deg_to": 10}, dpa=0.0)
    # a near-ring written in wrap form (100 → 80 = 340°) stays a ring after padding: a critic
    # box at PA 180–220 covers it (endpoint-first padding collapsed it to 80 → 100)
    ring = {"r_arcsec": 1.0, "pa_deg_from": 100, "pa_deg_to": 80}
    assert covers(ring, {"r_arcsec_from": 0.5, "r_arcsec_to": 2.0, "pa_deg_from": 180, "pa_deg_to": 220})
    assert covers(ring, {"r_arcsec_from": 0.5, "r_arcsec_to": 2.0, "pa_deg_from": 180, "pa_deg_to": 220}, dpa=0.0)
    assert not covers({"r_arcsec": 1.0, "pa_deg_from": 100, "pa_deg_to": 120},
                      {"r_arcsec_from": 0.5, "r_arcsec_to": 2.0, "pa_deg_from": 180, "pa_deg_to": 220})
    # U rows carry the run's 0–100 inspector confidence under any of its column names
    assert rank_score({"id": "u", "S": None, "pipe_inspector_conf": 95.0}) < 0 < rank_score({"S": 0.01})
    if verbose:
        for k, v in out.items():
            print(k, {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items()})
    return out


if __name__ == "__main__":
    self_test(verbose=True)
    print(f"aggregate_v2 {__version__}: self_test PASS")
