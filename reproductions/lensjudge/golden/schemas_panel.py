"""golden/schemas_panel.py — the model-record contract for the evidence-first JWST panel.

Four pydantic records, one per role the runner calls (`golden/run_truth_eval.py`,
`golden/panel.py`) and one for the incumbent baseline arm, plus the python-side result
object the runner writes rows from. Why a separate module from `common/schemas.py`:

1. **Nothing is coerced.** `common.schemas.ImageGrade` repairs model output (a stray letter
   becomes "D", unknown keys are dropped). These records are the SCORED FIELDS of the scheme
   — `p_evidence`, located items, `refutation_strength`, `accounts_for`, rulings — and a
   repaired field would silently move an item's score. Every record here is
   `extra="forbid"`; every enum is a `Literal`; an unknown alternative, ruling, panel name
   or letter is a validation error, which `common.parse.parse_model` turns into a parse
   FAILURE (one repair retry lives in `imaging/grader_direct`), never a default.
2. **The brief's rules are validators.** `scale_tension` is the only admissible θ_E critique
   and is capped at refutation_strength ≤ 0.4; an abstaining critic (`no_opinion`) names no
   alternative; a critic that names an alternative must give its strength AND a location box
   ("a named alternative requires a location" — without one it could never cover an item, so
   it would be reported as the contaminant while changing nothing); advocate item numbers
   `k` are unique (critics refer to them). A record that breaks a rule is rejected, not
   clamped — the monitors count parse failures per role.
3. **One row shape for every consumer.** `PanelResult` + `to_row` write `p_lens = S`,
   `grade_pred = letter`, `confidence = p_evidence`, `contaminant = alternative_final`,
   `escalate = needs_human` beside the v2 columns (S, S_arb, per-critic a/r/no_opinion,
   letter_source, parse_fail_roles, per-role system shas) so `imaging/run_batch._row_dict`-
   style consumers, `eval/score.recovery_at_fpr` and `eval/lensbench_gate.grade_flip_rate`
   (which need `name, grade_pred, p_lens`) work unchanged.

The arithmetic (S, S_arb, letters, ranking) lives in `golden/aggregate_v2.py`, which is
stdlib-only because a byte-identical copy ships to Nate's repo; `assemble()` here is the
lensjudge-side glue that applies the parse-failure policy and builds the PanelResult.

Parse-failure policy (pre-registered, stated once here): a row is a parse failure when ANY
called role — advocate, a critic, or the arbitrator — returned no record. Then S and S_arb
are NaN, the letters None, `parse_ok` False, and the row is kept with `parse_fail_roles`;
it is excluded from every recall/FPR number and counted in the parse-failure rate. An
arbitrator-only failure therefore also voids S (the primary does not need the arbitrator,
but one policy for every role is what was registered and what `parse_ok` reports).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from lensjudge.golden import aggregate_v2

CRITIC_ROLES = aggregate_v2.CRITIC_ROLES                 # ("artifact", "geometry", "morphology")
ROLES = aggregate_v2.ROLES                               # advocate, critics, arbitrator
PANELS = ("a", "b", "c", "d", "e", "f", "ctx")
SCALE_CLASSES = ("galaxy", "group", "cluster", "none")
ALTERNATIVES = ("spiral_arm", "ring_galaxy", "shell_tidal", "merger", "edge_on_disk",
                "companion_projection", "star_forming_clump", "diffraction_spike",
                "detector_artifact", "subtraction_residual", "psf_wing", "scale_tension", "other")
NO_OPINION_REASONS = ("outside_competence", "feature_not_in_my_views", "image_quality")
SCALE_TENSION_MAX_R = 0.4
CRITERIA_V2 = ("source_contrast", "low_surface_brightness", "curvature", "counter_image",
               "arc_morphology")

_FORBID = ConfigDict(extra="forbid")


def _blank_to_none(v):
    """JSON "" where the brief says null: the one normalisation allowed (no other coercion)."""
    return None if isinstance(v, str) and v.strip() == "" else v


# ------------------------------------------------------------------ advocate
class CriteriaV2(BaseModel):
    """The five NIRCam-adapted criteria, 0 (absent) – 10 (textbook)."""
    model_config = _FORBID
    source_contrast: int = Field(ge=0, le=10)
    low_surface_brightness: int = Field(ge=0, le=10)
    curvature: int = Field(ge=0, le=10)
    counter_image: int = Field(ge=0, le=10)
    arc_morphology: int = Field(ge=0, le=10)


class EvidenceItem(BaseModel):
    """One LOCATED piece of evidence: what, which panel, where (radius, PA span), whether it
    is traceable in an un-subtracted panel, and which criteria (1–5) it supports."""
    model_config = _FORBID
    k: int = Field(ge=1)
    what: str
    panel: Literal["a", "b", "c", "d", "e", "f", "ctx"]
    r_arcsec: float = Field(ge=0.0)
    pa_deg_from: float
    pa_deg_to: float
    visible_in_direct: bool
    criteria: list[int] = Field(default_factory=list)

    @field_validator("criteria")
    @classmethod
    def _crit_idx(cls, v):
        bad = [c for c in v if c not in (1, 2, 3, 4, 5)]
        if bad:
            raise ValueError(f"criteria indices must be 1..5, got {bad}")
        return v


class CounterImagePos(BaseModel):
    model_config = _FORBID
    r_arcsec: float = Field(ge=0.0)
    pa_deg: float


class AdvocateRecord(BaseModel):
    """The evidence scorer's record: criteria, located items, reported (never penalised)
    scale facts, and p_evidence. `nothing_because` names what the centre is when no item was
    located (the D rule needs it non-empty)."""
    model_config = _FORBID
    id: str
    persona: Literal["advocate"]
    criteria: CriteriaV2
    items: list[EvidenceItem]
    arc_radius_arcsec: Optional[float] = None
    arc_pa_span_deg: Optional[tuple[float, float]] = None
    counter_image_pos: Optional[CounterImagePos] = None
    centre_of_curvature_offset_arcsec: Optional[float] = None
    scale_class: Literal["galaxy", "group", "cluster", "none"]
    n_red_neighbours_10as: int = Field(ge=0)
    bcg_like_halo: bool
    deflector_is_centre: bool
    p_evidence: float = Field(ge=0.0, le=1.0)
    nothing_because: str = ""
    notes: str = ""

    @field_validator("items")
    @classmethod
    def _unique_k(cls, v):
        ks = [it.k for it in v]
        dup = sorted({k for k in ks if ks.count(k) > 1})
        if dup:
            raise ValueError(f"item k must be unique, duplicated: {dup}")
        return v


# ------------------------------------------------------------------ critics
class LocationBox(BaseModel):
    """Where the critic's alternative is: a radius range and a PA range (wraps at 360)."""
    model_config = _FORBID
    r_arcsec_from: float = Field(ge=0.0)
    r_arcsec_to: float = Field(ge=0.0)
    pa_deg_from: float
    pa_deg_to: float


class CriticRecord(BaseModel):
    """One critic's report: an abstention (`no_opinion` + reason), or a NAMED alternative with
    its location, the items it accounts for / leaves standing and a graded strength. A
    critic with `no_opinion=False` and `alternative=None` is the symmetric-mandate answer
    ("nothing in my competence fits") and is excluded from the product like an abstention."""
    model_config = _FORBID
    id: str
    persona: Literal["artifact", "geometry", "morphology"]
    no_opinion: bool = False
    no_opinion_reason: Optional[Literal["outside_competence", "feature_not_in_my_views",
                                        "image_quality"]] = None
    alternative: Optional[Literal["spiral_arm", "ring_galaxy", "shell_tidal", "merger",
                                  "edge_on_disk", "companion_projection", "star_forming_clump",
                                  "diffraction_spike", "detector_artifact", "subtraction_residual",
                                  "psf_wing", "scale_tension", "other"]] = None
    alternative_desc: str = ""
    location: Optional[LocationBox] = None
    accounts_for: list[int] = Field(default_factory=list)
    leaves_standing: list[int] = Field(default_factory=list)
    refutation_strength: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    measured: Optional[dict] = None
    scale_class: Optional[str] = None
    notes: str = ""

    @field_validator("alternative", "no_opinion_reason", mode="before")
    @classmethod
    def _blank(cls, v):
        return _blank_to_none(v)

    @model_validator(mode="after")
    def _brief_rules(self):
        if self.no_opinion and self.alternative is not None:
            raise ValueError("no_opinion=true must not name an alternative")
        if self.alternative is not None and self.refutation_strength is None:
            raise ValueError(f"alternative {self.alternative!r} needs a refutation_strength")
        if self.alternative is not None and self.location is None:
            raise ValueError(f"alternative {self.alternative!r} needs a location box (the brief: "
                             f"a named alternative requires a location)")
        if self.alternative == "scale_tension" and self.refutation_strength > SCALE_TENSION_MAX_R:
            raise ValueError(f"scale_tension is capped at refutation_strength <= {SCALE_TENSION_MAX_R} "
                             f"(got {self.refutation_strength})")
        if self.refutation_strength is None:
            self.refutation_strength = 0.0
        return self


# ------------------------------------------------------------------ arbitrator
class Ruling(BaseModel):
    model_config = _FORBID
    persona: Literal["artifact", "geometry", "morphology"]
    ruling: Literal["upheld", "partial", "overruled"]
    covers: list[int] = Field(default_factory=list)
    why: str = ""


class ArbitratorRecord(BaseModel):
    """Per-critic rulings (with the image), surviving items, an ADVISORY Huang-VI letter and
    the rationale that becomes the row's `rationale`. `scale_class_final` reconciles the
    advocate's and the geometry critic's scale_class."""
    model_config = _FORBID
    id: str
    persona: Literal["arbitrator"]
    rulings: list[Ruling]
    surviving_items: list[int] = Field(default_factory=list)
    letter_llm: Literal["A", "B", "C", "D"]
    scale_class_final: str = ""
    needs_human: bool = False
    rationale: str = ""

    @field_validator("rulings")
    @classmethod
    def _one_per_critic(cls, v):
        ps = [r.persona for r in v]
        dup = sorted({p for p in ps if ps.count(p) > 1})
        if dup:
            raise ValueError(f"one ruling per critic, duplicated: {dup}")
        return v


# ------------------------------------------------------------------ incumbent (arm a0)
class IncumbentVerdict(BaseModel):
    """The incumbent verifier's line (`verify_workflow.js` SCHEMA): id, persona, verdict,
    alternative, notes — aggregated by `aggregate_v2.passcount_incumbent`."""
    model_config = _FORBID
    id: str
    persona: Literal["artifact", "morphology", "geometry"]
    verdict: Literal["pass", "fail", "uncertain"]
    alternative: str = ""
    notes: str = ""


SCHEMA_FOR_ROLE = {"advocate": AdvocateRecord, "artifact": CriticRecord, "geometry": CriticRecord,
                   "morphology": CriticRecord, "arbitrator": ArbitratorRecord,
                   "incumbent": IncumbentVerdict}


# ------------------------------------------------------------------ the python-side result
@dataclass
class PanelResult:
    """One candidate through the panel. `critics` holds only the roles that were CALLED
    (None = called, failed to parse); `parse_failures` lists every failed role. S / S_arb
    are NaN and `letter` None whenever ANY called role failed, the arbitrator included (the
    pre-registered policy: excluded from recall/FPR, counted in the parse-failure rate)."""
    advocate: Optional[AdvocateRecord]
    critics: dict[str, Optional[CriticRecord]]
    arbitrator: Optional[ArbitratorRecord]
    S: float
    S_arb: float
    letter: Optional[str]
    letter_source: str
    a: dict                                   # role -> coverage fraction entering S
    parse_failures: list[str]
    cost_usd: float
    calls: int
    system_sha16s: dict                       # role -> sha16 of that role's system prompt
    r: dict = field(default_factory=dict)     # role -> refutation_strength (named critics)
    letter_arb: Optional[str] = None          # letter on S_arb with the arbitrator's rulings
    thresholds: dict = field(default_factory=dict)
    raw: dict = field(default_factory=dict)   # role -> raw model text (traces hold the full call)
    meta: dict = field(default_factory=dict)

    @property
    def parse_ok(self) -> bool:
        return not self.parse_failures and self.advocate is not None

    @property
    def p_evidence(self) -> Optional[float]:
        return None if self.advocate is None else self.advocate.p_evidence


def alternative_final(advocate, critics: dict, arbitrator=None) -> Optional[str]:
    """The contaminant to report: among critics that enter the product (arbitrated when an
    arbitrator exists: upheld/partial only), the alternative with the largest r·a; None when
    no critic names one. Ties: artifact, geometry, morphology order."""
    terms = aggregate_v2.critic_terms(advocate, critics, arbitrator)
    best, best_ra = None, -1.0
    for role in CRITIC_ROLES:
        t = terms.get(role)
        if not t or not t["included"] or t["alternative"] is None:
            continue
        ra = t["r"] * t["a"]
        if ra > best_ra:
            best, best_ra = str(t["alternative"]), ra
    return best


def assemble(advocate: Optional[AdvocateRecord], critics: dict, arbitrator: Optional[ArbitratorRecord],
             thresholds: dict, *, parse_failures: Optional[list] = None, cost_usd: float = 0.0,
             calls: int = 0, system_sha16s: Optional[dict] = None, raw: Optional[dict] = None,
             meta: Optional[dict] = None, arbitrator_called: Optional[bool] = None) -> PanelResult:
    """Score one candidate. `critics` = {role: record|None} for the roles that were CALLED;
    `parse_failures` lists failed roles (derived from None entries when not given);
    `arbitrator_called` says whether an arbitrator call was made (default: inferred — it was
    if a record exists or "arbitrator" is in parse_failures). `thresholds` is an
    `aggregate_v2.resolve_thresholds` dict."""
    critics = dict(critics or {})
    fails = list(parse_failures or [])
    if advocate is None and "advocate" not in fails:
        fails.append("advocate")
    for role, rec in critics.items():
        if rec is None and role not in fails:
            fails.append(role)
    if arbitrator_called is None:
        arbitrator_called = arbitrator is not None or "arbitrator" in fails
    if arbitrator_called and arbitrator is None and "arbitrator" not in fails:
        fails.append("arbitrator")
    fails = sorted(set(fails), key=lambda x: ROLES.index(x) if x in ROLES else 99)

    nan = float("nan")
    if fails:       # any failed role (advocate, critic OR arbitrator) ⇒ the row is a parse failure
        S, S_arb = nan, nan
    else:
        S = aggregate_v2.score_S(advocate, critics)
        S_arb = aggregate_v2.score_S_arb(advocate, critics, arbitrator)
    letter, source = aggregate_v2.assign_letter(S, advocate, critics, thresholds)
    letter_arb, _ = aggregate_v2.assign_letter(S_arb, advocate, critics, thresholds, arbitrator=arbitrator)
    terms = aggregate_v2.critic_terms(advocate, critics) if advocate is not None else {}
    a = {role: (t["a"] if t["parsed"] else nan) for role, t in terms.items()}
    r = {role: (t["r"] if t["parsed"] else nan) for role, t in terms.items()}
    return PanelResult(advocate=advocate, critics=critics, arbitrator=arbitrator, S=S, S_arb=S_arb,
                       letter=letter, letter_source=source, a=a, parse_failures=fails,
                       cost_usd=float(cost_usd), calls=int(calls), system_sha16s=dict(system_sha16s or {}),
                       r=r, letter_arb=letter_arb, thresholds=dict(thresholds), raw=dict(raw or {}),
                       meta=dict(meta or {}))


# The v2 columns every row carries (beside the run_batch._row_dict base + manifest join cols)
ROW_V2_COLS = (
    "S", "S_arb", "p_evidence", "scale_class", "scale_class_final", "letter_llm", "letter_arb",
    "letter_source", "alternative_final", "n_items", "n_surviving", "needs_human",
    *(f"a_{r}" for r in CRITIC_ROLES), *(f"r_{r}" for r in CRITIC_ROLES),
    *(f"alt_{r}" for r in CRITIC_ROLES), *(f"no_opinion_{r}" for r in CRITIC_ROLES),
    *(f"ruling_{r}" for r in CRITIC_ROLES), "parse_fail_roles", "calls", "cost_usd",
    *(f"system_sha16_{r}" for r in ROLES),
)
ROW_BASE_COLS = ("name", "grade_truth", "catalog", "region", "p_meta", "parse_ok", "turns",
                 "error", "grade_pred", "p_lens", "confidence", "contaminant", "escalate",
                 "rationale", *(f"crit_{c}" for c in CRITERIA_V2))
ROW_COLS = ROW_BASE_COLS + ROW_V2_COLS


def _nan_none(x):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else x


def to_row(result: PanelResult, cand: dict) -> dict:
    """One predictions row. The `run_batch._row_dict` keys first (p_lens = S, grade_pred =
    letter, confidence = p_evidence, contaminant = alternative_final, rationale = the
    arbitrator's or else the advocate's notes, escalate = needs_human), then the v2 columns.
    Every column is present on every row (None where a role was not called) so parquet
    flushes share one schema."""
    adv, arb = result.advocate, result.arbitrator
    rulings = {ru.persona: ru.ruling for ru in (arb.rulings if arb is not None else [])}
    row: dict[str, Any] = {
        "name": cand["name"], "grade_truth": cand.get("grade"), "catalog": cand.get("catalog"),
        "region": cand.get("region"), "p_meta": cand.get("p_meta"), "parse_ok": result.parse_ok,
        "turns": result.calls,
        "error": ("parse_fail:" + "+".join(result.parse_failures)) if result.parse_failures else None,
        "grade_pred": result.letter, "p_lens": result.S,
        "confidence": result.p_evidence,
        "contaminant": alternative_final(adv, result.critics, arb) if adv is not None else None,
        "escalate": bool(arb.needs_human) if arb is not None else False,
        "rationale": (arb.rationale if arb is not None and arb.rationale else
                      (adv.notes if adv is not None else "")),
    }
    crit = adv.criteria.model_dump() if adv is not None else {}
    row.update({f"crit_{c}": crit.get(c) for c in CRITERIA_V2})
    # the arbitrator reconciles scale_class; without one (or with an empty answer) the
    # advocate's report stands
    scale_final = adv.scale_class if adv is not None else None
    if arb is not None and arb.scale_class_final:
        scale_final = arb.scale_class_final
    row.update({
        "S": result.S, "S_arb": result.S_arb, "p_evidence": result.p_evidence,
        "scale_class": adv.scale_class if adv is not None else None,
        "scale_class_final": scale_final,
        "letter_llm": arb.letter_llm if arb is not None else None,
        "letter_arb": result.letter_arb, "letter_source": result.letter_source,
        "alternative_final": row["contaminant"],
        "n_items": len(adv.items) if adv is not None else None,
        "n_surviving": len(arb.surviving_items) if arb is not None else None,
        "needs_human": row["escalate"],
    })
    for role in CRITIC_ROLES:
        c = result.critics.get(role)
        called = role in result.critics
        row[f"a_{role}"] = _nan_none(result.a.get(role)) if called else None
        row[f"r_{role}"] = _nan_none(result.r.get(role)) if called else None
        row[f"alt_{role}"] = c.alternative if c is not None else None
        row[f"no_opinion_{role}"] = bool(c.no_opinion) if c is not None else None
        row[f"ruling_{role}"] = rulings.get(role)
    row.update({"parse_fail_roles": "+".join(result.parse_failures), "calls": result.calls,
                "cost_usd": result.cost_usd})
    row.update({f"system_sha16_{r}": result.system_sha16s.get(r) for r in ROLES})
    assert set(row) == set(ROW_COLS), sorted(set(row) ^ set(ROW_COLS))
    return {k: row[k] for k in ROW_COLS}          # one column order for every row
