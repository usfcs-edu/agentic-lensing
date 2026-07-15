"""Physicality domains for profile parameters, and construction-time validation.

Layer contract (misuse_register, physicality layer)
---------------------------------------------------
Every registered profile declares, next to ``_params``:

  - ``_domains``: per-parameter :class:`Domain` with a **hard** interval whose
    bounds are CODE-DERIVED — each rationale cites the kernel line (clip, mask,
    log/sqrt/power domain, division) it is derived from, and the behaviors were
    verified numerically (2026-07-10 kernel-pathology run; pinned by
    ``tests/validation/test_redteam_physicality.py``). Outside the hard interval
    the kernel does NOT compute the profile it claims: it NaNs, silently clips
    to a different model, or silently renders zero/garbage.
  - ``_joint_constraints``: :class:`JointConstraint` entries for validity
    regions no per-parameter box can express (e.g. ``e1**2 + e2**2 < 1``).

**Soft (plausibility) bands** are population/context-dependent judgments —
human-curated, NOT code-derived — so they are set sparingly. A first curated
set is registered: EPL ``gamma >= 1.1`` (``Domain.soft``), and two joint bands,
mass-profile axis ratio ``q >= 0.2`` and external-shear magnitude ``<= 0.2``
(``JointConstraint(severity="soft")``). Everything outside a soft band is
physically valid but atypical, so it **warns, never raises** — a plausibility
prompt, not a correctness guard. Most parameters still carry no soft band
(``Domain.soft is None``); adding one is a domain-knowledge decision.

Validation runs at ``LensModel`` construction — the last point where fixed
values are concrete (under jit every argument is a tracer, so kernel-level
guards are structurally impossible; see misuse_register, jit-blindness note):

  - **fixed value violates a hard bound → raise** (the model constructed would
    not be the model asked for);
  - **prior places mass outside a hard bound → warn** with the estimated mass
    (a finite-but-suspect configuration: the sampler will visit regions where
    the kernel computes a different model);
  - **fixed value / prior mass outside a soft band → soft warning** (valid but
    atypical; threshold ``SOFT_EPS_MASS``, a plausibility judgment, not the
    correctness-derived ``EPS_MASS``);
  - profiles not yet audited produce an *info* finding in the report only
    (ratcheted by the test suite's pending-audit list, not silently skipped).

The SAME checks run against posterior samples via
:func:`validate_posterior_samples` — given the structured (batched) params
pytree, it reports the *fraction* of the posterior landing outside each hard
domain / soft band / joint region. Unlike construction it never raises: a
completed fit is diagnosed, not rejected.

Threshold and probe-size derivation
-----------------------------------
A typical inference run makes ~1e6 posterior evaluations (this project's usual
production scale — an assumption, not a measured constant; revisit EPS_MASS if
run sizes grow), so prior mass ``eps`` inside an invalid region is visited
~``eps * 1e6`` times per run; we warn when the expected visit count reaches
order one: ``EPS_MASS = 1e-6``.
Scalar-prior mass is computed exactly via the distribution's ``cdf`` when it
has one. Joint constraints (and cdf-less priors) use a fixed-seed Monte-Carlo
probe: a zero-violation probe of size K certifies mass <~ 3/K at 95%
confidence (rule of three), so ``PROBE_DRAWS = 3 / EPS_MASS = 3e6`` matches
the probe's resolution to the warning threshold. Constraint evaluation is
parameter-space arithmetic (no lensing kernels), so 3e6 draws cost
milliseconds. The seed is FIXED: a construction-time warning must be
deterministic, not flicker between runs.
"""

import dataclasses
import math
import warnings
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

EPS_MASS = 1e-6
# Soft (plausibility) mass threshold. Unlike EPS_MASS — which is a CORRECTNESS
# floor ("~1 in 1e6 evaluations lands where the kernel computes a different
# model") — a soft band is a plausibility judgment, so one-in-a-million is far
# too strict: we warn only when a MEANINGFUL fraction of the prior/posterior
# sits in the valid-but-atypical region. 5% is a curation default, not derived.
SOFT_EPS_MASS = 0.05
PROBE_DRAWS = 3_000_000
_PROBE_SEED = 20260710


# --------------------------------------------------------------------------------
# Declaration types (profiles import these)
# --------------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class Domain:
    """Validity interval for one parameter.

    ``(lo, hi)`` with per-end openness is the HARD interval (code-derived; see
    module docstring). ``soft`` is the plausibility band — human-curated,
    deliberately ``None`` in this draft. ``rationale`` must cite the kernel
    line(s) the hard bounds derive from.
    """

    lo: float = -math.inf
    hi: float = math.inf
    lo_open: bool = False
    hi_open: bool = False
    soft: Optional[Tuple[float, float]] = None
    rationale: str = ""

    @property
    def unbounded(self) -> bool:
        return math.isinf(self.lo) and math.isinf(self.hi)

    def violates(self, value):
        """Elementwise: True where ``value`` is outside the hard interval.
        Works on scalars and numpy arrays. Non-finite values count as outside."""
        v = np.asarray(value)
        bad = ~np.isfinite(v)
        bad |= (v <= self.lo) if self.lo_open else (v < self.lo)
        bad |= (v >= self.hi) if self.hi_open else (v > self.hi)
        return bad

    def outside_soft(self, value):
        """Elementwise: True where ``value`` is outside the soft (plausibility)
        band ``soft=(lo, hi)``. All-False when no soft band is set. The band is
        treated as closed (its endpoints are plausible): only strictly-below-lo
        or strictly-above-hi is atypical. Endpoints at +/-inf don't constrain
        that side. Non-finite values are the hard check's concern, not this one
        (NaN compares False here), so soft never double-reports a hard failure."""
        v = np.asarray(value)
        if self.soft is None:
            return np.zeros(np.shape(v), dtype=bool)
        lo, hi = self.soft
        bad = np.zeros(np.shape(v), dtype=bool)
        if math.isfinite(lo):
            bad = bad | (v < lo)
        if math.isfinite(hi):
            bad = bad | (v > hi)
        return bad

    def interval_str(self) -> str:
        return (("(" if self.lo_open else "[") + f"{self.lo:g}, {self.hi:g}"
                + (")" if self.hi_open else "]"))

    def soft_str(self) -> str:
        lo, hi = self.soft
        return f"[{lo:g}, {hi:g}]"


@dataclasses.dataclass(frozen=True)
class JointConstraint:
    """A validity region over several parameters that no per-parameter box can
    express. ``ok(**values)`` must accept scalars and/or (K,)-arrays for each
    name in ``params`` (numpy broadcasting) and return True where SATISFIED.

    ``severity`` is ``"hard"`` (code-derived validity: a fixed violation raises,
    prior mass warns) or ``"soft"`` (human-curated plausibility: only ever a
    soft warning, never a raise)."""

    name: str
    params: Tuple[str, ...]
    ok: Callable[..., Any]
    rationale: str = ""
    severity: str = "hard"


def ellipticity_constraint(limit: float, *, exclude_circular: bool = False,
                           rationale: str = "") -> JointConstraint:
    """The recurring kernel constraint on (e1, e2): ``e1**2 + e2**2 < limit**2``,
    optionally excluding the exact circular point (SIE NaNs there)."""
    lim2 = float(limit) ** 2

    def ok(e1, e2):
        m2 = np.asarray(e1) ** 2 + np.asarray(e2) ** 2
        good = m2 < lim2
        if exclude_circular:
            good = good & (m2 > 0.0)
        return good

    return JointConstraint(name=f"e1^2+e2^2 < {limit}^2"
                                + (" (and > 0)" if exclude_circular else ""),
                           params=("e1", "e2"), ok=ok, rationale=rationale)


def axis_ratio_constraint(q_min: float, *, rationale: str = "") -> JointConstraint:
    """SOFT plausibility band on axis ratio: ``q >= q_min``. Across every
    gigalens mass profile the (e1, e2) -> q map is the same, ``c =
    sqrt(e1^2+e2^2)``, ``q = (1-c)/(1+c)`` (verified uniform 2026-07-13), so
    ``q >= q_min`` is exactly ``c <= (1-q_min)/(1+q_min)``. A very flattened mass
    profile (small q) is physically valid but rare, hence a soft warning."""
    c_max = (1.0 - q_min) / (1.0 + q_min)

    def ok(e1, e2):
        c = np.sqrt(np.asarray(e1) ** 2 + np.asarray(e2) ** 2)
        return c <= c_max

    return JointConstraint(name=f"axis ratio q >= {q_min:g}",
                           params=("e1", "e2"), ok=ok, rationale=rationale,
                           severity="soft")


def shear_magnitude_constraint(mag_max: float, *,
                               rationale: str = "") -> JointConstraint:
    """SOFT plausibility band on external-shear magnitude:
    ``sqrt(gamma1^2 + gamma2^2) <= mag_max``. The shear kernel is exactly linear
    for any finite shear (no hard bound is derivable), but |gamma| beyond a few
    tenths is astrophysically implausible, hence a soft warning."""

    def ok(gamma1, gamma2):
        mag = np.sqrt(np.asarray(gamma1) ** 2 + np.asarray(gamma2) ** 2)
        return mag <= mag_max

    return JointConstraint(name=f"|shear| <= {mag_max:g}",
                           params=("gamma1", "gamma2"), ok=ok,
                           rationale=rationale, severity="soft")


# --------------------------------------------------------------------------------
# Findings and report
# --------------------------------------------------------------------------------
@dataclasses.dataclass
class Finding:
    severity: str          # "error" | "warning" | "soft" | "info"
    profile: str           # profile name (and path when available)
    param: str             # parameter name or joint-constraint name
    kind: str              # "fixed-value" | "prior-mass" | "soft-prior-mass" | ...
    message: str
    mass: Optional[float] = None   # estimated prior mass in the invalid region
    method: str = ""               # "exact (cdf)" | "probe (K=...)"


@dataclasses.dataclass
class PhysicalityReport:
    """Structured result of a physicality validation pass. Attached to the
    ``LensModel`` (``model.physicality_report``) so downstream consumers (e.g.
    a model card) can persist what was checked, what fired, and — critically —
    the resolution floor of checks that came back clean."""

    findings: List[Finding] = dataclasses.field(default_factory=list)
    checks_run: List[str] = dataclasses.field(default_factory=list)

    def errors(self) -> List[Finding]:
        return [f for f in self.findings if f.severity == "error"]

    def warnings_(self) -> List[Finding]:
        return [f for f in self.findings if f.severity == "warning"]

    def soft_warnings_(self) -> List[Finding]:
        """Plausibility findings (soft bands). Never raise; warned separately
        from hard warnings so the two voices don't blur (see ``apply_report``)."""
        return [f for f in self.findings if f.severity == "soft"]

    def summary(self) -> str:
        n_e, n_w = len(self.errors()), len(self.warnings_())
        n_s = len(self.soft_warnings_())
        lines = [f"PhysicalityReport: {n_e} error(s), {n_w} warning(s), "
                 f"{n_s} soft/plausibility warning(s), "
                 f"{len(self.checks_run)} check(s) run clean or recorded"]
        lines += [f"  [{f.severity}] {f.profile}.{f.param}: {f.message}"
                  for f in self.findings]
        lines += [f"  [ok] {c}" for c in self.checks_run]
        return "\n".join(lines)


# --------------------------------------------------------------------------------
# Engine
# --------------------------------------------------------------------------------
def _mass_outside_cdf(dist, dom: Domain) -> Optional[float]:
    """Exact prior mass outside the hard interval via the distribution's cdf;
    None when the distribution has no usable cdf (falls back to the probe).
    Open vs closed endpoints coincide for continuous distributions."""
    try:
        below = float(dist.cdf(dom.lo)) if math.isfinite(dom.lo) else 0.0
        above = 1.0 - float(dist.cdf(dom.hi)) if math.isfinite(dom.hi) else 0.0
    except Exception:  # tfd raises NotImplementedError subclasses vary by dist
        return None
    if not (math.isfinite(below) and math.isfinite(above)):
        return None
    return below + above


def _mass_outside_soft_cdf(dist, dom: Domain) -> Optional[float]:
    """Exact prior mass outside the SOFT band via the cdf; None when the band
    is unset or the distribution has no usable cdf (caller falls back to the
    probe). The soft band is closed, but open vs closed endpoints coincide for
    continuous distributions."""
    if dom.soft is None:
        return 0.0
    lo, hi = dom.soft
    try:
        below = float(dist.cdf(lo)) if math.isfinite(lo) else 0.0
        above = 1.0 - float(dist.cdf(hi)) if math.isfinite(hi) else 0.0
    except Exception:
        return None
    if not (math.isfinite(below) and math.isfinite(above)):
        return None
    return below + above


class _DrawCache:
    """Fixed-seed draws per free PARAMETER identity. Sharing is HANDLE identity,
    not dist identity (grader, fifth-increment rd-1): the same shared() handle at
    several sites must reuse ONE draw set (breaking that would destroy exactly
    the correlation a joint constraint depends on), while DISTINCT handles that
    happen to wrap the same dist object are independent free params and must get
    INDEPENDENT draws (keying those by id(dist) silently correlated them).
    Callers pass ``key=id(handle)`` where a handle exists; bare distributions
    default to ``id(dist)`` (bare reuse across sites raises at derivation, §3.3,
    so a bare id never spans two free params)."""

    def __init__(self, n_draws: int, seed: int):
        self.n = int(n_draws)
        self._seed = seed
        self._cache: Dict[int, np.ndarray] = {}
        self._count = 0

    def draws(self, dist, key: Optional[int] = None) -> np.ndarray:
        if key is None:
            key = id(dist)
        if key not in self._cache:
            import jax  # deferred: keep module import light

            self._count += 1
            k = jax.random.fold_in(jax.random.PRNGKey(self._seed), self._count)
            self._cache[key] = np.asarray(dist.sample(self.n, seed=k))
        return self._cache[key]


def _resolve_param_values(comp, cache: _DrawCache):
    """Map each parameter name of a Component to a checkable value:
    float (fixed) | (K,)-array (scalar prior / shared / grouped-prior column).
    Mirrors the scene layer's classification (str vs tuple keys, SharedParam)."""
    # Local import: physicality is imported by profile modules, so importing the
    # scene here at module scope would be circular.
    from gigalens.jax.scene import SharedParam, CoupledSlot, _is_number

    values: Dict[str, Any] = {}
    dists: Dict[str, Any] = {}   # param -> the scalar dist it came from (for cdf)
    keys: Dict[str, int] = {}    # param -> draw-cache identity (handle id if shared)
    for key, val in comp.priors.items():
        names = key if isinstance(key, tuple) else (key,)
        # Identity BEFORE unwrapping (grader, fifth-increment rd-1): the same
        # shared() handle at several sites reuses one draw set; DISTINCT handles
        # wrapping the same dist object are independent free params and get
        # independent draws.
        draw_key = id(val)
        if isinstance(val, SharedParam):
            val = val.dist
        if isinstance(val, CoupledSlot):
            # A cross-component coupling member: probe its MARGINAL by drawing the group's
            # joint dist once (one draw set per group, keyed by group identity) and taking
            # its event column. Like a tuple-key group member, it has no scalar cdf.
            grp = val.group
            draws = cache.draws(grp.dist, key=id(grp))   # (K, k)
            values[names[0]] = draws[..., val.idx]
            dists[names[0]] = None
            continue
        if _is_number(val):
            values[names[0]] = float(val)
        elif isinstance(key, tuple):
            draws = cache.draws(val, key=draw_key)  # (K, len(names))
            for j, n in enumerate(names):
                values[n] = draws[..., j]
                dists[n] = None                    # joint: no scalar cdf
        else:
            values[key] = None                     # sampled lazily (cdf first)
            dists[key] = val
            keys[key] = draw_key
    return values, dists, keys


def validate_component(comp, cache: _DrawCache, *, label: str = "",
                       report: PhysicalityReport) -> None:
    """Validate one Component (profile + priors) against its registered
    domains and joint constraints, appending findings to ``report``."""
    profile = comp.profile
    pname = f"{getattr(profile, 'name', type(profile).__name__)}{label}"
    # Own-class lookup, deliberately NOT inherited: a subclass may add parameters
    # and kernel behavior its parent's audit never covered (e.g. CoreSersic vs
    # Sersic), so each class must declare its own _domains (assigning a shared
    # module-level dict in the class body, as EPL_sean does, is fine).
    domains = vars(type(profile)).get("_domains")
    if domains is None:
        report.findings.append(Finding(
            "info", pname, "*", "unregistered-profile",
            "no physicality domains registered for this profile "
            "(pending audit; see PENDING_PHYSICALITY_AUDIT)"))
        return

    fallback = getattr(profile, "_domain_fallback", None)
    values, dists, keys = _resolve_param_values(comp, cache)

    # --- per-parameter boxes -----------------------------------------------------
    for param in comp_params_covered(comp):
        dom = domains.get(param, fallback)
        if dom is None:
            report.findings.append(Finding(
                "info", pname, param, "unregistered-param",
                "parameter has no registered domain (pending audit)"))
            continue
        v = values.get(param)
        if isinstance(v, float):                       # fixed value
            if bool(dom.violates(v)):
                report.findings.append(_fixed_hard_finding(pname, param, dom, v))
            elif dom.soft is not None and bool(dom.outside_soft(v)):
                report.findings.append(_fixed_soft_finding(pname, param, dom, v))
            continue
        if dom.unbounded and dom.soft is None:
            continue
        dist = dists.get(param)
        # --- hard bound (skipped for jointly-bounded / unbounded params) ---------
        if not dom.unbounded:
            if dist is not None:                       # scalar prior: cdf first
                mass = _mass_outside_cdf(dist, dom)
                method = "exact (cdf)"
                if mass is None:
                    draws = cache.draws(dist, key=keys[param])
                    values[param] = draws              # reuse for joint checks
                    mass = float(np.mean(dom.violates(draws)))
                    method = f"probe (K={cache.n})"
            else:                                      # grouped-prior column
                mass = float(np.mean(dom.violates(values[param])))
                method = f"probe (K={cache.n})"
            _record_mass(report, pname, param, "prior-mass", mass, method,
                         dom.rationale,
                         f"prior mass outside hard domain {dom.interval_str()}")
        # --- soft (plausibility) band --------------------------------------------
        if dom.soft is not None:
            if dist is not None:
                smass = _mass_outside_soft_cdf(dist, dom)
                smethod = "exact (cdf)"
                if smass is None:
                    draws = cache.draws(dist, key=keys[param])
                    values[param] = draws
                    smass = float(np.mean(dom.outside_soft(draws)))
                    smethod = f"probe (K={cache.n})"
            else:
                smass = float(np.mean(dom.outside_soft(values[param])))
                smethod = f"probe (K={cache.n})"
            _record_mass(report, pname, param, "soft-prior-mass", smass, smethod,
                         dom.rationale,
                         f"prior mass outside plausible band {dom.soft_str()}",
                         severity="soft", threshold=SOFT_EPS_MASS,
                         advice=(f" — {dom.rationale} Physically valid but "
                                 f"atypical; check this is intended."))

    # --- joint constraints ---------------------------------------------------------
    for jc in getattr(profile, "_joint_constraints", ()):
        vals = {}
        for n in jc.params:
            v = values.get(n)
            if v is None:                              # scalar prior not yet drawn
                v = cache.draws(dists[n], key=keys[n])
                values[n] = v
            vals[n] = v
        good = np.asarray(jc.ok(**vals))
        _emit_joint(report, pname, jc, vals, good, method=f"probe (K={cache.n})")


# --------------------------------------------------------------------------------
# Finding builders and emitters — shared by the prior and posterior paths so both
# speak in identical message text and kinds.
# --------------------------------------------------------------------------------
def _fixed_hard_finding(pname, param, dom: Domain, v) -> Finding:
    return Finding(
        "error", pname, param, "fixed-value",
        f"fixed value {v:g} outside hard domain {dom.interval_str()} "
        f"— {dom.rationale}")


def _fixed_soft_finding(pname, param, dom: Domain, v) -> Finding:
    return Finding(
        "soft", pname, param, "soft-fixed-value",
        f"fixed value {v:g} outside plausible band {dom.soft_str()} (hard domain "
        f"{dom.interval_str()} satisfied) — {dom.rationale} Physically valid but "
        f"atypical; check this is intended.")


def _emit_joint(report, pname, jc: JointConstraint, vals, good, *, method,
                posterior: bool = False) -> None:
    """Emit findings for one joint constraint given the SATISFIED mask ``good``.
    Shared by prior (probe) and posterior (sample) paths; ``jc.severity`` selects
    hard (raise/warn) vs soft (plausibility warning only)."""
    soft = (jc.severity == "soft")
    what = "posterior" if posterior else "prior"
    if good.ndim == 0:                                 # all inputs fixed/constant
        if not bool(good):
            fixed = ", ".join(f"{n}={float(np.asarray(vals[n])):g}" for n in jc.params)
            if soft:
                report.findings.append(Finding(
                    "soft", pname, jc.name, "soft-fixed-value",
                    f"fixed values ({fixed}) outside plausible region {jc.name} "
                    f"— {jc.rationale} Physically valid but atypical; check this "
                    f"is intended."))
            else:
                report.findings.append(Finding(
                    "error", pname, jc.name, "fixed-value",
                    f"fixed values ({fixed}) violate {jc.name} — {jc.rationale}"))
        else:
            report.checks_run.append(f"{pname}: {jc.name} satisfied by fixed values")
        return
    mass = float(np.mean(~good))
    if soft:
        _record_mass(report, pname, jc.name, f"soft-joint-{what}-mass", mass, method,
                     jc.rationale, f"{what} mass outside plausible region {jc.name}",
                     severity="soft", threshold=SOFT_EPS_MASS,
                     advice=(f" — {jc.rationale} Physically valid but atypical; "
                             f"check this is intended."))
    elif posterior:
        _record_mass(report, pname, jc.name, "joint-posterior-mass", mass, method,
                     jc.rationale, f"posterior mass violating {jc.name}",
                     advice=(f" — {jc.rationale} the completed fit places this "
                             f"fraction on physically invalid parameters."))
    else:
        _record_mass(report, pname, jc.name, "joint-prior-mass", mass, method,
                     jc.rationale, f"prior mass violating {jc.name}")


def _record_mass(report, pname, param, kind, mass, method, rationale, what, *,
                 severity: str = "warning", threshold: float = EPS_MASS,
                 advice: Optional[str] = None):
    """Append a warning/soft finding when ``mass`` exceeds ``threshold``, else
    record a clean check with its resolution floor. ``advice`` overrides the
    trailing clause; the default reproduces the original hard-prior message."""
    if mass > threshold:
        if advice is None:
            advice = (f" (~{mass:.0e} of a typical run's ~1e6 evaluations land where "
                      f"the kernel computes a different model) — {rationale} Consider "
                      f"a bounded/truncated prior.")
        report.findings.append(Finding(
            severity, pname, param, kind,
            f"{what}: estimated mass {mass:.3g} ({method}) exceeds {threshold:g}"
            f"{advice}", mass=mass, method=method))
    else:
        if "cdf" in method:
            tail = "exact"
        elif mass == 0.0:
            tail = (f"zero violations: certifies mass < "
                    f"{3.0 / report_probe_n(method):.1g} at 95% (rule of three)")
        else:
            se = math.sqrt(mass * (1.0 - mass) / report_probe_n(method))
            tail = f"binomial se {se:.1g}"
        report.checks_run.append(
            f"{pname}.{param}: {what} = {mass:.3g} ({method}; {tail})")


def report_probe_n(method: str) -> float:
    if "K=" in method:
        return float(method.split("K=")[1].rstrip(")"))
    if "N=" in method:                                 # posterior sample count
        return float(method.split("N=")[1].rstrip(")"))
    return float("inf")


def comp_params_covered(comp) -> List[str]:
    out: List[str] = []
    for key in comp.priors:
        out.extend(key if isinstance(key, tuple) else (key,))
    return out


# --------------------------------------------------------------------------------
# Sampled-redshift geometry (the C1 guard's prior analogue)
# --------------------------------------------------------------------------------
_REDSHIFT_DOMAIN = Domain(lo=0.0, lo_open=True, rationale=(
    "redshift must be finite and > 0 (misuse_register C1): the cosmology layer's "
    "distance integrals and deflection_ratio are meaningless at z <= 0. The "
    "concrete-redshift guard (scene._validate_concrete_redshifts) RAISES there; "
    "this is its sampled-prior analogue."))

_ORDERING_RATIONALE = (
    "planes are documented observer->source (non-decreasing redshift); prior mass "
    "on the reversed ordering silently yields a NEGATIVE derived deflection ratio "
    "/ distance under jit (misuse_register C1 — the concrete guard raises on "
    "fixed values; this is its sampled-prior analogue).")


def _redshift_value(plane):
    """Classify a plane's redshift: ("none", None, None) | ("fixed", float, None)
    | ("dist", tfd dist, identity). ``identity`` is the object whose IDENTITY
    defines the free parameter — the shared() HANDLE when there is one (distinct
    handles wrapping one dist object are independent free params; grader rd-1),
    else the bare dist itself (bare reuse across sites raises at derivation)."""
    from gigalens.jax.scene import SharedParam, _is_number

    z = getattr(plane, "redshift", None)
    if z is None:
        return "none", None, None
    if isinstance(z, SharedParam):
        return "dist", z.dist, z
    if _is_number(z):
        return "fixed", float(z), None
    return "dist", z, z


def _one_sided_cdf(dist, c: float, *, upper: bool) -> Optional[float]:
    """P(X > c) (upper) or P(X < c) via the cdf; None -> caller probes."""
    try:
        v = float(dist.cdf(c))
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    return (1.0 - v) if upper else v


def validate_redshift_geometry(planes: Sequence, cache: _DrawCache, *,
                               report: PhysicalityReport) -> None:
    """Prior-mass checks for SAMPLED plane redshifts (cosmology-present models).

    ``scene._validate_concrete_redshifts`` raises on concrete violations but by
    design skips free (sampled) redshifts — a redshift PRIOR putting mass at
    z <= 0 or below the preceding plane's redshift previously passed silently
    (fourth-increment residual). Mirrors that guard's exact constraint set:

    - per plane: prior mass at z <= 0 (exact via cdf where available, else the
      fixed-seed probe; non-finite draws count as outside);
    - per ADJACENT plane pair: prior mass violating non-decreasing ordering,
      P(z_i < z_{i-1}) — a sequence is non-decreasing iff every adjacent pair
      is, so pairwise adjacency is complete for the ordering constraint. Exact
      one-sided cdf when the other side is fixed; the probe when both are
      sampled (independent free params => independent draw sets). The SAME
      shared() handle at both sites is structurally ordered (one free param,
      identical draws) and recorded as a clean check; DISTINCT handles — even
      wrapping the same dist object — are independent free params (each has
      its own prior entry) and are probed with independent draws (grader rd-1:
      the structural shortcut is HANDLE identity, never dist identity).
      Fixed-fixed pairs are the concrete guard's job (raise).

    Threshold: the same EPS_MASS as profile priors — mass eps in the reversed-
    ordering region is visited ~eps * 1e6 times per typical run, each computing
    a silently sign-flipped deflection.
    """
    resolved = [_redshift_value(p) for p in planes]

    # -- per-plane domain: z > 0, finite ------------------------------------------
    for i, (kind, val, ident) in enumerate(resolved):
        if kind != "dist":
            continue  # "none": ratio-mode model; "fixed": concrete guard raised already
        mass = _mass_outside_cdf(val, _REDSHIFT_DOMAIN)
        method = "exact (cdf)"
        if mass is None:
            draws = cache.draws(val, key=id(ident))
            mass = float(np.mean(_REDSHIFT_DOMAIN.violates(draws)))
            method = f"probe (K={cache.n})"
        _record_mass(report, "geometry", f"planes[{i}].redshift", "prior-mass",
                     mass, method, _REDSHIFT_DOMAIN.rationale,
                     f"prior mass outside hard domain "
                     f"{_REDSHIFT_DOMAIN.interval_str()}")

    # -- adjacent-pair ordering ----------------------------------------------------
    # NaN draws compare False in every ordering probe below (counted as NON-
    # violating); they are not lost — the per-plane domain check above counts
    # non-finite draws as outside the hard domain.
    for i in range(1, len(resolved)):
        (k0, v0, i0), (k1, v1, i1) = resolved[i - 1], resolved[i]
        if k0 == "none" or k1 == "none":
            continue  # ratio-mode model (no redshifts to order)
        if k0 == "fixed" and k1 == "fixed":
            continue  # concrete guard raises on violation
        name = f"planes[{i-1}].redshift <= planes[{i}].redshift"
        # Structural shortcut on HANDLE identity ONLY (grader rd-1): the same
        # shared() handle is one free param -> one draw set -> ordering cannot be
        # violated. Distinct handles wrapping the same dist object are TWO free
        # params (two prior entries) and fall through to the independent probe.
        if k0 == "dist" and k1 == "dist" and i0 is i1:
            report.checks_run.append(
                f"geometry: {name} — the same shared redshift handle at both "
                "planes (one free parameter); ordering violation structurally "
                "impossible (identical draws)")
            continue
        if k0 == "fixed":                              # P(z_i < c)
            mass = _one_sided_cdf(v1, v0, upper=False)
            method = "exact (cdf)"
            if mass is None:
                mass = float(np.mean(cache.draws(v1, key=id(i1)) < v0))
                method = f"probe (K={cache.n})"
        elif k1 == "fixed":                            # P(z_{i-1} > c)
            mass = _one_sided_cdf(v0, v1, upper=True)
            method = "exact (cdf)"
            if mass is None:
                mass = float(np.mean(cache.draws(v0, key=id(i0)) > v1))
                method = f"probe (K={cache.n})"
        else:                                          # both sampled, independent
            mass = float(np.mean(
                cache.draws(v1, key=id(i1)) < cache.draws(v0, key=id(i0))))
            method = f"probe (K={cache.n})"
        _record_mass(report, "geometry", name, "joint-prior-mass", mass, method,
                     _ORDERING_RATIONALE, f"prior mass violating {name}")


def validate_planes(planes: Sequence, *, draws: int = PROBE_DRAWS,
                    seed: int = _PROBE_SEED) -> PhysicalityReport:
    """Validate every mass/light Component across the planes, plus sampled-
    redshift geometry priors (``validate_redshift_geometry``). The cosmology
    Component is NOT handled here — its physicality flag is the C3 guard
    (``CosmoBase._flag_unphysical_densities``). CONCRETE plane redshifts are
    the C1 guard (raises at the scene layer); SAMPLED redshift priors are
    checked here for mass at z <= 0 and mass violating plane ordering.
    Returns the report; the caller decides raise/warn."""
    report = PhysicalityReport()
    cache = _DrawCache(draws, seed)
    validate_redshift_geometry(planes, cache, report=report)
    for i, plane in enumerate(planes):
        for kind, comps in (("mass", plane.mass), ("light", plane.light)):
            for j, comp in enumerate(comps):
                validate_component(comp, cache,
                                   label=f"[plane {i} {kind} {j}]", report=report)
    return report


# --------------------------------------------------------------------------------
# Posterior-sample checks — the SAME domains / joints / geometry, run against
# realized draws instead of the prior. Reports the FRACTION of the posterior
# outside each region; never raises (a completed fit is diagnosed, not rejected).
# The resolution floor is the sample count N (via ``report_probe_n``'s N= branch),
# far coarser than the prior probe's 3e6 — a fraction outside a HARD domain means
# the fit itself visited physically invalid parameters.
# --------------------------------------------------------------------------------
def _n_samples(sample_vals: Dict[str, Any]) -> int:
    for v in sample_vals.values():
        a = np.asarray(v)
        if a.ndim > 0:
            return int(a.size)
    return 0


def _get_path(d, path):
    """Read ``d[path[0]][path[1]]...``; None if any step is missing/not indexable."""
    cur = d
    for k in path:
        try:
            cur = cur[k]
        except (KeyError, IndexError, TypeError):
            return None
    return cur


def validate_component_samples(comp, sample_vals: Dict[str, Any], *,
                               label: str = "", report: PhysicalityReport) -> None:
    """Posterior analogue of :func:`validate_component`. ``sample_vals`` maps each
    parameter name of the Component to a float (a constant param) or an array of
    posterior draws. Runs the same per-parameter hard/soft boxes and hard/soft
    joint constraints, recording the empirical fraction outside each region."""
    profile = comp.profile
    pname = f"{getattr(profile, 'name', type(profile).__name__)}{label}"
    domains = vars(type(profile)).get("_domains")
    if domains is None:
        report.findings.append(Finding(
            "info", pname, "*", "unregistered-profile",
            "no physicality domains registered for this profile "
            "(pending audit; see PENDING_PHYSICALITY_AUDIT)"))
        return
    fallback = getattr(profile, "_domain_fallback", None)
    n = _n_samples(sample_vals)
    method = f"samples (N={n})" if n else "samples"

    for param in comp_params_covered(comp):
        dom = domains.get(param, fallback)
        if dom is None:
            report.findings.append(Finding(
                "info", pname, param, "unregistered-param",
                "parameter has no registered domain (pending audit)"))
            continue
        if param not in sample_vals:
            continue                                   # not present in this pytree
        arr = np.asarray(sample_vals[param])
        if arr.ndim == 0:                              # constant param
            fv = float(arr)
            if bool(dom.violates(fv)):
                report.findings.append(_fixed_hard_finding(pname, param, dom, fv))
            elif dom.soft is not None and bool(dom.outside_soft(fv)):
                report.findings.append(_fixed_soft_finding(pname, param, dom, fv))
            continue
        if not dom.unbounded:
            hmass = float(np.mean(dom.violates(arr)))
            _record_mass(report, pname, param, "posterior-mass", hmass, method,
                         dom.rationale,
                         f"posterior mass outside hard domain {dom.interval_str()}",
                         advice=(f" — {dom.rationale} the completed fit places this "
                                 f"fraction on physically invalid parameters."))
        if dom.soft is not None:
            smass = float(np.mean(dom.outside_soft(arr)))
            _record_mass(report, pname, param, "soft-posterior-mass", smass, method,
                         dom.rationale,
                         f"posterior mass outside plausible band {dom.soft_str()}",
                         severity="soft", threshold=SOFT_EPS_MASS,
                         advice=(f" — {dom.rationale} physically valid but atypical "
                                 f"for the fit."))

    for jc in getattr(profile, "_joint_constraints", ()):
        if any(pn not in sample_vals for pn in jc.params):
            continue
        vals = {pn: np.asarray(sample_vals[pn]) for pn in jc.params}
        good = np.asarray(jc.ok(**vals))
        _emit_joint(report, pname, jc, vals, good, method=method, posterior=True)


def validate_redshift_geometry_samples(model, params, *,
                                       report: PhysicalityReport) -> None:
    """Posterior analogue of :func:`validate_redshift_geometry`: per-plane mass at
    z <= 0 and adjacent-pair ordering violations, from realized redshift draws."""
    if getattr(model, "cosmo", None) is None:
        return
    zs = [_get_path(params, ("planes", i, "geometry", "redshift"))
          for i in range(len(model.planes))]
    zs = [None if z is None else np.asarray(z) for z in zs]
    n = next((int(z.size) for z in zs if z is not None and z.ndim > 0), 0)
    method = f"samples (N={n})" if n else "samples"
    for i, z in enumerate(zs):
        if z is None:
            continue
        mass = float(np.mean(_REDSHIFT_DOMAIN.violates(z)))
        _record_mass(report, "geometry", f"planes[{i}].redshift", "posterior-mass",
                     mass, method, _REDSHIFT_DOMAIN.rationale,
                     f"posterior mass outside hard domain "
                     f"{_REDSHIFT_DOMAIN.interval_str()}",
                     advice=(f" — {_REDSHIFT_DOMAIN.rationale} the completed fit "
                             f"places this fraction on physically invalid redshifts."))
    for i in range(1, len(zs)):
        z0, z1 = zs[i - 1], zs[i]
        if z0 is None or z1 is None:
            continue
        name = f"planes[{i-1}].redshift <= planes[{i}].redshift"
        mass = float(np.mean(np.asarray(z1) < np.asarray(z0)))
        _record_mass(report, "geometry", name, "joint-posterior-mass", mass, method,
                     _ORDERING_RATIONALE, f"posterior mass violating {name}",
                     advice=(f" — {_ORDERING_RATIONALE} the completed fit places "
                             f"this fraction on a reversed plane ordering."))


def validate_posterior_samples(model, params, *,
                               report: Optional[PhysicalityReport] = None
                               ) -> PhysicalityReport:
    """Run the physicality checks against POSTERIOR SAMPLES.

    ``params`` is the structured params pytree (the §5 layout produced by
    ``model.to_params`` and consumed by the simulator) with each leaf holding the
    posterior draws for that parameter — shape ``(N,)`` or ``(chains, draws)``;
    constant parameters may stay scalar. Reports, per profile parameter / joint
    region / sampled redshift, the FRACTION of the posterior outside the hard
    domain, outside the soft (plausibility) band, or violating a joint
    constraint. Never raises — a completed fit is diagnosed, not rejected.
    Returns the report; the caller decides how to surface it (e.g. print
    ``report.summary()`` or iterate ``report.warnings_()`` /
    ``report.soft_warnings_()``)."""
    report = report if report is not None else PhysicalityReport()
    for i, plane in enumerate(model.planes):
        for kind, comps in (("mass", plane.mass), ("light", plane.light)):
            for j, comp in enumerate(comps):
                sub = _get_path(params, ("planes", i, kind, j))
                if sub is None:
                    continue
                validate_component_samples(comp, sub, report=report,
                                           label=f"[plane {i} {kind} {j}]")
    validate_redshift_geometry_samples(model, params, report=report)
    return report


def apply_report(report: PhysicalityReport, *, stacklevel: int = 3) -> None:
    """Enforce the raise-vs-flag policy: aggregate hard fixed-value violations
    into one ValueError; emit each hard prior-mass finding as a UserWarning; and
    emit each soft (plausibility) finding as a distinct, lower-key UserWarning
    (prefixed ``[plausibility]``) — soft findings never raise."""
    errs = report.errors()
    if errs:
        raise ValueError(
            "physically invalid fixed parameter(s) (misuse_register, physicality "
            "layer):\n" + "\n".join(
                f"  - {f.profile}.{f.param}: {f.message}" for f in errs))
    for f in report.warnings_():
        warnings.warn(f"{f.profile}.{f.param}: {f.message}",
                      UserWarning, stacklevel=stacklevel)
    for f in report.soft_warnings_():
        warnings.warn(f"[plausibility] {f.profile}.{f.param}: {f.message}",
                      UserWarning, stacklevel=stacklevel)


# Profiles that exist in gigalens.jax.profiles but have NOT yet had their hard
# domains derived from their kernels. The test suite ratchets this list: a new
# profile class must either register _domains or be added here explicitly
# (reviewed), never silently skipped. Keys are the module path relative to
# gigalens.jax.profiles plus the class name — _name is not unique (two EPLs,
# two dPIEs, dynamic scaling names) and even module basenames collide (mass/
# and light/ both have a combined_profile.CombinedProfile).
PENDING_PHYSICALITY_AUDIT = frozenset({
    "light.sersic.CoreSersic",        # formula needs review first (R_sersic ** alpha ** 1.0)
    "light.sersic.DoubleSersic",      # combined profile — needs recursive validation
    "light.sersic_shapelets.SersicShapelets",
    "light.combined_profile.CombinedProfile",
    "mass.combined_profile.CombinedProfile",
    "mass.piemd.DPIS",
    "mass.piemd.DPIE",
    "mass.piep.DPIEP",
    "mass.tnfw.TNFW",
    "mass.tnfw_ellipse.TNFW_Ellipse",
    "mass.nfw_ellipse_slope.NFW_ELLIPSE_SLOPE",
    "mass.scaling_relation.ScalingRelation",
    "mass.scaling_series.ScalingRelationSeries",
    "mass.dpie_series.DPIESeries",
    "mass.dpie_subhalo.DPIESubhalo",
    "mass.dpie_subhalo_series.DPIESubhaloSeries",
})
