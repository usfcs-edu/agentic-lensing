"""Red-team — physicality layer: code-derived hard domains on profile parameters.

Contract (red-team, not parity): the oracle is the kernel itself. Every hard
bound registered in a profile's ``_domains`` / ``_joint_constraints`` derives
from a verified kernel behavior (clip, mask, NaN, silent zero), and this file
pins BOTH sides:

  1. the *pathology* — the kernel really does misbehave outside the domain
     (so if someone later fixes a kernel, the failing pin says "now relax the
     registered domain");
  2. the *guard* — LensModel construction raises on fixed values outside a
     hard domain, warns with a calibrated mass estimate when a prior places
     more than EPS_MASS of probability outside one, and stays silent for sane
     configurations.

Prior-mass assertions use analytically computable fixtures (uniform square vs.
disk; Gaussian tail vs. a fixed offset), asserted within the probe's binomial
5-sigma band — never a tuned tolerance.

Soft (plausibility) bounds are deliberately unset in this draft (human-curated).
"""
import math
import warnings

import numpy as np
import pytest

pytestmark = pytest.mark.redteam

PROBE_TOL_SIGMA = 5.0


# ---------------------------------------------------------------- helpers
def _tfd():
    from tensorflow_probability.substrates import jax as tfp
    return tfp.distributions


def _scene():
    import gigalens.jax.scene as scene
    return scene


EPL_OK = dict(theta_E=1.1, gamma=2.05, e1=0.04, e2=-0.03, center_x=0.0, center_y=0.0)
SRC_OK = dict(R_sersic=0.30, n_sersic=2.0, e1=0.05, e2=-0.02,
              center_x=0.05, center_y=-0.04)


def _model(mass_priors=None, light_priors=None, mass_cls=None, light_cls=None):
    """Minimal two-plane model: mass plane + source-light plane (ratio 1)."""
    from gigalens.jax.profiles.mass.epl import EPL
    from gigalens.jax.profiles.light.sersic import SersicEllipse
    scene = _scene()
    mass_cls = mass_cls or (lambda: EPL(50))
    light_cls = light_cls or (lambda: SersicEllipse(use_lstsq=True))
    planes = [
        scene.Plane(mass=[scene.Component(mass_cls(), dict(mass_priors or EPL_OK))]),
        scene.Plane(deflection_ratio=1.0,
                    light=[scene.Component(light_cls(), dict(light_priors or SRC_OK))]),
    ]
    return scene.LensModel(planes)


def _binomial_tol(p, k):
    return PROBE_TOL_SIGMA * math.sqrt(p * (1.0 - p) / k)


def _mass_of(report, param_contains):
    hits = [f for f in report.findings
            if f.severity == "warning" and param_contains in f.param]
    assert hits, (f"expected a warning finding on {param_contains!r}; report:\n"
                  + report.summary())
    return hits[0].mass, hits[0].method


# ---------------------------------------------------------------- 1. fixed values raise
@pytest.mark.parametrize(
    "bad_mass, match",
    [
        (dict(EPL_OK, theta_E=-1.5), "theta_E"),          # box violation
        (dict(EPL_OK, gamma=5.0), "gamma"),               # series pole domain
        (dict(EPL_OK, e1=0.8, e2=0.7), r"e1\^2\+e2\^2"),  # joint violation
    ],
)
def test_fixed_unphysical_mass_params_raise_at_construction(bad_mass, match):
    with pytest.raises(ValueError, match=match):
        _model(mass_priors=bad_mass)


def test_fixed_circular_sie_raises_at_construction():
    """SIE at exactly e1=e2=0 is all-NaN (kernel divides by sqrt(1-q^2)=0) —
    the error should fire at construction, not as NaNs mid-inference."""
    from gigalens.jax.profiles.mass.sie import SIE
    sie_circ = dict(theta_E=1.1, e1=0.0, e2=0.0, center_x=0.0, center_y=0.0)
    with pytest.raises(ValueError, match="e1"):
        _model(mass_priors=sie_circ, mass_cls=SIE)


def test_fixed_unphysical_light_params_raise_at_construction():
    with pytest.raises(ValueError, match="R_sersic"):
        _model(light_priors=dict(SRC_OK, R_sersic=-0.3))


def test_fixed_bpl_core_exceeding_b_raises_at_construction():
    """r_c > b is silently clamped to r_c = b by the kernel (verified
    bitwise-identical), so the constructed model would not be the model asked."""
    from gigalens.jax.profiles.mass.bpl import BPL
    bad = dict(b=1.5, alpha=2.0, alpha_c=0.5, r_c=2.0, e1=0.3, e2=0.0,
               center_x=0.0, center_y=0.0)
    with pytest.raises(ValueError, match="r_c"):
        _model(mass_priors=bad, mass_cls=BPL)


# ---------------------------------------------------------------- 2. prior mass warns, calibrated
def test_scalar_prior_mass_outside_domain_warns_exact_cdf():
    """theta_E ~ Normal(0.5, 0.25): mass below 0 is Phi(-2) = 0.02275, computed
    exactly via the distribution's cdf — no probe involved."""
    tfd = _tfd()
    priors = dict(EPL_OK, theta_E=tfd.Normal(0.5, 0.25))
    with pytest.warns(UserWarning, match="theta_E"):
        m = _model(mass_priors=priors)
    mass, method = _mass_of(m.physicality_report, "theta_E")
    expected = 0.5 * math.erfc(2.0 / math.sqrt(2.0))  # Phi(-2)
    assert "cdf" in method
    assert abs(mass - expected) < 1e-6, (mass, expected)


def test_joint_prior_mass_probe_matches_analytic_square_vs_disk():
    """e1, e2 ~ independent Uniform(-0.9, 0.9): the violating mass of
    e1^2+e2^2 >= 1 is the square-minus-disk area over the square area —
    computable in closed form. The probe estimate must land within its own
    binomial 5-sigma band of that truth."""
    tfd = _tfd()
    s = 0.9
    x0 = math.sqrt(1.0 - s * s)
    # quarter-plane area of {x^2+y^2<=1} ∩ [0,s]^2:
    quarter = x0 * s + 0.5 * ((s * math.sqrt(1 - s * s) + math.asin(s))
                              - (x0 * math.sqrt(1 - x0 * x0) + math.asin(x0)))
    expected = 1.0 - quarter / (s * s)  # violating fraction of the square
    priors = dict(EPL_OK, e1=tfd.Uniform(-s, s), e2=tfd.Uniform(-s, s))
    with pytest.warns(UserWarning, match="e1"):
        m = _model(mass_priors=priors)
    mass, method = _mass_of(m.physicality_report, "e1^2+e2^2")
    from gigalens.physicality import PROBE_DRAWS
    assert "probe" in method
    assert abs(mass - expected) < _binomial_tol(expected, PROBE_DRAWS), (
        mass, expected)


def test_mixed_fixed_plus_prior_conditional_mass_caught():
    """The case per-parameter boxes cannot see: e1 = 0.7 FIXED, e2 ~ N(0, 0.3).
    Marginally both look fine; jointly P(0.49 + e2^2 >= 1) = 2*Phi(-sqrt(0.51)/0.3)
    ~ 1.7e-2 of the prior sits where the kernel renders zero deflection."""
    tfd = _tfd()
    priors = dict(EPL_OK, e1=0.7, e2=tfd.Normal(0.0, 0.3))
    with pytest.warns(UserWarning, match="e1"):
        m = _model(mass_priors=priors)
    mass, _ = _mass_of(m.physicality_report, "e1^2+e2^2")
    expected = math.erfc(math.sqrt(0.51) / 0.3 / math.sqrt(2.0))  # 2*Phi(-z)
    from gigalens.physicality import PROBE_DRAWS
    assert abs(mass - expected) < _binomial_tol(expected, PROBE_DRAWS), (
        mass, expected)


def test_sane_priors_construct_silently():
    """Physical control: a typical configuration must produce NO physicality
    warnings — the derived EPS_MASS threshold separates sane Gaussian tails
    (Phi(-6) ~ 1e-9) from real leaks. Run with warnings-as-errors."""
    tfd = _tfd()
    mass_p = dict(theta_E=tfd.Normal(1.5, 0.25), gamma=tfd.TruncatedNormal(2.0, 0.15, 1.2, 2.8),
                  e1=tfd.Normal(0.0, 0.15), e2=tfd.Normal(0.0, 0.15),
                  center_x=0.0, center_y=0.0)
    light_p = dict(R_sersic=tfd.LogNormal(-1.0, 0.25), n_sersic=tfd.Uniform(0.5, 6.0),
                   e1=tfd.Normal(0.0, 0.15), e2=tfd.Normal(0.0, 0.15),
                   center_x=0.05, center_y=-0.04)
    with warnings.catch_warnings():
        # UserWarning only: the probe's tfp sampling emits an unrelated jax
        # DeprecationWarning; the physicality layer speaks in UserWarning.
        warnings.simplefilter("error", category=UserWarning)
        m = _model(mass_priors=mass_p, light_priors=light_p)
    # ... and the report must still carry the audit trail of what ran clean.
    assert len(m.physicality_report.checks_run) > 0
    assert not m.physicality_report.findings


def test_grouped_prior_mass_probe_matches_analytic_chi2():
    """Grouped tuple-key prior: (e1, e2) ~ MultivariateNormalDiag(0, 0.55 I).
    The violating mass of e1^2+e2^2 >= 1 is exp(-1/(2*0.55^2)) exactly
    (Rayleigh tail) — pins the grouped-prior column extraction path."""
    tfd = _tfd()
    sigma = 0.55
    priors = dict(theta_E=1.1, gamma=2.05, center_x=0.0, center_y=0.0)
    priors[("e1", "e2")] = tfd.MultivariateNormalDiag(
        loc=[0.0, 0.0], scale_diag=[sigma, sigma])
    with pytest.warns(UserWarning, match="e1"):
        m = _model(mass_priors=priors)
    mass, method = _mass_of(m.physicality_report, "e1^2+e2^2")
    expected = math.exp(-1.0 / (2.0 * sigma * sigma))
    from gigalens.physicality import PROBE_DRAWS
    assert "probe" in method
    assert abs(mass - expected) < _binomial_tol(expected, PROBE_DRAWS), (
        mass, expected)


def test_grouped_correlated_prior_preserves_joint_structure():
    """Correlated grouped prior: (r_c, b) ~ MVN with rho=0.5, sigma=0.1 each,
    means (0.6, 0.5). P(r_c > b) = Phi((0.6-0.5)/sqrt(2 sigma^2 (1-rho))) =
    Phi(1) exactly — a value that is WRONG (Phi(0.707)) if the engine breaks
    the correlation between the grouped columns."""
    tfd = _tfd()
    cov = [[0.01, 0.005], [0.005, 0.01]]
    tril = np.linalg.cholesky(np.asarray(cov)).tolist()
    priors = dict(alpha=2.0, alpha_c=0.5, e1=0.3, e2=0.0,
                  center_x=0.0, center_y=0.0)
    priors[("r_c", "b")] = tfd.MultivariateNormalTriL(
        loc=[0.6, 0.5], scale_tril=tril)
    from gigalens.jax.profiles.mass.bpl import BPL
    with pytest.warns(UserWarning, match="r_c"):
        m = _model(mass_priors=priors, mass_cls=BPL)
    mass, _ = _mass_of(m.physicality_report, "r_c <= b")
    expected = 0.5 * math.erfc(-1.0 / math.sqrt(2.0))          # Phi(+1)
    decoy = 0.5 * math.erfc(-1.0 / math.sqrt(2.0) / math.sqrt(2.0))  # Phi(1/sqrt2)
    from gigalens.physicality import PROBE_DRAWS
    tol = _binomial_tol(expected, PROBE_DRAWS)
    assert abs(mass - expected) < tol, (mass, expected)
    assert abs(mass - decoy) > 10 * tol, "correlation was broken (independent draws)"


def test_shared_handle_prior_checked_at_every_site():
    """A shared() theta_E used by two mass components must be flagged at BOTH
    sites, each with the exact Phi(-2) cdf mass."""
    tfd = _tfd()
    scene = _scene()
    from gigalens.jax.profiles.mass.epl import EPL
    from gigalens.jax.profiles.light.sersic import SersicEllipse
    h = scene.shared(tfd.Normal(0.5, 0.25))
    mk = lambda: scene.Component(EPL(50), dict(EPL_OK, theta_E=h))
    planes = [
        scene.Plane(mass=[mk(), mk()]),
        scene.Plane(deflection_ratio=1.0,
                    light=[scene.Component(SersicEllipse(use_lstsq=True),
                                           dict(SRC_OK))]),
    ]
    with pytest.warns(UserWarning, match="theta_E"):
        m = scene.LensModel(planes)
    hits = [f for f in m.physicality_report.findings
            if f.severity == "warning" and f.param == "theta_E"]
    assert len(hits) == 2, m.physicality_report.summary()
    expected = 0.5 * math.erfc(2.0 / math.sqrt(2.0))
    for f in hits:
        assert abs(f.mass - expected) < 1e-6, (f.mass, expected)


# ---------------------------------------------------------------- 3. report plumbing
def test_unregistered_profile_reported_not_silently_skipped():
    from gigalens import physicality
    scene = _scene()

    class _Stub:
        name = "STUB"
        params = ["p"]

    plane = scene.Plane(mass=[scene.Component(_Stub(), {"p": 1.0})])
    report = physicality.validate_planes([plane])
    kinds = [f.kind for f in report.findings]
    assert "unregistered-profile" in kinds


# ---------------------------------------------------------------- 4. kernel-truth pins
# These pin the PATHOLOGIES the domains derive from. If one of these starts
# failing, the kernel was fixed — relax/remove the corresponding domain entry
# and its rationale rather than deleting the test.
def _grid():
    import jax.numpy as jnp
    g = jnp.linspace(-2.0, 2.0, 11)
    return g, g


def test_kernel_truth_sie_circular_is_all_nan():
    from gigalens.jax.profiles.mass.sie import SIE
    x, y = _grid()
    fx, fy = SIE().deriv(x, y, 1.5, 0.0, 0.0, 0.0, 0.0)
    assert bool(np.all(np.isnan(np.asarray(fx)))) and bool(
        np.all(np.isnan(np.asarray(fy))))


def test_kernel_truth_epl_superextremal_ellipticity_renders_zero():
    from gigalens.jax.profiles.mass.epl import EPL
    x, y = _grid()
    fx, fy = EPL(50).deriv(x, y, 1.5, 2.0, 1.3, 0.0, 0.0, 0.0)
    assert np.all(np.asarray(fx) == 0.0) and np.all(np.asarray(fy) == 0.0)


def test_kernel_truth_negative_r_sersic_keeps_only_central_spike():
    from gigalens.jax.profiles.light.sersic import Sersic
    x, y = _grid()
    lit = np.asarray(Sersic(use_lstsq=True).light(x, y, -1.0, 4.0, 0.0, 0.0))
    ref = np.asarray(Sersic(use_lstsq=True).light(x, y, 1.0, 4.0, 0.0, 0.0))
    on_center = np.isclose(np.asarray(x), 0.0)
    assert np.all(lit[..., ~on_center] == 0.0), "off-center pixels should be zeroed"
    assert np.max(lit) == pytest.approx(np.max(ref)), "peak silently unchanged"


def test_kernel_truth_bpl_core_clamp_is_bitwise():
    from gigalens.jax.profiles.mass.bpl import BPL
    x, y = _grid()
    a1 = BPL().deriv(x, y, 1.5, 2.0, 0.5, 3.0, 0.3, 0.0, 0.0, 0.0)
    a2 = BPL().deriv(x, y, 1.5, 2.0, 0.5, 1.5, 0.3, 0.0, 0.0, 0.0)
    assert np.array_equal(np.asarray(a1[0]), np.asarray(a2[0]))
    assert np.array_equal(np.asarray(a1[1]), np.asarray(a2[1]))


# ---------------------------------------------------------------- 5. coverage ratchet
def test_every_profile_registers_domains_or_is_explicitly_pending():
    """Ratchet: a profile class must either declare _domains or appear in
    PENDING_PHYSICALITY_AUDIT — never silently opt out. Stale pending entries
    (audited since) must be removed so the list only shrinks."""
    import importlib
    import inspect
    import pkgutil

    import gigalens.jax.profiles as prof_pkg
    from gigalens import physicality
    from gigalens.profile import Parameterized

    found = {}
    for mod_info in pkgutil.walk_packages(prof_pkg.__path__,
                                          prefix=prof_pkg.__name__ + "."):
        mod = importlib.import_module(mod_info.name)
        for _, cls in inspect.getmembers(mod, inspect.isclass):
            if (issubclass(cls, Parameterized) and cls.__module__ == mod.__name__
                    and not inspect.isabstract(cls)):
                # Key = module path relative to the profiles package + class name:
                # basenames collide (mass/ and light/ combined_profile), and a
                # collision would let one twin's registration mask the other's.
                rel = mod.__name__[len(prof_pkg.__name__) + 1:]
                key = f"{rel}.{cls.__name__}"
                assert key not in found, f"ratchet key collision: {key}"
                found[key] = cls

    # Own-class declaration only (matches the engine): inheriting a parent's
    # _domains does not count — the subclass's params/kernel were not audited.
    registered = {k: ("_domains" in vars(c)) for k, c in found.items()}
    missing = sorted(k for k, c in found.items()
                     if not registered[k]
                     and k not in physicality.PENDING_PHYSICALITY_AUDIT)
    stale = sorted(k for k in physicality.PENDING_PHYSICALITY_AUDIT
                   if k in found and registered[k])
    ghost = sorted(k for k in physicality.PENDING_PHYSICALITY_AUDIT
                   if k not in found)
    assert not missing, f"profiles with neither _domains nor pending entry: {missing}"
    assert not stale, f"audited profiles still on the pending list: {stale}"
    assert not ghost, f"pending entries matching no discovered class: {ghost}"

    # Registered classes must cover every class-level parameter.
    for key, cls in found.items():
        doms = vars(cls).get("_domains")
        if doms is None:
            continue
        missing_params = set(cls._params) - set(doms)
        assert not missing_params, (
            f"{key}: _domains missing entries for {sorted(missing_params)}")
