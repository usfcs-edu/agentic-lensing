"""Soft (plausibility) physicality bands.

Contract: soft bands are human-curated plausibility judgments, NOT code-derived
hard bounds. A fixed value or prior mass outside a soft band is physically valid
but atypical, so it produces a ``severity="soft"`` finding and (at construction)
a distinct ``[plausibility]``-prefixed UserWarning — it NEVER raises. This file
pins the three bands registered in this increment:

  - EPL ``gamma >= 1.1`` (per-parameter ``Domain.soft``);
  - mass-profile axis ratio ``q >= 0.2`` (joint, all e1/e2 mass profiles);
  - external-shear magnitude ``<= 0.2`` (joint, SHEAR).

Masses are asserted via each distribution's exact cdf where available, so no
probe tolerance is needed for the scalar-prior cases.
"""
import math
import warnings

import pytest

pytestmark = pytest.mark.redteam


def _tfd():
    from tensorflow_probability.substrates import jax as tfp
    return tfp.distributions


EPL_OK = dict(theta_E=1.1, gamma=2.05, e1=0.04, e2=-0.03, center_x=0.0, center_y=0.0)
SRC_OK = dict(R_sersic=0.30, n_sersic=2.0, e1=0.05, e2=-0.02,
              center_x=0.05, center_y=-0.04)


def _model(mass_comps):
    """Two-plane model: a mass plane (given Components) + a ratio-1 source plane."""
    from gigalens.jax.profiles.light.sersic import SersicEllipse
    from gigalens.jax.scene import Component, LensModel, Plane
    return LensModel([
        Plane(mass=list(mass_comps)),
        Plane(deflection_ratio=1.0,
              light=[Component(SersicEllipse(use_lstsq=True), dict(SRC_OK))]),
    ])


def _epl(**over):
    from gigalens.jax.profiles.mass.epl import EPL
    from gigalens.jax.scene import Component
    p = dict(EPL_OK)
    p.update(over)
    return Component(EPL(50), p)


def _shear(gamma1, gamma2):
    from gigalens.jax.profiles.mass.shear import Shear
    from gigalens.jax.scene import Component
    return Component(Shear(), dict(gamma1=gamma1, gamma2=gamma2))


def _soft(report, param_contains):
    hits = [f for f in report.findings
            if f.severity == "soft" and param_contains in f.param]
    assert hits, (f"expected a soft finding on {param_contains!r}; report:\n"
                  + report.summary())
    return hits[0]


# ------------------------------------------------------------------ registration
def test_soft_bands_are_registered_where_expected():
    """The three soft bands must be declared on the right profiles/params."""
    from gigalens.jax.profiles.mass.bpl import BPL
    from gigalens.jax.profiles.mass.epl import EPL
    from gigalens.jax.profiles.mass.nfw import NFW_ELLIPSE
    from gigalens.jax.profiles.mass.shear import Shear
    from gigalens.jax.profiles.mass.sie import SIE

    assert EPL._domains["gamma"].soft == (1.1, 3.0)
    # axis-ratio joint on every audited ellipticity mass profile
    for cls in (EPL, SIE, NFW_ELLIPSE, BPL):
        names = [jc.name for jc in cls._joint_constraints if jc.severity == "soft"]
        assert any("axis ratio" in n for n in names), (cls.__name__, names)
    # shear magnitude joint
    shear_soft = [jc.name for jc in Shear._joint_constraints if jc.severity == "soft"]
    assert any("|shear|" in n for n in shear_soft), shear_soft


# ------------------------------------------------------------------ gamma (box)
def test_gamma_just_above_one_soft_warns_not_raises():
    """gamma=1.05: inside the hard domain (1,3) but below the soft floor 1.1 —
    a [plausibility] UserWarning and a soft finding, NEVER a ValueError."""
    with pytest.warns(UserWarning, match=r"\[plausibility\].*gamma"):
        m = _model([_epl(gamma=1.05)])
    f = _soft(m.physicality_report, "gamma")
    assert f.kind == "soft-fixed-value"
    assert not m.physicality_report.errors()


def test_gamma_soft_prior_mass_exact_cdf():
    """gamma ~ Uniform(1.0, 1.2): zero HARD mass (support is inside (1,3)) but half
    the prior sits below the soft floor 1.1 -> a soft-prior-mass finding whose mass
    is the exact cdf value 0.5, no probe."""
    tfd = _tfd()
    with pytest.warns(UserWarning, match=r"\[plausibility\].*gamma"):
        m = _model([_epl(gamma=tfd.Uniform(1.0, 1.2))])
    f = _soft(m.physicality_report, "gamma")
    assert f.kind == "soft-prior-mass"
    assert "cdf" in f.method
    assert abs(f.mass - 0.5) < 1e-6, f.mass    # tfp cdf evaluated in float32
    # and no HARD warning: Uniform(1.0,1.2) places no mass at gamma<=1
    assert not m.physicality_report.warnings_()


def test_gamma_sane_produces_no_soft_finding():
    """gamma=2.05 (EPL_OK): comfortably inside the plausible band -> no soft
    finding (a clean check instead), no warning of any kind."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", category=UserWarning)
        m = _model([_epl()])
    assert not m.physicality_report.soft_warnings_()
    assert not m.physicality_report.findings


# ------------------------------------------------------------------ axis ratio (joint)
def test_axis_ratio_fixed_soft_warns():
    """e1=0.7, e2=0: c=0.7 so q=(1-0.7)/(1+0.7)=0.176 < 0.2 (very flattened), yet
    c<1 keeps the HARD ellipticity bound satisfied -> a soft finding on the axis-
    ratio joint, no raise."""
    with pytest.warns(UserWarning, match=r"\[plausibility\].*axis ratio"):
        m = _model([_epl(e1=0.7, e2=0.0)])
    f = _soft(m.physicality_report, "axis ratio")
    assert f.kind == "soft-fixed-value"
    assert not m.physicality_report.errors()


def test_axis_ratio_prior_mass_probe():
    """(e1,e2) ~ N(0, 0.4) each: a large fraction has c>2/3 (q<0.2). The soft joint
    fires with a probe-estimated mass; the exact HARD ellipticity bound (c>=1) also
    catches its own smaller tail — the two are distinct findings."""
    tfd = _tfd()
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        m = _model([_epl(e1=tfd.Normal(0.0, 0.4), e2=tfd.Normal(0.0, 0.4))])
    soft = _soft(m.physicality_report, "axis ratio")
    assert soft.kind == "soft-joint-prior-mass"
    assert "probe" in soft.method
    assert soft.mass > 0.05
    # exact Rayleigh tail P(c > 2/3) with scale 0.4: exp(-(2/3)^2 / (2*0.4^2))
    expected = math.exp(-(2.0 / 3.0) ** 2 / (2.0 * 0.4 ** 2))
    assert abs(soft.mass - expected) < 0.02, (soft.mass, expected)


# ------------------------------------------------------------------ shear (joint)
def test_shear_magnitude_fixed_soft_warns():
    """External shear gamma1=0.3, gamma2=0.0: |shear|=0.3 > 0.2 -> a soft finding
    (the linear kernel is exact, so there is no hard bound to violate)."""
    with pytest.warns(UserWarning, match=r"\[plausibility\].*shear"):
        m = _model([_epl(), _shear(0.3, 0.0)])
    f = _soft(m.physicality_report, "shear")
    assert f.kind == "soft-fixed-value"
    assert not m.physicality_report.errors()


def test_shear_small_produces_no_soft_finding():
    """A realistic |shear| ~ 0.05 is inside the plausible band -> no soft finding."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", category=UserWarning)
        m = _model([_epl(), _shear(0.03, -0.04)])  # |shear| = 0.05
    assert not m.physicality_report.soft_warnings_()
