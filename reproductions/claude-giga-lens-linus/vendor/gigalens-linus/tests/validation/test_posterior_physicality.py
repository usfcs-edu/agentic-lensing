"""Posterior-sample physicality checks (``validate_posterior_samples``).

Contract: the SAME hard domains / soft bands / joint constraints that guard
construction are re-run against realized posterior draws, reporting the FRACTION
of the posterior outside each region. Unlike construction it NEVER raises — a
completed fit is diagnosed, not rejected. Input is the structured params pytree
(``model.to_params`` layout) with each leaf holding the draws for that parameter.

Fractions are asserted exactly from the hand-built sample arrays (no sampler,
no probe), so these are closed-form pins on the empirical-mass path.
"""
import numpy as np
import pytest

from gigalens.physicality import validate_posterior_samples

pytestmark = pytest.mark.redteam


def _tfd():
    from tensorflow_probability.substrates import jax as tfp
    return tfp.distributions


EPL_OK = dict(theta_E=1.1, gamma=2.05, e1=0.04, e2=-0.03, center_x=0.0, center_y=0.0)
SRC_OK = dict(R_sersic=0.30, n_sersic=2.0, e1=0.05, e2=-0.02,
              center_x=0.05, center_y=-0.04)


def _model():
    """Sane two-plane model — constructs silently; posterior arrays are fed later."""
    from gigalens.jax.profiles.light.sersic import SersicEllipse
    from gigalens.jax.profiles.mass.epl import EPL
    from gigalens.jax.scene import Component, LensModel, Plane
    return LensModel([
        Plane(mass=[Component(EPL(50), dict(EPL_OK))]),
        Plane(deflection_ratio=1.0,
              light=[Component(SersicEllipse(use_lstsq=True), dict(SRC_OK))]),
    ])


def _mass_samples(**over):
    """Params pytree carrying only the plane-0 EPL mass component's draws."""
    d = dict(theta_E=np.full(5, 1.1), gamma=np.full(5, 2.0),
             e1=np.full(5, 0.04), e2=np.full(5, -0.03),
             center_x=0.0, center_y=0.0)
    d.update(over)
    return {"planes": {0: {"mass": {0: d}}}}


def _find(report, param, kind):
    hits = [f for f in report.findings if f.param == param and f.kind == kind]
    assert hits, (f"expected a {kind!r} finding on {param!r}; report:\n"
                  + report.summary())
    return hits[0]


# ------------------------------------------------------------------ hard domain
def test_posterior_flags_hard_domain_fraction():
    """gamma draws with 1 of 5 at 0.9 (<= hard floor 1): a posterior-mass finding
    (severity 'warning') whose mass is exactly 0.2, and never a raise."""
    model = _model()
    params = _mass_samples(gamma=np.array([2.0, 2.1, 0.9, 2.2, 1.5]))
    report = validate_posterior_samples(model, params)
    f = _find(report, "gamma", "posterior-mass")
    assert f.severity == "warning"
    assert abs(f.mass - 0.2) < 1e-12, f.mass
    assert "N=5" in f.method
    assert not report.errors()          # posterior checks NEVER raise


# ------------------------------------------------------------------ soft band
def test_posterior_flags_soft_gamma_fraction():
    """gamma draws all inside (1,3) but 2 of 5 below the soft floor 1.1: a
    soft-posterior-mass finding (severity 'soft'), mass 0.4, and NO hard finding."""
    model = _model()
    params = _mass_samples(gamma=np.array([1.05, 1.08, 2.0, 2.0, 2.0]))
    report = validate_posterior_samples(model, params)
    f = _find(report, "gamma", "soft-posterior-mass")
    assert f.severity == "soft"
    assert abs(f.mass - 0.4) < 1e-12, f.mass
    assert not report.warnings_()       # nothing crosses the hard floor


# ------------------------------------------------------------------ soft joint
def test_posterior_flags_soft_axis_ratio_joint():
    """e1 draws with 2 of 5 at c in (2/3, 1) — q<0.2 but hard bound (c<1) satisfied:
    a soft axis-ratio joint finding, mass 0.4, and no hard ellipticity finding."""
    model = _model()
    params = _mass_samples(e1=np.array([0.70, 0.72, 0.10, 0.10, 0.10]),
                           e2=np.zeros(5))
    report = validate_posterior_samples(model, params)
    f = _find(report, "axis ratio q >= 0.2", "soft-joint-posterior-mass")
    assert f.severity == "soft"
    assert abs(f.mass - 0.4) < 1e-12, f.mass
    assert not report.warnings_()


# ------------------------------------------------------------------ hard joint
def test_posterior_flags_hard_ellipticity_joint():
    """e1 draws with 1 of 5 at c=1.2 (>= hard bound 1): a joint-posterior-mass
    finding (severity 'warning'), mass 0.2."""
    model = _model()
    params = _mass_samples(e1=np.array([1.2, 0.1, 0.1, 0.1, 0.1]), e2=np.zeros(5))
    report = validate_posterior_samples(model, params)
    f = _find(report, "e1^2+e2^2 < 1.0^2", "joint-posterior-mass")
    assert f.severity == "warning"
    assert abs(f.mass - 0.2) < 1e-12, f.mass


# ------------------------------------------------------------------ clean case
def test_posterior_clean_samples_no_findings():
    """A tight, sane posterior produces zero findings — only clean checks."""
    model = _model()
    rng = np.random.default_rng(0)
    params = _mass_samples(
        theta_E=1.0 + 0.02 * rng.standard_normal(2000),
        gamma=2.0 + 0.05 * rng.standard_normal(2000),
        e1=0.05 * rng.standard_normal(2000),
        e2=0.05 * rng.standard_normal(2000))
    report = validate_posterior_samples(model, params)
    assert not report.findings, report.summary()
    assert report.checks_run                     # audit trail still recorded


def test_posterior_never_raises_on_wild_samples():
    """Even a posterior riddled with hard violations must return a report, not
    raise: the fit already happened; we diagnose it."""
    model = _model()
    params = _mass_samples(theta_E=np.array([-3.0, -1.0, 0.0, 1.0, 2.0]),
                           gamma=np.array([0.2, 5.0, 2.0, 9.0, -1.0]),
                           e1=np.array([2.0, 3.0, 0.1, 0.1, 0.1]), e2=np.zeros(5))
    report = validate_posterior_samples(model, params)   # must not raise
    assert report.warnings_()                            # and it did flag them


# ------------------------------------------------------------------ redshift geometry
def test_posterior_flags_redshift_geometry():
    """Sampled source redshifts behind a z=0.5 lens: the posterior array leaks two
    fractions — mass at z<=0 (domain) and mass below the lens (ordering) — both
    reported as posterior findings, exactly matching the hand-built array."""
    from gigalens.jax.cosmo import wCDM_Cosmo
    from gigalens.jax.profiles.light.sersic import SersicEllipse
    from gigalens.jax.profiles.mass.epl import EPL
    from gigalens.jax.scene import Component, LensModel, Plane
    tfd = _tfd()
    # Tight source-z prior -> silent construction; the POSTERIOR array below is what
    # we actually check, and it deliberately misbehaves.
    model = LensModel(
        [Plane(mass=[Component(EPL(50), dict(EPL_OK))], redshift=0.5),
         Plane(light=[Component(SersicEllipse(use_lstsq=True), dict(SRC_OK))],
               redshift=tfd.Normal(2.0, 0.1))],
        cosmo=Component(wCDM_Cosmo(z_lens=0.5, z_source_ref=10.0),
                        dict(H0=70.0, Om0=0.3, k=0.0, w0=-1.0)))
    zsrc = np.array([2.0, 1.5, 0.3, 0.1, -0.2, 3.0, 0.5, 0.4, -0.5, 1.0])
    params = {"planes": {0: {"geometry": {"redshift": 0.5}},
                         1: {"geometry": {"redshift": zsrc}}}}
    report = validate_posterior_samples(model, params)

    dom = _find(report, "planes[1].redshift", "posterior-mass")
    assert abs(dom.mass - 0.2) < 1e-12, dom.mass       # 2/10 at z<=0
    order = _find(report, "planes[0].redshift <= planes[1].redshift",
                  "joint-posterior-mass")
    assert abs(order.mass - 0.5) < 1e-12, order.mass   # 5/10 below the lens
    assert not report.errors()
