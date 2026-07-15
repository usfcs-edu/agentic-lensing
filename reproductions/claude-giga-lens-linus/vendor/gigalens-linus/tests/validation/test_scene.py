"""Unit tests for the scene derivation (Phase 2): Component / Plane / shared / LensModel.

Not physics-vs-lenstronomy — these check the *structure* the new API derives: that
fixing pulls a param out of the prior, that sharing reduces sampled dimensionality and
is counted once (§4), that a reused bare distribution raises (§3.3), and that the
geometry/completeness validators raise rather than silently default (§3.1, §3.2).
"""
import numpy as np
import pytest
import tensorflow_probability.substrates.jax as tfp

tfd = tfp.distributions

from gigalens.jax.scene import Component, Plane, LensModel, shared


def _epl_priors():
    """Fresh free EPL priors (new objects each call so nothing is reused by identity)."""
    return dict(
        theta_E=tfd.LogNormal(np.log(1.2), 0.3), gamma=tfd.Normal(2.0, 0.1),
        e1=tfd.Normal(0.0, 0.1), e2=tfd.Normal(0.0, 0.1),
        center_x=tfd.Normal(0.0, 0.05), center_y=tfd.Normal(0.0, 0.05),
    )


def _sersic_priors(center_x=None, center_y=None):
    """Fresh Sérsic priors; pass a shared handle for center to link across components."""
    return dict(
        R_sersic=tfd.LogNormal(np.log(0.3), 0.3), n_sersic=tfd.Uniform(0.5, 6.0),
        e1=tfd.Normal(0.0, 0.1), e2=tfd.Normal(0.0, 0.1),
        center_x=center_x if center_x is not None else tfd.Normal(0.0, 0.3),
        center_y=center_y if center_y is not None else tfd.Normal(0.0, 0.3),
    )


def _epl(**over):
    from gigalens.jax.profiles.mass.epl import EPL
    p = _epl_priors(); p.update(over)
    return Component(EPL(50), p)


def _sersic_source(**over):
    from gigalens.jax.profiles.light.sersic import SersicEllipse
    p = _sersic_priors(); p.update(over)
    return Component(SersicEllipse(use_lstsq=True), p)


# -- fixing -----------------------------------------------------------------------
def test_fixing_pulls_param_out_of_prior():
    """A number is a constant (not in the prior); a distribution is free."""
    from gigalens.jax.profiles.mass.epl import EPL
    pri = _epl_priors(); pri["gamma"] = 2.0  # FIX gamma
    m = LensModel([Plane(mass=[Component(EPL(50), pri)]),
                   Plane(deflection_ratio=1.0, light=[_sersic_source()])])
    assert m.num_free_params == 5 + 6  # EPL: 6-1 fixed; Sersic(lstsq): 6
    # gamma is a constant at its path, and absent from the prior leaves.
    assert float(m.constants["planes"][0]["mass"][0]["gamma"]) == 2.0
    leaves = set(m.prior.model.keys())
    assert not any(k.endswith("gamma") for k in leaves)


# -- sharing (§4) -----------------------------------------------------------------
def test_sharing_reduces_dimensionality_and_is_counted_once():
    """Sharing center_x across two sources is ONE free param, not two; to_params fans
    the single value out to both sites."""
    cx = shared(tfd.Normal(0.0, 0.3))
    src_a = _sersic_source(center_x=cx)
    src_b = _sersic_source(center_x=cx)
    shared_model = LensModel([Plane(mass=[_epl()]),
                              Plane(deflection_ratio=1.0, light=[src_a, src_b])])

    # Independent baseline: same structure, distinct center_x distributions.
    indep_model = LensModel([Plane(mass=[_epl()]),
                             Plane(deflection_ratio=1.0,
                                   light=[_sersic_source(), _sersic_source()])])

    assert shared_model.num_free_params == indep_model.num_free_params - 1

    # The prior carries the shared param exactly once.
    n_shared_leaves = sum(1 for k in shared_model.prior.model if k.startswith("shared_"))
    assert n_shared_leaves == 1

    # to_params fans the one sampled value out to both light sites.
    from jax import random as jr
    unique = shared_model.prior.sample(seed=jr.PRNGKey(0))  # the unique-param dict
    params = shared_model.to_params(unique)
    a = params["planes"][1]["light"][0]["center_x"]
    b = params["planes"][1]["light"][1]["center_x"]
    assert float(a) == float(b)


# -- partial fix (G1 D1: LensModel.fix_to) ----------------------------------------
def test_fix_to_freezes_all_but_free_components():
    """fix_to(truth, free=[src]) pins every parameter to its truth value EXCEPT the
    free Component's params, which keep their priors. Exact/derived — no tolerance.

    Build EPL+Shear mass + Sersic lens light (plane 0) + Sersic source (plane 1), all
    free. Fix to a truth with only the source free; assert: (a) non-free params left the
    prior and became constants at their truth values, (b) the source's params remain in
    the prior, (c) num_free_params dropped by EXACTLY the count of fixed free params, and
    (d) geometry (deflection_ratio) is fixed to truth.
    """
    from gigalens.jax.profiles.mass.epl import EPL
    from gigalens.jax.profiles.mass.shear import Shear
    from gigalens.jax.profiles.light.sersic import SersicEllipse
    from jax import random as jr

    epl = _epl()
    shear = Component(Shear(), dict(gamma1=tfd.Normal(0.0, 0.1), gamma2=tfd.Normal(0.0, 0.1)))
    lens_light = Component(SersicEllipse(use_lstsq=True), _sersic_priors())
    src = _sersic_source()
    model = LensModel([
        Plane(mass=[epl, shear], light=[lens_light]),
        Plane(deflection_ratio=1.0, light=[src]),
    ])
    n_before = model.num_free_params  # EPL6 + Shear2 + LensLight6 + Source6 = 20

    # A concrete truth (structured params dict) from sampling the full prior.
    truth = model.to_params(model.prior.sample(seed=jr.PRNGKey(1)))

    # Count the free params that belong to the SOURCE (kept free).
    src_param_count = len(src.profile.params)  # 6 for SersicEllipse (lstsq amps excluded)

    fixed = model.fix_to(truth, free=[src])

    # (c) dimensionality dropped by exactly the number of NON-source free params.
    assert fixed.num_free_params == src_param_count, (
        f"expected {src_param_count} free (source only), got {fixed.num_free_params}")
    assert n_before - fixed.num_free_params == (n_before - src_param_count)

    # (b) the source's params remain free (present in the prior leaves).
    src_leaves = [k for k in fixed.prior.model if k.startswith("planes/1/light/0/")]
    assert len(src_leaves) == src_param_count, src_leaves

    # (a) non-free params are gone from the prior and pinned to their truth values.
    for k in fixed.prior.model:
        assert k.startswith("planes/1/light/0/"), f"non-source param still free: {k}"
    # EPL theta_E became a constant equal to truth.
    c = fixed.constants["planes"][0]["mass"][0]["theta_E"]
    assert float(c) == float(np.asarray(truth["planes"][0]["mass"][0]["theta_E"]))
    # Shear gamma1 pinned to truth.
    assert float(fixed.constants["planes"][0]["mass"][1]["gamma1"]) == \
        float(np.asarray(truth["planes"][0]["mass"][1]["gamma1"]))
    # Lens light R_sersic pinned to truth.
    assert float(fixed.constants["planes"][0]["light"][0]["R_sersic"]) == \
        float(np.asarray(truth["planes"][0]["light"][0]["R_sersic"]))

    # (d) geometry fixed to truth.
    assert float(fixed.constants["planes"][1]["geometry"]["deflection_ratio"]) == \
        float(np.asarray(truth["planes"][1]["geometry"]["deflection_ratio"]))


def test_source_plane_light_identifies_lensed_components():
    """source_plane_light() returns light on planes preceded by mass (by identity)."""
    from gigalens.jax.profiles.light.sersic import SersicEllipse
    lens_light = Component(SersicEllipse(use_lstsq=True), _sersic_priors())
    src = _sersic_source()
    model = LensModel([
        Plane(mass=[_epl()], light=[lens_light]),
        Plane(deflection_ratio=1.0, light=[src]),
    ])
    spl = model.source_plane_light()
    assert [id(c) for c in spl] == [id(src)], "source_plane_light must be exactly [src]"
    assert id(lens_light) not in {id(c) for c in spl}


def test_reused_bare_distribution_raises():
    """Reusing the SAME tfd object (not via shared()) is a loud error, not silent link."""
    d = tfd.Normal(0.0, 0.3)
    src_a = _sersic_source(center_x=d)
    src_b = _sersic_source(center_x=d)  # same object reused
    with pytest.raises(ValueError, match="reuse|shared"):
        LensModel([Plane(mass=[_epl()]),
                   Plane(deflection_ratio=1.0, light=[src_a, src_b])])


# -- completeness (§3.2) ----------------------------------------------------------
def test_missing_param_raises():
    from gigalens.jax.profiles.mass.epl import EPL
    pri = _epl_priors(); del pri["gamma"]
    with pytest.raises(ValueError, match="missing parameter"):
        LensModel([Plane(mass=[Component(EPL(50), pri)]),
                   Plane(deflection_ratio=1.0, light=[_sersic_source()])])


def test_unknown_param_raises():
    from gigalens.jax.profiles.mass.epl import EPL
    pri = _epl_priors(); pri["not_a_param"] = tfd.Normal(0.0, 1.0)
    with pytest.raises(ValueError, match="unknown parameter"):
        LensModel([Plane(mass=[Component(EPL(50), pri)]),
                   Plane(deflection_ratio=1.0, light=[_sersic_source()])])


# -- geometry (§3.1) --------------------------------------------------------------
def test_single_source_plane_defaults_deflection_ratio_to_one():
    m = LensModel([Plane(mass=[_epl()]), Plane(light=[_sersic_source()])])  # no dr given
    assert float(m.constants["planes"][1]["geometry"]["deflection_ratio"]) == 1.0


def test_two_lensed_planes_missing_deflection_ratio_raises():
    with pytest.raises(ValueError, match="deflection_ratio is required"):
        LensModel([Plane(mass=[_epl()]),
                   Plane(light=[_sersic_source()]),
                   Plane(light=[_sersic_source()])])


def test_cosmo_requires_redshift_on_every_plane():
    from gigalens.jax.cosmo import wCDM_Cosmo
    cosmo = Component(wCDM_Cosmo(z_lens=0.5, z_source_ref=10.0), dict(H0=70.0, Om0=0.3, k=0.0, w0=-1.0))
    with pytest.raises(ValueError, match="redshift is required"):
        LensModel([Plane(mass=[_epl()], redshift=0.5),
                   Plane(light=[_sersic_source()])],  # missing redshift
                  cosmo=cosmo)


def test_cosmo_rejects_deflection_ratio():
    from gigalens.jax.cosmo import wCDM_Cosmo
    cosmo = Component(wCDM_Cosmo(z_lens=0.5, z_source_ref=10.0), dict(H0=70.0, Om0=0.3, k=0.0, w0=-1.0))
    with pytest.raises(ValueError, match="deflection_ratio given but a cosmology"):
        LensModel([Plane(mass=[_epl()], redshift=0.5),
                   Plane(light=[_sersic_source()], redshift=2.0, deflection_ratio=0.8)],
                  cosmo=cosmo)
