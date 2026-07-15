"""Flat-``z`` bijector convention + grouped (tuple-key) priors.

Covers three things the scene API must guarantee:

1. **Flat-z convention.** ``model.bijector`` maps a flat ``z`` of shape ``(..., D)``
   to the constrained unique-param dict and back. Scalar-only models are unchanged
   (same ``num_free_params``, same ``z_param_names``, round-trip identity).

2. **Misuse-proofing.** The retired ``list(z.T)`` convention is adapted with a loud
   ``DeprecationWarning`` (and, under ``ZBijector._STRICT``, a hard error). Any other
   last-axis size raises immediately, so a silent column/dimension mismatch cannot pass.

3. **Grouped priors.** A tuple key ``("e1", "e2")`` binds one joint distribution (and
   its joint event-space bijector) to several profile params. Its components scatter
   back to the individual sites, so the profile still sees scalar ``e1``/``e2``; the
   coupling lives entirely in the prior/bijector/scatter layer.
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import warnings

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest
from tensorflow_probability.substrates.jax import distributions as tfd

from gigalens.jax.scene import Component, Plane, LensModel, ZBijector
from gigalens.jax.profiles.mass.epl import EPL

SEED = jax.random.PRNGKey(0)


# --------------------------------------------------------------------------------
# model builders (mass-only: we exercise only prior/bijector/scatter, not rendering)
# --------------------------------------------------------------------------------
def _epl_scalar():
    return Component(EPL(20), dict(
        theta_E=tfd.Normal(1.5, 0.1), gamma=tfd.Uniform(1.5, 2.5),
        e1=tfd.Normal(0., 0.1), e2=tfd.Normal(0., 0.1),
        center_x=tfd.Normal(0., 0.1), center_y=tfd.Normal(0., 0.1)))


def _epl_grouped(group_dist=None):
    if group_dist is None:
        group_dist = tfd.MultivariateNormalDiag(loc=[0., 0.], scale_diag=[0.1, 0.1])
    # NOTE: a tuple key requires a {...} dict literal — dict(**kwargs) can't hold one.
    return Component(EPL(20), {
        "theta_E": tfd.Normal(1.5, 0.1), "gamma": tfd.Uniform(1.5, 2.5),
        ("e1", "e2"): group_dist,
        "center_x": tfd.Normal(0., 0.1), "center_y": tfd.Normal(0., 0.1)})


def _model(comp):
    return LensModel([Plane(mass=[comp])])


def _flat_params(model, z):
    """forward a flat z through the bijector + scatter, return a flat name->value dict."""
    x = model.bijector.forward(z)
    p = model.to_params(x)
    out = {}

    def rec(d, pre):
        if isinstance(d, dict):
            for k, v in d.items():
                rec(v, (pre + "/" + str(k)) if pre else str(k))
        else:
            out[pre] = float(np.asarray(d).ravel()[0])

    rec(p, "")
    return out


# --------------------------------------------------------------------------------
# 1. flat-z convention
# --------------------------------------------------------------------------------
def test_scalar_num_free_params_and_names_unchanged():
    m = _model(_epl_scalar())
    assert m.num_free_params == 6
    # scalar entries keep their canonical unique-key names (old behaviour)
    assert set(m.z_param_names) == {
        f"planes/0/mass/0/{n}"
        for n in ("theta_E", "gamma", "e1", "e2", "center_x", "center_y")}


def test_flat_z_roundtrip_scalar():
    m = _model(_epl_scalar())
    start = m.prior.sample(5, seed=SEED)
    z = m.bijector.inverse(start)                      # flat (5, D)
    assert z.shape == (5, m.num_free_params)
    x = m.bijector.forward(z)
    z2 = m.bijector.inverse(x)
    assert jnp.allclose(z, z2, atol=1e-6)


def test_list_shim_matches_flat_and_warns():
    m = _model(_epl_scalar())
    z = m.bijector.inverse(m.prior.sample(3, seed=SEED))   # (3, D)
    flat = m.bijector.forward(z)
    with pytest.warns(DeprecationWarning, match="list"):
        via_list = m.bijector.forward(list(z.T))
    for k in flat:
        assert jnp.allclose(flat[k], via_list[k], atol=1e-6)


def test_wrong_dim_raises():
    m = _model(_epl_scalar())
    bad = jnp.zeros((4, m.num_free_params + 1))         # transposed / wrong width
    with pytest.raises(ValueError, match="last axis"):
        m.bijector.forward(bad)


def test_strict_mode_turns_list_into_hard_error():
    m = _model(_epl_scalar())
    z = m.bijector.inverse(m.prior.sample(2, seed=SEED))
    try:
        ZBijector._STRICT = True
        with pytest.raises(TypeError, match="retired"):
            m.bijector.forward(list(z.T))
    finally:
        ZBijector._STRICT = False


# --------------------------------------------------------------------------------
# 2. grouped-prior mechanics
# --------------------------------------------------------------------------------
def test_grouped_dim_and_scatter():
    mg = _model(_epl_grouped())
    # a 2-component group contributes 2 columns -> same total as the scalar model
    assert mg.num_free_params == 6
    start = mg.prior.sample(4, seed=SEED)
    x = mg.bijector.forward(mg.bijector.inverse(start))
    params = mg.to_params(x)["planes"][0]["mass"][0]
    # the profile sees scalar e1/e2 (not a vector) — scatter split the group
    assert np.asarray(params["e1"]).shape == (4,)
    assert np.asarray(params["e2"]).shape == (4,)
    # and they equal the two components of the sampled group vector
    grp = start["planes/0/mass/0/e1|planes/0/mass/0/e2"]
    got = mg.to_params(start)["planes"][0]["mass"][0]
    assert jnp.allclose(got["e1"], grp[..., 0])
    assert jnp.allclose(got["e2"], grp[..., 1])


def test_grouped_names_and_perturbation_identity():
    mg = _model(_epl_grouped())
    names = mg.z_param_names
    assert len(names) == mg.num_free_params
    # the two group columns are labelled by their member param paths
    assert "planes/0/mass/0/e1" in names
    assert "planes/0/mass/0/e2" in names
    # perturb each column; exactly the claimed physical param moves
    z0 = jnp.zeros(mg.num_free_params)
    base = _flat_params(mg, z0)
    for j, name in enumerate(names):
        zj = z0.at[j].set(0.3)
        moved = [k for k, v in _flat_params(mg, zj).items()
                 if abs(v - base[k]) > 1e-6]
        assert moved == [name], f"column {j} ({name}) moved {moved}"


def test_grouped_roundtrip_and_logprob():
    mg = _model(_epl_grouped())
    start = mg.prior.sample(3, seed=SEED)
    z = mg.bijector.inverse(start)
    assert z.shape == (3, mg.num_free_params)
    x = mg.bijector.forward(z)
    assert jnp.allclose(mg.bijector.inverse(x), z, atol=1e-6)
    # joint prior log_prob + fldj are finite (the log-prior path ProbModel uses)
    lp = mg.prior.log_prob(x) + mg.bijector.forward_log_det_jacobian(z)
    assert jnp.all(jnp.isfinite(lp))


def test_grouped_matches_scalar_forward():
    """A grouped model and the equivalent scalar model produce the same physical
    params for a matched z (the group just reparameterizes the same 2 columns)."""
    ms, mg = _model(_epl_scalar()), _model(_epl_grouped())
    # both are 6-dim; MVNDiag(0,0.1) group == independent Normal(0,0.1) scalars,
    # and column order is sorted-key identical, so a shared z maps identically.
    assert ms.z_param_names == mg.z_param_names
    z = jnp.linspace(-0.5, 0.5, ms.num_free_params)
    assert _flat_params(ms, z) == pytest.approx(_flat_params(mg, z), abs=1e-6)


# --------------------------------------------------------------------------------
# 3. validation / misuse of grouped keys
# --------------------------------------------------------------------------------
def test_group_wrong_event_shape_raises():
    bad = tfd.MultivariateNormalDiag(loc=[0., 0., 0.], scale_diag=[1., 1., 1.])  # dim 3
    with pytest.raises(ValueError, match="event_shape"):
        _model(_epl_grouped(bad))


def test_group_scalar_dist_raises():
    with pytest.raises(ValueError, match="event_shape"):
        _model(_epl_grouped(tfd.Normal(0., 0.1)))  # event_shape [] not [2]


def test_number_under_tuple_key_raises():
    comp = Component(EPL(20), {
        "theta_E": tfd.Normal(1.5, 0.1), "gamma": tfd.Uniform(1.5, 2.5),
        ("e1", "e2"): 0.0,
        "center_x": tfd.Normal(0., 0.1), "center_y": tfd.Normal(0., 0.1)})
    with pytest.raises(TypeError, match="cover multiple"):
        _model(comp)


def test_duplicate_coverage_raises():
    comp = Component(EPL(20), {
        "theta_E": tfd.Normal(1.5, 0.1), "gamma": tfd.Uniform(1.5, 2.5),
        "e1": tfd.Normal(0., 0.1),
        ("e1", "e2"): tfd.MultivariateNormalDiag(loc=[0., 0.], scale_diag=[.1, .1]),
        "center_x": tfd.Normal(0., 0.1), "center_y": tfd.Normal(0., 0.1)})
    with pytest.raises(ValueError, match="more than one"):
        _model(comp)


def test_single_name_tuple_raises():
    comp = Component(EPL(20), {
        "theta_E": tfd.Normal(1.5, 0.1), "gamma": tfd.Uniform(1.5, 2.5),
        "e1": tfd.Normal(0., 0.1), "e2": tfd.Normal(0., 0.1),
        ("center_x",): tfd.Normal(0., 0.1),
        "center_y": tfd.Normal(0., 0.1)})
    with pytest.raises(ValueError, match="at least two"):
        _model(comp)
