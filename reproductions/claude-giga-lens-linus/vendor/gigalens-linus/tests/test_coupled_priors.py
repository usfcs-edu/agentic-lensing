"""Cross-component coupling (§coupled-priors): coupled(dist, names=...) + soft_link().

A coupled group binds one joint distribution to scalar sites that live in DIFFERENT
components (here a mass EPL and a light Sersic sharing a centre). The group's event
components scatter back to the individual sites, exactly like a within-component tuple
key — so the profiles still see scalar center_x/center_y, and constrained()/to_params
report absolute per-site positions, not the internal k-vector.

The `soft_link` helper builds the coupling in OFFSET form, so the sampler's unconstrained
coordinates are (mass_centre, offset): the separation is its own tight, decorrelated axis
while the constrained space is absolute positions. That geometry is the whole point, so it
is asserted directly.
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest
from tensorflow_probability.substrates.jax import distributions as tfd, bijectors as tfb

from gigalens.jax.scene import (
    Component, Plane, LensModel, coupled, soft_link, CoupledSlot)
from gigalens.jax.profiles.mass.epl import EPL
from gigalens.jax.profiles.light.sersic import Sersic

SEED = jax.random.PRNGKey(0)


# --------------------------------------------------------------------------------
# builders
# --------------------------------------------------------------------------------
def _mass(center):
    """EPL mass component; `center` is a {center_x, center_y: slot} dict to splat."""
    return Component(EPL(20), {
        "theta_E": tfd.Normal(1.5, 0.1), "gamma": tfd.Uniform(1.5, 2.5),
        "e1": tfd.Normal(0., 0.1), "e2": tfd.Normal(0., 0.1), **center})


def _light(center):
    return Component(Sersic(use_lstsq=True), {
        "R_sersic": tfd.Normal(0.3, 0.05), "n_sersic": tfd.Normal(3.0, 0.3), **center})


def _anchor(sigma_pos):
    """One Normal prior per centre coordinate (soft_link requires per-param anchors)."""
    return (tfd.Normal(0.0, sigma_pos), tfd.Normal(0.0, sigma_pos))


def _softlink_model(separation=0.02, sigma_pos=0.3):
    mass_c, light_c = soft_link(
        ("center_x", "center_y"), anchor=_anchor(sigma_pos), separation=separation)
    return LensModel([Plane(mass=[_mass(mass_c)], light=[_light(light_c)])])


# --------------------------------------------------------------------------------
# soft_link: return shape + wiring
# --------------------------------------------------------------------------------
def test_softlink_returns_splatable_participant_dicts():
    mass_c, light_c = soft_link(
        ("center_x", "center_y"), anchor=_anchor(0.3), separation=0.02)
    assert set(mass_c) == set(light_c) == {"center_x", "center_y"}
    for d in (mass_c, light_c):
        assert all(isinstance(v, CoupledSlot) for v in d.values())
    # the two participants are members of the SAME group but distinct event components
    assert mass_c["center_x"].group is light_c["center_x"].group
    assert mass_c["center_x"].idx != light_c["center_x"].idx


def test_softlink_model_builds_and_counts_free_params():
    m = _softlink_model()
    # 4 mass scalars + 2 light scalars + 4 coupled centre coords = 10
    assert m.num_free_params == 10


# --------------------------------------------------------------------------------
# scatter: constrained() gives absolute per-site positions, not k-vectors
# --------------------------------------------------------------------------------
def test_constrained_scatters_to_absolute_per_site_centres():
    m = _softlink_model()
    z = jnp.zeros((m.num_free_params,))
    params = m.constrained(z)
    mass = params["planes"][0]["mass"][0]
    light = params["planes"][0]["light"][0]
    # each centre coordinate is a scalar at its own site (not a shared 2-/4-vector)
    for leaf in (mass["center_x"], mass["center_y"], light["center_x"], light["center_y"]):
        assert jnp.asarray(leaf).shape == ()


def test_unconstrained_inverts_constrained_roundtrip():
    m = _softlink_model()
    z = jax.random.normal(SEED, (m.num_free_params,))
    z2 = m.unconstrained(m.constrained(z))
    assert jnp.allclose(z, z2, atol=1e-5)


def test_unconstrained_inverts_constrained_batched():
    m = _softlink_model()
    z = jax.random.normal(SEED, (7, m.num_free_params))
    z2 = m.unconstrained(m.constrained(z))
    assert z2.shape == z.shape
    assert jnp.allclose(z, z2, atol=1e-5)


def test_to_unique_regroups_coupled_members():
    # to_unique must gather the scattered per-site centres back into the group k-vector,
    # matching the pre-scatter bijector.forward output exactly.
    m = _softlink_model()
    z = jax.random.normal(SEED, (m.num_free_params,))
    unique_fwd = m.bijector.forward(z)
    unique_gathered = m.to_unique(m.constrained(z))
    assert set(unique_fwd) == set(unique_gathered)
    for k in unique_fwd:
        assert jnp.allclose(jnp.asarray(unique_fwd[k]),
                            jnp.asarray(unique_gathered[k]), atol=1e-6)


def test_constrained_equals_to_params_of_forward():
    m = _softlink_model()
    z = jax.random.normal(SEED, (m.num_free_params,))
    a = m.constrained(z)
    b = m.to_params(m.bijector.forward(z))
    assert jnp.allclose(a["planes"][0]["light"][0]["center_x"],
                        b["planes"][0]["light"][0]["center_x"])
    assert jnp.allclose(a["planes"][0]["mass"][0]["theta_E"],
                        b["planes"][0]["mass"][0]["theta_E"])


# --------------------------------------------------------------------------------
# the geometry claim: unconstrained = (mass centre, offset); constrained = absolute
# --------------------------------------------------------------------------------
def test_unconstrained_uses_relative_offset_constrained_uses_absolute():
    # Only the coupled centres are free, so z is exactly (mass_cx, mass_cy, dx, dy).
    sep, sigma_pos = 0.02, 0.3
    mass_c, light_c = soft_link(
        ("center_x", "center_y"), anchor=_anchor(sigma_pos), separation=sep)
    mass = Component(EPL(20), {
        "theta_E": 1.5, "gamma": 2.0, "e1": 0.0, "e2": 0.0, **mass_c})
    light = Component(Sersic(use_lstsq=True), {"R_sersic": 0.3, "n_sersic": 3.0, **light_c})
    m = LensModel([Plane(mass=[mass], light=[light])])
    assert m.num_free_params == 4

    unique = m.prior.sample(6000, seed=SEED)          # constrained unique dict
    z = m.bijector.inverse(unique)                     # (6000, 4) unconstrained
    zc = np.asarray(z)
    std = zc.std(0)
    # unconstrained: broad anchor (cols 0,1 ~ sigma_pos), tight offset (cols 2,3 ~ sep)
    assert np.allclose(std[:2], sigma_pos, rtol=0.15)
    assert np.allclose(std[2:], sep, rtol=0.15)
    # and the offset is decorrelated from the anchor in unconstrained space
    assert abs(np.corrcoef(zc[:, 0], zc[:, 2])[0, 1]) < 0.1

    # constrained centres: mass=(z0,z1), light=(z0+dx, z1+dy) -> separation is tight,
    # absolute positions are broad and stiffly correlated across the two bodies.
    cons = m.constrained(z)
    mcx = np.asarray(cons["planes"][0]["mass"][0]["center_x"])
    lcx = np.asarray(cons["planes"][0]["light"][0]["center_x"])
    assert (lcx - mcx).std() == pytest.approx(sep, rel=0.15)
    assert mcx.std() == pytest.approx(sigma_pos, rel=0.15)
    assert np.corrcoef(mcx, lcx)[0, 1] > 0.95


# --------------------------------------------------------------------------------
# the coupled(...) primitive: same result, hand-built offset-form distribution
# --------------------------------------------------------------------------------
def test_coupled_primitive_matches_softlink_construction():
    x0, spos, ssep = 0.0, 0.3, 0.02
    # jnp.array (not a python list) so loc/scale keep the x64 float64 dtype the model
    # promotes z to — a bare list would pin the dist to float32 and clash.
    base = tfd.MultivariateNormalDiag(
        loc=jnp.array([x0, x0, 0.0, 0.0]),
        scale_diag=jnp.array([spos, spos, ssep, ssep]))
    L = jnp.array([[1., 0., 0., 0.], [0., 1., 0., 0.],
                   [1., 0., 1., 0.], [0., 1., 0., 1.]])
    dist = tfd.TransformedDistribution(base, tfb.ScaleMatvecTriL(scale_tril=L))
    g = coupled(dist, names=["m/cx", "m/cy", "l/cx", "l/cy"])
    mass = Component(EPL(20), {
        "theta_E": 1.5, "gamma": 2.0, "e1": 0.0, "e2": 0.0,
        "center_x": g["m/cx"], "center_y": g["m/cy"]})
    light = Component(Sersic(use_lstsq=True), {
        "R_sersic": 0.3, "n_sersic": 3.0,
        "center_x": g["l/cx"], "center_y": g["l/cy"]})
    m = LensModel([Plane(mass=[mass], light=[light])])
    assert m.num_free_params == 4
    unique = m.prior.sample(4000, seed=SEED)
    cons = m.constrained(m.bijector.inverse(unique))
    mcx = np.asarray(cons["planes"][0]["mass"][0]["center_x"])
    lcx = np.asarray(cons["planes"][0]["light"][0]["center_x"])
    assert (lcx - mcx).std() == pytest.approx(ssep, rel=0.15)
    assert np.corrcoef(mcx, lcx)[0, 1] > 0.95


# --------------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------------
def test_missing_member_raises():
    g = coupled(tfd.MultivariateNormalDiag(loc=[0., 0.], scale_diag=[1., 1.]),
                names=["a", "b"])
    # only member "a" is placed; "b" is wired to nothing
    mass = Component(EPL(20), {
        "theta_E": 1.5, "gamma": 2.0, "e1": 0.0, "e2": 0.0,
        "center_x": g["a"], "center_y": tfd.Normal(0., 0.1)})
    with pytest.raises(ValueError, match="never placed at a site"):
        LensModel([Plane(mass=[mass])])


def test_reused_member_raises():
    g = coupled(tfd.MultivariateNormalDiag(loc=[0., 0.], scale_diag=[1., 1.]),
                names=["a", "b"])
    mass = Component(EPL(20), {
        "theta_E": 1.5, "gamma": 2.0, "e1": 0.0, "e2": 0.0,
        "center_x": g["a"], "center_y": g["a"]})   # member "a" at two sites, "b" nowhere
    with pytest.raises(ValueError, match="more than one site|never placed"):
        LensModel([Plane(mass=[mass])])


def test_wrong_event_shape_raises():
    with pytest.raises(ValueError, match="event_shape"):
        coupled(tfd.MultivariateNormalDiag(loc=[0., 0., 0.], scale_diag=[1., 1., 1.]),
                names=["a", "b"])   # 3-dim dist for 2 names


def test_softlink_rejects_non_normal_anchor():
    with pytest.raises(TypeError, match="tfd.Normal"):
        soft_link("center_x", anchor=tfd.Uniform(-1., 1.), separation=0.02)


def test_softlink_rejects_single_anchor_for_multiple_params():
    # a single distribution may not be broadcast across center_x AND center_y
    with pytest.raises(TypeError, match="one prior PER parameter"):
        soft_link(("center_x", "center_y"), anchor=tfd.Normal(0.0, 0.3), separation=0.02)


def test_softlink_rejects_anchor_length_mismatch():
    with pytest.raises(ValueError, match="one prior per parameter"):
        soft_link(("center_x", "center_y"),
                  anchor=(tfd.Normal(0.0, 0.3),), separation=0.02)


def test_softlink_rejects_nonpositive_separation():
    with pytest.raises(ValueError, match="separation"):
        soft_link("center_x", anchor=tfd.Normal(0., 0.3), separation=0.0)


# --------------------------------------------------------------------------------
# constrained() edge case: no free params -> constants only
# --------------------------------------------------------------------------------
def test_constrained_no_free_params_returns_constants():
    mass = Component(EPL(20), {
        "theta_E": 1.5, "gamma": 2.0, "e1": 0.0, "e2": 0.0,
        "center_x": 0.0, "center_y": 0.0})
    m = LensModel([Plane(mass=[mass])])
    assert m.bijector is None
    params = m.constrained(jnp.zeros((0,)))
    assert float(params["planes"][0]["mass"][0]["theta_E"]) == 1.5
