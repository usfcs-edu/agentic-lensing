"""DiskBijector / DiskEllipticity: the disk-bounded ellipticity grouped prior.

Checks the map itself (round-trip, analytic log-det vs autodiff, hard cap) and its
integration as a scene tuple-key prior (constraint on arbitrary z, num_free_params,
z-column labels, round-trip, and — the load-bearing one — that the scene bijector's
forward_log_det_jacobian through Split/Reshape/pack + DiskBijector equals autodiff).
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest
from tensorflow_probability.substrates.jax import distributions as tfd

from gigalens.jax.scene import Component, Plane, LensModel
from gigalens.jax.grouped_priors import DiskBijector, DiskEllipticity
from gigalens.jax.profiles.mass.epl import EPL

KEY = jax.random.PRNGKey(0)
R = 0.5
Q_FLOOR = (1 - R) / (1 + R)


# --------------------------------------------------------------------------------
# the bijector
# --------------------------------------------------------------------------------
def test_disk_bijector_roundtrip_and_cap():
    b = DiskBijector(R)
    u = jax.random.normal(KEY, (200, 2)) * 5.0        # large -> pushes the boundary
    e = b.forward(u)
    c = jnp.sqrt(jnp.sum(e ** 2, axis=-1))
    assert jnp.all(c < R)                              # hard cap, always
    assert jnp.allclose(b.inverse(e), u, atol=1e-6)   # inverse recovers u


def test_disk_bijector_inverse_uncached():
    # inverse on constructed values (NOT via forward, so no bijector-cache hit) actually
    # runs _inverse — the path that hits jnp.maximum/clip and that bij.inverse(true_dict)
    # takes in practice.
    b = DiskBijector(R)
    e = jnp.array([[0.1, -0.2], [0.3, 0.0], [-0.45, 0.1]])
    u = b.inverse(e)
    assert jnp.allclose(b.forward(u), e, atol=1e-6)


def test_disk_bijector_logdet_matches_autodiff():
    b = DiskBijector(R)
    for u in jax.random.normal(KEY, (6, 2)) * 2.0:
        jac = jax.jacfwd(b.forward)(u)
        _, ld = jnp.linalg.slogdet(jac)
        assert jnp.allclose(ld, b.forward_log_det_jacobian(u), atol=1e-6)


def test_disk_bijector_regular_at_origin():
    b = DiskBijector(R)
    # e ~ R*u near 0 (no singularity, unlike a polar (c, phi) parameterization)
    small = jnp.array([1e-4, -2e-4])
    assert jnp.allclose(b.forward(small), R * small, rtol=1e-3, atol=1e-8)


def test_bad_radius_raises():
    with pytest.raises(ValueError, match="radius"):
        DiskBijector(0.0)


# --------------------------------------------------------------------------------
# the distribution
# --------------------------------------------------------------------------------
def test_disk_ellipticity_samples_inside_and_centering():
    de = DiskEllipticity(e_max=R, scale=0.3, mode=(0.2, -0.1))
    s = de.sample(500, seed=KEY)
    assert s.shape == (500, 2)
    assert jnp.all(jnp.sqrt(jnp.sum(s ** 2, axis=-1)) < R)
    # the mode maps back to the requested ellipticity
    e0 = de.bijector.forward(de.distribution.loc)
    assert jnp.allclose(e0, jnp.array([0.2, -0.1]), atol=1e-6)
    assert jnp.all(jnp.isfinite(de.log_prob(s)))


def test_mode_outside_disk_raises():
    with pytest.raises(ValueError, match="strictly inside"):
        DiskEllipticity(e_max=0.3, mode=(0.4, 0.0))


# --------------------------------------------------------------------------------
# scene integration
# --------------------------------------------------------------------------------
def _f64(x):
    return jnp.asarray(x, jnp.float64)


def _disk_model(e_max=R, scale=0.25, mode=(0.0, 0.0)):
    # A float64 model (x64 on): DiskEllipticity defaults to the ambient float (float64),
    # so the scalar priors are float64 too for dtype-uniformity — mirrors real models
    # (e.g. sim_carousel) and guards the reported float32/float64 concat crash.
    comp = Component(EPL(20), {
        "theta_E": tfd.Normal(_f64(1.5), _f64(0.1)),
        "gamma": tfd.Uniform(_f64(1.5), _f64(2.5)),
        ("e1", "e2"): DiskEllipticity(e_max=e_max, scale=scale, mode=mode),
        "center_x": tfd.Normal(_f64(0.), _f64(0.1)),
        "center_y": tfd.Normal(_f64(0.), _f64(0.1))})
    return LensModel([Plane(mass=[comp])])


def test_disk_model_uniform_float64_inverse():
    """Repro guard: bij.inverse(prior.sample()) on a float64 model with DiskEllipticity
    must not choke concatenating a float32 group with float64 scalars."""
    m = _disk_model()
    z = m.bijector.inverse(m.prior.sample(64, seed=KEY))
    assert z.shape == (64, 6) and z.dtype == jnp.float64


def test_disk_ellipticity_dtype_follows_x64_and_override():
    """Under x64 the default dtype is float64 (so it matches typical float64 models);
    an explicit dtype overrides. Guards the float32/float64 concat crash at its root."""
    assert DiskEllipticity(e_max=0.5).sample(seed=KEY).dtype == jnp.float64
    assert DiskEllipticity(e_max=0.5, dtype=jnp.float32).sample(seed=KEY).dtype == jnp.float32


def test_scene_disk_constraint_for_arbitrary_z():
    m = _disk_model()
    assert m.num_free_params == 6
    z = jax.random.normal(KEY, (300, m.num_free_params)) * 6.0   # far into unconstrained space
    p = m.to_params(m.bijector.forward(z))["planes"][0]["mass"][0]
    assert np.asarray(p["e1"]).shape == (300,)                   # profile sees scalars
    c = jnp.sqrt(p["e1"] ** 2 + p["e2"] ** 2)
    assert jnp.all(c < R)                                         # never trips |e|>=R
    q = (1 - c) / (1 + c)
    assert jnp.all(q > Q_FLOOR - 1e-9)


def test_scene_disk_names_and_roundtrip():
    m = _disk_model()
    assert "planes/0/mass/0/e1" in m.z_param_names
    assert "planes/0/mass/0/e2" in m.z_param_names
    start = m.prior.sample(4, seed=KEY)
    z = m.bijector.inverse(start)
    assert z.shape == (4, 6)
    assert jnp.allclose(m.bijector.inverse(m.bijector.forward(z)), z, atol=1e-6)


def test_scene_fldj_matches_autodiff():
    """The whole forward log-det (ev-space bijector incl. DiskBijector, through the
    volume-preserving Split/Reshape/pack) equals slogdet of the autodiff Jacobian."""
    m = _disk_model()
    z = jax.random.normal(KEY, (m.num_free_params,))

    def f(zz):  # flat z -> flat constrained vector (dim-preserving => square Jacobian)
        x = m.bijector.forward(zz)
        return jnp.concatenate([jnp.reshape(x[k], (-1,)) for k in sorted(x)])

    _, logdet = jnp.linalg.slogdet(jax.jacfwd(f)(z))
    assert jnp.allclose(logdet, m.bijector.forward_log_det_jacobian(z), atol=1e-5)


def test_scene_disk_logprob_finite():
    m = _disk_model()  # float64 model; float64 z matches
    z = jax.random.normal(KEY, (5, m.num_free_params)) * 3.0
    x = m.bijector.forward(z)
    lp = m.prior.log_prob(x) + m.bijector.forward_log_det_jacobian(z)
    assert lp.shape == (5,)
    assert jnp.all(jnp.isfinite(lp))


def _full_prior_logprob(m, n=8):
    """Mirror ProbModel.log_prior: inverse (promotes) -> forward -> native-dtype cast
    -> prior.log_prob + fldj. Returns the log-prior; raises on any dtype clash."""
    z = m.bijector.inverse(m.prior.sample(n, seed=KEY))          # promotes mixed -> wider
    x = m.bijector.forward(z)
    return (z, m.prior.log_prob(m.cast_free_to_native(x))
            + m.bijector.forward_log_det_jacobian(z))


def test_mixed_float32_scalars_float64_group_full_path():
    """The reported MAP case: float32 scalar priors + a float64 DiskEllipticity. inverse
    promotes to float64 and the full log-prior path stays finite."""
    comp = Component(EPL(20), {
        "theta_E": tfd.Normal(1.5, 0.1), "gamma": tfd.Uniform(1.5, 2.5),    # float32
        ("e1", "e2"): DiskEllipticity(e_max=0.5),                            # float64 (x64)
        "center_x": tfd.Normal(0., 0.1), "center_y": tfd.Normal(0., 0.1)})   # float32
    z, lp = _full_prior_logprob(LensModel([Plane(mass=[comp])]))
    assert z.dtype == jnp.float64 and jnp.all(jnp.isfinite(lp))


def test_mixed_float32_group_needs_native_cast():
    """Reverse/pathological: float64 scalars + a float32 (MVNDiag-strict) group. The flat
    z promotes to float64; the native-dtype cast is what lets prior.log_prob succeed."""
    comp = Component(EPL(20), {
        "theta_E": tfd.Normal(_f64(1.5), _f64(0.1)),
        "gamma": tfd.Uniform(_f64(1.5), _f64(2.5)),
        ("e1", "e2"): DiskEllipticity(e_max=0.5, dtype=jnp.float32),
        "center_x": tfd.Normal(_f64(0.), _f64(0.1)),
        "center_y": tfd.Normal(_f64(0.), _f64(0.1))})
    _, lp = _full_prior_logprob(LensModel([Plane(mass=[comp])]))
    assert jnp.all(jnp.isfinite(lp))
