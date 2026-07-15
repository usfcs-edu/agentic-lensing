"""Forward-mode shapelet rendering must work for FIXED (unbatched) amplitudes.

A shapelet source with sampled amplitudes carries a trailing sample/batch axis, and the
render contracts the shapelet order against it (einsum ``ij,i...j->...j``). A *fixed*
amplitude (e.g. rendering a known/truth source, or the ``fix_to`` path) is a scalar with
no batch axis, which used to make that einsum raise. This pins that the fixed-amplitude
render (1) works and (2) equals both the batched-amplitude render and the lstsq basis
combined with the same coefficients.
"""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from gigalens.jax.profiles.light.shapelets import Shapelets

pytestmark = pytest.mark.physics


def _grid(n=40):
    g = jnp.linspace(-1, 1, n).astype(jnp.float64)
    xx, yy = jnp.meshgrid(g, g)
    return xx[..., None], yy[..., None]


def test_fixed_amplitude_render_matches_batched_and_lstsq():
    n_max = 4
    fwd = Shapelets(n_max=n_max, use_lstsq=False, interpolate=False)
    lstsq = Shapelets(n_max=n_max, use_lstsq=True, interpolate=False)
    names = fwd._amp_names
    coeffs = np.random.default_rng(0).standard_normal(fwd.depth)
    xx, yy = _grid()
    kw = dict(center_x=0.0, center_y=0.0, beta=0.1)

    # (1) fixed (scalar) amplitudes render without error
    fixed = np.squeeze(np.asarray(fwd.light(xx, yy, **kw,
                       **{n: float(coeffs[i]) for i, n in enumerate(names)})))
    assert fixed.shape == (40, 40) and np.all(np.isfinite(fixed))

    # (2a) equals the batched-amplitude render (each amp a length-1 array = batch 1)
    batched = np.squeeze(np.asarray(fwd.light(xx, yy, **kw,
                         **{n: jnp.asarray([coeffs[i]]) for i, n in enumerate(names)})))
    np.testing.assert_allclose(fixed, batched, rtol=1e-10, atol=1e-12)

    # (2b) equals the lstsq basis contracted with the same coefficients
    basis = np.asarray(lstsq.light(xx, yy, **kw))            # (depth, 40, 40, 1)
    manual = np.squeeze((coeffs[:, None, None, None] * basis).sum(0))
    np.testing.assert_allclose(fixed, manual, rtol=1e-8, atol=1e-10)
