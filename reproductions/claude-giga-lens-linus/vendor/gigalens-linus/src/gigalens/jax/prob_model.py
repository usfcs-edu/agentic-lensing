"""Legacy prob-model module — classes removed with the old gigalens API.

The ``ProbModel`` / ``ForwardProbModel`` / ``BackwardProbModel`` family was removed when
the old API was dropped; the scene path uses
``gigalens.jax.scene_prob_model.ProbModel`` instead. Only the small precision helpers
below remain (they are self-contained and were the load-bearing float64 likelihood
pieces); nothing on the working path imports this module. Restore from git b82397c if the
old classes are needed.
"""
import jax
from jax import numpy as jnp
from tensorflow_probability.substrates.jax import distributions as tfd


def _likelihood_precision(simulator):
    """Resolve the likelihood precision of ``simulator`` (default ``"float64"``)."""
    return getattr(simulator, "likelihood_precision", "float64")


def _cast_params_for_precision(params, prec):
    """Cast the (bijector-forward) params to the forward dtype for ``prec``.

    MCLMC positions are seeded from a float32 sample, so without this the
    "float64" forward would silently run in float32 (only the reduction below
    would be float64). It also keeps the basis dtype matched to a float64 PSF
    kernel, which the strict-dtype convolution in the simulator requires.
    """
    dt = jnp.float32 if prec == "float32" else jnp.float64
    return jax.tree_util.tree_map(lambda a: jnp.asarray(a).astype(dt), params)


def _independent_normal_log_prob_f64(im_masked, observed_masked, error_masked):
    """``tfd.Independent(tfd.Normal(observed, error), 1).log_prob(im)`` in float64.

    Runs the *same* likelihood expression as the float32 path, only in float64, so
    the masked-array broadcasting is identical by construction. The ~40000-pixel
    sum to a float32 scalar of magnitude ~1.5e5 quantizes logp at ~0.016 (the ulp
    staircase that breaks the EEVPD signal); accumulating in float64 removes it.
    Requires jax_enable_x64.
    """
    f64 = jnp.float64
    dist = tfd.Independent(
        tfd.Normal(observed_masked.astype(f64), error_masked.astype(f64)),
        reinterpreted_batch_ndims=1,
    )
    return dist.log_prob(im_masked.astype(f64))
