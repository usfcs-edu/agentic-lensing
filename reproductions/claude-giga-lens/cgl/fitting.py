"""Generic MAP / SVI phases against the zoo `log_prob_batch` surface.

Faithful reimplementation of the vendored gigalens ModellingSequence.MAP/SVI
(single device, no shard_map) so any zoo target -- not just gigalens-built
ones -- can run the published recipe. Used by cgl/samplers/baseline_gigalens
(S0) and reusable by the P2b neural-transport / SMC adapters.

Fidelity notes:
  * MAP: adabelief multistart on unconstrained z; per-step per-sample best
    tracked exactly like gigalens output_type="best". Loss is -mean(logp)
    WITHOUT the gigalens 1/n_pixels factor (no image on the generic surface;
    adabelief's per-parameter adaptive scaling absorbs constants).
  * SVI: full-rank MVN q via the same FillScaleTriL(Exp, shift=1e-6)
    parameterization, init_scales on the diagonal, best-loss iterate kept.
  * Gradient counts are returned analytically (steps x batch).
"""
from __future__ import annotations

import numpy as np


def run_map(log_prob_batch, z0, steps: int, lr: float, fdtype):
    """Multistart MAP. z0: (K, dim) starts. Returns
    (z_best (dim,) np, lp_best float, lp_hist (steps,) np, n_grad)."""
    import jax
    import jax.numpy as jnp
    import optax

    z0 = jnp.asarray(z0, dtype=fdtype)
    K = int(z0.shape[0])
    # Warm the target's per-batch-size machinery EAGERLY (gigalens targets
    # build a LensSimulator per batch size; constructing one inside the scan
    # trace is a TracerArrayConversionError). Also a finite-start sanity eval.
    lp0 = log_prob_batch(z0)
    if not bool(np.any(np.isfinite(np.asarray(lp0)))):
        raise RuntimeError("run_map: no finite log-prob among the starts")
    opt = optax.adabelief(lr, b1=0.95, b2=0.99)

    def loss_fn(Z):
        lp = log_prob_batch(Z)
        return -jnp.mean(lp), lp

    loss_grad = jax.value_and_grad(loss_fn, has_aux=True)

    def one_step(carry, _):
        Z, opt_state, best_z, best_lp = carry
        (_, lp), g = loss_grad(Z)
        updates, opt_state = opt.update(g, opt_state)
        Z = optax.apply_updates(Z, updates)
        i = jnp.nanargmax(lp)
        better = lp[i] > best_lp
        best_z = jnp.where(better, Z[i], best_z)
        best_lp = jnp.where(better, lp[i], best_lp)
        return (Z, opt_state, best_z, best_lp), jnp.nanmax(lp)

    @jax.jit
    def run(Z0):
        carry = (Z0, opt.init(Z0), Z0[0], jnp.asarray(-jnp.inf, dtype=fdtype))
        (_, _, best_z, best_lp), lp_hist = jax.lax.scan(
            one_step, carry, None, length=steps)
        return best_z, best_lp, lp_hist

    best_z, best_lp, lp_hist = run(z0)
    best_z.block_until_ready()
    return (np.asarray(best_z), float(best_lp), np.asarray(lp_hist),
            steps * K)


def run_svi(log_prob_batch, z_map, key, dim: int, steps: int, n_vi: int,
            lr: float, init_scales: float, fdtype):
    """Full-rank MVN SVI from a MAP point. Returns
    (loc (dim,) f64, cov (dim,dim) f64, best_neg_elbo, loss_hist, n_grad)."""
    import jax
    import jax.numpy as jnp
    import optax
    import tensorflow_probability.substrates.jax as tfp

    tfd, tfb = tfp.distributions, tfp.bijectors

    # diag_shift must carry the target dtype: tfp infers the bijector dtype
    # from it, and a python float means float32 (breaks f64 targets).
    cov_bij = tfb.FillScaleTriL(
        diag_bijector=tfb.Exp(),
        diag_shift=np.asarray(1e-6, dtype=np.dtype(jnp.dtype(fdtype))))
    scale0 = jnp.eye(dim, dtype=fdtype) * jnp.asarray(init_scales, dtype=fdtype)
    params0 = jnp.concatenate([jnp.asarray(z_map, dtype=fdtype),
                               cov_bij.inverse(scale0)])
    # eager warmup of the (n_vi, dim) batch path (see run_map note)
    log_prob_batch(jnp.broadcast_to(jnp.asarray(z_map, dtype=fdtype),
                                    (n_vi, dim)))
    opt = optax.adabelief(lr, b1=0.95, b2=0.99)

    def neg_elbo(params, k):
        loc = params[:dim]
        tril = cov_bij.forward(params[dim:])
        qz = tfd.MultivariateNormalTriL(loc=loc, scale_tril=tril)
        z = qz.sample(n_vi, seed=k)
        return jnp.mean(qz.log_prob(z) - log_prob_batch(z))

    neg_elbo_grad = jax.value_and_grad(neg_elbo)

    def one_step(carry, _):
        params, opt_state, k, best_params, best_loss = carry
        k, sub = jax.random.split(k)
        loss, g = neg_elbo_grad(params, sub)
        better = loss < best_loss
        best_params = jax.lax.select(better, params, best_params)
        best_loss = jax.lax.select(better, loss, best_loss)
        updates, opt_state = opt.update(g, opt_state)
        params = optax.apply_updates(params, updates)
        return (params, opt_state, k, best_params, best_loss), loss

    @jax.jit
    def run(params0, key):
        carry = (params0, opt.init(params0), key, params0,
                 jnp.asarray(jnp.inf, dtype=fdtype))
        (_, _, _, best_params, best_loss), loss_hist = jax.lax.scan(
            one_step, carry, None, length=steps)
        return best_params, best_loss, loss_hist

    best_params, best_loss, loss_hist = run(params0, key)
    best_params.block_until_ready()
    loc = np.asarray(best_params[:dim], dtype=np.float64)
    tril = np.asarray(cov_bij.forward(best_params[dim:]), dtype=np.float64)
    return (loc, tril @ tril.T, float(best_loss), np.asarray(loss_hist),
            steps * n_vi)
