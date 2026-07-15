#!/usr/bin/env python3
"""End-to-end smoke test for the ModellingSequence rewiring (prob.log_prob(z), no
make_lens_sim). Confirms MAP / SVI / HMC run through the batch-flexible per-dataset
simulators, and that a TWO-dataset ProbModel's log_prob is now reachable (the multi
-dataset `sees` loop), which the old explicit-single-simulator injection forbade.

Run in the canonical Shifter container on a GPU node.
"""
from __future__ import annotations
import os
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import jax
import jax.numpy as jnp
import optax
import tensorflow_probability.substrates.jax as tfp

from profile_scene_likelihood import build_problem
from gigalens.jax.inference import ModellingSequence
from gigalens.jax.scene_prob_model import ImageData, ProbModel

tfd = tfp.distributions


def main():
    print("JAX", jax.__version__, "| devices", jax.devices())
    model, prob, cfg, z_vec, _ = build_problem(num_pix=48, supersample=2, n_max=6)
    seq = ModellingSequence.from_scene(model, prob, cfg)

    # --- MAP (batched n_samples through prob.log_prob) ---
    z_map, lp_map, _ = seq.MAP(optax.adam(1e-2), n_samples=64, num_steps=15,
                               seed=0, output_type="best", pbar_interval=0)
    z_map = np.asarray(z_map)
    print(f"MAP   ok: z_map shape {z_map.shape}, lp finite={np.isfinite(np.asarray(lp_map)).all()}")

    # --- SVI (n_vi through prob.log_prob) ---
    q_z, loss_hist = seq.SVI(start=jnp.asarray(z_map), optimizer=optax.adam(1e-2),
                             n_vi=64, num_steps=15, seed=0, pbar_interval=0)
    print(f"SVI   ok: final -ELBO {float(np.asarray(loss_hist)[-1]):.3f}, "
          f"finite={np.isfinite(np.asarray(loss_hist)).all()}")

    # Sanitize surrogate before HMC (host rebuild -> drops shard_map sharding tag).
    # Symmetrize + jitter so the short-run cov is PD for Cholesky (test artifact, not
    # an inference concern -- a real SVI run uses hundreds of steps).
    loc = np.asarray(jax.device_get(q_z.mean()))
    cov = np.asarray(jax.device_get(q_z.covariance()))
    cov = 0.5 * (cov + cov.T) + 1e-6 * np.eye(cov.shape[0])
    q_z = tfd.MultivariateNormalTriL(loc=jnp.asarray(loc),
                                     scale_tril=jnp.asarray(np.linalg.cholesky(cov)))

    # --- HMC (n_hmc through prob.log_prob) ---
    samples = seq.HMC(q_z, n_hmc=8, num_burnin_steps=8, num_results=8,
                      seed=0, pbar_interval=0)
    samples = np.asarray(samples)
    print(f"HMC   ok: samples shape {samples.shape}, finite={np.isfinite(samples).all()}")

    # --- multi-dataset reachability: 2-band lstsq ProbModel, log_prob(z) directly ---
    # Two datasets that each see ALL light (lstsq amplitudes solved per band, so flux
    # sharing is allowed). Build from the same model + cfg; jitter the second image.
    ds0 = prob.datasets[0]
    img2 = np.asarray(ds0.image) + 0.001 * np.random.default_rng(1).standard_normal(ds0.image.shape)
    d_a = ImageData(np.asarray(ds0.image), cfg, error_map=np.asarray(ds0.error_map), sees="all")
    d_b = ImageData(img2, cfg, error_map=np.asarray(ds0.error_map), sees="all")
    prob2 = ProbModel(model, [d_a, d_b], mode="lstsq")
    z_batch = jnp.asarray(np.asarray(z_vec)[None, :] + 0.01 *
                          np.random.default_rng(0).standard_normal((4, z_vec.shape[0])))
    lp2, rc2 = prob2.log_prob(z_batch)          # simulator=None -> loops prob2.simulators
    lp1, rc1 = prob.log_prob(z_batch)           # single-band reference
    print(f"MULTI ok: n_sims={len(prob2.simulators)}, lp2 shape {lp2.shape}, "
          f"finite={np.isfinite(np.asarray(lp2)).all()}; "
          f"2-band logp < 1-band? {bool(np.all(np.asarray(lp2) < np.asarray(lp1)))}")

    # --- multi-dataset MAP end-to-end (exercises num_pixels normalization) ---
    seq2 = ModellingSequence.from_scene(model, prob2, cfg)
    z_map2, lp_map2, _ = seq2.MAP(optax.adam(1e-2), n_samples=32, num_steps=10,
                                  seed=0, output_type="best", pbar_interval=0)
    print(f"MULTI-MAP ok: num_pixels={prob2.num_pixels} (== 2x {prob.num_pixels} single), "
          f"z shape {np.asarray(z_map2).shape}, lp finite={np.isfinite(np.asarray(lp_map2)).all()}")
    print("\nALL SMOKE CHECKS PASSED")


if __name__ == "__main__":
    main()
