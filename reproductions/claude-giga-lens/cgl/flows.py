"""Normalizing-flow helpers (flowjax 19) for the S8 neutra / S9 glnt adapters.

Conventions:
  * flowjax 19 requires NEW-STYLE typed PRNG keys (jax.random.key); passing
    legacy uint32 PRNGKey raises TypeError (P0 stage log).
  * Flows are ALWAYS fit on standardized data ((x - mean) / std); the affine
    de-standardization is part of the FlowBundle transform and its constant
    log-det is included, so pullback densities are exact.
  * The flow itself is a coupling NSF (RationalQuadraticSpline transformer):
    analytic + fast in BOTH directions, which the NeuTra pullback needs
    (u -> x per leapfrog step inside the kernel).
  * Flow training/pushforward performs ZERO target evaluations; adapters
    ledger it as wallclock only (per the P2b budget convention).
  * dtype follows the process x64 state (single-dtype-per-process): under
    GIGALENS_X64=1 the flow weights and transforms are float64.

Pullback density (the NeuTra construction; unit-tested in
tests/test_samplers.py against numerical Jacobians):

    logp_u(u) = logp_target(T(u)) + log|det J_T(u)|,   T(u) = mean + std * f(u)
"""
from __future__ import annotations

import dataclasses
import time
from typing import Any, Callable, Optional

import numpy as np


@dataclasses.dataclass
class FlowBundle:
    """A fitted flow + its standardization affine + training record."""
    flow: Any                       # flowjax Transformed distribution
    mean: np.ndarray                # (dim,) float64
    std: np.ndarray                 # (dim,) float64
    losses: dict                    # {"train": [...], "val": [...]}
    fit_seconds: float
    config: dict

    # ---- u -> (x, logdet) on (N, dim) batches -------------------------------
    def forward_batch(self, U):
        import jax
        import jax.numpy as jnp

        bij = self.flow.bijection

        def one(u):
            x, ld = bij.transform_and_log_det(u)
            return x, ld

        xs, ld = jax.vmap(one)(U)
        mean = jnp.asarray(self.mean, dtype=U.dtype)
        std = jnp.asarray(self.std, dtype=U.dtype)
        x = mean[None, :] + std[None, :] * xs
        return x, ld + jnp.sum(jnp.log(std))

    def make_pullback(self, log_prob_batch: Callable) -> Callable:
        """(N, dim) u-space -> (N,) pullback log-density."""
        def pullback(U):
            x, ld = self.forward_batch(U)
            return log_prob_batch(x) + ld

        return pullback

    def push_samples(self, U_tc: np.ndarray, chunk: int = 4096) -> np.ndarray:
        """(T, C, dim) u-draws -> (T, C, dim) target-space draws (chunked)."""
        import jax.numpy as jnp

        T, C, dim = U_tc.shape
        flat = np.asarray(U_tc).reshape(T * C, dim)
        out = np.empty_like(flat, dtype=np.float64)
        for i in range(0, flat.shape[0], chunk):
            x, _ = self.forward_batch(jnp.asarray(flat[i:i + chunk]))
            out[i:i + chunk] = np.asarray(x, dtype=np.float64)
        return out.reshape(T, C, dim)


def fit_nsf(seed: int, samples: np.ndarray, *, flow_layers: int = 6,
            knots: int = 8, interval: float = 5.0, nn_width: int = 64,
            nn_depth: int = 1, max_epochs: int = 100, max_patience: int = 8,
            batch_size: int = 256, learning_rate: float = 3e-4,
            val_prop: float = 0.1, show_progress: bool = False) -> FlowBundle:
    """Fit a coupling NSF to (N, dim) samples (standardized internally)."""
    import jax
    import jax.numpy as jnp
    from flowjax.bijections import RationalQuadraticSpline
    from flowjax.distributions import Normal
    from flowjax.flows import coupling_flow
    from flowjax.train import fit_to_data

    cfg = dict(flow_layers=flow_layers, knots=knots, interval=interval,
               nn_width=nn_width, nn_depth=nn_depth, max_epochs=max_epochs,
               max_patience=max_patience, batch_size=batch_size,
               learning_rate=learning_rate, val_prop=val_prop)
    x = np.asarray(samples, dtype=np.float64)
    n, dim = x.shape
    mean = x.mean(axis=0)
    std = np.maximum(x.std(axis=0), 1e-12)
    xs = (x - mean) / std

    key = jax.random.key(int(seed))            # typed key (flowjax 19)
    flow_key, train_key = jax.random.split(key)
    flow = coupling_flow(
        flow_key,
        base_dist=Normal(jnp.zeros(dim)),
        transformer=RationalQuadraticSpline(knots=int(knots),
                                            interval=float(interval)),
        flow_layers=int(flow_layers), nn_width=int(nn_width),
        nn_depth=int(nn_depth),
    )
    t0 = time.time()
    flow, losses = fit_to_data(
        train_key, flow, jnp.asarray(xs),
        max_epochs=int(max_epochs), max_patience=int(max_patience),
        batch_size=min(int(batch_size), max(32, n // 2)),
        learning_rate=float(learning_rate), val_prop=float(val_prop),
        show_progress=show_progress,
    )
    fit_s = time.time() - t0
    losses = {k: [float(v) for v in vs] for k, vs in losses.items()}
    if not np.all(np.isfinite(losses.get("train", [0.0]))):
        raise RuntimeError("NSF fit produced non-finite training loss")
    return FlowBundle(flow=flow, mean=mean, std=std, losses=losses,
                      fit_seconds=fit_s, config=cfg)
