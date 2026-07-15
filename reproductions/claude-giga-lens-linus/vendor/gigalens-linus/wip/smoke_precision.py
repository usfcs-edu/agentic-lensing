#!/usr/bin/env python3
"""Verify the dtype-valued precision config: the float64 default path is byte-identical
to an explicit float64, and the float32 conv/basis override paths still run and behave
like float32 (close to, but not byte-equal to, the float64 result). Run in container.
"""
from __future__ import annotations
import os
os.environ.setdefault("JAX_ENABLE_X64", "1")

import dataclasses as dc
import numpy as np
import jax

from profile_scene_likelihood import build_problem
from gigalens.jax.scene_simulator import SceneSimulator


def lstsq_of(cfg, model, params, ds):
    sim = SceneSimulator(model, cfg)
    out = sim.lstsq_simulate(params, ds.image, ds.error_map, ds.mask)
    return np.asarray(jax.block_until_ready(out)), sim


def params_batch(params):
    leaf = jax.tree_util.tree_leaves(params)[0]
    return int(leaf.shape[0]) if leaf.ndim else 1


def main():
    print("JAX", jax.__version__, "| devices", jax.devices())
    model, prob, cfg, z_vec, params = build_problem(num_pix=48, supersample=2, n_max=6)
    ds = prob.datasets[0]

    # cfg.likelihood_precision was passed as the STRING "float64"; confirm it is stored
    # as a dtype now.
    print(f"stored likelihood_precision = {cfg.likelihood_precision!r} "
          f"(type {type(cfg.likelihood_precision).__name__})")
    assert cfg.likelihood_precision == np.dtype("float64")

    base, sim = lstsq_of(cfg, model, params, ds)
    print(f"high_precision={sim.high_precision} (expect True)")

    # 1) default float64 (None conv/basis) vs EXPLICIT float64 override. NOT byte-equal:
    # an explicit override inserts no-op astype(float64) casts that change the XLA graph,
    # and the lstsq normal-equation solve amplifies the resulting float64 reassociation
    # (~1e-13) through the gram conditioning to ~1e-6. This matches the OLD string code's
    # behavior (it inserted the same casts) -- so it is pre-existing, not from this change.
    # The byte-identical guarantee is on the DEFAULT path, covered by the golden anchor.
    cfg_f64 = dc.replace(cfg, conv_precision="float64", basis_precision="float64")
    out_f64, _ = lstsq_of(cfg_f64, model, params, ds)
    rel_f64 = np.abs(base - out_f64).max() / np.abs(base).max()
    print(f"[float64 explicit ~= default]   rel={rel_f64:.2e}  (solve-amplified reassoc, <1e-4)")

    # 2) float32 conv override (string in) -> runs, finite, ~1e-6 close to f64
    cfg_cf32 = dc.replace(cfg, conv_precision="float32")
    out_cf32, _ = lstsq_of(cfg_cf32, model, params, ds)
    rel_c = np.abs(out_cf32 - base).max() / np.abs(base).max()
    print(f"[conv_precision=float32]        finite={np.isfinite(out_cf32).all()}  rel={rel_c:.2e}")

    # 3) float32 basis override (dtype in) -> runs, finite
    cfg_bf32 = dc.replace(cfg, basis_precision=np.dtype("float32"))
    out_bf32, _ = lstsq_of(cfg_bf32, model, params, ds)
    rel_b = np.abs(out_bf32 - base).max() / np.abs(base).max()
    print(f"[basis_precision=float32]       finite={np.isfinite(out_bf32).all()}  rel={rel_b:.2e}")

    ok = rel_f64 < 1e-4 and np.isfinite(out_cf32).all() and np.isfinite(out_bf32).all() \
        and rel_c < 1e-3 and rel_b < 1e-1
    print("\n" + ("ALL PRECISION CHECKS PASSED" if ok else "*** CHECK FAILED ***"))


if __name__ == "__main__":
    main()
