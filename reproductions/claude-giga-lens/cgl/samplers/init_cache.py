"""MAP+SVI init cache: computed ONCE per (T1 target, seed), reused by every
init-consuming adapter, and BILLED IDENTICALLY to each consumer (protocol,
P2b). The schedule is EXACTLY the S0 chees-mode MAP/SVI (gu-2022 recipe +
guard floor), with the same seed-derived keys S0 would use, so the billed
cost is the cost S0 itself pays inline.

CLI (one process per target; dtype/GPU pinned before jax import):

    /raid/benson/.venvs/cgl/bin/python -m cgl.samplers.init_cache \
        --target gu2022_sys006 --seed 0 --gpu 9

Writes data/init_cache/<target>_s<seed>.npz with z_map, svi loc/cov (RAW
covariance -- consumers apply guards.floor_svi_covariance, identical to S0),
gradient counts and wallclock for billing.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np

# S0 chees-mode defaults (baseline_gigalens.DEFAULT_CONFIG subset)
CACHE_CONFIG = dict(n_map=128, map_steps=250, map_lr=1e-2,
                    n_vi=200, svi_steps=None, svi_lr=1e-3, init_scales=1e-3)


def build_init_cache(target, seed: int, config: dict | None = None) -> dict:
    """Run MAP+SVI on `target` with S0's keys for `seed`; save + return."""
    import jax

    from cgl import fitting, guards
    from cgl.samplers.common import INIT_CACHE_DIR, init_cache_path
    from cgl.zoo.api import np_dtype

    cfg = {**CACHE_CONFIG, **(config or {})}
    fdtype = np_dtype(target.dtype)
    dim = target.dim
    # SAME key derivation as S0.run_cell (k_map, k_svi are splits 0,1 of 4)
    key = jax.random.PRNGKey(int(seed))
    k_map, k_svi, _, _ = jax.random.split(key, 4)

    t0 = time.time()
    z0 = target.init.prior_sample_fn(k_map, int(cfg["n_map"]))
    z_map, lp_map, _, g_map = fitting.run_map(
        target.log_prob_batch, z0, int(cfg["map_steps"]), cfg["map_lr"],
        fdtype)
    map_s = time.time() - t0

    t0 = time.time()
    svi_steps = int(cfg["svi_steps"] or max(500, 150 * dim))
    guards.check_svi_schedule(svi_steps, dim)
    svi_loc, svi_cov, best_neg_elbo, _, g_svi = fitting.run_svi(
        target.log_prob_batch, z_map, k_svi, dim, svi_steps,
        int(cfg["n_vi"]), cfg["svi_lr"], cfg["init_scales"], fdtype)
    svi_s = time.time() - t0

    cfg["svi_steps"] = svi_steps
    INIT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = init_cache_path(target.name, seed)
    np.savez_compressed(
        path, z_map=np.asarray(z_map, dtype=np.float64), lp_map=float(lp_map),
        svi_loc=np.asarray(svi_loc, dtype=np.float64),
        svi_cov=np.asarray(svi_cov, dtype=np.float64),
        best_neg_elbo=float(best_neg_elbo),
        n_grad_map=int(g_map), n_grad_svi=int(g_svi),
        map_s=float(map_s), svi_s=float(svi_s),
        config_json=json.dumps(cfg), target=target.name, seed=int(seed))
    return dict(path=str(path), lp_map=float(lp_map),
                best_neg_elbo=float(best_neg_elbo), n_grad=int(g_map + g_svi),
                map_s=map_s, svi_s=svi_s)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--gpu", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    from cgl.zoo import get_target_info
    from cgl.zoo.runtime import setup_process_env

    info = get_target_info(args.target)
    setup_process_env(info.dtype, args.gpu)

    from cgl.paths import REPRO
    from cgl.samplers.common import init_cache_path
    from cgl.zoo import get_target
    from cgl.zoo.runtime import setup_jax_cache

    if init_cache_path(args.target, args.seed).exists() and not args.force:
        print(f"init cache exists for {args.target} s{args.seed}; skipping")
        return 0
    setup_jax_cache(REPRO)
    t0 = time.time()
    target = get_target(args.target)
    print(f"built {target.name} in {time.time()-t0:.0f}s", flush=True)
    out = build_init_cache(target, args.seed)
    print(f"init cache {out['path']}: lp_map={out['lp_map']:.2f} "
          f"neg_elbo={out['best_neg_elbo']:.2f} n_grad={out['n_grad']} "
          f"(map {out['map_s']:.0f}s, svi {out['svi_s']:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
