#!/usr/bin/env python3
"""Microbench gate: does making SceneSimulator's batch size flexible cost anything?

Compares the CURRENT simulator (grid pre-tiled to ``bs`` at __init__) against a
candidate where the grid carries a trailing SINGLETON axis and the batch is driven
by the params' batch dimension (broadcast at trace time). The simulator code is the
only thing that differs between the two runs; this script is identical for both.

For each batch size B it builds ONE ``SceneSimulator(model, cfg)``, feeds
batched params/z of batch B (with per-sample jitter so XLA can't constant-fold the
batch), and times the production work units, device-synchronized and warmed:

    lstsq_simulate(params)            # amplitude solve (lstsq mode forward)
    log_prob(z, simulator=sim)        # full masked-Gaussian likelihood
    grad(sum log_prob)(z)             # the MAP / MCMC per-step work (VJP through bs)

It saves the outputs (lstsq image, log_prob vector, grad) to an .npz tagged by
``--tag`` and records XLA ``cost_analysis`` (FLOPs / bytes accessed) for the compiled
lstsq kernel. ``--compare A.npz B.npz`` reports max abs/rel diff per array.

Run inside the canonical Shifter container (JAX 0.10) on a GPU node with the canonical
PYTHONPATH (sidecar + gigalens/src + conda site-packages).
"""
from __future__ import annotations

import os
os.environ.setdefault("JAX_ENABLE_X64", "1")

import argparse
import statistics
import time

import numpy as np
import jax
import jax.numpy as jnp

from profile_scene_likelihood import build_problem


def time_fn(fn, *args, n_warmup=3, n_iter=30):
    """Median + min ms per call, device-synchronized."""
    jax.block_until_ready(fn(*args))
    for _ in range(n_warmup - 1):
        jax.block_until_ready(fn(*args))
    ts = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        ts.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(ts), min(ts)


def lstsq_cost(fn, params):
    """XLA static cost of the compiled lstsq kernel (FLOPs, GB accessed)."""
    comp = jax.jit(fn).lower(params).compile()
    ca = comp.cost_analysis()
    if isinstance(ca, (list, tuple)):
        ca = ca[0]
    if not isinstance(ca, dict):
        return float("nan"), float("nan")
    flops = float(ca.get("flops", float("nan")))
    gbytes = float(ca.get("bytes accessed", float("nan"))) / 1e9
    return flops, gbytes


def run(args):
    from gigalens.jax.scene_simulator import SceneSimulator

    print("JAX", jax.__version__, "| x64:", jax.config.jax_enable_x64,
          "| devices:", jax.devices())
    model, prob, cfg, z_vec, _ = build_problem(
        num_pix=args.num_pix, supersample=args.supersample, n_max=args.n_max)
    n_params = int(z_vec.shape[0])
    ds = prob.datasets[0]
    img, err, mask = ds.image, ds.error_map, ds.mask
    print(f"tag={args.tag} | grid={args.num_pix*args.supersample}^2 | "
          f"depth={prob.simulators[0].depth} | n_params={n_params}")

    # Deterministic batched z (same across both runs): fixed base + fixed jitter, so
    # the batch elements differ (defeats constant-folding) yet are reproducible.
    rng = np.random.default_rng(0)

    saved = {}
    print(f"\n{'bs':>5}{'grid_axis':>11}{'lstsq ms':>10}{'logp ms':>9}"
          f"{'grad ms':>9}{'lstsq GFLOP':>12}{'lstsq GB':>9}")
    print("-" * 65)
    for B in args.bs:
        sim = SceneSimulator(model, cfg)
        z_batch = jnp.asarray(
            np.asarray(z_vec)[None, :] + 0.01 * rng.standard_normal((B, n_params)))
        params_b = model.to_params(model.bijector.forward(list(z_batch.T)))

        f_lstsq = lambda p: sim.lstsq_simulate(p, img, err, mask)
        f_logp = lambda z: prob.log_prob(z, simulator=sim)[0]
        f_grad = jax.grad(lambda z: jnp.sum(prob.log_prob(z, simulator=sim)[0]))

        out_lstsq = f_lstsq(params_b)
        out_logp = f_logp(z_batch)
        out_grad = f_grad(z_batch)
        jax.block_until_ready((out_lstsq, out_logp, out_grad))

        t_lstsq, _ = time_fn(f_lstsq, params_b, n_iter=args.n_iter)
        t_logp, _ = time_fn(f_logp, z_batch, n_iter=args.n_iter)
        t_grad, _ = time_fn(f_grad, z_batch, n_iter=args.n_iter)
        flops, gb = lstsq_cost(f_lstsq, params_b)

        print(f"{B:>5}{str(sim.img_X.shape[-1]):>11}{t_lstsq:>10.3f}{t_logp:>9.3f}"
              f"{t_grad:>9.3f}{flops/1e9:>12.2f}{gb:>9.3f}")

        saved[f"lstsq_bs{B}"] = np.asarray(out_lstsq)
        saved[f"logp_bs{B}"] = np.asarray(out_logp)
        saved[f"grad_bs{B}"] = np.asarray(out_grad)
        saved[f"flops_bs{B}"] = np.asarray(flops)
        saved[f"gb_bs{B}"] = np.asarray(gb)

    out_path = os.path.join(args.out_dir, f"{args.tag}.npz")
    os.makedirs(args.out_dir, exist_ok=True)
    np.savez(out_path, **saved)
    print(f"\nsaved outputs -> {out_path}")


def compare(path_a, path_b):
    a = np.load(path_a)
    b = np.load(path_b)
    keys = [k for k in a.files if not (k.startswith("flops") or k.startswith("gb"))]
    print(f"\nNUMERICAL COMPARISON  {os.path.basename(path_a)} vs {os.path.basename(path_b)}")
    print(f"{'array':>14}{'max|abs|':>14}{'max|rel|':>14}{'allclose 1e-10':>16}")
    print("-" * 58)
    worst_rel = 0.0
    for k in keys:
        x, y = a[k], b[k]
        if x.shape != y.shape:
            print(f"{k:>14}  SHAPE MISMATCH {x.shape} vs {y.shape}")
            continue
        absd = np.abs(x - y)
        denom = np.maximum(np.abs(x), np.abs(y))
        reld = np.where(denom > 0, absd / denom, 0.0)
        ac = bool(np.allclose(x, y, rtol=1e-10, atol=1e-12))
        worst_rel = max(worst_rel, float(reld.max()))
        print(f"{k:>14}{absd.max():>14.3e}{reld.max():>14.3e}{str(ac):>16}")
    print("-" * 58)
    print(f"worst relative diff across all arrays: {worst_rel:.3e}")
    print("\nCOST (should be ~identical if same kernel):")
    print(f"{'key':>14}{'A':>16}{'B':>16}")
    for k in [f for f in a.files if f.startswith(("flops", "gb"))]:
        print(f"{k:>14}{float(a[k]):>16.4g}{float(b[k]):>16.4g}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-pix", type=int, default=60)
    ap.add_argument("--supersample", type=int, default=2)
    ap.add_argument("--n-max", type=int, default=8)
    ap.add_argument("--n-iter", type=int, default=30)
    ap.add_argument("--bs", type=int, nargs="+", default=[1, 32, 128, 256])
    ap.add_argument("--tag", default="tiled")
    ap.add_argument("--out-dir", default="wip/bench_out")
    ap.add_argument("--compare", nargs=2, metavar=("A.npz", "B.npz"))
    args = ap.parse_args()
    if args.compare:
        compare(*args.compare)
    else:
        run(args)


if __name__ == "__main__":
    main()
