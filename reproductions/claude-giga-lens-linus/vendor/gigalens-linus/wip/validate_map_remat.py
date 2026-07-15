#!/usr/bin/env python3
"""Gates for MAP-scoped remat (ProbModel.with_map_remat + inference.MAP wiring).
Checkpoint 2026-07-09 (GIGALens-Code docs/logs/compute-profiling.md).

  G1 twin identity : with_map_remat twin vs base at (200,ss2,15): f64 grad L2
                     <= 1e-12; production-f32 grad L2 <= 1e-5; finite.
  G2 ss>=3 no-op   : at (120,ss4,30) with_map_remat() returns SELF (identity) —
                     the +23% fused-recompute regime keeps the user's config.
  G3 memory wiring : MAP-loss grad at bs=16 via the twin: XLA static peak
                     ~4934 MB (C-16 remat row) vs base ~9197 MB; falsifier:
                     twin peak > 0.7x base => wiring failed.
  G4 end-to-end    : ModellingSequence.MAP(n_samples=48, num_steps=3) on a
                     default (remat_basis=False) config MUST run: base needs
                     ~28 GB static at bs=48 (bs=32 already OOM'd at 18.4 GB in
                     C-16) — success proves MAP executes rematerialized.
                     Falsifier: OOM with >= 20 GB device free => wiring bug
                     (with < 20 GB free, record contention and rely on G3).
"""
import os
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from validate_map_throughput import (build_problem, logp_fn, grad_l2,
                                     xla_peak_mb, _free_gpu_mb)

import json
import numpy as np
import jax
import jax.numpy as jnp
from jax import random as jr
import optax

from gigalens.jax.inference import ModellingSequence


def main():
    print("device:", jax.devices()[0], "| jax", jax.__version__)
    out = {}

    print("\n== G1: twin identity @ (200,ss2,15) ==")
    g1 = {}
    for conv, gate in [(None, 1e-12), ("float32", 1e-5)]:
        model, prob, cfg, z0 = build_problem(200, 2, 15, conv_precision=conv)
        twin = prob.with_map_remat()
        assert twin is not prob and twin.simulators[0].remat_basis, "twin not built"
        draws = [jnp.stack(model.bijector.inverse(model.prior.sample(seed=jr.PRNGKey(900 + s))))
                 for s in range(3)]
        gb = jax.jit(jax.grad(logp_fn(prob)))
        gt = jax.jit(jax.grad(logp_fn(twin)))
        worst = max(grad_l2(gt(z), gb(z)) for z in draws)
        fin = all(np.all(np.isfinite(np.asarray(gt(z)))) for z in draws)
        ok = worst <= gate and fin
        print(f"  conv={conv}: grad L2 {worst:.2e} (gate {gate:.0e}) finite={fin} "
              f"-> {'PASS' if ok else 'FAIL'}")
        g1[str(conv)] = dict(grad=worst, finite=fin, gate_pass=ok)
    out["g1"] = g1

    print("\n== G2: ss>=3 no-op ==")
    prob4 = build_problem(120, 4, 30)[1]
    noop = prob4.with_map_remat() is prob4
    print(f"  with_map_remat() is prob (ss=4): {noop} -> {'PASS' if noop else 'FAIL'}")
    out["g2"] = dict(noop=noop)

    print("\n== G3: memory wiring @ bs=16 MAP loss ==")
    model, prob, cfg, z1v = build_problem(200, 2, 15)
    twin = prob.with_map_remat()

    def mk_loss(p):
        def loss(z):
            lp, _ = p.log_prob(z)
            return -jnp.mean(lp) / p.num_pixels
        return loss

    z16 = jnp.tile(z1v[None, :], (16, 1)) + 1e-3 * jr.normal(jr.PRNGKey(2), (16, z1v.shape[0]))
    peak_base = xla_peak_mb(jax.jit(jax.grad(mk_loss(prob))), z16)
    peak_twin = xla_peak_mb(jax.jit(jax.grad(mk_loss(twin))), z16)
    ok3 = peak_twin < 0.7 * peak_base
    print(f"  base {peak_base:.0f} MB vs twin {peak_twin:.0f} MB "
          f"(C-16 expect ~9197 vs ~4934) -> {'PASS' if ok3 else 'FAIL'}")
    out["g3"] = dict(peak_base=peak_base, peak_twin=peak_twin, gate_pass=ok3)

    print("\n== G4: end-to-end MAP @ n_samples=48 with numeric peak gate ==")
    # Grading fix: on an idle 40GB device the UNREMAT bs=48 (~27.6 GB static)
    # would also run — success alone is not proof of remat. Gate the actual
    # device peak during MAP at < 20 GB (remat expectation ~14.8 GB = 3x the
    # C-16 bs=16 twin peak; base ~27.6 GB; >= 5 GB margin to both).
    assert len(jax.devices()) == 1, "G4 premise requires a single device"
    free = _free_gpu_mb()
    print(f"  device free before: {free:.0f} MB")
    mseq = ModellingSequence(prob)  # default config: remat_basis=False
    try:
        samples, lps, chisqs = mseq.MAP(optax.adam(3e-3), n_samples=48,
                                        num_steps=3, pbar_interval=0)
        stats = jax.local_devices()[0].memory_stats() or {}
        peak_gb = stats.get("peak_bytes_in_use", float("nan")) / 1e9
        fin = bool(np.all(np.isfinite(np.asarray(lps))))
        ok4 = fin and peak_gb < 20.0
        print(f"  MAP ran; lps finite={fin}; device peak {peak_gb:.1f} GB "
              f"(gate < 20; remat expect ~15, unremat ~28) -> "
              f"{'PASS' if ok4 else 'FAIL'}")
        out["g4"] = dict(ran=True, finite=fin, peak_gb=peak_gb, free_mb=free,
                         gate_pass=ok4)
    except Exception as e:
        msg = str(e).splitlines()[0][:140]
        print(f"  MAP FAILED (free {free:.0f} MB): {msg}")
        out["g4"] = dict(ran=False, free_mb=free, error=msg, gate_pass=False)

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "validate_map_remat_results.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print("\nresults written to", path)


if __name__ == "__main__":
    main()
