#!/usr/bin/env python3
"""Gates for flipping remat_basis default -> True (checkpoint 2026-07-09,
GIGALens-Code docs/logs/compute-profiling.md). Closes the two C-16 unexercised
corners:

  (i)  remat x fused conv+pool (ss>=3): equivalence remat-on vs remat-off
       (BOTH sides fused — the C-15 OOM was the UNFUSED f64 reference, so f64
       runs at (120,ss4,30) conservatively; f32 also checked at (200,ss4,30)).
       f64 chi2 rel & grad L2 <= 1e-12; f32 grad L2 <= 1e-5 + finiteness.
       bs=1 latency (200,ss4,30) on/off: penalty > 10% => do NOT default-on at
       ss>=3 (auto-off rule).
  (ii) vmapped-chains sampler shape: jit(vmap(grad(log_prob))) over 8 chains at
       (200,ss2,15) on/off: penalty > 10% => don't default-on for that shape.

Thresholds inherit from graded C-15/C-16 precedents. Companion (run in the
launcher): pytest tests/validation under the new default (golden anchor now
exercises remat; predicted pass ~1e-13 vs 1e-10 tolerance).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from validate_map_throughput import (build_problem, chi2_fn, logp_fn, rel,
                                     grad_l2, time_fn, xla_peak_mb)

import json
import numpy as np
import jax
import jax.numpy as jnp
from jax import random as jr


def draws_for(model, n, base_seed=700):
    return [jnp.stack(model.bijector.inverse(model.prior.sample(seed=jr.PRNGKey(base_seed + s))))
            for s in range(n)]


def eval_pair(cell, conv_precision, n_draws=6):
    """chi2/grad for remat True vs False at one cell (same data both sides)."""
    npx, ss, nm = cell
    model, prob0, cfg, z0 = build_problem(npx, ss, nm, conv_precision=conv_precision)
    data = np.asarray(prob0.datasets[0].image)
    dr = draws_for(model, n_draws)
    out = {}
    for remat in (True, False):
        prob = build_problem(npx, ss, nm, data_image=data,
                             conv_precision=conv_precision, remat_basis=remat)[1]
        fc, fg = jax.jit(chi2_fn(prob)), jax.jit(jax.grad(logp_fn(prob)))
        out[remat] = [dict(c=float(fc(z)), g=np.asarray(fg(z))) for z in dr]
    worst_c = max(rel(out[True][s]["c"], out[False][s]["c"]) for s in range(n_draws))
    worst_g = max(grad_l2(out[True][s]["g"], out[False][s]["g"]) for s in range(n_draws))
    nan = any(not np.all(np.isfinite(out[True][s]["g"])) for s in range(n_draws))
    print(f"  cell={cell} conv={conv_precision}: chi2 rel {worst_c:.2e}  "
          f"grad L2 {worst_g:.2e}  NaN={nan}")
    return dict(cell=list(cell), conv=str(conv_precision), chi2=worst_c,
                grad=worst_g, nan=nan)


def latency_pair(cell, shape, n_iter=30):
    """bs=1 or vmapped-8-chain grad latency, remat on vs off."""
    npx, ss, nm = cell
    res = {}
    for remat in (True, False):
        prob = build_problem(npx, ss, nm, remat_basis=remat)[1]
        z1 = jnp.stack(prob.model.bijector.inverse(
            prob.model.prior.sample(seed=jr.PRNGKey(0))))
        if shape == "bs1":
            g = jax.jit(jax.grad(logp_fn(prob)))
            z = z1
        else:  # vmap over 8 chains (MCLMC per-device shape)
            g = jax.jit(jax.vmap(jax.grad(logp_fn(prob))))
            z = jnp.tile(z1[None, :], (8, 1)) + 1e-3 * jr.normal(
                jr.PRNGKey(1), (8, z1.shape[0]))
        med, mn = time_fn(g, z, n_iter=n_iter)
        peak = xla_peak_mb(g, z)
        fin = bool(np.all(np.isfinite(np.asarray(g(z)))))
        res[remat] = dict(med=med, mn=mn, peak=peak, finite=fin)
        print(f"  cell={cell} shape={shape} remat={remat}: grad {med:7.2f} ms "
              f"(min {mn:.2f}) peak {peak:6.0f} MB finite={fin}")
    pen_med = (res[True]["med"] - res[False]["med"]) / res[False]["med"]
    pen_min = (res[True]["mn"] - res[False]["mn"]) / res[False]["mn"]
    print(f"  -> penalty: median {100*pen_med:+.1f}%  min {100*pen_min:+.1f}%")
    return dict(cell=list(cell), shape=shape, penalty_med=pen_med,
                penalty_min=pen_min, on=res[True], off=res[False])


def main():
    print("device:", jax.devices()[0], "| jax", jax.__version__)
    out = {}
    print("\n== corner (i): remat x fused equivalence ==")
    out["equiv"] = [eval_pair((120, 4, 30), None),
                    eval_pair((120, 4, 30), "float32"),
                    eval_pair((200, 4, 30), "float32")]
    e = out["equiv"]
    ok = (e[0]["chi2"] <= 1e-12 and e[0]["grad"] <= 1e-12
          and all(x["grad"] <= 1e-5 and not x["nan"] for x in e[1:])
          and not e[0]["nan"])
    print("  equivalence:", "PASS" if ok else "FAIL")
    out["equiv_pass"] = ok
    print("\n== corner (i): ss4 bs=1 latency ==")
    out["ss4_latency"] = latency_pair((200, 4, 30), "bs1")
    print("\n== corner (ii): vmapped 8-chain shape @ ss2 ==")
    out["vmap_latency"] = latency_pair((200, 2, 15), "vmap8")
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "validate_remat_default_results.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print("\nresults written to", path)


if __name__ == "__main__":
    main()
