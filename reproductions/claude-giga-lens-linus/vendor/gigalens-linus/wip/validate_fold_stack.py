#!/usr/bin/env python3
"""Pre-registered equivalence + speedup gates: fused conv+pool (spectral fold)
and scatter-free shapelet recurrence (stack). Checkpoint draft:
$PSCRATCH/claude_perf/c15_checkpoint_draft.md (HOME at quota; merge to
GIGALens-Code docs/logs/compute-profiling.md when space frees).

Gates (falsifiers):
  EA  fold identity, f64 (conv_precision=None): chi2 rel > 1e-10 or grad L2
      rel > 1e-10 vs conv->crop->average_pool           -> fold NOT shippable.
  EA32 fold at production conv f32: chi2 rel > 2e-5 (pooling moves into f32;
      f32 eps x reassociation) or any VJP NaN            -> fold NOT shippable.
  EB  stack identity, f64: chi2 rel > 1e-12 or grad L2 > 1e-12 vs the buffer/
      scatter recurrence (same ops, same order)          -> stack NOT shippable.
  EC  combined new-vs-old grad L2 > 1e-8 (f64)           -> investigate.
  S   speed at production f32: fold < 5% grad gain at (200,ss4,30) -> revert
      fold; stack < 2% at (200,ss4,30) -> revert stack. Predictions: fold
      -15..25%, stack -5..15% at the vela cell.

Variants are module-global toggles (scene_simulator._FUSE_CONV_POOL;
shapelets._phi_basis_1d <-> _phi_basis_1d_buffer). jit traces at FIRST CALL:
every jit for a variant is built, traced, and evaluated inside its context.
"""
import os
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse
import contextlib
import json
import statistics
import time

import numpy as np
import jax
import jax.numpy as jnp
from jax import random as jr

import gigalens.jax.scene_simulator as scene_sim
import gigalens.jax.profiles.light.shapelets as shp

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from validate_fast_lstsq import build_problem  # noqa: E402


@contextlib.contextmanager
def unfused():
    saved = scene_sim._FUSE_CONV_POOL
    scene_sim._FUSE_CONV_POOL = False
    try:
        yield
    finally:
        scene_sim._FUSE_CONV_POOL = saved


@contextlib.contextmanager
def buffer_phi():
    saved = shp._phi_basis_1d
    shp._phi_basis_1d = shp._phi_basis_1d_buffer
    try:
        yield
    finally:
        shp._phi_basis_1d = saved


def chi2_fn(prob):
    def f(z):
        _, red = prob.log_like(z)
        return red * prob.event_size
    return f


def logp_fn(prob):
    return lambda z: prob.log_prob(z)[0]


def rel(a, b):
    return float(abs(a - b) / max(abs(b), 1e-300))


def grad_l2(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def time_fn(fn, *args, n_warmup=3, n_iter=30):
    jax.block_until_ready(fn(*args))
    for _ in range(n_warmup - 1):
        jax.block_until_ready(fn(*args))
    ts = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        ts.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(ts), min(ts)


def eval_variant(ctxs, npx, ss, nm, conv_precision, draws, data_image):
    """Build + trace + evaluate chi2 and grad inside the given contexts."""
    with contextlib.ExitStack() as st:
        for c in ctxs:
            st.enter_context(c())
        prob = build_problem(npx, ss, nm, data_image=data_image,
                             conv_precision=conv_precision)[1]
        fc = jax.jit(chi2_fn(prob))
        fg = jax.jit(jax.grad(logp_fn(prob)))
        return [dict(c=float(fc(z)), g=np.asarray(fg(z))) for z in draws]


def equivalence(npx, ss, nm, n_draws=6):
    print(f"\n-- equivalence @ ({npx}, ss{ss}, n_max={nm}) --")
    model, prob0, cfg, z0 = build_problem(npx, ss, nm, conv_precision=None)
    data = np.asarray(prob0.datasets[0].image)
    draws = [jnp.stack(model.bijector.inverse(model.prior.sample(seed=jr.PRNGKey(300 + s))))
             for s in range(n_draws)]

    new = eval_variant([], npx, ss, nm, None, draws, data)              # fold+stack
    unf = eval_variant([unfused], npx, ss, nm, None, draws, data)       # EA ref
    buf = eval_variant([buffer_phi], npx, ss, nm, None, draws, data)    # EB ref
    old = eval_variant([unfused, buffer_phi], npx, ss, nm, None, draws, data)  # EC ref
    new32 = eval_variant([], npx, ss, nm, "float32", draws, data)
    unf32 = eval_variant([unfused], npx, ss, nm, "float32", draws, data)

    worst = dict(ea_chi2=0.0, ea_grad=0.0, eb_chi2=0.0, eb_grad=0.0,
                 ec_grad=0.0, ea32_chi2=0.0, nan=False)
    for s in range(n_draws):
        worst["ea_chi2"] = max(worst["ea_chi2"], rel(new[s]["c"], unf[s]["c"]))
        worst["ea_grad"] = max(worst["ea_grad"], grad_l2(new[s]["g"], unf[s]["g"]))
        worst["eb_chi2"] = max(worst["eb_chi2"], rel(new[s]["c"], buf[s]["c"]))
        worst["eb_grad"] = max(worst["eb_grad"], grad_l2(new[s]["g"], buf[s]["g"]))
        worst["ec_grad"] = max(worst["ec_grad"], grad_l2(new[s]["g"], old[s]["g"]))
        worst["ea32_chi2"] = max(worst["ea32_chi2"], rel(new32[s]["c"], unf32[s]["c"]))
        worst["nan"] = worst["nan"] or not (
            np.all(np.isfinite(new[s]["g"])) and np.all(np.isfinite(new32[s]["g"])))
    ok = (worst["ea_chi2"] <= 1e-10 and worst["ea_grad"] <= 1e-10
          and worst["eb_chi2"] <= 1e-12 and worst["eb_grad"] <= 1e-12
          and worst["ec_grad"] <= 1e-8 and worst["ea32_chi2"] <= 2e-5
          and not worst["nan"])
    print("  worst:", {k: (f"{v:.2e}" if isinstance(v, float) else v)
                       for k, v in worst.items()}, "->", "PASS" if ok else "FAIL")
    return dict(cell=[npx, ss, nm], **worst, gate_pass=ok)


def time_variant(ctxs, npx, ss, nm, n_iter):
    with contextlib.ExitStack() as st:
        for c in ctxs:
            st.enter_context(c())
        prob = build_problem(npx, ss, nm)[1]   # production conv f32
        g = jax.jit(jax.grad(logp_fn(prob)))
        z = jnp.stack(prob.model.bijector.inverse(
            prob.model.prior.sample(seed=jr.PRNGKey(0))))
        med, mn = time_fn(g, z, n_iter=n_iter)
        comp = g.lower(z).compile()
        try:
            ma = comp.memory_analysis()
            peak = (ma.temp_size_in_bytes + ma.output_size_in_bytes) / 1e6
        except Exception:
            peak = float("nan")
    return dict(grad_ms=med, grad_ms_min=mn, xla_peak_mb=peak)


def speed(n_iter=30):
    rows = []
    print("\n-- speed 2x2 @ (200, ss4, n_max=30), production conv f32 --")
    for fold_name, fctx in [("fold", []), ("unfold", [unfused])]:
        for phi_name, pctx in [("stack", []), ("buffer", [buffer_phi])]:
            r = time_variant(fctx + pctx, 200, 4, 30, n_iter)
            r.update(fold=fold_name, phi=phi_name, cell=[200, 4, 30])
            rows.append(r)
            print(f"  {fold_name:>6}+{phi_name:<7} | grad {r['grad_ms']:8.2f} ms "
                  f"(min {r['grad_ms_min']:.2f})  peak {r['xla_peak_mb']:8.0f} MB")
    print("\n-- combined new vs old at anchor cells --")
    for (npx, ss, nm) in [(200, 2, 15), (200, 2, 30)]:
        for label, ctxs in [("new", []), ("old", [unfused, buffer_phi])]:
            r = time_variant(ctxs, npx, ss, nm, n_iter)
            r.update(variant=label, cell=[npx, ss, nm])
            rows.append(r)
            print(f"  ({npx},ss{ss},nmax{nm}) {label}: grad {r['grad_ms']:8.2f} ms "
                  f"(min {r['grad_ms_min']:.2f})  peak {r['xla_peak_mb']:8.0f} MB")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="all", choices=["all", "equiv", "speed"])
    ap.add_argument("--n-iter", type=int, default=30)
    args = ap.parse_args()
    print("device:", jax.devices()[0], "| jax", jax.__version__)
    out = dict(device=str(jax.devices()[0]))
    if args.part in ("all", "equiv"):
        # ss4 identity cell at 120px: the (200,ss4,30) f64 UNFUSED reference
        # needs > 40 GB (complex128 spectra of the 496-basis stack) — the
        # identity is size-independent, so the fold/conditioning coverage
        # (ss=4, depth 496) moves to a feasible grid; (200,ss4,30) itself is
        # exercised at production f32 by the speed cells + pytest goldens.
        # Pre-registered as checkpoint amendment 7 (feasibility, not results).
        out["equiv"] = [equivalence(200, 2, 15), equivalence(120, 4, 30)]
    if args.part in ("all", "speed"):
        out["speed"] = speed(args.n_iter)
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "validate_fold_stack_results.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print("\nresults written to", path)


if __name__ == "__main__":
    main()
