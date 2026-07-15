#!/usr/bin/env python3
"""Profile the scene-API likelihood gradient for a representative single-plane fit.

Phase 0 (static, no device timing needed): AOT-lower + compile the forward and
gradient of ``ProbModel.log_prob`` and read XLA's own ``cost_analysis`` (FLOPs,
bytes accessed) and ``memory_analysis`` (peak temp). Sweeping n_max and
supersample at this stage gives a *first-principles scaling prediction* of where
the FLOPs go (gram ~ depth^2 vs basis ~ depth), before any device run.

Phase 1 (device timing): warm up and time a single bs=1 likelihood-gradient eval
(the fundamental MCLMC per-chain work unit; MCLMC builds the sim at bs=1 and vmaps
over chains/device), decomposed into stages by *differencing* progressively larger
jitted sub-computations:

    basis+conv+pool   <- lstsq_simulate(return_stacked=True, no_deflection=True)
    + EPL deflection  <- lstsq_simulate(return_stacked=True)
    + gram + solve    <- lstsq_simulate()
    + reduction+prior <- log_prob(z)            (forward)
    + backward (VJP)  <- grad(log_prob)(z)

A jax.profiler trace is also written to profile-data/ for fine-grained per-op
attribution in TensorBoard / Perfetto.

This is a measurement harness: it asserts nothing about correctness. Run it inside
the canonical Shifter container (JAX 0.10) on a GPU node.
"""
from __future__ import annotations

import os
# float64 pipeline must be requested before jax is imported.
os.environ.setdefault("JAX_ENABLE_X64", "1")

import argparse
import statistics
import time

import numpy as np
import jax
import jax.numpy as jnp
from jax import random as jr

import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions

from gigalens.simulator import SimulatorConfig
from gigalens.jax.profiles.mass.epl import EPL
from gigalens.jax.profiles.mass.shear import Shear
from gigalens.jax.profiles.light.sersic import SersicEllipse
from gigalens.jax.profiles.light.shapelets import Shapelets
from gigalens.jax.scene import Component, Plane, LensModel
from gigalens.jax.scene_prob_model import ImageData, ProbModel


# ----------------------------------------------------------------------------- model
def gaussian_psf(npix=21, fwhm_pix=3.0):
    """A simple normalized Gaussian PSF kernel (odd size)."""
    assert npix % 2 == 1
    sigma = fwhm_pix / 2.3548
    ax = np.arange(npix) - (npix - 1) / 2.0
    xx, yy = np.meshgrid(ax, ax)
    k = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    return (k / k.sum()).astype(np.float64)


def build_problem(num_pix=200, supersample=2, n_max=15, delta_pix=0.05,
                  conv_precision="float32", basis_precision=None, seed=0):
    """A representative single-deflector lstsq fit:
    EPL + Shear mass, Sersic lens light (lstsq), n_max shapelets source (lstsq)."""
    epl = Component(EPL(50), dict(
        theta_E=tfd.LogNormal(np.log(1.1), 0.3), gamma=tfd.Normal(2.05, 0.1),
        e1=tfd.Normal(0.0, 0.1), e2=tfd.Normal(0.0, 0.1),
        center_x=tfd.Normal(0.0, 0.05), center_y=tfd.Normal(0.0, 0.05)))
    shear = Component(Shear(), dict(gamma1=tfd.Normal(0.0, 0.05),
                                    gamma2=tfd.Normal(0.0, 0.05)))
    lens_light = Component(SersicEllipse(use_lstsq=True), dict(
        R_sersic=tfd.LogNormal(np.log(1.2), 0.3), n_sersic=tfd.Uniform(0.5, 6.0),
        e1=tfd.Normal(0.0, 0.1), e2=tfd.Normal(0.0, 0.1),
        center_x=tfd.Normal(0.0, 0.05), center_y=tfd.Normal(0.0, 0.05)))
    source = Component(Shapelets(n_max=n_max, use_lstsq=True), dict(
        beta=tfd.LogNormal(np.log(0.1), 0.3),
        center_x=tfd.Normal(0.05, 0.1), center_y=tfd.Normal(-0.04, 0.1)))

    model = LensModel([
        Plane(mass=[epl, shear], light=[lens_light]),
        Plane(deflection_ratio=1.0, light=[source]),
    ])

    cfg = SimulatorConfig(delta_pix=delta_pix, num_pix=num_pix, supersample=supersample,
                          kernel=gaussian_psf(), likelihood_precision="float64",
                          conv_precision=conv_precision, basis_precision=basis_precision)

    # Synthetic observation: forward-render the prior mean-ish sample, add noise.
    z = model.prior.sample(seed=jr.PRNGKey(seed))
    z_vec = jnp.stack(model.bijector.inverse(z))
    bg_rms, exp_time = 0.01, 1000.0
    # Build a throwaway sim to render a plausible image to fit against.
    from gigalens.jax.scene_simulator import SceneSimulator
    sim0 = SceneSimulator(model, cfg)
    x = model.bijector.forward(list(z_vec.T))
    params = model.to_params(x)
    # render each basis at unit amplitude, sum -> a non-trivial image
    stacked = np.asarray(sim0.lstsq_simulate(
        params, jnp.zeros((num_pix, num_pix)), jnp.ones((num_pix, num_pix)),
        return_stacked=True))
    image = stacked.sum(axis=-1).squeeze().astype(np.float64)
    rng = np.random.default_rng(seed)
    err = np.sqrt(bg_rms ** 2 + np.clip(image, 0, None) / exp_time)
    image = image + rng.normal(size=image.shape) * err

    ds = ImageData(image, cfg, background_rms=bg_rms, exp_time=exp_time, sees="all")
    prob = ProbModel(model, ds, mode="lstsq")
    return model, prob, cfg, z_vec, params


# ----------------------------------------------------------------------------- timing
def time_fn(fn, *args, n_warmup=3, n_iter=30):
    """Median ms per call, device-synchronized."""
    out = fn(*args)
    jax.block_until_ready(out)
    for _ in range(n_warmup - 1):
        jax.block_until_ready(fn(*args))
    ts = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        ts.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(ts), min(ts)


# ----------------------------------------------------------------------------- phase 0
def phase0(num_pix, n_max_list, supersample_list):
    print("\n" + "=" * 78)
    print("PHASE 0 — static cost model (XLA cost_analysis / memory_analysis)")
    print("=" * 78)
    hdr = f"{'n_max':>6} {'ss':>3} {'depth':>6} {'fwd GFLOP':>11} {'grad GFLOP':>11} {'grad GB':>9} {'peak MB':>9}"
    print(hdr)
    print("-" * len(hdr))
    for ss in supersample_list:
        for n_max in n_max_list:
            model, prob, cfg, z_vec, _ = build_problem(
                num_pix=num_pix, supersample=ss, n_max=n_max)
            depth = prob.simulators[0].depth

            def logp(z):
                return prob.log_prob(z)[0]

            def grad_logp(z):
                return jax.grad(logp)(z)

            def _cost(fn):
                comp = jax.jit(fn).lower(z_vec).compile()
                ca = comp.cost_analysis()
                if isinstance(ca, (list, tuple)):
                    ca = ca[0]
                flops = float(ca.get("flops", float("nan"))) if isinstance(ca, dict) else float("nan")
                gbytes = (float(ca.get("bytes accessed", float("nan"))) / 1e9
                          if isinstance(ca, dict) else float("nan"))
                peak_mb = float("nan")
                try:
                    ma = comp.memory_analysis()
                    peak_mb = (ma.temp_size_in_bytes + ma.output_size_in_bytes) / 1e6
                except Exception:
                    pass
                return flops, gbytes, peak_mb

            fwd_flops, _, _ = _cost(logp)
            g_flops, g_gb, g_peak = _cost(grad_logp)
            print(f"{n_max:>6} {ss:>3} {depth:>6} {fwd_flops/1e9:>11.2f} "
                  f"{g_flops/1e9:>11.2f} {g_gb:>9.2f} {g_peak:>9.1f}")


# ----------------------------------------------------------------------------- phase 1
def phase1(num_pix, supersample, n_max, trace_dir, n_iter):
    print("\n" + "=" * 78)
    print(f"PHASE 1 — device timing breakdown  (num_pix={num_pix}, ss={supersample}, "
          f"n_max={n_max})")
    print("=" * 78)
    print("device:", jax.devices()[0])

    model, prob, cfg, z_vec, params = build_problem(
        num_pix=num_pix, supersample=supersample, n_max=n_max)
    sim = prob.simulators[0]
    depth = sim.depth
    ds = prob.datasets[0]
    img, err, mask = ds.image, ds.error_map, ds.mask
    print(f"depth (n basis components) = {depth}, "
          f"grid = {num_pix * supersample}x{num_pix * supersample}, "
          f"num_free_params = {model.num_free_params}")

    f_stack_nd = jax.jit(lambda p: sim.lstsq_simulate(
        p, img, err, mask, return_stacked=True, no_deflection=True))
    f_stack = jax.jit(lambda p: sim.lstsq_simulate(
        p, img, err, mask, return_stacked=True))
    f_lstsq = jax.jit(lambda p: sim.lstsq_simulate(p, img, err, mask))
    f_fwd = jax.jit(lambda z: prob.log_prob(z)[0])
    f_grad = jax.jit(lambda z: jax.grad(lambda zz: prob.log_prob(zz)[0])(z))

    t_stack_nd, _ = time_fn(f_stack_nd, params, n_iter=n_iter)
    t_stack, _ = time_fn(f_stack, params, n_iter=n_iter)
    t_lstsq, _ = time_fn(f_lstsq, params, n_iter=n_iter)
    t_fwd, _ = time_fn(f_fwd, z_vec, n_iter=n_iter)
    t_grad, _ = time_fn(f_grad, z_vec, n_iter=n_iter)

    rows = [
        ("basis + conv + pool", t_stack_nd),
        ("EPL deflection trace", t_stack - t_stack_nd),
        ("gram form + solve", t_lstsq - t_stack),
        ("reduction + prior", t_fwd - t_lstsq),
        ("backward (VJP)", t_grad - t_fwd),
    ]
    print(f"\n{'stage':<24}{'ms (incremental)':>18}{'% of grad':>12}")
    print("-" * 54)
    for name, ms in rows:
        print(f"{name:<24}{ms:>18.3f}{100 * ms / t_grad:>11.1f}%")
    print("-" * 54)
    print(f"{'forward total':<24}{t_fwd:>18.3f}")
    print(f"{'GRAD TOTAL (per eval)':<24}{t_grad:>18.3f}{100.0:>11.1f}%")
    print(f"\n~MCLMC step (2 grad evals, mclachlan): {2 * t_grad:.2f} ms")

    # Fine-grained profiler trace for TensorBoard / Perfetto.
    os.makedirs(trace_dir, exist_ok=True)
    with jax.profiler.trace(trace_dir):
        for _ in range(20):
            jax.block_until_ready(f_grad(z_vec))
    print(f"\nprofiler trace written to: {trace_dir}")
    print("  view with: tensorboard --logdir", trace_dir)


# ----------------------------------------------------------------- basis-precision sweep
def _stage_breakdown(num_pix, supersample, n_max, basis_precision, n_iter):
    """Time the stage decomposition of one grad eval at a given basis precision."""
    model, prob, cfg, z_vec, params = build_problem(
        num_pix=num_pix, supersample=supersample, n_max=n_max,
        basis_precision=basis_precision)
    sim = prob.simulators[0]
    ds = prob.datasets[0]
    img, err, mask = ds.image, ds.error_map, ds.mask

    f_stack_nd = jax.jit(lambda p: sim.lstsq_simulate(
        p, img, err, mask, return_stacked=True, no_deflection=True))
    f_lstsq = jax.jit(lambda p: sim.lstsq_simulate(p, img, err, mask))
    f_grad = jax.jit(lambda z: jax.grad(lambda zz: prob.log_prob(zz)[0])(z))

    t_basis, _ = time_fn(f_stack_nd, params, n_iter=n_iter)   # basis + conv + pool
    t_lstsq, _ = time_fn(f_lstsq, params, n_iter=n_iter)
    t_grad, _ = time_fn(f_grad, z_vec, n_iter=n_iter)
    return dict(depth=sim.depth, basis=t_basis, grad=t_grad,
                rest=t_grad - t_basis)


def phase_basis_compare(num_pix, supersample, n_max, n_iter):
    print("\n" + "=" * 78)
    print(f"BASIS-PRECISION COMPARISON  (num_pix={num_pix}, ss={supersample}, "
          f"n_max={n_max}, grid={num_pix * supersample}x{num_pix * supersample})")
    print("=" * 78)
    f64 = _stage_breakdown(num_pix, supersample, n_max, None, n_iter)
    f32 = _stage_breakdown(num_pix, supersample, n_max, "float32", n_iter)
    print(f"depth = {f64['depth']}")
    print(f"\n{'stage':<22}{'f64 ms':>10}{'f32 ms':>10}{'speedup':>10}")
    print("-" * 52)
    for key, label in (("basis", "basis+conv+pool"), ("rest", "rest of grad"),
                       ("grad", "GRAD TOTAL")):
        print(f"{label:<22}{f64[key]:>10.3f}{f32[key]:>10.3f}"
              f"{f64[key] / f32[key]:>9.2f}x")
    print("-" * 52)
    print(f"basis stage as % of grad:  f64 {100*f64['basis']/f64['grad']:.0f}%   "
          f"f32 {100*f32['basis']/f32['grad']:.0f}%")


# --------------------------------------------------------------- backward breakdown
def phase_backward_breakdown(num_pix, supersample, n_max, n_iter):
    """Attribute the gradient's backward pass to stages by differencing grad-times of
    nested scalar losses, with a no-PSF simulator twin to split conv VJP from basis VJP.

    L0: sum(basis, no conv, no deflection)        [basis-gen + pool]
    L1: sum(basis, +conv, no deflection)          [+ convolution]
    L2: sum(basis, +conv, +deflection)            [+ EPL deflection]
    L3: log_prob(z)  (full: + gram solve + chi2)  [+ gram solve + reduction]
    backward(stage) = (grad_time - fwd_time) differenced across L0..L3.
    """
    import dataclasses as _dc
    from gigalens.jax.scene_simulator import SceneSimulator

    print("\n" + "=" * 78)
    print(f"BACKWARD-PASS BREAKDOWN  (num_pix={num_pix}, ss={supersample}, n_max={n_max}, "
          f"grid={num_pix * supersample}x{num_pix * supersample})")
    print("=" * 78)

    model, prob, cfg, z_vec, _ = build_problem(
        num_pix=num_pix, supersample=supersample, n_max=n_max)
    sim = prob.simulators[0]
    sim_nopsf = SceneSimulator(model, _dc.replace(cfg, kernel=None))
    ds = prob.datasets[0]
    img, err, mask = ds.image, ds.error_map, ds.mask

    def params_of(z):
        return model.to_params(prob.bij.forward(list(z.T)))

    # Weight map fed as a RUNTIME ARGUMENT (not a closed-over constant): a constant W —
    # incl. jnp.sum's all-ones — lets XLA constant-fold the pool/conv VJP (the cotangent
    # into those stages becomes a compile-time constant), so the convolution backward
    # measures as ~0 (an artifact). Passing W as a traced arg keeps every stage's upstream
    # cotangent runtime-varying, matching the real chi2-driven backward of L3.
    _sample = sim.lstsq_simulate(params_of(z_vec), img, err, mask, return_stacked=True)
    W = jax.random.normal(jax.random.PRNGKey(0), _sample.shape, dtype=_sample.dtype)

    def s0(z):
        return sim_nopsf.lstsq_simulate(
            params_of(z), img, err, mask, return_stacked=True, no_deflection=True)

    def s1(z):
        return sim.lstsq_simulate(
            params_of(z), img, err, mask, return_stacked=True, no_deflection=True)

    def s2(z):
        return sim.lstsq_simulate(
            params_of(z), img, err, mask, return_stacked=True, no_deflection=False)

    # weighted surrogate (W is a runtime arg → not folded)
    def _wsum(s):
        return lambda z, w: jnp.sum(w * s(z))

    fwd, bwd = {}, {}
    for name, s in (("L0 basis-gen+pool", s0), ("L1 +convolution", s1),
                    ("L2 +EPL deflection", s2)):
        f = _wsum(s)
        fj = jax.jit(f)
        gj = jax.jit(jax.grad(f, argnums=0))
        ft, _ = time_fn(lambda z: fj(z, W), z_vec, n_iter=n_iter)
        gt, _ = time_fn(lambda z: gj(z, W), z_vec, n_iter=n_iter)
        fwd[name], bwd[name] = ft, gt - ft

    # L3 is the real log_prob (already z-dependent cotangents — no surrogate needed).
    name = "L3 +gram-solve+reduction"
    ft, _ = time_fn(jax.jit(lambda z: prob.log_prob(z)[0]), z_vec, n_iter=n_iter)
    gt, _ = time_fn(jax.jit(jax.grad(lambda z: prob.log_prob(z)[0])), z_vec, n_iter=n_iter)
    fwd[name], bwd[name] = ft, gt - ft

    names = ["L0 basis-gen+pool", "L1 +convolution",
             "L2 +EPL deflection", "L3 +gram-solve+reduction"]

    # incremental backward per stage
    stage_bwd = [("basis-gen + pool", bwd[names[0]]),
                 ("convolution", bwd[names[1]] - bwd[names[0]]),
                 ("EPL deflection", bwd[names[2]] - bwd[names[1]]),
                 ("gram solve + reduction", bwd[names[3]] - bwd[names[2]])]
    total_bwd = bwd[names[3]]
    print(f"depth = {sim.depth}\n")
    print(f"{'cumulative loss':<28}{'fwd ms':>9}{'bwd ms':>9}")
    print("-" * 46)
    for name in names:
        print(f"{name:<28}{fwd[name]:>9.3f}{bwd[name]:>9.3f}")
    print(f"\n{'backward stage':<26}{'ms':>9}{'% of total bwd':>16}")
    print("-" * 51)
    for label, ms in stage_bwd:
        print(f"{label:<26}{ms:>9.3f}{100 * ms / total_bwd:>15.1f}%")
    print("-" * 51)
    print(f"{'TOTAL BACKWARD':<26}{total_bwd:>9.3f}{100.0:>15.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-pix", type=int, default=200)
    ap.add_argument("--supersample", type=int, default=2)
    ap.add_argument("--n-max", type=int, default=15)
    ap.add_argument("--n-iter", type=int, default=30)
    ap.add_argument("--trace-dir", default="profile-data/scene_likelihood")
    ap.add_argument("--phase0-only", action="store_true")
    ap.add_argument("--phase1-only", action="store_true")
    ap.add_argument("--basis-compare", action="store_true",
                    help="only run the f64-vs-f32 basis-precision timing comparison")
    ap.add_argument("--backward-breakdown", action="store_true",
                    help="only run the backward-pass stage attribution")
    args = ap.parse_args()

    print("JAX", jax.__version__, "| x64:", jax.config.jax_enable_x64,
          "| devices:", jax.devices())

    if args.backward_breakdown:
        for ss in sorted(set([1, args.supersample])):
            phase_backward_breakdown(args.num_pix, ss, args.n_max, args.n_iter)
        return

    if args.basis_compare:
        for ss in sorted(set([1, args.supersample])):
            phase_basis_compare(args.num_pix, ss, args.n_max, args.n_iter)
        return

    if not args.phase1_only:
        phase0(args.num_pix, n_max_list=[8, 12, 15, 20],
               supersample_list=[1, args.supersample])
    if not args.phase0_only:
        phase1(args.num_pix, args.supersample, args.n_max, args.trace_dir, args.n_iter)


if __name__ == "__main__":
    main()
