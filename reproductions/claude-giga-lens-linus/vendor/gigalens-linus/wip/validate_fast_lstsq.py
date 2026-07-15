#!/usr/bin/env python3
"""Pre-registered equivalence + speedup gate: direct-chi2 lstsq and rfft2 conv.

Design checkpoint: "(2026-07-09) Direct-chi2 lstsq + rfft2 convolution" in
GIGALens-Code docs/logs/compute-profiling.md. Gates (falsifiers):

  E1 chi2 identity : rel|chi2_direct - chi2_image| > 1e-8 on any of 8 prior
                     draws (anchor AND n_max=30 cell)      -> NOT shippable.
                     (Pre-registered reading: failure ONLY at n_max=30 ->
                     investigate cross-graph solve rounding x gram conditioning
                     before concluding "bug".)
  E2 conv identity : at conv_precision=None (f64, the regime the thresholds are
                     derived for): chi2 rel > 1e-10 or grad L2 rel ||dg||/||g||
                     > 1e-10, or any NaN in the rfft2 VJP  -> revert to complex.
  E3 grad identity : combined grad(log_prob) max rel err (floored elementwise)
                     > 1e-6 at f64.
  E4 production dtype: at conv_precision="float32" (the shipped config, C-6):
                     rfft-vs-complex chi2 rel > 1e-5 (f32 eps ~1.2e-7 x FFT
                     growth) or any VJP NaN               -> revert to complex.
  S  speedup       : 2x2 variant matrix {complex,rfft} x {image-chi2,direct-chi2}
                     at the anchor (production conv f32); winner also at
                     (30,ss2) and (30,ss4).
                     Direct-chi2 < 5% grad gain -> re-profile, don't ship.

  Mandatory companion gate: `python -m pytest tests/validation -q` (includes the
  frozen golden anchor — the only pre-change reference; E1/E3 alone cannot catch
  a bug shared by the refactored lstsq_simulate and lstsq_chi2).

Variants are realized by (a) monkeypatching gigalens.jax.simulator's module-global
``_manual_fftconvolve_same`` to the retained ``_manual_fftconvolve_same_complex``
(fresh ProbModel per variant => fresh traces pick up the patch), and (b) an
image-path log_prob replicating the pre-change ``_dataset_chi2``.

Run inside the canonical Shifter container (JAX 0.10) on a GPU node.
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

import tensorflow_probability.substrates.jax as tfp

tfd = tfp.distributions

import gigalens.jax.simulator as sim_mod
from gigalens.simulator import SimulatorConfig
from gigalens.jax.profiles.mass.epl import EPL
from gigalens.jax.profiles.mass.shear import Shear
from gigalens.jax.profiles.light.sersic import SersicEllipse
from gigalens.jax.profiles.light.shapelets import Shapelets
from gigalens.jax.scene import Component, Plane, LensModel
from gigalens.jax.scene_prob_model import ImageData, ProbModel
from gigalens.jax.scene_simulator import SceneSimulator


# ------------------------------------------------------------------ problem
def gaussian_psf(npix=21, fwhm_pix=3.0):
    assert npix % 2 == 1
    sigma = fwhm_pix / 2.3548
    ax = np.arange(npix) - (npix - 1) / 2.0
    xx, yy = np.meshgrid(ax, ax)
    k = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    return (k / k.sum()).astype(np.float64)


def build_problem(num_pix=200, supersample=2, n_max=15, delta_pix=0.05,
                  niter=18, seed=0, data_image=None, conv_precision="float32"):
    """Same synthetic single-plane lstsq fit as the Phase-2 profiling driver.

    ``data_image``: reuse a previously generated observed image so two variant
    builds (rfft vs complex conv) fit BYTE-IDENTICAL data — otherwise the
    1e-14-level conv difference leaks into the synthetic data and contaminates
    the equivalence comparison.

    ``conv_precision``: "float32" is the production dtype (C-6) and is what the
    speed matrix times; the f64 equivalence gates pass ``None`` because their
    1e-10-scale thresholds are derived for float64 arithmetic (grading fix F1 —
    at f32 the rfft-vs-complex difference is float32-rounding scale, ~1e-6)."""
    epl = Component(EPL(niter), dict(
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
                          conv_precision=conv_precision, basis_precision=None)

    z = model.prior.sample(seed=jr.PRNGKey(seed))
    z_vec = jnp.stack(model.bijector.inverse(z))
    bg_rms, exp_time = 0.01, 1000.0
    if data_image is None:
        sim0 = SceneSimulator(model, cfg)
        x = model.bijector.forward(list(z_vec.T))
        params = model.to_params(x)
        stacked = np.asarray(sim0.lstsq_simulate(
            params, jnp.zeros((num_pix, num_pix)), jnp.ones((num_pix, num_pix)),
            return_stacked=True))
        clean = stacked.sum(axis=-1).squeeze().astype(np.float64)
        err = np.sqrt(bg_rms ** 2 + np.clip(clean, 0, None) / exp_time)
        image = clean + np.random.default_rng(seed).normal(size=clean.shape) * err
    else:
        image = np.asarray(data_image)
    ds = ImageData(image, cfg, background_rms=bg_rms, exp_time=exp_time, sees="all")
    prob = ProbModel(model, ds, mode="lstsq")
    return model, prob, cfg, z_vec


@contextlib.contextmanager
def complex_conv():
    """Route _convolve_components through the retained full-complex FFT path."""
    saved = sim_mod._manual_fftconvolve_same
    sim_mod._manual_fftconvolve_same = sim_mod._manual_fftconvolve_same_complex
    try:
        yield
    finally:
        sim_mod._manual_fftconvolve_same = saved


def image_path_fns(prob):
    """The pre-change path: lstsq image reconstruction + pixel-space chi2.

    Returns (chi2_fn, logp_fn). NOTE: any jit of these must be first CALLED
    while the intended conv patch is active — tracing happens at first call,
    not at ProbModel construction."""
    ds = prob.datasets[0]
    sim = prob.simulators[0]
    mask = ds.mask.astype(ds.error_map.dtype)
    norm = jnp.sum(jnp.log(2 * jnp.pi * ds.error_map ** 2) * mask, axis=(-2, -1))

    def chi2_fn(z):
        x = prob.bij.forward(z)
        params = prob.model.to_params(x)
        im = sim.lstsq_simulate(params, ds.image, ds.error_map, ds.mask)
        diff = (im - ds.image) / ds.error_map
        return jnp.sum(diff ** 2 * mask, axis=(-2, -1))

    def logp(z):
        return -0.5 * (chi2_fn(z) + norm) + prob.log_prior(z)

    return chi2_fn, logp


def direct_fns(prob):
    """The new path: chi2 via the normal-equation identity (ProbModel.log_prob)."""
    def chi2_fn(z):
        _, red_chi2 = prob.log_like(z)
        return red_chi2 * prob.event_size

    def logp(z):
        return prob.log_prob(z)[0]

    return chi2_fn, logp


# ------------------------------------------------------------------ metrics
def rel_scalar(a, b):
    return float(abs(a - b) / max(abs(b), 1e-300))

def grad_rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    floor = 1e-6 * np.max(np.abs(b))
    max_el = float(np.max(np.abs(a - b) / np.maximum(np.abs(b), floor)))
    l2 = float(np.linalg.norm(a - b) / np.linalg.norm(b))
    return max_el, l2

def time_fn(fn, *args, n_warmup=3, n_iter=30):
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

def xla_peak_mb(jitted, z):
    comp = jitted.lower(z).compile()
    try:
        ma = comp.memory_analysis()
        return (ma.temp_size_in_bytes + ma.output_size_in_bytes) / 1e6
    except Exception:
        return float("nan")


# ------------------------------------------------------------------ E-gates
def equivalence(num_pix, supersample, n_max, n_draws=8):
    """E1/E2/E3 at conv_precision=None (float64 — matches the threshold
    derivation; grading fix F1), then E4 at the production conv_precision
    ="float32": rfft-vs-complex chi2 rel <= 1e-5 (float32 eps ~1.2e-7 x FFT
    growth) + no NaN in the production-dtype rfft VJP."""
    print(f"\n-- equivalence @ npix={num_pix} ss={supersample} n_max={n_max} --")
    model, prob, cfg, z0 = build_problem(num_pix, supersample, n_max,
                                         conv_precision=None)
    shared_image = np.asarray(prob.datasets[0].image)  # identical data for all variants

    draws = [jnp.stack(model.bijector.inverse(model.prior.sample(seed=jr.PRNGKey(100 + s))))
             for s in range(n_draws)]

    # rfft-conv f64 variants (production module state): trace + evaluate now.
    c_img, l_img = image_path_fns(prob)
    c_dir, l_dir = direct_fns(prob)
    f_c_img, f_c_dir = jax.jit(c_img), jax.jit(c_dir)
    g_img, g_dir = jax.jit(jax.grad(l_img)), jax.jit(jax.grad(l_dir))
    vals = [dict(c_img=float(f_c_img(z)), c_dir=float(f_c_dir(z)),
                 g_img=np.asarray(g_img(z)), g_dir=np.asarray(g_dir(z)))
            for z in draws]

    # complex-conv f64 reference: build, TRACE, and evaluate entirely inside the
    # patch (jit traces at first call — a trace outside the context would
    # silently use the rfft path and void the E2 comparison).
    with complex_conv():
        prob_c = build_problem(num_pix, supersample, n_max,
                               data_image=shared_image, conv_precision=None)[1]
        c_img_c, l_img_c = image_path_fns(prob_c)
        f_c_img_c = jax.jit(c_img_c)
        g_img_c = jax.jit(jax.grad(l_img_c))
        for v, z in zip(vals, draws):
            v["c_img_c"] = float(f_c_img_c(z))
            v["g_img_c"] = np.asarray(g_img_c(z))

    # E4: production dtype (conv f32) — rfft vs complex chi2 + VJP NaN check.
    prob32 = build_problem(num_pix, supersample, n_max,
                           data_image=shared_image, conv_precision="float32")[1]
    c32, l32 = image_path_fns(prob32)
    f_c32, g32 = jax.jit(c32), jax.jit(jax.grad(l32))
    vals32 = [dict(c=float(f_c32(z)), g=np.asarray(g32(z))) for z in draws]
    with complex_conv():
        prob32_c = build_problem(num_pix, supersample, n_max,
                                 data_image=shared_image,
                                 conv_precision="float32")[1]
        c32c = jax.jit(image_path_fns(prob32_c)[0])
        for v, z in zip(vals32, draws):
            v["c_c"] = float(c32c(z))

    worst = dict(e1_chi2=0.0, e2_fwd=0.0, e2_grad_l2=0.0, e3_grad=0.0,
                 e4_chi2_f32=0.0, nan=False, nan_f32=False)
    for s, (v, v32) in enumerate(zip(vals, vals32)):
        e1 = rel_scalar(v["c_dir"], v["c_img"])          # direct vs image chi2 (same conv)
        e2f = rel_scalar(v["c_img"], v["c_img_c"])       # rfft vs complex (same chi2 path)
        _, e2l2 = grad_rel(v["g_img"], v["g_img_c"])     # L2 metric (grading fix F2)
        e3m, e3l2 = grad_rel(v["g_dir"], v["g_img_c"])   # combined new vs old
        e4 = rel_scalar(v32["c"], v32["c_c"])            # production-dtype conv delta
        nan = bool(np.any(~np.isfinite(v["g_dir"])))
        nan32 = bool(np.any(~np.isfinite(v32["g"])))
        worst["e1_chi2"] = max(worst["e1_chi2"], e1)
        worst["e2_fwd"] = max(worst["e2_fwd"], e2f)
        worst["e2_grad_l2"] = max(worst["e2_grad_l2"], e2l2)
        worst["e3_grad"] = max(worst["e3_grad"], e3m)
        worst["e4_chi2_f32"] = max(worst["e4_chi2_f32"], e4)
        worst["nan"] = worst["nan"] or nan
        worst["nan_f32"] = worst["nan_f32"] or nan32
        print(f"  draw {s}: E1 chi2 {e1:.2e} | E2 fwd {e2f:.2e} gradL2 {e2l2:.2e} "
              f"| E3 grad max {e3m:.2e} (L2 {e3l2:.2e}) | E4 f32 {e4:.2e} "
              f"| NaN={nan}/{nan32}")
    ok = (worst["e1_chi2"] <= 1e-8 and worst["e2_fwd"] <= 1e-10
          and worst["e2_grad_l2"] <= 1e-10 and worst["e3_grad"] <= 1e-6
          and worst["e4_chi2_f32"] <= 1e-5
          and not worst["nan"] and not worst["nan_f32"])
    print(f"  WORST: {worst}  ->  {'PASS' if ok else 'FAIL'}")
    return dict(num_pix=num_pix, ss=supersample, n_max=n_max, **worst, gate_pass=ok)


# ------------------------------------------------------------------ S-gate
def _time_variant(npx, ss, nm, conv_ctx, which_fns, n_iter, fwd_too=False):
    """Build + trace + time one variant WITH the conv patch active throughout."""
    with conv_ctx():
        prob = build_problem(npx, ss, nm)[1]
        _, logp = which_fns(prob)
        zf = jnp.stack(prob.model.bijector.inverse(
            prob.model.prior.sample(seed=jr.PRNGKey(0))))
        g = jax.jit(jax.grad(logp))
        tg, tg_min = time_fn(g, zf, n_iter=n_iter)
        tf_ = float("nan")
        if fwd_too:
            fw = jax.jit(logp)
            tf_, _ = time_fn(fw, zf, n_iter=n_iter)
        peak = xla_peak_mb(g, zf)
    return dict(grad_ms=tg, grad_ms_min=tg_min, fwd_ms=tf_, xla_peak_mb=peak)


def speed_matrix(n_iter=30):
    print("\n-- speedup: 2x2 variants @ anchor (200, ss2, n_max15, niter18) --")
    rows = []
    for conv_name, ctx in [("rfft", contextlib.nullcontext), ("complex", complex_conv)]:
        for chi2_name, fns in [("direct", direct_fns), ("image", image_path_fns)]:
            r = _time_variant(200, 2, 15, ctx, fns, n_iter, fwd_too=True)
            r.update(conv=conv_name, chi2=chi2_name, cell=(200, 2, 15))
            rows.append(r)
            print(f"  conv={conv_name:8s} chi2={chi2_name:7s} | fwd {r['fwd_ms']:7.2f}  "
                  f"grad {r['grad_ms']:7.2f} ms (min {r['grad_ms_min']:.2f})  "
                  f"peak {r['xla_peak_mb']:7.0f} MB")
    print("\n-- winner (rfft+direct) vs baseline (complex+image) at big cells --")
    for (npx, ss, nm) in [(200, 2, 30), (200, 4, 30)]:
        for label, ctx, fns in [("new", contextlib.nullcontext, direct_fns),
                                ("old", complex_conv, image_path_fns)]:
            r = _time_variant(npx, ss, nm, ctx, fns, n_iter)
            r.update(variant=label, cell=(npx, ss, nm))
            rows.append(r)
            print(f"  ({npx},ss{ss},nmax{nm}) {label}: grad {r['grad_ms']:8.2f} ms "
                  f"(min {r['grad_ms_min']:.2f})  peak {r['xla_peak_mb']:8.0f} MB")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="all", choices=["all", "equiv", "speed"])
    ap.add_argument("--n-iter", type=int, default=30)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    print("device:", jax.devices()[0], "| jax", jax.__version__)
    results = dict(device=str(jax.devices()[0]), jax=jax.__version__)
    if args.part in ("all", "equiv"):
        results["equiv"] = [equivalence(200, 2, 15), equivalence(200, 2, 30)]
    if args.part in ("all", "speed"):
        results["speed"] = speed_matrix(n_iter=args.n_iter)
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "validate_fast_lstsq_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nresults written to {out}")


if __name__ == "__main__":
    main()
