#!/usr/bin/env python
"""25_run_l0g2_scene.py -- P3 L0-G2: the v3b CORRELATED per-basin refit on the
scene API (Perlmutter Path A, cgl2-pm venv, A100 hbm80g; deferred from phoenix
where only ~56 particles fit -- measured 367.6 MB/particle,
data/l0_v3b_memory_smoke.json).

Question (PLAN §6 P3): does the ported correlated likelihood ON THE SCENE API
reproduce the certified old-stack money number gamma_binned(corr,low) =
1.1032 within 2*sqrt(sigma_stat^2+sigma_seed^2) = 0.0172, with the low-basin
logZ preference retained (sign of dlogZ(steep-low) < 0)? L0-G2 licenses all
X1-class real-lens claims on the new substrate.

Design = the PRODUCTION sampler recipe, mirrored parameter-for-parameter from
cgl.e2.run_correlated_smc (basin-local q -> posterior anneal, blackjax
adaptive tempered SMC, HMC mutation steps 4 x 8 leapfrog @ step 0.1, metric =
q covariance, target_ess 0.7, max 400 stages; the VERBATIM driver copy
cgl2.samplers.common.run_adaptive_tempered_smc) -- the sampler is held fixed
so the cell isolates the SUBSTRATE question (old stack vs scene API), not a
kernel change. Likelihood = build_pm('v3b', diagonal=False) from
10_anchor_arbitration.py: the parity-certified marg scene model +
CorrelatedImageData with the ported whitener_v3b bundle (checkpointed
whitening), parity grid/PSF conventions applied.

q hand-off (documented deviation, checkpoint 2026-07-16): the production q
lives in OLD-stack unconstrained z; the scene bijector differs, so q cannot be
copied. Step 1 (25a_export_v3b_basin_x46.py, OLD venv) exported the production
q-fit draws in CONSTRAINED x46; here each draw maps through
param_map.scene_z_from_old_labels into scene-z, a Gaussian is refit
(cov * 3.0 inflate = the production --smc-cov-inflate, guard-floored), and
particles are drawn FRESH from that q (the lambda=0 law is exactly q by
construction; logZ_basin and the lambda=1 posterior are q-independent).

Legs: low @ 128 particles (the gamma gate leg, production count) then
steep @ 96 (the logZ-sign comparator leg, production count), seed 2.
No gate math here; harvest draws figs first.

Run: python 25_run_l0g2_scene.py --x46-npz <l0g2_basin_x46.npz> --seed 2 \
         --out $OUTDIR/l0g2_v3b_scene
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import time
from pathlib import Path

import numpy as np

# env BEFORE jax init (t02/A100 SMC production flags; module below setdefaults too)
_flags = os.environ.get("XLA_FLAGS", "")
for f in ("--xla_gpu_autotune_level=0", "--xla_disable_hlo_passes=priority-fusion"):
    if f.split("=")[0] not in _flags:
        _flags = (_flags + " " + f).strip()
os.environ["XLA_FLAGS"] = _flags
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("GIGALENS_X64", "1")

HERE = Path(__file__).resolve().parent

# import the L0 builder module (bootstraps vendor + guards at import)
_spec = importlib.util.spec_from_file_location(
    "anchor_arb", HERE / "10_anchor_arbitration.py")
arb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(arb)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import tensorflow_probability.substrates.jax as tfp  # noqa: E402

from cgl2 import guards, param_map  # noqa: E402
from cgl2.samplers import common  # noqa: E402

tfd = tfp.distributions

# frozen production SMC config (cgl.e2.run_correlated_smc mirror)
COV_INFLATE = 3.0
TARGET_ESS = 0.7
NUM_MCMC_STEPS = 4
HMC_INTEGRATION_STEPS = 8
HMC_STEP_SIZE = 0.1
MAX_LAMBDA_STEPS = 400
LEGS = (("low", 128), ("steep", 96))
GAMMA_SPLIT = 1.9          # campaign v3b basin-split convention


def _floor_svi_covariance(cov, rel_floor: float = 1e-10):
    """Verbatim copy with attribution: ../claude-giga-lens/cgl/guards.py::
    floor_svi_covariance (Bug-2 eigenvalue floor; cgl2.guards does not carry
    it). Floor at rel_floor * lambda_max, rebuild, Cholesky."""
    cov64 = np.asarray(cov, dtype=np.float64)
    cov64 = 0.5 * (cov64 + cov64.T)
    w, v = np.linalg.eigh(cov64)
    floor = rel_floor * float(w.max())
    n_floored = int((w < floor).sum())
    w = np.maximum(w, floor)
    cov_reg = (v * w) @ v.T
    cov_reg = 0.5 * (cov_reg + cov_reg.T)
    chol = np.linalg.cholesky(cov_reg)
    return cov_reg, chol, n_floored


def _weighted_quantile(x, w, qs):
    """Verbatim cgl.e2._weighted_quantile (attribution: old campaign)."""
    x = np.asarray(x, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    i = np.argsort(x)
    x, w = x[i], w[i]
    cw = (np.cumsum(w) - 0.5 * w) / max(w.sum(), 1e-300)
    return np.interp(qs, cw, x)


def scene_q_from_x46(model, x46, basin):
    """Constrained x46 draws -> scene-z draws -> production-convention q."""
    zs = np.stack([
        param_map.scene_z_from_old_labels(
            model, param_map.labeled_from_vec46(row))
        for row in x46])
    assert np.all(np.isfinite(zs)), f"{basin}: non-finite scene-z map"
    loc = zs.mean(axis=0)
    cov = np.cov(zs, rowvar=False) * COV_INFLATE
    cov_reg, chol, n_floored = _floor_svi_covariance(cov)
    return loc, cov_reg, chol, int(n_floored), zs.shape[0]


def run_leg(pm, basin, n_particles, loc, cov_reg, chol, seed):
    logprior_fn, loglik_fn = arb.make_closures(pm)

    def scene_logpost(z):
        return logprior_fn(z) + loglik_fn(z)

    q = tfd.MultivariateNormalTriL(
        loc=jnp.asarray(loc, dtype=jnp.float64),
        scale_tril=jnp.asarray(chol, dtype=jnp.float64))

    def logprior_pt(z):
        return jnp.squeeze(q.log_prob(jnp.asarray(z, dtype=jnp.float64)))

    def loglike_pt(z):
        zz = jnp.asarray(z, dtype=jnp.float64)
        return jnp.squeeze(scene_logpost(zz)) - jnp.squeeze(q.log_prob(zz))

    key = jax.random.PRNGKey(int(seed))
    k_seed, k_smc = jax.random.split(key)
    eps = jax.random.normal(k_seed, (int(n_particles), loc.size),
                            dtype=jnp.float64)
    particles = (jnp.asarray(loc)[None, :] + eps @ jnp.asarray(chol).T)

    t0 = time.time()
    out = common.run_adaptive_tempered_smc(
        logprior_pt, loglike_pt, particles, k_smc,
        target_ess=TARGET_ESS, num_mcmc_steps=NUM_MCMC_STEPS,
        hmc_step_size=HMC_STEP_SIZE,
        inverse_mass_matrix=jnp.asarray(cov_reg, dtype=jnp.float64),
        hmc_num_integration_steps=HMC_INTEGRATION_STEPS,
        max_lambda_steps=MAX_LAMBDA_STEPS)
    wall = time.time() - t0

    parts = np.asarray(out["particles"], dtype=np.float64)
    w = np.asarray(out["weights"], dtype=np.float64)
    w = w / max(w.sum(), 1e-300)
    w_ess = common.weight_ess(w)
    rng = np.random.default_rng(int(seed) + 20260710)   # e2 equal-weight conv.
    idx = rng.choice(int(n_particles), size=int(n_particles), p=w)
    gam = arb.gamma_of(pm, parts)
    gmed, g16, g84 = _weighted_quantile(gam, w, [0.5, 0.16, 0.84])
    gmean = float(np.sum(w * gam))
    st = jax.local_devices()[0].memory_stats() or {}
    summary = dict(
        basin=basin, n_particles=int(n_particles), seed=int(seed),
        logZ=float(out["log_evidence"]),
        n_lambda_steps=int(out["n_steps"]),
        gamma_median=float(gmed), gamma_q16=float(g16), gamma_q84=float(g84),
        gamma_mean=gmean,
        gamma_sigma=float(np.sqrt(max(np.sum(w * (gam - gmean) ** 2), 0.0))),
        ess_weight=float(w_ess), n_unique=int(len(np.unique(idx))),
        frac_gamma_gt_split=float(np.sum(w * (gam > GAMMA_SPLIT))),
        n_grad=int(out["n_grad"]), n_logp=int(out["n_logp"]),
        wall_s=round(wall, 1),
        peak_mb=round(float(st.get("peak_bytes_in_use", 0)) / 2**20, 1),
        config=dict(cov_inflate=COV_INFLATE, target_ess=TARGET_ESS,
                    num_mcmc_steps=NUM_MCMC_STEPS,
                    hmc_integration_steps=HMC_INTEGRATION_STEPS,
                    hmc_step_size=HMC_STEP_SIZE,
                    max_lambda_steps=MAX_LAMBDA_STEPS))
    arrays = dict(particles=parts[idx],            # equal-weight (e2 convention)
                  particles_weighted=parts, weights=w, gamma=gam,
                  lambdas=np.asarray(out["lambdas"], dtype=np.float64))
    return summary, arrays


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--x46-npz", required=True,
                    help="output of 25a_export_v3b_basin_x46.py")
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--out", required=True, help="output prefix (no suffix)")
    args = ap.parse_args()

    guards.require_gpu()
    print(f"[l0g2] devices={jax.devices()} x64={jax.config.jax_enable_x64}",
          flush=True)

    refs = arb.load_refs()
    pm, mv, _ = arb.build_pm("v3b", refs, diagonal=False, checkpoint=True)
    xz = np.load(args.x46_npz, allow_pickle=True)
    x46_prov = json.loads(str(xz["provenance"]))
    print(f"[l0g2] x46 export provenance: {x46_prov['inputs']}", flush=True)

    out = dict(script="25_run_l0g2_scene.py",
               generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               jax=jax.__version__, device=str(jax.local_devices()[0]),
               whitener="whitener_v3b.npz (ported, 02 manifest)",
               x46_provenance=x46_prov, config=vars(args), legs={})
    arrays_all = {}
    for basin, n_particles in LEGS:
        print(f"\n[l0g2] === leg {basin} @ {n_particles} ===", flush=True)
        loc, cov_reg, chol, n_floored, n_fit = scene_q_from_x46(
            mv.model, np.asarray(xz[f"{basin}_x46"], dtype=np.float64), basin)
        gq = float(arb.gamma_of(pm, loc[None, :])[0])
        print(f"[l0g2] q({basin}): n_fit={n_fit} gamma(loc)={gq:.4f} "
              f"n_floored={n_floored}", flush=True)
        summary, arrays = run_leg(pm, basin, n_particles, loc, cov_reg, chol,
                                  args.seed)
        summary["q"] = dict(n_fit=n_fit, n_floored=n_floored, gamma_loc=gq,
                            cov_inflate=COV_INFLATE)
        out["legs"][basin] = summary
        for k, v in arrays.items():
            arrays_all[f"{basin}_{k}"] = v
        print(f"[l0g2] {basin}: gamma={summary['gamma_median']:.4f} "
              f"[{summary['gamma_q16']:.4f},{summary['gamma_q84']:.4f}] "
              f"logZ={summary['logZ']:.2f} steps={summary['n_lambda_steps']} "
              f"({summary['wall_s']:.0f}s)", flush=True)

    if {"low", "steep"} <= set(out["legs"]):
        dlogZ = out["legs"]["steep"]["logZ"] - out["legs"]["low"]["logZ"]
        out["H"] = dict(dlogZ_steep_minus_low=float(dlogZ),
                        favors_low=bool(dlogZ < 0.0))
        print(f"\n[l0g2] dlogZ(steep-low) = {dlogZ:.2f} "
              f"(favors {'LOW' if dlogZ < 0 else 'STEEP'})", flush=True)

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    np.savez(outp.with_suffix(".npz"), **arrays_all)
    json.dump(out, open(outp.with_suffix(".json"), "w"), indent=1, default=float)
    print(f"[l0g2] wrote {outp}.npz + .json", flush=True)


if __name__ == "__main__":
    main()
