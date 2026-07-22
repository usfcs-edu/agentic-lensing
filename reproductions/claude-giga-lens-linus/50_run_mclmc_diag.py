#!/usr/bin/env python
"""50_run_mclmc_diag.py — E1: PI-requested MCLMC diagonal fits (phoenix L4s).

Design checkpoint: CAMPAIGN.md stage-log entry 2026-07-22 "E1 — PI-REQUESTED
MCLMC DIAGONAL FITS" (frozen BEFORE launch). Sampler = the PI's OWN vendored
MCLMC (gigalens-linus experimental `full_mclmc_with_adapt_sharded` +
`_build_kernel_shardmap(isokinetic_mclachlan_smart)`, UNPATCHED per D1),
driven directly with the parity-certified scene-API diagonal logdensity —
the old-API-coupled MCLMC_JIT wrapper is bypassed, the driver/kernel/adaptation
code runs verbatim. Targets = `10_anchor_arbitration.build_pm(tag, refs,
diagonal=True)` for tag in {v2d, v3b}. Warm starts = physical basin ONLY
(single-basin by design; E1-G2 containment check reports any escapee chain).

Stages (run in order):
  prep   : OLD-venv (cgl, CPU, bijector only — NEVER the likelihood) warm-start
           extraction: subsample foundry-i hmc_v13_{tag} z74 draws (v3b:
           mass_gamma<1.7 conditional), map via cgl.e2.paper_z_to_marg_z ->
           x46 constrained (KEYED: index_labels stored, order never assumed)
                                   -> data/mclmc_warm_{tag}_x46.npz
  smoke  : 4 chains x 50 burn + 50 draws on the L4 — finite, NaN-health,
           basin containment, grad-timing horizon projection
                                   -> data/mclmc_diag_{tag}_smoke.json
  run    : production — 2 seed groups x 32 chains x (2000 burn + 4000 draws),
           sequential groups (memory-safe); arviz R-hat/ESS worst-param over
           46 scene-z + 8 physical mass params, pooled 64 chains + per group
                                   -> data/mclmc_diag_{tag}.{npz,json}

Run (one product per L4; v3b -> GPU 8, v2d -> GPU 9):
  PYTHONUNBUFFERED=1 GIGALENS_X64=1 CGL2_GPU=8 CUDA_DEVICE_ORDER=PCI_BUS_ID \
  /raid/benson/.venvs/cgl2/bin/python 50_run_mclmc_diag.py run --tag v3b
Escalation (ONE allowed per product, per checkpoint): --draws 8000 --seed-offset 10
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path("/raid/benson/git/agentic-lensing/reproductions/claude-giga-lens-linus")
OLD = Path("/raid/benson/git/agentic-lensing/reproductions/claude-giga-lens")
FI_DATA = Path("/raid/benson/git/agentic-lensing/reproductions/foundry-i/data")
DATA = ROOT / "data"
OLD_PY = "/raid/benson/.venvs/cgl/bin/python"

# ---- frozen E1 config (checkpoint 2026-07-22; changes = ledgered deviation) ----
N_MAP = 512                 # warm draws mapped per product
MAP_SEED = {"v2d": 424242, "v3b": 424243}
V3B_GAMMA_CUT = 1.7         # physical-branch conditional (fermat-teaser convention)
N_CHAINS = 32               # per seed group
N_GROUPS = 2
BURN = 2000
DRAWS = 4000
FRAC_TUNE = (0.2, 0.6, 0.2)         # MCLMC_JIT wrapper defaults
DESIRED_ENERGY_VAR = 5e-4
NUM_EFF_SAMPLES = 100
SVI_MM_WEIGHT = 10.0 * N_CHAINS     # wrapper default 10*n_chains
RNG_KEY_SEED = 220722               # + group index (+ --seed-offset on escalation)
INIT_ROW_SEED = 880                 # + group index (+ --seed-offset)
BASIN_LO, BASIN_HI = 1.15, 2.0      # E1-G2 containment band
RHAT_GATE = 1.01
ESS_GATE = 1000.0
ROUNDTRIP_TOL = 1e-8                # paper_z_to_marg_z dgamma fence (measured 0.0)
# E1 amendment 2 (ledgered, CAMPAIGN.md 2026-07-22): the PI's own F4 fix.
# First smoke reproduced their documented failure ("windows 2/3 accumulate from
# an empty Welford with no shrinkage ... rank-deficient ... cholesky NaN ->
# rejection cascade", vendored mclmc.py): window-2 cov from 24 samples in 46
# dims -> kernel_nonan 0.19 (healthy exactly until the window-2 install) ->
# zeroed-dE step-size runaway to eps~2174. regularize_mass_matrix=True is
# their opt-in Stan-style window shrinkage designed for exactly this.
REGULARIZE_MM = True
# smoke sized so every mm window has >> dim samples (w2 ~ 34 steps x 8 chains)
SMOKE_CHAINS, SMOKE_BURN, SMOKE_DRAWS = 8, 300, 100
SMOKE_KERNEL_NONAN_GATE = 0.99      # kernel-level (pre-zeroing) non-NaN fraction

MASS_KEYS = [  # the 8 physical mass params (scene unique keys, KEYED access)
    ("theta_E", "planes/0/mass/0/theta_E"), ("gamma", "planes/0/mass/0/gamma"),
    ("e1", "planes/0/mass/0/e1"), ("e2", "planes/0/mass/0/e2"),
    ("center_x", "planes/0/mass/0/center_x"), ("center_y", "planes/0/mass/0/center_y"),
    ("gamma1", "planes/0/mass/1/gamma1"), ("gamma2", "planes/0/mass/1/gamma2"),
]

# comparison values recorded in the checkpoint (harvest-time use ONLY; no gate here)
REF = {"v2d": dict(med=1.4330, ci68=[1.3995, 1.4685],
                   source="foundry-i hmc_v13_v2d (rhat 1.0163, ess 5714)"),
       "v3b": dict(med=1.2941, ci68=[1.2832, 1.3085],
                   source="foundry-i hmc_v13_v3b mass_gamma<1.7 conditional "
                          "(GLOBALLY UNCONVERGED rhat 843 — init cloud only; "
                          "this run is the first convergence-certified diagonal "
                          "fit of this branch)")}


def md5(path, chunk=1 << 22):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def utcnow():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# =========================================================================== #
# Stage PREP — OLD venv (cgl), CPU, bijector only (never the likelihood)
# =========================================================================== #
def stage_prep(tag):
    if sys.executable != OLD_PY:
        env = dict(os.environ, JAX_PLATFORMS="cpu", GIGALENS_X64="1",
                   PYTHONUNBUFFERED="1")
        env.pop("CUDA_VISIBLE_DEVICES", None)
        print(f"[prep:{tag}] re-exec under {OLD_PY} (CPU, bijector only)", flush=True)
        subprocess.run([OLD_PY, str(ROOT / "50_run_mclmc_diag.py"), "prep",
                        "--tag", tag], env=env, cwd=str(OLD), check=True)
        return
    assert os.environ.get("GIGALENS_X64") == "1"
    import jax.numpy as jnp

    from cgl import e2
    from cgl.likelihood import _flat_named46

    src = FI_DATA / f"hmc_v13_{tag}.npz"
    src_md5 = md5(src)
    hz = np.load(src, allow_pickle=True)
    z74_all = np.asarray(hz["samples"], dtype=np.float64).reshape(-1, 74)
    g74_all = np.asarray(hz["mass_gamma"], dtype=np.float64).ravel()
    if tag == "v3b":
        pool = np.flatnonzero(g74_all < V3B_GAMMA_CUT)   # physical branch only
    else:
        pool = np.arange(z74_all.shape[0])               # healthy anchor chain
    rng = np.random.default_rng(MAP_SEED[tag])
    idx = np.sort(rng.choice(pool, size=N_MAP, replace=False))
    if tag == "v3b":
        assert np.all(g74_all[idx] < V3B_GAMMA_CUT)

    tgt = e2.build_target(tag)          # bijector + conversion_factor only
    model = tgt.model
    cf = float(tgt.conversion_factor)
    labels = list(model.index_labels)

    x46 = np.empty((N_MAP, 46), dtype=np.float64)
    g46 = np.empty(N_MAP, dtype=np.float64)
    worst_rt = 0.0
    t0 = time.time()
    for k, i in enumerate(idx):
        z46, rep = e2.paper_z_to_marg_z(z74_all[i], model, ie_scale=cf)
        named = dict(_flat_named46(model.bij.forward(list(jnp.asarray(z46)[:, None]))))
        x46[k] = [float(named[lab]) for lab in labels]
        g46[k] = rep["gamma"]
        worst_rt = max(worst_rt, float(rep["roundtrip_dgamma"]))
        if (k + 1) % 64 == 0:
            print(f"[prep:{tag}] mapped {k + 1}/{N_MAP} "
                  f"({(time.time() - t0) / (k + 1):.2f} s/draw, "
                  f"worst_rt={worst_rt:.3e})", flush=True)
    if worst_rt > ROUNDTRIP_TOL:
        raise RuntimeError(f"paper_z_to_marg_z roundtrip fence: {worst_rt:.3e} "
                           f"> {ROUNDTRIP_TOL}")
    meta = dict(generated_utc=utcnow(), tag=tag, source=str(src),
                source_md5=src_md5, n_map=N_MAP, map_seed=MAP_SEED[tag],
                selection=("mass_gamma<1.7 conditional (physical branch)"
                           if tag == "v3b" else "all draws (healthy chain)"),
                pool_size=int(pool.size), ie_scale=cf,
                worst_roundtrip_dgamma=worst_rt,
                gamma46_med=float(np.median(g46)),
                gamma46_q16=float(np.quantile(g46, 0.16)),
                gamma46_q84=float(np.quantile(g46, 0.84)),
                path="z74 -> cgl.e2.paper_z_to_marg_z -> x46 (S1-certified; "
                     "KEYED via labels46, order never assumed)")
    out = DATA / f"mclmc_warm_{tag}_x46.npz"
    np.savez_compressed(out, x46=x46, labels46=np.array(labels),
                        z74_idx=idx, gamma74=g74_all[idx], gamma46=g46,
                        meta=np.array(json.dumps(meta)))
    print(f"[prep:{tag}] wrote {out} ({(time.time() - t0):.0f} s): "
          f"gamma46 med {meta['gamma46_med']:.4f} "
          f"[{meta['gamma46_q16']:.4f}, {meta['gamma46_q84']:.4f}]", flush=True)


# =========================================================================== #
# GPU-side shared machinery (cgl2 venv)
# =========================================================================== #
def _gpu_setup():
    os.environ.setdefault("GIGALENS_X64", "1")
    os.environ.setdefault("JAX_ENABLE_X64", "1")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("CGL2_GPU", "9"))
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ.setdefault("XLA_FLAGS", "--xla_gpu_autotune_level=0")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


def _build(tag):
    """Target + warm scene-z cloud. Returns (pm, mv, logdens, Z_scene, warm_meta)."""
    sys.path.insert(0, str(ROOT))
    from cgl2 import guards, param_map, paths
    paths.bootstrap_vendor()
    guards.require_vendor_ref()
    guards.require_jax_pin()
    import jax
    import jax.numpy as jnp
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "anchor_arb", ROOT / "10_anchor_arbitration.py")
    arb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(arb)

    refs = arb.load_refs()
    pm, mv, _ = arb.build_pm(tag, refs, diagonal=True)
    logprior_fn, loglik_fn = arb.make_closures(pm)

    def logdens(z):
        return logprior_fn(z) + loglik_fn(z)

    warm_file = DATA / f"mclmc_warm_{tag}_x46.npz"
    warm_md5 = md5(warm_file)
    wz = np.load(warm_file, allow_pickle=True)
    warm_meta = json.loads(str(wz["meta"]))
    warm_meta["warm_x46_md5"] = warm_md5
    labels = [str(s) for s in wz["labels46"]]
    x46 = np.asarray(wz["x46"], dtype=np.float64)

    cache = DATA / f"mclmc_warm_{tag}_scenez.npz"
    if cache.exists():
        cz = np.load(cache, allow_pickle=True)
        if str(cz["x46_md5"]) == warm_md5:
            Z = np.asarray(cz["Z_scene"], dtype=np.float64)
            print(f"[build:{tag}] scene-z cloud from cache {cache}", flush=True)
            return pm, mv, arb, logdens, Z, warm_meta
    t0 = time.time()
    Z = np.empty((x46.shape[0], 46), dtype=np.float64)
    for k in range(x46.shape[0]):
        labeled = dict(zip(labels, x46[k]))          # KEYED (C-8 lesson)
        Z[k] = param_map.scene_z_from_old_labels(mv.model, labeled)
        if (k + 1) % 128 == 0:
            print(f"[build:{tag}] scene-z mapped {k + 1}/{x46.shape[0]}", flush=True)
    np.savez_compressed(cache, Z_scene=Z, x46_md5=np.array(warm_md5))
    print(f"[build:{tag}] scene-z cloud built in {time.time() - t0:.0f} s "
          f"-> {cache}", flush=True)
    return pm, mv, arb, logdens, Z, warm_meta


def _apply_vma_workaround():
    """LEDGERED DEVIATION (E1 amendment, 2026-07-22, CAMPAIGN.md): disable
    shard_map's varying-manual-axes replication CHECK for the vendored driver.

    Root cause (reproduced minimally on the real v2d target BEFORE this fix):
    jax 0.6.2 `jax.shard_map(..., check_vma=True)` + the scene likelihood's
    FFT convolution vjp raises `TypeError: cotangent type does not match
    function output ... complex128[1,33,H,W] ... {V:device}` when
    value_and_grad runs inside the shard_map body. check_vma is a STATIC
    correctness check (replication/varying-axis bookkeeping), not numerics;
    on this run's SINGLE-DEVICE mesh (1 L4 = 1 shard) every collective is the
    identity and the check is vacuous. The vendored file stays byte-identical
    (D1): we rebind the module-level `_shard_map` symbol at runtime from the
    campaign script only."""
    import functools
    import inspect
    from gigalens.jax.experimental import mclmc as _vmclmc
    if getattr(_vmclmc, "_e1_vma_workaround", False):
        return
    try:
        params = inspect.signature(_vmclmc._shard_map).parameters
    except (TypeError, ValueError):
        params = {}
    kw = ("check_vma" if "check_vma" in params
          else "check_rep" if "check_rep" in params else "check_vma")
    _vmclmc._shard_map = functools.partial(_vmclmc._shard_map, **{kw: False})
    _vmclmc._e1_vma_workaround = True
    print(f"[mclmc] E1 VMA workaround active: vendored _shard_map rebound "
          f"with {kw}=False (single-device mesh; ledgered deviation)", flush=True)


def _mclmc_run(logdens, positions, cov_reg, svi_mean, rng_seed, n_chains,
               burn, draws):
    """One seed group through the vendored driver. Returns (samples, params_final)."""
    import jax
    import jax.numpy as jnp
    from blackjax.adaptation.mclmc_adaptation import MCLMCAdaptationState
    from gigalens.jax.experimental.blackjax_updated_utils import (
        _build_kernel_shardmap, init_multi, isokinetic_mclachlan_smart)
    from gigalens.jax.experimental.mclmc import full_mclmc_with_adapt_sharded
    _apply_vma_workaround()

    dim = positions.shape[1]
    kernel = lambda imm: _build_kernel_shardmap(          # noqa: E731
        logdensity_fn=logdens, integrator=isokinetic_mclachlan_smart,
        inverse_mass_matrix=imm)
    rng_key = jax.random.key(int(rng_seed))
    init_key, run_key = jax.random.split(rng_key)
    state = init_multi(jnp.asarray(positions, dtype=jnp.float64), init_key, logdens)
    ld0 = np.asarray(state.logdensity, dtype=np.float64)
    if not np.all(np.isfinite(ld0)):
        raise RuntimeError(f"non-finite logdensity at init: {ld0}")
    print(f"[mclmc] init logdensity range [{ld0.min():.2f}, {ld0.max():.2f}]",
          flush=True)
    params_init = MCLMCAdaptationState(
        L=jnp.sqrt(float(dim)), step_size=0.25 * jnp.sqrt(float(dim)),
        inverse_mass_matrix=jnp.asarray(cov_reg, dtype=jnp.float64))
    return full_mclmc_with_adapt_sharded(
        kernel=kernel, num_burnin_steps=burn, num_results=draws,
        state_init=state, params_init=params_init,
        svi_mean=jnp.asarray(svi_mean, dtype=jnp.float64), rng_key=run_key,
        frac_tune1=FRAC_TUNE[0], frac_tune2=FRAC_TUNE[1], frac_tune3=FRAC_TUNE[2],
        desired_energy_var=DESIRED_ENERGY_VAR, num_chains=n_chains,
        num_effective_samples=NUM_EFF_SAMPLES,
        svi_mass_matrix_weight=SVI_MM_WEIGHT,
        step_size_adapt_use_psmile=False, windowed_mass_matrix=True,
        regularize_mass_matrix=REGULARIZE_MM, progress_bar=False)


def _mass_of(pm, Z_flat, chunk=16384):
    """(N,46) flat-z -> (N,8) physical mass params, KEYED via z_param_names."""
    import jax.numpy as jnp
    cols = []
    for i in range(0, Z_flat.shape[0], chunk):
        u = pm.model.bijector.forward(jnp.asarray(Z_flat[i:i + chunk],
                                                  dtype=jnp.float64))
        cols.append(np.stack([np.asarray(u[k], dtype=np.float64)
                              for _, k in MASS_KEYS], axis=1))
    return np.concatenate(cols, axis=0)


def _gpu_peak_mb():
    import jax
    st = jax.local_devices()[0].memory_stats() or {}
    return float(st.get("peak_bytes_in_use", 0)) / 2**20


def _cov_and_mean(Z, arb_common_n):
    from cgl2.samplers import common
    cov = np.cov(Z.T)
    cov_reg = common.regularize_cov(cov, float(arb_common_n))
    return cov_reg, Z.mean(axis=0)


def _hist_np(samples):
    """Pull the small per-step histories host-side (skip the (S,C,46,46) mm hist)."""
    out = {}
    for k in ("step_size", "L", "nonan", "xi", "energy_change_raw",
              "kernel_nonan", "step_norm"):
        out[k] = np.asarray(getattr(samples, k))
    return out


# =========================================================================== #
# Stage SMOKE
# =========================================================================== #
def stage_smoke(tag):
    _gpu_setup()
    import jax
    import jax.numpy as jnp
    pm, mv, arb, logdens, Z, warm_meta = _build(tag)
    dev = str(jax.local_devices()[0])
    rep = dict(generated_utc=utcnow(), stage="smoke", tag=tag, device=dev,
               CUDA_VISIBLE_DEVICES=os.environ["CUDA_VISIBLE_DEVICES"],
               n_chains=SMOKE_CHAINS, burn=SMOKE_BURN, draws=SMOKE_DRAWS,
               warm=warm_meta)

    g_init = arb.gamma_of(pm, Z)
    rep["warm_cloud_gamma"] = dict(med=float(np.median(g_init)),
                                   min=float(g_init.min()),
                                   max=float(g_init.max()))
    cov_reg, svi_mean = _cov_and_mean(Z, N_MAP)
    rng = np.random.default_rng(INIT_ROW_SEED)
    pos = Z[rng.choice(Z.shape[0], SMOKE_CHAINS, replace=False)]

    t0 = time.time()
    samples, params_final = _mclmc_run(logdens, pos, cov_reg, svi_mean,
                                       rng_seed=990, n_chains=SMOKE_CHAINS,
                                       burn=SMOKE_BURN, draws=SMOKE_DRAWS)
    jax.block_until_ready(samples.position)
    wall = time.time() - t0
    hist = _hist_np(samples)
    draws = np.asarray(samples.position)[:, -SMOKE_DRAWS:, :]
    gam = arb.gamma_of(pm, draws.reshape(-1, draws.shape[-1]))
    res_slice = slice(-SMOKE_DRAWS, None)
    rep.update(
        wall_s=round(wall, 1),
        finite_draws=bool(np.all(np.isfinite(draws))),
        nonan_frac_total=float(np.mean(hist["nonan"])),
        kernel_nonan_frac=float(np.mean(hist["kernel_nonan"])),
        step_size_final=[float(v) for v in hist["step_size"][:, -1]],
        L_final=float(np.asarray(params_final.L)),
        xi_results_med=float(np.median(hist["xi"][:, res_slice])),
        energy_change_raw_results=dict(
            mean_abs=float(np.mean(np.abs(hist["energy_change_raw"][:, res_slice]))),
            max_abs=float(np.max(np.abs(hist["energy_change_raw"][:, res_slice])))),
        gamma_draws=dict(med=float(np.median(gam)), min=float(gam.min()),
                         max=float(gam.max())),
        basin_contained=bool(np.all((gam > BASIN_LO) & (gam < BASIN_HI))),
        peak_mb_smoke=round(_gpu_peak_mb(), 1),
    )

    # horizon probe: batched value_and_grad at PRODUCTION chain count
    vg = jax.jit(jax.vmap(jax.value_and_grad(logdens)))
    zp = jnp.asarray(np.repeat(pos, N_CHAINS // SMOKE_CHAINS, axis=0))
    v, g = vg(zp); jax.block_until_ready(v)          # compile
    t0 = time.time()
    n_rep = 10
    for _ in range(n_rep):
        v, g = vg(zp)
    jax.block_until_ready(v)
    t_batch = (time.time() - t0) / n_rep
    total_steps = BURN + DRAWS
    proj_h_group = total_steps * 2 * t_batch * 1.2 / 3600.0   # mclachlan = 2 grads/step
    rep["grad_probe"] = dict(n_chains=N_CHAINS, t_batch_s=round(t_batch, 4),
                             grads_per_step=2, margin=1.2)
    rep["projected_hours_per_group"] = round(proj_h_group, 2)
    rep["projected_hours_production"] = round(2 * proj_h_group, 2)
    rep["peak_mb_with_probe"] = round(_gpu_peak_mb(), 1)
    hist_mb = total_steps * N_CHAINS * (46 * 46 + 46 + 7) * 8 / 2**20
    rep["analytic_history_mb_production"] = round(hist_mb, 0)

    rep["smoke_ok"] = bool(rep["finite_draws"]
                           and rep["kernel_nonan_frac"] >= SMOKE_KERNEL_NONAN_GATE
                           and rep["nonan_frac_total"] >= 0.99
                           and rep["basin_contained"])
    out = DATA / f"mclmc_diag_{tag}_smoke.json"
    out.write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))
    if not rep["smoke_ok"]:
        sys.exit(1)


# =========================================================================== #
# Stage RUN (production)
# =========================================================================== #
def stage_run(tag, draws_n, seed_offset):
    _gpu_setup()
    import arviz as az
    import jax
    smoke = json.loads((DATA / f"mclmc_diag_{tag}_smoke.json").read_text())
    if not smoke.get("smoke_ok"):
        raise RuntimeError("smoke gates not passed — refusing to run")
    escalated = (draws_n != DRAWS) or (seed_offset != 0)
    pm, mv, arb, logdens, Z, warm_meta = _build(tag)
    cov_reg, svi_mean = _cov_and_mean(Z, N_MAP)
    t_all = time.time()

    groups = []
    for g in range(N_GROUPS):
        rng = np.random.default_rng(INIT_ROW_SEED + seed_offset + g)
        idx = rng.choice(Z.shape[0], N_CHAINS, replace=False)
        pos = Z[idx]
        print(f"[run:{tag}] === seed group {g}: rng_key "
              f"{RNG_KEY_SEED + seed_offset + g}, {N_CHAINS} chains, "
              f"burn {BURN} + draws {draws_n} ===", flush=True)
        t0 = time.time()
        samples, params_final = _mclmc_run(
            logdens, pos, cov_reg, svi_mean,
            rng_seed=RNG_KEY_SEED + seed_offset + g, n_chains=N_CHAINS,
            burn=BURN, draws=draws_n)
        jax.block_until_ready(samples.position)
        wall = time.time() - t0
        hist = _hist_np(samples)
        kept = np.asarray(samples.position)[:, -draws_n:, :]     # (C, D, 46)
        imm_final = np.asarray(samples.inverse_mass_matrix[0, -1])  # device-side index
        del samples
        groups.append(dict(g=g, idx=idx, pos_kept=kept, hist=hist,
                           imm_final=imm_final,
                           L_final=float(np.asarray(params_final.L)),
                           wall_s=wall))
        print(f"[run:{tag}] group {g} done in {wall / 3600.0:.2f} h; "
              f"nonan {np.mean(hist['nonan']):.4f}; "
              f"peak {_gpu_peak_mb():.0f} MB", flush=True)

    # ---- diagnostics -------------------------------------------------------
    print(f"[run:{tag}] mapping physical mass params + diagnostics", flush=True)
    z_names = list(pm.model.z_param_names)
    mass_names = [n for n, _ in MASS_KEYS]
    for grp in groups:
        C, D, dim = grp["pos_kept"].shape
        grp["mass_kept"] = _mass_of(pm, grp["pos_kept"].reshape(-1, dim)
                                    ).reshape(C, D, 8)

    pooled_z = np.concatenate([grp["pos_kept"] for grp in groups], axis=0)
    pooled_m = np.concatenate([grp["mass_kept"] for grp in groups], axis=0)
    gam = pooled_m[:, :, mass_names.index("gamma")]              # (2C, D)

    # E1-G2 basin containment (report, never drop)
    viol = []
    for c in range(gam.shape[0]):
        out_frac = float(np.mean((gam[c] <= BASIN_LO) | (gam[c] >= BASIN_HI)))
        if out_frac > 0:
            viol.append(dict(chain=int(c), group=int(c // N_CHAINS),
                             frac_outside=out_frac,
                             gamma_min=float(gam[c].min()),
                             gamma_max=float(gam[c].max())))

    def rhat_ess(zc, mc):
        """zc (C,D,46), mc (C,D,8) -> per-param rhat/ess dicts (worst-param)."""
        x = np.concatenate([zc, mc], axis=2)
        names = z_names + [f"mass:{n}" for n in mass_names]
        ds = az.convert_to_dataset(x)
        r = np.asarray(az.rhat(ds)["x"])
        e = np.asarray(az.ess(ds, method="bulk")["x"])
        iw, ie_ = int(np.argmax(r)), int(np.argmin(e))
        return dict(rhat_worst=float(r[iw]), rhat_worst_param=names[iw],
                    ess_min=float(e[ie_]), ess_min_param=names[ie_],
                    rhat=dict(zip(names, [float(v) for v in r])),
                    ess=dict(zip(names, [float(v) for v in e])))

    diag_pooled = rhat_ess(pooled_z, pooled_m)
    diag_groups = [rhat_ess(grp["pos_kept"], grp["mass_kept"]) for grp in groups]
    gate_g1 = (diag_pooled["rhat_worst"] < RHAT_GATE
               and diag_pooled["ess_min"] >= ESS_GATE)
    gate_g2 = (len(viol) == 0)

    qs = {q: float(np.quantile(gam, q)) for q in (0.025, 0.16, 0.5, 0.84, 0.975)}
    per_group_gamma = [dict(med=float(np.median(gam[i * N_CHAINS:(i + 1) * N_CHAINS])),
                            q16=float(np.quantile(gam[i * N_CHAINS:(i + 1) * N_CHAINS], 0.16)),
                            q84=float(np.quantile(gam[i * N_CHAINS:(i + 1) * N_CHAINS], 0.84)))
                       for i in range(N_GROUPS)]

    stem = DATA / f"mclmc_diag_{tag}"
    npz = {"z_names": np.array(z_names), "mass_names": np.array(mass_names),
           "cov_reg_init": cov_reg, "svi_mean": svi_mean}
    for grp in groups:
        g = grp["g"]
        npz[f"pos_g{g}"] = grp["pos_kept"]
        npz[f"mass_g{g}"] = grp["mass_kept"]
        npz[f"init_idx_g{g}"] = grp["idx"]
        npz[f"imm_final_g{g}"] = grp["imm_final"]
        for k, v in grp["hist"].items():
            npz[f"{k}_g{g}"] = v
    np.savez_compressed(stem.with_suffix(".npz"), **npz)

    summary = dict(
        generated_utc=utcnow(), stage="run", tag=tag,
        checkpoint="CAMPAIGN.md 2026-07-22 E1 design checkpoint (pre-registered)",
        device=str(jax.local_devices()[0]),
        CUDA_VISIBLE_DEVICES=os.environ["CUDA_VISIBLE_DEVICES"],
        config=dict(n_chains=N_CHAINS, n_groups=N_GROUPS, burn=BURN,
                    draws=draws_n, frac_tune=FRAC_TUNE,
                    desired_energy_var=DESIRED_ENERGY_VAR,
                    num_effective_samples=NUM_EFF_SAMPLES,
                    svi_mass_matrix_weight=SVI_MM_WEIGHT,
                    windowed_mass_matrix=True,
                    regularize_mass_matrix=REGULARIZE_MM,  # ledgered E1 amendment 2 (PI's F4 fix)
                    step_size_adapt_use_psmile=False,
                    integrator="isokinetic_mclachlan_smart",
                    init_L="sqrt(46)", init_step_size="0.25*sqrt(46)",
                    inverse_mass_init="regularize_cov(warm scene-z cov, n=512)",
                    rng_key_seeds=[RNG_KEY_SEED + seed_offset + g
                                   for g in range(N_GROUPS)],
                    init_row_seeds=[INIT_ROW_SEED + seed_offset + g
                                    for g in range(N_GROUPS)],
                    escalated=escalated, seed_offset=seed_offset,
                    vma_check_disabled=True,  # ledgered E1 amendment (see _apply_vma_workaround)
                    sampler="vendored gigalens-linus experimental MCLMC "
                            "(full_mclmc_with_adapt_sharded, UNPATCHED on disk, "
                            "runtime _shard_map check_vma=False rebind, "
                            "MCLMC_JIT wrapper bypassed)"),
        provenance=dict(
            warm=warm_meta,
            parity_refs_md5=md5(DATA / "parity_refs.npz"),
            warm_x46_md5=md5(DATA / f"mclmc_warm_{tag}_x46.npz"),
            source_chain_md5=warm_meta.get("source_md5"),
            script="50_run_mclmc_diag.py"),
        reference_for_harvest=REF[tag],
        wall_h=[round(grp["wall_s"] / 3600.0, 3) for grp in groups],
        wall_h_total=round((time.time() - t_all) / 3600.0, 3),
        peak_mb=round(_gpu_peak_mb(), 1),
        nonan_frac=[float(np.mean(grp["hist"]["nonan"])) for grp in groups],
        kernel_nonan_frac=[float(np.mean(grp["hist"]["kernel_nonan"]))
                           for grp in groups],
        L_final=[grp["L_final"] for grp in groups],
        step_size_final_med=[float(np.median(grp["hist"]["step_size"][:, -1]))
                             for grp in groups],
        gates=dict(
            E1_G1=dict(rhat_gate=RHAT_GATE, ess_gate=ESS_GATE,
                       rhat_worst=diag_pooled["rhat_worst"],
                       rhat_worst_param=diag_pooled["rhat_worst_param"],
                       ess_min=diag_pooled["ess_min"],
                       ess_min_param=diag_pooled["ess_min_param"],
                       passed=bool(gate_g1)),
            E1_G2=dict(band=[BASIN_LO, BASIN_HI], violations=viol,
                       passed=bool(gate_g2)),
            note="gamma quotable ONLY on E1-G1 pass (checkpoint rule); "
                 "harvest/verdict happens in a later session, plots first"),
        diagnostics=dict(pooled=diag_pooled, per_group=diag_groups),
        gamma=dict(pooled_quantiles=qs, per_group=per_group_gamma,
                   quotable=bool(gate_g1)),
    )
    stem.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    slim = {k: v for k, v in summary.items() if k != "diagnostics"}
    slim["diagnostics_pooled"] = {k: v for k, v in diag_pooled.items()
                                  if k in ("rhat_worst", "rhat_worst_param",
                                           "ess_min", "ess_min_param")}
    print(json.dumps(slim, indent=2))
    print(f"[run:{tag}] COMPLETE -> {stem}.npz/.json "
          f"(wall {summary['wall_h_total']:.2f} h)", flush=True)


# =========================================================================== #
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prep", "smoke", "run"])
    ap.add_argument("--tag", required=True, choices=["v2d", "v3b"])
    ap.add_argument("--draws", type=int, default=DRAWS,
                    help="escalation only (checkpoint: single doubling allowed)")
    ap.add_argument("--seed-offset", type=int, default=0,
                    help="escalation only (checkpoint: +10)")
    a = ap.parse_args()
    if a.stage == "prep":
        stage_prep(a.tag)
    elif a.stage == "smoke":
        stage_smoke(a.tag)
    else:
        stage_run(a.tag, a.draws, a.seed_offset)
