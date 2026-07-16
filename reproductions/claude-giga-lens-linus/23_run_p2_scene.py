#!/usr/bin/env python
"""23_run_p2_scene.py -- P2 scene-API benchmark cells B1/B2: one target, one arm,
one GPU (Perlmutter Path A, cgl2-pm venv; design checkpoints in CAMPAIGN.md,
written BEFORE submission -- P2 wave 2026-07-16).

Arms:
  s1  : prior-seeded MC-SMC, MAMS mutation, frozen P0 protocol
        (cgl2.samplers.common.run_tempered_smc, target_ess 0.7, 4 draws/stage,
        per-lambda ensemble preconditioning, pilot 64, eps0 0.5, boot_seed
        20260715+seed) with the CHUNKED-MUTATION MAMS subclass from 22_run_b3.py
        (identical math + identical per-particle PRNG keys; execution order
        only -- bit-identity vs the stock kernel verified in the B3 checkpoint).
  s6b : MAMS-alone-warm baseline (B1 only). Two stages, ALL cost billed:
        (a) MAP: multi-start optax.adabelief(1e-2) (their pipeline default,
            replicated from GIGALens-Code pipeline.py:1328-1336 with
            attribution -- the vendored inference.py MAP does not run under
            jax 0.6.2, ledgered defect E3) on ProbModel.log_prob, chunked
            batched value_and_grad; n_grad billed = starts * steps.
        (b) MAMS-alone at lambda=1: C chains started from the C best MAP
            endpoints, ensemble-preconditioned (regularized weighted particle
            covariance -- the SAME preconditioning the SMC kernel uses, so the
            comparison isolates tempering+resampling, not the metric); burn
            rounds re-tune (dual-averaging pilot) + re-estimate (mu, chol);
            sampling rounds run with tuning FROZEN; draws recorded per round.
        ESS for the efficiency gate is computed at HARVEST (tfp
        effective_sample_size on the saved draws; figs first, house rule).

No gate math here: the runner writes raw artifacts + sampler-health traces
only. Harvest (phoenix) draws figs FIRST, then evaluates the pre-registered
B1/B2 gates.

B1 DATA NOTE (honesty rule): carousel33 runs on the SEEDED MOCK stand-in --
the real MUSE cutouts remain permission-blocked on Perlmutter (re-verified
2026-07-16: /global/u1/l/linusu/... Permission denied). zoo.py records this
in target.provenance/meta, and the run json carries both verbatim.

Run (Perlmutter shared QOS, 1 GPU):
  python 23_run_p2_scene.py --target carousel33 --arm s1 --n 512 --seed 2 \
      --chunk 64 --out $OUTDIR/b1_carousel33_s1_seed2
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# env BEFORE jax init (A100 SMC policy: autotune off + priority-fusion disable,
# the old campaign's validated flag set; harmless if already exported)
_flags = os.environ.get("XLA_FLAGS", "")
for f in ("--xla_gpu_autotune_level=0", "--xla_disable_hlo_passes=priority-fusion"):
    if f.split("=")[0] not in _flags:
        _flags = (_flags + " " + f).strip()
os.environ["XLA_FLAGS"] = _flags
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("GIGALENS_X64", "1")

HERE = Path(__file__).resolve().parent

import cgl2  # noqa: F401,E402  x64 bootstrap BEFORE jax import
from cgl2 import guards, paths  # noqa: E402

paths.bootstrap_vendor()
guards.require_vendor_ref()
guards.require_jax_pin()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from cgl2 import zoo  # noqa: E402
from cgl2.samplers import common, smc_micro  # noqa: E402

# chunked MAMS: reuse 22_run_b3.py's checkpointed subclass (never copy)
_spec = importlib.util.spec_from_file_location("run_b3", HERE / "22_run_b3.py")
_run_b3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_run_b3)
make_chunked_mams = _run_b3.make_chunked_mams

# frozen s6b recipe constants (design checkpoint, CAMPAIGN.md 2026-07-16)
S6B_MAP_STARTS = 64
S6B_MAP_STEPS = 350           # classic inference.py default (B3 convention)
S6B_CHAINS = 64
S6B_BURN_ROUNDS = 30          # ensemble re-tune + re-precondition per round
S6B_SAMPLE_ROUNDS = 120       # tuning FROZEN; 4 MAMS draws per round
NUM_MCMC_STEPS = smc_micro.NUM_MCMC_STEPS   # 4 (frozen P0)


def _jsonable(x):
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return x


def gpu_peak_mb():
    st = jax.local_devices()[0].memory_stats() or {}
    return float(st.get("peak_bytes_in_use", 0)) / 2**20


def constrained_readout(target, Z):
    """Best-effort constrained-space arrays for the harvest (never load-bearing:
    harvest recomputes from z + the deterministic target builder). Grouped/tuple
    prior keys (dspl ratio coords) may not flatten -- then only z is saved."""
    out = {}
    try:
        pm = target.prob_model
        u = pm.model.bijector.forward(jnp.asarray(Z, dtype=jnp.float64))
        for k, v in u.items():
            try:
                arr = np.asarray(v, dtype=np.float64)
            except Exception:
                continue
            if arr.ndim == 1 and arr.shape[0] == np.shape(Z)[0]:
                out[str(k).replace("/", "__")] = arr
    except Exception as e:  # readout is optional by design
        print(f"[readout] constrained readout skipped: {e!r}", flush=True)
    return out


def make_chunked_loglik_batch(loglik_fn, chunk: int):
    """Execution-order-only chunked weight-step evaluator (lax.map over the
    SAME per-particle vmap; per-particle math identical by construction).
    Passed through run_tempered_smc's keyword-only loglik_batch_fn seam
    (default None = the validated full-vmap path, bit-for-bit)."""
    ll_vmap = jax.vmap(loglik_fn)

    @jax.jit
    def batch(Z):
        n, d = Z.shape
        k = n // chunk
        out = jax.lax.map(ll_vmap, Z[: k * chunk].reshape(k, chunk, d))
        out = jnp.reshape(out, (-1,))
        if n % chunk:
            out = jnp.concatenate([out, ll_vmap(Z[k * chunk:])])
        return out

    return batch


def arm_s1(target, args):
    key = jax.random.PRNGKey(int(args.seed))
    z0 = np.asarray(target.prior_sample_fn(key, int(args.n)), dtype=np.float64)
    kern = make_chunked_mams(int(args.chunk))   # instance; num_mcmc_steps=4 frozen
    llb = (make_chunked_loglik_batch(target.loglik_fn, int(args.forward_chunk))
           if int(args.forward_chunk) > 0 else None)
    t0 = time.time()
    res = common.run_tempered_smc(
        target.logprior_fn, target.loglik_fn, z0, jax.random.PRNGKey(int(args.seed)),
        kernel=kern, target_ess=smc_micro.TARGET_ESS, max_stages=400,
        n_boot=200, pilot_size=64, eps0=0.5, precondition=True,
        boot_seed=20260715 + int(args.seed), loglik_batch_fn=llb)
    wall = time.time() - t0
    parts = np.asarray(res.pop("particles"), dtype=np.float64)
    summary = dict(
        arm="s1", kernel=res.pop("kernel"), wall_s=round(wall, 1),
        n_particles=int(args.n), chunk=int(args.chunk),
        forward_chunk=int(args.forward_chunk), seed=int(args.seed),
        peak_mb=round(gpu_peak_mb(), 1), **_jsonable(res))
    extras = constrained_readout(target, parts)
    return parts, summary, extras


def _their_map_optimizer():
    """GIGALens-Code pipeline.py::_default_map_optimizer, replicated with
    attribution (same as 22_run_b3.py): adabelief(1e-2), nesterov when
    supported."""
    import inspect

    import optax
    kwargs = ({"nesterov": True}
              if "nesterov" in inspect.signature(optax.adabelief).parameters
              else {})
    return optax.adabelief(1e-2, **kwargs)


def arm_s6b(target, args):
    """MAP multistart (billed) -> MAMS-alone at lambda=1 (ensemble-preconditioned)."""
    import optax

    dim = int(target.z_dim)
    chunk = int(args.chunk)
    logprior_fn, loglik_fn = target.logprior_fn, target.loglik_fn

    def logpost(z):
        return logprior_fn(z) + loglik_fn(z)

    # ---- (a) MAP multistart, chunked batched value_and_grad -------------------
    k_map = jax.random.PRNGKey(int(args.seed) + 1000)
    Z = jnp.asarray(np.asarray(
        target.prior_sample_fn(k_map, S6B_MAP_STARTS), dtype=np.float64))
    vg_one = jax.value_and_grad(lambda z: -logpost(z))

    def chunked_vg(Zb):
        n = Zb.shape[0]
        Zc = Zb.reshape(n // chunk if n % chunk == 0 else 1, -1, dim) \
            if n % chunk == 0 else Zb.reshape(1, n, dim)
        vals, grads = jax.lax.map(lambda zc: jax.vmap(vg_one)(zc), Zc)
        return vals.reshape(n), grads.reshape(n, dim)

    opt = _their_map_optimizer()
    state = opt.init(Z)

    @jax.jit
    def map_step(Z, state):
        vals, grads = chunked_vg(Z)
        updates, state = opt.update(grads, state, Z)
        return optax.apply_updates(Z, updates), state, vals

    t0 = time.time()
    best_val = np.full(S6B_MAP_STARTS, np.inf)
    best_Z = np.asarray(Z, dtype=np.float64).copy()
    for it in range(S6B_MAP_STEPS):
        Z, state, vals = map_step(Z, state)
        v = np.asarray(vals, dtype=np.float64)
        upd = v < best_val                     # per-start best tracking
        if upd.any():
            best_val[upd] = v[upd]
            best_Z[upd] = np.asarray(Z, dtype=np.float64)[upd]
        if it % 50 == 0:
            print(f"[s6b-map] step {it}: best -logpost {best_val.min():.3f}",
                  flush=True)
    map_wall = time.time() - t0
    n_grad_map = S6B_MAP_STARTS * S6B_MAP_STEPS
    order = np.argsort(best_val)
    chains0 = best_Z[order[:S6B_CHAINS]] if S6B_CHAINS <= S6B_MAP_STARTS \
        else best_Z[order]
    print(f"[s6b-map] done {map_wall:.0f}s best logpost {-best_val.min():.3f} "
          f"({n_grad_map} grads billed)", flush=True)

    # ---- (b) MAMS-alone at lambda=1, ensemble preconditioning ----------------
    kern = make_chunked_mams(chunk)             # instance; num_mcmc_steps=4 frozen
    kern.build(logprior_fn, loglik_fn, dim)
    from scipy.linalg import solve_triangular

    z = chains0.copy()
    C = z.shape[0]
    key = jax.random.PRNGKey(int(args.seed) + 2000)
    rng = np.random.default_rng(20260716 + int(args.seed))
    grad_evals = dict(tune=0, mutate=0)
    eps_warm = 0.5
    draws = []
    trace = []
    tuning = None
    mu = chol = None
    t0 = time.time()
    for r in range(S6B_BURN_ROUNDS + S6B_SAMPLE_ROUNDS):
        burn = r < S6B_BURN_ROUNDS
        if burn or mu is None:
            mu = z.mean(axis=0)
            cov = common.regularize_cov(np.cov(z, rowvar=False), float(C))
            chol = np.linalg.cholesky(cov)
        U = jnp.asarray(solve_triangular(chol, (z - mu[None, :]).T, lower=True).T)
        key, k_pilot, k_mut = jax.random.split(key, 3)
        if burn:
            pilot_idx = rng.choice(C, size=min(64, C), replace=False)
            tuning = kern.tune(k_pilot, 1.0, jnp.asarray(mu), jnp.asarray(chol),
                               U[pilot_idx], eps_warm)
            grad_evals["tune"] += tuning.n_grad
            eps_warm = tuning.step_size
        U_new, stats = kern.mutate(k_mut, 1.0, jnp.asarray(mu),
                                   jnp.asarray(chol), U, tuning)
        grad_evals["mutate"] += int(stats["n_grad"])
        z = np.asarray(mu[None, :]
                       + np.asarray(U_new, dtype=np.float64) @ chol.T)
        if not burn:
            draws.append(z.copy())
        trace.append(dict(round=r, burn=bool(burn),
                          accept=float(stats["accept_mean"]),
                          eps=float(tuning.step_size), n_int=int(tuning.n_int)))
        if r % 20 == 0:
            print(f"[s6b-mams] round {r} accept {trace[-1]['accept']:.3f} "
                  f"eps {trace[-1]['eps']:.4f} n_int {trace[-1]['n_int']}",
                  flush=True)
    mams_wall = time.time() - t0
    draws = np.stack(draws)                      # (T, C, dim)
    grad_evals["map"] = n_grad_map
    grad_evals["total"] = sum(grad_evals.values())
    summary = dict(
        arm="s6b", kernel=kern.name, seed=int(args.seed), chunk=chunk,
        map_starts=S6B_MAP_STARTS, map_steps=S6B_MAP_STEPS,
        map_best_logpost=float(-best_val.min()), map_wall_s=round(map_wall, 1),
        chains=C, burn_rounds=S6B_BURN_ROUNDS, sample_rounds=S6B_SAMPLE_ROUNDS,
        num_mcmc_steps=NUM_MCMC_STEPS, mams_wall_s=round(mams_wall, 1),
        grad_evals=grad_evals, round_trace=trace,
        peak_mb=round(gpu_peak_mb(), 1),
        billing_note="grad ledger includes the MAP warm-start cost "
                     "(pre-registered: S6b MAP cost billed)")
    extras = dict(draws=draws, map_best_z=chains0,
                  **constrained_readout(target, draws[-1]))
    return draws[-1], summary, extras


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", required=True,
                    choices=["carousel33", "dspl20_orig", "dspl20_ratio"])
    ap.add_argument("--arm", required=True, choices=["s1", "s6b"])
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--chunk", type=int, default=64,
                    help="mutation (and s6b MAP) chunk size; execution order only")
    ap.add_argument("--forward-chunk", type=int, default=0,
                    help="0 = stock full-vmap weight step (validated path); "
                         ">0 = lax.map-chunked weight step (memory only)")
    ap.add_argument("--out", required=True, help="output prefix (no suffix)")
    args = ap.parse_args()

    guards.require_gpu()
    print(f"[23] devices={jax.devices()} x64={jax.config.jax_enable_x64}",
          flush=True)
    t0 = time.time()
    target = zoo.build(args.target)
    print(f"[23] built {target.name} dim={target.z_dim} in {time.time()-t0:.0f}s",
          flush=True)
    print(f"[23] provenance: {target.provenance}", flush=True)

    if args.arm == "s1":
        parts, summary, extras = arm_s1(target, args)
    else:
        if args.target != "carousel33":
            raise SystemExit("s6b is a B1 (carousel33) arm only")
        parts, summary, extras = arm_s6b(target, args)

    out = dict(script="23_run_p2_scene.py",
               generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               target=target.name, z_dim=int(target.z_dim),
               provenance=target.provenance, meta=_jsonable(target.meta),
               jax=jax.__version__, device=str(jax.local_devices()[0]),
               config=vars(args), **summary)
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    npz = dict(particles=parts.astype(np.float64),
               z_param_names=np.array(target.meta.get("z_param_names", []),
                                      dtype=object))
    for k, v in extras.items():
        npz[k] = v
    np.savez(outp.with_suffix(".npz"), **npz)
    json.dump(out, open(outp.with_suffix(".json"), "w"), indent=1, default=float)
    print(f"[23] wrote {outp}.npz + .json "
          f"(logZ={out.get('logZ', 'n/a')}, stages={out.get('n_stages', 'n/a')})",
          flush=True)


if __name__ == "__main__":
    main()
