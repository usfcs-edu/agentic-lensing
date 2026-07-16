#!/usr/bin/env python
"""22_run_b3.py -- P2 B3 cell runner (Front C): one hs2 system, one arm, one GPU.

Pre-registered design: research/checkpoints_b3.md (WRITTEN BEFORE any run).
Arms:
  ref : classic scene-API MAP -> SVI -> HMC via the VENDORED
        gigalens.jax.inference.ModellingSequence (diagonal likelihood; the
        GIGA-Lens baseline recipe = inference.py defaults, optimizers = their
        pipeline defaults adabelief(1e-2)/adabelief(1e-4, b1=.95, b2=.99)
        replicated from GIGALens-Code src/gigalens_research/inference_utils/
        pipeline.py:1328-1336 -- replicated, never imported).
  smc : prior-seeded MC-SMC, MAMS mutation, N=512, frozen P0 protocol
        (cgl2.samplers.common.run_tempered_smc) with a CHUNKED-MUTATION MAMS
        subclass (chunk=128): identical math + identical per-particle PRNG keys
        as smc_micro._MAMSKernel; only the vmap over particles executes as
        sequential 128-chunks inside one jit (memory fit -- batched grad at 512
        measured OOM on L4 23GB / A16 15.3GB; see checkpoint sizing probe).

Writes: data/b3_run_<arm>_sys<i>_<devtag>.json (+ .npz samples/particles).
No gate math here -- harvest (22_run_b3_harvest.py) draws figs FIRST, then gates.

Run:  CUDA_VISIBLE_DEVICES=<d> GIGALENS_X64=1 python 22_run_b3.py \
          --sys 0 --arm smc --devtag a16-0
"""
from __future__ import annotations

import argparse
import json
import os
import time
import traceback

import numpy as np

# env BEFORE jax init (house patterns; aarch64 XLA livelock + PCI device order)
if "--xla_disable_hlo_passes" not in os.environ.get("XLA_FLAGS", ""):
    os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "")
                               + " --xla_disable_hlo_passes=priority-fusion").strip()
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import cgl2  # noqa: F401,E402  x64 bootstrap BEFORE jax import
from cgl2 import guards, paths  # noqa: E402

# frozen recipe constants (checkpoint; inference.py classic defaults)
REF_MAP_N, REF_MAP_STEPS = 500, 350
REF_SVI_NVI, REF_SVI_STEPS = 250, 500
REF_HMC_N, REF_HMC_BURN, REF_HMC_RES = 50, 250, 750
REF_HMC_INIT_L, REF_HMC_MAX_L = 3, 30
REF_ESCALATED_BURN, REF_ESCALATED_RES = 500, 1500   # their hundredsystems values
SMC_N, SMC_CHUNK = 512, 128
RHAT_MAX, ESS_MIN = 1.05, 200.0


def _jsonable(x):
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return x


def _their_map_optimizer():
    """Replicates GIGALens-Code pipeline.py::_default_map_optimizer (attribution
    in module docstring): adabelief(1e-2), nesterov=True when supported."""
    import inspect
    import optax
    kwargs = ({"nesterov": True}
              if "nesterov" in inspect.signature(optax.adabelief).parameters
              else {})
    return optax.adabelief(1e-2, **kwargs)


def _their_svi_optimizer():
    import optax
    return optax.adabelief(1e-4, b1=0.95, b2=0.99)


# --------------------------------------------------------------------------- #
# chunked MAMS mutation (checkpoint-pre-registered execution-order-only change)
# --------------------------------------------------------------------------- #
def make_chunked_mams(chunk: int = SMC_CHUNK):
    from cgl2.samplers import smc_micro

    class ChunkedMAMS(smc_micro._MAMSKernel):
        """_MAMSKernel with the particle vmap executed in sequential chunks.

        Identical math: same per-particle keys (jax.random.split(rng, N) exactly
        as the parent), same kernel/integrator/tuning; jax.lax.map over
        N//chunk chunks replaces one N-wide vmap (memory, not numerics).
        Gradient billing inherited unchanged."""

        name = f"mams_chunk{chunk}"

        def _get_mutate(self, n_int: int, n_mcmc: int):
            key = (n_int, n_mcmc)
            if key in self._cache:
                return self._cache[key]
            import blackjax
            import jax
            import jax.numpy as jnp
            from blackjax.mcmc.integrators import isokinetic_mclachlan

            logprior_fn, loglik_fn = self.logprior_fn, self.loglik_fn

            def mutate(rng, U, lam, mu, chol, eps):
                def logdens(u):
                    z = mu + chol @ u
                    return logprior_fn(z) + lam * loglik_fn(z)

                kern = blackjax.adjusted_mclmc.build_kernel(
                    logdens, integrator=isokinetic_mclachlan,
                    inverse_mass_matrix=1.0)

                def one(u, k):
                    st = blackjax.adjusted_mclmc.init(u, logdens)

                    def body(st, kk):
                        st2, info = kern(kk, st, step_size=eps,
                                         num_integration_steps=n_int)
                        return st2, info.acceptance_rate

                    st, accs = jax.lax.scan(body, st,
                                            jax.random.split(k, n_mcmc))
                    return st.position, jnp.mean(accs)

                keys = jax.random.split(rng, U.shape[0])   # == parent keys
                n, dim = U.shape
                if n <= chunk:
                    U_new, accs = jax.vmap(one)(U, keys)
                    return U_new, jnp.mean(accs)
                assert n % chunk == 0, (n, chunk)
                Uc = U.reshape(n // chunk, chunk, dim)
                Kc = keys.reshape(n // chunk, chunk, *keys.shape[1:])
                U_new, accs = jax.lax.map(
                    lambda t: jax.vmap(one)(t[0], t[1]), (Uc, Kc))
                return U_new.reshape(n, dim), jnp.mean(accs)

            import jax
            fn = jax.jit(mutate)
            self._cache[key] = fn
            return fn

    return ChunkedMAMS()


# --------------------------------------------------------------------------- #
# arms
# --------------------------------------------------------------------------- #
def run_smc_arm(tgt, sys_idx: int) -> dict:
    import jax
    from cgl2.samplers import common

    z0 = np.asarray(tgt.prior_sample_fn(jax.random.PRNGKey(4000 + sys_idx),
                                        SMC_N), dtype=np.float64)
    kern = make_chunked_mams(SMC_CHUNK)
    t0 = time.time()
    out = common.run_tempered_smc(
        tgt.logprior_fn, tgt.loglik_fn, z0, jax.random.PRNGKey(4100 + sys_idx),
        kernel=kern, target_ess=0.7, max_stages=400, n_boot=200,
        pilot_size=64, eps0=0.5, precondition=True,
        boot_seed=20260715 + sys_idx)
    wall = time.time() - t0
    parts = out.pop("particles")
    res = dict(
        arm="smc", n_particles=SMC_N, chunk=SMC_CHUNK,
        seed_z0=4000 + sys_idx, seed_run=4100 + sys_idx,
        wall_s=wall, **{k: _jsonable(v) for k, v in out.items()})
    res["n_unique_final"] = int(len(np.unique(parts.round(12), axis=0)))
    return res, dict(particles=parts, z0=z0)


def _hmc_chains_first_to_samples(x, n_hmc, num_results):
    """Normalize their HMC output to (num_results, n_chains, dim)."""
    x = np.asarray(x)
    if x.ndim != 3:
        raise RuntimeError(f"unexpected HMC output shape {x.shape}")
    if x.shape[0] == n_hmc and x.shape[1] == num_results:
        return np.swapaxes(x, 0, 1)
    if x.shape[0] == num_results and x.shape[1] == n_hmc:
        return x
    raise RuntimeError(f"cannot orient HMC output shape {x.shape} "
                       f"(n_hmc={n_hmc}, num_results={num_results})")


def _map_replicated(prob_model, optimizer, n_samples, num_steps, seed):
    """Classic multi-start MAP, replicated with attribution from the vendored
    gigalens/jax/inference.py::ModellingSequence.MAP (output_type='best'),
    minus the shard_map wrapper (1-device mesh => semantically identity; the
    vendored path raises a jax-0.6.2 shard_map cotangent TypeError -- see the
    checkpoint amendment). Their exact step order is kept, including the
    record-after-update pairing of (params, lp) inside the scan."""
    import jax
    import jax.numpy as jnp
    import optax

    start = prob_model.prior.sample(int(n_samples),
                                    seed=jax.random.PRNGKey(seed))
    params = prob_model.bij.inverse(start)
    map_prob = (prob_model.with_map_remat()
                if hasattr(prob_model, "with_map_remat") else prob_model)

    def loss(z):
        lp, chisq = map_prob.log_prob(z)
        return -jnp.mean(lp) / map_prob.loss_normalization, (lp, chisq)

    loss_and_grad = jax.value_and_grad(loss, has_aux=True)

    def one_step(carry, _):
        p, opt_state = carry
        (_, (lp, chisq)), grads = loss_and_grad(p)
        updates, opt_state = optimizer.update(grads, opt_state)
        p = optax.apply_updates(p, updates)
        i = jnp.nanargmax(lp)              # their post-update pairing, kept
        return (p, opt_state), (p[i], lp[i], chisq[i])

    @jax.jit
    def run(p0):
        opt_state = optimizer.init(p0)
        _, best = jax.lax.scan(one_step, (p0, opt_state), length=num_steps)
        return best

    bp, blp, bchi = run(params)
    j = int(jnp.nanargmax(blp))
    return np.asarray(bp[j]), float(blp[j]), np.asarray(bchi)


def _svi_replicated(prob_model, start, optimizer, n_vi, num_steps, seed,
                    init_scales=1e-3):
    """Classic SVI (full-rank MVN-TriL surrogate), replicated with attribution
    from inference.py::ModellingSequence.SVI, dev_cnt=1 path (same surrogate
    construction: FillScaleTriL(Exp, diag_shift=1e-6), init scale diag*1e-3;
    same best-by-loss tracking BEFORE the optimizer update; same per-step key
    split pattern)."""
    import jax
    import jax.numpy as jnp
    import optax
    import tensorflow_probability.substrates.jax as tfp

    tfd, tfb = tfp.distributions, tfp.bijectors
    prec = jnp.float64 if prob_model.high_precision else jnp.float32
    start_np = np.asarray(jnp.squeeze(start))
    n_params = int(start_np.size)
    scale = (jnp.diag(jnp.ones(n_params, dtype=prec))
             * jnp.asarray(init_scales, prec))
    cov_bij = tfb.FillScaleTriL(diag_bijector=tfb.Exp(),
                                diag_shift=jnp.asarray(1e-6, prec))
    params0 = jnp.asarray(np.concatenate(
        [start_np, np.asarray(cov_bij.inverse(scale))], axis=0)).astype(prec)

    def neg_elbo(params, key):
        mean, cov_raw = params[:n_params], params[n_params:]
        qz = tfd.MultivariateNormalTriL(loc=mean,
                                        scale_tril=cov_bij.forward(cov_raw))
        z = qz.sample(int(n_vi), seed=key)
        return jnp.mean(qz.log_prob(z) - prob_model.log_prob(z)[0])

    vg = jax.value_and_grad(neg_elbo)

    def one(carry, _):
        params, opt_state, key, best_p, best_l = carry
        key, curr = jax.random.split(key)
        my_key = jax.random.split(curr, 1)[0]   # their dev_cnt=1 stream
        val, grad = vg(params, my_key)
        better = val < best_l
        best_p = jax.lax.select(better, params, best_p)
        best_l = jax.lax.select(better, val, best_l)
        updates, opt_state = optimizer.update(grad, opt_state)
        params = optax.apply_updates(params, updates)
        return (params, opt_state, key, best_p, best_l), val

    @jax.jit
    def run(p0, key):
        opt_state = optimizer.init(p0)
        (_, _, _, bp, _), hist = jax.lax.scan(
            one, (p0, opt_state, key, p0, jnp.inf), length=num_steps)
        return bp, hist

    bp, hist = run(params0, jax.random.PRNGKey(seed))
    qz = tfd.MultivariateNormalTriL(
        loc=bp[:n_params], scale_tril=cov_bij.forward(bp[n_params:]))
    return qz, np.asarray(hist)


def _hmc_replicated(prob_model, qz, n_hmc, burn, res, init_eps, init_l,
                    max_l, seed):
    """Classic HMC stage, replicated with attribution from
    inference.py::ModellingSequence.HMC, dev_cnt=1 path (identical tfp kernel
    stack: PreconditionedHMC(momentum = inv(qz cov)) wrapped by
    GradientBasedTrajectoryLengthAdaptation(0.8*burnin, max_leapfrog) and
    DualAveragingStepSizeAdaptation; identical fold_in seed derivation)."""
    import jax
    import jax.numpy as jnp
    import tensorflow_probability.substrates.jax as tfp

    tfd, tfe = tfp.distributions, tfp.experimental
    momentum_distribution = tfd.MultivariateNormalFullCovariance(
        loc=jnp.zeros_like(qz.mean()),
        covariance_matrix=jnp.linalg.inv(qz.covariance()))

    def log_prob(z):
        return prob_model.log_prob(z)[0]

    seed0 = jax.random.split(
        jax.random.fold_in(jax.random.PRNGKey(seed), 0), 1)[0]
    num_adapt = int(burn * 0.8)
    kern = tfe.mcmc.PreconditionedHamiltonianMonteCarlo(
        target_log_prob_fn=log_prob,
        momentum_distribution=momentum_distribution,
        step_size=init_eps, num_leapfrog_steps=init_l)
    kern = tfe.mcmc.GradientBasedTrajectoryLengthAdaptation(
        kern, num_adaptation_steps=num_adapt, max_leapfrog_steps=max_l)
    kern = tfp.mcmc.DualAveragingStepSizeAdaptation(
        inner_kernel=kern, num_adaptation_steps=num_adapt)

    @jax.jit
    def run(k):
        start = qz.sample(int(n_hmc), seed=k)
        return tfp.mcmc.sample_chain(
            num_results=int(res), num_burnin_steps=int(burn),
            current_state=start, trace_fn=None, seed=k, kernel=kern)

    return np.asarray(jax.block_until_ready(run(seed0)))


def run_ref_arm(tgt, sys_idx: int) -> dict:
    import jax
    import jax.numpy as jnp
    import tensorflow_probability.substrates.jax as tfp

    pm = tgt.prob_model
    log = dict(arm="ref",
               recipe="classic MAP->SVI->HMC, replicated single-device from "
                      "vendored inference.py (checkpoint amendment: vendored "
                      "MAP raises shard_map cotangent TypeError on jax 0.6.2)")
    oom_fallbacks = []

    def _oom_retry(fn, label, *sizes):
        """Run fn(size) with the pre-registered halving fallback on OOM."""
        for s in sizes:
            try:
                return s, fn(s)
            except Exception as e:  # noqa: BLE001
                if "RESOURCE_EXHAUSTED" in str(e) or "Out of memory" in str(e):
                    oom_fallbacks.append(dict(stage=label, failed_size=s,
                                              error=str(e)[:200]))
                    continue
                raise
        raise RuntimeError(f"{label}: all sizes OOM ({sizes})")

    # ---- MAP -------------------------------------------------------------------
    t0 = time.time()
    n_map, (map_best, map_lp, map_chisq) = _oom_retry(
        lambda n: _map_replicated(pm, _their_map_optimizer(), n,
                                  REF_MAP_STEPS, seed=0),
        "map", REF_MAP_N, 256, 128)
    log["map"] = dict(n_samples=n_map, num_steps=REF_MAP_STEPS, seed=0,
                      wall_s=time.time() - t0,
                      best_lp=float(map_lp),
                      final_chisq_min=float(np.nanmin(map_chisq)),
                      grad_evals=n_map * REF_MAP_STEPS)

    # ---- SVI -------------------------------------------------------------------
    t0 = time.time()
    n_vi, (qz, svi_loss) = _oom_retry(
        lambda n: _svi_replicated(pm, map_best, _their_svi_optimizer(), n,
                                  REF_SVI_STEPS, seed=1),
        "svi", REF_SVI_NVI, 128, 64)
    log["svi"] = dict(n_vi=n_vi, num_steps=REF_SVI_STEPS, seed=1,
                      wall_s=time.time() - t0,
                      final_neg_elbo=float(np.asarray(svi_loss)[-1]),
                      best_neg_elbo=float(np.nanmin(np.asarray(svi_loss))),
                      grad_evals=n_vi * REF_SVI_STEPS)

    # ---- HMC (+ one pre-registered escalation on Rhat/ESS failure) --------------
    def _hmc(burn, res):
        t0 = time.time()
        raw = _hmc_replicated(pm, qz, REF_HMC_N, burn, res, init_eps=0.3,
                              init_l=REF_HMC_INIT_L,
                              max_l=REF_HMC_MAX_L, seed=2)
        wall = time.time() - t0
        chains = _hmc_chains_first_to_samples(raw, REF_HMC_N, res)
        rhat = np.asarray(tfp.mcmc.potential_scale_reduction(
            jnp.asarray(chains), independent_chain_ndims=1,
            split_chains=True))
        ess = np.asarray(tfp.mcmc.effective_sample_size(
            jnp.asarray(chains), cross_chain_dims=1))
        return chains, dict(
            n_hmc=REF_HMC_N, num_burnin_steps=burn, num_results=res, seed=2,
            init_l=REF_HMC_INIT_L, max_leapfrog_steps=REF_HMC_MAX_L,
            wall_s=wall, rhat_worst=float(np.max(rhat)),
            ess_min=float(np.min(ess)), rhat=rhat.tolist(), ess=ess.tolist(),
            grad_evals_lo=REF_HMC_N * (burn + res) * REF_HMC_INIT_L,
            grad_evals_hi=REF_HMC_N * (burn + res) * REF_HMC_MAX_L)

    chains, hlog = _hmc(REF_HMC_BURN, REF_HMC_RES)
    accepted = hlog["rhat_worst"] < RHAT_MAX and hlog["ess_min"] >= ESS_MIN
    log["hmc"] = hlog
    if not accepted:
        chains2, hlog2 = _hmc(REF_ESCALATED_BURN, REF_ESCALATED_RES)
        log["hmc_escalated"] = hlog2
        if hlog2["rhat_worst"] < RHAT_MAX and hlog2["ess_min"] >= ESS_MIN:
            chains, accepted = chains2, True
            log["accepted_stage"] = "hmc_escalated"
        else:
            log["accepted_stage"] = "NONE (blocked-by-reference)"
    else:
        log["accepted_stage"] = "hmc"

    log["reference_accepted"] = bool(accepted)
    log["oom_fallbacks"] = oom_fallbacks
    acc = log.get("accepted_stage")
    hacc = log["hmc_escalated"] if acc == "hmc_escalated" else log["hmc"]
    log["grad_evals_accepted"] = dict(
        map=log["map"]["grad_evals"], svi=log["svi"]["grad_evals"],
        hmc_lo=hacc["grad_evals_lo"], hmc_hi=hacc["grad_evals_hi"],
        total_lo=log["map"]["grad_evals"] + log["svi"]["grad_evals"]
                 + hacc["grad_evals_lo"],
        total_hi=log["map"]["grad_evals"] + log["svi"]["grad_evals"]
                 + hacc["grad_evals_hi"])
    log["wall_s"] = (log["map"]["wall_s"] + log["svi"]["wall_s"]
                     + hacc["wall_s"])
    log["wall_s_incl_failed"] = (log["map"]["wall_s"] + log["svi"]["wall_s"]
                                 + log["hmc"]["wall_s"]
                                 + log.get("hmc_escalated", {}).get("wall_s", 0.0))
    arrays = dict(chains=chains, map_best=np.asarray(map_best),
                  qz_mean=np.asarray(qz.mean()),
                  qz_cov=np.asarray(qz.covariance()))
    return log, arrays


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sys", type=int, required=True, choices=range(4))
    ap.add_argument("--arm", required=True, choices=("ref", "smc"))
    ap.add_argument("--devtag", required=True,
                    help="device label for artifacts, e.g. a16-0 / l4-8")
    args = ap.parse_args()

    guards.require_vendor_ref()
    guards.require_jax_pin()
    guards.require_x64()
    guards.require_gpu()
    guards.require_single_device()

    import jax
    from cgl2 import zoo

    name = f"hs2_sys{args.sys}"
    tag = f"{args.arm}_sys{args.sys}_{args.devtag}"
    print(f"[b3] {tag}: building {name} on {jax.devices()}", flush=True)
    tgt = zoo.build(name)

    row = dict(script="22_run_b3.py", target=name, sys=args.sys,
               devtag=args.devtag,
               date=time.strftime("%Y-%m-%d %H:%M:%S"),
               jax_version=jax.__version__,
               backend=jax.default_backend(),
               device=str(jax.devices()[0]),
               x64=bool(jax.config.jax_enable_x64),
               z_dim=int(tgt.z_dim),
               z_true=tgt.meta.get("z_true"),
               provenance=tgt.provenance)
    t0 = time.time()
    try:
        if args.arm == "smc":
            res, arrays = run_smc_arm(tgt, args.sys)
        else:
            res, arrays = run_ref_arm(tgt, args.sys)
        row.update(res)
        row["status"] = "OK"
    except Exception as e:  # noqa: BLE001 -- recorded, never faked
        row.update(status="ERROR", error=f"{type(e).__name__}: {e}",
                   trace=traceback.format_exc(limit=12))
        arrays = None
    row["wall_total_s"] = time.time() - t0

    paths.DATA_DIR.mkdir(exist_ok=True)
    if arrays is not None:
        np.savez_compressed(paths.DATA_DIR / f"b3_{tag}.npz", **arrays)
    with open(paths.DATA_DIR / f"b3_run_{tag}.json", "w") as fh:
        json.dump(_jsonable(row), fh, indent=1)
    print(f"[b3] {tag}: status={row['status']} wall={row['wall_total_s']:.0f}s "
          f"-> data/b3_run_{tag}.json", flush=True)


if __name__ == "__main__":
    main()
