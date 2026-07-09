#!/usr/bin/env python
"""23_pt_reference_v3b.py -- the T3 (foundry_v3b74) parallel-tempering REFERENCE.

The linchpin of P2c-T3: the stored v3b chains have ZERO inter-basin migrations,
so their basin fractions are start artifacts, NOT posterior mass. This script
runs a LONG TFP ReplicaExchangeMC over the batched PHMC kernel with
LIKELIHOOD-ONLY tempering (per-replica logp = log_prior + beta * log_like, the
prior is never tempered) over a geometric beta-ladder (beta 1 -> beta_min), and
measures the true low/steep mode weights from the equilibrated beta=1 (cold)
chain. Because the basins are Delta chi^2_v ~ 0.08 close, beta_min defaults to
0.05 (tune so adjacent-swap acceptance in [0.2, 0.6]).

Round-trip stop signal: beta=1 round trips (coldest -> hottest -> coldest
excursions of a tracked walker) are reconstructed per chain from the accepted
adjacent-swap trace (cgl.metrics.pt_walker_temps_from_adjacent +
count_pt_round_trips). The run length (--max-steps) is sized so the total
post-burn round trips reach --target-round-trips (>=100 for production); the
output flags whether the target was met.

Output (data/results/pt_reference/v3b_s<seed>.json + .npz): the beta-ladder,
per-pair swap acceptance, per-chain / total round trips, the cold basin
occupancy (the reference mode weights) with a chain-bootstrap sigma, and a
thinned copy of the cold samples. Acceptance of the reference (per the P2c
brief) is decided at analysis time: PT weights vs the 24_basin_evidence_v3b SMC
weights must agree within 2 sigma.

Single-target-per-process: env (x64 OFF for f32 T3, XLA flags) is set from the
registry dtype BEFORE the first jax import. Priority-fusion is disabled by
default per the Perlmutter A100 policy (harmless on f32; CAMPAIGN.md P2b).

Run (phoenix smoke, GPU 4):
  /raid/benson/.venvs/cgl/bin/python 23_pt_reference_v3b.py --gpu 4 \
      --n-replicas 4 --beta-min 0.05 --n-chains 4 --n-burn 20 --max-steps 50 \
      --smoke
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPRO = Path(__file__).resolve().parent
TARGET = "foundry_v3b74"
GAMMA_THRESHOLD = 1.8   # low basin < 1.31, steep basin > 2.32 (mid-gap)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu", default=None, help="CUDA index (default registry)")
    ap.add_argument("--n-replicas", type=int, default=10)
    ap.add_argument("--beta-min", type=float, default=0.05,
                    help="hottest inverse temperature (geometric ladder)")
    ap.add_argument("--n-chains", type=int, default=8,
                    help="independent PT systems (each carries n_replicas)")
    ap.add_argument("--n-burn", type=int, default=500)
    ap.add_argument("--max-steps", type=int, default=8000,
                    help="post-burn step cap (sized to reach target round trips)")
    ap.add_argument("--num-leapfrog", type=int, default=8)
    ap.add_argument("--eps0", type=float, default=0.15)
    ap.add_argument("--eps-scale-power", type=float, default=-0.5)
    ap.add_argument("--eps-max", type=float, default=2.0)
    ap.add_argument("--mass", default="svi_diag",
                    choices=["svi_diag", "svi"],
                    help="inner-kernel momentum precond (svi_diag = diagraw-"
                         "style diagonal precision; svi = full inv(SVI cov))")
    ap.add_argument("--start-jitter", type=float, default=0.3,
                    help="SVI-chol-scaled jitter around the multi-basin MAPs")
    ap.add_argument("--target-round-trips", type=int, default=100)
    ap.add_argument("--thin", type=int, default=1, help="thin cold samples in T")
    ap.add_argument("--n-boot", type=int, default=2000,
                    help="chain-bootstrap resamples for the weight sigma")
    ap.add_argument("--out", default=None)
    ap.add_argument("--smoke", action="store_true",
                    help="tiny run: just prove swaps happen + round-trip "
                         "counter works + finite output")
    return ap.parse_args()


def main():
    args = parse_args()

    # ---- env BEFORE jax (f32 target; single-target-per-process) -------------
    from cgl.zoo import get_target_info
    from cgl.zoo.runtime import setup_process_env

    info = get_target_info(TARGET)
    assert info.dtype == "float32", info.dtype
    import os as _os
    _flags = _os.environ.get("XLA_FLAGS", "")
    if "--xla_disable_hlo_passes" not in _flags:   # A100 priority-fusion policy
        _os.environ["XLA_FLAGS"] = (
            _flags + " --xla_disable_hlo_passes=priority-fusion").strip()
    setup_process_env(info.dtype, args.gpu)

    import jax
    import jax.numpy as jnp
    import numpy as np
    import tensorflow_probability.substrates.jax as tfp

    from cgl import guards, metrics
    from cgl.samplers import common
    from cgl.samplers.remc_pt import geometric_ladder, make_replica_logp
    from cgl.zoo import get_target
    from cgl.zoo.api import np_dtype
    from cgl.zoo.runtime import setup_jax_cache

    tfd = tfp.distributions
    tfe = tfp.experimental
    setup_jax_cache(REPRO)
    guards.require_single_device()
    print(f"devices={jax.devices()} x64={jax.config.jax_enable_x64}", flush=True)

    fdtype = np_dtype(info.dtype)
    R = int(args.n_replicas)
    C = int(args.n_chains)
    L = int(args.num_leapfrog)
    n_burn, max_steps = int(args.n_burn), int(args.max_steps)

    # ---- build target -------------------------------------------------------
    t0 = time.time()
    target = get_target(TARGET)
    dim = target.dim
    print(f"built {target.name} dim={dim} in {time.time()-t0:.0f}s", flush=True)

    betas = geometric_ladder(R, float(args.beta_min))
    print(f"beta ladder ({R}): {np.array2string(betas, precision=4)}", flush=True)

    key = jax.random.PRNGKey(int(args.seed))
    k_start, k_chain = jax.random.split(key)

    # ---- SVI momentum preconditioner + multi-basin starts -------------------
    ledger = metrics.BudgetLedger()
    timing = {}
    ginit = common.gaussian_init(target, int(args.seed), ledger, timing)
    cov_reg = np.asarray(ginit.cov_reg, dtype=np.float64)
    if args.mass == "svi":
        prec = np.linalg.solve(cov_reg, np.eye(dim))
        prec = 0.5 * (prec + prec.T)
        momentum = tfd.MultivariateNormalFullCovariance(
            loc=jnp.zeros(dim, dtype=fdtype),
            covariance_matrix=jnp.asarray(prec, dtype=fdtype))
    else:   # svi_diag: diagonal precision (diagraw-style, f32-safe in 74-dim)
        scale = 1.0 / np.sqrt(np.maximum(np.diag(cov_reg), 1e-30))
        momentum = tfd.MultivariateNormalDiag(
            loc=jnp.zeros(dim, dtype=fdtype),
            scale_diag=jnp.asarray(scale, dtype=fdtype))

    ms = target.init.multi_starts
    if ms is not None and len(ms) >= 1:
        reps = int(np.ceil(C / len(ms)))
        centers = np.tile(np.asarray(ms, dtype=np.float64), (reps, 1))[:C]
        start_note = f"multi-basin MAP starts x{len(ms)} cycled over {C} chains"
    else:
        centers = np.tile(np.asarray(ginit.loc, dtype=np.float64), (C, 1))
        start_note = "SVI-loc starts (no multi_starts on target)"
    eps_j = jax.random.normal(k_start, (C, dim), dtype=fdtype)
    start = (jnp.asarray(centers, dtype=fdtype)
             + float(args.start_jitter)
             * eps_j @ jnp.asarray(ginit.chol, dtype=fdtype).T)

    # ---- tempered/untempered closures + REMC kernel -------------------------
    tempered, untempered, _ = make_replica_logp(target, betas)
    common.warmup_batch(target, C, R * C)      # zoo batch-size warmup contract

    eps = np.minimum(float(args.eps0) * betas ** float(args.eps_scale_power),
                     float(args.eps_max))
    eps_arr = jnp.asarray(eps, dtype=fdtype).reshape(R, 1, 1)
    print(f"step-size ladder: {np.array2string(eps, precision=4)}", flush=True)

    def make_kernel_fn(target_log_prob_fn):
        return tfe.mcmc.PreconditionedHamiltonianMonteCarlo(
            target_log_prob_fn=target_log_prob_fn,
            momentum_distribution=momentum,
            step_size=eps_arr, num_leapfrog_steps=L)

    remc = tfp.mcmc.ReplicaExchangeMC(
        target_log_prob_fn=None,
        inverse_temperatures=jnp.asarray(betas, dtype=fdtype),
        make_kernel_fn=make_kernel_fn,
        tempered_log_prob_fn=tempered,
        untempered_log_prob_fn=untempered)

    def trace_fn(_, pkr):
        return {
            "is_swap_accepted_adjacent": pkr.is_swap_accepted_adjacent,
            "is_swap_proposed_adjacent": pkr.is_swap_proposed_adjacent,
        }

    n_steps = n_burn + max_steps

    @jax.jit
    def run_chain(seed_key, start):
        return tfp.mcmc.sample_chain(
            num_results=n_steps, num_burnin_steps=0,
            current_state=start, kernel=remc, trace_fn=trace_fn, seed=seed_key)

    t0 = time.time()
    samples_all, trace = run_chain(k_chain, start)
    samples_all = jax.block_until_ready(samples_all)
    remc_s = time.time() - t0
    per_step_ms = 1e3 * remc_s / max(n_steps, 1)
    print(f"REMC {n_steps} steps in {remc_s:.1f}s ({per_step_ms:.1f} ms/step) "
          f"[{R} replicas x {C} chains x L={L}]", flush=True)

    # ---- swap health --------------------------------------------------------
    acc = np.asarray(trace["is_swap_accepted_adjacent"])   # (T, R-1, C)
    prop = np.asarray(trace["is_swap_proposed_adjacent"])
    assert acc.shape[0] == n_steps and acc.shape[1] == R - 1, acc.shape
    with np.errstate(invalid="ignore"):
        pair_rates = (acc.astype(np.float64).sum(axis=(0, 2))
                      / np.maximum(prop.astype(np.float64).sum(axis=(0, 2)), 1.0))
    swap_mean = float(np.mean(pair_rates))
    swaps_healthy = bool(np.all((pair_rates >= 0.2) & (pair_rates <= 0.6)))
    print(f"swap acceptance per pair: "
          f"{np.array2string(pair_rates, precision=3)}  mean={swap_mean:.3f}  "
          f"in[0.2,0.6]={swaps_healthy}", flush=True)

    # ---- beta=1 round trips (post-burn), per chain from accepted swaps -------
    acc_pb = acc[n_burn:]                                   # (Tk, R-1, C)
    rt_per_chain = []
    reached_hot_chains = 0
    for c in range(C):
        temps_c = metrics.pt_walker_temps_from_adjacent(acc_pb[:, :, c])
        rtc = metrics.count_pt_round_trips(temps_c)
        rt_per_chain.append(rtc["total_round_trips"])
        reached_hot_chains += int(rtc["n_walkers_reaching_hot"] > 0)
    total_round_trips = int(np.sum(rt_per_chain))
    rt_rate = total_round_trips / max(max_steps, 1)
    met_target = total_round_trips >= int(args.target_round_trips)
    print(f"round trips (post-burn): total={total_round_trips} "
          f"per_chain={rt_per_chain}  rate={rt_rate:.3e}/step  "
          f"target={args.target_round_trips} met={met_target}", flush=True)

    # ---- cold basin occupancy = the reference mode weights ------------------
    cold = np.asarray(samples_all)[n_burn:]                 # (Tk, C, dim)
    Tk = cold.shape[0]
    # inner-HMC exploration check: RMS step-to-step move of the cold chain. If
    # ~0 the leapfrog steps are diverging/rejected (states frozen) -> "round
    # trips" are meaningless swap shuffles of near-identical states, and the
    # basins can never mix. Compare against the per-dim SVI scale.
    if Tk >= 2:
        step_rms = float(np.sqrt(np.mean(np.diff(cold, axis=0) ** 2)))
    else:
        step_rms = float("nan")
    svi_scale = float(np.sqrt(np.mean(np.diag(cov_reg))))
    print(f"cold-chain exploration: step_rms={step_rms:.4g} "
          f"(SVI per-dim scale ~{svi_scale:.4g}; ratio "
          f"{step_rms / max(svi_scale, 1e-30):.3f}) -- ~0 => inner HMC frozen",
          flush=True)
    gamma = np.asarray(target.to_physical(
        cold.reshape(-1, dim))["gamma"], dtype=np.float64).reshape(Tk, C)
    low = gamma < GAMMA_THRESHOLD                          # (Tk, C) bool
    w_low = float(low.mean())
    w_low_per_chain = low.mean(axis=0)
    # chain bootstrap sigma (resample the C cold chains with replacement)
    rng = np.random.default_rng(int(args.seed) + 23_20260709)
    boot = np.array([low[:, rng.integers(0, C, C)].mean()
                     for _ in range(int(args.n_boot))])
    w_low_sigma = float(boot.std())
    print(f"cold basin occupancy: w_low={w_low:.4f} +/- {w_low_sigma:.4f} "
          f"(w_steep={1-w_low:.4f})  per-chain w_low="
          f"{np.array2string(w_low_per_chain, precision=3)}", flush=True)

    # ---- basin-MIXING gate (the reference's real validity check) ------------
    # Temperature round trips only certify the LADDER mixes; the WEIGHT is
    # trustworthy only if the cold (beta=1) state actually SWITCHES basins --
    # i.e. the hot replica is prior-dominated enough to cross the gamma barrier
    # (the stored chains had ZERO such crossings; that is the whole problem).
    col_low = low.astype(int)
    beta1_basin_transitions = int(
        np.count_nonzero(col_low[1:] != col_low[:-1]))
    per_chain_frac = low.mean(axis=0)
    n_basin_mixing_chains = int(np.sum((per_chain_frac > 0.0)
                                       & (per_chain_frac < 1.0)))
    basins_mixing = n_basin_mixing_chains >= 1
    print(f"basin MIXING at beta=1: {n_basin_mixing_chains}/{C} chains visit "
          f"BOTH basins, {beta1_basin_transitions} cold basin transitions -> "
          f"weights_trustworthy={basins_mixing and met_target}", flush=True)

    # ---- save ---------------------------------------------------------------
    g = n_steps * R * C * L
    ledger.add("remc", n_grad=g, n_logp=n_steps * R * C,
               note=f"{n_steps} steps x {R} replicas x {C} chains x L={L}")

    out_json = Path(args.out) if args.out else \
        REPRO / "data" / "results" / "pt_reference" / f"v3b_s{args.seed}.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_npz = out_json.with_suffix(".npz")

    cold_store = cold[:: max(int(args.thin), 1)].astype(np.float32)
    np.savez_compressed(
        out_npz, cold_samples=cold_store, gamma=gamma.astype(np.float32),
        betas=betas, pair_swap_acceptance=pair_rates,
        w_low_per_chain=w_low_per_chain)

    payload = dict(
        script="23_pt_reference_v3b.py", target=TARGET, seed=int(args.seed),
        smoke=bool(args.smoke),
        config=dict(n_replicas=R, beta_min=float(args.beta_min), n_chains=C,
                    n_burn=n_burn, max_steps=max_steps, num_leapfrog=L,
                    eps0=float(args.eps0),
                    eps_scale_power=float(args.eps_scale_power),
                    eps_max=float(args.eps_max), mass=args.mass,
                    start_jitter=float(args.start_jitter), thin=int(args.thin)),
        betas=betas.tolist(), step_size_ladder=eps.tolist(),
        start_note=start_note,
        swap=dict(pair_acceptance=pair_rates.tolist(), mean=swap_mean,
                  healthy_0p2_0p6=swaps_healthy),
        round_trips=dict(total=total_round_trips, per_chain=rt_per_chain,
                         rate_per_step=rt_rate,
                         n_chains_reaching_hot=int(reached_hot_chains),
                         target=int(args.target_round_trips),
                         target_met=bool(met_target)),
        mode_weights=dict(
            w_low=w_low, w_steep=float(1.0 - w_low), sigma_w_low=w_low_sigma,
            w_low_per_chain=w_low_per_chain.tolist(),
            gamma_threshold=GAMMA_THRESHOLD, estimator="pt_cold_occupancy",
            n_cold_samples=int(Tk * C)),
        basin_mixing=dict(
            n_basin_mixing_chains=n_basin_mixing_chains, n_chains=C,
            beta1_basin_transitions=beta1_basin_transitions,
            basins_mix=bool(basins_mixing),
            weights_trustworthy=bool(basins_mixing and met_target)),
        inner_hmc=dict(cold_step_rms=step_rms, svi_per_dim_scale=svi_scale,
                       step_rms_over_scale=float(step_rms / max(svi_scale,
                                                                1e-30))),
        timing=dict(remc_s=remc_s, per_step_ms=per_step_ms,
                    total_s=remc_s + sum(v for v in timing.values())),
        budget=ledger.as_dict(),
        provenance="likelihood-only tempered ReplicaExchangeMC over batched "
                   "PHMC (cgl.samplers.remc_pt kernel); cold occupancy = "
                   "reference mode weights. Cross-check vs 24_basin_evidence "
                   "SMC weights (accept iff agree within 2 sigma).",
        npz=out_npz.name)
    out_json.write_text(json.dumps(payload, indent=2, default=str))
    print(f"wrote {out_json}\n      {out_npz}", flush=True)

    if not swaps_healthy:
        why = ("too HIGH -> ladder too fine / beta_min too COLD (go hotter, "
               "fewer replicas)" if bool(np.any(pair_rates > 0.6))
               else "too LOW -> add replicas / raise beta_min")
        print(f"WARNING: swap acceptance outside [0.2,0.6] on some pairs "
              f"({why}) before the production run.", flush=True)
    if not basins_mixing:
        print("WARNING: cold chain does NOT switch basins (weights reflect the "
              "starts, NOT posterior mass -- same failure as the stored "
              "chains). Lower beta_min until the hot replica is prior-dominated "
              "and cold basin transitions appear, THEN trust the weights.",
              flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
