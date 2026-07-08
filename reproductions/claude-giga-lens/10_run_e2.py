"""10_run_e2.py — E2 real-data cross-scale unification (P1c pillar-1 headline).

Modes:
  smoke  : build all three E2 targets, map both basin starts each, ASSERT logp
           finiteness + rendered-model sanity at every start, run N HMC steps
           per (target, basin) to measure grad/step wall-time. Writes a smoke
           report; burns no production budget. This is the Perlmutter debug-QOS
           pre-flight (CAMPAIGN.md P1c execution plan).
  prod   : full production fit of ONE product tag: map both basin starts,
           L-BFGS MAP polish under the correlated likelihood, Laplace evidence
           per basin, two-stage re-preconditioned PHMC per basin, gamma
           posteriors + basin masses. Writes data/results/e2_<tag>.npz.

Usage:
  python 10_run_e2.py --mode smoke --hmc-steps 50 --out data/results/e2_smoke.json
  python 10_run_e2.py --mode prod --tag v3b --chains 24 --keep 2000 \
        --out data/results/e2_v3b.npz

ALWAYS run under GIGALENS_X64=1 on a GPU (never CPU: the lensing likelihood is
216x slower on CPU — guard-enforced).
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np


def _finite(x):
    return bool(np.all(np.isfinite(np.asarray(x))))


def render_sanity(target, z, label):
    """Diagnostic (diagonal) chi2_pp + whitened logL_data at z (numpy, eager)."""
    import jax.numpy as jnp
    m = target.model
    internals = m.marg_internals(z)
    mimg = np.asarray(m.model_image(jnp.asarray(z, dtype=jnp.float64)))
    Y = np.asarray(m.Y)
    err = np.asarray(m.masked_err_map)
    keep = np.asarray(m.keep_mask)
    chi2_pp = float(np.mean(((Y - mimg)[keep] / err[keep]) ** 2))
    rw = internals["Rw"]
    nkw = int(np.sum(np.asarray(target.keep_w)))
    return dict(
        label=label, chi2_pp_diag=chi2_pp,
        logL_data=internals["logL_data"], log_prior=internals["log_prior"],
        logpost=internals["logpost"], logdetA=internals["logdetA"],
        u2_over_nkw=float(np.sum(rw ** 2) / max(nkw, 1)),
        n_keep_w=nkw, model_finite=_finite(mimg),
        model_max=float(np.max(np.abs(mimg))))


def smoke(args):
    from cgl import e2
    import jax, jax.numpy as jnp

    tags = args.tags.split(",") if args.tags else ["v3b", "v2d", "v3"]
    report = dict(mode="smoke", hmc_steps=args.hmc_steps,
                  host=os.uname().nodename, gigalens_x64=os.environ.get(
                      "GIGALENS_X64"), targets={})
    all_ok = True
    for tag in tags:
        print(f"\n===== SMOKE {tag} ({e2.PRODUCTS[tag]['label']}) =====",
              flush=True)
        t_build = time.time()
        target = e2.build_target(tag)
        t_build = time.time() - t_build
        print(f"  target built in {t_build:.1f}s; whitener={target.whiten_meta}",
              flush=True)
        tr = dict(build_s=t_build, whiten=target.whiten_meta, basins={})
        starts = {}
        for basin in ("steep", "low"):
            z, rep = e2.make_start(tag, basin, target)
            lp = float(target.model.target_log_prob_fn(
                jnp.asarray(z, dtype=jnp.float64)))
            san = render_sanity(target, z, f"{tag}/{basin}")
            ok = _finite(z) and np.isfinite(lp) and san["model_finite"]
            all_ok &= ok
            starts[basin] = z
            print(f"  [{basin:5s}] {rep['artifact']:20s} "
                  f"gamma={rep['gamma']:.4f} (tgt {rep['gamma_target']:.4f} "
                  f"drt {rep['roundtrip_dgamma']:.1e}) thetaE={rep['theta_E']:.4f} "
                  f"| logp={lp:.1f} chi2pp={san['chi2_pp_diag']:.3f} "
                  f"u2/nkw={san['u2_over_nkw']:.3f} {'OK' if ok else 'BAD'}",
                  flush=True)
            tr["basins"][basin] = dict(start=rep, logp=lp, sanity=san, ok=bool(ok))

        # timing: compile + N HMC steps, both basins in one 2*? batch? keep per-basin
        chains = args.smoke_chains
        st = np.concatenate([np.repeat(starts["steep"][None], chains // 2, 0),
                             np.repeat(starts["low"][None], chains - chains // 2, 0)],
                            axis=0)
        st = st + 1e-3 * np.random.default_rng(0).standard_normal(st.shape)
        t_hmc, step_ms, grad_ms = _time_hmc(target, st, args.hmc_steps)
        tr["hmc_timing"] = dict(chains=chains, n_steps=args.hmc_steps,
                                total_s=t_hmc, per_step_ms=step_ms,
                                per_grad_ms=grad_ms)
        print(f"  HMC {chains}ch x {args.hmc_steps} steps: {t_hmc:.1f}s "
              f"({step_ms:.1f} ms/step, ~{grad_ms:.2f} ms/grad-est)", flush=True)
        report["targets"][tag] = tr

    report["all_starts_ok"] = bool(all_ok)
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(outp, "w"), indent=1, default=float)
    print(f"\n[smoke] all_starts_ok={all_ok}; wrote {outp}", flush=True)
    return report


def _time_hmc(target, start, n_steps):
    """Compile + run n_steps of batched PHMC (fixed leapfrog=8); return
    (total_s, ms/step, ms/grad-estimate)."""
    import jax, jax.numpy as jnp
    import tensorflow_probability.substrates.jax as tfp
    tfe = tfp.experimental
    ndim = target.model.ndim
    chains = start.shape[0]
    leap = 8
    momentum = tfp.distributions.MultivariateNormalDiag(
        loc=jnp.zeros(ndim), scale_diag=jnp.ones(ndim))
    kernel = tfe.mcmc.PreconditionedHamiltonianMonteCarlo(
        target_log_prob_fn=target.batched_lp, momentum_distribution=momentum,
        step_size=np.float64(0.05), num_leapfrog_steps=leap)

    @jax.jit
    def run(s0, key):
        return tfp.mcmc.sample_chain(
            num_results=n_steps, num_burnin_steps=0, current_state=s0,
            kernel=kernel, trace_fn=None, seed=key)
    s0 = jnp.asarray(start, dtype=jnp.float64)
    # compile
    d = run(s0, jax.random.PRNGKey(0)); d.block_until_ready()
    t0 = time.time()
    d = run(s0, jax.random.PRNGKey(1)); d.block_until_ready()
    total = time.time() - t0
    step_ms = 1e3 * total / n_steps
    grad_ms = step_ms / (leap + 1)   # ~leap grads + 1 per step
    return total, step_ms, grad_ms


def prod(args):
    from cgl import e2
    import jax.numpy as jnp

    tag = args.tag
    print(f"===== PROD {tag} ({e2.PRODUCTS[tag]['label']}) =====", flush=True)
    target = e2.build_target(tag, whitener_file=(args.whitener or None))
    gi = e2.gamma_index(target.model)
    ti = e2.theta_e_index(target.model)
    labels = list(target.model.index_labels)
    out = dict(tag=tag, label=e2.PRODUCTS[tag]["label"], whiten=target.whiten_meta,
               ndim=int(target.model.ndim), gamma_index=gi, basins={},
               config=vars(args))
    basin_draws = {}
    basins = [b.strip() for b in args.basins.split(",") if b.strip()]
    for basin in basins:
        print(f"\n--- basin {basin} (seed {args.seed}) ---", flush=True)
        z0, rep = e2.make_start(tag, basin, target)
        print(f"  start {rep['artifact']} gamma={rep['gamma']:.4f} "
              f"(over={rep['mass_override']})", flush=True)
        mp = e2.map_polish(target, z0, rounds=args.map_rounds,
                           iters=args.map_iters)
        xm = target.model.bij.forward(list(jnp.asarray(mp["z_map"])[:, None]))
        gamma_map = float(np.asarray(xm[0][0]["gamma"]).reshape(-1)[0])
        gamma_seed = rep["gamma"]
        # basin-crossing guard: keep the MAP in its seed basin. If L-BFGS drifted
        # across the steep/low split, seed HMC at the (in-basin) mapped start and
        # flag it (the marg landscape has saddles: weak-likelihood products let
        # L-BFGS wander toward the gamma prior mean 2.0).
        crossed = ((gamma_seed < e2.GAMMA_STEEP_LOW_SPLIT)
                   != (gamma_map < e2.GAMMA_STEEP_LOW_SPLIT))
        z_seed_hmc = z0 if crossed else mp["z_map"]
        print(f"  MAP: logp {mp['logp0']:.1f} -> {mp['logp_map']:.1f} "
              f"gamma_map={gamma_map:.4f} ({mp['wall_s']:.0f}s) "
              f"{'CROSSED->reseed@start' if crossed else 'in-basin'}", flush=True)
        lap = e2.laplace_evidence(target, z_seed_hmc)
        print(f"  Laplace: logZ={lap['log_evidence']:.2f} logdetH={lap['logdet_H']:.2f} "
              f"min_eig={lap['min_eig']:.2e} n_floored={lap['n_floored']}", flush=True)
        st = e2.run_staged(
            target, z_seed_hmc, lap["cov"], chains=args.chains, seed=args.seed,
            stage1_burn=args.stage1_burn, stage1_keep=args.stage1_keep,
            burn=args.burn, keep=args.keep, step_size=args.step_size,
            num_leapfrog=args.num_leapfrog, jitter=args.jitter)
        draws = st["draws"]                       # (T, C, ndim)
        basin_draws[basin] = draws
        # robust basin MAP: highest-logpost draw over a small even subsample
        # (memory-safe: batched_lp on the full pooled set OOMs). n_probe chains
        # per eval fit comfortably on a 40 GB A100 for all three products.
        flat = draws.reshape(-1, draws.shape[-1])
        n_probe = 96
        step = max(1, flat.shape[0] // n_probe)
        sub = flat[::step][:n_probe]
        lp_flat = np.asarray(target.batched_lp(jnp.asarray(sub, dtype=jnp.float64)))
        best_sub = int(np.argmax(lp_flat))
        z_best = sub[best_sub]
        logp_best = float(lp_flat[best_sub])
        gd = draws[:, :, gi].reshape(-1)
        td = draws[:, :, ti].reshape(-1)
        # to physical gamma/thetaE via bijector
        phys = target.model.to_physical_mass(draws.reshape(-1, draws.shape[-1]))
        gph = np.asarray(phys["gamma"]); tph = np.asarray(phys["theta_E"])
        gamma_best = float(np.asarray(target.model.to_physical_mass(
            z_best[None, :])["gamma"]).reshape(-1)[0])
        summ = dict(
            start=rep, logp_map=mp["logp_map"], gamma_map=gamma_map,
            map_crossed_basin=bool(crossed), reseeded_at_start=bool(crossed),
            logp_best=logp_best, gamma_best=gamma_best,
            map_round_lps=mp["round_lps"], map_wall_s=mp["wall_s"],
            laplace=dict(log_evidence=lap["log_evidence"], logdet_H=lap["logdet_H"],
                         min_eig=lap["min_eig"], n_floored=lap["n_floored"]),
            gamma_median=float(np.median(gph)),
            gamma_q16=float(np.quantile(gph, 0.16)),
            gamma_q84=float(np.quantile(gph, 0.84)),
            gamma_std=float(np.std(gph)),
            thetaE_median=float(np.median(tph)),
            thetaE_std=float(np.std(tph)),
            rhat_max=st["stage2_rhat_max"], ess_min=st["stage2_ess_min"],
            rhat_gamma=float(st["rhat"][gi]), ess_gamma=float(st["ess"][gi]),
            stage1_rhat_max=st["stage1_rhat_max"], stage1_ess_min=st["stage1_ess_min"],
            step_final=st["step_final"], n_floored=st["n_floored"],
            wall_s=st["wall_s"], n_draws=int(gph.size),
            frac_gamma_gt_split=float(np.mean(gph > e2.GAMMA_STEEP_LOW_SPLIT)))
        print(f"  posterior gamma={summ['gamma_median']:.4f} "
              f"[{summ['gamma_q16']:.4f},{summ['gamma_q84']:.4f}] "
              f"std={summ['gamma_std']:.4f} | Rhat_max={summ['rhat_max']:.4f} "
              f"ESS_min={summ['ess_min']:.0f} ESS_gamma={summ['ess_gamma']:.0f} "
              f"({st['wall_s']:.0f}s)", flush=True)
        out["basins"][basin] = summ

    # H1 numbers (only when BOTH basins were sampled). Pre-registered test:
    # corrected Delta logL(steep-low) <= 0 at basin MAPs. Robust basin MAP =
    # highest-logpost kept draw per basin (the L-BFGS MAP is saddle-prone on the
    # weak-likelihood products). Also report the Laplace log-evidence difference
    # (basin mass proxy; caveated when the Hessian is non-PD) and the fraction of
    # low-basin posterior mass in the steep region (occupancy under separated
    # basins reflects seeds, so reported, not used as the mass).
    if "low" not in out["basins"] or "steep" not in out["basins"]:
        out["H1"] = dict(note=f"single-basin run (basins={basins}); H1 not "
                         "computed (needs both). This product is a per-basin "
                         "replicate (e.g. cross-seed robustness of gamma).")
        outp = Path(args.out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        np.savez(outp.with_suffix(".npz"),
                 **{f"{b}_draws": basin_draws[b].astype(np.float32) for b in basins},
                 labels=np.array(labels), summary=json.dumps(out, default=float))
        json.dump(out, open(outp.with_suffix(".json"), "w"), indent=1, default=float)
        print(f"[prod] wrote {outp.with_suffix('.npz')} + .json (single-basin)",
              flush=True)
        return out
    lo, hi = out["basins"]["low"], out["basins"]["steep"]
    dlogpost_best = hi["logp_best"] - lo["logp_best"]
    dlogZ = hi["laplace"]["log_evidence"] - lo["laplace"]["log_evidence"]
    mass_steep_laplace = float(1.0 / (1.0 + np.exp(-dlogZ)))
    out["H1"] = dict(
        dlogpost_best_steep_minus_low=float(dlogpost_best),
        dlogpost_map_steep_minus_low=float(hi["logp_map"] - lo["logp_map"]),
        dlogZ_laplace_steep_minus_low=float(dlogZ),
        laplace_mass_steep=mass_steep_laplace,
        steep_map_crossed=bool(hi["map_crossed_basin"]),
        low_map_crossed=bool(lo["map_crossed_basin"]),
        gamma_best_low=lo["gamma_best"], gamma_best_steep=hi["gamma_best"],
        verdict_dlogL_le_0=bool(dlogpost_best <= 0.0))
    print(f"\n[H1] dlogpost_best(steep-low)={dlogpost_best:.2f} "
          f"(<=0 -> {dlogpost_best<=0}); dlogZ_laplace={dlogZ:.2f}; "
          f"Laplace mass_steep={mass_steep_laplace:.4f}", flush=True)

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    np.savez(outp.with_suffix(".npz"),
             low_draws=basin_draws["low"].astype(np.float32),
             steep_draws=basin_draws["steep"].astype(np.float32),
             labels=np.array(labels), summary=json.dumps(out, default=float))
    json.dump(out, open(outp.with_suffix(".json"), "w"), indent=1, default=float)
    print(f"[prod] wrote {outp.with_suffix('.npz')} + .json", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke", "prod"], required=True)
    ap.add_argument("--tag", choices=["v3", "v3b", "v2d"], default="v3b")
    ap.add_argument("--tags", type=str, default="")
    ap.add_argument("--whitener", type=str, default="",
                    help="override whitener npz (e.g. whitener_v2d.npz strict)")
    ap.add_argument("--out", type=str, default="data/results/e2_out.json")
    # smoke
    ap.add_argument("--hmc-steps", type=int, default=50)
    ap.add_argument("--smoke-chains", type=int, default=8)
    # prod
    ap.add_argument("--chains", type=int, default=24)
    ap.add_argument("--stage1-burn", type=int, default=500)
    ap.add_argument("--stage1-keep", type=int, default=500)
    ap.add_argument("--burn", type=int, default=500)
    ap.add_argument("--keep", type=int, default=2000)
    ap.add_argument("--step-size", type=float, default=0.1)
    ap.add_argument("--num-leapfrog", type=int, default=16)  # foundry-i marg recipe
    ap.add_argument("--jitter", type=float, default=0.5)
    ap.add_argument("--map-rounds", type=int, default=4)
    ap.add_argument("--map-iters", type=int, default=200)
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--basins", type=str, default="low,steep",
                    help="which basins to sample (e.g. 'low' for a cross-seed "
                    "replicate of the money product on the otherwise-idle GPU3)")
    args = ap.parse_args()

    if os.environ.get("GIGALENS_X64") != "1":
        raise SystemExit("set GIGALENS_X64=1 (real-data marg is float64-only)")
    if args.mode == "smoke":
        smoke(args)
    else:
        prod(args)


if __name__ == "__main__":
    main()
