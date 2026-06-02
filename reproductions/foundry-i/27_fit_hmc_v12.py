"""v12: single-chain fixed-leapfrog Preconditioned HMC for the foundry-i lens.

This is the production fixed-leapfrog HMC counterpart to the v11f NUTS run. The
compile diagnostic (26_compile_diagnostic.py) established that the fixed-leapfrog
PHMC kernel (kernel B) lowers/compiles cheaply, unlike the original gigalens
GradientBasedTrajectoryLengthAdaptation stack (kernel A). Here we actually RUN it.

Kernel:
  PreconditionedHamiltonianMonteCarlo(num_leapfrog_steps=L, momentum=Sigma_hat^-1)
    -> DualAveragingStepSizeAdaptation (step-size only; NO trajectory-length
       adaptation, so no ChEES, so a single chain is fine).

Momentum modes (see _hmc_lib.momentum_distribution):
  inv  (default, CORRECT): momentum cov = Sigma_hat^-1  (mass matrix = Sigma_hat)
  fwd  (v11f-bug control) : momentum cov = Sigma_hat
  diag                    : momentum cov = diag(1/var_i)

All model / target_log_prob construction is imported from _hmc_lib.build_model();
this file does NOT duplicate the forward model.

Argparse flags:
  --mode {inv,fwd,diag}  default inv
  --num-leapfrog INT     default 16
  --step-size FLOAT      default 1e-3
  --burn INT             default 500
  --keep INT             default 800
  --out PATH             default data/hmc_v12_{mode}.npz
"""
# ---------------------------------------------------------------------------
# Enable the JAX persistent compile cache BEFORE importing jax-heavy modules.
# ---------------------------------------------------------------------------
import jax
jax.config.update(
    'jax_compilation_cache_dir',
    '/raid/benson/git/agentic-lensing/reproductions/foundry-i/.jax_cache',
)
jax.config.update('jax_persistent_cache_min_compile_time_secs', 1.0)

import argparse
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import tensorflow_probability.substrates.jax as tfp

import _hmc_lib

tfd = tfp.distributions
tfe = tfp.experimental
REPRO = Path(__file__).parent


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["inv", "fwd", "diag"], default="inv",
                    help="momentum distribution mode (default: inv = correct)")
    ap.add_argument("--num-leapfrog", type=int, default=16,
                    help="number of fixed leapfrog steps per HMC iteration")
    ap.add_argument("--step-size", type=float, default=1e-3,
                    help="initial leapfrog step size")
    ap.add_argument("--burn", type=int, default=500,
                    help="number of burn-in steps")
    ap.add_argument("--keep", type=int, default=800,
                    help="number of kept (post-burn) samples")
    ap.add_argument("--out", type=str, default=None,
                    help="output .npz path (default: data/hmc_v12_{mode}.npz)")
    return ap.parse_args()


def main():
    args = parse_args()
    out_path = (Path(args.out) if args.out is not None
                else REPRO / "data" / f"hmc_v12_{args.mode}.npz")
    if not out_path.is_absolute():
        out_path = REPRO / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"devices: {jax.devices()}", flush=True)
    print(f"\nv12 fixed-leapfrog PHMC config:", flush=True)
    print(f"  mode (momentum)     = {args.mode}", flush=True)
    print(f"  num_leapfrog_steps  = {args.num_leapfrog}", flush=True)
    print(f"  init step_size      = {args.step_size}", flush=True)
    print(f"  num_burnin / kept   = {args.burn} / {args.keep}", flush=True)
    print(f"  num_adaptation_steps= {int(0.8 * args.burn)} (step-size only)", flush=True)
    print(f"  out                 = {out_path}", flush=True)

    # ----- model + momentum --------------------------------------------------
    m = _hmc_lib.build_model()
    momentum = _hmc_lib.momentum_distribution(args.mode)

    lp_start = m.target_log_prob_fn(m.qz_start)
    lp_start.block_until_ready()
    print(f"\nndim={m.ndim}  log_p(qz_start)={float(lp_start):.1f}", flush=True)

    # ----- kernel ------------------------------------------------------------
    kernel = tfe.mcmc.PreconditionedHamiltonianMonteCarlo(
        target_log_prob_fn=m.target_log_prob_fn,
        momentum_distribution=momentum,
        step_size=args.step_size,
        num_leapfrog_steps=args.num_leapfrog,
    )
    # Step-size adaptation only. No GradientBasedTrajectoryLengthAdaptation, so
    # no ChEES, so a single chain is valid. PHMC wraps its leapfrog kernel in a
    # MetropolisHastings step. DualAveragingStepSizeAdaptation's DEFAULT
    # getter/setter/log_accept_prob fns already understand the
    # MetropolisHastingsKernelResults -> accepted_results.step_size structure and,
    # crucially, re-broadcast step_size correctly during bootstrap (the raw PHMC
    # bootstrap leaves accepted_results.step_size == [] which the custom lambdas
    # propagated, driving the adapted step_size to +inf and acceptance to 0).
    # Use the defaults.
    adapted = tfp.mcmc.DualAveragingStepSizeAdaptation(
        inner_kernel=kernel,
        num_adaptation_steps=int(0.8 * args.burn),
        target_accept_prob=jnp.float32(0.75),
    )

    def trace_fn(_, pkr):
        # pkr is the DASSA results; pkr.inner_results is the
        # MetropolisHastingsKernelResults wrapping the PHMC leapfrog kernel.
        ir = pkr.inner_results
        out = {
            "is_accepted":     ir.is_accepted,
            "target_log_prob": ir.accepted_results.target_log_prob,
            "step_size":       pkr.new_step_size,
        }
        # per-step accept ratio (MH exposes log_accept_ratio at the top level).
        lar = getattr(ir, "log_accept_ratio", None)
        if lar is not None:
            out["accept_ratio"] = jnp.exp(jnp.minimum(lar, 0.0))
        return out

    # ----- run ---------------------------------------------------------------
    print(f"\nRunning HMC sample_chain (compile + run) ...", flush=True)
    t0 = time.time()
    samples, trace = tfp.mcmc.sample_chain(
        num_results=args.keep,
        num_burnin_steps=args.burn,
        current_state=m.qz_start,
        kernel=adapted,
        trace_fn=trace_fn,
        seed=jax.random.PRNGKey(0),
    )
    samples.block_until_ready()
    elapsed = time.time() - t0
    print(f"HMC done in {elapsed:.1f}s ({elapsed / 60:.2f} min)", flush=True)

    # ----- diagnostics -------------------------------------------------------
    samples_np = np.asarray(samples)
    is_acc = np.asarray(trace["is_accepted"])
    lps = np.asarray(trace["target_log_prob"])
    ss = np.asarray(trace["step_size"])
    has_ar = "accept_ratio" in trace
    accept_ratio = np.asarray(trace["accept_ratio"]) if has_ar else None

    diff = np.linalg.norm(np.diff(samples_np, axis=0), axis=1)
    per_param_std = samples_np.std(axis=0)
    final_ss = float(ss[-1])
    step_collapsed = final_ss < 1e-5
    any_nan = bool(np.any(np.isnan(samples_np)))

    print(f"\nDiagnostics over {args.keep} kept samples:", flush=True)
    print(f"  Acceptance rate      : {float(is_acc.mean()):.3f}", flush=True)
    if has_ar:
        print(f"  mean per-step accept : {float(np.nanmean(accept_ratio)):.3f}",
              flush=True)
    print(f"  target_log_prob      : min={lps.min():.1f}, "
          f"median={np.median(lps):.1f}, max={lps.max():.1f}", flush=True)
    print(f"  step_size            : start={float(ss[0]):.5g}, "
          f"final={final_ss:.5g}", flush=True)
    print(f"  step_collapsed (<1e-5): {step_collapsed}", flush=True)
    print(f"  ||Δz|| step-to-step  : median={np.median(diff):.4f}, "
          f"max={diff.max():.4f}", flush=True)
    print(f"  per-param sample std : min={per_param_std.min():.4e}, "
          f"max={per_param_std.max():.4e}", flush=True)
    print(f"  any NaN              : {any_nan}", flush=True)

    # ESS of the 6 physical mass params.
    combined = _hmc_lib.to_physical_mass(samples_np, prob_model=m.prob_model)
    ess = {}
    print(f"\n  ESS (effective_sample_size) on 6 physical mass params:", flush=True)
    for k in ["theta_E", "gamma", "e1", "e2", "gamma1", "gamma2"]:
        if k not in combined:
            continue
        ess_k = float(np.asarray(
            tfp.mcmc.effective_sample_size(jnp.asarray(combined[k]))))
        ess[k] = ess_k
        print(f"    {k:>10s}: ESS={ess_k:8.1f}", flush=True)

    # ----- comparison table --------------------------------------------------
    print("", flush=True)
    _hmc_lib.print_comparison(combined)

    # ----- save --------------------------------------------------------------
    save_kw = dict(
        samples=samples_np,
        is_accepted=is_acc,
        target_log_prob=lps,
        step_size=ss,
        mode=args.mode,
        num_leapfrog=args.num_leapfrog,
        init_step_size=args.step_size,
        adapted_step_size=final_ss,
        step_collapsed=step_collapsed,
        elapsed=elapsed,
        burn=args.burn,
        keep=args.keep,
        initial_state=np.asarray(m.qz_start),
        ess_keys=np.array(list(ess.keys())),
        ess_vals=np.array(list(ess.values()), dtype=np.float64),
    )
    if has_ar:
        save_kw["accept_ratio"] = accept_ratio
    # physical mass params, prefixed to avoid key collisions
    for k, v in combined.items():
        save_kw[f"mass_{k}"] = np.asarray(v)
    np.savez(out_path, **save_kw)
    print(f"\nSaved {out_path}", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
