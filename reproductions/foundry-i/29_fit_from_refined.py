"""Stage 4 sampler: flexible HMC/NUTS runner started from the refined MAP.

Stage 3 found the v7 "paper-mode MAP" qz_v7 is a SADDLE (15 large negative
Hessian eigenvalues of -log_post). 28_refine_map.py drove ||grad|| down ~3 orders
of magnitude into a near-stationary point qz_refined (data/map_refined.npz) whose
Hessian has NO large negative eigenvalues, and computed a PD-floored Hessian mass
matrix (data/hess_refined.npz['H_reg']). This runner samples from that better
geometry to see whether a clean start + curvature-matched preconditioner finally
lets HMC/NUTS mix (ESS up, gamma std toward the paper's 0.023).

Flags:
  --kernel {hmc,nuts}        default hmc
  --start {v7,refined}       default refined (loads data/map_refined.npz['qz_refined'])
  --massmatrix {hess_refined,hess_v7,inv,diag}
        default hess_refined: momentum COVARIANCE = the PD-floored refined Hessian
            H_reg via MultivariateNormalTriL(scale_tril=chol(H_reg)). Under the TFP
            preconditioned-HMC convention the optimal mass matrix M = Sigma_post^-1
            ~= H (Laplace precision); momentum p ~ N(0, M).
        hess_v7: same construction on data/hess_massmatrix.npz (the v7-saddle Hessian).
        inv / diag: _hmc_lib.momentum_distribution('inv'|'diag') over Sigma_hat.
  --num-leapfrog INT         hmc, default 16
  --max-tree-depth INT       nuts, default 8
  --target-accept FLOAT      default 0.8
  --step-size FLOAT          default 1e-3
  --burn INT                 default 500
  --keep INT                 default 800
  --out PATH                 default data/refined_{kernel}_{start}_{massmatrix}.npz

Smoke test:
  CUDA_DEVICE_ORDER=PCI_BUS_ID XLA_FLAGS=--xla_gpu_autotune_level=0 \
    CUDA_VISIBLE_DEVICES=1 /raid/benson/.venvs/gigalens/bin/python 29_fit_from_refined.py \
    --kernel hmc --start refined --massmatrix hess_refined --burn 30 --keep 50 \
    --out data/refined_smoke.npz
"""
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
DATA = REPRO / "data"


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kernel", choices=["hmc", "nuts"], default="hmc")
    ap.add_argument("--start", choices=["v7", "refined"], default="refined")
    ap.add_argument("--massmatrix",
                    choices=["hess_refined", "hess_v7", "inv", "diag"],
                    default="hess_refined")
    ap.add_argument("--num-leapfrog", type=int, default=16,
                    help="fixed leapfrog steps per HMC iteration (hmc only)")
    ap.add_argument("--max-tree-depth", type=int, default=8,
                    help="NUTS max tree depth (nuts only)")
    ap.add_argument("--target-accept", type=float, default=0.8)
    ap.add_argument("--step-size", type=float, default=1e-3)
    ap.add_argument("--burn", type=int, default=500)
    ap.add_argument("--keep", type=int, default=800)
    ap.add_argument("--out", type=str, default=None)
    return ap.parse_args()


def build_momentum(massmatrix):
    """Return a tfd.Distribution over R^74 (the HMC momentum distribution)."""
    if massmatrix == "hess_refined":
        d = np.load(DATA / "hess_refined.npz")
        H_reg = jnp.asarray(d["H_reg"], dtype=jnp.float32)
        ndim = H_reg.shape[0]
        chol = jnp.linalg.cholesky(H_reg)
        return tfd.MultivariateNormalTriL(
            loc=jnp.zeros(ndim, dtype=jnp.float32), scale_tril=chol)
    if massmatrix == "hess_v7":
        # the v7-saddle Hessian (regularized) cached by _hmc_lib.laplace_hessian.
        return _hmc_lib.momentum_distribution("hess")
    # inv / diag over Sigma_hat.
    return _hmc_lib.momentum_distribution(massmatrix)


def main():
    args = parse_args()
    default_name = f"refined_{args.kernel}_{args.start}_{args.massmatrix}.npz"
    out_path = Path(args.out) if args.out is not None else DATA / default_name
    if not out_path.is_absolute():
        out_path = REPRO / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"devices: {jax.devices()}", flush=True)

    # ----- model -------------------------------------------------------------
    m = _hmc_lib.build_model()

    # ----- start point -------------------------------------------------------
    if args.start == "refined":
        rd = np.load(DATA / "map_refined.npz")
        start = jnp.asarray(rd["qz_refined"], dtype=jnp.float32)
        print(f"start = refined MAP (data/map_refined.npz)  "
              f"saved ||grad||={float(rd['grad_norm_after']):.4e} "
              f"log_p={float(rd['logp_after']):.2f}", flush=True)
    else:
        start = jnp.asarray(m.qz_start, dtype=jnp.float32)
        print(f"start = v7 paper-mode saddle (m.qz_start)", flush=True)

    momentum = build_momentum(args.massmatrix)

    lp_start = m.target_log_prob_fn(start)
    lp_start.block_until_ready()
    print(f"\nconfig:", flush=True)
    print(f"  kernel              = {args.kernel}", flush=True)
    print(f"  start               = {args.start}", flush=True)
    print(f"  massmatrix          = {args.massmatrix}", flush=True)
    if args.kernel == "hmc":
        print(f"  num_leapfrog_steps  = {args.num_leapfrog}", flush=True)
    else:
        print(f"  max_tree_depth      = {args.max_tree_depth}", flush=True)
    print(f"  target_accept_prob  = {args.target_accept}", flush=True)
    print(f"  init step_size      = {args.step_size}", flush=True)
    print(f"  num_burnin / kept   = {args.burn} / {args.keep}", flush=True)
    print(f"  num_adaptation_steps= {int(0.8 * args.burn)} (step-size only)", flush=True)
    print(f"  out                 = {out_path}", flush=True)
    print(f"  ndim={m.ndim}  log_p(start)={float(lp_start):.2f}", flush=True)

    # ----- kernel + step-size adaptation -------------------------------------
    if args.kernel == "hmc":
        # PHMC wraps its leapfrog kernel in a MetropolisHastings step. DASSA's
        # DEFAULT getter/setter understand MetropolisHastingsKernelResults ->
        # accepted_results.step_size and re-broadcast correctly during bootstrap
        # (the 27_fit_hmc_v12.py fix). Use the defaults for hmc.
        inner = tfe.mcmc.PreconditionedHamiltonianMonteCarlo(
            target_log_prob_fn=m.target_log_prob_fn,
            momentum_distribution=momentum,
            step_size=args.step_size,
            num_leapfrog_steps=args.num_leapfrog,
        )
        adapted = tfp.mcmc.DualAveragingStepSizeAdaptation(
            inner_kernel=inner,
            num_adaptation_steps=int(0.8 * args.burn),
            target_accept_prob=jnp.float32(args.target_accept),
        )

        def trace_fn(_, pkr):
            ir = pkr.inner_results
            out = {
                "is_accepted":     ir.is_accepted,
                "target_log_prob": ir.accepted_results.target_log_prob,
                "step_size":       pkr.new_step_size,
            }
            lar = getattr(ir, "log_accept_ratio", None)
            if lar is not None:
                out["accept_ratio"] = jnp.exp(jnp.minimum(lar, 0.0))
            return out
    else:
        # PreconditionedNoUTurnSampler: step_size lives directly on the kernel
        # results (no MetropolisHastings wrap), so use the explicit v11f
        # getter/setter/log_accept lambdas.
        inner = tfe.mcmc.PreconditionedNoUTurnSampler(
            target_log_prob_fn=m.target_log_prob_fn,
            momentum_distribution=momentum,
            step_size=args.step_size,
            max_tree_depth=args.max_tree_depth,
        )
        adapted = tfp.mcmc.DualAveragingStepSizeAdaptation(
            inner_kernel=inner,
            num_adaptation_steps=int(0.8 * args.burn),
            target_accept_prob=jnp.float32(args.target_accept),
            step_size_setter_fn=lambda pkr, new_ss: pkr._replace(step_size=new_ss),
            step_size_getter_fn=lambda pkr: pkr.step_size,
            log_accept_prob_getter_fn=lambda pkr: pkr.log_accept_ratio,
        )

        def trace_fn(_, pkr):
            ir = pkr.inner_results
            return {
                "is_accepted":     ir.is_accepted,
                "target_log_prob": ir.target_log_prob,
                "step_size":       pkr.new_step_size,
                "accept_ratio":    jnp.exp(jnp.minimum(ir.log_accept_ratio, 0.0)),
            }

    # ----- run ---------------------------------------------------------------
    print(f"\nRunning {args.kernel} sample_chain (compile + run) ...", flush=True)
    t0 = time.time()
    samples, trace = tfp.mcmc.sample_chain(
        num_results=args.keep,
        num_burnin_steps=args.burn,
        current_state=start,
        kernel=adapted,
        trace_fn=trace_fn,
        seed=jax.random.PRNGKey(0),
    )
    samples.block_until_ready()
    elapsed = time.time() - t0
    print(f"{args.kernel} done in {elapsed:.1f}s ({elapsed / 60:.2f} min)", flush=True)

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
    print(f"  Acceptance rate       : {float(is_acc.mean()):.3f}", flush=True)
    if has_ar:
        print(f"  mean per-step accept  : {float(np.nanmean(accept_ratio)):.3f}",
              flush=True)
    print(f"  target_log_prob       : min={lps.min():.1f}, "
          f"median={np.median(lps):.1f}, max={lps.max():.1f}", flush=True)
    print(f"  step_size             : start={float(ss[0]):.5g}, "
          f"final={final_ss:.5g}", flush=True)
    print(f"  step_collapsed (<1e-5): {step_collapsed}", flush=True)
    print(f"  ||Δz|| step-to-step   : median={np.median(diff):.4f}, "
          f"max={diff.max():.4f}", flush=True)
    print(f"  per-param sample std  : min={per_param_std.min():.4e}, "
          f"max={per_param_std.max():.4e}", flush=True)
    print(f"  any NaN               : {any_nan}", flush=True)

    # ESS of the 6 physical mass params + gamma std + theta_E median.
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

    gamma_std = float(np.std(combined["gamma"])) if "gamma" in combined else float("nan")
    theta_E_median = (float(np.median(combined["theta_E"]))
                      if "theta_E" in combined else float("nan"))
    print(f"\n  gamma posterior std   : {gamma_std:.6e}  (paper 0.023)", flush=True)
    print(f"  theta_E median        : {theta_E_median:.6f}  (paper 2.6463)", flush=True)

    # ----- comparison table --------------------------------------------------
    print("", flush=True)
    _hmc_lib.print_comparison(combined)

    # ----- save --------------------------------------------------------------
    save_kw = dict(
        samples=samples_np,
        is_accepted=is_acc,
        target_log_prob=lps,
        step_size=ss,
        kernel=args.kernel,
        start=args.start,
        massmatrix=args.massmatrix,
        num_leapfrog=args.num_leapfrog,
        max_tree_depth=args.max_tree_depth,
        target_accept=args.target_accept,
        init_step_size=args.step_size,
        adapted_step_size=final_ss,
        step_collapsed=step_collapsed,
        median_dz=float(np.median(diff)),
        gamma_std=gamma_std,
        theta_E_median=theta_E_median,
        any_nan=any_nan,
        elapsed=elapsed,
        burn=args.burn,
        keep=args.keep,
        initial_state=np.asarray(start),
        ess_keys=np.array(list(ess.keys())),
        ess_vals=np.array(list(ess.values()), dtype=np.float64),
    )
    if has_ar:
        save_kw["accept_ratio"] = accept_ratio
    for k, v in combined.items():
        save_kw[f"mass_{k}"] = np.asarray(v)
    np.savez(out_path, **save_kw)
    print(f"\nSaved {out_path}", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
