"""Stage 5 sampler: flexible HMC/NUTS runner for the lstsq-REDUCED model.

_hmc_lib_lstsq.build_model_lstsq() marginalizes the 33 linear light amplitudes
via least squares per log_prob call, so the sampler explores only the 41
NONLINEAR params. The reduced Hessian has NO flat directions (conditioning
fixed). 30_refine_lstsq.py drove the projected full-model MAP from a 12-negative
saddle to a PD mode (data/map_refined_lstsq.npz) and cached the PD-floored
reduced Hessian (data/hess_refined_lstsq.npz['H_reg']). This runner samples from
that geometry: a clean PD start + curvature-matched preconditioner should finally
let HMC mix (ESS up, gamma std toward the paper's 0.023, vs the baseline ESS_min~4
of 800 and gamma std ~1e-4).

Flags:
  --kernel {hmc,nuts}        default hmc
  --massmatrix {hess_refined_lstsq, diag, identity}
        default hess_refined_lstsq: momentum COVARIANCE = the PD-floored reduced
            Hessian H_reg via MultivariateNormalTriL(scale_tril=chol(H_reg)).
            Under the TFP preconditioned-HMC convention the optimal mass matrix
            M = Sigma_post^-1 ~= H (Laplace precision); momentum p ~ N(0, M).
        diag: momentum covariance = diag(H_reg) (diagonal precision only).
        identity: momentum covariance = I.
  --num-leapfrog INT         hmc, default 16
  --max-tree-depth INT       nuts, default 8
  --target-accept FLOAT      default 0.8
  --step-size FLOAT          default 1e-3
  --burn INT                 default 500
  --keep INT                 default 800
  --out PATH                 default data/lstsq_{kernel}_{massmatrix}.npz

Smoke test:
  CUDA_DEVICE_ORDER=PCI_BUS_ID XLA_FLAGS=--xla_gpu_autotune_level=0 \
    CUDA_VISIBLE_DEVICES=1 /raid/benson/.venvs/gigalens/bin/python 31_fit_lstsq.py \
    --kernel hmc --burn 30 --keep 50 --out data/lstsq_smoke.npz

--x64: run the whole reduced model + sampler in float64 and read the float64 MAP /
  Hessian (data/map_refined_lstsq64.npz, data/hess_refined_lstsq64.npz). Required
  to match 30_refine_lstsq.py --x64 (cond ~1e10 only fits float64). A16 FP64 is
  slow -- size chains using sec_per_grad reported by 30_refine_lstsq.py --x64.
"""
import argparse
import os

# --x64 must enable jax_enable_x64 BEFORE _hmc_lib_lstsq is imported (it sets x64
# at its own import based on GIGALENS_X64, before any jnp array). Parse it first.
_ap0 = argparse.ArgumentParser(add_help=False)
_ap0.add_argument("--x64", action="store_true",
                  help="run reduced model + sampler in float64")
_args0, _ = _ap0.parse_known_args()
if _args0.x64:
    os.environ["GIGALENS_X64"] = "1"

import jax
if _args0.x64:
    jax.config.update("jax_enable_x64", True)
jax.config.update(
    'jax_compilation_cache_dir',
    '/raid/benson/git/agentic-lensing/reproductions/foundry-i/.jax_cache',
)
jax.config.update('jax_persistent_cache_min_compile_time_secs', 1.0)

import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import tensorflow_probability.substrates.jax as tfp

import _hmc_lib
import _hmc_lib_lstsq

tfd = tfp.distributions
tfe = tfp.experimental
REPRO = Path(__file__).parent
DATA = REPRO / "data"


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--x64", action="store_true",
                    help="run reduced model + sampler in float64 (read *64.npz)")
    ap.add_argument("--kernel", choices=["hmc", "nuts"], default="hmc")
    ap.add_argument("--massmatrix",
                    choices=["hess_refined_lstsq", "diag", "identity"],
                    default="hess_refined_lstsq")
    ap.add_argument("--num-leapfrog", type=int, default=16,
                    help="fixed leapfrog steps per HMC iteration (hmc only)")
    ap.add_argument("--max-tree-depth", type=int, default=8,
                    help="NUTS max tree depth (nuts only)")
    ap.add_argument("--target-accept", type=float, default=0.8)
    ap.add_argument("--step-size", type=float, default=1e-3)
    ap.add_argument("--burn", type=int, default=500)
    ap.add_argument("--keep", type=int, default=800)
    ap.add_argument("--out", type=str, default=None)
    # --pd: start from the saddle-free-Newton PD mode (32_saddlefree_newton.py)
    # instead of the optax-refined saddle, and use its Hessian as the
    # hess_refined_lstsq mass matrix.  Shorthand for
    #   --start-file data/map_pd_lstsq64.npz --mass-file data/hess_pd_lstsq64.npz
    # with the matching key (qz_pd).  --start-file / --mass-file override it.
    ap.add_argument("--pd", action="store_true",
                    help="start from the saddle-free-Newton PD mode "
                         "(data/map_pd_lstsq64.npz) and use data/hess_pd_lstsq64.npz "
                         "as the hess_refined_lstsq mass matrix")
    # --trust: start from a 33_trust_refine.py trust-region PD mode and use its
    # Hessian as the mass matrix. Shorthand for
    #   --start-file data/map_trust_{seed}.npz --mass-file data/hess_trust_{seed}.npz
    # (the trust npz stores the start under qz_refined and the Hessian under
    # H_reg/chol, the keys this runner already reads). --start-file / --mass-file
    # override it.
    ap.add_argument("--trust", choices=["paper", "current"], default=None,
                    help="start from the trust-region PD mode "
                         "(data/map_trust_{seed}.npz) and use "
                         "data/hess_trust_{seed}.npz as the hess_refined_lstsq "
                         "mass matrix; SEED selects the paper- or current-seeded run")
    ap.add_argument("--start-file", type=str, default=None,
                    help="npz with the start vector (qz/qz_pd/qz_refined key); "
                         "overrides the default refined-MAP start")
    ap.add_argument("--mass-file", type=str, default=None,
                    help="npz with H_reg (+chol) for the hess_refined_lstsq mass "
                         "matrix; overrides the default refined Hessian")
    return ap.parse_args()


def build_momentum(massmatrix, ndim, fdtype, hess_path):
    """Return a tfd.Distribution over R^ndim (the HMC momentum distribution).

    Under the TFP preconditioned-HMC convention the optimal mass matrix is the
    posterior precision M = Sigma_post^-1 ~= H (Laplace), and the momentum is
    p ~ N(0, M). So the momentum distribution's COVARIANCE must equal the
    PD-floored reduced Hessian H_reg.  fdtype/hess_path follow --x64 / --mass-file.
    """
    if massmatrix == "hess_refined_lstsq":
        d = np.load(hess_path)
        H_reg = jnp.asarray(d["H_reg"], dtype=fdtype)
        # Prefer the cached cholesky if present (already PD by construction);
        # otherwise recompute it from H_reg.
        if "chol" in d:
            chol = jnp.asarray(d["chol"], dtype=fdtype)
        else:
            chol = jnp.linalg.cholesky(H_reg)
        return tfd.MultivariateNormalTriL(
            loc=jnp.zeros(ndim, dtype=fdtype), scale_tril=chol)
    if massmatrix == "diag":
        d = np.load(hess_path)
        H_reg = np.asarray(d["H_reg"], dtype=np.float64)
        # momentum covariance = diag(H_reg); scale_diag = sqrt(diag).
        scale_diag = jnp.asarray(np.sqrt(np.maximum(np.diag(H_reg), 1e-30)),
                                 dtype=fdtype)
        return tfd.MultivariateNormalDiag(
            loc=jnp.zeros(ndim, dtype=fdtype), scale_diag=scale_diag)
    # identity: covariance = I.
    return tfd.MultivariateNormalDiag(
        loc=jnp.zeros(ndim, dtype=fdtype),
        scale_diag=jnp.ones(ndim, dtype=fdtype))


def main():
    args = parse_args()
    x64 = bool(args.x64)
    fdtype = jnp.float64 if x64 else jnp.float32
    suffix = "64" if x64 else ""
    # --trust / --pd are shorthands for a PD-mode start + its Hessian; explicit
    # --start-file / --mass-file (if given) take precedence over all of them.
    # --trust takes precedence over --pd.
    if args.trust is not None:
        map_name = f"map_trust_{args.trust}.npz"
        hess_name = f"hess_trust_{args.trust}.npz"
    elif args.pd:
        map_name = "map_pd_lstsq64.npz"
        hess_name = "hess_pd_lstsq64.npz"
    else:
        map_name = f"map_refined_lstsq{suffix}.npz"
        hess_name = f"hess_refined_lstsq{suffix}.npz"
    start_path = Path(args.start_file) if args.start_file else DATA / map_name
    if not start_path.is_absolute():
        start_path = REPRO / start_path
    mass_path = Path(args.mass_file) if args.mass_file else DATA / hess_name
    if not mass_path.is_absolute():
        mass_path = REPRO / mass_path
    default_name = f"lstsq_{args.kernel}_{args.massmatrix}{suffix}.npz"
    out_path = Path(args.out) if args.out is not None else DATA / default_name
    if not out_path.is_absolute():
        out_path = REPRO / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"devices: {jax.devices()}", flush=True)
    print(f"x64={x64}  jax_enable_x64={jax.config.jax_enable_x64}", flush=True)

    # ----- reduced model -----------------------------------------------------
    m = _hmc_lib_lstsq.build_model_lstsq(x64=x64)

    # ----- start point: PD mode (--pd) or refined MAP ------------------------
    rd = np.load(start_path, allow_pickle=True)
    # The saddle-free PD-mode npz stores the start under 'qz_pd'; the refined-MAP
    # and trust-region npz under 'qz_refined' (the trust npz also stores 'qz').
    # Accept any so --start-file / --pd / --trust all work.
    if "qz_pd" in rd:
        start = jnp.asarray(rd["qz_pd"], dtype=fdtype)
        start_kind = "saddle-free-Newton PD mode"
    elif "qz_refined" in rd:
        start = jnp.asarray(rd["qz_refined"], dtype=fdtype)
        start_kind = ("trust-region PD mode" if "nit" in rd
                      else "refined lstsq MAP")
    else:
        start = jnp.asarray(rd["qz"], dtype=fdtype)
        start_kind = "PD mode (qz)"
    gn = float(rd["grad_norm_after"]) if "grad_norm_after" in rd else float("nan")
    lp_saved = (float(rd["logp_after"]) if "logp_after" in rd
                else (float(rd["logp"]) if "logp" in rd else float("nan")))
    nlneg = int(rd["n_large_negative"]) if "n_large_negative" in rd else -1
    is_pd = bool(rd["is_pd_mode"]) if "is_pd_mode" in rd else False
    print(f"start = {start_kind} ({start_path})  "
          f"saved ||grad||={gn:.4e} log_p={lp_saved:.2f}  "
          f"n_large_negative={nlneg}  is_pd_mode={is_pd}", flush=True)
    print(f"mass-matrix source     = {mass_path}", flush=True)

    momentum = build_momentum(args.massmatrix, m.ndim, fdtype, mass_path)

    lp_start = m.target_log_prob_fn(start)
    lp_start.block_until_ready()
    print(f"\nconfig:", flush=True)
    print(f"  kernel              = {args.kernel}", flush=True)
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
        # (the 29_fit_from_refined.py / 27_fit_hmc_v12.py fix). Use the defaults.
        inner = tfe.mcmc.PreconditionedHamiltonianMonteCarlo(
            target_log_prob_fn=m.target_log_prob_fn,
            momentum_distribution=momentum,
            step_size=args.step_size,
            num_leapfrog_steps=args.num_leapfrog,
        )
        adapted = tfp.mcmc.DualAveragingStepSizeAdaptation(
            inner_kernel=inner,
            num_adaptation_steps=int(0.8 * args.burn),
            target_accept_prob=jnp.asarray(args.target_accept, dtype=fdtype),
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
        # getter/setter/log_accept lambdas (25_fit_nuts_v11f.py / 29).
        inner = tfe.mcmc.PreconditionedNoUTurnSampler(
            target_log_prob_fn=m.target_log_prob_fn,
            momentum_distribution=momentum,
            step_size=args.step_size,
            max_tree_depth=args.max_tree_depth,
        )
        adapted = tfp.mcmc.DualAveragingStepSizeAdaptation(
            inner_kernel=inner,
            num_adaptation_steps=int(0.8 * args.burn),
            target_accept_prob=jnp.asarray(args.target_accept, dtype=fdtype),
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
    combined = m.to_physical_mass(samples_np)
    ess = {}
    print(f"\n  ESS (effective_sample_size) on 6 physical mass params:", flush=True)
    for k in ["theta_E", "gamma", "e1", "e2", "gamma1", "gamma2"]:
        if k not in combined:
            continue
        ess_k = float(np.asarray(
            tfp.mcmc.effective_sample_size(jnp.asarray(combined[k]))))
        ess[k] = ess_k
        print(f"    {k:>10s}: ESS={ess_k:8.1f}", flush=True)
    ess_min = float(min(ess.values())) if ess else float("nan")
    print(f"    {'ESS_min':>10s}: {ess_min:8.1f}  (baseline to beat ~4 of "
          f"{args.keep})", flush=True)

    gamma_std = float(np.std(combined["gamma"])) if "gamma" in combined else float("nan")
    theta_E_median = (float(np.median(combined["theta_E"]))
                      if "theta_E" in combined else float("nan"))
    print(f"\n  gamma posterior std   : {gamma_std:.6e}  (paper 0.023, baseline ~1e-4)",
          flush=True)
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
        x64=np.bool_(x64),
        kernel=args.kernel,
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
        ess_min=ess_min,
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
