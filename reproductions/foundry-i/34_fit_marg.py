"""Stage 5 (marginal): refine + HMC driver for the ridge-regularized Gaussian-
marginalized model (_hmc_lib_marg.build_model_marg, ndim=46).

WHY THIS MODEL.  The lstsq-profiled model (_hmc_lib_lstsq.build_model_lstsq) uses
gigalens lstsq_simulate = pinv on a near-rank-deficient 33-column amplitude design
matrix.  pinv truncates tiny singular values, which is NON-SMOOTH in theta:
||grad(-log_post)|| floors at ~1e5 even in float64, the reduced Hessian stays
indefinite, and no genuine PD mode / HMC mixing is reachable (33_trust_refine.py
on the lstsq target stalled with persistent negative eigenvalues).

THE FIX (_hmc_lib_marg).  Marginalize ONLY the 28 SHAPELET amps analytically using
their Gaussian priors Normal(0, sigma_i), sigma_i = 5/sqrt(i+1), as Tikhonov
regularization.  The shapelet normal matrix A = X^T W X + Lambda (Lambda_ii =
(i+1)/25) is ALWAYS positive-definite, so the amplitude solve is a SMOOTH Cholesky
solve (never pinv).  The marginal log-likelihood carries the Gaussian-evidence /
Occam term -0.5*logdet(A) that gigalens OMITS; this is what removes the indefinite
curvature.  The 5 Sersic Ie are SAMPLED (LogNormal -> strictly positive,
non-degenerate).  Sampled dim = 41 nonlinear + 5 Sersic Ie = 46.

This script mirrors the two proven runners on this NEW smooth target:

  --mode refine : scipy.optimize.minimize(method='trust-exact') -- the SAME
        trust-region Newton 33_trust_refine.py uses (More-Sorensen exact subproblem
        solve; handles indefinite/stiff Hessians).  On the smooth marginal target a
        genuine PD minimum should now exist, so trust-exact should drive ||grad||
        to a true stationary point with n_negative=0.  Saves the refined point +
        Hessian as data/map_marg_pd.npz / data/hess_marg_pd.npz (default --out), in
        the same key layout 31_fit_lstsq.py / this script's hmc mode read.

  --mode hmc    : PreconditionedHamiltonianMonteCarlo (fixed leapfrog) +
        DualAveragingStepSizeAdaptation -- the SAME PHMC 31_fit_lstsq.py uses.
        Mass matrix = the PD Hessian at the refined mode (momentum covariance =
        H_reg via MultivariateNormalTriL(scale_tril=chol(H_reg)); TFP PHMC optimal
        mass matrix M = Sigma_post^-1 ~= H).  Start = the refined PD mode.

ALWAYS float64 (--x64 default on): the curvature is stiff; float64 is required for
both the trust-region solve and a meaningful gradient (float32 saturates near
~1e4).  x64 is enabled BEFORE _hmc_lib_marg is imported (the module sets x64 at its
own import from GIGALENS_X64, before any jnp array is created).

Run refine (A16 index 4):
  CUDA_DEVICE_ORDER=PCI_BUS_ID XLA_FLAGS=--xla_gpu_autotune_level=0 \
    CUDA_VISIBLE_DEVICES=4 /raid/benson/.venvs/gigalens/bin/python 34_fit_marg.py \
      --mode refine --x64 --maxiter 80 --out data/map_marg_pd.npz

Run hmc from the refined PD mode (A16 index 4):
  CUDA_DEVICE_ORDER=PCI_BUS_ID XLA_FLAGS=--xla_gpu_autotune_level=0 \
    CUDA_VISIBLE_DEVICES=4 /raid/benson/.venvs/gigalens/bin/python 34_fit_marg.py \
      --mode hmc --x64 --num-leapfrog 16 --burn 500 --keep 800 \
      --start-file data/map_marg_pd.npz --mass-file data/hess_marg_pd.npz
"""
import argparse
import os

# --x64 (default ON) must enable jax_enable_x64 BEFORE _hmc_lib_marg is imported
# (the module sets x64 at its own import based on GIGALENS_X64, before any jnp
# array is created).  Parse it first, then set the env var.
_ap0 = argparse.ArgumentParser(add_help=False)
_ap0.add_argument("--x64", dest="x64", action="store_true", default=True)
_ap0.add_argument("--no-x64", dest="x64", action="store_false")
_args0, _ = _ap0.parse_known_args()
if _args0.x64:
    os.environ["GIGALENS_X64"] = "1"

import jax  # noqa: E402

if _args0.x64:
    jax.config.update("jax_enable_x64", True)
jax.config.update(
    "jax_compilation_cache_dir",
    "/raid/benson/git/agentic-lensing/reproductions/foundry-i/.jax_cache",
)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)

import time  # noqa: E402
from pathlib import Path  # noqa: E402

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import scipy.optimize  # noqa: E402
import tensorflow_probability.substrates.jax as tfp  # noqa: E402

import _hmc_lib_marg  # noqa: E402

tfd = tfp.distributions
tfe = tfp.experimental
REPRO = Path(__file__).parent
DATA = REPRO / "data"

# Huang 2025a (foundry-i) published PHYSICAL mass params (for reference printing).
PAPER_GAMMA = 1.372
PAPER_THETA_E = 2.6463

LARGE_NEG_FRAC = 1e-6  # PD floor / large-negative threshold (matches 30/32/33).


# --------------------------------------------------------------------------- #
# args
# --------------------------------------------------------------------------- #
def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["refine", "hmc"], required=True)
    ap.add_argument("--x64", dest="x64", action="store_true", default=True,
                    help="run the marginal model + sampler in float64 (DEFAULT)")
    ap.add_argument("--no-x64", dest="x64", action="store_false",
                    help="run in float32 (NOT recommended; gradient saturates)")
    # --- refine (trust-exact) flags ---
    ap.add_argument("--maxiter", type=int, default=80,
                    help="[refine] scipy trust-exact max iterations")
    ap.add_argument("--gtol", type=float, default=1e-3,
                    help="[refine] scipy gradient-norm convergence tolerance")
    # --- hmc flags ---
    ap.add_argument("--massmatrix",
                    choices=["hess_marg_pd", "diag", "diagraw", "hesscorr", "identity"],
                    default="hess_marg_pd",
                    help="[hmc] momentum COVARIANCE source (default the PD Hessian "
                         "at the refined mode). 'diagraw' = diagonal mass matrix "
                         "from the UN-FLOORED Hessian diagonal |H_raw_ii| -- "
                         "float64-safe (per-param scalars, no matrix ops), "
                         "collapses the dominant (diagonal) cond~1e14 from the "
                         "over/under-constrained lens-light nuisance params")
    ap.add_argument("--seed", type=int, default=0,
                    help="[hmc] PRNG seed (vary across parallel chains for R-hat)")
    ap.add_argument("--num-leapfrog", type=int, default=16,
                    help="[hmc] fixed leapfrog steps per HMC iteration")
    ap.add_argument("--target-accept", type=float, default=0.8, help="[hmc]")
    ap.add_argument("--step-size", type=float, default=1e-3, help="[hmc]")
    ap.add_argument("--burn", type=int, default=500, help="[hmc]")
    ap.add_argument("--keep", type=int, default=800, help="[hmc]")
    # --- shared I/O ---
    ap.add_argument("--start-file", type=str, default=None,
                    help="[hmc] npz with start vector (qz/qz_pd/qz_refined). "
                         "default data/map_marg_pd.npz")
    ap.add_argument("--mass-file", type=str, default=None,
                    help="[hmc] npz with H_reg (+chol) for the mass matrix. "
                         "default data/hess_marg_pd.npz")
    ap.add_argument("--out", type=str, default=None,
                    help="output npz. refine default data/map_marg_pd.npz "
                         "(Hessian alongside -> data/hess_marg_pd.npz); hmc default "
                         "data/marg_hmc_{massmatrix}.npz")
    return ap.parse_args()


# --------------------------------------------------------------------------- #
# shared helpers
# --------------------------------------------------------------------------- #
def eig_report(H):
    """(w, V, n_large_negative, n_negative, eig_min, eig_max) for symmetric H."""
    w, V = np.linalg.eigh(np.asarray(H, dtype=np.float64))
    eig_max = float(w.max())
    thresh = LARGE_NEG_FRAC * eig_max
    n_large_negative = int(np.sum(w < -thresh))
    n_negative = int(np.sum(w < 0.0))
    return w, V, n_large_negative, n_negative, float(w.min()), eig_max


def phys_gamma_thetaE(m, qz):
    phys = m.to_physical_mass(np.asarray(qz)[None, :])
    return float(phys["gamma"][0]), float(phys["theta_E"][0]), phys


# --------------------------------------------------------------------------- #
# refine mode: scipy trust-exact (same as 33_trust_refine.py)
# --------------------------------------------------------------------------- #
def run_refine(args, m, fdtype):
    ndim = int(m.ndim)
    x0 = np.asarray(m.qz_start, dtype=np.float64)

    def f_scalar(z):
        return -m.target_log_prob_fn(z)

    f_jit = jax.jit(f_scalar)
    grad_jit = jax.jit(jax.grad(f_scalar))
    hess_jit = jax.jit(jax.hessian(f_scalar))

    def f_np(z):
        return float(f_jit(jnp.asarray(z, dtype=jnp.float64)))

    def jac_np(z):
        return np.asarray(grad_jit(jnp.asarray(z, dtype=jnp.float64)),
                          dtype=np.float64)

    def hess_np(z):
        H = hess_jit(jnp.asarray(z, dtype=jnp.float64))
        H = 0.5 * (H + H.T)
        return np.asarray(H, dtype=np.float64)

    # gradient dtype sanity (the whole point is escaping the f32 / pinv floor).
    g0 = grad_jit(jnp.asarray(x0, dtype=jnp.float64))
    g0.block_until_ready()
    if args.x64 and str(g0.dtype) != "float64":
        raise RuntimeError(
            f"grad dtype is {g0.dtype}, not float64; float64 did not propagate "
            "through the marginal simulator.")

    logp_before = float(m.target_log_prob_fn(jnp.asarray(x0, dtype=jnp.float64)))
    grad_norm_before = float(np.linalg.norm(np.asarray(g0, dtype=np.float64)))
    g_b, tE_b, _ = phys_gamma_thetaE(m, x0)
    print(f"\nBEFORE trust-region Newton (marginal target):", flush=True)
    print(f"  log_p={logp_before:.4f}  ||grad||={grad_norm_before:.6e}", flush=True)
    print(f"  gamma={g_b:.4f}  theta_E={tE_b:.4f}", flush=True)

    iter_state = {"n": 0, "t0": time.time()}

    def callback(xk):
        iter_state["n"] += 1
        gn = float(np.linalg.norm(jac_np(xk)))
        fk = f_np(xk)
        gk, tk, _ = phys_gamma_thetaE(m, xk)
        print(f"  [trust {iter_state['n']:4d}]  f={fk:.4f}  log_p={-fk:.4f}  "
              f"||grad||={gn:.6e}  gamma={gk:.4f}  theta_E={tk:.4f}  "
              f"({time.time()-iter_state['t0']:.0f}s)", flush=True)

    print(f"\n--- scipy trust-exact: maxiter={args.maxiter}, gtol={args.gtol:g} "
          f"(46-dim smooth marginal target) ---", flush=True)
    t0 = time.time()
    res = scipy.optimize.minimize(
        f_np, x0, method="trust-exact", jac=jac_np, hess=hess_np,
        callback=callback,
        options={"maxiter": args.maxiter, "gtol": args.gtol},
    )
    wall_s = time.time() - t0
    print(f"\ntrust-exact done in {wall_s:.1f}s ({wall_s/60:.2f} min, "
          f"{res.nit} iters)", flush=True)
    print(f"  scipy status={res.status}  success={res.success}", flush=True)
    print(f"  message: {res.message}", flush=True)

    # final diagnostics at the converged point
    qz = jnp.asarray(res.x, dtype=jnp.float64)
    logp_after = float(m.target_log_prob_fn(qz))
    g_after = np.asarray(grad_jit(qz), dtype=np.float64)
    grad_norm_after = float(np.linalg.norm(g_after))
    dist_from_start = float(np.linalg.norm(np.asarray(qz) - x0))

    print(f"\n--- recomputing Hessian of -log_post at the converged point "
          f"({ndim}x{ndim}) ---", flush=True)
    th = time.time()
    H = hess_np(np.asarray(qz))
    print(f"  hessian computed in {time.time()-th:.1f}s", flush=True)
    w, V, n_large_negative, n_negative, eig_min, eig_max = eig_report(H)
    is_pd_mode = bool(n_negative == 0)

    print(f"\nAFTER trust-region Newton:", flush=True)
    print(f"  log_p={logp_after:.4f}  ||grad||={grad_norm_after:.6e}", flush=True)
    print(f"  ||grad|| reduction      : {grad_norm_before:.4e} -> "
          f"{grad_norm_after:.4e} "
          f"(x{grad_norm_before/max(grad_norm_after,1e-30):.3e})", flush=True)
    print(f"  log_p change            : {logp_before:.4f} -> {logp_after:.4f} "
          f"({logp_after-logp_before:+.4f})", flush=True)
    print(f"  ||Delta|| from start    : {dist_from_start:.6e}", flush=True)
    print(f"  eig_min={eig_min:.6e}  eig_max={eig_max:.6e}", flush=True)
    print(f"  n_negative (any <0)     : {n_negative}  (want 0 -> PD mode)", flush=True)
    print(f"  n_large_negative(<-1e-6*max): {n_large_negative}", flush=True)
    print(f"  most negative eigs      : {np.sort(w)[:8]}", flush=True)
    print(f"  smallest |eigs|         : {np.sort(np.abs(w))[:8]}", flush=True)
    print(f"\n  is_pd_mode (n_negative==0) = {is_pd_mode}", flush=True)

    gamma, theta_E, phys = phys_gamma_thetaE(m, qz)
    print(f"\nPhysical mass params at the converged point:", flush=True)
    print(f"  {'param':>10s} {'value':>12s} {'paper':>12s}", flush=True)
    print(f"  {'gamma':>10s} {gamma:>12.4f} {PAPER_GAMMA:>12.4f}", flush=True)
    print(f"  {'theta_E':>10s} {theta_E:>12.4f} {PAPER_THETA_E:>12.4f}", flush=True)
    for k in ["e1", "e2", "center_x", "center_y", "gamma1", "gamma2"]:
        if k in phys:
            print(f"  {k:>10s} {float(phys[k][0]):>12.4f}", flush=True)

    # marginal-mode shapelet amps (negative is fine; the 5 Ie are SAMPLED positive).
    amps28 = np.asarray(m.shapelet_amps(qz)).astype(np.float64)
    n_neg_shapelet = int(np.sum(amps28 < 0.0))
    print(f"\nmarginal-mode shapelet amps a* (28): n_negative={n_neg_shapelet} "
          f"(fine), range [{amps28.min():.4f}, {amps28.max():.4f}]", flush=True)
    # the 5 sampled Ie are strictly positive by the LogNormal bijector.
    n_neg_sersic_ie = 0  # LogNormal -> always positive by construction

    # PD-floored Hessian + cholesky (for the HMC mass matrix).
    floor = LARGE_NEG_FRAC * eig_max
    eig_floored = np.maximum(w, floor)
    H_reg = (V * eig_floored) @ V.T
    H_reg = 0.5 * (H_reg + H_reg.T)
    try:
        chol = np.linalg.cholesky(H_reg)
        chol_ok = True
    except np.linalg.LinAlgError:
        chol = None
        chol_ok = False
    eig_reg = np.linalg.eigvalsh(H_reg)
    cond_after = float(eig_reg.max() / max(eig_reg.min(), 1e-300))
    print(f"\n  PD-floored mass matrix: floor={floor:.6e}  "
          f"cond_after_floor={cond_after:.6e}  chol_ok={chol_ok}", flush=True)

    # ----- save (paths) ------------------------------------------------------
    DATA.mkdir(parents=True, exist_ok=True)
    if args.out is not None:
        map_path = Path(args.out)
        if not map_path.is_absolute():
            map_path = REPRO / map_path
    else:
        map_path = DATA / "map_marg_pd.npz"
    # Hessian alongside: hess_*.npz derived from the map stem.
    if "map" in map_path.stem:
        hess_path = map_path.with_name(map_path.stem.replace("map", "hess", 1)
                                       + ".npz")
    else:
        hess_path = map_path.with_name(map_path.stem + "_hess.npz")

    np.savez(
        map_path,
        qz=np.asarray(qz, dtype=np.float64),
        qz_refined=np.asarray(qz, dtype=np.float64),  # key the hmc reader expects
        qz_start=x0.astype(np.float64),
        x64=np.bool_(bool(args.x64)),
        grad=g_after.astype(np.float64),
        grad_norm_before=np.float64(grad_norm_before),
        grad_norm_after=np.float64(grad_norm_after),
        logp=np.float64(logp_after),
        logp_before=np.float64(logp_before),
        logp_after=np.float64(logp_after),
        gamma=np.float64(gamma),
        theta_E=np.float64(theta_E),
        dist_from_start=np.float64(dist_from_start),
        nit=np.int64(int(res.nit)),
        scipy_status=np.int64(int(res.status)),
        scipy_success=np.bool_(bool(res.success)),
        wall_s=np.float64(wall_s),
        eig_raw=w.astype(np.float64),
        eig_min=np.float64(eig_min),
        eig_max=np.float64(eig_max),
        n_negative=np.int64(n_negative),
        n_large_negative=np.int64(n_large_negative),
        is_pd_mode=np.bool_(is_pd_mode),
        amps_shapelet=amps28.astype(np.float64),
        n_neg_shapelet=np.int64(n_neg_shapelet),
        n_neg_sersic_ie=np.int64(n_neg_sersic_ie),
        cond_after_floor=np.float64(cond_after),
        index_labels=np.array(m.index_labels),
    )
    save_hess = dict(
        H=H.astype(np.float64),
        H_raw=H.astype(np.float64),
        H_reg=H_reg.astype(np.float64),  # key the hmc mass-matrix reader expects
        x64=np.bool_(bool(args.x64)),
        eig_raw=w.astype(np.float64),
        eig_floored=eig_floored.astype(np.float64),
        eig_min_raw=np.float64(eig_min),
        eig_max_raw=np.float64(eig_max),
        n_negative_eigs=np.int64(n_negative),
        n_large_negative=np.int64(n_large_negative),
        eig_floor=np.float64(floor),
        cond_after_floor=np.float64(cond_after),
        chol_ok=np.bool_(chol_ok),
    )
    if chol is not None:
        save_hess["chol"] = chol.astype(np.float64)
    np.savez(hess_path, **save_hess)
    print(f"\nSaved {map_path}", flush=True)
    print(f"Saved {hess_path}", flush=True)

    # ----- verdict -----------------------------------------------------------
    print(f"\n=== VERDICT (marginal refine) ===", flush=True)
    if is_pd_mode and grad_norm_after < 1e3:
        print(f"  The regularized MARGINAL target REACHED a genuine PD minimum "
              f"(n_negative=0, ||grad||={grad_norm_after:.3e}) at gamma={gamma:.4f}, "
              f"theta_E={theta_E:.4f}.", flush=True)
    else:
        print(f"  Marginal refine did NOT reach a clean PD minimum "
              f"(||grad||={grad_norm_after:.3e}, n_negative={n_negative}); "
              f"gamma={gamma:.4f}, theta_E={theta_E:.4f}.", flush=True)
    print("Done.", flush=True)

    return dict(
        grad_final=grad_norm_after, n_negative=n_negative, is_pd_mode=is_pd_mode,
        gamma=gamma, theta_E=theta_E, logp=logp_after, wall_s=wall_s,
        map_path=str(map_path), hess_path=str(hess_path),
    )


# --------------------------------------------------------------------------- #
# hmc mode: PHMC fixed-leapfrog + DualAveraging (same as 31_fit_lstsq.py)
# --------------------------------------------------------------------------- #
def build_momentum(massmatrix, ndim, fdtype, hess_path):
    """Momentum distribution over R^ndim.  PHMC optimal mass matrix M = Sigma^-1
    ~= H, momentum p ~ N(0, M), so the momentum COVARIANCE = the PD Hessian H_reg."""
    if massmatrix == "hess_marg_pd":
        d = np.load(hess_path)
        H_reg = jnp.asarray(d["H_reg"], dtype=fdtype)
        chol = (jnp.asarray(d["chol"], dtype=fdtype) if "chol" in d
                else jnp.linalg.cholesky(H_reg))
        return tfd.MultivariateNormalTriL(
            loc=jnp.zeros(ndim, dtype=fdtype), scale_tril=chol)
    if massmatrix == "diag":
        d = np.load(hess_path)
        H_reg = np.asarray(d["H_reg"], dtype=np.float64)
        scale_diag = jnp.asarray(np.sqrt(np.maximum(np.diag(H_reg), 1e-30)),
                                 dtype=fdtype)
        return tfd.MultivariateNormalDiag(
            loc=jnp.zeros(ndim, dtype=fdtype), scale_diag=scale_diag)
    if massmatrix == "diagraw":
        # Diagonal mass matrix from the UN-FLOORED Hessian diagonal. The dominant
        # conditioning (cond~1e14: lens-light centers ~1e12 vs Sersic indices ~1e-2)
        # is DIAGONAL, so M = diag(|H_raw_ii|) rescales each param to its true
        # marginal width. This is float64-SAFE: per-param scalars only (sqrt, 1/x),
        # NO eigendecomposition/cholesky/inverse of the ill-conditioned matrix, so
        # the roundoff that crippled the full-matrix preconditioners never occurs.
        d = np.load(hess_path)
        H_raw = np.asarray(d["H_raw"], dtype=np.float64)
        diag = np.abs(np.diag(H_raw))
        diag = np.maximum(diag, 1.0)  # tiny positive floor for any ~0 nuisance dir
        scale_diag = jnp.asarray(np.sqrt(diag), dtype=fdtype)
        print(f"  diagraw mass: diag(|H_raw|) in [{diag.min():.3e}, {diag.max():.3e}], "
              f"cond={diag.max()/diag.min():.3e}", flush=True)
        return tfd.MultivariateNormalDiag(
            loc=jnp.zeros(ndim, dtype=fdtype), scale_diag=scale_diag)
    if massmatrix == "hesscorr":
        # Full-Hessian mass matrix (scales AND correlations) built float64-SAFELY
        # via the diagonally-scaled correlation matrix. The dominant cond~1e14 is
        # diagonal; factor it out first so the only Cholesky touches a
        # well-conditioned matrix:
        #   D = diag(|H_ii|);  C = D^-1/2 H D^-1/2 (unit-ish diag, MILD cond = just
        #   the correlations);  chol(H) = diag(sqrt(D)) @ chol(C).
        # momentum cov = chol_H @ chol_H^T = D^1/2 C D^1/2 = H  => the proper
        # Laplace mass matrix M=Sigma^-1, capturing the lensing degeneracies that
        # diagonal preconditioning ('diagraw') leaves uncorrected.
        d = np.load(hess_path)
        H = np.asarray(d["H_raw"], dtype=np.float64)
        H = 0.5 * (H + H.T)
        Dh = np.sqrt(np.maximum(np.abs(np.diag(H)), 1.0))
        C = H / np.outer(Dh, Dh)
        C = 0.5 * (C + C.T)
        wC = np.linalg.eigvalsh(C)
        ridge = max(0.0, -wC.min()) + 1e-8 * max(wC.max(), 1.0)
        C = C + ridge * np.eye(C.shape[0])
        Lc = np.linalg.cholesky(C)
        chol_H = (Dh[:, None] * Lc).astype(np.float64)
        condC = float((wC.max() + ridge) / (wC.min() + ridge))
        print(f"  hesscorr mass: corr-matrix C cond={condC:.3e} (vs raw H cond~1e14), "
              f"ridge={ridge:.3e}, chol(H) via diag(sqrt(D))@chol(C)", flush=True)
        return tfd.MultivariateNormalTriL(
            loc=jnp.zeros(ndim, dtype=fdtype),
            scale_tril=jnp.asarray(chol_H, dtype=fdtype))
    return tfd.MultivariateNormalDiag(
        loc=jnp.zeros(ndim, dtype=fdtype), scale_diag=jnp.ones(ndim, dtype=fdtype))


def run_hmc(args, m, fdtype):
    start_path = (Path(args.start_file) if args.start_file
                  else DATA / "map_marg_pd.npz")
    if not start_path.is_absolute():
        start_path = REPRO / start_path
    mass_path = (Path(args.mass_file) if args.mass_file
                 else DATA / "hess_marg_pd.npz")
    if not mass_path.is_absolute():
        mass_path = REPRO / mass_path
    out_path = (Path(args.out) if args.out is not None
                else DATA / f"marg_hmc_{args.massmatrix}.npz")
    if not out_path.is_absolute():
        out_path = REPRO / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rd = np.load(start_path, allow_pickle=True)
    if "qz_pd" in rd:
        start = jnp.asarray(rd["qz_pd"], dtype=fdtype)
    elif "qz_refined" in rd:
        start = jnp.asarray(rd["qz_refined"], dtype=fdtype)
    else:
        start = jnp.asarray(rd["qz"], dtype=fdtype)
    gn = float(rd["grad_norm_after"]) if "grad_norm_after" in rd else float("nan")
    is_pd = bool(rd["is_pd_mode"]) if "is_pd_mode" in rd else False
    print(f"start = refined PD mode ({start_path})  saved ||grad||={gn:.4e}  "
          f"is_pd_mode={is_pd}", flush=True)
    print(f"mass-matrix source = {mass_path}", flush=True)

    momentum = build_momentum(args.massmatrix, m.ndim, fdtype, mass_path)
    lp_start = m.target_log_prob_fn(start)
    lp_start.block_until_ready()
    print(f"\nconfig:", flush=True)
    print(f"  num_leapfrog_steps   = {args.num_leapfrog}", flush=True)
    print(f"  target_accept_prob   = {args.target_accept}", flush=True)
    print(f"  init step_size       = {args.step_size}", flush=True)
    print(f"  num_burnin / kept    = {args.burn} / {args.keep}", flush=True)
    print(f"  num_adaptation_steps = {int(0.8 * args.burn)} (step-size only)",
          flush=True)
    print(f"  massmatrix           = {args.massmatrix}", flush=True)
    print(f"  out                  = {out_path}", flush=True)
    print(f"  ndim={m.ndim}  log_p(start)={float(lp_start):.2f}", flush=True)

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

    print(f"\nRunning hmc sample_chain (compile + run) ...", flush=True)
    t0 = time.time()
    samples, trace = tfp.mcmc.sample_chain(
        num_results=args.keep,
        num_burnin_steps=args.burn,
        current_state=start,
        kernel=adapted,
        trace_fn=trace_fn,
        seed=jax.random.PRNGKey(args.seed),
    )
    samples.block_until_ready()
    wall_s = time.time() - t0
    print(f"hmc done in {wall_s:.1f}s ({wall_s/60:.2f} min)", flush=True)

    samples_np = np.asarray(samples)
    is_acc = np.asarray(trace["is_accepted"])
    lps = np.asarray(trace["target_log_prob"])
    ss = np.asarray(trace["step_size"])
    accept_ratio = (np.asarray(trace["accept_ratio"])
                    if "accept_ratio" in trace else None)
    diff = np.linalg.norm(np.diff(samples_np, axis=0), axis=1)
    per_param_std = samples_np.std(axis=0)
    final_ss = float(ss[-1])
    step_collapsed = final_ss < 1e-5
    any_nan = bool(np.any(np.isnan(samples_np)))

    print(f"\nDiagnostics over {args.keep} kept samples:", flush=True)
    print(f"  Acceptance rate       : {float(is_acc.mean()):.3f}", flush=True)
    if accept_ratio is not None:
        print(f"  mean per-step accept  : {float(np.nanmean(accept_ratio)):.3f}",
              flush=True)
    print(f"  target_log_prob       : min={lps.min():.1f}, "
          f"median={np.median(lps):.1f}, max={lps.max():.1f}", flush=True)
    print(f"  step_size             : start={float(ss[0]):.5g}, final={final_ss:.5g}",
          flush=True)
    print(f"  step_collapsed (<1e-5): {step_collapsed}", flush=True)
    print(f"  ||dz|| step-to-step   : median={np.median(diff):.4f}, "
          f"max={diff.max():.4f}", flush=True)
    print(f"  per-param sample std  : min={per_param_std.min():.4e}, "
          f"max={per_param_std.max():.4e}", flush=True)
    print(f"  any NaN               : {any_nan}", flush=True)

    combined = m.to_physical_mass(samples_np)
    ess = {}
    print(f"\n  ESS on 6 physical mass params:", flush=True)
    for k in ["theta_E", "gamma", "e1", "e2", "gamma1", "gamma2"]:
        if k not in combined:
            continue
        ess_k = float(np.asarray(
            tfp.mcmc.effective_sample_size(jnp.asarray(combined[k]))))
        ess[k] = ess_k
        print(f"    {k:>10s}: ESS={ess_k:8.1f}", flush=True)
    ess_min = float(min(ess.values())) if ess else float("nan")
    print(f"    {'ESS_min':>10s}: {ess_min:8.1f}", flush=True)

    gamma_std = float(np.std(combined["gamma"])) if "gamma" in combined else float("nan")
    theta_E_median = (float(np.median(combined["theta_E"]))
                      if "theta_E" in combined else float("nan"))
    gamma_median = (float(np.median(combined["gamma"]))
                    if "gamma" in combined else float("nan"))
    print(f"\n  gamma posterior std   : {gamma_std:.6e}  (paper 0.023)", flush=True)
    print(f"  gamma median          : {gamma_median:.6f}  (paper {PAPER_GAMMA})",
          flush=True)
    print(f"  theta_E median        : {theta_E_median:.6f}  (paper {PAPER_THETA_E})",
          flush=True)

    save_kw = dict(
        samples=samples_np, is_accepted=is_acc, target_log_prob=lps, step_size=ss,
        x64=np.bool_(bool(args.x64)), kernel="hmc", massmatrix=args.massmatrix,
        num_leapfrog=args.num_leapfrog, target_accept=args.target_accept,
        init_step_size=args.step_size, adapted_step_size=final_ss,
        step_collapsed=step_collapsed, median_dz=float(np.median(diff)),
        gamma_std=gamma_std, gamma_median=gamma_median,
        theta_E_median=theta_E_median, ess_min=ess_min, any_nan=any_nan,
        elapsed=wall_s, burn=args.burn, keep=args.keep,
        initial_state=np.asarray(start), ess_keys=np.array(list(ess.keys())),
        ess_vals=np.array(list(ess.values()), dtype=np.float64),
    )
    if accept_ratio is not None:
        save_kw["accept_ratio"] = accept_ratio
    for k, v in combined.items():
        save_kw[f"mass_{k}"] = np.asarray(v)
    np.savez(out_path, **save_kw)
    print(f"\nSaved {out_path}", flush=True)
    print("Done.", flush=True)

    return dict(
        grad_final=float("nan"), n_negative=-1, is_pd_mode=is_pd,
        gamma=gamma_median, theta_E=theta_E_median,
        logp=float(np.median(lps)), wall_s=wall_s, out_path=str(out_path),
    )


def main():
    args = parse_args()
    fdtype = jnp.float64 if args.x64 else jnp.float32
    print(f"devices: {jax.devices()}", flush=True)
    print(f"mode={args.mode}  x64={args.x64}  "
          f"jax_enable_x64={jax.config.jax_enable_x64}", flush=True)

    m = _hmc_lib_marg.build_model_marg(x64=args.x64)
    print(f"ndim={m.ndim}  (expect 46 = 41 nonlinear + 5 Sersic Ie)", flush=True)
    print(f"design_matrix_source: {m.design_matrix_source[:120]}...", flush=True)
    if m.missing_labels:
        print(f"WARNING missing_labels: {m.missing_labels}", flush=True)

    if args.mode == "refine":
        run_refine(args, m, fdtype)
    else:
        run_hmc(args, m, fdtype)


if __name__ == "__main__":
    main()
