"""Stage 5 saddle-free Newton: escape the reduced-model SADDLE to a true PD mode.

30_refine_lstsq.py drove the 41-dim lstsq-reduced objective f(z) = -log_post(z)
from the projected full-model MAP to data/map_refined_lstsq64.npz, but that point
is STILL A SADDLE: its reduced Hessian has 2 LARGE-negative eigenvalues
(eig_min ~ -8.4e7 vs eig_max ~ 2.2e9) and ||grad|| ~ 9.2e4. First-order optax (and
even damped Newton / Levenberg-Marquardt, which only ADDS lambda*I to push H toward
PD) cannot follow a negative-curvature direction to LEAVE a saddle: along a
negative-curvature eigendirection the LM step (H+lambda*I)^-1 g points UPHILL once
lambda dominates -|lambda_neg|, so the iterate just sits on the ridge.

SADDLE-FREE NEWTON (Dauphin et al. 2014, "Identifying and attacking the saddle
point problem in high-dimensional non-convex optimization"): replace H by its
"absolute" curvature |H| = V diag(|lambda_i|) V^T.  The step

    step = (V diag(1/(|lambda_i| + damping)) V^T) grad        x <- x - alpha*step

DESCENDS along every eigendirection: along a positive-curvature direction it is the
usual Newton step (toward the minimum), and along a NEGATIVE-curvature direction it
flips the sign (|lambda| > 0) so the step moves DOWNHILL away from the saddle ridge
instead of uphill toward it.  This is the only second-order step that can escape a
saddle, which is exactly what the LM polish in 30 could not do.

This requires float64: the reduced objective has cond ~1e10, which exceeds float32's
~7 digits (the f32 gradient saturates at a ~1.2e4 noise floor); float64's ~16 digits
fit cond ~1e10, so the eigendecomposition and step are meaningful.

Algorithm (per iter, up to --max-iters, default 60):
  g = grad f(x);  H = hessian f(x);  H = (H+H^T)/2;  w,V = eigh(H)
  damping  = max(DAMP_FRAC * |w|.max(), DAMP_MIN)
  inv      = V diag(1/(|w|+damping)) V^T
  step     = inv @ g
  trust-region: if ||step|| > TRUST cap it to TRUST (tiny-curvature dirs can blow up)
  backtracking Armijo line search over alpha in {1,1/2,1/4,...,~1e-4}:
      accept the LARGEST alpha with  f(x - alpha*step) < f(x) - C1*alpha*(g.step)
      (g.step > 0 since step is along +grad through |H|^{-1}, so this is a descent
       direction; fall back to plain decrease f(x-alpha*step) < f(x) if needed)
  x <- x - alpha*step
Stop when ||g|| is small AND n_large_negative == 0 (a genuine PD mode), else iters
exhausted.

Outputs (float64):
  data/map_pd_lstsq64.npz  : qz_pd, grad, logp, gamma, theta_E (+ before/after,
                             eig spectrum, n_large_negative, is_pd_mode, iters).
  data/hess_pd_lstsq64.npz : H (raw symmetrized), H_reg (PD-floored), chol(H_reg).

Run (one idle A16; A16 FP64 ~0.39 s/grad so a 41-dim hessian ~16 s, 60 iters ~20 min):
  CUDA_DEVICE_ORDER=PCI_BUS_ID XLA_FLAGS=--xla_gpu_autotune_level=0 \
    CUDA_VISIBLE_DEVICES=1 /raid/benson/.venvs/gigalens/bin/python 32_saddlefree_newton.py

Then 31_fit_lstsq.py --pd starts an HMC from this PD mode with H as the mass matrix
(this script does NOT run a chain).
"""
import argparse
import os

# x64 must enable jax_enable_x64 BEFORE _hmc_lib_lstsq is imported (it sets x64 at
# its own import from GIGALENS_X64, before any jnp array). This script is ALWAYS
# float64 -- the saddle-free eigendecomposition of a cond~1e10 Hessian needs it.
os.environ["GIGALENS_X64"] = "1"

import jax  # noqa: E402

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

import _hmc_lib_lstsq  # noqa: E402

REPRO = Path(__file__).parent
DATA = REPRO / "data"

# ---- tunables ---------------------------------------------------------------
# damping is kept SMALL so the LARGE-negative-curvature directions (this saddle's
# negatives reach |lambda| ~ 1e8) still receive a meaningful escape step
# |g_d|/(|lambda_d|+damping); a large damping would shrink the very steps needed to
# leave the saddle.  Runaway of the ~20 near-flat (|lambda|~1) directions is
# bounded instead by (a) a cap on the per-direction inverse curvature (--inv-cap)
# and (b) the trust-region cap on ||step|| (--trust).
DAMP_FRAC = 1e-5        # damping = max(DAMP_FRAC*|w|.max(), DAMP_MIN)
DAMP_MIN = 1e-3
TRUST = 1.0             # cap ||step|| so tiny-|lambda| directions can't explode
INV_CAP = 1.0           # cap 1/(|lambda|+damping) so flat dirs take a bounded step
ARMIJO_C1 = 1e-4        # sufficient-decrease constant
ALPHAS = [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 1e-2, 1e-3, 1e-4]
GRAD_TOL = 1e3          # "small" gradient relative to the f64-achievable floor
# A large-negative eig is one below -LARGE_NEG_FRAC * eig_max (the same 1e-6
# threshold 30_refine_lstsq.py uses to count n_large_negative; magnitudes below it
# are float64 curvature noise, not a genuine descent direction).
LARGE_NEG_FRAC = 1e-6


def eig_report(H):
    """(w, V, n_large_negative, n_negative, eig_min, eig_max) for symmetric H."""
    w, V = np.linalg.eigh(np.asarray(H, dtype=np.float64))
    eig_max = float(w.max())
    thresh = LARGE_NEG_FRAC * eig_max
    n_large_negative = int(np.sum(w < -thresh))
    n_negative = int(np.sum(w < 0.0))
    return w, V, n_large_negative, n_negative, float(w.min()), eig_max


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-iters", type=int, default=60)
    ap.add_argument("--trust", type=float, default=TRUST)
    ap.add_argument("--damp-frac", type=float, default=DAMP_FRAC)
    ap.add_argument("--inv-cap", type=float, default=INV_CAP,
                    help="cap on 1/(|lambda|+damping) so near-flat directions take "
                         "a bounded (not exploding) step")
    args = ap.parse_args()

    print(f"devices: {jax.devices()}", flush=True)
    print(f"jax_enable_x64={jax.config.jax_enable_x64}", flush=True)

    m = _hmc_lib_lstsq.build_model_lstsq(x64=True)
    ndim = int(m.ndim)

    rd = np.load(DATA / "map_refined_lstsq64.npz", allow_pickle=True)
    x = jnp.asarray(rd["qz_refined"], dtype=jnp.float64)
    print(f"\nstart = data/map_refined_lstsq64.npz['qz_refined']  ndim={ndim}", flush=True)
    print(f"  saved grad_norm_after={float(rd['grad_norm_after']):.4e}  "
          f"logp_after={float(rd['logp_after']):.4f}  "
          f"n_large_negative={int(rd['n_large_negative'])}  "
          f"is_pd_mode={bool(rd['is_pd_mode'])}", flush=True)

    # f(z) = -log_post(z); MINIMIZE f.  Saddle-free Newton escapes the saddle.
    def f(z):
        return -m.target_log_prob_fn(z)

    f_jit = jax.jit(f)
    grad_jit = jax.jit(jax.grad(f))
    hess_jit = jax.jit(jax.hessian(f))

    # ---- gradient dtype sanity (the whole point is escaping the f32 floor) ----
    g0 = grad_jit(x)
    g0.block_until_ready()
    if str(g0.dtype) != "float64":
        raise RuntimeError(
            f"grad dtype is {g0.dtype}, not float64; float64 did not propagate "
            "through the simulator. Saddle-free Newton on a cond~1e10 Hessian is "
            "meaningless in float32.")

    logp_before = float(m.target_log_prob_fn(x))
    grad_norm_before = float(jnp.linalg.norm(g0))
    H0 = hess_jit(x)
    H0 = 0.5 * (H0 + H0.T)
    _, _, nlneg0, nneg0, emin0, emax0 = eig_report(H0)
    print(f"\nBEFORE saddle-free Newton:", flush=True)
    print(f"  log_p={logp_before:.4f}  ||grad||={grad_norm_before:.6e}", flush=True)
    print(f"  eig_min={emin0:.4e}  eig_max={emax0:.4e}  "
          f"n_large_negative={nlneg0}  n_negative={nneg0}", flush=True)

    # ---- saddle-free Newton iterations ----------------------------------------
    eye_unused = None  # keep numpy-only linear algebra below
    fx = logp_before * -1.0  # f(x) = -log_p
    converged = False
    iters_run = 0
    t_loop = time.time()
    print(f"\n--- saddle-free Newton: up to {args.max_iters} iters "
          f"(trust={args.trust}, damp_frac={args.damp_frac}) ---", flush=True)
    for it in range(args.max_iters):
        iters_run = it + 1
        g = np.asarray(grad_jit(x), dtype=np.float64)
        gn = float(np.linalg.norm(g))
        H = hess_jit(x)
        H = 0.5 * (H + H.T)
        w, V, nlneg, nneg, emin, emax = eig_report(H)

        # already a PD mode (small grad AND no large-negative curvature)?
        if gn < GRAD_TOL and nlneg == 0:
            print(f"  [it {it:3d}] CONVERGED: ||grad||={gn:.4e} < {GRAD_TOL:g} AND "
                  f"n_large_negative=0 -> PD mode.", flush=True)
            converged = True
            break

        # saddle-free step: rescale by |lambda| (negative curvature flipped to a
        # positive magnitude) so the step descends along ALL eigendirections.
        damping = max(args.damp_frac * float(np.abs(w).max()), DAMP_MIN)
        inv_diag = 1.0 / (np.abs(w) + damping)
        # cap the inverse curvature so the ~20 near-flat (|lambda|~1) directions
        # take a BOUNDED step rather than a 1/damping ~ 1e3 sized one (which the
        # trust region would then absorb entirely, starving the escape directions).
        inv_diag = np.minimum(inv_diag, args.inv_cap)
        # step = V diag(inv_diag) V^T g  (compute via (V*inv_diag) @ (V^T g))
        step = (V * inv_diag) @ (V.T @ g)
        step_norm = float(np.linalg.norm(step))
        # trust-region cap
        if step_norm > args.trust:
            step = step * (args.trust / step_norm)
            step_norm_capped = args.trust
        else:
            step_norm_capped = step_norm

        # directional derivative along -step: g.step (>0 => descent for x-alpha*step)
        g_dot_step = float(g @ step)

        # backtracking Armijo line search over decreasing alpha.
        x_np = np.asarray(x, dtype=np.float64)
        step_j = jnp.asarray(step, dtype=jnp.float64)
        accepted_alpha = 0.0
        f_new_best = fx
        for alpha in ALPHAS:
            x_try = jnp.asarray(x_np - alpha * step, dtype=jnp.float64)
            f_try = float(f_jit(x_try))
            if not np.isfinite(f_try):
                continue
            armijo = fx - ARMIJO_C1 * alpha * g_dot_step
            # accept on Armijo sufficient decrease, else fall back to plain decrease.
            if f_try < armijo or f_try < fx:
                accepted_alpha = alpha
                f_new_best = f_try
                break

        if accepted_alpha > 0.0:
            x = jnp.asarray(x_np - accepted_alpha * step, dtype=jnp.float64)
            fx = f_new_best
        # if no alpha decreased f, x is unchanged this iter (step too aligned with a
        # tiny-curvature direction); the next iter's Hessian re-eval may still help,
        # but report it.

        print(f"  [it {it:3d}] ||grad||={gn:.4e}  log_p={-fx:.4f}  "
              f"n_large_neg={nlneg}  n_neg={nneg}  eig_min={emin:.3e}  "
              f"damp={damping:.3e}  ||step||={step_norm_capped:.4e}  "
              f"alpha={accepted_alpha:g}  g.step={g_dot_step:+.3e}", flush=True)

        # If the line search found NO alpha that decreases f, the iterate is frozen
        # (re-evaluating the same Hessian at the same x gives the same rejected
        # step), so further iters are no-ops -- stop and report this point.
        if accepted_alpha == 0.0:
            print(f"  [it {it:3d}] no alpha decreased f (line search exhausted); "
                  f"iterate frozen -> stopping.", flush=True)
            break

    print(f"\nsaddle-free Newton: {iters_run} iters in "
          f"{time.time()-t_loop:.1f}s  converged={converged}", flush=True)

    # ---- final diagnostics at x ------------------------------------------------
    qz_pd = jnp.asarray(x, dtype=jnp.float64)
    logp_after = float(m.target_log_prob_fn(qz_pd))
    g_after = np.asarray(grad_jit(qz_pd), dtype=np.float64)
    grad_norm_after = float(np.linalg.norm(g_after))
    H = hess_jit(qz_pd)
    H = np.asarray(0.5 * (H + H.T), dtype=np.float64)
    w, V, n_large_negative, n_negative, eig_min, eig_max = eig_report(H)
    dist_from_start = float(np.linalg.norm(np.asarray(qz_pd) - np.asarray(rd["qz_refined"])))

    # is_pd_mode: small gradient AND no large-negative curvature (a genuine PD mode).
    is_pd_mode = bool(grad_norm_after < GRAD_TOL and n_large_negative == 0)

    print(f"\nAFTER saddle-free Newton:", flush=True)
    print(f"  log_p={logp_after:.4f}  ||grad||={grad_norm_after:.6e}", flush=True)
    print(f"  ||grad|| reduction      : {grad_norm_before:.4e} -> {grad_norm_after:.4e} "
          f"(x{grad_norm_before/max(grad_norm_after,1e-30):.3e})", flush=True)
    print(f"  log_p change            : {logp_before:.4f} -> {logp_after:.4f} "
          f"({logp_after-logp_before:+.4f})", flush=True)
    print(f"  ||Delta|| from start    : {dist_from_start:.6e}", flush=True)
    print(f"  eig_min={eig_min:.6e}  eig_max={eig_max:.6e}", flush=True)
    print(f"  n_large_negative BEFORE : {nlneg0}  -> AFTER : {n_large_negative} "
          f"(want 0 for a true mode)", flush=True)
    print(f"  n_negative (any <0)     : {nneg0} -> {n_negative}", flush=True)
    print(f"  most negative eigs      : {np.sort(w)[:8]}", flush=True)
    print(f"  smallest |eigs|         : {np.sort(np.abs(w))[:8]}", flush=True)
    print(f"\n  is_pd_mode (grad<{GRAD_TOL:g} & n_large_negative==0) = {is_pd_mode}",
          flush=True)

    # ---- physics at the PD mode ------------------------------------------------
    phys = m.to_physical_mass(np.asarray(qz_pd)[None, :])
    gamma = float(phys["gamma"][0])
    theta_E = float(phys["theta_E"][0])
    print(f"\nPhysical mass params at the PD mode:", flush=True)
    print(f"  {'param':>10s} {'PD mode':>12s} {'paper':>12s}", flush=True)
    print(f"  {'gamma':>10s} {gamma:>12.4f} {1.372:>12.4f}", flush=True)
    print(f"  {'theta_E':>10s} {theta_E:>12.4f} {2.6463:>12.4f}", flush=True)
    for k in ["e1", "e2", "gamma1", "gamma2"]:
        if k in phys:
            print(f"  {k:>10s} {float(phys[k][0]):>12.4f}", flush=True)

    # ---- PD-floored Hessian + cholesky (for the HMC mass matrix) --------------
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

    # ---- save ------------------------------------------------------------------
    DATA.mkdir(parents=True, exist_ok=True)
    map_path = DATA / "map_pd_lstsq64.npz"
    hess_path = DATA / "hess_pd_lstsq64.npz"
    np.savez(
        map_path,
        qz_pd=np.asarray(qz_pd, dtype=np.float64),
        qz_start=np.asarray(rd["qz_refined"], dtype=np.float64),
        x64=np.bool_(True),
        grad=g_after.astype(np.float64),
        grad_norm_before=np.float64(grad_norm_before),
        grad_norm_after=np.float64(grad_norm_after),
        logp=np.float64(logp_after),
        logp_before=np.float64(logp_before),
        logp_after=np.float64(logp_after),
        gamma=np.float64(gamma),
        theta_E=np.float64(theta_E),
        dist_from_start=np.float64(dist_from_start),
        iters_run=np.int64(iters_run),
        converged=np.bool_(converged),
        eig_raw=w.astype(np.float64),
        eig_min=np.float64(eig_min),
        eig_max=np.float64(eig_max),
        n_large_negative_before=np.int64(nlneg0),
        n_large_negative=np.int64(n_large_negative),
        n_negative=np.int64(n_negative),
        is_pd_mode=np.bool_(is_pd_mode),
        cond_after_floor=np.float64(cond_after),
        reduced_index_labels=np.array(m.reduced_index_labels),
    )
    save_hess = dict(
        H=H.astype(np.float64),
        H_raw=H.astype(np.float64),
        # Alias H_reg under the name 31_fit_lstsq.py's build_momentum reads.
        H_reg=H_reg.astype(np.float64),
        x64=np.bool_(True),
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

    # ---- verdict ---------------------------------------------------------------
    print(f"\n=== VERDICT ===", flush=True)
    if is_pd_mode:
        print("  Saddle-free Newton REACHED a genuine PD mode "
              "(small gradient AND no large-negative Hessian eigenvalues).", flush=True)
    else:
        print("  Saddle-free Newton did NOT reach a genuine PD mode "
              f"(grad_norm={grad_norm_after:.3e}, n_large_negative={n_large_negative}).",
              flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
