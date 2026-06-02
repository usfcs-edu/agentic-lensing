"""Stage 5 trust-region Newton: reach a PD mode near the paper's gamma~1.37.

The 41-dim lstsq-reduced objective f(z) = -log_post(z) is EXTREMELY stiff
(reduced-Hessian cond ~1e9, eig_max ~2.2e9).  data/map_refined_lstsq64.npz sits at
gamma=1.866, theta_E=2.598 with ||grad||~9.2e4 -- NOT a saddle in the
"flat-direction" sense, but an UNCONVERGED point whose gradient is dominated by
stiff eigendirections.  First-order optax (30_refine_lstsq.py) and additive-damped
Newton / Levenberg-Marquardt (the LM polish in 30, and even the saddle-free Newton
in 32) STALL: along an indefinite/stiff spectrum an ADDITIVE damping (H+lambda*I)
must grow lambda to ~eig_max to stay descending, which collapses the step.

FIX: scipy.optimize.minimize(method='trust-exact') -- a trust-region Newton that
solves the trust-region subproblem EXACTLY (the More-Sorensen secular-equation
solve).  It (a) handles INDEFINITE Hessians (escapes saddles by stepping along
negative-curvature directions inside the trust radius) and (b) handles STIFFNESS
(the exact subproblem solve rescales each eigendirection correctly instead of a
single global lambda).  It needs the FULL Hessian; jax.hessian supplies it in
float64 (~55 s once, fine for the few trust-region iters).

SEED: --seed paper OVERRIDES the 6 EPL mass params (theta_E, gamma, e1, e2,
center_x, center_y) + 2 shear params (gamma1, gamma2) with Huang 2025a's published
PHYSICAL values, mapped into unconstrained z via the model's constraining bijector
(prob_model.bij forward at the base -> override the structured mass block ->
bij.inverse).  The bijector is blockwise, so ONLY z0..z7 change; z8..z40 stay from
the base (data/map_refined_lstsq64.npz['qz_refined']).  This targets the right
basin (the current point's gamma=1.866 is the wrong basin).  --seed current uses
the base unchanged.

Outputs (float64):
  data/map_trust_{seed}.npz  : qz, grad, logp, gamma, theta_E (+ before/after,
                               eig spectrum, n_negative, is_pd_mode, n_neg_sersic_ie).
  data/hess_trust_{seed}.npz : H (raw symmetrized), H_reg (PD-floored), chol(H_reg).

Then 31_fit_lstsq.py --start-file data/map_trust_{seed}.npz \
        --mass-file data/hess_trust_{seed}.npz  runs an HMC from this PD mode.

Run (one idle A16; A16 FP64 ~0.39 s/grad, one jax.hessian ~55 s, few trust iters):
  CUDA_DEVICE_ORDER=PCI_BUS_ID XLA_FLAGS=--xla_gpu_autotune_level=0 \
    CUDA_VISIBLE_DEVICES=1 /raid/benson/.venvs/gigalens/bin/python 33_trust_refine.py \
      --seed paper --maxiter 60 --out data/map_trust_paper.npz

Smoke (2 iters, current seed):
  CUDA_DEVICE_ORDER=PCI_BUS_ID XLA_FLAGS=--xla_gpu_autotune_level=0 \
    CUDA_VISIBLE_DEVICES=1 /raid/benson/.venvs/gigalens/bin/python 33_trust_refine.py \
      --seed current --maxiter 2 --out data/trust_smoke.npz
"""
import argparse
import os

# This script is ALWAYS float64: a trust-region Newton on a cond~1e9 Hessian needs
# float64 (cond~1e9 exceeds float32's ~7 digits; the f32 gradient saturates at a
# ~1.2e4 noise floor).  Enable x64 BEFORE _hmc_lib_lstsq is imported (it sets x64
# at its own import from the GIGALENS_X64 env var, before any jnp array is created).
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
import scipy.optimize  # noqa: E402

import _hmc_lib_lstsq  # noqa: E402

REPRO = Path(__file__).parent
DATA = REPRO / "data"

# Huang 2025a (foundry-i) published PHYSICAL mass params.
PAPER_MASS = dict(
    theta_E=2.6463, gamma=1.372, e1=0.1091, e2=-0.1320,
    center_x=0.0, center_y=0.0,
)
PAPER_SHEAR = dict(gamma1=0.0657, gamma2=-0.0939)

LARGE_NEG_FRAC = 1e-6  # PD floor / large-negative threshold (matches 30/32).


def build_paper_seed(m, base):
    """Map the paper PHYSICAL mass params into unconstrained z, blockwise.

    Forward the base z (41-dim) through the model's constraining bijector
    prob_model.bij to get the structured event-space point x; OVERRIDE the EPL
    (theta_E, gamma, e1, e2, center_x, center_y) and shear (gamma1, gamma2)
    physical values with the paper values; inverse the whole structured point
    back to a flat 41-vector z.  Because the bijector acts per-parameter, only
    z0..z7 (the mass block) change; z8..z40 round-trip from the base unchanged.

    Returns (z_paper float64 ndarray, changed_indices ndarray).
    """
    bij = m.prob_model.bij
    z = jnp.asarray(np.asarray(base, dtype=np.float64)[:, None], dtype=jnp.float64)
    x = bij.forward(list(z))

    mass_main = dict(x[0][0])
    mass_shear = dict(x[0][1])
    for k, v in PAPER_MASS.items():
        mass_main[k] = jnp.asarray([v], dtype=jnp.float64)
    for k, v in PAPER_SHEAR.items():
        mass_shear[k] = jnp.asarray([v], dtype=jnp.float64)
    x_new = [[mass_main, mass_shear], x[1], x[2]]

    z_new = np.array(
        [float(np.asarray(v).squeeze()) for v in bij.inverse(x_new)],
        dtype=np.float64)
    if not np.all(np.isfinite(z_new)):
        raise RuntimeError(
            "paper-seed bijector inverse produced non-finite z; a paper mass "
            "value is outside its prior support (e.g. gamma outside [1.0, 2.7]).")
    changed = np.where(np.abs(z_new - np.asarray(base, dtype=np.float64)) > 1e-9)[0]
    return z_new, changed


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
    ap.add_argument("--seed", choices=["paper", "current"], default="current",
                    help="paper: override mass block with Huang 2025a physical "
                         "values (right basin, gamma~1.37); current: use the "
                         "refined-MAP base unchanged.")
    ap.add_argument("--maxiter", type=int, default=60,
                    help="scipy trust-exact max iterations.")
    ap.add_argument("--gtol", type=float, default=1e-2,
                    help="scipy gradient-norm convergence tolerance.")
    ap.add_argument("--out", type=str, default=None,
                    help="output npz for the refined point (map_trust_{seed}.npz "
                         "by default); the Hessian goes to hess_trust_{seed}.npz "
                         "alongside it.")
    args = ap.parse_args()

    print(f"devices: {jax.devices()}", flush=True)
    print(f"jax_enable_x64={jax.config.jax_enable_x64}", flush=True)

    m = _hmc_lib_lstsq.build_model_lstsq(x64=True)
    ndim = int(m.ndim)

    # ----- base start vector -------------------------------------------------
    rd = np.load(DATA / "map_refined_lstsq64.npz", allow_pickle=True)
    base = np.asarray(rd["qz_refined"], dtype=np.float64)
    print(f"\nbase = data/map_refined_lstsq64.npz['qz_refined']  ndim={ndim}", flush=True)
    print(f"  saved grad_norm_after={float(rd['grad_norm_after']):.4e}  "
          f"logp_after={float(rd['logp_after']):.4f}  "
          f"gamma={float(rd['gamma_refined']):.4f}  "
          f"theta_E={float(rd['theta_E_refined']):.4f}", flush=True)

    # ----- seed: paper override or current base ------------------------------
    paper_seed_constructed = False
    if args.seed == "paper":
        x0, changed = build_paper_seed(m, base)
        paper_seed_constructed = True
        labels = m.reduced_index_labels
        print(f"\nseed=paper: overrode mass block with Huang 2025a physical values.",
              flush=True)
        print(f"  changed z indices : {changed.tolist()}", flush=True)
        print(f"  changed labels    : {[labels[i] for i in changed]}", flush=True)
        # verify the override round-trips back to the paper physical values.
        phys_seed = m.to_physical_mass(x0[None, :])
        print(f"  paper-seed physical mass (round-tripped through to_physical_mass):",
              flush=True)
        for k in ["theta_E", "gamma", "e1", "e2", "center_x", "center_y",
                  "gamma1", "gamma2"]:
            if k in phys_seed:
                print(f"    {k:>10s} = {float(phys_seed[k][0]):.6f}", flush=True)
    else:
        x0 = base.copy()
        print(f"\nseed=current: using the refined-MAP base unchanged.", flush=True)

    x0 = np.asarray(x0, dtype=np.float64)

    # ----- objective + jax grad/hessian (numpy-returning float64 wrappers) ---
    def f_scalar(z):
        return -m.target_log_prob_fn(z)

    f_jit = jax.jit(f_scalar)
    grad_jit = jax.jit(jax.grad(f_scalar))
    hess_jit = jax.jit(jax.hessian(f_scalar))

    def f_np(z):
        return float(f_jit(jnp.asarray(z, dtype=jnp.float64)))

    def jac_np(z):
        g = grad_jit(jnp.asarray(z, dtype=jnp.float64))
        return np.asarray(g, dtype=np.float64)

    def hess_np(z):
        H = hess_jit(jnp.asarray(z, dtype=jnp.float64))
        H = 0.5 * (H + H.T)  # symmetrize
        return np.asarray(H, dtype=np.float64)

    # gradient dtype sanity (the whole point is escaping the f32 floor).
    g0 = grad_jit(jnp.asarray(x0, dtype=jnp.float64))
    g0.block_until_ready()
    if str(g0.dtype) != "float64":
        raise RuntimeError(
            f"grad dtype is {g0.dtype}, not float64; float64 did not propagate "
            "through the simulator. Trust-region Newton on a cond~1e9 Hessian is "
            "meaningless in float32.")

    logp_before = float(m.target_log_prob_fn(jnp.asarray(x0, dtype=jnp.float64)))
    grad_norm_before = float(np.linalg.norm(np.asarray(g0, dtype=np.float64)))
    print(f"\nBEFORE trust-region Newton:", flush=True)
    print(f"  log_p={logp_before:.4f}  ||grad||={grad_norm_before:.6e}", flush=True)

    # ----- trust-region Newton (scipy trust-exact) ---------------------------
    # trust-exact solves the trust-region subproblem EXACTLY (More-Sorensen),
    # which handles INDEFINITE Hessians (escapes saddles) AND stiffness (no single
    # global damping).  Per-iter ||grad|| via the callback.
    iter_state = {"n": 0, "t0": time.time()}

    def callback(xk):
        iter_state["n"] += 1
        gn = float(np.linalg.norm(jac_np(xk)))
        fk = f_np(xk)
        print(f"  [trust {iter_state['n']:4d}]  f={fk:.4f}  log_p={-fk:.4f}  "
              f"||grad||={gn:.6e}  ({time.time()-iter_state['t0']:.0f}s)", flush=True)

    print(f"\n--- scipy trust-exact: maxiter={args.maxiter}, gtol={args.gtol:g} ---",
          flush=True)
    t0 = time.time()
    res = scipy.optimize.minimize(
        f_np, x0, method="trust-exact", jac=jac_np, hess=hess_np,
        callback=callback,
        options={"maxiter": args.maxiter, "gtol": args.gtol},
    )
    elapsed = time.time() - t0
    print(f"\ntrust-exact done in {elapsed:.1f}s  "
          f"({elapsed/60:.2f} min, {res.nit} iters)", flush=True)
    print(f"  scipy status={res.status}  success={res.success}", flush=True)
    print(f"  message: {res.message}", flush=True)

    # ----- final diagnostics at the converged point --------------------------
    qz = jnp.asarray(res.x, dtype=jnp.float64)
    logp_after = float(m.target_log_prob_fn(qz))
    g_after = np.asarray(grad_jit(qz), dtype=np.float64)
    grad_norm_after = float(np.linalg.norm(g_after))
    dist_from_start = float(np.linalg.norm(np.asarray(qz) - x0))

    print(f"\n--- recomputing reduced Hessian of -log_post at the converged point "
          f"({ndim}x{ndim}) ---", flush=True)
    t0 = time.time()
    H = hess_np(np.asarray(qz))
    print(f"  hessian computed in {time.time()-t0:.1f}s", flush=True)
    w, V, n_large_negative, n_negative, eig_min, eig_max = eig_report(H)

    # is_pd_mode: no NEGATIVE eigenvalues at all (a genuine PD mode).
    is_pd_mode = bool(n_negative == 0)

    print(f"\nAFTER trust-region Newton:", flush=True)
    print(f"  log_p={logp_after:.4f}  ||grad||={grad_norm_after:.6e}", flush=True)
    print(f"  ||grad|| reduction      : {grad_norm_before:.4e} -> {grad_norm_after:.4e} "
          f"(x{grad_norm_before/max(grad_norm_after,1e-30):.3e})", flush=True)
    print(f"  log_p change            : {logp_before:.4f} -> {logp_after:.4f} "
          f"({logp_after-logp_before:+.4f})", flush=True)
    print(f"  ||Delta|| from start    : {dist_from_start:.6e}", flush=True)
    print(f"  eig_min={eig_min:.6e}  eig_max={eig_max:.6e}", flush=True)
    print(f"  n_negative (any <0)     : {n_negative}  (want 0 -> PD mode)", flush=True)
    print(f"  n_large_negative (<-1e-6*max) : {n_large_negative}", flush=True)
    print(f"  most negative eigs      : {np.sort(w)[:8]}", flush=True)
    print(f"  smallest |eigs|         : {np.sort(np.abs(w))[:8]}", flush=True)
    print(f"\n  is_pd_mode (n_negative==0) = {is_pd_mode}", flush=True)

    # ----- physics at the converged point ------------------------------------
    phys = m.to_physical_mass(np.asarray(qz)[None, :])
    gamma = float(phys["gamma"][0])
    theta_E = float(phys["theta_E"][0])
    print(f"\nPhysical mass params at the converged point:", flush=True)
    print(f"  {'param':>10s} {'value':>12s} {'paper':>12s}", flush=True)
    print(f"  {'gamma':>10s} {gamma:>12.4f} {1.372:>12.4f}", flush=True)
    print(f"  {'theta_E':>10s} {theta_E:>12.4f} {2.6463:>12.4f}", flush=True)
    for k in ["e1", "e2", "center_x", "center_y", "gamma1", "gamma2"]:
        if k in phys:
            print(f"  {k:>10s} {float(phys[k][0]):>12.4f}", flush=True)

    # ----- lstsq amplitudes (n negative Sersic Ie) ---------------------------
    # amps[0:5] = 4 lens-light + 1 source Sersic Ie (UNPHYSICAL if negative);
    # amps[5:33] = 28 source shapelet amps (negative is fine).
    amps = np.asarray(m.lstsq_amps(qz)).astype(np.float64)
    sersic_ie = amps[:5]
    n_neg_sersic_ie = int(np.sum(sersic_ie < 0.0))
    n_neg_shapelet = int(np.sum(amps[5:] < 0.0))
    print(f"\nlstsq amplitudes at the converged point (33 total):", flush=True)
    print(f"  Sersic Ie (amps[0:5])  = {np.round(sersic_ie, 5)}", flush=True)
    print(f"  n_neg Sersic Ie        = {n_neg_sersic_ie}  (UNPHYSICAL if >0)", flush=True)
    print(f"  n_neg shapelet (of 28) = {n_neg_shapelet}  (expected/fine)", flush=True)

    # ----- PD-floored Hessian + cholesky (for the HMC mass matrix) -----------
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

    # ----- save --------------------------------------------------------------
    DATA.mkdir(parents=True, exist_ok=True)
    if args.out is not None:
        map_path = Path(args.out)
        if not map_path.is_absolute():
            map_path = REPRO / map_path
    else:
        map_path = DATA / f"map_trust_{args.seed}.npz"
    # Hessian goes alongside the map output: hess_trust_{seed}.npz next to it, or
    # derived from the --out stem so a custom --out keeps its Hessian paired.
    if args.out is not None:
        hess_path = map_path.with_name(map_path.stem.replace("map", "hess", 1)
                                       + ".npz") if "map" in map_path.stem \
            else map_path.with_name(map_path.stem + "_hess.npz")
    else:
        hess_path = DATA / f"hess_trust_{args.seed}.npz"

    np.savez(
        map_path,
        # 31_fit_lstsq.py reads qz_refined (or qz_pd); save both names so
        # --start-file accepts this file directly.
        qz=np.asarray(qz, dtype=np.float64),
        qz_refined=np.asarray(qz, dtype=np.float64),
        qz_start=x0.astype(np.float64),
        seed=np.array(args.seed),
        x64=np.bool_(True),
        paper_seed_constructed=np.bool_(paper_seed_constructed),
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
        eig_raw=w.astype(np.float64),
        eig_min=np.float64(eig_min),
        eig_max=np.float64(eig_max),
        n_negative=np.int64(n_negative),
        n_large_negative=np.int64(n_large_negative),
        is_pd_mode=np.bool_(is_pd_mode),
        amps=amps.astype(np.float64),
        n_neg_sersic_ie=np.int64(n_neg_sersic_ie),
        n_neg_shapelet=np.int64(n_neg_shapelet),
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

    # ----- verdict -----------------------------------------------------------
    print(f"\n=== VERDICT ===", flush=True)
    if is_pd_mode and grad_norm_after < 1e3:
        print(f"  trust-exact REACHED a PD mode (n_negative=0, ||grad||="
              f"{grad_norm_after:.3e}) at gamma={gamma:.4f}, theta_E={theta_E:.4f}.",
              flush=True)
    else:
        print(f"  trust-exact did NOT reach a clean PD mode "
              f"(||grad||={grad_norm_after:.3e}, n_negative={n_negative}); "
              f"gamma={gamma:.4f}, theta_E={theta_E:.4f}.", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
