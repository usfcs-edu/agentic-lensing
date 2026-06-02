"""Stage: is the PAPER's gamma~1.37 mode REACHABLE in the SMOOTH marginal model?

CONTEXT.  _hmc_lib_marg.build_model_marg(x64=True) is the SMOOTH 46-dim target
(41 nonlinear + 5 SAMPLED positive Sersic Ie) that does EXACT ridge-regularized
Gaussian marginalization of the 28 degenerate shapelet amps
(A = X^T W X + Lambda, Cholesky solve + -0.5 logdet evidence term).  Unlike the
old pinv/lstsq path (||grad|| floored at ~1e5, Hessian persistently indefinite),
this target is genuinely smooth: trust-region Newton drove ||grad|| from ~9.2e4
to ~400 and the Hessian to 44/46 POSITIVE eigenvalues (the 3 "negative" eigs are
float64 round-off vs eig_max ~2.3e12, cond ~1e18 -> effectively a PD mode).  That
refined point sits in the CURRENT basin at gamma=1.866, theta_E=2.600,
logp=-45841 (data/map_marg_pd.npz, the marginal's constant convention).

QUESTION this script answers.  The paper (Huang 2025a, foundry-i) reports
gamma=1.372+/-0.023, theta_E=2.6463.  Our refined mode is at gamma=1.866.  Is the
gamma gap (a) a SAMPLER/optimizer artifact (the paper mode is a real optimum of
OUR smooth model that our current basin just failed to find) or (b) an
UNPUBLISHED-MODEL-DIFFERENCE (the paper mode is simply not a good fit to our model
setup, so no amount of better sampling reaches it)?

METHOD.  Seed scipy trust-exact (the SAME exact-subproblem trust-region Newton 33
used: jax grad + full jax.hessian, float64) from a PAPER-SEEDED start, and let it
refine the smooth marginal target.  The start is build_model_marg's own qz_start
(46-dim) with ONLY the mass block OVERRIDDEN to the paper's PHYSICAL values
(theta_E=2.6463, gamma=1.372, e1=0.1091, e2=-0.1320, center_x~0, center_y~0, shear
gamma1=0.0657, gamma2=-0.0939), mapped into unconstrained z via the model's
blockwise constraining bijector (forward the base z -> override the structured
mass block -> bij.inverse).  Because the bijector acts per-parameter, ONLY the 8
mass-block z-indices change; the 5 Ie and the remaining 33 nonlinear dims
round-trip from qz_start unchanged.  (Same construction 33_trust_refine.py used,
adapted to the 46-dim marginal model: mass main x[0][0], shear x[0][1].)

INTERPRETATION (migrated_to_paper).
  TRUE  -> after refining, the point STAYED near gamma~1.37 with a
           competitive-or-better log_p than the current basin's -45841.  The paper
           mode IS a real, reachable optimum of our smooth setup => the gamma gap
           is a SAMPLER issue (our current basin just missed it).
  FALSE -> the point either DRIFTED back toward gamma~1.86, or stayed near
           gamma~1.37 but at a MUCH-WORSE log_p.  The paper mode is NOT a good fit
           to our model setup => the gamma gap is an UNPUBLISHED-MODEL-DIFFERENCE,
           not a sampler issue.

Outputs (float64):
  data/map_marg_paper.npz  : qz, grad, logp, gamma, theta_E (+ before/after, eig
                             spectrum, n_negative, is_pd_mode, shapelet amps).
  data/hess_marg_paper.npz : H (raw symmetrized), H_reg (PD-floored), chol(H_reg).

Run (one idle A16; A16 FP64 ~0.39 s/grad, one jax.hessian ~1 min, ~80 trust iters):
  CUDA_DEVICE_ORDER=PCI_BUS_ID XLA_FLAGS=--xla_gpu_autotune_level=0 \
    CUDA_VISIBLE_DEVICES=5 /raid/benson/.venvs/gigalens/bin/python \
      35_refine_paper_marg.py --maxiter 80

This script does NOT edit or import _hmc_lib_marg's mutable state; it only calls
build_model_marg(x64=True).  It never touches 34_fit_marg.py.
"""
import argparse
import os

# This script is ALWAYS float64.  Trust-region Newton on a cond~1e18 Hessian needs
# float64 (the f32 gradient saturates at a ~1e4 noise floor).  Enable x64 BEFORE
# _hmc_lib_marg is imported (it reads GIGALENS_X64 at import, before any jnp array).
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

import _hmc_lib_marg  # noqa: E402

REPRO = Path(__file__).parent
DATA = REPRO / "data"

# Huang 2025a (foundry-i) published PHYSICAL mass params.
PAPER_MASS = dict(
    theta_E=2.6463, gamma=1.372, e1=0.1091, e2=-0.1320,
    center_x=0.0, center_y=0.0,
)
PAPER_SHEAR = dict(gamma1=0.0657, gamma2=-0.0939)

PAPER_GAMMA = 1.372
PAPER_THETA_E = 2.6463
CURRENT_BASIN_LOGP = -45840.98400599839  # data/map_marg_pd.npz logp (gamma=1.866).
CURRENT_BASIN_GAMMA = 1.8655244015301542

LARGE_NEG_FRAC = 1e-6  # PD floor / large-negative threshold (matches 33/34).
# "near the paper" gamma window: the paper gamma 1.372 vs the current basin 1.866
# are ~0.49 apart; "stayed near 1.37" means it did NOT drift the bulk of the way
# back toward 1.86.  Use the paper +/-0.10 window (covers the published 0.023 std
# with margin) as "near paper".
GAMMA_NEAR_PAPER = 0.10
# log_p tolerance for "competitive": within a small slack of the current basin
# (the marginal logp scale is ~ -45841; a few units of slack is generous).
LOGP_COMPETITIVE_SLACK = 5.0


def build_paper_seed(m, base):
    """Map the paper PHYSICAL mass params into unconstrained z, blockwise.

    Forward the base z (46-dim) through the model's constraining bijector
    m.prob_model.bij to the structured event-space point x; OVERRIDE the EPL
    (theta_E, gamma, e1, e2, center_x, center_y) and shear (gamma1, gamma2)
    physical values with the paper values; inverse the whole structured point back
    to a flat 46-vector z.  Because the bijector acts per-parameter, only the 8
    mass-block z-indices change; the 5 Sersic Ie and the other 33 nonlinear dims
    round-trip from the base unchanged.

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
    # Preserve the marginal model's nesting: x = [[mass_main, mass_shear],
    # lens_light_list, [src_sersic, src_shapelet]].
    x_new = [[mass_main, mass_shear], x[1], x[2]]

    z_new = np.array(
        [float(np.asarray(v).squeeze()) for v in bij.inverse(x_new)],
        dtype=np.float64)
    if not np.all(np.isfinite(z_new)):
        raise RuntimeError(
            "paper-seed bijector inverse produced non-finite z; a paper mass value "
            "is outside its prior support (e.g. gamma outside [1.0, 2.7]).")
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
    ap.add_argument("--maxiter", type=int, default=80,
                    help="scipy trust-exact max iterations.")
    ap.add_argument("--gtol", type=float, default=1e-2,
                    help="scipy gradient-norm convergence tolerance.")
    ap.add_argument("--out", type=str, default=None,
                    help="output npz for the refined point "
                         "(data/map_marg_paper.npz by default); the Hessian goes to "
                         "data/hess_marg_paper.npz alongside it.")
    args = ap.parse_args()

    print(f"devices: {jax.devices()}", flush=True)
    print(f"jax_enable_x64={jax.config.jax_enable_x64}", flush=True)

    m = _hmc_lib_marg.build_model_marg(x64=True)
    ndim = int(m.ndim)
    labels = list(m.index_labels)
    assert ndim == 46, f"expected 46-dim marginal model, got {ndim}"

    # ----- base start vector: build_model_marg's own 46-dim qz_start ----------
    base = np.asarray(m.qz_start, dtype=np.float64)
    logp_base = float(m.target_log_prob_fn(jnp.asarray(base, dtype=jnp.float64)))
    phys_base = m.to_physical_mass(base[None, :])
    print(f"\nbase = build_model_marg.qz_start  ndim={ndim}", flush=True)
    print(f"  base log_p={logp_base:.4f}  gamma={float(phys_base['gamma'][0]):.4f}  "
          f"theta_E={float(phys_base['theta_E'][0]):.4f}", flush=True)

    # ----- paper-seeded start: override the mass block ------------------------
    x0, changed = build_paper_seed(m, base)
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
    x0 = np.asarray(x0, dtype=np.float64)

    # ----- objective + jax grad/hessian (numpy-returning float64 wrappers) ----
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
            "through the simulator. Trust-region Newton on a cond~1e18 Hessian is "
            "meaningless in float32.")

    logp_before = float(m.target_log_prob_fn(jnp.asarray(x0, dtype=jnp.float64)))
    grad_norm_before = float(np.linalg.norm(np.asarray(g0, dtype=np.float64)))
    # logp_paper_basin is the log_p AT the paper-seed point (the reported quantity).
    logp_paper_seed = logp_before
    print(f"\nBEFORE trust-region Newton (paper-seed point):", flush=True)
    print(f"  log_p={logp_before:.4f}  ||grad||={grad_norm_before:.6e}", flush=True)
    print(f"  (current basin logp={CURRENT_BASIN_LOGP:.4f} at gamma="
          f"{CURRENT_BASIN_GAMMA:.4f})", flush=True)

    # ----- trust-region Newton (scipy trust-exact) ---------------------------
    iter_state = {"n": 0, "t0": time.time()}

    def callback(xk):
        iter_state["n"] += 1
        gn = float(np.linalg.norm(jac_np(xk)))
        fk = f_np(xk)
        ph = m.to_physical_mass(np.asarray(xk)[None, :])
        print(f"  [trust {iter_state['n']:4d}]  f={fk:.4f}  log_p={-fk:.4f}  "
              f"||grad||={gn:.6e}  gamma={float(ph['gamma'][0]):.4f}  "
              f"theta_E={float(ph['theta_E'][0]):.4f}  "
              f"({time.time()-iter_state['t0']:.0f}s)", flush=True)

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

    print(f"\n--- recomputing Hessian of -log_post at the converged point "
          f"({ndim}x{ndim}) ---", flush=True)
    t0 = time.time()
    H = hess_np(np.asarray(qz))
    print(f"  hessian computed in {time.time()-t0:.1f}s", flush=True)
    w, V, n_large_negative, n_negative, eig_min, eig_max = eig_report(H)

    # is_pd_mode: NO large-negative eigenvalues (the few tiny negatives are float64
    # round-off vs eig_max ~1e12, cond ~1e18 -- exactly the situation documented for
    # data/map_marg_pd.npz, which is "effectively a PD mode").
    is_pd_mode = bool(n_large_negative == 0)

    print(f"\nAFTER trust-region Newton (paper-seeded):", flush=True)
    print(f"  log_p={logp_after:.4f}  ||grad||={grad_norm_after:.6e}", flush=True)
    print(f"  ||grad|| reduction      : {grad_norm_before:.4e} -> {grad_norm_after:.4e} "
          f"(x{grad_norm_before/max(grad_norm_after,1e-30):.3e})", flush=True)
    print(f"  log_p change            : {logp_before:.4f} -> {logp_after:.4f} "
          f"({logp_after-logp_before:+.4f})", flush=True)
    print(f"  ||Delta|| from start    : {dist_from_start:.6e}", flush=True)
    print(f"  eig_min={eig_min:.6e}  eig_max={eig_max:.6e}", flush=True)
    print(f"  n_negative (any <0)     : {n_negative}", flush=True)
    print(f"  n_large_negative (<-1e-6*max) : {n_large_negative}  (0 -> PD mode)",
          flush=True)
    print(f"  most negative eigs      : {np.sort(w)[:8]}", flush=True)
    print(f"  smallest |eigs|         : {np.sort(np.abs(w))[:8]}", flush=True)
    print(f"\n  is_pd_mode (n_large_negative==0) = {is_pd_mode}", flush=True)

    # ----- physics at the converged point ------------------------------------
    phys = m.to_physical_mass(np.asarray(qz)[None, :])
    gamma = float(phys["gamma"][0])
    theta_E = float(phys["theta_E"][0])
    print(f"\nPhysical mass params at the converged point:", flush=True)
    print(f"  {'param':>10s} {'value':>12s} {'paper':>12s}", flush=True)
    print(f"  {'gamma':>10s} {gamma:>12.4f} {PAPER_GAMMA:>12.4f}", flush=True)
    print(f"  {'theta_E':>10s} {theta_E:>12.4f} {PAPER_THETA_E:>12.4f}", flush=True)
    for k in ["e1", "e2", "center_x", "center_y", "gamma1", "gamma2"]:
        if k in phys:
            print(f"  {k:>10s} {float(phys[k][0]):>12.4f}", flush=True)

    # ----- marginal shapelet amps + sampled Sersic Ie at the converged point --
    # The 28 shapelet amps are the analytic marginal-mode solution a*(z) (negative
    # is fine -- shapelets are signed).  The 5 Sersic Ie are now SAMPLED z-coords
    # (LogNormal -> strictly positive by construction), so n_neg_sersic_ie==0 unless
    # something is badly wrong.
    amps_shapelet = np.asarray(m.shapelet_amps(qz)).astype(np.float64)
    n_neg_shapelet = int(np.sum(amps_shapelet < 0.0))
    ie_labels = ["LL0.Ie", "LL1.Ie", "LL2.Ie", "LL3.Ie", "srcS.Ie"]
    sersic_ie = np.array([float(phys.get(l, [np.nan])[0]) if l in phys else
                          float(m.to_physical_mass(np.asarray(qz)[None, :]).get(l, [np.nan])[0])
                          for l in ie_labels])
    # to_physical_mass only returns mass params, so pull Ie from the bijector forward.
    xfull = m.prob_model.bij.forward(list(np.asarray(qz, dtype=np.float64)[:, None]))
    flat = dict(_hmc_lib_marg._flat_named_46(xfull))
    sersic_ie = np.array([float(flat[l]) for l in ie_labels], dtype=np.float64)
    n_neg_sersic_ie = int(np.sum(sersic_ie < 0.0))
    print(f"\nmarginal-mode shapelet amps (28) + sampled Sersic Ie (5) at converged:",
          flush=True)
    print(f"  Sersic Ie              = {np.round(sersic_ie, 5)}", flush=True)
    print(f"  n_neg Sersic Ie        = {n_neg_sersic_ie}  (sampled LogNormal -> 0)",
          flush=True)
    print(f"  n_neg shapelet (of 28) = {n_neg_shapelet}  (signed -> fine)", flush=True)

    # ----- PD-floored Hessian + cholesky (for an HMC mass matrix) ------------
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

    # ----- migrated_to_paper verdict -----------------------------------------
    # stayed near gamma~1.37?
    gamma_near_paper = bool(abs(gamma - PAPER_GAMMA) <= GAMMA_NEAR_PAPER)
    drifted_toward_current = bool(abs(gamma - CURRENT_BASIN_GAMMA) <
                                  abs(gamma - PAPER_GAMMA))
    # competitive-or-better log_p vs the current basin -45841?
    logp_competitive = bool(logp_after >= CURRENT_BASIN_LOGP - LOGP_COMPETITIVE_SLACK)
    migrated_to_paper = bool(gamma_near_paper and not drifted_toward_current
                             and logp_competitive)

    # logp_paper_basin: the converged log_p of the paper-seeded refine (the basin
    # the paper seed settled into); logp_current_basin: the documented -45841.
    logp_paper_basin = logp_after
    logp_current_basin = CURRENT_BASIN_LOGP

    # ----- save --------------------------------------------------------------
    DATA.mkdir(parents=True, exist_ok=True)
    if args.out is not None:
        map_path = Path(args.out)
        if not map_path.is_absolute():
            map_path = REPRO / map_path
    else:
        map_path = DATA / "map_marg_paper.npz"
    hess_path = DATA / "hess_marg_paper.npz"

    np.savez(
        map_path,
        qz=np.asarray(qz, dtype=np.float64),
        qz_refined=np.asarray(qz, dtype=np.float64),
        qz_start=x0.astype(np.float64),
        seed=np.array("paper"),
        x64=np.bool_(True),
        paper_seed_constructed=np.bool_(True),
        grad=g_after.astype(np.float64),
        grad_norm_before=np.float64(grad_norm_before),
        grad_norm_after=np.float64(grad_norm_after),
        logp=np.float64(logp_after),
        logp_before=np.float64(logp_before),
        logp_after=np.float64(logp_after),
        logp_paper_seed=np.float64(logp_paper_seed),
        logp_paper_basin=np.float64(logp_paper_basin),
        logp_current_basin=np.float64(logp_current_basin),
        gamma=np.float64(gamma),
        theta_E=np.float64(theta_E),
        gamma_paper=np.float64(PAPER_GAMMA),
        theta_E_paper=np.float64(PAPER_THETA_E),
        dist_from_start=np.float64(dist_from_start),
        nit=np.int64(int(res.nit)),
        scipy_status=np.int64(int(res.status)),
        scipy_success=np.bool_(bool(res.success)),
        wall_s=np.float64(elapsed),
        eig_raw=w.astype(np.float64),
        eig_min=np.float64(eig_min),
        eig_max=np.float64(eig_max),
        n_negative=np.int64(n_negative),
        n_large_negative=np.int64(n_large_negative),
        is_pd_mode=np.bool_(is_pd_mode),
        amps_shapelet=amps_shapelet.astype(np.float64),
        sersic_ie=sersic_ie.astype(np.float64),
        n_neg_shapelet=np.int64(n_neg_shapelet),
        n_neg_sersic_ie=np.int64(n_neg_sersic_ie),
        cond_after_floor=np.float64(cond_after),
        gamma_near_paper=np.bool_(gamma_near_paper),
        drifted_toward_current=np.bool_(drifted_toward_current),
        logp_competitive=np.bool_(logp_competitive),
        migrated_to_paper=np.bool_(migrated_to_paper),
        index_labels=np.array(m.index_labels),
    )
    save_hess = dict(
        H=H.astype(np.float64),
        H_raw=H.astype(np.float64),
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
    print(f"  paper-seed point     : gamma={PAPER_GAMMA:.4f}, theta_E={PAPER_THETA_E:.4f}, "
          f"log_p={logp_paper_seed:.4f}", flush=True)
    print(f"  converged point      : gamma={gamma:.4f}, theta_E={theta_E:.4f}, "
          f"log_p={logp_after:.4f}", flush=True)
    print(f"  current basin (ref)  : gamma={CURRENT_BASIN_GAMMA:.4f}, "
          f"log_p={CURRENT_BASIN_LOGP:.4f}", flush=True)
    print(f"  gamma_near_paper={gamma_near_paper}  "
          f"drifted_toward_current={drifted_toward_current}  "
          f"logp_competitive={logp_competitive}", flush=True)
    print(f"  is_pd_mode={is_pd_mode}  ||grad||={grad_norm_after:.4e}  "
          f"n_negative={n_negative}", flush=True)
    print(f"\n  migrated_to_paper = {migrated_to_paper}", flush=True)
    if migrated_to_paper:
        print(f"  => The paper's gamma~1.37 mode IS a real, reachable optimum of our "
              f"smooth marginal model.\n     The gamma gap is a SAMPLER/optimizer "
              f"issue (the current basin just missed it).", flush=True)
    else:
        if drifted_toward_current:
            print(f"  => The paper-seeded point DRIFTED back toward gamma~1.86.",
                  flush=True)
        elif not logp_competitive:
            print(f"  => The paper-seeded point stayed near gamma~1.37 but at a "
                  f"MUCH-WORSE log_p ({logp_after-CURRENT_BASIN_LOGP:+.1f} vs the "
                  f"current basin).", flush=True)
        print(f"  => The paper's gamma~1.37 mode is NOT a good fit to our model "
              f"setup.\n     The gamma gap is an UNPUBLISHED-MODEL-DIFFERENCE, not a "
              f"sampler issue.", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
