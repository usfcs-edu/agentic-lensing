"""Stage 5: refine the 41-dim lstsq-reduced model to a TRUE PD mode.

The full 74-dim model samples 33 LINEAR light amplitudes that leave ~56 near-flat
Hessian directions at the MAP, which is why fixed-leapfrog HMC mixes terribly.
_hmc_lib_lstsq.build_model_lstsq() profiles-out (marginalizes via least squares
per log_prob call) all 33 amplitudes, so HMC samples only the 41 NONLINEAR params.
The reduced-space Hessian has NO flat directions (smallest |eig|~1.0 vs the full
model's 56 near-zero) -- conditioning is FIXED.

BUT m.qz_start_nonlinear is the full-model MAP projected into the 41-dim space, so
it is a SADDLE in the reduced space: 12 negative Hessian eigenvalues, ||grad||~2.8e5.
You cannot mix an HMC from a saddle.

This script drives ||grad|| of -log_post down by orders of magnitude (the same
3-phase optax recipe used in 28_refine_map.py: adabelief -> adam -> adam, tracking
the best point by minimum gradient norm), then re-examines the reduced Hessian:
  - n_negative before (12) vs after (want ~0 -> a PD mode)?
  - n_tiny (flat) directions, condition number?
  - is_pd_mode (grad~0 AND no large negative eigs)?
And the physics: gamma + theta_E at the refined mode via to_physical_mass -- did
the lstsq marginalization move gamma off the full-model ~2.18?
And: how many of the 5 Sersic Ie linear amplitudes solve NEGATIVE at the mode
(unphysical)? amps[0:5] are the 4 lens-light + 1 source Sersic Ie; amps[5:33]
are the 28 source shapelet amps (negative shapelet amps are expected and fine).

Outputs:
  data/map_refined_lstsq.npz  : qz_refined (41-vec), grad/logp before+after,
                                gamma/theta_E at mode, eig spectrum, n_neg
                                before/after, amps, n_neg_sersic_ie.
  data/hess_refined_lstsq.npz : H_raw, H_reg (PD-floored), chol, eig spectrum.

Run (one idle A16):
  CUDA_DEVICE_ORDER=PCI_BUS_ID XLA_FLAGS=--xla_gpu_autotune_level=0 \
    CUDA_VISIBLE_DEVICES=1 /raid/benson/.venvs/gigalens/bin/python 30_refine_lstsq.py --x64

--x64: run the entire reduced model + MAP refinement in float64 (cond ~1e10 fits
  float64's ~16 digits but NOT float32's ~7; without it ||grad|| stalls at a
  ~1.2e4 f32 noise floor and no PD mode is reachable).  When --x64 is passed,
  outputs are written to data/map_refined_lstsq64.npz and hess_refined_lstsq64.npz.
"""
import argparse
import os

# --x64 must enable jax_enable_x64 BEFORE _hmc_lib_lstsq is imported (it sets x64
# at its own import based on the GIGALENS_X64 env var, before any jnp array), so
# parse it first and set the env var.
_ap0 = argparse.ArgumentParser(add_help=False)
_ap0.add_argument("--x64", action="store_true",
                  help="run reduced model + refinement in float64")
_ap0.add_argument("--newton", action="store_true",
                  help="add an f64 damped-Newton (Levenberg-Marquardt) polish "
                       "after the optax phases. The reduced objective has "
                       "cond~1e10, far too stiff for first-order optax to reach a "
                       "stationary point; in f64 (~16 digits) the Hessian is "
                       "invertible to enough precision that (H+lambda*I)dz=-g gives "
                       "a descending step, driving ||grad|| orders of magnitude "
                       "below the f32 1.2e4 floor. (f32 Newton failed only because "
                       "lambda had to grow to ~1e10 to stay PD, destroying the step "
                       "-- exactly the precision wall float64 removes.)")
_ap0.add_argument("--newton-steps", type=int, default=60)
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
import optax

import _hmc_lib_lstsq

REPRO = Path(__file__).parent
DATA = REPRO / "data"


def newton_polish(f_jit, grad_jit, f_for_hess, best, n_steps, x64):
    """f64 damped-Newton (Levenberg-Marquardt) polish from the optax best point.

    Solves (H + lambda*I) dz = -grad each step (H = Hessian of f = -log_post),
    adapting lambda by the trust ratio.  On a cond~1e10 objective this is the
    ONLY thing that drives ||grad|| toward 0; it needs f64 to invert H.
    Tracks/returns the running best by minimum ||grad|| (same convention as the
    optax phases).
    """
    best_z, best_f, best_gn = best
    hess_jit = jax.jit(jax.hessian(f_for_hess))
    n = int(best_z.shape[0])
    eye = jnp.eye(n, dtype=best_z.dtype)
    z = best_z
    lam = 1e3 if x64 else 1e6  # damping; f64 can use far less damping.
    print(f"\n--- phase4: f64 damped-Newton, {n_steps} steps (start lambda={lam:g}) ---",
          flush=True)
    fz = float(f_jit(z))
    t0 = time.time()
    for i in range(n_steps):
        g = grad_jit(z)
        gn = float(jnp.linalg.norm(g))
        if np.isfinite(gn) and gn < best_gn:
            best_gn, best_f, best_z = gn, float(f_jit(z)), z
        H = hess_jit(z)
        H = 0.5 * (H + H.T)
        # try the LM step; if it doesn't reduce f, grow lambda and retry.
        accepted = False
        for _retry in range(12):
            try:
                dz = jnp.linalg.solve(H + lam * eye, -g)
            except Exception:
                lam *= 10.0
                continue
            z_new = z + dz
            f_new = float(f_jit(z_new))
            if np.isfinite(f_new) and f_new < fz:
                z, fz = z_new, f_new
                lam = max(lam / 3.0, 1e-8)
                accepted = True
                break
            lam *= 10.0
        if i % 5 == 0 or i == n_steps - 1:
            print(f"  [p4 {i:4d}]  f={fz:.4f}  log_p={-fz:.4f}  ||grad||={gn:.4e}  "
                  f"lambda={lam:.3e}  (best ||grad||={best_gn:.4e})", flush=True)
        if not accepted and lam > 1e15:
            print(f"  [p4 {i:4d}] lambda saturated ({lam:.2e}); stopping.", flush=True)
            break
    # final check at the last z.
    g = grad_jit(z)
    gn = float(jnp.linalg.norm(g))
    if np.isfinite(gn) and gn < best_gn:
        best_gn, best_f, best_z = gn, float(f_jit(z)), z
    print(f"  phase4 done in {time.time()-t0:.1f}s  best ||grad||={best_gn:.4e}",
          flush=True)
    return (best_z, best_f, best_gn)


def main():
    args = _ap0.parse_args()
    x64 = bool(args.x64)
    fdtype = jnp.float64 if x64 else jnp.float32
    suffix = "64" if x64 else ""
    print(f"devices: {jax.devices()}", flush=True)
    print(f"x64={x64}  jax_enable_x64={jax.config.jax_enable_x64}  "
          f"fdtype={fdtype.__name__ if hasattr(fdtype,'__name__') else fdtype}",
          flush=True)
    m = _hmc_lib_lstsq.build_model_lstsq(x64=x64)
    z0 = jnp.asarray(m.qz_start_nonlinear, dtype=fdtype)
    ndim = int(z0.shape[0])

    # f(z) = -log_post(z); we MINIMIZE f.
    def f(z):
        return -m.target_log_prob_fn(z)

    f_jit = jax.jit(f)
    grad_jit = jax.jit(jax.grad(f))

    logp_before = float(m.target_log_prob_fn(z0))
    g0 = grad_jit(z0)
    g0.block_until_ready()
    grad_norm_before = float(jnp.linalg.norm(g0))
    grad_dtype = str(g0.dtype)
    print(f"\nndim={ndim}  (reduced nonlinear params)", flush=True)
    # VERIFY the gradient dtype is float64 under --x64. If this prints float32 the
    # whole point (escaping the f32 ~1.2e4 noise floor) is lost.
    print(f"GRAD DTYPE = {grad_dtype}  (expect float64 under --x64)", flush=True)
    print(f"z0 dtype   = {z0.dtype}", flush=True)
    if x64 and grad_dtype != "float64":
        raise RuntimeError(
            f"--x64 set but grad dtype is {grad_dtype}, not float64; "
            "float64 did not propagate through the simulator.")
    print(f"BEFORE refine:  log_p={logp_before:.4f}  ||grad||={grad_norm_before:.6e}",
          flush=True)

    # ----- optimizer: adabelief w/ cosine-decay LR + grad clipping ---------------
    # Same recipe as 28_refine_map.py: clip the global grad norm, anneal LR to a
    # small floor, track the best point by ||grad|| (a low-loss point can still
    # have huge gradient in a stiff valley; for a stationary point we want min
    # gradient), then a long small-LR adam polish.
    def run_phase(z, opt, n_steps, label, log_every, best):
        best_z, best_f, best_gn = best
        opt_state = opt.init(z)

        @jax.jit
        def step(z, opt_state):
            updates, opt_state = opt.update(grad_jit(z), opt_state, z)
            z_new = optax.apply_updates(z, updates)
            loss_new = f_jit(z_new)
            gn_new = jnp.linalg.norm(grad_jit(z_new))
            return z_new, opt_state, loss_new, gn_new

        t0 = time.time()
        for i in range(n_steps):
            z, opt_state, loss, gn = step(z, opt_state)
            fi = float(loss)
            gni = float(gn)
            if np.isfinite(fi) and np.isfinite(gni) and gni < best_gn:
                best_gn = gni
                best_f = fi
                best_z = z
            if i % log_every == 0 or i == n_steps - 1:
                print(f"  [{label} {i:5d}]  f={fi:.4f}  log_p={-fi:.4f}  "
                      f"||grad||={gni:.4e}  (best ||grad||={best_gn:.4e})", flush=True)
        z.block_until_ready()
        print(f"  {label} done in {time.time()-t0:.1f}s", flush=True)
        return z, (best_z, best_f, best_gn)

    best = (z0, float(f_jit(z0)), grad_norm_before)

    # First-order optax CANNOT reach a stationary point on this cond~1e10 objective
    # (||grad|| oscillates ~1e5 regardless of precision); its only job is to nudge
    # the projected start into the right basin.  The actual gradient minimization is
    # done by the f64 damped-Newton phase (--newton).  So with --x64 --newton keep
    # the optax budget SHORT (a brief adabelief warmup) and let Newton do the work.
    if x64 and args.newton:
        N1, N2, N3 = 800, 400, 200
    else:
        N1 = 2500 if x64 else 4000
        N2 = 2000 if x64 else 3000
        N3 = 1500 if x64 else 2000

    # Phase 1: adabelief, cosine decay, grad-clip to 1e4.
    PEAK_LR = 2e-3
    sched1 = optax.warmup_cosine_decay_schedule(
        init_value=PEAK_LR * 0.1, peak_value=PEAK_LR,
        warmup_steps=100, decay_steps=N1, end_value=PEAK_LR * 1e-2)
    opt1 = optax.chain(optax.clip_by_global_norm(1e4),
                       optax.adabelief(learning_rate=sched1))
    print(f"\n--- phase1: adabelief+clip, {N1} steps, peak_lr={PEAK_LR} ---", flush=True)
    z, best = run_phase(z0, opt1, N1, "p1", 200, best)

    # Phase 2: adam small fixed LR + tighter clip, from the best point so far.
    opt2 = optax.chain(optax.clip_by_global_norm(1e3),
                       optax.adam(learning_rate=1e-4))
    print(f"\n--- phase2: adam lr=1e-4 + clip, {N2} steps ---", flush=True)
    z, best = run_phase(best[0], opt2, N2, "p2", 200, best)

    # Phase 3: adam tiny LR, final polish to minimize gradient.
    opt3 = optax.chain(optax.clip_by_global_norm(3e2),
                       optax.adam(learning_rate=2e-5))
    print(f"\n--- phase3: adam lr=2e-5 + clip, {N3} steps ---", flush=True)
    z, best = run_phase(best[0], opt3, N3, "p3", 200, best)

    # ----- phase 4 (optional): f64 damped-Newton / Levenberg-Marquardt polish -----
    # First-order optax cannot reach a stationary point on a cond~1e10 objective:
    # the step that descends the soft directions is ~1e10x too large for the stiff
    # ones, so ||grad|| oscillates around ~1e5 (seen in phases 1-3) and never
    # approaches 0.  A (damped) Newton step (H+lambda*I)^-1 g rescales every
    # eigendirection by ~1/lambda_i, which is the only way to descend a stiff
    # valley.  This REQUIRES f64: cond~1e10 needs ~10 digits just to represent H
    # accurately and more to invert it, which f32's ~7 digits cannot supply (the
    # f32 attempt forced lambda->1e10, collapsing the step).  We adapt lambda by
    # trust-ratio (accept & shrink lambda on improvement, reject & grow on failure).
    if args.newton:
        best = newton_polish(f_jit, grad_jit, f, best, args.newton_steps, x64)

    best_z, best_f, best_gn = best

    # NOTE on the achievable gradient floor.  The reduced (amplitude-marginalized)
    # objective is EXTREMELY stiff: at the optax best point the Hessian eigenvalues
    # span ~[-1.4e10, +3.6e10] (cond ~1e10).  The gigalens simulator + lstsq solve
    # run in float32, so the gradient of -log_post saturates at a float32 noise floor
    # of ||grad|| ~ 1e4 here; below that, neither further optax nor a float64
    # damped-Newton / saddle-free-Newton polish reduces ||grad|| (every Newton step
    # is rejected as lambda -> 1e10, verified empirically).  Consequently a handful
    # of Hessian eigenvalues remain slightly negative (their magnitudes -1.1 .. -29
    # are ~1e-10 of eig_max, i.e. float32 curvature noise; two genuinely large
    # negatives ~1e10/1e9 mark a stiff ridge optax cannot leave in f32).  The point
    # is therefore the best-achievable reduced MAP in float32, NOT a perfectly PD
    # mode.  This is fine for HMC: the sampler's mass matrix is the PD-FLOORED
    # Hessian H_reg (eigenvalues floored to 1e-6*eig_max -> cond 1e6, chol_ok=True),
    # which is positive-definite by construction; and the start being a near-MAP with
    # a curvature-matched preconditioner is what fixes mixing, not f32-exact PD-ness.
    qz_refined = jnp.asarray(best_z, dtype=fdtype)
    logp_after = float(m.target_log_prob_fn(qz_refined))
    g_after = grad_jit(qz_refined)
    g_after.block_until_ready()
    grad_norm_after = float(jnp.linalg.norm(g_after))
    dist_from_start = float(jnp.linalg.norm(qz_refined - z0))

    print(f"\nAFTER refine:   log_p={logp_after:.4f}  ||grad||={grad_norm_after:.6e}",
          flush=True)
    print(f"  log_p improvement     = {logp_after - logp_before:+.4f}", flush=True)
    print(f"  ||grad|| reduction    = {grad_norm_before:.4e} -> {grad_norm_after:.4e} "
          f"(x{grad_norm_before/max(grad_norm_after,1e-30):.2e})", flush=True)
    print(f"  ||Delta|| from start  = {dist_from_start:.6e}", flush=True)

    # ----- physical mass params at the start vs the refined point -----------------
    phys_start = m.to_physical_mass(np.asarray(z0)[None, :])
    phys_ref = m.to_physical_mass(np.asarray(qz_refined)[None, :])
    gamma_start = float(phys_start["gamma"][0])
    theta_E_start = float(phys_start["theta_E"][0])
    gamma_refined = float(phys_ref["gamma"][0])
    theta_E_refined = float(phys_ref["theta_E"][0])
    print(f"\nPhysical mass params (did lstsq move gamma off the full-model ~2.18?):",
          flush=True)
    print(f"  {'param':>10s} {'start':>12s} {'refined':>12s} {'paper':>12s}", flush=True)
    print(f"  {'gamma':>10s} {gamma_start:>12.4f} {gamma_refined:>12.4f} {1.372:>12.4f}",
          flush=True)
    print(f"  {'theta_E':>10s} {theta_E_start:>12.4f} {theta_E_refined:>12.4f} "
          f"{2.6463:>12.4f}", flush=True)
    for k in ["e1", "e2", "gamma1", "gamma2"]:
        print(f"  {k:>10s} {float(phys_start[k][0]):>12.4f} "
              f"{float(phys_ref[k][0]):>12.4f}", flush=True)

    # ----- lstsq amplitudes at the refined mode -----------------------------------
    # Ordering from gigalens lstsq_simulate: lens_light first (4 Sersic Ie), then
    # source (1 Sersic Ie + 28 shapelet amps). So amps[0:5] = the 5 Sersic Ie
    # (unphysical if negative); amps[5:33] = 28 shapelet amps (neg is fine).
    amps = np.asarray(m.lstsq_amps(qz_refined)).astype(np.float64)
    sersic_ie = amps[:5]
    shapelet_amps = amps[5:]
    n_neg_sersic_ie = int(np.sum(sersic_ie < 0.0))
    n_neg_shapelet = int(np.sum(shapelet_amps < 0.0))
    print(f"\nlstsq amplitudes at the refined mode (33 total):", flush=True)
    print(f"  Sersic Ie (amps[0:5])  = {np.round(sersic_ie, 5)}", flush=True)
    print(f"  n_neg Sersic Ie        = {n_neg_sersic_ie}  (UNPHYSICAL if >0)", flush=True)
    print(f"  n_neg shapelet (of 28) = {n_neg_shapelet}  (expected/fine)", flush=True)

    # ----- Hessian at the refined point -------------------------------------------
    print(f"\n--- reduced Hessian of -log_post at refined point ({ndim}x{ndim}) ---",
          flush=True)
    t0 = time.time()
    H = np.asarray(jax.hessian(f)(qz_refined)).astype(np.float64)
    H = 0.5 * (H + H.T)  # symmetrize
    print(f"  hessian computed in {time.time()-t0:.1f}s", flush=True)

    # n_negative BEFORE (re-eval at the start saddle) for the report.
    H0 = np.asarray(jax.hessian(f)(z0)).astype(np.float64)
    H0 = 0.5 * (H0 + H0.T)
    eig0 = np.linalg.eigvalsh(H0)
    n_negative_before = int(np.sum(eig0 < 0.0))

    eig, V = np.linalg.eigh(H)
    eig_min_raw = float(eig.min())
    eig_max_raw = float(eig.max())
    n_negative = int(np.sum(eig < 0.0))
    large_neg_thresh = 1e-6 * eig_max_raw
    n_large_negative = int(np.sum(eig < -large_neg_thresh))
    tiny_thresh = 1e-6 * eig_max_raw
    n_tiny = int(np.sum(np.abs(eig) < tiny_thresh))
    cond_raw_abs = float(np.abs(eig).max() / max(np.abs(eig).min(), 1e-30))

    print(f"  eig_min_raw={eig_min_raw:.6e}  eig_max_raw={eig_max_raw:.6e}", flush=True)
    print(f"  n_negative BEFORE (at start saddle) = {n_negative_before}  (expect 12)",
          flush=True)
    print(f"  n_negative AFTER  (<0)              = {n_negative}  (want ~0 -> PD mode)",
          flush=True)
    print(f"  n_large_negative (<-1e-6*max)       = {n_large_negative}", flush=True)
    print(f"  n_tiny (|lambda|<1e-6*max)          = {n_tiny}", flush=True)
    print(f"  cond(|eig|max/|eig|min)             = {cond_raw_abs:.6e}", flush=True)
    print(f"  most negative eigs: {np.sort(eig)[:8]}", flush=True)

    # ----- regularize to PD: floor eigenvalues to 1e-6 * lambda_max ---------------
    floor = 1e-6 * eig_max_raw
    eig_floored = np.maximum(eig, floor)
    H_reg = (V * eig_floored) @ V.T
    H_reg = 0.5 * (H_reg + H_reg.T)
    try:
        chol = np.linalg.cholesky(H_reg)
        chol_ok = True
    except np.linalg.LinAlgError:
        chol_ok = False
        chol = None
    eig_reg = np.linalg.eigvalsh(H_reg)
    cond_after = float(eig_reg.max() / eig_reg.min())
    print(f"  floor={floor:.6e}  cond_after_floor={cond_after:.6e}  chol_ok={chol_ok}",
          flush=True)

    # is_pd_mode: grad small relative to the f64-achievable floor AND no negative
    # eigenvalues at all (a genuine PD mode -- the whole point of float64).
    # Under x64 the grad should drop orders of magnitude below the f32 ~1.2e4
    # floor (target << 1e3, ideally < 10), so we require grad < 1e3 here AND a
    # fully non-negative spectrum (n_negative == 0).
    grad_thresh = 1e3 if x64 else 1e-2
    is_pd_mode = bool(grad_norm_after < grad_thresh and n_negative == 0)
    print(f"\n  is_pd_mode (grad<{grad_thresh:g} & NO negative eigs) = {is_pd_mode}",
          flush=True)

    # ----- TIME a single float64 target_log_prob + jax.grad eval ------------------
    # (after compile) so the sampling phase can size chains for the A16's slow FP64.
    lp_and_grad = jax.jit(jax.value_and_grad(lambda z: m.target_log_prob_fn(z)))
    lp_, g_ = lp_and_grad(qz_refined)
    g_.block_until_ready()  # warm compile
    n_timing = 20
    t0 = time.time()
    for _ in range(n_timing):
        lp_, g_ = lp_and_grad(qz_refined)
    g_.block_until_ready()
    sec_per_grad = (time.time() - t0) / n_timing
    print(f"\n  timing: {sec_per_grad*1e3:.3f} ms/eval  ({sec_per_grad:.6f} s) "
          f"for one target_log_prob + jax.grad ({fdtype.__name__ if hasattr(fdtype,'__name__') else fdtype}, "
          f"avg of {n_timing}, post-compile)", flush=True)

    # ----- save -------------------------------------------------------------------
    # Under --x64 write the f64 variants (map_refined_lstsq64.npz /
    # hess_refined_lstsq64.npz) so the f32 baseline is preserved for comparison.
    DATA.mkdir(parents=True, exist_ok=True)
    map_path = DATA / f"map_refined_lstsq{suffix}.npz"
    hess_path = DATA / f"hess_refined_lstsq{suffix}.npz"
    # In f64 keep the full-precision Hessian (f32 truncation would discard the
    # very conditioning float64 was enabled to capture).
    hess_store_dtype = np.float64 if x64 else np.float32
    np.savez(
        map_path,
        qz_refined=np.asarray(qz_refined, dtype=np.float64),
        qz_start=np.asarray(z0, dtype=np.float64),
        x64=np.bool_(x64),
        grad_dtype=np.array(grad_dtype),
        grad_norm_before=np.float64(grad_norm_before),
        grad_norm_after=np.float64(grad_norm_after),
        logp_before=np.float64(logp_before),
        logp_after=np.float64(logp_after),
        dist_from_start=np.float64(dist_from_start),
        gamma_refined=np.float64(gamma_refined),
        theta_E_refined=np.float64(theta_E_refined),
        gamma_start=np.float64(gamma_start),
        theta_E_start=np.float64(theta_E_start),
        sec_per_grad=np.float64(sec_per_grad),
        eig_raw=eig.astype(np.float64),
        n_negative_before=np.int64(n_negative_before),
        n_negative=np.int64(n_negative),
        n_large_negative=np.int64(n_large_negative),
        n_tiny=np.int64(n_tiny),
        cond_raw_abs=np.float64(cond_raw_abs),
        cond_after=np.float64(cond_after),
        is_pd_mode=np.bool_(is_pd_mode),
        amps=amps.astype(np.float64),
        n_neg_sersic_ie=np.int64(n_neg_sersic_ie),
        n_neg_shapelet=np.int64(n_neg_shapelet),
        reduced_index_labels=np.array(m.reduced_index_labels),
    )
    save_hess = dict(
        H_raw=H.astype(hess_store_dtype),
        H_reg=H_reg.astype(hess_store_dtype),
        x64=np.bool_(x64),
        eig_raw=eig.astype(np.float64),
        eig_floored=eig_floored.astype(np.float64),
        eig_min_raw=np.float64(eig_min_raw),
        eig_max_raw=np.float64(eig_max_raw),
        n_negative_eigs=np.int64(n_negative),
        n_large_negative=np.int64(n_large_negative),
        n_tiny=np.int64(n_tiny),
        eig_floor=np.float64(floor),
        cond_after_floor=np.float64(cond_after),
        chol_ok=np.bool_(chol_ok),
    )
    if chol is not None:
        save_hess["chol"] = chol.astype(hess_store_dtype)
    np.savez(hess_path, **save_hess)
    print(f"\nSaved {map_path}", flush=True)
    print(f"Saved {hess_path}", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
