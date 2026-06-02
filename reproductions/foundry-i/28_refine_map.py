"""Stage 4: refine the v7 "paper-mode MAP" saddle into a true stationary point.

BACKGROUND (Stages 1-3): The fixed-leapfrog PHMC kernel compiles and runs, but
HMC mixes terribly for EVERY momentum preconditioner (inv/diag/fwd/exact Hessian)
-- so the mass matrix is not the bottleneck. ROOT CAUSE: the Laplace Hessian of
-log_post at the v7 start qz_v7 (data/map_v7_paper_mode.npz['best_params']) has
15 LARGE negative eigenvalues. qz_v7 is a SADDLE, not a maximum. You cannot mix
an HMC from a saddle.

This script drives the gradient of -log_post down by orders of magnitude using a
strong optimizer (optax adabelief with a decaying LR + an adam polish), then
re-examines the Hessian at the refined point:
  - how many negative eigenvalues remain (was 15)?
  - how many near-zero (flat/degenerate) directions (the 31 linear shapelet amps)?
  - condition number?
  - is_true_mode (grad~0 and no large negative eigs)?
And the KEY physics question: does refining move gamma (EPL slope) toward the
paper's 1.372, or does it stay ~2.25 (confirming a distinct mode from the paper)?

Outputs:
  data/map_refined.npz  : qz_refined, grad_norm_before/after, logp_before/after,
                          dist_from_v7, gamma/theta_E at refined, eig spectrum.
  data/hess_refined.npz : H_raw, H_reg (floored to PD), eig spectrum, chol_ok.

Run (one idle A16):
  CUDA_DEVICE_ORDER=PCI_BUS_ID XLA_FLAGS=--xla_gpu_autotune_level=0 \
    CUDA_VISIBLE_DEVICES=1 /raid/benson/.venvs/gigalens/bin/python 28_refine_map.py
"""
import jax
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

import _hmc_lib

REPRO = Path(__file__).parent
DATA = REPRO / "data"


def main():
    print(f"devices: {jax.devices()}", flush=True)
    m = _hmc_lib.build_model()
    z0 = jnp.asarray(m.qz_start, dtype=jnp.float32)
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
    print(f"\nndim={ndim}", flush=True)
    print(f"BEFORE refine:  log_p={logp_before:.4f}  ||grad||={grad_norm_before:.6e}",
          flush=True)

    # ----- optimizer: adabelief w/ cosine-decay LR + grad clipping ---------------
    # The posterior is extremely stiff: eig_max(Hessian) ~ 1e11 along the linear
    # shapelet-amplitude directions, eig spread ~1e13. Plain large-LR steps blow up
    # in the stiff directions, so we (a) clip the global grad norm, (b) anneal the LR
    # to a small floor, (c) track the best point by ||grad|| (a low-loss point can
    # still have huge gradient in a stiff valley; for a stationary point we want
    # min gradient), and (d) run a long small-LR adam polish.
    def run_phase(z, opt, n_steps, label, log_every, best):
        best_z, best_f, best_gn = best
        opt_state = opt.init(z)

        @jax.jit
        def step(z, opt_state):
            updates, opt_state = opt.update(grad_jit(z), opt_state, z)
            z_new = optax.apply_updates(z, updates)
            # grad/loss evaluated AT the new iterate so the tracked best point
            # genuinely has the reported (low) gradient (no off-by-one step).
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

    # Phase 1: adabelief, cosine decay, grad-clip to 1e4.
    N1, PEAK_LR = 4000, 2e-3
    sched1 = optax.warmup_cosine_decay_schedule(
        init_value=PEAK_LR * 0.1, peak_value=PEAK_LR,
        warmup_steps=100, decay_steps=N1, end_value=PEAK_LR * 1e-2)
    opt1 = optax.chain(optax.clip_by_global_norm(1e4),
                       optax.adabelief(learning_rate=sched1))
    print(f"\n--- phase1: adabelief+clip, {N1} steps, peak_lr={PEAK_LR} ---", flush=True)
    z, best = run_phase(z0, opt1, N1, "p1", 200, best)

    # Phase 2: adam small fixed LR + tighter clip, from the best point so far.
    N2 = 3000
    opt2 = optax.chain(optax.clip_by_global_norm(1e3),
                       optax.adam(learning_rate=1e-4))
    print(f"\n--- phase2: adam lr=1e-4 + clip, {N2} steps ---", flush=True)
    z, best = run_phase(best[0], opt2, N2, "p2", 200, best)

    # Phase 3: adam tiny LR, final polish to minimize gradient.
    N3 = 2000
    opt3 = optax.chain(optax.clip_by_global_norm(3e2),
                       optax.adam(learning_rate=2e-5))
    print(f"\n--- phase3: adam lr=2e-5 + clip, {N3} steps ---", flush=True)
    z, best = run_phase(best[0], opt3, N3, "p3", 200, best)

    best_z, best_f, best_gn = best
    qz_refined = jnp.asarray(best_z, dtype=jnp.float32)
    logp_after = float(m.target_log_prob_fn(qz_refined))
    g_after = grad_jit(qz_refined)
    g_after.block_until_ready()
    grad_norm_after = float(jnp.linalg.norm(g_after))
    dist_from_v7 = float(jnp.linalg.norm(qz_refined - z0))

    print(f"\nAFTER refine:   log_p={logp_after:.4f}  ||grad||={grad_norm_after:.6e}",
          flush=True)
    print(f"  log_p improvement     = {logp_after - logp_before:+.4f}", flush=True)
    print(f"  ||grad|| reduction    = {grad_norm_before:.4e} -> {grad_norm_after:.4e} "
          f"(x{grad_norm_before/max(grad_norm_after,1e-30):.2e})", flush=True)
    print(f"  ||Delta|| from qz_v7  = {dist_from_v7:.6e}", flush=True)

    # ----- physical mass params at the refined point ------------------------------
    phys_v7 = _hmc_lib.to_physical_mass(np.asarray(z0)[None, :], prob_model=m.prob_model)
    phys_ref = _hmc_lib.to_physical_mass(np.asarray(qz_refined)[None, :],
                                         prob_model=m.prob_model)
    gamma_v7 = float(phys_v7["gamma"][0])
    theta_E_v7 = float(phys_v7["theta_E"][0])
    gamma_refined = float(phys_ref["gamma"][0])
    theta_E_refined = float(phys_ref["theta_E"][0])
    print(f"\nPhysical mass params (KEY QUESTION: did gamma move toward paper 1.372?):",
          flush=True)
    print(f"  {'param':>10s} {'qz_v7':>12s} {'refined':>12s} {'paper':>12s}", flush=True)
    print(f"  {'gamma':>10s} {gamma_v7:>12.4f} {gamma_refined:>12.4f} {1.372:>12.4f}",
          flush=True)
    print(f"  {'theta_E':>10s} {theta_E_v7:>12.4f} {theta_E_refined:>12.4f} "
          f"{2.6463:>12.4f}", flush=True)
    for k in ["e1", "e2", "gamma1", "gamma2"]:
        print(f"  {k:>10s} {float(phys_v7[k][0]):>12.4f} "
              f"{float(phys_ref[k][0]):>12.4f}", flush=True)

    # ----- Hessian at the refined point -------------------------------------------
    print(f"\n--- Hessian of -log_post at refined point ---", flush=True)
    t0 = time.time()
    H = np.asarray(jax.hessian(f)(qz_refined)).astype(np.float64)
    H = 0.5 * (H + H.T)  # symmetrize
    print(f"  hessian computed in {time.time()-t0:.1f}s", flush=True)

    eig, V = np.linalg.eigh(H)
    eig_min_raw = float(eig.min())
    eig_max_raw = float(eig.max())
    n_negative = int(np.sum(eig < 0.0))
    # "large negative": magnitude not negligible vs the spectrum scale.
    large_neg_thresh = 1e-6 * eig_max_raw
    n_large_negative = int(np.sum(eig < -large_neg_thresh))
    # near-zero / flat (degenerate) directions: |lambda| < 1e-6 * lambda_max.
    tiny_thresh = 1e-6 * eig_max_raw
    n_tiny = int(np.sum(np.abs(eig) < tiny_thresh))

    print(f"  eig_min_raw={eig_min_raw:.6e}  eig_max_raw={eig_max_raw:.6e}", flush=True)
    print(f"  n_negative (<0)            = {n_negative}  (was 15 at qz_v7 saddle)",
          flush=True)
    print(f"  n_large_negative (<-1e-6*max) = {n_large_negative}", flush=True)
    print(f"  n_tiny (|lambda|<1e-6*max) = {n_tiny}  (flat/degenerate, ~31 shapelet amps)",
          flush=True)
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

    # is_true_mode: grad ~ 0 AND no LARGE negative eigenvalues.
    is_true_mode = bool(grad_norm_after < 1e-2 and n_large_negative == 0)
    print(f"\n  is_true_mode (grad<1e-2 & no large neg eigs) = {is_true_mode}", flush=True)

    # ----- save -------------------------------------------------------------------
    DATA.mkdir(parents=True, exist_ok=True)
    np.savez(
        DATA / "map_refined.npz",
        qz_refined=np.asarray(qz_refined),
        qz_v7=np.asarray(z0),
        grad_norm_before=np.float64(grad_norm_before),
        grad_norm_after=np.float64(grad_norm_after),
        logp_before=np.float64(logp_before),
        logp_after=np.float64(logp_after),
        dist_from_v7=np.float64(dist_from_v7),
        gamma_refined=np.float64(gamma_refined),
        theta_E_refined=np.float64(theta_E_refined),
        gamma_v7=np.float64(gamma_v7),
        theta_E_v7=np.float64(theta_E_v7),
        eig_raw=eig.astype(np.float64),
        n_negative=np.int64(n_negative),
        n_large_negative=np.int64(n_large_negative),
        n_tiny=np.int64(n_tiny),
        cond_after=np.float64(cond_after),
        is_true_mode=np.bool_(is_true_mode),
    )
    save_hess = dict(
        H_raw=H.astype(np.float32),
        H_reg=H_reg.astype(np.float32),
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
        save_hess["chol"] = chol.astype(np.float32)
    np.savez(DATA / "hess_refined.npz", **save_hess)
    print(f"\nSaved {DATA / 'map_refined.npz'}", flush=True)
    print(f"Saved {DATA / 'hess_refined.npz'}", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
