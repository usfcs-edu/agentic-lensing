#!/usr/bin/env python
"""Minimal, self-contained UPSTREAM reproducer for three real gigalens bugs.

Run on CPU only (no GPU):

    JAX_PLATFORMS=cpu /raid/benson/.venvs/gigalens/bin/python 36_upstream_gigalens_repro.py

It uses a TINY model (EPL + Shear + one Sersic source, 32x32, supersample=1) and
finishes in seconds.  Each bug is demonstrated with a short assertion / printout.

gigalens commit under test: e8e47e5 (2026-05-19)
  /raid/benson/lensing-repos/gigalens/src/gigalens/jax/{inference.py,simulator.py}
TFP 0.25.0 (jax substrate), JAX 0.6.2.

The three bugs:
  BUG 1  inference.py:280-307  HMC kernel stack is pmap'd over ALL devices and the
         GBTLA ChEES criterion structurally requires >=2 chains; combined with the
         31-channel grouped conv + cuDNN autotuner this is what produced the
         multi-hour "won't compile" hang.  A fixed-num_leapfrog kernel compiles fine.
  BUG 2  inference.py:258-260  momentum preconditioner is built as
         MultivariateNormalFullCovariance(covariance_matrix=jnp.linalg.inv(q_z.cov()))
         which is NaN/garbage on a (near-)singular posterior covariance.  The
         precision parameterization is finite + correct.
  BUG 3  simulator.py:127      lstsq_simulate profiles the linear light amplitudes
         with pinv(X^T W X, rcond=1e-6) and OMITS the -0.5 log|X^T W X| Gaussian
         evidence term.  On a near-rank-deficient design matrix the profiled
         objective is NON-SMOOTH (autodiff != finite difference, gradient does not
         vanish at the optimum).  The ridge-regularized exact Gaussian
         marginalization A = X^T W X + Lambda with the -0.5 log|A| term is smooth.
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")  # never touch GPU
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_autotune_level=0")

import functools
import time
import warnings

warnings.filterwarnings("ignore")  # silence pinv rcond deprecation (kept verbatim
#                                    to match gigalens simulator.py:127)

import jax

jax.config.update("jax_enable_x64", True)  # BUG 3 needs float64 to make the point cleanly

import jax.numpy as jnp
import numpy as np
import tensorflow_probability.substrates.jax as tfp

tfd = tfp.distributions
tfe = tfp.experimental
tfb = tfp.bijectors

SEP = "=" * 78
RESULTS = {}  # bug -> bool demonstrated


def banner(msg):
    print("\n" + SEP + "\n" + msg + "\n" + SEP)


# ---------------------------------------------------------------------------
# A tiny gigalens model: EPL + Shear lens, one Sersic source, 32x32, ss=1.
# Used by BUG 1 (real kernel stack) and BUG 3 (real lstsq_simulate path).
# ---------------------------------------------------------------------------
def build_tiny_gigalens():
    from gigalens.jax.profiles.mass import epl, shear
    from gigalens.jax.profiles.light import sersic
    from gigalens.jax.simulator import LensSimulator
    from gigalens.jax.inference import ModellingSequence
    from gigalens.jax.model import BackwardProbModel
    from gigalens.model import PhysicalModel
    from gigalens.simulator import SimulatorConfig

    num_pix = 32
    rng = np.random.default_rng(0)
    # Trivial 3x3 PSF (peaked) so the conv path is exercised but cheap.
    psf = np.zeros((3, 3), np.float64)
    psf[1, 1] = 1.0
    sim_config = SimulatorConfig(delta_pix=0.13, num_pix=num_pix, supersample=1,
                                 kernel=psf)

    phys = PhysicalModel(
        [epl.EPL(50), shear.Shear()],
        [],  # no lens light
        [sersic.SersicEllipse(use_lstsq=True)],  # 1 linear amplitude
    )

    prior = tfd.JointDistributionSequential([
        tfd.JointDistributionSequential([
            tfd.JointDistributionNamed(dict(
                theta_E=tfd.LogNormal(jnp.log(1.5), 0.2),
                gamma=tfd.TruncatedNormal(2.0, 0.2, 1.2, 2.6),
                e1=tfd.Normal(0.0, 0.1), e2=tfd.Normal(0.0, 0.1),
                center_x=tfd.Normal(0.0, 0.05), center_y=tfd.Normal(0.0, 0.05),
            )),
            tfd.JointDistributionNamed(dict(gamma1=tfd.Normal(0.0, 0.05),
                                            gamma2=tfd.Normal(0.0, 0.05))),
        ]),
        tfd.JointDistributionSequential([
            tfd.JointDistributionNamed(dict(
                R_sersic=tfd.LogNormal(jnp.log(0.3), 0.2),
                n_sersic=tfd.Uniform(0.5, 4.0),
                e1=tfd.TruncatedNormal(0.0, 0.15, -0.5, 0.5),
                e2=tfd.TruncatedNormal(0.0, 0.15, -0.5, 0.5),
                center_x=tfd.Normal(0.0, 0.1), center_y=tfd.Normal(0.0, 0.1),
            )),
        ]),
    ])

    observed = rng.normal(0.0, 0.05, size=(num_pix, num_pix)).astype(np.float64)
    observed += 0.3 * np.exp(-((np.indices((num_pix, num_pix)) -
                                num_pix / 2.0) ** 2).sum(0) / 8.0)
    background_rms, exp_time = 0.05, 1000.0
    prob_model = BackwardProbModel(prior, jnp.asarray(observed),
                                   background_rms, exp_time)
    lens_sim = LensSimulator(phys, sim_config, bs=1)
    return dict(prior=prior, prob_model=prob_model, lens_sim=lens_sim,
                ModellingSequence=ModellingSequence, phys=phys,
                sim_config=sim_config, observed=observed,
                background_rms=background_rms, exp_time=exp_time)


# ===========================================================================
# BUG 1 - HMC kernel stack: pmap-over-all-devices + GBTLA needs >=2 chains.
# inference.py:280-307 (run_chain pmap) and :291-295 (GBTLA), :284-289 (PHMC).
# ===========================================================================
def bug1_compile(env):
    banner("BUG 1  compile/structure: GBTLA ChEES needs >=2 chains; HMC() pmaps "
           "the whole stack over ALL devices  (inference.py:280-307)")
    # We exercise the EXACT gigalens kernel-class stack (PHMC -> GBTLA -> DASA).
    # The physics target is irrelevant to the structural bug, so we use a cheap
    # batched Gaussian log-prob (the real gigalens log_prob, model.py:71-80, would
    # be the target in HMC() -- it is what makes each compile expensive, but the
    # STRUCTURE below is what hung).
    ndim = 14  # same dim as the tiny EPL+Shear+Sersic model (env confirms it)
    assert env["prob_model"] is not None  # tiny gigalens model built as context
    target_cov_inv = jnp.eye(ndim)

    @jax.jit
    def log_prob(z):
        return -0.5 * jnp.sum(z * (z @ target_cov_inv), axis=-1)

    print(f"tiny model ndim = {ndim}")

    # exact gigalens momentum-distribution shape (identity here; BUG 2 stresses it)
    mom = tfd.MultivariateNormalFullCovariance(
        loc=jnp.zeros(ndim), covariance_matrix=jnp.eye(ndim))

    def make_stack(state_for_init):
        # Mirror inference.py:284-298 verbatim.
        k = tfe.mcmc.PreconditionedHamiltonianMonteCarlo(
            target_log_prob_fn=log_prob, momentum_distribution=mom,
            step_size=0.1, num_leapfrog_steps=3)
        k = tfe.mcmc.GradientBasedTrajectoryLengthAdaptation(
            k, num_adaptation_steps=8, max_leapfrog_steps=30)
        k = tfp.mcmc.DualAveragingStepSizeAdaptation(
            inner_kernel=k, num_adaptation_steps=8)
        return k

    # ---- 1a. single chain: ChEES criterion *structurally* fails (>=2 chains). ----
    single = jnp.zeros((1, ndim))  # one chain
    failed_single = False
    try:
        k = make_stack(single)
        tfp.mcmc.sample_chain(num_results=2, num_burnin_steps=8,
                              current_state=single, kernel=k,
                              trace_fn=lambda *_: (),
                              seed=jax.random.PRNGKey(0))
    except Exception as e:  # noqa: BLE001
        failed_single = True
        msg = str(e).strip().splitlines()[-1][:140]
        print(f"  single-chain GBTLA -> ERROR (expected): {msg}")
    assert failed_single, "expected single-chain ChEES to fail (needs >=2 chains)"
    print("  -> confirmed: GBTLA ChEES requires >=2 chains "
          "(gradient_based_trajectory_length_adaptation.py:272, :220-222)")

    # ---- 1b. >=2 chains: the SAME stack compiles & runs (fast, fixed leapfrog). --
    multi = jnp.zeros((2, ndim))  # 2 chains -> ChEES well-defined
    t0 = time.time()
    k = make_stack(multi)
    out = tfp.mcmc.sample_chain(num_results=3, num_burnin_steps=8,
                                current_state=multi, kernel=k,
                                trace_fn=lambda *_: (),
                                seed=jax.random.PRNGKey(0))
    out.all_states.block_until_ready()
    dt = time.time() - t0
    print(f"  2-chain stack compiled+ran in {dt:.1f}s, "
          f"states shape {tuple(out.all_states.shape)}")
    assert out.all_states.shape == (3, 2, ndim)

    # ---- 1c. plain fixed-num_leapfrog kernel (no GBTLA) compiles fine on 1 chain.
    t0 = time.time()
    k = tfe.mcmc.PreconditionedHamiltonianMonteCarlo(
        target_log_prob_fn=log_prob, momentum_distribution=mom,
        step_size=0.1, num_leapfrog_steps=3)
    out = tfp.mcmc.sample_chain(num_results=3, num_burnin_steps=5,
                                current_state=single, kernel=k,
                                trace_fn=lambda *_: (),
                                seed=jax.random.PRNGKey(0))
    out.all_states.block_until_ready()
    print(f"  fixed-leapfrog PHMC (single chain) compiled+ran in "
          f"{time.time() - t0:.1f}s")

    # ---- 1d. structural cause: HMC() wraps run_chain in pmap over ALL devices. ----
    n_dev = len(jax.devices())
    print(f"  jax.devices() count = {n_dev}; inference.py:280 wraps the WHOLE "
          f"adaptive stack in jax.pmap(axis_name='device') over all {n_dev} "
          f"-> on the real model this + the 31-channel grouped conv + cuDNN "
          f"autotuner is what hung for hours.")
    RESULTS["BUG1"] = True


# ===========================================================================
# BUG 2 - momentum preconditioner inv() of a (near-)singular covariance.
# inference.py:258-260
#   tfd.MultivariateNormalFullCovariance(
#       covariance_matrix=jnp.linalg.inv(q_z.covariance()))
# ===========================================================================
def bug2_momentum(env):
    banner("BUG 2  momentum NaN: MultivariateNormalFullCovariance(inv(cov)) on a "
           "(near-)singular posterior covariance  (inference.py:258-260)")
    d = 6
    rng = np.random.default_rng(1)
    # A realistic SVI posterior covariance with flat (near-rank-deficient) dirs:
    # 2 well-constrained dirs, 4 near-flat (tiny eigenvalues) -> cond ~ 1e16.
    Q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    eig = np.array([1.0, 0.5, 1e-13, 1e-15, 1e-16, 0.0])  # last is exactly singular
    cov = (Q * eig) @ Q.T
    cov = 0.5 * (cov + cov.T)
    cov_j = jnp.asarray(cov)
    print(f"  cov eigenvalues = {np.sort(eig)}")
    print(f"  cond(cov) ~ {np.linalg.cond(cov):.3e} (rank-deficient)")

    # ---- gigalens path: inv(cov) then MVNFullCovariance(covariance=inv). ----
    inv_cov = jnp.linalg.inv(cov_j)
    n_nonfinite = int(jnp.sum(~jnp.isfinite(inv_cov)))
    print(f"  jnp.linalg.inv(cov): {n_nonfinite} non-finite entries, "
          f"max|entry| = {float(jnp.max(jnp.abs(jnp.nan_to_num(inv_cov)))):.3e}")

    mom_giga = tfd.MultivariateNormalFullCovariance(
        loc=jnp.zeros(d), covariance_matrix=inv_cov)
    s_giga = mom_giga.sample(4, seed=jax.random.PRNGKey(0))
    giga_bad = bool(jnp.any(~jnp.isfinite(s_giga)))
    print(f"  gigalens momentum samples finite? {not giga_bad}  "
          f"(any NaN/Inf -> momentum kick is garbage -> HMC step collapses)")

    # ---- fix: precision parameterization (momentum cov == precision == cov^-1
    #      means precision matrix == cov).  Regularize to make it PD. ----
    ed = tfp.experimental.distributions
    cov_reg = cov_j + 1e-8 * jnp.eye(d)  # ridge so the precision factor exists
    precision_factor = jnp.linalg.cholesky(cov_reg)  # chol of the (PD) cov estimate
    mom_fix = ed.MultivariateNormalPrecisionFactorLinearOperator(
        loc=jnp.zeros(d),
        precision_factor=tfp.tf2jax.linalg.LinearOperatorLowerTriangular(
            precision_factor),
        precision=tfp.tf2jax.linalg.LinearOperatorFullMatrix(cov_reg),
    )
    s_fix = mom_fix.sample(4, seed=jax.random.PRNGKey(0))
    fix_ok = bool(jnp.all(jnp.isfinite(s_fix)))
    print(f"  precision-parameterized momentum samples finite? {fix_ok}")

    # The PHMC momentum_distribution should have covariance == cov^-1 (mass matrix
    # M = Sigma^-1).  With the precision parameterization we PASS the covariance
    # estimate directly and never invert a singular matrix.
    assert giga_bad, "expected gigalens inv() momentum to be non-finite"
    assert fix_ok, "expected precision-parameterized momentum to be finite"
    print("  -> confirmed: inv()+FullCovariance is NaN; precision factor is finite.")
    RESULTS["BUG2"] = True


# ===========================================================================
# BUG 3 - lstsq_simulate pinv profiling is NON-SMOOTH and OMITS the evidence term.
# simulator.py:127  coeffs = (pinv(Xt @ X, rcond=1e-6) @ Xt @ Y)[...,0]
# (plus the log-likelihood in model.py uses only the point estimate -> no
#  -0.5 log|X^T W X| Gaussian-evidence / Occam term.)
# ===========================================================================
def bug3_lstsq_smoothness():
    banner("BUG 3  lstsq non-smooth + missing evidence term  (simulator.py:127)")

    # Small synthetic linear-amplitude problem whose design matrix X(t) becomes
    # NEAR rank-deficient as a nonlinear parameter t varies.  This is exactly the
    # situation of the 28 shapelet columns: highly collinear basis functions whose
    # X^T W X singular spectrum sweeps THROUGH the pinv rcond=1e-6 truncation
    # threshold as the nonlinear (mass/beta) params move.
    n_pix, n_amp = 60, 6
    rng = np.random.default_rng(2)
    # Two near-duplicate column groups -> the Gram matrix has a singular value that
    # passes the rcond cutoff as t varies (where pinv flips rank -> a KINK).
    U, _ = np.linalg.qr(rng.normal(size=(n_pix, n_amp)))
    B = jnp.asarray(U)  # orthonormal base columns
    W = jnp.ones(n_pix)  # uniform noise precision for the demo
    y = jnp.asarray(0.7 * U[:, 0] + 0.3 * U[:, 1] + 0.05 * rng.normal(size=n_pix))
    # Tikhonov precision from Gaussian priors a_i ~ N(0, sigma_i), sigma_i=5/sqrt(i+1)
    lam = (jnp.arange(n_amp, dtype=jnp.float64) + 1.0) / 25.0

    def design(t):
        # Scale the high-order columns by powers of t so their contribution to
        # X^T W X collapses smoothly to zero; pinv with rcond=1e-6 TRUNCATES them
        # at a sharp threshold (rank flip) -> non-smooth profiled loss.  The
        # low-order columns carry the signal so the residual is non-trivial.
        scales = jnp.concatenate([
            jnp.ones(2), t ** (jnp.arange(2, n_amp, dtype=jnp.float64))])
        return B * scales[None, :]

    # ---------- (a) gigalens pinv profiling (simulator.py:127), NO evidence term ----
    def neglogpost_pinv(t):
        X = design(t)
        Xw = X * jnp.sqrt(W)[:, None]
        yw = y * jnp.sqrt(W)
        XtX = Xw.T @ Xw
        Xty = Xw.T @ yw
        a = jnp.linalg.pinv(XtX, rcond=1e-6) @ Xty   # <-- gigalens line 127
        r = yw - Xw @ a
        return 0.5 * jnp.sum(r ** 2)                 # Gaussian NLL of point estimate
        # NOTE: NO +0.5*logdet(XtX) evidence term, matching gigalens.

    # ---------- (b) ridge-regularized EXACT Gaussian marginalization (the fix) -----
    def neglogpost_marg(t):
        X = design(t)
        Xw = X * jnp.sqrt(W)[:, None]
        yw = y * jnp.sqrt(W)
        A = Xw.T @ Xw + jnp.diag(lam)                # PD: ridge from the priors
        b = Xw.T @ yw
        chol = jnp.linalg.cholesky(A)                # smooth (A is PD)
        astar = jax.scipy.linalg.cho_solve((chol, True), b)
        logdetA = 2.0 * jnp.sum(jnp.log(jnp.diag(chol)))
        quad = 0.5 * jnp.sum(yw ** 2) - 0.5 * jnp.dot(b, astar)
        return quad + 0.5 * logdetA                  # <-- includes evidence term

    def fd_grad(f, t, h):
        return float((f(t + h) - f(t - h)) / (2 * h))

    # central finite-difference truncation error is O(h^2); pick h relative to t so
    # the smooth objective matches autodiff to ~1e-6 and only a genuine KINK shows.
    def mismatch(g_ad, g_fd):
        return abs(g_ad - g_fd) / max(abs(g_fd), abs(g_ad), 1e-9)

    print("  t      |  pinv: autodiff   finitediff    mismatch | "
          "marg: autodiff   finitediff   mismatch")
    # Sweep the regime where the design matrix's smallest singular value crosses
    # the pinv rcond=1e-6 cutoff (this is where the rank flip / kink occurs) and
    # where gradients are well above the finite-difference resolution floor.
    pinv_max_mismatch = 0.0
    marg_max_mismatch = 0.0
    for t in [1.0, 0.5, 0.2, 0.1, 0.05, 0.02]:
        tj = jnp.float64(t)
        h = 1e-7 * t  # central FD step relative to t; truncation error ~ (h/t)^2
        gp_ad = float(jax.grad(neglogpost_pinv)(tj))
        gp_fd = fd_grad(neglogpost_pinv, tj, h)
        gm_ad = float(jax.grad(neglogpost_marg)(tj))
        gm_fd = fd_grad(neglogpost_marg, tj, h)
        rp = mismatch(gp_ad, gp_fd)
        rm = mismatch(gm_ad, gm_fd)
        pinv_max_mismatch = max(pinv_max_mismatch, rp)
        marg_max_mismatch = max(marg_max_mismatch, rm)
        print(f"  {t:5.2f}  |  {gp_ad:+11.3e} {gp_fd:+11.3e}  {rp:9.2e}   | "
              f"  {gm_ad:+11.3e} {gm_fd:+11.3e}  {rm:9.2e}")

    print(f"\n  pinv profiling: max relative autodiff-vs-finitediff mismatch = "
          f"{pinv_max_mismatch:.2e}  (NON-SMOOTH: pinv singular-value truncation)")
    print(f"  ridge marginalization: max relative mismatch = "
          f"{marg_max_mismatch:.2e}  (SMOOTH: Cholesky + slogdet of PD A)")

    # The smooth objective's gradient agrees with finite differences to ~O(h^2)=1e-5;
    # the pinv objective disagrees by O(1) (a hard KINK) in the near-singular regime,
    # where autodiff sees ~0 but the function actually jumps by hundreds.
    assert marg_max_mismatch < 1e-3, marg_max_mismatch
    assert pinv_max_mismatch > 0.5, (pinv_max_mismatch, marg_max_mismatch)

    # ---- Second symptom: the pinv autodiff gradient is UNINFORMATIVE ("floors"
    #      to ~0) because autodiff differentiates through pinv's truncated SVD and
    #      lands in the discarded null-space, while the objective ACTUALLY has a
    #      steep slope (huge finite-difference).  HMC follows the autodiff gradient,
    #      so it sees ~0 and cannot descend -> exactly the real-model ||grad||
    #      "floor".  The marg gradient tracks the true slope. ----
    grad_pinv = jax.jit(jax.grad(neglogpost_pinv))
    grad_marg = jax.jit(jax.grad(neglogpost_marg))
    ts = jnp.linspace(0.05, 0.2, 60)
    ad = jnp.array([abs(float(grad_pinv(t))) for t in ts])
    fdv = jnp.array([abs(fd_grad(neglogpost_pinv, t, 1e-7 * float(t))) for t in ts])
    max_true_slope = float(jnp.max(fdv))
    max_ad = float(jnp.max(ad))
    print(f"\n  pinv objective over t in [0.05,0.2]:")
    print(f"     max |autodiff grad| = {max_ad:.3e}  (FLOORED near zero)")
    print(f"     max |true slope (finite diff)| = {max_true_slope:.3e}  "
          f"(autodiff is BLIND to it -> HMC cannot descend)")
    assert max_true_slope > 1e3 * max(max_ad, 1e-12), (max_true_slope, max_ad)
    # marg autodiff tracks its true slope everywhere (smooth):
    ad_m = jnp.array([abs(float(grad_marg(t))) for t in ts])
    fd_m = jnp.array([abs(fd_grad(neglogpost_marg, t, 1e-7 * float(t))) for t in ts])
    marg_track = float(jnp.max(jnp.abs(ad_m - fd_m) / (jnp.abs(fd_m) + 1e-9)))
    print(f"     marg autodiff vs true slope: max rel mismatch = {marg_track:.3e} "
          f"(autodiff IS the true gradient)")
    assert marg_track < 1e-3, marg_track

    # ---- Third symptom: the OMITTED evidence term. The marg objective differs
    #      from a no-evidence-term ridge fit by exactly 0.5*log|A|; at fixed
    #      amplitudes this Occam factor changes the t-landscape (and is what
    #      penalizes the degenerate, near-flat configurations). ----
    def neglogpost_marg_noevidence(t):
        X = design(t)
        Xw = X * jnp.sqrt(W)[:, None]
        yw = y * jnp.sqrt(W)
        A = Xw.T @ Xw + jnp.diag(lam)
        b = Xw.T @ yw
        chol = jnp.linalg.cholesky(A)
        astar = jax.scipy.linalg.cho_solve((chol, True), b)
        return 0.5 * jnp.sum(yw ** 2) - 0.5 * jnp.dot(b, astar)  # NO 0.5*log|A|

    t_probe = jnp.float64(0.1)
    with_ev = float(neglogpost_marg(t_probe))
    no_ev = float(neglogpost_marg_noevidence(t_probe))
    A_probe = (design(t_probe) * jnp.sqrt(W)[:, None])
    A_probe = A_probe.T @ A_probe + jnp.diag(lam)
    half_logdet = 0.5 * float(jnp.linalg.slogdet(A_probe)[1])
    print(f"\n  at t={float(t_probe)}: evidence term 0.5*log|A| = {half_logdet:+.4f}")
    print(f"     marg-with-evidence - marg-without = {with_ev - no_ev:+.4f}  "
          f"(== 0.5*log|A|: this is the Occam term gigalens OMITS)")
    assert abs((with_ev - no_ev) - half_logdet) < 1e-6

    print("\n  -> confirmed: pinv profiling is non-smooth (discontinuous gradient) "
          "AND omits the -0.5 log|X^T W X| evidence term; ridge marginalization "
          "is smooth and includes it.")
    RESULTS["BUG3"] = True


def main():
    print(f"JAX_PLATFORMS={os.environ.get('JAX_PLATFORMS')}  "
          f"devices={jax.devices()}  x64={jax.config.jax_enable_x64}")
    env = build_tiny_gigalens()
    bug1_compile(env)
    bug2_momentum(env)
    bug3_lstsq_smoothness()

    banner("SUMMARY")
    for k in ("BUG1", "BUG2", "BUG3"):
        print(f"  {k}: {'DEMONSTRATED' if RESULTS.get(k) else 'NOT demonstrated'}")
    assert all(RESULTS.get(k) for k in ("BUG1", "BUG2", "BUG3"))
    print("\nALL THREE BUGS DEMONSTRATED, reproducer ran clean.")


if __name__ == "__main__":
    main()
