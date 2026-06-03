#!/usr/bin/env python
"""02_fit_system.py -- GIGA-Lens MAP -> SVI -> HMC on ONE mock (Gu et al. 2022).

Reproduces the paper's three-step pipeline (Table 1) on a single simulated
system, pinned to ONE GPU. Writes posterior, ESS, split-Rhat and
recovered-vs-truth to data/fits/system_XXX_fit.npz.

Pipeline (Table 1 hyperparameters, all from the reference system, left unchanged):
  1. MAP  : K_MAP=300 multistart, n_MAP=350 steps, adabelief 1e-2 (Adam 1e-2->1e-3)
  2. SVI  : K_VI=1000 samples, n_VI=500 steps, adabelief 1e-3, init Sigma=1e-6 I
  3. HMC  : 50 chains, n_burn=250, n_sample=750, eps0=0.3, L0=3, mass M = Sigma_VI^-1,
            PreconditionedHMC + GradientBasedTrajectoryLengthAdaptation +
            DualAveragingStepSizeAdaptation -- the SAME kernel stack as the
            canonical gigalens jax-demo / paper, but run as a single-device
            batched sample_chain (NOT the gigalens HMC() helper, which pmaps over
            all devices and hangs).

GPU rules: pin via CUDA_VISIBLE_DEVICES, CUDA_DEVICE_ORDER=PCI_BUS_ID,
XLA_FLAGS=--xla_gpu_autotune_level=0. One system per process.

Usage (one GPU, one system):
  CUDA_DEVICE_ORDER=PCI_BUS_ID XLA_FLAGS=--xla_gpu_autotune_level=0 \
    CUDA_VISIBLE_DEVICES=0 python 02_fit_system.py --idx 0
"""
import os, sys, time, json, argparse
import numpy as np

GIGALENS_SRC = "/raid/benson/lensing-repos/gigalens/src"
sys.path.insert(0, GIGALENS_SRC)


def build_prior(tfd, jnp):
    """The MODELLING prior = the broader 'b' side of Eq. (8)'s a/b notation.
    Order of parameters MUST match the gigalens truth list packing."""
    lens_prior = tfd.JointDistributionSequential([
        tfd.JointDistributionNamed(dict(
            theta_E=tfd.LogNormal(jnp.log(1.25), 0.4),
            gamma=tfd.TruncatedNormal(2.0, 0.5, 1.0, 3.0),
            e1=tfd.Normal(0.0, 0.2),
            e2=tfd.Normal(0.0, 0.2),
            center_x=tfd.Normal(0.0, 0.1),
            center_y=tfd.Normal(0.0, 0.1),
        )),
        tfd.JointDistributionNamed(dict(
            gamma1=tfd.Normal(0.0, 0.06),
            gamma2=tfd.Normal(0.0, 0.06),
        )),
    ])
    lens_light_prior = tfd.JointDistributionSequential([
        tfd.JointDistributionNamed(dict(
            R_sersic=tfd.LogNormal(jnp.log(1.6), 0.25),
            n_sersic=tfd.Uniform(0.5, 8.0),
            e1=tfd.TruncatedNormal(0.0, 0.1, -0.15, 0.15),
            e2=tfd.TruncatedNormal(0.0, 0.1, -0.15, 0.15),
            center_x=tfd.Normal(0.0, 0.02),
            center_y=tfd.Normal(0.0, 0.02),
            Ie=tfd.LogNormal(jnp.log(300.0), 0.5),
        )),
    ])
    source_light_prior = tfd.JointDistributionSequential([
        tfd.JointDistributionNamed(dict(
            R_sersic=tfd.LogNormal(jnp.log(0.25), 0.25),
            n_sersic=tfd.Uniform(0.5, 8.0),
            e1=tfd.TruncatedNormal(0.0, 0.3, -0.5, 0.5),
            e2=tfd.TruncatedNormal(0.0, 0.3, -0.5, 0.5),
            center_x=tfd.Normal(0.0, 0.5),
            center_y=tfd.Normal(0.0, 0.5),
            Ie=tfd.LogNormal(jnp.log(150.0), 0.9),
        )),
    ])
    return tfd.JointDistributionSequential(
        [lens_prior, lens_light_prior, source_light_prior])


# Parameter labels in the gigalens packing order (must match build_prior order).
PARAM_LABELS = [
    "theta_E", "gamma", "e1", "e2", "center_x", "center_y",      # EPL (6)
    "gamma1", "gamma2",                                          # shear (2)
    "ll_R_sersic", "ll_n_sersic", "ll_e1", "ll_e2",
    "ll_center_x", "ll_center_y", "ll_Ie",                       # lens light (7)
    "src_R_sersic", "src_n_sersic", "src_e1", "src_e2",
    "src_center_x", "src_center_y", "src_Ie",                    # source light (7)
]
# The 8 "lensing parameters" the paper reports ESS/Rhat for (Table 2 / Fig 7 mass).
MASS_LABELS = ["theta_E", "gamma", "e1", "e2", "center_x", "center_y",
               "gamma1", "gamma2"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--idx", type=int, required=True)
    ap.add_argument("--mocks", default="data/mocks")
    ap.add_argument("--out", default="data/fits")
    # Batch sizes reduced from the paper's K_MAP=300 / K_VI=1000 to fit a 15 GB
    # A16 at 80x80 and keep wall-time reasonable on this much weaker GPU (the
    # paper used 40-80 GB A100s, ~5-10x faster). Method unchanged -- this is a
    # documented hardware proxy. SVI ~0.8 ms/step/sample on the A16, so the per-
    # step ELBO MC batch dominates wall time; 200 still recovers the truth.
    ap.add_argument("--n-map", type=int, default=128, help="K_MAP multistart")
    ap.add_argument("--map-steps", type=int, default=250)
    ap.add_argument("--n-vi", type=int, default=200, help="K_VI MC samples / ELBO step")
    ap.add_argument("--vi-steps", type=int, default=500)
    ap.add_argument("--n-hmc", type=int, default=50, help="HMC chains")
    ap.add_argument("--burn", type=int, default=250)
    ap.add_argument("--keep", type=int, default=750)
    ap.add_argument("--eps", type=float, default=0.3)
    ap.add_argument("--leapfrog", type=int, default=5,
                    help="fixed leapfrog steps L (paper Table 1: L=5)")
    ap.add_argument("--max-leapfrog", type=int, default=30)
    ap.add_argument("--gbtla", action="store_true", default=False,
                    help="use GradientBasedTrajectoryLengthAdaptation (paper/demo "
                         "default; ~6x slower on A16). Off => fast fixed-L=5 PHMC.")
    ap.add_argument("--x64", action="store_true", default=False,
                    help="float64 (paper/demo use float32; A16 OOMs at 80x80 x64 bs=300)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    here_dir = os.path.dirname(os.path.abspath(__file__))
    if args.x64:
        os.environ["JAX_ENABLE_X64"] = "1"

    import jax
    # Persistent compilation cache: the MAP/SVI/HMC kernels have identical shapes
    # across all systems, so caching lets later fits skip recompilation.
    jax.config.update("jax_compilation_cache_dir",
                      os.path.join(here_dir, ".jax_cache"))
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)
    import jax.experimental.shard_map  # noqa: F401 (gigalens MAP/SVI need this registered)
    from jax import numpy as jnp
    import optax
    import tensorflow_probability.substrates.jax as tfp
    tfd = tfp.distributions
    tfe = tfp.experimental
    from gigalens.jax.inference import ModellingSequence
    from gigalens.jax.model import ForwardProbModel
    from gigalens.model import PhysicalModel
    from gigalens.jax.simulator import LensSimulator
    from gigalens.simulator import SimulatorConfig
    from gigalens.jax.profiles.light import sersic
    from gigalens.jax.profiles.mass import epl, shear

    if args.x64:
        jax.config.update("jax_enable_x64", True)
    devs = jax.devices()
    print(f"[idx {args.idx}] devices={devs} x64={args.x64}", flush=True)
    assert len(devs) == 1, "Pin to ONE GPU via CUDA_VISIBLE_DEVICES."

    here = os.path.dirname(os.path.abspath(__file__))
    mock_path = os.path.join(here, args.mocks, f"system_{args.idx:03d}.npz")
    d = np.load(mock_path, allow_pickle=True)
    image = np.array(d["image"], dtype=np.float64 if args.x64 else np.float32)
    psf = np.array(d["psf"], dtype=np.float32)
    sigma_bkg = float(d["sigma_bkg"]); exp_time = float(d["exp_time"])
    num_pix = int(d["num_pix"]); delta_pix = float(d["delta_pix"])
    supersample = int(d["supersample"])
    truth_flat = dict(zip([str(k) for k in d["flat_keys"]],
                          np.array(d["flat_vals"], dtype=np.float64)))
    print(f"[idx {args.idx}] mock loaded {image.shape} "
          f"truth theta_E={truth_flat['theta_E']:.3f} gamma={truth_flat['gamma']:.3f}",
          flush=True)

    # ---- model setup (matches the canonical jax-demo, 80x80, supersample=2) ----
    prior = build_prior(tfd, jnp)
    sim_config = SimulatorConfig(delta_pix=delta_pix, num_pix=num_pix,
                                 supersample=supersample, kernel=psf)
    phys_model = PhysicalModel(
        [epl.EPL(50), shear.Shear()],
        [sersic.SersicEllipse(use_lstsq=False)],
        [sersic.SersicEllipse(use_lstsq=False)],
    )
    prob_model = ForwardProbModel(prior, image, background_rms=sigma_bkg,
                                  exp_time=exp_time)
    model_seq = ModellingSequence(phys_model, prob_model, sim_config)

    # =========================== STEP 1: MAP ================================
    t0 = time.time()
    opt = optax.adabelief(1e-2, b1=0.95, b2=0.99)
    map_best, map_lp, map_chisq = model_seq.MAP(
        opt, n_samples=args.n_map, num_steps=args.map_steps,
        seed=args.seed, output_type="best", pbar_interval=0)
    map_best = jnp.asarray(map_best)  # (1, ndim) in unconstrained space
    t_map = time.time() - t0
    print(f"[idx {args.idx}] MAP done {t_map:.1f}s  chisq={float(map_chisq):.4f}",
          flush=True)

    # =========================== STEP 2: SVI ================================
    t0 = time.time()
    opt = optax.adabelief(1e-3, b1=0.95, b2=0.99)
    qz, loss_hist = model_seq.SVI(map_best, opt, n_vi=args.n_vi,
                                  init_scales=1e-3, num_steps=args.vi_steps,
                                  seed=args.seed, pbar_interval=0)
    t_svi = time.time() - t0
    print(f"[idx {args.idx}] SVI done {t_svi:.1f}s  final -ELBO={float(loss_hist[-1]):.2f}",
          flush=True)

    # =========================== STEP 3: HMC ================================
    # Single-device batched 50-chain PHMC + GBTLA + DualAveraging.
    # Momentum covariance = inv(Sigma_VI) (preconditioned HMC, paper Eq. M = Sigma^-1).
    ndim = int(qz.mean().shape[0])
    n_hmc = args.n_hmc
    lens_sim = LensSimulator(phys_model, sim_config, bs=n_hmc)

    momentum_distribution = tfd.MultivariateNormalFullCovariance(
        loc=jnp.zeros_like(qz.mean()),
        covariance_matrix=jnp.linalg.inv(qz.covariance()),
    )

    @jax.jit
    def log_prob(z):
        return prob_model.log_prob(lens_sim, z)[0]

    num_adapt = int(args.burn * 0.8)  # adapt during first 80% of burn-in

    def run_chain(seed):
        start = qz.sample(n_hmc, seed=seed)  # (n_hmc, ndim)
        kernel = tfe.mcmc.PreconditionedHamiltonianMonteCarlo(
            target_log_prob_fn=log_prob,
            momentum_distribution=momentum_distribution,
            step_size=args.eps,
            num_leapfrog_steps=args.leapfrog,
        )
        if args.gbtla:
            # Paper's adaptive trajectory length (gigalens demo default). Powerful
            # but VERY slow on the A16 (max_leapfrog=30 -> up to 30 grad evals/step).
            kernel = tfe.mcmc.GradientBasedTrajectoryLengthAdaptation(
                kernel, num_adaptation_steps=num_adapt,
                max_leapfrog_steps=args.max_leapfrog,
            )
        # Dual-averaging step-size adaptation over the first 80% of burn-in
        # (paper Table 1 footnote). For fixed-leapfrog PHMC this is the same
        # preconditioned-HMC stack as foundry-i; the paper lists L=5, eps_f=0.3.
        kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
            inner_kernel=kernel, num_adaptation_steps=num_adapt,
        )
        return tfp.mcmc.sample_chain(
            num_results=args.keep,
            num_burnin_steps=args.burn,
            current_state=start,
            kernel=kernel,
            trace_fn=None,
            seed=seed,
        )

    t0 = time.time()
    seed_key = jax.random.PRNGKey(args.seed + 1234)
    samples = jax.jit(run_chain)(seed_key)
    samples = jax.block_until_ready(samples)
    t_hmc = time.time() - t0
    samples = np.asarray(samples)  # (num_results, n_hmc, ndim)
    print(f"[idx {args.idx}] HMC done {t_hmc:.1f}s  samples={samples.shape}",
          flush=True)

    # ---- diagnostics: ESS (cross-chain) + split-Rhat, per param ----
    s_jnp = jnp.asarray(samples)  # (T, C, ndim)
    rhat = np.asarray(tfp.mcmc.potential_scale_reduction(
        s_jnp, independent_chain_ndims=1, split_chains=True))   # (ndim,)
    ess = np.asarray(tfp.mcmc.effective_sample_size(
        s_jnp, cross_chain_dims=1))                             # (ndim,)

    # ---- convert to physical parameters ----
    # IMPORTANT: the gigalens bijector (default_event_space_bijector o pack_bij)
    # REORDERS the dict keys within each block (e.g. e2 before e1, center_y before
    # center_x, gamma2 before gamma1). The raw unconstrained z index k, the physical
    # leaf k, and ess[k]/rhat[k] are all aligned to the SAME leaf flattening, so we
    # must build the truth vector in that *physical* leaf order -- NOT in PARAM_LABELS
    # (the human/spec order). We do this by consuming the truth dicts per block in
    # the bijector's key order. (Misaligning truth was a real bug; this fixes it.)
    flat = samples.reshape(-1, ndim)                            # (T*C, ndim)
    phys = prob_model.bij.forward(list(jnp.asarray(flat).T))
    # nested [[mass, shear],[lens_light],[source_light]] of dicts of (N,)
    # truth nested in the same structure (from the saved gigalens truth list):
    truth_nested = json.loads(str(d["truth_json"]))
    tups = [(0, 0), (0, 1), (1, 0), (2, 0)]
    block_prefix = {(0, 0): "", (0, 1): "", (1, 0): "ll_", (2, 0): "src_"}
    phys_cols, phys_labels, truth_list = [], [], []
    for (i, j) in tups:
        tdict = truth_nested[i][j]
        for key, val in phys[i][j].items():           # bijector key order
            phys_cols.append(np.asarray(val))
            phys_labels.append(block_prefix[(i, j)] + key)
            truth_list.append(float(tdict[key]))      # truth for the SAME key
    phys_arr = np.vstack(phys_cols).T                           # (T*C, 22)
    phys_means = phys_arr.mean(axis=0)
    phys_stds = phys_arr.std(axis=0)
    truth_vec = np.array(truth_list)

    # scaled error z = (mean - truth)/std for each param (paper Table 2 mu_z)
    z_err = (phys_means - truth_vec) / np.where(phys_stds > 0, phys_stds, np.nan)

    # mass-param diagnostics (the 8 the paper reports) -- find by physical label
    mass_idx = [phys_labels.index(m) for m in MASS_LABELS]
    print(f"\n[idx {args.idx}] === RECOVERY (8 lensing params) ===", flush=True)
    print(f"{'param':10s} {'truth':>9s} {'mean':>9s} {'std':>9s} "
          f"{'z':>7s} {'ESS':>8s} {'Rhat':>7s}", flush=True)
    for k in mass_idx:
        print(f"{phys_labels[k]:10s} {truth_vec[k]:9.4f} {phys_means[k]:9.4f} "
              f"{phys_stds[k]:9.4f} {z_err[k]:7.2f} {ess[k]:8.0f} {rhat[k]:7.4f}",
              flush=True)
    print(f"[idx {args.idx}] mass-param min ESS={ess[mass_idx].min():.0f} "
          f"max Rhat={rhat[mass_idx].max():.4f}", flush=True)
    print(f"[idx {args.idx}] all-22 min ESS={ess.min():.0f} "
          f"max Rhat={rhat.max():.4f}", flush=True)

    # ---- save ----
    outdir = os.path.join(here, args.out)
    os.makedirs(outdir, exist_ok=True)
    out_path = os.path.join(outdir, f"system_{args.idx:03d}_fit.npz")
    np.savez(
        out_path,
        samples_unconstrained=samples.astype(np.float32),   # (T, C, ndim)
        phys_samples=phys_arr.astype(np.float32),           # (T*C, 22)
        phys_labels=phys_labels,
        param_labels=PARAM_LABELS,
        ess=ess, rhat=rhat,
        phys_means=phys_means, phys_stds=phys_stds,
        truth_vec=truth_vec, z_err=z_err,
        mass_idx=np.array(mass_idx),
        map_chisq=float(map_chisq), svi_neg_elbo=float(loss_hist[-1]),
        t_map=t_map, t_svi=t_svi, t_hmc=t_hmc,
        n_hmc=n_hmc, burn=args.burn, keep=args.keep,
        idx=args.idx,
    )
    print(f"[idx {args.idx}] saved -> {out_path}  "
          f"(t_map={t_map:.0f}s t_svi={t_svi:.0f}s t_hmc={t_hmc:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
