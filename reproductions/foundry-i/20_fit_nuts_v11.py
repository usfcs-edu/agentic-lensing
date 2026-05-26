"""v11: TFP NUTS replacement for the failed v10 HMC stage.

Why this exists: gigalens 2.0's HMC method wraps
   pmap(scan(mcmc_sample_chain(PreconditionedHMC +
                                GradientBasedTrajectoryLengthAdaptation)))
in a single JIT and at the v6+ 74-parameter model size it never compiles.

This script uses tfp.experimental.mcmc.PreconditionedNoUTurnSampler
with a STATIC max_tree_depth=6, DualAveragingStepSizeAdaptation over
the burn-in, and no pmap. Validation run: single chain × 500 results
+ 200 burn-in, initialized at the v10 SVI posterior mean, preconditioned
by the v10 SVI Cholesky factor (apples-to-apples with what gigalens HMC
was attempting).

Reuses the v9/v10 model verbatim (74 free non-linear parameters,
empirical PSF, central 1.5-px mask, ForwardProbModel with masked
log_prob monkey-patch).
"""
import time
from pathlib import Path

import jax
import jax.experimental.shard_map  # noqa: F401  gigalens 2.0 / JAX 0.6.2 compat
import jax.numpy as jnp
import numpy as np
import tensorflow_probability.substrates.jax as tfp
from astropy.io import fits

from gigalens.jax.model import ForwardProbModel
from gigalens.jax.profiles.light import sersic, shapelets
from gigalens.jax.profiles.mass import epl, shear
from gigalens.jax.simulator import LensSimulator
from gigalens.model import PhysicalModel
from gigalens.simulator import SimulatorConfig

tfd = tfp.distributions
REPRO = Path(__file__).parent
DATA = REPRO / "data"


# ===================================================================
# Model = v10 / v9 verbatim
# ===================================================================
with fits.open(DATA / "cutout_F140W.fits") as h:
    sci = h["SCI"].data.astype(np.float32)
    wht = h["WHT"].data.astype(np.float32)
sky = float(np.median(sci))
data_arr = sci - sky
background_rms = float(np.sqrt(np.median(1.0 / np.where(wht > 0, wht, np.nan))))
EXP_TIME = 1197.7

kernel = np.load(DATA / "empirical_psf.npy").astype(np.float32)
print(f"PSF: shape={kernel.shape} (empirical)")

nb = np.load(DATA / "nearby_galaxy_loc.npz")
NEAR_X, NEAR_Y = float(nb["arcsec_x"]), float(nb["arcsec_y"])

NUM_PIX = data_arr.shape[0]
sim_config = SimulatorConfig(delta_pix=0.13, num_pix=NUM_PIX, supersample=2, kernel=kernel)
N_MAX = 6
phys_model = PhysicalModel(
    [epl.EPL(50), shear.Shear()],
    [sersic.SersicEllipse(use_lstsq=False)] * 4,
    [sersic.SersicEllipse(use_lstsq=False),
     shapelets.Shapelets(n_max=N_MAX, use_lstsq=False, interpolate=False)],
)


def sersic_prior(R_med, R_sig, Ie_med, Ie_sig, n_lo=0.5, n_hi=8.0,
                 cx_mean=0.0, cy_mean=0.0, c_sig=0.05):
    return tfd.JointDistributionNamed(dict(
        R_sersic=tfd.LogNormal(jnp.log(R_med), R_sig),
        n_sersic=tfd.Uniform(n_lo, n_hi),
        e1=tfd.TruncatedNormal(0.0, 0.15, -0.4, 0.4),
        e2=tfd.TruncatedNormal(0.0, 0.15, -0.4, 0.4),
        center_x=tfd.Normal(cx_mean, c_sig), center_y=tfd.Normal(cy_mean, c_sig),
        Ie=tfd.LogNormal(jnp.log(Ie_med), Ie_sig),
    ))


lens_mass_prior = tfd.JointDistributionSequential([
    tfd.JointDistributionNamed(dict(
        theta_E=tfd.LogNormal(jnp.log(2.5), 0.25),
        gamma=tfd.TruncatedNormal(2.0, 0.25, 1.0, 2.7),
        e1=tfd.Normal(0.0, 0.1), e2=tfd.Normal(0.0, 0.1),
        center_x=tfd.Normal(0.0, 0.02), center_y=tfd.Normal(0.0, 0.02),
    )),
    tfd.JointDistributionNamed(dict(gamma1=tfd.Normal(0.0, 0.05),
                                     gamma2=tfd.Normal(0.0, 0.05))),
])
lens_light_prior = tfd.JointDistributionSequential([
    sersic_prior(0.4, 0.3, 5.0, 0.5, c_sig=0.02),
    sersic_prior(2.0, 0.3, 2.0, 0.5, c_sig=0.02),
    sersic_prior(0.3, 0.3, 1.0, 0.5, cx_mean=NEAR_X, cy_mean=NEAR_Y, c_sig=0.1),
    sersic_prior(0.6, 0.3, 0.5, 0.5, cx_mean=NEAR_X, cy_mean=NEAR_Y, c_sig=0.1),
])
src_sersic_prior = tfd.JointDistributionNamed(dict(
    R_sersic=tfd.LogNormal(jnp.log(0.5), 0.3), n_sersic=tfd.Uniform(0.5, 6.0),
    e1=tfd.TruncatedNormal(0.0, 0.15, -0.5, 0.5),
    e2=tfd.TruncatedNormal(0.0, 0.15, -0.5, 0.5),
    center_x=tfd.Normal(0.0, 0.1), center_y=tfd.Normal(0.0, 0.1),
    Ie=tfd.LogNormal(jnp.log(2.0), 0.5),
))
shp_amp_names = shapelets.Shapelets(n_max=N_MAX)._amp_names
amp_priors = {name: tfd.Normal(0.0, 5.0 / float(jnp.sqrt(i + 1)))
              for i, name in enumerate(shp_amp_names)}
src_shp_prior = tfd.JointDistributionNamed(dict(
    beta=tfd.LogNormal(jnp.log(0.1), 0.1),
    center_x=tfd.Normal(0.0, 0.05), center_y=tfd.Normal(0.0, 0.05),
    **amp_priors,
))
source_light_prior = tfd.JointDistributionSequential([src_sersic_prior, src_shp_prior])
prior = tfd.JointDistributionSequential([lens_mass_prior, lens_light_prior, source_light_prior])

yy, xx = np.indices(data_arr.shape)
r_center = np.sqrt((xx - NUM_PIX // 2) ** 2 + (yy - NUM_PIX // 2) ** 2)
keep_mask = r_center > 1.5
prob_model = ForwardProbModel(prior, data_arr, background_rms=background_rms, exp_time=EXP_TIME)

import functools as _ft, types as _types
from jax import jit as _jit
INF_ERR = jnp.float32(1e10)
mask_jax = jnp.asarray(keep_mask)


@_ft.partial(_jit, static_argnums=(0, 1))
def masked_log_prob(self, simulator, z):
    z = list(z.T)
    x = self.bij.forward(z)
    im_sim = simulator.simulate(x)
    err_map = jnp.sqrt(self.background_rms ** 2 + im_sim / self.exp_time)
    err_map = jnp.where(mask_jax, err_map, INF_ERR)
    log_like = tfd.Independent(
        tfd.Normal(im_sim, err_map), reinterpreted_batch_ndims=2
    ).log_prob(self.observed_image)
    log_prior = self.prior.log_prob(x) + self.bij.forward_log_det_jacobian(z)
    chisq = jnp.mean(jnp.where(mask_jax, ((im_sim - self.observed_image) / err_map) ** 2, 0.0),
                     axis=(-2, -1))
    return log_like + log_prior, chisq


prob_model.log_prob = _types.MethodType(masked_log_prob, prob_model)
print("Model built: 74 params, empirical PSF, central mask, masked log_prob")


# ===================================================================
# NUTS-specific: target_log_prob_fn taking a flat (74,) vector
# ===================================================================
lens_sim_bs1 = LensSimulator(phys_model, sim_config, bs=1)


@jax.jit
def target_log_prob_fn(z):
    """Scalar log-prob over a flat unconstrained 74-vector."""
    z_batched = z[None, :]
    lp, _ = prob_model.log_prob(lens_sim_bs1, z_batched)
    return jnp.squeeze(lp)


# Warm up + sanity check
print("\nSmoke test of target_log_prob_fn at v10 SVI mean...")
svi_v10 = np.load(DATA / "svi_v10_paper_mode_empirical.npz")
qz_mean = jnp.asarray(svi_v10["mean"], dtype=jnp.float32)
qz_cov = jnp.asarray(svi_v10["cov"], dtype=jnp.float32)
scale_tril = jnp.linalg.cholesky(qz_cov).astype(jnp.float32)
print(f"  qz_mean shape: {qz_mean.shape}, dtype: {qz_mean.dtype}")
print(f"  qz_cov shape:  {qz_cov.shape}")
print(f"  scale_tril diag stats: min={float(jnp.min(jnp.diag(scale_tril))):.4e}, "
      f"max={float(jnp.max(jnp.diag(scale_tril))):.4e}")

t0 = time.time()
lp0 = target_log_prob_fn(qz_mean)
lp0.block_until_ready()
print(f"  first target_log_prob_fn(qz_mean): {float(lp0):+.2f} in {time.time()-t0:.2f}s (incl. JIT)")
t0 = time.time()
lp1 = target_log_prob_fn(qz_mean + 0.01 * jnp.ones_like(qz_mean))
lp1.block_until_ready()
print(f"  second target_log_prob_fn(perturbed): {float(lp1):+.2f} in {time.time()-t0:.2f}s")


# ===================================================================
# Build PreconditionedNoUTurnSampler
# ===================================================================
MAX_TREE_DEPTH = 6        # static — gives at most 2^6 = 64 leapfrog steps
INIT_STEP_SIZE = 0.05
NUM_BURN = 200
NUM_KEEP = 500
TARGET_ACCEPT = 0.8

momentum_distribution = tfd.MultivariateNormalTriL(
    loc=jnp.zeros_like(qz_mean),
    scale_tril=scale_tril,
)

print(f"\nNUTS config:")
print(f"  max_tree_depth      = {MAX_TREE_DEPTH} (static; <= 2^{MAX_TREE_DEPTH} = {2**MAX_TREE_DEPTH} leapfrog steps/iter)")
print(f"  init step_size      = {INIT_STEP_SIZE}")
print(f"  num_burnin_steps    = {NUM_BURN}")
print(f"  num_results         = {NUM_KEEP}")
print(f"  target_accept_prob  = {TARGET_ACCEPT}")
print(f"  preconditioner      = MultivariateNormalTriL(0, scale_tril=cholesky(v10 SVI cov))")

nuts_kernel = tfp.experimental.mcmc.PreconditionedNoUTurnSampler(
    target_log_prob_fn=target_log_prob_fn,
    momentum_distribution=momentum_distribution,
    step_size=INIT_STEP_SIZE,
    max_tree_depth=MAX_TREE_DEPTH,
)

adapted_kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
    inner_kernel=nuts_kernel,
    num_adaptation_steps=int(0.8 * NUM_BURN),
    target_accept_prob=jnp.float32(TARGET_ACCEPT),
    step_size_setter_fn=lambda pkr, new_ss: pkr._replace(step_size=new_ss),
    step_size_getter_fn=lambda pkr: pkr.step_size,
    log_accept_prob_getter_fn=lambda pkr: pkr.log_accept_ratio,
)


def trace_fn(_, pkr):
    inner = pkr.inner_results
    return {
        "is_accepted":      inner.is_accepted,
        "target_log_prob":  inner.target_log_prob,
        "step_size":        pkr.new_step_size,
        "tree_depth":       inner.leapfrogs_taken,
    }


# ===================================================================
# Run sample_chain
# ===================================================================
print(f"\nStarting NUTS sample_chain ({NUM_BURN} burn-in + {NUM_KEEP} kept) ...")
t0 = time.time()
samples, trace = tfp.mcmc.sample_chain(
    num_results=NUM_KEEP,
    num_burnin_steps=NUM_BURN,
    current_state=qz_mean,
    kernel=adapted_kernel,
    trace_fn=trace_fn,
    seed=jax.random.PRNGKey(0),
)
samples.block_until_ready()
elapsed = time.time() - t0
print(f"NUTS done in {elapsed:.1f}s ({elapsed/60:.2f} min)")
print(f"samples shape: {samples.shape}")

samples_np = np.asarray(samples)
is_accepted = np.asarray(trace["is_accepted"])
target_lp   = np.asarray(trace["target_log_prob"])
step_size   = np.asarray(trace["step_size"])
leapfrogs   = np.asarray(trace["tree_depth"])

print(f"\nAcceptance rate over kept samples: {float(is_accepted.mean()):.3f}")
print(f"target_log_prob stats over kept samples: "
      f"min={target_lp.min():.1f}, median={np.median(target_lp):.1f}, max={target_lp.max():.1f}")
print(f"step_size final: {float(step_size[-1]):.6f}")
print(f"leapfrogs per step: min={leapfrogs.min()}, median={int(np.median(leapfrogs))}, max={leapfrogs.max()}")

np.savez(
    DATA / "nuts_v11_samples.npz",
    samples=samples_np,
    is_accepted=is_accepted,
    target_log_prob=target_lp,
    step_size=step_size,
    leapfrogs=leapfrogs,
    initial_state=np.asarray(qz_mean),
    scale_tril=np.asarray(scale_tril),
    elapsed_sec=elapsed,
)
print(f"\nSaved {DATA / 'nuts_v11_samples.npz'}")


# ===================================================================
# Bijector forward to physical mass params + comparison
# ===================================================================
print("\nMapping samples to physical mass parameters via bijector...")
arr = jnp.asarray(samples_np)
physical_samples = prob_model.bij.forward(list(arr.T))
mass_main = physical_samples[0][0]
mass_shear = physical_samples[0][1]
combined = {**{k: np.asarray(mass_main[k]) for k in mass_main.keys()},
            **{k: np.asarray(mass_shear[k]) for k in mass_shear.keys()}}
np.savez(DATA / "nuts_v11_posterior_mass.npz", **combined)

paper = dict(
    theta_E=(2.6463, 0.0017), gamma=(1.372, 0.023),
    e1=(0.1091, 0.0020), e2=(-0.1320, 0.0020),
    gamma1=(0.0657, 0.0024), gamma2=(-0.0939, 0.0022),
)
v10 = dict(
    theta_E=(+2.5659, 0.0003), gamma=(+2.1507, 0.0004),
    e1=(+0.1064, 0.0004), e2=(-0.0533, 0.0002),
    gamma1=(+0.0399, 0.0003), gamma2=(-0.0676, 0.0002),
)
print("\nv11 NUTS posterior (500 kept samples, paper-mode + empirical PSF):")
print(f"  {'param':>10s}    {'v11 median':>10s}  ({'+1σ':>7s}/{'-1σ':>7s})    "
      f"{'v10 SVI':>17s}    {'paper':>17s}    {'v11−paper':>10s}")
for k, (mu_p, sig_p) in paper.items():
    a = combined.get(k)
    if a is None: continue
    med = float(np.median(a))
    lo = float(np.percentile(a, 16))
    hi = float(np.percentile(a, 84))
    v10m, v10s = v10[k]
    print(f"  {k:>10s}    {med:+7.4f}  ({hi-med:+7.4f}/{lo-med:+7.4f})    "
          f"{v10m:+7.4f}±{v10s:.4f}    {mu_p:+7.4f}±{sig_p:.4f}    {med-mu_p:+8.4f}")
print("\nDone.")
