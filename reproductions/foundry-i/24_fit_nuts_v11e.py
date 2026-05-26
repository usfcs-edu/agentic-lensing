"""v11e: NUTS with JITTERED preconditioner (fix for rank-deficient v10 SVI cov).

Root cause from v11/v11b/v11c/v11d scan: v10 SVI covariance is rank 53 of 74
(min diagonal 1.16e-8, min eigenvalue -4.4e-13). cholesky() silently produces
NaN. TFP's MultivariateNormalTriL(scale_tril=NaN) sampler then proposes
NaN momentum, which always rejects, freezing the chain.

Fix: add a small diagonal jitter to cov before Cholesky. Jitter magnitude:
1e-6, comparable to the smallest non-degenerate diagonal entry — small enough
to leave the well-identified directions un-perturbed, large enough to fill in
the 21 nearly-degenerate directions with a sane prior.

This version also goes back to DualAveragingStepSizeAdaptation (now with real
non-NaN momentum, adaptation should converge to a sensible step_size rather
than crashing to ~0).
"""
import time
from pathlib import Path

import jax
import jax.experimental.shard_map  # noqa: F401
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

with fits.open(DATA / "cutout_F140W.fits") as h:
    sci = h["SCI"].data.astype(np.float32)
    wht = h["WHT"].data.astype(np.float32)
sky = float(np.median(sci))
data_arr = sci - sky
background_rms = float(np.sqrt(np.median(1.0 / np.where(wht > 0, wht, np.nan))))
EXP_TIME = 1197.7
kernel = np.load(DATA / "empirical_psf.npy").astype(np.float32)
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
lens_sim_bs1 = LensSimulator(phys_model, sim_config, bs=1)

@jax.jit
def target_log_prob_fn(z):
    z_batched = z[None, :]
    lp, _ = prob_model.log_prob(lens_sim_bs1, z_batched)
    return jnp.squeeze(lp)


# ===================================================================
# SVI artifact + JITTERED preconditioner
# ===================================================================
svi_v10 = np.load(DATA / "svi_v10_paper_mode_empirical.npz")
qz_mean_np = svi_v10["mean"]
qz_cov_np = svi_v10["cov"]
NDIM = qz_mean_np.shape[0]

# Diagnose cov
print(f"v10 SVI cov: dtype={qz_cov_np.dtype}, shape={qz_cov_np.shape}")
print(f"  diag stats: min={np.diag(qz_cov_np).min():.4e}, max={np.diag(qz_cov_np).max():.4e}")
eigs = np.linalg.eigvalsh(qz_cov_np)
print(f"  eigenvalue range: [{eigs.min():.4e}, {eigs.max():.4e}], rank ~ {int(np.sum(eigs > 1e-10))}/{NDIM}")

# Jitter: add 1e-6 * I — comparable to smallest legitimate diag, fixes negative eigenvalues
JITTER = 1e-6
print(f"\nJittering: cov_fixed = cov + {JITTER} * I  (raises min eigenvalue above 0)")
qz_cov_fixed = qz_cov_np + JITTER * np.eye(NDIM, dtype=qz_cov_np.dtype)
eigs_fixed = np.linalg.eigvalsh(qz_cov_fixed)
print(f"  eigenvalue range after jitter: [{eigs_fixed.min():.4e}, {eigs_fixed.max():.4e}]")

# Cholesky in numpy first to verify
L_np = np.linalg.cholesky(qz_cov_fixed)
print(f"  numpy chol(jittered) diag: min={np.diag(L_np).min():.4e}, max={np.diag(L_np).max():.4e}")
print(f"  any NaN in L: {np.any(np.isnan(L_np))}")

qz_mean = jnp.asarray(qz_mean_np, dtype=jnp.float32)
scale_tril = jnp.asarray(L_np, dtype=jnp.float32)

# Warmup
lp0 = target_log_prob_fn(qz_mean); lp0.block_until_ready()
print(f"\nlog_p at SVI mean = {float(lp0):.2f}")

# ===================================================================
# NUTS with adapter (now should work)
# ===================================================================
MAX_TREE_DEPTH = 6
INIT_STEP_SIZE = 0.1
NUM_BURN = 200
NUM_KEEP = 500
TARGET_ACCEPT = 0.8

momentum_distribution = tfd.MultivariateNormalTriL(
    loc=jnp.zeros_like(qz_mean),
    scale_tril=scale_tril,
)

print(f"\nNUTS config (v11e: jittered preconditioner + DualAveraging adapter):")
print(f"  jitter on cov diag     = {JITTER}")
print(f"  max_tree_depth         = {MAX_TREE_DEPTH} (static)")
print(f"  init step_size         = {INIT_STEP_SIZE}")
print(f"  num_burnin / num_kept  = {NUM_BURN} / {NUM_KEEP}")
print(f"  target_accept_prob     = {TARGET_ACCEPT}")

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
    return {
        "is_accepted":      pkr.inner_results.is_accepted,
        "target_log_prob":  pkr.inner_results.target_log_prob,
        "step_size":        pkr.new_step_size,
        "leapfrogs":        pkr.inner_results.leapfrogs_taken,
    }

print(f"\nRunning NUTS sample_chain ...")
t0 = time.time()
samples, trace = tfp.mcmc.sample_chain(
    num_results=NUM_KEEP, num_burnin_steps=NUM_BURN,
    current_state=qz_mean, kernel=adapted_kernel,
    trace_fn=trace_fn, seed=jax.random.PRNGKey(0),
)
samples.block_until_ready()
elapsed = time.time() - t0
print(f"NUTS done in {elapsed:.1f}s ({elapsed/60:.2f} min)")

samples_np = np.asarray(samples)
is_acc = np.asarray(trace["is_accepted"])
lps = np.asarray(trace["target_log_prob"])
ss = np.asarray(trace["step_size"])
leap = np.asarray(trace["leapfrogs"])

# Diagnostics
diff = np.linalg.norm(np.diff(samples_np, axis=0), axis=1)
print(f"\nDiagnostics over {NUM_KEEP} kept samples:")
print(f"  Acceptance rate     : {float(is_acc.mean()):.3f}")
print(f"  target_log_prob     : min={lps.min():.1f}, median={np.median(lps):.1f}, max={lps.max():.1f}")
print(f"  step_size           : start={float(ss[0]):.5g}, final={float(ss[-1]):.5g}")
print(f"  leapfrogs/iter      : min={int(leap.min())}, median={int(np.median(leap))}, max={int(leap.max())}")
print(f"  ||Δz|| step-to-step : median={np.median(diff):.4f}, max={diff.max():.4f}")
print(f"  per-param sample std: min={samples_np.std(axis=0).min():.4e}, max={samples_np.std(axis=0).max():.4e}")
print(f"  any NaN in samples  : {bool(np.any(np.isnan(samples_np)))}")

np.savez(DATA / "nuts_v11e_samples.npz",
         samples=samples_np, is_accepted=is_acc, target_log_prob=lps,
         step_size=ss, leapfrogs=leap, initial_state=np.asarray(qz_mean),
         scale_tril=np.asarray(scale_tril), jitter=JITTER, elapsed_sec=elapsed)
print(f"\nSaved {DATA / 'nuts_v11e_samples.npz'}")

# Physical posterior
arr = jnp.asarray(samples_np)
physical = prob_model.bij.forward(list(arr.T))
mass_main = physical[0][0]
mass_shear = physical[0][1]
combined = {**{k: np.asarray(mass_main[k]) for k in mass_main.keys()},
            **{k: np.asarray(mass_shear[k]) for k in mass_shear.keys()}}
np.savez(DATA / "nuts_v11e_posterior_mass.npz", **combined)

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
print(f"\nv11e NUTS posterior (jittered preconditioner, 500 kept samples):")
print(f"  {'param':>10s}    {'v11e median ± 1σ':>22s}    {'v10 SVI':>17s}    "
      f"{'paper (HMC)':>17s}    {'v11e−paper':>10s}")
for k, (mu_p, sig_p) in paper.items():
    a = combined.get(k)
    if a is None: continue
    med = float(np.median(a))
    lo = float(np.percentile(a, 16))
    hi = float(np.percentile(a, 84))
    v10m, v10s = v10[k]
    print(f"  {k:>10s}    {med:+7.4f} (+{hi-med:.4f}/{lo-med:+.4f})    "
          f"{v10m:+7.4f}±{v10s:.4f}    {mu_p:+7.4f}±{sig_p:.4f}    {med-mu_p:+8.4f}")
print("\nDone.")
