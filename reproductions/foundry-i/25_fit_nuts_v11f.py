"""v11f: NUTS starting from v7 paper-mode chain, loose target_accept, bigger step_size.

v11/v11e pathology: starting at v10 SVI mean (log_p=-45225) is far from the
typical-set, the chain burns through low-density landscape, dual-averaging
adapter sees divergences and crashes step_size.

Fixes for v11f:
  1. Start at v7's paper-mode MAP chain (log_p=-21789), a much higher-density
     point. Chain doesn't need to traverse a huge log_p gradient during burnin.
  2. target_accept_prob = 0.60 (looser; encourages bigger steps even with some
     divergences).
  3. init_step_size = 0.5 (much larger; adapter has to SHRINK, not grow).
  4. num_burnin = 400 (more time for adaptation to stabilize).
  5. Same jittered preconditioner (cov + 1e-6 I) from v11e.

If v11f still crashes step_size to ~1e-5 or smaller, the v10 SVI cov is just
too poor a preconditioner. We'd need windowed-adaptive mass matrix estimation,
which requires restructuring as a TFP JointDistribution — beyond current scope.
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
# Starting point: v7 paper-mode MAP chain (log_p=-21789, much better than SVI mean)
# ===================================================================
v7pm = np.load(DATA / "map_v7_paper_mode.npz")
qz_start_np = v7pm["best_params"].astype(np.float32)
print(f"v7 paper-mode start: log_p={float(v7pm['best_lp']):.1f}, chi2={float(v7pm['best_chi']):.3f}")
qz_start = jnp.asarray(qz_start_np)

# Jittered preconditioner (from v11e)
svi_v10 = np.load(DATA / "svi_v10_paper_mode_empirical.npz")
qz_cov_np = svi_v10["cov"]
NDIM = qz_start_np.shape[0]
JITTER = 1e-6
qz_cov_jittered = qz_cov_np + JITTER * np.eye(NDIM, dtype=qz_cov_np.dtype)
L_np = np.linalg.cholesky(qz_cov_jittered)
scale_tril = jnp.asarray(L_np, dtype=jnp.float32)
print(f"Preconditioner: jittered v10 SVI cov + {JITTER}*I, chol diag in [{np.diag(L_np).min():.4e}, {np.diag(L_np).max():.4e}]")

# Warmup
lp_start = target_log_prob_fn(qz_start)
lp_start.block_until_ready()
print(f"log_p at v7 paper-mode start = {float(lp_start):.2f}")


# ===================================================================
# NUTS with looser target_accept + bigger init step_size + longer burnin
# ===================================================================
MAX_TREE_DEPTH = 6
INIT_STEP_SIZE = 0.5
NUM_BURN = 400
NUM_KEEP = 500
TARGET_ACCEPT = 0.60

momentum_distribution = tfd.MultivariateNormalTriL(
    loc=jnp.zeros_like(qz_start), scale_tril=scale_tril,
)

print(f"\nNUTS config (v11f):")
print(f"  starting point         = v7 paper-mode MAP (log_p={float(lp_start):.0f})")
print(f"  jitter on cov          = {JITTER}")
print(f"  max_tree_depth         = {MAX_TREE_DEPTH} (static)")
print(f"  init step_size         = {INIT_STEP_SIZE} (larger than v11e's 0.1)")
print(f"  num_burnin / num_kept  = {NUM_BURN} / {NUM_KEEP}")
print(f"  target_accept_prob     = {TARGET_ACCEPT} (looser than v11e's 0.8)")

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
    current_state=qz_start, kernel=adapted_kernel,
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
diff = np.linalg.norm(np.diff(samples_np, axis=0), axis=1)

print(f"\nDiagnostics over {NUM_KEEP} kept samples:")
print(f"  Acceptance rate     : {float(is_acc.mean()):.3f}")
print(f"  target_log_prob     : min={lps.min():.1f}, median={np.median(lps):.1f}, max={lps.max():.1f}")
print(f"  step_size           : start={float(ss[0]):.5g}, final={float(ss[-1]):.5g}")
print(f"  leapfrogs/iter      : min={int(leap.min())}, median={int(np.median(leap))}, max={int(leap.max())}")
print(f"  ||Δz|| step-to-step : median={np.median(diff):.4f}, max={diff.max():.4f}")
print(f"  per-param sample std: min={samples_np.std(axis=0).min():.4e}, max={samples_np.std(axis=0).max():.4e}")
print(f"  any NaN             : {bool(np.any(np.isnan(samples_np)))}")

np.savez(DATA / "nuts_v11f_samples.npz",
         samples=samples_np, is_accepted=is_acc, target_log_prob=lps,
         step_size=ss, leapfrogs=leap, initial_state=qz_start_np,
         scale_tril=np.asarray(scale_tril), jitter=JITTER, elapsed_sec=elapsed)
print(f"\nSaved {DATA / 'nuts_v11f_samples.npz'}")

# Physical posterior
arr = jnp.asarray(samples_np)
physical = prob_model.bij.forward(list(arr.T))
mass_main = physical[0][0]
mass_shear = physical[0][1]
combined = {**{k: np.asarray(mass_main[k]) for k in mass_main.keys()},
            **{k: np.asarray(mass_shear[k]) for k in mass_shear.keys()}}
np.savez(DATA / "nuts_v11f_posterior_mass.npz", **combined)

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
print(f"\nv11f NUTS posterior (v7 paper-mode start, looser target, jittered precond, 500 kept):")
print(f"  {'param':>10s}    {'v11f median ± 1σ':>22s}    {'v10 SVI':>17s}    "
      f"{'paper':>17s}    {'v11f−paper':>10s}")
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
