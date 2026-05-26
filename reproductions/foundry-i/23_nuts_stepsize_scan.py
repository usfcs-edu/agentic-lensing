"""Scan step_size to find where NUTS actually accepts proposals.

v11  (adapted, ended at 2e-6):  acceptance=1.000  but chain frozen (step too small)
v11b (fixed 0.05):               acceptance=0.000, leapfrogs=1 (step too big)
v11c (fixed 0.005):              acceptance=0.000, leapfrogs=1 (still too big??)

So the working window must be somewhere below 0.005 but well above 2e-6.
Try a coarse log-spaced scan and report acceptance + step-to-step movement.
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

# ----- model & priors (copy from v10/v11) -----
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

svi_v10 = np.load(DATA / "svi_v10_paper_mode_empirical.npz")
qz_mean = jnp.asarray(svi_v10["mean"], dtype=jnp.float32)
qz_cov = jnp.asarray(svi_v10["cov"], dtype=jnp.float32)
scale_tril = jnp.linalg.cholesky(qz_cov).astype(jnp.float32)

# Warmup JIT
lp0 = target_log_prob_fn(qz_mean); lp0.block_until_ready()
print(f"log_p at SVI mean = {float(lp0):.2f}")
print(f"scale_tril diag: min={float(jnp.min(jnp.diag(scale_tril))):.4e}, "
      f"max={float(jnp.max(jnp.diag(scale_tril))):.4e}")

# ----- Step size scan -----
NUM_BURN = 50
NUM_KEEP = 100   # short — just want diagnostic, not posterior
MAX_TREE_DEPTH = 6

# Coarse log scan: 8 step sizes spanning 1e-5 to 1e-1
step_sizes = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]
print(f"\nScan: num_burn={NUM_BURN}, num_keep={NUM_KEEP}, max_tree_depth={MAX_TREE_DEPTH}")
print(f"  {'step_size':>10s}  {'wall_s':>8s}  {'accept':>7s}  {'leapfrogs(med)':>15s}  "
      f"{'||Δz||(med)':>13s}  {'lp(min..max)':>20s}")

results = []
for ss in step_sizes:
    momentum_distribution = tfd.MultivariateNormalTriL(
        loc=jnp.zeros_like(qz_mean),
        scale_tril=scale_tril,
    )
    nuts_kernel = tfp.experimental.mcmc.PreconditionedNoUTurnSampler(
        target_log_prob_fn=target_log_prob_fn,
        momentum_distribution=momentum_distribution,
        step_size=float(ss),
        max_tree_depth=MAX_TREE_DEPTH,
    )
    def trace_fn(_, pkr):
        return {"is_accepted": pkr.is_accepted, "lp": pkr.target_log_prob,
                "leap": pkr.leapfrogs_taken}
    t0 = time.time()
    samples, trace = tfp.mcmc.sample_chain(
        num_results=NUM_KEEP, num_burnin_steps=NUM_BURN, current_state=qz_mean,
        kernel=nuts_kernel, trace_fn=trace_fn, seed=jax.random.PRNGKey(0),
    )
    samples.block_until_ready()
    wall = time.time() - t0
    s_np = np.asarray(samples)
    is_acc = np.asarray(trace["is_accepted"])
    lps = np.asarray(trace["lp"])
    leap = np.asarray(trace["leap"])
    diff = np.linalg.norm(np.diff(s_np, axis=0), axis=1)
    print(f"  {ss:>10.1e}  {wall:>8.1f}  {float(is_acc.mean()):>7.3f}  "
          f"{int(np.median(leap)):>15d}  {float(np.median(diff)):>13.5f}  "
          f"{lps.min():>+8.1f}..{lps.max():+8.1f}")
    results.append((ss, float(is_acc.mean()), int(np.median(leap)),
                    float(np.median(diff)), float(lps.min()), float(lps.max())))

print("\nDone.")
