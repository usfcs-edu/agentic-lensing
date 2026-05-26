"""v1 GIGA-Lens MAP fit for Foundry I demo system DESI-165.4754-06.0423.

Simplifications vs Huang et al. 2025a:
  - 1 Sersic lens-light (paper uses 2)
  - 1 Sersic source-light, no shapelets (paper uses Sersic + n_max=6 shapelets)
  - No nearby-galaxy Sersic components (paper uses 2)
  - 20 multi-start chains, 200 grad steps (paper used 500 chains, 350 steps default)
  - TinyTim F140W PSF supersampled to 0.065" from gigalens package (paper used empirical PSF
    from field stars; matches in band/instrument)

Goal: get a sensible posterior on θ_E and γ_EPL. Paper targets:
  θ_E = 2.6463 ± 0.0017"  (intermediate-axis convention)
  γ_EPL = 1.372 ± 0.023   (deprojected 3D slope)
"""
import time
from pathlib import Path
import gigalens

import jax
import jax.experimental.shard_map  # noqa: F401 — gigalens 2.0 accesses jax.experimental.shard_map.shard_map by attribute; JAX 0.6.2 requires explicit submodule import to populate the attribute. Upstream issue.
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import tensorflow_probability.substrates.jax as tfp
from astropy.io import fits

from gigalens.jax.inference import ModellingSequence
from gigalens.jax.model import ForwardProbModel
from gigalens.jax.profiles.light import sersic
from gigalens.jax.profiles.mass import epl, shear
from gigalens.jax.simulator import LensSimulator
from gigalens.model import PhysicalModel
from gigalens.simulator import SimulatorConfig

tfd = tfp.distributions

REPRO = Path(__file__).parent
DATA = REPRO / "data"
FIGS = REPRO / "figs"
FIGS.mkdir(exist_ok=True)

# --- load data ---
with fits.open(DATA / "cutout_F140W.fits") as h:
    sci = h["SCI"].data.astype(np.float32)
    wht = h["WHT"].data.astype(np.float32)
sky = float(np.median(sci))
data = sci - sky  # subtract sky background
print(f"Data: shape={data.shape}, sky-subtracted range=[{data.min():.3f}, {data.max():.3f}]")
print(f"Sky median (subtracted): {sky:.4f} e-/s")

# Per-pixel noise: sigma^2 = 1/WHT  for drizzled MAST products (WHT is inverse-variance)
# We pass this to gigalens as: background_rms = sqrt(median(1/WHT))
median_var = float(np.median(1.0 / np.where(wht > 0, wht, np.nan)))
background_rms = float(np.sqrt(median_var))
print(f"Median per-pixel noise: background_rms={background_rms:.5f} e-/s")

EXP_TIME = 1197.7  # 3 x 399.23 s, F140W

# --- PSF ---
PSF_PATH = Path(gigalens.__file__).parent / "assets" / "psf.npy"
kernel = np.load(PSF_PATH).astype(np.float32)
kernel /= kernel.sum()
print(f"PSF: shape={kernel.shape} (TinyTim F140W, supersampled to 0.065\")")

# --- simulator config ---
# Data is at native 0.13"/px. Set delta_pix to match data; supersample=2 lifts internal grid to 0.065".
NUM_PIX = data.shape[0]
DELTA_PIX = 0.13
SUPERSAMPLE = 2
sim_config = SimulatorConfig(delta_pix=DELTA_PIX, num_pix=NUM_PIX, supersample=SUPERSAMPLE, kernel=kernel)
print(f"SimulatorConfig: num_pix={NUM_PIX}, delta_pix={DELTA_PIX}, supersample={SUPERSAMPLE}")

# --- physical model ---
phys_model = PhysicalModel(
    [epl.EPL(50), shear.Shear()],
    [sersic.SersicEllipse(use_lstsq=False)],
    [sersic.SersicEllipse(use_lstsq=False)],
)

# --- priors (Huang 2025a Table 2, single-sersic simplification) ---
lens_prior = tfd.JointDistributionSequential(
    [
        tfd.JointDistributionNamed(
            dict(
                theta_E=tfd.LogNormal(jnp.log(2.5), 0.25),
                gamma=tfd.TruncatedNormal(2.0, 0.25, 1.0, 2.7),
                e1=tfd.Normal(0.0, 0.1),
                e2=tfd.Normal(0.0, 0.1),
                center_x=tfd.Normal(0.0, 0.05),
                center_y=tfd.Normal(0.0, 0.05),
            )
        ),
        tfd.JointDistributionNamed(
            dict(gamma1=tfd.Normal(0.0, 0.05), gamma2=tfd.Normal(0.0, 0.05))
        ),
    ]
)
lens_light_prior = tfd.JointDistributionSequential(
    [
        tfd.JointDistributionNamed(
            dict(
                R_sersic=tfd.LogNormal(jnp.log(1.0), 0.3),
                n_sersic=tfd.Uniform(0.5, 8.0),
                e1=tfd.TruncatedNormal(0.0, 0.15, -0.4, 0.4),
                e2=tfd.TruncatedNormal(0.0, 0.15, -0.4, 0.4),
                center_x=tfd.Normal(0.0, 0.1),
                center_y=tfd.Normal(0.0, 0.1),
                Ie=tfd.LogNormal(jnp.log(5.0), 0.5),
            )
        )
    ]
)
source_light_prior = tfd.JointDistributionSequential(
    [
        tfd.JointDistributionNamed(
            dict(
                R_sersic=tfd.LogNormal(jnp.log(0.25), 0.25),
                n_sersic=tfd.Uniform(0.5, 6.0),
                e1=tfd.TruncatedNormal(0.0, 0.15, -0.5, 0.5),
                e2=tfd.TruncatedNormal(0.0, 0.15, -0.5, 0.5),
                center_x=tfd.Normal(0.0, 0.25),
                center_y=tfd.Normal(0.0, 0.25),
                Ie=tfd.LogNormal(jnp.log(2.0), 0.5),
            )
        )
    ]
)
prior = tfd.JointDistributionSequential([lens_prior, lens_light_prior, source_light_prior])

prob_model = ForwardProbModel(prior, data, background_rms=background_rms, exp_time=EXP_TIME)
model_seq = ModellingSequence(phys_model, prob_model, sim_config)

# Sanity check: forward-sim with prior median
print("\nSanity-check forward sim with prior medians...")
sample = prior.sample(1, seed=jax.random.PRNGKey(0))
sim_check = LensSimulator(phys_model, sim_config, bs=1).simulate(sample)
sim_check.block_until_ready()
print(f"  simulated range: [{float(sim_check.min()):.3f}, {float(sim_check.max()):.3f}]")
print(f"  observed range:  [{data.min():.3f}, {data.max():.3f}]")

# --- MAP ---
N_SAMPLES = 20
NUM_STEPS = 200
print(f"\nMAP: n_samples={N_SAMPLES} chains, num_steps={NUM_STEPS}")
opt = optax.adabelief(1e-2, b1=0.95, b2=0.99)
t0 = time.time()
map_all = model_seq.MAP(opt, n_samples=N_SAMPLES, num_steps=NUM_STEPS, seed=0, output_type="best")
print("MAP returned; type=", type(map_all))
elapsed = time.time() - t0
print(f"MAP done in {elapsed:.1f} s ({elapsed/60:.2f} min)")

# output_type="best" returns just the best chain across the run.
best_params, best_lp, best_chi = map_all
print(f"MAP output shapes: params={best_params.shape}, lp={best_lp.shape}, chi={best_chi.shape}")

# Convert to numpy immediately to avoid further async surprises
best_params_np = np.asarray(best_params).reshape(-1)
best_lp_np = float(np.asarray(best_lp).squeeze())
best_chi_np = float(np.asarray(best_chi).squeeze())
print(f"Best log_p={best_lp_np:.2f}, chi2={best_chi_np:.2f}")

# Save raw params for offline analysis
np.savez(DATA / "map_result.npz",
         best_params=best_params_np, best_lp=best_lp_np, best_chi=best_chi_np)
print(f"Saved raw MAP params to {DATA / 'map_result.npz'}")

# Now map back to physical parameters via the bijector
print("Applying bijector forward (unconstrained -> physical) ...")
t_bij = time.time()
physical = prob_model.bij.forward(list(best_params_np[:, None]))
print(f"  done in {time.time()-t_bij:.2f}s")

# Print headline mass parameters
mass = physical[0]
print("\nBest-fit mass model:")
for k, v in mass[0].items():
    val = float(jnp.asarray(v).squeeze())
    print(f"  {k:>10s} = {val:.4f}")
print("  -- external shear --")
for k, v in mass[1].items():
    val = float(jnp.asarray(v).squeeze())
    print(f"  {k:>10s} = {val:.4f}")

# Compare to paper
paper = dict(theta_E=(2.6463, 0.0017), gamma=(1.372, 0.023),
             e1=(0.1091, 0.0020), e2=(-0.1320, 0.0020),
             gamma1=(0.0657, 0.0024), gamma2=(-0.0939, 0.0022))
print("\nComparison to Huang 2025a Table 3:")
print(f"  {'param':>10s}  {'fit':>8s}   {'paper':>15s}   {'delta':>8s}")
fit = {**{k: float(jnp.asarray(v).squeeze()) for k, v in mass[0].items()},
       **{k: float(jnp.asarray(v).squeeze()) for k, v in mass[1].items()}}
for k, (mu, sig) in paper.items():
    fv = fit.get(k, np.nan)
    print(f"  {k:>10s}  {fv:>+8.4f}   {mu:>+8.4f}±{sig:.4f}   {(fv-mu)/sig:>+8.2f}σ")

# Residual plot moved to a separate script (04_residual_plot.py) to keep MAP fast.
print("\nDone. Run 04_residual_plot.py for the residual figure.")
