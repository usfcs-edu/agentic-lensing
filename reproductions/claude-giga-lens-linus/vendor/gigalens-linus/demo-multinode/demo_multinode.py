import jax

jax.distributed.initialize()

import sys

# TODO: Change to local gigalens package
sys.path.append("/global/homes/e/eldenyap/gigalens/src")

from gigalens.jax.inference import ModellingSequence
from gigalens.jax.model import ForwardProbModel
from gigalens.jax.simulator import LensSimulator
from gigalens.simulator import SimulatorConfig
from gigalens.model import PhysicalModel
from gigalens.jax.profiles.light import sersic
from gigalens.jax.profiles.mass import epl, shear

import corner as corner
import tensorflow_probability.substrates.jax as tfp
from jax import numpy as jnp
import time
import numpy as np
import optax
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import norm, kstest

tfd = tfp.distributions

# Showing all available devices
total_devices = jax.device_count()
verbose = jax.process_index() == 0
print(f"{jax.process_index()}: local devices: {jax.local_devices()}")
if verbose:
    print(f"Global devices: {jax.devices()}")

# Create priors for lens mass, lens light, source light
lens_prior = tfd.JointDistributionSequential(
    [
        tfd.JointDistributionNamed(
            dict(
                theta_E=tfd.LogNormal(jnp.log(1.25), 0.4),
                gamma=tfd.TruncatedNormal(2, 0.5, 1, 3),
                e1=tfd.Normal(0, 0.2),
                e2=tfd.Normal(0, 0.2),
                center_x=tfd.Normal(0, 0.06),
                center_y=tfd.Normal(0, 0.06),
            )
        ),
        tfd.JointDistributionNamed(
            dict(gamma1=tfd.Normal(0, 0.1), gamma2=tfd.Normal(0, 0.1))
        ),
    ]
)
lens_light_prior = tfd.JointDistributionSequential(
    [
        tfd.JointDistributionNamed(
            dict(
                R_sersic=tfd.LogNormal(jnp.log(1.6), 0.25),
                n_sersic=tfd.Uniform(0.5, 8),
                e1=tfd.TruncatedNormal(0, 0.1, -0.15, 0.15),
                e2=tfd.TruncatedNormal(0, 0.1, -0.15, 0.15),
                center_x=tfd.Normal(0, 0.02),
                center_y=tfd.Normal(0, 0.02),
                Ie=tfd.LogNormal(jnp.log(300.0), 0.5),
            )
        )
    ]
)
source_light_prior = tfd.JointDistributionSequential(
    [
        tfd.JointDistributionNamed(
            dict(
                R_sersic=tfd.LogNormal(jnp.log(0.25), 0.25),
                n_sersic=tfd.Uniform(0.5, 8),
                e1=tfd.TruncatedNormal(0, 0.3, -0.5, 0.5),
                e2=tfd.TruncatedNormal(0, 0.3, -0.5, 0.5),
                center_x=tfd.Normal(0, 0.5),
                center_y=tfd.Normal(0, 0.5),
                Ie=tfd.LogNormal(jnp.log(150.0), 0.9),
            )
        )
    ]
)
prior = tfd.JointDistributionSequential(
    [lens_prior, lens_light_prior, source_light_prior]
)

# TODO: Change to point to local directory
kernel = np.load("./assets/psf.npy").astype(np.float32)
observed_img = np.load("./assets/demo.npy")

# Modeling Parameters
background_rms = 0.2
exp_time = 100
delta_pix = 0.065
num_pix = 60
supersample = 1

phys_model = PhysicalModel(
    [epl.EPL(50), shear.Shear()],
    [sersic.SersicEllipse(use_lstsq=False)],
    [sersic.SersicEllipse(use_lstsq=False)],
)
prob_model = ForwardProbModel(
    prior, observed_img, background_rms=background_rms, exp_time=exp_time
)
sim_config = SimulatorConfig(
    delta_pix=delta_pix, num_pix=num_pix, supersample=supersample, kernel=kernel
)
lens_sim = LensSimulator(phys_model, sim_config, bs=1)

model_seq = ModellingSequence(phys_model, prob_model, sim_config)

results = {}

# MAP
print("Starting MAP")

map_opt = optax.adabelief(1e-2, b1=0.95, b2=0.99, nesterov=True)
map_kwargs = {
    "optimizer": map_opt,
    "n_samples": 16000,
    "num_steps": 1000,
    "seed": 0,
    "output_type": "best_step",
}

map_estimate, map_lps_hist, map_chisq_hist = model_seq.MAP(**map_kwargs)

map_best_z = map_estimate[jnp.nanargmax(map_lps_hist)][None]
map_best_x = model_seq.prob_model.bij.forward(list(map_best_z.T))

# SVI
print("Starting SVI")

svi_opt = optax.adabelief(1e-4, b1=0.95, b2=0.99)
svi_kwargs = {
    "n_vi": 16000,
    "num_steps": 1500,
    "seed": 0,
}

qz, svi_loss_hist = model_seq.SVI(map_best_z, svi_opt, **svi_kwargs)

svi_samples_z = qz.sample(36000, seed=jax.random.PRNGKey(0))
svi_samples_x = prob_model.bij.forward(list(svi_samples_z.T))
svi_mean = prob_model.bij.forward(list(qz.mean().T))

# HMC
print("Starting HMC")

hmc_kwargs = {
    "n_hmc": 75,
    "num_burnin_steps": 250,
    "num_results": 750,
    "seed": 0,
}

hmc_samples_z = model_seq.HMC(qz, **hmc_kwargs)

rhat = tfp.mcmc.potential_scale_reduction(hmc_samples_z, independent_chain_ndims=2)
print(f"The rhats are:\n{rhat}")

smp = hmc_samples_z.reshape((-1, 22))
hmc_samples_x = prob_model.bij.forward(list(smp.T))
hmc_median_x = prob_model.bij.forward(list(np.median(smp, axis=0)))

# Display MAP Results
fig, axs = plt.subplots(1, 4)
fig.set_size_inches(12, 3)

predicted_img = lens_sim.simulate(map_best_x)
noise_map = np.sqrt(observed_img / exp_time + background_rms**2)
residual = (observed_img - predicted_img) / noise_map
chisq = np.sum(np.square(residual))
dof = observed_img.shape[0] * observed_img.shape[1] - 22

axs[0].imshow(observed_img)
axs[0].set_title(f"True Image")
axs[1].imshow(predicted_img)
axs[1].set_title(f"MAP Model Fit (Red. Chisq:{chisq / dof:.3f})")
axs[2].imshow(residual)
axs[2].set_title(f"MAP Normalized Residual")

flat_residual = residual.flatten()
mu, std = norm.fit(flat_residual)
p = kstest(flat_residual, norm.cdf).pvalue
dummy_x = np.linspace(np.min(flat_residual), np.max(flat_residual), 100)
axs[3].hist(
    flat_residual,
    bins=50,
    density=True,
    label=f"mu={mu:.4f} \nstd={std:.4f} \np={p:.4f}",
)
axs[3].plot(dummy_x, norm.pdf(dummy_x, mu, std))
axs[3].set_title("MAP Global Gaussianity Test")
axs[3].legend()

plt.savefig("results/map_results.png")
plt.close(fig)

# Display HMC Results
fig, axs = plt.subplots(1, 4)
fig.set_size_inches(12, 3)

predicted_img = lens_sim.simulate(hmc_median_x)
residual = (observed_img - predicted_img) / noise_map
chisq = np.sum(np.square(residual))

axs[0].imshow(observed_img)
axs[0].set_title(f"True Image")
axs[1].imshow(predicted_img)
axs[1].set_title(f"HMC Model Fit (Red. Chisq:{chisq / dof:.3f})")
axs[2].imshow(residual)
axs[2].set_title(f"HMC Normalized Residual")

flat_residual = residual.flatten()
mu, std = norm.fit(flat_residual)
p = kstest(flat_residual, norm.cdf).pvalue
dummy_x = np.linspace(np.min(flat_residual), np.max(flat_residual), 100)
axs[3].hist(
    flat_residual,
    bins=50,
    density=True,
    label=f"mu={mu:.4f} \nstd={std:.4f} \np={p:.4f}",
)
axs[3].plot(dummy_x, norm.pdf(dummy_x, mu, std))
axs[3].set_title("HMC Global Gaussianity Test")
axs[3].legend()

plt.savefig("results/hmc_results.png")
plt.close(fig)

# Plot loss histories
fig, axs = plt.subplots(1, 2)
axs[0].plot(map_chisq_hist)
axs[0].set_title("MAP Loss History")
axs[0].set_xlabel("Step")
axs[0].set_ylabel("Chi-squared Loss")
axs[0].set_ylim(bottom=0, top=3)

axs[1].plot(svi_loss_hist)
axs[1].set_title("SVI Loss History")
axs[1].set_xlabel("Step")
axs[1].set_ylabel("ELBO")

plt.savefig("results/map_svi_loss_histories.png")
plt.close(fig)

# Plot cornerplots
tups = [(0, 0), (0, 1), (1, 0), (2, 0)]
label_prefixes = ["", "", "lens_", "src_"]
labels = []
for (i, j), label_prefix in zip(tups, label_prefixes):
    labels.extend((label_prefix + key for key in map_best_x[i][j].keys()))

svi_color = "#4169E1"  # Royal Blue
svi_plt_samples = np.vstack(
    [np.array(list(svi_samples_x[i][j].values())) for i, j in tups]
).T
fig = corner.corner(
    svi_plt_samples, show_titles=True, title_fmt=".3f", labels=labels, color=svi_color
)

map_color = "red"
best_map_pts = []
for i, j in tups:
    best_map_pts.extend((arr.item() for arr in map_best_x[i][j].values()))
best_map_pts = np.array(best_map_pts)
corner.overplot_points(
    fig,
    best_map_pts[np.newaxis],
    marker="*",
    markersize=12,
    mfc=map_color,
    mec=map_color,
)

hmc_color = "#228B22"  # Forest Green
std_quantiles = [0.159, 0.841]  # Quantiles for ±1 standard deviation
hmc_plt_samples = np.vstack(
    [np.array(list(hmc_samples_x[i][j].values())) for i, j in tups]
).T
corner.corner(
    hmc_plt_samples,
    fig=fig,
    show_titles=False,
    title_fmt=".3f",
    labels=labels,
    quantiles=std_quantiles,
    color=hmc_color,
)

# Create legend handles and add to figure
svi_patch = mpatches.Patch(color=svi_color, label="SVI")
hmc_patch = mpatches.Patch(color=hmc_color, label="HMC")
map_handle = plt.Line2D(
    [0],
    [0],
    marker="*",
    color="w",
    markerfacecolor=map_color,
    markeredgecolor=map_color,
    markersize=12,
    label="MAP",
)
fig.legend(
    handles=[svi_patch, hmc_patch, map_handle],
    loc="upper right",
    frameon=True,
    fontsize=20,
    handlelength=3,
    handleheight=1.5,
    labelspacing=1.5,
)

plt.savefig("results/cornerplot.png")
plt.close()
