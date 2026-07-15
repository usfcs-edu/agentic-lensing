"""Recovery stage for the 3-plane demo: can gigalens inference recover the truth
parameters of the lenstronomy-simulated system?

Staged, with criteria DERIVED before the run (see tests/multiplane_demo/README.md and
the proposal in the lab record). Everything here is UNCERTIFIED (Mode B): this module
PROPOSES results; the human grades the artifacts.

  Stage 0  noiseless identity (deterministic): at the true nonlinear params, the lstsq
           gigalens model reproduces the noiseless lenstronomy truth to the Sersic b_n
           floor (max|d|/peak ~ 3.3e-3). Falsifier: structured residual > ~5e-3 => a
           likelihood / lstsq wiring bug, not the b_n offset. This is a cheap gate that
           must pass before MAP is meaningful.

  Stage 1  MAP point estimate (a WAY-STATION, not a recovery certification --
           project-standards §2: point estimates can be biased). Metric: reduced chi^2
           at the MAP on the noisy data, pre-committed to 1 +/- 3*sqrt(2/N). |MAP-truth|
           is reported but NOT certified. The certifiable recovery claim needs MCMC
           (Stage 2, separate).

Priors are deliberately NOT centered on truth; positive-support distributions are used
for positive params so the default-event-space bijector cannot wander negative. lstsq
solves the two Sersic Ie amplitudes, so only the 24 nonlinear params are sampled.

Run (inside the canonical Shifter container; see GIGALens-Code/docs/env_setup.md):

    /usr/bin/python3 -m tests.multiplane_demo.recover        # from ~/gigalens
"""
import os
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
from jax import numpy as jnp
import optax
import tensorflow_probability.substrates.jax as tfp

from gigalens.simulator import SimulatorConfig
from gigalens.jax.cosmo import wCDM_Cosmo
from gigalens.jax.profiles.mass.epl import EPL as GEPL
from gigalens.jax.profiles.light.sersic import SersicEllipse
from gigalens.jax.scene import Component, Plane, LensModel
from gigalens.jax.scene_simulator import SceneSimulator
from gigalens.jax.scene_prob_model import ImageData, ProbModel
from gigalens.jax.inference import ModellingSequence

from . import config as C
from .noise_model import error_map, BACKGROUND_RMS, EXP_TIME, NOISE_SEED

tfd = tfp.distributions

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(HERE, "artifacts")

MAP_SEED = 1
# 100 particles fits one 40GB A100 (batch=n_samples drives memory: the supersampled
# multiplane sim x GEPL(50) series x float64 needs ~0.19 GiB/particle). 500 OOM'd at 94 GiB.
MAP_N_SAMPLES = 100
MAP_NUM_STEPS = 500

# truth lookup (which config dict supplies each (plane, role, comp) component) -----
_TRUTH_SRC = {(0, "mass", 0): C.EPL_Z1, (1, "mass", 0): C.EPL_Z2,
              (1, "light", 0): C.LENS_LIGHT_Z2, (2, "light", 0): C.SRC_LIGHT_Z3}


# ---- priors (NOT centered on truth) ------------------------------------------------
def _prior_for(name):
    """Physically-motivated prior for one nonlinear param, by name. Positive params get
    positive-support distributions (LogNormal / TruncatedNormal). See README table."""
    f = lambda v: jnp.asarray(v, jnp.float64)
    if name == "theta_E":          # positive; median 0.5" (!= either truth 1.0/0.2)
        return tfd.LogNormal(f(np.log(0.5)), f(np.log(2.0)))
    if name == "gamma":            # EPL slope; centered at ISOTHERMAL (physical, not truth)
        return tfd.TruncatedNormal(f(2.0), f(0.3), low=f(1.5), high=f(2.5))
    if name == "R_sersic":         # positive; median 0.4" (between truths 1.0/0.15)
        return tfd.LogNormal(f(np.log(0.4)), f(np.log(2.5)))
    if name == "n_sersic":         # Sersic index over the physical range
        return tfd.TruncatedNormal(f(3.0), f(1.5), low=f(0.5), high=f(8.0))
    if name in ("e1", "e2"):
        return tfd.Normal(f(0.0), f(0.2))
    if name in ("center_x", "center_y"):
        return tfd.Normal(f(0.0), f(0.3))
    raise ValueError(f"no prior defined for param {name!r}")


def _epl_priors():
    return {k: _prior_for(k) for k in
            ("theta_E", "gamma", "e1", "e2", "center_x", "center_y")}


def _sersic_priors():   # lstsq solves Ie -> Ie is NOT a sampled param
    return {k: _prior_for(k) for k in
            ("R_sersic", "n_sersic", "e1", "e2", "center_x", "center_y")}


def build_recovery_model():
    """LensModel with priors over the 24 nonlinear params; lstsq Sersic amplitudes;
    cosmology + redshifts FIXED (modelling matches the simulation, per the spec)."""
    cosmo = Component(wCDM_Cosmo(z_lens=C.Z1, z_source_ref=10.0),
                      dict(H0=C.COSMO["H0"], Om0=C.COSMO["Om0"], k=0.0, w0=C.COSMO["w0"]))
    model = LensModel([
        Plane(redshift=C.Z1, mass=[Component(GEPL(50), _epl_priors())]),
        Plane(redshift=C.Z2, mass=[Component(GEPL(50), _epl_priors())],
              light=[Component(SersicEllipse(use_lstsq=True), _sersic_priors())]),
        Plane(redshift=C.Z3,
              light=[Component(SersicEllipse(use_lstsq=True), _sersic_priors())]),
    ], cosmo=cosmo)
    cfg = SimulatorConfig(delta_pix=C.DELTA_PIX, num_pix=C.NUM_PIX,
                          supersample=C.SUPERSAMPLE,
                          kernel=__import__("tests.multiplane_demo.build_demo",
                                            fromlist=["gaussian_kernel"]).gaussian_kernel(),
                          likelihood_precision="float64")
    return model, cfg


def truth_dict(model):
    """Map each free prior key (e.g. 'planes/0/mass/0/theta_E') to its config truth."""
    out = {}
    for key in model.prior.model.keys():
        p = key.split("/")            # planes / <i> / <role> / <j> / <param>
        src = _TRUTH_SRC[(int(p[1]), p[2], int(p[3]))]
        out[key] = jnp.asarray(float(src[p[4]]), jnp.float64)
    return out


def constrained(model, z):
    """Unconstrained vector -> constrained {key: value} dict (one position)."""
    x = model.bijector.forward(jnp.atleast_2d(z))
    return {k: float(np.asarray(v).reshape(-1)[0]) for k, v in x.items()}


# ---- Stage 0: noiseless identity ---------------------------------------------------
def stage0_noiseless(model, cfg, img, sigma):
    sim = SceneSimulator(model, cfg)
    tdict = truth_dict(model)
    z_true = model.bijector.inverse(tdict)
    ds = ImageData(img, cfg, error_map=sigma, sees="all")
    prob = ProbModel(model, ds, mode="lstsq")

    _, red_chi2 = prob.log_like(z_true, simulator=sim)
    params = model.to_params(model.bijector.forward(z_true))
    recon = np.asarray(sim.lstsq_simulate(params, jnp.asarray(img),
                                          jnp.asarray(sigma), ds.mask))
    max_over_peak = np.max(np.abs(recon - img)) / np.max(np.abs(img))
    print("\n----- Stage 0: noiseless identity (deterministic) -----")
    print(f"  lstsq reconstruction max|d|/peak = {max_over_peak:.3e}   "
          f"(threshold ~b_n floor 3.3e-3; falsifier > 5e-3)")
    print(f"  reduced chi^2 at truth (noiseless) = {float(red_chi2):.3e}  (~0 expected)")
    verdict = "PROPOSE PASS" if max_over_peak < 5e-3 else "PROPOSE FAIL"
    print(f"  [{verdict}, UNCERTIFIED]")
    return z_true, recon


# ---- Stage 1: MAP ------------------------------------------------------------------
def stage1_map(model, cfg, noisy, sigma, z_true):
    ds = ImageData(noisy, cfg, error_map=sigma, sees="all")
    prob = ProbModel(model, ds, mode="lstsq")
    seq = ModellingSequence.from_scene(prob)

    optimizer = optax.adabelief(1e-2, b1=0.95, b2=0.99)   # repo-standard MAP optimizer
    print("\n----- Stage 1: MAP (way-station, NOT a recovery certification) -----")
    print(f"  optimizer=adabelief(1e-2,0.95,0.99)  n_samples={MAP_N_SAMPLES}  "
          f"num_steps={MAP_NUM_STEPS}  seed={MAP_SEED}  devices={len(jax.devices())}")
    z_map, lp_map, _ = seq.MAP(optimizer, n_samples=MAP_N_SAMPLES,
                               num_steps=MAP_NUM_STEPS, seed=MAP_SEED,
                               output_type="best", pbar_interval=0)
    z_map = np.asarray(z_map)

    _, red_chi2 = prob.log_like(jnp.asarray(z_map))
    red = float(np.asarray(red_chi2).reshape(-1)[0])
    N = ds.event_size
    spread = np.sqrt(2.0 / N)
    in_band = abs(red - 1.0) < 3 * spread
    print(f"\n  reduced chi^2 at MAP = {red:.4f}   "
          f"(pre-committed 1 +/- 3*sqrt(2/N) = 1 +/- {3*spread:.4f})")
    print(f"  [{'in band' if in_band else 'OUT OF BAND'}; "
          f"good-fit line red-chi2 < 1.2 -> {'ok' if red < 1.2 else 'FAIL'}]")

    # |MAP - truth| per param (reported, NOT certified -- no uncertainty from a point est.)
    cmap = constrained(model, jnp.asarray(z_map))
    ctru = constrained(model, z_true)
    print("\n  param recovery (MAP vs truth; UNCERTIFIED, point estimate only):")
    print(f"    {'param':40s} {'truth':>9s} {'MAP':>9s} {'|d|':>9s}")
    worst = (0.0, "")
    for k in model.prior.model.keys():
        d = abs(cmap[k] - ctru[k])
        short = k.replace("planes/", "p").replace("/mass/0/", ".M.").replace(
            "/light/0/", ".L.")
        print(f"    {short:40s} {ctru[k]:9.4f} {cmap[k]:9.4f} {d:9.4f}")
        # track worst on a scale-relative basis for the dominant lens / source
        if d > worst[0]:
            worst = (d, short)
    print(f"    worst abs deviation: {worst[1]} = {worst[0]:.4f}")
    return z_map, cmap, ctru, red


def main():
    npz = np.load(os.path.join(ART, "demo_3plane.npz"))
    nnpz = np.load(os.path.join(ART, "demo_3plane_noisy.npz"))
    img = npz["img"]                       # noiseless lenstronomy truth (the oracle)
    noisy, sigma = nnpz["noisy"], nnpz["sigma"]

    model, cfg = build_recovery_model()
    print(f"recovery model: {model.num_free_params} free nonlinear params "
          f"(+2 lstsq Ie amplitudes); cosmology+redshifts fixed")

    z_true, recon = stage0_noiseless(model, cfg, img, sigma)
    z_map, cmap, ctru, red = stage1_map(model, cfg, noisy, sigma, z_true)

    np.savez(os.path.join(ART, "demo_3plane_map.npz"),
             z_map=np.asarray(z_map), z_true=np.asarray(z_true),
             recon_noiseless=recon, red_chi2_map=red,
             map_seed=MAP_SEED, noise_seed=NOISE_SEED)
    print(f"\nsaved: {os.path.join(ART, 'demo_3plane_map.npz')}")
    print("\nAll results UNCERTIFIED (Mode B). MAP is a way-station; the certifiable "
          "recovery claim requires Stage 2 (MCMC + R-hat/ESS).")


if __name__ == "__main__":
    main()
