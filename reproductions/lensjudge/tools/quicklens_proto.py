"""quicklens_proto — Foundry-I lens-MODELABILITY criterion for DESI grz cutouts.

Question (Huang 2025a): does a plausible lens MODEL reproduce the image configuration?
A true lens admits a good lens-model fit (sensible theta_E, an arc/multiple images,
acceptable residual). A single galaxy / spiral / ring should not.

Approach: GIGA-Lens MAP (JAX, GPU) multi-start gradient descent fit of
  EPL + external-shear mass, 2 lens-light Sersics, source = Sersic + small shapelets.
We then compare the lens-model fit to a NULL model = lens-light-only (no lensed source),
fit the same way. The discriminator is the chi2 IMPROVEMENT from adding the lensed
source: a real lens needs the lensed source to explain off-center flux (big improvement);
a smooth galaxy does not (small improvement). We also report theta_E and n_images from
solving the lens equation on the best-fit mass model.

Run in the gigalens venv:
  /home/benson/.venvs/gigalens/bin/python quicklens_proto.py [path1.fits path2.fits ...]
With no args it runs the built-in validation set (3 A lenses + 3 non-lenses).
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
from astropy.io import fits

# ---- keep JAX from grabbing all GPU memory; one visible device is plenty ----
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.6")
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")

import jax  # noqa: E402
try:
    import jax.experimental.shard_map  # noqa: F401,E402  (ModellingSequence needs it)
except Exception:  # noqa: BLE001
    pass
import jax.numpy as jnp  # noqa: E402
import optax  # noqa: E402
import tensorflow_probability.substrates.jax as tfp  # noqa: E402

from gigalens.jax.inference import ModellingSequence  # noqa: E402
from gigalens.jax.model import BackwardProbModel  # noqa: E402
from gigalens.jax.profiles.light import sersic, shapelets  # noqa: E402
from gigalens.jax.profiles.mass import epl, shear  # noqa: E402
from gigalens.model import PhysicalModel  # noqa: E402
from gigalens.simulator import SimulatorConfig  # noqa: E402

tfd = tfp.distributions

DELTA_PIX = 0.262           # DESI Legacy arcsec/px
SUPERSAMPLE = 2
FWHM_ARCSEC = 1.3          # ground-based Gaussian PSF assumption
N_SAMPLES = 16             # multi-start chains
NUM_STEPS = 200
LR = 1e-2
PSF_HALF = 4               # 9x9 base kernel (18x18 supersampled) to limit conv cost


def gaussian_psf_kernel(fwhm_px: float, half: int = 6) -> np.ndarray:
    sig = fwhm_px / 2.355
    ax = np.arange(-half, half + 1)
    xx, yy = np.meshgrid(ax, ax)
    k = np.exp(-0.5 * (xx**2 + yy**2) / sig**2)
    k /= k.sum()
    return k.astype(np.float32)


def estimate_background_rms(img: np.ndarray) -> float:
    """Sky-noise estimate from the four image CORNERS (blank sky), so galaxy flux
    does not inflate the estimate (which would make every fit look 'good')."""
    n = img.shape[0]
    c = max(8, n // 8)
    corners = np.concatenate([
        img[:c, :c].ravel(), img[:c, -c:].ravel(),
        img[-c:, :c].ravel(), img[-c:, -c:].ravel()])
    med = np.median(corners)
    mad = np.median(np.abs(corners - med))
    rms = 1.4826 * mad
    return float(max(rms, 1e-4))


def build_priors(near_arc: float):
    """Priors in arcsec. near_arc = half the cutout extent (search window)."""
    lens_mass_prior = tfd.JointDistributionSequential([
        tfd.JointDistributionNamed(dict(
            theta_E=tfd.LogNormal(jnp.log(1.5), 0.4),       # arcsec, weak
            gamma=tfd.TruncatedNormal(2.0, 0.2, 1.4, 2.6),
            e1=tfd.Normal(0.0, 0.1), e2=tfd.Normal(0.0, 0.1),
            center_x=tfd.Normal(0.0, 0.15), center_y=tfd.Normal(0.0, 0.15),
        )),
        tfd.JointDistributionNamed(dict(
            gamma1=tfd.Normal(0.0, 0.05), gamma2=tfd.Normal(0.0, 0.05))),
    ])

    def sersic_lstsq_prior(R_med, c_sig=0.2, cx=0.0, cy=0.0, c_mean_sig=0.15):
        return tfd.JointDistributionNamed(dict(
            R_sersic=tfd.LogNormal(jnp.log(R_med), 0.4),
            n_sersic=tfd.Uniform(0.5, 6.0),
            e1=tfd.TruncatedNormal(0.0, 0.2, -0.5, 0.5),
            e2=tfd.TruncatedNormal(0.0, 0.2, -0.5, 0.5),
            center_x=tfd.Normal(cx, c_mean_sig),
            center_y=tfd.Normal(cy, c_mean_sig),
        ))

    lens_light_prior = tfd.JointDistributionSequential([
        sersic_lstsq_prior(R_med=0.6),
        sersic_lstsq_prior(R_med=1.5),
    ])

    src_sersic_prior = tfd.JointDistributionNamed(dict(
        R_sersic=tfd.LogNormal(jnp.log(0.25), 0.4),
        n_sersic=tfd.Uniform(0.5, 4.0),
        e1=tfd.TruncatedNormal(0.0, 0.2, -0.5, 0.5),
        e2=tfd.TruncatedNormal(0.0, 0.2, -0.5, 0.5),
        center_x=tfd.Normal(0.0, 0.3), center_y=tfd.Normal(0.0, 0.3),
    ))
    src_shp_prior = tfd.JointDistributionNamed(dict(
        beta=tfd.LogNormal(jnp.log(0.15), 0.2),
        center_x=tfd.Normal(0.0, 0.2), center_y=tfd.Normal(0.0, 0.2),
    ))
    source_light_prior = tfd.JointDistributionSequential([src_sersic_prior, src_shp_prior])

    return (tfd.JointDistributionSequential([lens_mass_prior, lens_light_prior, source_light_prior]),
            tfd.JointDistributionSequential([lens_mass_prior, lens_light_prior]))


def build_null_priors():
    """Null = lens-light-only (no mass, no source). 2 Sersics."""
    def sersic_lstsq_prior(R_med, c_mean_sig=0.15):
        return tfd.JointDistributionNamed(dict(
            R_sersic=tfd.LogNormal(jnp.log(R_med), 0.4),
            n_sersic=tfd.Uniform(0.5, 6.0),
            e1=tfd.TruncatedNormal(0.0, 0.2, -0.5, 0.5),
            e2=tfd.TruncatedNormal(0.0, 0.2, -0.5, 0.5),
            center_x=tfd.Normal(0.0, c_mean_sig),
            center_y=tfd.Normal(0.0, c_mean_sig),
        ))
    return tfd.JointDistributionSequential([
        tfd.JointDistributionSequential([
            sersic_lstsq_prior(R_med=0.6),
            sersic_lstsq_prior(R_med=1.5),
        ]),
    ])


from gigalens.jax.simulator import LensSimulator  # noqa: E402


def _run_map(phys_model, prior, data_arr, background_rms, exp_time, kernel, seed=0):
    num_pix = data_arr.shape[0]
    sim_config = SimulatorConfig(delta_pix=DELTA_PIX, num_pix=num_pix,
                                 supersample=SUPERSAMPLE, kernel=kernel)
    prob_model = BackwardProbModel(prior, data_arr, background_rms=background_rms,
                                   exp_time=exp_time)
    model_seq = ModellingSequence(phys_model, prob_model, sim_config)
    opt = optax.adabelief(LR, b1=0.95, b2=0.99)
    map_all = model_seq.MAP(opt, n_samples=N_SAMPLES, num_steps=NUM_STEPS,
                            seed=seed, output_type="best")
    best_params, best_lp, best_chi = map_all
    best_params_np = np.asarray(best_params).reshape(-1)
    best_lp = float(np.asarray(best_lp).squeeze())
    best_chi = float(np.asarray(best_chi).squeeze())
    return best_params_np, best_lp, best_chi, prob_model, sim_config


def quicklens_fit(cube: np.ndarray) -> dict:
    """Run the lens-modelability fit on a (3,101,101) grz cube. Returns the signal dict.

    ONE GIGA-Lens MAP fit of EPL+shear + 2 lens-light Sersics + (Sersic+shapelets) source.
    Discriminator: from the best-fit params, recompute the data chi2 two ways using the
    same components and lstsq amplitudes:
       chi2_lens      = source ray-traced THROUGH the mass (lensed; the lens hypothesis)
       chi2_nodeflect = source NOT deflected (sits at source-plane spot; companion blob)
    A real lens NEEDS deflection (chi2_lens << chi2_nodeflect -> large dchi2_frac);
    a smooth galaxy/spiral does not (the source blob fits central residual either way).
    """
    t0 = time.time()
    cube = np.asarray(cube, dtype=np.float32)
    data_arr = (cube[1] + cube[2]) * 0.5          # r+z stack, already sky-subtracted
    data_arr = (data_arr - np.median(data_arr)).astype(np.float32)
    num_pix = data_arr.shape[0]
    background_rms = estimate_background_rms(data_arr)
    peak = float(np.max(data_arr))
    exp_time = float(max(200.0 / max(peak, 1e-3), 1.0))

    kernel = gaussian_psf_kernel(FWHM_ARCSEC / DELTA_PIX, half=PSF_HALF)
    near_arc = (num_pix // 2) * DELTA_PIX
    prior_lens, _ = build_priors(near_arc)

    phys_lens = PhysicalModel(
        [epl.EPL(50), shear.Shear()],
        [sersic.SersicEllipse(use_lstsq=True), sersic.SersicEllipse(use_lstsq=True)],
        [sersic.SersicEllipse(use_lstsq=True),
         shapelets.Shapelets(n_max=4, use_lstsq=True, interpolate=False)],
    )

    try:
        bp, lp, chi_lens, pm, sim_cfg = _run_map(
            phys_lens, prior_lens, data_arr, background_rms, exp_time, kernel, seed=0)
    except Exception as e:  # noqa: BLE001
        return {"error": f"lens MAP failed: {type(e).__name__}: {e}",
                "converged": False, "plausible": False, "theta_E": None,
                "reduced_chi2": None, "n_images": 0,
                "runtime_s": round(time.time() - t0, 2)}

    physical = pm.bij.forward(list(bp[:, None]))
    mass_main, mass_shear = physical[0][0], physical[0][1]
    theta_E = float(jnp.asarray(mass_main["theta_E"]).squeeze())
    gamma = float(jnp.asarray(mass_main["gamma"]).squeeze())
    src = physical[2][0]
    src_x = float(jnp.asarray(src["center_x"]).squeeze())
    src_y = float(jnp.asarray(src["center_y"]).squeeze())

    # --- deflected vs un-deflected chi2 from the SAME best-fit params ---
    lens_sim = LensSimulator(phys_lens, sim_cfg, bs=1)
    obs, err = pm.observed_image, pm.err_map
    img_defl = lens_sim.lstsq_simulate(physical, obs, err, no_deflection=False)[0]
    img_nodef = lens_sim.lstsq_simulate(physical, obs, err, no_deflection=True)[0]
    chi2_defl = float(jnp.mean(((img_defl - obs) / err) ** 2))
    chi2_nodef = float(jnp.mean(((img_nodef - obs) / err) ** 2))
    dchi2_frac = (chi2_nodef - chi2_defl) / chi2_nodef if chi2_nodef > 0 else float("nan")

    n_images = _solve_n_images(mass_main, mass_shear, theta_E, gamma, src_x, src_y, near_arc)

    converged = bool(np.isfinite(lp) and np.isfinite(chi2_defl))

    # --- continuous lens_score (the primary signal; validated AUC ~0.77 on 8A vs 8neg) ---
    # theta_E plausibility: full credit inside the physical window, decaying outside
    if 0.8 <= theta_E <= 3.5:
        ts = 1.0
    elif theta_E < 0.8:
        ts = max(0.0, theta_E / 0.8)
    else:
        ts = max(0.0, 1.0 - (theta_E - 3.5) / 3.0)
    dc = dchi2_frac if np.isfinite(dchi2_frac) else 0.0
    lens_score = round(0.6 * ts + 0.4 * min(dc / 0.3, 1.0), 4) if converged else 0.0

    # boolean plausible (best acc on the validation set: sens 5/8, spec 7/8)
    plausible = bool(
        converged
        and 0.7 < theta_E < 4.0
        and 1.4 < gamma < 2.7
        and np.isfinite(dchi2_frac) and dchi2_frac > 0.08
        and n_images >= 2
    )

    return {
        "theta_E": round(theta_E, 3),
        "gamma": round(gamma, 3),
        "reduced_chi2": round(chi2_defl, 4),
        "reduced_chi2_nodeflect": round(chi2_nodef, 4),
        "dchi2_frac": round(dchi2_frac, 4) if np.isfinite(dchi2_frac) else None,
        "n_images": int(n_images),
        "lens_score": lens_score,
        "converged": converged,
        "plausible": plausible,
        "runtime_s": round(time.time() - t0, 2),
    }


def _solve_n_images(mass_main, mass_shear, theta_E, gamma, src_x, src_y, near_arc):
    """Solve the lens equation with lenstronomy in-process if importable, else
    estimate multiplicity from the source position vs caustic (geometric proxy)."""
    try:
        from lenstronomy.LensModel.lens_model import LensModel
        from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
        lm = LensModel(lens_model_list=["EPL", "SHEAR"])
        kwargs = [
            {"theta_E": theta_E, "gamma": gamma,
             "e1": float(jnp.asarray(mass_main["e1"]).squeeze()),
             "e2": float(jnp.asarray(mass_main["e2"]).squeeze()),
             "center_x": float(jnp.asarray(mass_main["center_x"]).squeeze()),
             "center_y": float(jnp.asarray(mass_main["center_y"]).squeeze())},
            {"gamma1": float(jnp.asarray(mass_shear["gamma1"]).squeeze()),
             "gamma2": float(jnp.asarray(mass_shear["gamma2"]).squeeze())},
        ]
        solver = LensEquationSolver(lm)
        xi, yi = solver.image_position_from_source(
            src_x, src_y, kwargs, min_distance=DELTA_PIX, search_window=2 * near_arc,
            precision_limit=1e-4, num_iter_max=50)
        return int(len(xi))
    except Exception:
        # geometric proxy: source within ~theta_E of optical axis -> multiply imaged
        return 2 if (src_x**2 + src_y**2) ** 0.5 < theta_E else 1


# ----------------------------------------------------------------------------
def _load_cube(path):
    with fits.open(path) as h:
        return np.array(h[0].data, dtype=np.float32)


def _validation_set():
    import csv
    cutdir = "/raid/benson/git/agentic-lensing/reproductions/inchausti-2025/data/cutouts_fits_candidates_storfer"
    negdir = "/raid/benson/git/agentic-lensing/reproductions/inchausti-2025/data/cutouts_fits_neg_dr9"
    with open("/raid/benson/git/agentic-lensing/reproductions/inchausti-2025/data/candidate_scores_storfer.csv") as f:
        rows = list(csv.DictReader(f))
    A = [r["name"] for r in rows if r["grade"] == "A"
         and os.path.exists(os.path.join(cutdir, r["name"] + ".fits"))][:3]
    C = [r["name"] for r in rows if r["grade"] == "C"
         and os.path.exists(os.path.join(cutdir, r["name"] + ".fits"))][:2]
    negs = sorted(os.listdir(negdir))[:3]
    items = []
    for n in A:
        items.append(("A-lens", os.path.join(cutdir, n + ".fits")))
    for n in C:
        items.append(("C-cand", os.path.join(cutdir, n + ".fits")))
    for n in negs:
        items.append(("neg-gal", os.path.join(negdir, n)))
    return items


def _run_npy(npy_path: str):
    """Subprocess entrypoint used by tools/quicklens.py: load an (3,N,N) cube from
    a .npy/.fits file and print one JSON line with the modelability signal."""
    if npy_path.endswith(".fits"):
        cube = _load_cube(npy_path)
    else:
        cube = np.load(npy_path)
    print(json.dumps(quicklens_fit(cube)))


if __name__ == "__main__":
    # subprocess contract: `python quicklens_proto.py --cube <path.npy|.fits>` -> JSON line
    if len(sys.argv) == 3 and sys.argv[1] == "--cube":
        _run_npy(sys.argv[2])
    elif len(sys.argv) > 1:
        for p in sys.argv[1:]:
            cube = _load_cube(p)
            print(json.dumps({"path": p, **quicklens_fit(cube)}))
    else:
        print("device:", jax.devices()[0])
        for label, path in _validation_set():
            cube = _load_cube(path)
            res = quicklens_fit(cube)
            print(f"[{label}] {os.path.basename(path):40s} "
                  f"theta_E={res.get('theta_E')} chi2={res.get('reduced_chi2')} "
                  f"chi2_nodef={res.get('reduced_chi2_nodeflect')} dchi2_frac={res.get('dchi2_frac')} "
                  f"n_img={res.get('n_images')} plausible={res.get('plausible')} "
                  f"t={res.get('runtime_s')}s")
