#!/usr/bin/env python
"""01_gen_mocks.py -- Gu et al. 2022 (GIGA-Lens) §3 benchmark mock generator.

Simulate EPL + external-shear lenses with a Sersic source + Sersic lens light
on the paper's grid:

    pixel scale 0.065"/px, cutout 80 x 80 (5.2" x 5.2"), supersample = 2,
    sigma_bkg = 0.2, t_exp = 100 s, G = 1 (HST WFC3/F140W-class),
    PSF = the GIGA-Lens TinyTim kernel (gigalens/assets/psf.npy),
    typical arc SNR ~ 100 (set by the light-amplitude simulation distribution).

Parameters are drawn from the SIMULATION distribution of Eq. (8) (the "a" side of
the a/b notation; the broader "b" side is the modelling PRIOR used in 02_fit_system.py).

SIMULATOR CHOICE (--simulator):
  gigalens (DEFAULT): simulate with the SAME gigalens forward model that 02_fit_system.py
    fits, so the recovery test is self-consistent (no simulator-mismatch floor).
    Verified: at supersample=1 lenstronomy and gigalens agree to chi2=0.0, but at
    the paper's supersample=2 they differ (chi2~0.94 at identical truth) because
    lenstronomy's subgrid-kernel supersampled convolution and gigalens's internal
    supersampling are NOT bit-identical. Fitting a lenstronomy mock with gigalens
    then leaves an irreducible ~0.94 chi2 floor that biases the recovered params.
    The paper's own benchmark simulates and infers within one framework; we do the
    same with gigalens.
  lenstronomy: simulate with lenstronomy (the literal text of the paper). Use only
    at supersample=1, or accept the cross-simulator floor.

Truth (gigalens-format param list), the noiseless model, the noisy image, and the
noise level are saved per system to data/mocks/system_XXX.npz.

Usage:
    python 01_gen_mocks.py --n 12  --seed 0                      # validation (gigalens)
    python 01_gen_mocks.py --n 100 --seed 0                      # full paper benchmark
    python 01_gen_mocks.py --n 12  --seed 0 --simulator lenstronomy
"""
import os, sys, argparse, json
import numpy as np

GIGA_SRC = "/raid/benson/lensing-repos/gigalens/src"

GIGALENS_SRC = "/raid/benson/lensing-repos/gigalens/src"
PSF_PATH = "/raid/benson/lensing-repos/gigalens/src/gigalens/assets/psf.npy"

# ---- the paper's grid / noise constants (Fig. 10 caption + §2.2) -------------
DELTA_PIX = 0.065
NUM_PIX = 80
SUPERSAMPLE = 2
SIGMA_BKG = 0.2
EXP_TIME = 100.0
TN = lambda *a: a  # marker, unused


# ---- Eq. (8) SIMULATION distribution -----------------------------------------
# Each entry samples one physical parameter from the "a" (simulation) side.
def sample_truth(rng):
    """Return (gigalens_truth_list, flat_dict) drawn from the Eq.(8) sim dist."""
    def trunc_normal(mu, sd, lo, hi):
        # rejection sampling for a truncated normal
        while True:
            x = rng.normal(mu, sd)
            if lo <= x <= hi:
                return float(x)

    # --- EPL mass ---
    theta_E = float(np.exp(rng.normal(np.log(1.25), 0.25)))
    gamma   = trunc_normal(2.0, 0.25, 1.0, 3.0)
    e1      = float(rng.normal(0.0, 0.1))
    e2      = float(rng.normal(0.0, 0.1))
    cx      = float(rng.normal(0.0, 0.05))
    cy      = float(rng.normal(0.0, 0.05))
    # --- external shear ---
    g1 = float(rng.normal(0.0, 0.03))
    g2 = float(rng.normal(0.0, 0.03))
    # --- lens light (Sersic) ---
    Rl  = float(np.exp(rng.normal(np.log(1.6), 0.15)))
    nl  = float(rng.uniform(2.0, 6.0))
    le1 = trunc_normal(0.0, 0.05, -0.15, 0.15)
    le2 = trunc_normal(0.0, 0.05, -0.15, 0.15)
    lcx = float(rng.normal(0.0, 0.01))
    lcy = float(rng.normal(0.0, 0.01))
    Il  = float(np.exp(rng.normal(np.log(300.0), 0.3)))
    # --- source light (Sersic) ---
    Rs  = float(np.exp(rng.normal(np.log(0.25), 0.15)))
    ns  = float(rng.uniform(0.5, 4.0))
    se1 = trunc_normal(0.0, 0.15, -0.5, 0.5)
    se2 = trunc_normal(0.0, 0.15, -0.5, 0.5)
    scx = float(rng.normal(0.0, 0.25))
    scy = float(rng.normal(0.0, 0.25))
    Is  = float(np.exp(rng.normal(np.log(150.0), 0.5)))

    truth = [
        [
            {"theta_E": theta_E, "gamma": gamma, "e1": e1, "e2": e2,
             "center_x": cx, "center_y": cy},
            {"gamma1": g1, "gamma2": g2},
        ],
        [
            {"R_sersic": Rl, "n_sersic": nl, "e1": le1, "e2": le2,
             "center_x": lcx, "center_y": lcy, "Ie": Il},
        ],
        [
            {"R_sersic": Rs, "n_sersic": ns, "e1": se1, "e2": se2,
             "center_x": scx, "center_y": scy, "Ie": Is},
        ],
    ]
    flat = dict(
        theta_E=theta_E, gamma=gamma, e1=e1, e2=e2, center_x=cx, center_y=cy,
        gamma1=g1, gamma2=g2,
        ll_R_sersic=Rl, ll_n_sersic=nl, ll_e1=le1, ll_e2=le2,
        ll_center_x=lcx, ll_center_y=lcy, ll_Ie=Il,
        src_R_sersic=Rs, src_n_sersic=ns, src_e1=se1, src_e2=se2,
        src_center_x=scx, src_center_y=scy, src_Ie=Is,
    )
    return truth, flat


def build_lenstronomy_image(truth, psf_kernel):
    """Simulate the noiseless model with lenstronomy on the supersampled grid."""
    from lenstronomy.LensModel.lens_model import LensModel
    from lenstronomy.LightModel.light_model import LightModel
    from lenstronomy.ImSim.image_model import ImageModel
    from lenstronomy.Data.imaging_data import ImageData
    from lenstronomy.Data.psf import PSF
    from lenstronomy.Util import util as l_util
    from lenstronomy.Util import simulation_util as sim_util

    (mass, shear_), (lens_light,), (source_light,) = truth

    lens_model = LensModel(lens_model_list=["EPL", "SHEAR"])
    kwargs_lens = [
        dict(theta_E=mass["theta_E"], gamma=mass["gamma"], e1=mass["e1"],
             e2=mass["e2"], center_x=mass["center_x"], center_y=mass["center_y"]),
        dict(gamma1=shear_["gamma1"], gamma2=shear_["gamma2"]),
    ]
    ll_model = LightModel(light_model_list=["SERSIC_ELLIPSE"])
    src_model = LightModel(light_model_list=["SERSIC_ELLIPSE"])
    kwargs_ll = [dict(amp=lens_light["Ie"], R_sersic=lens_light["R_sersic"],
                      n_sersic=lens_light["n_sersic"], e1=lens_light["e1"],
                      e2=lens_light["e2"], center_x=lens_light["center_x"],
                      center_y=lens_light["center_y"])]
    kwargs_src = [dict(amp=source_light["Ie"], R_sersic=source_light["R_sersic"],
                       n_sersic=source_light["n_sersic"], e1=source_light["e1"],
                       e2=source_light["e2"], center_x=source_light["center_x"],
                       center_y=source_light["center_y"])]

    # data grid (NUM_PIX) centered on (0,0)
    kwargs_data = sim_util.data_configure_simple(NUM_PIX, DELTA_PIX, EXP_TIME,
                                                 SIGMA_BKG, inverse=False)
    data = ImageData(**kwargs_data)
    psf = PSF(psf_type="PIXEL", kernel_point_source=psf_kernel,
              point_source_supersampling_factor=SUPERSAMPLE)
    kwargs_numerics = dict(supersampling_factor=SUPERSAMPLE,
                           supersampling_convolution=True,
                           point_source_supersampling_factor=SUPERSAMPLE)
    image_model = ImageModel(data, psf, lens_model_class=lens_model,
                             source_model_class=src_model,
                             lens_light_model_class=ll_model,
                             kwargs_numerics=kwargs_numerics)
    model = image_model.image(kwargs_lens, kwargs_src, kwargs_ll)
    # lensed-source-only component (arcs), for the arc-SNR diagnostic
    arc_only = image_model.image(kwargs_lens, kwargs_src, kwargs_ll,
                                 lens_light_add=False)
    return np.array(model, dtype=np.float64), np.array(arc_only, dtype=np.float64)


_GIGA = {}  # cache the gigalens simulators (built once)


def build_gigalens_image(truth, psf_kernel):
    """Simulate the noiseless model with the SAME gigalens forward model the fitter
    uses (EPL+SHEAR mass, SersicEllipse lens+source light, supersample=2)."""
    if not _GIGA:
        sys.path.insert(0, GIGA_SRC)
        import gigalens.jax.simulator as gsim
        from gigalens.simulator import SimulatorConfig
        from gigalens.model import PhysicalModel
        from gigalens.jax.profiles.light import sersic
        from gigalens.jax.profiles.mass import epl, shear
        cfg = SimulatorConfig(delta_pix=DELTA_PIX, num_pix=NUM_PIX,
                              supersample=SUPERSAMPLE,
                              kernel=psf_kernel.astype(np.float32))
        pm = PhysicalModel([epl.EPL(50), shear.Shear()],
                           [sersic.SersicEllipse(use_lstsq=False)],
                           [sersic.SersicEllipse(use_lstsq=False)])
        _GIGA["full"] = gsim.LensSimulator(pm, cfg, bs=1)
        # lens-light-only sim (no lensing of a source): used to subtract for the
        # arc-SNR diagnostic. One Sersic lens light, no source.
        pm_ll = PhysicalModel([epl.EPL(50), shear.Shear()],
                              [sersic.SersicEllipse(use_lstsq=False)], [])
        _GIGA["ll"] = gsim.LensSimulator(pm_ll, cfg, bs=1)
    # full model
    model = np.array(_GIGA["full"].simulate(truth), dtype=np.float64)
    # lens-light-only -> arcs = full - lens light (diagnostic only)
    truth_ll = [truth[0], truth[1], []]
    ll_only = np.array(_GIGA["ll"].simulate(truth_ll), dtype=np.float64)
    arc_only = np.clip(model - ll_only, 0.0, None)
    return model, arc_only


def add_noise(model, rng):
    """Gaussian background + Poisson shot noise (G=1), paper §2.2."""
    err_map = np.sqrt(SIGMA_BKG ** 2 + np.clip(model, 0, None) / EXP_TIME)
    noisy = model + rng.normal(0.0, 1.0, size=model.shape) * err_map
    return noisy.astype(np.float64), err_map.astype(np.float64)


def arc_snr(arc_only, err_map):
    """Integrated arc SNR over the LENSED-SOURCE-ONLY image (lens light excluded):

        SNR = sum(arc_flux on arc pixels) / sqrt(sum(noise^2 on arc pixels))

    Arc pixels are those of the source-only image above 1 sigma of the per-pixel
    noise. This is the standard 'SNR of the arc' that the paper tunes to ~100
    (range 30-200). Reported as a sanity diagnostic."""
    mask = arc_only > err_map  # > 1 sigma per pixel
    if mask.sum() == 0:
        return 0.0
    sig = arc_only[mask].sum()
    noise = np.sqrt((err_map[mask] ** 2).sum())
    return float(sig / noise)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/mocks")
    ap.add_argument("--simulator", choices=["gigalens", "lenstronomy"],
                    default="gigalens",
                    help="gigalens (default, self-consistent) or lenstronomy")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    outdir = os.path.join(here, args.out)
    os.makedirs(outdir, exist_ok=True)

    psf_kernel = np.load(PSF_PATH).astype(np.float64)
    psf_kernel = psf_kernel / psf_kernel.sum()

    summary = []
    for i in range(args.n):
        rng = np.random.default_rng(args.seed * 100000 + i)
        truth, flat = sample_truth(rng)
        if args.simulator == "gigalens":
            model, arc_only = build_gigalens_image(truth, psf_kernel)
        else:
            model, arc_only = build_lenstronomy_image(truth, psf_kernel)
        noisy, err_map = add_noise(model, rng)
        snr = arc_snr(arc_only, err_map)
        path = os.path.join(outdir, f"system_{i:03d}.npz")
        np.savez(
            path,
            image=noisy.astype(np.float32),
            model=model.astype(np.float32),
            err_map=err_map.astype(np.float32),
            psf=psf_kernel.astype(np.float32),
            truth_json=json.dumps(truth),
            flat_keys=list(flat.keys()),
            flat_vals=np.array(list(flat.values()), dtype=np.float64),
            delta_pix=DELTA_PIX, num_pix=NUM_PIX, supersample=SUPERSAMPLE,
            sigma_bkg=SIGMA_BKG, exp_time=EXP_TIME, arc_snr=snr,
            simulator=args.simulator,
        )
        summary.append((i, flat["theta_E"], flat["gamma"], snr))
        print(f"[{i:03d}] theta_E={flat['theta_E']:.3f} gamma={flat['gamma']:.3f} "
              f"arc_snr~{snr:6.1f}  -> {os.path.basename(path)}", flush=True)

    snrs = np.array([s[3] for s in summary])
    print(f"\nGenerated {args.n} mocks in {outdir}")
    print(f"arc_snr (diagnostic): median={np.median(snrs):.1f} "
          f"range=[{snrs.min():.1f}, {snrs.max():.1f}]  (paper target ~100, 30-200)")


if __name__ == "__main__":
    main()
