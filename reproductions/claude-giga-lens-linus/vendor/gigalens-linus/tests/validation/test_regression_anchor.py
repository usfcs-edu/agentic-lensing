"""§6b refactor regression anchor: the NEW scene path == a FROZEN golden master of the
OLD prob_model/simulator path on a representative forward model, so the Phase-6 deletion
of the old API is provably safe (we keep proof the new code reproduces the old).

This is the grader the Phase-6 deletion depends on. It is a **code-path-equivalence**
claim (did the refactor change results?), distinct from the lenstronomy physics suite
(is the physics right?). See ``design/phase1-model-data-api.md`` §6b and §8.

**Golden-master freeze (Phase 6b Step 2).** The OLD API (``LensSimulator`` /
``BackwardProbModel`` / ``PhysicalModel``) is deleted in Phase 6b, so the anchor can no
longer construct it live. Before deletion the OLD path was run ONCE and its outputs
saved to ``fixtures/anchor_golden.npz`` (regenerate with the committed
``tools``/scratchpad ``make_golden.py``). This module now builds ONLY the NEW scene path
and asserts it reproduces those frozen OLD arrays — byte-/behavior-identical at the
float64 floor. The golden was produced on the SAME synthesized data and the SAME random
draws (PRNGKey(7)) this test uses, so the comparison is exact.

Scene (load-bearing machinery, mirroring the active ``vela_shapelets``
``BackwardProbModel`` build without needing real vela FITS — data is synthesized by a
forward render, as in the Phase-4/5 tests):

  * mass:        EPL + external Shear
  * lens light:  Sérsic (lstsq amplitude)
  * source:      Shapelets (lstsq amplitudes, n_max=6 → 28 basis components)
  * PSF:         a real (normalized Moffat) kernel
  * supersample: 2
  * precision:   likelihood_precision="float64"

Two graders:

  1. ``test_regression_anchor_new_vs_golden`` — fully-FIXED params: the rendered lstsq
     best-fit IMAGE, the lstsq coefficient vector (lens-light amp + 28 shapelet amps),
     and the log-likelihood / reduced χ², vs the frozen OLD arrays.
  2. ``test_regression_anchor_probmodel_free_params_vs_golden`` — FREE params: exercises
     the bijector + prior path (which the fixed build cannot), comparing the NEW scene
     ``ProbModel(mode="lstsq")`` ``log_like``/``log_prior`` against the frozen OLD
     ``BackwardProbModel`` values, draw-for-draw.

Pre-registration (method-discipline §3):
  * cause hypothesis: the new path calls the SAME ops as the old (EPL/Shear deriv,
    Sérsic/Shapelets light, subgrid_kernel(num_iter=100) → _convolve_components →
    average_pool_2d, _solve_normal_eq_with_fallback), visiting light in the SAME order
    (lens first, then source), and reduces the SAME masked Gaussian. A faithful refactor
    ⇒ results identical to the frozen golden up to the float64 reduction floor.
  * prediction: image ≲ 1e-10·peak, coeffs ≲ 1e-8 rel, log_like ≲ 1e-8 rel, free-param
    log_like/log_prior ≲ 1e-10 rel (tolerances.ANCHOR_*; derived there).
  * falsifier: an O(1) or structurally-localized residual ⇒ the new path is NOT
    equivalent — reported as a FINDING, not tuned away.
"""
import os

import numpy as np
import pytest
from jax import numpy as jnp

from . import tolerances as tol

pytestmark = pytest.mark.physics

_GOLDEN_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "anchor_golden.npz")


def _load_golden():
    if not os.path.exists(_GOLDEN_PATH):
        pytest.skip(
            f"anchor golden fixture missing at {_GOLDEN_PATH}; regenerate with the "
            "Phase-6b make_golden.py before the OLD API was deleted.")
    return np.load(_GOLDEN_PATH)


# Representative physical parameters (shared by the data-synthesis + fixed-param build).
EPL0 = dict(theta_E=1.1, gamma=2.05, e1=0.04, e2=-0.03, center_x=0.0, center_y=0.0)
SHEAR0 = dict(gamma1=0.03, gamma2=-0.02)
LL = dict(R_sersic=1.2, n_sersic=3.0, e1=0.0, e2=0.0, center_x=0.0, center_y=0.0)
SRC = dict(beta=0.15, center_x=0.05, center_y=-0.04)  # Shapelets geometry
N_MAX = 6           # 28 shapelet basis components
DR = 1.0            # single source plane, deflection_ratio reference (§3.1)
A_LL = 500.0        # planted lens-light amplitude (forward render of the data)
A_SRC0 = 200.0      # planted shapelet amp on the n=0 mode
_BG, _EXP = 0.01, 1000.0


def _moffat_psf(half=10, fwhm=3.0, beta=3.5):
    """A real normalized Moffat PSF kernel (not a Gaussian) — exercises subgrid_kernel."""
    yy, xx = np.mgrid[-half:half + 1, -half:half + 1]
    alpha = fwhm / (2.0 * np.sqrt(2.0 ** (1.0 / beta) - 1.0))
    k = (1.0 + (xx ** 2 + yy ** 2) / alpha ** 2) ** (-beta)
    return (k / k.sum()).astype(np.float64)


def _cfg(num_pix=60, supersample=2):
    from gigalens.simulator import SimulatorConfig
    return SimulatorConfig(delta_pix=0.05, num_pix=num_pix, supersample=supersample,
                           kernel=_moffat_psf(), likelihood_precision="float64")


# --------------------------------------------------------------------------------
# Data synthesis: forward-render a clean image (NEW path, use_lstsq=False), then add a
# real noise realization. The frozen golden was produced from this SAME synthesis.
# --------------------------------------------------------------------------------
def _synthesize_data():
    from gigalens.jax.profiles.mass.epl import EPL
    from gigalens.jax.profiles.mass.shear import Shear
    from gigalens.jax.profiles.light.sersic import SersicEllipse
    from gigalens.jax.scene import Component, Plane, LensModel
    from gigalens.jax.scene_simulator import SceneSimulator

    cfg = _cfg()
    SRC_SERSIC = dict(R_sersic=0.25, n_sersic=1.5, e1=0.05, e2=-0.02,
                      center_x=0.05, center_y=-0.04)
    fwd = LensModel([
        Plane(mass=[Component(EPL(50), EPL0), Component(Shear(), SHEAR0)],
              light=[Component(SersicEllipse(use_lstsq=False), dict(LL, Ie=A_LL))]),
        Plane(deflection_ratio=DR,
              light=[Component(SersicEllipse(use_lstsq=False),
                               dict(SRC_SERSIC, Ie=A_SRC0))]),
    ])
    clean = np.asarray(SceneSimulator(fwd, cfg).simulate(fwd.to_params({})))
    err = np.sqrt(_BG ** 2 + np.clip(clean, 0, np.inf) / _EXP)
    rng = np.random.default_rng(20260623)
    noisy = clean + rng.standard_normal(clean.shape) * err
    return np.asarray(clean), np.asarray(noisy), np.asarray(err)


# --------------------------------------------------------------------------------
# NEW path: scene LensModel + SceneSimulator (lstsq) + scene ProbModel.
# --------------------------------------------------------------------------------
def _build_new(observed, err):
    from gigalens.jax.profiles.mass.epl import EPL
    from gigalens.jax.profiles.mass.shear import Shear
    from gigalens.jax.profiles.light.sersic import SersicEllipse
    from gigalens.jax.profiles.light.shapelets import Shapelets
    from gigalens.jax.scene import Component, Plane, LensModel
    from gigalens.jax.scene_simulator import SceneSimulator
    from gigalens.jax.scene_prob_model import ImageData, ProbModel

    cfg = _cfg()
    model = LensModel([
        Plane(mass=[Component(EPL(50), EPL0), Component(Shear(), SHEAR0)],
              light=[Component(SersicEllipse(use_lstsq=True), LL)]),
        Plane(deflection_ratio=DR,
              light=[Component(Shapelets(n_max=N_MAX, use_lstsq=True), SRC)]),
    ])
    sim = SceneSimulator(model, cfg)
    params = model.to_params({})  # fully fixed → constants only; no free params
    obs = jnp.asarray(observed)
    e = jnp.asarray(err)
    image = np.asarray(sim.lstsq_simulate(params, obs, e))
    coeffs = np.squeeze(np.asarray(sim.lstsq_simulate(params, obs, e, return_coeffs=True)))

    ds = ImageData(observed, cfg, error_map=err, sees="all")
    prob = ProbModel(model, ds, mode="lstsq")
    mask = ds.mask.astype(image.dtype)
    diff = (image - observed) / err
    chi2 = float(np.sum(diff ** 2 * np.asarray(mask)))
    norm = float(np.sum(np.log(2 * np.pi * err ** 2) * np.asarray(mask)))
    log_like = -0.5 * (chi2 + norm)
    red_chi2 = chi2 / prob.event_size
    return dict(image=image, coeffs=np.atleast_1d(coeffs),
                log_like=log_like, red_chi2=red_chi2, event_size=prob.event_size)


# --------------------------------------------------------------------------------
# The fixed-param anchor: NEW scene path vs the FROZEN golden master.
# --------------------------------------------------------------------------------
def test_regression_anchor_new_vs_golden(plotter):
    """NEW scene path == FROZEN OLD prob_model/simulator outputs on the representative
    build.

    Asserts agreement on image, lstsq coeffs, and log_like/reduced-χ² at the float64
    reduction floor (tolerances.ANCHOR_*). A mismatch is a FINDING — reported with its
    magnitude and where it lives — not a tolerance to loosen.
    """
    golden = _load_golden()
    # Freeze the INPUT: load the exact noisy image + error map the golden was generated
    # from, instead of re-synthesizing. A JAX float64 forward render is NOT bit-reproducible
    # across separate processes (XLA reduction/FMA ordering differs at ~1e-14), so
    # re-synthesizing and demanding rtol=0/atol=0 equality against the frozen copy is an
    # invalid invariant (it passes only in the same process that wrote the golden). The
    # golden-master contract is: fix the input, run the code on it, compare the OUTPUT to the
    # frozen expected output at the derived ANCHOR_* floors. (Verified: a fresh synthesis
    # matches this frozen noisy to ~2e-14 — pure ULP nondeterminism, not a code change.)
    noisy = np.asarray(golden["noisy"])
    err = np.asarray(golden["err"])

    new = _build_new(noisy, err)

    new_img = np.squeeze(new["image"])
    old_img = np.squeeze(golden["image"])
    peak = float(np.max(np.abs(old_img))) or 1.0
    img_atol = tol.ANCHOR_IMAGE_ATOL_PEAK_FRAC * peak

    assert new["event_size"] == int(golden["event_size"]) == noisy.size, (
        f"event_size mismatch: new {new['event_size']} golden {int(golden['event_size'])} "
        f"pix {noisy.size}")

    plotter.image("anchor_image__new_vs_golden", old_img, new_img,
                  rtol=tol.ANCHOR_IMAGE_RTOL, atol=img_atol,
                  labels=("OLD (golden)", "NEW SceneSimulator"))

    old_coeffs = np.atleast_1d(np.asarray(golden["coeffs"]))
    img_max_abs = float(np.max(np.abs(new_img - old_img)))
    coeff_max_abs = float(np.max(np.abs(new["coeffs"] - old_coeffs)))
    old_log_like = float(golden["log_like"])
    old_red_chi2 = float(golden["red_chi2"])
    ll_rel = abs(new["log_like"] - old_log_like) / (abs(old_log_like) + 1e-300)
    rchi_rel = abs(new["red_chi2"] - old_red_chi2) / (abs(old_red_chi2) + 1e-300)

    # (1) image
    assert np.allclose(new_img, old_img, rtol=tol.ANCHOR_IMAGE_RTOL, atol=img_atol), (
        f"IMAGE mismatch: max|Δ|={img_max_abs:.3e} (peak={peak:.3e}, "
        f"max|Δ|/peak={img_max_abs/peak:.3e}) — FINDING, not a tolerance to loosen.")

    # (2) lstsq coefficients (lens-light amp + 28 shapelet amps, in the SAME order)
    assert new["coeffs"].shape == old_coeffs.shape, \
        f"coeff vector length differs: new {new['coeffs'].shape} golden {old_coeffs.shape}"
    assert np.allclose(new["coeffs"], old_coeffs,
                       rtol=tol.ANCHOR_COEFF_RTOL, atol=tol.ANCHOR_COEFF_ATOL), (
        f"COEFF mismatch: max|Δ|={coeff_max_abs:.3e}; "
        f"per-coeff Δ={np.array2string(new['coeffs']-old_coeffs, precision=2)}")

    # (3) log_like / reduced chi2
    assert ll_rel < tol.ANCHOR_LOGLIKE_RTOL, (
        f"LOG_LIKE mismatch: new {new['log_like']:.10e} vs golden {old_log_like:.10e} "
        f"(rel {ll_rel:.3e})")
    assert rchi_rel < tol.ANCHOR_LOGLIKE_RTOL, (
        f"RED_CHI2 mismatch: new {new['red_chi2']:.10e} vs golden {old_red_chi2:.10e} "
        f"(rel {rchi_rel:.3e})")

    print(f"\n[anchor] image max|Δ|={img_max_abs:.3e} (max|Δ|/peak={img_max_abs/peak:.3e}); "
          f"coeff max|Δ|={coeff_max_abs:.3e}; log_like rel={ll_rel:.3e}; "
          f"red_chi2 new={new['red_chi2']:.6f} golden={old_red_chi2:.6f} rel={rchi_rel:.3e}")


# --------------------------------------------------------------------------------
# §6b prob-model FREE-param companion vs the FROZEN golden: exercises the bijector +
# prior path that the fully-fixed anchor above cannot.
# --------------------------------------------------------------------------------
def _new_free_model():
    from gigalens.jax.profiles.mass.epl import EPL
    from gigalens.jax.profiles.mass.shear import Shear
    from gigalens.jax.profiles.light.sersic import SersicEllipse
    from gigalens.jax.profiles.light.shapelets import Shapelets
    from gigalens.jax.scene import Component, Plane, LensModel
    from tensorflow_probability.substrates.jax import distributions as tfd

    return LensModel([
        Plane(mass=[Component(EPL(50), dict(
                    theta_E=tfd.LogNormal(jnp.log(1.1), 0.4),
                    gamma=tfd.TruncatedNormal(2.05, 0.5, 1.0, 3.0),
                    e1=tfd.TruncatedNormal(0.04, 0.2, -0.5, 0.5),
                    e2=tfd.TruncatedNormal(-0.03, 0.2, -0.5, 0.5),
                    center_x=tfd.Normal(0.0, 0.06),
                    center_y=tfd.Normal(0.0, 0.06))),
                 Component(Shear(), dict(
                    gamma1=tfd.TruncatedNormal(0.03, 0.1, -0.5, 0.5),
                    gamma2=tfd.Normal(-0.02, 0.1)))],
              light=[Component(SersicEllipse(use_lstsq=True), dict(
                    R_sersic=tfd.LogNormal(jnp.log(1.2), 0.25),
                    n_sersic=tfd.Uniform(0.5, 8.0),
                    e1=tfd.TruncatedNormal(0.0, 0.1, -0.2, 0.2),
                    e2=tfd.TruncatedNormal(0.0, 0.1, -0.2, 0.2),
                    center_x=tfd.Normal(0.0, 0.02),
                    center_y=tfd.Normal(0.0, 0.02)))]),
        Plane(deflection_ratio=DR,
              light=[Component(Shapelets(n_max=N_MAX, use_lstsq=True), dict(
                    beta=tfd.LogNormal(jnp.log(0.15), 0.4),
                    center_x=tfd.Normal(0.0, 0.5),
                    center_y=tfd.Normal(0.0, 0.5)))]),
    ])


def test_regression_anchor_probmodel_free_params_vs_golden(plotter):
    """NEW scene ProbModel.log_like/log_prob == FROZEN OLD BackwardProbModel through the
    bijector + prior (FREE params), closing the 6a tautology gap.

    The golden froze the OLD ``log_like``/``log_prior`` at the matched physical params
    for each of the deterministic PRNGKey(7) draws; this test re-draws the SAME ``z``,
    evaluates the NEW scene ProbModel, and compares draw-for-draw.

    Pre-registration (method-discipline §3):
      * cause hypothesis: both prob models reduce the SAME masked Gaussian on the SAME
        lstsq image and add the SAME log-prior (identical 1-D marginals + per-coordinate
        Jacobian).
      * prediction: log_like and log_prior agree at the float64 reduction floor
        (≲1e-12 rel; tolerances.ANCHOR_PROB_*).
      * falsifier: a ≳1e-8 rel gap ⇒ a real semantic difference — a FINDING.
    """
    from gigalens.jax.scene_simulator import SceneSimulator
    from gigalens.jax.scene_prob_model import ImageData, ProbModel
    import jax

    golden = _load_golden()
    n_draws = int(golden["prob_n_draws"])
    seed = int(golden["prob_seed"])
    ll_old_v = np.asarray(golden["prob_log_like"], dtype=np.float64)
    lp_old_v = np.asarray(golden["prob_log_prior"], dtype=np.float64)

    clean, noisy, err = _synthesize_data()
    cfg = _cfg()
    model = _new_free_model()
    sim_new = SceneSimulator(model, cfg)
    ds = ImageData(noisy, cfg, error_map=err, sees="all")
    prob_new = ProbModel(model, ds, mode="lstsq")

    assert prob_new.event_size == int(golden["event_size"]) == noisy.size, (
        f"event_size mismatch: new {prob_new.event_size} golden "
        f"{int(golden['event_size'])} pix {noisy.size}")

    key = jax.random.PRNGKey(seed)
    ll_new_v, lp_new_v = [], []
    for _ in range(n_draws):
        key, sk = jax.random.split(key)
        z_new = jax.random.normal(sk, (model.num_free_params,), dtype=jnp.float64)
        ll_new, _ = prob_new.log_like(z_new[None, :], simulator=sim_new)
        lp_new = prob_new.log_prior(z_new[None, :])
        ll_new_v.append(float(np.asarray(ll_new).squeeze()))
        lp_new_v.append(float(np.asarray(lp_new).squeeze()))
    ll_new_v = np.array(ll_new_v)
    lp_new_v = np.array(lp_new_v)
    idx = np.arange(n_draws)

    plotter.curve("anchor_prob_loglike__new_vs_golden", idx, ll_old_v, ll_new_v,
                  xlabel="z draw", ylabel="log_like",
                  labels=("OLD (golden)", "NEW scene ProbModel"))
    plotter.curve("anchor_prob_logprior__new_vs_golden", idx, lp_old_v, lp_new_v,
                  xlabel="z draw", ylabel="log_prior",
                  labels=("OLD (golden)", "NEW scene ProbModel"))

    ll_rel = np.abs(ll_new_v - ll_old_v) / (np.abs(ll_old_v) + 1e-300)
    lp_rel = np.abs(lp_new_v - lp_old_v) / (np.abs(lp_old_v) + 1e-300)

    assert np.max(ll_rel) < tol.ANCHOR_PROB_LOGLIKE_RTOL, (
        f"LOG_LIKE (free params) mismatch: max rel={np.max(ll_rel):.3e} — FINDING. "
        f"per-draw rel={np.array2string(ll_rel, precision=2)}")
    assert np.max(lp_rel) < tol.ANCHOR_PROB_LOGPRIOR_RTOL, (
        f"LOG_PRIOR (free params) mismatch: max rel={np.max(lp_rel):.3e} — FINDING. "
        f"per-draw rel={np.array2string(lp_rel, precision=2)}")

    print(f"\n[anchor-prob] free-param NEW==golden over {n_draws} draws: "
          f"log_like max rel={np.max(ll_rel):.3e}; log_prior max rel={np.max(lp_rel):.3e}")
