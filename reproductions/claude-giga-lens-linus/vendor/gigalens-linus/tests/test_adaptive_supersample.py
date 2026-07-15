"""Adaptive supersampling (experimental): correctness + misuse-proofing.

Correctness is anchored to the existing, validated uniform pipeline:

* a factor-1 map must reproduce the ``supersample=1`` simulator EXACTLY (same
  points, same native-res PSF conv — any deviation is a wiring bug);
* a uniform factor-``f`` map uses the SAME sub-pixel quadrature as uniform
  ``supersample=f`` (equal up to float summation order, unconvolved);
* an all-subsample map equals the coarse-grid render replicated across blocks;
* the data-driven map on the bundled demo system must match a uniform
  ``supersample=8`` reference (full supersampled-PSF pipeline) to
  ``|delta|/sigma < 0.1`` on every pixel — the pre-registered falsifier for the
  default SNR thresholds AND the bin-first PSF approximation together.

Misuse-proofing follows the repo's raise-never-default pattern: wrong shapes,
unsupported factors, misaligned subsample blocks, unvalidated raw arrays,
non-monotone threshold specs, and composing a global supersample with an
adaptive map all fail loudly at construction.
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import pytest

import jax

jax.config.update("jax_enable_x64", True)

from jax import numpy as jnp
import tensorflow_probability.substrates.jax as tfp

tfd = tfp.distributions

import gigalens
from gigalens.simulator import SimulatorConfig
from gigalens.jax.profiles.mass.epl import EPL
from gigalens.jax.profiles.mass.shear import Shear
from gigalens.jax.profiles.light.sersic import SersicEllipse
from gigalens.jax.scene import Component, Plane, LensModel
from gigalens.jax.scene_simulator import SceneSimulator
from gigalens.jax.scene_prob_model import ProbModel
from gigalens.jax.experimental.adaptive_supersample import (
    ALLOWED_FACTORS,
    DEFAULT_SNR_LEVELS,
    AdaptiveGrid,
    AdaptiveImageData,
    AdaptiveSceneSimulator,
    compare_to_reference,
    plot_factor_map,
)

ROOT = gigalens.__path__[0]
BACKGROUND_RMS, EXP_TIME = 0.2, 100.0

TRUTH = {
    "planes/0/mass/0/theta_E": 1.1, "planes/0/mass/0/gamma": 2.0,
    "planes/0/mass/0/e1": 0.1, "planes/0/mass/0/e2": 0.1,
    "planes/0/mass/0/center_x": 0.1, "planes/0/mass/0/center_y": 0.0,
    "planes/0/mass/1/gamma1": -0.01, "planes/0/mass/1/gamma2": 0.03,
    "planes/0/light/0/R_sersic": 0.8, "planes/0/light/0/n_sersic": 2.5,
    "planes/0/light/0/e1": 0.09534746574143645,
    "planes/0/light/0/e2": 0.14849487967198177,
    "planes/0/light/0/center_x": 0.1, "planes/0/light/0/center_y": 0.0,
    "planes/0/light/0/Ie": 499.3695906504067,
    "planes/1/light/0/R_sersic": 0.25, "planes/1/light/0/n_sersic": 1.5,
    "planes/1/light/0/e1": 0.0, "planes/1/light/0/e2": 0.0,
    "planes/1/light/0/center_x": 0.09566681002252231,
    "planes/1/light/0/center_y": -0.0639623054267272,
    "planes/1/light/0/Ie": 149.58828877085668,
}


def _demo_model(use_lstsq=False):
    """The demos/simple_demo.ipynb system: EPL + shear, Sersic lens light,
    Sersic source."""
    epl = Component(EPL(50), dict(
        theta_E=tfd.LogNormal(jnp.log(1.25), 0.25),
        gamma=tfd.TruncatedNormal(2, 0.25, 1, 3),
        e1=tfd.Normal(0, 0.1), e2=tfd.Normal(0, 0.1),
        center_x=tfd.Normal(0, 0.05), center_y=tfd.Normal(0, 0.05)))
    shear = Component(Shear(), dict(
        gamma1=tfd.Normal(0, 0.05), gamma2=tfd.Normal(0, 0.05)))
    light_kw = dict(
        R_sersic=tfd.LogNormal(jnp.log(1.0), 0.15), n_sersic=tfd.Uniform(2, 6),
        e1=tfd.TruncatedNormal(0, 0.1, -0.3, 0.3),
        e2=tfd.TruncatedNormal(0, 0.1, -0.3, 0.3),
        center_x=tfd.Normal(0, 0.05), center_y=tfd.Normal(0, 0.05))
    source_kw = dict(
        R_sersic=tfd.LogNormal(jnp.log(0.25), 0.15), n_sersic=tfd.Uniform(0.5, 4),
        e1=tfd.TruncatedNormal(0, 0.15, -0.5, 0.5),
        e2=tfd.TruncatedNormal(0, 0.15, -0.5, 0.5),
        center_x=tfd.Normal(0, 0.25), center_y=tfd.Normal(0, 0.25))
    if not use_lstsq:
        light_kw["Ie"] = tfd.LogNormal(jnp.log(500.0), 0.3)
        source_kw["Ie"] = tfd.LogNormal(jnp.log(150.0), 0.5)
    lens_light = Component(SersicEllipse(use_lstsq=use_lstsq), light_kw)
    source_light = Component(SersicEllipse(use_lstsq=use_lstsq), source_kw)
    return LensModel([
        Plane(mass=[epl, shear], light=[lens_light]),
        Plane(deflection_ratio=1.0, light=[source_light]),
    ])


def _truth_params(model, batch=None):
    def leaf(v):
        return jnp.full((batch,), v) if batch else jnp.asarray(v)
    return model.to_params({n: leaf(TRUTH[n]) for n in model.z_param_names})


@pytest.fixture(scope="module")
def demo():
    kernel = np.load(f"{ROOT}/assets/psf.npy").astype(np.float32)
    observed = np.load(f"{ROOT}/assets/demo.npy")
    err_map = np.sqrt(BACKGROUND_RMS ** 2 + np.clip(observed, 0, np.inf) / EXP_TIME)
    cfg = SimulatorConfig(delta_pix=0.065, num_pix=60, supersample=1, kernel=kernel)
    return dict(kernel=kernel, observed=observed, err_map=err_map, cfg=cfg,
                model=_demo_model(use_lstsq=False),
                model_lstsq=_demo_model(use_lstsq=True))


# ---------------------------------------------------------------------------
# correctness against the uniform pipeline
# ---------------------------------------------------------------------------

def test_factor1_map_reproduces_base_exactly(demo):
    """All-ones map = the supersample=1 simulator: identical points, identical
    native-res PSF conv. Expected bitwise; tolerance only for op-fusion slack."""
    grid = AdaptiveGrid(np.ones((60, 60)))
    base = SceneSimulator(demo["model"], demo["cfg"])
    adap = AdaptiveSceneSimulator(demo["model"], demo["cfg"], grid)
    p = _truth_params(demo["model"])
    a = np.asarray(adap.simulate(p))
    b = np.asarray(base.simulate(p))
    np.testing.assert_allclose(a, b, rtol=1e-13, atol=0)


@pytest.mark.parametrize("f", [2, 4, 8])
def test_uniform_factor_matches_uniform_supersample_unconvolved(demo, f):
    """A uniform factor-f map uses the same sub-pixel points as supersample=f;
    unconvolved (kernel=None) the two pipelines differ only in float summation
    order (weighted scatter-add vs average_pool)."""
    cfg1 = SimulatorConfig(delta_pix=0.065, num_pix=60, supersample=1, kernel=None)
    cfgf = SimulatorConfig(delta_pix=0.065, num_pix=60, supersample=f, kernel=None)
    grid = AdaptiveGrid(np.full((60, 60), float(f)))
    base = SceneSimulator(demo["model"], cfgf)
    adap = AdaptiveSceneSimulator(demo["model"], cfg1, grid)
    p = _truth_params(demo["model"])
    np.testing.assert_allclose(np.asarray(adap.simulate(p)),
                               np.asarray(base.simulate(p)), rtol=1e-12)


def test_all_subsample_equals_replicated_coarse_grid(demo):
    """An all-1/4 map evaluates one point per aligned 4x4 block, at the block
    center — which is exactly the pixel grid of a 15x15 image at 4*delta_pix.
    The unconvolved render must equal that coarse render replicated blockwise."""
    cfg1 = SimulatorConfig(delta_pix=0.065, num_pix=60, supersample=1, kernel=None)
    cfg_coarse = SimulatorConfig(delta_pix=0.065 * 4, num_pix=15, supersample=1,
                                 kernel=None)
    grid = AdaptiveGrid(np.full((60, 60), 0.25))
    adap = AdaptiveSceneSimulator(demo["model"], cfg1, grid)
    coarse = SceneSimulator(demo["model"], cfg_coarse)
    p = _truth_params(demo["model"])
    a = np.asarray(adap.simulate(p))
    # coarse conversion_factor is (4*delta)^2; the adaptive map keeps native
    # pixel area — rescale to compare the quadrature (positions + values).
    c = np.asarray(coarse.simulate(p)) / 16.0
    np.testing.assert_allclose(a, np.kron(c, np.ones((4, 4))), rtol=1e-12)


def test_demo_falsifier_adaptive_matches_ss8_reference(demo):
    """THE acceptance gate: on the demo system at truth, the data-driven map must
    match the same-PSF-convention (bin-first) uniform supersample=8 reference to
    |delta|/sigma < 0.1 everywhere, at a fraction of the evaluation points."""
    grid = AdaptiveGrid.from_snr(demo["observed"], demo["err_map"])
    fm = grid.factor_map
    # the map must be genuinely adaptive on this system: refined arcs/center AND
    # subsampled sky (else the test is vacuous)
    assert (fm >= 2).sum() > 0, "no supersampled pixels — driver broken"
    assert (fm < 1).sum() > 0, "no subsampled pixels — driver broken"
    adap = AdaptiveSceneSimulator(demo["model"], demo["cfg"], grid)
    rep = compare_to_reference(adap, _truth_params(demo["model"]),
                               reference_supersample=8,
                               error_map=demo["err_map"])
    assert rep["max_abs_delta_over_sigma"] < 0.1, (
        f"max |delta|/sigma = {rep['max_abs_delta_over_sigma']:.3f} — factor map "
        "mis-assigned somewhere; plot rep['delta_over_sigma'] to locate it")
    # the efficiency claim: <0.1 sigma uniformly needs supersample=8 (64 pts/pix
    # — supersample=4 leaves 0.22 sigma errors); adaptive must beat that by >3x.
    assert grid.n_points < rep["n_points_reference"] / 3, (
        f"{grid.n_points} points vs {rep['n_points_reference']} for the uniform "
        "reference — the adaptive map no longer pays for itself")


def test_subgrid_psf_convention_gap_is_exposed(demo):
    """The bin-first vs subgrid PSF conventions REALLY differ on this system
    (~4 sigma at the lens-light cusp, measured 2026-07-12). compare_to_reference
    must expose that gap rather than hide it — if this number ever collapses,
    the demo/PSF changed and the module docstring's caveat needs re-measuring."""
    grid = AdaptiveGrid.from_snr(demo["observed"], demo["err_map"])
    adap = AdaptiveSceneSimulator(demo["model"], demo["cfg"], grid)
    rep = compare_to_reference(adap, _truth_params(demo["model"]),
                               reference_supersample=8, psf_mode="subgrid",
                               error_map=demo["err_map"])
    assert rep["psf_mode"] == "subgrid"
    assert rep["max_abs_delta_over_sigma"] > 1.0
    with pytest.raises(ValueError, match="psf_mode"):
        compare_to_reference(adap, _truth_params(demo["model"]),
                             psf_mode="banana")


def test_snr_driver_targets_bright_pixels(demo):
    grid = AdaptiveGrid.from_snr(demo["observed"], demo["err_map"])
    iy, ix = np.unravel_index(np.argmax(demo["observed"]), demo["observed"].shape)
    assert grid.factor_map[iy, ix] >= 2


def test_gradients_flow_through_adaptive_render(demo):
    grid = AdaptiveGrid.from_snr(demo["observed"], demo["err_map"])
    adap = AdaptiveSceneSimulator(demo["model"], demo["cfg"], grid)
    names = list(demo["model"].z_param_names)

    def loss(theta_e):
        vals = {n: jnp.asarray(theta_e if n == "planes/0/mass/0/theta_E"
                               else TRUTH[n]) for n in names}
        return jnp.sum(adap.simulate(demo["model"].to_params(vals)) ** 2)

    g = jax.grad(loss)(jnp.asarray(1.1))
    assert np.isfinite(float(g)) and float(g) != 0.0


def test_batched_params_match_unbatched(demo):
    grid = AdaptiveGrid.from_snr(demo["observed"], demo["err_map"])
    adap = AdaptiveSceneSimulator(demo["model"], demo["cfg"], grid)
    batched = np.asarray(adap.simulate(_truth_params(demo["model"], batch=3)))
    single = np.asarray(adap.simulate(_truth_params(demo["model"])))
    assert batched.shape == (3, 60, 60)
    for k in range(3):
        np.testing.assert_allclose(batched[k], single, rtol=1e-12)


# ---------------------------------------------------------------------------
# lstsq path (inherited solve over the overridden basis)
# ---------------------------------------------------------------------------

def test_lstsq_factor1_map_reproduces_base(demo):
    grid = AdaptiveGrid(np.ones((60, 60)))
    base = SceneSimulator(demo["model_lstsq"], demo["cfg"])
    adap = AdaptiveSceneSimulator(demo["model_lstsq"], demo["cfg"], grid)
    p = _truth_params(demo["model_lstsq"])
    a = np.asarray(adap.lstsq_simulate(p, demo["observed"], demo["err_map"]))
    b = np.asarray(base.lstsq_simulate(p, demo["observed"], demo["err_map"]))
    np.testing.assert_allclose(a, b, rtol=1e-10)


def test_lstsq_adaptive_close_to_uniform8_reference(demo):
    """lstsq quadrature convergence within the bin-first convention: the
    data-driven map vs a uniform all-8 map (both through the SAME adaptive conv
    path, whose factor-1 case is pinned to the base pipeline above)."""
    grid = AdaptiveGrid.from_snr(demo["observed"], demo["err_map"])
    ref_grid = AdaptiveGrid(np.full((60, 60), 8.0))
    adap = AdaptiveSceneSimulator(demo["model_lstsq"], demo["cfg"], grid)
    ref = AdaptiveSceneSimulator(demo["model_lstsq"], demo["cfg"], ref_grid)
    p = _truth_params(demo["model_lstsq"])
    a = np.asarray(adap.lstsq_simulate(p, demo["observed"], demo["err_map"]))
    r = np.asarray(ref.lstsq_simulate(p, demo["observed"], demo["err_map"]))
    assert np.max(np.abs(a - r) / demo["err_map"]) < 0.1


def test_lstsq_remat_matches_nonremat(demo):
    """remat_basis re-runs identical ops through the extracted _render_basis_nchw
    seam — exactness must survive the refactor, for base AND adaptive."""
    import dataclasses
    cfg_r = dataclasses.replace(demo["cfg"], remat_basis=True)
    p = _truth_params(demo["model_lstsq"])
    for cls, args in ((SceneSimulator, ()),
                      (AdaptiveSceneSimulator, (AdaptiveGrid(np.ones((60, 60))),))):
        plain = cls(demo["model_lstsq"], demo["cfg"], *args)
        remat = cls(demo["model_lstsq"], cfg_r, *args)
        np.testing.assert_allclose(
            np.asarray(remat.lstsq_simulate(p, demo["observed"], demo["err_map"])),
            np.asarray(plain.lstsq_simulate(p, demo["observed"], demo["err_map"])),
            rtol=1e-12)


# ---------------------------------------------------------------------------
# ProbModel integration
# ---------------------------------------------------------------------------

def test_probmodel_end_to_end_uses_adaptive_simulator(demo):
    ds = AdaptiveImageData(demo["observed"], demo["cfg"],
                           background_rms=BACKGROUND_RMS, exp_time=EXP_TIME,
                           sees="all")
    prob = ProbModel(demo["model"], ds, mode="forward")
    assert isinstance(prob.simulators[0], AdaptiveSceneSimulator)
    z = jnp.zeros(demo["model"].num_free_params)
    lp, red_chi2 = prob.log_prob(z)
    assert np.isfinite(float(lp)) and np.isfinite(float(red_chi2))


def test_with_map_remat_preserves_adaptive_quadrature(demo):
    """MAP's remat twin must not silently swap the adaptive simulator for a
    uniform one (the replace_config seam)."""
    ds = AdaptiveImageData(demo["observed"], demo["cfg"],
                           background_rms=BACKGROUND_RMS, exp_time=EXP_TIME,
                           sees="all")
    prob = ProbModel(demo["model"], ds, mode="forward")
    twin = prob.with_map_remat()
    sim = twin.simulators[0]
    assert isinstance(sim, AdaptiveSceneSimulator)
    assert sim.remat_basis is True
    assert sim.n_points == prob.simulators[0].n_points


# ---------------------------------------------------------------------------
# misuse-proofing (raise, never default)
# ---------------------------------------------------------------------------

def test_rejects_global_supersample_composition(demo):
    cfg2 = SimulatorConfig(delta_pix=0.065, num_pix=60, supersample=2,
                           kernel=demo["kernel"])
    with pytest.raises(ValueError, match="supersample must be 1"):
        AdaptiveSceneSimulator(demo["model"], cfg2, AdaptiveGrid(np.ones((60, 60))))


def test_rejects_wrong_shape_factor_map(demo):
    with pytest.raises(ValueError, match="factor map shape"):
        AdaptiveSceneSimulator(demo["model"], demo["cfg"],
                               AdaptiveGrid(np.ones((30, 30))))


def test_rejects_unvalidated_raw_array(demo):
    with pytest.raises(TypeError, match="AdaptiveGrid"):
        AdaptiveSceneSimulator(demo["model"], demo["cfg"], np.ones((60, 60)))


@pytest.mark.parametrize("bad", [3.0, 0.0, -1.0, 16.0])
def test_grid_rejects_unsupported_factors(bad):
    fm = np.ones((8, 8))
    fm[2, 2] = bad
    with pytest.raises(ValueError, match="unsupported factors"):
        AdaptiveGrid(fm)


def test_grid_rejects_nonfinite_and_wrong_ndim():
    fm = np.ones((8, 8))
    fm[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        AdaptiveGrid(fm)
    with pytest.raises(ValueError, match="2D"):
        AdaptiveGrid(np.ones(64))


def test_grid_rejects_misaligned_subsample_blocks():
    fm = np.ones((8, 8))
    fm[1:3, 0:2] = 0.5  # rows 1:3 straddle two aligned 2x2 blocks
    with pytest.raises(ValueError, match="aligned"):
        AdaptiveGrid(fm)
    fm = np.ones((10, 10))
    fm[2:6, 3:7] = 0.25  # cols 3:7 misaligned for a 4x4 block
    with pytest.raises(ValueError, match="aligned"):
        AdaptiveGrid(fm)
    fm = np.ones((5, 5))
    fm[3:5, 3:5] = 0.5  # aligned start would be row 4 — block leaves the image
    with pytest.raises(ValueError, match="aligned"):
        AdaptiveGrid(fm)


def test_grid_accepts_aligned_blocks_near_odd_edges():
    fm = np.ones((5, 5))
    fm[0:2, 0:2] = 0.5
    AdaptiveGrid(fm)  # must not raise: the block is aligned and fully inside


def test_quantize_promotes_never_demotes():
    d = np.ones((8, 8))
    d[1, 1] = 0.5                      # isolated wish: promoted to 1
    d[0:2, 4:6] = 0.5                  # aligned 2x2: kept
    d[4:8, 4:8] = 0.25                 # aligned 4x4: kept
    d[0, 0] = 4.0                      # >=1 wishes: untouched
    out = AdaptiveGrid.quantize(d)
    assert out[1, 1] == 1.0
    assert (out[0:2, 4:6] == 0.5).all()
    assert (out[4:8, 4:8] == 0.25).all()
    assert out[0, 0] == 4.0
    AdaptiveGrid(out)  # quantize output is always constructible


def test_from_snr_input_validation(demo):
    obs, err = demo["observed"], demo["err_map"]
    with pytest.raises(ValueError, match="strictly positive"):
        AdaptiveGrid.from_snr(obs, np.zeros_like(err))
    with pytest.raises(ValueError, match="identical 2D shapes"):
        AdaptiveGrid.from_snr(obs, err[:30, :30])
    with pytest.raises(ValueError, match="descending"):
        AdaptiveGrid.from_snr(obs, err, snr_levels=((8.0, 2.0), (40.0, 4.0),
                                                    (-np.inf, 1.0)))
    with pytest.raises(ValueError, match="descending"):
        AdaptiveGrid.from_snr(obs, err, snr_levels=((40.0, 2.0), (8.0, 4.0),
                                                    (-np.inf, 1.0)))
    with pytest.raises(ValueError, match=r"\(-inf, floor_factor\)"):
        AdaptiveGrid.from_snr(obs, err, snr_levels=((40.0, 4.0), (0.0, 1.0)))
    with pytest.raises(ValueError, match="not in"):
        AdaptiveGrid.from_snr(obs, err, snr_levels=((40.0, 3.0), (-np.inf, 1.0)))


def test_adaptive_imagedata_validation(demo):
    with pytest.raises(TypeError, match="AdaptiveGrid"):
        AdaptiveImageData(demo["observed"], demo["cfg"],
                          adaptive_grid=np.ones((60, 60)),
                          background_rms=BACKGROUND_RMS, exp_time=EXP_TIME,
                          sees="all")
    with pytest.raises(ValueError, match="shape"):
        AdaptiveImageData(demo["observed"], demo["cfg"],
                          adaptive_grid=AdaptiveGrid(np.ones((30, 30))),
                          background_rms=BACKGROUND_RMS, exp_time=EXP_TIME,
                          sees="all")


def test_snr_levels_can_disable_subsampling(demo):
    grid = AdaptiveGrid.from_snr(demo["observed"], demo["err_map"],
                                 snr_levels=((8.0, 2.0), (-np.inf, 1.0)))
    assert set(np.unique(grid.factor_map)) <= {1.0, 2.0}


# ---------------------------------------------------------------------------
# diagnostics utilities
# ---------------------------------------------------------------------------

def test_compare_to_reference_bookkeeping(demo):
    grid = AdaptiveGrid(np.ones((60, 60)))
    adap = AdaptiveSceneSimulator(demo["model"], demo["cfg"], grid)
    rep = compare_to_reference(adap, _truth_params(demo["model"]),
                               reference_supersample=4)
    assert rep["n_points"] == 3600
    assert rep["n_points_reference"] == 3600 * 16
    assert rep["adaptive"].shape == rep["reference"].shape == (60, 60)
    assert adap.n_points == grid.n_points  # two computations of one quantity


def test_bin_first_reference_pinned_to_base_at_ss1(demo):
    """The bin_first reference path (unconvolved render + post-hoc native conv)
    must reduce to the validated base pipeline when the quadrature is trivial —
    pins the reference's reshape/conv plumbing so the falsifier can't drift."""
    grid = AdaptiveGrid(np.ones((60, 60)))
    adap = AdaptiveSceneSimulator(demo["model"], demo["cfg"], grid)
    rep = compare_to_reference(adap, _truth_params(demo["model"]),
                               reference_supersample=1, psf_mode="bin_first")
    base = SceneSimulator(demo["model"], demo["cfg"])
    np.testing.assert_allclose(
        rep["reference"], np.asarray(base.simulate(_truth_params(demo["model"]))),
        rtol=1e-13)


def test_compact_peak_is_not_smoothed_below_refinement(demo):
    """A genuine single-pixel peak must keep its refinement tier: smoothing is
    for noise suppression at the LOW thresholds and must not average a real
    cusp/point source below the supersample thresholds (found in review: a 3x3
    box mean alone demotes an isolated SNR-30 peak to factor 1)."""
    err = np.full((60, 60), 1.0)
    img = np.zeros((60, 60))
    img[20, 20] = 30.0  # isolated compact peak, SNR 30 in a dark field
    grid = AdaptiveGrid.from_snr(img, err)
    assert grid.factor_map[20, 20] == 8.0


def test_quantize_rejects_nonfinite():
    d = np.ones((8, 8))
    d[3, 3] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        AdaptiveGrid.quantize(d)


def test_plot_factor_map_renders(demo, tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    grid = AdaptiveGrid.from_snr(demo["observed"], demo["err_map"])
    fig, ax = plot_factor_map(grid)
    fig.savefig(tmp_path / "factor_map.png", dpi=72)
    labels = [t.get_text() for t in fig.axes[-1].get_yticklabels()]
    assert any("native" in l for l in labels)  # identity is labeled, not color-alone
