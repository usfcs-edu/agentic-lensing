"""Validation for the lensed point-source scene integration (gigalens.jax.point_source).

Covers:
  * Link A — the ported physics (det A, Fermat potential) vs lenstronomy, and the
    differentiable cosmology (D_d/D_s/D_ds) vs astropy.
  * Scene wiring — a point-source-only ProbModel builds, log_prob is scalar-for-scalar /
    batched-for-batched, the H0 gradient flows through the time-delay distance (cosmology
    is genuinely differentiable), and num_pixels falls back to the observable count.
  * The two policy guards — a point-source dataset may not share a ProbModel with imaging,
    and a source off the cosmo reference plane is rejected.

Runs in float32 (no x64 toggling); tolerances are set for float32. lenstronomy/astropy are
required — the module skips if they are unavailable.
"""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp
pytest.importorskip("lenstronomy")
pytest.importorskip("astropy")

from lenstronomy.LensModel.lens_model import LensModel as LtLensModel
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
from astropy.cosmology import FlatLambdaCDM

from gigalens.jax import point_source as ps
from gigalens.jax.cosmo import wCDM_Cosmo
from gigalens.jax.profiles.mass.epl import EPL
from gigalens.jax.profiles.mass.shear import Shear
from gigalens.jax.profiles.light.point_source import PointSource
from gigalens.jax.scene import Component, Plane, LensModel
from gigalens.jax.scene_prob_model import ProbModel

tfd = tfp.distributions

EPL_T = dict(theta_E=1.167, gamma=2.0, e1=0.049, e2=0.083, center_x=0.005, center_y=0.011)
SHEAR_T = dict(gamma1=0.078, gamma2=0.015)
AMP_T, H0_T = 1.0, 70.0
Z_LENS, Z_SOURCE, OM0 = 0.2262, 0.3544, 0.3
SRC_X_T, SRC_Y_T = 0.0, 0.006


def _image_positions():
    lm = LtLensModel(lens_model_list=["EPL", "SHEAR"])
    kw = [dict(**EPL_T), dict(gamma1=SHEAR_T["gamma1"], gamma2=SHEAR_T["gamma2"], ra_0=0, dec_0=0)]
    cx, cy = LensEquationSolver(lm).image_position_from_source(
        SRC_X_T, SRC_Y_T, kw, min_distance=0.01, search_window=6, precision_limit=1e-10)
    return lm, kw, np.asarray(cx[:4]), np.asarray(cy[:4])


def _mass():
    return ([EPL(niter=18), Shear()],
            {0: {k: jnp.float32(v) for k, v in EPL_T.items()},
             1: {k: jnp.float32(v) for k, v in SHEAR_T.items()}})


def _cosmo_params():
    return dict(H0=jnp.float32(H0_T), Om0=jnp.float32(OM0),
                k=jnp.float32(0.0), w0=jnp.float32(-1.0))


def test_link_a_magnification_and_fermat_vs_lenstronomy():
    lm, kw, xs, ys = _image_positions()
    profs, mp = _mass()
    X, Y = jnp.asarray(xs, jnp.float32), jnp.asarray(ys, jnp.float32)

    det_a = np.asarray(ps._magnification(profs, mp, X, Y))
    det_a_lt = 1.0 / np.asarray(lm.magnification(xs, ys, kw))
    np.testing.assert_allclose(det_a, det_a_lt, rtol=1e-4,
                               err_msg="det A disagrees with lenstronomy")

    fermat = np.asarray(ps._fermat_potential(profs, mp, X, Y,
                                             jnp.float32(SRC_X_T), jnp.float32(SRC_Y_T)))
    fermat_lt = np.asarray(lm.fermat_potential(xs, ys, kw, x_source=SRC_X_T, y_source=SRC_Y_T))
    np.testing.assert_allclose(fermat, fermat_lt, atol=1e-4,
                               err_msg="Fermat potential disagrees with lenstronomy")


def test_link_a_cosmology_vs_astropy():
    cosmo = wCDM_Cosmo(z_lens=Z_LENS, z_source_ref=Z_SOURCE)
    cp = _cosmo_params()
    apc = FlatLambdaCDM(H0=H0_T, Om0=OM0, Ob0=0.0)
    # trapezoid (n=1000) vs astropy quad -> agree to ~1e-4.
    np.testing.assert_allclose(float(cosmo.angular_distance(Z_LENS, **cp)),
                               apc.angular_diameter_distance(Z_LENS).value, rtol=1e-3)
    np.testing.assert_allclose(float(cosmo.angular_distance(Z_SOURCE, **cp)),
                               apc.angular_diameter_distance(Z_SOURCE).value, rtol=1e-3)
    np.testing.assert_allclose(float(cosmo.angular_distance_z1z2(Z_LENS, Z_SOURCE, **cp)),
                               apc.angular_diameter_distance_z1z2(Z_LENS, Z_SOURCE).value, rtol=1e-3)


def _build_scene():
    ps_comp = Component(PointSource(use_lstsq=False), dict(amp=tfd.Normal(1.0, 0.1)))
    epl_comp = Component(EPL(niter=18), dict(
        theta_E=tfd.Uniform(0.5, 2.0), gamma=tfd.TruncatedNormal(2.0, 0.25, 1.5, 2.5),
        e1=tfd.Normal(0.0, 0.1), e2=tfd.Normal(0.0, 0.1),
        center_x=tfd.Normal(0.0, 0.1), center_y=tfd.Normal(0.0, 0.1)))
    shear_comp = Component(Shear(), dict(gamma1=tfd.Normal(0.0, 0.1), gamma2=tfd.Normal(0.0, 0.1)))
    cosmo_comp = Component(wCDM_Cosmo(z_lens=Z_LENS, z_source_ref=Z_SOURCE),
                           dict(H0=tfd.Uniform(0.0, 150.0), Om0=OM0, k=0.0, w0=-1.0))
    model = LensModel([Plane(redshift=Z_LENS, mass=[epl_comp, shear_comp]),
                       Plane(redshift=Z_SOURCE, light=[ps_comp])], cosmo=cosmo_comp)
    return model, ps_comp


def _data(ps_comp):
    lm, kw, xs, ys = _image_positions()
    profs, mp = _mass()
    # Fixed flat-LambdaCDM data: use the SAME astropy path the term selects, so the fit is
    # self-consistent (truth recovers exactly). The source position used for the truth-data
    # time delays is the TRUE source; the term derives its own from the delensed mean, which
    # equals the true source at truth.
    cl, cs = ps.precompute_comoving_distances(Z_LENS, Z_SOURCE, OM0)

    def tdall(x, y):
        f = ps._fermat_potential(profs, mp, x, y, jnp.float32(SRC_X_T), jnp.float32(SRC_Y_T))
        return ps._time_delay_days_fixed(f, jnp.float32(H0_T), cl, cs, Z_LENS, Z_SOURCE)

    X, Y = jnp.asarray(xs, jnp.float32), jnp.asarray(ys, jnp.float32)
    order = np.argsort(np.asarray(tdall(X, Y)))
    xs_s, ys_s = xs[order], ys[order]
    Xs, Ys = jnp.asarray(xs_s, jnp.float32), jnp.asarray(ys_s, jnp.float32)
    flux_obs = np.asarray(ps._magnification(profs, mp, Xs, Ys)) ** 2 / AMP_T ** 2
    td = np.asarray(tdall(Xs, Ys))
    return ps.PointSourceData(ps_comp, xs_s, ys_s, flux_obs, td - td[0], 3e3, 5e9, 3e3)


def test_scene_wiring_logprob_shapes_h0_gradient_and_num_pixels(recwarn):
    model, ps_comp = _build_scene()
    prob = ProbModel(model, _data(ps_comp))

    names = list(model.z_param_names)
    # num_pixels is a LITERAL imaging-pixel count -> 0 for a point-source-only model;
    # loss_normalization (what MAP divides by) falls back to the observable count.
    assert prob.num_pixels == 0
    assert prob.loss_normalization == prob.event_size > 0
    assert isinstance(prob.high_precision, bool)  # resolves without an imaging simulator

    truth_by_path = {"cosmo/H0": H0_T, "planes/0/mass/0/theta_E": EPL_T["theta_E"],
        "planes/0/mass/0/gamma": EPL_T["gamma"], "planes/0/mass/0/e1": EPL_T["e1"],
        "planes/0/mass/0/e2": EPL_T["e2"], "planes/0/mass/0/center_x": EPL_T["center_x"],
        "planes/0/mass/0/center_y": EPL_T["center_y"], "planes/0/mass/1/gamma1": SHEAR_T["gamma1"],
        "planes/0/mass/1/gamma2": SHEAR_T["gamma2"], "planes/1/light/0/amp": AMP_T}
    assert len(model.z_param_names) == 10  # source position is NOT sampled
    draw = model.prior.sample(seed=jax.random.PRNGKey(0))
    z = np.asarray(model.bijector.inverse(
        {k: jnp.asarray(truth_by_path[k], dtype=draw[k].dtype) for k in draw})).astype(np.float32)

    lp_s, _ = prob.log_prob(jnp.asarray(z))
    assert np.ndim(lp_s) == 0                       # scalar z -> scalar logp
    lp_b, _ = prob.log_prob(jnp.stack([z, z + 0.01]))
    assert np.shape(lp_b) == (2,)                   # batched z -> vector logp

    g = jax.grad(lambda zz: prob.log_prob(zz)[0])(jnp.asarray(z))
    gi = names.index("cosmo/H0")
    assert np.isfinite(float(g[gi])) and abs(float(g[gi])) > 0  # H0 differentiable via D_dt

    # truth is a local max of the (penalized) log-posterior
    lp_pert, _ = prob.log_prob(jnp.asarray(z) + 0.02)
    assert float(lp_s) > float(lp_pert)


def test_pointsource_only_probmodel_constructs():
    model, ps_comp = _build_scene()
    prob = ProbModel(model, _data(ps_comp))   # no imaging term required (chi2 stand-in)
    assert prob._chi2_event_size > 0


def test_mixing_pointsource_with_imaging_raises():
    from gigalens.jax.profiles.light.sersic import SersicEllipse
    from gigalens.jax.scene_prob_model import ImageData
    from gigalens.simulator import SimulatorConfig

    ll = Component(SersicEllipse(use_lstsq=True), dict(
        R_sersic=tfd.Uniform(0.1, 2.0), n_sersic=tfd.Uniform(1.0, 6.0),
        e1=tfd.Normal(0.0, 0.1), e2=tfd.Normal(0.0, 0.1),
        center_x=tfd.Normal(0.0, 0.1), center_y=tfd.Normal(0.0, 0.1)))
    ps_comp = Component(PointSource(use_lstsq=False), dict(amp=tfd.Normal(1.0, 0.1)))
    epl_comp = Component(EPL(niter=18), dict(
        theta_E=tfd.Uniform(0.5, 2.0), gamma=2.0, e1=0.0, e2=0.0, center_x=0.0, center_y=0.0))
    cosmo_comp = Component(wCDM_Cosmo(z_lens=Z_LENS, z_source_ref=Z_SOURCE),
                           dict(H0=tfd.Uniform(0.0, 150.0), Om0=OM0, k=0.0, w0=-1.0))
    model = LensModel([Plane(redshift=Z_LENS, mass=[epl_comp], light=[ll]),
                       Plane(redshift=Z_SOURCE, light=[ps_comp])], cosmo=cosmo_comp)
    data = ps.PointSourceData(ps_comp, [1.0, -1.0, 0.5, -0.5], [0.5, -0.5, 1.0, -1.0],
                              [0.01, 0.01, 0.01, 0.01], [0.0, 1.0, 2.0, 3.0], 3e3, 5e9, 3e3)
    img = ImageData(np.ones((20, 20), np.float32),
                    SimulatorConfig(delta_pix=0.05, num_pix=20, kernel=np.ones((3, 3), np.float32) / 9),
                    background_rms=0.01, exp_time=100.0, sees=[ll])
    with pytest.raises(NotImplementedError, match="point-source"):
        ProbModel(model, [data, img])


def test_source_off_cosmo_reference_plane_raises():
    # z_source_ref != source-plane redshift -> deflection_ratio != 1 -> rejected.
    ps_comp = Component(PointSource(use_lstsq=False), dict(amp=tfd.Normal(1.0, 0.1)))
    epl_comp = Component(EPL(niter=18), dict(
        theta_E=tfd.Uniform(0.5, 2.0), gamma=2.0, e1=0.0, e2=0.0, center_x=0.0, center_y=0.0))
    cosmo_comp = Component(wCDM_Cosmo(z_lens=Z_LENS, z_source_ref=Z_SOURCE + 0.5),
                           dict(H0=tfd.Uniform(0.0, 150.0), Om0=OM0, k=0.0, w0=-1.0))
    model = LensModel([Plane(redshift=Z_LENS, mass=[epl_comp]),
                       Plane(redshift=Z_SOURCE, light=[ps_comp])], cosmo=cosmo_comp)
    data = ps.PointSourceData(ps_comp, [1.0, -1.0, 0.5, -0.5], [0.5, -0.5, 1.0, -1.0],
                              [0.01, 0.01, 0.01, 0.01], [0.0, 1.0, 2.0, 3.0], 3e3, 5e9, 3e3)
    with pytest.raises(ValueError, match="z_source_ref"):
        ProbModel(model, data)


# ---------------------------------------------------------------- renders capability
def _sersic_and_pointsource_source_plane():
    """A source plane holding one RENDERABLE Sersic and one non-rendering PointSource,
    both fully fixed (constants), behind a fixed EPL lens. No cosmology (imaging idiom:
    deflection_ratio=1.0) — this is about rendering, not time delays."""
    from gigalens.jax.profiles.light.sersic import SersicEllipse
    sersic = Component(SersicEllipse(use_lstsq=False), dict(
        R_sersic=0.3, n_sersic=1.0, e1=0.0, e2=0.0, center_x=0.05, center_y=0.0, Ie=150.0))
    point = Component(PointSource(use_lstsq=False), dict(amp=1.0))
    lens = Component(EPL(niter=18), dict(
        theta_E=1.1, gamma=2.0, e1=0.03, e2=0.0, center_x=0.0, center_y=0.0))
    return sersic, point, lens


def test_pointsource_renders_false_and_base_light_raises():
    p = PointSource(use_lstsq=False)
    assert p.renders is False
    assert p.params == ["amp"]     # only amp is sampled; the source position is derived
    # PointSource defines no light(); the base (renders-aware) method raises.
    with pytest.raises(NotImplementedError, match="renders"):
        p.light(jnp.zeros(3), jnp.zeros(3))


def test_imaging_sees_all_excludes_point_source():
    from gigalens.jax.scene_prob_model import ImageData
    from gigalens.simulator import SimulatorConfig
    sersic, point, lens = _sersic_and_pointsource_source_plane()
    model = LensModel([Plane(mass=[lens]),
                       Plane(deflection_ratio=1.0, light=[sersic, point])])
    cfg = SimulatorConfig(delta_pix=0.05, num_pix=20, kernel=np.ones((3, 3), np.float32) / 9)
    seen = ImageData(np.ones((20, 20), np.float32), cfg,
                     background_rms=0.01, exp_time=100.0, sees="all").resolve_sees(model)
    seen_ids = {id(c) for c in seen}
    assert id(sersic) in seen_ids          # renderable light is seen
    assert id(point) not in seen_ids       # non-rendering emitter is NOT imaged by 'all'


def test_imaging_explicit_sees_pointsource_raises():
    from gigalens.jax.scene_prob_model import ImageData
    from gigalens.simulator import SimulatorConfig
    sersic, point, lens = _sersic_and_pointsource_source_plane()
    model = LensModel([Plane(mass=[lens]),
                       Plane(deflection_ratio=1.0, light=[sersic, point])])
    cfg = SimulatorConfig(delta_pix=0.05, num_pix=20, kernel=np.ones((3, 3), np.float32) / 9)
    with pytest.raises(ValueError, match="non-rendering"):
        ImageData(np.ones((20, 20), np.float32), cfg, background_rms=0.01, exp_time=100.0,
                  sees=[point]).resolve_sees(model)


def test_simulator_skips_non_rendering_light_in_basis_and_render():
    """A non-rendering PointSource in the light list contributes NOTHING to the image and
    NOTHING to the lstsq basis/depth — the render is identical to the plain (Sersic-only)
    model. This pins the silent-amplitude-bias hazard: a non-renderer must never enter the
    basis."""
    from gigalens.jax.scene_simulator import SceneSimulator
    from gigalens.simulator import SimulatorConfig
    sersic, point, lens = _sersic_and_pointsource_source_plane()
    cfg = SimulatorConfig(delta_pix=0.05, num_pix=20, kernel=np.ones((3, 3), np.float32) / 9)

    model_ps = LensModel([Plane(mass=[lens]),
                          Plane(deflection_ratio=1.0, light=[sersic, point])])
    model_plain = LensModel([Plane(mass=[lens]),
                             Plane(deflection_ratio=1.0, light=[sersic])])
    sim_ps = SceneSimulator(model_ps, cfg)
    sim_plain = SceneSimulator(model_plain, cfg)

    # the point source is excluded from the render bookkeeping and the depth total
    assert len(sim_ps._light) == len(sim_plain._light) == 1
    assert sim_ps.depth == sim_plain.depth
    assert all(id(comp) != id(point) for _, _, comp, _ in sim_ps._light)

    img_ps = np.asarray(sim_ps.simulate(model_ps.to_params({})))
    img_plain = np.asarray(sim_plain.simulate(model_plain.to_params({})))
    np.testing.assert_allclose(img_ps, img_plain, rtol=1e-6, atol=1e-6,
                               err_msg="non-rendering point source perturbed the image")


def test_pointsource_data_rejects_rendering_source():
    """A point-source dataset must observe a non-rendering emitter, not extended light."""
    from gigalens.jax.profiles.light.sersic import SersicEllipse
    sersic = Component(SersicEllipse(use_lstsq=False), dict(
        R_sersic=tfd.Uniform(0.1, 2.0), n_sersic=tfd.Uniform(1.0, 6.0), e1=0.0, e2=0.0,
        center_x=0.0, center_y=0.0, Ie=100.0))
    lens = Component(EPL(niter=18), dict(
        theta_E=1.1, gamma=2.0, e1=0.0, e2=0.0, center_x=0.0, center_y=0.0))
    cosmo_comp = Component(wCDM_Cosmo(z_lens=Z_LENS, z_source_ref=Z_SOURCE),
                           dict(H0=tfd.Uniform(0.0, 150.0), Om0=OM0, k=0.0, w0=-1.0))
    model = LensModel([Plane(redshift=Z_LENS, mass=[lens]),
                       Plane(redshift=Z_SOURCE, light=[sersic])], cosmo=cosmo_comp)
    data = ps.PointSourceData(sersic, [1.0, -1.0, 0.5, -0.5], [0.5, -0.5, 1.0, -1.0],
                              [0.01, 0.01, 0.01, 0.01], [0.0, 1.0, 2.0, 3.0], 3e3, 5e9, 3e3)
    with pytest.raises(ValueError, match="RENDERING|renders"):
        data.resolve_sees(model)


# ---------------------------------------------------------------- lensing potential
def test_mass_profile_without_potential_raises_by_name():
    """A mass profile that defines deriv but not potential (e.g. SIS) must fail loudly and
    BY NAME when the point-source path asks for a lensing potential — no silent fallback,
    and no parameter-name sniffing that a look-alike profile could fool."""
    from gigalens.jax.profiles.mass.sis import SIS
    with pytest.raises(NotImplementedError, match="SIS"):
        ps._lensing_potential([SIS()], {0: dict(theta_E=1.0, center_x=0.0, center_y=0.0)},
                              jnp.zeros(3), jnp.zeros(3))


def test_potential_gradient_equals_deflection():
    """alpha = grad(psi) by definition — the check that the moved EPL/Shear potential
    formulas are the true lensing potentials (independent of lenstronomy)."""
    epl, epl_kw = EPL(niter=18), EPL_T
    shear, sh_kw = Shear(), SHEAR_T
    for (x, y) in [(0.7, -0.3), (1.1, 0.9), (-0.5, 0.4)]:
        x, y = jnp.float32(x), jnp.float32(y)
        for prof, kw in ((epl, epl_kw), (shear, sh_kw)):
            gx, gy = jax.grad(lambda a, b: prof.potential(a, b, **kw), argnums=(0, 1))(x, y)
            ax, ay = prof.deriv(x, y, **kw)
            np.testing.assert_allclose([float(gx), float(gy)], [float(ax), float(ay)],
                                       rtol=1e-4, atol=1e-6,
                                       err_msg=f"grad(potential) != deriv for {prof}")


# ---------------------------------------------------------------- faithfulness to notebook
def test_fixed_flat_lcdm_selects_astropy_time_delay():
    """A fixed flat-LambdaCDM cosmology (Om0/k/w0 fixed) uses the astropy path — the one
    that is bit-faithful to the reference notebook."""
    model, ps_comp = _build_scene()
    prob = ProbModel(model, _data(ps_comp))
    assert prob.terms[0].cosmology_mode == "fixed-flat-lcdm-astropy"


def test_sampled_cosmology_uses_differentiable_trapezoid():
    """Freeing Om0 switches to the differentiable trapezoid cosmology (inference mode)."""
    ps_comp = Component(PointSource(use_lstsq=False), dict(amp=tfd.Normal(1.0, 0.1)))
    epl_comp = Component(EPL(niter=18), dict(
        theta_E=tfd.Uniform(0.5, 2.0), gamma=tfd.TruncatedNormal(2.0, 0.25, 1.5, 2.5),
        e1=tfd.Normal(0.0, 0.1), e2=tfd.Normal(0.0, 0.1),
        center_x=tfd.Normal(0.0, 0.1), center_y=tfd.Normal(0.0, 0.1)))
    shear_comp = Component(Shear(), dict(gamma1=tfd.Normal(0.0, 0.1), gamma2=tfd.Normal(0.0, 0.1)))
    cosmo_comp = Component(wCDM_Cosmo(z_lens=Z_LENS, z_source_ref=Z_SOURCE),
                           dict(H0=tfd.Uniform(0.0, 150.0), Om0=tfd.Uniform(0.1, 0.5), k=0.0, w0=-1.0))
    model = LensModel([Plane(redshift=Z_LENS, mass=[epl_comp, shear_comp]),
                       Plane(redshift=Z_SOURCE, light=[ps_comp])], cosmo=cosmo_comp)
    prob = ProbModel(model, _data(ps_comp))
    assert prob.terms[0].cosmology_mode == "differentiable-trapezoid"


def test_fixed_time_delay_matches_notebook_astropy_formula():
    """The fixed-cosmology time delay equals the notebook's D_dt formula built directly
    from astropy angular-diameter distances (bit-faithful, not the trapezoid)."""
    cl, cs = ps.precompute_comoving_distances(Z_LENS, Z_SOURCE, OM0)
    fermat = 0.037  # arcsec^2, arbitrary
    td = float(ps._time_delay_days_fixed(
        jnp.float64(fermat), jnp.float64(H0_T), cl, cs, Z_LENS, Z_SOURCE))
    apc = FlatLambdaCDM(H0=H0_T, Om0=OM0, Ob0=0.0)
    d_d = apc.angular_diameter_distance(Z_LENS).value
    d_s = apc.angular_diameter_distance(Z_SOURCE).value
    d_ds = apc.angular_diameter_distance_z1z2(Z_LENS, Z_SOURCE).value
    d_dt = (1 + Z_LENS) * d_d * d_s / d_ds  # Mpc
    mpc_m, c, arcsec, day = 3.0856775800000002e22, 2.9979246e8, 4.84813681109536e-06, 86400.0
    td_ref = d_dt * mpc_m / c * fermat * arcsec ** 2 / day
    np.testing.assert_allclose(td, td_ref, rtol=1e-4)


def test_loss_near_zero_at_truth_and_worse_when_perturbed():
    """The weighted loss (compactness + flux + time delay) is ~0 at the truth params and
    vastly larger away from them — confirms the restored consecutive-scatter L_D and
    derived source recover the reference behavior. (The sqrt-based L_D is float32-noise
    limited at truth, hence a loose absolute bound plus a large ratio.)"""
    model, ps_comp = _build_scene()
    prob = ProbModel(model, _data(ps_comp))
    truth_by_path = {"cosmo/H0": H0_T, "planes/0/mass/0/theta_E": EPL_T["theta_E"],
        "planes/0/mass/0/gamma": EPL_T["gamma"], "planes/0/mass/0/e1": EPL_T["e1"],
        "planes/0/mass/0/e2": EPL_T["e2"], "planes/0/mass/0/center_x": EPL_T["center_x"],
        "planes/0/mass/0/center_y": EPL_T["center_y"], "planes/0/mass/1/gamma1": SHEAR_T["gamma1"],
        "planes/0/mass/1/gamma2": SHEAR_T["gamma2"], "planes/1/light/0/amp": AMP_T}
    draw = model.prior.sample(seed=jax.random.PRNGKey(0))
    z = np.asarray(model.bijector.inverse(
        {k: jnp.asarray(truth_by_path[k], dtype=draw[k].dtype) for k in draw})).astype(np.float32)
    _, rc_truth = prob.log_prob(jnp.asarray(z))         # red_chi2 == the weighted loss stand-in
    _, rc_pert = prob.log_prob(jnp.asarray(z) + 0.02)
    assert float(rc_truth) < 1e-2                        # near-zero fit at truth
    assert float(rc_pert) > 1e3 * float(rc_truth)        # vastly worse away from truth
