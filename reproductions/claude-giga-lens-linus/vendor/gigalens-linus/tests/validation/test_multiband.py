"""Multi-dataset / ``sees`` validation (Phase 5).

The Phase-5 surface adds a per-dataset light *view* (``sees``) and a dataset loop in
``ProbModel``: the shared lens mass + cosmology are traced for every dataset, while each
dataset renders only the light Components it sees, on its own grid, with its own PSF, and
(in lstsq mode) solves its amplitudes independently. These tests pin that the right
light is rendered for each band, that sharing a parameter genuinely reduces the sampled
dimensionality (no double-counting), that an lstsq Component shared across bands keeps
ONE shape but FREE per-band flux, that a 1-dataset ProbModel built the new way reduces to
the Phase-4 result, and that the §3.5/§3.6/§3.7 validators raise on the offending
constructions.

Claim types (method-discipline §2 — name the link):
  - render-vs-oracle  → deterministic identity vs a lenstronomy-deflection render of THAT
                        band's components, held at IMAGE_NOPSF_F64 (no PSF, supersample=1
                        isolates trace+assembly+sees-filtering from sersic/PSF parity).
  - dimensionality    → exact integer identity (z length / num_free_params); no tolerance.
  - shared-lstsq flux → identical basis (REDUCTION floor) but per-band amplitudes that
                        differ by exactly the planted ratio (LINEARITY floor).
  - reduction guard   → 1-dataset new-way == Phase-4 single-sim path (REDUCTION floor).
  - validators        → software contract (pytest.raises), no tolerance.
"""
import numpy as np
import pytest

from . import tolerances as tol

pytestmark = pytest.mark.physics

EPL0 = dict(theta_E=1.1, gamma=2.05, e1=0.04, e2=-0.03, center_x=0.0, center_y=0.0)
LL = dict(R_sersic=1.2, n_sersic=3.0, e1=0.0, e2=0.0, center_x=0.0, center_y=0.0)
# Two source morphologies that SHARE a centre but differ in shape (different bands).
SRC_A = dict(R_sersic=0.30, n_sersic=2.0, e1=0.05, e2=-0.02)   # centre supplied via shared
SRC_B = dict(R_sersic=0.18, n_sersic=1.2, e1=-0.04, e2=0.03)   # centre supplied via shared
CX, CY = 0.05, -0.04


def _gauss_psf(sigma=2.0, half=10):
    yy, xx = np.mgrid[-half:half + 1, -half:half + 1]
    k = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return k / k.sum()


def _cfg(num_pix=50, delta_pix=0.05, supersample=1, kernel=None):
    from gigalens.simulator import SimulatorConfig
    return SimulatorConfig(delta_pix=delta_pix, num_pix=num_pix, supersample=supersample,
                           kernel=kernel, likelihood_precision="float64")


# ----------------------------------------------------------------------------------
# 1. Two bands, shared source position, different morphology per band.
# ----------------------------------------------------------------------------------
def test_two_band_shared_position_different_morphology(plotter):
    """Each band's image renders ONLY its own seen source, lensed through the shared
    mass, and matches a lenstronomy-deflection render of THAT band's component.

    Metric: max relative error of the per-band gigalens image vs an independent render
    that uses gigalens light at a lenstronomy-computed β (so the sersic profile itself is
    not under test — only trace + sees-filtering + assembly + flux scale). Threshold:
    IMAGE_NOPSF_F64 (~1e-9), ~6 orders above float64 eps. Falsifier / pre-committed
    appearance: a structureless residual under 1e-9; ANY band leaking the other band's
    source, or rendering at the wrong β, shows up as O(1) structured residual far above
    1e-9.
    """
    pytest.importorskip("lenstronomy")
    from lenstronomy.LensModel.Profiles.epl import EPL as LEPL
    from gigalens.jax.profiles.mass.epl import EPL
    from gigalens.jax.profiles.light.sersic import SersicEllipse
    from gigalens.jax.scene import Component, Plane, LensModel, shared
    from gigalens.jax.scene_prob_model import ImageData, ProbModel
    import tensorflow_probability.substrates.jax as tfp
    tfd = tfp.distributions

    cx = shared(tfd.Normal(0.0, 0.5))
    cy = shared(tfd.Normal(0.0, 0.5))
    src_a = Component(SersicEllipse(use_lstsq=False),
                      dict({k: tfd.Normal(float(v), 0.3) for k, v in SRC_A.items()},
                           Ie=tfd.LogNormal(np.log(200.0), 0.3), center_x=cx, center_y=cy))
    src_b = Component(SersicEllipse(use_lstsq=False),
                      dict({k: tfd.Normal(float(v), 0.3) for k, v in SRC_B.items()},
                           Ie=tfd.LogNormal(np.log(120.0), 0.3), center_x=cx, center_y=cy))
    model = LensModel([
        Plane(mass=[Component(EPL(50), {k: tfd.Normal(float(v), 0.3) for k, v in EPL0.items()})]),
        Plane(deflection_ratio=1.0, light=[src_a, src_b]),
    ])

    cfg = _cfg()
    img0 = np.zeros((cfg.num_pix, cfg.num_pix))
    err0 = np.ones_like(img0)
    da = ImageData(img0, cfg, error_map=err0, sees=[src_a])
    db = ImageData(img0, cfg, error_map=err0, sees=[src_b])
    prob = ProbModel(model, [da, db], mode="forward")

    # Truth at the shared centre + per-band shapes/fluxes.
    IE_A, IE_B = 200.0, 120.0
    truth = {
        "planes/0/mass/0/theta_E": EPL0["theta_E"], "planes/0/mass/0/gamma": EPL0["gamma"],
        "planes/0/mass/0/e1": EPL0["e1"], "planes/0/mass/0/e2": EPL0["e2"],
        "planes/0/mass/0/center_x": EPL0["center_x"], "planes/0/mass/0/center_y": EPL0["center_y"],
        "planes/1/light/0/R_sersic": SRC_A["R_sersic"], "planes/1/light/0/n_sersic": SRC_A["n_sersic"],
        "planes/1/light/0/e1": SRC_A["e1"], "planes/1/light/0/e2": SRC_A["e2"],
        "planes/1/light/0/Ie": IE_A,
        "planes/1/light/1/R_sersic": SRC_B["R_sersic"], "planes/1/light/1/n_sersic": SRC_B["n_sersic"],
        "planes/1/light/1/e1": SRC_B["e1"], "planes/1/light/1/e2": SRC_B["e2"],
        "planes/1/light/1/Ie": IE_B,
        "shared_%d" % cx.uid: CX, "shared_%d" % cy.uid: CY,
    }
    import jax.numpy as jnp
    unique = {k: jnp.asarray(float(truth[k])) for k in model.prior.model.keys()}
    params = model.to_params(unique)

    got_a = np.asarray(prob.simulators[0].simulate(params))
    got_b = np.asarray(prob.simulators[1].simulate(params))

    # Independent per-band reference: gigalens light at a lenstronomy β, that band only.
    sim = prob.simulators[0]
    gx = np.asarray(sim.img_X)[..., 0]
    gy = np.asarray(sim.img_Y)[..., 0]
    ax, ay = LEPL().derivatives(x=gx, y=gy, **EPL0)
    bx, by = gx - ax, gy - ay
    cf = float(np.linalg.det(np.asarray(sim.transform_pix2angle)))
    ref_a = np.asarray(src_a.profile.light(bx[..., None], by[..., None],
                                           Ie=IE_A, center_x=CX, center_y=CY, **SRC_A))[..., 0] * cf
    ref_b = np.asarray(src_b.profile.light(bx[..., None], by[..., None],
                                           Ie=IE_B, center_x=CX, center_y=CY, **SRC_B))[..., 0] * cf

    plotter.image("band_A_seen_src_a_only", ref_a, got_a,
                  rtol=tol.IMAGE_NOPSF_F64_RTOL, atol=tol.IMAGE_NOPSF_F64_ATOL,
                  labels=("lenstronomy-β render (src_a)", "gigalens band A"))
    plotter.image("band_B_seen_src_b_only", ref_b, got_b,
                  rtol=tol.IMAGE_NOPSF_F64_RTOL, atol=tol.IMAGE_NOPSF_F64_ATOL,
                  labels=("lenstronomy-β render (src_b)", "gigalens band B"))
    assert np.allclose(got_a, ref_a, rtol=tol.IMAGE_NOPSF_F64_RTOL, atol=tol.IMAGE_NOPSF_F64_ATOL), \
        f"band A max|Δ|={np.max(np.abs(got_a-ref_a)):.2e}"
    assert np.allclose(got_b, ref_b, rtol=tol.IMAGE_NOPSF_F64_RTOL, atol=tol.IMAGE_NOPSF_F64_ATOL), \
        f"band B max|Δ|={np.max(np.abs(got_b-ref_b)):.2e}"
    # Cross-leak guard: band A must NOT contain src_b's flux (and vice versa). If sees
    # leaked, got_a would equal ref_a+ref_b; assert it is far from that.
    assert np.max(np.abs(got_a - (ref_a + ref_b))) > 1e-3 * np.max(np.abs(ref_b)), \
        "band A appears to also render src_b — sees filtering leaked"


def test_shared_centre_reduces_dimensionality():
    """Sharing the source centre via shared(...) reduces the unique free-param count by
    exactly 2 (one shared contributes ONE z slot, not one per site) — the §4 / §4-rigor
    double-counting falsifier. Exact integer identity; no tolerance.
    """
    from gigalens.jax.profiles.mass.epl import EPL
    from gigalens.jax.profiles.light.sersic import SersicEllipse
    from gigalens.jax.scene import Component, Plane, LensModel, shared
    import tensorflow_probability.substrates.jax as tfp
    import jax.numpy as jnp
    tfd = tfp.distributions

    def _mass():
        return Component(EPL(50), {k: tfd.Normal(float(v), 0.3) for k, v in EPL0.items()})

    def _src(shape, cx, cy):
        return Component(SersicEllipse(use_lstsq=False),
                         dict({k: tfd.Normal(float(v), 0.3) for k, v in shape.items()},
                              Ie=tfd.LogNormal(0.0, 0.3), center_x=cx, center_y=cy))

    # Unlinked: each source has its own free center_x/center_y (4 centre params).
    unlinked = LensModel([
        _mass_plane := Plane(mass=[_mass()]),
        Plane(deflection_ratio=1.0, light=[
            _src(SRC_A, tfd.Normal(0., .5), tfd.Normal(0., .5)),
            _src(SRC_B, tfd.Normal(0., .5), tfd.Normal(0., .5))]),
    ])
    # Linked: both sources share ONE centre (2 centre params).
    cx, cy = shared(tfd.Normal(0., .5)), shared(tfd.Normal(0., .5))
    linked = LensModel([
        Plane(mass=[_mass()]),
        Plane(deflection_ratio=1.0, light=[_src(SRC_A, cx, cy), _src(SRC_B, cx, cy)]),
    ])

    # Sharing two centre coords (each was 2 free params across the 2 sites) saves 2.
    assert linked.num_free_params == unlinked.num_free_params - 2, \
        f"linked={linked.num_free_params}, unlinked={unlinked.num_free_params}"

    # And the prior carries exactly one term per unique param: z length == num_free_params.
    z = linked.bijector.inverse(linked.prior.sample(seed=linked._seed))
    assert z.shape[0] == linked.num_free_params, \
        f"z dim {z.shape[0]} != num_free_params {linked.num_free_params}"
    # The two shared centre coords appear once each in the prior's named params.
    shared_keys = [k for k in linked.prior.model.keys() if k.startswith("shared_")]
    assert len(shared_keys) == 2, f"expected 2 shared params, got {shared_keys}"


# ----------------------------------------------------------------------------------
# 2. Different source plane per band (different deflection_ratio per seen plane).
# ----------------------------------------------------------------------------------
def test_different_source_plane_per_band(plotter):
    """Two source planes at DIFFERENT deflection_ratio; band A sees only plane-1 light,
    band B only plane-2 light. Each band's image must use its plane's deflection_ratio.

    Metric/threshold: per-band image vs a lenstronomy-β render scaled by that plane's
    deflection_ratio, IMAGE_NOPSF_F64. Falsifier: a band rendered at the wrong plane's
    dr (or both planes summed) → O(1) structured residual.
    """
    pytest.importorskip("lenstronomy")
    from lenstronomy.LensModel.Profiles.epl import EPL as LEPL
    from gigalens.jax.profiles.mass.epl import EPL
    from gigalens.jax.profiles.light.sersic import SersicEllipse
    from gigalens.jax.scene import Component, Plane, LensModel
    from gigalens.jax.scene_prob_model import ImageData, ProbModel
    import jax.numpy as jnp

    DR1, DR2 = 0.8, 1.0
    src1 = Component(SersicEllipse(use_lstsq=False), dict(SRC_A, center_x=CX, center_y=CY, Ie=200.0))
    src2 = Component(SersicEllipse(use_lstsq=False), dict(SRC_B, center_x=-0.03, center_y=0.06, Ie=150.0))
    model = LensModel([
        Plane(mass=[Component(EPL(50), EPL0)]),
        Plane(deflection_ratio=DR1, light=[src1]),
        Plane(deflection_ratio=DR2, light=[src2]),
    ])
    cfg = _cfg()
    img0 = np.zeros((cfg.num_pix, cfg.num_pix)); err0 = np.ones_like(img0)
    da = ImageData(img0, cfg, error_map=err0, sees=[src1])
    db = ImageData(img0, cfg, error_map=err0, sees=[src2])
    prob = ProbModel(model, [da, db], mode="forward")
    params = model.to_params({})

    got_a = np.asarray(prob.simulators[0].simulate(params))
    got_b = np.asarray(prob.simulators[1].simulate(params))

    sim = prob.simulators[0]
    gx = np.asarray(sim.img_X)[..., 0]; gy = np.asarray(sim.img_Y)[..., 0]
    ax, ay = LEPL().derivatives(x=gx, y=gy, **EPL0)
    cf = float(np.linalg.det(np.asarray(sim.transform_pix2angle)))
    bx1, by1 = gx - DR1 * ax, gy - DR1 * ay
    bx2, by2 = gx - DR2 * ax, gy - DR2 * ay
    ref_a = np.asarray(src1.profile.light(bx1[..., None], by1[..., None],
                                          center_x=CX, center_y=CY, Ie=200.0, **SRC_A))[..., 0] * cf
    ref_b = np.asarray(src2.profile.light(bx2[..., None], by2[..., None],
                                          center_x=-0.03, center_y=0.06, Ie=150.0, **SRC_B))[..., 0] * cf

    plotter.image("plane1_dr0p8_band_A", ref_a, got_a,
                  rtol=tol.IMAGE_NOPSF_F64_RTOL, atol=tol.IMAGE_NOPSF_F64_ATOL)
    plotter.image("plane2_dr1p0_band_B", ref_b, got_b,
                  rtol=tol.IMAGE_NOPSF_F64_RTOL, atol=tol.IMAGE_NOPSF_F64_ATOL)
    assert np.allclose(got_a, ref_a, rtol=tol.IMAGE_NOPSF_F64_RTOL, atol=tol.IMAGE_NOPSF_F64_ATOL), \
        f"band A (dr={DR1}) max|Δ|={np.max(np.abs(got_a-ref_a)):.2e}"
    assert np.allclose(got_b, ref_b, rtol=tol.IMAGE_NOPSF_F64_RTOL, atol=tol.IMAGE_NOPSF_F64_ATOL), \
        f"band B (dr={DR2}) max|Δ|={np.max(np.abs(got_b-ref_b)):.2e}"


# ----------------------------------------------------------------------------------
# 3. Shared lens-light (lstsq) across bands: ONE shape, FREE per-band flux.
# ----------------------------------------------------------------------------------
def test_shared_lstsq_lens_light_independent_amplitude(plotter):
    """A single lstsq lens-light Component seen by BOTH bands: its rendered basis (shape)
    is identical across bands, but the amplitude is solved independently per band.

    We plant two DIFFERENT per-band amplitudes by forward-rendering each band's data, then
    let lstsq recover. Claims: (i) the stacked basis is bit-identical across the two
    per-dataset simulators (same grid+cfg) → REDUCTION floor; (ii) recovered coeffs equal
    the planted Ie×pixel_area per band, i.e. the bands are NOT forced to share flux →
    LINEARITY floor. Falsifier for (ii): coeffs equal each other (flux wrongly linked) or
    miss the planted value.
    """
    from gigalens.jax.profiles.mass.epl import EPL
    from gigalens.jax.profiles.light.sersic import SersicEllipse
    from gigalens.jax.scene import Component, Plane, LensModel
    from gigalens.jax.scene_simulator import SceneSimulator
    from gigalens.jax.scene_prob_model import ImageData, ProbModel
    import jax.numpy as jnp

    cfg = _cfg()
    IE_A, IE_B = 500.0, 300.0  # different lens-light flux per band
    # Forward truth scenes to plant data (same lens-light SHAPE, different Ie per band).
    fwdA = LensModel([Plane(mass=[Component(EPL(50), EPL0)],
                            light=[Component(SersicEllipse(use_lstsq=False), dict(LL, Ie=IE_A))])])
    fwdB = LensModel([Plane(mass=[Component(EPL(50), EPL0)],
                            light=[Component(SersicEllipse(use_lstsq=False), dict(LL, Ie=IE_B))])])
    dataA = np.asarray(SceneSimulator(fwdA, cfg).simulate(fwdA.to_params({})))
    dataB = np.asarray(SceneSimulator(fwdB, cfg).simulate(fwdB.to_params({})))

    ll = Component(SersicEllipse(use_lstsq=True), LL)  # ONE shared lstsq component
    model = LensModel([Plane(mass=[Component(EPL(50), EPL0)], light=[ll])])
    err = np.ones_like(dataA)
    da = ImageData(dataA, cfg, error_map=err, sees=[ll])
    db = ImageData(dataB, cfg, error_map=err, sees=[ll])
    prob = ProbModel(model, [da, db], mode="lstsq")
    params = model.to_params({})

    # (i) identical basis across the two per-dataset simulators (same cfg, same sees).
    stackA = np.asarray(prob.simulators[0].lstsq_simulate(
        params, jnp.asarray(dataA), jnp.asarray(err), return_stacked=True))
    stackB = np.asarray(prob.simulators[1].lstsq_simulate(
        params, jnp.asarray(dataB), jnp.asarray(err), return_stacked=True))
    assert np.allclose(stackA, stackB, rtol=tol.REDUCTION_RTOL, atol=tol.REDUCTION_ATOL), \
        f"shared lstsq basis differs across bands max|Δ|={np.max(np.abs(stackA-stackB)):.2e}"

    # (ii) per-band amplitudes solved independently == planted Ie × pixel_area.
    cf = float(np.linalg.det(np.asarray(prob.simulators[0].transform_pix2angle)))
    cA = np.squeeze(np.asarray(prob.simulators[0].lstsq_simulate(
        params, jnp.asarray(dataA), jnp.asarray(err), return_coeffs=True)))
    cB = np.squeeze(np.asarray(prob.simulators[1].lstsq_simulate(
        params, jnp.asarray(dataB), jnp.asarray(err), return_coeffs=True)))
    plotter.image("shared_lstsq_band_A_recon", dataA,
                  np.asarray(prob.simulators[0].lstsq_simulate(params, jnp.asarray(dataA), jnp.asarray(err))),
                  rtol=1e-6, atol=1e-9)
    plotter.image("shared_lstsq_band_B_recon", dataB,
                  np.asarray(prob.simulators[1].lstsq_simulate(params, jnp.asarray(dataB), jnp.asarray(err))),
                  rtol=1e-6, atol=1e-9)
    assert np.allclose(float(cA), IE_A * cf, rtol=tol.LSTSQ_AMPLITUDE_RTOL), \
        f"band A coeff {float(cA)} != planted {IE_A*cf}"
    assert np.allclose(float(cB), IE_B * cf, rtol=tol.LSTSQ_AMPLITUDE_RTOL), \
        f"band B coeff {float(cB)} != planted {IE_B*cf}"
    # The two recovered amplitudes genuinely differ (flux is free per band, not linked).
    assert abs(float(cA) - float(cB)) > 0.1 * IE_B * cf, \
        f"per-band amplitudes nearly equal ({cA} vs {cB}) — flux appears wrongly linked"


# ----------------------------------------------------------------------------------
# 4. Reduction guard: 1-dataset new-way == Phase-4 single-sim path.
# ----------------------------------------------------------------------------------
def test_one_dataset_reduces_to_phase4(plotter):
    """A 1-dataset ProbModel built the Phase-5 way (internal per-dataset simulator)
    yields the SAME log-likelihood / reduced χ² as the Phase-4 path that passes an
    explicit simulator. Exact float64 identity (same code, same sim) → REDUCTION floor.
    """
    from gigalens.jax.profiles.mass.epl import EPL
    from gigalens.jax.profiles.light.sersic import SersicEllipse
    from gigalens.jax.scene import Component, Plane, LensModel
    from gigalens.jax.scene_simulator import SceneSimulator
    from gigalens.jax.scene_prob_model import ImageData, ProbModel
    import tensorflow_probability.substrates.jax as tfp
    import jax.numpy as jnp
    tfd = tfp.distributions

    SRC = dict(R_sersic=0.3, n_sersic=2.0, e1=0.05, e2=-0.02, center_x=0.05, center_y=-0.04)
    cfg = _cfg(num_pix=60, supersample=2, kernel=_gauss_psf())
    fwd = LensModel([
        Plane(mass=[Component(EPL(50), EPL0)],
              light=[Component(SersicEllipse(use_lstsq=False), dict(LL, Ie=500.0))]),
        Plane(deflection_ratio=1.0,
              light=[Component(SersicEllipse(use_lstsq=False), dict(SRC, Ie=200.0))]),
    ])
    clean = np.asarray(SceneSimulator(fwd, cfg).simulate(fwd.to_params({})))
    err = np.sqrt(0.01**2 + np.clip(clean, 0, np.inf) / 1000.0)

    def _free(d):
        return {k: tfd.Normal(float(v), 0.3) for k, v in d.items()}
    model = LensModel([
        Plane(mass=[Component(EPL(50), _free(EPL0))],
              light=[ll := Component(SersicEllipse(use_lstsq=True), _free(LL))]),
        Plane(deflection_ratio=1.0,
              light=[sc := Component(SersicEllipse(use_lstsq=True), _free(SRC))]),
    ])
    _TRUTH = {(0, "mass", 0): EPL0, (0, "light", 0): LL, (1, "light", 0): SRC}

    def truth_val(key):
        p = key.split("/")
        return float(_TRUTH[(int(p[1]), p[2], int(p[3]))][p[4]])
    true_unique = {k: jnp.asarray(truth_val(k)) for k in model.prior.model.keys()}
    z_true = model.bijector.inverse(true_unique)

    ds = ImageData(clean, cfg, error_map=err, sees="all")
    prob = ProbModel(model, ds, mode="lstsq")
    # Phase-5 path: no explicit simulator (uses prob.simulators[0]).
    ll_new, rc_new = prob.log_like(z_true)
    # Phase-4 path: explicit simulator built the old way.
    ext = SceneSimulator(model, cfg)
    ll_old, rc_old = prob.log_like(z_true, simulator=ext)

    assert np.allclose(float(ll_new), float(ll_old), rtol=tol.REDUCTION_RTOL, atol=tol.REDUCTION_ATOL), \
        f"log_like new {float(ll_new)} != old {float(ll_old)}"
    assert np.allclose(float(rc_new), float(rc_old), rtol=tol.REDUCTION_RTOL, atol=tol.REDUCTION_ATOL), \
        f"red_chi2 new {float(rc_new)} != old {float(rc_old)}"
    plotter.image("reduction_guard_recon", clean,
                  np.asarray(prob.simulators[0].lstsq_simulate(
                      model.to_params(model.bijector.forward(z_true)),
                      jnp.asarray(clean), jnp.asarray(err), ds.mask)),
                  rtol=1e-6, atol=1e-9, err=err)


# ----------------------------------------------------------------------------------
# 5. Validators §3.5 / §3.6 / §3.7 raise on the offending constructions.
# ----------------------------------------------------------------------------------
def _two_src_model():
    from gigalens.jax.profiles.mass.epl import EPL
    from gigalens.jax.profiles.light.sersic import SersicEllipse
    from gigalens.jax.scene import Component, Plane, LensModel
    src_a = Component(SersicEllipse(use_lstsq=False), dict(SRC_A, center_x=CX, center_y=CY, Ie=200.0))
    src_b = Component(SersicEllipse(use_lstsq=False), dict(SRC_B, center_x=CX, center_y=CY, Ie=120.0))
    model = LensModel([
        Plane(mass=[Component(EPL(50), EPL0)]),
        Plane(deflection_ratio=1.0, light=[src_a, src_b]),
    ])
    return model, src_a, src_b


def test_validator_sees_required_and_nonempty():
    """§3.5: missing/None or empty sees raises (no silent 'sees everything')."""
    from gigalens.jax.scene_prob_model import ImageData, ProbModel
    model, src_a, src_b = _two_src_model()
    cfg = _cfg(); img0 = np.zeros((cfg.num_pix, cfg.num_pix)); err0 = np.ones_like(img0)
    with pytest.raises(ValueError, match=r"sees is required"):
        ProbModel(model, ImageData(img0, cfg, error_map=err0, sees=None), mode="forward")
    with pytest.raises(ValueError, match=r"sees is empty"):
        ProbModel(model, ImageData(img0, cfg, error_map=err0, sees=[]), mode="forward")


def test_validator_dead_entity_raises():
    """§3.6: a light Component seen by NO dataset raises (wiring bug)."""
    from gigalens.jax.scene_prob_model import ImageData, ProbModel
    model, src_a, src_b = _two_src_model()
    cfg = _cfg(); img0 = np.zeros((cfg.num_pix, cfg.num_pix)); err0 = np.ones_like(img0)
    # Only src_a is ever seen → src_b is dead.
    with pytest.raises(ValueError, match=r"seen by no dataset"):
        ProbModel(model, ImageData(img0, cfg, error_map=err0, sees=[src_a]), mode="forward")


def test_validator_forward_flux_sharing_raises():
    """§3.7: a non-lstsq Component seen by >1 dataset raises (identical flux per band)."""
    from gigalens.jax.scene_prob_model import ImageData, ProbModel
    model, src_a, src_b = _two_src_model()
    cfg = _cfg(); img0 = np.zeros((cfg.num_pix, cfg.num_pix)); err0 = np.ones_like(img0)
    da = ImageData(img0, cfg, error_map=err0, sees=[src_a, src_b])  # src_a forward, both bands
    db = ImageData(img0, cfg, error_map=err0, sees=[src_a, src_b])
    with pytest.raises(ValueError, match=r"forward-mode .* seen by more than one dataset"):
        ProbModel(model, [da, db], mode="forward")


def test_validator_lstsq_shared_across_bands_ok():
    """§3.7 carve-out: an lstsq Component seen by multiple datasets is FINE (amplitude
    solved per dataset). A valid multi-band construction must NOT raise."""
    from gigalens.jax.profiles.mass.epl import EPL
    from gigalens.jax.profiles.light.sersic import SersicEllipse
    from gigalens.jax.scene import Component, Plane, LensModel
    from gigalens.jax.scene_prob_model import ImageData, ProbModel
    ll = Component(SersicEllipse(use_lstsq=True), LL)
    model = LensModel([Plane(mass=[Component(EPL(50), EPL0)], light=[ll])])
    cfg = _cfg(); img0 = np.zeros((cfg.num_pix, cfg.num_pix)); err0 = np.ones_like(img0)
    da = ImageData(img0, cfg, error_map=err0, sees=[ll])
    db = ImageData(img0, cfg, error_map=err0, sees=[ll])
    ProbModel(model, [da, db], mode="lstsq")  # must not raise


def test_validator_sees_foreign_component_raises():
    """§3.5: a sees entry that is not a light Component of this model raises."""
    from gigalens.jax.profiles.light.sersic import SersicEllipse
    from gigalens.jax.scene import Component
    from gigalens.jax.scene_prob_model import ImageData, ProbModel
    model, src_a, src_b = _two_src_model()
    foreign = Component(SersicEllipse(use_lstsq=False), dict(SRC_A, center_x=0.0, center_y=0.0, Ie=1.0))
    cfg = _cfg(); img0 = np.zeros((cfg.num_pix, cfg.num_pix)); err0 = np.ones_like(img0)
    with pytest.raises(ValueError, match=r"not a light Component of this model"):
        ProbModel(model, ImageData(img0, cfg, error_map=err0, sees=[src_a, src_b, foreign]),
                  mode="forward")
