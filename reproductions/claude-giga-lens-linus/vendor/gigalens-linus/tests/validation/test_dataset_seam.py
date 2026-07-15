"""Dataset/LikelihoodTerm seam (Phase A) + trace-at-positions (Phase B) validation.

Phase A claim: ``ProbModel`` is now a SUM of per-observation likelihood terms and no
longer hardwires "observation = image". The imaging path must be unchanged (that claim
is held by the frozen golden anchor + the Phase-4/5 suites, which run unmodified); this
module tests the NEW degrees of freedom only:

  1. **Term summation:** a ProbModel with [imaging, toy-Gaussian-constraint] equals
     imaging-only ``log_like`` plus the toy term (derived N·eps tolerance; see in-test note).
  2. **Diagnostics channel:** ``red_chi2`` is defined over chi²-reporting terms only —
     adding a non-Gaussian term must NOT change it (falsifier: any drift means the toy
     term leaked into the χ² channel).
  3. **No-Gaussian-term guard:** a ProbModel with no chi²-reporting term raises loudly
     (the (log_like, red_chi2) channel would be vacuous) rather than emitting 0/0.
  4. **Per-dataset mode:** ``ImageData(mode=...)`` overrides the ProbModel default and is
     exactly equivalent to setting the same mode globally.

Phase B claim: ``SceneSimulator.trace(params, positions=(x, y))`` runs the SAME ray
trace on caller-supplied positions. Prediction (pre-registered): tracing a SUBSET of
grid points returns bitwise-identical β to the corresponding entries of the full-grid
trace, because the trace is elementwise in the image-plane coordinates (no cross-pixel
reduction anywhere in ``_alpha`` / the recursion). Falsifier: ANY elementwise
difference ⇒ a shape-dependent code path in the trace — a FINDING to investigate
(not a tolerance to widen), since it would break the planned non-grid consumers
(lens-equation solving, catalog positions, adaptive supersampling).
"""
import numpy as np
import pytest
from jax import numpy as jnp

pytestmark = pytest.mark.physics

EPL0 = dict(theta_E=1.1, gamma=2.05, e1=0.04, e2=-0.03, center_x=0.0, center_y=0.0)
LL = dict(R_sersic=1.2, n_sersic=3.0, e1=0.0, e2=0.0, center_x=0.0, center_y=0.0)
SRC = dict(R_sersic=0.3, n_sersic=2.0, e1=0.05, e2=-0.02, center_x=0.05, center_y=-0.04)
_BG, _EXP = 0.01, 1000.0


def _gauss_psf(sigma=2.0, half=10):
    yy, xx = np.mgrid[-half:half + 1, -half:half + 1]
    k = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return k / k.sum()


def _cfg(num_pix=60, supersample=2, kernel="psf"):
    from gigalens.simulator import SimulatorConfig
    return SimulatorConfig(delta_pix=0.05, num_pix=num_pix, supersample=supersample,
                           kernel=_gauss_psf() if kernel == "psf" else None,
                           likelihood_precision="float64")


def _free(d):
    import tensorflow_probability.substrates.jax as tfp
    tfd = tfp.distributions
    return {k: tfd.Normal(float(v), 0.3) for k, v in d.items()}


def _imaging_scene(use_lstsq=True):
    """(observed, err, model, z_draw) — the Phase-4 single-band scene, free params."""
    from gigalens.jax.profiles.mass.epl import EPL
    from gigalens.jax.profiles.light.sersic import SersicEllipse
    from gigalens.jax.scene import Component, Plane, LensModel
    from gigalens.jax.scene_simulator import SceneSimulator

    fwd = LensModel([
        Plane(mass=[Component(EPL(50), EPL0)],
              light=[Component(SersicEllipse(use_lstsq=False), dict(LL, Ie=500.0))]),
        Plane(deflection_ratio=1.0,
              light=[Component(SersicEllipse(use_lstsq=False), dict(SRC, Ie=200.0))]),
    ])
    clean = np.asarray(SceneSimulator(fwd, _cfg()).simulate(fwd.to_params({})))
    err = np.sqrt(_BG**2 + np.clip(clean, 0, np.inf) / _EXP)

    light_priors = (_free(LL), _free(SRC)) if use_lstsq else (
        _free(dict(LL, Ie=500.0)), _free(dict(SRC, Ie=200.0)))
    model = LensModel([
        Plane(mass=[Component(EPL(50), _free(EPL0))],
              light=[Component(SersicEllipse(use_lstsq=use_lstsq), light_priors[0])]),
        Plane(deflection_ratio=1.0,
              light=[Component(SersicEllipse(use_lstsq=use_lstsq), light_priors[1])]),
    ])
    # A fixed, valid z: the bijector-inverse of a prior draw (seeded; value irrelevant —
    # every claim below is an identity at the SAME z, not a statement about the fit).
    import jax
    draw = model.prior.sample(seed=jax.random.PRNGKey(3))
    z = np.asarray(model.bijector.inverse(draw))
    return clean, err, model, jnp.asarray(z)


def _toy_theta_e_observation(mass_component, value, sigma):
    """A minimal non-image Dataset: Gaussian constraint on the EPL theta_E.

    Deliberately tiny — it exists to exercise the seam (sees resolution, term
    summation, chi²-channel exclusion), not to model anything physical."""
    from gigalens.jax.scene_prob_model import LikelihoodTerm, Dataset

    class _ThetaETerm(LikelihoodTerm):
        reports_chi2 = False
        event_size = 1

        def log_like(self, params):
            v = params["planes"][0]["mass"][0]["theta_E"]
            r = (v - value) / sigma
            return -0.5 * (r ** 2) - 0.5 * jnp.log(2 * jnp.pi * sigma ** 2), None

    class _ThetaEDataset(Dataset):
        event_size = 1

        def __init__(self):
            self.sees = [mass_component]

        def resolve_sees(self, model):
            return list(self.sees)

        def build_term(self, model, seen, *, default_mode="lstsq"):
            return _ThetaETerm()

    return _ThetaEDataset()


def toy_ll(z, model, value, sigma):
    params = model.to_params(model.bijector.forward(z))
    v = np.asarray(params["planes"][0]["mass"][0]["theta_E"])
    return -0.5 * ((v - value) / sigma) ** 2 - 0.5 * np.log(2 * np.pi * sigma ** 2)


# ---------------------------------------------------------------- Phase A: the seam
def test_probmodel_sums_terms_and_chi2_channel_excludes_non_gaussian():
    """Claims 1 + 2: mixed [imaging, toy] log_like == imaging log_like + toy term
    (exact: one extra float64 add), and red_chi2 is unchanged by the toy term."""
    from gigalens.jax.scene_prob_model import ImageData, ProbModel
    clean, err, model, z = _imaging_scene()
    ds = ImageData(clean, _cfg(), error_map=err, sees="all")
    toy = _toy_theta_e_observation(model.planes[0].mass[0], value=1.1, sigma=0.05)

    prob_img = ProbModel(model, ds, mode="lstsq")
    prob_mix = ProbModel(model, [ds, toy], mode="lstsq")

    # Tolerance (derived): the two ProbModels compile lstsq_simulate separately, and
    # the compiled Gram matmul + Σ over N≈60² pixels is reproducible across
    # compilations only to the reduction-order floor. For an all-positive float64 sum
    # the relative error bound is N·eps ≈ 3600·2.2e-16 ≈ 8e-13 → rtol=1e-12. A real
    # seam bug (missing/double-counted term, toy leak) is ≳1e-6 relative here —
    # 6 orders above the band. (Measured wobble: ~6e-14; bitwise equality across
    # separate jit compilations was falsified, see the worklog.)
    for zz in (z, jnp.stack([z, z + 0.1])):  # scalar AND batched draws
        ll_img, red_img = prob_img.log_like(zz)
        ll_mix, red_mix = prob_mix.log_like(zz)
        expected = np.asarray(ll_img) + toy_ll(zz, model, 1.1, 0.05)
        np.testing.assert_allclose(np.asarray(ll_mix), expected, rtol=1e-12,
                                   err_msg="term summation is not imaging + toy")
        np.testing.assert_allclose(
            np.asarray(red_mix), np.asarray(red_img), rtol=1e-12,
            err_msg="red_chi2 changed when a non-chi² term was added — the toy term "
                    "leaked into the χ² diagnostic channel")

    # event_size counts every observation; the χ² denominator only Gaussian ones.
    assert prob_mix.event_size == prob_img.event_size + 1
    assert prob_mix._chi2_event_size == prob_img.event_size
    # Back-compat: the imaging simulator list is unchanged by the extra term.
    assert len(prob_mix.simulators) == 1


def test_no_gaussian_term_raises():
    """Claim 3: a ProbModel whose terms report no chi² refuses construction loudly."""
    from gigalens.jax.profiles.mass.epl import EPL
    from gigalens.jax.scene import Component, Plane, LensModel
    from gigalens.jax.scene_prob_model import ProbModel

    mass = Component(EPL(50), _free(EPL0))
    model = LensModel([Plane(mass=[mass]), Plane(deflection_ratio=1.0, light=[])])
    toy = _toy_theta_e_observation(mass, value=1.1, sigma=0.05)
    with pytest.raises(NotImplementedError, match="chi²-reporting"):
        ProbModel(model, [toy])


def test_dataset_mode_overrides_probmodel_default():
    """Claim 4: ImageData(mode='forward') under ProbModel(mode='lstsq') ==
    ProbModel(mode='forward') with a plain dataset, exactly."""
    from gigalens.jax.scene_prob_model import ImageData, ProbModel
    clean, err, model, z = _imaging_scene(use_lstsq=False)

    ds_override = ImageData(clean, _cfg(), error_map=err, sees="all", mode="forward")
    ds_plain = ImageData(clean, _cfg(), error_map=err, sees="all")

    prob_override = ProbModel(model, ds_override, mode="lstsq")
    prob_global = ProbModel(model, ds_plain, mode="forward")
    assert prob_override.terms[0].mode == "forward"

    ll_o, red_o = prob_override.log_like(z)
    ll_g, red_g = prob_global.log_like(z)
    assert np.array_equal(np.asarray(ll_o), np.asarray(ll_g))
    assert np.array_equal(np.asarray(red_o), np.asarray(red_g))

    with pytest.raises(ValueError, match="mode"):
        ImageData(clean, _cfg(), error_map=err, sees="all", mode="bogus")


def test_with_map_remat_ports_to_terms():
    """The MAP-scoped remat twin (PR #24) must survive the seam port: it swaps the
    imaging terms' simulators (remat_basis forced ON at ss < fuse threshold), leaves
    ``self`` and the shared datasets untouched, is idempotent, and is
    likelihood-exact (remat re-runs identical ops; same derived N·eps band as above
    since the twin's simulator is a separate compilation)."""
    from gigalens.jax.scene_prob_model import ImageData, ProbModel
    clean, err, model, z = _imaging_scene()
    ds = ImageData(clean, _cfg(), error_map=err, sees="all")  # ss=2 < fuse threshold
    prob = ProbModel(model, ds, mode="lstsq")

    twin = prob.with_map_remat()
    assert twin is not prob
    assert prob.simulators[0].remat_basis is False, "self was mutated"
    assert twin.simulators[0].remat_basis is True
    assert twin.terms[0] is not prob.terms[0]
    assert twin.datasets is prob.datasets  # shared, not copied
    assert twin.with_map_remat() is twin   # idempotent: nothing left to change

    ll0, red0 = prob.log_like(z)
    ll1, red1 = twin.log_like(z)
    np.testing.assert_allclose(np.asarray(ll1), np.asarray(ll0), rtol=1e-12,
                               err_msg="remat twin is not likelihood-exact")
    np.testing.assert_allclose(np.asarray(red1), np.asarray(red0), rtol=1e-12)


# ------------------------------------------------------- Phase B: trace at positions
def _subset_positions(sim, idx):
    """K grid points as (K, 1) position arrays (trailing broadcast axis, as the grid)."""
    X = np.asarray(sim.img_X).reshape(-1, 1)
    Y = np.asarray(sim.img_Y).reshape(-1, 1)
    return jnp.asarray(X[idx]), jnp.asarray(Y[idx])


def _assert_subset_matches_grid(sim, params, idx, planes):
    grid = sim.trace(params)
    sub = sim.trace(params, positions=_subset_positions(sim, idx))
    for i in planes:
        for a in (0, 1):
            full = np.asarray(grid[i][a]).reshape(-1, 1)[idx]
            got = np.asarray(sub[i][a])
            assert np.array_equal(got, full), (
                f"plane {i} axis {a}: traced subset differs from grid trace "
                f"(max |Δ| = {np.max(np.abs(got - full)):.3e}) — shape-dependent "
                "trace code path (FINDING; do not widen to a tolerance)")


def test_trace_positions_single_deflector_matches_grid():
    """Single-deflector trace: default positions == stored grid (wiring), and a
    64-point subset is bitwise-identical to the corresponding grid entries."""
    from gigalens.jax.profiles.mass.epl import EPL
    from gigalens.jax.profiles.light.sersic import SersicEllipse
    from gigalens.jax.scene import Component, Plane, LensModel
    from gigalens.jax.scene_simulator import SceneSimulator

    model = LensModel([
        Plane(mass=[Component(EPL(50), EPL0)],
              light=[Component(SersicEllipse(use_lstsq=False), dict(LL, Ie=500.0))]),
        Plane(deflection_ratio=1.0,
              light=[Component(SersicEllipse(use_lstsq=False), dict(SRC, Ie=200.0))]),
    ])
    sim = SceneSimulator(model, _cfg(num_pix=40, kernel=None))
    params = model.to_params({})
    assert sim.trace_mode == "deflection_ratio"

    grid = sim.trace(params)
    explicit = sim.trace(params, positions=(sim.img_X, sim.img_Y))
    for i in grid:
        assert np.array_equal(np.asarray(grid[i][0]), np.asarray(explicit[i][0]))
        assert np.array_equal(np.asarray(grid[i][1]), np.asarray(explicit[i][1]))

    idx = np.linspace(0, np.asarray(sim.img_X).size - 1, 64).astype(int)
    _assert_subset_matches_grid(sim, params, idx, planes=list(grid.keys()))

    # no_deflection returns the supplied positions unchanged.
    px, py = _subset_positions(sim, idx)
    nd = sim.trace(params, no_deflection=True, positions=(px, py))
    for i in nd:
        assert np.array_equal(np.asarray(nd[i][0]), np.asarray(px))
        assert np.array_equal(np.asarray(nd[i][1]), np.asarray(py))


def test_trace_positions_multiplane_matches_grid():
    """Multiplane recursion at caller positions: 64-point subset bitwise-identical to
    the grid trace on a 3-plane (2-deflector) model."""
    from gigalens.jax.cosmo import wCDM_Cosmo
    from gigalens.jax.profiles.mass.epl import EPL
    from gigalens.jax.scene import Component, Plane, LensModel
    from gigalens.jax.scene_simulator import SceneSimulator

    z1, z2, z_src = 0.45, 0.8, 3.0
    epl1 = dict(theta_E=0.6, gamma=2.0, e1=-0.04, e2=0.02, center_x=0.3, center_y=-0.2)
    cosmo = Component(wCDM_Cosmo(z_lens=z1, z_source_ref=10.0),
                      dict(H0=70.0, Om0=0.3, k=0.0, w0=-1.0))
    model = LensModel([
        Plane(redshift=z1, mass=[Component(EPL(50), EPL0)]),
        Plane(redshift=z2, mass=[Component(EPL(50), epl1)]),
        Plane(redshift=z_src, light=[]),
    ], cosmo=cosmo)
    sim = SceneSimulator(model, _cfg(num_pix=40, supersample=1, kernel=None))
    params = model.to_params({})
    assert sim.trace_mode == "multiplane"

    idx = np.linspace(0, np.asarray(sim.img_X).size - 1, 64).astype(int)
    _assert_subset_matches_grid(sim, params, idx, planes=[0, 1, 2])
