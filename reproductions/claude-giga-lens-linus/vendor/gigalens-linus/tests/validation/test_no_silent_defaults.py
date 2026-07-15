"""The no-silent-defaults guards must RAISE (R10).

project-standards §7 / spec §3: a missing scientific input must raise at construction,
never default to a degraded-but-plausible model. These tests assert the *absence* of a
silent fallback — they pass only when the constructor refuses bad input. Written
against the new API (LensModel/Plane/ImageData/shared) and skipped until Phase 2.

This is a behavioral spec, not a parity check: there is no oracle, the assertion is
``pytest.raises``. Each test names the exact failure mode it forbids.
"""
import numpy as np
import pytest

pytestmark = pytest.mark.redteam


# -- construction helpers (shared by the §3.1–§3.4 guards) ------------------------
def _epl():
    from gigalens.jax.scene import Component
    from gigalens.jax.profiles.mass.epl import EPL
    return Component(EPL(50), dict(theta_E=1.1, gamma=2.05, e1=0.04, e2=-0.03,
                                   center_x=0.0, center_y=0.0))


def _sersic(**over):
    from gigalens.jax.scene import Component
    from gigalens.jax.profiles.light.sersic import SersicEllipse
    p = dict(R_sersic=0.3, n_sersic=2.0, e1=0.0, e2=0.0, center_x=0.0, center_y=0.0,
             Ie=1.0)
    p.update(over)
    return Component(SersicEllipse(use_lstsq=False), p)


def _cosmo_component():
    """A valid (all-constant) wCDM cosmology Component for the §3.1 geometry guards."""
    from gigalens.jax.scene import Component
    from gigalens.jax.cosmo import wCDM_Cosmo
    return Component(wCDM_Cosmo(z_lens=0.5, z_source_ref=10.0),
                     dict(H0=70.0, Om0=0.3, k=0.0, w0=-1.0))


def test_missing_deflection_ratio_with_multiple_planes_raises():
    """§3.1: ≥2 lensed-light planes, no cosmo, one omits deflection_ratio → raise.

    The single-plane 1.0 default applies only to a *lone* source plane; with two source
    planes a missing ratio is ambiguous and must not silently default. Falsifier: a model
    with two lensed-light planes (one without deflection_ratio) constructs without error.
    """
    from gigalens.jax.scene import Plane, LensModel
    with pytest.raises(ValueError, match=r"deflection_ratio is required"):
        LensModel([
            Plane(mass=[_epl()]),
            Plane(deflection_ratio=1.0, light=[_sersic()]),
            Plane(light=[_sersic()]),               # second source, no ratio → raise
        ])


def test_redshift_without_cosmo_raises_and_deflection_ratio_with_cosmo_raises():
    """§3.1: geometry kind must match cosmo presence — both mismatches raise.

    Falsifier: a redshift accepted with no cosmology (distances undefined), or a
    deflection_ratio accepted alongside a cosmology (two competing geometry sources).
    """
    from gigalens.jax.scene import Plane, LensModel
    # redshift but NO cosmology
    with pytest.raises(ValueError, match=r"redshift given but no cosmology"):
        LensModel([Plane(mass=[_epl()]),
                   Plane(redshift=2.0, light=[_sersic()])])
    # deflection_ratio WITH a cosmology
    with pytest.raises(ValueError, match=r"deflection_ratio given but a cosmology"):
        LensModel([Plane(redshift=0.5, mass=[_epl()]),
                   Plane(deflection_ratio=1.0, light=[_sersic()])],
                  cosmo=_cosmo_component())
    # cosmology present but a plane omits its redshift
    with pytest.raises(ValueError, match=r"redshift is required when a cosmology"):
        LensModel([Plane(redshift=0.5, mass=[_epl()]),
                   Plane(light=[_sersic()])],          # no redshift → raise
                  cosmo=_cosmo_component())


def test_missing_or_unknown_param_raises():
    """§3.2: a Component.priors missing a profile param, or naming an unknown one, raises.

    Falsifier: either constructs without error, letting an unspecified physical parameter
    take a silent default (the no-silent-defaults failure mode).
    """
    from gigalens.jax.scene import Component, Plane, LensModel
    from gigalens.jax.profiles.mass.epl import EPL
    # unknown parameter
    with pytest.raises(ValueError, match=r"unknown parameter"):
        LensModel([Plane(mass=[Component(EPL(50), dict(
            theta_E=1.1, gamma=2.05, e1=0.04, e2=-0.03, center_x=0.0, center_y=0.0,
            bogus_param=1.0))])])
    # missing parameter (drop center_y)
    with pytest.raises(ValueError, match=r"missing parameter"):
        LensModel([Plane(mass=[Component(EPL(50), dict(
            theta_E=1.1, gamma=2.05, e1=0.04, e2=-0.03, center_x=0.0))])])


def test_reused_bare_distribution_raises():
    """§3.3: the same tfd.Distribution object at >1 site raises (must wrap in shared()).

    Reusing one distribution object silently *links* two parameters (identity = sharing);
    if that was not intended the model is mis-specified. Falsifier: the same object at two
    sites constructs without error. Wrapping in shared() is the explicit way to link.
    """
    import tensorflow_probability.substrates.jax as tfp
    from gigalens.jax.scene import Plane, LensModel
    tfd = tfp.distributions
    d = tfd.Normal(0.0, 0.1)                 # one object...
    with pytest.raises(ValueError, match=r"same tfd.Distribution object is reused"):
        LensModel([Plane(mass=[_epl()]),
                   Plane(deflection_ratio=1.0,
                         light=[_sersic(center_x=d), _sersic(center_y=d)])])  # ...reused


def test_dataset_without_noise_raises():
    """§3.4: a ImageData with neither error_map nor (background_rms + exp_time) raises.

    Falsifier: a ImageData constructs with no noise model, leaving the Gaussian likelihood
    to invent a silent default sigma.
    """
    from gigalens.simulator import SimulatorConfig
    from gigalens.jax.scene_prob_model import ImageData
    cfg = SimulatorConfig(delta_pix=0.05, num_pix=8, supersample=1, kernel=None,
                          likelihood_precision="float64")
    with pytest.raises(ValueError, match=r"either error_map or"):
        ImageData(np.zeros((8, 8)), cfg, sees=[_sersic()])


def test_empty_sees_and_dead_entity_raise():
    """Empty/absent `sees` raises (no silent sees-all); a Component/plane no dataset
    sees raises (dead-entity wiring guard). Implemented Phase 5; full behavioral
    coverage (including the foreign-component and string-'all' cases) lives in
    test_multiband.py — these are the risk-register R-slots for §3.5/§3.6."""
    import numpy as np
    from gigalens.simulator import SimulatorConfig
    from gigalens.jax.profiles.mass.epl import EPL
    from gigalens.jax.profiles.light.sersic import SersicEllipse
    from gigalens.jax.scene import Component, Plane, LensModel
    from gigalens.jax.scene_prob_model import ImageData, ProbModel

    epl = dict(theta_E=1.1, gamma=2.05, e1=0.04, e2=-0.03, center_x=0.0, center_y=0.0)
    s = dict(R_sersic=0.3, n_sersic=2.0, e1=0.0, e2=0.0, center_x=0.0, center_y=0.0)
    a = Component(SersicEllipse(use_lstsq=False), dict(s, Ie=1.0))
    b = Component(SersicEllipse(use_lstsq=False), dict(s, Ie=1.0))
    model = LensModel([Plane(mass=[Component(EPL(50), epl)]),
                       Plane(deflection_ratio=1.0, light=[a, b])])
    cfg = SimulatorConfig(delta_pix=0.05, num_pix=40, supersample=1, kernel=None,
                          likelihood_precision="float64")
    img = np.zeros((40, 40)); err = np.ones_like(img)
    with pytest.raises(ValueError, match=r"sees is required"):
        ProbModel(model, ImageData(img, cfg, error_map=err, sees=None), mode="forward")
    # b is seen by no dataset → dead entity.
    with pytest.raises(ValueError, match=r"seen by no dataset"):
        ProbModel(model, ImageData(img, cfg, error_map=err, sees=[a]), mode="forward")


def test_forward_mode_flux_sharing_raises():
    """A non-lstsq light Component seen by >1 dataset raises (would equate flux across
    bands); shared geometry must go through shared() with per-band profiles.
    Implemented Phase 5 (the §3.7 risk-register slot)."""
    import numpy as np
    from gigalens.simulator import SimulatorConfig
    from gigalens.jax.profiles.mass.epl import EPL
    from gigalens.jax.profiles.light.sersic import SersicEllipse
    from gigalens.jax.scene import Component, Plane, LensModel
    from gigalens.jax.scene_prob_model import ImageData, ProbModel

    epl = dict(theta_E=1.1, gamma=2.05, e1=0.04, e2=-0.03, center_x=0.0, center_y=0.0)
    s = dict(R_sersic=0.3, n_sersic=2.0, e1=0.0, e2=0.0, center_x=0.0, center_y=0.0)
    fwd = Component(SersicEllipse(use_lstsq=False), dict(s, Ie=1.0))
    model = LensModel([Plane(mass=[Component(EPL(50), epl)]),
                       Plane(deflection_ratio=1.0, light=[fwd])])
    cfg = SimulatorConfig(delta_pix=0.05, num_pix=40, supersample=1, kernel=None,
                          likelihood_precision="float64")
    img = np.zeros((40, 40)); err = np.ones_like(img)
    da = ImageData(img, cfg, error_map=err, sees=[fwd])
    db = ImageData(img, cfg, error_map=err, sees=[fwd])
    with pytest.raises(ValueError, match=r"forward-mode .* seen by more than one dataset"):
        ProbModel(model, [da, db], mode="forward")


@pytest.mark.skip(reason="new API multiplane non-flat render — Phase 3")
def test_multiplane_nonflat_cosmo_supported():
    """Non-flat multi-plane is SUPPORTED (curvature implemented in Phase 1, validated
    in test_cosmology). This Phase-3 test renders a non-flat (k != 0) multi-plane image
    and checks it against lenstronomy — i.e. confirms there is no longer a silent flat
    approximation *and* no spurious raise. (Superseded the old 'must raise until
    curvature implemented' guard.)"""
    raise NotImplementedError
