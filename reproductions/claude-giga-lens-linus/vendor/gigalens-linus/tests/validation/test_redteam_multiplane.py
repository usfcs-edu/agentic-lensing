"""Red-team M2 — redshift-specified planes must actually be deflected.

The encountered pitfall (misuse_register M2 / encountered_pitfalls P2): a lensed-light
plane specified by ``redshift`` (+ cosmology) was returned on the UNDEFLECTED grid
unless its geometry dict literally contained ``"deflection_ratio"`` — especially toxic
under ``lstsq_simulate``, whose auto-solved amplitudes refit the wrong-geometry basis
into a plausible-looking image. The library fix computes ratios from redshifts inside
the trace; these tests pin that behavior against regression from the OUTSIDE (rendered
images), not by inspecting internals.

Contract (red-team, not parity): the oracle is gigalens' own explicit-``deflection_ratio``
path plus the cosmology's ratio — no lenstronomy needed. Assertions are identities and
loud-difference checks, never a bare "looks plausible".
"""
import numpy as np
import pytest

from . import tolerances as tol

pytestmark = pytest.mark.redteam

# Geometry: reference source plane AT z=1.0 (ratio exactly 1 there), second source
# behind it at z=3.0 (ratio > 1). Lens mass at the cosmology's z_lens.
Z_LENS, Z_A, Z_B = 0.5, 1.0, 3.0
GL_KW = dict(H0=70.0, Om0=0.3, k=0.0, w0=-1.0)

EPL0 = dict(theta_E=1.1, gamma=2.05, e1=0.04, e2=-0.03, center_x=0.0, center_y=0.0)
SRC_A = dict(R_sersic=0.30, n_sersic=2.0, e1=0.05, e2=-0.02, center_x=0.05, center_y=-0.04)
SRC_B = dict(R_sersic=0.25, n_sersic=1.5, e1=-0.03, e2=0.04, center_x=-0.30, center_y=0.20)

# Derived loud-difference floor: the naive (ratio=1) render differs from the correct
# one by source B moving through (r_B - 1)·α ≈ 0.5 · 1.1" ≈ 0.55" ≈ 11 pixels at
# delta_pix=0.05 — an O(1) relative change in pixels near source B. 1e-3 of the image
# peak is ~3 orders below that signal and ~6 orders above the float64 render floor
# (IMAGE_NOPSF_F64), so it separates the two regimes with wide margin either way.
LOUD_FRAC = 1e-3


def _cfg():
    from gigalens.simulator import SimulatorConfig
    return SimulatorConfig(delta_pix=0.05, num_pix=50, supersample=1, kernel=None,
                           likelihood_precision="float64")


def _cosmo_component():
    from gigalens.jax.scene import Component
    from gigalens.jax.cosmo import wCDM_Cosmo
    return Component(wCDM_Cosmo(z_lens=Z_LENS, z_source_ref=Z_A), dict(GL_KW))


def _ratio_b():
    """The cosmology's own deflection ratio for plane B (shared by both variants)."""
    from gigalens.jax.cosmo import wCDM_Cosmo
    cosmo = wCDM_Cosmo(z_lens=Z_LENS, z_source_ref=Z_A)
    return float(np.asarray(cosmo.deflection_ratio(Z_B, **GL_KW)).reshape(-1)[0])


def _planes(lstsq, geometry):
    """The three planes with geometry either by redshift or explicit ratio.

    ``geometry``: "redshift" | ("ratios", r_A, r_B).
    """
    from gigalens.jax.scene import Component, Plane
    from gigalens.jax.profiles.mass.epl import EPL
    from gigalens.jax.profiles.light.sersic import SersicEllipse

    a = dict(SRC_A) if lstsq else dict(SRC_A, Ie=200.0)
    b = dict(SRC_B) if lstsq else dict(SRC_B, Ie=300.0)
    light = lambda p: Component(SersicEllipse(use_lstsq=lstsq), p)
    if geometry == "redshift":
        geoms = [dict(redshift=Z_LENS), dict(redshift=Z_A), dict(redshift=Z_B)]
    else:
        _, r_a, r_b = geometry
        geoms = [{}, dict(deflection_ratio=r_a), dict(deflection_ratio=r_b)]
    return [Plane(mass=[Component(EPL(50), EPL0)], **geoms[0]),
            Plane(light=[light(a)], **geoms[1]),
            Plane(light=[light(b)], **geoms[2])]


def _render(lstsq, geometry, observed=None):
    from gigalens.jax.scene import LensModel
    from gigalens.jax.scene_simulator import SceneSimulator
    import jax.numpy as jnp

    cosmo = _cosmo_component() if geometry == "redshift" else None
    model = LensModel(_planes(lstsq, geometry), cosmo=cosmo)
    sim = SceneSimulator(model, _cfg())
    params = model.to_params({})
    if lstsq:
        obs = jnp.asarray(observed)
        return np.asarray(sim.lstsq_simulate(params, obs, jnp.ones_like(obs)))
    return np.asarray(sim.simulate(params))


def test_redshift_planes_are_actually_deflected_simulate():
    """Claim: identity + loud difference, under ``simulate``.

    (1) The redshift-specified render equals the render with the SAME ratios given
    explicitly (ratio r_A = 1 at the reference plane, r_B = cosmo.deflection_ratio(z_B)),
    to the float64 render floor — pins that the redshift path derives and APPLIES the
    cosmology's ratio. (2) It differs loudly from the naive both-ratios-1 render — the
    P2 regression (plane B on an effectively unscaled deflection) cannot pass silently.
    Falsifiers: (1) any >IMAGE_NOPSF_F64 disagreement ⇒ redshift→ratio derivation or
    threading is wrong; (2) max|Δ| below LOUD_FRAC·peak ⇒ ratio collapsed to 1.
    """
    r_b = _ratio_b()
    assert r_b > 1.05, f"test geometry degenerate: ratio(z_B)={r_b} too close to 1"

    img_z = _render(False, "redshift")
    img_explicit = _render(False, ("ratios", 1.0, r_b))
    img_naive = _render(False, ("ratios", 1.0, 1.0))

    assert np.allclose(img_z, img_explicit,
                       rtol=tol.IMAGE_NOPSF_F64_RTOL, atol=tol.IMAGE_NOPSF_F64_ATOL), (
        f"redshift render != explicit-ratio render: "
        f"max|Δ|={np.max(np.abs(img_z - img_explicit)):.2e}")
    loud = np.max(np.abs(img_z - img_naive))
    assert loud > LOUD_FRAC * np.max(img_z), (
        f"redshift render is indistinguishable from the UNDEFLECTED-scaling render "
        f"(max|Δ|={loud:.2e} <= {LOUD_FRAC}·peak): plane B's ratio collapsed to 1 (P2)")


def test_redshift_planes_are_actually_deflected_lstsq():
    """Claim: identity + loud difference, under ``lstsq_simulate`` — the toxic case.

    The observed image is a forward render of the redshift model with planted
    amplitudes. (1) Identity: the redshift lstsq reconstruction equals the
    explicit-ratio lstsq reconstruction (same bases ⇒ same solve), to the float64
    render floor — pins the redshift→ratio path inside the lstsq branch. (2) The
    redshift model's round-trip residual is below LOUD_FRAC·peak. It is NOT at the
    float floor: ``_solve_normal_eq_with_fallback`` deliberately jitters the Gram
    diagonal by ε=1e-6·mean(diag) (simulator.py), which biases the smaller-norm
    basis's coefficient by ε·mean(diag)/G_ii ≈ 2e-5 here → measured image error
    1.0e-5·peak; LOUD_FRAC=1e-3 gives ~100× headroom over that documented solver
    bias. (3) The naive both-ratios-1 lstsq model must NOT reproduce the image: its
    plane-B basis is at the wrong position and the amplitude refit cannot absorb the
    geometry error (measured residual 0.58·peak, ~580× above the floor). Falsifier
    for (3): residual <= LOUD_FRAC·peak ⇒ lstsq is hiding an undeflected plane
    (exactly how P2 stayed invisible).
    """
    r_b = _ratio_b()
    observed = _render(False, "redshift")
    peak = np.max(observed)

    recon_z = _render(True, "redshift", observed=observed)
    recon_explicit = _render(True, ("ratios", 1.0, r_b), observed=observed)
    assert np.allclose(recon_z, recon_explicit,
                       rtol=tol.IMAGE_NOPSF_F64_RTOL, atol=tol.IMAGE_NOPSF_F64_ATOL), (
        f"redshift lstsq != explicit-ratio lstsq: "
        f"max|Δ|={np.max(np.abs(recon_z - recon_explicit)):.2e}")

    resid_z = np.max(np.abs(recon_z - observed))
    assert resid_z < LOUD_FRAC * peak, (
        f"lstsq round-trip on the redshift model exceeds the solver-bias band: "
        f"max|Δ|={resid_z:.2e} vs {LOUD_FRAC}·peak={LOUD_FRAC * peak:.2e}")

    recon_naive = _render(True, ("ratios", 1.0, 1.0), observed=observed)
    resid_naive = np.max(np.abs(recon_naive - observed))
    assert resid_naive > LOUD_FRAC * peak, (
        f"lstsq with an undeflected plane-B basis reconstructed the observed image to "
        f"max|Δ|={resid_naive:.2e} <= {LOUD_FRAC}·peak — amplitude refit is absorbing "
        f"the geometry error (P2's failure mode)")


# --- C1 scene path: concrete redshifts guarded at CONSTRUCTION (jit skips first-use) ---
def test_front_of_lens_plane_raises_at_construction():
    """Claim: contract. A CONCRETE front-of-lens redshift on a plane listed after the
    mass plane must raise at ``LensModel`` construction — grader probe P2 showed the
    cosmology layer's first-use guard is structurally skipped on this path (``simulate``
    is jitted, so even a fixed redshift is a tracer at ``deflection_ratio`` time), and
    the model silently rendered with deflection_ratio = -1.357. Construction is the
    last point the value is concrete. Also: any concrete z <= 0 / non-finite raises.
    Falsifier: any construction below succeeding.
    """
    from gigalens.jax.scene import Component, Plane, LensModel
    from gigalens.jax.profiles.mass.epl import EPL
    from gigalens.jax.profiles.light.sersic import SersicEllipse

    light = lambda: Component(SersicEllipse(use_lstsq=False), dict(SRC_A, Ie=200.0))
    mass = lambda: Plane(redshift=Z_LENS, mass=[Component(EPL(50), EPL0)])
    # the P2 probe: light plane at z=0.3 in FRONT of the z=0.5 lens, listed after it
    with pytest.raises(ValueError, match="ordered observer"):
        LensModel([mass(), Plane(redshift=0.3, light=[light()])],
                  cosmo=_cosmo_component())
    # non-positive / non-finite concrete redshifts
    with pytest.raises(ValueError, match="finite and > 0"):
        LensModel([mass(), Plane(redshift=0.0, light=[light()])],
                  cosmo=_cosmo_component())
    with pytest.raises(ValueError, match="finite and > 0"):
        LensModel([mass(), Plane(redshift=float("nan"), light=[light()])],
                  cosmo=_cosmo_component())


def test_ordered_foreground_light_plane_is_legitimate_and_undeflected():
    """Claim: identity (scope pin for the C1 ordering guard). A foreground light plane
    listed BEFORE the mass plane (correct observer→source order) is legitimate use:
    it must construct, and its light must be rendered UNDEFLECTED — additive on top of
    the lensed scene. Identity: image(foreground + lensed scene) − image(lensed scene)
    == image(foreground alone, no mass), to the float64 render floor. Guards against
    the ordering check over-rejecting, and against a foreground plane being spuriously
    deflected. Falsifier: construction raising, or the identity failing.
    """
    from gigalens.jax.scene import Component, Plane, LensModel
    from gigalens.jax.scene_simulator import SceneSimulator
    from gigalens.jax.profiles.mass.epl import EPL
    from gigalens.jax.profiles.light.sersic import SersicEllipse

    fg = dict(R_sersic=0.4, n_sersic=1.8, e1=0.0, e2=0.0, center_x=0.6, center_y=0.5,
              Ie=150.0)
    def img(planes, cosmo):
        model = LensModel(planes, cosmo=cosmo)
        sim = SceneSimulator(model, _cfg())
        return np.asarray(sim.simulate(model.to_params({})))

    fg_plane = lambda: Plane(redshift=0.3,
                             light=[Component(SersicEllipse(use_lstsq=False), fg)])
    mass_plane = lambda: Plane(redshift=Z_LENS, mass=[Component(EPL(50), EPL0)])
    src_plane = lambda: Plane(redshift=Z_A, light=[Component(
        SersicEllipse(use_lstsq=False), dict(SRC_A, Ie=200.0))])

    with_fg = img([fg_plane(), mass_plane(), src_plane()], _cosmo_component())
    without_fg = img([mass_plane(), src_plane()], _cosmo_component())
    fg_alone = img([Plane(redshift=0.3, light=[Component(
        SersicEllipse(use_lstsq=False), fg)])], _cosmo_component())
    assert np.allclose(with_fg - without_fg, fg_alone,
                       rtol=tol.IMAGE_NOPSF_F64_RTOL, atol=tol.IMAGE_NOPSF_F64_ATOL), (
        f"foreground plane is not additive/undeflected: "
        f"max|Δ|={np.max(np.abs(with_fg - without_fg - fg_alone)):.2e}")


# --- C3 scene path: concrete unphysical cosmo constants flagged at CONSTRUCTION ----
def test_unphysical_cosmo_constants_flagged_at_model_construction():
    """Claim: contract. Fixed (constant) unphysical cosmology parameters must warn at
    ``LensModel`` construction — grader probe P3 showed the distance-layer flag is dead
    under jit (params are tracers), i.e. dead on the whole modeling path; construction
    is where constants are last concrete. Control: the physical cosmology used by every
    other test must NOT warn. Falsifier: no warning on Om0=1.5, or a spurious one on
    the control.
    """
    import warnings
    from gigalens.jax.scene import Component, Plane, LensModel
    from gigalens.jax.cosmo import wCDM_Cosmo
    from gigalens.jax.profiles.mass.epl import EPL
    from gigalens.jax.profiles.light.sersic import SersicEllipse

    planes = lambda: [
        Plane(redshift=Z_LENS, mass=[Component(EPL(50), EPL0)]),
        Plane(redshift=Z_A, light=[Component(SersicEllipse(use_lstsq=False),
                                             dict(SRC_A, Ie=200.0))])]
    bad = Component(wCDM_Cosmo(z_lens=Z_LENS, z_source_ref=Z_A),
                    dict(H0=70.0, Om0=1.5, k=0.0, w0=-1.0))
    with pytest.warns(UserWarning, match="Ode0"):
        LensModel(planes(), cosmo=bad)
    # grader round-2 F1: a SAMPLED H0 with a concrete unphysical Om0 (the common
    # inference configuration) must still flag — the check substitutes a nominal H0.
    from tensorflow_probability.substrates.jax import distributions as tfd
    mixed = Component(wCDM_Cosmo(z_lens=Z_LENS, z_source_ref=Z_A),
                      dict(H0=tfd.Normal(70.0, 5.0), Om0=-0.5, k=0.0, w0=-1.0))
    with pytest.warns(UserWarning, match="Om0"):
        LensModel(planes(), cosmo=mixed)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        LensModel(planes(), cosmo=_cosmo_component())
