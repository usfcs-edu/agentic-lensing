"""SceneSimulator validation (Phase 3): trace + render core, single- and multi-plane.

Layered to isolate the new plumbing from profile parity (method-discipline §2 "name the
link"): the single-plane render test uses gigalens' OWN light at a lenstronomy-computed
β, so it checks *trace + assembly + grid + flux scaling* at tight tolerance without the
sersic-parity floor. The lstsq test is an oracle-free amplitude round-trip. The
multiplane test checks the recursion's source-plane β against lenstronomy.
"""
import jax.numpy as jnp
import numpy as np
import pytest

from . import tolerances as tol
from .oracles import lenstronomy_multiplane_ray_shooting

pytestmark = pytest.mark.physics

EPL0 = dict(theta_E=1.1, gamma=2.05, e1=0.04, e2=-0.03, center_x=0.0, center_y=0.0)
SRC = dict(R_sersic=0.3, n_sersic=2.0, e1=0.05, e2=-0.02, center_x=0.05, center_y=-0.04)
LL = dict(R_sersic=1.2, n_sersic=3.0, e1=0.0, e2=0.0, center_x=0.0, center_y=0.0)


def _cfg(num_pix=50, delta_pix=0.05, supersample=1, kernel=None):
    from gigalens.simulator import SimulatorConfig
    return SimulatorConfig(delta_pix=delta_pix, num_pix=num_pix, supersample=supersample,
                           kernel=kernel, likelihood_precision="float64")


def test_single_plane_render_assembly_no_psf(plotter):
    """No-PSF single-plane image: lens light at θ + source at β=θ−α, ×pixel-area.

    Reference uses gigalens light at a lenstronomy-computed deflection, so a tight
    IMAGE_NOPSF_F64 tolerance probes the simulator's trace/assembly/scaling, not sersic
    parity. Falsifier: any >1e-9 disagreement ⇒ a grid/deflection-sign/flux-scale bug.
    """
    pytest.importorskip("lenstronomy")
    from lenstronomy.LensModel.Profiles.epl import EPL as LEPL
    from gigalens.jax.profiles.mass.epl import EPL
    from gigalens.jax.profiles.light.sersic import SersicEllipse
    from gigalens.jax.scene import Component, Plane, LensModel
    from gigalens.jax.scene_simulator import SceneSimulator

    src_prof = SersicEllipse(use_lstsq=False)
    ll_prof = SersicEllipse(use_lstsq=False)
    model = LensModel([
        Plane(mass=[Component(EPL(50), EPL0)], light=[Component(ll_prof, dict(LL, Ie=500.0))]),
        Plane(deflection_ratio=1.0, light=[Component(src_prof, dict(SRC, Ie=200.0))]),
    ])
    cfg = _cfg()
    sim = SceneSimulator(model, cfg)
    params = model.to_params({})
    got = np.asarray(sim.simulate(params))

    gx = np.asarray(sim.img_X)[..., 0]
    gy = np.asarray(sim.img_Y)[..., 0]
    ax, ay = LEPL().derivatives(x=gx, y=gy, **EPL0)        # independent deflection
    bx, by = gx - ax, gy - ay
    src = np.asarray(src_prof.light(bx[..., None], by[..., None], Ie=200.0, **SRC))[..., 0]
    ll = np.asarray(ll_prof.light(gx[..., None], gy[..., None], Ie=500.0, **LL))[..., 0]
    ref = (src + ll) * float(np.linalg.det(np.asarray(sim.transform_pix2angle)))

    plotter.image("single_plane_image_no_psf", ref, got,
                  rtol=tol.IMAGE_NOPSF_F64_RTOL, atol=tol.IMAGE_NOPSF_F64_ATOL)
    assert np.allclose(got, ref, rtol=tol.IMAGE_NOPSF_F64_RTOL, atol=tol.IMAGE_NOPSF_F64_ATOL), \
        f"max|Δ|={np.max(np.abs(got-ref)):.2e}"


def test_lstsq_recovers_known_amplitudes(plotter):
    """Plant amplitudes via a forward render (no noise), recover them via lstsq.

    Oracle-free round-trip on ``_solve_normal_eq_with_fallback``. Falsifier: recovered
    amplitudes differ from the planted ones beyond the conditioning floor ⇒ a
    design-matrix / weighting bug.
    """
    from gigalens.jax.profiles.mass.epl import EPL
    from gigalens.jax.profiles.light.sersic import SersicEllipse
    from gigalens.jax.scene import Component, Plane, LensModel
    from gigalens.jax.scene_simulator import SceneSimulator

    A_ll, A_src = 500.0, 200.0
    fwd = LensModel([
        Plane(mass=[Component(EPL(50), EPL0)],
              light=[Component(SersicEllipse(use_lstsq=False), dict(LL, Ie=A_ll))]),
        Plane(deflection_ratio=1.0,
              light=[Component(SersicEllipse(use_lstsq=False), dict(SRC, Ie=A_src))]),
    ])
    cfg = _cfg()
    observed = np.asarray(SceneSimulator(fwd, cfg).simulate(fwd.to_params({})))

    lstsq = LensModel([
        Plane(mass=[Component(EPL(50), EPL0)],
              light=[Component(SersicEllipse(use_lstsq=True), LL)]),
        Plane(deflection_ratio=1.0,
              light=[Component(SersicEllipse(use_lstsq=True), SRC)]),
    ])
    sim = SceneSimulator(lstsq, cfg)
    import jax.numpy as jnp
    obs = jnp.asarray(observed)
    err = jnp.ones_like(obs)
    lp = lstsq.to_params({})

    # The lstsq best-fit image reproduces the (noiseless, in-span) observed image.
    recon = np.asarray(sim.lstsq_simulate(lp, obs, err))
    plotter.image("lstsq_observed_vs_reconstruction", observed, recon, rtol=1e-6, atol=1e-9)
    assert np.allclose(recon, observed, rtol=1e-6, atol=1e-9), \
        f"reconstruction max|Δ|={np.max(np.abs(recon-observed)):.2e}"

    # Units: the basis is rendered WITHOUT the pixel-area conversion_factor (matching the
    # legacy lstsq path), so a recovered coeff equals Ie × pixel_area. Document + check.
    cf = float(np.linalg.det(np.asarray(sim.transform_pix2angle)))
    coeffs = np.squeeze(np.asarray(sim.lstsq_simulate(lp, obs, err, return_coeffs=True)))
    assert np.allclose(coeffs, [A_ll * cf, A_src * cf], rtol=1e-5), f"coeffs={coeffs}, cf={cf}"


@pytest.mark.slow
def test_multiplane_source_plane_vs_lenstronomy(plotter):
    """SceneSimulator's MultiPlaneTrace source-plane β vs lenstronomy ray_shooting.

    Two EPL deflectors at z=0.3, 0.6; source at z=2.0. Distances from gigalens' own
    cosmology (matched to a radiation-matched flat astropy for lenstronomy). Claim type:
    deterministic identity → RAY_SHOOTING_RTOL. Falsifier: structured β residual ⇒ a
    recursion/distance bug in the *new* trace (vs the experimental one tested elsewhere).
    """
    pytest.importorskip("lenstronomy")
    from astropy.cosmology import FlatwCDM
    import astropy.units as u
    from gigalens.jax.profiles.mass.epl import EPL
    from gigalens.jax.profiles.light.sersic import SersicEllipse
    from gigalens.jax.cosmo import wCDM_Cosmo
    from gigalens.jax.scene import Component, Plane, LensModel
    from gigalens.jax.scene_simulator import SceneSimulator

    z1, z2, zs = 0.3, 0.6, 2.0
    epl1 = dict(theta_E=0.6, gamma=2.0, e1=-0.04, e2=0.02, center_x=0.3, center_y=-0.2)
    cosmo = Component(wCDM_Cosmo(z_lens=z1, z_source_ref=10.0), dict(H0=70.0, Om0=0.3, k=0.0, w0=-1.0))
    model = LensModel([
        Plane(redshift=z1, mass=[Component(EPL(50), EPL0)]),
        Plane(redshift=z2, mass=[Component(EPL(50), epl1)]),
        Plane(redshift=zs, light=[Component(SersicEllipse(use_lstsq=True), SRC)]),
    ], cosmo=cosmo)
    assert SceneSimulator(model, _cfg(num_pix=40)).trace_mode == "multiplane"
    sim = SceneSimulator(model, _cfg(num_pix=40))
    params = model.to_params({})
    pos = sim.trace(params)
    bx_gl = np.asarray(pos[2][0])[..., 0]
    by_gl = np.asarray(pos[2][1])[..., 0]

    gx = np.asarray(sim.img_X)[..., 0]
    gy = np.asarray(sim.img_Y)[..., 0]
    ap = FlatwCDM(H0=70 * u.km / u.s / u.Mpc, Om0=0.3, w0=-1.0, Tcmb0=2.7255, Neff=3.046)
    bx_le, by_le = lenstronomy_multiplane_ray_shooting(
        gx, gy, ["EPL", "EPL"], [EPL0, epl1], [z1, z2], zs, ap)

    plotter.image("scene_multiplane_beta_x", bx_le, bx_gl,
                  rtol=tol.RAY_SHOOTING_RTOL, atol=tol.RAY_SHOOTING_ATOL)
    plotter.image("scene_multiplane_beta_y", by_le, by_gl,
                  rtol=tol.RAY_SHOOTING_RTOL, atol=tol.RAY_SHOOTING_ATOL)
    assert np.allclose(bx_gl, bx_le, rtol=tol.RAY_SHOOTING_RTOL, atol=tol.RAY_SHOOTING_ATOL), \
        f"max|Δβx|={np.max(np.abs(bx_gl-bx_le)):.2e}"
    assert np.allclose(by_gl, by_le, rtol=tol.RAY_SHOOTING_RTOL, atol=tol.RAY_SHOOTING_ATOL)


def test_single_plane_cosmo_autoscales_deflection_ratio():
    """Single deflector + cosmology: the deflection_ratio trace must DERIVE the source
    plane's scaling from the cosmology (r = deflection_ratio(z_s) normalized to
    z_source_ref) and apply β = θ − r·α — equivalent to passing that same ratio
    explicitly with no cosmology.

    Falsifier: if the cosmo model's traced source position differs from the explicit-ratio
    model's (beyond float64 noise), the auto-scaling is mis-wired; if it equals the
    undeflected grid, the deflection is silently dropped (the original bug). z_source_ref
    is deliberately ≠ z_source so the ratio is not the trivial 1.0.
    """
    from gigalens.jax.cosmo import wCDM_Cosmo
    from gigalens.jax.profiles.mass.epl import EPL
    from gigalens.jax.profiles.light.sersic import SersicEllipse
    from gigalens.jax.scene import Component, Plane, LensModel
    from gigalens.jax.scene_simulator import SceneSimulator

    z_lens, z_src, z_ref = 0.4, 1.5, 2.0
    cosmo = Component(wCDM_Cosmo(z_lens=z_lens, z_source_ref=z_ref),
                      dict(H0=70.0, Om0=0.3, k=0.0, w0=-1.0))
    r = float(np.asarray(cosmo.profile.deflection_ratio(
        jnp.asarray(float(z_src)), H0=70.0, Om0=0.3, k=0.0, w0=-1.0)).ravel()[0])
    assert abs(r - 1.0) > 0.05, f"pick z_ref so the ratio is non-trivial; got r={r}"

    cosmo_model = LensModel([
        Plane(redshift=z_lens, mass=[Component(EPL(50), EPL0)]),
        Plane(redshift=z_src, light=[Component(SersicEllipse(use_lstsq=True), SRC)]),
    ], cosmo=cosmo)
    explicit_model = LensModel([
        Plane(mass=[Component(EPL(50), EPL0)]),
        Plane(deflection_ratio=r, light=[Component(SersicEllipse(use_lstsq=True), SRC)]),
    ])

    sim_c = SceneSimulator(cosmo_model, _cfg(num_pix=40))
    sim_e = SceneSimulator(explicit_model, _cfg(num_pix=40))
    assert sim_c.trace_mode == "deflection_ratio"

    pos_c = sim_c.trace(cosmo_model.to_params({}))[1]
    pos_e = sim_e.trace(explicit_model.to_params({}))[1]
    gx, gy = np.asarray(sim_c.img_X), np.asarray(sim_c.img_Y)
    # (1) deflection is actually applied (not the undeflected grid).
    assert np.max(np.abs(np.asarray(pos_c[0]) - gx)) > 1e-3, "no deflection applied"
    # (2) auto-from-cosmo == explicit ratio, to float64 precision.
    assert np.allclose(np.asarray(pos_c[0]), np.asarray(pos_e[0]), rtol=1e-10, atol=1e-10), \
        f"βx mismatch max={np.max(np.abs(np.asarray(pos_c[0])-np.asarray(pos_e[0]))):.2e}"
    assert np.allclose(np.asarray(pos_c[1]), np.asarray(pos_e[1]), rtol=1e-10, atol=1e-10)


def test_z_source_ref_is_required():
    """z_source_ref has no silent default: constructing a cosmology without it must fail."""
    from gigalens.jax.cosmo import wCDM_Cosmo
    with pytest.raises(TypeError):
        wCDM_Cosmo(z_lens=0.4)  # missing z_source_ref


def test_single_deflector_zlens_mismatch_raises():
    """The mass plane redshift must match cosmo.z_lens in deflection_ratio mode."""
    from gigalens.jax.cosmo import wCDM_Cosmo
    from gigalens.jax.profiles.mass.epl import EPL
    from gigalens.jax.profiles.light.sersic import SersicEllipse
    from gigalens.jax.scene import Component, Plane, LensModel
    from gigalens.jax.scene_simulator import SceneSimulator

    cosmo = Component(wCDM_Cosmo(z_lens=0.962, z_source_ref=2.0),
                      dict(H0=70.0, Om0=0.3, k=0.0, w0=-1.0))
    model = LensModel([
        Plane(redshift=0.49, mass=[Component(EPL(50), EPL0)]),   # ≠ cosmo z_lens
        Plane(redshift=1.5, light=[Component(SersicEllipse(use_lstsq=True), SRC)]),
    ], cosmo=cosmo)
    with pytest.raises(ValueError, match="z_lens"):
        SceneSimulator(model, _cfg(num_pix=20))


@pytest.mark.skip(reason="legacy LensSimulator/PhysicalModel removed with the old API; the "
                         "new==old single-plane-PSF parity claim is now covered by the "
                         "certified §6b golden anchor test_regression_anchor.py (NEW vs frozen "
                         "golden). Restore the old API from git b82397c to run this.")
def test_single_plane_with_psf_matches_legacy(plotter):
    """End-to-end: full lensed image (PSF + supersampling) — new SceneSimulator vs the
    trusted legacy LensSimulator on the *same* physical model and parameters.

    This is the all-encompassing single-plane diagnostic: it exercises the entire
    pipeline (deflection → source/lens light → PSF convolution → supersample pooling →
    pixel-area scaling) and doubles as a partial §6b anchor (new == old). The new and
    legacy paths share the conv/pool/solver helpers, so they should agree to ~float64
    machine precision. Falsifier: any structured residual ⇒ the new render diverges
    from the validated one. The plot shows the realistic lensed image for inspection.

    RETIRED: the legacy LensSimulator/PhysicalModel were removed with the old gigalens
    API, so this NEW-vs-OLD parity check is no longer runnable. Its claim (the scene
    render reproduces the old render at the float64 floor) is now carried by the certified
    §6b golden anchor (``test_regression_anchor.py``, NEW vs frozen golden, scene-only).
    The body is kept under the skip to document the original intent; the now-broken
    old-API imports are removed so collection does not error.
    """
    import numpy as _np
    from gigalens.jax.profiles.mass.epl import EPL
    from gigalens.jax.profiles.light.sersic import SersicEllipse
    from gigalens.jax.scene import Component, Plane, LensModel
    from gigalens.jax.scene_simulator import SceneSimulator
    # NOTE: the legacy ``PhysicalModel`` / ``LensSimulator`` imports were removed (old API
    # dropped). The legacy-render half of the body below is dead under the skip.

    # A small normalized Gaussian PSF, and supersampling on, to exercise the full path.
    yy, xx = _np.mgrid[-10:11, -10:11]
    psf = _np.exp(-(xx**2 + yy**2) / (2 * 2.0**2)); psf = psf / psf.sum()
    cfg = _cfg(num_pix=60, supersample=2, kernel=psf)

    # New API
    model = LensModel([
        Plane(mass=[Component(EPL(50), EPL0)],
              light=[Component(SersicEllipse(use_lstsq=False), dict(LL, Ie=500.0))]),
        Plane(deflection_ratio=1.0,
              light=[Component(SersicEllipse(use_lstsq=False), dict(SRC, Ie=200.0))]),
    ])
    new_img = np.asarray(SceneSimulator(model, cfg).simulate(model.to_params({})))

    # Legacy API (source via the old is_source/deflection_ratio mechanism)
    src = SersicEllipse(use_lstsq=False, is_source=True, cosmo_sample=False)
    phys = PhysicalModel([EPL(50)], [SersicEllipse(use_lstsq=False)], [src])
    legacy_params = {
        "lens_mass": {"0": EPL0},
        "lens_light": {"0": dict(LL, Ie=500.0)},
        "source_light": {"0": dict(SRC, Ie=200.0, deflection_ratio=1.0)},
    }
    old_img = np.asarray(LensSimulator(phys, cfg, bs=1).simulate(legacy_params))

    plotter.image("single_plane_with_psf__new_vs_legacy", old_img, new_img,
                  rtol=1e-9, atol=1e-9, labels=("legacy LensSimulator", "new SceneSimulator"))
    assert np.allclose(new_img, old_img, rtol=1e-9, atol=1e-9), \
        f"new vs legacy max|Δ|={np.max(np.abs(new_img-old_img)):.2e}"
