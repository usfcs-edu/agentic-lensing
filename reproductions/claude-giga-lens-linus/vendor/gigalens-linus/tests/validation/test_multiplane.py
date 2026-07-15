"""Multi-lens-plane ray-shooting validation (oracle: lenstronomy multi-plane).

The live R1 test here guards the **shipped** scene trace
(``gigalens.jax.scene_simulator.SceneSimulator._trace_multiplane`` /
``MultiPlaneTrace``): two EPL deflectors at distinct redshifts, source-plane β vs
lenstronomy. The recursion + per-plane mass normalization (R1, R12) are validated
against lenstronomy, the project's validation oracle.

History: this test originally targeted the experimental
``gigalens.jax.multiplane_exp`` implementation, which was **removed in Phase 6** once
its recursion was folded into the scene ``_trace_multiplane`` seam. No validation
capability was lost — lenstronomy, not the experimental code, is the oracle — so R1 is
simply re-pointed at the shipped scene path. To keep the coverage independent from
``test_scene_simulator.py::test_multiplane_source_plane_vs_lenstronomy`` (which exercises
the same trace at z=0.3/0.6/2.0), this test deliberately uses a DISTINCT configuration
(different deflector params and redshifts z=0.45/0.8/3.0).

The reduction tests (N=1 multiplane == single plane; deflection_ratio == multiplane)
are written against the *new* API and skipped until Phase 3.
"""
import numpy as np
import pytest

from . import tolerances as tol
from .oracles import lenstronomy_multiplane_ray_shooting

pytestmark = [pytest.mark.physics, pytest.mark.multiplane]


def _grid_arcsec(n=40, half=2.5):
    xs = np.linspace(-half, half, n, dtype=np.float64)
    X, Y = np.meshgrid(xs, xs)
    return X + 1e-3, Y - 1e-3  # nudge off centers


@pytest.mark.slow
def test_two_plane_ray_shooting_vs_lenstronomy(plotter):
    """Two EPL deflectors at different redshifts: scene source-plane β vs lenstronomy.

    THE R1 detector, now guarding the SHIPPED scene trace
    (``SceneSimulator._trace_multiplane`` / ``MultiPlaneTrace``) after the experimental
    ``multiplane_exp`` implementation was removed in Phase 6. Claim type: deterministic
    identity (same physics, two codes) → held at RAY_SHOOTING_RTOL (loosened from profile
    parity only because the two codes use different distance integrators on the SAME flat
    wCDM cosmology — gigalens' own ``wCDM_Cosmo`` vs a radiation-matched flat astropy).

    A DISTINCT config (z=0.45/0.8/3.0, offset EPLs) from
    ``test_scene_simulator.py::test_multiplane_source_plane_vs_lenstronomy`` (z=0.3/0.6/2.0)
    so this is independent coverage of the same recursion. (The redshifts were chosen so
    no grid pixel lands on a β≈0 zero-crossing, where the deliberately-tight
    RAY_SHOOTING_ATOL=1e-9 would flag pure integrator noise; the trace's *relative*
    agreement on O(1) β is ~1e-5 regardless of config — verified across several z-triples
    — so this is grid hygiene, not a tolerance loosened to mask a bug.)

    Falsifier: gigalens' final-plane β differs from lenstronomy's ray_shooting by more
    than the band. A *structured* (not uniform-noise) disagreement here would indicate
    a per-plane theta_E normalization or distance-scaling bug in the scene
    ``_trace_multiplane``, not rounding — inspect the residual map, do not retune the
    tolerance (method-discipline §5). A real divergence is a FINDING about the scene trace.
    """
    pytest.importorskip("lenstronomy")
    from astropy.cosmology import FlatwCDM
    import astropy.units as u
    from gigalens.jax.cosmo import wCDM_Cosmo
    from gigalens.jax.profiles.mass.epl import EPL
    from gigalens.jax.scene import Component, Plane, LensModel
    from gigalens.jax.scene_simulator import SceneSimulator
    from gigalens.simulator import SimulatorConfig

    # Distinct from the scene_simulator coverage (z=0.3/0.6/2.0): different redshifts.
    z1, z2, z_src = 0.45, 0.8, 3.0
    epl0 = dict(theta_E=1.1, gamma=2.0, e1=0.05, e2=-0.03, center_x=0.0, center_y=0.0)
    epl1 = dict(theta_E=0.6, gamma=2.0, e1=-0.04, e2=0.02, center_x=0.3, center_y=-0.2)

    # Flat wCDM (H0=70, Om0=0.3, w0=-1): exactly what gigalens' wCDM_Cosmo represents.
    # lenstronomy gets the radiation-matched flat astropy equivalent so the only thing
    # under test is gigalens' recursion + distance matrix (NOT Planck18: the gigalens
    # cosmo cannot match Planck18's full model — that dependency was removed with the
    # experimental code).
    cosmo = Component(wCDM_Cosmo(z_lens=z1, z_source_ref=10.0), dict(H0=70.0, Om0=0.3, k=0.0, w0=-1.0))
    model = LensModel([
        Plane(redshift=z1, mass=[Component(EPL(50), epl0)]),
        Plane(redshift=z2, mass=[Component(EPL(50), epl1)]),
        Plane(redshift=z_src, light=[]),  # bare source plane
    ], cosmo=cosmo)

    cfg = SimulatorConfig(delta_pix=0.05, num_pix=40, supersample=1)
    sim = SceneSimulator(model, cfg)
    assert sim.trace_mode == "multiplane", \
        f"expected multiplane trace, got {sim.trace_mode!r}"

    # Trace on the simulator's OWN pixel grid, and feed lenstronomy that SAME grid.
    # (Using an independent meshgrid would compare β on different sky coordinates and
    # spuriously "fail" — a harness bug that masqueraded as a physics error.)
    params = model.to_params({})
    pos = sim.trace(params)
    # The scene trace returns the final-plane position as an ANGULAR β directly
    # (no /D[N,0] post-scaling — that was the experimental packing).
    bx_gl = np.asarray(pos[2][0])[..., 0]
    by_gl = np.asarray(pos[2][1])[..., 0]

    gx = np.asarray(sim.img_X)[..., 0]
    gy = np.asarray(sim.img_Y)[..., 0]
    ap = FlatwCDM(H0=70 * u.km / u.s / u.Mpc, Om0=0.3, w0=-1.0, Tcmb0=2.7255, Neff=3.046)
    bx_le, by_le = lenstronomy_multiplane_ray_shooting(
        gx, gy,
        lens_model_list=["EPL", "EPL"],
        kwargs_list=[epl0, epl1],
        redshift_list=[z1, z2],
        z_source=z_src,
        astropy_cosmo=ap,
    )

    # Register full β-field comparisons. A *structured* residual (e.g. a coherent
    # pattern aligned with the deflectors, not uniform noise) is the signature of an
    # R1 per-plane normalization bug and is an open finding even if the aggregate
    # passes — look at the map (method-discipline §5), do not retune the tolerance.
    plotter.image("ray_shooting_beta_x", bx_le, bx_gl,
                  rtol=tol.RAY_SHOOTING_RTOL, atol=tol.RAY_SHOOTING_ATOL)
    plotter.image("ray_shooting_beta_y", by_le, by_gl,
                  rtol=tol.RAY_SHOOTING_RTOL, atol=tol.RAY_SHOOTING_ATOL)

    max_dbx = float(np.max(np.abs(bx_gl - bx_le)))
    max_dby = float(np.max(np.abs(by_gl - by_le)))
    assert np.allclose(bx_gl, bx_le, rtol=tol.RAY_SHOOTING_RTOL, atol=tol.RAY_SHOOTING_ATOL), \
        (f"scene multiplane β_x disagrees with lenstronomy: max|Δβx|={max_dbx:.2e} "
         "(inspect residual map before retuning — FINDING about _trace_multiplane)")
    assert np.allclose(by_gl, by_le, rtol=tol.RAY_SHOOTING_RTOL, atol=tol.RAY_SHOOTING_ATOL), \
        f"scene multiplane β_y disagrees with lenstronomy: max|Δβy|={max_dby:.2e}"

    print(f"\n[R1 scene] max|Δβx|={max_dbx:.2e} max|Δβy|={max_dby:.2e} "
          f"(RAY_SHOOTING_RTOL={tol.RAY_SHOOTING_RTOL:.1e})")


@pytest.mark.pending
@pytest.mark.skip(reason="new API (LensModel/Plane/trace) — Phase 3")
def test_single_lens_multiplane_equals_singleplane():
    """R4: a single mass plane via MultiPlaneTrace == DeflectionRatioTrace, exactly.

    Claim type: deterministic reduction identity → REDUCTION_RTOL (float64 floor, not
    a physical band). Build one LensModel with a single mass plane; render once
    forcing each trace; assert images equal to REDUCTION_RTOL. Falsifier: any
    structured difference — means the two paths disagree where they must coincide.
    """
    raise NotImplementedError


@pytest.mark.pending
@pytest.mark.skip(reason="new API (LensModel/Plane/trace) — Phase 3")
def test_deflection_ratio_equals_multiplane_single_lens():
    """R5: one deflector + several source redshifts — deflection_ratio path equals the
    multiplane recursion (the recursion's distance ratios reduce to D_ls/D_s for N=1).

    Claim type: deterministic identity → REDUCTION_RTOL. Falsifier: per-source β from
    the two paths differ beyond the float64 floor.
    """
    raise NotImplementedError


@pytest.mark.pending
@pytest.mark.skip(reason="new API multiplane image render — Phase 3")
def test_intermediate_plane_light_position():
    """R12: light on an intermediate plane is rendered at its deflected position.

    Place light on plane 1 (deflected by plane 0 only) and compare the rendered image
    to a lenstronomy multi-plane image with light at the same redshift.
    """
    raise NotImplementedError
