"""Cosmological-distance validation (oracle: astropy).

These run against the *current* ``gigalens.jax.cosmo`` — they are part of Phase 0
and will surface existing issues (notably the missing curvature term, R2) before the
multiplane trace is built on top of these distances.

Links under test are kept separate (method-discipline §2):
  - integrator accuracy (is the 1000-pt trapezoid fine enough?)  -> self-convergence
  - model + integrator (flat)                                    -> vs astropy
  - curvature handling (non-flat)                                -> vs astropy (xfail)

Radiation caveat: gigalens always adds Or0 = 2.469e-5 h^-2 (1+0.2271 Neff); astropy
adds radiation only via Tcmb0/Neff. We match them as closely as the APIs allow; the
residual radiation-model difference (~few e-5 in Ω) is part of what COSMO_DISTANCE_RTOL
absorbs — if a comparison fails at that band, suspect a *model* mismatch, not the
integrator.
"""
import numpy as np
import pytest

from . import tolerances as tol

pytestmark = pytest.mark.cosmology

# Flat wCDM expressed for gigalens efunc(z, H0, Om0, k, w0); k=0 is flat.
GL_KW = dict(H0=70.0, Om0=0.3, k=0.0, w0=-1.0)
Z_GRID = [0.2, 0.5, 1.0, 2.0, 4.0]


def _gl_cosmo():
    from gigalens.jax.cosmo import wCDM_Cosmo
    return wCDM_Cosmo(z_lens=0.5, z_source_ref=10.0)


def test_distance_integral_self_convergence():
    """Integrator-only check: the default trapezoid vs a far finer one.

    Claim type: numerical-convergence. Falsifier: default-grid comoving distance
    differs from an 8x-finer grid by more than COSMO_INTEGRATOR_RTOL, i.e. n_grid=1000
    is too coarse and distances carry integrator error above their stated band.
    """
    cosmo = _gl_cosmo()

    def integrand(z):
        return 1.0 / cosmo.efunc(z, **GL_KW)

    for z in Z_GRID:
        coarse = float(cosmo._integrate(integrand, 0.0, z, n_grid=1000))
        fine = float(cosmo._integrate(integrand, 0.0, z, n_grid=8000))
        assert np.isclose(coarse, fine, rtol=tol.COSMO_INTEGRATOR_RTOL), (
            f"trapezoid not converged at z={z}: {coarse} vs {fine}")


def test_comoving_distance_vs_astropy_flat(astropy_cosmo, plotter):
    """Model+integrator check (flat): line-of-sight comoving distance vs astropy.

    Comoving distance has no (1+z) or curvature subtlety, so this isolates the
    1/E(z) model + integrator. Falsifier: >COSMO_DISTANCE_RTOL relative disagreement
    at any z (a model/units bug, since the integrator is shown converged above).
    """
    from .oracles import astropy_comoving_distance
    cosmo = _gl_cosmo()
    zs = np.linspace(0.1, 5.0, 25)
    gl = np.array([float(cosmo.comoving_distance(z, **GL_KW)) for z in zs])
    ap = np.array([astropy_comoving_distance(astropy_cosmo, z) for z in zs])
    # Plot the full D_C(z) curve + fractional residual (eyeball any z-dependent drift,
    # which a per-point allclose would not localize — method-discipline §5).
    plotter.curve("comoving_distance_vs_z", zs, ap, gl,
                  xlabel="z", ylabel="D_C(z) [Mpc]")
    for z, g, a in zip(zs, gl, ap):
        assert np.isclose(g, a, rtol=tol.COSMO_DISTANCE_RTOL), (
            f"comoving D_C(z={z:.2f}) Mpc: gigalens {g} vs astropy {a}")


def test_angular_diameter_distance_vs_astropy_flat(astropy_cosmo):
    """Model+integrator check (flat): D_A(0,z) and D_A(z1,z2) vs astropy.

    D_A(z1,z2) is what the multiplane distance matrix is built from. For a FLAT
    cosmology gigalens' ``comoving_z1z2/(1+z2)`` equals astropy. Falsifier:
    >COSMO_DISTANCE_RTOL disagreement.
    """
    from .oracles import (
        astropy_angular_diameter_distance,
        astropy_angular_diameter_distance_z1z2,
    )
    cosmo = _gl_cosmo()
    for z in Z_GRID:
        gl = float(cosmo.angular_distance(z, **GL_KW))
        ap = astropy_angular_diameter_distance(astropy_cosmo, z)
        assert np.isclose(gl, ap, rtol=tol.COSMO_DISTANCE_RTOL), (
            f"D_A(0,{z}): gigalens {gl} vs astropy {ap}")

    for z1, z2 in [(0.5, 1.0), (0.5, 2.0), (1.0, 4.0)]:
        gl = float(cosmo.angular_distance_z1z2(z1, z2, **GL_KW))
        ap = astropy_angular_diameter_distance_z1z2(astropy_cosmo, z1, z2)
        assert np.isclose(gl, ap, rtol=tol.COSMO_DISTANCE_RTOL), (
            f"D_A({z1},{z2}): gigalens {gl} vs astropy {ap}")


def _open_wcdm_pair():
    """A matched OPEN (Omega_k0 = +0.05) gigalens/astropy pair, radiation-matched.

    gigalens k maps to Omega_k0 = -k/H0^2, so Omega_k0 = +0.05 ⇒ k = -0.05 H0^2.
    astropy's Ode0 is set so its Omega_k0 equals gigalens' (using gigalens' own Or0 so
    the two radiation densities agree to ~1e-6), and Tcmb0/Neff match the suite's
    radiation convention. Returns (gigalens_cosmo, gl_kwargs, astropy_cosmo).
    """
    from astropy.cosmology import wCDM
    import astropy.units as u

    H0, Om0, w0, Ok0 = 70.0, 0.3, -1.0, 0.05
    k = -Ok0 * H0**2
    gl_kw = dict(H0=H0, Om0=Om0, k=k, w0=w0)
    cosmo = _gl_cosmo()
    Or0 = float(cosmo.omega_rad0(H0))            # gigalens' radiation density
    ode0 = 1.0 - Om0 - Ok0 - Or0                  # ⇒ astropy Omega_k0 ≈ 0.05
    ap = wCDM(H0=H0 * u.km / u.s / u.Mpc, Om0=Om0, Ode0=ode0, w0=w0,
              Tcmb0=2.7255, Neff=3.046)
    return cosmo, gl_kw, ap


def test_nonflat_transverse_comoving_distance_vs_astropy():
    """R2 (curvature): transverse comoving distance D_M(0,z) for an OPEN universe.

    Direct check of the S_k (sinh) transform against astropy's
    ``comoving_transverse_distance``. Claim type: model identity → COSMO_DISTANCE_RTOL.
    Falsifier: >1e-4 disagreement ⇒ the curvature transform is wrong. (Was an xfail
    ratchet while curvature was unimplemented; implemented in Phase 1, now a real test.)
    """
    cosmo, gl_kw, ap = _open_wcdm_pair()
    for z in [0.5, 1.0, 2.0, 4.0]:
        gl = float(cosmo._transverse_comoving_distance(z, **gl_kw))
        a = float(ap.comoving_transverse_distance(z).value)
        assert np.isclose(gl, a, rtol=tol.COSMO_DISTANCE_RTOL), (
            f"open D_M(0,{z}): gigalens {gl} vs astropy {a}")


def test_nonflat_angular_diameter_distance_vs_astropy():
    """R2 (curvature): D_A(z1,z2) for an OPEN universe vs astropy's curved value.

    Exercises the full Hogg eq.19 (the form the multiplane trace relies on). Claim
    type: model identity → COSMO_DISTANCE_RTOL. Falsifier: >1e-4 disagreement.
    """
    from .oracles import astropy_angular_diameter_distance_z1z2
    cosmo, gl_kw, ap = _open_wcdm_pair()
    for z1, z2 in [(0.5, 1.0), (0.5, 2.0), (1.0, 4.0)]:
        gl = float(cosmo.angular_distance_z1z2(z1, z2, **gl_kw))
        a = astropy_angular_diameter_distance_z1z2(ap, z1, z2)
        assert np.isclose(gl, a, rtol=tol.COSMO_DISTANCE_RTOL), (
            f"open D_A({z1},{z2}): gigalens {gl} vs astropy {a}")


def test_distance_matrix_vs_astropy_flat(astropy_cosmo):
    """The multi-plane distance matrix (T_xy) built from gigalens cosmo vs astropy.

    Validates the exact object MultiPlaneTrace will consume (Phase 3): D[j,i] =
    transverse comoving distance D_M(z_i, z_j), with T_xy = (1+z2)·D_A(z1,z2) as the
    astropy oracle. Claim type: model identity → COSMO_DISTANCE_RTOL. Falsifier: any
    lower-triangular entry disagrees > band (a units/layout/curvature bug in the
    builder), or an on/above-diagonal entry is non-zero.
    """
    cosmo = _gl_cosmo()
    zs = [0.3, 0.6, 2.0]
    D = np.asarray(cosmo.distance_matrix(zs, **GL_KW))
    full = [0.0] + zs
    assert D.shape == (4, 4)
    for j in range(4):
        for i in range(4):
            if j > i:
                tij = (1 + full[j]) * float(
                    astropy_cosmo.angular_diameter_distance_z1z2(full[i], full[j]).value)
                assert np.isclose(D[j, i], tij, rtol=tol.COSMO_DISTANCE_RTOL), (
                    f"T_xy(z{i}={full[i]}, z{j}={full[j]}): gigalens {D[j, i]} vs astropy {tij}")
            else:
                assert D[j, i] == 0.0, f"D[{j},{i}] should be 0 (on/above diagonal)"


def test_shared_batch_redshift_and_cosmo_params_elementwise():
    """Batching convention: a SAMPLED redshift and SAMPLED cosmo params share one batch
    axis, so distances combine element-wise -> shape (N,), never an (N, N) outer product.

    Regression for the ``_trace_deflection_ratio`` shape blow-up: when a source/plane
    redshift and Om0/w0 are sampled together, ``comoving_distance_z1z2`` used to give the
    param batch its own trailing axis on top of the redshift-batched quadrature grid,
    yielding (n_grid, N, N). Falsifier: any (N, N) shape, or a batched value that differs
    from the per-sample scalar computation beyond float64 noise.
    """
    import jax
    import jax.numpy as jnp
    from gigalens.jax.cosmo import wCDM_Cosmo

    cosmo = wCDM_Cosmo(z_lens=0.5, z_source_ref=1.2)
    N = 5
    rng = np.random.default_rng(0)
    z = jnp.asarray(1.3 + 2.0 * rng.random(N))          # sampled source redshift
    Om0 = jnp.asarray(0.1 + 0.5 * rng.random(N))        # sampled cosmology
    w0 = jnp.asarray(-1.3 + 0.4 * rng.random(N))
    kw = dict(H0=70.0, k=0.0)

    dr = cosmo.deflection_ratio(z, Om0=Om0, w0=w0, **kw)
    assert np.shape(dr) == (N,), f"expected (N,), got {np.shape(dr)} — outer-product regression"
    ref = np.array([float(np.asarray(cosmo.deflection_ratio(
        z[i], Om0=Om0[i], w0=w0[i], **kw)).ravel()[0]) for i in range(N)])
    np.testing.assert_allclose(np.asarray(dr), ref, rtol=1e-10,
                               err_msg="batched deflection_ratio != per-sample scalar computation")

    # raw distances share the same element-wise contract
    for fn in ("angular_distance", "_transverse_comoving_distance"):
        v = getattr(cosmo, fn)(z, Om0=Om0, w0=w0, **kw)
        assert np.shape(v) == (N,), f"{fn} gave {np.shape(v)}"
    dz = cosmo.angular_distance_z1z2(jnp.full(N, 0.5), z, Om0=Om0, w0=w0, **kw)
    assert np.shape(dz) == (N,)

    # a FIXED redshift with batched cosmo (the common inference case) is unchanged: (N,)
    dr_fixed = cosmo.deflection_ratio(jnp.asarray(2.3), Om0=Om0, w0=w0, **kw)
    assert np.shape(np.squeeze(np.asarray(dr_fixed))) == (N,)

    # still differentiable in a sampled cosmo param
    g = jax.grad(lambda o: jnp.sum(cosmo.deflection_ratio(z, Om0=o, w0=w0, **kw)))(Om0)
    assert np.all(np.isfinite(np.asarray(g)))
