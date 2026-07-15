"""Profile parity at float64 (R11).

The existing ``tests/test_profiles.py`` checks profiles in float32 at rtol=1e-5. The
framework is now float64-first, so analytic profiles must agree with lenstronomy near
machine precision; a float64 run that only reaches the float32 band would hide a
formula/normalization error inside the old loose tolerance. This module re-checks the
core profiles at PROFILE_F64_RTOL and is the template for extending coverage to the
profiles the old suite misses (NFW/tNFW/PIEMD/SIE-variants/shapelets in f64).

Runnable now against current profiles.
"""
import numpy as np
import pytest

from . import tolerances as tol

pytestmark = pytest.mark.physics


@pytest.mark.parametrize("params", [
    dict(theta_E=1.0, gamma=2.0, e1=0.0, e2=0.0, center_x=0.0, center_y=0.0),
    dict(theta_E=1.2, gamma=2.2, e1=-0.1, e2=0.1, center_x=0.05, center_y=-0.05),
    dict(theta_E=0.8, gamma=1.8, e1=0.2, e2=-0.15, center_x=0.0, center_y=0.0),
])
def test_epl_deriv_vs_lenstronomy_f64(rng, params):
    """EPL reduced deflection vs lenstronomy at float64.

    Claim type: deterministic identity → PROFILE_F64_RTOL (≈5 orders above float64
    eps; see tolerances.py). Falsifier: relative disagreement above the band away from
    the center (we sample radii in [0.2, 5] arcsec to avoid the central singularity).
    """
    pytest.importorskip("lenstronomy")
    from lenstronomy.LensModel.Profiles.epl import EPL as LEPL
    from gigalens.jax.profiles.mass.epl import EPL

    r = rng.uniform(0.2, 5.0, size=4000)
    phi = rng.uniform(0, 2 * np.pi, size=4000)
    x = (r * np.cos(phi)).astype(np.float64)
    y = (r * np.sin(phi)).astype(np.float64)

    gx, gy = EPL(100).deriv(x=x, y=y, **params)
    lx, ly = LEPL().derivatives(x=x, y=y, **params)
    assert np.allclose(np.asarray(gx), lx, rtol=tol.PROFILE_F64_RTOL, atol=tol.PROFILE_F64_ATOL)
    assert np.allclose(np.asarray(gy), ly, rtol=tol.PROFILE_F64_RTOL, atol=tol.PROFILE_F64_ATOL)


@pytest.mark.pending
@pytest.mark.skip(reason="extend coverage: NFW, tNFW, PIEMD, shapelets at f64")
def test_remaining_profiles_vs_lenstronomy_f64():
    """Placeholder enumerating the profiles still lacking float64 parity tests.

    Each gets a parametrized parity test like ``test_epl_deriv_vs_lenstronomy_f64``.
    Light profiles compare ``.light`` (with use_lstsq=False) to the lenstronomy
    LightModel equivalent; mass profiles compare deriv AND hessian (convergence/shear
    are derived from the hessian and must be checked too, not just deflection).
    """
    raise NotImplementedError
