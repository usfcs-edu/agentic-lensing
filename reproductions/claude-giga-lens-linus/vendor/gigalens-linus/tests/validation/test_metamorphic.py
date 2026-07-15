"""Oracle-free invariants + randomized differential sweeps (the unforeseen-error net).

These catch bugs lenstronomy parity can miss: a relation that *must* hold for any
correct simulator, asserted over seeded random valid parameter draws rather than one
hand-picked point. Where an invariant holds in exact arithmetic, it is held at the
float64 floor (REDUCTION_/LINEARITY_ tolerances), not a physical band.

The equivariance/reduction invariants target the new unified simulator and are
skipped until Phase 3; amplitude linearity is runnable against the current simulator
now and is included as the template.
"""
import numpy as np
import pytest

from . import tolerances as tol

pytestmark = pytest.mark.metamorphic


@pytest.mark.pending
@pytest.mark.skip(reason="new unified simulator — Phase 3")
def test_translation_equivariance():
    """Shifting every center (lens + source) by an integer-pixel (dx,dy) shifts the
    model image by exactly (dx,dy) pixels (no PSF, supersample=1).

    Invariant (no oracle). Falsifier: residual between shifted-image and
    image-of-shifted-model above REDUCTION floor → a coordinate/origin bug (R8).
    """
    raise NotImplementedError


@pytest.mark.pending
@pytest.mark.skip(reason="new unified simulator — Phase 3")
def test_rotation_equivariance():
    """Rotating all positions and ellipticity phases by θ rotates the (no-PSF) image
    by θ. Falsifier: above REDUCTION floor → an angle-convention bug."""
    raise NotImplementedError


@pytest.mark.pending
@pytest.mark.skip(reason="new unified simulator — Phase 3")
def test_zero_mass_plane_is_noop():
    """Adding a plane with zero mass (and no light) leaves every downstream β and the
    image unchanged. Falsifier: any change → a distance-matrix indexing bug."""
    raise NotImplementedError


@pytest.mark.pending
@pytest.mark.skip(reason="new unified simulator — Phase 3")
def test_component_and_plane_reordering_invariance():
    """Reordering light components within a plane, and reordering planes of equal
    redshift, does not change the summed image. Falsifier: any change → an
    order-dependent accumulation bug."""
    raise NotImplementedError


@pytest.mark.pending
@pytest.mark.skip(reason="new unified simulator — Phase 3")
def test_multiplane_N1_reduces_to_single_plane():
    """Mirror of test_multiplane::test_single_lens_multiplane_equals_singleplane, as a
    metamorphic relation over a random parameter sweep rather than one point."""
    raise NotImplementedError


def test_amplitude_linearity_current_simulator(rng):
    """Light is linear in its amplitude: scaling a (non-lstsq) light amplitude by c
    scales that component's image contribution by exactly c.

    Runnable against the current simulator as the live template. Claim type: exact
    identity → LINEARITY_RTOL. Falsifier: ratio deviates from c beyond the float64
    floor → a non-linearity / state-leak in light evaluation.
    """
    pytest.importorskip("lenstronomy")
    from gigalens.jax.profiles.light.sersic import SersicEllipse

    se = SersicEllipse(use_lstsq=False)
    p = dict(R_sersic=1.0, n_sersic=2.5, e1=0.05, e2=-0.05, center_x=0.0, center_y=0.0)
    x = rng.uniform(-3, 3, size=(20, 20, 1)).astype(np.float64)
    y = rng.uniform(-3, 3, size=(20, 20, 1)).astype(np.float64)

    base = np.asarray(se.light(x=x, y=y, Ie=1.0, **p))
    for c in (2.0, 0.5, 7.3):
        scaled = np.asarray(se.light(x=x, y=y, Ie=c, **p))
        assert np.allclose(scaled, c * base, rtol=tol.LINEARITY_RTOL, atol=1e-12), \
            f"Sersic light not linear in Ie at c={c}"
