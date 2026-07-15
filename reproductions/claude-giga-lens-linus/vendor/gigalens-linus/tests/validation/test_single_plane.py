"""Single-plane forward-simulation validation.

Layered so a failure localizes (method-discipline §5): the pre-convolution image is
held tightly (pure profile + deflection composition), the full PSF image loosely
(the subgrid-kernel pipeline is the dominant non-physics discrepancy), and the lstsq
solver is checked by a noiseless amplitude round-trip rather than against an oracle.

Most tests target the new unified simulator (Phase 3); the lstsq round-trip can be
adapted to run against the current ``BackwardProbModel`` path during Phase 0 as an
extra anchor.
"""
import numpy as np
import pytest

from . import tolerances as tol

pytestmark = pytest.mark.physics


@pytest.mark.pending
@pytest.mark.skip(reason="new unified simulator — Phase 3")
def test_preconvolution_image_vs_lenstronomy_f64():
    """No-PSF, supersample=1 image (EPL+Shear lens, Sersic source) vs a hand-built
    lenstronomy render (β = θ − α; Sersic at β).

    Claim type: deterministic identity → IMAGE_NOPSF_F64_RTOL. This is the tight,
    informative image test; the full-PSF test below is only a backstop.
    """
    raise NotImplementedError


@pytest.mark.slow
@pytest.mark.pending
@pytest.mark.skip(reason="new unified simulator — Phase 3")
def test_full_image_vs_lenstronomy():
    """Full PSF + supersample image vs lenstronomy ImageModel (R6).

    Held at IMAGE_RTOL with an absolute floor of IMAGE_ATOL_PEAK_FRAC × peak — loose
    because lenstronomy's iterative subgrid_kernel + conv edge handling differ from
    ours. Per §5, look at the residual map; a structured residual (e.g. ringing at
    bright arcs) is a finding even if the aggregate passes.
    """
    raise NotImplementedError


@pytest.mark.pending
@pytest.mark.skip(reason="new unified simulator — Phase 3")
def test_supersample_converges():
    """Image converges as supersample → ∞ (Richardson-style): the supersample=2→4→8
    sequence should converge, and high-supersample should match a direct fine-grid
    render. Falsifier: non-convergence or a fixed offset → a pooling/normalization bug.
    """
    raise NotImplementedError


@pytest.mark.pending
@pytest.mark.skip(reason="new unified simulator — Phase 3")
def test_precision_modes_agree():
    """R7: float32 and float64 model images agree to the float32 noise floor, and the
    float64 path is bit-stable across runs. Guards the precision plumbing the rework
    must preserve verbatim.
    """
    raise NotImplementedError


@pytest.mark.pending
@pytest.mark.skip(reason="lstsq for new simulator — Phase 3 (adapt to current path in Phase 0)")
def test_lstsq_recovers_known_amplitudes():
    """R9: render a noiseless image from known light amplitudes, then solve for them.

    Claim type: deterministic round-trip → recovered amplitudes match the planted ones
    to the solver's conditioning floor. This is the oracle-free correctness check on
    ``_solve_normal_eq_with_fallback`` (no lenstronomy equivalent). Falsifier:
    recovered ≠ planted beyond the conditioning-derived tolerance → a design-matrix /
    weighting / masking bug in the lstsq path.
    """
    raise NotImplementedError
