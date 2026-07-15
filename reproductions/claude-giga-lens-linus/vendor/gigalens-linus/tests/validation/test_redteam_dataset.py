"""Red-team (adversarial) ImageData/noise tests — see ``misuse_register.md`` (M3).

Contract: a non-expert who hands the ``ImageData`` a poisoned noise map or image must get
a loud ``ValueError`` at construction, not a silent all-NaN likelihood downstream. The
guard is scoped to the UNMASKED region (the pixels that enter the Gaussian likelihood).

The "allowed" tests are not filler: they pin the guard's *scope* so a later tightening
cannot silently start rejecting valid data — negative observed pixels (background
subtraction / reduction) and NaN padding in masked-out pixels are both legitimate.
"""
import numpy as np
import pytest

pytestmark = pytest.mark.redteam


def _cfg():
    from gigalens.simulator import SimulatorConfig
    return SimulatorConfig(delta_pix=0.05, num_pix=8, supersample=1, kernel=None,
                           likelihood_precision="float64")


def _dataset(image=None, error_map=None, **kw):
    from gigalens.jax.scene_prob_model import ImageData
    if image is None:
        image = np.zeros((8, 8))
    if error_map is None:
        error_map = np.ones((8, 8))
    return ImageData(image, _cfg(), error_map=error_map, **kw)


# --- M3: poisoned inputs must RAISE -----------------------------------------------
def test_nan_error_map_raises():
    """Claim: contract. A NaN in the unmasked error_map raises (the real Pitfall-3
    incident). Falsifier: construction succeeds, deferring an all-NaN likelihood."""
    err = np.ones((8, 8)); err[3, 3] = np.nan
    with pytest.raises(ValueError, match=r"error_map.*non-finite"):
        _dataset(error_map=err)


def test_inf_error_map_raises():
    """Claim: contract. An inf sigma in the unmasked region raises (an inf sigma is not
    a valid likelihood weight here; mask the pixel out instead)."""
    err = np.ones((8, 8)); err[0, 0] = np.inf
    with pytest.raises(ValueError, match=r"error_map.*non-finite"):
        _dataset(error_map=err)


def test_zero_or_negative_sigma_raises():
    """Claim: contract. A zero or negative noise sigma raises — a 0/neg sigma gives an
    infinite/ill-defined Gaussian likelihood. Falsifier: either is accepted."""
    err0 = np.ones((8, 8)); err0[2, 2] = 0.0
    with pytest.raises(ValueError, match=r"error_map.*non-positive"):
        _dataset(error_map=err0)
    errneg = np.ones((8, 8)); errneg[2, 2] = -1.0
    with pytest.raises(ValueError, match=r"error_map.*non-positive"):
        _dataset(error_map=errneg)


def test_nan_image_raises():
    """Claim: contract. A NaN in the unmasked observed image raises (it propagates into
    the residual and poisons the likelihood just as a NaN sigma does)."""
    img = np.zeros((8, 8)); img[5, 5] = np.nan
    with pytest.raises(ValueError, match=r"image.*non-finite"):
        _dataset(image=img)


def test_derived_noise_with_bad_exptime_raises():
    """Claim: contract. The derived-noise path (background_rms + exp_time) is guarded
    too: a non-positive exp_time yields a non-finite/zero sigma, which must raise rather
    than silently produce NaN sigmas."""
    from gigalens.jax.scene_prob_model import ImageData
    img = np.ones((8, 8))
    with pytest.raises(ValueError):
        ImageData(img, _cfg(), background_rms=1.0, exp_time=-10.0)


# --- M3 scope: valid-but-unusual inputs must NOT raise ----------------------------
def test_negative_image_pixels_allowed():
    """Claim: scope. Negative observed pixels are valid (background subtraction /
    reduction). The guard checks image FINITENESS, not sign. Falsifier: a negative-pixel
    image with a valid error_map raises ⇒ the guard over-rejects real data."""
    img = np.full((8, 8), -0.5)
    _dataset(image=img)  # must not raise


def test_nonfinite_in_masked_region_allowed():
    """Claim: scope. NaN/inf in MASKED-OUT pixels is legitimate (they never enter the
    likelihood). Falsifier: masked-out NaN padding raises ⇒ the guard ignores the mask."""
    err = np.ones((8, 8)); err[0, 0] = np.nan      # poisoned pixel...
    mask = np.ones((8, 8), dtype=bool); mask[0, 0] = False  # ...but masked out
    _dataset(error_map=err, mask=mask)  # must not raise


# --- §3.8: shape agreement must RAISE (broadcastable-but-wrong is the trap) --------
def test_error_map_shape_mismatch_raises():
    """Claim: contract. A wrong-shaped error_map raises ONE actionable ValueError at
    construction. Verified pre-guard behavior (2026-07-10, the kernel is the oracle):
    an (8, 8, 1) map was accepted SILENTLY and the likelihood residual broadcast to
    (8, 8, 8) — a 512-element pseudo-likelihood; an (8,) map crashed with a cryptic
    IndexError inside the finite-noise check (loud but unactionable). Falsifier:
    either shape constructs, or raises anything but the §3.8 message."""
    err_singleton = np.ones((8, 8, 1))         # the verified SILENT case
    with pytest.raises(ValueError, match=r"error_map shape.*!= image shape"):
        _dataset(error_map=err_singleton)
    err_row = np.ones(8)                       # the verified cryptic-IndexError case
    with pytest.raises(ValueError, match=r"error_map shape.*!= image shape"):
        _dataset(error_map=err_row)


def test_mask_shape_mismatch_raises():
    """Claim: contract. A wrong-shaped boolean mask raises. The trap is real and
    verified (2026-07-10, pre-guard kernel): an (8,) boolean mask was ACCEPTED
    silently with event_size=8 — it indexes the FIRST AXIS of the (8, 8) image (whole
    rows), so the finite-noise guard and the likelihood run over the wrong pixel set
    with no exception anywhere. Falsifier: construction succeeds."""
    mask_row = np.ones(8, dtype=bool)
    with pytest.raises(ValueError, match=r"mask shape.*!= image shape"):
        _dataset(mask=mask_row)


def test_derived_noise_shape_mismatch_raises():
    """Claim: contract. The derived-noise path is guarded too: a per-ROW background_rms
    (broadcastable, wrong) raises; a scalar or full-shape background_rms is fine."""
    from gigalens.jax.scene_prob_model import ImageData
    img = np.ones((8, 8))
    with pytest.raises(ValueError, match=r"background_rms.*scalar or match"):
        ImageData(img, _cfg(), background_rms=np.ones(8), exp_time=100.0)
    ImageData(img, _cfg(), background_rms=1.0, exp_time=100.0)            # scalar ok
    ImageData(img, _cfg(), background_rms=np.ones((8, 8)), exp_time=100.0)  # full ok


# --- §3.9: an all-masked dataset must RAISE (it silently contributes nothing) ------
def test_fully_masked_dataset_raises():
    """Claim: contract. A mask excluding EVERY pixel raises. Verified (2026-07-10,
    pre-guard kernel): an all-False mask was ACCEPTED silently with event_size=0 —
    the dataset's chi2/ll terms are identically zero among other live datasets, so the
    model silently ignores an observation the user believes is constraining it (the
    all-datasets-masked case only raises later, at ProbModel, via the chi2-channel
    guard; a single dead dataset among live ones raised nowhere). Falsifier:
    construction succeeds."""
    with pytest.raises(ValueError, match=r"excludes every pixel"):
        _dataset(mask=np.zeros((8, 8), dtype=bool))


def test_one_masked_pixel_is_enough():
    """Claim: scope. The §3.9 guard is exactly event_size == 0 — a single unmasked pixel
    constructs fine. Falsifier: the guard over-rejects heavily-masked real data."""
    mask = np.zeros((8, 8), dtype=bool); mask[4, 4] = True
    _dataset(mask=mask)  # must not raise
