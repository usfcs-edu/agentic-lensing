"""CPU unit tests: cgl.marg.marg_loglik vs pure numpy, and the delta-kernel
conv whitener vs the diagonal path (16x16 toys; conftest forces CPU+x64)."""
import jax.numpy as jnp
import numpy as np

from cgl.marg import marg_loglik
from cgl.whiten import make_conv_whitener


def _numpy_reference(X, R, lam):
    b = X.T @ R
    A = X.T @ X + np.diag(lam)
    a_star = np.linalg.solve(A, b)
    sign, logdetA = np.linalg.slogdet(A)
    assert sign > 0
    logL = -0.5 * R @ R + 0.5 * b @ a_star - 0.5 * logdetA
    return b, A, a_star, logdetA, logL


def test_marg_loglik_matches_numpy():
    rng = np.random.default_rng(7)
    n_pix, k = 64, 7
    for trial in range(5):
        X = rng.standard_normal((n_pix, k))
        R = rng.standard_normal(n_pix) * (1.0 + trial)
        lam = (np.arange(k) + 1.0) / 25.0  # the foundry-i ridge (i+1)/25
        b_np, A_np, a_np, ld_np, logL_np = _numpy_reference(X, R, lam)

        logL, a_star, logdetA = marg_loglik(
            jnp.asarray(X), jnp.asarray(R), jnp.asarray(lam))

        # b and A as computed on the jax side (same expressions)
        b_j = np.asarray(jnp.asarray(X).T @ jnp.asarray(R))
        A_j = np.asarray(jnp.asarray(X).T @ jnp.asarray(X) + jnp.diag(jnp.asarray(lam)))
        np.testing.assert_allclose(b_j, b_np, rtol=0, atol=1e-12)
        np.testing.assert_allclose(A_j, A_np, rtol=0, atol=1e-12)

        np.testing.assert_allclose(np.asarray(a_star), a_np, rtol=0, atol=1e-12)
        assert abs(float(logdetA) - ld_np) <= 1e-12 * max(1.0, abs(ld_np))
        assert abs(float(logL) - logL_np) <= 1e-12 * max(1.0, abs(logL_np))
        # a* must satisfy the normal equations
        np.testing.assert_allclose(A_np @ np.asarray(a_star), b_np,
                                   rtol=0, atol=1e-10)


def test_marg_loglik_with_random_ridge():
    rng = np.random.default_rng(11)
    n_pix, k = 64, 7
    X = rng.standard_normal((n_pix, k))
    X[:, -1] = X[:, 0]  # exactly collinear columns: only the ridge saves A
    R = rng.standard_normal(n_pix)
    lam = rng.uniform(0.5, 2.0, k)
    _, _, a_np, ld_np, logL_np = _numpy_reference(X, R, lam)
    logL, a_star, logdetA = marg_loglik(
        jnp.asarray(X), jnp.asarray(R), jnp.asarray(lam))
    np.testing.assert_allclose(np.asarray(a_star), a_np, rtol=0, atol=1e-10)
    assert abs(float(logdetA) - ld_np) <= 1e-12 * max(1.0, abs(ld_np))
    assert abs(float(logL) - logL_np) <= 1e-12 * max(1.0, abs(logL_np))


def test_delta_kernel_whitener_equals_diagonal():
    """h=[[1.0]] + keep_w=keep_mask must reproduce the diagonal sqrt(W) path
    (CPU miniature of parity gate D)."""
    rng = np.random.default_rng(0)
    err = rng.uniform(0.5, 1.5, (16, 16))
    keep = rng.random((16, 16)) > 0.2
    masked_err = np.where(keep, err, 1e10)  # the err->1e10 mask convention
    sqrtW = 1.0 / masked_err  # likelihood convention: multiply by reciprocal
    wf = make_conv_whitener(np.ones((1, 1)), sqrtW, keep)
    for _ in range(3):
        img = rng.standard_normal((16, 16)) * 5.0
        u = np.asarray(wf(jnp.asarray(img)))
        # exact (bitwise) against the kept-pixel diagonal path
        ref_kept = ((img * sqrtW) * keep).reshape(-1)
        np.testing.assert_allclose(u, ref_kept, rtol=0, atol=0)
        # and equal to the full diagonal path up to the masked pixels' 1e-10
        # down-weighting (their whitened values are img*1e-10; the whitener
        # zeroes them exactly instead — a bounded, squared-away difference)
        ref_diag = (img * sqrtW).reshape(-1)
        assert np.max(np.abs(u - ref_diag)) <= np.max(np.abs(img)) * 1e-10


def test_conv_whitener_matches_scipy_correlate():
    """Lock the conv semantics: 'SAME' conv_general_dilated == zero-padded
    cross-correlation (scipy.ndimage.correlate, mode='constant')."""
    from scipy.ndimage import correlate

    rng = np.random.default_rng(1)
    h = rng.standard_normal((3, 3))  # deliberately asymmetric
    sqrt_d_inv = rng.uniform(0.5, 2.0, (16, 16))
    keep = np.ones((16, 16), dtype=bool)
    img = rng.standard_normal((16, 16))
    wf = make_conv_whitener(h, sqrt_d_inv, keep)
    u = np.asarray(wf(jnp.asarray(img))).reshape(16, 16)
    ref = correlate(img * sqrt_d_inv, h, mode="constant", cval=0.0)
    np.testing.assert_allclose(u, ref, rtol=0, atol=1e-12)


def test_conv_whitener_rejects_even_kernels():
    import pytest

    with pytest.raises(ValueError):
        make_conv_whitener(np.ones((2, 2)), np.ones((8, 8)), np.ones((8, 8)))
