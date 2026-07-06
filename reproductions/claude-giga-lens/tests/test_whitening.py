"""CPU tests for cgl.whiten.build_whitener / erode_keep (P1a) plus the
delta-kernel regression that protects parity gate D. Small toys only
(conftest forces CPU+x64)."""
import numpy as np
import pytest

from cgl.whiten import (
    _embed_kernel,
    _spectrum_of_taps,
    build_whitener,
    erode_keep,
    make_conv_whitener,
)


def _toy_rho():
    """Mild 5x5 correlation kernel: normalized autocorrelation of a 3x3
    smoothing tap (PSD by construction)."""
    from scipy.signal import correlate2d

    g = np.outer([0.22, 1.0, 0.22], [0.22, 1.0, 0.22])
    acf = correlate2d(g, g, mode="full")
    return acf / acf[2, 2]


def test_build_whitener_toy_reaches_gate():
    rho = _toy_rho()
    w = build_whitener(rho, M=6, s_floor=0.05, grid=256)
    assert w["e_op"] <= 0.02, w["e_op"]
    assert w["e_op"] <= w["e_op_init"] + 1e-15
    # symmetric taps
    np.testing.assert_allclose(w["h"], w["h"][::-1, ::-1], atol=1e-14)
    # logdet is the mean log spectrum of the (floored) kernel
    S = np.maximum(np.real(np.fft.fft2(_embed_kernel(rho, 256))), w["floor"])
    assert abs(w["logdet_per_pix"] - np.mean(np.log(S))) < 1e-12


def test_build_whitener_whitens_white_noise_mc():
    """Whitened synthetic correlated noise on 32x32 crops: Var ~ 1 and
    neighbor correlations ~ 0 (MC over many draws)."""
    from scipy.ndimage import correlate

    rho = _toy_rho()
    w = build_whitener(rho, M=6, s_floor=0.05, grid=256)
    grid = 256
    S = np.maximum(np.real(np.fft.fft2(_embed_kernel(rho, grid))), w["floor"])
    rng = np.random.default_rng(7)
    us = []
    for _ in range(40):
        wn = rng.standard_normal((grid, grid))
        x = np.real(np.fft.ifft2(np.sqrt(S) * np.fft.fft2(wn)))
        u = correlate(x, w["h"], mode="wrap")
        # 32x32 crops away from any structure
        us.append(u[100:132, 100:132])
    us = np.asarray(us)
    var = us.var()
    assert abs(var - 1.0) < 0.03
    c01 = np.mean(us[:, :, :-1] * us[:, :, 1:]) / var
    c10 = np.mean(us[:, :-1, :] * us[:, 1:, :]) / var
    c11 = np.mean(us[:, :-1, :-1] * us[:, 1:, 1:]) / var
    for c in (c01, c10, c11):
        assert abs(c) < 0.02, (c01, c10, c11)


def test_e_op_definition_matches_spectrum():
    rho = _toy_rho()
    w = build_whitener(rho, M=5, s_floor=0.05, grid=256)
    S = np.maximum(np.real(np.fft.fft2(_embed_kernel(rho, 256))), w["floor"])
    H = _spectrum_of_taps(w["h"], 256)
    assert abs(w["e_op"] - np.max(np.abs(S * H * H - 1.0))) < 1e-14


def test_erode_keep_border_and_interior():
    keep = np.ones((16, 16), dtype=bool)
    keep[8, 8] = False
    er = erode_keep(keep, M=2)
    # border margin M gone
    assert not er[:2, :].any() and not er[:, -2:].any()
    # Chebyshev-2 neighborhood of the interior hole gone
    assert not er[6:11, 6:11].any()
    # a safe pixel survives
    assert er[3, 3]
    # M=0 is the identity
    np.testing.assert_array_equal(erode_keep(keep, 0), keep)


def test_erode_keep_matches_conv_support():
    """Every surviving whitened pixel must be computable from kept, in-grid
    pixels only: whitening an image that is NaN outside keep must stay finite
    on the eroded mask."""
    rng = np.random.default_rng(3)
    keep = rng.random((24, 24)) > 0.15
    M = 3
    er = erode_keep(keep, M)
    img = rng.standard_normal((24, 24))
    img[~keep] = np.nan
    from scipy.ndimage import correlate

    h = rng.standard_normal((2 * M + 1, 2 * M + 1))
    u = correlate(np.nan_to_num(img, nan=np.nan), h, mode="constant", cval=0.0)
    assert np.isfinite(u[er]).all()
    # and the complement inside the interior does touch NaN/border influence
    # (sanity that the erosion is not overly generous)
    assert er.sum() < keep.sum()


def test_delta_kernel_regression_gate_d():
    """Gate D protection: build_whitener on the identity kernel returns the
    delta tap, and make_conv_whitener with it reproduces the diagonal path."""
    import jax.numpy as jnp

    rho = np.zeros((5, 5))
    rho[2, 2] = 1.0
    w = build_whitener(rho, M=1, s_floor=0.05, grid=64)
    assert w["e_op"] < 1e-12
    np.testing.assert_allclose(w["h"][1, 1], 1.0, atol=1e-12)
    assert np.abs(w["h"]).sum() == pytest.approx(1.0, abs=1e-12)

    rng = np.random.default_rng(0)
    err = rng.uniform(0.5, 1.5, (16, 16))
    keep = rng.random((16, 16)) > 0.2
    masked_err = np.where(keep, err, 1e10)
    sqrtW = 1.0 / masked_err
    keep_w = erode_keep(keep, 1)
    wf = make_conv_whitener(w["h"], sqrtW, keep_w)
    img = rng.standard_normal((16, 16)) * 3.0
    u = np.asarray(wf(jnp.asarray(img)))
    ref = ((img * sqrtW) * keep_w).reshape(-1)
    np.testing.assert_allclose(u, ref, rtol=0, atol=1e-12)


def test_build_whitener_rejects_bad_grid():
    rho = _toy_rho()
    with pytest.raises(ValueError):
        build_whitener(rho, M=40, s_floor=0.05, grid=64)


def test_exact_ref_conv_matrix_matches_scipy():
    """cgl.exact_ref.conv_matrix must reproduce the SAME zero-padded
    cross-correlation semantics (the independent CPU rep of the whitener)."""
    from scipy.ndimage import correlate

    from cgl.exact_ref import conv_matrix

    rng = np.random.default_rng(2)
    h = rng.standard_normal((5, 3))                 # asymmetric on purpose
    img = rng.standard_normal((12, 14))
    A = conv_matrix(h, img.shape)
    got = (A @ img.reshape(-1)).reshape(img.shape)
    ref = correlate(img, h, mode="constant", cval=0.0)
    np.testing.assert_allclose(got, ref, rtol=0, atol=1e-13)


def test_exact_ref_dense_whitened_marg_matches_marg_loglik():
    """dense_whitened_marg (scipy) equals cgl.marg.marg_loglik (jax) on the
    same whitened inputs — the 04 hard-gate functional, in miniature."""
    import jax.numpy as jnp

    from cgl.exact_ref import dense_whitened_marg, whitening_operator
    from cgl.marg import marg_loglik

    rng = np.random.default_rng(4)
    H = W = 12
    err = rng.uniform(0.5, 1.5, (H, W))
    keep = rng.random((H, W)) > 0.1
    masked_err = np.where(keep, err, 1e10)
    h = rng.standard_normal((3, 3))
    keep_w = erode_keep(keep, 1)
    G_e, _ = whitening_operator(h, 1.0 / masked_err, keep_w)

    Y = rng.standard_normal((H, W))
    M_det = rng.standard_normal((H, W))
    ret = rng.standard_normal((H, W, 4))
    lam = rng.uniform(0.5, 2.0, 4)

    ref = dense_whitened_marg(G_e, Y, M_det, ret, lam)

    wf = make_conv_whitener(h, 1.0 / masked_err, keep_w)
    Rw = np.asarray(wf(jnp.asarray(Y - M_det)))
    Xw = np.stack([np.asarray(wf(jnp.asarray(ret[:, :, k])))
                   for k in range(4)], axis=1)
    # dense path keeps only eroded rows; conv path zeroes non-eroded -> the
    # marg quantities are identical
    logL, a_star, logdetA = marg_loglik(jnp.asarray(Xw), jnp.asarray(Rw),
                                        jnp.asarray(lam))
    assert abs(float(logL) - ref["logL"]) < 1e-9
    np.testing.assert_allclose(np.asarray(a_star), ref["a_star"], atol=1e-10)
    assert abs(float(logdetA) - ref["logdetA"]) < 1e-10
