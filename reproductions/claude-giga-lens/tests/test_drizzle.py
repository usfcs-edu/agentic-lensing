"""CPU tests for cgl.noise: drizzle-operator enumeration, the t(1) closed
form, the 2x2 block-sum covariance transform, the masked-ACF port, and the
PSD projection. All pure numpy (conftest forces CPU+x64)."""
import numpy as np
import pytest

from cgl.noise import (
    binned_kernel_from_fine,
    drizzle_acf,
    drizzle_overlap_matrix_1d,
    fit_kernel,
    masked_acf_2d,
    masked_autocorr_full,
    psd_project,
    rho_model,
)


# --------------------------------------------------------------------- drizzle
def test_overlap_matrix_flux_conservation():
    """Row sums = output pixel size s (full coverage); interior column sums =
    pixfrac (each drop fully deposited)."""
    for r, p, phase in [(3.2075, 1.0, 0.0), (3.2075, 1.0, 0.37),
                        (3.0, 1.0, 0.5), (0.98692, 1.0, 0.11),
                        (2.5, 0.8, 0.2)]:
        A, _ = drizzle_overlap_matrix_1d(r, p, 60, phase)
        if p == 1.0:
            # drops tile the line -> every output pixel fully covered
            np.testing.assert_allclose(A.sum(axis=1), 1.0 / r,
                                       rtol=0, atol=1e-12)
        else:
            # pixfrac < 1 leaves gaps: coverage varies but never exceeds s
            assert np.all(A.sum(axis=1) <= 1.0 / r + 1e-12)
        # flux conservation: each interior drop deposits its full width p
        interior = A.sum(axis=0)[4:-4]
        np.testing.assert_allclose(interior, p, rtol=0, atol=1e-12)


def test_t1_closed_form_r3_and_r321():
    """Phase-averaged t(1) = (r-1)/(r-1/3) at r=3 and r=3.21 (valid r>=2)."""
    for r in (3.0, 3.21):
        d = drizzle_acf(r, pixfrac=1.0, n_frames=3, offsets=None,
                        max_lag=5, n_phase=64)
        assert abs(d["t1"] - d["t1_analytic"]) < 1e-3, (r, d["t1"])
        assert abs(d["t1_analytic"] - (r - 1.0) / (r - 1.0 / 3.0)) < 1e-15
    # spot value from the campaign design note
    d = drizzle_acf(3.21, 1.0, 3, None, max_lag=3, n_phase=64)
    assert abs(d["t1_analytic"] - 0.768) < 5e-4


def test_drizzle_acf_is_normalized_and_symmetric():
    d = drizzle_acf(3.2075, 1.0, 3, None, max_lag=6, n_phase=16)
    rho = d["rho"]
    L = 6
    assert rho.shape == (13, 13)
    assert rho[L, L] == 1.0
    np.testing.assert_allclose(rho, rho[::-1, ::-1], atol=1e-14)
    np.testing.assert_allclose(rho, rho.T, atol=1e-14)
    # PSD-ish: periodic spectrum of the enumerated ACF must be >= -tiny
    emb = np.zeros((64, 64))
    for dy in range(-L, L + 1):
        for dx in range(-L, L + 1):
            emb[dy % 64, dx % 64] = rho[L + dy, L + dx]
    S = np.real(np.fft.fft2(emb))
    assert S.min() > -5e-3


def test_drizzle_integer_r3_offsets_matches_exact_average():
    """r=3, pixfrac=1, integer fine offsets: the enumerated stacked kernel
    must equal the exact 3x3 block-average tent structure of the mock design
    (support |lag|_inf <= 2)."""
    offs = np.array([[0, 0], [1, 2], [2, 1]]) / 3.0
    d = drizzle_acf(3.0, 1.0, 3, offs, max_lag=4)
    rho = d["rho"]
    L = 4
    # support exactly |lag| <= 2
    assert np.abs(rho[L, L + 3:]).max() < 1e-12
    assert np.abs(rho[L + 3:, :]).max() < 1e-12
    # 1-D axis cut is the tent (3-|d|)/3 for these Latin-square offsets
    np.testing.assert_allclose(rho[L, L:L + 3], [1.0, 2.0 / 3.0, 1.0 / 3.0],
                               atol=1e-12)


# ------------------------------------------------------------------ block-sum
def test_binned_kernel_from_fine_toy():
    """Direct MC-free check: build a small stationary field covariance
    densely, block-sum it, compare against the transform."""
    rng = np.random.default_rng(3)
    # toy fine kernel: PSD by construction (autocorrelation of a random taps)
    g = rng.standard_normal((3, 3))
    cov_full = np.zeros((9, 9))
    from scipy.signal import correlate2d

    acf = correlate2d(g, g, mode="full")          # 5x5 autocorrelation, PSD
    Lf = 2
    n = 12                                         # fine grid 12x12
    # dense stationary covariance on the fine grid
    C = np.zeros((n * n, n * n))
    for i in range(n * n):
        yi, xi = divmod(i, n)
        for j in range(n * n):
            yj, xj = divmod(j, n)
            dy, dx = yi - yj, xi - xj
            if abs(dy) <= Lf and abs(dx) <= Lf:
                C[i, j] = acf[Lf + dy, Lf + dx]
    # block-sum operator B: (n/2)^2 x n^2
    m = n // 2
    B = np.zeros((m * m, n * n))
    for bi in range(m * m):
        by, bx = divmod(bi, m)
        for oy in (0, 1):
            for ox in (0, 1):
                B[bi, (2 * by + oy) * n + (2 * bx + ox)] = 1.0
    Cb = B @ C @ B.T
    # interior binned lags vs the transform
    got = binned_kernel_from_fine(acf)
    Lb = (Lf - 1) // 2   # = 0 ... too small; extend acf to 9x9 with zeros
    acf9 = np.zeros((9, 9))
    acf9[2:7, 2:7] = acf
    got = binned_kernel_from_fine(acf9)            # Lb = 1
    ctr = (m // 2) * m + (m // 2)                  # interior binned pixel
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            j = (m // 2 + dy) * m + (m // 2 + dx)
            np.testing.assert_allclose(got[1 + dy, 1 + dx], Cb[ctr, j],
                                       rtol=0, atol=1e-12)


def test_binned_variance_identity():
    """Cov_b(0) = sum over the 4x4 intra-block lag table of Cov_f."""
    rng = np.random.default_rng(11)
    g = rng.standard_normal((3, 3))
    from scipy.signal import correlate2d

    acf = correlate2d(g, g, mode="full")
    acf9 = np.zeros((9, 9))
    acf9[2:7, 2:7] = acf
    got = binned_kernel_from_fine(acf9)
    expect = 0.0
    for oy in (0, 1):
        for ox in (0, 1):
            for oy2 in (0, 1):
                for ox2 in (0, 1):
                    expect += acf9[4 + oy - oy2, 4 + ox - ox2]
    assert abs(got[1, 1] - expect) < 1e-12


# ------------------------------------------------------------------ maskedACF
def test_masked_acf_recovers_known_kernel():
    """ACF of a synthetic correlated field (known kernel, ~40% masked)
    recovers the truth to a few 1e-2."""
    rng = np.random.default_rng(0)
    n = 256
    white = rng.standard_normal((n, n))
    k = np.outer([0.25, 1.0, 0.25], [0.25, 1.0, 0.25])
    from scipy.signal import fftconvolve

    field = fftconvolve(white, k, mode="same")
    mask = rng.random((n, n)) > 0.4
    rho, counts = masked_acf_2d(field, mask, max_lag=3)
    from scipy.signal import correlate2d

    truth = correlate2d(k, k, mode="full")         # 5x5
    truth = truth / truth[2, 2]
    got_win = rho[1:6, 1:6]                        # +/-2 window of the L=3 out
    np.testing.assert_allclose(got_win, truth, atol=0.04)
    assert counts[3, 3] == pytest.approx(mask.sum(), rel=1e-12)


def test_masked_acf_matches_foundry_port():
    """Bit-level agreement with foundry-i 46_noise_audit.masked_autocorr on
    identical inputs (the port contract)."""
    import importlib.util
    from cgl.paths import FOUNDRY_I

    spec = importlib.util.spec_from_file_location(
        "fi_noise_audit", FOUNDRY_I / "46_noise_audit.py")
    mod = importlib.util.module_from_spec(spec)
    import sys

    sys.path.insert(0, str(FOUNDRY_I))             # for its `import _data_lib`
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.path.remove(str(FOUNDRY_I))
    rng = np.random.default_rng(42)
    img = rng.standard_normal((64, 64))
    mask = rng.random((64, 64)) > 0.3
    rho_fi, center_fi = mod.masked_autocorr(img, mask)
    rho_us, center_us, _ = masked_autocorr_full(img, mask)
    assert center_fi == center_us
    np.testing.assert_array_equal(np.isnan(rho_fi), np.isnan(rho_us))
    np.testing.assert_allclose(np.nan_to_num(rho_us), np.nan_to_num(rho_fi),
                               rtol=0, atol=0)


# ------------------------------------------------------------ mock pipeline
def test_mock_pipeline_matches_exact_average_and_units():
    """DrizzleMockPipeline (built from the noise.py 1-D overlap enumeration)
    must equal the exact 3x3-block-average construction at r=3 integer, and
    conserve surface brightness on a uniform scene."""
    from cgl.mocks import (CROP0, FINE_RENDER, N_FINE, N_NATIVE,
                           OFFSETS_FINE, DrizzleMockPipeline)

    pipe = DrizzleMockPipeline()
    rng = np.random.default_rng(5)
    scene = rng.random((FINE_RENDER, FINE_RENDER))

    out = np.zeros((N_FINE, N_FINE))
    for (oy, ox) in OFFSETS_FINE:
        sub = scene[oy:oy + 3 * N_NATIVE, ox:ox + 3 * N_NATIVE]
        nat = sub.reshape(N_NATIVE, 3, N_NATIVE, 3).sum(axis=(1, 3))
        up = np.repeat(np.repeat(nat, 3, 0), 3, 1)
        out += up[CROP0 - oy:CROP0 - oy + N_FINE,
                  CROP0 - ox:CROP0 - ox + N_FINE]
    ref = out / (len(OFFSETS_FINE) * 9)
    np.testing.assert_allclose(pipe(scene), ref, rtol=0, atol=1e-13)

    # full-phase dither: the stack must be EXACTLY a convolution — the
    # separable 3x3 tent — so an effective PSF exists (mock design contract)
    from scipy.signal import fftconvolve

    tent = np.outer([1, 2, 3, 2, 1], [1, 2, 3, 2, 1]) / 81.0   # sum = 1
    conv = fftconvolve(scene, tent, mode="same")[
        CROP0:CROP0 + N_FINE, CROP0:CROP0 + N_FINE]
    interior = (slice(4, -4), slice(4, -4))
    np.testing.assert_allclose(pipe(scene)[interior], conv[interior],
                               rtol=0, atol=1e-12)

    u = pipe(np.ones((FINE_RENDER, FINE_RENDER)))
    np.testing.assert_allclose(u, 1.0, rtol=0, atol=1e-13)


def test_mock_pipeline_variance_and_kernel():
    """Analytic per-pixel variance and stationary kernel of the mock stack
    agree with Monte Carlo."""
    from cgl.mocks import N_NATIVE, DrizzleMockPipeline

    pipe = DrizzleMockPipeline()
    n_f = len(pipe.offsets)
    rng = np.random.default_rng(1)
    errs = [np.full((N_NATIVE, N_NATIVE), 0.3) for _ in range(n_f)]
    var = pipe.fine_var(errs)
    assert var.std() < 1e-15                       # constant sigma -> constant
    draws = np.array([pipe.drizzle(
        [rng.normal(0, 0.3, (N_NATIVE, N_NATIVE)) for _ in range(n_f)])
        for _ in range(300)])
    assert abs(draws.var(axis=0).mean() / var.mean() - 1.0) < 0.05
    rho = pipe.fine_rho(4)
    c01 = np.mean(draws[:, :, :-1] * draws[:, :, 1:]) / draws.var()
    assert abs(c01 - rho[4, 5]) < 0.02
    np.testing.assert_allclose(rho[4, 5], 2.0 / 3.0, atol=1e-12)


# ------------------------------------------------------------ model + psd
def test_rho_model_delta_limit_and_center():
    d = drizzle_acf(3.2075, 1.0, 3, None, max_lag=8, n_phase=8)
    m = rho_model(0.0, 1e-9, d["rho"], 4)
    expect = np.zeros((9, 9))
    expect[4, 4] = 1.0
    np.testing.assert_allclose(m, expect, atol=1e-14)
    m = rho_model(0.7, 0.9, d["rho"], 4)
    assert abs(m[4, 4] - 1.0) < 1e-12               # center always 1
    assert m[4, 5] > 0.0


def test_fit_kernel_recovers_synthetic_hyperparams():
    d = drizzle_acf(3.2075, 1.0, 3, None, max_lag=14, n_phase=8)
    truth_w, truth_s = 0.85, 0.7
    rho_true = rho_model(truth_w, truth_s, d["rho"], 6)
    counts = np.full_like(rho_true, 1e4)
    fit = fit_kernel(rho_true, counts, d["rho"], max_lag=6)
    assert abs(fit["w"] - truth_w) < 1e-6
    assert abs(fit["sigma_e"] - truth_s) < 1e-5
    assert fit["max_abs_resid"] < 1e-8
    assert fit["gate_le_0p05"]


def test_psd_project_output_is_psd_and_close():
    d = drizzle_acf(3.2075, 1.0, 3, None, max_lag=10, n_phase=8)
    rho = rho_model(0.9, 0.8, d["rho"], 6)
    noisy = rho + np.random.default_rng(1).normal(0, 0.01, rho.shape)
    noisy = 0.5 * (noisy + noisy[::-1, ::-1])
    noisy[6, 6] = 1.0
    proj, diag = psd_project(noisy, support=6, s_floor=0.0, grid=128,
                             n_iter=40)
    emb = np.zeros((128, 128))
    for dy in range(-6, 7):
        for dx in range(-6, 7):
            emb[dy % 128, dx % 128] = proj[6 + dy, 6 + dx]
    S = np.real(np.fft.fft2(emb))
    # POCS converges slowly; residual negativity must be tiny vs S.max()
    # (the whitener construction floors the spectrum anyway)
    assert S.min() > -5e-3
    assert S.min() > diag["s_min_input"]            # strictly improved
    assert np.max(np.abs(proj - noisy)) < 0.05      # stayed close to input
