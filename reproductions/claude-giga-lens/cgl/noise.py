"""Correlated-noise kernels for drizzled HST products (P1a).

Four layers, each validated against a reference before the next is built:

1. ``masked_acf_2d`` — masked, mask-deconvolved 2-D autocorrelation of a
   residual image. PORTED WITH ATTRIBUTION from foundry-i
   ``46_noise_audit.masked_autocorr`` (same padding / weight-deconvolution /
   validity conventions; verified against it bit-for-bit in
   tests/test_drizzle.py and in 02_fit_noise_kernels.py on the real inputs).
   ``detrend_sky`` is the matching Background2D detrender (46 conventions).

2. ``drizzle_acf`` — the drizzle noise autocorrelation obtained by
   NUMERICALLY enumerating the actual drizzle operator (square kernel,
   drop-overlap areas) on a test patch, phase-averaged over sub-pixel
   registrations. Anchor: for the 1-D separable square kernel at pixfrac=1
   the phase-averaged nearest-neighbour correlation has the closed form
   t(1) = (r-1)/(r-1/3) (r = native/output pixel-scale ratio; 0.768 at the
   fine-skycell r=3.21), which the enumeration must reproduce
   (tests/test_drizzle.py).

3. ``rho_model`` / ``fit_kernel`` — the two-hyperparameter stationary kernel
   family rho = (1-w)*delta + w*normalize(rho_drz (*) Gaussian(sigma_e)),
   positive-semidefinite by construction (rho_drz is an autocorrelation ->
   nonneg spectrum; Gaussian has nonneg spectrum; convolution multiplies
   spectra; the delta adds a nonneg constant for w in [0,1]). Fitted to a
   measured ACF by weighted least squares over the fit window.

4. ``binned_kernel_from_fine`` — the exact 2x2 block-sum covariance
   transform Cov_b(D) = sum_{o,o' in {0,1}^2} Cov_f(2D + o - o'), and
   ``psd_project`` — the nonparametric FFT clip-retruncate validation kernel.

All CPU numpy/scipy: kernel fitting is one of the sanctioned CPU tasks
(the lensing likelihood itself never runs here).
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.optimize import least_squares

RHO_FLOOR = 0.02  # 46_noise_audit convention (report-level, not used in fits)


# --------------------------------------------------------------------------- #
# (1) masked ACF + detrending  (port of foundry-i 46_noise_audit)
# --------------------------------------------------------------------------- #
def masked_autocorr_full(img, mask):
    """Mask-aware autocorrelation, zero-padded to (2N,2N).

    PORTED VERBATIM (modulo names) from foundry-i 46_noise_audit.py
    ``masked_autocorr``: n = img - mean(img[mask]) on mask px (0 elsewhere);
    A = IFFT2(|FFT2(n)|^2), W = IFFT2(|FFT2(w)|^2) deconvolves the mask
    geometry; rho = (A/W)/(A/W)[0,0], valid only where W > 0.05*W[0,0]
    (NaN elsewhere). Returns (rho fftshifted with lag (0,0) at index (N,N),
    center index N, W fftshifted = pair counts per lag).
    """
    img = np.asarray(img, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    n_pix = img.shape[0]
    n = np.where(mask, img - img[mask].mean(), 0.0)
    pad_n = np.zeros((2 * n_pix, 2 * n_pix))
    pad_w = np.zeros((2 * n_pix, 2 * n_pix))
    pad_n[:n_pix, :n_pix] = n
    pad_w[:n_pix, :n_pix] = mask.astype(np.float64)
    acf = np.real(np.fft.ifft2(np.abs(np.fft.fft2(pad_n)) ** 2))
    wgt = np.real(np.fft.ifft2(np.abs(np.fft.fft2(pad_w)) ** 2))
    valid = wgt > 0.05 * wgt[0, 0]
    cov = np.where(valid, acf / np.where(valid, wgt, 1.0), np.nan)
    rho = cov / cov[0, 0]
    return np.fft.fftshift(rho), n_pix, np.fft.fftshift(wgt)


def masked_acf_2d(residual, keep_mask, max_lag):
    """Windowed masked ACF: (2L+1, 2L+1) rho + per-lag pair counts.

    Args:
        residual: (N, N) image (e.g. model-subtracted, detrended, per-pixel
            sigma-normalized residual).
        keep_mask: (N, N) bool; only these pixels enter the estimate.
        max_lag: L, half-width of the returned window.

    Returns:
        (rho_win, counts_win): both (2L+1, 2L+1), center = lag (0,0).
        counts_win are the mask-pair counts per lag (WLS fit weights).
    """
    rho, center, wgt = masked_autocorr_full(residual, keep_mask)
    L = int(max_lag)
    sl = slice(center - L, center + L + 1)
    return rho[sl, sl].copy(), wgt[sl, sl].copy()


def detrend_sky(img, sky, box):
    """Remove ~1"-scale local background estimated from the sky pixels.

    Port of foundry-i 46_noise_audit.detrend_sky: photutils Background2D
    (non-sky masked, 3x3 median filter, exclude_percentile=95), falling back
    to a global sky-median subtraction if the tiling fails.

    Returns (detrended_img, method_string).
    """
    img = np.asarray(img, dtype=np.float64)
    sky = np.asarray(sky, dtype=bool)
    try:
        from photutils.background import Background2D, MedianBackground

        bkg = Background2D(img, box, mask=~sky, filter_size=3,
                           bkg_estimator=MedianBackground(),
                           exclude_percentile=95.0)
        return img - bkg.background, "background2d"
    except Exception as exc:  # pragma: no cover
        return img - np.median(img[sky]), f"global-median ({exc})"


# --------------------------------------------------------------------------- #
# (2) drizzle-operator enumeration
# --------------------------------------------------------------------------- #
def drizzle_overlap_matrix_1d(scale_ratio, pixfrac, n_out, phase=0.0):
    """1-D drop-overlap matrix of the square-kernel drizzle operator.

    Native (input) pixels have unit size; output pixels have size s = 1/r
    (r = scale_ratio = native/output). Input pixel k drops its flux as a
    top-hat of width pixfrac centered on its center c_k = k + 1/2 + phase.
    Entry [j, k] = overlap length of that drop with output pixel
    [j*s, (j+1)*s). Rows must be normalized (by coverage) before use as the
    signal/noise-combination operator.

    Returns (A (n_out, n_in), k_offsets (n_in,) native indices).
    """
    r = float(scale_ratio)
    p = float(pixfrac)
    if r <= 0 or p <= 0:
        raise ValueError("scale_ratio and pixfrac must be positive")
    s = 1.0 / r
    k_lo = int(np.floor(-p / 2.0 - phase)) - 1
    k_hi = int(np.ceil(n_out * s + p / 2.0 - phase)) + 1
    k = np.arange(k_lo, k_hi + 1)
    c = k + 0.5 + phase                     # drop centers (native units)
    j = np.arange(n_out)
    lo = np.maximum(j[:, None] * s, (c - p / 2.0)[None, :])
    hi = np.minimum((j[:, None] + 1) * s, (c + p / 2.0)[None, :])
    return np.clip(hi - lo, 0.0, None), k


def _cov1d_single_frame(scale_ratio, pixfrac, phase, max_lag, n_out=None):
    """cov(o_j, o_{j+d}) for d=0..max_lag, one frame at a given phase.

    o_j = sum_k a~_{jk} n_k with unit-variance iid native noise n and
    a~ = A / rowsum(A). Averaged over interior output rows (margins excluded
    so the finite patch has no edge effect).
    """
    r = float(scale_ratio)
    margin = int(np.ceil((pixfrac / 2.0 + 1.0) * r)) + 2
    if n_out is None:
        n_out = 4 * (max_lag + margin)
    A, _ = drizzle_overlap_matrix_1d(r, pixfrac, n_out, phase)
    rows = A.sum(axis=1)
    At = A / rows[:, None]
    G = At @ At.T                            # cov of all output pairs
    j = np.arange(margin, n_out - margin - max_lag)
    cov = np.empty(max_lag + 1)
    for d in range(max_lag + 1):
        cov[d] = G[j, j + d].mean()
    return cov


def drizzle_acf(scale_ratio, pixfrac=1.0, n_frames=3, offsets=None,
                max_lag=8, n_phase=16, n_out=None):
    """Noise autocorrelation of the (stacked) square-kernel drizzle operator.

    The operator is enumerated numerically (drop-overlap matrix on a test
    patch); the square kernel makes the 2-D operator the tensor product of
    two 1-D operators, so 2-D covariances are exact outer products of the
    per-axis 1-D covariances (per frame, then summed over frames).

    Args:
        scale_ratio: r = native / output pixel scale (3.21 fine skycell).
        pixfrac: drizzle pixfrac (D001PIXF; 1.0 for the HAP products).
        n_frames: number of stacked exposures (NDRIZIM; equal weights).
        offsets: (n_frames, 2) dither offsets in NATIVE-pixel units
            ((dy, dx); only the fractional parts matter). None = unknown
            registration: phase-average covariance and variance separately
            over an (n_phase x n_phase) uniform sub-pixel phase grid — the
            convention under which t(1) = (r-1)/(r-1/3) holds exactly.
        max_lag: L; the returned rho is (2L+1, 2L+1).
        n_phase: phase-grid resolution per axis (offsets=None only).
        n_out: test-patch length in output px (default: auto, interior-safe).

    Returns dict:
        rho: (2L+1, 2L+1) stacked-noise autocorrelation, center = lag 0.
        rho_single: same for a single frame.
        t1d: (L+1,) phase/frame-averaged 1-D axis correlation t(d).
        t1: t1d[1]; t1_analytic: (r-1)/(r-1/3).
        var_out_over_var_in: output/input variance ratio of the stack.
    """
    r = float(scale_ratio)
    L = int(max_lag)

    if offsets is None:
        phases = (np.arange(n_phase) + 0.5) / n_phase
        cov_ax = np.mean([_cov1d_single_frame(r, pixfrac, ph, L, n_out)
                          for ph in phases], axis=0)
        # phase-averaged frames are exchangeable: per-frame cov identical
        cov2d_single = np.outer(cov_ax, cov_ax)
        cov2d_stack = cov2d_single / float(n_frames)
        t1d = cov_ax / cov_ax[0]
    else:
        offsets = np.asarray(offsets, dtype=np.float64)
        if offsets.shape != (n_frames, 2):
            raise ValueError(f"offsets must be ({n_frames}, 2), "
                             f"got {offsets.shape}")
        cov2d_stack = np.zeros((L + 1, L + 1))
        covy_list = []
        for f in range(n_frames):
            covy = _cov1d_single_frame(r, pixfrac, offsets[f, 0] % 1.0, L, n_out)
            covx = _cov1d_single_frame(r, pixfrac, offsets[f, 1] % 1.0, L, n_out)
            covy_list.append((covy, covx))
            cov2d_stack += np.outer(covy, covx)
        cov2d_stack /= float(n_frames) ** 2
        cov2d_single = np.outer(*covy_list[0])
        t_stack = np.sum([cy for cy, _ in covy_list], axis=0)
        t1d = t_stack / t_stack[0]

    def _mirror(q):
        """(L+1, L+1) first-quadrant array -> (2L+1, 2L+1) symmetric."""
        full = np.zeros((2 * L + 1, 2 * L + 1))
        for sy in (-1, 1):
            for sx in (-1, 1):
                full[L + sy * np.arange(L + 1)[:, None],
                     L + sx * np.arange(L + 1)[None, :]] = q
        return full

    rho = _mirror(cov2d_stack / cov2d_stack[0, 0])
    rho_single = _mirror(cov2d_single / cov2d_single[0, 0])
    return dict(
        rho=rho, rho_single=rho_single, t1d=t1d, t1=float(t1d[1]),
        t1_analytic=float((r - 1.0) / (r - 1.0 / 3.0)),
        var_out_over_var_in=float(cov2d_stack[0, 0]),
        scale_ratio=r, pixfrac=float(pixfrac), n_frames=int(n_frames),
    )


# --------------------------------------------------------------------------- #
# (3) kernel family + WLS fit
# --------------------------------------------------------------------------- #
def rho_model(w, sigma_e, rho_drz, max_lag):
    """rho = (1-w)*delta + w*normalize(rho_drz (*) Gaussian(sigma_e)).

    Args:
        w: correlated-fraction weight in [0, 1].
        sigma_e: Gaussian smoothing sigma (px) applied to the drizzle ACF
            (phenomenological extra correlation: PSF-correlated sky,
            resampling, charge diffusion). sigma_e <= 1e-3 means none.
        rho_drz: (2Ld+1, 2Ld+1) drizzle ACF; Ld must be >= max_lag (the
            Gaussian tail beyond Ld is folded by 'nearest' padding — keep
            Ld >= max_lag + 4*sigma_e for accuracy).
        max_lag: L of the returned (2L+1, 2L+1) kernel.

    PSD by construction for w in [0,1] (see module docstring).
    """
    rho_drz = np.asarray(rho_drz, dtype=np.float64)
    Ld = (rho_drz.shape[0] - 1) // 2
    L = int(max_lag)
    if Ld < L:
        raise ValueError(f"rho_drz half-width {Ld} < max_lag {L}")
    if sigma_e > 1e-3:
        base = gaussian_filter(rho_drz, sigma=float(sigma_e),
                               mode="constant", cval=0.0)
    else:
        base = rho_drz.copy()
    base = base / base[Ld, Ld]
    sl = slice(Ld - L, Ld + L + 1)
    out = float(w) * base[sl, sl]
    out[L, L] += 1.0 - float(w)
    return out


def fit_kernel(acf_2d, lag_counts, rho_drz, max_lag=None,
               sigma_bounds=(1e-6, 6.0)):
    """WLS fit of (w, sigma_e) to a measured ACF over the fit window.

    Args:
        acf_2d: (2L+1, 2L+1) measured rho (masked_acf_2d output). NaNs
            (invalid lags) are excluded.
        lag_counts: same-shape pair counts -> WLS weights (sqrt(counts)).
        rho_drz: drizzle-anchor ACF, half-width >= L + Gaussian tail.
        max_lag: fit-window half-width L (default: from acf_2d shape).
        sigma_bounds: bounds for sigma_e (px).

    Returns dict: w, sigma_e, rho_fit ((2L+1,2L+1)), max_abs_resid,
        rms_resid, per-lag residual map, gate_le_0p05, n_lags_fit, cost.
    """
    acf_2d = np.asarray(acf_2d, dtype=np.float64)
    L = (acf_2d.shape[0] - 1) // 2 if max_lag is None else int(max_lag)
    c = (acf_2d.shape[0] - 1) // 2
    sl = slice(c - L, c + L + 1)
    meas = acf_2d[sl, sl]
    counts = np.asarray(lag_counts, dtype=np.float64)[sl, sl]
    ok = np.isfinite(meas) & (counts > 0)
    wgt = np.sqrt(counts / counts[ok].max())

    def resid(theta):
        w, se = theta
        m = rho_model(w, se, rho_drz, L)
        return ((m - np.where(ok, meas, m)) * wgt)[ok]

    best = None
    for w0 in (0.3, 0.6, 0.9, 0.99):
        for s0 in (0.05, 0.4, 0.8, 1.6):
            try:
                res = least_squares(
                    resid, x0=[w0, s0],
                    bounds=([0.0, sigma_bounds[0]], [1.0, sigma_bounds[1]]),
                    xtol=1e-14, ftol=1e-14, gtol=1e-14)
            except Exception:
                continue
            if best is None or res.cost < best.cost:
                best = res
    if best is None:
        raise RuntimeError("kernel fit failed from every start point")

    w_fit, se_fit = float(best.x[0]), float(best.x[1])
    rho_fit = rho_model(w_fit, se_fit, rho_drz, L)
    resid_map = np.where(ok, rho_fit - meas, np.nan)
    max_abs = float(np.nanmax(np.abs(resid_map)))
    rms = float(np.sqrt(np.nanmean(resid_map ** 2)))
    return dict(
        w=w_fit, sigma_e=se_fit, rho_fit=rho_fit, resid_map=resid_map,
        max_abs_resid=max_abs, rms_resid=rms,
        gate_le_0p05=bool(max_abs <= 0.05),
        n_lags_fit=int(ok.sum()), cost=float(best.cost),
    )


def gaussian2d(sig_y, sig_x, max_lag, c=0.0):
    """Bivariate Gaussian correlation, center-normalized to 1.

    G(dy, dx) = exp(-1/2 * (dy^2/sy^2 - 2c dy dx/(sy sx) + dx^2/sx^2)/(1-c^2))

    PSD for |c| < 1 (Gaussian of a positive-definite quadratic form has a
    positive Gaussian spectrum; discrete sampling wraps it, staying > 0).
    """
    L = int(max_lag)
    sy = max(float(sig_y), 1e-9)
    sx = max(float(sig_x), 1e-9)
    c = float(np.clip(c, -0.99, 0.99))
    d = np.arange(-L, L + 1, dtype=np.float64)
    dy = d[:, None]
    dx = d[None, :]
    q = (dy ** 2 / sy ** 2 - 2.0 * c * dy * dx / (sy * sx)
         + dx ** 2 / sx ** 2) / (1.0 - c * c)
    return np.exp(-0.5 * q)


def rho_model2(w_d, sig_ey, sig_ex, c_e, w_b, sig_by, sig_bx, c_b,
               rho_drz, max_lag):
    """Two-component kernel family (P1a deviation, justified in
    data/noise_kernel_report.json):

        rho = (1 - w_d - w_b) * delta
              + w_d * normalize(rho_drz (*) G(sig_ey, sig_ex, c_e))   [core]
              + w_b * G(sig_by, sig_bx, c_b)                    [background]

    The bivariate-Gaussian-smoothed drizzle core absorbs the (rotated-frame)
    core anisotropy; the second component models the medium-scale correlated
    background the ~1" detrending leaves in the model-subtracted residuals
    (isotropic pedestal on the fine/binned skycell products; column-direction
    stripe correlation on the native product). Each component is PSD; a
    nonnegative mixture with w_d + w_b <= 1 is PSD.
    """
    from scipy.signal import fftconvolve

    rho_drz = np.asarray(rho_drz, dtype=np.float64)
    Ld = (rho_drz.shape[0] - 1) // 2
    L = int(max_lag)
    if Ld < L:
        raise ValueError(f"rho_drz half-width {Ld} < max_lag {L}")
    sig_norm = float(np.hypot(sig_ey, sig_ex))
    if sig_norm > 1e-3:
        Lg = min(int(np.ceil(4.0 * max(sig_ey, sig_ex))) + 1, Ld)
        g = gaussian2d(sig_ey, sig_ex, Lg, c_e)
        base = fftconvolve(rho_drz, g, mode="same")
    else:
        base = rho_drz.copy()
    base = base / base[Ld, Ld]
    sl = slice(Ld - L, Ld + L + 1)
    out = float(w_d) * base[sl, sl] \
        + float(w_b) * gaussian2d(sig_by, sig_bx, L, c_b)
    out[L, L] += 1.0 - float(w_d) - float(w_b)
    return out


def fit_kernel2(acf_2d, lag_counts, rho_drz, max_lag=None,
                sigma_e_bounds=(1e-6, 6.0), sig_b_bounds=(0.3, 40.0),
                c_bound=0.8):
    """WLS fit of the two-component family
    (w_d, sig_ey, sig_ex, c_e, w_b, sig_by, sig_bx, c_b) over the fit window.

    The simplex constraint w_d + w_b <= 1 is enforced exactly by the smooth
    reparameterization w_d = u0*f, w_b = u0*(1-f) with u0, f in [0,1].

    Returns dict like fit_kernel plus the extra hyperparameters.
    """
    acf_2d = np.asarray(acf_2d, dtype=np.float64)
    L = (acf_2d.shape[0] - 1) // 2 if max_lag is None else int(max_lag)
    c = (acf_2d.shape[0] - 1) // 2
    sl = slice(c - L, c + L + 1)
    meas = acf_2d[sl, sl]
    counts = np.asarray(lag_counts, dtype=np.float64)[sl, sl]
    ok = np.isfinite(meas) & (counts > 0)
    wgt = np.sqrt(counts / counts[ok].max())

    def unpack(theta):
        u0, f, sey, sex, ce, sby, sbx, cb = theta
        return (u0 * f, sey, sex, ce, u0 * (1.0 - f), sby, sbx, cb)

    def resid(theta):
        p = unpack(theta)
        m = rho_model2(*p, rho_drz, L)
        return ((m - np.where(ok, meas, m)) * wgt)[ok]

    se_lo, se_hi = sigma_e_bounds
    sb_lo, sb_hi = sig_b_bounds
    lo = [0.0, 0.0, se_lo, se_lo, -c_bound, sb_lo, sb_lo, -c_bound]
    hi = [1.0, 1.0, se_hi, se_hi, c_bound, sb_hi, sb_hi, c_bound]
    best = None
    for u0 in (0.7, 0.95):
        for f in (0.5, 0.8):
            for s0 in (0.1, 0.8):
                for sb0 in (2.0, 6.0, 25.0):
                    for cb0 in (-0.2, 0.0, 0.2):
                        x0 = [u0, f, s0, s0, 0.0, sb0, sb0, cb0]
                        try:
                            res = least_squares(
                                resid, x0=x0, bounds=(lo, hi),
                                xtol=1e-13, ftol=1e-13, gtol=1e-13)
                        except Exception:
                            continue
                        if best is None or res.cost < best.cost:
                            best = res
    if best is None:
        raise RuntimeError("kernel fit (2-component) failed everywhere")

    p = unpack(best.x)
    w_d, sig_ey, sig_ex, c_e, w_b, sig_by, sig_bx, c_b = p
    rho_fit = rho_model2(*p, rho_drz, L)
    resid_map = np.where(ok, rho_fit - meas, np.nan)
    max_abs = float(np.nanmax(np.abs(resid_map)))
    return dict(
        w_d=float(w_d), sig_ey=float(sig_ey), sig_ex=float(sig_ex),
        c_e=float(c_e), w_b=float(w_b), sig_by=float(sig_by),
        sig_bx=float(sig_bx), c_b=float(c_b),
        params=[float(v) for v in p],
        rho_fit=rho_fit, resid_map=resid_map,
        max_abs_resid=max_abs,
        rms_resid=float(np.sqrt(np.nanmean(resid_map ** 2))),
        gate_le_0p05=bool(max_abs <= 0.05),
        n_lags_fit=int(ok.sum()), cost=float(best.cost),
    )


# --------------------------------------------------------------------------- #
# (4) block-sum transform + nonparametric PSD projection
# --------------------------------------------------------------------------- #
def block_sum(a, k):
    """k x k block-sum of a square array whose side is divisible by k
    (generalizes foundry-i 40c_make_cutout_binned.blocksum, k=2)."""
    a = np.asarray(a)
    n = a.shape[0] // k
    m = a.shape[1] // k
    return a[:n * k, :m * k].reshape(n, k, m, k).sum(axis=(1, 3))


def block_all(mask, k):
    """k x k block-AND of a boolean mask (40c blockall, generalized)."""
    mask = np.asarray(mask, dtype=bool)
    n = mask.shape[0] // k
    m = mask.shape[1] // k
    return mask[:n * k, :m * k].reshape(n, k, m, k).all(axis=(1, 3))


def binned_kernel_from_fine(cov_fine):
    """Exact 2x2 block-sum covariance transform.

    Cov_b(D) = sum_{o, o' in {0,1}^2} Cov_f(2D + o - o')   (stationary case)

    Args:
        cov_fine: (2Lf+1, 2Lf+1) fine-grid stationary covariance (or rho —
            the transform is linear; normalize the output yourself if you
            need a correlation).

    Returns:
        cov_binned: (2Lb+1, 2Lb+1) with Lb = (Lf - 1) // 2.
    """
    cov_fine = np.asarray(cov_fine, dtype=np.float64)
    Lf = (cov_fine.shape[0] - 1) // 2
    Lb = (Lf - 1) // 2
    if Lb < 0:
        raise ValueError("cov_fine too small")
    out = np.zeros((2 * Lb + 1, 2 * Lb + 1))
    offs = (0, 1)
    for dy in range(-Lb, Lb + 1):
        for dx in range(-Lb, Lb + 1):
            tot = 0.0
            for oy in offs:
                for ox in offs:
                    for oy2 in offs:
                        for ox2 in offs:
                            tot += cov_fine[Lf + 2 * dy + oy - oy2,
                                            Lf + 2 * dx + ox - ox2]
            out[Lb + dy, Lb + dx] = tot
    return out


def psd_project(acf, support, s_floor=0.0, grid=256, n_iter=5):
    """Nonparametric PSD-projected validation kernel (FFT clip re-truncate).

    Alternating projections: embed the (odd-square) ACF periodically on a
    grid, clip its spectrum at max(s_floor * mean(S), 0), transform back,
    re-truncate to the (2*support+1)^2 window, renormalize the center to 1.

    Returns (acf_projected, diagnostics dict).
    """
    a = np.asarray(acf, dtype=np.float64)
    L = (a.shape[0] - 1) // 2
    Ms = int(support)
    if Ms > L:
        raise ValueError(f"support {Ms} > input half-width {L}")
    cur = a[L - Ms:L + Ms + 1, L - Ms:L + Ms + 1].copy()
    cur[~np.isfinite(cur)] = 0.0
    s_min_first = None
    for _ in range(int(n_iter)):
        emb = np.zeros((grid, grid))
        for dy in range(-Ms, Ms + 1):
            for dx in range(-Ms, Ms + 1):
                emb[dy % grid, dx % grid] = cur[Ms + dy, Ms + dx]
        S = np.real(np.fft.fft2(emb))
        if s_min_first is None:
            s_min_first = float(S.min())
        floor = max(float(s_floor) * float(S.mean()), 0.0)
        S = np.maximum(S, floor)
        back = np.real(np.fft.ifft2(S))
        new = np.empty_like(cur)
        for dy in range(-Ms, Ms + 1):
            for dx in range(-Ms, Ms + 1):
                new[Ms + dy, Ms + dx] = back[dy % grid, dx % grid]
        cur = new / new[Ms, Ms]
    diag = dict(s_min_input=s_min_first, grid=grid, n_iter=int(n_iter),
                s_floor=float(s_floor),
                max_change=float(np.max(np.abs(
                    cur - a[L - Ms:L + Ms + 1, L - Ms:L + Ms + 1]))))
    return cur, diag
