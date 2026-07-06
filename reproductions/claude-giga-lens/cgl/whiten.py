"""Convolutional whitening operators (P1 correlated-noise likelihood).

The whitener contract used by ``cgl.likelihood.build_marg_model(whiten_fn=...)``:

    whiten_fn(image_hw) -> (n_pix,) flattened whitened vector

applied IDENTICALLY to the residual R = Y - M_det and to each shapelet design
column, so the marginalization math in ``cgl.marg.marg_loglik`` sees a
consistently whitened linear model whatever the noise model is.

``make_conv_whitener(h, sqrt_d_inv, keep_w)`` builds the stationary-kernel
whitener for C = D^{1/2} K D^{1/2}:

    u = keep_w * conv2d_SAME(h, sqrt_d_inv * image)

where ``sqrt_d_inv`` ~ D^{-1/2} (per-pixel 1/sigma), ``h`` ~ the whitening
kernel with hhat ~ 1/sqrt(Khat), and ``keep_w`` masks pixels whose whitened
value is invalid (mask interior +, later, conv-support edge trims).

Parity anchor (gate D): h = [[1.0]] (1x1 delta kernel) with
keep_w = keep_mask and sqrt_d_inv = 1/masked_err_map reproduces the validated
diagonal sqrt(W) path (masked pixels are exactly zeroed here vs down-weighted
by 1/1e10 there; the log-posterior difference is O(1e-16), gated at 1e-10).

Convolution semantics: ``jax.lax.conv_general_dilated`` with 'SAME' padding is
CROSS-CORRELATION (no kernel flip), zero-padded at the edges — i.e.
``scipy.ndimage.correlate(v, h, mode='constant', cval=0)`` for odd-shaped h.
Symmetric whitening kernels make the distinction moot; asymmetric callers must
pass h in correlation orientation (locked by tests/test_marg.py).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from scipy.ndimage import binary_erosion


def make_conv_whitener(h, sqrt_d_inv, keep_w):
    """Build whiten_fn(image_hw) -> (n_pix,) for a stationary conv whitener.

    Args:
        h: (kh, kw) whitening kernel, odd-shaped (correlation orientation).
        sqrt_d_inv: (H, W) per-pixel D^{-1/2} = 1/sigma diagonal factor.
        keep_w: (H, W) 0/1 (or bool) mask of pixels kept in the whitened
            vector; excluded pixels are exactly zeroed after the convolution.

    Returns:
        whiten_fn: image (H, W) -> flattened whitened vector (H*W,).
    """
    h = jnp.asarray(h)
    if h.ndim != 2 or h.shape[0] % 2 != 1 or h.shape[1] % 2 != 1:
        raise ValueError(
            f"whitening kernel must be 2-D and odd-shaped, got {tuple(h.shape)}"
        )
    sqrt_d_inv = jnp.asarray(sqrt_d_inv)
    keep_w = jnp.asarray(keep_w)
    kern = h[:, :, None, None]  # (kh, kw, in=1, out=1) for 'HWIO'

    def whiten_fn(image_hw):
        v = image_hw * sqrt_d_inv
        u = jax.lax.conv_general_dilated(
            v[None, None, :, :].astype(v.dtype),
            kern.astype(v.dtype),
            window_strides=(1, 1),
            padding="SAME",
            dimension_numbers=("NCHW", "HWIO", "NCHW"),
        )
        u = u[0, 0] * keep_w.astype(v.dtype)
        return jnp.reshape(u, (-1,))

    return whiten_fn


# --------------------------------------------------------------------------- #
# P1a: whitening-kernel construction for a stationary correlation kernel rho
# --------------------------------------------------------------------------- #
def _embed_kernel(k, grid):
    """Place an odd-square centered kernel at lag (0,0) of a periodic grid."""
    k = np.asarray(k, dtype=np.float64)
    L = (k.shape[0] - 1) // 2
    if 2 * L + 1 > grid:
        raise ValueError(f"kernel half-width {L} too large for grid {grid}")
    emb = np.zeros((grid, grid))
    for dy in range(-L, L + 1):
        emb[dy % grid, (np.arange(-L, L + 1)) % grid] = k[L + dy, :]
    return emb


def _spectrum_of_taps(h, grid):
    """H(omega) on the grid for symmetric real taps h (real spectrum)."""
    return np.real(np.fft.fft2(_embed_kernel(h, grid)))


def build_whitener(rho, M, s_floor=0.05, grid=512, n_gn=40):
    """Truncated convolutional whitening kernel for stationary correlation rho.

    Construction (design doc, P1a):
      1. Embed rho periodically on (grid, grid); S = FFT (real, symmetric).
      2. Floor: S <- max(S, s_floor * mean(S)) (mean(S) = rho[0,0] = 1).
      3. hhat = S^(-1/2); h = IFFT (real, symmetric); truncate to (2M+1)^2.
      4. Gauss-Newton refine of the truncated taps on the least-squares
         objective ||S * H_M(omega)^2 - 1||_2 (all-FFT normal equations;
         Levenberg damping), followed by Lawson-IRLS rounds (reweighting by
         |r|) that push the CHEBYSHEV ripple down — the gate quantity is
         e_op = max_omega |S * H_M(omega)^2 - 1|, and the plain L2 optimum
         concentrates its worst ripple on the few sharp-spectral-peak
         frequencies (error ~ 2 sqrt(S) * deltaH is sqrt(S)-amplified).
         The best-e_op iterate over all stages is returned.

    Args:
        rho: (2L+1, 2L+1) stationary correlation kernel (center = 1).
        M: half-width of the truncated whitening kernel ((2M+1)^2 taps).
        s_floor: relative spectral floor (fraction of mean(S)).
        grid: periodic embedding size (must exceed both supports comfortably).
        n_gn: max Gauss-Newton iterations.

    Returns dict:
        h: (2M+1, 2M+1) whitening taps (correlation orientation; symmetric,
            so conv == corr).
        e_op: max_omega |S*H^2 - 1| after refinement (gate: <= 0.02).
        e_op_init: before refinement.
        logdet_per_pix: mean_omega log S (the theta-independent per-pixel
            log-det of the floored stationary kernel K; Szego / circulant
            approximation of log|K|/n).
        s_min, s_max, floor_frac: spectrum diagnostics (pre-floor min).
        n_gn_used: accepted Gauss-Newton steps.
    """
    rho = np.asarray(rho, dtype=np.float64)
    Lr = (rho.shape[0] - 1) // 2
    M = int(M)
    if grid < 4 * max(Lr, M):
        raise ValueError(f"grid {grid} too small for L={Lr}, M={M}")

    S_raw = np.real(np.fft.fft2(_embed_kernel(rho, grid)))
    s_mean = float(S_raw.mean())
    floor = s_floor * s_mean
    S = np.maximum(S_raw, floor)

    # ---- step 3: spectral inverse square root, truncated -------------------
    h_full = np.real(np.fft.ifft2(1.0 / np.sqrt(S)))
    h_shift = np.fft.fftshift(h_full)
    c = grid // 2
    h = h_shift[c - M:c + M + 1, c - M:c + M + 1].copy()
    h = 0.5 * (h + h[::-1, ::-1])                    # enforce exact symmetry

    def e_op_of(h_taps):
        H = _spectrum_of_taps(h_taps, grid)
        return float(np.max(np.abs(S * H * H - 1.0)))

    e_op_init = e_op_of(h)

    # ---- step 4: Gauss-Newton on ||S H^2 - 1||_2 via FFT normal equations --
    # Unique taps under point symmetry: theta_0 = h[0,0] (basis c_0 = 1),
    # theta_k = h[k] for k in the upper half-plane (basis c_k = 2cos(w.k)).
    ks = [(0, 0)]
    for dy in range(0, M + 1):
        for dx in range(-M, M + 1):
            if dy == 0 and dx <= 0:
                continue
            ks.append((dy, dx))
    n_u = len(ks)

    def taps_to_theta(h_taps):
        th = np.empty(n_u)
        for i, (dy, dx) in enumerate(ks):
            th[i] = h_taps[M + dy, M + dx]
        return th

    def theta_to_taps(th):
        out = np.zeros((2 * M + 1, 2 * M + 1))
        for i, (dy, dx) in enumerate(ks):
            out[M + dy, M + dx] = th[i]
            out[M - dy, M - dx] = th[i]
        return out

    def cos_sums(W):
        """G(m) = sum_omega W(omega) cos(omega . m) for all lags m (FFT)."""
        return grid * grid * np.real(np.fft.ifft2(W))

    def gn_rounds(theta, Wfreq, n_iter):
        """Weighted Gauss-Newton (Levenberg) on sum Wfreq * r^2; returns the
        updated theta and the number of accepted steps."""
        lam = 1e-8
        used = 0
        cost_prev = None
        for _ in range(int(n_iter)):
            H = _spectrum_of_taps(theta_to_taps(theta), grid)
            r = S * H * H - 1.0
            cost = float(np.sum(Wfreq * r * r))
            if cost_prev is not None and \
                    cost_prev - cost < 1e-14 * max(cost, 1.0):
                break
            cost_prev = cost
            GW = cos_sums(Wfreq * 4.0 * S * S * H * H)
            GV = cos_sums(Wfreq * 2.0 * S * H * r)
            JtJ = np.empty((n_u, n_u))
            Jtr = np.empty(n_u)
            for i, ki in enumerate(ks):
                Jtr[i] = GV[ki[0] % grid, ki[1] % grid] * \
                    (1.0 if i == 0 else 2.0)
                for j, kj in enumerate(ks):
                    if i == 0 and j == 0:
                        JtJ[0, 0] = GW[0, 0]
                    elif i == 0:
                        JtJ[0, j] = 2.0 * GW[kj[0] % grid, kj[1] % grid]
                    elif j == 0:
                        JtJ[i, 0] = 2.0 * GW[ki[0] % grid, ki[1] % grid]
                    else:
                        dm = ((ki[0] - kj[0]) % grid, (ki[1] - kj[1]) % grid)
                        pm = ((ki[0] + kj[0]) % grid, (ki[1] + kj[1]) % grid)
                        JtJ[i, j] = 2.0 * (GW[dm] + GW[pm])
            d = np.diag(JtJ).copy()
            d[d <= 0] = 1.0
            stepped = False
            for _try in range(8):
                try:
                    delta = np.linalg.solve(JtJ + lam * np.diag(d), -Jtr)
                except np.linalg.LinAlgError:
                    lam *= 10.0
                    continue
                th_new = theta + delta
                H_new = _spectrum_of_taps(theta_to_taps(th_new), grid)
                r_new = S * H_new * H_new - 1.0
                if float(np.sum(Wfreq * r_new * r_new)) < cost:
                    theta = th_new
                    lam = max(lam * 0.3, 1e-12)
                    stepped = True
                    used += 1
                    break
                lam *= 10.0
            if not stepped:
                break
        return theta, used

    # stage 1: plain L2 Gauss-Newton
    theta = taps_to_theta(h)
    ones = np.ones_like(S)
    theta, n_used = gn_rounds(theta, ones, n_gn)
    best_h = theta_to_taps(theta)
    best_e = e_op_of(best_h)
    if e_op_init < best_e:
        best_h, best_e = h, e_op_init

    # stage 2: Lawson-IRLS toward the Chebyshev (max-ripple) optimum
    Wfreq = np.ones_like(S)
    th_l = theta.copy()
    n_lawson = 0
    for _ in range(8):
        H = _spectrum_of_taps(theta_to_taps(th_l), grid)
        r = np.abs(S * H * H - 1.0)
        if r.max() < 1e-13:          # already exact (e.g. delta kernel)
            break
        Wfreq = Wfreq * (r + 1e-6 * r.max())
        Wfreq = Wfreq / Wfreq.mean()
        th_l, used = gn_rounds(th_l, Wfreq, 6)
        n_lawson += used
        e_l = e_op_of(theta_to_taps(th_l))
        if e_l < best_e:
            best_h, best_e = theta_to_taps(th_l), e_l

    return dict(
        h=best_h, e_op=best_e, e_op_init=e_op_init,
        n_lawson_used=int(n_lawson),
        logdet_per_pix=float(np.mean(np.log(S))),
        s_min=float(S_raw.min()), s_max=float(S_raw.max()),
        s_mean=s_mean, floor=float(floor),
        floor_frac=float(np.mean(S_raw < floor)),
        grid=int(grid), s_floor=float(s_floor), M=M,
        n_gn_used=int(n_used),
    )


def erode_keep(keep_mask, M):
    """Whitened-domain keep mask: binary erosion by the (2M+1)^2 support,
    intersected with the M-pixel interior (SAME-conv zero-padding border).

    A pixel survives iff every pixel its whitening kernel touches is kept and
    in-grid — so kept whitened values never see masked pixels (whose
    down-weighted values are ~0 instead of data) or the zero-padded border.
    binary_erosion with border_value=0 enforces both; the explicit border
    trim is kept as a belt-and-braces guard.
    """
    keep = np.asarray(keep_mask, dtype=bool)
    M = int(M)
    if M == 0:
        return keep.copy()
    structure = np.ones((2 * M + 1, 2 * M + 1), dtype=bool)
    er = binary_erosion(keep, structure=structure, border_value=0)
    er[:M, :] = False
    er[-M:, :] = False
    er[:, :M] = False
    er[:, -M:] = False
    return er
