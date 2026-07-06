"""Exact dense-covariance CPU reference machinery (P1a validation only).

CPU float64 by design — this module is the sanctioned-CPU independent
implementation the GPU conv-whitened likelihood is validated against
(04_validate_exact_smallgrid.py). It is deliberately jax-free: everything is
numpy/scipy so no operation can share code (or bugs) with the JAX path.

Objects:
  * conv_matrix(h, shape)       — the SAME zero-padded cross-correlation as a
                                  sparse matrix (independent re-implementation
                                  of the jax.lax conv semantics).
  * whitening_operator(...)     — G_e = R_erode @ C_h @ diag(sqrt_d_inv): the
                                  dense representation of the conv whitener.
                                  The likelihood's implied data-space precision
                                  is C^{-1} := G_e^T G_e (rank n_e).
  * dense_whitened_marg(...)    — the whitened ridge-marginalized logL via
                                  scipy cho_factor (the 04 hard-gate quantity).
  * dense_C_kept / ExactGLS     — the physical dense covariance
                                  C = D^{1/2} K D^{1/2} on kept pixels, exact
                                  GLS with all log-det terms (Cholesky once,
                                  reused across draws).
  * sigma_u_audit(...)          — Sigma_u = G_e C G_e^T: per-pixel Var(u) and
                                  off-diagonal correlations at |lag|<=3.
  * sample_stationary_batch(...)— exact FFT draws from the stationary kernel
                                  (for the 260^2 Monte-Carlo whiteness path).
  * circulant_cg_quad(...)      — FFT-circulant-preconditioned CG for spot
                                  quadratic forms R^T C^{-1} R on grids too
                                  large to factor densely (260^2).
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.linalg import cho_factor, cho_solve
from scipy.signal import fftconvolve
from scipy.sparse.linalg import LinearOperator, cg


# --------------------------------------------------------------------------- #
# conv whitener as a dense/sparse operator
# --------------------------------------------------------------------------- #
def _shift_1d(n, k):
    """Sparse (n, n) matrix E with E[i, i+k] = 1 (zero outside the grid)."""
    if abs(k) >= n:
        return sp.csr_matrix((n, n))
    return sp.eye(n, n, k=k, format="csr")


def conv_matrix(h, shape):
    """Sparse matrix of the 'SAME' zero-padded 2-D CROSS-CORRELATION by h.

    Matches jax.lax.conv_general_dilated / scipy.ndimage.correlate
    (mode='constant'): out[i, j] = sum_{ky,kx} h[M+ky, M+kx] * in[i+ky, j+kx].
    """
    h = np.asarray(h, dtype=np.float64)
    kh, kw = h.shape
    if kh % 2 != 1 or kw % 2 != 1:
        raise ValueError("kernel must be odd-shaped")
    My, Mx = kh // 2, kw // 2
    H, W = shape
    A = sp.csr_matrix((H * W, H * W))
    for iy in range(kh):
        Ey = _shift_1d(H, iy - My)
        for ix in range(kw):
            w = h[iy, ix]
            if w == 0.0:
                continue
            A = A + w * sp.kron(Ey, _shift_1d(W, ix - Mx), format="csr")
    return A.tocsr()


def whitening_operator(h, sqrt_d_inv, keep_w):
    """G_e (n_e, n_pix) sparse: rows of C_h @ diag(sqrt_d_inv) at eroded px.

    Returns (G_e, eroded_flat_idx).
    """
    sqrt_d_inv = np.asarray(sqrt_d_inv, dtype=np.float64)
    keep_w = np.asarray(keep_w, dtype=bool)
    Ch = conv_matrix(h, sqrt_d_inv.shape)
    G = Ch @ sp.diags(sqrt_d_inv.reshape(-1))
    idx = np.flatnonzero(keep_w.reshape(-1))
    return G.tocsr()[idx], idx


def dense_whitened_marg(G_e, Y, M_det, ret, Lambda_diag):
    """The whitened ridge-marginalized data logL via scipy (04 hard gate).

    Computes exactly the functional the GPU path implements — the data-space
    precision C^{-1} := G_e^T G_e — with independent CPU linear algebra:
        u  = G_e (Y - M_det)          (whitened residual, eroded support)
        Xw = G_e X                    (whitened design columns)
        b = Xw^T u;  A = Xw^T Xw + diag(Lambda);  cho_factor(A)
        logL = -1/2 ||u||^2 + 1/2 b.a* - 1/2 logdet A
    """
    Y = np.asarray(Y, dtype=np.float64).reshape(-1)
    M = np.asarray(M_det, dtype=np.float64).reshape(-1)
    depth = ret.shape[-1]
    X = np.asarray(ret, dtype=np.float64).reshape(-1, depth)
    u = G_e @ (Y - M)
    Xw = G_e @ X
    b = Xw.T @ u
    A = Xw.T @ Xw + np.diag(np.asarray(Lambda_diag, dtype=np.float64))
    c, low = cho_factor(A, lower=True)
    a_star = cho_solve((c, low), b)
    logdetA = 2.0 * float(np.sum(np.log(np.diag(c))))
    logL = -0.5 * float(u @ u) + 0.5 * float(b @ a_star) - 0.5 * logdetA
    return dict(logL=logL, a_star=a_star, logdetA=logdetA, u=u, b=b, A=A,
                n_e=int(u.size))


# --------------------------------------------------------------------------- #
# physical dense covariance C = D^{1/2} K D^{1/2}
# --------------------------------------------------------------------------- #
def _rho_lookup(rho):
    L = (rho.shape[0] - 1) // 2

    def look(dy, dx):
        out = np.zeros(np.broadcast(dy, dx).shape)
        m = (np.abs(dy) <= L) & (np.abs(dx) <= L)
        out[m] = rho[L + dy[m], L + dx[m]]
        return out

    return look


def dense_C_kept(rho, err_map, keep_idx, shape):
    """Dense C = D^{1/2} K D^{1/2} restricted to keep_idx (float64).

    K_ij = rho(delta_ij) (stationary, truncated at the kernel support);
    D = diag(err^2) with the UNMASKED per-pixel sigma.
    """
    err = np.asarray(err_map, dtype=np.float64).reshape(-1)
    H, W = shape
    yy = (keep_idx // W).astype(np.int64)
    xx = (keep_idx % W).astype(np.int64)
    look = _rho_lookup(np.asarray(rho, dtype=np.float64))
    dy = yy[:, None] - yy[None, :]
    dx = xx[:, None] - xx[None, :]
    K = look(dy, dx)
    s = err[keep_idx]
    return K * s[:, None] * s[None, :]


class ExactGLS:
    """Exact GLS with the physical dense covariance (Cholesky once)."""

    def __init__(self, C_kept):
        self.n = C_kept.shape[0]
        self._cho = cho_factor(np.asarray(C_kept, dtype=np.float64),
                               lower=True)
        self.logdetC = 2.0 * float(np.sum(np.log(np.diag(self._cho[0]))))

    def solve(self, v):
        return cho_solve(self._cho, np.asarray(v, dtype=np.float64))

    def quad(self, r):
        """r^T C^{-1} r."""
        return float(np.asarray(r) @ self.solve(r))

    def marg_loglik(self, R_kept, X_kept, Lambda_diag, with_constants=True):
        """Exact ridge-marginalized GLS logL on kept pixels.

        logL = -1/2 R^T C^{-1} R + 1/2 b.a* - 1/2 logdet A
               [- 1/2 logdet C - n/2 log 2pi   if with_constants]
        with b = X^T C^{-1} R, A = X^T C^{-1} X + diag(Lambda).
        """
        R = np.asarray(R_kept, dtype=np.float64)
        X = np.asarray(X_kept, dtype=np.float64)
        CiR = self.solve(R)
        CiX = self.solve(X)
        b = X.T @ CiR
        A = X.T @ CiX + np.diag(np.asarray(Lambda_diag, dtype=np.float64))
        c, low = cho_factor(A, lower=True)
        a_star = cho_solve((c, low), b)
        logdetA = 2.0 * float(np.sum(np.log(np.diag(c))))
        logL = -0.5 * float(R @ CiR) + 0.5 * float(b @ a_star) - 0.5 * logdetA
        consts = -0.5 * self.logdetC - 0.5 * self.n * np.log(2.0 * np.pi)
        return dict(logL_data=logL,
                    logL=logL + (consts if with_constants else 0.0),
                    quad=float(R @ CiR), b=b, a_star=a_star,
                    logdetA=logdetA, logdetC=self.logdetC,
                    const_terms=float(consts), n=self.n)


# --------------------------------------------------------------------------- #
# whitened-covariance audit  Sigma_u = G_e C G_e^T
# --------------------------------------------------------------------------- #
def sigma_u_audit(G_e, eroded_idx, rho, err_map, shape, max_lag=3):
    """Exact Sigma_u = G_e C G_e^T diagnostics (dense; 80^2/130^2 grids).

    C uses the UNMASKED err_map (the physical noise; masked pixels leak into
    u only through the ~1e-10 down-weighting, which the erosion eliminates).

    Returns per-pixel Var(u) stats and, for every lag 0 < |d|_inf <= max_lag,
    the mean signed off-diagonal correlation over all eroded pixel pairs at
    that lag, plus the overall mean |corr|.
    """
    H, W = shape
    err = np.asarray(err_map, dtype=np.float64)
    n_e = G_e.shape[0]
    # B[:, j] = C @ G_e^T e_j ;  apply K by FFT convolution per column image
    Gt = G_e.T.tocsc()
    sig = err.reshape(-1)
    rho = np.asarray(rho, dtype=np.float64)
    B = np.empty((H * W, n_e))
    for j in range(n_e):
        col = np.asarray(Gt[:, j].todense()).reshape(-1) * sig
        img = col.reshape(H, W)
        conv = fftconvolve(img, rho, mode="same")     # rho symmetric: conv=corr
        B[:, j] = (conv.reshape(-1)) * sig
    Sigma = np.asarray(G_e @ B)                        # (n_e, n_e)

    var = np.diag(Sigma).copy()
    stats = dict(var_mean=float(var.mean()), var_min=float(var.min()),
                 var_max=float(var.max()), var_std=float(var.std()),
                 n_e=int(n_e))

    pos = np.full(H * W, -1, dtype=np.int64)
    pos[eroded_idx] = np.arange(n_e)
    ey = eroded_idx // W
    ex = eroded_idx % W
    lag_corr = {}
    abs_list = []
    for dy in range(-max_lag, max_lag + 1):
        for dx in range(-max_lag, max_lag + 1):
            if dy == 0 and dx == 0:
                continue
            if (dy, dx) < (0, 0):
                continue                     # symmetric: count each lag once
            ny, nx = ey + dy, ex + dx
            ok = (ny >= 0) & (ny < H) & (nx >= 0) & (nx < W)
            j = pos[np.where(ok, ny * W + nx, 0)]
            ok &= j >= 0
            i_idx = np.arange(n_e)[ok]
            j_idx = j[ok]
            if i_idx.size == 0:
                continue
            c = Sigma[i_idx, j_idx] / np.sqrt(var[i_idx] * var[j_idx])
            lag_corr[f"{dy},{dx}"] = dict(mean=float(c.mean()),
                                          mean_abs=float(np.abs(c).mean()),
                                          n_pairs=int(c.size))
            abs_list.append(np.abs(c))
    stats["offdiag_mean_abs"] = float(np.concatenate(abs_list).mean())
    stats["offdiag_worst_lag_mean"] = float(max(
        abs(v["mean"]) for v in lag_corr.values()))
    stats["lags"] = lag_corr
    return stats


# --------------------------------------------------------------------------- #
# FFT sampling + circulant-preconditioned CG (260^2 path)
# --------------------------------------------------------------------------- #
def stationary_spectrum(rho, grid):
    """Periodic-embedding spectrum of a centered odd-square kernel."""
    rho = np.asarray(rho, dtype=np.float64)
    L = (rho.shape[0] - 1) // 2
    if 2 * L + 1 > grid:
        raise ValueError("grid too small")
    emb = np.zeros((grid, grid))
    for dy in range(-L, L + 1):
        emb[dy % grid, (np.arange(-L, L + 1)) % grid] = rho[L + dy, :]
    return np.real(np.fft.fft2(emb))


def sample_stationary_batch(rho, shape, n_draws, rng, grid=512):
    """Exact FFT draws of a stationary field with autocovariance rho.

    Returns (n_draws, H, W) crops of periodic grid^2 fields whose circular
    covariance equals the embedded kernel (spectrum clipped at 0 if the
    truncated kernel is marginally indefinite; clip fraction is tiny and
    reported by the caller via stationary_spectrum).
    """
    H, W = shape
    S = np.clip(stationary_spectrum(rho, grid), 0.0, None)
    sqrtS = np.sqrt(S)
    out = np.empty((n_draws, H, W))
    for i in range(n_draws):
        wn = rng.standard_normal((grid, grid))
        x = np.real(np.fft.ifft2(sqrtS * np.fft.fft2(wn)))
        out[i] = x[:H, :W]
    return out


def circulant_cg_quad(rho, err_map, keep_mask, r_kept, tol=1e-9,
                      maxiter=400, grid=512):
    """R^T C^{-1} R on kept pixels via FFT-preconditioned CG (no dense factor).

    matvec: v -> [D^{1/2} K D^{1/2}]_kept v with K applied as an FFT
    convolution on the full grid (exact for the truncated stationary kernel).
    Preconditioner: the circulant inverse (mask ignored, sigma frozen at its
    median) applied via FFT with the spectrum floored at 1e-3 of its mean.
    Returns (quad, n_iter, rel_resid).
    """
    err = np.asarray(err_map, dtype=np.float64)
    keep = np.asarray(keep_mask, dtype=bool)
    H, W = err.shape
    kidx = np.flatnonzero(keep.reshape(-1))
    sig = err.reshape(-1)[kidx]
    rho = np.asarray(rho, dtype=np.float64)

    def matvec(v):
        full = np.zeros(H * W)
        full[kidx] = sig * v
        conv = fftconvolve(full.reshape(H, W), rho, mode="same")
        return sig * conv.reshape(-1)[kidx]

    S = np.maximum(stationary_spectrum(rho, grid),
                   1e-3 * float(np.abs(stationary_spectrum(rho, grid)).mean()))
    s_med = float(np.median(sig))

    def precond(v):
        full = np.zeros((grid, grid))
        img = np.zeros(H * W)
        img[kidx] = v / s_med
        full[:H, :W] = img.reshape(H, W)
        sol = np.real(np.fft.ifft2(np.fft.fft2(full) / S))[:H, :W]
        return sol.reshape(-1)[kidx] / s_med

    n = kidx.size
    A = LinearOperator((n, n), matvec=matvec)
    Mpre = LinearOperator((n, n), matvec=precond)
    it_count = [0]

    def cb(_):
        it_count[0] += 1

    x, info = cg(A, r_kept, rtol=tol, maxiter=maxiter, M=Mpre, callback=cb)
    resid = float(np.linalg.norm(matvec(x) - r_kept)
                  / max(np.linalg.norm(r_kept), 1e-300))
    return float(r_kept @ x), it_count[0], resid, info
