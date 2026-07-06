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
