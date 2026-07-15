import functools
from typing import List, Dict

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit
from jax import lax
from lenstronomy.Util.kernel_util import subgrid_kernel
from objax.constants import ConvPadding
from objax.functional import average_pool_2d

import gigalens.model
import gigalens.simulator
    
def _next_fast_len(n):
    """Smallest 7-smooth integer >= n (2^a 3^b 5^c 7^d: cuFFT and pocketfft both
    have fast radix kernels for these; larger prime factors fall to slow generic
    paths). Runs at trace time only (shapes are static under jit)."""
    m = n
    while True:
        k = m
        for p in (2, 3, 5, 7):
            while k % p == 0:
                k //= p
        if k == 1:
            return m
        m += 1


def _next_fast_len_multiple(n, mult):
    """Smallest 7-smooth integer >= n that is also a multiple of ``mult``.

    The fused conv+pool transform length must be divisible by the pooling
    stride so the spectral fold N -> N/ss is integral. ``mult`` itself must be
    7-smooth or no such integer exists (loud guard; ss is 2 or 4 in practice)."""
    k = mult
    for p in (2, 3, 5, 7):
        while k % p == 0:
            k //= p
    if k != 1:
        raise ValueError(f"pool stride {mult} has a prime factor > 7; no 7-smooth "
                         "FFT length is divisible by it.")
    m = n + (-n) % mult
    while True:
        k = m
        for p in (2, 3, 5, 7):
            while k % p == 0:
                k //= p
        if k == 1:
            return m
        m += mult


def _fold_indices(Ny, Nx, ss, oy, ox):
    """Trace-time constants for the spectral fold (see _rfft_convolve_pool_same).

    Returns gather indices into the rfft2 half-spectrum (gy, gx), the
    conjugation mask for Hermitian-reconstructed bins, and the phase/fold
    weights, all shaped (My, Mx//2+1, ss, ss)."""
    My, Mx = Ny // ss, Nx // ss
    fyp = np.arange(My)[:, None, None, None]
    fxp = np.arange(Mx // 2 + 1)[None, :, None, None]
    ry = np.arange(ss)[None, None, :, None]
    rx = np.arange(ss)[None, None, None, :]
    fy = fyp + ry * My                      # in [0, Ny)
    fx = fxp + rx * Mx                      # in [0, Nx)
    use_conj = fx > Nx // 2                 # beyond the stored half-spectrum
    gy = np.where(use_conj, (Ny - fy) % Ny, fy)
    gx = np.where(use_conj, Nx - fx, fx)
    phase = np.exp(2j * np.pi * (fy * oy / Ny + fx * ox / Nx)) / ss ** 2
    return gy, gx, use_conj, phase


def _rfft_convolve_pool_same(img, kernel, ss):
    """'same' PSF convolution + ss x ss average pooling in ONE Fourier pass,
    with the inverse transform evaluated DIRECTLY on the pooled grid.

    Identity: with the ss-box folded into the kernel (g = k * box/ss^2),

        pooled[p, q] = (img (*) g)[oy + p*ss, ox + q*ss],
        oy = (kh-1)//2 + ss - 1,  ox = (kw-1)//2 + ss - 1

    (the 'same' crop offset composed with the causal box-convolution index; the
    average_pool blocks [p*ss, (p+1)*ss) are exactly the box sums). Sampling an
    N-point circular convolution at stride ss equals an M = N/ss-point inverse
    transform of the phase-shifted ss-fold-summed spectrum, per axis:

        B[f'] = (1/ss) * sum_r A[f' + r*M] * exp(+2i*pi*(f'+r*M)*o/N)

    The forward stays rfft2 (real input); full-spectrum bins needed by the fold
    along the last axis are reconstructed from Hermitian symmetry
    A[fy, fx] = conj(A[(-fy) % Ny, (-fx) % Nx]) via trace-time gather indices.
    This removes the LARGE (N-point) inverse transform, the separate pooling
    pass, and the full-resolution crop intermediate: the c2r shrinks by ss^2.
    Exactness gate: wip/validate_fold_stack.py (vs conv -> crop -> average_pool;
    f64 rel <= 1e-10, production-f32 chi2 rel <= 2e-5).

    img: (..., H, W) real, H and W divisible by ss. kernel: (kh, kw).
    Returns (..., H//ss, W//ss).
    """
    H, W = img.shape[-2:]
    kh, kw = kernel.shape[-2:]
    assert H % ss == 0 and W % ss == 0, (H, W, ss)
    Ny = _next_fast_len_multiple(H + kh + ss - 2, ss)
    Nx = _next_fast_len_multiple(W + kw + ss - 2, ss)
    My, Mx = Ny // ss, Nx // ss
    oy = (kh - 1) // 2 + ss - 1
    ox = (kw - 1) // 2 + ss - 1

    cplx = jnp.complex64 if img.dtype == jnp.float32 else jnp.complex128
    # box/ss^2: the AVERAGE-pool normalization lives here; the /ss^2 inside the
    # fold phase (see _fold_indices) is the spectral-fold normalization — two
    # distinct factors, do not merge them.
    box = jnp.full((ss, ss), 1.0 / ss ** 2, dtype=img.dtype)
    khat = (jnp.fft.rfft2(kernel, s=(Ny, Nx))
            * jnp.fft.rfft2(box, s=(Ny, Nx))).astype(cplx)  # constants: XLA folds
    G = jnp.fft.rfft2(img, s=(Ny, Nx)) * khat

    gy, gx, use_conj, phase = _fold_indices(Ny, Nx, ss, oy, ox)
    vals = G[..., gy, gx]
    vals = jnp.where(use_conj, jnp.conj(vals), vals)
    B = jnp.sum(vals * jnp.asarray(phase, dtype=cplx), axis=(-2, -1))
    out = jnp.fft.irfft2(B, s=(My, Mx))
    return out[..., : H // ss, : W // ss]


def _convolve_pool_components(img, flat_kernel, ss, source_light_depths=None):
    """Fused variant of ``_convolve_components`` + average pooling for the
    single-shared-PSF case (``SceneSimulator`` always builds flat_kernel of
    shape (1, 1, kh, kw)). Multi-kernel inputs RAISE: those branches would be
    unreachable dead code today, are exercised by no gate, and a first draft
    contained a latent shape divergence vs the legacy path (2026-07-09 grading)
    — route such callers through the unfused path (_FUSE_CONV_POOL=False) or
    add the branch WITH its own equivalence gate. Output spatial dims are the
    pooled (detector) grid."""
    if flat_kernel is None:
        raise ValueError("fused conv+pool requires a PSF kernel; pool separately.")
    if flat_kernel.shape[0] != 1:
        raise NotImplementedError(
            "fused conv+pool is only validated for a single shared PSF "
            "(flat_kernel shape (1,1,kh,kw)); for multiple kernels use the "
            "unfused path (scene_simulator._FUSE_CONV_POOL = False) or add a "
            "gated multi-kernel branch.")
    return _rfft_convolve_pool_same(img, flat_kernel[0, 0], ss)


def _manual_fftconvolve_same_complex(img, kernel):
    """
    Performs exact convolution using native JAX FFTs.
    Uses full complex FFTs to avoid JAX VJP (autodiff) bugs with rfft2.
    generated by Gemini pro 3.1

    Retained as the reference implementation for the rfft2 equivalence gate
    (wip/validate_fast_lstsq.py) and as the fallback if the rfft2 VJP ever
    regresses; the production path is ``_manual_fftconvolve_same`` below.
    """
    H, W = img.shape[-2:]
    kh, kw = kernel.shape[-2:]

    # Target shape for linear convolution
    shape = (H + kh - 1, W + kw - 1)

    # Use full complex FFT2 to keep the gradient engine happy
    im_fft = jnp.fft.fft2(img, s=shape, axes=(-2, -1))
    k_fft = jnp.fft.fft2(kernel, s=shape, axes=(-2, -1))

    # Multiply in frequency domain, IFFT, and extract the real component
    out = jnp.fft.ifft2(im_fft * k_fft, s=shape, axes=(-2, -1)).real

    # Slice back to the "same" image shape
    start_y = (kh - 1) // 2
    start_x = (kw - 1) // 2
    return out[..., start_y : start_y + H, start_x : start_x + W]


def _manual_fftconvolve_same(img, kernel):
    """Exact 'same' linear convolution via REAL FFTs on 7-smooth padded sizes.

    Numerically the same linear convolution as the full-complex version above:
    padding beyond the linear length (H+kh-1, W+kw-1) only appends zeros, which
    cannot change the cropped 'same' region, and rfft2/irfft2 of real inputs is
    the same transform restricted to the real subspace. Real FFTs halve the
    transform work and the frequency-domain bytes; 7-smooth sizes avoid slow
    prime-length paths. The original full-complex code dates from a JAX rfft2
    VJP bug; on JAX 0.10 the rfft2 VJP is exercised directly by the pre-registered
    gate in wip/validate_fast_lstsq.py (falsifier: any grad NaN or rel err
    > 1e-10 vs the complex path -> revert to _manual_fftconvolve_same_complex).
    The kernel FFT is left to XLA constant folding (the kernel is a captured
    constant under jit).
    """
    H, W = img.shape[-2:]
    kh, kw = kernel.shape[-2:]
    shape = (_next_fast_len(H + kh - 1), _next_fast_len(W + kw - 1))

    im_fft = jnp.fft.rfft2(img, s=shape, axes=(-2, -1))
    k_fft = jnp.fft.rfft2(kernel, s=shape, axes=(-2, -1))
    out = jnp.fft.irfft2(im_fft * k_fft, s=shape, axes=(-2, -1))

    start_y = (kh - 1) // 2
    start_x = (kw - 1) // 2
    return out[..., start_y : start_y + H, start_x : start_x + W]

def _convolve_components(img, flat_kernel, source_light_depths=None):
    """
    High-performance 64-bit convolution handling both single and multiple unique PSFs.
    generated by Gemini Pro 3.1
    """
    if flat_kernel is None:
        return img
    
    # Case 1: Single PSF shared across all components globally
    if flat_kernel.shape[0] == 1:
        # Extract 2D kernel and expand to allow broadcasting over bs and depth axes
        k = flat_kernel[0, 0][jnp.newaxis, jnp.newaxis, :, :] 
        return _manual_fftconvolve_same(img, k)
        
    # Case 2: Multiple PSFs (One distinct PSF assigned per source plane / band)
    else:
        if source_light_depths is None:
            # Fallback/Used when depth == num_kernels
            img_t = jnp.transpose(img, (1, 0, 2, 3)) # (num_kernels, bs, h, w)
            kernels_4d = flat_kernel[:, 0, :, :][:, jnp.newaxis, :, :] # (num_kernels, 1, kh, kw)
            
            convolved_t = _manual_fftconvolve_same(img_t, kernels_4d)
            return jnp.transpose(convolved_t, (1, 0, 2, 3))
            
        else:
            outputs = []
            start = 0
            
            for k_idx, d in enumerate(source_light_depths):
                if d == 0:
                    continue
                end = start + d
                sub_img = img[:, start:end, :, :]  # (bs, d, h, w)
                
                # Extract specific 2D kernel and expand to broadcast over bs and d
                k = flat_kernel[k_idx, 0][jnp.newaxis, jnp.newaxis, :, :] 
                convolved_sub = _manual_fftconvolve_same(sub_img, k)
                outputs.append(convolved_sub)
                
                start = end
                
            return jnp.concatenate(outputs, axis=1)

def _regularize_gram(gram, jitter_scale=1e-6):
    gram = 0.5 * (gram + jnp.swapaxes(gram, -1, -2))
    diag_mean = jnp.mean(jnp.diagonal(gram, axis1=-2, axis2=-1), axis=-1, keepdims=True)
    jitter = jitter_scale * jnp.maximum(diag_mean, 1.0)
    return gram + jitter[..., jnp.newaxis] * jnp.eye(gram.shape[-1], dtype=gram.dtype)


def _solve_normal_eq_with_fallback(gram, rhs):
    # NOTE: We use jnp.linalg.solve (LU with partial pivoting) rather than a
    # Cholesky-based solver because we need a well-conditioned VJP, not just a
    # well-conditioned forward pass. For very large shapelet bases (n_max>~25)
    # the normal-equation gram matrix X^T X has cond ~ 1e16 in float32, and
    # even though the small diagonal jitter is enough to make the forward
    # Cholesky finite, the Cholesky VJP divides by L_{ii}^2 and explodes to
    # NaN. The lstsq fallback path has the same problem (SVD VJP with
    # near-zero singular values). jnp.linalg.solve has a backward pass that is
    # just another solve against the same matrix, so it stays finite anywhere
    # the forward stays finite. (Vela n_max=30 modeling: NaN gradients used to
    # appear in 6-7 of 8 random MAP starts; with this solver they go to 0.)
    def solve_one(gram_i, rhs_i):
        gram_i = _regularize_gram(gram_i)
        return jnp.linalg.solve(gram_i, rhs_i)

    if gram.shape[0] == 1:
        return solve_one(gram[0], rhs[0])[jnp.newaxis, ...]

    return jax.vmap(solve_one)(gram, rhs)


# The ``LensSimulator`` class and its ``multiband_simulate`` / ``multiband_lstsq_simulate``
# methods were removed with the old gigalens API; the scene path uses
# ``gigalens.jax.scene_simulator.SceneSimulator``. The module-level helpers above
# (_manual_fftconvolve_same / _convolve_components / _regularize_gram /
# _solve_normal_eq_with_fallback) are PRESERVED — scene_simulator and voronoi import them.
# An import-safe stub is kept so a module that merely imports the name still imports;
# constructing it raises. Restore from git b82397c if the old class is needed.


class LensSimulator:
    """Removed with the old gigalens API — use ``SceneSimulator``."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "gigalens.jax.simulator.LensSimulator was removed with the old gigalens API; "
            "use gigalens.jax.scene_simulator.SceneSimulator. Restore from git b82397c "
            "if needed.")
