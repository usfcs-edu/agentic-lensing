#!/usr/bin/env python3
"""Change-3 implementability probe (2026-07-09): is the FFT 1/N normalization a
user-fusible op? Compile grad of irfft2(rfft2(x)*K) and inspect optimized HLO.

Result (first run, jax 0.10.0.dev20260709, A100 login node):
  FFT-related HLO lines: 6
    fft(...IRFFT...) / fft(...IFFT...) pairs with input_transpose_fusion /
    wrapped_transpose between them (irfft2 lowers SEPARABLY: IFFT on one axis,
    IRFFT on the other, transposes in between)
  candidate scale lines: 0
VERDICT: the 1/N lives INSIDE the backend lowering of the fft HLO op (the
cuBLAS scal kernels seen in C-14 traces); there is NO user-visible multiply to
fold constants into, and XLA does not move K/N across an fft op algebraically.
Change 3 is NOT IMPLEMENTABLE at the JAX/XLA-API level. Side finding: the
separable-lowering transposes explain part of C-14's copy/transpose class.
"""
import os
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import jax
import jax.numpy as jnp


def main():
    rng = np.random.default_rng(0)
    K = jnp.asarray(rng.normal(size=(220, 111)) + 1j * rng.normal(size=(220, 111)))

    def f(x):
        return jnp.sum(jnp.fft.irfft2(jnp.fft.rfft2(x, s=(220, 220)) * K,
                                      s=(220, 220)) ** 2)

    g = jax.jit(jax.grad(f))
    txt = g.lower(jnp.ones((4, 220, 220))).compile().as_text()
    fft_lines = [l.strip()[:150] for l in txt.splitlines() if "fft(" in l]
    print("FFT-related HLO lines:", len(fft_lines))
    for l in fft_lines:
        print(" ", l)
    scale = [l.strip()[:150] for l in txt.splitlines()
             if "multiply" in l and "constant" in l]
    print("candidate scale lines:", len(scale))
    for l in scale[:6]:
        print(" ", l)


if __name__ == "__main__":
    main()
