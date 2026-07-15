#!/usr/bin/env python3
"""CPU brute-force unit test for _rfft_convolve_pool_same (grading item 1,
2026-07-09): fused 'same' conv + ss-average-pool vs direct linear convolution
+ crop + block-mean, over odd/even kernels, rectangular grids, ss in {2,4},
with batch dims. Run on CPU (JAX_PLATFORMS=cpu), f64.

First run (2026-07-09, jax 0.10.0.dev20260709, container jax-2026-04-13):
  H16 W16 k5x5 ss2: rel 1.07e-15
  H24 W16 k7x5 ss4: rel 1.95e-15
  H16 W24 k5x7 ss2: rel 9.23e-16
  H32 W32 k9x9 ss4: rel 1.54e-15
  H20 W20 k4x6 ss2: rel 7.56e-16
  H28 W28 k6x4 ss4: rel 1.81e-15
  WORST 1.95e-15 PASS
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import jax.numpy as jnp

from gigalens.jax.simulator import _rfft_convolve_pool_same


def brute(img, k, ss):
    H, W = img.shape[-2:]
    kh, kw = k.shape
    full = np.zeros(img.shape[:-2] + (H + kh - 1, W + kw - 1))
    for idx in np.ndindex(img.shape[:-2]):
        for i in range(kh):
            for j in range(kw):
                full[idx][i:i + H, j:j + W] += img[idx] * k[i, j]
    cy, cx = (kh - 1) // 2, (kw - 1) // 2
    same = full[..., cy:cy + H, cx:cx + W]
    return same.reshape(*img.shape[:-2], H // ss, ss, W // ss, ss).mean(axis=(-3, -1))


def main():
    rng = np.random.default_rng(0)
    worst = 0.0
    cases = [(16, 16, 5, 5, 2), (24, 16, 7, 5, 4), (16, 24, 5, 7, 2),
             (32, 32, 9, 9, 4), (20, 20, 4, 6, 2), (28, 28, 6, 4, 4)]
    for (H, W, kh, kw, ss) in cases:
        img = rng.normal(size=(2, 3, H, W))
        k = rng.normal(size=(kh, kw))
        k = k / k.sum()
        got = np.asarray(_rfft_convolve_pool_same(jnp.asarray(img), jnp.asarray(k), ss))
        want = brute(img, k, ss)
        err = np.max(np.abs(got - want)) / np.max(np.abs(want))
        worst = max(worst, err)
        print(f"H{H} W{W} k{kh}x{kw} ss{ss}: rel {err:.2e}")
    print("WORST", f"{worst:.2e}", "PASS" if worst < 1e-12 else "FAIL")
    assert worst < 1e-12


if __name__ == "__main__":
    main()
