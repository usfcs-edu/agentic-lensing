"""Parity-harness wrapper (GPU) + CPU-friendly marg_loglik reference test.

The GPU test shells out to 01_parity_harness.py in a fresh process so jax is
initialized with the right env (x64 + one pinned GPU) and the pytest process
itself never claims GPU memory. ./00_run_tests.sh --gpu provides
CUDA_VISIBLE_DEVICES / GIGALENS_X64 / XLA_FLAGS; sensible defaults are set
here for direct `pytest -m gpu` invocations.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPRO = Path(__file__).resolve().parent.parent
HARNESS = REPRO / "01_parity_harness.py"


@pytest.mark.gpu
def test_parity_harness_all_hard_gates_pass():
    env = os.environ.copy()
    env["GIGALENS_X64"] = "1"
    env.setdefault("CUDA_VISIBLE_DEVICES", env.get("CGL_GPU", "8"))
    env.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    env.setdefault("XLA_FLAGS", "--xla_gpu_autotune_level=0")
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    env.pop("JAX_PLATFORMS", None)   # never inherit the CPU-test forcing
    env.pop("CGL_ALLOW_CPU", None)

    r = subprocess.run([sys.executable, str(HARNESS)], env=env, cwd=str(REPRO),
                       capture_output=True, text=True, timeout=3600)
    tail = "\n".join(r.stdout.splitlines()[-30:])
    assert r.returncode == 0, (
        f"parity harness exited {r.returncode}\n--- stdout tail ---\n{tail}"
        f"\n--- stderr tail ---\n{chr(10).join(r.stderr.splitlines()[-30:])}")

    report = json.loads((REPRO / "data" / "parity_report.json").read_text())
    assert report["all_hard_gates_pass"] is True
    for k in ("A", "B", "C", "D", "E"):
        g = report["gates"][k]
        assert g["pass"], f"gate {k}: achieved {g['achieved']} > {g['threshold']}"


def test_marg_loglik_numpy_reference_small():
    """CPU-friendly: marg_loglik vs a pure-numpy reference on random small
    matrices (tolerance 1e-12)."""
    import jax.numpy as jnp

    from cgl.marg import marg_loglik

    rng = np.random.default_rng(42)
    for n_pix, k in [(32, 4), (48, 6), (64, 7)]:
        X = rng.standard_normal((n_pix, k))
        R = rng.standard_normal(n_pix)
        lam = rng.uniform(0.4, 3.0, k)
        logL, a_star, logdetA = marg_loglik(
            jnp.asarray(X), jnp.asarray(R), jnp.asarray(lam))

        b = X.T @ R
        A = X.T @ X + np.diag(lam)
        a_np = np.linalg.solve(A, b)
        sign, ld_np = np.linalg.slogdet(A)
        assert sign > 0
        logL_np = -0.5 * R @ R + 0.5 * b @ a_np - 0.5 * ld_np

        np.testing.assert_allclose(np.asarray(a_star), a_np, rtol=0, atol=1e-12)
        assert abs(float(logdetA) - ld_np) <= 1e-12 * max(1.0, abs(ld_np))
        assert abs(float(logL) - logL_np) <= 1e-12 * max(1.0, abs(logL_np))
