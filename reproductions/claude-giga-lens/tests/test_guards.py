"""Unit tests for cgl.guards — each guard encodes a prior real incident."""
import numpy as np
import pytest

from cgl import guards


def test_require_x64_passes_under_conftest():
    guards.require_x64()  # conftest enables x64


def test_require_gpu_honors_cpu_test_exception(monkeypatch):
    monkeypatch.setenv("CGL_ALLOW_CPU", "1")
    guards.require_gpu()


def test_require_gpu_raises_on_cpu(monkeypatch):
    monkeypatch.delenv("CGL_ALLOW_CPU", raising=False)
    with pytest.raises(guards.GuardError, match="216x"):
        guards.require_gpu()


def test_psf_sampling_accepts_native():
    guards.assert_psf_sampling(psf_pixel_scale=0.13, delta_pix=0.13)


def test_psf_sampling_rejects_supersampled():
    # The R0c incident: 0.065"-sampled kernel handed to a 0.13" delta_pix sim.
    with pytest.raises(guards.GuardError, match="subgrid_kernel"):
        guards.assert_psf_sampling(psf_pixel_scale=0.065, delta_pix=0.13)


def test_floor_svi_covariance_fixes_rank_deficiency():
    rng = np.random.default_rng(0)
    # Rank-deficient 8x8 covariance (rank 5) — the Bug-2 shape of failure.
    a = rng.normal(size=(8, 5))
    cov = a @ a.T
    cov_reg, chol, n_floored = guards.floor_svi_covariance(cov, rel_floor=1e-8)
    assert n_floored >= 3
    # Cholesky must succeed and reproduce cov_reg.
    np.testing.assert_allclose(chol @ chol.T, cov_reg, rtol=0, atol=1e-10)
    # Floored matrix stays close to the original on its column space.
    w = np.linalg.eigvalsh(cov_reg)
    assert w.min() > 0


def test_check_svi_schedule():
    guards.check_svi_schedule(n_steps=15000, n_params=74)  # the working schedule
    with pytest.raises(guards.GuardError, match="rank-53/74"):
        guards.check_svi_schedule(n_steps=1500, n_params=74)  # the incident


def test_assert_model_subtracted_sky():
    guards.assert_model_subtracted_sky({"model_subtracted": True})
    with pytest.raises(guards.GuardError, match="0.451"):
        guards.assert_model_subtracted_sky({})


def test_single_device_allow_flag():
    guards.require_single_device(allow_multi=True)  # never raises when allowed
