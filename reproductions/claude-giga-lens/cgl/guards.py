"""Hard guards encoding prior real incidents from the GIGA-Lens reproductions.

Every guard here exists because the failure it prevents actually happened and
cost real debugging time (foundry-i UPSTREAM_GIGALENS_ISSUE.md, PERLMUTTER_
CAMPAIGN.md Phase R; gu-2022 supersample cross-simulator mismatch). Driver
scripts import and call these instead of re-learning the incidents.
"""
from __future__ import annotations

import os

import numpy as np


class GuardError(RuntimeError):
    """A campaign guard tripped. The message names the prior incident."""


def require_x64() -> None:
    """Real-data marginalized posteriors need float64 (cond ~1e14).

    Incident: float32 floors ||grad|| at ~1.2e4 on the foundry-i reduced
    objective; HMC cannot mix. GIGALENS_X64=1 must be set before jax import.
    """
    import jax

    if not jax.config.jax_enable_x64:
        raise GuardError(
            "jax_enable_x64 is OFF. Set GIGALENS_X64=1 in the environment before "
            "importing cgl/jax modules (foundry-i: f32 grad floor ~1.2e4, no HMC)."
        )


def require_gpu() -> None:
    """The lensing likelihood must never run on CPU.

    Incident: grouped-conv XLA pathology makes CPU 216x slower than one L4
    (foundry-i). Unit tests on 16x16 toy grids are the sole sanctioned
    exception (they set CGL_ALLOW_CPU=1 via tests/conftest.py).
    """
    if os.environ.get("CGL_ALLOW_CPU") == "1":
        return
    import jax

    backend = jax.default_backend()
    if backend != "gpu":
        raise GuardError(
            f"default jax backend is {backend!r}, not 'gpu'. The lensing likelihood "
            "is BANNED on CPU (216x slower than one L4). Set CGL_ALLOW_CPU=1 only "
            "for 16x16 toy-grid unit tests."
        )


def require_single_device(allow_multi: bool = False) -> None:
    """Refuse to run sampling with >1 visible device unless explicitly allowed.

    Incident: gigalens HMC() pmaps GBTLA over ALL visible devices; on the
    10-device phoenix host this is a multi-hour 'won't compile' hang (Bug 1).
    The proven recipe is a single-device batched sample_chain.
    """
    if allow_multi:
        return
    import jax

    n = len(jax.devices())
    if n != 1:
        raise GuardError(
            f"{n} jax devices visible. Pin one GPU (CUDA_VISIBLE_DEVICES=<id>, "
            "CUDA_DEVICE_ORDER=PCI_BUS_ID, XLA_FLAGS=--xla_gpu_autotune_level=0) "
            "or pass --allow-multidevice (foundry-i Bug 1: pmap-over-all-devices hang)."
        )


def assert_psf_sampling(psf_pixel_scale: float, delta_pix: float,
                        atol: float = 1e-9) -> None:
    """PSF kernels handed to the simulator MUST be sampled at delta_pix.

    Incident (twice): lenstronomy subgrid_kernel assumes the kernel is sampled
    at delta_pix and upsamples internally by `supersample`. foundry-i R0c ran
    every native fit with a 2x-broadened effective PSF (chi2_nu floor 3.4 that
    was really 1.05); gu-2022 hit the same convention family at supersample=2.
    Refuse supersampled kernels at the door.
    """
    if abs(psf_pixel_scale - delta_pix) > atol:
        raise GuardError(
            f"PSF kernel sampled at {psf_pixel_scale}\" but delta_pix={delta_pix}\". "
            "subgrid_kernel upsamples internally; passing a supersampled kernel "
            "double-applies the refinement (foundry-i R0c: 2x-broadened PSF, "
            "gamma-ESS 259 vs 5714). Resample the kernel to delta_pix first "
            "(see foundry-i/40d_fix_v2_psf.py)."
        )


def floor_svi_covariance(cov: np.ndarray, rel_floor: float = 1e-10):
    """Eigenvalue-floor an SVI covariance; return (cov_reg, chol) in float64.

    Incident (Bug 2): gigalens builds HMC momentum as
    MultivariateNormalFullCovariance(inv(qz.covariance())) which NaNs on the
    rank-deficient covariance a short SVI schedule produces (rank 53/74 at
    1500 steps). Floor at rel_floor * lambda_max, rebuild, Cholesky.
    """
    cov64 = np.asarray(cov, dtype=np.float64)
    cov64 = 0.5 * (cov64 + cov64.T)
    w, v = np.linalg.eigh(cov64)
    floor = rel_floor * float(w.max())
    n_floored = int((w < floor).sum())
    w = np.maximum(w, floor)
    cov_reg = (v * w) @ v.T
    cov_reg = 0.5 * (cov_reg + cov_reg.T)
    chol = np.linalg.cholesky(cov_reg)
    return cov_reg, chol, n_floored


def check_svi_schedule(n_steps: int, n_params: int, min_steps_per_param: int = 150) -> None:
    """A too-short SVI schedule silently yields a rank-deficient covariance.

    Incident: 1500 steps -> rank 53/74 variational covariance that silently
    broke the HMC preconditioner; the working schedule was 15000 steps for 74
    params (~200 steps/param). Warn-level guard: raises below the floor.
    """
    if n_steps < min_steps_per_param * n_params:
        raise GuardError(
            f"SVI schedule {n_steps} steps for {n_params} params is below the "
            f"{min_steps_per_param}/param floor (foundry-i: 1500 steps -> rank-53/74 "
            "covariance, silently broken HMC preconditioner; 15000 steps -> full rank)."
        )


def assert_model_subtracted_sky(meta: dict) -> None:
    """Noise kernels/calibrations must come from MODEL-SUBTRACTED residuals.

    Incident: the celebrated chi2_nu=0.451 was an artifact of calibrating the
    sky sigma on raw-image fluctuations that were ~70% diffuse lens-wing flux
    (honest recalibration: 0.92). Any noise-kernel artifact must carry
    meta['model_subtracted'] == True.
    """
    if not meta.get("model_subtracted", False):
        raise GuardError(
            "noise-kernel/calibration artifact lacks model_subtracted=True. "
            "Raw-sky ACF/variance is wing-contaminated (foundry-i R0: fake "
            "chi2=0.451 vs honest 0.92). Fit kernels on model-subtracted residuals."
        )
