"""Process-environment plumbing for zoo driver scripts (20/21/22).

One process = one target dtype (x64 cannot be mixed after jax import), so
every driver either sets its own env from the registry's STATIC dtype before
importing jax, or spawns one subprocess per target with the right env.

All functions here touch os.environ / subprocess only -- importing this
module never imports jax.
"""
from __future__ import annotations

import json
import os
import sys
from typing import Optional

import numpy as np

DEFAULT_GPU = "9"   # campaign L4; GPU 8 belongs to another agent
_AUTOTUNE = "--xla_gpu_autotune_level=0"
_NOFUSION = "--xla_disable_hlo_passes=priority-fusion"
_NOTRITON = "--xla_gpu_enable_triton_gemm=false"


def _xla_flags(existing: str, dtype: str) -> str:
    flags = existing or ""
    if "--xla_gpu_autotune_level" not in flags:
        flags += f" {_AUTOTUNE}"
    # jaxlib 0.6.2 priority-fusion livelock: f64 random.normal fused with a
    # reduction (CAMPAIGN.md P0). Harmless pass-disable for every x64 process.
    if dtype == "float64" and "--xla_disable_hlo_passes" not in flags:
        flags += f" {_NOFUSION}"
    # STACK DEFECT (found P2a, 2026-07-06): jaxlib 0.6.2 XLA triton GEMM
    # aborts with 'CANCELLED: Too small divisible part of the contracting
    # dimension' on tiny f32 dots (e.g. the dim-2 SVI scale_tril matmuls of
    # t0_mix2) under --xla_gpu_autotune_level=0 on the L4. f64 is immune
    # (cuBLAS path). Fall back to cuBLAS everywhere in zoo processes.
    if "--xla_gpu_enable_triton_gemm" not in flags:
        flags += f" {_NOTRITON}"
    return flags.strip()


def setup_process_env(dtype: str, gpu: Optional[str] = None) -> None:
    """Set GIGALENS_X64 / GPU pin / XLA flags. MUST run before jax import."""
    if "jax" in sys.modules:
        import jax
        x64 = bool(jax.config.jax_enable_x64)
        want = dtype == "float64"
        if x64 != want:
            raise RuntimeError(
                f"jax already imported with x64={x64} but target dtype is "
                f"{dtype}; env must be set before the first jax import "
                "(single-target-per-process).")
        return
    x64 = "1" if dtype == "float64" else "0"
    os.environ["GIGALENS_X64"] = x64
    os.environ["JAX_ENABLE_X64"] = x64   # honored by jax itself at import;
    # covers targets that never import cgl.likelihood (T0 synthetic).
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    else:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", DEFAULT_GPU)
    os.environ["XLA_FLAGS"] = _xla_flags(os.environ.get("XLA_FLAGS", ""), dtype)


def child_env(dtype: str, gpu: Optional[str] = None) -> dict:
    """Env dict for a per-target subprocess."""
    env = dict(os.environ)
    x64 = "1" if dtype == "float64" else "0"
    env["GIGALENS_X64"] = x64
    env["JAX_ENABLE_X64"] = x64
    env.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    env["CUDA_VISIBLE_DEVICES"] = str(gpu) if gpu is not None else \
        env.get("CUDA_VISIBLE_DEVICES", DEFAULT_GPU)
    env["XLA_FLAGS"] = _xla_flags(env.get("XLA_FLAGS", ""), dtype)
    return env


def setup_jax_cache(repro_dir) -> None:
    """Persistent compilation cache (after jax import; same dir as P0)."""
    import jax

    jax.config.update("jax_compilation_cache_dir",
                      str(repro_dir / ".jax_cache"))
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)


# --------------------------------------------------------------------------- #
# freeze reference points (deterministic; values are ALSO stored explicitly
# in zoo_freeze.json, so readers never regenerate)
# --------------------------------------------------------------------------- #
FREEZE_SEED0 = 20260706
N_FREEZE_POINTS = 3


def make_freeze_points(target) -> np.ndarray:
    """(3, dim) reference z-points: anchor + scale * standard_normal, rng
    seeded per point index. anchor = init.map_z (scale 0.1) when available,
    else 0 (scale 0.5)."""
    if target.init.map_z is not None:
        anchor, scale = np.asarray(target.init.map_z, dtype=np.float64), 0.1
    else:
        anchor, scale = np.zeros(target.dim), 0.5
    pts = []
    for k in range(N_FREEZE_POINTS):
        rng = np.random.default_rng(FREEZE_SEED0 + k)
        pts.append(anchor + scale * rng.standard_normal(target.dim))
    return np.asarray(pts)


def load_freeze(path) -> dict:
    return json.loads(path.read_text())


def freeze_entry_for(freeze: dict, name: str) -> dict:
    if name not in freeze.get("targets", {}):
        raise KeyError(
            f"target {name!r} missing from zoo freeze {sorted(freeze.get('targets', {}))}; "
            "run 20_build_zoo.py first")
    return freeze["targets"][name]
