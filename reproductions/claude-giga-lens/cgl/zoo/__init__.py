"""Lens-posterior zoo registry.

Import-light on purpose (stdlib only): `list_targets()` returns STATIC
metadata without touching jax; `get_target(name)` imports the target module
lazily and builds it. dtype discipline: one process cannot mix x64 states,
so callers must consult `info["dtype"]` and set GIGALENS_X64 BEFORE the
first jax import (single-target-per-process; see cgl/zoo/api.py).
"""
from __future__ import annotations

import dataclasses
import importlib
from typing import Optional

__all__ = ["get_target", "get_target_info", "list_targets", "TargetInfo"]


@dataclasses.dataclass(frozen=True)
class TargetInfo:
    name: str
    tier: str
    dim: int
    dtype: str            # "float32" | "float64"
    noise_model: str
    module: str
    factory: str
    available: bool
    note: str = ""
    kwargs: tuple = ()    # ((key, value), ...) for the factory


def _entries():
    e = [
        TargetInfo("t0_mix2", "T0", 2, "float32", "diag",
                   "cgl.zoo.synthetic", "build_mix2", True,
                   "2-mode Gaussian mixture w=0.8/0.2, analytic logZ/moments"),
        TargetInfo("t0_mix2_f64", "T0", 2, "float64", "diag",
                   "cgl.zoo.synthetic", "build_mix2_f64", True,
                   "float64 variant (f64 kernel-path / MCLMC testing)"),
        TargetInfo("t0_mix22", "T0", 22, "float32", "diag",
                   "cgl.zoo.synthetic", "build_mix22", True,
                   "22-D mixture at gu-2022 dimensionality"),
        TargetInfo("t0_funnel10", "T0", 10, "float32", "diag",
                   "cgl.zoo.synthetic", "build_funnel10", True,
                   "Neal funnel, logZ=0 exact"),
        TargetInfo("t0_illcond46", "T0", 46, "float64", "diag",
                   "cgl.zoo.synthetic", "build_illcond46", True,
                   "diagonal cond-1e14 Gaussian (mimics T2)"),
    ]
    for i in range(12):
        e.append(TargetInfo(
            f"gu2022_sys{i:03d}", "T1", 22, "float32", "diag",
            "cgl.zoo.gu2022", "build", True,
            "gu-2022 mock, ForwardProbModel 80x80 ss2 (GPU)",
            kwargs=(("idx", i),)))
    e.append(TargetInfo(
        "foundry_marg46", "T2", 46, "float64", "diag",
        "cgl.zoo.foundry_marg", "build", True,
        "foundry-i 46-dim ridge-marginalized real posterior, cond~1e14 (GPU)"))
    e.append(TargetInfo(
        "foundry_v3b74", "T3", 74, "float32", "diag",
        "cgl.zoo.foundry_v3b", "build", True,
        "foundry-i 74-dim bimodal v3b posterior (GPU)"))
    e.append(TargetInfo(
        "euclid_q1", "T4", -1, "float32", "diag",
        "cgl.zoo.euclid", "build", False,
        "P3 scope; NOT available in P2a"))
    return e


_REGISTRY = {t.name: t for t in _entries()}


def list_targets(available_only: bool = False):
    """Static registry metadata (list of dicts); NO jax work."""
    out = []
    for t in _REGISTRY.values():
        if available_only and not t.available:
            continue
        out.append(dataclasses.asdict(t))
    return out


def get_target_info(name: str) -> TargetInfo:
    if name not in _REGISTRY:
        raise KeyError(
            f"unknown zoo target {name!r}; known: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def get_target(name: str):
    """Build and return the LensPosterior (lazy module import; jax work)."""
    info = get_target_info(name)
    mod = importlib.import_module(info.module)
    factory = getattr(mod, info.factory)
    target = factory(**dict(info.kwargs))
    assert target.name == name and target.dim == info.dim \
        and target.dtype == info.dtype, \
        f"registry/static mismatch for {name}: built " \
        f"({target.name},{target.dim},{target.dtype})"
    return target
