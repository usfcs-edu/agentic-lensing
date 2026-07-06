"""Sampler adapters. Uniform signature (frozen for P2; see baseline_gigalens):

    run_cell(target: LensPosterior, seed: int, budget: dict,
             config: dict | None = None, freeze_points: list | None = None)
        -> cgl.io.CellResult

Import-light registry: adapters are imported lazily by name.
"""
from __future__ import annotations

import importlib

_ADAPTERS = {
    "s0_baseline": ("cgl.samplers.baseline_gigalens", "run_cell"),
    # P2b: s1_nuts, s2_mclmc, s3_adjusted_mclmc, s4_smc, s5_pt_remc,
    #      s6_flowmc, s7_nautilus, s8_neutra  (built against the same API)
}


def list_samplers():
    return sorted(_ADAPTERS)


def get_run_cell(name: str):
    if name not in _ADAPTERS:
        raise KeyError(f"unknown sampler {name!r}; known: {list_samplers()}")
    module, fn = _ADAPTERS[name]
    return getattr(importlib.import_module(module), fn)
