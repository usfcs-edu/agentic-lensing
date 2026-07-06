"""Sampler adapters. Uniform signature (frozen for P2; see baseline_gigalens):

    run_cell(target: LensPosterior, seed: int, budget: dict,
             config: dict | None = None, freeze_points: list | None = None)
        -> cgl.io.CellResult

Import-light registry: adapters are imported lazily by name.

P2b roster (S5 pocoMC SKIPPED -- pre-approved drop, S4 covers
preconditioned-SMC scientifically; recorded in the pool report):
    s0_baseline      S0  MAP->SVI->ChEES-PHMC (the published GIGA-Lens recipe)
    bj_nuts          S1  blackjax window-adapted NUTS (vmapped chains)
    bj_mclmc         S2  blackjax MCLMC (unadjusted tuner / adjusted static)
    flowmc_runner    S3  flowMC RQSpline_MALA (local MALA + flow global moves)
    bj_smc           S4  blackjax adaptive tempered SMC (HMC kernel, logZ)
    nautilus_runner  S6  nautilus (gradient-free, unit-cube face, logZ)
    remc_pt          S7  TFP ReplicaExchangeMC over batched PHMC (like-only
                         tempering)
    neutra           S8  NSF fit to floored-SVI samples -> ChEES-HMC in
                         flow-pullback space
    glnt             S9  MAP->SVI->short tempered-SMC anneal->NSF->ChEES-HMC
                         in flow space (the CGL recipe candidate)

SCALE_KEYS: per-adapter budget fields that Track B multiplies by 4x
(pre-registered convergence-track protocol; config stays frozen).
"""
from __future__ import annotations

import importlib

_ADAPTERS = {
    "s0_baseline": ("cgl.samplers.baseline_gigalens", "run_cell"),
    "bj_nuts": ("cgl.samplers.bj_nuts", "run_cell"),
    "bj_mclmc": ("cgl.samplers.bj_mclmc", "run_cell"),
    "flowmc_runner": ("cgl.samplers.flowmc_runner", "run_cell"),
    "bj_smc": ("cgl.samplers.bj_smc", "run_cell"),
    "nautilus_runner": ("cgl.samplers.nautilus_runner", "run_cell"),
    "remc_pt": ("cgl.samplers.remc_pt", "run_cell"),
    "neutra": ("cgl.samplers.neutra", "run_cell"),
    "glnt": ("cgl.samplers.glnt", "run_cell"),
}

# Track-B budget fields scaled 4x (s0_baseline's lives here because the S0
# module predates the convention and stays untouched).
SCALE_KEYS = {
    "s0_baseline": ("n_keep",),
    "bj_nuts": ("n_keep",),
    "bj_mclmc": ("n_steps",),
    "flowmc_runner": ("n_production_loops",),
    "bj_smc": ("n_particles",),
    "nautilus_runner": ("n_like_max", "n_eff"),
    "remc_pt": ("n_keep",),
    "neutra": ("n_keep",),
    "glnt": ("n_keep",),
}


def list_samplers():
    return sorted(_ADAPTERS)


def get_run_cell(name: str):
    if name not in _ADAPTERS:
        raise KeyError(f"unknown sampler {name!r}; known: {list_samplers()}")
    module, fn = _ADAPTERS[name]
    return getattr(importlib.import_module(module), fn)


def get_scale_keys(name: str):
    if name not in _ADAPTERS:
        raise KeyError(f"unknown sampler {name!r}; known: {list_samplers()}")
    return SCALE_KEYS.get(name, ())
