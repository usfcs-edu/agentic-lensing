"""Per-cell result schema + storage for the sampler benchmark.

One benchmark cell = (sampler, target, seed, track). Each cell writes an
npz/json PAIR under data/results/<sampler>/<target>/ (data/ is gitignored;
small gate evidence is quoted into CAMPAIGN.md, never bulk arrays):

    s<seed>_<track>.npz   samples (T, C, dim) in the TARGET dtype (thinned in
                          T if the array would exceed SAMPLES_MAX_BYTES) +
                          diagnostics arrays (prefixed "diag_")
    s<seed>_<track>.json  metrics, budget ledger, timing, config + hash, env,
                          freeze-check evidence, provenance

The json is self-contained for ledger/reporting; the npz carries bulk arrays.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import platform
import time
from pathlib import Path
from typing import Optional

import numpy as np

from cgl import paths

SAMPLES_MAX_BYTES = 512 * 1024 * 1024
SCHEMA_VERSION = 1


def config_hash(config: dict) -> str:
    """sha256 of the canonical-json config (sorted keys, no whitespace)."""
    blob = json.dumps(config, sort_keys=True, separators=(",", ":"),
                      default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


def env_info() -> dict:
    """Runtime provenance snapshot (safe to call after jax import only)."""
    import jax

    dev = jax.devices()[0]
    return dict(
        hostname=platform.node(),
        python=platform.python_version(),
        jax=jax.__version__,
        numpy=np.__version__,
        backend=jax.default_backend(),
        device=str(dev),
        device_kind=dev.device_kind,
        n_devices=len(jax.devices()),
        x64=bool(jax.config.jax_enable_x64),
        CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES"),
        XLA_FLAGS=os.environ.get("XLA_FLAGS"),
        GIGALENS_X64=os.environ.get("GIGALENS_X64"),
    )


@dataclasses.dataclass
class CellResult:
    """The uniform result every sampler adapter returns from run_cell."""
    sampler: str
    target: str
    seed: int
    track: str
    samples: np.ndarray              # (T, C, dim) unconstrained, post-burn
    labels: list
    mass_labels: list
    diagnostics: dict                # name -> np.ndarray (per-step traces)
    budget: dict                     # BudgetLedger.as_dict()
    timing: dict                     # phase -> seconds
    config: dict                     # full resolved adapter config
    env: dict
    freeze_check: dict               # zoo-freeze assertion evidence
    metrics: dict = dataclasses.field(default_factory=dict)
    thinned_by: int = 1
    notes: str = ""

    @property
    def config_sha(self) -> str:
        return config_hash(self.config)


def cell_paths(sampler: str, target: str, seed: int, track: str,
               root: Optional[Path] = None):
    root = Path(root) if root is not None else paths.RESULTS
    d = root / sampler / target
    stem = f"s{seed}_{track}"
    return d / f"{stem}.npz", d / f"{stem}.json"


def save_cell_result(res: CellResult, root: Optional[Path] = None):
    """Write the npz/json pair; returns (npz_path, json_path)."""
    npz_path, json_path = cell_paths(res.sampler, res.target, res.seed,
                                     res.track, root)
    npz_path.parent.mkdir(parents=True, exist_ok=True)

    samples = np.asarray(res.samples)
    thin = res.thinned_by
    while samples.nbytes > SAMPLES_MAX_BYTES:
        samples = samples[::2]
        thin *= 2

    arrays = {"samples": samples}
    for k, v in (res.diagnostics or {}).items():
        arrays[f"diag_{k}"] = np.asarray(v)
    np.savez_compressed(npz_path, **arrays)

    payload = dict(
        schema_version=SCHEMA_VERSION,
        written_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        sampler=res.sampler, target=res.target, seed=res.seed, track=res.track,
        samples_shape=list(samples.shape),
        samples_dtype=str(samples.dtype),
        thinned_by=int(thin),
        labels=list(res.labels), mass_labels=list(res.mass_labels),
        diagnostics_keys=sorted((res.diagnostics or {}).keys()),
        metrics=res.metrics,
        budget=res.budget,
        timing=res.timing,
        config=res.config,
        config_sha=res.config_sha,
        env=res.env,
        freeze_check=res.freeze_check,
        notes=res.notes,
        npz=str(npz_path.name),
    )
    json_path.write_text(json.dumps(payload, indent=2, default=str))
    return npz_path, json_path


def load_cell_result(sampler: str, target: str, seed: int, track: str,
                     root: Optional[Path] = None) -> CellResult:
    npz_path, json_path = cell_paths(sampler, target, seed, track, root)
    meta = json.loads(json_path.read_text())
    z = np.load(npz_path)
    diagnostics = {k[len("diag_"):]: z[k] for k in z.files
                   if k.startswith("diag_")}
    return CellResult(
        sampler=meta["sampler"], target=meta["target"], seed=meta["seed"],
        track=meta["track"], samples=z["samples"],
        labels=meta["labels"], mass_labels=meta["mass_labels"],
        diagnostics=diagnostics, budget=meta["budget"], timing=meta["timing"],
        config=meta["config"], env=meta["env"],
        freeze_check=meta["freeze_check"], metrics=meta["metrics"],
        thinned_by=meta["thinned_by"], notes=meta.get("notes", ""),
    )
