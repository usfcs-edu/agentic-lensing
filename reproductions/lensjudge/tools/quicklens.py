"""quick_lensmodel - the Foundry-I (Huang 2025a) lens-MODELABILITY criterion.

Question: does a plausible strong-lens MODEL reproduce the image configuration?
A GIGA-Lens MAP fit (EPL+shear mass, 2 lens-light Sersics, source = Sersic+shapelets)
is run on the candidate's grz cutout; from the best fit we report the Einstein radius,
the reduced chi2, the chi2 IMPROVEMENT from deflecting the source through the mass
(dchi2_frac - large for a real lens, ~0 for a smooth galaxy), the predicted number of
images, and a continuous ``lens_score``.

Heavy deps (JAX / GIGA-Lens) are kept OUT of this SDK process: the cube is written to a
temp .npy and we subprocess into the GIGA-Lens venv (config.GIGALENS_PY), which runs
outputs/quicklens_proto.py --cube and prints one JSON line. Results are cached by
candidate name under cache/quicklens/.

VALIDATION / CAVEAT (measured, do not over-claim):
  * A-lens vs random-galaxy negative: AUC ~0.77 (real, useful signal).
  * A-lens vs grade-D human-reject:   AUC ~0.44 (DOES NOT separate them - grade-D
    rejects admit lens-model fits just as well as true lenses on ground-based grz).
  So this is a real-vs-random-galaxy filter, NOT a reliable real-vs-hard-reject filter.
  Runtime ~40-55 s per cutout (GPU, JIT-compiled); cached after first run.
"""
from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from claude_agent_sdk import tool

from lensjudge import config
from lensjudge.common import fetch

_CACHE = config.CACHE / "quicklens"

_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "survey": {"type": "string"},
        "ra": {"type": "number"},
        "dec": {"type": "number"},
    },
    "required": ["name"],
}


def _parse_last_json(stdout: str) -> dict | None:
    """The GIGA-Lens process also prints 'Final Chi-squared: ...'; take the last
    line that parses as a JSON object."""
    for line in reversed(stdout.strip().splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return None


def _run_subprocess(cube: np.ndarray) -> dict:
    _CACHE.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".npy", dir=_CACHE, delete=False) as tf:
        npy = Path(tf.name)
    try:
        np.save(npy, np.asarray(cube, dtype=np.float32))
        env = {
            "CUDA_VISIBLE_DEVICES": config.QUICKLENS_GPU,
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.6",
            "TF_GPU_ALLOCATOR": "cuda_malloc_async",
            "PATH": "/usr/bin:/bin",
        }
        proc = subprocess.run(
            [str(config.GIGALENS_PY), str(config.QUICKLENS_SCRIPT), "--cube", str(npy)],
            capture_output=True, text=True, timeout=config.QUICKLENS_TIMEOUT_S, env=env)
        res = _parse_last_json(proc.stdout)
        if res is None:
            return {"error": "no JSON from quicklens subprocess",
                    "stderr_tail": proc.stderr[-400:], "converged": False,
                    "plausible": False}
        return res
    finally:
        npy.unlink(missing_ok=True)


@tool("quick_lensmodel",
      "Run the Foundry-I lens-MODELABILITY check: fit a strong-lens model "
      "(EPL+shear mass, Sersic lens light, lensed source) to the candidate's grz cutout "
      "and report theta_E (Einstein radius, arcsec), reduced_chi2, dchi2_frac "
      "(chi2 improvement from lensing the source - high for a real lens), n_images, a "
      "continuous lens_score (0-1), and a plausible flag. NOTE: this reliably separates "
      "real lenses from random galaxies (AUC~0.77) but NOT from hard human-rejected "
      "candidates (AUC~0.44); treat it as supporting, not decisive, evidence. ~40-55s, "
      "cached.", _SCHEMA)
async def quick_lensmodel(args: dict) -> dict:
    name = args.get("name")
    survey = args.get("survey") or "storfer"
    _CACHE.mkdir(parents=True, exist_ok=True)
    cache_file = _CACHE / f"{name}.json" if name else None

    if cache_file is not None and cache_file.exists():
        res = json.loads(cache_file.read_text())
    else:
        cube = fetch.get_cube(name=name, ra=args.get("ra"), dec=args.get("dec"),
                              survey=survey)
        if cube is None:
            return {"content": [{"type": "text",
                                 "text": "ERROR: no cutout for lens-model fit."}],
                    "is_error": True}
        res = _run_subprocess(cube)
        if cache_file is not None and "error" not in res:
            cache_file.write_text(json.dumps(res))

    res.setdefault("note",
                   "modelability signal: high lens_score / plausible=True means a "
                   "strong-lens model reproduces the configuration. Reliable vs random "
                   "galaxies, weak vs hard human-rejects - use as supporting evidence.")
    return {"content": [{"type": "text", "text": json.dumps(res)}]}
