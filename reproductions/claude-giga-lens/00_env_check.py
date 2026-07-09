#!/usr/bin/env python
"""Environment gate: asserts pins, GPU, sampler imports; writes data/env_freeze_<host>.txt.

Run on every host before anything else:
    phoenix:    source /raid/benson/.venvs/cgl/bin/activate && python 00_env_check.py
    perlmutter: module load python && source ~/claude-giga-lens/venv/bin/activate && \
                python 00_env_check.py --host perlmutter

Exit code 0 = gate passed. Non-fatal warnings are printed but do not fail the gate
(e.g. missing optional 'drizzle' package -> reproject fallback).
"""
from __future__ import annotations

import argparse
import importlib
import os
import platform
import socket
import subprocess
import sys
from pathlib import Path

REPRO = Path(__file__).resolve().parent

CORE_PINS = {
    "jax": "0.6.2",
    "jaxlib": "0.6.2",
    "numpy": "2.4.6",
    "scipy": "1.17.1",
    "optax": "0.2.8",
    "objax": "1.8.0",
    "astropy": "7.2.0",
    "lenstronomy": "1.14.0",
}
TFP_PIN = "0.25.0"
SAMPLER_PINS = {  # exact-or-prefix match; blackjax 1.3.x accepted
    "blackjax": "1.3",
    "flowMC": "0.4.5",
    "flowjax": "19.0.0",
    "equinox": "0.13.8",
    "nautilus": "1.0.6",   # import name for nautilus-sampler
}
OPTIONAL = ["drizzle", "reproject", "arviz", "corner", "coolest"]

FAIL = []
WARN = []


DIST_NAME = {"nautilus": "nautilus-sampler"}  # import name -> pip distribution name


def check(name: str, expected: str, exact: bool = False) -> None:
    try:
        mod = importlib.import_module(name)
    except Exception as e:  # noqa: BLE001
        FAIL.append(f"{name}: IMPORT FAILED ({e!r})")
        return
    ver = getattr(mod, "__version__", None)
    if ver is None:  # e.g. flowMC ships no __version__
        from importlib.metadata import version as dist_version
        try:
            ver = dist_version(DIST_NAME.get(name, name))
        except Exception:  # noqa: BLE001
            ver = "?"
    ok = ver == expected if exact else ver.startswith(expected)
    if not ok:
        FAIL.append(f"{name}: {ver} != pinned {expected}")
    print(f"  {name:24s} {ver:12s} {'OK' if ok else 'PIN MISMATCH'}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=None, help="override host label (e.g. perlmutter)")
    ap.add_argument("--skip-gpu", action="store_true",
                    help="skip the GPU-visibility check (login nodes)")
    args = ap.parse_args()
    host = args.host or socket.gethostname().split(".")[0]

    print(f"== cgl env check: host={host} machine={platform.machine()} "
          f"python={sys.version.split()[0]} ==")

    print("-- core pins (must match the blessed gigalens stack) --")
    for name, pin in CORE_PINS.items():
        check(name, pin, exact=True)
    try:
        import tensorflow_probability.substrates.jax as tfp  # noqa: F401
        import tensorflow_probability as tfp_pkg
        ok = tfp_pkg.__version__ == TFP_PIN
        print(f"  {'tfp[jax]':24s} {tfp_pkg.__version__:12s} {'OK' if ok else 'PIN MISMATCH'}")
        if not ok:
            FAIL.append(f"tensorflow-probability: {tfp_pkg.__version__} != {TFP_PIN}")
    except Exception as e:  # noqa: BLE001
        FAIL.append(f"tfp[jax] substrate import failed: {e!r}")

    print("-- sampler pins --")
    for name, pin in SAMPLER_PINS.items():
        check(name, pin, exact=False)

    print("-- optional --")
    for name in OPTIONAL:
        try:
            mod = importlib.import_module(name)
            print(f"  {name:24s} {getattr(mod, '__version__', '?'):12s} OK (optional)")
        except Exception:  # noqa: BLE001
            WARN.append(f"optional package {name} not importable")
            print(f"  {name:24s} {'-':12s} MISSING (optional)")

    # GPU + tiny jitted logp
    if not args.skip_gpu:
        print("-- gpu --")
        import jax
        import jax.numpy as jnp
        backend = jax.default_backend()
        devs = jax.devices()
        print(f"  backend={backend} devices={[d.device_kind for d in devs]}")
        if backend != "gpu":
            FAIL.append(f"jax default backend is {backend!r}, not gpu")
        else:
            x = jnp.linspace(-1.0, 1.0, 256)
            val = jax.jit(lambda z: jnp.sum(-0.5 * z ** 2))(x)
            val.block_until_ready()
            print(f"  jitted toy logp OK ({float(val):.3f})")

    # vendored ref assert
    print("-- vendor --")
    sys.path.insert(0, str(REPRO))
    from cgl.paths import bootstrap_vendor, EXPECTED_VENDOR_REF
    bootstrap_vendor()
    import gigalens  # noqa: F401
    gl_path = Path(gigalens.__file__).resolve()
    if str(REPRO / "vendor") not in str(gl_path):
        FAIL.append(f"gigalens imported from {gl_path}, not the campaign vendor tree")
    print(f"  gigalens from {gl_path}")
    print(f"  vendored ref {EXPECTED_VENDOR_REF[:12]} OK")

    # Perlmutter-specific: slurm account must be an APPROVED account.
    # deepsrch_g (default, more hours) OR cosmo_g (better fairshare when deepsrch_g
    # is LensJudge-congested; both verified with ample balance for this campaign).
    if host.startswith("perlmutter") or host.startswith("login") or host.startswith("nid"):
        print("-- perlmutter account --")
        approved = ("deepsrch_g", "cosmo_g")
        templates = sorted((REPRO / "slurm").glob("*.slurm"))
        bad = [t.name for t in templates
               if not any(f"-A {a}" in t.read_text() or f"--account={a}" in t.read_text()
                          for a in approved)]
        if bad:
            FAIL.append(f"slurm templates charge no approved account {approved}: {bad}")
        else:
            print(f"  {len(templates)} slurm templates all charge an approved account OK")

    # freeze
    (REPRO / "data").mkdir(exist_ok=True)
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"],
                            capture_output=True, text=True).stdout
    out = REPRO / "data" / f"env_freeze_{host}.txt"
    out.write_text(freeze)
    print(f"-- wrote {out} ({len(freeze.splitlines())} packages) --")

    for w in WARN:
        print(f"WARN: {w}")
    if FAIL:
        for f in FAIL:
            print(f"FAIL: {f}")
        print("== ENV GATE: FAIL ==")
        return 1
    print("== ENV GATE: PASS ==")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
