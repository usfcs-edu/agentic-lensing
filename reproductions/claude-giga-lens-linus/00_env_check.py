#!/usr/bin/env python
"""Environment gate: pins, vendor ref, imports, backend. Raises on any mismatch.

Run first in every session and at the top of every slurm job (house rule).
Writes data/env_freeze_<host>.txt for the ledger.
"""
import os
import platform
import socket
import subprocess
import sys

os.environ.setdefault("JAX_ENABLE_X64", "1")

from cgl2 import paths  # noqa: E402  (x64 bootstrap via cgl2.__init__)


def main() -> int:
    paths.bootstrap_vendor()
    paths.require_jax_pin()

    import jax
    import blackjax
    import numpy
    import tensorflow_probability.substrates.jax as tfp  # noqa: F401

    try:
        import tensorflow  # noqa: F401

        raise RuntimeError("tensorflow present — the vendored dep must stay inert (D2).")
    except ImportError:
        pass

    from blackjax import adjusted_mclmc  # noqa: F401  (MAMS mutation kernel dependency)
    import gigalens
    from gigalens.jax import scene, scene_simulator, scene_prob_model  # noqa: F401

    from cgl2 import guards

    guards.require_x64()
    guards.require_where_mask_semantics()

    backend = jax.default_backend()
    lines = [
        f"host={socket.gethostname()} platform={platform.machine()}",
        f"python={sys.version.split()[0]}",
        f"jax={jax.__version__} backend={backend} devices={len(jax.devices())}",
        f"blackjax={blackjax.__version__} numpy={numpy.__version__}",
        f"gigalens={gigalens.__file__}",
        f"vendor_ref={paths.EXPECTED_VENDOR_REF}",
    ]
    freeze = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True
    ).stdout
    paths.DATA_DIR.mkdir(exist_ok=True)
    out = paths.DATA_DIR / f"env_freeze_{socket.gethostname()}.txt"
    out.write_text("\n".join(lines) + "\n\n# pip freeze\n" + freeze)
    print("\n".join(lines))
    print(f"ENV CHECK: PASS -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
