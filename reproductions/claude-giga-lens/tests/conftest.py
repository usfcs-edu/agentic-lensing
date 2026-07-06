"""Test bootstrap: CPU-only toy grids are the sole sanctioned CPU exception.

Unit tests run on 16x16 toys with JAX_PLATFORMS=cpu + CGL_ALLOW_CPU=1 so the
guard in cgl.guards.require_gpu stays honest everywhere else. GPU-marked tests
(parity harness) unset these and pin one device.
"""
import os

# Must be set BEFORE any jax import in the test session.
os.environ.setdefault("GIGALENS_X64", "1")
if os.environ.get("CGL_TEST_GPU") != "1":
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("CGL_ALLOW_CPU", "1")

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
