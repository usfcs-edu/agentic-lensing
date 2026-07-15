"""Test env bootstrap: CPU toys, x64 ON, before any jax import (house pattern
carried from ../claude-giga-lens/tests/). Unit tests never touch a GPU
(CGL2_ALLOW_CPU=1 sanctions the 16x16 toys)."""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CGL2_ALLOW_CPU", "1")
os.environ.setdefault("GIGALENS_X64", "1")
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_autotune_level=0")

import numpy as np  # noqa: E402
import pytest  # noqa: E402


@pytest.fixture(scope="session")
def toy():
    """Session-scoped 16x16 toy scene (shared across correlated-term tests)."""
    from cgl2 import scene_build

    return scene_build.build_toy_scene(n_pix=16, n_max=3, seed=3,
                                       delta_pix=0.25)


@pytest.fixture(scope="session")
def rng():
    return np.random.default_rng(0)
