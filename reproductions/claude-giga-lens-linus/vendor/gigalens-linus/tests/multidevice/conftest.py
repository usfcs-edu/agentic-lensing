"""Fixtures + device setup for the multi-device / sharding suite.

Ported from the old-API ``tests/essentials/{conftest,inference_fns_test,multihost_test}.py``
(gigalens-old, ``dev`` branch) onto the new "scene" API. The behaviour under test --
``ModellingSequence`` sharding the MAP/SVI walker batch across a 1-D ``'device'`` mesh,
replicated point estimates vs. sharded ``output_type='all'`` histories -- is unchanged
between the two APIs; only model *construction* was rewritten (LensModel/Plane/Component
+ ImageData + ProbModel instead of PhysicalModel/ForwardProbModel/LensSimulator).

Two ways to get >=2 devices:
  * **faked CPU devices** (this file) -- ``--xla_force_host_platform_device_count`` lets
    ``test_inference_sharding.py`` run in plain ``pytest`` on a login node / CI with no
    GPU. Harmless on a GPU node (the flag only splits the CPU *host* platform; JAX still
    selects GPU when one is present) and in the multi-host driver (each worker pins a GPU
    and calls ``jax.distributed.initialize`` before importing jax here, so the flag is a
    no-op there).
  * **real GPUs** -- ``test_multihost.py`` spawns one ``jax.distributed`` process per GPU;
    it is gated behind >=2 real GPUs (no interactive prompt; see that file).

The model/start builders are module-level (NOT fixtures/closures) so the spawned
multi-host workers can import and call them in a fresh process, exactly like the old
essentials conftest ("outside of their wrappers so that they can be spun up on the
multihost process").
"""
import os

# Must be set before jax is first imported in this process.
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"  # else JAX grabs the whole GPU for
                                                       # the parent, starving the workers
os.environ.setdefault("JAX_ENABLE_X64", "1")

# Fake N CPU devices unless the caller opted out. Override the count with
# GIGALENS_TEST_FAKE_DEVICES=<n>; set GIGALENS_TEST_FORCE_CPU=0 to use whatever real
# devices are present instead (e.g. a real multi-GPU standalone run). 16 // 4 == 4 (see
# the coprime-vs-dev_cnt note in the sharding test), so 4 keeps every batched array
# evenly divisible.
#
# JAX_PLATFORMS=cpu is REQUIRED to make the fake-device flag bite: --xla_force_host_
# platform_device_count only splits the CPU host platform, so on a node that also has a
# GPU (e.g. a NERSC login node) JAX would otherwise select the single GPU and ignore it.
# The multi-host driver sets GIGALENS_TEST_FORCE_CPU=0 so its workers use the real GPUs.
_FAKE_DEVICES = int(os.environ.get("GIGALENS_TEST_FAKE_DEVICES", "4"))
_FORCE_CPU = os.environ.get("GIGALENS_TEST_FORCE_CPU", "1") == "1"
if _FORCE_CPU:
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    if "xla_force_host_platform_device_count" not in os.environ.get("XLA_FLAGS", ""):
        os.environ["XLA_FLAGS"] = (
            os.environ.get("XLA_FLAGS", "").strip()
            + f" --xla_force_host_platform_device_count={_FAKE_DEVICES}"
        ).strip()

import sys

import pytest
import jax

# tests/multidevice is on sys.path (pytest prepend mode, rootdir anchored by pytest.ini),
# so the shared, pytest-free model builders import directly -- the same module the
# standalone multi-node driver (multinode_validate.py) uses, so both build identical models.
sys.path.insert(0, os.path.dirname(__file__))
from _models import build_single_lensplane, build_multiplane, start_vector  # noqa: E402

try:  # x64 for parity with the rest of the JAX suite; harmless if already set
    jax.config.update("jax_enable_x64", True)
except Exception:  # pragma: no cover
    pass


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "multidevice: exercises ModellingSequence / MCLMC sharding; needs >=2 (real or "
        "faked) JAX devices, and the multi-host driver needs >=2 real GPUs.",
    )


# Fixtures build once per module. Single-lens-plane is the default system; multiplane
# (two mass planes) is used by the multi-lens-plane sharding checks.
@pytest.fixture(scope="module")
def shared_model_seq():
    return build_single_lensplane()[3]


@pytest.fixture(scope="module")
def shared_start():
    return start_vector(build_single_lensplane()[2], seed=0)


@pytest.fixture(scope="module")
def multiplane_seq():
    return build_multiplane()[3]


@pytest.fixture(scope="module")
def multiplane_start():
    return start_vector(build_multiplane()[2], seed=0)
