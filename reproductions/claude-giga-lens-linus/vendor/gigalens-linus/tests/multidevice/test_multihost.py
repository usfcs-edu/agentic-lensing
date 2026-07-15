"""Real multi-host driver: run the sharding suite under ``jax.distributed``.

Ported from gigalens-old ``tests/essentials/multihost_test.py``. Spawns ``num_processes``
separate Python processes (one per GPU), each initialising ``jax.distributed`` and
re-running ``test_inference_sharding.py`` in a genuine multi-process JAX environment, then
asserts every rank exited 0. This is the only test that exercises the real cross-host code
paths in ``ModellingSequence`` (``process_allgather`` in HMC, per-process seed folding).

Differences from the original:
  * The interactive ``input("Skip Test? (y/n)")`` gate is replaced by a proper
    ``skipif`` on the real GPU count -- the prompt hangs under any non-interactive runner
    (CI, ``pytest -p no:cacheprovider``, nohup). Faked CPU devices do NOT satisfy this:
    ``jax.distributed`` needs real, separately-addressable devices, so this test targets
    real GPUs only and is skipped otherwise.
  * It re-runs this directory (``tests/multidevice``) rather than the old ``essentials``
    dir. ``--confcutdir`` keeps the workers from collecting the TF-era top-level conftest.
"""
import os
from pathlib import Path
import sys
import traceback
import multiprocessing as mp

import jax
import pytest

NUM_PROCESSES = int(os.environ.get("GIGALENS_MULTIHOST_PROCS", "2"))
_current_file = os.path.abspath(__file__)
_this_dir = str(Path(_current_file).parent)
ctx = mp.get_context("spawn")


def _gpu_count():
    """Real GPUs visible to this process, 0 if the GPU backend is absent.

    jax.devices(backend='gpu') *raises* RuntimeError when JAX_PLATFORMS=cpu (the faked-CPU
    standalone mode) rather than returning an empty list, so guard it."""
    try:
        return len(jax.devices(backend="gpu"))
    except RuntimeError:
        return 0


def _multihost_worker(rank, num_processes, queue, pytest_args):
    """Run pytest inside one rank of a distributed JAX environment."""
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        # Workers use the real GPU backend: do NOT force CPU or fake devices.
        os.environ["GIGALENS_TEST_FORCE_CPU"] = "0"
        os.environ["GIGALENS_TEST_FAKE_DEVICES"] = "1"

        import jax as _jax
        _jax.distributed.initialize(
            coordinator_address="127.0.0.1:12345",
            num_processes=num_processes,
            process_id=rank,
        )
        exit_code = pytest.main(pytest_args)
        queue.put((rank, int(exit_code), None))
    except Exception:
        queue.put((rank, 1, traceback.format_exc()))
    finally:
        try:
            import jax as _jax
            _jax.experimental.multihost_utils.barrier()
            _jax.distributed.shutdown()
        except Exception:
            pass


@pytest.mark.multidevice
def test_multihost_sharding_suite():
    # The GPU-count gate lives INSIDE the test, not in a @skipif decorator: a spawned
    # worker re-imports this module, and any module-level jax.devices() call would
    # initialise the XLA backend before jax.distributed.initialize() -- which then raises.
    gpus = _gpu_count()
    if gpus < NUM_PROCESSES:
        pytest.skip(
            f"multi-host test needs >={NUM_PROCESSES} real GPUs (jax.distributed cannot "
            f"use faked CPU devices); found {gpus}. Run on a multi-GPU node before pushing."
        )

    pytest_args = [
        _this_dir,
        f"--ignore={_current_file}",   # don't recurse into this driver
        f"--confcutdir={_this_dir}",   # skip the TF-era top-level conftest
        "-v",
    ]
    queue = ctx.Queue()
    processes = []
    for rank in range(NUM_PROCESSES):
        p = ctx.Process(target=_multihost_worker,
                        args=(rank, NUM_PROCESSES, queue, pytest_args))
        p.start()
        processes.append(p)

    results = []
    for _ in range(NUM_PROCESSES):
        try:
            results.append(queue.get(timeout=600))
        except mp.queues.Empty:
            results.append((-1, 1, "worker timeout"))
    for p in processes:
        p.join()

    for rank, exit_code, error in results:
        assert exit_code == 0, f"rank {rank} failed with exit code {exit_code}\n{error}"


if __name__ == "__main__":
    sys.exit(pytest.main([_current_file, "-v", "-s"]))
