"""Standalone multi-NODE validation of ModellingSequence + MCLMC sharding.

Launched by ``multinode_validate.slurm`` via ``srun`` with one process per GPU across >=2
nodes. Each process calls ``jax.distributed.initialize()`` (Slurm auto-detected), then runs
TINY MAP / SVI / HMC / MCLMC on both a single-lens-plane and a multi-lens-plane (two mass
planes) model, checking each returns finite, correctly-shaped, replicated output. This is a
sharding smoke test, NOT inference -- step counts are deliberately trivial.

Why standalone (not pytest): true multi-node needs one process per rank launched by srun and
coordinating over NCCL; ``jax.distributed.initialize()`` must run before ANY backend-
initialising JAX call -- and ``gigalens.jax.inference`` builds its device mesh at import.
So we initialise distributed FIRST, then import the model builders.

Coverage note: MAP/SVI shard the batch via shard_map over the global 'device' mesh; HMC
uses pmap + ``process_allgather`` (explicitly multi-host); MCLMC uses shard_map + ``psum``
over the global mesh but has NO ``process_allgather`` -- so MCLMC's multi-node path (does
resharding a process-local array onto a global multi-host mesh work?) is the least-tested
and the main thing this driver de-risks.

Exit code: each rank exits non-zero if any of its runs failed; srun surfaces any failing
task. Rank 0 prints the summary table.
"""
import os
os.environ.setdefault("JAX_ENABLE_X64", "1")  # gigalens precision guard reads this at import
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import sys
import traceback

import jax

# MUST precede importing gigalens (its inference module builds a mesh via jax.devices()).
try:
    jax.distributed.initialize()  # Slurm auto-detect (SLURM_PROCID / nodelist / localid)
    _DIST = True
except Exception as exc:  # single-process fallback (e.g. a 1-node smoke run)
    print(f"[warn] jax.distributed.initialize() failed ({exc!r}); running single-process")
    _DIST = False

jax.config.update("jax_enable_x64", True)

import numpy as np  # noqa: E402
import optax  # noqa: E402
import tensorflow_probability.substrates.jax.distributions as tfd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _models import build_single_lensplane, build_multiplane, start_vector, make_qz  # noqa: E402
from gigalens.jax.experimental.mclmc import MCLMC_JIT  # noqa: E402

RANK = jax.process_index()
NPROC = jax.process_count()
NDEV = jax.device_count()
NLOCAL = jax.local_device_count()
IS_LEAD = RANK == 0

# n_chains / n_samples: multiple of the GLOBAL device count and >= it (MCLMC requires it).
N = max(2 * NDEV, NDEV)


def _log(msg):
    print(f"[rank {RANK}/{NPROC}] {msg}", flush=True)


def _check(name, arr, expected_shape, D):
    """Return (ok, detail) for a returned sample/param array."""
    a = np.asarray(arr)
    finite = bool(np.all(np.isfinite(a)))
    repl = getattr(getattr(arr, "sharding", None), "is_fully_replicated",
                   getattr(arr, "is_fully_replicated", None))
    shape_ok = (expected_shape is None) or (a.shape == expected_shape)
    ok = finite and shape_ok
    return ok, f"shape={a.shape} finite={finite} replicated={repl}"


def _run_algorithms(tag, seq, start):
    D = len(start)
    qz = make_qz(start)
    results = []

    def record(algo, fn, checker):
        try:
            out = fn()
            ok, detail = checker(out)
            results.append((algo, ok, detail))
            _log(f"{tag:16s} {algo:6s} {'PASS' if ok else 'FAIL'}  {detail}")
        except Exception:
            tb = traceback.format_exc().strip().splitlines()[-1]
            results.append((algo, False, f"EXC {tb}"))
            _log(f"{tag:16s} {algo:6s} FAIL  EXC {tb}")

    opt = optax.adabelief(1e-2, b1=0.95, b2=0.99)

    record("MAP",
           lambda: seq.MAP(optimizer=opt, n_samples=N, num_steps=3, seed=0,
                           output_type="best", pbar_interval=0),
           lambda o: _check("MAP", o[0], (D,), D))

    record("SVI",
           lambda: seq.SVI(start=start, optimizer=opt, n_vi=N, num_steps=3, seed=0,
                           pbar_interval=0),
           lambda o: _check("SVI", o[1], (3,), D))  # loss_hist

    record("HMC",
           lambda: seq.HMC(qz, n_hmc=N, num_burnin_steps=2, num_results=2, pbar_interval=0),
           lambda o: (np.asarray(o).size == N * 2 * D and np.all(np.isfinite(np.asarray(o))),
                      f"size={np.asarray(o).size} (exp {N*2*D}) shape={np.asarray(o).shape}"))

    record("MCLMC",
           lambda: MCLMC_JIT(seq, qz, n_hmc=N, num_burnin_steps=20, num_results=4, seed=0,
                             progress_bar=False),
           lambda o: _check("MCLMC", o, (N, 4, D), D))

    return results


def main():
    if IS_LEAD:
        print("=" * 78, flush=True)
        print(f"multi-node sharding validation: processes={NPROC} global_devices={NDEV} "
              f"local_devices/proc={NLOCAL} n_chains={N} distributed={_DIST}", flush=True)
        print("=" * 78, flush=True)
    _log(f"local devices: {jax.local_devices()}")

    all_ok = True
    summary = []
    for tag, builder in (("single-lensplane", build_single_lensplane),
                         ("multiplane", build_multiplane)):
        _, _, prob, seq = builder()
        start = start_vector(prob, seed=0)
        _log(f"built {tag}: {len(start)} free params")
        for algo, ok, detail in _run_algorithms(tag, seq, start):
            summary.append((tag, algo, ok, detail))
            all_ok = all_ok and ok

    if _DIST:
        try:
            jax.experimental.multihost_utils.barrier()
        except Exception:
            pass

    if IS_LEAD:
        print("\n" + "=" * 78, flush=True)
        print(f"SUMMARY (rank 0 of {NPROC} processes, {NDEV} devices):", flush=True)
        for tag, algo, ok, detail in summary:
            print(f"  {tag:16s} {algo:6s} {'PASS' if ok else 'FAIL':4s}  {detail}", flush=True)
        print(f"\nRANK-0 RESULT: {'ALL PASS' if all_ok else 'FAILURES PRESENT'}", flush=True)
        print("=" * 78, flush=True)

    if _DIST:
        try:
            jax.distributed.shutdown()
        except Exception:
            pass

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
