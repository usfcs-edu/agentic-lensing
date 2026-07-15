# `tests/multidevice` — sharding & multi-node parallelism

Ported from the old-API `tests/essentials/{conftest,inference_fns_test,multihost_test}.py`
(gigalens-old, `dev` branch) onto the new "scene" API. What is under test —
`gigalens.jax.inference.ModellingSequence` sharding the MAP/SVI walker batch across a 1-D
`'device'` mesh, replicated point estimates vs. sharded `output_type='all'` histories, and
the HMC cross-host gather — is structurally unchanged between the two APIs; only model
*construction* was rewritten (`LensModel`/`Plane`/`Component` + `ImageData` + `ProbModel`).

## Files

- `test_inference_sharding.py` — asserts the sharding/shape/replication contract of MAP,
  SVI, HMC **and MCLMC** (`gigalens.jax.experimental.mclmc.MCLMC_JIT`) outputs, on both a
  single-lens-plane and a multi-lens-plane model. Device-count-agnostic; runs on faked CPU
  devices, so it works in plain `pytest` with no GPU.
- `test_multihost.py` — spawns one `jax.distributed` process per GPU and re-runs
  `test_inference_sharding.py` inside a real multi-process environment (the only *pytest*
  that exercises `process_allgather` / per-process seed folding). Needs **≥2 real GPUs**;
  skips otherwise (no interactive prompt, unlike the old suite).
- `multinode_validate.py` + `multinode_validate.slurm` — standalone **multi-NODE** driver:
  one process per GPU across ≥2 nodes, `jax.distributed.initialize()` before importing
  gigalens, then tiny MAP/SVI/HMC/MCLMC on single-lensplane and multiplane, checking
  finite/shape/replicated per rank. Cross-node NCCL smoke, not inference.
- `_models.py` — shared, pytest-free model builders (`build_single_lensplane`,
  `build_multiplane` = two mass planes / recursive ray-shooting) used by BOTH the pytest
  suite and the multi-node driver, so they build identical models.
- `conftest.py` — device setup + fixtures over `_models`.

## Two ways to get ≥2 devices

**Faked CPU devices (default).** `conftest.py` sets `JAX_PLATFORMS=cpu` +
`--xla_force_host_platform_device_count=4`, so the sharding-property test runs multi-device
on CPU with no allocation. `JAX_PLATFORMS=cpu` is required because the fake-device flag only
splits the CPU host platform — on a node that also has a GPU (e.g. a NERSC login node) JAX
would otherwise pick the single GPU and ignore the flag.

**Real GPUs.** Set `GIGALENS_TEST_FORCE_CPU=0` on a multi-GPU node. The parent then sees the
real GPUs, so `test_multihost.py` runs, and its workers exercise the genuine distributed
paths end-to-end (`jax.distributed.initialize`, per-process seed folding, `process_allgather`,
NCCL) and check that the gathered results are correct.

Knobs: `GIGALENS_TEST_FAKE_DEVICES` (fake CPU device count, default 4),
`GIGALENS_TEST_FORCE_CPU` (`1`/`0`), `GIGALENS_MULTIHOST_PROCS` (processes/GPUs, default 2).

## Running

Login node / CI (faked CPU devices; multi-host test skips):

```bash
JAX_ENABLE_X64=1 pytest tests/multidevice --confcutdir=tests/multidevice
```

Multi-GPU node (full contract, including the real multi-host driver):

```bash
GIGALENS_TEST_FORCE_CPU=0 JAX_ENABLE_X64=1 pytest tests/multidevice --confcutdir=tests/multidevice
```

`--confcutdir=tests/multidevice` (and this dir's `pytest.ini`) keep the TF-era top-level
`tests/conftest.py` from being collected. On the NERSC login node, see
`~/.claude` memory `run-new-api-gigalens-tests-on-login-node` for the pytest-shim recipe.

Multi-NODE (≥2 nodes, real cross-node NCCL — the `test_multihost.py` pytest only covers
multiple GPUs on ONE node):

```bash
sbatch --account=<acct> --export=ALL,GIGALENS_REPO=$PWD \
       tests/multidevice/multinode_validate.slurm       # 2 nodes x 4 GPUs, debug qos
```

`GIGALENS_PY` overrides the interpreter (default: the `gigalens_oldapi` conda python — GPU
JAX 0.9.1 + blackjax). Rank 0 prints a `PASS`/`FAIL` table per (model, algorithm) and the
job exits non-zero if any rank failed.

## Verified

- **CPU-faked (4 devices, login node):** 8 passed incl. MCLMC + multiplane MAP.
- **2 GPUs, one node** (`test_multihost.py`): passed — both `jax.distributed` workers ran
  the full suite (incl. MCLMC + multiplane) and exited 0.
- **2 nodes × 4 A100 = 8 GPUs** (`multinode_validate.slurm`, job on nid[002921,002924]):
  `RANK-0 RESULT: ALL PASS` — MAP / SVI / HMC / MCLMC all replicated & finite on **both**
  single-lensplane and multi-lens-plane, exit `0:0`. Confirms MCLMC (shard_map + `psum`
  over the global mesh, no `process_allgather`) reshards correctly across hosts.

## What CPU-faking does and does not cover

## What CPU-faking does and does not cover

CPU-faking validates that multi-device execution runs, that shapes are correct, that
outputs are returned replicated, and that they are placed on every local device
(`addressable_shards == local_device_count`).

Note the new API gathers MAP/SVI/HMC outputs to **fully replicated** regardless of device
or process count — the batch sharding is internal to the compute, so there is no observable
output *partition* to assert (the old-API `not is_fully_replicated` check is obsolete;
confirmed replicated even under a real 2-process `jax.distributed` run). What CPU-faking
does **not** cover is the real distributed machinery — `jax.distributed.initialize`,
`process_allgather`, per-process seed folding, NCCL — which is exercised only by
`test_multihost.py` on real GPUs. Run it before trusting a multinode change.
