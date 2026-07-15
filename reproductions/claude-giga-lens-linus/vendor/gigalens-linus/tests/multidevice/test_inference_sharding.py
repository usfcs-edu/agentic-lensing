"""Sharding-correctness of ModellingSequence MAP / SVI / HMC outputs.

Ported from gigalens-old ``tests/essentials/inference_fns_test.py`` onto the new scene
API. ``ModellingSequence`` still shards the walker/chain batch across a 1-D ``'device'``
mesh internally, but the port had to update the OUTPUT contract the old test asserted --
verified on 4 faked CPU devices, 4 real A100s (single process), and a real 2-GPU
``jax.distributed`` run (via ``test_multihost.py``):

  * MAP ``output_type='best_history'`` was renamed ``'best_step'``
    (``gigalens.jax.inference.ModellingSequence.MAP``).

  * The new API returns MAP outputs **fully replicated for every output_type, including
    'all'** -- ``MAP`` reshards all three arrays to ``P()`` before the ``'all'`` branch
    (for JAX 0.10 strict-gather compatibility of the ``'best'`` indexing that follows).
    The old-API test asserted ``not is_fully_replicated`` for ``'all'``; that no longer
    holds -- confirmed replicated even with ``process_count()==2``. The sharding is now a
    purely internal compute detail; the observable contract is "replicated, right shape".
    (The reshard-to-replicated on the ``'all'`` path looks unnecessary -- that path skips
    the indexing that needs it -- and defeats the memory saving of returning sharded
    histories. Flagged to the maintainer, not worked around here.)

  * New-API HMC returns a topology-dependent array carrying a device axis
    (``(num_results, num_local_devices, chains_per_device, D)``) that the consumer
    normalises downstream (see ``tests/multiplane_demo/mcmc.py``). We therefore assert the
    recovered *content* (element count + trailing param axis) and replication, not the old
    fixed ``(n_chains, n_results, D)`` layout. Under multi-host this is the real check that
    ``process_allgather`` correctly reassembles chains split across processes.

The value of the real multi-host run (``test_multihost.py``) is end-to-end exercise of the
distributed paths -- ``jax.distributed.initialize``, per-process seed folding,
``process_allgather``, NCCL -- and that the gathered results are correct, not a distinct
output-partition assertion (there is none to make on the new API).

Sizes follow the original's note: multiples of the device count for batched axes, small
coprime step counts elsewhere. With 4 faked CPU devices (conftest) ``16`` divides evenly.
"""
import optax
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import tensorflow_probability.substrates.jax.distributions as tfd

pytestmark = pytest.mark.multidevice

# New-API MAP/SVI/HMC outputs are gathered to fully-replicated regardless of device or
# process count (the batch sharding is internal to the compute). The only device-count-
# dependent, observable fact is that a replicated array is placed on every local device,
# i.e. addressable_shards == local_device_count -- a weak "it ran on the mesh" check.
_multi_dev = jax.device_count() > 1


def test_map_best_output(shared_model_seq, shared_start):
    opt = optax.adabelief(1e-2, b1=0.95, b2=0.99)
    num_steps, n_samples = 5, 16

    best, lps, chisq = shared_model_seq.MAP(
        optimizer=opt, n_samples=n_samples, num_steps=num_steps, seed=0,
        output_type="best", pbar_interval=0,
    )

    assert best.sharding.is_fully_replicated, "best params should be replicated"
    assert lps.sharding.is_fully_replicated, "log-probabilities should be replicated"
    assert chisq.sharding.is_fully_replicated, "chi-squared should be replicated"

    assert best.shape == (len(shared_start),)
    assert lps.shape == () or lps.shape == (1,)
    assert chisq.shape == (num_steps,)


def test_map_best_step_output(shared_model_seq, shared_start):
    opt = optax.adabelief(1e-2, b1=0.95, b2=0.99)
    num_steps, n_samples = 5, 16

    best, lps, chisq = shared_model_seq.MAP(
        optimizer=opt, n_samples=n_samples, num_steps=num_steps, seed=0,
        output_type="best_step", pbar_interval=0,
    )

    assert best.shape == (num_steps, len(shared_start))
    assert lps.shape == (num_steps,)
    assert chisq.shape == (num_steps,)

    assert best.sharding.is_fully_replicated, "best params should be replicated"
    assert lps.sharding.is_fully_replicated, "log-probabilities should be replicated"
    assert chisq.sharding.is_fully_replicated, "chi-squared should be replicated"


def test_map_all_output(shared_model_seq, shared_start):
    opt = optax.adabelief(1e-2, b1=0.95, b2=0.99)
    num_steps, n_samples = 5, 16

    best, lps, chisq = shared_model_seq.MAP(
        optimizer=opt, n_samples=n_samples, num_steps=num_steps, seed=0,
        output_type="all", pbar_interval=0,
    )

    assert best.shape == (n_samples, num_steps, len(shared_start))
    assert lps.shape == (n_samples, num_steps)
    assert chisq.shape == (n_samples, num_steps)

    # New API gathers the full history to replicated (see module docstring) -- true on
    # faked CPU, single multi-GPU, and real multi-host alike.
    assert best.sharding.is_fully_replicated, "output_type='all' is returned replicated"
    assert lps.sharding.is_fully_replicated
    assert chisq.sharding.is_fully_replicated

    # Weak placement check: a replicated array lives on every local device.
    if _multi_dev:
        n_local = jax.local_device_count()
        assert len(best.addressable_shards) == n_local, "should span the local devices"
        assert len(lps.addressable_shards) == n_local
        assert len(chisq.addressable_shards) == n_local


def test_svi_output(shared_model_seq, shared_start):
    opt = optax.adabelief(1e-4, b1=0.95, b2=0.99)
    n_steps, n_vi = 5, 16

    qz, loss_hist = shared_model_seq.SVI(
        optimizer=opt, n_vi=n_vi, num_steps=n_steps, start=shared_start, pbar_interval=0,
    )

    assert qz.mean().is_fully_replicated, "qz should be replicated"
    assert loss_hist.is_fully_replicated, "loss history should be replicated"

    assert loss_hist.shape == (n_steps,)
    assert isinstance(qz, tfd.Distribution)


def test_hmc_output(shared_model_seq, shared_start):
    mock_qz = tfd.MultivariateNormalDiag(shared_start, jnp.ones_like(shared_start) * 1e-2)
    n_chains, n_burnin, n_results = 16, 5, 9
    D = len(shared_start)

    samples = shared_model_seq.HMC(
        mock_qz, n_hmc=n_chains, num_results=n_results, num_burnin_steps=n_burnin,
        pbar_interval=0,
    )

    # HMC gathers across hosts (process_allgather) -> fully addressable everywhere.
    assert samples.is_fully_replicated, "samples should be replicated after the gather"
    # The device axis makes the raw layout topology-dependent; assert the content instead:
    # every (chain, draw) position of the D-vector must be present, params on the last axis.
    assert samples.shape[-1] == D, "params must be the trailing axis"
    assert samples.size == n_chains * n_results * D, (
        f"expected {n_chains}*{n_results}*{D} sample values, got shape {samples.shape}"
    )
    assert np.all(np.isfinite(np.asarray(samples)))


def test_full_inference_pipeline(shared_model_seq, shared_start):
    map_opt = optax.adabelief(1e-2, b1=0.95, b2=0.99)
    best, _, _ = shared_model_seq.MAP(
        optimizer=map_opt, n_samples=8, num_steps=2, seed=0, output_type="best",
        pbar_interval=0,
    )

    svi_opt = optax.adabelief(1e-4, b1=0.95, b2=0.99)
    qz, _ = shared_model_seq.SVI(
        start=best, optimizer=svi_opt, n_vi=8, num_steps=2, seed=0, pbar_interval=0,
    )

    samples = shared_model_seq.HMC(qz, n_hmc=8, num_burnin_steps=2, num_results=2,
                                   pbar_interval=0)
    assert samples is not None
    assert np.all(np.isfinite(np.asarray(samples)))


def test_mclmc_output(shared_model_seq, shared_start):
    # MCLMC (gigalens.jax.experimental) shards chains across the 'device' mesh via
    # shard_map + jax.lax.psum reductions, then reshards the samples to replicated. Unlike
    # HMC it has NO process_allgather -- its cross-node correctness is exercised by the
    # multi-node driver; here we check the single-process multi-device contract.
    from gigalens.jax.experimental.mclmc import MCLMC_JIT

    qz = tfd.MultivariateNormalDiag(shared_start, jnp.ones_like(shared_start) * 1e-2)
    n_chains, n_burnin, n_results = 16, 20, 4  # burnin split 4/12/4 across the tune stages
    D = len(shared_start)

    samples = MCLMC_JIT(
        shared_model_seq, qz, n_hmc=n_chains, num_burnin_steps=n_burnin,
        num_results=n_results, seed=0, progress_bar=False,
    )

    assert samples.sharding.is_fully_replicated, "MCLMC reshards samples to replicated"
    assert samples.shape == (n_chains, n_results, D)
    assert np.all(np.isfinite(np.asarray(samples)))


def test_multiplane_map_output(multiplane_seq, multiplane_start):
    # Multi-lens-plane (two mass planes -> recursive ray-shooting) MUST shard the MAP
    # walker batch exactly like the single-plane model. Smoke it here; the full
    # multiplane x {MAP,SVI,HMC,MCLMC} cross-node validation is the sbatch driver.
    opt = optax.adabelief(1e-2, b1=0.95, b2=0.99)
    best, lps, chisq = multiplane_seq.MAP(
        optimizer=opt, n_samples=16, num_steps=3, seed=0, output_type="best",
        pbar_interval=0,
    )
    assert best.sharding.is_fully_replicated
    assert best.shape == (len(multiplane_start),)
    assert chisq.shape == (3,)
    assert np.all(np.isfinite(np.asarray(best)))
