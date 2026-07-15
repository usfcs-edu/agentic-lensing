import functools
import sys

import jax.random
import numpy as np
import optax
from tensorflow_probability.python.internal import unnest
import tensorflow_probability.substrates.jax as tfp
import time
from typing import Literal
from jax import jit, pmap
from jax import numpy as jnp
from jax.sharding import NamedSharding 
from jax.sharding import PartitionSpec as P
from tensorflow_probability.substrates.jax import (
    distributions as tfd,
    bijectors as tfb,
    experimental as tfe,
)
from tqdm.auto import tqdm

# JAX 0.10 promoted shard_map to the top level and removed
# `jax.experimental.shard_map` as an importable submodule. Fall back to the
# experimental path on older JAX so this file works on both kernels.
try:
    _shard_map = jax.shard_map  # type: ignore[attr-defined]
except AttributeError:
    from jax.experimental.shard_map import shard_map as _shard_map  # noqa: F401

# `jax.reshard` is the JAX 0.10+ API for changing an array's sharding outside
# of jit. Older JAX releases don't have it; `jax.device_put` with the same
# NamedSharding is the equivalent and is a no-op when already on those devices.
_reshard = getattr(jax, 'reshard', jax.device_put)


# `jax.lax.pvary(x, axis_name)` was removed in JAX 0.10; the replacement is
# `jax.lax.pcast(x, axis_name, to='varying')`. JAX 0.10's pcast also errors if
# the source is already 'varying' (whereas pvary was idempotent), so swallow
# that case to preserve the original semantics on both JAX versions.
def _pvary(x, axis_name):
    pcast = getattr(jax.lax, 'pcast', None)
    if pcast is not None:
        try:
            return pcast(x, axis_name, to='varying')
        except ValueError as e:
            if 'from=varying' in str(e):
                return x
            raise
    return jax.lax.pvary(x, axis_name)  # type: ignore[attr-defined]


import gigalens.inference
import gigalens.jax.simulator as sim
import gigalens.model

import warnings

if (jax.process_count() > 1) and (not jax.distributed.is_initialized()):
    warnings.warn('jax.distributed.initialize() was not called. For multinode, please call it before running any JAX functions.')
mesh = jax.make_mesh((len(jax.devices()),), ('device',))


def sharding(obj):
    if hasattr(obj,"sharding"): return obj.sharding
    else: return None

def is_ipython():
    try: 
        from IPython import get_ipython
        return (get_ipython() != None)
    except(ImportError,NameError): return False
        
class ModellingSequence(gigalens.inference.ModellingSequenceInterface):

    # SCENE-ONLY (old gigalens API dropped). A ModellingSequence is built from a scene
    # LensModel + scene ProbModel via ``from_scene``. MAP / SVI / HMC call ``log_prob(z)``
    # directly: the ProbModel's per-dataset SceneSimulators are batch-flexible (the grid
    # carries a singleton batch axis that broadcasts from the params), so they serve any
    # MCMC/MAP batch size without a per-call rebuild -- and the multi-dataset ``sees`` loop
    # is reachable for free. Inference touches only the prob_model surface
    # (prior / bij / log_prob(z) / observed_image), on the scene path.
    scene_model = None  # the scene LensModel (set in __init__)

    def __init__(self, prob_model):
        """Build a scene ModellingSequence from a scene ProbModel alone.

        ``prob_model`` is a :class:`gigalens.jax.scene_prob_model.ProbModel` built from
        a scene :class:`gigalens.jax.scene.LensModel`. Everything else is derived from it:
        ``scene_model`` / ``phys_model`` are set to ``prob_model.model``, and
        ``sim_config`` is taken from the first dataset (``prob_model.datasets[0].sim_config``).

        All four attributes (``phys_model``, ``prob_model``, ``sim_config``,
        ``scene_model``) are populated for downstream consumers
        (``InferenceContext.from_modelling_sequence``, the ``Posterior`` view, and the
        legacy normalizing-flows path which reads ``sim_config``). The authoritative
        per-band grids/PSFs for the likelihood are ``prob_model.simulators``, built from
        each ImageData's own ``sim_config``; the single stored ``sim_config`` is used only
        for plotting extent / source-plane FOV defaults."""
        self.prob_model = prob_model
        self.phys_model = prob_model.model
        # sim_config is used only for plotting extent / source-plane FOV defaults. Take the
        # first dataset that actually has one; a point-source-only ProbModel has none of an
        # imaging grid, so this is None there (tolerated — nothing on the fit path reads it).
        self.sim_config = next(
            (getattr(d, "sim_config", None) for d in prob_model.datasets
             if getattr(d, "sim_config", None) is not None), None)
        self.scene_model = prob_model.model

    @classmethod
    def from_scene(cls, prob_model):
        """Back-compat alias for ``ModellingSequence(prob_model)``."""
        return cls(prob_model)

    @property
    def is_scene_backed(self) -> bool:
        return self.scene_model is not None

    def MAP(
            self,
            optimizer: optax.GradientTransformation,
            start=None,
            n_samples=500,
            num_steps=350,
            seed=0,
            output_type: Literal["all", "best_step", "best"] = "best",
            pbar_interval = 100,
    ):        
        dev_cnt = len(jax.devices())
        n_samples = (n_samples // dev_cnt) * dev_cnt
        jax_seed = jax.random.PRNGKey(seed)

        start = (
            self.prob_model.prior.sample(n_samples, seed=jax_seed)
            if start is None
            else start
        )
        params = self.prob_model.bij.inverse(start)  # constrained dict -> flat (n_samples, D)

        # MAP-scoped remat (compute-profiling C-16/C-17): batched MAP is
        # memory-bound, so run it on a twin prob model with remat_basis forced
        # on where measured beneficial (ss < 3); samplers elsewhere keep the
        # user's config untouched. Exact — remat re-runs identical ops.
        map_prob = (self.prob_model.with_map_remat()
                    if hasattr(self.prob_model, "with_map_remat")
                    else self.prob_model)
        # Printed, never silent (§0): the MAP-time remat decision is invisible
        # to the dataset sim_config (and hence to InferenceContext.hash and the
        # model card) — say what MAP actually runs.
        _n_remat = sum(bool(getattr(s, "remat_basis", False))
                       for s in getattr(map_prob, "simulators", []))
        _n_sims = len(getattr(map_prob, "simulators", [])) or 1
        print(f"MAP: remat_basis effective for {_n_remat}/{_n_sims} simulator(s) "
              f"({'MAP-scoped override, ss<3' if map_prob is not self.prob_model else 'config values unchanged'}).")

        def loss(z):
            lp, chisq = map_prob.log_prob(z)
            # loss_normalization is the observation-agnostic MAP scale (imaging pixels when
            # present, else the observable count) — so this never divides by zero for a
            # point-source-only or otherwise pixel-less model.
            return -jnp.mean(lp) / map_prob.loss_normalization, (lp, chisq)

        loss_and_grad = jit(jax.value_and_grad(loss, has_aux=True))
        
        # Explicit sharding of params
        params = jax.device_put(params, NamedSharding(mesh, P("device")))
        

        
        pbar_run = (pbar_interval>0) and (jax.process_index()==0) and (sys.stdout.isatty() or is_ipython()) 
        pbar = tqdm(total=num_steps,position=0,leave=True,miniters=pbar_interval,disable=not pbar_run,mininterval=0.1) 
        def pbar_display(args):
                pbar.n = args[0].item()
                pbar.set_description(f"Chi-squared: {float(jnp.nanmin(args[1]).item()):.3f}")
                return None

        @functools.partial(jax.jit, static_argnums=(1, 2))
        @functools.partial(_shard_map, mesh=mesh, in_specs=(P('device'), None, None), out_specs=P('device'))
        def run_map(params, optimizer, output_type):
            opt_state = optimizer.init(params)

            # Explicitly tell JAX that opt_state varies across devices and is not replicated.
            # _pvary handles the JAX 0.10 rename of pvary -> pcast(..., to='varying').
            pvary = lambda x: _pvary(x, 'device') if isinstance(x, jax.Array) else x
            opt_state = jax.tree_util.tree_map(pvary, opt_state)
            n_iter = 1
            
            def one_step(carry, _):
                params, opt_state, n_iter = carry
                (_, (lp, chisq)), grads = loss_and_grad(params)
                updates, opt_state = optimizer.update(grads, opt_state)
                params = optax.apply_updates(params, updates)

                jax.lax.cond(pbar_run & (jax.lax.axis_index('device')==0) & (n_iter%pbar_interval == 0),
                             lambda:jax.debug.callback(pbar_display,(n_iter,chisq)), lambda: None)

                carry = (params, opt_state,n_iter+1)

                # Saves memory by not materializing full histories in memory
                if output_type in ["best_step", "best"]:
                    best_lp_idx = jnp.nanargmax(lp)
                    b = (params[best_lp_idx][None], lp[best_lp_idx][None], chisq[best_lp_idx][None])
                else:
                    b = (params, lp, chisq)

                return carry, b
            
            _, b = jax.lax.scan(one_step, (params, opt_state,n_iter), length=num_steps)
            # (num_steps, num_samples) -> (num_samples, num_steps)
            b = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), b)
            return b

        # first 2 dims are shape (total_samples, num_steps) or (num_devices, num_steps)
        map_samples, map_lps, map_chisqs = run_map(params, optimizer, output_type)
        # Reshard outputs to fully-replicated. Without this, JAX 0.10's strict
        # gather/sharding rules raise ShardingTypeError on the indexing below
        # (e.g. `map_samples[best_lp_sample_idxs, jnp.arange(num_steps)]`)
        # because the chain axis is sharded across 'device'. No-op on older JAX.
        _replicated = NamedSharding(mesh, P())
        map_samples = _reshard(map_samples, _replicated)
        map_lps = _reshard(map_lps, _replicated)
        map_chisqs = _reshard(map_chisqs, _replicated)
        pbar.close()
        
        final_chisq = jnp.nanmin(map_chisqs[:, -1])
        final_chisq = jax.block_until_ready(final_chisq)

        print(f"Final Chi-squared: {final_chisq:.5f}")
                
        if output_type == "all":
            return map_samples, map_lps, map_chisqs
        
        best_lp_sample_idxs = jnp.nanargmax(map_lps, axis=0)

        # map_samples: (num_steps, num_dims), map_lps and map_chisqs: (num_steps,)
        idx = jnp.array((best_lp_sample_idxs, jnp.arange(num_steps)))
        map_samples = map_samples[*idx]
        map_lps = map_lps[*idx]
        map_chisqs = map_chisqs[*idx]

        if output_type == 'best_step':
            return map_samples, map_lps, map_chisqs
        
        elif output_type == 'best':
            best_sample_step_idx = jnp.nanargmax(map_lps)
            return map_samples[best_sample_step_idx], map_lps[best_sample_step_idx], map_chisqs
        
        raise ValueError(f"Illegal value for output_type: {output_type}")

    def SVI(
            self,
            start,
            optimizer: optax.GradientTransformation,
            n_vi=250,
            init_scales=1e-3,
            num_steps=500,
            seed=0,
            pbar_interval=100,
    ):
        dev_cnt = len(jax.devices())

        n_vi = (n_vi // dev_cnt) * dev_cnt

        # Build the surrogate at the simulator's working precision (float64 when
        # likelihood_precision="float64", else float32) so the variational
        # parameters, samples, and ELBO gradient follow sim_config rather than the
        # global jax_enable_x64 flag. Without this the surrogate is always float64
        # under x64 -- mismatching a float32 config, and tripping a dtype clash in
        # the cov bijector below.
        # Ask the ProbModel for its working precision rather than reaching into
        # simulators[0] (which IndexErrors for a point-source-only model with no imaging
        # simulator); high_precision resolves for any dataset mix.
        prec_dtype = jnp.float64 if self.prob_model.high_precision else jnp.float32

        pbar_run = (pbar_interval>0) and (jax.process_index()==0) and (sys.stdout.isatty() or is_ipython()) 
        pbar = tqdm(total=num_steps,position=0,leave=True,miniters=pbar_interval,disable=not pbar_run, mininterval=0.1)
        def pbar_display(args):
                pbar.n = args[0].item()
                pbar.set_description(f"-ELBO: {float(args[1].item()):.3f}")
                return None
        
        scale = (
            jnp.diag(jnp.ones(jnp.size(start), dtype=prec_dtype)) * jnp.asarray(init_scales, prec_dtype)
            if jnp.size(init_scales) == 1
            else jnp.asarray(init_scales, dtype=prec_dtype)
        )
        # diag_shift must share the surrogate's dtype: a float32 constant clashes
        # with a float64 scale inside FillScaleTriL (TypeError under jax_enable_x64).
        cov_bij = tfp.bijectors.FillScaleTriL(
            diag_bijector=tfb.Exp(), diag_shift=jnp.asarray(1e-6, prec_dtype)
        )
        # Build replicated_params without inheriting any Explicit-axis tag
        # from `start`. Without this, JAX 0.10 propagates the outer Explicit
        # context through scan-in-shard_map and trips the closing-over check
        # on the very first slice (`params[:n_params]`).
        _start_np = np.asarray(jnp.squeeze(start))
        _cov_inv_np = np.asarray(cov_bij.inverse(scale))
        replicated_params = jnp.asarray(
            np.concatenate([_start_np, _cov_inv_np], axis=0)
        ).astype(prec_dtype)

        n_params = int(_start_np.size)

        # Take mean and cov_chol_raw as separate args. This avoids slicing
        # `qz_params[:n_params]` *inside* the value_and_grad trace, which in
        # JAX 0.10 trips "Closing over inputs to shard_map" because the
        # inner trace doesn't preserve the outer shard_map's Manual-axis
        # context for the sliced operand.
        def neg_elbo(mean, cov_chol_raw, jax_seed):
            cov = cov_bij.forward(cov_chol_raw)
            qz = tfd.MultivariateNormalTriL(loc=mean, scale_tril=cov)
            jax_seed = jax_seed if jax_seed.ndim == 1 else jax_seed[0]
            z = qz.sample(n_vi // dev_cnt, seed=jax_seed)
            lps = qz.log_prob(z)
            return jnp.mean(lps - self.prob_model.log_prob(z)[0])

        neg_elbo_and_grad = jax.value_and_grad(neg_elbo, argnums=(0, 1))

        # Restructure for JAX 0.10: put scan *inside* shard_map (rather than
        # shard_map inside scan inside jit). With the previous nesting, JAX
        # 0.10 can't bridge the outer Explicit-axis context into the inner
        # shard_map's Manual mode and raises "Closing over inputs to shard_map
        # where the input is sharded on Explicit axes". Putting scan inside
        # shard_map keeps every per-step op in Manual mode, which works on
        # both JAX versions and preserves the original RNG stream.
        def one_step_sharded(carry, _):
            params, opt_state, key, best_params, best_loss, n_iter = carry
            key, curr = jax.random.split(key)
            # Same per-device sub-keys as the original (jax.random.split on
            # the parent key is identical), then each device selects its own.
            keys = jax.random.split(curr, dev_cnt)
            my_key = keys[jax.lax.axis_index('device')]

            mean = params[:n_params]
            cov_chol_raw = params[n_params:]
            val, (grad_mean, grad_cov) = neg_elbo_and_grad(mean, cov_chol_raw, my_key)
            grad = jnp.concatenate([grad_mean, grad_cov])
            loss = jax.lax.pmean(val, axis_name='device')
            grad = jax.lax.pmean(grad, axis_name='device')

            better = loss < best_loss
            best_params = jax.lax.select(better, params, best_params)
            best_loss = jax.lax.select(better, loss, best_loss)

            updates, opt_state = optimizer.update(grad, opt_state)
            params = optax.apply_updates(params, updates)

            # Only one device fires the pbar callback to avoid duplicate prints.
            jax.lax.cond(
                pbar_run
                & (jax.lax.axis_index('device') == 0)
                & (n_iter % pbar_interval == 0),
                lambda: jax.debug.callback(pbar_display, (n_iter, loss)),
                lambda: None,
            )

            return (params, opt_state, key, best_params, best_loss, n_iter + 1), loss

        @jax.jit
        @functools.partial(
            _shard_map,
            mesh=mesh,
            in_specs=(P(), P(), P()),
            out_specs=(P(), P(), P()),
        )
        def run_svi(initial_params, initial_opt_state, initial_key):
            initial_carry = (
                initial_params, initial_opt_state, initial_key,
                initial_params, jnp.inf, 1,
            )
            (final_carry, loss_hist) = jax.lax.scan(
                one_step_sharded, initial_carry, length=num_steps,
            )
            _, _, _, best_params, best_loss, _ = final_carry
            return best_params, best_loss, loss_hist

        opt_state = optimizer.init(replicated_params)
        key = jax.random.PRNGKey(seed)

        best_params, best_loss, loss_hist = run_svi(replicated_params, opt_state, key)
        pbar.close()
        
        best_loss = jax.block_until_ready(best_loss)

        mean = best_params[:n_params]
        cov = cov_bij.forward(best_params[n_params:])
        qz = tfd.MultivariateNormalTriL(loc=mean, scale_tril=cov)
        
        return qz, loss_hist
    
    def HMC(
            self,
            q_z,
            init_eps=0.3,
            init_l=3,
            n_hmc=50,
            num_burnin_steps=250,
            num_results=750,
            max_leapfrog_steps=30,
            num_steps_between_results=0,
            seed=0,
            pbar_interval = 100,
    ):
        dev_cnt = len(jax.devices())
        local_dev_cnt = len(jax.local_devices())
        # seeds are per process (node)
        seeds = jax.random.split(jax.random.fold_in(jax.random.PRNGKey(seed), jax.process_index()), local_dev_cnt)
        n_hmc = (n_hmc // dev_cnt) * dev_cnt

        momentum_distribution = tfd.MultivariateNormalFullCovariance(
            loc=jnp.zeros_like(q_z.mean()),
            covariance_matrix=jnp.linalg.inv(q_z.covariance()),
        )

        pbar_run = (pbar_interval>0) and (jax.process_index()==0) and (sys.stdout.isatty() or is_ipython())
        pbar=tqdm(total=num_results+num_burnin_steps, position=0, leave=True, miniters=pbar_interval, disable = not pbar_run, desc="HMC Progress",mininterval=0.1)
        def pbar_display_fn(curr, pkr):
            def pbar_display(pkr):
                pbar.n = int(pkr.step)
                pbar.refresh()
                return None
            jax.lax.cond(pbar_run & (jax.lax.axis_index('device')==0) & (pkr.step%pbar_interval == 0),
                             (lambda:jax.debug.callback(pbar_display,pkr)), lambda: None)
            return ()

        trace_fn = pbar_display_fn if pbar_run else lambda curr,pkr: None

        @jit
        def log_prob(z):
            return self.prob_model.log_prob(z)[0]

        @functools.partial(jax.pmap, axis_name='device')
        def run_chain(seed):
            start = q_z.sample(n_hmc // dev_cnt, seed=seed)
            num_adaptation_steps = int(num_burnin_steps * 0.8)
            mc_kernel = tfe.mcmc.PreconditionedHamiltonianMonteCarlo(
                target_log_prob_fn=log_prob,
                momentum_distribution=momentum_distribution,
                step_size=init_eps,
                num_leapfrog_steps=init_l,
            )

            mc_kernel = tfe.mcmc.GradientBasedTrajectoryLengthAdaptation(
                mc_kernel,
                num_adaptation_steps=num_adaptation_steps,
                max_leapfrog_steps=max_leapfrog_steps,
            )
            mc_kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
                inner_kernel=mc_kernel, num_adaptation_steps=num_adaptation_steps
            )
            
            return tfp.mcmc.sample_chain(
                num_results=num_results,
                num_burnin_steps=num_burnin_steps,
                current_state=start,
                trace_fn= trace_fn,
                seed=seed,
                num_steps_between_results=num_steps_between_results,
                kernel=mc_kernel,
            )

        start = time.time()
        samples = run_chain(seeds)
        end = time.time()
        pbar.close()
        print(f"Sampling took {(end - start):.1f}s")
        # The line below aggregates over all devices
        samples = jax.experimental.multihost_utils.process_allgather(samples.all_states)
        # Output from the line below is (num_devices, num_steps, n_hmc_per_device, num_params)
        samples = samples.reshape(samples.shape[0] * samples.shape[1], *samples.shape[2:])
        # Final output is (num_steps, num_devices, n_hmc_per_device, num_params)
        samples = jnp.swapaxes(samples, 0, 1)
        return samples

    # Alternative HMC that performs an adaptive burn-in phase and optionally re-estimates the momentum preconditioner from burn-in samples,
    # unlike HMC_multi which uses the fixed variational covariance.
    def HMC_alt_multi(
            self,
            q_z,
            init_eps=0.3,
            init_l=3,
            n_hmc=50,
            n_vi=1000,
            num_burnin_steps=250,
            proportion_burnin_to_use=0.9,
            num_results=750,
            max_leapfrog_steps=30,
            num_steps_between_results=0,
            seed=0,
            force_use_burnin=False,
    ):
        dev_cnt = len(jax.devices())
        local_dev_cnt = len(jax.local_devices())
        seeds = jax.random.split(jax.random.fold_in(jax.random.PRNGKey(seed), jax.process_index()), local_dev_cnt)
        n_hmc = (n_hmc // dev_cnt) * dev_cnt

        momentum_distribution = tfd.MultivariateNormalFullCovariance(
            loc=jnp.zeros_like(q_z.mean()),
            covariance_matrix=jnp.linalg.inv(q_z.covariance()),
        )

        @jit
        def log_prob(z):
            return self.prob_model.log_prob(z)[0]

        @pmap
        def run_burnin_chain(seed):
            start = q_z.sample(n_hmc // dev_cnt, seed=seed)
            num_adaptation_steps = int(num_burnin_steps * 0.8)

            mc_kernel = tfe.mcmc.PreconditionedHamiltonianMonteCarlo(
                target_log_prob_fn=log_prob,
                momentum_distribution=momentum_distribution,
                step_size=init_eps,
                num_leapfrog_steps=init_l
            )

            mc_kernel = tfe.mcmc.GradientBasedTrajectoryLengthAdaptation(
                inner_kernel=mc_kernel,
                num_adaptation_steps=num_adaptation_steps,
                max_leapfrog_steps=max_leapfrog_steps
            )

            mc_kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
                inner_kernel=mc_kernel,
                num_adaptation_steps=num_adaptation_steps
            )
            
            results = tfp.mcmc.sample_chain(
                num_results=num_burnin_steps,
                current_state=start,
                kernel=mc_kernel,
                trace_fn=lambda curr,pkr: None,
                return_final_kernel_results=True,
                num_steps_between_results=num_steps_between_results,
                seed=seed
            )    
            kernel_results = results.final_kernel_results
            step_size = unnest.get_innermost(kernel_results, 'step_size')
            num_leapfrog_steps = unnest.get_innermost(kernel_results, 'num_leapfrog_steps')
            
            return results.all_states, step_size, num_leapfrog_steps
            
        
        @pmap
        def run_chain(seed, dev_idx):            
            final_kernel = tfe.mcmc.PreconditionedHamiltonianMonteCarlo(
                target_log_prob_fn=log_prob,
                momentum_distribution=momentum_distribution,
                step_size=step_size[dev_idx],
                num_leapfrog_steps=num_leapfrog_steps[dev_idx]
            )

            return tfp.mcmc.sample_chain(
                num_results=num_results,
                current_state=all_states[dev_idx, -1],
                kernel=final_kernel,
                trace_fn=lambda curr,pkr: None,
                num_steps_between_results=num_steps_between_results,
                seed=seed
            ).all_states
            

        start = time.time()
        num_burnin_to_use = int(proportion_burnin_to_use * num_burnin_steps)
        mle_cov = None
        if num_burnin_steps > 0:
            # tuple: (all_states, step_size, num_leapfrog_steps), dim 0 of each tensor is device idx 
            all_states, step_size, num_leapfrog_steps = run_burnin_chain(seeds)
            # gather_samples is (num_processes, num_devices_per_process, num_steps, n_hmc_per_device, n_dims)
            gather_samples = jax.experimental.multihost_utils.process_allgather(all_states)
            gather_samples = jnp.moveaxis(gather_samples, 2, 0)

            if num_burnin_to_use < gather_samples.shape[0]:
                gather_samples = gather_samples[num_burnin_to_use: ]
            
                all_samples = gather_samples.reshape(-1, gather_samples.shape[-1])
                mle_cov = jnp.cov(all_samples, rowvar=False)
                proposed_normal_distribution = tfd.MultivariateNormalFullCovariance(
                    loc=jnp.median(all_samples, axis=0),
                    covariance_matrix=mle_cov,
                )
                print(f"force_use_burnin: {force_use_burnin}")
                if force_use_burnin:
                    momentum_distribution = tfd.MultivariateNormalFullCovariance(
                        loc=jnp.zeros_like(q_z.mean()),
                        covariance_matrix=jnp.linalg.inv(mle_cov),
                    )
                    print('Switched to burnin model')
                    # print(f'mle_cov: {mle_cov}')
                else:
                    # pick lower elbo distribution
                    sharding = NamedSharding(mesh, P('device'))


                    @jit
                    @functools.partial(_shard_map, mesh=mesh, in_specs=(None, None, P('device')), out_specs=P())
                    def elbo(loc, cov, jax_seed):
                        qz = tfd.MultivariateNormalFullCovariance(loc=loc, covariance_matrix=cov)
                        z = qz.sample(n_vi // dev_cnt, seed=jax_seed[0])
                        lps = qz.log_prob(z)
                        return jax.lax.pmean(jnp.mean(lps - self.prob_model.log_prob(z)[0]), axis_name="device")

                    
                    # @functools.partial(jit, static_argnums=(0,))
                    # def elbo(qz):
                    #     z = qz.sample(n_vi, seed=jax.random.PRNGKey(0))
                    #     lps = qz.log_prob(z)
                    #     return jnp.mean(lps - self.prob_model.log_prob(z, simulator=elbo_lens_sim)[0])
                    jax_seeds = jax.random.split(jax.random.PRNGKey(0), dev_cnt)
                    # JAX 0.10's shard_map enforces strict in_specs matching;
                    # pre-shard `jax_seeds` so its sharding matches P('device').
                    # No-op on older JAX (jax.device_put with same sharding).
                    jax_seeds = _reshard(jax_seeds, sharding)
                    burnin_elbo = elbo(jnp.median(all_samples, axis=0), mle_cov, jax_seeds)
                    q_z_elbo = elbo(q_z.loc, q_z.covariance(), jax_seeds)
                    print(f'Burn-in elbo: {burnin_elbo}')
                    print(f'q_z elbo: {q_z_elbo}')
                    if burnin_elbo < q_z_elbo:
                        momentum_distribution = tfd.MultivariateNormalFullCovariance(
                        loc=jnp.zeros_like(q_z.mean()),
                        covariance_matrix=jnp.linalg.inv(mle_cov),
                    )
                        print('Switched to burnin model')
        

        dev_idxs = jnp.arange(local_dev_cnt)
        all_states = run_chain(seeds, dev_idxs)
        end = time.time()
        print(f"Sampling took {(end - start):.1f}s")
        all_samples = jax.experimental.multihost_utils.process_allgather(all_states)
        # device_partitioned_samples is (num_devices, num_steps, n_hmc_per_device, num_params)
        device_partitioned_samples = all_samples.reshape(all_samples.shape[0] * all_samples.shape[1], *all_samples.shape[2:])
        # Final device_partitioned_samples is (num_steps, num_devices, n_hmc_per_device, num_params)
        device_partitioned_samples = jnp.swapaxes(device_partitioned_samples, 0, 1)
        return device_partitioned_samples#, mle_cov    # mle_cov for debugging only
