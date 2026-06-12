import functools
import sys

import jax.random
import optax
from tensorflow_probability.python.internal import unnest
import tensorflow_probability.substrates.jax as tfp
import time
from typing import Literal
from jax import jit, pmap
from jax import numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P
from tensorflow_probability.substrates.jax import (
    distributions as tfd,
    bijectors as tfb,
    experimental as tfe,
)
from tqdm import tqdm


import gigalens.inference
import gigalens.jax.simulator as sim
import gigalens.model

import warnings

if (jax.process_count() > 1) and (not jax.distributed.is_initialized()):
    warnings.warn('jax.distributed.initialize() was not called. For multinode, please call it before running any JAX functions.')
mesh = jax.make_mesh((len(jax.devices()),), ('device',))

def is_ipython():
    try: 
        from IPython import get_ipython
        return (get_ipython() != None)
    except(ImportError,NameError): return False

class ModellingSequence(gigalens.inference.ModellingSequenceInterface):

    def MAP(
            self,
            optimizer: optax.GradientTransformation,
            start=None,
            n_samples=500,
            num_steps=350,
            seed=0,
            output_type: Literal["all", "best_step", "best"] = "best",
            pbar_interval = 5
    ):        
        dev_cnt = len(jax.devices())
        n_samples = (n_samples // dev_cnt) * dev_cnt
        lens_sim = sim.LensSimulator(
            self.phys_model,
            self.sim_config,
            bs=n_samples // dev_cnt,
        )
        jax_seed = jax.random.PRNGKey(seed)

        start = (
            self.prob_model.prior.sample(n_samples, seed=jax_seed)
            if start is None
            else start
        )
        params = jnp.stack(self.prob_model.bij.inverse(start)).T

        def loss(z):
            lp, chisq = self.prob_model.log_prob(lens_sim, z)
            return -jnp.mean(lp) / jnp.size(self.prob_model.observed_image), (lp, chisq)

        loss_and_grad = jit(jax.value_and_grad(loss, has_aux=True))
        
        # Explicit sharding of params
        params = jax.device_put(params, NamedSharding(mesh, P("device")))

        
        pbar_run = (pbar_interval>0) and (jax.process_index()==0) and (sys.stdout.isatty() or is_ipython()) 
        pbar = tqdm(total=num_steps,position=0,leave=True,miniters=pbar_interval,disable=not pbar_run) 
        def pbar_display(args):
                pbar.n = args[0].item()
                pbar.set_description(f"Chi-squared: {float(jnp.nanmin(args[1]).item()):.3f}")
                return None

        @functools.partial(jax.jit, static_argnums=(1, 2))
        @functools.partial(jax.experimental.shard_map.shard_map, mesh=mesh, in_specs=(P('device'), None, None), out_specs=P('device'))
        def run_map(params, optimizer, output_type):
            opt_state = optimizer.init(params)

            # Explicitly tell JAX that opt_state varies across devices and is not replicated
            pvary = lambda x: jax.lax.pvary(x, 'device') if isinstance(x, jax.Array) else x
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
        pbar.close()
        
        final_chisq = jnp.nanmin(map_chisqs[:, -1])
        final_chisq = jax.block_until_ready(final_chisq)

        print(f"Final Chi-squared: {final_chisq:.5f}")
                
        if output_type == "all":
            return map_samples, map_lps, map_chisqs
        
        best_lp_sample_idxs = jnp.nanargmax(map_lps, axis=0)

        # map_samples: (num_steps, num_dims), map_lps and map_chisqs: (num_steps,)
        map_samples = map_samples[best_lp_sample_idxs, jnp.arange(num_steps)]
        map_lps = map_lps[best_lp_sample_idxs, jnp.arange(num_steps)]
        map_chisqs = map_chisqs[best_lp_sample_idxs, jnp.arange(num_steps)]

        if output_type == 'best_step':
            return map_samples, map_lps, map_chisqs
        
        elif output_type == 'best':
            best_sample_step_idx = jnp.nanargmax(map_lps)
            return map_samples[best_sample_step_idx][jnp.newaxis, :], map_lps[best_sample_step_idx], map_chisqs[best_sample_step_idx]
        
        raise ValueError(f"Illegal value for output_type: {output_type}")

    def SVI(
            self,
            start,
            optimizer: optax.GradientTransformation,
            n_vi=250,
            init_scales=1e-3,
            num_steps=500,
            seed=0,
            pbar_interval=5,
    ):
        dev_cnt = len(jax.devices())

        n_vi = (n_vi // dev_cnt) * dev_cnt
        lens_sim = sim.LensSimulator(
            self.phys_model,
            self.sim_config,
            bs=n_vi // dev_cnt
        )

        pbar_run = (pbar_interval>0) and (jax.process_index()==0) and (sys.stdout.isatty() or is_ipython()) 
        pbar = tqdm(total=num_steps,position=0,leave=True,miniters=pbar_interval,disable=not pbar_run)
        def pbar_display(args):
                pbar.n = args[0].item()
                pbar.set_description(f"-ELBO: {float(args[1].item()):.3f}")
                return None
        
        scale = (
            jnp.diag(jnp.ones(jnp.size(start))) * init_scales
            if jnp.size(init_scales) == 1
            else init_scales
        )
        cov_bij = tfp.bijectors.FillScaleTriL(diag_bijector=tfb.Exp(), diag_shift=1e-6)
        replicated_params = jnp.concatenate(
            [jnp.squeeze(start), cov_bij.inverse(scale)], axis=0
        )

        n_params = jnp.size(start)

        # Returns a scalar
        def neg_elbo(qz_params, jax_seed):
            mean = qz_params[:n_params]
            cov = cov_bij.forward(qz_params[n_params:])
            qz = tfd.MultivariateNormalTriL(loc=mean, scale_tril=cov)
            jax_seed = jax_seed if jax_seed.ndim == 1 else jax_seed[0]
            z = qz.sample(n_vi // dev_cnt, seed=jax_seed)
            lps = qz.log_prob(z)
            return jnp.mean(lps - self.prob_model.log_prob(lens_sim, z)[0])

        neg_elbo_and_grad = jit(jax.value_and_grad(neg_elbo, argnums=0))

        @functools.partial(jax.experimental.shard_map.shard_map, mesh=mesh, in_specs=(None, P('device')), out_specs=P())
        def get_update(qz_params, jax_seed):
            val, grad = neg_elbo_and_grad(qz_params, jax_seed)
            return jax.lax.pmean(val, axis_name="device"), jax.lax.pmean(grad, axis_name="device")

        def one_step(carry, _):
            params, opt_state, key, best_params, best_loss,n_iter = carry
            key, curr = jax.random.split(key)
            keys = jax.random.split(curr, dev_cnt)

            loss, grad = get_update(params, keys)

            better = loss < best_loss
            best_params = jax.lax.select(better, params, best_params)
            best_loss = jax.lax.select(better, loss, best_loss)

            updates, opt_state = optimizer.update(grad, opt_state)
            params = optax.apply_updates(params, updates)
            jax.lax.cond(pbar_run & (n_iter%pbar_interval == 0), lambda:jax.debug.callback(pbar_display,(n_iter,loss)), lambda: None)

            return (params, opt_state, key, best_params, best_loss, n_iter+1), loss

        @jax.jit
        def run_svi(initial_carry):
            return jax.lax.scan(one_step, initial_carry, length=num_steps)
        
        opt_state = optimizer.init(replicated_params)
        key = jax.random.PRNGKey(seed)
        initial_carry = (replicated_params, opt_state, key, replicated_params, jnp.inf,1)

        ((_, _, _, best_params, best_loss, _), loss_hist) = run_svi(initial_carry)
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
            seed=0,
            pbar_interval = 0, # progress bars are not performant here and should not be used for long HMC samplings
    ):
        dev_cnt = len(jax.devices())
        local_dev_cnt = len(jax.local_devices())
        # seeds are per process (node)
        seeds = jax.random.split(jax.random.fold_in(jax.random.PRNGKey(seed), jax.process_index()), local_dev_cnt)
        n_hmc = (n_hmc // dev_cnt) * dev_cnt
        lens_sim = sim.LensSimulator(
            self.phys_model,
            self.sim_config,
            bs=n_hmc // dev_cnt,
        )
        momentum_distribution = tfd.MultivariateNormalFullCovariance(
            loc=jnp.zeros_like(q_z.mean()),
            covariance_matrix=jnp.linalg.inv(q_z.covariance()),
        )

        pbar_run = (pbar_interval>0) and (jax.process_index()==0) and (sys.stdout.isatty() or is_ipython())
        pbar=tqdm(total=num_results+num_burnin_steps, position=0, leave=True, miniters=pbar_interval, disable = not pbar_run, desc="HMC Progress")
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
            return self.prob_model.log_prob(lens_sim, z)[0]

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
            seed=0,
            force_use_burnin=False,
    ):
        dev_cnt = len(jax.devices())
        local_dev_cnt = len(jax.local_devices())
        seeds = jax.random.split(jax.random.fold_in(jax.random.PRNGKey(seed), jax.process_index()), local_dev_cnt)
        n_hmc = (n_hmc // dev_cnt) * dev_cnt
        lens_sim = sim.LensSimulator(
            self.phys_model,
            self.sim_config,
            bs=n_hmc // dev_cnt,
        )
        momentum_distribution = tfd.MultivariateNormalFullCovariance(
            loc=jnp.zeros_like(q_z.mean()),
            covariance_matrix=jnp.linalg.inv(q_z.covariance()),
        )

        @jit
        def log_prob(z):
            return self.prob_model.log_prob(lens_sim, z)[0]

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
                    elbo_lens_sim = sim.LensSimulator(
                        self.phys_model,
                        self.sim_config,
                        bs=1,
                    )

                    sharding = NamedSharding(mesh, P('device'))


                    @jit
                    @functools.partial(jax.experimental.shard_map.shard_map, mesh=mesh, in_specs=(None, None, P('device')), out_specs=P())
                    def elbo(loc, cov, jax_seed):
                        qz = tfd.MultivariateNormalFullCovariance(loc=loc, covariance_matrix=cov)
                        z = qz.sample(n_vi // dev_cnt, seed=jax_seed[0])
                        lps = qz.log_prob(z)
                        return jax.lax.pmean(jnp.mean(lps - self.prob_model.log_prob(elbo_lens_sim, z)[0]), axis_name="device")

                    
                    # @functools.partial(jit, static_argnums=(0,))
                    # def elbo(qz):
                    #     z = qz.sample(n_vi, seed=jax.random.PRNGKey(0))
                    #     lps = qz.log_prob(z)
                    #     return jnp.mean(lps - self.prob_model.log_prob(elbo_lens_sim, z)[0])
                    jax_seeds = jax.random.split(jax.random.PRNGKey(0), dev_cnt)
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
        end = time.time()
        print(f"Sampling took {(end - start):.1f}s")

        dev_idxs = jnp.arange(local_dev_cnt)
        all_states = run_chain(seeds, dev_idxs)
        all_samples = jax.experimental.multihost_utils.process_allgather(all_states)
        # device_partitioned_samples is (num_devices, num_steps, n_hmc_per_device, num_params)
        device_partitioned_samples = all_samples.reshape(all_samples.shape[0] * all_samples.shape[1], *all_samples.shape[2:])
        # Final device_partitioned_samples is (num_steps, num_devices, n_hmc_per_device, num_params)
        device_partitioned_samples = jnp.swapaxes(device_partitioned_samples, 0, 1)
        return device_partitioned_samples#, mle_cov    # mle_cov for debugging only
