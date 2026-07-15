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

from gigalens.jax.inference import ModellingSequence
from jax.sharding import NamedSharding, PartitionSpec
import equinox as eqx
import paramax
from tqdm import tqdm

mesh = jax.make_mesh((len(jax.devices()),), ('device',))
dev_cnt = len(jax.devices())

class ModellingSequenceNF(ModellingSequence):
    def SVI(
            self,
            flow,
            optimizer: optax.GradientTransformation,
            n_vi=100,
            steps=1000,
            seed=0,
            show_progress: bool = True,
            path_estimator: bool = False,
    ):
        seed = jax.random.key(seed)
        key, subkey = jax.random.split(seed)
        
        sharding = NamedSharding(mesh, PartitionSpec('device'))

        n_vi = (n_vi // dev_cnt) * dev_cnt
        lens_sim = sim.LensSimulator(
            self.phys_model,
            self.sim_config,
            bs=1#n_vi // dev_cnt
        )
        
        params, static = eqx.partition(
            flow,
            eqx.is_inexact_array,
            is_leaf=lambda leaf: isinstance(leaf, paramax.NonTrainable),
        )
        @eqx.filter_jit
        def normal_elbo(params, static, key):
            dist = eqx.combine(params, static)
            samples, log_probs = dist.sample_and_log_prob(key, (n_vi,))
            target_density = self.prob_model.log_prob(samples, simulator=lens_sim)[0]
            return (log_probs - target_density).mean()
            
        @eqx.filter_jit
        def path_elbo(params, static, key):
            dist = eqx.combine(params, static)
            samples = dist.sample(key, (n_vi,))
            dist = eqx.combine(jax.lax.stop_gradient(params), static)
            log_probs = dist.log_prob(samples)
            target_density = self.prob_model.log_prob(samples, simulator=lens_sim)[0]
            return (log_probs - target_density).mean()
            
        loss_and_grad = eqx.filter_value_and_grad(path_elbo) if path_estimator else eqx.filter_value_and_grad(normal_elbo)
        
        @eqx.filter_jit
        def step(params, static, key, opt, opt_state):
            loss, grads = loss_and_grad(params, static, key)
            updates, opt_state = optimizer.update(grads, opt_state, params=params)
            params = eqx.apply_updates(params, updates)
            return params, opt_state, loss
            
        opt_state = optimizer.init(params)
        losses = []
        min_loss = float('inf')
        best_params = params
    
        keys = tqdm(jax.random.split(key, steps), disable=not show_progress)
        for key in keys:
            params, opt_state, loss = step(params, static, key, optimizer, opt_state)
            if loss < min_loss:
                    best_params = params
                    min_loss = loss
            losses.append(loss.item())
            keys.set_postfix({"loss": loss.item()})
        return eqx.combine(best_params, static), losses

    dev_cnt = len(jax.local_devices())
    def NeutraHMC(self, dist, bij_forward, num_chains=50, num_burnin=250, num_results=750, num_steps_between_results=0, seed=0):
        num_chains = num_chains//dev_cnt
        
        lens_sim = sim.LensSimulator(self.phys_model, self.sim_config, bs=num_chains)
        @jax.jit
        def log_prob(z):
            z, log_det = bij_forward(z)
            return self.prob_model.log_prob(z, simulator=lens_sim)[0] + log_det
            
        seed = jax.random.key(seed)
        num_adaptation_steps = int(num_burnin * 0.8)
        mc_kernel = tfe.mcmc.PreconditionedHamiltonianMonteCarlo(
                    target_log_prob_fn=log_prob,
                    momentum_distribution=dist,
                    step_size=0.1,
                    num_leapfrog_steps=3,
                )
    
        mc_kernel = tfe.mcmc.GradientBasedTrajectoryLengthAdaptation(
                    mc_kernel,
                    num_adaptation_steps=num_adaptation_steps,
                    max_leapfrog_steps=30,
                )
        mc_kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
                    inner_kernel=mc_kernel, num_adaptation_steps=num_adaptation_steps
                )
        @jax.pmap
        def run_chain(seed):
            start = dist.sample(num_chains,seed)
            samples = tfp.mcmc.sample_chain(
                num_results=num_results,
                num_burnin_steps=num_burnin,
                num_steps_between_results=num_steps_between_results,
                current_state=start,
                kernel=mc_kernel,
                trace_fn=None,
                seed=seed)
            return samples
            
        seeds = jax.random.split(seed, dev_cnt)
        return run_chain(seeds)