import functools
import sys

import jax
from jax.sharding import NamedSharding, PartitionSpec as P
import jax.numpy as jnp
import optax
import tensorflow_probability.substrates.jax as tfp
from tensorflow_probability.substrates.jax import (
    distributions as tfd,
    bijectors as tfb,
)
from tqdm.auto import tqdm

import warnings

import gigalens.jax.simulator as sim

if (jax.process_count() > 1) and (not jax.distributed.is_initialized()):
    warnings.warn(
        'jax.distributed.initialize() was not called. '
        'For multinode, please call it before running any JAX functions.'
    )
mesh = jax.make_mesh((len(jax.devices()),), ('device',))

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _is_ipython():
    try:
        from IPython import get_ipython
        return get_ipython() is not None
    except (ImportError, NameError):
        return False

# JOON: helper functions with NaN safety, might help

def _stick_breaking_weights(eta):
    ETA_MAX = 20.0
    eta = jnp.clip(eta, -ETA_MAX, ETA_MAX)
    v = jax.nn.sigmoid(eta)
 
    # Numerically stable log(v) and log(1-v) via softplus identities
    log_v           = -jax.nn.softplus(-eta)   # log(sigmoid(eta))
    log_one_minus_v = -jax.nn.softplus(eta)    # log(1 - sigmoid(eta))
 
    cum    = jnp.concatenate([jnp.zeros(1), jnp.cumsum(log_one_minus_v)])
    log_pi = jnp.concatenate([log_v + cum[:-1], cum[-1:]])
 
    # Log-softmax normalisation: safe even when all log_pi → -inf
    log_pi = log_pi - jax.nn.logsumexp(log_pi)
    pi = jnp.exp(log_pi)
    return pi, v


def _dp_log_prior(v, alpha):
    """
    Log-prior for stick variables under the DP: sum_k log Beta(v_k; 1, alpha).
 
    log p(v_k) ∝ (alpha - 1) * log(1 - v_k)
    """
    eta = jax.scipy.special.logit(jnp.clip(v, 1e-7, 1 - 1e-7))
    log_one_minus_v = -jax.nn.softplus(eta)
    raw = (alpha - 1.0) * jnp.sum(log_one_minus_v)
    return jnp.clip(raw, -1e6, 1e6)

def GMM(
    self,
    start,
    optimizer: optax.GradientTransformation,
    n_vi: int = 250,
    init_scales=1e-3,
    num_steps: int = 500,
    seed: int = 0,
    pbar_interval: int = 100,
    # ── DP-specific ──────────────────────────────────────────────────────────
    T: int = 20,
    alpha: float = 1.0,
    weight_threshold: float = 0.02,
    # ── same as SVI_GMM ──────────────────────────────────────────────────────
    spread_scales=1e-2,
    repulsion_strength: float = 0.1,
    cov_bij=tfp.bijectors.FillScaleTriL(
        diag_bijector=tfb.Softplus(), diag_shift=1e-6
    ),
    output_type="best"
):
    """
    Multi-device Dirichlet Process GMM variational inference.

    Parameters
    ----------
    start            : jnp.ndarray, shape (n_starts, n_params)
    optimizer        : optax.GradientTransformation
    n_vi             : int   — total VI samples per step (divided across devices)
    init_scales      : float or matrix
    num_steps        : int
    seed             : int
    pbar_interval    : int   — display update frequency (0 = off)
    T                : int   — truncation ceiling
    alpha            : float — DP concentration
    weight_threshold : float — pi_k > threshold → "active"
    spread_scales    : float
    repulsion_strength : float
    cov_bij          : tfb.Bijector

    Returns
    -------
    qz       : tfd.MixtureSameFamily
    loss_hist : list[float]
    """
    dev_cnt  = len(jax.devices())
    jax_seed = jax.random.PRNGKey(seed)

    n_vi         = (n_vi // (dev_cnt*T)) * (dev_cnt*T) #just a safeguard in case anyone uses raw n_vi
    if n_vi/dev_cnt//T < 1: raise ValueError("too few samples assigned to cover all components on each device, consider increasing n_vi or decreasing T")

    # lens_sim runs inside shard_map → sees only the per-device batch
    lens_sim = sim.LensSimulator(
        self.phys_model,
        self.sim_config,
        bs=n_vi//dev_cnt,
    )

    # ── Parameter layout ─────────────────────────────────────────────────────
    # qz_params = [ means (T*n_params) | stick_eta (T-1) | covariances (T*tri) ]
    n_params = jnp.size(start[0]) if jnp.ndim(start)>1 else len(start)
    n_starts = len(start) if jnp.ndim(start)>1 else 1
    if n_starts > T:
        start = start[:T]

    key, subkey = jax.random.split(jax_seed)
    perturbations = (
        jnp.array(
            self.prob_model.bij.inverse(
                self.prob_model.prior.sample((T - n_starts), subkey)
            )
        ).T
        - start[0]
    ) * spread_scales
    
    map_start = jnp.concatenate([
        start.flatten(),
        (jnp.squeeze(start[0]) + perturbations).flatten(),
    ])

    stick_init = jnp.array([
        jnp.log(1.0 / (T - k)) - jnp.log(1.0 - 1.0 / (T - k))
        for k in range(T - 1)
    ])
    
    scale_template = (
        cov_bij.inverse(jnp.diag(jnp.ones(n_params)) * init_scales)
        if jnp.size(init_scales) == 1
        else cov_bij.inverse(init_scales)
    )
    scale_init = jnp.ravel(jnp.tile(scale_template, (T,)))

    qz_params         = jnp.concatenate([map_start, stick_init, scale_init])
    replicated_params = jax.device_get(jnp.array(qz_params))

    idx_means  = int(T * n_params)
    idx_sticks = idx_means + (T - 1)

    opt_state  = optimizer.init(replicated_params)

    def _parse(params):
        return params[:idx_means].reshape(T, n_params), params[idx_means:idx_sticks], jnp.reshape(params[idx_sticks:], (T,-1))
        

    def build_qz(params):
        means, eta, scales = _parse(params)
        pi,_v = _stick_breaking_weights(eta)
        cov    = cov_bij.forward(scales)
        return tfd.MixtureSameFamily(
            tfd.Categorical(probs=pi),
            tfd.MultivariateNormalTriL(loc=means, scale_tril=cov),
        )
    def neg_elbo(params, jax_seed, step):
        means, eta, scales = _parse(params)
        pi,v = _stick_breaking_weights(eta)
        cov = cov_bij.forward(scales)

        temp    = jnp.clip(2.0 * (1.0 - step / num_steps), 1.0, 2.0)
        pi_temp = pi ** (1.0 / temp)
        pi_temp = (pi_temp / jnp.sum(pi_temp))
        
        components = tfd.MultivariateNormalTriL(loc=means, scale_tril=cov)
        # mixture = tfd.MixtureSameFamily(
        #     mixture_distribution=tfd.Categorical(probs=pi_temp),
        #     components_distribution=components)
        z = components.sample(n_vi//(dev_cnt*T), seed=jax_seed[0])
        
        log_p       = self.prob_model.log_prob(lens_sim, jnp.swapaxes(z,0,1).reshape((-1,n_params)))[0].reshape((T,n_vi//(dev_cnt*T)))
        log_q       = jnp.clip(components.log_prob(z),-1e12,1e12).T
        
        loss = jnp.sum(pi_temp*jnp.mean(log_q-log_p, axis=1))
        dp_prior = _dp_log_prior(v, alpha)

        return loss - dp_prior

    
    neg_elbo_and_grad = jax.value_and_grad(neg_elbo)

    def dirichlet_process(params, jax_seed): # currently separate, will merge things later to try to optimize the parameter changing
        means, eta, scales = _parse(params)
        pi,v = _stick_breaking_weights(eta)
        cov = cov_bij.forward(scales)
        
        components = tfd.MultivariateNormalTriL(loc=means, scale_tril=cov)
        z = components.sample(n_vi//(dev_cnt*T), seed=jax_seed[0])

        log_p       = self.prob_model.log_prob(lens_sim, z.reshape((-1,n_params)))[0].reshape((n_vi//(dev_cnt*T),T))
        log_q       = jnp.clip(components.log_prob(z),-1e12,1e12)

        local_worst_z = z.reshape(-1,n_params)[jnp.argmax(log_p-log_q)]
        local_worst_estimate = jnp.max(log_p-log_q)
        global_worst_estimate = jax.lax.pmax(local_worst_estimate, axis_name="device")
        global_worst_z = jax.lax.psum(jax.lax.select(local_worst_estimate==global_worst_estimate, 1,0)*local_worst_z, axis_name="device")
        
        sm_idx = jnp.argmin(pi)
        change_mask = (jnp.arange(T) == sm_idx+1)[:,jnp.newaxis]# for array shape compatibility
        cm2 = jnp.arange(T-1) == sm_idx
        lg_idx = jnp.argmax(pi)

        new_means_candidate = jnp.where(change_mask, global_worst_z, means)
        new_eta_candidate = jnp.where(cm2, jnp.ones(T-1)*jnp.max(eta)/2, eta)
        new_scales_candidate = jnp.where(change_mask, scales[lg_idx], scales)

        new_means = jax.lax.select(pi[sm_idx]<weight_threshold, new_means_candidate, means)
        new_eta = jax.lax.select(pi[sm_idx]<weight_threshold, new_eta_candidate, eta)
        new_scales = jax.lax.select(pi[sm_idx]<weight_threshold, new_scales_candidate, scales)

        return jnp.concat([new_means.flatten(),new_eta,new_scales.flatten()])

    @jax.jit
    @functools.partial(
        jax.shard_map,
        mesh=mesh,
        in_specs=(P(), P('device'), P()),
        out_specs=(P(),P(),P()),
    )
    def get_update(qz_params, jax_seed, step):

        new_params = dirichlet_process(qz_params, jax_seed)
        
        loss, grad = neg_elbo_and_grad(new_params, jax_seed, step)
        return (new_params, jax.lax.pmean(loss,axis_name='device'), jax.lax.pmean(grad,axis_name='device'),)

    pbar_run = ((pbar_interval > 0) and (sys.stdout.isatty() or _is_ipython()))
    pbar = tqdm(total=num_steps,position=0,leave=True,miniters=pbar_interval,disable=not pbar_run, mininterval=0.1)
    def pbar_display(args):
        pbar.n = args[0].item()
        pbar.set_description(f"-ELBO: {float(args[1].item()):.3f}")

    def one_step(carry, _):
        params, opt_state, best_params, key, best_loss, n_iter = carry
        key, curr = jax.random.split(key)
        keys = jax.device_put(jax.random.split(curr, dev_cnt), NamedSharding(mesh, P("device")))


        params, loss, grad = get_update(params, keys, jnp.int32(n_iter))

        better = loss < best_loss
        best_params = jax.lax.select(better, params, best_params)
        best_loss   = jax.lax.select(better, loss, best_loss)

        updates, opt_state = optimizer.update(grad, opt_state)
        params = optax.apply_updates(params, updates)

        jax.lax.cond(
            pbar_run & (n_iter % pbar_interval == 0), 
            lambda: jax.debug.callback(pbar_display, (n_iter, loss)), 
            lambda: None
        )

        return (params, opt_state, best_params, key, best_loss, n_iter + 1), loss

    @jax.jit
    def run_svi(initial_carry):
        return jax.lax.scan(one_step, initial_carry, length=num_steps)

    # ── Execution ────────────────────────────────────────────────────────────
    opt_state = optimizer.init(replicated_params)
    initial_carry = (replicated_params, opt_state, replicated_params, key, jnp.inf, 1)

    # Launch the compiled loop
    ((_, _, best_params, _, best_loss, _), loss_hist) = jax.device_get(run_svi(initial_carry))
    pbar.close()

    # ── Build final mixture and report ────────────────────────────────────────
    qz       = build_qz(best_params)
    final_pi = qz.mixture_distribution.probs
    n_active = int(jnp.sum(final_pi > weight_threshold))

    sorted_pi = jnp.sort(final_pi)[::-1]
    print(f"\n[SVI_GMM_DP_multi] Converged with {n_active} active component(s) "
          f"(threshold = {weight_threshold}, T = {T}, alpha = {alpha})")
    print(f"                   Top weights: "
          + "  ".join(f"{float(p):.4f}" for p in sorted_pi[:min(n_active + 2, T)]))
    
    if output_type=="all":
        return qz, loss_hist,qz_hist,n_active_hist

    else:
        return qz, loss_hist,
