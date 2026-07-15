# Run MCLMC instead of HMC

Microcanonical Langevin Monte Carlo (MCLMC) is often the fastest sampler for
these posteriors once you have a good starting surrogate from MAP → SVI. It lives
in `gigalens.jax.experimental.mclmc`.

## The call

```python
from gigalens.jax.experimental.mclmc import MCLMC_JIT   # MCLMC is an alias for MCLMC_JIT

samples = MCLMC_JIT(seq, qz,
                    n_hmc=8,                 # number of chains
                    num_burnin_steps=1000,
                    num_results=2000,
                    seed=0,
                    progress_bar=False)
# samples.shape == (n_chains, num_results, num_free_params)  — unconstrained z-space
```

- `seq` is a `ModellingSequence` (MCLMC uses `seq.prob_model.log_prob(z)` internally).
- `qz` is the **initialisation surrogate** — it must provide `.sample((n,), seed=)`,
  `.mean()`, and `.covariance()`. Its mean seeds the chains and its covariance
  seeds the inverse mass matrix. Build it from your SVI result (or a MAP point):

```python
qz = tfd.MultivariateNormalFullCovariance(
    loc=np.asarray(jax.device_get(qz_svi.mean())),
    covariance_matrix=np.asarray(jax.device_get(qz_svi.covariance())))
```

- `n_hmc` (chains) must be ≥ the device count and is floored to a multiple of it.

## Useful knobs

| Argument | Default | Purpose |
|---|---|---|
| `desired_energy_variance` | `5e-4` | Energy-variance setpoint for step-size tuning. |
| `init_L`, `init_step_size` | `sqrt(dim)`, `0.25·sqrt(dim)` | Trajectory length / step size seeds. |
| `frac_tune1/2/3` | `0.2 / 0.6 / 0.2` | Fractions of burn-in spent in each tuning stage. |
| `debug_output` | `False` | Return the full history object instead of just samples. |

With `debug_output=True` the return is a history object; take
`np.asarray(hist.position)[:, -num_results:, :]` for the kept draws.

```{admonition} Recommended defaults
:class: tip
For these posteriors, 8 chains with 2000 burn-in and 2000 results, initialised
from a well-converged MAP → SVI surrogate, is a solid starting point. Map the
`z`-space samples to physical values with `model.bijector.forward(...)`.
```

```{seealso}
{doc}`../tutorials/point-source` runs MCLMC end to end. The built-in `seq.HMC`
(see {doc}`../tutorials/first-fit`) is the simpler default if you don't need MCLMC.
```
