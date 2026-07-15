# `gigalens.jax.inference`

The inference driver. `ModellingSequence` wraps a scene `ProbModel` and exposes
the optimisation and sampling stages — `MAP`, `SVI`, `HMC` — that all run through
the model's differentiable `log_prob(z)`.

```{eval-rst}
.. currentmodule:: gigalens.jax.inference

.. autosummary::
   :nosignatures:

   ModellingSequence
```

```{eval-rst}
.. automodule:: gigalens.jax.inference
   :members:
   :show-inheritance:
   :member-order: bysource
```

```{seealso}
Langevin/Hamiltonian samplers beyond `HMC` live in
{doc}`samplers` (`gigalens.jax.experimental`).
```
