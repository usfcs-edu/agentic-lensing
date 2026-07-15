# `gigalens.jax.experimental` — samplers & supersampling

Experimental / advanced components: gradient-based samplers beyond the built-in
`HMC`, and adaptive supersampling. These modules move faster than the core scene
API; treat their signatures as less stable.

## MCLMC

Microcanonical Langevin Monte Carlo — often the fastest sampler for these
posteriors once a good MAP/SVI starting point is available.

```{eval-rst}
.. automodule:: gigalens.jax.experimental.mclmc
   :members:
   :member-order: bysource
```

## MAMS

```{eval-rst}
.. automodule:: gigalens.jax.experimental.mams
   :members:
   :member-order: bysource
```

## Adaptive supersampling

SNR-driven per-pixel supersampling: refine the render grid only where the signal
warrants it.

```{eval-rst}
.. automodule:: gigalens.jax.experimental.adaptive_supersample
   :members:
   :member-order: bysource
```
