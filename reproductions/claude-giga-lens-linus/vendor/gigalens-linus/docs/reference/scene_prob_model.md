# `gigalens.jax.scene_prob_model`

Observations and the likelihood. A `Dataset` (imaging: `ImageData`) holds one
observation and declares which light entities it `sees`; `ProbModel` ties one
`LensModel` to one or more datasets and sums their likelihood terms into a
differentiable `log_prob(z)`.

```{eval-rst}
.. currentmodule:: gigalens.jax.scene_prob_model

.. autosummary::
   :nosignatures:

   ProbModel
   Dataset
   ImageData
   LikelihoodTerm
   ImageLikelihoodTerm
```

```{eval-rst}
.. automodule:: gigalens.jax.scene_prob_model
   :members:
   :show-inheritance:
   :member-order: bysource
```
