# `gigalens.jax.scene`

The parameter/structure layer: `Component`s and `Plane`s compose into a
`LensModel`, which derives the prior, the flat-`z` `ZBijector`, and the constants.
Parameter sharing (`shared`, `coupled`, `soft_link`) also lives here.

```{eval-rst}
.. currentmodule:: gigalens.jax.scene

.. autosummary::
   :nosignatures:

   LensModel
   Component
   Plane
   shared
   SharedParam
   coupled
   CoupledGroup
   CoupledSlot
   soft_link
   ZBijector
```

```{eval-rst}
.. automodule:: gigalens.jax.scene
   :members:
   :show-inheritance:
   :member-order: bysource
```
