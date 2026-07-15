# `gigalens.jax.scene_simulator`

Trace and render. `SceneSimulator` ray-traces the model to each plane and renders
the image (`simulate`), or renders a basis and solves linear light amplitudes
(`lstsq_simulate`). It selects deflection-ratio vs multi-plane tracing from the
model's geometry.

```{eval-rst}
.. currentmodule:: gigalens.jax.scene_simulator

.. autosummary::
   :nosignatures:

   SceneSimulator
```

```{eval-rst}
.. automodule:: gigalens.jax.scene_simulator
   :members:
   :show-inheritance:
   :member-order: bysource
```
