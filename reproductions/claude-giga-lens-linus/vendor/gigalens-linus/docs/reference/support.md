# Configuration & support

Backend-agnostic pieces shared across the scene API: the imaging/grid
configuration, the profile base classes, and physicality checking.

## `SimulatorConfig` — grid, PSF, precision

The grid, pixel scale, PSF kernel, supersampling, and precision knobs for
rendering. Passed to every `ImageData` / `SceneSimulator`.

```{eval-rst}
.. currentmodule:: gigalens.simulator

.. autosummary::
   :nosignatures:

   SimulatorConfig
```

```{eval-rst}
.. automodule:: gigalens.simulator
   :members:
   :show-inheritance:
   :member-order: bysource
```

## Profile base classes

`Parameterized`, `LightProfile`, `MassProfile` — the interfaces every profile
implements (parameter names, evaluation).

```{eval-rst}
.. automodule:: gigalens.profile
   :members:
   :show-inheritance:
```

## Physicality

Domain constraints and soft plausibility checking. Physicality reporting is
non-fatal by design — it flags implausible regions rather than raising.

```{eval-rst}
.. automodule:: gigalens.physicality
   :members:
   :show-inheritance:
```
