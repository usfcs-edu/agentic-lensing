# Light profiles

Source and lens-light profiles. Wrap one in a `Component` with priors and place
it in a `Plane`'s `light=[...]`. A profile's amplitude is either **sampled**
(`use_lstsq=False`, adding an amplitude parameter like `Ie` you give a prior) or
**solved by linear least squares** at render time (`use_lstsq=True`).

## Sérsic family

`Sersic`, `SersicEllipse`, `CoreSersic`, `DoubleSersic`.

```{eval-rst}
.. automodule:: gigalens.jax.profiles.light.sersic
   :members:
   :show-inheritance:
```

## Shapelets

Flexible basis for complex source morphology (typically `use_lstsq=True`).

```{eval-rst}
.. automodule:: gigalens.jax.profiles.light.shapelets
   :members:
   :show-inheritance:

.. automodule:: gigalens.jax.profiles.light.sersic_shapelets
   :members:
   :show-inheritance:
```

## Point source

Lensed point sources (quasars, supernovae).

```{eval-rst}
.. automodule:: gigalens.jax.profiles.light.point_source
   :members:
   :show-inheritance:
```

## Combining light profiles

```{eval-rst}
.. automodule:: gigalens.jax.profiles.light.combined_profile
   :members:
   :show-inheritance:
```
