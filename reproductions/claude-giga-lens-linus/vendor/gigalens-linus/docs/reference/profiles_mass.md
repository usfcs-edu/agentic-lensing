# Mass profiles

Deflector mass profiles. Each is a `Parameterized` profile you wrap in a
`Component` with priors and place in a `Plane`'s `mass=[...]`. Parameter names
(e.g. `theta_E`, `gamma`, `e1`, `e2`, `center_x`, `center_y`) are the keys your
prior dict must cover.

## Elliptical power law (EPL)

```{eval-rst}
.. automodule:: gigalens.jax.profiles.mass.epl
   :members:
   :show-inheritance:
```

## Singular isothermal ellipsoid / sphere (SIE, SIS)

```{eval-rst}
.. automodule:: gigalens.jax.profiles.mass.sie
   :members:
   :show-inheritance:

.. automodule:: gigalens.jax.profiles.mass.sis
   :members:
   :show-inheritance:
```

## External shear

```{eval-rst}
.. automodule:: gigalens.jax.profiles.mass.shear
   :members:
   :show-inheritance:
```

## NFW family

```{eval-rst}
.. automodule:: gigalens.jax.profiles.mass.nfw
   :members:
   :show-inheritance:

.. automodule:: gigalens.jax.profiles.mass.nfw_ellipse_slope
   :members:
   :show-inheritance:

.. automodule:: gigalens.jax.profiles.mass.tnfw
   :members:
   :show-inheritance:

.. automodule:: gigalens.jax.profiles.mass.tnfw_ellipse
   :members:
   :show-inheritance:
```

```{admonition} Also available
:class: note
Additional deflectors — dPIE/PIEMD, subhalo and scaling-relation profiles, and
their series-expansion variants (`dpie_series`, `dpie_subhalo`, `scaling_relation`,
`scaling_series`) — live under `gigalens.jax.profiles.mass`. They follow the same
`Component(profile, priors)` pattern.
```
