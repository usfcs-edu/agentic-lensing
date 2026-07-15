# `gigalens.jax.cosmo`

Cosmology components for multi-plane and cosmological inference. A cosmology is a
`Component` attached to a `LensModel` via `cosmo=`; its parameters (e.g. `H0`,
matter density, dark-energy equation of state) can be sampled like any other.

```{admonition} `z_source_ref` is required
:class: note
`wCDM_Cosmo(z_lens, z_source_ref)` takes a mandatory reference source redshift —
there is no silent default.
```

```{eval-rst}
.. currentmodule:: gigalens.jax.cosmo

.. autosummary::
   :nosignatures:

   wCDM_Cosmo
   w0waCDM_Cosmo
```

```{eval-rst}
.. automodule:: gigalens.jax.cosmo
   :members:
   :show-inheritance:
   :member-order: bysource
```
