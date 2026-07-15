# API reference

Generated from the source docstrings. The **scene API** is the stable, documented
surface; the profiles and support modules round it out.

```{toctree}
:maxdepth: 1
:caption: Core scene API

scene
scene_simulator
scene_prob_model
grouped_priors
```

```{toctree}
:maxdepth: 1
:caption: Inference

inference
samplers
```

```{toctree}
:maxdepth: 1
:caption: Profiles & cosmology

profiles_mass
profiles_light
cosmo
```

```{toctree}
:maxdepth: 1
:caption: Configuration & support

support
```

```{admonition} Not documented here: the superseded old API
:class: warning
`gigalens.jax.physical_model`, `gigalens.jax.prob_model`, `gigalens.jax.simulator`
(`LensSimulator`) and `gigalens.jax.point_source` are pre-merge remnants that the
scene API replaces. They are intentionally excluded — see
`design/leftover-code-inventory.md` for their removal status.
```
