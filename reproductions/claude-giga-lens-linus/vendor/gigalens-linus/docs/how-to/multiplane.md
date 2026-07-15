# Multi-lens-plane (line-of-sight) fits

For a genuine multi-plane system — two or more **mass** planes at different
redshifts — attach a cosmology and give every plane a redshift. gigalens then
ray-traces recursively through the planes.

## Build it

```python
from gigalens.jax.cosmo import wCDM_Cosmo
from gigalens.jax.scene import Component, Plane, LensModel

cosmo = Component(wCDM_Cosmo(z_lens=z1, z_source_ref=10.0),
                  dict(H0=70.0, Om0=0.3, k=0.0, w0=-1.0))
model = LensModel([
    Plane(redshift=z1, mass=[Component(EPL(50), epl1_priors)]),
    Plane(redshift=z2, mass=[Component(EPL(50), epl2_priors)], light=[lens_light]),
    Plane(redshift=z3, light=[source]),
], cosmo=cosmo)
```

## The rules

- **A cosmology forces redshift geometry.** With `cosmo=` set, every plane must
  use `redshift=` and none may use `deflection_ratio=`. Redshifts must be finite,
  positive, and non-decreasing observer→source.
- **Trace mode is chosen from the number of mass planes:**
  - **≥ 2 mass planes → `multiplane`** (recursive ray-trace; requires a cosmology).
  - **0 or 1 mass planes → `deflection_ratio`** (single-deflector; with a cosmology
    the ratio is derived from redshifts, otherwise from the explicit
    `deflection_ratio` value).

  You can check which was selected: `SceneSimulator(model, cfg).trace_mode`.
- **The cosmology `Component`** wraps a profile exposing `z_lens`,
  `distance_matrix(...)`, and `deflection_ratio(...)` — `wCDM_Cosmo` or
  `w0waCDM_Cosmo`. Its prior dict must cover exactly that profile's parameters
  (`H0`, `Om0`, `k`, `w0`[, `wa`]), each a number, distribution, or `shared` handle.

```{admonition} Single deflector but want cosmology?
:class: note
With one mass plane the model stays in `deflection_ratio` mode. If you add a
cosmology anyway, the mass plane's redshift must equal the cosmology's `z_lens`.
Cosmological *parameter* inference from a single deflector is the time-delay
route — see {doc}`../tutorials/point-source`.
```

Everything downstream (priors, bijector, `ProbModel`, inference) is identical to
the single-plane case; only the geometry changes.
