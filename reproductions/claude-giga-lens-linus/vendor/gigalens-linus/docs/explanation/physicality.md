# Physicality

Some parameter combinations are numerically invalid (a profile's kernel divides by
zero); others are merely astrophysically implausible (a near-zero axis ratio, a
huge external shear). gigalens separates these into **hard domain constraints** and
**soft plausibility bands**, and checks them where the values are concrete — at
model construction and, optionally, on a finished posterior.

## Hard vs soft

- **Hard domains** are per-parameter intervals *derived from the code* — each cites
  the kernel line that makes the bound necessary (e.g. a mass floor `1e-6`). A
  value outside a hard domain is genuinely invalid.
- **Soft plausibility bands** are *human-curated* and deliberately sparse: EPL
  `gamma ≥ 1.1`, mass axis ratio `q ≥ 0.2`, external-shear magnitude `≤ 0.2`. They
  encode "unlikely", not "impossible", and the default soft tolerance (5%) is a
  curation choice, not a derived quantity.

Multi-parameter regions (ellipticity magnitude, axis ratio, shear magnitude) are
expressed as joint constraints with a `hard` or `soft` severity.

## The raise-vs-report policy

Checking happens at `LensModel` construction (via `validate_planes` →
`apply_report`), because that is where fixed values are concrete — under `jit`
every argument is a tracer, so kernel-level guards are impossible. The policy:

| Situation | Outcome |
|---|---|
| A **fixed value** outside a **hard** domain | **`ValueError`** (aggregated) |
| **Prior mass** past a hard bound | `UserWarning` |
| Anything **soft** (fixed value or prior mass) | `[plausibility]` `UserWarning` — never raises |

So a genuinely invalid constant stops you immediately; everything else warns. The
full report is attached as `model.physicality_report` for model-card persistence.
Two adjacent construction-time guards live in `scene.py`: concrete redshifts must
be positive and ordered, and a concrete cosmology is flagged for unphysical
densities.

## Diagnosing a posterior

Physicality checking of a *fit* never raises. `validate_posterior_samples(model,
params)` runs the same domains and joint constraints against realised draws and
reports the **fraction** of the posterior outside each region:

```python
from gigalens.physicality import validate_posterior_samples
report = validate_posterior_samples(model, constrained_samples)
report.summary()        # inspect; a completed fit is diagnosed, not rejected
```

```{admonition} Why never reject a fit
:class: note
A posterior grazing a soft band is information about the fit (or the data), not a
reason to throw samples away. Physicality reports; you decide. This mirrors the
project's method discipline — surface the diagnostic, don't silently "fix" it.
```
