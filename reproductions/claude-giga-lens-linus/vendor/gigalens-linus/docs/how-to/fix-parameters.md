# Fix or freeze parameters

There are two ways to hold parameters constant, depending on whether you want to
fix **one value** or freeze **most of a model**.

## Fix a single parameter — pass a number

In a `Component`'s prior dict, any value may be a distribution (free), a
`shared()`/`coupled()` handle (free + linked), or a **plain number** (a fixed
constant). A number contributes no free parameter:

```python
epl = Component(EPL(50), {
    "theta_E": 1.2,                       # fixed constant
    "gamma":   tfd.TruncatedNormal(2, 0.25, 1, 3),   # free
    "e1": tfd.Normal(0, 0.1), "e2": tfd.Normal(0, 0.1),
    "center_x": 0.0, "center_y": 0.0,     # fixed
})
```

The same rule applies to plane geometry — `Plane(redshift=0.5, ...)` or
`Plane(deflection_ratio=1.0, ...)` fix those as constants.

```{admonition} Grouped (tuple-key) priors can't take a number
:class: note
A tuple key like `("e1", "e2")` must map to a distribution or a `shared` handle.
To fix its members, give each an individual scalar key with a number.
```

## Freeze most of a model — `LensModel.fix_to`

To pin an entire model to a set of truth values and free only a few components
(e.g. hold the mass fixed, fit only the source), use `fix_to`:

```python
LensModel.fix_to(truth_scene, free=())
```

- `truth_scene` is a **structured params dict** in the `planes`/`cosmo` layout —
  the shape produced by `model.to_params(...)` or `model.constrained(z_map)`. It
  must cover every parameter being fixed.
- `free` is a list of `Component` objects (matched by **identity**) whose
  parameters keep their priors; everything else is fixed to its truth value.
- Plane **geometry** (redshift / deflection_ratio) is always fixed to truth,
  regardless of `free`.
- Returns a **new** `LensModel`.

```python
truth = model.constrained(best_z)          # or model.to_params(truth_unique)
# freeze everything except the source-light component `src`:
source_only = model.fix_to(truth, free=[src])
```

```{admonition} Shared parameters
:class: warning
If one `shared()` handle is referenced by both a free and a fixed component,
`fix_to` raises rather than half-fixing it — free or fix the shared value as a
whole.
```
