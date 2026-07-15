# Priors, bijectors, and the flat-`z` space

Inference in gigalens runs in an **unconstrained** vector space `z`, not in the
physical parameters directly. This page explains how a `LensModel` turns a
collection of per-parameter priors into one joint prior plus a bijector, and why
that indirection is what makes gradient sampling work.

## From per-component priors to one joint prior

When you build a `LensModel`, it walks every plane and component and classifies
each prior-dict entry into one of three kinds:

- a bare `tfd.Distribution` → a **unique free parameter**, keyed by its site path;
- a `shared()` handle → **one** free parameter registered once, so N reuse sites
  collapse to a single sampled value (the value is fanned back out to every site);
- a plain number → a **constant**, recorded in `model.constants`, contributing no
  free parameter.

The unique free parameters become a single joint prior,
`tfd.JointDistributionNamed`, stored as `model.prior`. Independence is the
default; only `shared()`, `coupled()`, and tuple-key (grouped) priors introduce
coupling — and that coupling lives in the *distribution*, not in the profile.

```{admonition} Identity = sharing
:class: note
Reusing the same Python object links; a fresh object stays independent. Reusing a
*bare* `tfd.Distribution` at two sites is a loud error — wrap it in `shared()` to
link deliberately. See {doc}`../how-to/share-parameters`.
```

## What the bijector does

`model.bijector` (a `ZBijector`) maps a flat unconstrained array `z` of shape
`(..., num_free_params)` to the constrained parameter dict. It is built by
chaining two pieces:

1. a **repack** that splits the flat vector and reshapes it into the joint prior's
   named structure, and
2. the joint prior's **default event-space bijector**, which maps each
   unconstrained coordinate to its constrained support (e.g. a `LogNormal`
   parameter's `z` lives on all of ℝ and is exponentiated into ℝ₊).

`ZBijector` also supplies `forward_log_det_jacobian` — the volume correction that
keeps the posterior correct when you change variables.

- **`num_free_params`** is the total number of *unconstrained columns*, which is
  not the same as the number of prior entries: a scalar prior is one column, a
  grouped/multivariate prior contributes its unconstrained event size.
- **`model.z_param_names`** is the authoritative column→name map, in the sampler's
  canonical (tree-flattened) order. Always index samples through it — never
  reconstruct order from insertion.

```{admonition} Flat-`z`, not list-of-columns
:class: important
Pass `z` as a `(..., num_free_params)` array. The older list-of-columns form
(`bijector.forward(list(z.T))`) is retired: it emits a `DeprecationWarning` and
will become an error.
```

## Why inference lives in `z`-space

`z` is an unconstrained ℝ^D vector, so gradient-based samplers (HMC, MCLMC) and
optimisers explore freely without hitting boundary walls (positivity, bounded
ellipticity, simplex constraints). The bijector maps each proposal back to
physically valid parameters and the Jacobian term corrects the density. The prior
log-density is evaluated as:

```text
log_prior(z) = prior.log_prob( bijector.forward(z) ) + bijector.forward_log_det_jacobian(z)
```

`model.constrained(z)` is the convenience wrapper that produces a
simulator-ready params dict from a flat `z` (it is `to_params(bijector.forward(z))`).

## Grouped priors and disk ellipticity

A tuple key such as `("e1", "e2")` maps two (or more) sites to a single joint
prior with matching event shape; its default event-space bijector *is* the
coupling. `DiskEllipticity` (in `gigalens.jax.grouped_priors`) is the
recommended ellipticity prior: it pushes a Gaussian through a smooth map onto the
open disk `|e| < radius`, so the hard axis-ratio cap holds for every `z`, the map
is regular at the centre, and there is **no finite reflecting wall** for the
sampler to fight — a common failure mode for naive bounded ellipticity priors.
`coupled()` and `soft_link()` extend the same mechanism to sites in *different*
components.
