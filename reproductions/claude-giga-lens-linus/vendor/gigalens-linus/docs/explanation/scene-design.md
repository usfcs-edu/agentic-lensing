# The scene design: model / data orthogonality

The scene API replaces the pre-merge tangle of `PhysicalModel`,
`ForwardProbModel` / `BackwardProbModel`, `JointModel`, and `multiband` with
**one** source of truth for the prior, bijector, and constants, plus **one**
simulator and **one** probabilistic-model code path that subsumes single-image,
multi-band, and multi-plane fits.

## Three modules, three responsibilities

| Module | Objects | Role |
|---|---|---|
| `gigalens.jax.scene` | `Component`, `Plane`, `LensModel`, `shared`/`coupled`/`soft_link`, `ZBijector` | the parameter/structure layer — derives `prior`, `bijector`, `constants` |
| `gigalens.jax.scene_simulator` | `SceneSimulator` | trace + render: `simulate`, `lstsq_simulate` |
| `gigalens.jax.scene_prob_model` | `Dataset` / `ImageData`, `ProbModel` | observation(s) + the Gaussian likelihood |

## Model and data are orthogonal

The redesign separates two things the old code entangled:

- **Model** (`LensModel`) — the shared physical universe: what entities exist
  (mass profiles, light profiles, source-plane geometry, cosmology) and their
  priors / constants. It is the **sole source of truth** for the prior, the
  bijector, and the constants dict. **It contains no data.**
- **Data** (`Dataset` subclasses; imaging: `ImageData`) — one observation: an
  image + grid/PSF (`sim_config`) + noise + mask, plus a *view* (`sees`)
  declaring which light entities of the shared model this observation renders.
  **It contains no model definition.**

A fit ties **one** `LensModel` to **one or more** datasets via `ProbModel`. One
dataset that `sees="all"` reduces to the old single-image path; N datasets subsume
both the old `multiband` and `JointModel` mechanisms with one code path.

## Identity = sharing

The guiding rule across the whole API:

- reuse a `Component` (the same Python object) across datasets → the *same*
  physical entity (shared morphology);
- reuse a `shared(dist)` handle across sites → the *same* sampled parameter;
- reuse a plain number → the value is simply equal (a fixed constant).

Independence is the default; sharing is always **explicit**. Reusing the same
*bare* `tfd.Distribution` object at two sites is a **loud error**, not a silent
link — wrap it in `shared()` to link, or build a fresh distribution to keep the
two independent. See {doc}`../how-to/share-parameters`.

## What this replaces

```{admonition} Superseded old-API objects
:class: note
`PhysicalModel` → `LensModel`; `ForwardProbModel` / `BackwardProbModel` /
`JointModel` → `ProbModel`; `LensSimulator` → `SceneSimulator`; the `multiband`
path → multiple `ImageData`s under one `ProbModel`. The old classes remain in the
tree only as removal-pending stubs (`design/leftover-code-inventory.md`).
```

```{admonition} Provenance
:class: seealso
This page supersedes the earlier hand-written `docs/scene-api.md`, whose concepts
are preserved here but whose code examples had drifted (removed `bs=` arguments,
the retired `list(z.T)` bijector form, the single-arg `ModellingSequence`, and
`wCDM_Cosmo`'s now-required `z_source_ref`). The design spec of record is
`design/phase1-model-data-api.md`.
```
