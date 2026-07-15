# Share parameters across components

By default every prior in every `Component` is an **independent** free parameter.
Sharing is always explicit, and the mechanism you pick depends on *what* you want
to share. The guiding rule is **identity = sharing**: reusing the same Python
object links; building a fresh one keeps things independent.

```{admonition} Reusing a bare distribution is an error
:class: warning
Passing the *same* bare `tfd.Distribution` instance at two sites raises — it is an
ambiguous foot-gun (link? or template?). Wrap it in `shared()` to link, or build a
fresh distribution to keep the two sites independent.
```

## One shared value at several sites — `shared()`

Use {py:func}`~gigalens.jax.scene.shared` when two or more parameters should be
the **same sampled value** (e.g. a common centre for lens mass and lens light).
Create one handle and reuse *that instance*:

```python
from gigalens.jax.scene import shared

cx = shared(tfd.Normal(0, 0.05))   # one handle
cy = shared(tfd.Normal(0, 0.05))
mass  = Component(EPL(50), dict(..., center_x=cx, center_y=cy))
light = Component(SersicEllipse(use_lstsq=False), dict(..., center_x=cx, center_y=cy))
# mass and light now share a single (center_x, center_y) — one draw, used in both.
```

## A joint prior over sites in different components — `coupled()`

Use {py:func}`~gigalens.jax.scene.coupled` when several scalar sites are drawn
**jointly** from one multivariate prior, even across different components or
planes. It returns a group; place its per-name slots into the prior dicts:

```python
from gigalens.jax.scene import coupled

grp = coupled(some_joint_dist, names=["a", "b"])
comp1 = Component(profile1, dict(..., x=grp["a"]))
comp2 = Component(profile2, dict(..., y=grp["b"]))
```

## Grouped priors within one component — tuple keys

For parameters that share a single (possibly multivariate) prior *within* one
component — the canonical case being ellipticity `(e1, e2)` — use a **tuple key**
in the prior dict:

```python
Component(SersicEllipse(use_lstsq=False), {
    ("e1", "e2"): some_ellipticity_dist,   # drawn together
    "R_sersic": tfd.LogNormal(jnp.log(1.0), 0.15),
    ...
})
```

## Position anchoring — `soft_link()`

{py:func}`~gigalens.jax.scene.soft_link` is an ergonomic helper for anchoring
positions with a soft separation prior (e.g. subhalo positions near a host). It
returns `n` dicts of slots you splice into each component's priors.

```python
from gigalens.jax.scene import soft_link
# returns n prior-dict fragments coupling the positions to an anchor ± separation
```

```{seealso}
Exact signatures and arguments: {doc}`../reference/scene` — `shared`,
`SharedParam`, `coupled`, `CoupledGroup`, `CoupledSlot`, `soft_link`. The concepts
behind "identity = sharing" are in {doc}`../explanation/scene-design`.
```
