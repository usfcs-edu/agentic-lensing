---
sd_hide_title: true
---

# gigalens

```{div} sd-text-center sd-fs-3 sd-font-weight-bold
gigalens
```

```{div} sd-text-center sd-fs-5 sd-text-muted
Gradient-Informed, GPU-Accelerated strong-lens modelling in JAX
```

**gigalens** models strong gravitational lenses by expressing the entire
forward problem — mass and light profiles, multi-plane geometry, cosmology,
priors, and the imaging likelihood — as a single differentiable JAX program,
then fitting it with gradient-based optimisation (MAP), variational inference
(SVI), and Hamiltonian/Langevin samplers (HMC, MCLMC) that scale across GPUs
and nodes.

This site documents the **scene API** — the redesigned, model/data-orthogonal
interface built around `LensModel`, `SceneSimulator`, and `ProbModel`.

:::{admonition} Status: new-API documentation
:class: note
These docs cover the current scene API. The pre-merge (`PhysicalModel` /
`ForwardProbModel` / `multiband`) interface is superseded — see
{doc}`explanation/scene-design` for the redesign and what it replaces.
:::

## Get started

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} {octicon}`rocket` Tutorials
:link: tutorials/index
:link-type: doc
Learning-oriented walkthroughs. Start here: build and fit your first lens.
:::

:::{grid-item-card} {octicon}`tools` How-to guides
:link: how-to/index
:link-type: doc
Task-oriented recipes: share parameters, fix values, run MCLMC, multi-band.
:::

:::{grid-item-card} {octicon}`book` Explanation
:link: explanation/index
:link-type: doc
The concepts: model/data orthogonality, identity = sharing, priors & bijectors.
:::

:::{grid-item-card} {octicon}`code` API reference
:link: reference/index
:link-type: doc
Every public class and function, generated from the source.
:::

::::

## Quickstart

A complete single-plane fit — build a model, attach one image, run MAP → SVI →
HMC. Every line is taken from the verified `demos/simple_demo.ipynb`; the
walkthrough is in {doc}`tutorials/first-fit`.

```python
import jax, numpy as np, optax
from jax import numpy as jnp
import tensorflow_probability.substrates.jax as tfp; tfd = tfp.distributions

from gigalens.simulator import SimulatorConfig
from gigalens.jax.profiles.mass.epl import EPL
from gigalens.jax.profiles.mass.shear import Shear
from gigalens.jax.profiles.light.sersic import SersicEllipse
from gigalens.jax.scene import Component, Plane, LensModel
from gigalens.jax.scene_prob_model import ImageData, ProbModel
from gigalens.jax.inference import ModellingSequence

# 1. A model = Components (profile + priors) arranged into Planes.
epl = Component(EPL(50), dict(
    theta_E=tfd.LogNormal(jnp.log(1.25), 0.25), gamma=tfd.TruncatedNormal(2, 0.25, 1, 3),
    e1=tfd.Normal(0, 0.1), e2=tfd.Normal(0, 0.1),
    center_x=tfd.Normal(0, 0.05), center_y=tfd.Normal(0, 0.05)))
shear = Component(Shear(), dict(gamma1=tfd.Normal(0, 0.05), gamma2=tfd.Normal(0, 0.05)))
source = Component(SersicEllipse(use_lstsq=False), dict(...))   # source-plane light

model = LensModel([
    Plane(mass=[epl, shear], light=[...]),   # lens plane (deflectors + lens light)
    Plane(deflection_ratio=1.0, light=[source]),  # source plane
])

# 2. Attach one observation; declare what it sees. Wrap in a ProbModel.
sim_config = SimulatorConfig(delta_pix=0.065, num_pix=60, supersample=1, kernel=psf)
ds   = ImageData(observed_img, sim_config, background_rms=0.2, exp_time=100, sees="all")
prob = ProbModel(model, ds, mode="forward")
seq  = ModellingSequence(prob)

# 3. Optimise (MAP) → variational fit (SVI) → sample (HMC).
best_z, best_lp, _ = seq.MAP(optax.adabelief(1e-2, b1=0.95, b2=0.99), seed=0, output_type="best")
qz, _   = seq.SVI(best_z, optax.adabelief(1e-4, b1=0.95, b2=0.99), n_vi=1000, num_steps=1500)
samples = seq.HMC(qz, num_burnin_steps=250, num_results=750)

# 4. Sampler works in a flat "z" space; map back to physical values.
constrained = model.bijector.forward(jnp.asarray(samples).reshape(-1, model.num_free_params))
```


```{toctree}
:hidden:
:caption: Documentation

tutorials/index
how-to/index
explanation/index
reference/index
```
