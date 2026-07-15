# Adaptive supersampling

Uniform supersampling (`SimulatorConfig(supersample=f)`) refines *every* pixel
`f×f`, which is wasteful when only the high-SNR arcs need it. The experimental
`gigalens.jax.experimental.adaptive_supersample` module refines the render grid
**per pixel**, driven by the observed signal-to-noise.

```{admonition} Experimental
:class: warning
This module moves faster than the core scene API; treat its interface as less
stable. It requires `SimulatorConfig(supersample=1)` — the per-pixel factor map
*is* the supersampling spec.
```

## Build a factor map from SNR

```python
from gigalens.jax.experimental.adaptive_supersample import (
    AdaptiveGrid, AdaptiveSceneSimulator, AdaptiveImageData, compare_to_reference)

grid = AdaptiveGrid.from_snr(observed_img, error_map)   # per-pixel factor in {0.25,0.5,1,2,4}
```

`from_snr` thresholds the observed SNR into allowed factors (≥2 supersample, 1 is
native, <1 subsample). You can also pass an explicit map: `AdaptiveGrid(factor_map)`.

## Two ways to use it

**Directly**, as a `SceneSimulator` drop-in (note `sim_config.supersample == 1`):

```python
sim = AdaptiveSceneSimulator(model, sim_config, grid)
img = sim.simulate(params)
```

**End to end**, as an `ImageData` that plugs into the normal `ProbModel` stack —
it derives its own grid from its image/noise if you don't pass one:

```python
ds   = AdaptiveImageData(observed_img, sim_config, background_rms=0.2, exp_time=100, sees='all')
prob = ProbModel(model, ds, mode='forward')
```

## Calibrate before you trust it

`compare_to_reference` renders the same params at a high uniform supersample and
reports the per-pixel difference — the falsifier that the adaptive grid is fine
enough:

```python
rep = compare_to_reference(sim, params, error_map=error_map)
assert rep['max_abs_delta_over_sigma'] < 0.1
```

```{admonition} Method discipline
:class: tip
Adaptive supersampling trades accuracy for speed. Always run
`compare_to_reference` (against `reference_supersample=8`) on a representative
parameter set before using an adaptive grid in a fit — a coarse grid can bias the
likelihood in exactly the high-curvature regions you care about.
```
