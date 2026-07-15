# How-to guides

Task-oriented recipes. Each answers a single "how do I…?" against the current
scene API. They assume you have worked through {doc}`../tutorials/first-fit`.

```{toctree}
:maxdepth: 1

share-parameters
fix-parameters
run-mclmc
multiplane
adaptive-supersampling
```

- **{doc}`share-parameters`** — link parameters with `shared()`, `coupled()`, tuple-key priors, `soft_link()`.
- **{doc}`fix-parameters`** — inline constants and `LensModel.fix_to`.
- **{doc}`run-mclmc`** — sample with MCLMC instead of HMC.
- **{doc}`multiplane`** — multi-lens-plane / line-of-sight geometry.
- **{doc}`adaptive-supersampling`** — SNR-driven per-pixel render refinement.

For multi-band / multi-dataset and cosmology fits, see the tutorials
{doc}`../tutorials/multiband` and {doc}`../tutorials/cosmology`.
