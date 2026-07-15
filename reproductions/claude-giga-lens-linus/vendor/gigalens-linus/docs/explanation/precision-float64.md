# Precision and the float64 pipeline

Strong-lens likelihoods can be sensitive to floating-point precision: a float32
least-squares basis carries a noise floor that biases high-dynamic-range fits (and
high-`n_max` shapelet sampling in particular). gigalens therefore runs a
configurable-precision forward model, with a canonical **float64** pipeline.

## The canonical setting

```python
# environment (or jax.config.update("jax_enable_x64", True))
JAX_ENABLE_X64=1

SimulatorConfig(..., likelihood_precision="float64")
```

`likelihood_precision="float64"` **requires** `jax_enable_x64` — it is the working
precision of the whole forward pass and reduction. Setting it removes the basis
noise floor at roughly 2× the forward memory and compute of float32.

## The knobs (all on `SimulatorConfig`)

| Knob | Effect |
|---|---|
| `likelihood_precision` | Working precision of the forward + reduction. `None` → float64; `"float32"` is the baseline; `"float64"` is the recommended default (needs x64). |
| `conv_precision` | PSF-convolution-only override, independent of the above. `"float32"` moves the FLOP-heavy convolution off the slow FP64 path with a benign (~1e-7) error; basis/solve/reduction stay float64. |
| `basis_precision` | Light-basis-only override. `"float32"` casts the basis eval (and its VJP) to float32; the solve and χ² are promoted back to float64. **An open question — not on by default.** |
| `remat_basis` | Not precision: rematerialises the basis+conv+pool pipeline to cut peak memory (~40%) at the cost of recompute. Exact. |

Each precision knob accepts only `"float32"`, `"float64"`, a matching dtype, or
`None` — an unrecognised value raises, so there is no silent precision surprise.

## What runs where

Under `likelihood_precision="float64"`, basis generation, PSF
convolution/pooling, the gram/least-squares solve, and the χ² reduction all run
in float64; the pixel-grid WCS coordinates are always float64. The `conv_precision`
and `basis_precision` overrides selectively drop the convolution or the basis eval
to float32 while the float64 noise/observed arrays re-promote the solve and
reduction. `ProbModel` also casts sampler positions to the forward dtype before
evaluating the likelihood, so a float32-seeded MCLMC chain doesn't silently run
the forward in float32.

```{admonition} Method-discipline note
:class: tip
Precision is a *structural* choice, not a knob to tune for a nicer number. If a
result changes between float32 and float64, that is diagnostic — investigate the
conditioning (basis noise floor, near-degenerate light components) rather than
picking whichever precision gives the answer you expected. The `conv_precision` /
`basis_precision` float32 options are **performance** opt-ins (Ampere FP64 is
half-rate), justified by an error budget, not accuracy shortcuts.
```
