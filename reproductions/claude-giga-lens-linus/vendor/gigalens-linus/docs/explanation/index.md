# Explanation

The concepts behind the scene API — the "why", not the "how". Read these when you
want to understand the design rather than accomplish a specific task.

```{toctree}
:maxdepth: 1

scene-design
priors-bijectors
precision-float64
physicality
```

- **{doc}`scene-design`** — model/data orthogonality and what the scene API replaces.
- **{doc}`priors-bijectors`** — how the joint prior and `ZBijector` are derived, and why inference runs in unconstrained `z`-space.
- **{doc}`precision-float64`** — the float64 pipeline and the `SimulatorConfig` precision knobs.
- **{doc}`physicality`** — hard domains, soft plausibility bands, and posterior diagnostics.
