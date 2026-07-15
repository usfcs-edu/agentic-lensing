# `tests/validation/` — physics validation against lenstronomy

**Status: UNCERTIFIED.** This suite is a *skeleton* built before the API migration
(see `design/phase1-model-data-api.md`, §6a). It has **not** been run to a clean
pass yet; do not read the presence of a test as evidence the physics is correct.
Running it and recording the results — including existing failures of the *current*
code — is Phase 0.

## Purpose

Validate the gigalens **forward simulation** (single-plane *and* multi-lens-plane)
against an independent reference. It deliberately does **not** exercise inference
algorithms (MAP/SVI/HMC/MCLMC): those have no lenstronomy counterpart and are
costly. Inference correctness is a separate concern, diagnosed per
`GIGALens-Code/docs/inference-diagnostics.md`.

## The oracle

**lenstronomy** is the correctness oracle for everything it implements (profile
deflections/light, multiplane ray-shooting). **astropy** is the oracle for
cosmological distances. Where no oracle exists (e.g. our specific PSF-subgrid +
pooling pipeline), we fall back to **metamorphic** relations and **round-trip**
tests (below).

## Design (how real test suites are built, applied here)

1. **Layered to isolate the failure (a test pyramid).** A failure should point at
   *one* stage, not "the image is wrong." Order, tightest/most-local first:
   - `test_cosmology.py` — distances vs astropy.
   - `test_profiles*` — per-profile deflection/light vs lenstronomy (the existing
     `tests/test_profiles.py` already covers several in float32; this suite adds
     float64 and the profiles it misses).
   - `test_coordinates.py` — pixel grid orientation/units, `conversion_factor`.
   - `test_multiplane.py` — the recursive trace vs lenstronomy `ray_shooting`.
   - `test_single_plane.py` — pre-convolution image (tight) then full image (loose),
     and the lstsq amplitude round-trip.
   We prefer asserting an *early* layer tightly over leaning on a loose end-to-end
   number (method-discipline §5: the disagreement must not hide in a metric's blind
   spot, §4).

2. **Derived tolerances only.** `tolerances.py` holds every numeric threshold with a
   written derivation (float eps + accumulation, trapezoid order, subgrid-kernel
   error). *A threshold you cannot derive is not a test* (method-discipline §3). We
   do **not** inherit the old `rtol=1e-5/atol=1e-4` float32 band as a default — the
   framework is float64-first, so analytic parity is held near machine precision and
   the loose bands are used only for explicit float32-mode tests.

3. **Metamorphic + randomized differential tests for the unforeseen.**
   `test_metamorphic.py` asserts invariants that hold with no oracle and catch bugs
   we did not anticipate: translation/rotation equivariance, amplitude linearity,
   zero-mass-plane no-op, plane/component reordering invariance, the
   single-deflector ⇄ multiplane reduction (N=1), and `deflection_ratio` ≡ multiplane
   for one lens + multiple source redshifts. `conftest.py` provides seeded random
   *valid* parameter draws so these run as a differential sweep, not a single point.

4. **Risk traceability.** `risk_register.md` maps every physics risk flagged in the
   design to the test(s) that guard it. "Guards against all flagged errors" is only
   meaningful if it is auditable — that file is the audit.

5. **Pre-registration in each test.** Each test's docstring states, briefly, the
   claim type (deterministic identity / limit / distributional — method-discipline
   §2), the falsifier, and the derived tolerance it uses. Movement toward agreement
   is not a pass; the derived threshold is.

## Conventions

- Markers (registered in `conftest.py`): `physics`, `multiplane`, `cosmology`,
  `metamorphic`, `slow`, `pending` (new-API tests not yet implementable — skipped
  with a Phase reference, ready to flip on).
- New-API tests import the new modules **inside** the test body / behind skips, so
  collection works before those modules exist.
- Seeds are fixed and recorded (project-standards §5). float64 is enabled in
  `conftest.py` *before* `jax.numpy` is imported (project-standards §8).

## Comparison plots (method-discipline §5)

The numeric `allclose` is the automated gate; **plots are diagnostic artifacts**, not
the pass/fail decision. They exist because an aggregate metric is blind to *structured*
residuals that stay under tolerance (§4) — you have to look. A test registers what it
compared via the `plotter` fixture:

```python
def test_xxx(plotter):
    plotter.image("final_image", ref, model, rtol=..., atol=..., err=...)  # 2-D
    plotter.curve("D_C_vs_z", z, ref, model, xlabel="z", ylabel="D_C")     # 1-D
```

Image plots render 5 panels — reference, gigalens, difference, normalized residual
(`/σ` if an error map is given, else relative), and a residual histogram — titled with
PASS/FAIL, max |Δ|, and max relative Δ. Rendering happens **only** when `--plots` is
passed (so passing tests can be eyeballed) **or the test failed** (auto, for
debugging); otherwise nothing is rendered and matplotlib is never imported. A passing
aggregate with a structured residual map is an **open finding**, not a pass.

## Running

```bash
cd ~/gigalens
JAX_ENABLE_X64=1 pytest tests/validation -q                 # all (skips pending)
JAX_ENABLE_X64=1 pytest tests/validation -m "cosmology"     # one layer
JAX_ENABLE_X64=1 pytest tests/validation -m "not slow"      # fast subset
JAX_ENABLE_X64=1 pytest tests/validation --plots            # also write comparison plots
JAX_ENABLE_X64=1 pytest tests/validation --plots --plot-dir /tmp/val_plots
```

Plots default to `tests/validation/_artifacts/` (gitignored).

Record outcomes (including failures of current code) in the relevant
`GIGALens-Code` lab-notebook log; a failing oracle comparison is a finding to file,
not a number to tune away.
