# Misuse register — red-team (adversarial) traceability

**Status: PROPOSED / UNCERTIFIED.** Drafted by a proposing agent (Mode B); the human
grades. Nothing here is a certified pass. This is the design artifact for a *red-team*
layer that is distinct in contract from `risk_register.md`:

- `risk_register.md` asks: **when used correctly, does gigalens compute the right physics?**
  (oracle = lenstronomy / astropy; assertion = `allclose`).
- This register asks: **when used incorrectly or incompletely by a non-expert, does
  gigalens fail loudly instead of silently returning plausible-but-wrong physics?**
  (oracle = the spec / physics contract; assertion = `pytest.raises` *or* "the degraded
  condition is surfaced", never a silent number).

Each row is one naive-misuse path. "We guard naive misuse" is only an auditable claim if
this table maps every guarded misuse to its test — that is the audit.

## The organizing distinction: *absent* vs. *poisoned* input

The existing no-silent-defaults work guards **absent** scientific inputs (missing PSF /
noise / cosmo / param / `sees`) — and those guards are already implemented (see Status).
The failures the user actually hit are mostly **present-but-poisoned** inputs (a NaN
error map; a redshift that yields no deflection; a reference redshift silently defaulted
to a wrong value). Poisoned-content guards are largely **missing**. The red-team layer's
main job is to close that second class.

Status legend: **guard-exists** = validator already in code, test dormant (un-skip & pin);
**gap** = no guard yet, proposed test will (correctly) FAIL against current code until a
guard is added — that failure is the finding, not something to tune away;
**invariant** = no oracle, asserts a physics/identity property.

---

## A. Derived from real encountered pitfalls (`scratch_mds/encountered_pitfalls.md`)

| # | Naive misuse | Forbidden silent behavior | Guard status (code) | Proposed test | Status |
|---|---|---|---|---|---|
| M1 | Construct a cosmology without passing the reference source redshift | `z_source_ref` silently defaulted to 10 → all `theta_E` normalized to the wrong plane; model is not what the user thinks, no signal | Now required positional in `CosmoBase.__init__` (`src/gigalens/cosmo.py:9-17`); omission → `TypeError` | `test_redteam_cosmology.py::test_cosmo_requires_reference_redshift` — assert constructing `wCDM_Cosmo(z_lens=...)` without `z_source_ref` raises; **also** sweep every construction path (factory/config/YAML loader) for a surviving default | guard-exists + **investigate other paths** |
| M2 | Specify planes by `redshift` (+ cosmology) but no explicit `deflection_ratio` | A lensed-light plane returned the *undeflected* grid unless its geometry dict literally contained `"deflection_ratio"`; especially toxic under `lstsq_simulate`, whose auto-solved intensities hide it by refitting | Ratios now computed from redshift; `§3.1` forces `redshift` when cosmo present (`src/gigalens/jax/scene.py:164-187`); single source defaults `deflection_ratio=1.0` (`:184`) | `test_redteam_multiplane.py::test_redshift_planes_are_actually_deflected_{simulate,lstsq}` — cosmo + two source redshifts (ref plane ratio 1, z=3 plane ratio 1.64): redshift render ≡ explicit-ratio render (float64 floor), and differs loudly (>1e-3·peak, derived) from a both-ratios-1 render, under **both** `simulate` and `lstsq_simulate`. Falsifier: undeflected grid / ratio collapses to 1 / lstsq refit absorbing the geometry error | **live (passing)** |
| M3 | Pass an `error_map` containing NaN/inf (or zero/negative) into an `ImageData` | `lstsq_simulate` returned an all-NaN image with no prior warning | **IMPLEMENTED** — `ImageData._validate_finite_noise` raises on non-finite image / non-finite or ≤0 sigma over the **unmasked** region; the derived-noise path now also validates `background_rms`/`exp_time` at the source (`scene_prob_model.py`) | `test_redteam_dataset.py` (7 tests, **passing**): NaN/inf/≤0 error_map raise; NaN image raises; bad exp_time raises; **negative image pixels & masked-out NaN allowed** (scope pins) | **live (passing)** |
| M4 | Leave a cosmological parameter (e.g. `H0`) free to be sampled with a prior | Distance integral was not vectorized → raised on batched cosmo leaves (here it *raised*, but the fix risks a silent mis-batch: the param batch colliding with the 1000-pt z grid — see the warning comment at `src/gigalens/cosmo.py:27-40`) | Integral vectorized (`comoving_distance_z1z2`, `distance_matrix` give the batch its own trailing axes) | `test_redteam_cosmology.py::test_batched_distances_equal_looped` — differential: batched cosmo params vs a Python loop over the same scalars must match to machine precision, for `comoving_distance`, `deflection_ratio`, and `distance_matrix`. Guards against a **silently wrong batched distance during HMC of cosmology** (→ biased posterior, invisible) | invariant (differential) |

## B. Cosmology adversarial — "incorrect functions that still return plausible numbers"

The astropy-parity tests (`test_cosmology.py`) check `D_C`, `D_A`, `D_M`, and the distance
matrix *individually* at valid redshifts. They do **not** check (a) invalid geometry, (b)
the *assembly* of those distances into `deflection_ratio`, or (c) unphysical parameters.
A `lensing_distance` with `D_s`/`D_ls` swapped can match astropy on every individual
distance and still produce the wrong ratio — the user's stated main worry.

| # | Naive misuse / latent bug | Forbidden silent behavior | Guard status (code) | Proposed test | Status |
|---|---|---|---|---|---|
| C1 | Source at/in front of the lens: `z_source ≤ z_lens`, or `z_source_ref ≤ z_lens`, or any `z ≤ 0` | `angular_distance_z1z2(z_lens, z_source)` with `z2 < z1` returns a negative / nonsensical distance; `deflection_ratio` goes negative — physically a source in front of the lens is not lensed | **IMPLEMENTED (two layers)** — eager: `CosmoBase.__init__` raises on `z_lens ≤ 0` / `z_source_ref ≤ z_lens`; `_validate_source_redshift` raises on `z_source < z_lens` / `z ≤ 0` / non-finite at first use (`z_source == z_lens` stays allowed: ratio exactly 0, pinned by C2). Scene path (jit makes even fixed redshifts tracers, grader probe P2): `LensModel._validate_concrete_redshifts` raises at construction on concrete `z ≤ 0`/non-finite/ordering violations. Residual: sampled redshifts (prior's job) | `test_redteam_cosmology.py::test_invalid_redshift_ordering_raises` (8 eager cases) + `test_redteam_multiplane.py::test_front_of_lens_plane_raises_at_construction`, `::test_ordered_foreground_light_plane_is_legitimate_and_undeflected` | **live (passing)** |
| C2 | (latent) `lensing_distance` / `deflection_ratio` assembled wrong while individual distances stay correct | Plausible ratios that pass astropy parity but mis-scale every mass | **invariant** — assert ratio-assembly identities directly | `test_redteam_cosmology.py::test_deflection_ratio_physicality` — `deflection_ratio(z_source_ref)==1` exactly; `deflection_ratio(z_lens)==0`; strictly increasing in `z_source` on `(z_lens, ∞)`; `∈(0,1)` for `z_lens<z_source<z_ref`, `>1` beyond `z_ref`. Catches swapped `D_s`/`D_ls` that parity misses | invariant |
| C3 | Unphysical cosmological parameters, e.g. `Om0 > 1` (⇒ `Ode0 < 0`) | `efunc` takes `sqrt` of a negative → `NaN` propagates silently into every distance and into the likelihood | **IMPLEMENTED (two layers)** — eager: `CosmoBase._flag_unphysical_densities` at the distance-integral entry point warns (`UserWarning`, per resolved FLAG policy) on `Om0 < 0` or `Ode0 = 1−Om0−Or0−Ok0 < 0`; distances still compute. Scene path (flag is dead under jit, grader probe P3): `LensModel.__init__` flags concrete cosmo constants at construction. Residuals: sampled params (prior's job); model-card persistence field. Caveat: EdS `Om0=1.0` warns (Ode0 = −Or0) | `test_redteam_cosmology.py::test_unphysical_density_is_flagged_not_silent` + `test_redteam_multiplane.py::test_unphysical_cosmo_constants_flagged_at_model_construction` (both with must-NOT-warn controls) | **live (passing)** |
| C4 | (structural) multi-plane `distance_matrix` mis-built (layout/units/curvature) for a config with no astropy run | Wrong tracing distances; subtle multi-plane bias | partial — `test_distance_matrix_vs_astropy_flat` covers flat parity | `test_redteam_cosmology.py::test_distance_matrix_structure` — strictly lower-triangular; zero on/above diagonal; all sub-diagonal entries positive and increasing down each column. Structural invariant for non-astropy configs | **live (passing)** |

## C. First-batch target — un-skip & pin the existing no-silent-defaults guards

These validators are **already implemented**; their asserting tests in
`test_no_silent_defaults.py` are dormant (`@pytest.mark.skip`, "Phase 2"). The first batch
lifts the skips and pins each guard (assert the raise *and its message*), converting dead
spec into live regression protection. Expect most to pass immediately (they certify
existing guards); any that don't are findings.

| Spec | Guard in code | Dormant test to activate |
|---|---|---|
| §3.1 geometry: `deflection_ratio` with cosmo / `redshift` without cosmo / missing geometry | `src/gigalens/jax/scene.py:164-187` | `test_missing_deflection_ratio_with_multiple_planes_raises`, `test_redshift_without_cosmo_raises_and_deflection_ratio_with_cosmo_raises` |
| §3.2 missing/unknown profile param | `src/gigalens/jax/scene.py:217-219` (+ doc `:59`) | `test_missing_or_unknown_param_raises` |
| §3.3 reused bare `tfd.Distribution` (must wrap in `shared()`) | `src/gigalens/jax/scene.py:198-204` | `test_reused_bare_distribution_raises` |
| §3.4 dataset without noise | `src/gigalens/jax/scene_prob_model.py:55-66` | `test_dataset_without_noise_raises` |

## Grader decisions (resolved 2026-06-25)

1. **Raise vs. flag.** Resolved: **flag (warn + model card) by default; raise only
   conservatively** — i.e. raise iff the input cannot yield a valid computation
   (non-finite, domain-violating, missing-required); flag when finite-but-suspect. So:
   raise → M3 (NaN/inf/neg error map), C1 (`z≤0`, `z_source<z_lens`); flag → C3 (`Ode0<0`
   finite), `z_source==z_lens`, negative net light (project-standards §3).
2. **Guard timing.** Resolved: **construction-time** (matches the no-silent-defaults
   convention). M3 guard in `ImageData.__init__`; C1 guard at cosmo construction / first use.
3. **Inference layer.** Resolved: **hold.** The grader has separate plans for the
   adversarial inference layer (it is compute-heavier). Not in scope for this batch.

   *Flag mechanism confirmed:* `warnings.warn(UserWarning)` is the established channel
   (e.g. `gigalens/jax/profiles/light/shapelets.py:51`; `z_scores` warns on absent
   params); `model_card()` (`GIGALens-Code/.../inference_utils/pipeline.py:322`) is the
   persistence surface — a physicality-flags field there is a small build item, not a
   blocker. Tests assert flags via `pytest.warns`.

## First increment — run outcome (PROPOSED / UNCERTIFIED)

`tests/validation/test_redteam_cosmology.py`, run in-container (float64), 2026-06-25:
**4 passed, 2 xfailed.**

- **C2 deflection_ratio assembly invariants — PASS (proposed).** `ratio(z_ref)==1`,
  `ratio(z_lens)==0`, strictly increasing & bounded all hold at the float64 floor. This
  is a *positive* finding for the stated #1 worry: the ratio assembly in the current code
  is **not** silently wrong (a D_s/D_ls swap would have broken these). Now locked against
  regression.
- **M4 batched==looped — PASS (proposed).** Vectorized distances (`comoving_distance`,
  `deflection_ratio`, `distance_matrix`) match the per-element scalar computation to the
  float64 floor. The Pitfall-4 vectorization is correct, not silently mis-batched.
- **C1 invalid redshift ordering — XFAIL (gap, tracked).** No domain guard yet; ratchet
  flips to a hard failure (prompting un-xfail) when the construction-time guard lands.
- **C3 unphysical-density flag — XFAIL (gap, tracked).** No physicality warning yet;
  ratchet per the resolved flag policy.

**Landed since (grader-approved):**
- **M3 guard — DONE.** `ImageData._validate_finite_noise` + derived-noise input validation
  in `scene_prob_model.py`; `test_redteam_dataset.py` (7 tests) passing. A sub-finding
  surfaced *by* the test: the derived-noise path silently accepted a negative `exp_time`
  (the background term masked it into a finite sigma) — now raised at the source, not left
  to the downstream sigma check.
- **§3.1–§3.4 guards ACTIVATED.** The five dormant `test_no_silent_defaults.py` stubs are
  implemented against the existing validators (all passing); module marker `pending`→
  `redteam`. These were already-built guards with no live test — now pinned against
  regression.

**Full suite after this batch:** 61 passed / 16 skipped / 2 xfailed / 0 failed (was
45/21/0/0). No regressions from the `ImageData` change.

## Second increment — run outcome (PROPOSED / UNCERTIFIED, 2026-07-09)

Human grader approved landing C1+C3 and writing M2+C4 (user directive, 2026-07-09:
"go ahead with landing C1+C3 and write M2+C4", terminal session b85291ca). First
rigor-grader round returned NEEDS-MORE with two structural findings — both guards
were **dead on the jitted scene path** (under jit every argument is a tracer,
including a plain-number redshift the user fixed, so the tracer-skip disabled the
guard on exactly the path the misuse rows describe): a fixed light plane at z=0.3
behind-listed of a z_lens=0.5 mass rendered silently with deflection_ratio −1.357
(probe P2), and traced `Om0=1.5` produced 0 warnings (probe P3). Fixed by adding the
scene-construction layer below, where the values are last concrete. Run in-container
(float64), branch `redteam-c1c3`:

- **C1 guard — DONE (two layers).** (a) Eager cosmology API: construction-time raises
  in `CosmoBase.__init__` (`z_lens ≤ 0`, `z_source_ref ≤ z_lens`, non-finite) +
  first-use raises in `_validate_source_redshift` via `lensing_distance`
  (`z_source < z_lens`, `z ≤ 0`, non-finite); `z_source == z_lens` deliberately stays
  valid (ratio 0 is the physical answer, C2 pins it). (b) Scene path (the grader-round
  fix): `LensModel._validate_concrete_redshifts` raises at model construction on any
  concrete plane redshift that is non-finite/≤ 0, or that breaks the documented
  observer→source ordering — mis-ordering is precisely the silent-negative-ratio
  misuse, and ordering makes a correctly-placed foreground light plane render
  undeflected (legitimate use pinned by a scope test). **Residual (documented, not
  guarded): genuinely sampled plane redshifts** — bounding those is the prior's job.
  Ratchet flipped: xfail removed; 8 eager raise cases + 3 construction raise cases +
  1 foreground-legitimacy identity live.
- **C3 flag — DONE (two layers).** (a) Eager: `CosmoBase._flag_unphysical_densities`
  at the `comoving_distance_z1z2` choke point warns on `Om0 < 0` or `Ode0 < 0`; never
  raises (resolved FLAG policy); physical control case pinned NOT to warn. (b) Scene
  path (grader-round fix): `LensModel.__init__` flags concrete (constant) cosmo
  params at construction — the jitted distance calls can never warn. Rigor-grader
  round 2 (F1) caught the first version gating this on a concrete `H0`, which
  silently disabled the flag for the COMMON configuration (sampled H0, fixed
  unphysical Om0); fixed by substituting a nominal H0=70 (it only enters the check
  through the ~1e-4-scale radiation/curvature terms) and pinned by the mixed-case
  test. **Residuals (documented): fully sampled cosmo params** (prior support is the
  guard; a warning cannot fire inside jit), and the model-card persistence field
  (small open build item).
  **Caveat:** `Om0=1.0` flat (EdS) warns — with radiation, Ode0 = −Or0 ≈ −8.5e-5 < 0,
  consistent with `efunc`'s own parameterization (that configuration is genuinely
  slightly over-closed).
- **M2 — DONE (2 tests).** Redshift-geometry render ≡ explicit-ratio render at the
  float64 floor AND loudly ≠ a both-ratios-1 render, under both `simulate` and
  `lstsq_simulate`. Loud floor derived: (r_B−1)·α ≈ 0.55" ≈ 11 px shift ⇒ O(1) pixel
  changes; 1e-3·peak sits ≥2 orders above solver noise and ~580× below the measured
  geometry signal (0.58·peak).
- **C4 — DONE.** `distance_matrix` structural invariants (strict lower-triangularity,
  positivity, per-column monotonicity) for the multiplane trace's input object.
- **Sub-finding (scope pin, worth knowing):** `lstsq_simulate`'s reconstruction is NOT
  at the float floor — `_solve_normal_eq_with_fallback` adds a deliberate
  `1e-6·mean(diag)` Gram jitter (the shapelet-VJP safeguard, `jax/simulator.py`),
  which biases the coefficient of a basis whose norm sits below the diagonal mean by
  `ε·mean(diag)/G_ii` (measured 1.8e-5 here, image error 1.0e-5·peak; exact solve of
  the same system: 1.4e-14). Equality/round-trip tests on multi-basis lstsq output
  must budget for this documented solver bias, not assume machine precision — same
  class as the jit-recompile non-bitwise finding in the GIGALens-Code record.

**Full suite after this batch: 68 passed / 16 skipped / 0 xfailed** + the 1 known
pre-existing baseline failure (`test_regression_anchor` LOG_PRIOR max rel 1.258e-07,
TFP-nightly vs golden freeze — signature unchanged, predates this batch). Was
61/16/2 + same failure. No regressions from the `cosmo.py` / `scene.py` guards; the
foreground-light identity test pins that the ordering guard does not over-reject
legitimate use.

**Remaining (unscheduled):** model-card physicality-flags field (C3 persistence);
mutation-testing sensitivity pass (below); adversarial inference layer (held, user's
own plans). The C2/M4 invariant tests needed no product change.

## Note on suite sensitivity (proposed meta-check, not yet scheduled)

The user's core complaint is "the suite didn't catch everything." Suite *size* ≠
sensitivity. A mutation-testing pass (`mutmut` / `cosmic-ray`) over the cosmology +
guard modules would report which mutations survive with the suite green — the real holes.
Proposed as a follow-up once the first batch lands, to measure whether the red-team layer
actually deepened coverage.

## Third increment — physicality layer (PROPOSED / UNCERTIFIED, 2026-07-10)

**Scope (user-approved draft):** per-profile physicality domains with CODE-DERIVED hard
bounds; soft (plausibility) bounds deliberately left unset for human curation. Session
b85291ca ("can you build a draft of this validation in a worktree? Please go with
code-derived bounds for the hard bounds, and leave the soft bounds to me").

**Design (as discussed and approved in-session):**
- `src/gigalens/physicality.py`: `Domain` (hard interval + empty `soft` slot +
  line-citing rationale), `JointConstraint` (e.g. `e1^2+e2^2 < 1`, BPL `r_c <= b`),
  probe engine, `PhysicalityReport` (attached to `LensModel` as
  `model.physicality_report` — the model-card persistence hook).
- Registration is **own-class only** (`vars(cls)`), not inherited: a subclass's params/
  kernel were not audited by its parent's entry (caught live: `CoreSersic` inheriting
  `Sersic._domains` while adding `Rb/alpha/gamma`). Unaudited profiles are ratcheted by
  `PENDING_PHYSICALITY_AUDIT` + a meta-test (missing/stale/ghost all fail).
- Enforcement at `LensModel` construction (same jit-blindness rationale as C1/C3):
  fixed value outside hard domain → **raise**; prior mass outside → **warn** with the
  estimated mass. Scalar priors via exact `cdf` where available; joint constraints and
  cdf-less priors via a fixed-seed MC probe. **Derived thresholds:** warn at
  `EPS_MASS = 1e-6` (≈1 expected excursion per typical ~1e6-eval run); probe size
  `K = 3/EPS_MASS = 3e6` (rule of three: zero violations certifies mass ≲ 1e-6 at 95%).
  Report records the resolution floor of clean checks (absence of evidence made explicit).

**Hard bounds are verified, not read:** every registered bound cites a kernel line AND a
2026-07-10 numerical verification (`kernel_pathology_check.py`, job tmp; pinned as
kernel-truth tests). The verification **overturned three code-reading predictions** —
the reader is not the authority, the kernel is:
- SIE at exactly `e1=e2=0` is ALL-NaN (`b/sqrt(1-q^2)` → 0/0); EPL circular is fine.
- EPL `theta_E<0` does NOT NaN at `gamma=2` (exponent `t-1=0`): it silently renders the
  sign-flipped *repulsive* lens; NaN only for `gamma≠2`. `theta_E=0` renders all-zero.
- Sersic `R_sersic<0` does not zero the image: the NaN-safety mask zeroes every
  OFF-CENTER pixel, silently collapsing the profile to its central spike with the peak
  value unchanged.
Also verified: EPL `gamma=5` series-pole NaN; `|e|≥1` renders identically-zero images
(EPL deflection, SersicEllipse light); NFW `Rs=0` → inf, `Rs<0` → silent ~1e-20
deflection; NFW_ELLIPSE_EINSTEIN `theta_E<0` → silent ~9e10 garbage; Shapelets `beta=0`
NaN / `beta<0` silent parity-mirror; BPL `r_c>b` bitwise-identical to `r_c=b` (silent
clamp) and `alpha=3`/`alpha_c=3` poles.

**Test calibration (never tuned):** prior-mass assertions use analytically computable
fixtures — Uniform(-0.9,0.9)² vs unit disk (violating mass 0.1029 closed-form), fixed
`e1=0.7` + `e2~N(0,0.3)` conditional mass `2Φ(-√0.51/0.3)=0.0173`, `Normal(0.5,0.25)`
theta_E tail `Φ(-2)` exact via cdf — asserted within the probe's own binomial 5σ band.

**FINDING (fixtures, needs human decision):** the layer immediately flags the existing
suite fixtures' priors as leaking mass into kernel-invalid regions: `R_sersic` normals
put **0.159–0.274** of their mass at ≤0 (spike-collapse region), ellipticity normals
put ~4e-3 jointly at `|e|≥1` (zero-image region), `gamma` 1e-3 outside (1,3), `theta_E`
1.2e-4 at ≤0. These warnings now fire during the suite (tests pass; warn-not-raise per
policy). Fixtures were deliberately NOT re-priored here — golden anchors depend on prior
log-probs; re-prioring is a separate, human-approved change.

**Costs / residuals:** suite 62s → 91s (probe cost at K=3e6 per free-prior component;
knob = `physicality.PROBE_DRAWS`/`validate_planes(draws=...)`); 15 profiles remain on
the pending-audit ratchet; posterior-time excursions (likelihood dragging samples to a
clip boundary the prior barely touches) are NOT covered — that is a run-time diagnostic,
proposed as a follow-on register row; soft bounds all unset (user-curated next). The
pending-audit list is 16 keys / 15 distinct classes (mass and light `combined_profile`
twins are now separate keys).

**Definition/theory-based bounds (grader round-1 correction — NOT clip/mask/NaN-derived,
all flagged for human review):** `Sersic.Ie ≥ 0`; `SIS.theta_E ≥ 0` and `SIE.theta_E ≥ 0`
(the kernel *faithfully* computes the sign-flipped repulsive model — raising blocks
deliberate negative-mass components); `NFW.alpha_Rs ≥ 0` (same); the interval choices of
`EPL.gamma ∈ (1,3)` (the γ=5 pole is code-derived; the (1,3) endpoints are Tessore &
Metcalf validity + integrability) and `BPL.alpha`/`alpha_c` ranges. All were numerically
verified to change the rendered model; none derive from a guard line in the kernel. The
original claim that `Ie ≥ 0` was "the one" definition-based bound was **wrong** and is
withdrawn; the in-code rationales now carry an explicit DEFINITION-BASED marker.

**Grader round-1 (CERTIFY-RECOMMENDED, scope narrowed) — catches, all applied:** grader
independently reproduced the closed forms (Φ(-2)=0.0227501; square-vs-disk 0.102874 +
its own 2e7-draw MC 0.10277; conditional 0.017290), the suite counts, the fixture-leak
masses, and fixed-seed determinism (bitwise-equal across constructions). (1) The
mis-categorized definition-based bounds above. (2) Grouped tuple-key and shared()
physicality paths were verified correct by grader probe (correlated-MVN joint mass
0.32652 vs analytic 0.32633; shared handle flagged at both sites) but were **unpinned by
the suite** — three tests added: grouped Rayleigh-tail exact (0.19147), correlated
(r_c, b) at Φ(1)=0.8413 with an explicit broken-correlation decoy at Φ(1/√2)=0.7602,
shared theta_E flagged at BOTH sites at Φ(-2). (3) Pending-audit ratchet key collision
(mass/ and light/ combined_profile shared one key) — keys now carry the subpackage path
and the walker asserts collision-freedom. (4) Rule-of-three "95%" wording was applied to
nonzero probe counts — clean-check strings now say exact / binomial-se / rule-of-three by
case, and the ~1e6 evals/run figure behind EPS_MASS is marked as an assumption.

**Full suite after this increment: 93 passed / 16 skipped / 0 xfailed** + the same 1
known pre-existing baseline failure (LOG_PRIOR max rel 1.258e-07, signature unchanged).

## Fourth increment — sampled geometry + dataset-shape guards (PROPOSED / UNCERTIFIED, 2026-07-10)

**Origin (user coverage audit):** the user asked whether the suite checks (1) parameter
sharing across profiles, (2) edge cases in datasets sharing/splitting observables,
(3) directly sampling deflection ratios with their own priors (no cosmology), and
(4) over-constrained cosmology (ratios AND cosmo params sampled together). Audit
verdicts: (1) covered, minus a cross-role (mass↔light) share; (4) structurally
impossible — `_validate_geometry` raises on `deflection_ratio` + cosmology, pinned by
two tests; (2) covered for multi-band imaging + sees-validators but with unguarded
shape/mask edges; (3) the feature exists (`_classify` accepts a tfd/shared
`deflection_ratio`; the trace consumes it from params) but had ZERO tests — every
existing use is a fixed float. This increment closes (3), the (1) gap (STRUCTURAL
only: prior tree + `to_params` fan-out, not a render-level assertion on the shared
centroid), and the (2) edges. User approval: "Yes, that red-team file is a good
idea. Go ahead with it!"

**New guards (`ImageData.__init__`, `scene_prob_model.py`):**
- **§3.8 shape agreement** — `error_map` and `mask` must match the image shape
  EXACTLY (derived-noise `background_rms`/`exp_time`: scalar or image-shape).
- **§3.9 all-masked dataset** — `event_size == 0` raises: among live datasets its
  chi2/ll terms are identically zero, so the observation is silently ignored (the
  chi2-channel `NotImplementedError` only fires when ALL datasets are empty).

**Pre-guard kernel behavior verified, not read** (stash + probe, 2026-07-10; one
code-reading prediction overturned): an `(8,)` error_map did NOT pass silently — it
crashed with a cryptic IndexError in the finite-noise check; the SILENT case is an
`(8,8,1)` error_map, accepted with the likelihood residual broadcasting to `(8,8,8)`
(512-element pseudo-likelihood). The `(8,)` boolean mask WAS silent (event_size=8,
whole rows — first-axis indexing), and the all-False mask WAS silent (event_size=0).
Test docstrings state the verified truths, not the predicted ones.

**New tests:**
- `test_redteam_dataset.py` +5: error_map mismatch (both the silent `(8,8,1)` and the
  cryptic-IndexError `(8,)` shapes now raise §3.8), mask mismatch, derived-noise
  shape mismatch (scalar and full-shape allowed — scope pin), fully-masked raises
  (§3.9), single-unmasked-pixel constructs (scope pin: the guard is exactly
  `event_size == 0`).
- `test_redteam_sampled_geometry.py` (NEW, 4): a `deflection_ratio` prior is a
  genuine free parameter (prior leaf at the canonical path key, one z column,
  `z_param_names` labels it, `to_params` fan-out exact, flat-z round trip at the
  REDUCTION floor); the sampled ratio DRIVES the render (sampled==fixed at the same
  value at the REDUCTION floor; response bound derived: Δr=0.4 ⇒ Δβ≈0.4·θ_E=0.44″ >
  R_sersic=0.30″ ⇒ order-unity image change, asserted >0.1·peak, vs ΔI≡0 for a trace
  reading geometry from constants); a `shared()` ratio across two source planes is ONE
  parameter fanned to both geometry sites; a mass↔light shared centroid ("mass follows
  light") is one parameter per coordinate fanned to both role sites.

**Residuals (documented, NOT fixed — future increments/decisions):**
- **Sampled redshifts skip the ordering guard**: `_validate_concrete_redshifts` only
  checks concrete values, so a redshift PRIOR putting mass below `z_lens` silently
  yields a negative derived deflection ratio. Same class as the pre-increment (3) gap
  (sampled-geometry paths systematically less checked than fixed ones); needs a
  physicality-style prior-mass check on redshift ordering.
- `deflection_ratio` has no hard physicality Domain: a negative sampled ratio renders
  the mirror (repulsive) deflection faithfully — definition-based-bound territory,
  parked with the soft-constraint decision.
- Heterogeneous observables beyond imaging (time delays) and multi-system sharing
  remain untested (no such Dataset type exists yet).

**Grader round-1 (CERTIFY-RECOMMENDED, scope narrowed) — all catches applied:** the
grader independently rebuilt the PRE-guard kernel in a shadow tree and reproduced all
four claimed silent/cryptic modes (plus a bonus: row `background_rms` was also silent
pre-guard, now §3.8-covered); re-derived the response geometry (ring radii 0.816″ /
0.395″, both inside the 2.086″ field; measured response **0.98·peak** vs the 0.1·peak
bound; sampled==fixed **bitwise 0.0**, so REDUCTION tolerance is not load-bearing);
recounted the integer free-param assertions from `profile.py` (lstsq drops `Ie`);
reproduced all suite counts. Catches (all low, applied): (1) test-2 docstring claimed
partial-application "also caught" — false for responses in (0.1·peak, peak); reworded
to the verified scope. (2) "closes the (1) gap" scoped to STRUCTURAL (no render-level
shared-centroid assertion). (3) explicit doubt report added (below).

**Doubt report:** the increment's empirical core ("these modes were silent before")
was produced by the same party that wrote the guards (stash-and-probe); the grader
closed this by independently reproducing all four on the HEAD kernel. Remaining
exposure: the sampled-geometry tests certify construction→fan-out→render WIRING on one
easy scene (single deflector, forward mode, no PSF) — nothing here tests INFERENCE
through a geometry prior (`log_prob`/bijector under a real sampler), which is where
the feature would actually be used; "the sampled ratio drives the render" must not be
read as "a sampled ratio is inferable."

**Full suite after this increment: 102 passed / 16 skipped / 0 xfailed** (+9: 4
sampled-geometry, 5 dataset-shape/mask) + the same 1 known pre-existing baseline
failure (LOG_PRIOR max rel 1.258e-07, signature unchanged); 92.9s (vs 91s — the new
tests avoid probe cost by fixing every parameter not under test).

## Fifth increment — sampled-redshift physicality (PROPOSED / UNCERTIFIED, 2026-07-10)

**Origin:** the fourth increment's documented residual, user-requested: "Can you also
add the prior physicality check for sampled redshift?" `_validate_concrete_redshifts`
(C1, scene path) raises on concrete violations but by design skips free redshifts, so
a redshift PRIOR putting mass at z ≤ 0 or below the preceding plane's redshift passed
silently — the silent-negative-deflection-ratio misuse, in prior form.

**Design (`physicality.validate_redshift_geometry`, run inside `validate_planes` so
it shares the report, the raise/warn policy, and the fixed-seed draw cache):** the
exact constraint set of the concrete guard, in prior-mass form —
- per plane with a sampled redshift: prior mass at z ≤ 0 (exact cdf where available,
  else probe; non-finite draws count as outside, mirroring `Domain.violates`);
- per ADJACENT plane pair with ≥1 sampled side: mass violating non-decreasing
  ordering, P(z_i < z_{i-1}). Adjacency is COMPLETE for the ordering constraint (a
  sequence is non-decreasing iff every adjacent pair is), so no pair set was chosen
  by judgment. Exact one-sided cdf when the other side is fixed; fixed-seed probe
  when both are sampled (independent free params ⇒ independent draw sets); the SAME
  shared() handle at both sites is structurally ordered (one free param, identical
  draws) — recorded as a clean check, not skipped. DISTINCT handles wrapping one
  dist object are two independent free params and are probed independently (grader
  rd-1 — see below). Fixed–fixed pairs stay with the concrete guard (raise).
- Threshold: the same derived EPS_MASS = 1e-6 (identical visited-region argument:
  mass eps in the reversed-ordering region ⇒ ~eps·1e6 evaluations per typical run,
  each computing a silently sign-flipped deflection). Severity: warn, never raise —
  same policy as all prior-mass findings. Findings carry profile="geometry".
- Coverage note: the mass plane's redshift participates in the pairwise chain, and
  §3.1 requires every plane to carry a redshift when a cosmology is present, so
  "source prior below the deflector" is covered without referencing `cosmo.z_lens`.

**Hole verified, not read (stash probe, 2026-07-10):** pre-change, a source redshift
~ Normal(2, 1) behind a fixed z=0.5 lens — 6.68% of its mass in the reversed-ordering
region, 2.28% at z ≤ 0 — constructed with **zero** UserWarnings and an empty geometry
report. Post-change: exactly two warnings whose report masses match the closed forms.

**Test calibration (4 new tests in `test_redteam_sampled_geometry.py`, never tuned):**
ordering mass Φ(-1.5) = 0.0668072 and domain mass Φ(-2) = 0.0227501 exact (cdf path;
independent oracle = `math.erfc` in-test); both-sampled ordering forced onto the probe
path with closed form P = Φ(-0.5/√0.02) = 2.03e-4 asserted within the probe's own
binomial 5σ band; shared-handle ordering asserted structurally clean under
warnings-as-errors (falsifier: independent draws at shared sites would poison every
shared-handle joint check); sane control (Normal(2, 0.1) behind z=0.5, masses
~1e-51/1e-89) silent under warnings-as-errors with the clean checks recorded in the
audit trail.

**Grader round-1 (REJECT) — the catch was real and is FIXED + PINNED:** the grader
confirmed the closed forms, the 5σ band, the adjacency-completeness argument (probing
the mixed chain [fixed 0.5, sampled N(3,0.1), fixed 1.0] — pair (1,2) flagged at mass
1.0, correct direction), the cdf directions, the suite counts, and reproduced the
pre-change hole stash-free (shadow HEAD copy) — but caught a MAJOR defect: the
structural shortcut tested DIST-OBJECT identity, not HANDLE identity. Two distinct
`shared()` handles wrapping ONE dist object are two independent free redshifts (two
uids, two prior entries — the true ordering-violation mass is exactly 1/2 by iid
symmetry), yet the pre-fix branch declared them "structurally impossible" and wrote a
false justification into the audit trail — a new silent clean inside the guard built
to eliminate silent cleans. Fix: `_DrawCache.draws` now keys on the HANDLE where one
exists (same handle ⇒ one draw set; distinct handles ⇒ independent draws) and the
structural branch compares handles. The same `id(dist)` keying pre-dated this
increment in `_resolve_param_values` for COMPONENT params (falsely correlated
joint-constraint masses in the same configuration) — fixed by the same mechanism.
Both directions pinned: redshift pair probed at 0.5 ± 5σ with the structural line
asserted ABSENT; component ellipticity probed at the independent closed form
exp(-1/(2σ²)) = 3.8659e-3 with the correlated decoy 2Φ(-1/(σ√2)) = 1.842e-2 asserted
>10 bands away. Producer-honesty note (grader rd-1, on the record): the defect
surfaced via grading, not the producer's own doubt report.

**Grader round-2 (CERTIFY-RECOMMENDED, 2026-07-10 — awaiting human certification):**
grader re-ran the rd-1 counterexample live (mass 0.49993 within the 5σ band of
exactly 1/2, structural line absent, 2 free params, 1 warning); differentially
confirmed the component decoy IS the pre-fix value (shadow HEAD physicality.py:
0.018438 vs decoy 0.018422; post-fix 0.0038537 vs independent 0.0038659); verified
cross-kind handle sharing (one handle at a redshift site AND a mass-param site) still
yields one free param; reproduced 29-passed and 108/16/1 suite counts. Remaining
cosmetic note: a DIRECT `validate_planes` call (outside LensModel) with one BARE dist
object at two redshift planes would take the structural shortcut with "shared handle"
wording — unreachable via LensModel (bare reuse raises at derivation, before
validate_planes); cosmetic only. Rd-2 scope caveat, quoted for the record: all
calibration oracles are Normal cdfs; truncated/transformed/mixture priors exercise
the cdf-fallback and probe in regimes no test pins, and warn-never-raise means even a
mass-1.0 geometry constructs — "108 passed" is not coverage of configurations the
tests never touch.

**Residuals:** cosmology-mode single-deflector models also require the mass plane's
redshift to EQUAL `cosmo.z_lens` (a SceneSimulator-construction check on concrete
values; verified to skip sampled redshifts) — a sampled mass-plane redshift prior
straddling z_lens is not flagged by this increment (different contract:
equality-to-a-constant, not ordering; needs its own derivation of "how much prior
mass away from z_lens matters"). A both-sampled/mixed pair with ordering mass ≈ 1 (a
priori almost-surely invalid geometry) still constructs with a WARNING, not a raise —
deliberate consistency with the prior-mass policy (grader rd-1 catch 3: named here so
the asymmetry with the all-concrete raise is on record; escalating mass ≈ 1 to error
severity is a future policy decision). Non-imaging datasets and the deflection_ratio
definition-based bound remain as before.

**Full suite after this increment: 108 passed / 16 skipped / 0 xfailed** (+6: 4
redshift-physicality, 2 grader-rd-1 handle-identity pins) + the same 1 known
pre-existing baseline failure (LOG_PRIOR max rel 1.258e-07, signature unchanged);
120s (vs 92.9s — three extra 3e6-draw probe pairs across the new tests).
