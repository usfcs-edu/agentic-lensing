# AstroMark lens profile, version 1.0 — draft

**Status: draft for review. Not adopted.**
Profile of `astromark/1.0`. Vocabulary: `astromark-vocab/lens-1.0`.

This profile supplies the terms for annotating strong gravitational lens candidates. The core
supplies geometry, portrayal separation, provenance and the render contract, and defines no domain
terms.

## 1. Families

Every role belongs to one family. The family, not the individual role, is what survives at thumbnail
size, so it carries the coarsest and most important distinction.

| family | contains | the distinction it protects |
|---|---|---|
| **lens mass** | deflector, second deflector, satellite, lens-light region | mass that bends light |
| **lensed light** | lensed image, arc, counter-image, counter-arc, knot, secondary ring, lensed-light region | deflected light |
| **obstruction** | dust lane, artifact | something that hides or imitates |
| **field** | field galaxy, nearby galaxy, companion galaxy, star, ambiguous structure | not part of the system |
| **measurement** | Einstein ring | a quantity, not a feature |

**A lens-mass object is never a lensed image.** This is the single most consequential rule in the
profile and the one most often broken: a galaxy that is bending light is labelled as a lens, whatever
its position relative to the visible arcs.

## 2. Roles

The full table with definitions is generated at `generated/vocabulary.md`. Four terms carry rules
that are not obvious from their names:

**`lens:satellite`** — a galaxy bound to the deflector. It is lens mass and a separate light
component for modelling. **It is never a mask target.** Circling it is an error, not a preference.

**`lens:counter_image`** — the image of the *same* source lying across the deflector. It may lie at a
**greater** radius than the main image. Lensing permits this, so a larger radius is never a ground
for rejecting a candidate, and never a reason to shrink the ring to it.

**`lens:knot`** — a compact bright region *within* a lensed image. A whole compact lensed image is a
lensed image, not a knot.

**`lens:dust_lane`** — an obscuring band inside the deflector. It explains why part of a lensed
image may be invisible. **It is not evidence against lensing.** A system carrying a dust-lane mark
records `lens:dust_lane_case` in `hard_case`.

## 3. Polarity, and what a negative mark is for

A negative mark is how the record states a refutation:

```json
{"id": "m8", "type": "vector", "role": "lens:arc",
 "polarity": "core:negative", "alternative": "lens:spiral_arm",
 "tail": [0.851, 0.358], "head": [0.739, 0.282]}
```

*There is an arc-shaped feature here; it argues against lensing, because it is a spiral arm.*
An annotation of a non-lens is as informative as an annotation of a lens, and the format must be
able to carry the reasoning either way.

`core:ambiguous` is the class for evidence that supports two readings at once, and is the natural
class for a borderline system: the record shows the feature **and** its alternative rather than
choosing silently.

## 4. The Einstein radius

`system.theta_e` carries `value_arcsec`, `lower_arcsec`, `upper_arcsec` and a coded `method`.

**θ_E is a radius, never a diameter.**

The nominal value is taken at the **radial midpoint of the main lensed image** — not its inner edge,
not its outer edge, and not half the separation between the main image and the opposite one. That
last rule is available as a cross-check (`lens:half_separation`) but is not the primary rule, because
the opposite image may lie farther out than the main image.

Bounds are the inner and outer edges of the main image. `lower ≤ value ≤ upper` is a validation
constraint. The term `alt` is deprecated: a bound whose direction is unstated is not a bound.

Precision is recognition-level, not model-fit level. A ring drawn to a few tenths of an arcsecond is
correct; one tuned past that is over-claiming, since a circle is already an approximation to an
elliptical critical curve.

**Ring marks.** At most one nominal ring per source, plus optional `lower` and `upper` bound rings.
Every ring uses `center_ref` to track a deflector mark. A ring imprinted by a *second* deflector is a
mark with role `lens:secondary_ring`, not a second nominal ring.

**The drawn circle is not the observed ring.** These are two different things and the name invites
confusing them:

| what you are marking | role | carries polarity? |
|---|---|---|
| the **circle recording how large θ_E is** | `lens:einstein_ring` | **no** |
| the **ring of light visible in the data** | `lens:arc` / `lens:lensed_image` marks, or a `lens:lensed_light` region, with the source's `config` set to `ring` | **yes** |

So a near-complete Einstein ring in the image is marked as lensed light in a ring configuration, and
the `einstein_ring` mark beside it records only its size. Marking the observed ring *only* as an
`einstein_ring` loses the evidence: it records the measurement and drops the thing being measured.

The asymmetry has a reason. θ_E is **downstream** of the lensing interpretation — it can only be
computed once a lens has been assumed — so it presupposes the claim and cannot be evidence for it.
That is also why the size of θ_E is supporting evidence and never the sole ground for a verdict, a
rule this profile puts in the schema rather than leaving to a prompt.

Doubt is carried accordingly, on three separate channels: uncertainty in the **measurement** by
`lower_arcsec`/`upper_arcsec` and the coded `method`; doubt about the **interpretation** by polarity
on the evidential marks; and a ring simply drawn in the wrong place or at the wrong size by a review
verdict of `wrong_size`.

## 5. Counterpart search

`system.counter_image` records `core:found`, `core:not_found` or `core:not_searched`.

`core:not_found` asserts that a search was carried out — across the deflector from every marked
lensed image, at all radii out to the edge of the cutout — and that none was located. It is not the
default, and it is not the same statement as `core:not_searched`.

When a candidate is tested and fails, it is recorded with a negative or ambiguous polarity and the
reason, rather than left unmarked. A test that produced a result is evidence; silence is not.

## 6. Sources and multiplicity

`source` tags a mark as an image of a particular physical source (`S1`, `S2`, …), and
`system.sources` declares each source with its image count and configuration.

Two lensed features that cannot be paired are **both** marked. Whether they come from one source or
two is a separate question, recorded in `sources` or left open — it never justifies leaving a clearly
lensed feature unmarked.

Deflector-side roles never carry a source tag: a lens is not an image of anything.

## 7. Field objects and their treatment

A `circle` mark on a field object carries `treatment`:

- `core:mask` — exclude these pixels from the fit.
- `core:model` — the object's light reaches the lens system, so fit it as its own component.

**Never circled:** the deflector, its extended halo, a bound satellite, any lensed image, any
counter-image, or the diffraction spikes of an already-marked star. These are lens mass or lensed
light; they are modelled, not removed. The first four are validation errors rather than style
warnings.

The deflector's light extends well beyond its bright core. Treating the visible core as the whole
galaxy is a modelling error the record should not encourage.

## 8. Hard cases

`system.hard_case` records the morphologies that make a system difficult but leave it a lens: a dust
lane, a second deflector, a merged pair, a faint counterpart, a counterpart outside the main image,
two sources, a single giant arc, group scale, low signal-to-noise, an obscured arc, dominating lens
light.

These are for retrieval as much as for reading. A corpus that cannot be filtered to its hard cases
cannot be used to teach them.

## 9. Segmentation

Two region classes, with **deliberately asymmetric tolerance**:

- `lens:lens_light` — the deflector's light. Tolerant: roughly half the visible halo. Tracing the
  faintest isophote claims a boundary the data does not support.
- `lens:lensed_light` — lensed light, including faint opposite images. **Recall matters here.** A
  missed faint counterpart is a real error; an approximate outer boundary is not.

## 10. What the profile deliberately does not decide

Recorded because a specification that pretends these are settled would be wrong.

1. **The grade scale.** Grades are carried as `A`–`D`, but only the borderline class has a definition
   on record. The boundaries need setting before grades can be compared across annotators.
2. **The probability bands** attached to verdicts are a proposal for consistency, not measured
   values. They need ratifying.
3. **The nominal ring's colour**, which is a style question and belongs in the style document, but on
   which practice has differed.
4. **Which roles take polarity is settled but unratified.** Fifteen of the twenty roles declare
   `takes_polarity: true`. The five that do not are `field_galaxy`, `nearby_galaxy`, `star`,
   `artifact` and `einstein_ring`.

   The principle: a role takes polarity when the mark can bear on whether the lensing interpretation
   is true. **Inventory roles cannot** — they assert only that something is present and how it
   should be treated. **Measurement roles cannot either, for a different reason:** θ_E is downstream
   of the interpretation, computable only once a lens has been assumed, so it presupposes the claim
   and cannot be evidence for it. Excluding it also puts an existing domain rule into the schema
   rather than a prompt — that the size of θ_E is supporting evidence and never the sole ground for
   a verdict.

   **The distinction the name `einstein_ring` invites, and which must not be lost:** the *drawn
   circle* recording how large the ring is takes no polarity, while the *observed ring of light in
   the data* is powerful evidence — and it is a different mark. An observed ring is marked as
   `arc` / `lensed_image` marks, or a `lensed_light` region, with the source's `config` set to
   `ring`; all of those take polarity.

   Two boundary calls are judgement and the room should confirm them: `companion_galaxy` takes
   polarity because it records the outcome of a failed counter-image test, while `nearby_galaxy`
   does not because it is a modelling instruction.