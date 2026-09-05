# AstroMark core, version 1.0 — draft

**Status: draft for review. Not adopted.**

AstroMark records annotations on an astronomical image as vector geometry with controlled semantic
terms, separately from how those annotations are drawn. This document defines the core. Domain
vocabulary lives in a profile; the lens profile is `astromark/lens/1.0`.

## 1. Purpose and scope

The record exists so that one person can communicate precisely with another about what is visible in
an image, and so that a model can receive that judgement in a form it can be corrected on. It serves
both search and modelling: roles, treatments and image-relative coordinates are modelling inputs,
not only descriptions.

This specification is written to be handed to a model on its own. Every semantic rule is in this
document and in the vocabulary file, not only in a prompt.

## 2. The three-file contract

| file | role |
|---|---|
| `<id>.png` | the original image, never modified |
| `<id>.astromark.json` | the record: geometry, terms, provenance |
| `<id>.annot.png` | a deterministic render, derived from the record |

**The record is primary and the render is derived, never the reverse.** Annotations are an overlay;
they are never burned into the image. A consumer that has only the render has lost the record.

## 3. Coordinates and units

Geometry is **normalized**: `[u, v]` with `u` rightward, `v` downward, origin at the top-left corner,
so `[0,0]` is the top-left and `[1,1]` the bottom-right of the image. Geometry is therefore
independent of pixel dimensions and of any resampling.

Sizes are in **arcseconds**, never pixels. Conversion uses `pixel_scale_arcsec`, which must equal
`cutout_arcsec / width`.

Geometry is **image-relative on purpose**. Lens modelling consumes image-centred offsets, so
converting to sky coordinates at rest would lose the frame the modeller works in. Sky position is
recorded **once per image**, in `image.wcs`, and derived per mark by exporters when needed.

Orientation is declared by `north_up` and `east_left`. When both are true, north is toward smaller
`v` and east toward smaller `u`, so north-east is the upper **left**.

## 4. Marks

A mark is one statement about one place in the image. Every mark carries:

| field | meaning |
|---|---|
| `id` | stable within the document |
| `type` | `vector`, `point`, `circle`, or `polygon` |
| `role` | a term from the profile's `role` scheme: what the feature is |
| `polarity` | `core:positive`, `core:negative`, or `core:ambiguous`: what the mark does to the interpretation. Only on roles that take it — see below |
| `alternative` | required when polarity is not positive: the competing reading |

Roles and polarity are **orthogonal in meaning but not independent in applicability**.
`role: lens:arc` with `polarity: core:negative` and `alternative: lens:spiral_arm` states that an
arc-shaped feature is a spiral arm. That statement is first-class; it is not encoded in a label.

**Polarity applies only to roles that can bear on the interpretation.** Each role in a profile
declares `takes_polarity`. Where it is true, `polarity` is required. Where it is false — a field
galaxy to be masked, a star, a detector artefact, a measured ring — `polarity` **must be absent**,
because such a mark asserts that something is there and to be treated a certain way, and says
nothing for or against the interpretation. Demanding a value there would make one of the values mean
two different things, and would apply it to the majority of marks in a typical annotation.

### 4.1 Three geometry families, and why the distinction matters

| family | types | is size meaningful? |
|---|---|---|
| **designating** | `vector`, `point` | no — a pointer's length is presentational |
| **extent** | `circle` | **yes — the radius is the datum** |
| **region** | `polygon` | **yes — the vertices are the datum** |

This distinction governs the emphasis rule in §7 and the render rules. A `circle`'s
`radius_arcsec` is a measurement; nothing presentational may alter it.

### 4.2 Reference rather than duplication

A circle may carry `center_ref` instead of `center`, naming the mark whose position it tracks. An
Einstein ring centred on a deflector mark uses `center_ref`, so the two **cannot drift apart**. A
dangling `center_ref` is a validation error, not a warning.

The same principle governs measurements: **a measured quantity is recorded in exactly one place.**
The Einstein radius lives in `system.theta_e`. A ring mark carries geometry; it does not carry a
second copy of the measurement.

## 5. The system block

One per document: the claim being made about the image as a whole — a verdict, a graded confidence,
the measured quantity with its bounds and coded estimation method, the search status for expected
counterparts, and free text for people.

Three-valued search status is deliberate. "Not found" must be distinguishable from "did not look",
and in a structured-output schema a null and an absent key are the same thing to many producers, so
absence is an explicit term rather than a missing field.

## 6. Portrayal is not in the record

The record says what is true. A separate, versioned **style document** says how it is drawn, and is
referenced by id and hash. Two consequences, both intended:

1. **No field of the record may carry meaning through its appearance.** A colour cannot mean a role.
   Deleting every presentational value from a document must not change what the document says.
2. A style document can be revised — thinner strokes, a different palette — without touching a
   single record, and a record renders identically under a pinned style forever.

## 7. Emphasis

`emphasis` is `core:muted`, `core:normal` or `core:key`, and it is **presentational**.

Every drawn scalar is classified:

| PRESENTATIONAL — emphasis scales these | METRIC — emphasis never touches these |
|---|---|
| stroke width, glyph size, label size | circle radius |
| casing width, dot radius and spacing | polygon vertices |
| layer alpha, pointer tail position | pointer head position |

A ring at `core:key` is drawn with heavier, closer-spaced dots and a stronger casing. **Its radius
does not change.** Emphasis scales the stroke, never the path.

Two constraints follow. A dotted circle's dot radius is capped at `0.08 × radius`, because
oversized dots bias the radius a reader measures off the ink. And emphasis is part of the record, so
it is inside the content hash and changes the render — which is correct, and is why a viewer's
transient highlight must never be written to the record.

## 8. Provenance and review

Every mark records what produced it and, when reviewed, a verdict from the review vocabulary with
the magnitude of any correction. This is what makes a propose-then-correct workflow measurable
rather than anecdotal: the disagreement between a proposal and a reviewer is data, and it is
retained even when the mark is rejected.

## 9. Free text

Free text exists for people. `system.description` and per-mark `rationale` are prose fields.

**Every other field is a term or a number.** This is a structural commitment, not a convention: it
makes "remove all free text" a closed allow-list over keys rather than a judgement about content. A
consumer that must not receive prose can therefore be given a provably clean document.

## 10. Determinism

The same record and the same style document produce a byte-identical render. The record's content
hash covers everything except the render block. A render records the hash of the record it was made
from, so a stale render is detectable rather than merely suspected.

## 11. Profiles

The core defines geometry, coordinates, provenance, portrayal separation and the render contract. It
defines **no domain terms**.

A profile supplies vocabulary. Terms are **prefixed strings** — `lens:arc`, `core:negative` — so a
reader that does not implement a profile can tell that a term belongs to one, and skip the mark,
rather than failing validation. A bare enum cannot express "not mine"; it can only be invalid.

Adding a profile is a vocabulary release. It requires no change to this document.

## 12. Conformance

A conforming producer:

1. emits geometry in normalized coordinates and sizes in arcseconds;
2. uses only terms defined in the core and in a declared profile;
3. supplies `alternative` whenever polarity is not positive;
4. records a measured quantity in exactly one place;
5. writes no meaning into any presentational field.

A conforming consumer:

6. resolves `center_ref` rather than assuming a coordinate;
7. treats an unknown profile prefix as skippable, not invalid;
8. never alters a metric value when rendering emphasis.
