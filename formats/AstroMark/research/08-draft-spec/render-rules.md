# AstroMark render rules, version 1.0 — draft

**Status: draft for review. Not adopted.**

How an AstroMark record is drawn. These rules live in a versioned **style document**, separate from
the record, so the notation can be revised without touching a single annotation and a record renders
identically under a pinned style forever.

Numbers are fractions of `m = min(width, height)` of the **output** image unless stated otherwise.

## 1. Output geometry

The canonical render is **2× the cutout**, with the base upsampled by nearest-neighbour — astronomical
pixels are never interpolated — and the overlay drawn at 3× supersample of that, then downsampled
with a Lanczos filter and composited.

The 2× factor is forced by arithmetic, not preference. A stroke narrower than about **1.25 output
pixels** aliases into a grey smear indistinguishable from noise. At the native size of a typical
cutout, three elements of this notation fall below that floor:

| element | fraction of m | at 403 px | at 806 px |
|---|---:|---:|---:|
| bound-ring dot radius | 0.0016 | 0.65 px ✗ | 1.3 px ✓ |
| Einstein-ring dot radius | 0.0022 | 0.9 px ✗ | 1.8 px ✓ |
| leader hairline | 0.0022 | 0.9 px ✗ | 1.8 px ✓ |

Any proposal to thin the notation at 1× is unsound for this reason.

## 2. The territory rule

> **No ink inside the image rectangle that is not about a specific location in that image.**

This is the governing constraint, and it is derived from measurement: in the current renders the
on-image legend plate occupies 3.7–6.6% of the panel, which is roughly half the total ink and, in
most cases, more than every mark combined.

Consequences:

1. **Identification is in-place.** Every mark that needs naming carries its label near the mark. A
   label is a role name, never a colour name.
2. **The key is a caption band below the image**, costing zero image pixels. Canonical output is
   `width × (height + caption)`.
3. **The notation key is printed once per sheet**, not once per panel. Nine copies of a constant key
   is the failure mode to avoid.
4. **The one exception is the scale bar**, which may sit inside the image because a length comparison
   requires physical adjacency to the pixels being compared. A compass does not qualify: it is a
   property of the whole rectangle and belongs in the band.

## 3. Stroke widths and casing

| element | fraction of m |
|---|---:|
| pointer shaft, primary stroke | 0.0040 |
| glyph outline | 0.0038 |
| measured circle | 0.0036 |
| leader hairline, anchor | 0.0022 |
| region outline | 0.0028 |
| attention bracket | 0.0030 |

Every stroke is drawn twice: a **casing** pass at `w + 2 × 0.0022·m` in a contrasting ink at moderate
alpha, then the stroke itself. Over sky the casing is nearly invisible; over a bright core it
supplies the contrast a bare thin stroke lacks. This replaces a hard, opaque text halo, which is
legible but heavy.

Dotted circles take a one-pixel casing only. A full casing on a dotted ring doubles its apparent ink
and makes it read as a dashed line.

**Casing colour is resolved at authoring time**, not at render time, and written into the record as
`dark`, `light` or a resolved value. A renderer that samples the base image to choose is no longer a
pure function of its inputs, and the same record over a re-stretched image would produce different
ink.

## 4. Keep-clear

No mark may cover the pixels it refers to.

- A pointer's tip stops `0.010·m` short of its target; the terminator extends *backwards* from
  there.
- A measured circle rides the boundary of what it encloses.
- A region polygon is outline-only, or filled at no more than 6% alpha.

## 5. Line style carries polarity

| polarity | shaft | terminator |
|---|---|---|
| positive | solid | as the role family specifies |
| **negative** | long dash, 0.020 on / 0.010 off | plus a cross-bar struck through it |
| **ambiguous** | dotted, 0.0024 dot / 0.0055 spacing | drawn hollow where the positive form is filled |

Three numeric rules govern the texture channel, checked when a style document is released:

1. any two textures used together must differ in **period by at least 2.0×** — a full octave;
2. every period must be **at least 3× the stroke width**;
3. every texture must survive the rasterisation floor at the smallest intended output size.

Rule 1 is the one most easily violated by intuition: a dashed circle and a dotted circle that look
different at full size can converge at thumbnail scale if their periods are close. The existing
dashed-galaxy versus dotted-star convention should be measured against it rather than assumed.

Negative gets two redundant channels: it is the rarest mark and the most consequential.

**Polarity is carried by line style: texture is the channel that survives downscaling best.**
Measured discriminability thresholds, for a 0.030·m glyph:

| channel | minimum panel |
|---|---:|
| stroke texture | **167 px** |
| orientation, 30° steps | 235 px |
| glyph shape | 300 px |
| shape family only | 200 px |
| text | **367 px** |

At 260 px per panel — an ordinary contact sheet — text is unreadable. Any distinction that exists
only in a label has vanished at that size.

## 6. Colour is redundant

Role family is carried by **glyph shape**, polarity by **stroke texture**. Colour reinforces and
never carries alone (WCAG 2.2 SC 1.4.1).

Palette: Okabe-Ito. Lens mass `#009E73`, lensed light `#56B4E9`, obstruction `#CC79A7`, field
`#D55E00`, modelled-not-masked `#E69F00`, measurement `#F0E442`, bounds `#BFBFBF`.

**The conformance test is mechanical**, not a review step: for every pair of marks that can share a
panel, if their inks are closer than ΔE2000 = 15 under normal vision, deuteranopia, protanopia,
tritanopia or greyscale, then their glyph shape, stroke texture or orientation must differ. The
floor is 15 rather than 12 since pure red against pure green still measures 12.9 under
deuteranopia — nearly all of it a lightness difference — and a lower floor would pass the exact
confusion the rule exists to catch.

## 7. Emphasis

Presentational multipliers only:

| | stroke | glyph | label | dot | layer alpha |
|---|---:|---:|---:|---:|---:|
| `core:muted` | ×0.75 | ×0.85 | ×0.85 | ×0.80 | 0.55 |
| `core:normal` | ×1.00 | ×1.00 | ×1.00 | ×1.00 | 1.00 |
| `core:key` | ×1.60 | ×1.25 | ×1.08 | ×1.40 | 1.00 |

**Three levels, and exactly three.** This is a bound, not a preference. At a 3:1 contrast ratio
between adjacent grades, the achievable relative luminances between black and white are 0.000, 0.100
and 0.400; a fourth grade would need a luminance of 1.30, which does not exist. An ordinal emphasis
channel carried by lightness therefore cannot have four values.

Never applied: a circle's radius, a polygon's vertices, a pointer's head position.

**Guard rail.** A dotted circle's dot radius is capped at `0.08 × radius`, because oversized dots
bias the radius a reader measures off the ink. A small circle therefore cannot be emphasised past a
point, which is correct.

**Attention mark.** At `core:key` only, four corner brackets forming a square around the feature.
Not a ring: circles are already spent on measurement here, and a new circle reads as another
measured radius. The square's size derives from the feature it wraps, so it introduces no second
free size channel.

**Compositing.** Each emphasis level draws into its own sublayer at full alpha; the layer's alpha is
scaled once and the layers composited bottom-up. Per-stroke alpha would let two overlapping muted
strokes double-composite and darken — an order-dependent defect that is difficult to reproduce.

Draw order: labels, then muted, normal and key geometry, then key brackets, then the caption band.
Labels sit *below* geometry so a label can never hide a mark.

## 8. Labels

Sizes: primary 0.023·m, alternative-reading second line 0.019·m, index digit 0.017·m. **Floor: 11
output pixels for any string that must be read.**

Regular weight throughout. Bold is reserved for the single `core:key` label — one reservation makes
bold mean something; universal bold means nothing.

**Placement** is a greedy search: preferred side first, then perpendiculars and diagonals, at
increasing offsets. Candidates are rejected against a **keep-out set** that includes other placed
labels, every pointer head and apex, and — importantly — the **annulus** of every measured circle at
`radius ± 0.012·m`. A label lying on a ring is read as naming a point on that ring.

The keep-out for a circle is its annulus, **not its bounding box**. A label may legitimately sit
inside a large mask circle; it must only avoid the ring itself. Using the bounding box blocks most
placements on a crowded panel and forces needless demotions.

**Deterministic failure.** When no placement is found, the label **demotes to an index digit** at the
mark and its text moves to the caption band. Crowding then degrades predictably instead of producing
overlap. The renderer reports how many labels demoted, so a lint can flag a panel that is carrying
too much.

## 9. Capacity

Guidance, from ink budget and reading effort:

- ink ≤ 3.5% of the panel, hard ceiling 5%
- occlusion of informative pixels below 1.5% where the content allows
- at most 5 labelled marks nominal, 7 hard — beyond that, demote to indices
- texture marks (masks, rings) are read as a set, not individually; ~25 is legible
- at most 2 region polygons
- at most 6 distinct glyph types visible at once
- two labelled marks closer than 0.09·m cannot both carry a full label

## 10. Determinism and traceability

Same record plus same style document produces byte-identical output. No background sampling at
render time, no system fonts, no ordering that depends on a hash map's iteration.

**The render carries its own provenance.** Alongside the pixels, the image file embeds, in a text
chunk:

| key | value |
|---|---|
| `astromark.record` | the content hash of the record that produced this render |
| `astromark.style` | the id and version of the style document used |
| `astromark.renderer` | the renderer version |
| `astromark.ref` | *optional* — a resolver reference for fetching the record |

This is what stops a flattened render from being an orphan. A figure lifted into a talk, a paper or a
message can still be traced to the document, the style and the code that made it.

It does not compromise determinism: every embedded value derives from the inputs rather than from the
output, so the same record under the same style still produces identical bytes. Note the one
practical consequence — a renderer that currently strips **all** metadata for byte-stability must
switch to writing exactly this set and nothing else, and existing golden hashes are re-pinned once.

Two assertions belong in the test suite:

1. A record with no `emphasis` renders byte-identical to one with `core:normal` throughout, so
   adding the field does not invalidate existing renders.
2. The radius recovered from the rendered ink is identical across all three emphasis levels to
   within 0.1%. This is what catches a presentational multiplier leaking into a metric path.
