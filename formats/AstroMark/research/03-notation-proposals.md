# The symbolic notation: four proposals, measured

**Status:** four working prototypes, rendered and measured, 2026-09-03.
Reproduce with `examples/render_proposals.py` then `examples/measure.py all`.

Every figure referenced here renders the **same twenty marks** from one neutral content file
(`examples/content.json`), compiled to each notation. The sheets therefore compare notations rather
than annotators — which is the discipline that makes any of the numbers below mean anything.

---

## 0. What the measurements changed

I expected "the annotation is too heavy" to be about stroke width. It is not.

| | ink over the panel | of which, the on-image legend plate |
|---|---:|---:|
| LensMark today, deck-01 | 7.7% | **4.6%** |
| deck-04 | 10.8% | **6.6%** |
| deck-08 | 9.7% | **5.3%** |
| deck-07 (legend off) | 9.0% | 0% |
| The agent campaign the room liked | **1.2–2.3%** | 0% — its key sits *below* the image |

**The legend plate is roughly half the total ink, and in most decks it is more than every arrow,
circle, ring and label combined.** Note deck-07: thirty items and no plate, and it still lands below
the eleven-item decks that have one.

That single measurement produces the rule that organises everything else:

> ### No ink inside the image rectangle that is not about a specific location in that image.

This resolves the apparent contradiction in the record — *"labels go on the object"* and *"the
legend obscured the image"* are both true, and they are about different things. Identification
belongs on the object; the key belongs in a caption band **below** the image, where it costs zero
image pixels. The notation key is printed once per *sheet*, not once per panel; the real sin of the
campaign's legend was not that it existed but that it repeated nine identical copies of information
that was constant across the set.

One principled exception: **the scale bar may sit inside the image**, because a length comparison
requires physical adjacency to the pixels being compared. A compass does not qualify — it is a
property of the whole rectangle, so it goes in the band.

---

## 1. The shared substrate

All four proposals inherit these, so the comparison has one variable.

**The canonical render is 2×.** This is forced by arithmetic, not taste. A stroke must land at
≥ 1.25 output pixels or it aliases into a grey smear indistinguishable from noise. At 403 px the
bound-ring dots (0.65 px), θ_E dots (0.9 px) and leader hairlines (0.9 px) are *all* below that
floor. At 806 px they clear it. Any proposal to thin the notation at 1× is arithmetically unsound.

**Three geometry families, and the distinction is load-bearing everywhere:**

| family | what it asserts | is its size meaningful? |
|---|---|---|
| **designating** | a location — deflector, arc, counter-image, knot, dust lane | no, size is free |
| **extent** | a measurement — mask radius, θ_E and its bounds | **yes, the radius IS the datum** |
| **region** | a boundary — segmentation polygons | **yes, the vertices ARE the datum** |

**Keep-clear.** No mark may cover the pixels it refers to. Pointers stop short of their target,
circles ride the boundary, polygons are outline-only.

**Casing, not halo.** Every stroke is drawn twice — a casing pass in a contrasting ink at moderate
alpha, then the stroke. This is what replaces the current hard black text halo, which is half of the
"blocky" complaint on the record.

---

## 2. The emphasis mechanic

Greg asked for a way to resize marks to draw the eye. The difficulty is that **some marks encode a
measurement in their size** — the ring radius *is* θ_E — so scaling for emphasis would corrupt data.

The answer is a partition, not a multiplier. Every drawn scalar is tagged:

| PRESENTATIONAL — emphasis scales these | METRIC — emphasis never touches these |
|---|---|
| stroke width, glyph size, label size and face | circle radius (mask, θ_E, bounds) |
| casing width, dot radius and spacing | polygon vertices |
| layer alpha, shaft length, tail position | pointer head position |

So a θ_E ring at `key` gets heavier, closer-spaced dots and a stronger casing — it visibly shouts —
while **its radius does not move by one pixel**. Emphasise the stroke, never the path.

Design decisions inside that, each with a reason:

- **`emphasis: muted | normal | key`, an ordinal enum, not a float.** A float invites a model to
  emit `1.37`, and makes cross-panel comparison meaningless. Three steps is also about the limit of
  what a reader can rank reliably.
- **The attention mark is corner brackets, not a halo ring.** This notation has already spent
  circles on measurement; a new circle around a feature reads as *another measured radius*. A square
  never does. Its size derives from the feature it wraps, so it is not a second free size channel.
- **A guard rail:** `dot_r ≤ 0.08 × r_metric`. Fattening a dotted circle's dots biases the radius a
  reader measures off the ink; the cap bounds that. A small mask circle therefore *cannot* be
  emphasised past a point, which is correct — a 7 px circle is not what you want the eye on.
- **Per-layer alpha, never per-stroke.** Two overlapping muted strokes at per-stroke alpha
  double-composite and darken — an order-dependent bug that is painful to reproduce. Each emphasis
  level gets its own sublayer, drawn at full alpha, with the layer's alpha scaled once.
- **Emphasis is a document field, so it is inside the content hash.** Changing it changes the
  render, which is correct and consistent. A viewer-side hover highlight, if ever wanted, lives in
  the browser preview and never touches the canonical PNG.

Two golden assertions come with it: `emphasis` absent must render byte-identical to
`emphasis: "normal"` (or every existing deck's golden breaks), and the radius recovered from the ink
must be identical across all three levels to 0.1% (which is what catches an emphasis multiplier
leaking into a metric path).

---

## 3. The four proposals

### N1 — Terminator alphabet
*Keep the pointer; move all semantic weight into the shape at the business end.*

The arrow is the one mark the room has consistently endorsed, so N1 keeps it and thins everything
else. Semantics live in the **terminator**, inscribed in a circle of constant silhouette area so no
role looks heavier than another.

- **Lens mass — filled silhouettes** ("mass is solid"): deflector a filled triangle; second
  deflector the same with a notched base; satellite a smaller triangle.
- **Lensed light — open, curved forms** ("light is thin and curved"): arc a crescent tick; lensed
  image an open chevron; counter-image a nested double chevron ("the other one"); knot a small open
  circle.
- **Obstruction — bars** ("a bar is a wall, not an arrow"): dust lane a double bar.

Polarity is the shaft texture — solid, long-dash, dotted — reinforced on negatives by a cross-bar
struck through the terminator. Two channels, because negative is the rarest and most consequential
mark.

**Does well:** highest role resolution of the four; evolutionary, so adoption is cheapest; the
terminator sits where the eye already goes. **Breaks:** below ~9 px the individual shapes collapse
and only the three *families* survive, so the families must be designed as the thumbnail layer;
seventeen shapes is a real vocabulary needing a key card.

### N2 — Bertin ledger
*Assign each visual variable to exactly one semantic dimension; the assignment table is the spec.*

Rather than starting from a shape catalogue, N2 starts from Bertin's seven variables and forbids any
dimension from claiming a second. The mark inventory falls out of the table.

The move that makes it genuinely different: **orientation is derived from the physics.** Each
designating mark is a two-stroke tick whose angle comes from the radius vector to the nearest
lens-mass item — **tangential means lensed light** (which really is tangentially stretched),
**radial means lens mass**. The mark *is* the physics, which makes it mnemonic for astronomers and
cheap to teach.

**Does well:** orientation survives downscaling far better than shape — 235 px per panel versus
300 px — so it is the most thumbnail-robust; the ledger makes "which variable is free?" a one-word
question, which matters when the standard is extended by people who were not in the room.
**Breaks:** orientation is undefined at the deflector itself, where the radius vector is degenerate
(the prototype handles this explicitly — a lens-mass mark is measured from the *primary* deflector,
and the primary falls back to a fixed vertical tick); a tangential tick lying along a tangential arc
can be read as part of it, so it must be offset outward.

### N3 — Station model
*One compact badge per object; one fixation instead of three.*

Modelled on the WMO surface station plot. A badge sits off the object on the ray from panel centre,
joined by a hairline stem, with slots: a role ideogram, a **polarity underline or strike** (one
stroke — this is what makes it cheap), a **source index letter**, and a treatment mark (`×` mask,
`f` fit). A badge appears only when it carries something the mark's own style does not.

**Does well:** lowest ink per object; carries **source membership**, which nothing in the current
system can express at all; it is what a discipline-wide notation actually looks like, which matters
for the stated adoption goal. **Breaks:** badges sit off the object, so the reader must trust the
stem, and stems cross things on crowded panels; it demands literacy — an unbriefed astronomer cannot
read `A ⌒⌒ ×`, which is in direct tension with the room's preference for plain labels on the object.

### N4 — Evidence graph
*Annotate the argument, not the objects. The primitive is a link.*

A strong-lens claim is the assertion "these blobs are images of one source behind that mass". N4
draws that assertion and lets the objects be implied by their participation in it.

- an **anchor** (a small open ring) is the only mark that touches a feature;
- a **source chord** — a hairline arc concentric with the θ_E ring, with radial ties down to its
  anchors — says "these are images of one source, tangential, at this radius" in one stroke;
- a **stem** from each lens-mass anchor to a filled dot at the centre says "this is mass"; two stems
  converging on one dot *draws* a group-scale deflector;
- a **refutation stroke** across a chord says "these are not images of one source, because ⟨…⟩".

Polarity is structural rather than decorative: a positive panel has an unbroken chord, a negative
panel's chord is struck. **From across a room, the non-lens is the panel whose graph is broken.**

**Does well:** the only proposal where negative evidence is a first-class visual object rather than a
styling of a positive one — which is the defect that motivated this whole exercise; lowest ink of
the four; topology survives downscaling when glyph shape does not. **Breaks:** poor role resolution
— it separates lens mass, lensed light, field and obstruction and nothing finer, so knots and second
deflectors need a badge borrowed from N3; it needs a θ_E estimate before a chord can be drawn; it is
the most unfamiliar and has the highest explanation cost.

---

## 4. Measured results

Rendered at 806 px, measured against the un-annotated null arm.

| arm | statements | ink % | **ink/statement** | occlusion % | contrast p5 |
|---|---:|---:|---:|---:|---:|
| N1 Terminator alphabet | 20 | 3.55 | 0.178 | 10.95 | 0.136 |
| N2 Bertin ledger | 20 | 3.07 | 0.154 | 11.12 | 0.137 |
| N3 Station model | 20 | 3.27 | 0.163 | 11.59 | 0.137 |
| **N4 Evidence graph** | 20 | **2.57** | **0.128** | 11.35 | 0.110 |
| R-CURRENT (LensMark today) | 15 | 6.23 | **0.416** | 12.77 | 0.120 |
| R-CAMPAIGN (agent campaign) | 15 | 1.99 | 0.133 | **6.16** | 0.091 |

**Read the ink/statement column, not the ink column.** Raw ink penalises an arm for saying more, and
the two reference arms *cannot express* bound rings, segmentation polygons, treatment, polarity or
source grouping — five of the twenty statements. They look thinner for free.

Normalised, the result is clean: **today's notation costs 3.2× more ink per unit of meaning than N4,
and 2.3× more than the worst candidate.** All four candidates sit in the same band as the campaign
the room liked, while expressing five more things.

Two honest caveats:

- **Occlusion is worse for every candidate (≈11%) than for the campaign (6.2%)** — for the same
  reason. The candidates draw bound rings and two segmentation polygons that necessarily lie over
  informative pixels. This is a real cost of expressing more, not an artefact, and the room should
  decide whether bounds and segmentation earn their occlusion.
- The harness validates against reality: R-CURRENT measures 6.23% against 6.4–10.8% for real
  LensMark renders, and R-CAMPAIGN 1.99% against the campaign's measured 1.2–2.3%. The prototype
  reference arms reproduce the real thing closely enough to trust the comparison.

### Thumbnail thresholds

What survives downscaling, measured against a 0.030·m glyph:

| channel | minimum panel size |
|---|---|
| **stroke texture** (solid / dash / dot) | **167 px** — most robust |
| **orientation** (30° steps) | 235 px |
| shape | 300 px |
| shape *family* only | 200 px |
| **text** | **367 px** |

At 260 px per panel — a 3×3 contact sheet on a laptop — **text is unreadable in every arm**. Only
the visual channel remains. That is the empirical case for putting polarity in texture and role in
shape or orientation, and against any design that relies on reading a label.

---

## 5. Recommendation on the notation

**Prototype N1 and N4 as the two poles; treat N2 as the specification method and N3 as the crowding
fallback.**

- **N1 is the evolutionary pole.** It keeps the arrow, fixes polarity, roles and bounds at the
  lowest adoption cost, and is what ships if the standard has to be presentable at the January AAS
  meeting.
- **N4 is the revolutionary pole.** It is the only one that fixes the actual named defect — a viewer
  flipping through nine panels cannot tell which is the non-lens — at the level of the mark rather
  than the label. Its source chord is the one genuinely new idea in the option space.
- **N2 is not really a fifth notation; it is how to write the spec for whichever wins.** The ledger
  table is what makes the accessibility constraint structural instead of a checklist.
- **N3's source-index slot is worth stealing into whichever wins**, because source membership is
  inexpressible today and costs two characters.

**Six changes are notation-independent** and worth doing whichever proposal is chosen, in this
order: the 2× canonical render → METRIC/PRESENTATIONAL scalar tagging → the emphasis enum with
layered compositing → keep-out geometry in the label solver → demotion-to-index → the caption band
and `legend.mode`. Only the sign inventory is a bet.

*(On keep-out geometry: implementing it exposed a real bug. Using each circle's bounding box rather
than its annulus blocked most label placements and demoted four labels in N2 that had no need to
demote. A label may sit inside a large mask circle; it must only avoid lying **on** the ring, where
it would read as naming a point on it.)*

## 6. Figures

| file | what |
|---|---|
| `examples/reference/n1.png` … `n4.png` | each proposal on the reference scene, with caption band |
| `examples/reference/r-current.png` | today's LensMark notation at its own constants |
| `examples/reference/r-campaign.png` | the agent campaign's style |
| `examples/reference/null.png` | the un-annotated cutout — the arm that asks which marks earn their ink |
| `examples/reference/contact-{528,328,260}.png` | the thumbnail test at three sizes |
| `examples/reference/polarity-triad.png` | positive / negative / ambiguous, **labels removed** |
| `examples/reference/cvd-sheet.png` | every arm under four vision conditions |
| `examples/reference/metrics.json` | the numbers above, machine-readable |
