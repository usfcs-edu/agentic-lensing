# Executive summary

**AstroMark: the symbolic notation and the metadata format.** Research, proposals and a draft spec,
for the team to choose from. 2026-09-03.

**Nothing here is adopted.** Recommendations are recommendations; the measurements are real and
reproducible; the gaps are named rather than smoothed over.

---

## The recommendation in one table

| | Pick | Why |
|---|---|---|
| Notation | **N1 Terminator alphabet**, with **N4 Evidence graph** prototyped beside it | N1 keeps the arrow and costs least to adopt; N4 is the only one that fixes the defect that started this |
| Container | **P1 evolved bespoke** | the cheapest decision; converters are 57–161 lines |
| Identity | **prefixed terms in one versioned vocabulary file that generates every other surface** | this is where the real problem is |
| Model surfaces | **read** the compact line form, **write** JSON under a flat closed schema | the only arrangement that satisfies both the schema requirement and the token budget |
| Portrayal | **out of the record**, into a versioned style document | so no field can carry meaning through its appearance |
| First thing to build | **the vocabulary file** | the only artifact every proposal shares |

---

## Four findings that changed the design

### 1. The container barely matters; the vocabulary is everything

Every structural problem in the current format is a **vocabulary-identity** problem: the enums are
defined in six uncoordinated places, θ_E is stored twice and kept in sync by a prompt rule, the
estimation method is a free string, and what counts as a lens galaxy is decided by

```python
def _is_deflector(label): return bool(label) and "deflector" in label.lower()
```

duplicated in Python and TypeScript — so a label reading `spiral arm, NOT a deflector` renders in the
colour that means lens mass. **Not one of these would have been prevented by GeoJSON, by Web
Annotation, or by tables.** Choose the container for cheapness; put the rigour in the vocabulary,
where a CI check can hold it.

### 2. "Too heavy" was not about stroke width

The on-image legend plate is **3.7–6.6% of the panel — roughly half the total ink, and in most decks
more than every arrow, circle, ring and label combined.** The annotations the room liked best are
3–8× thinner inside the image mainly because they moved the key *below* it.

That produces one rule that settles the apparent contradiction in the record between *"labels go on
the object"* and *"the legend obscured the image"* — both true, and about different things:

> **No ink inside the image rectangle that is not about a specific location in that image.**

Normalised for how much each notation actually says, **today's costs 3.2× more ink per unit of
meaning than the best candidate, while expressing five fewer things.**

### 3. Emphasis is a partition, not a multiplier

Greg asked for marks that can be resized to draw the eye. The obstacle is that some marks encode a
measurement in their size — the ring radius *is* θ_E. So every drawn scalar is tagged **METRIC** or
**PRESENTATIONAL**, and emphasis scales only the presentational ones. A ring at `key` gets heavier,
closer-spaced dots and a stronger casing while its radius does not move a pixel.

**Emphasise the stroke, never the path.**

### 4. Accessibility is testable, and the test is worth having

Of 136 pairs of marks that can share a panel, **123 — ninety percent — are not reliably separable by
colour** under normal vision, deuteranopia, protanopia, tritanopia or greyscale. **Zero are separated
by colour alone**, because role is carried by glyph shape and polarity by stroke texture.

The floor had to be calibrated rather than guessed: pure red against pure green still measures
ΔE2000 = 12.9 under deuteranopia, almost all of it *lightness*, so a floor at 12 would wave through
the canonical confusion. It is set at 15.

---

## What exists now

**Working code**, all deterministic, in `examples/`:
a synthetic reference cutout containing every hard case the notation must express (dust lane, second
deflector, bound satellite, a counter-image *farther out* than the arc, a spiral arm that mimics an
arc); a prototype renderer implementing all four notations plus two reference arms and a null arm;
colour-vision simulation with CIEDE2000 verified against the Sharma et al. reference pairs; the
measurement harness; and an embargo checker.

**Figures:** every arm rendered, contact sheets at three sizes, a polarity triad with all labels
removed, and CVD sheets.

**A draft spec** in `08-draft-spec/`: core, lens profile, render rules, an 80-term vocabulary across
10 schemes, and a generator that emits the JSON Schema, pydantic Literals, TypeScript unions and
documentation *from that one file*, with `--check` as the staleness gate. The schema is flat
(0 `$ref`) and closed, and the reference annotation validates against it.

---

## Confidence

**Measured and reproducible:** ink, occlusion, contrast, encoding sizes in characters, the colour
gate, the reference arms against real LensMark renders (R-CURRENT measures 6.23% against a real
6.4–10.8%; R-CAMPAIGN 1.99% against a real 1.2–2.3%).

**Derived:** token counts. No Claude tokenizer offline; character ratios are exact and are what the
decision turns on.

**Untested:** whether image plus coordinate metadata teaches a model better than an annotated PNG.
This is Greg's hypothesis on the record, the format is partly designed around it, and it should be
run **before** the format freezes. The experiment is cheap now that all the encodings exist.

**Judgement, declared as such:** N1 versus N4. The measurements rule out the status quo; they do not
choose between two good candidates.

---

## Six decisions for the review

1. Notation: **N1**, **N4**, or N1 with N4's source chord grafted in?
2. Container: accept **P1 plus prefixed-term identity**, or spend the argument on the container?
3. Do the six **notation-independent changes** land regardless? *(2× render; METRIC/PRESENTATIONAL
   tagging; the emphasis enum; annulus keep-out in the label solver; demotion-to-index; the caption
   band.)* I would say yes.
4. Is the **read/write asymmetry** acceptable, given it means two surfaces and a round-trip test?
5. Does **polarity need a fourth, neutral value** for a dust lane — a real feature that neither
   supports nor refutes? *(This surfaced while building the reference annotation and is not in the
   workshop record.)*
6. Who runs the **few-shot experiment**, and does the freeze wait for it?

Start with the vocabulary file either way. It is the part that is not in dispute.
