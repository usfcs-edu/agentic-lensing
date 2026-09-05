# Prior art: computer vision and web standards

## Computer vision

The domain closest in mechanics to LensMark, and therefore most useful for where it falls short of a
*scientific record* rather than a training set.

**LVIS** scopes exhaustiveness explicitly, with two arrays per image: categories confirmed **absent**,
and categories **present but not exhaustively annotated** — with unknown as the default third state
that is never scored. This is the direct analogue of the counter-image search status, and it is the
mechanism that lets a corpus be honest about what it did not look at.

**Open Images** carries three ideas worth taking:
- every qualifier is a **trit** — present / absent / unknown, encoded 1 / 0 / −1 — so a negative
  statement is a first-class row with the same weight as a positive one;
- raw disagreement is stored rather than a consensus;
- **`IsDepiction`** marks that an object has the *appearance* of a term without being an instance of
  it. That is exactly the spiral-arm-that-looks-like-an-arc case, and it argues for encoding it as a
  structured pair — `{appearance_of: arc, is: spiral_arm}` — rather than as a negated label string.

**Label Studio** splits geometry from semantics into separate entries **in one flat array**, linked by
a shared id, with a field naming which control produced each statement. And it makes grouping a
**typed, directed edge record** living in that same array — `{type: relation, from_id, to_id,
labels: [...]}` — rather than a scalar group id on each mark. Directed typed edges express
"counter-image *of*" in a way a shared tag cannot.

**LabelMe**, from 2005, already had non-destructive rejection: `<deleted>0</deleted>` and
`<verified>0</verified>` as flags rather than removal, with author and time on the mark *and* on the
geometry separately.

**CVAT/Datumaro** keeps one lossless core document, generates ~26 export formats from it, and
publishes a machine-readable table of exactly what each projection loses. That is the honest version
of what `exports/` already does informally.

**The universal weakness:** none of these formats has a place for uncertainty, for a reviewer's
disagreement, or for why a negative is a negative. They are training-set formats, and a scientific
record needs more.

## Web and geospatial standards

**OGC Symbology Encoding / SLD** is the purest statement of record-portrayal separation: the style is
one separately-versioned rule set where each rule is a *predicate over the data's own coded fields*
plus a drawing instruction. Data files carry no style at all.

Its warning is equally clear: SE requires every element to be an expression, and leaves those
expressions untyped, which makes it verbose to emit, hard to validate, and effectively unusable by a
generative client. **Take the architecture, and make the mapping a flat lookup table.**

**W3C Web Annotation** contributes two things beyond the body/target model:

- **Three typed provenance fields, not one author string**: `creator` (the agent whose intent the
  claim expresses), `generator` (the software that produced the record), and `renderedVia` (the tool
  that drew it). A model proposal and a human proposal are then the same shape with a different
  `creator` type — which is what makes propose-then-correct measurable.
- **Grouping with a three-way distinction**: *Composite* (these marks jointly constitute the claim),
  *List* (ordered), *Independents* (each stands alone). "These four marks are images of one source" is
  a Composite; that is a different assertion from a mere tag.

**GeoJSON** contributes a negative result and a positive discipline. The negative: it has **no circle
primitive**, which disqualifies it here since nine of twenty marks are circles whose radius is the
datum. The positive: RFC 7946 §7 closes the type vocabulary absolutely — implementations MUST NOT
extend the fixed set — and publishes a numbered example corpus as a primary artifact. A closed type
list plus worked examples is why GeoJSON interoperates and SE does not.

**SKOS** supplies the field discipline for the vocabulary file: a stable opaque identity that never
changes, a `notation` (the short code that is actually serialised), a `prefLabel` for display,
`altLabel` for synonyms, a definition, and broader/narrower relations. The draft vocabulary follows
this shape.
