# The metadata representation: six proposals, and the axis that actually decides

**Status:** option space designed, all six encoded and measured, 2026-09-03.
Reproduce with `examples/make_encodings.py`; the encodings are in `examples/encodings/`.

---

## 0. The finding that reframes the question

I expected this to be a choice between containers — bespoke JSON, GeoJSON, Web Annotation, tables.
Auditing what actually went wrong with `lensmark/1.0` says otherwise. Here is every structural
problem in the current format, and what each one really is:

| Audit finding | What it actually is |
|---|---|
| Enums defined in six uncoordinated places | term identity, unowned |
| θ_E stored twice and unlinked | identity of a *measured quantity* |
| `theta_e.method` a free string despite three named rules | a term with no vocabulary |
| green = deflector; `mask_red` reserved by item type | a term smuggled into portrayal |
| prose refers to arrows *by colour* | a term with no referenceable id |
| the whole style table copied into every data file | portrayal had no identity, so it was inlined |
| no `role` field — semantics in a 40-char label, matched by substring | a term hidden in free text |
| no polarity | a missing term |
| no polygon, no secondary ring, no source grouping | missing terms, and a missing grouping term |

**Nine for nine, they are vocabulary-identity problems. Not one would have been prevented by
GeoJSON, by Web Annotation, or by tables.**

The sharpest instance is worth seeing in full. This is how the current system decides what is a lens
galaxy:

```python
# lensmark/validate.py:75
def _is_deflector(label): return bool(label) and "deflector" in label.lower()
```

mirrored independently in `frontend/src/geometry.ts`. A label reading `spiral arm, NOT a deflector`
is therefore coloured green — the colour that means lens mass. The semantics of the standard ride on
substring-matching an optional 40-character string, in two languages, with no shared definition.

So the axis that decides this is not the container:

> **Where does semantic identity live — in a separately versioned vocabulary artifact that is the
> single generator of every other surface, or in the shape of the document?**

Choose the container for cheapness. Choose the vocabulary machinery for rigour.

---

## 1. The six proposals

Each is encoded in full in `examples/encodings/`, all saying the same thing about the same image.

### P1 — Evolved bespoke
Keep the discriminated union on `type`. Add semantics as *orthogonal* fields (`role`, `polarity`,
`alternative`, `source`) rather than new types. Hoist the style table and the vocabulary out into
separately versioned side documents referenced by hash.

```json
{"id": "m-mimic", "type": "vector", "role": "lens:arc", "polarity": "negative",
 "alternative": "lens:spiral_arm",
 "tail": [0.851, 0.358], "head": [0.739, 0.282]}
```

**Easy:** smallest diff from what exists — the renderer, frontend and all four exporters change by a
field lookup, not a rewrite; golden render tests keep working; humans already read this shape.
**Hard:** every new geometry is a new class in three or more places with no generator, so it
inherits the exact failure being fixed unless code generation is added; the core/profile boundary is
a naming convention, not a mechanism. **Failure mode:** it silently reverts to lens-only, and the
third enum copy reappears the first time the frontend needs a `role` dropdown.

### P2 — GeoJSON-shaped
`FeatureCollection`; the core owns `geometry`, the profile owns `properties`. The boundary is
structural rather than conventional, which is genuinely attractive.

**It fails on a fact about this data.** GeoJSON has no circle, and **nine of the twenty marks are
circles whose radius IS the datum** — mask radii and θ_E. The options are a 64-gon, which destroys
the record (the exporters need `radius_arcsec`, not vertices), or `Point` + `properties.radius`, at
which point the geometry member no longer holds the geometry:

```json
{"type": "Feature", "id": "m-ring",
 "geometry": {"type": "Point", "coordinates": [0.5, 0.51]},
 "properties": {"radius_arcsec": 1.45,
                "_geojson_note": "radius is not representable in GeoJSON geometry"}}
```

RFC 7946 also mandates WGS84 lon/lat and deliberately removed the `crs` member, so using it for
normalized image coordinates is a knowing violation. And measured: **it is the largest encoding of
the six.** You pay the standard's tax and get none of its tooling. **Failure mode:** the first
colleague who opens it in QGIS files a bug you answer with "it isn't really GeoJSON."

### P3 — W3C Web Annotation (JSON-LD)
`body` + `target` + `selector`. Review is *native*: a critique is an Annotation whose target is
another Annotation's IRI — exactly the propose→critique→accept loop, expressed without inventing
anything, and it generalises to critiques-of-critiques for free. Every term is a dereferenceable
IRI, so vocabulary versioning and profile membership are solved by the URL.

**Hard:** JSON-LD is open by construction — `@context` remapping, value-or-array polymorphism at
nearly every site. There is no plausible flat, closed JSON Schema for it, so it fails the
structured-output requirement unless you define a sub-profile that is no longer Web Annotation, at
which point the standard's authority — the entire reason to pick it — is gone. Geometry via
`SvgSelector` embeds SVG-in-a-string, which is hostile to `radius_arcsec` (an SVG `r` is in pixels)
and to content hashing. **Failure mode:** correct and unusable; every implementer ships a private
subset no other consumer can read, which is the observed history of most adopters.

### P4 — Markup / annotation split with coded concepts
The DICOM SR and NCI AIM lineage. Geometry (**markup**) carries no meaning at all; meaning
(**observations**) is a set of coded concepts that *reference* markup. Every term is a
`{scheme, code, meaning}` triple where identity is scheme+code and the English string is display
only.

```json
"markup":       [{"id": "k8", "shape": "vector", "points": [[0.851,0.358],[0.739,0.282]]}],
"observations": [{"id": "o-m-mimic", "about": ["k8"],
                  "concept":  {"scheme":"AM-LENS","code":"ARC","meaning":"arc"},
                  "polarity": {"scheme":"AM-CORE","code":"NEG","meaning":"negative"},
                  "because":  {"scheme":"AM-LENS","code":"SPIRAL_ARM","meaning":"spiral arm"}}]
```

**This is the only proposal in which "green means deflector" is structurally impossible**, because
portrayal is not in the record and semantics can only be a code. Vocabulary versioning is genuinely
solved — a scheme versions independently of the schema, so a morphology profile is a vocabulary
release with *zero* core-schema change. And the θ_E double-storage bug is unrepresentable: a
measurement is an observation with value, unit, bounds and coded method, which cannot be duplicated
into a geometry field.

**Hard:** 9.4× the compact form; the model must emit a three-key object where an enum string would
do, with three chances to be inconsistent; two id spaces to track. **Failure mode, honestly:** the
ceremony is abandoned under deadline. Someone adds `"role": "arc"` beside the coded concept "just
for the frontend", and within a release you have both, unlinked — which *is* the θ_E bug, reproduced.

> **The salvage — and the single most valuable idea in this document.** What is worth having is
> *term identity*, not the triple. `"role": "lens:arc"` — one prefixed string resolved in a
> versioned, SKOS-shaped vocabulary file carrying label, definition, profile, status and
> `replaced_by` — keeps token cost at enum level while retaining everything that matters: profile
> membership visible in the term, graceful degradation (an unknown `morph:` prefix is recognisably
> not-mine rather than merely invalid), and vocabulary versioned separately from schema. Call it
> **P4-lite**. It composes with P1, P2 and P6.

### P5 — Dual surface
Canonical JSON plus a normative line notation with enforced round-tripping. Fully covered in
`06-llm-ergonomics.md`; the measured result is 2.1 kB against 14.4 kB, and a 50-example payload of
~30k tokens against ~205k. **It is orthogonal to P1–P4 and P6**, which is itself a finding — this is
not a competing container but a second surface any container can have.

### P6 — Relational / columnar
Not a document: `marks` / `assertions` / `reviews` / `sources` as typed rows, shipped as FITS
BINTABLE, VOTable or Parquet. Astronomy-native — this is how every catalogue in the field ships, and
what an archive accepts without argument. Corpus queries become trivial, and those queries *are* the
research questions: "every counter-image mark the reviewer called wrong_position", "θ_E residual
versus hard-case class", stratified few-shot selection.

**Hard:** hostile to single-document editing and to git; a human cannot read one cutout's annotation
without performing a join in their head; the content hash and deterministic render need a
*document* to hash, so you must define a canonical flattening — which is a document format.
**Failure mode:** you build the document format anyway and the tables become a derived warehouse.
Which is fine — but then P6 was never the record, it was the export. **Its real contribution is as a
criterion:** measure every other candidate against P6's query and degradation story.

### Considered and rejected as primaries

**DS9 region format, extended.** The incumbent, and the adoption precedent the room itself cited. It
has grouping (`tag={}`, multi-valued and many-to-many — genuinely good) and inline portrayal. It is
often said to have polarity, and it does not: the `-` prefix on a region "flags the region with a
boolean NOT for later analysis" — it is **set arithmetic on a pixel mask**, an instruction to exclude
those pixels from an aperture, not an assertion about what the feature is. But it has no provenance,
no review state, no uncertainty, no grouping semantics beyond a string bag — and `exports/ds9.py`
already abuses `tag={id:…}` to smuggle identity through it. Extending it means forking a format you
do not control whose reference parser is C inside SAOimageDS9. **Keep it as an export — it already
is one, at 161 lines — never as the record.** It does prove that a line-oriented notation is
culturally acceptable in this field, which is evidence for P5.

**RDF/OWL, ontology-first.** Right for vocabulary governance, wrong for everything else: no flat
schema, no token budget, no human-editable file. **Steal the SKOS shape** — a term has `prefLabel`,
`definition`, `broader`, `status`, `deprecated`, `replacedBy` — as a plain JSON vocabulary file.
That artifact is P4-lite's backbone and is compatible with all six proposals.

---

## 2. Measured size

| Encoding | chars | ×smallest | 50-example payload (derived) |
|---|---:|---:|---:|
| P5 model-read surface | 2,091 | 1.0 | ~30k tokens |
| P6 relational | 8,089 | 3.9 | ~116k |
| P1 evolved bespoke | 14,362 | 6.9 | ~205k |
| P3 Web Annotation | 17,777 | 8.5 | ~254k |
| P4 coded concepts | 19,678 | 9.4 | ~281k |
| P2 GeoJSON-shaped | 19,762 | 9.5 | ~282k |

Character counts exact; token figures derived at 3.5 chars/token, not measured.

---

## 3. How to decide, if the team wants to decide by evidence

**Four gates. Fail one and the candidate is out regardless of its other merits.**

| gate | test |
|---|---|
| **G1** flat, closed structured-output schema | 0 `$ref`, `additionalProperties:false` everywhere, honoured by a live API call |
| **G2** round-trip fidelity | convert the 19 real decks out and back; the result must re-render **byte-identical** to the nine golden SHA-256s |
| **G3** embargo separability | `strip_free_text` provable by a **closed key allow-list**, not a type-sniffing walk |
| **G4** expressiveness | all 22 fixture statements expressible; the 16 P0 statements as first-class constructs |

Predicted casualties, recorded before the gates are run: **P3 fails G1** (no flat closed schema is
possible without ceasing to be Web Annotation); **P2 fails G2** (circles are the majority of marks
and GeoJSON cannot hold a radius in its geometry).

**No weighted sum.** Report the matrix and decide by dominance; where nothing dominates, name the
trade in a sentence and take it to the room as a decision rather than a number. A weighted sum here
would only encode the weighter's prior with false precision.

### The 22 statements the notation must be able to make

The first sixteen are P0 — they must be first-class constructs, not conventions.

1 main deflector · 2 second deflector · 3 satellite (lens mass, never masked) · 4 arc of source S1 ·
5 counter-image of S1, **farther out than the arc** · 6 both arcs lensed but unpairable ·
7 two sources · 8 **NOT an arc — a spiral arm** · 9 ambiguous: counter-image or projection ·
10 θ_E = 1.45″ nominal, ≥1.15, ≤1.80, by arc-midline · 11 the ring is centred on the deflector
*mark* and follows it · 12 a secondary ring from a second deflector · 13 a dust lane, a hard case
and **not** evidence against · 14 mask this galaxy, **model** that one · 15 lens-light versus
lensed-light polygons · 16 **searched** for a counter-image and did not find one (≠ did not look) ·
17 provenance and reviewer verdict · 18 this mark is an edit of that one · 19 two hard-case classes
at once · 20 the counter-image's colour disagrees · 21 sky position and PA · 22 **everything above,
with all free text removed, still says 1–16 and 19** ← the embargo test, and the sharpest
discriminator in the list.

The current format scores zero on statements 2, 6, 7, 8, 9, 12, 15 and 16.

---

## 4. Recommendation

**Container: P1 evolved. Identity: P4-lite. Read surface: P5. Export: P6 and DS9 unchanged.**

Concretely:

- a **vocabulary file is built first and is the single generator** of the pydantic Literals, the
  JSON Schema enums, the TypeScript unions, the prompt's vocabulary section and the docs table, with
  a CI check that fails on staleness;
- terms are **prefixed strings** — `lens:arc`, `core:negative` — not bare enums and not triples;
- **portrayal leaves the data file** and becomes a versioned, hash-referenced style document;
- models **read** the line form and **write** JSON under a flat closed schema.

The reasoning is that all six containers would work, and what killed `lensmark/1.0` was not a wrong
choice but **erosion** — nothing prevented drift. So the axis that matters is the one where a
mechanism can be installed. That mechanism is compatible with every proposal here, and it is worth
more than the choice among them.

**One thing to do regardless of which container the team picks:** build the vocabulary file. It is
the only artifact all six proposals share, and the only one whose absence caused every problem in
the audit. Building it before the container is chosen is not premature — it is the part that is not
in dispute.
