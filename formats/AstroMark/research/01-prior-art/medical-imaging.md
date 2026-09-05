# Prior art: radiology and medical imaging

Greg asked whether other fields have notations for image markup, "like in radiology". They do, and
medicine solved this problem thirty years ago — thoroughly enough that most of AstroMark's open
questions have a settled answer here. It also contains the single most instructive failure in the
survey.

---

## 1. The coded entry: a term is never a string

The foundational move. In DICOM, every semantic term is a **coded entry** of three mandatory fields
plus an optional version:

| field | tag | role |
|---|---|---|
| Code Value | (0008,0100) | the opaque, stable identity |
| Coding Scheme Designator | (0008,0102) | which vocabulary it comes from |
| Code Meaning | (0008,0104) | the human-readable string |
| Coding Scheme Version | (0008,0103) | optional |

And the rule that makes it work is normative, in the standard's own words, with the loophole closed
in the next sentence:

> "the Value of Code Meaning (0008,0104) **shall never be used as a key, index or decision value**,
> rather the combination of Coding Scheme Designator (0008,0102) and Code Value (0008,0100) … may be
> used. Code Meaning (0008,0104) is a purely annotative, descriptive Attribute." … "This does not
> imply that Code Meaning (0008,0104) can be filled with arbitrary free text."
> — PS3.3 §8.3

Identity is `scheme + value`; the English string is documentation. Two systems that disagree about
the wording still agree about the concept.

Compare the current LensMark position, where identity *is* the English string and is tested by
substring match. Those two designs differ as a vocabulary differs from a convention.

**Versioning.** The code is never versioned; the *list it was drawn from* is. Context Identifier +
Context UID + Context Group Version + Mapping Resource travel with the instance, so a reader can
reconstruct which vocabulary was in force when the record was written. That is "grade B meant
something different before v0.4" as a runtime mechanism instead of a documentation promise.

**Local extension without forking.** Coding Scheme Designators beginning `99` are reserved for
private schemes, and a Context Group Extension Flag makes a local addition self-declaring. A survey
can add its own terms and remain readable.

**Deprecation.** PS3.3 §8.11, "Retired Codes and Expected Behavior": retired codes may still be sent
and receivers are expected to keep recognising them. A term is never deleted, because deleting one
silently changes the meaning of every record that used it.

**One source, many artifacts.** PS3.16 is authored once in DocBook, and every context group page
ships generated exports — HTML, FHIR JSON, FHIR XML, IHE SVS XML — plus a stable keyword. This is
the "single generator" principle at standards-body scale, and it is the model for
`08-draft-spec/generate.py`.

## 2. GSPS: portrayal as a separate object

A Grayscale Softcopy Presentation State is a **separate instance with its own identity** that
*references* images. The pixels are never modified. This is the overlay-not-burned-in rule,
standardised around 1999.

Two details worth stealing outright:

**Meaning rides on a Graphic Layer, not on colour.** A layer carries an id, an order, a description
and *two* recommended display values — a greyscale value **and** a CIELab colour. Redundant
achromatic encoding is not an accessibility afterthought here; it is in the data model. DICOM also
retired its RGB attribute in favour of CIELab in 2004, so "the same red" means the same red on two
monitors.

**Compound graphics carry their own fallback.** A semantic mark must ship an alternate rendering in
primitive shapes, linked by the same instance id. A renderer that understands the semantic type
draws it richly; one that does not still draws something correct. For AstroMark the Einstein-radius
circle is a compound graphic in exactly this sense — a measurement whose pixel radius is *computed*,
not drawn.

## 3. Where the measurement lives

TID 300: a measurement is a single numeric content item whose concept name is the coded quantity,
whose value carries a **UCUM** coded unit, with method as a coded modifier hung off the same node and
asymmetric uncertainty in a dedicated context group.

The part that matters most for AstroMark is the **direction of derivation**:

> The number is primary, and the geometry is `INFERRED FROM` it. **The shape is a witness, not a
> second copy.**

That is precisely the fix for θ_E being stored twice and kept in sync by a prompt rule. Put the
measurement on the finding; let the drawn circle be derived.

## 4. Negation and uncertainty are coded values

CID 240 "Present-Absent" is non-extensible and has exactly three codes: **Present, Absent,
Undetermined.**

AIM had `isPresent` as a boolean on its observation entity — the right instinct, the wrong type,
because **a boolean cannot say "undetermined"**. DICOM's three-valued coded version is strictly
better, and the standard even records the earlier version it fixed.

There is more: a numeric value may be replaced entirely by a *qualifier* code — "not a number",
"measurement failure", "value out of range" — so a failed measurement is recorded as a failure rather
than as a missing field or a zero.

And TID 1500's "Qualitative Evaluations" container holds coded questions with coded answers, which
makes *"Is this an arc?" → "No — spiral arm"* structurally identical to the positive case.

**This is the answer to negative evidence**, and it is better than what any astronomy format offers.

## 5. Two identities per mark

DICOM carries both a **Tracking Identifier** (this mark) and a **Tracking Unique Identifier** (the
physical thing this mark is of), replicated onto GSPS graphic objects and SEG segments.

Grouping is then *emergent*: four marks that share a tracking UID are four images of one object, and
the same mechanism links marks across files, sessions and observers. Many-to-many works without
contorting the document into a graph. This is a better answer than AstroMark's current `source` tag,
and it generalises to comparing two annotators' marks on the same feature.

The honest caveat from practice: DICOM's tracking ids are frequently left empty, because something
has to *propose* the shared identity when a human draws the second image. It is a tooling
obligation, not a schema one.

## 6. Provenance on four independent axes

DICOM keeps these separate and never collapses them: **who** observed (person or device, inheritable
and overridable per mark), **how** it was made, **what** produced it (algorithm name, version,
parameters), and **what it superseded** (predecessor documents — correction is a new instance citing
the old, not an in-place edit).

The "how" axis is three-valued: **AUTOMATIC, SEMIAUTOMATIC, MANUAL.**

> That middle value is the one people always forget and always need. A model-proposed arc that a
> human nudged is neither machine-made nor hand-made, and with a two-valued enum it is mislabelled
> forever.

This is a real gap in the current design. LensMark's `created_by.kind` plus `status: edited` covers
it by accident; making it explicit is cheap and it is exactly the propose-then-correct workflow the
project runs.

## 7. Factoring the long tail

DICOM's bulk annotation model requires — *mandatorily* — that a group of near-identical marks factor
out their commonality: one shape type, one property code, one generation method, then bulk coordinate
arrays. Re-stating the shared fields per mark is illegal.

A cutout with two hundred field galaxies to mask costs an order of magnitude fewer tokens this way.
Directly relevant to the few-shot budget in `06-llm-ergonomics.md`.

---

## 8. The cautionary tale: AIM

**NCI's Annotation and Image Markup got the central abstraction exactly right and died anyway.**

AIM's founding thesis is the record/portrayal split this project has independently arrived at:
*markup* is the graphical symbols placed over an image; *annotation* is the explanatory information
about the pixels. That is the correct architecture, stated clearly, in 2008.

It is dead. The reference implementation repository is archived; its last substantive commit was
**2014-04-29**.

Three verifiable reasons, each a warning:

**It violated its own thesis in its own schema.** `GeometricShapeEntity` — the *markup* class,
the one that was supposed to carry no meaning — carries `lineColor`, `lineOpacity`, `lineStyle` and
`lineThickness`. The standard that invented the separation put portrayal on the geometry object
anyway. If the schema permits it, it will happen.

**It made every relationship its own class.** The AIM 4.0 schema has 124 complex types, of which
**58 are `…Statement` classes**, with names like
`ImagingObservationEntityIsIdentifiedByTwoDimensionGeometricShapeEntityStatement`. Expressiveness
paid for with an unwritable surface.

**It imported ISO/HL7 datatype machinery because it looked rigorous.** Every scalar became an ISO
21090 type requiring a companion schema. The ceremony exceeded what its users would carry.

> This is the sharpest evidence for the recommendation in `07-recommendation.md` to take P4's
> **term identity** and leave P4's **ceremony**. The coded-concept architecture is right. The
> full apparatus is what killed the one project that tried it.

## 9. Two more failures worth naming

**RT Structure Set: the optional coded field loses to the required free-text one.** In the most
widely deployed structured annotation object in medicine, ROI Name is Type 2 — *must be present, may
be empty, free text* — while the coded semantics are Type 3, *optional*. Predictably, the name became
the de facto key and downstream systems match on it.

That is the LensMark substring-match bug, in a thirty-year-old international standard, for the same
structural reason: **whichever field is mandatory becomes the identity, regardless of what the
designers intended.**

**GSPS is excellent and under-used.** It has been in the standard since ~1999, is supported by
essentially every PACS, and is the right design. The workflow that dominates in practice is still
burning annotations into a flattened capture image, because it is easier and every viewer can show
it.

> A correct design does not win by being correct. If AstroMark's overlay path is harder than
> exporting a PNG with the marks drawn on, people will export the PNG. The three-file contract has
> to be the path of least resistance, not the virtuous one.

---

## 10. What to take

| take | from |
|---|---|
| Term identity is `scheme + opaque code`; the human string is non-normative | DICOM coded entry |
| Version the *context group*, not the code; retire, never delete | PS3.3 §8.11 |
| Reserve a prefix for local terms so a survey can extend without forking | `99`-prefixed schemes |
| One authored source generating every downstream artifact | PS3.16 → HTML / FHIR / SVS |
| Portrayal is a separate, referenced, versioned object | GSPS |
| Ship a greyscale value *beside* the colour, in the data model | Graphic Layer's paired display values |
| A semantic mark carries a primitive fallback rendering | Compound Graphic Sequence |
| The measurement is primary; the drawn shape is inferred from it | TID 300 |
| Presence is three-valued and coded, never a boolean | CID 240 |
| Two identities: this mark, and the thing it is of | Tracking Identifier / Tracking UID |
| Generation method is three-valued: automatic, semiautomatic, manual | SR observer context |
| Factor near-identical marks; forbid restating shared fields | bulk annotations |

| avoid | evidence |
|---|---|
| Declaring a record/portrayal split and then putting style on the geometry object | AIM 4.0's own XSD |
| Expressiveness that produces an unwritable surface | 58 `…Statement` classes |
| A mandatory free-text name beside an optional coded one | RT Structure Set |
| Assuming a correct overlay design will be adopted because it is correct | GSPS vs burned-in captures |
