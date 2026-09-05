# Prior art: astronomy metadata and vocabulary standards

The register AstroMark must sound like, and the bodies it must not offend.

## 1. AVM — the nearest existing neighbour

**Astronomy Visualization Metadata** is the closest thing that already exists: metadata embedded
directly in astronomical image files, covering title, caption, credit, colour-to-wavelength
assignment, coordinate projection and a subject taxonomy. It is what any reviewer will ask about
first.

What it does **not** do is the whole gap AstroMark fills: it describes *the image*, not *features
within the image*. There is no geometry, no per-feature semantics, no polarity, no provenance per
mark, no render contract. The spec should say this plainly and cite AVM rather than ignore it.

Its transferable mechanism is the **`X` local-extension hierarchy**, with normative wording worth
copying almost verbatim: general AVM-compliant readers **must ignore** terms in the local extension
space and **must not** error on them. That is the graceful-degradation rule for unknown profiles,
already written by an astronomy standard.

## 2. IVOA Vocabularies — one source, generated everything

A term **is a resolvable URI**; there is exactly one authoritative source file; every downstream
artifact is generated from it by content negotiation on that URI. This is the same "single generator"
principle as DICOM's PS3.16, in the register this project actually publishes in.

Two details:

**`desise` — "dead simple semantics".** Alongside the rich RDF rendering, IVOA ships a **flat,
generated, RDF-free JSON** version with a fixed minimal shape: `{uri, flavour, terms: {id: {label,
description, wider, narrower}}}`. A consumer that wants the vocabulary without an RDF stack gets one.
That is precisely what a model-facing enum list needs, and it is generated, not maintained.

**The VEP process** requires every proposed term to arrive with (a) id, label and definition, (b) its
relation to existing terms, and critically (c) a **"Used-In" URI proving the term is already in real
use.** A vocabulary that only accepts terms already in use cannot inflate.

## 3. UCD — the two-clock split, proven over twenty years

The **grammar** has been frozen as a Recommendation since 2005. The **word list** is maintained
separately and updated regularly. Two artifacts, two cadences, one stable and one alive.

This is the strongest available evidence for splitting AstroMark's JSON Schema (frozen; changes
almost never) from its vocabulary file (alive; changes with the science). It is the same conclusion
`04-metadata-proposals.md` reaches from the audit, reached here by twenty years of practice.

UCD also composes: the uncertainty of a quantity **derives its identity from the quantity** —
`stat.error;pos.angDistance` is the error on `pos.angDistance`. AstroMark could get θ_E's uncertainty
term for free by composition rather than by declaring a separate one.

## 4. VOEvent — a graded assertion, already standardised in astronomy

VOEvent's `<Why>` / `<Inference>` models the assertion as a **separate object from the geometry**,
referencing it, with **N assertions permitted over one observation**, each carrying a
controlled-vocabulary concept and a probability.

An astronomy standard already does the thing AstroMark needs — a graded, referenced,
multiple-interpretation claim about an observation — and it is the right precedent to cite when
someone asks why the format separates marks from findings.

## 5. Do not mint terms that already exist

The **IVOA object-type vocabulary** already publishes URIs for the objects being marked, including
`grav-lens` and `lensed-image` (with `lensed-g`, `lensed-q` variants), alongside the Unified
Astronomy Thesaurus.

**AstroMark should reference these rather than invent parallel class terms.** Minting `lens:deflector`
where `http://www.ivoa.net/rdf/object-type#grav-lens` exists is how a standard makes itself
un-interoperable and irritates the body whose recommendation it needs.

The distinction to hold: AstroMark's own vocabulary should cover the things that are *specific to
annotation* — polarity, treatment, estimation method, review verdict, emphasis, hard-case class — and
should **borrow** the astronomical object classes. That is a change to the draft vocabulary worth
making before it is circulated.

## 6. FITS and WCS

The keyword model, and how a cutout records its sky position and orientation. AstroMark's `image.wcs`
block and the `north_up` / `east_left` flags are a thin projection of this; the spec should state the
correspondence so a reader can round-trip to a FITS header without guessing.
