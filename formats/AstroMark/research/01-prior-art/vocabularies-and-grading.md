# Prior art: controlled vocabularies and graded assessment

Medicine's *semantic* half, as distinct from its geometry half. This is where the answer to
AstroMark's undefined grade boundaries lives.

## 1. Term identity and lifecycle

**RadElement** mints a permanent code for every enumerated *value*, derived from its parent term's id
plus an ordinal assigned at first registration: `RDE1695.2` = "indeterminate". Documents store the
code; the display name is a mutable annotation on it.

**Three independent clocks, never conflated:**

| clock | what it versions |
|---|---|
| a monotonic integer per **term** | that term changed |
| a lifecycle **status** per term, with its own date and a preserved history array | where that term is in its life |
| a semver version for the **format** | the grammar changed |

**RadLex Playbook** runs a four-value lifecycle with a **TRIAL** tier and a written policy: historical
codes are never overwritten or deleted; TRIAL codes may change without notice; anything else needs
committee approval. A trial tier is what lets a three-person group add a term without committing to
it forever — directly applicable to the `provisional` status in the draft vocabulary.

**SNOMED** records replacement as a **typed** link, not a bare pointer, and three types cover almost
everything: *same as* (it was a duplicate), *replaced by* (one unambiguous successor), *possibly
equivalent to* (the old term was vaguer than any new one). "This term was replaced" and "this term
was a duplicate" are different facts and a single `replaced_by` field cannot tell them apart.

**RadElement** also defines a legal, obviously-provisional identifier form — `TO_BE_DETERMINED123` —
that an author or model may emit *before* the registry has assigned a real id, documented as being
rewritten on registration. That is a neat answer to "what does a model write when it needs a term
that does not exist yet", which otherwise becomes a free-text leak.

## 2. Graded assessment — the answer to the grade problem

**BI-RADS defines its scale as a three-column table:**

| ordinal category | numeric probability band | the specific next observation that would settle it |

plus a **published expected prevalence** for each grade, and it keeps "incomplete / cannot assess" as
a category of its own rather than a missing value.

That is precisely the shape the workshop record is missing. Grades A–D are used but only the
borderline class has a definition, "what would confirm it" lives in prose, and there is no expected
prevalence to check a grader against.

**PI-RADS goes further and makes the overall grade a DERIVED field**, computed from coded component
scores by a published lookup table. The evidence is stored; the verdict is computed. Every numeric
threshold is named once and referenced.

> For a project whose stated goal is consistent grading between two people, the derived model is
> strictly stronger: it makes disagreement *locatable*. Two graders who disagree on the verdict but
> agree on the evidence have a different problem from two who disagree on the evidence.

**ACR BI-RADS publishes a term-level, side-by-side old-versus-new changelog** as a first-class release
artifact, with a stated reason for each removal. Not a diff — a rationale.

## 3. Presence is four-valued, and it is a different axis from the term

RadElement's near-universal value set: **present / absent / indeterminate / unknown**, where
*indeterminate* means "I assessed it and cannot decide" and *unknown* means "I did not assess it".

This is the finding that changes an AstroMark recommendation. The current `polarity` field conflates
two questions — *is this feature really there?* and *does it bear on the lensing claim?* A dust lane
is unambiguously present and bears neither way. **These are two axes.** See `README.md` §"The finding
that changed a recommendation".

## 4. Structure and governance

**IHE MRRT** splits the document into geometry with stable ids and coded content referencing them —
the same split medicine reached three other times.

**OME-Zarr** (the computational-pathology contrast case) keeps `image-label.colors` as a list keyed by
label value, parallel to the labels rather than inside them, and carries exactly **one** version stamp
for the whole document rather than a version key per block.

**The RSNA/ACR review process** is sized for a small group and worth copying almost verbatim: a
completeness check by whoever holds the registry; **one** reviewer checking conformance to the written
standard, explicitly instructed not to relitigate the science; then publication. Three stages, named
roles, an explicit scope limit on the reviewer.
