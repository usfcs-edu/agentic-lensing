# Recommendation

**Status: a recommendation for the team, not a decision.** Where the evidence is thin, that is said.

---

## The short version

| | Recommendation |
|---|---|
| **Notation** | **N1 Terminator alphabet** for adoption now; **N4 Evidence graph** prototyped alongside it: the only one that fixes the named defect |
| **Metadata container** | **P1 evolved bespoke** — the cheapest decision, and the one that will absorb the most argument |
| **Semantic identity** | **P4-lite**: prefixed terms (`lens:arc`) resolved in one versioned vocabulary file that *generates* every other surface |
| **Model surfaces** | models **read** the compact line form (~30k tokens for 50 examples), **write** JSON under a flat closed schema |
| **Portrayal** | leaves the record entirely, into a versioned style document referenced by hash |
| **Accessibility** | hard constraint: role by shape, polarity by texture, colour redundant — enforced by an automated ΔE test |
| **Do first, regardless** | **build the vocabulary file** |

---

## 1. Why the container is the least important decision

I began expecting this to be a choice between GeoJSON, Web Annotation, coded concepts and bespoke
JSON. The audit says otherwise. Every structural problem in `lensmark/1.0` — six uncoordinated
copies of the vocabulary, θ_E stored twice and unlinked, a free-string estimation method, colour
carrying semantics, no role field, no polarity, no grouping — is a **vocabulary-identity** problem.
**None would have been prevented by any of the containers.**

The sharpest illustration is the current definition of a lens galaxy:

```python
def _is_deflector(label): return bool(label) and "deflector" in label.lower()
```

duplicated independently in Python and TypeScript. A label reading `spiral arm, NOT a deflector`
renders green — the colour meaning lens mass. The semantics of the standard ride on substring-matching
an optional 40-character string.

So: **choose the container for cheapness, and the vocabulary machinery for rigour.** Converters
between containers are 57–161 lines each. The vocabulary is the part that decays.

What killed the current format was not a wrong choice; it was **erosion** — nothing prevented drift.
The axis that matters is therefore the one where a mechanism can be installed:

> One vocabulary file → generated pydantic Literals, generated JSON Schema enums, generated
> TypeScript unions, generated prompt vocabulary, generated documentation, with a CI check that
> fails when any of them is stale.

`08-draft-spec/generate.py` implements exactly this, and `--check` is the gate. Adding a role touches
**one file**. Today it touches six.

## 2. Why P1 rather than the more principled options

**P4 (coded concepts, the DICOM SR and NCI AIM lineage) is architecturally the best answer and I am
recommending against it.** It is the only proposal in which "green means deflector" is *structurally
impossible*, and in which the θ_E double-storage bug is unrepresentable. That is real.

It costs 9.4× the compact form, demands two id spaces, and — decisively — the medical standards
sustain that discipline because an institution curates the vocabularies. A three-person group is not
that institution. The honest prediction is that the ceremony gets abandoned under deadline, someone
adds `"role": "arc"` beside the coded concept "just for the frontend", and within a release there
are two unlinked representations. Which *is* the bug being fixed.

**So take the identity and leave the ceremony.** `"role": "lens:arc"` — one prefixed string resolved
in a versioned, SKOS-shaped vocabulary — retains what matters: profile membership visible in the
term, graceful degradation on an unknown prefix, and vocabulary versioned separately from schema. At
enum cost.

**P2 (GeoJSON) fails on a fact about this data.** GeoJSON has no circle, and nine of twenty marks are
circles whose radius *is* the datum. And measured, it is the **largest** encoding of the six — you
pay the standard's tax and get none of its tooling.

**P3 (Web Annotation) fails the flat-schema gate.** Its native review model — a critique is an
annotation targeting another annotation — is genuinely the best answer to that problem, and worth
stealing conceptually. But no flat closed JSON Schema exists for JSON-LD without ceasing to be Web
Annotation, at which point the standard's authority is gone.

**P6 (relational) is the right export, not the record.** Best corpus-query story by far; hostile to
per-cutout editing and to git. Keep it as the thing an archive ingests.

## 3. The notation

**Adopt N1 now; prototype N4 alongside.**

N1 keeps the pointer — the one mark the room has consistently endorsed — and moves the semantics into
the terminator shape, with polarity in the shaft texture. It has the highest role resolution of the
four and the lowest adoption cost, and it is what can be presented at the January meeting.

But **N4 is the only proposal that fixes the defect that motivated this work.** Flipping through nine
panels of the current output, a viewer cannot tell which one is the non-lens: the marks on a
lookalike use exactly the same palette and weight as the marks on a real lens. N4 makes a positive
panel one with an unbroken chord and a negative panel one whose chord is struck — visible at
thumbnail size, in greyscale, from across a room. Its source chord is the one genuinely new idea in
the option space, and it should not be discarded because it is unfamiliar.

**N2 is not a fifth notation; it is how to write the spec for whichever wins.** Assigning each of
Bertin's visual variables to exactly one semantic dimension is what makes the accessibility
constraint structural rather than a checklist.

**N3's source-index slot is worth stealing regardless**, because source membership is inexpressible
today and costs two characters.

### The measured case

| arm | ink/statement | statements |
|---|---:|---:|
| N4 Evidence graph | **0.128** | 20 |
| N2 Bertin ledger | 0.154 | 20 |
| N3 Station model | 0.163 | 20 |
| N1 Terminator alphabet | 0.178 | 20 |
| R-CAMPAIGN | 0.133 | 15 |
| **R-CURRENT (today)** | **0.416** | 15 |

Today's notation costs **3.2× more ink per unit of meaning** than the best candidate while expressing
five fewer things. Read the normalised column: raw ink penalises an arm for saying more.

## 4. Six changes worth making whichever proposal wins

Notation-independent, individually valuable, in dependency order:

1. **2× canonical render.** Forced by arithmetic — three stroke elements are below the aliasing floor
   at native size.
2. **Tag every drawn scalar METRIC or PRESENTATIONAL.** This is the whole answer to "resize for
   emphasis": a ring can shout without its radius moving.
3. **The `emphasis` enum with layered compositing.**
4. **Keep-out geometry in the label solver**, using each circle's *annulus* rather than its bounding
   box. Roughly a twenty-line change and the largest single legibility win available.
5. **Demotion-to-index** so crowding degrades deterministically.
6. **The caption band**, and remove the on-image legend plate from the contract. The plate is ~half
   the current ink.

Only the sign inventory is a bet. These six are not.

## 5. Migration

`lensmark/1.0` → `astromark/lens/1.0` is mechanical for everything except the parts that were never
expressible.

| today | becomes | lossless? |
|---|---|---|
| `schema_version: "lensmark/1.0"` | `schema` + `profile` | yes |
| `items[].type` | `marks[].type` | yes |
| `items[].label` | `role` (inferred from the label vocabulary) + `label` for display | **lossy where a label is free text** — needs a human pass over existing decks |
| `items[].color` | dropped from the record; derived from `role` via the style document | **intentionally lossy** |
| `mask_circle.kind` | `role` + `treatment` | yes; `treatment` defaults to mask |
| `system.theta_e.alt_arcsec` | `lower_arcsec` or `upper_arcsec` by comparison with the nominal | yes, and the two decks carrying `(alt)` rings are fixed by it |
| `einstein_ring.theta_e_arcsec` | geometry only; the measurement lives in `system.theta_e` | yes — this removes the duplicate |
| `style_defaults` (inlined) | `style_ref` by id and hash | yes |
| `legend.show` | `legend.mode` | yes |
| — | `polarity`, `alternative`, `source`, `hard_case`, `counter_image`, `bound`, polygons | **new; nothing to migrate** |

The label→role pass is the only part needing human judgement, and it is bounded: nineteen decks.

## 6. What I could not settle

Stated plainly, because a recommendation that hides its gaps is worse than one that admits them.

1. **Greg's few-shot hypothesis is untested.** Whether image + coordinate metadata teaches a model
   better than an annotated PNG is the question the format is partly designed around, and nothing
   here tests it. `06-llm-ergonomics.md` names the experiment; it is cheap now that the encodings
   exist, and it should run **before** the format is frozen.
2. **Token counts are derived, not measured.** No Claude tokenizer offline. Character ratios are
   exact and are what the decision turns on, but the absolute numbers carry an assumption.
3. **N1 versus N4 is a judgement, not a measurement.** The metrics narrow the field and rule out the
   status quo; they do not choose between two good candidates. The nine-panel forced-choice test —
   can a reader pick the non-lens from thumbnails alone — would discriminate them, and needs human
   readers to be worth anything.
4. **The grade boundaries remain undefined** for A, B and D. The format carries grades; it cannot
   make them carry one meaning for two annotators. That is a separate piece of work and it blocks
   any inter-annotator agreement measurement.
5. **Where a dust lane sits in the polarity scheme.** It is a real feature (so not negative) that
   does not support lensing (so positive reads oddly). The current encoding marks it positive and
   tags the system, which works — but it suggests polarity may want a fourth, neutral value. This
   surfaced while authoring the reference annotation and is **not** in the workshop record.
6. **Occlusion is worse for every candidate than for the campaign reference** (≈11% against 6.2%),
   because the candidates draw bound rings and segmentation polygons the references cannot express.
   Whether those earn their occlusion is a judgement for the room.

## 6b. What the prior-art survey changed

Sixty-six formats across eight domains (`01-prior-art/`). Five findings altered the design rather
than merely confirming it.

**Presence and polarity are two axes, not one.** RadElement separates *present / absent /
indeterminate / unknown* from the term itself. AstroMark's `polarity` currently conflates "is this
feature really there?" with "does it bear on the lensing claim?" — which is exactly why a dust lane
sits awkwardly. **This is a better answer than adding a fourth polarity value**, and it supersedes
what open question 5 originally proposed.

**Borrow the object classes; do not mint them.** The IVOA object-type vocabulary already publishes
`grav-lens` and `lensed-image` as URIs, alongside the Unified Astronomy Thesaurus. AstroMark's own
vocabulary should cover what is specific to *annotation* — polarity, treatment, estimation method,
review verdict, emphasis, hard case — and reference the existing astronomical classes. Minting
parallel terms is how a standard makes itself un-interoperable and irritates the body whose
recommendation it will need. Recorded as an action in the vocabulary file.

**Generation method is three-valued.** DICOM's AUTOMATIC / SEMIAUTOMATIC / MANUAL. The middle value
is the one everyone omits and everyone needs — a model-proposed arc that a human nudged is neither
machine-made nor hand-made, and a two-valued enum mislabels it forever. That is *precisely* the
propose-then-correct workflow this project runs, so the field should be explicit rather than inferred
from `status: edited`.

**The grade problem has a worked answer.** BI-RADS defines its scale as a three-column table —
ordinal category, numeric probability band, **and the specific next observation that would settle
it** — plus a published expected prevalence per grade. PI-RADS goes further and *derives* the overall
category from coded component scores by a published lookup table, storing the evidence rather than
only the verdict. For a project whose goal is consistent grading between annotators, the derived
model makes disagreement locatable: two graders who differ on the verdict but agree on the evidence
have a different problem from two who differ on the evidence. **This is the most actionable
unclaimed idea in the survey**, and it addresses the undefined A/B/D boundaries directly.

**Two hard numbers for the render rules.** Exactly three lightness levels fit between black and white
at 3:1 contrast (0.000, 0.100, 0.400; a fourth needs 1.30, which does not exist) — which independently
confirms that the emphasis channel gets three values and cannot get four. And two textures used
together must differ in period by a full octave, with every period at least 3× the stroke width.
Both are now in `08-draft-spec/render-rules.md`.

And one warning worth carrying into the room: **a correct design does not win by being correct.**
DICOM GSPS has been the right overlay answer since 1999, is supported by every PACS, and lost in
practice to burning annotations into a flattened image, because that was easier. If AstroMark's
overlay path is harder than exporting a PNG with the marks drawn on, people will export the PNG.

## 7. What to decide at the review

1. Notation: **N1 now**, or **N4**, or N1 with N4's chord grafted in?
2. Container: accept **P1 + P4-lite identity**, or spend the argument on the container?
3. Do the **six notation-independent changes** land regardless? (I would say yes.)
4. Is the **read/write asymmetry** — models read lines, write JSON — acceptable, given it means two
   surfaces and a round-trip test in CI?
5. Should **presence and polarity become two separate axes**, per the RadElement finding?
6. Who runs the **few-shot experiment**, and does the format freeze wait for it?

## 8. The one thing to do first

**Build the vocabulary file.** It is the only artifact all six proposals share, and the only one
whose absence caused every problem in the audit. Building it before the container is chosen is not
premature — it is the part that is not in dispute.

A draft is already at `08-draft-spec/astromark-vocab-lens-1.0.json`: 80 terms across 10 schemes, with
definitions, families, statuses, deprecations and a governance rule sized for three people. It
generates a flat closed schema that the reference annotation validates against today.
