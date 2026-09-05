# Requirements register

**Status:** consolidated 2026-09-03 from the workshop record, the LensMark codebase, and the
annotation campaigns. Every row traces to a source. Nothing here is new invention.

Sources, by shorthand:
`W04` `workshops/2026-08-31-LensMark/04-annotation-standard.md` (the standard as discussed) ·
`W02` `.../02-inspection-method.md` (the evidence hierarchy) ·
`W03` `.../03-hubble9-rounds.md` (the five annotation rounds) ·
`W10` `.../10-decisions-actions-open-questions.md` ·
`PROP` `.../analysis/proposals.json` (43 adjudicated proposals in groups `contract_schema`,
`propose_prompt`) ·
`CODE` `apps/LensMark/` as it stands · `CAMP` the Fable annotation campaigns.

---

## A. What the notation must be able to SAY

| # | Requirement | Source | Today |
|---|---|---|---|
| R1 | A controlled **role** vocabulary, orthogonal to the display label — ~17 lens roles | `PROP contract_schema-01` (P0) | free-text label, 40 chars, matched by substring |
| R2 | **Lens mass is never a lensed image.** deflector / second deflector / satellite are one family | `W04 §2.1` | one deflector assumed |
| R3 | **Polarity**: positive / negative / ambiguous, as a typed field — on the roles that can bear on the claim, and absent on those that cannot | `PROP -02, -03` (P0/P1) | polarity only in label wording |
| R4 | An **alternative** enum naming the non-lens reading, required when polarity ≠ positive | `PROP contract_schema-03` | absent |
| R5 | **Source grouping** — these marks are images of one source; a system may have two sources | `PROP contract_schema-13` | inexpressible |
| R6 | An unpaired but definitely-lensed feature is still marked | `W04 §2.1`, `PROP propose_prompt-03` | second arc left unmarked |
| R7 | A **secondary ring** around a second deflector | `W04 §2.5`, `PROP contract_schema-09` | "exactly one ring" — inexpressible |
| R8 | A **dust lane** inside the deflector, recorded as a hard case and **not** as evidence against lensing | `W04`, `PROP propose_prompt-02` (P0) | no label, no rule |
| R9 | **hard_case** tags for the morphologies that are still lenses | `PROP contract_schema-11` (P0) | absent |
| R10 | Mask **treatment**: cut out, or fit as its own light component | `PROP contract_schema-15` | all masks identical |
| R11 | **Never-mask** rule: the deflector's halo and bound satellites are lens mass | `W04 §2.8` | a satellite can be masked |
| R12 | **Segmentation polygons**, lens light versus lensed light, with asymmetric tolerance | `W04 §2.9`, `PROP contract_schema-20` | no polygon type |
| R13 | Counter-image **search status**: found / not found / **not searched** | `PROP contract_schema-10` | unrecorded |

R13 deserves emphasis: *"no counter-image found"* must be distinguishable from *"did not look"*.
That is a three-valued absence, and in a structured-output schema a null and a missing key are the
same thing to many model outputs — so it has to be an explicit enum member.

## B. Measurement

| # | Requirement | Source | Today |
|---|---|---|---|
| R14 | θ_E is a **radius**, never a diameter | `W04 §2.5` | correct, undocumented |
| R15 | The ring is centred on the deflector's light and **follows the deflector mark** if it moves | `PROP contract_schema-06` | `center_ref` exists, lint-only |
| R16 | Radius = the **radial midpoint of the main image**, not its edge and not half the separation | `PROP contract_schema-06` | prompt makes half-separation primary — not the room's convention |
| R17 | Explicit **lower / upper bounds** replace the disowned term `alt` | `W04 §2.5`, `PROP -07/-08/-09` | `alt_arcsec`; two decks carry `(alt)` rings |
| R18 | A **coded estimation method**, not a free string | `PROP contract_schema-08` | free string |
| R19 | **A measurement lives in exactly one place.** | audit | θ_E stored twice, unlinked, synced by a prompt rule |
| R20 | The expected radial ordering is a guide, **never a rejection ground** — a counter-image may lie farther out than the arc | `W02 L3` | absent |

## C. Portrayal, and its separation from the record

| # | Requirement | Source | Today |
|---|---|---|---|
| R21 | Marks are a **vector overlay, never burned into pixels** | `W04 §2.7` | satisfied; purpose unstated |
| R22 | **Colour must not carry meaning alone**; role by shape, polarity by texture | `PROP contract_schema-05`, WCAG 1.4.1 | green = deflector, enforced by substring match |
| R23 | Thin but legible; the overlay must let the image through | `W04 §2.3` | 6.4–10.8% ink; the legend plate is ~half of it |
| R24 | **Resizable for emphasis, without disturbing any measured size** | Greg, 2026-09-03 | no emphasis channel |
| R25 | Labels **on the object**; the key **below the image**, once per sheet | `W04 §2.3` + `CAMP` | legend on by default, plate on the image |
| R26 | Deterministic render: same document → byte-identical PNG | `CODE`, 9 golden SHA-256s | satisfied |
| R27 | Portrayal is a **versioned artifact**, not inlined per file | audit | the whole style table is copied into every file (10.4%) |

## D. Provenance, review, process

| # | Requirement | Source | Today |
|---|---|---|---|
| R28 | Every mark records **who or what made it**, at what effort, in which run | `CODE` | satisfied |
| R29 | A **reviewer verdict per mark**, with the edit delta, through propose→critique→accept | `CODE` | satisfied |
| R30 | **RA/Dec travels with the annotation**, recorded once per image | `W04 §2.6` (D2) | optional and absent from all nine decks |
| R31 | Item geometry stays **image-relative**, because modelling consumes image-centred offsets | `W02 L0` | satisfied |
| R32 | The arcsec scale must be **supplied and its provenance recorded** | `W04 §2.7` | satisfied (`scale_source: assumed`) |
| R33 | Multiple **renderings of one cutout** share one geometry frame | `PROP contract_schema-21` | one image per cutout |

## E. Machine and governance

| # | Requirement | Source |
|---|---|---|
| R34 | Models emit under a **flat, closed JSON Schema** — no `$ref` | `CODE`, tested |
| R35 | A **50-example few-shot payload** must be affordable | `W10` A11 |
| R36 | **Free text for people, stripped for models** — provably, by a closed key allow-list | golden README §EMBARGO rule 1 |
| R37 | **One generator** for the vocabulary; adding a term must touch one file | audit: six uncoordinated copies |
| R38 | A **general core plus domain profiles**; the lens vocabulary is not in the core | `naming-study.md` |
| R39 | A core-only reader must **degrade gracefully** on an unknown profile | R38 |
| R40 | Round-trip with the existing exporters: COCO, DS9, masks, few-shot | `CODE` (57–161 LOC each) |
| R41 | The spec must be **usable standalone by a model** — every semantic rule in the document, not only in a prompt | `W10` D18 |

---

## F. The two hard constraints

**The embargo.** `reproductions/lensjudge/golden/README.md` §EMBARGO rule 1: Xiaosheng's free text,
explicitly including meeting transcripts, may never be quoted, paraphrased or summarised into any
model-facing artifact — prompt, rubric, exemplar, or trace. A notation spec is meant to be handed to
a model, so **every sentence of the spec must be mechanism-level**. Enum names, geometry rules and
procedure ordering are mechanisms and are fine. Wording is not.

This is why R36 matters structurally rather than as hygiene: if free text and semantics can share a
field, "strip the free text" is a judgement call. If they cannot, it is a key allow-list and the
guarantee is by construction.

Verification: `workshops/2026-08-31-LensMark/build/ngram_check.py` (4-gram against the transcript
turns) plus `reproductions/lensjudge/golden/banned_lexicon.txt` (298 entries).

**STScI.** The nine Hubble cutouts are embargoed pending a public feature — internal only, never
published or uploaded. Hence the synthetic reference cutout: it carries no embargo, is reproducible
from a seed, and can ship with the standard permanently. Renders over the real nine live in
`examples/internal/`, which is gitignored.

---

## G. What the record does *not* settle

Carried forward honestly, because a spec that pretends these are decided would be wrong.

1. **The nominal ring's colour** — grey in Xiaosheng's prompt sequence, white in LensMark, green in
   one campaign round. The adjudicated proposals fix only the *bound* rings. Open.
2. **The A/B/D grade boundaries.** Grade C is defined on the record; A, B and D are not. `W02` calls
   this the largest hole in the capture.
3. **The few-shot payload format**, and Greg's hypothesis that image + coordinate metadata beats an
   annotated PNG. Untested. `06-llm-ergonomics.md` prices the encodings but does not test the
   hypothesis.
4. **Whether the live marker offsets were systematic** — a renderer-side question nothing addresses.
5. **The p_lens bands** in the adjudicated proposals are the distiller's proposal for consistency,
   explicitly *not* values from the room. They need ratifying.
6. **Which roles polarity applies to — now settled, and worth recording how.** The first draft made
   `polarity` mandatory on *every* mark. That is wrong, and the counterexample is not an edge case:
   a field star or a masked galaxy asserts nothing for or against lensing, and in a real deck **82%
   of items are mask circles**. Forcing them to be `positive` made that value mean two things at
   once.
   The fix is in the vocabulary, not the schema shape: each role carries `takes_polarity`, and
   `polarity` is required where it is true and **must be absent** where it is false. Eleven of the
   twenty marks in the reference scene carry polarity; the other nine are inventory or measurement.
   A fourth "neutral" value, and a split into separate presence and bearing axes, were both
   considered and rejected — presence does not vary here, and where it genuinely does it is already
   carried better by `counter_image: found | not_found | not_searched` at system level.
   **Neither the fault nor the fix is in the workshop record**; both came out of building the
   reference annotation and then being challenged on field objects.
