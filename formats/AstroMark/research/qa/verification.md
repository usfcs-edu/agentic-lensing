# Verification

Run 2026-09-03. Every check is reproducible from `examples/` and `08-draft-spec/`.

| # | Check | Result |
|---|---|---|
| 1 | **Determinism.** Rebuild the reference cutout from its seed; compare bytes. | PASS — identical (`322450f9f5ee4344…`) |
| 2 | **CIEDE2000 correctness.** Nine reference pairs from Sharma, Wu & Dalal (2005), including the pairs that break naive implementations. | PASS |
| 3 | **Accessibility gate.** Every pair of marks that can share a panel, under normal vision + deuteranopia + protanopia + tritanopia + greyscale. | PASS — 136 pairs, **123 below the ΔE floor, 0 separated by colour alone** |
| 4 | **Generator staleness.** All four downstream artifacts regenerated from the vocabulary file and compared. | PASS — 4/4 current |
| 5 | **Gate G1: structured-output shape.** `$ref` count and open objects in the generated schema. | PASS — 0 `$ref`, 0 open objects |
| 6 | **Schema usability.** The reference annotation validated against the generated schema. | PASS |
| 7 | **Embargo, model-facing.** Both spec documents and the render rules, against the transcript 4-grams and the 298-entry banned lexicon. | PASS — 0 lexicon hits, 0 collisions of any kind |
| 8 | **Repo hygiene.** Changes outside `formats/`; embargoed imagery under `formats/`. | PASS — 0 changes, `internal/` ignored |

## Reference-arm validation

The prototype's reference arms reproduce the real thing closely enough to trust the comparison:

| arm | prototype | real |
|---|---|---|
| R-CURRENT (LensMark today) | 6.23% ink | 6.4–10.8% across the nine decks |
| R-CAMPAIGN (agent campaign) | 1.99% ink | 1.2–2.3% measured on the campaign renders |

## Migration, executed rather than asserted

`examples/migrate_lensmark.py` converts a real `lensmark/1.0` deck. On deck-08 (29 marks, the
group-scale two-deflector ring): all four arrow labels mapped to roles lexically, 24 mask circles
defaulted to `treatment: mask`, and **four system-level fields need a human pass** —
`counter_image`, `n_images`, `hard_case` and `sources`. Those four are not conversion failures; they
are the things `lensmark/1.0` had no way to record.

## A verification gap this exposed

The first build of `AstroMark-Research.pptx` **rendered correctly through LibreOffice to a 59-page
PDF, and PowerPoint refused to open it.** One caption box on a code slide had been given a negative
height (`cy = -146304` EMU) when the code block ran long enough to push it past the slide edge.

LibreOffice silently tolerates a non-positive shape extent. PowerPoint treats it as corruption and
offers to repair the file. **So rendering a deck to PDF is not sufficient verification that it
opens.**

Two fixes, both in `build_deck.py`:

1. every textbox clamps its width and height to a positive minimum, and the code-slide layout budgets
   its vertical space before drawing rather than after;
2. `Deck.validate()` runs **before every save** and refuses to write a file containing a
   non-positive extent, an off-slide origin, or a shape overrunning the bottom margin.

The structural checks that now pass on the built file: XML well-formed in every part, no
non-positive extents, no duplicate shape ids, every table's row cell-count matching its column
count, and every relationship id resolving.

## The same gap, caught a second time

Building the Word report hit the identical class of problem. The link flattener — which turns
cross-document Markdown links into plain text, since inside one document they point at nothing —
used a regex that also matched the `[caption](path)` half of an image reference `![caption](path)`.
Every one of the fifteen figures was silently stripped before pandoc ever saw it.

The build reported success, because the assertion compared the number of image references in the
assembled Markdown (0, after stripping) against the number of shapes in the document (0). **Two
zeros agreeing is not a check.**

Fixed twice over: the flattener now refuses to match after a `!`, and the assertion is three-way —
figures *injected*, references *surviving into the Markdown*, and shapes *present in the document*
must all agree. A figure eaten before pandoc, or one that fails to resolve after, now fails the build.

Structural checks that pass on the built `.docx`: every XML part well-formed, all 15 media files
present with no dangling relationship ids, 15 drawings with no non-positive extents, and
`updateFields` set so the table of contents fills on first open.

## The three-way assertion earned its keep immediately

On its first run after being added, the figure assertion failed: *15 injected, 16 references in the
markdown, 15 shapes in the document.* The sixteenth was a literal `!` + `[caption](path)` inside a
code span in this very file, where the image-stripping bug above is documented. The count was
right — 15 figures really did render — but the counter was counting image syntax inside code.

Fixed by stripping fenced blocks and inline code before counting. Worth recording because a check
that had only ever seen agreement had not been shown to detect anything; this one detected a real
discrepancy the first time the inputs changed underneath it.

## What is NOT verified

- **Token counts are derived, not measured.** No Claude tokenizer is available offline. Character
  counts are exact and the ratios between encodings are what the decision turns on.
- **No live structured-output call** was made against the generated schema. It is flat and closed by
  static check, and validates the reference document, but gate G1's live half is unrun.
- **No human legibility testing.** The thumbnail forced-choice test — can a reader pick the non-lens
  from a contact sheet — is specified but needs human readers to mean anything.
- **The few-shot hypothesis is untested.** See `06-llm-ergonomics.md` §4.
- **Anomalous trichromacy** (deuteranomaly, the most common deficiency) is not simulated; the gate
  tests dichromacy at severity 1.0, which is conservative but not the same thing.
