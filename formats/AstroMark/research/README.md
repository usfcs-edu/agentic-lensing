# AstroMark — research

Design research for **AstroMark**: an open standard for annotating astronomical images, with a
symbolic notation and a vector metadata format. Strong gravitational lenses first
(`astromark/lens/1.0`), astronomy-general later.

**Everything here is a proposal for team review. Nothing has been adopted.** The recommendations are
recommendations; where the evidence is thin or the record is silent, that is said rather than
papered over.

## Read in this order

| | file | what it is |
|---|---|---|
| 1 | [`00-executive-summary.md`](00-executive-summary.md) | the recommendation and what to decide, in two pages |
| 2 | [`03-notation-proposals.md`](03-notation-proposals.md) | **the symbolic notation** — four proposals, rendered and measured |
| 3 | [`04-metadata-proposals.md`](04-metadata-proposals.md) | **the metadata format** — six proposals, encoded and measured |
| 4 | [`09-jwst-examples.md`](09-jwst-examples.md) | **worked examples on real JWST candidates**, every mark derived from a recorded measurement |
| 5 | [`07-recommendation.md`](07-recommendation.md) | the pick, the trade named, the migration |
| — | [`AstroMark-Research.pptx`](AstroMark-Research.pptx) | **the whole thing as a 59-slide walkthrough**, for the team review |
| — | [`AstroMark-Research.docx`](AstroMark-Research.docx) | **every document in this directory as one Word report** — 91 pages, 15 figures inline, for reading and commenting |

Supporting:

| file | what it is |
|---|---|
| [`naming-study.md`](naming-study.md) | why "AstroMark", and the core-plus-profiles architecture (earlier study) |
| [`01-prior-art/`](01-prior-art/) | how other fields solved this — medical imaging, computer vision, web standards, astronomy, notation design, accessibility |
| [`02-requirements.md`](02-requirements.md) | the requirement register, R1–R41, each traced to its source |
| [`05-accessibility-design.md`](05-accessibility-design.md) | colour as reinforcement, and the automated test that proves it |
| [`06-llm-ergonomics.md`](06-llm-ergonomics.md) | what the format costs a model to read and write |
| [`08-draft-spec/`](08-draft-spec/) | the recommended option written out, ready to implement or reject |
| [`examples/`](examples/) | the reference cutout, every rendered arm, every encoding, and the tools that made them |

## The three questions Greg asked, answered

**Should we consider accessibility — a notation that doesn't rely on colour, with colour to help
differentiate?** Yes, and the reasoning changes the notation rather than just the palette. Role is
carried by glyph shape, polarity by stroke texture; colour reinforces and never carries alone. This
is testable rather than aspirational: of 136 pairs of marks that can share a panel, **123 are not
reliably separable by colour under some vision condition, and zero are separated by colour alone**.
See `05-accessibility-design.md`.

**Can the symbols be concise but readable?** They can be considerably thinner than today, but the
measurement says the problem was misdiagnosed. The on-image legend plate is 3.7–6.6% of the panel —
**about half the total ink, and in most decks more than every arrow, circle and label combined**. The
fix is a territory rule, not thinner strokes: *no ink inside the image rectangle that is not about a
specific location in it*, with the key moved to a caption band below. See `03-notation-proposals.md`.

**Can marks be resized to emphasise them?** Yes, once you notice that some marks encode a
measurement in their size — the ring radius *is* θ_E. So every drawn scalar is tagged METRIC or
PRESENTATIONAL, and emphasis scales only the presentational ones. A ring can shout without its
radius moving a pixel: **emphasise the stroke, never the path.**

## Reproducing the numbers

```bash
cd examples
python make_reference_cutout.py     # the synthetic scene, from a seed
python render_proposals.py          # all seven arms
python measure.py all               # metrics + contact, CVD, polarity and A/B sheets
python make_encodings.py            # the same annotation in every metadata proposal
python make_jwst_examples.py --variants   # the three JWST candidates + the six-panel case
python migrate_lensmark.py <deck>   # convert a real lensmark/1.0 deck, and report what it costs
python cvd.py selftest              # CIEDE2000 against Sharma et al. reference pairs
python cvd.py pairtest reference/marks-table.json    # the accessibility gate
python embargo_check.py --spec      # the model-facing spec against both protected corpora

cd .. && ~/.venvs/workshop/bin/python build_deck.py   # rebuild the presentation
         ~/.venvs/workshop/bin/python build_docx.py   # rebuild the Word report
```

Requires `~/.venvs/lensmark` (PIL, numpy). Everything is deterministic from a seed.

## Two notices

**STScI embargo.** The nine Hubble cutouts the project has been annotating are embargoed pending a
public feature. Nothing derived from them is committed here: `examples/internal/` holds those renders
and is gitignored. The committed examples use a **synthetic reference cutout** that carries no
embargo and can ship with the standard permanently.

**Golden-grader embargo.** Per `reproductions/lensjudge/golden/README.md` §EMBARGO rule 1, meeting
transcripts may never reach a model-facing artifact. A notation spec *is* model-facing, so the draft
spec in `08-draft-spec/` is written at mechanism level from the adjudicated enum artifacts, not from
the transcript-quoting workshop prose, and is checked mechanically with `ngram_check.py` and the
banned lexicon. The research prose in this directory is human-facing and cites the workshop reports
by path.

## Status of the evidence

Honest about what is measured and what is not:

- **Measured:** ink coverage, occlusion, stroke contrast, encoding sizes in characters, the
  colour-difference gate, the reference arms against real LensMark renders.
- **Derived, not measured:** token counts (no Claude tokenizer available offline; character ratios
  are reported, which is what the decision turns on).
- **Not tested:** whether image + coordinate metadata teaches a model better than an annotated PNG.
  That is Greg's hypothesis on the record and remains open; `06-llm-ergonomics.md` names the
  experiment that would settle it.
- **Judgement, declared as such:** which of the four notations to adopt. The measurements narrow it;
  they do not decide it.
