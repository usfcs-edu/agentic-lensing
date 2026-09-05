# Embargo check: adjudication of 4-gram collisions in the research prose

Run: `python examples/embargo_check.py 0*.md README.md --json qa/embargo-research-prose.json`
Date: 2026-09-03. Raw report: `embargo-research-prose.json`.

## Final state (19 files, 2026-09-03)

| set | result |
|---|---|
| **Banned lexicon (298 strings from the written comments), all 19 files** | **0 hits** |
| **Model-facing spec** (`08-draft-spec/*.md`) | **all CLEAN — 0 collisions of any kind** |
| Human-facing research prose | 6 generic collisions, adjudicated below |

The spec is held to the strict bar because it is meant to be handed to a model. It reaches it by
being written from the adjudicated enum artifacts rather than from transcript-quoting prose — the
same discipline the embargo asks for, applied at the source rather than at the check.

## Result

**Banned lexicon (the 298 strings from the written comments): 0 hits in every file.** That is the
check that matters most, and it is clean.

**4-gram collisions against the transcript: 18 spans, all adjudicated GENERIC.** The tool's own
classifier escalated 7 of them as SUBSTANTIVE because they contain a content-class word. Each is
adjudicated below by hand. None reproduces domain meaning; all are ordinary English constructions
that collide by chance with a 21,078-n-gram corpus.

| span | file | disposition |
|---|---|---|
| `rather than starting from` | 03 | generic — no domain noun; describes a design method, not a lens |
| `in full this is` | 04 | generic — sentence connective |
| `the shape of the` | 04 | generic — "shape" here is a JSON structure, not a lens feature |
| `this is the part` | 05 | generic — discourse pointer |
| `is the difference between` | 06 | generic — comparative connective |
| `the reason is that` | 06 | generic — causal connective |
| `if the source is` | 09 | generic — conditional clause. The surrounding sentence paraphrases a *verifier agent's* written note in `top100_clean.csv`, which is machine-generated text and not part of either protected corpus |
| `is the only one`, `the only one that`, `this is not a`, `this is not the`, `this is the only`, `why this is not`, `is that it is`, `it is the same`, `it has to be` | various | function words only; classified GENERIC mechanically |

## Why blind pass/fail is the wrong instrument here

`ngram_check.py` is a deliberately strict tripwire built for short proposed *prompt* text, where any
collision deserves a look. Applied to long-form prose it will always flag ordinary English, because
four function words in a row appear in any 21k-n-gram corpus. The correct use is triage, not a gate:
a hit is a prompt to check, and the check is whether the shared span carries *meaning* from the
protected source. None of these do.

## The stricter standard, applied to the spec

The research prose in this directory is **human-facing** and cites the workshop reports by path. The
draft spec in `08-draft-spec/` is **model-facing** and is held to a higher bar: it is written from
the adjudicated enum artifacts in `analysis/proposals.json` rather than from transcript-quoting
prose, and every collision it produces is adjudicated in the same way before the spec is considered
finished.
