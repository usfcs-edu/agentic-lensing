# Machine ergonomics: what the format costs a model to read and write

**Status:** measured, 2026-09-03. Reproduce with `examples/make_encodings.py`.

Greg's stated goal for the format is that it be *"useful for humans and for LLMs to consume and
learn from."* Those two audiences pull in opposite directions, and this section is where the tension
gets priced rather than argued about.

## 1. The measurement

One annotation — the reference scene, 20 marks covering 12 roles, both polarities, all three
geometry kinds — encoded in every candidate. Same content, so the differences are the format.

| Encoding | chars | ×smallest | derived tokens | 50-example payload |
|---|---:|---:|---:|---:|
| **P5 model-read surface** | **2,091** | **1.0** | ~600 | **~30k** |
| P5 line notation (full) | 2,173 | 1.0 | ~620 | ~31k |
| P6 relational | 8,089 | 3.9 | ~2,300 | ~116k |
| **P1 evolved bespoke** | 14,362 | 6.9 | ~4,100 | ~205k |
| P3 W3C Web Annotation | 17,777 | 8.5 | ~5,100 | ~254k |
| P4 coded concepts | 19,678 | 9.4 | ~5,600 | ~281k |
| P2 GeoJSON-shaped | 19,762 | 9.5 | ~5,600 | ~282k |

*Character counts are exact. Token figures are **derived** at 3.5 chars/token, not measured — no
Claude tokenizer is available offline and a different tokenizer family would give a misleading
absolute. The ratios are what the decision turns on, and ratios are near tokenizer-independent.*

### Three results worth stating plainly

**The compact form is not a marginal saving.** A 50-example few-shot payload is the difference
between roughly 30k tokens and roughly 205k. That is not "cheaper"; it is the difference between the
few-shot experiment the team has planned being routine and being a budget decision every time it
runs.

**GeoJSON is the *most* expensive encoding, not the cheapest.** This surprised me. The wrapper
(`{"type":"Feature","geometry":{...},"properties":{...}}` around every mark) costs more than the
bespoke schema it was supposed to simplify, and it buys nothing here because nine of the twenty
marks are circles — and GeoJSON has no circle. Their radius, which *is* the datum for a mask and for
θ_E, gets exiled into `properties`, so the geometry member no longer holds the geometry. You pay the
standard's tax and get none of its tooling.

**The verbosity of the rigorous option is real.** P4's coded concepts — the DICOM SR and NCI AIM
lineage, where every term is a `{scheme, code, meaning}` triple — costs 9.4× the compact form. That
buys something genuine (see `04-metadata-proposals.md`), but it should be bought knowingly.

## 2. The asymmetry that resolves the tension

The requirement that models emit under a **flat, closed JSON Schema** is not negotiable — it is what
makes malformed output impossible rather than merely unlikely, and LensMark already depends on it.
The requirement that fifty examples fit in a payload is also not negotiable.

Those look contradictory. They are not, because **the read surface and the write surface do not have
to be the same surface**:

> **Models READ the compact line form and WRITE the JSON form.**

Reading is where the volume is — fifty examples, every one of them costing tokens. Writing is where
the correctness matters — one document, schema-enforced, repairable. Splitting them satisfies both
constraints at once, and nothing else in the option space does.

This is not a novel trick; it is what the existing pipeline already half-does. `exports/fewshot.py`
already emits a prose markdown card alongside each example precisely because the raw JSON is a poor
thing to learn from. The proposal is to replace that ad-hoc card with a *normative, round-trippable*
line form.

## 3. What the line form looks like

The entire reference annotation — every mark, both polarities, the bounds, the source grouping, the
review verdict — in 2.1 kB:

```
#astromark lens/1.0  frame=norm,tl,+x-right,+y-down  size=arcsec
#system likely_lens grade=A p=0.92 cimg=found n=2 hard=dust_lane,second_deflector,counter_image_outside_arc
#source S1 images=2 config=double thE=1.45"
defl     + 0.500,0.350 -> 0.500,0.510
dust     + 0.624,0.349 -> 0.530,0.526
defl2    + 0.235,0.657 -> 0.353,0.591
sat      + 0.244,0.460 -> 0.376,0.486
arc      + 0.394,0.709 -> 0.458,0.590  S1  ; rev=wrong_position 0.31"
knot     + 0.480,0.700 -> 0.490,0.600  S1
cimg     + 0.614,0.296 -> 0.550,0.415  S1  emph=key
arc      - 0.851,0.358 -> 0.739,0.282  alt=spiral_arm
ring     + @m-defl r=1.45"  S1  [1.15,1.8] method=arc_midline
ring     + @m-defl r=1.15"  S1  bound=lower
gal      + 0.181,0.241 r=1.1"  treat=mask
star     + 0.713,0.638 r=0.55"  treat=mask
```

Read the eighth line: `arc - ... alt=spiral_arm`. That single line says "there is an arc-shaped
feature here, it argues AGAINST lensing, and the reason is that it is a spiral arm." The current
format cannot say that at all, in any number of characters.

### Honest costs

- **Polygons dominate it.** Three segmentation polygons account for about 1.4 kB of the 2.1 kB.
  Without them the annotation is roughly 800 characters. Segmentation is inherently verbose in any
  text form; if the few-shot payload is tight, polygons are the first thing to drop from the read
  surface.
- **No API-level grammar enforcement.** Structured output applies to JSON, not to text. A text
  surface needs a parser and a repair layer. Mitigating fact: `lensmark/validate.py` is *already* a
  267-line repair layer whose `n_repaired` and `n_invalid` counters are treated as dataset columns —
  the project has already accepted repair-on-parse as design rather than failure.
- **Two surfaces can drift.** This is the real risk and it has a real answer: generate both from one
  definition, and put a round-trip property test over the whole corpus in CI. If a field can be
  added to one surface without the other, the design has already failed.

## 4. What this does *not* settle

Greg's hypothesis on the record — that giving a model the image plus independent coordinate metadata
beats giving it an annotated PNG — is **untested**, and nothing here tests it. This section measures
what each encoding *costs*, not what it *teaches*. The experiment that would settle it is:

> the same cutouts, the same model, three payload variants — annotated PNG only; original PNG plus
> the line form; original PNG plus the line form plus the annotated PNG — scored on the existing
> propose pipeline against the human decks.

That experiment is cheap now that the encodings exist, and it is the single highest-value thing to
run before the format is frozen. Until it is run, the read/write asymmetry is justified by token
cost alone — which is sufficient, but it is a weaker claim than the one the team actually wants.

### One distinction the experiment should keep separate

**The payload that teaches the notation is not the payload for doing the task**, and conflating them
is how a few-shot budget gets spent on the wrong thing.

*Teaching* plausibly wants all three artifacts together: the original shows what was there, the
rendered overlay demonstrates the visual grammar — which marks go where, how a negative reads, what
restraint looks like — and the coordinate record supplies the precision the render cannot. The
render is doing real work here, because the grammar is a visual thing and describing it in prose
costs more tokens than showing it.

*Doing the task* wants the original cutout plus the specification, and **not** a rendered example of
a different system. A worked render of some other candidate is decoration at that point: it consumes
roughly a thousand tokens to restate a grammar the spec already states, on a system the model is not
being asked about.

So the arms worth measuring are not three but four, and the fourth is the one most likely to be
missed:

| arm | teaching payload | working input |
|---|---|---|
| A | annotated PNG only | original |
| B | original + line form | original |
| C | original + line form + annotated PNG | original |
| **D** | **C** | **original + spec, no exemplar render** |

The interesting comparison is C against D: whether the annotated render earns its tokens at
*inference* time, or only at *teaching* time.
