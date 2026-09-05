# Accessibility: colour as reinforcement, and the test that proves it

**Status:** designed and verified, 2026-09-03. Reproduce with `examples/cvd.py` and
`examples/measure.py sheets`.

Greg asked: *should we have a symbol notation that doesn't rely on colour, but can use colour to
help further differentiate?* The answer is yes, and it is worth being precise about why, because the
reasoning changes the notation rather than just the palette.

## 1. Why this is not a courtesy

The workshop deferred accessibility behind reliability, and that was the right call at the time. But
three facts make it cheaper to do now than later:

1. **The current colour key is already broken on its own terms.** In the campaign that produced the
   best annotations so far, red and green were locked to deflector and arc, but four other colours
   were assigned opportunistically — and cyan meant "unrelated field object" on some panels and
   "lensed feature" on others *within the same nine-panel set*. A key that is not self-consistent
   for normal vision is not a colour-vision problem; it is a vocabulary problem that colour was
   asked to solve and could not.
2. **Roughly 8% of men have a colour-vision deficiency.** In a three-person group that is a
   coin-flip; in the community the standard is aimed at, it is a certainty.
3. **Figures get printed, photocopied, and projected.** Greyscale is not an edge case in astronomy;
   it is the second most common way a figure is seen.

The governing standard is **WCAG 2.2 SC 1.4.1, "Use of Color"**: colour must not be *the only visual
means* of conveying information. Note what it does and does not say — it does not ban colour, and it
does not require an ugly palette. It requires redundancy.

## 2. The channel assignment

| Semantic dimension | Primary channel | Reinforced by |
|---|---|---|
| **role family** (lens mass / lensed light / obstruction / field / measurement) | **glyph shape** — filled, open-curved, bar, circle | colour |
| **role within family** | glyph detail, or tick orientation in N2 | — |
| **polarity** (positive / negative / ambiguous) | **stroke texture** — solid / dashed / dotted | a strike through the terminator on negatives |
| **measured quantity** (θ_E, mask radius) | **size** — reserved exclusively for measurement | — |
| **emphasis** | stroke weight, glyph scale, corner brackets | — |
| **source membership** | an index letter, or a chord in N4 | colour |

Two consequences fall out of the table rather than being imposed:

**Size cannot carry emphasis**, because size is already spoken for by measurement. That is the whole
reason the emphasis mechanic is a stroke-weight multiplier and not a scale factor.

**Colour cannot carry polarity.** The workshop had already reached this conclusion independently —
the proposal to reserve white for "ambiguous" and grey for "NOT" was rejected in favour of line
style. The accessibility analysis arrives at the same place from a different direction, which is
mild evidence that it is right.

## 3. The palette

Okabe-Ito, the standard colour-vision-safe qualitative set. Used as *reinforcement only*, so nothing
breaks if a viewer cannot separate two of them.

| Family | Hex | Role |
|---|---|---|
| lens mass | `#009E73` bluish green | deflector, second deflector, satellite, lens-light segmentation |
| lensed light | `#56B4E9` sky blue | arc, knot, counter-image, lensed-light segmentation |
| obstruction | `#CC79A7` reddish purple | dust lane |
| field object | `#D55E00` vermillion | galaxy and star masks |
| modelled, not masked | `#E69F00` orange | a field galaxy fitted rather than cut out |
| measurement | `#F0E442` yellow | the nominal Einstein ring |
| bound | `#BFBFBF` grey | lower and upper θ_E rings |

## 4. The test

This is the part worth keeping regardless of which notation wins, because it converts a review step
into an assertion.

```
for every pair of semantically distinct marks that can appear on one panel:
    compute CIEDE2000 between their inks under
        normal vision, deuteranopia, protanopia, tritanopia, greyscale
    if the smallest of those is below the floor:
        assert glyph shape OR stroke texture OR orientation differs
```

The criterion is deliberately **not** "colour difference is large". A palette does not pass by being
colourful. It passes by never letting colour be the *only* thing separating two meanings.

Simulation uses **Machado, Oliveira & Fernandes (2009)** severity-1.0 matrices, inlined in the repo
rather than imported — a library that changes version would make the verification itself
irreproducible. Matrices operate on linear RGB, so sRGB is decoded before and re-encoded after;
skipping that step is the most common error in CVD code and it materially changes the answer.

### Calibrating the floor, which produced a finding

The floor was initially set at ΔE2000 = 12 and the test passed a pair it obviously should have
failed. Measuring the canonical case explains why:

> **Pure red `#FF0000` against pure green `#00E000` still measures ΔE2000 = 12.9 under deuteranopia
> simulation** — because almost everything that survives is a *lightness* difference (L\* 53 vs 78),
> not hue.

So a floor at 12 waves through the exact confusion the criterion exists to catch. The floor is set
at **15**, above that residual. Large filled areas would tolerate less; this notation is made of
thin strokes and small terminators, which tolerate more.

### Result

```
GATE: PASS
pairs checked: 136 | below the ΔE floor: 123 | separated by colour alone: 0
```

**123 of 136 mark pairs — 90% — are not reliably separable by colour under some vision condition.**
Every one of them is separated by shape, texture or orientation instead. That number is the argument
for the whole design: a notation that leaned on colour would be illegible to a large fraction of its
audience nine times out of ten, and would not know it.

## 5. Visual evidence

- `examples/reference/polarity-triad.png` — the same document rendered three times, all marks
  positive, all negative, all ambiguous, **with every label removed**. Solid, dashed and dotted are
  unmistakable with no text at all. If this sheet were ambiguous, the polarity channel would have
  failed.
- `examples/reference/cvd-sheet.png` — every arm through all four simulations.
- `examples/reference/contact-260.png` — at 260 px per panel, text is unreadable in *every* arm.
  Only the visual channel survives, which is the empirical case for putting polarity in texture and
  role in shape.

## 6. Limits, stated

- The gate tests **inks**, not rendered pixels. Two marks with distinguishable inks can still be
  confusable if one sits on a saturated galaxy core. That is what the contrast metric is for, and it
  is a separate measurement.
- Machado severity 1.0 models dichromacy. **Anomalous trichromacy — deuteranomaly, the most common
  deficiency at ~4.6% of men — is milder**, so testing at severity 1.0 is conservative in the right
  direction but is not the same as testing the most common case.
- Nothing here addresses non-visual access. A screen-reader path would mean a text serialization of
  the annotation, which the line form in `06-llm-ergonomics.md` happens to provide almost for free —
  an unplanned second use worth noting, not a claim to have solved it.
