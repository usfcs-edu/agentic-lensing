# Prior art: accessible visual encoding

## The normative rule

**WCAG 2.2 SC 1.4.1, "Use of Color"**: colour must not be the *only* visual means of conveying
information. Its conformance test is operational and is exactly the right one here — *remove the
colour; is the information still there?*

That test is implemented in `examples/cvd.py` and reported in `05-accessibility-design.md`.

## Simulation models

- **Brettel, Viénot & Mollon (1997)** and **Viénot, Brettel & Mollon (1999)** — the projection-based
  dichromat models.
- **Machado, Oliveira & Fernandes (2009)** — a physiologically-based model with severity parameters,
  and the one used here. Matrices are inlined in the repo rather than imported, because a library
  that changes version would make the verification irreproducible.

All operate on **linear RGB**. Skipping the sRGB decode is the most common error in CVD code and it
materially changes the answer.

## Palettes

**Okabe–Ito Color Universal Design** (8 colours) and **Paul Tol's** qualitative schemes are the
defensible choices. The important fine print, routinely missed: a palette labelled "colour-blind
safe" is safe *up to a stated number of colours*. **ColorBrewer encodes this explicitly**, publishing
a per-cardinality `blind` array — a scheme safe at 3 colours may not be safe at 6.

The transferable mechanism: **attach machine-readable, cardinality-indexed accessibility claims to
the vocabulary itself**, and have tooling read them rather than re-derive them.

## Two hard numbers

**Exactly three lightness levels exist.** At a 3:1 contrast ratio between adjacent grades, the
achievable relative luminances between black and white are L = 0.000, 0.100, 0.400 — a fourth would
require L ≥ 1.30, which does not exist. This is a mathematical bound, and it independently confirms
that the emphasis channel gets **three** values (`muted`, `normal`, `key`) and cannot get four.

**Texture needs an octave.** Three rules, enforced at style-release time:
1. any two textures used together must differ in period by **≥ 2.0×** — a full octave;
2. every period must be **≥ 3× the stroke width**;
3. every texture must survive the rasterisation floor at the smallest intended output size.

The current dashed-galaxy versus dotted-star convention should be checked against rule 1 rather than
assumed.

## Casing, not halo

Cartographic practice: draw every stroke and glyph two-tone — a core ink plus a casing at the opposite
lightness pole, with contrast(core, casing) ≥ 3:1 and ideally ≥ 7:1. LensMark already does this for
*labels* (a dark halo) but not for *geometry*, which is why thin strokes fail over bright cores. The
render rules extend it to every stroke.

## Ship a mono profile

The strongest single recommendation: publish a second, **mandatory** portrayal profile called `mono`
that maps every ink to one achromatic value, and make the build assert that **term recovery from the
mono render equals term recovery from the default render**. That converts WCAG's operational test
into a build-time gate rather than a review step.

## What astronomy offers

Nothing. Not one region format addresses colour-vision deficiency, and DS9 and CRTF both default
everything to green. The *channels* exist and round-trip widely — DS9 point shapes
(`circle|box|diamond|cross|x|arrow|boxcircle`), dash patterns with explicit dash lists, line widths;
Aladin Lite catalogue shapes; CRTF line styles — so the work is to make redundancy a normative rule
and a conformance test, not to invent new machinery.
