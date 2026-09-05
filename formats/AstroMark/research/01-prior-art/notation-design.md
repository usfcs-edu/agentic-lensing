# Prior art: how durable symbolic notations are designed

The theory, and the sign systems that actually survived. This informs the visual notation rather than
the file format.

## 1. Bertin's visual variables

Position, size, shape, value (lightness), colour hue, orientation, texture — each with different
properties. Some are *selective* (you can pick out all marks sharing that value at a glance), some
*ordered* (you can rank them), some only *associative*.

The consequences that shape AstroMark:

- **Size is ordered and quantitative**, which is why it is the natural carrier of a measurement — and
  therefore why it cannot also carry emphasis.
- **Shape is associative only** — not selective, not ordered. It is good for "which kind is this?" and
  bad for "find all of these quickly", which is why role families rather than individual roles must
  be the coarse layer.
- **Texture is selective *and* ordered** — which is what makes solid > dashed > dotted read as
  decreasing certainty rather than as three arbitrary categories.
- **Value is ordered and selective**, and it is the right home for emphasis for exactly that reason.

## 2. The WMO station model — the best worked example of density

A surface station plot packs about ten variables into one compact glyph, legible at a glance after
brief training. Three transferable mechanisms:

**Block the code space by family so a new member is guessable without a legend.** The present-weather
table is 100 codes in ten decades: 00–19 no precipitation now, 20–29 precipitation in the past hour
but not now, 30–39 duststorm or blowing snow, and so on. A code you have never seen is still placed
by its decade.

**Give "obscured" and "not observed" distinct first-class codes, rather than one null.** Cloud cover
runs 0–8 oktas, then **9 = sky obscured** — an observation, not a missing value. Compare
`counter_image: not_found` versus `not_searched`.

**Hold units and estimation method once, in a fixed header slot, as a coded indicator governing the
measurement reported elsewhere.** One indicator says both "estimated versus anemometer" and "m/s
versus knots", and it governs the wind value. The estimation method travels with the reading, not in
prose.

## 3. FGDC geologic symbols — two kinds of doubt on two variables

The single most directly applicable rule in the survey:

> **Put two independent kinds of doubt on two independent visual variables, and forbid either
> variable from carrying the other.**

FGDC uses **dash pattern for locational accuracy** — solid = accurately located, long dash =
approximate, short dash = inferred — entirely independently of what the line *is*. The identity of the
feature and the confidence in its position are two questions, and conflating them loses both.

AstroMark currently has one texture channel doing polarity. There is a second kind of doubt in this
domain — how well a mark's *position* is known — and it has no channel at all. Worth deciding
deliberately rather than by omission.

## 4. Standards that made a sign system survive

- **ISO 19117 Portrayal** exists solely to separate data from its depiction — the same architecture as
  DICOM GSPS and the DS9 Symbol Editor, arrived at independently.
- **ISO 7010** registers safety symbols centrally with a stated review process; its lesson is that a
  sign system needs a registry, not just a design.
- **IHO chart symbols** (INT 1) demonstrate a legend printed *once per chart series*, not once per
  chart — the same conclusion the ink measurements force here.
- **SMuFL** gives music notation a stable code point per glyph with the visual design left to the
  font: term identity separated from portrayal, in a notation eight centuries old.
- **Heraldic blazon** is a *textual* notation from which any herald redraws the same image. It is the
  historical proof that a compact text form and a visual form can be two renderings of one record —
  which is the read/write asymmetry in `06-llm-ergonomics.md`.
- **FIDE algebraic notation** is the minimal case: a closed vocabulary, one line per move, learnable
  in minutes, and unchanged for decades because it never tried to express anything else.
