# Worked examples on real JWST data

**Status:** three candidates annotated, 2026-09-03.
Reproduce with `examples/make_jwst_examples.py` then `examples/render_proposals.py`.

The synthetic reference scene (`examples/reference/`) exists because it can ship with the standard
and because it can be built to contain every hard case. These examples exist for the opposite
reason: to show the notation carrying **measurements that already exist**, on systems the team knows.

## Source, and why it is safe to use

`jwst-strong-lens-search/top100_clean/` — the top hundred candidates from the one-shot search of the
**public JWST NIRCam archive** (4.48 deg², 5,391 targets, ten surviving adversarial verification).
Public archive data with no embargo, unlike the nine Hubble cutouts.

Each figure is 752×562 holding **six 240×240 panels**: three stretches of the 10″ field on the top
row, three of the 3.5″ zoom below. North up, east left. The accompanying `top100_clean.csv` carries
41 columns per candidate, including the verifier grade, the blind Einstein radius with its stated
method, the arc radius and position angle, whether a counter-image was seen, and the verifier's
written reasoning.

## The claim these examples make

> **Every mark is derived from a recorded measurement, not placed by eye.**

Arc positions come from `blind_arc_radius_arcsec` and `blind_arc_pa_deg`; the nominal ring from
`blind_theta_E_arcsec` and `blind_theta_E_method`; counter-image positions from the coordinates the
verifier wrote into their own note. The transform is stated in the script: for a feature at position
angle PA and radius r in a panel of width FOV, `u = 0.5 − r·sin(PA)/FOV`, `v = 0.5 − r·cos(PA)/FOV`.

So these figures test whether the notation can express what was *already measured* — which is the
actual claim, and a stronger test than annotating by eye.

## The three, and what each one strains

### rank 1 — `J3440482-522486`, a published lens
SL2S J02176-0513, grade A, 3 of 3 verifiers. A blue tangential crescent east of a red elliptical with
a compact counter-image nearly opposite; θ_E = 1.15″ by **half-separation**.

The clean positive case, and it exercises three things at once: `emphasis: key` on the counter-image
(corner brackets, and its label demoting to an index because it sits too near the panel edge — the
designed failure, working); a coded `method` recording that half-separation was the rule used; and a
**negative** mark for the verifier's own correction, that the NE streak runs radially rather than
tangentially and is a field galaxy, not a second arc.

### rank 2 — `J15199556+2122210`, searched and not found
Grade A. A single thin tangential arc, ridge 1.44–1.52″, mean 1.48″ by **arc midline**.

`counter_image: not_found` — and that is a different statement from `not_searched`. The verifier
looked at every stretch and none was visible, which is why θ_E rests on the arc radius alone and
"could be ~10% high if the source is offset". Hard case: `single_giant_arc`.

This is the three-valued absence that DICOM, RadElement, WMO and Galaxy Zoo all arrived at
independently (`01-prior-art/README.md` §5), and here it carries real consequence.

### rank 3 — `J34707505-219476`, the counter-image is farther out than the arc
Grade A. Five blue knots at r 1.24–1.49″ over PA 95–162, plus a blue blob antipodal to the chain at
**r = 1.88″** — the same colour as the knots, and therefore a plausible counter-image, but at a
*greater* radius than the arc, which is the reverse of the usual configuration.

The verifier explicitly declined half-separation for that reason. The annotation records exactly
that: the blob is `polarity: ambiguous` with `alternative: companion_projection` — drawn with a
dotted shaft and a hollow struck terminator — and the ring carries `method: arc_midline`, so the
decision is recoverable rather than lost. Had the blob been accepted, θ_E would be nearer 1.6″.

## The finding this produced

**The rank-3 counter-image at r = 1.88″ falls outside the 3.5″ zoom panel.** The same annotation is
simply unrenderable there; it is rendered on the 10″ field instead.

That is the clearest possible demonstration that a mark's coordinates are meaningless without a
declared frame — and that the two field-of-view groups in this figure do **not** share one. It also
shows why the six-panel layout exists: the zoom shows the arc, the wide field shows the counter-image.

The current format has **one image per record**. It cannot say "these six renderings are of one sky",
cannot say which panel the marks were placed on, and cannot carry a per-panel scale.
`examples/jwst/variants-*.json` writes that case out. This is requirement R33, made concrete by real
data rather than argued for.

## Files

| file | what |
|---|---|
| `examples/jwst/<id>.png` | the extracted colour 3.5″ panel |
| `examples/jwst/<id>-wide.png` | the colour 10″ panel, where a mark falls outside the zoom |
| `examples/jwst/content-<id>.json` | the annotation, in the neutral content form |
| `examples/jwst/<id>-n1.png`, `-n4.png` | rendered under two notations |
| `examples/jwst/<id>-panels/` | all six panels, extracted |
| `examples/jwst/variants-<id>.json` | the six-renderings-of-one-sky case |
