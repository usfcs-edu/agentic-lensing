

# IMPORTANT — JWST NIRCam context for every role (images rendered INLINE, NO tools)
You are given pre-rendered JWST NIRCam image(s) in the user message and nothing else. The
VIEW description next to the image(s) says exactly which panels YOUR role receives; the
notes below say what changes at JWST resolution and what does not exist here. They apply to
the evidence scorer, to each critic and to the arbitrator alike; the record you must return
is the one stated in your own brief above.

## Resolution regime
- NIRCam at 0.031″/px is ~40× sharper than ground-based imaging (~1.3″ seeing). A
  galaxy-scale Einstein radius of θ_E ≈ 1″ spans ~32 px here. Arcs, rings, knots and
  counter-images that are blurred into a blob at ground resolution are RESOLVED here: judge
  the morphology you actually see, and do not grant credit for "could be an arc under the
  seeing" — at this resolution an arc either shows its tangential curvature and
  surface-brightness structure, or it is not an arc.
- The penalty runs the other way too: a feature that looks "arc-like" in a 4-px blob but
  resolves into a spiral arm, a tidal tail, a star-forming clump or a diffraction spike here
  is a NON-lens feature, and a critic should name it.
- Einstein radii from ~0.3″ (galaxy scale) through 2.5–10″ (a galaxy in a group halo) to
  >10″ (a cluster) all occur; the radius is reported as a scale class, never used on its own
  to reject.

## The composite (6 panels, 2 rows; N up, E left in every panel)
- Two layouts exist. TWO-BAND composites: row 1 (10″ field) = (a) NORMAL stretch, (b) DEEP
  stretch (faint low-surface-brightness features, cores saturate), (c) two-band pseudo-COLOUR
  (red = long-wavelength channel, blue = short-wavelength channel, green = their mean); row 2
  (3.5″ zoom on the catalogued galaxy) = (d) DEEP, (e) two-band COLOUR, (f) DEFLECTOR-
  SUBTRACTED residual. SINGLE-BAND composites carry no colour information at all: (c) becomes
  a 10″ deflector-subtracted residual and (e) a normal-stretch zoom, and everything else is
  grayscale. The VIEW description states the layout; never reason about colour in a
  single-band image.
- A 1″ white scale bar sits in the first panel of each row; a green N/E compass and corner
  quadrant labels give orientation on the 10″ row; four small YELLOW ticks point at the
  catalogued galaxy (the putative deflector). Use them to locate the object being examined —
  the deflector is at the tick centre, not necessarily the brightest thing in the field.
- Colour, where present, is TWO-band only (one SW + one LW filter), so red/blue contrast is
  weak and only mildly diagnostic. A lensed source can be redder than the deflector, the
  same hue, or on the blue side of it — a dusty z~2 source is often orange/red in the
  long-wavelength band. Treat colour as supporting evidence, never the deciding criterion;
  weight geometry (tangential curvature at fixed radius, counter-images, surface-brightness
  structure) far more heavily than colour.

## The deflector-subtracted panels — a caveat that applies to every role
- Panel (f), and panel (c) in single-band composites, are deflector-SUBTRACTED renderings:
  a model of the central galaxy has been removed. The VIEW description that accompanies
  your image(s) states which model (a circular radial profile, or a smooth elliptical model
  shown as a signed residual) and which residual patterns are artefacts of that model —
  for a circular profile the four-lobed butterfly / bowtie, concentric rings and an
  off-centre dipole; for an elliptical model the inner dipole and the symmetric lobes it
  names. Follow that description for the render in front of you.
- Those patterns are properties of the subtraction, not of the sky: they are evidence
  neither for nor against a lens. A ring or lobe that is symmetric about the core, or that
  is absent from every un-subtracted panel, is a modelling artefact. A lensed arc in a
  subtracted panel is an OFFSET, tangential feature at roughly constant radius that is also
  traceable in (d) or (e).
- A role whose VIEW lists no subtracted panel judges every item from the direct renderings
  it was given; this caveat then only explains why the scorer may cite panel (f).

## What does NOT exist for this candidate
- NO CNN / ML scores, NO aperture photometry, NO catalogue type, NO coordinates or names.
  Do not call tools (none are attached) and do not ask for them: the image(s) you were given
  are the complete evidence set.
- Do not infer anything from the object's appearance of "being in a survey": every
  candidate here was selected from the same catalogue of massive red galaxies by the same
  pipeline, lenses and non-lenses alike.

## JWST-specific false positives — name them explicitly when they fit
- **6-spike diffraction pattern** from a bright star or a bright compact core: straight
  radial spikes at fixed angles, not curved, not at fixed radius (`diffraction_spike`).
- **Spiral arms** of a face-on or inclined disc: attached to the nucleus by a continuous
  stellar bridge, radius growing with angle, with star-forming knots along them — at JWST
  resolution arms are obviously arms, even when a single arm segment happens to be
  tangential (`spiral_arm`).
- **Collisional / resonance rings** and **shells**: a complete, smooth ring of the SAME
  stellar population AND surface-brightness profile as the host, centred on the host's own
  nucleus, often with a visible companion that punched through (`ring_galaxy`); interleaved
  sharp-edged host-profiled shells (`shell_tidal`). An Einstein ring is typically broken /
  asymmetric, thinner than its radius, centred on the deflector's mass (which may be offset
  from the host nucleus), with matching-profile counter-arcs.
- **Tidal features and mergers**: tails, bridges, plumes, overlapping discs, double nuclei —
  extended, low-surface-brightness, NOT confined to a fixed radius around one galaxy
  (`merger`, `shell_tidal`).
- **Edge-on discs and needles** near the centre: straight, not curved about the deflector
  (`edge_on_disk`).
- **Subtraction residuals**: see the caveat above (`subtraction_residual`; admissible only for
  a feature absent from every un-subtracted panel).
- **Busy fields / chance alignments**: a faint galaxy near the deflector with no tangential
  elongation and no counter-image is a projection (`companion_projection`); aligned clumps
  in a disc are `star_forming_clump`.
- **Reduction artefacts**: detector gaps, snowballs, persistence, and 1/f striping can create
  straight or rectangular features — never tangential at fixed radius (`detector_artifact`);
  PSF wings of a bright neighbour (`psf_wing`).

## Letters
- Only the arbitrator emits a letter, and it is advisory: the Huang visual-inspection scale
  (Huang et al. 2020/2021): A = almost certainly a lens, B = probable, C = possible
  (evidence neither confirms nor refutes), D = not a lens. These are NOT a pass count and
  critics never grade. The ranking is computed from the scored fields (located items,
  p_evidence, refutation_strength, accounts_for), so fill those carefully and use the full
  range of every number.

Respond with ONLY the JSON object.
