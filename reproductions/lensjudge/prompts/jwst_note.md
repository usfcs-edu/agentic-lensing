

# IMPORTANT — this is a JWST NIRCam composite, rendered INLINE (NOT DESI grz, NO tools)
The rubric above was written for ground-based DESI grz cutouts fetched through tools. For
this candidate you are given ONE pre-rendered JWST composite in the user message and nothing
else. Apply the SAME five criteria and the SAME A/B/C/D scale; the notes below say what
changes at JWST resolution and what does not exist here.

## Resolution regime
- NIRCam at 0.031″/px is ~40× sharper than DESI (~1.3″ seeing). A galaxy-scale Einstein
  radius of θ_E ≈ 1″ spans ~32 px here, versus 4–8 px in the DESI rubric's "small arcs"
  caveat. Arcs, rings, knots and counter-images that are blurred into a blob at ground
  resolution are RESOLVED here: judge the morphology you actually see, and do not grant
  credit for "could be an arc under the seeing" — at this resolution an arc either shows its
  tangential curvature and surface-brightness structure, or it is not an arc.
- The penalty runs the other way too: a feature that looks "arc-like" in a 4-px DESI blob
  but resolves into a spiral arm, a tidal tail, a star-forming clump or a diffraction spike
  here is a NON-lens.

## The single composite you are shown (6 panels, 2 rows; N up, E left in every panel)
- Row 1, 10″ field: (a) NORMAL stretch, (b) DEEP stretch (faint low-surface-brightness
  features, cores saturate), (c) two-band pseudo-COLOUR (red = long-wavelength channel,
  blue = short-wavelength channel, green = their mean).
- Row 2, 3.5″ zoom on the catalogued galaxy: (d) DEEP, (e) two-band COLOUR,
  (f) DEFLECTOR-SUBTRACTED residual (a radial-profile model of the central galaxy removed —
  the best panel for faint arcs close to the core, but see the over-subtraction caveat).
- A 1″ white scale bar sits in each row; a green N/E compass and corner quadrant labels give
  orientation; four small YELLOW ticks point at the catalogued galaxy (the putative deflector).
  Use them to locate the object being graded — the deflector is at the tick centre, not
  necessarily the brightest thing in the field.
- A minority of composites lack one channel; those show extra grayscale renderings in place
  of the two colour panels and carry no colour information at all.
- Colour is TWO-band only (one SW + one LW filter), so red/blue contrast is weaker and less
  diagnostic than in DESI grz. An old deflector population is typically red/orange here and a
  lensed star-forming source blue/white, but treat colour as supporting evidence, never the
  deciding criterion; weight geometry (tangential curvature at fixed radius, counter-images,
  surface-brightness structure) far more heavily than colour.

## What does NOT exist for this candidate
- NO CNN / ML scores, NO aperture photometry, NO Tractor type, NO coordinates or names. Do
  not call tools (none are attached) and do not ask for them. The `get_photometry` and
  `fetch_cutout` instructions in the rubric are void for this candidate: the composite is
  the complete evidence set.
- Do not infer anything from the object's appearance of "being in a survey": every
  candidate here was selected from the same catalogue of massive red galaxies by the same
  pipeline, lenses and non-lenses alike.

## Grades: the Huang visual-inspection scale (A/B/C/D)
These are the group's VI letters (Huang et al. 2020/2021): A = almost certainly a lens,
B = probable, C = possible (evidence neither confirms nor refutes), D = not a lens. They are
NOT a pass count. Any reference examples in the user message carry a single expert's score on
the same 4-point scale (4 = A, 3 = B, 2 = C, 1 = D) plus that expert's confidence in it;
calibrate to those, not to the CNN-prior language of the DESI rubric.

## JWST-specific false positives — rule these out explicitly
- **6-spike diffraction pattern** from a bright star or a bright compact core: straight
  radial spikes at fixed angles, not curved, not at fixed radius → D (`star_halo`).
- **Spiral arms** of a face-on or inclined disc: attached to the centre, winding inward, with
  star-forming knots along them — at JWST resolution arms are obviously arms; grade D
  (`spiral`) even when a single arm segment happens to be tangential.
- **Collisional / resonance rings** and **shells**: a complete, smooth ring of the SAME colour
  and stellar population as the host, centred on it, often with a visible companion that
  punched through → D (`ring_galaxy`). An Einstein ring is typically bluer, broken /
  asymmetric, thinner than its radius, with matching-colour counter-arcs.
- **Tidal features and mergers**: tails, bridges, plumes, overlapping discs, double nuclei —
  extended, low-surface-brightness, NOT confined to a fixed radius around one galaxy → D
  (`merger`).
- **Symmetric over-subtraction residuals** in panel (f): the radial-profile model leaves
  concentric positive/negative rings and butterfly-shaped residuals around ANY galaxy with an
  ellipticity, a bar, or a disc. A ring in panel (f) that is absent from panels (d)/(e), or
  that is perfectly symmetric about the core, is a modelling artefact, not an arc. Require
  the arc to be visible in at least one un-subtracted panel too.
- **Busy fields / chance alignments**: a faint blue galaxy near a red one, with no tangential
  elongation and no counter-image, is a projection → C at best, usually D.
- **Reduction artefacts**: detector gaps, snowballs, persistence, and 1/f striping can create
  straight or rectangular features — never tangential at fixed radius → D (`other`).

## Calibration guidance
- Default expectation remains C or D. Reserve A/B for tangential arcs / counter-images /
  rings that survive every family above and are visible in more than one panel.
- `p_lens` is your probability that this is a true strong lens; `confidence` is how sure
  you are of the LETTER. Use the full range of both. Set `escalate_to_human=true` when a
  reviewer or deeper data could resolve a genuine ambiguity, not as a hedge on every C.
- The criteria scores (`blue_source`, `low_surface_brightness`, `curvature`,
  `counter_images`, `arc_morphology`) keep their meaning; score them from the composite.
