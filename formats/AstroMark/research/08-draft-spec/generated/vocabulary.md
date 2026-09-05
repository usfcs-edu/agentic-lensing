# astromark-vocab/lens-1.0 — term tables

GENERATED from `astromark-vocab-lens-1.0.json`. Do not edit.

## role

What a mark asserts the feature IS. Orthogonal to polarity: a role says what kind of thing this would be, polarity says whether it supports the lensing interpretation.

| term | label | definition | status |
|---|---|---|---|
| `lens:deflector` | deflector | The foreground galaxy whose mass bends the light. The whole galaxy, including its extended halo — not only the bright core. | stable |
| `lens:second_deflector` | second deflector | An additional galaxy contributing lensing mass to the same system. Labelled as a lens, never as a lensed image. | stable |
| `lens:satellite` | satellite | A small galaxy gravitationally bound to the deflector. A separate light and mass component for modelling, and never a mask target. | stable |
| `lens:lensed_image` | lensed image | Light originating behind the deflector and deflected by it. The general class; an arc is the tangentially extended special case. | stable |
| `lens:arc` | arc | A tangentially stretched lensed image, curved about the deflector. | stable |
| `lens:counter_image` | counter-image | The image of the SAME source lying across the deflector from the main image. It may lie at a greater radius than the main image; radius is never a ground for rejecting it. | stable |
| `lens:counter_arc` | counter-arc | A counter-image that is itself tangentially extended. | stable |
| `lens:knot` | knot | A bright compact star-forming region WITHIN a lensed image. A whole compact lensed image is a lensed image, not a knot. | stable |
| `lens:secondary_ring` | secondary ring | The small ring a secondary deflector imprints on already-lensed light. | stable |
| `lens:dust_lane` | dust lane | An obscuring band inside the deflector that can hide part of a lensed image. A reason the lensed light may be partly invisible; NOT a reason against lensing. | stable |
| `lens:companion_galaxy` | companion galaxy | A galaxy near the system that is not lens mass and not lensed. Applied to a counter-image candidate only after the same-source test has failed. | stable |
| `lens:nearby_galaxy` | nearby galaxy | The brightest field galaxy close enough that its light reaches the lens system, so the modeller must choose between fitting and masking it. | stable |
| `lens:field_galaxy` | field galaxy | An unrelated extended object to be treated during modelling. | stable |
| `lens:star` | star | A point source, usually with diffraction spikes. | stable |
| `lens:diffuse_candidate` | diffuse candidate | Bluish, spread-out light near the system, unlike a compact field galaxy. A flag for the modeller rather than a claim. | provisional |
| `lens:ambiguous_structure` | ambiguous structure | A feature whose class cannot be settled from the imaging available. | stable |
| `lens:artifact` | artifact | A detector or processing artefact: a spike, a hot pixel, a ghost, a subtraction residual. | stable |
| `lens:einstein_ring` | Einstein ring | The measured Einstein radius, drawn as a circle. A MEASUREMENT, not a feature: its radius is the datum. | stable |
| `lens:lens_light` | lens light | A segmentation region covering the deflector's light. Deliberately tolerant — about half the visible halo, not the faintest isophote. | provisional |
| `lens:lensed_light` | lensed light | A segmentation region covering lensed light, including faint opposite images. Recall matters here in a way it does not for lens light. | provisional |

## polarity

What the mark does to the lensing interpretation. Present only on roles whose `takes_polarity` is true. Independent of role: role=arc with polarity=negative and alternative=spiral_arm reads 'this arc-like thing is a spiral arm'.

| term | label | definition | status |
|---|---|---|---|
| `core:positive` | positive | The mark ASSERTS rather than refutes: this feature is here and is what the role says it is. It does not additionally claim to prove lensing. | stable |
| `core:negative` | negative | Argues AGAINST lensing, with a named alternative reading. Requires `alternative`. | stable |
| `core:ambiguous` | ambiguous | Could be lensing or the named alternative. The natural class for a grade-C system's evidence. Requires `alternative`. | stable |

## alternative

The non-lens reading of a feature. Required when polarity is negative or ambiguous.

| term | label | definition | status |
|---|---|---|---|
| `lens:spiral_arm` | spiral arm | An arm of a face-on spiral, curved about its own bulge rather than about a foreground mass. | stable |
| `lens:ring_galaxy` | ring galaxy | An intrinsic stellar ring, not a lensed image. | stable |
| `lens:shell_tidal` | shell or tidal feature | A shell or tidal tail from a past interaction. | stable |
| `lens:merger` | merger | An interacting pair mimicking multiple images. | stable |
| `lens:edge_on_disk` | edge-on disk | An edge-on disk galaxy mimicking an arc. | stable |
| `lens:companion_projection` | chance projection | An unrelated object that happens to lie across the deflector. | stable |
| `lens:star_forming_clump` | star-forming clump | A clump in the deflector itself. | stable |
| `lens:lens_galaxy_as_image` | lens galaxy read as an image | A lensing galaxy mistaken for a lensed image. The known model failure mode. | stable |
| `core:diffraction_spike` | diffraction spike | A spike from a bright star. | stable |
| `core:detector_artifact` | detector artifact | A defect of the detector or the exposure. | stable |
| `core:subtraction_residual` | subtraction residual | A residual left by an imperfect model subtraction. | stable |
| `core:psf_wing` | PSF wing | The wing of a bright source's point-spread function. | stable |
| `lens:scale_tension` | scale tension | The deflector's apparent mass is implausible for the Einstein radius. SUPPORTING evidence only; never a verdict on its own. | stable |
| `core:other` | other | Named in free text. Its presence in a corpus is a signal that a term is missing. | stable |

## hard_case

Morphologies that make a system hard to read but are still lenses. Emitted verbatim by the exemplar exporter so rare cases can be found and stratified.

| term | label | definition | status |
|---|---|---|---|
| `lens:dust_lane_case` | dust lane |  | stable |
| `lens:second_deflector_case` | second deflector |  | stable |
| `lens:merging_pair` | merging pair | Two merged images; the same colour at both tips is the mirror-parity signature, which is positive evidence. | stable |
| `lens:faint_counter_image` | faint counter-image |  | stable |
| `lens:counter_image_outside_arc` | counter-image outside the arc |  | stable |
| `lens:two_sources` | two sources |  | stable |
| `lens:single_giant_arc` | single giant arc | One very large, clearly tangentially stretched arc with no counterpart found after a search. | stable |
| `lens:group_scale` | group scale |  | stable |
| `lens:low_snr` | low signal to noise |  | stable |
| `lens:arc_obscured` | arc obscured |  | stable |
| `lens:lens_light_dominates` | lens light dominates |  | stable |

## theta_e_method

How the Einstein radius was estimated. Replaces a free string.

| term | label | definition | status |
|---|---|---|---|
| `lens:arc_midline` | arc midline | Radius to the radial MIDPOINT of the main lensed image. The preferred rule. | stable |
| `lens:arc_bounds` | arc bounds | From the inner and outer edges of the main image, giving the lower and upper bounds. | stable |
| `lens:half_separation` | half separation | Half the arc-to-opposite-image distance. Secondary, and a cross-check only: the opposite image may lie farther out than the arc. | stable |
| `lens:ring_mean` | ring mean | Mean radius of a full or partial ring. | stable |
| `lens:model` | model | From a lens model fit. | stable |
| `lens:human` | human | Set by an expert without a stated rule. | stable |
| `lens:geometric` | geometric | Retained because the nine example decks use it. Mapped on load. | deprecated → `lens:arc_midline` |
| `lens:arc_radius` | arc radius | Retained for the same reason. 'The arc's radius' names no specific part of the arc, which is why it was replaced. | deprecated → `lens:arc_midline` |

## treatment

What a modeller should do with a field object.

| term | label | definition | status |
|---|---|---|---|
| `core:mask` | mask | Exclude these pixels from the fit. | stable |
| `core:model` | model | The object's light reaches the lens system, so fit it as its own component rather than cutting it out. Extended objects only. | stable |

## counter_image_search

Three-valued, because 'not found' must be distinguishable from 'did not look'. In a structured-output schema a null and a missing key are the same thing to many model outputs, so this is an explicit enum.

| term | label | definition | status |
|---|---|---|---|
| `core:found` | found |  | stable |
| `core:not_found` | not found | Searched, at all radii out to the cutout edge, and none was located. | stable |
| `core:not_searched` | not searched |  | stable |

## source_config

The image configuration of one source.

| term | label | definition | status |
|---|---|---|---|
| `lens:double` | double |  | stable |
| `lens:quad` | quad |  | stable |
| `lens:cusp` | cusp |  | stable |
| `lens:fold` | fold |  | stable |
| `lens:cross` | cross |  | stable |
| `lens:ring` | ring |  | stable |
| `lens:partial_ring` | partial ring |  | stable |
| `core:unknown` | unknown |  | stable |

## emphasis

PRESENTATIONAL only. Scales stroke weight and glyph size; never a measured radius or a polygon vertex. Ordinal, three values — a continuous multiplier would invite arbitrary values and make cross-panel comparison meaningless.

| term | label | definition | status |
|---|---|---|---|
| `core:muted` | muted |  | stable |
| `core:normal` | normal |  | stable |
| `core:key` | key | At most one per panel, lint-warned above that. | stable |

## review_verdict

A reviewer's judgement of one proposed mark.

| term | label | definition | status |
|---|---|---|---|
| `core:correct` | correct |  | stable |
| `core:wrong_position` | wrong position |  | stable |
| `core:wrong_label` | wrong label |  | stable |
| `core:wrong_type` | wrong type |  | stable |
| `core:wrong_size` | wrong size |  | stable |
| `core:spurious` | spurious |  | stable |
| `core:redundant` | redundant |  | stable |
| `core:missed_by_model` | missed by model | On a human-added mark: the recall signal. | stable |
