// GENERATED from astromark-vocab/lens-1.0. Do not edit.

export type Role = "lens:deflector" | "lens:second_deflector" | "lens:satellite" | "lens:lensed_image" | "lens:arc" | "lens:counter_image" | "lens:counter_arc" | "lens:knot" | "lens:secondary_ring" | "lens:dust_lane" | "lens:companion_galaxy" | "lens:nearby_galaxy" | "lens:field_galaxy" | "lens:star" | "lens:diffuse_candidate" | "lens:ambiguous_structure" | "lens:artifact" | "lens:einstein_ring" | "lens:lens_light" | "lens:lensed_light";
export type Polarity = "core:positive" | "core:negative" | "core:ambiguous";
export type Alternative = "lens:spiral_arm" | "lens:ring_galaxy" | "lens:shell_tidal" | "lens:merger" | "lens:edge_on_disk" | "lens:companion_projection" | "lens:star_forming_clump" | "lens:lens_galaxy_as_image" | "core:diffraction_spike" | "core:detector_artifact" | "core:subtraction_residual" | "core:psf_wing" | "lens:scale_tension" | "core:other";
export type HardCase = "lens:dust_lane_case" | "lens:second_deflector_case" | "lens:merging_pair" | "lens:faint_counter_image" | "lens:counter_image_outside_arc" | "lens:two_sources" | "lens:single_giant_arc" | "lens:group_scale" | "lens:low_snr" | "lens:arc_obscured" | "lens:lens_light_dominates";
export type ThetaEMethod = "lens:arc_midline" | "lens:arc_bounds" | "lens:half_separation" | "lens:ring_mean" | "lens:model" | "lens:human";
export type Treatment = "core:mask" | "core:model";
export type CounterImageSearch = "core:found" | "core:not_found" | "core:not_searched";
export type SourceConfig = "lens:double" | "lens:quad" | "lens:cusp" | "lens:fold" | "lens:cross" | "lens:ring" | "lens:partial_ring" | "core:unknown";
export type Emphasis = "core:muted" | "core:normal" | "core:key";
export type ReviewVerdict = "core:correct" | "core:wrong_position" | "core:wrong_label" | "core:wrong_type" | "core:wrong_size" | "core:spurious" | "core:redundant" | "core:missed_by_model";
