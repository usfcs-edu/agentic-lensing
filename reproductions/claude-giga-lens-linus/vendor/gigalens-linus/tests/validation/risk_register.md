# Physics risk register — validation traceability

Every physics risk flagged during the API-rework design, mapped to the test(s) that
guard it. "The suite guards against all flagged errors" is only a real claim if it
is auditable; this is the audit. Update it whenever a risk or test changes.

Status legend: **live** = test implemented & runnable against current code;
**pending** = test written but skipped until the new API / trace exists (Phase ref).

| # | Risk (what could be silently wrong) | Guarding test(s) | Oracle | Status |
|---|---|---|---|---|
| R1 | Multiplane per-plane mass normalization — a plane-`k` lens's `theta_E` is degenerate with distance scaling; treating `deriv` as "reduced α" × `Ds/D_{ks}` may misnormalize | `test_multiplane.py::test_two_plane_ray_shooting_vs_lenstronomy`, `::test_single_lens_multiplane_equals_singleplane` | lenstronomy `LensModel(multi_plane=True)` | pending (P3) |
| R2 | Transverse vs LOS comoving distance / curvature for non-flat cosmology | `test_cosmology.py::test_nonflat_transverse_comoving_distance_vs_astropy`, `::test_nonflat_angular_diameter_distance_vs_astropy`, `::test_distance_matrix_vs_astropy_flat` | astropy `wCDM` | **RESOLVED (Phase 1)** — curvature (Hogg eq.16/19) implemented in `CosmoBase`; non-flat D_M/D_A match astropy to ~1e-6. xfail ratchet retired. |
| R3 | Distance-integral accuracy — 1000-pt trapezoid may be too coarse | `test_cosmology.py::test_distance_integral_self_convergence` (+ flat vs-astropy tests) | astropy / self | live (passing) |
| R4 | Single-deflector ⇄ multiplane reduction — N=1 multiplane must equal the single-plane path exactly | `test_multiplane.py::test_single_lens_multiplane_equals_singleplane`, `test_metamorphic.py::test_multiplane_N1_reduces_to_single_plane` | gigalens single-plane + lenstronomy | pending (P3) |
| R5 | `deflection_ratio` ≡ multiplane for one lens + several source redshifts | `test_multiplane.py::test_deflection_ratio_equals_multiplane_single_lens` | self-consistency + astropy distances | pending (P3) |
| R6 | PSF convolution + supersample/pooling numerics | `test_single_plane.py::test_full_image_vs_lenstronomy`, `::test_supersample_converges` | lenstronomy / Richardson | pending (P3) |
| R7 | float32/float64 consistency & float64 noise floor | `test_single_plane.py::test_precision_modes_agree`, profile tests parametrized over dtype | self (cross-dtype) | pending (P3) |
| R8 | Coordinate-grid orientation/units & pixel-area `conversion_factor` | `test_coordinates.py::test_pixel_grid_matches_wcs`, `::test_flux_scales_with_pixel_area` | lenstronomy pixel grid / analytic | live (grid) / pending (image) |
| R9 | lstsq amplitude recovery — solver must recover known amplitudes from a noiseless render | `test_scene_simulator.py::test_lstsq_recovers_known_amplitudes` | analytic round-trip | live (passing; coeff = Ie·pixel_area) |
| R10 | No-silent-defaults guards actually raise (missing distance/noise, bare-dist reuse, dead entity, forward-mode flux share) | `test_no_silent_defaults.py` (§3.1–§3.4 now live & passing); poisoned-content (NaN/≤0 noise) in `test_redteam_dataset.py`; see `misuse_register.md` | spec §3 | live (passing); multiplane-nonflat render still pending (P3) |
| R11 | Profile parity at float64 (existing suite only checks float32) | `test_profiles_f64.py` (extends `tests/test_profiles.py`) | lenstronomy | live |
| R12 | Multiplane light placement — light on intermediate planes rendered at the correct deflected position | `test_multiplane.py::test_intermediate_plane_light_position` | lenstronomy multi-plane image | pending (P3) |

Unforeseen-error coverage (no single risk row — these are the safety net):
`test_metamorphic.py` (equivariance, linearity, no-op/reorder invariances) +
randomized differential sweeps over seeded valid parameter draws (`conftest.py`).
