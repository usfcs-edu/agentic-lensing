# P3 recon — Euclid Q1 demo targets + COOLEST export (2026-07-08)

De-risk recon for Phase P3 (CGL recipe end-to-end + 2–3 Euclid Q1 lenses + COOLEST export).
Persisted for the P3 implementation agent. Source: recon agent, verified against on-disk data.

## SCOPING DECISION (load-bearing)
**Euclid VIS is native 0.1″/px (VIS detector plate scale), NOT drizzle-resampled** → its RMS
map is diagonal/instrumental. Therefore **P3-Euclid is a DIAGONAL-likelihood demonstration of
the SAMPLER recipe (Pillar 2), not of the P1 correlated likelihood.** (Only the NIR bands,
resampled 0.3″→0.1″, would carry resampling correlation — and we fit VIS only.) This is the
cleaner framing: Euclid independently validates the recipe on real data; the correlated
likelihood's real-data test is the HST cross-scale experiment (P1c). Report §8 states this.

## Data (reproductions/euclid-q1/data, = cgl.paths.EUCLID_Q1_DATA)
- `lens/` = 336 galaxy-scale dirs, 322 with SIE model, **185 grade-A with model** = the P3 pool.
  `group/` (43) = multi-deflector, AVOID. `unsuccess/` (145) = no model. `recenter/` (18) =
  lens-light-recentered variants (useful for the centering offset).
- Cutout `{id}.fits`: 13 ext = PRIMARY + {VIS,NIR_Y,NIR_J,NIR_H}×{FLUX,PSF,RMS}. FLUX/RMS
  300×300 @0.1″; VIS_PSF 21×21, NIR_PSF 33×33. RMS = per-px σ (~0.0041) with 1e16 sentinel on
  ~1.6% bad px. Load VIS: img=VIS_FLUX, err=VIS_RMS, psf=VIS_PSF/psf.sum(),
  keep=(RMS<1e15)&circle(r=mask_radius/0.1=40px, center)&~mask_extra_galaxies.
- `info.json`: mask_radius=4.0″, mask_centre=[0,0]. Published model = PyAutoLens/autofit,
  single-plane: `result_lens_mass.json` has einstein_radius AND einstein_radius_effective
  (area-equiv, convention-robust — USE THIS), ell_comps_0/1, centre_0/1, shear_gamma_1/2;
  arcsec; origin=cutout center; autolens (y,x) order. z_lens=0.5/z_src=1.0 FIXED defaults
  (only angular θ_E meaningful). Source = pixelized (we don't reproduce; anchor on mass θ_E).
  `raw/modeling_lens_mass.csv` = bulk θ_E table keyed by id_str.

## Flagship trio (grade-A, clean single-deflector, θ_E,eff arcsec)
| ID | θ_E,eff | q | |shear| | VIS S/N | note |
|---|---|---|---|---|---|
| 102020061_NEG607087127495633316 | 1.24 | 0.79 | 0.043 | 294 | flagship: mid θ_E, high S/N, low shear |
| 102157952_2658211530641487553 | 1.12 | 0.82 | 0.057 | 425 | highest S/N clean system |
| 102157958_2719195933641972975 | 0.85 | 0.56 | 0.084 | 241 | small-θ_E / elliptical end |
Alternates: 102044825_… (θ_E 1.08, shear 0.021), 102159192_… (shear 0.005),
102158274_… (q 0.88 round), 102159486_… (θ_E 1.92). Confirm morphology via rgb/VIS png first.

## Fit-gap vs HST F140W pipeline
1. delta_pix 0.13→0.1; meta psf_pixel_scale=0.1 → guard passes (renormalize PSF to sum=1;
   raw sum ≈0.956). VERIFY PSF not oversampled via MAP chi2_pp-floor test (foundry-i R0c
   signature); resample if needed.
2. **Priors MUST be re-centered** (the one real code change): current _build_prior hardcodes
   theta_E~LogNormal(log2.5,0.25) [3σ off a ~1″ lens] + mass.center~N(0,0.02) [too tight vs
   ~0.1″ cutout offset]. Parameterize build_marg_model with theta_E_med≈1.0, mass_center_sig
   ~0.1, near_xy off-field for isolated targets — defaults preserve HST parity bit-for-bit.
3. Monochromatic: fit VIS only.
4. Diagonal likelihood (whiten_fn=None) — correct + cheaper than E2's correlated fits.

## COOLEST export (coolest 0.1.11)
- Container COOLEST(mode='MAP', coordinates_origin, lensing_entities, observation, instrument,
  cosmology, ...). Profiles coolest.template.classes.profiles.mass.{PEMD,SIE,ExternalShear},
  .light.{Sersic,Shapelets,...}. Galaxy(light_model=LightModel('Sersic'),
  mass_model=MassModel('PEMD','ExternalShear')); MassField for external shear;
  LensingEntityList; Instrument(pixel_size=0.1, psf=PixelatedPSF); Observation; Cosmology.
  Params: profile.parameters['x'].point_estimate=PointEstimate(value) +
  .posterior_stats=PosteriorStatistics(mean,median,pctl16,pctl84).
  Serialize: JSONSerializer(path_no_ext, obj, indent=2).dump_simple().
- Map: EPL(theta_E,gamma,e1,e2,cx,cy)→PEMD(theta_E,gamma,q,phi,center) [check gamma def:
  gigalens 3D slope ↔ PEMD slope at θ_E]; shear(g1,g2)→ExternalShear(gamma_ext=hypot,
  phi_ext=½atan2 — check PA sign vs east-of-north); 4 lens-light + source Sérsic→Sersic;
  source shapelets→Shapelets(n_max, beta, amps=a_star[28]). Posteriors from PHMC chains.
- Ridge-marginalized shapelet amps: not a sampled block → export marginal-mode a_star at MAP z
  into Shapelets.amps as a MAP file + ship chains npz + MAP model-image FITS; note
  marginalization/logdetA in metadata. Comparison to published model is at MASS level (θ_E).

## Execution plan
- cgl/euclid_io.py: load_euclid_vis(id_str) → cgl.paths.load_product layout (build_marg_model
  consumes unchanged).
- extend build_marg_model/_build_prior with theta_E_med/mass_center_sig/near_xy overrides
  (defaults = HST parity path unchanged — do NOT break P0 parity).
- 30_recipe_e2e.py: CGL recipe end-to-end on HST v2-class system, reuse P1c infra
  (cgl.e2 build_target/map_polish/laplace_evidence/run_staged), --sampler = P2c winner.
- 31_fit_euclid.py: per target, diagonal build_marg_model(delta_pix=0.1, theta_E_med≈1.0) →
  map_polish → Laplace → run_staged (two-stage PHMC, 24 chains; unimodal galaxy-scale → PHMC
  is the reliable default; nautilus only as logZ cross-check — it's P2b-disqualified on 46-dim
  marg). Writes θ_E(ours) vs einstein_radius_effective(published).
- 32_coolest_export.py: posterior npz + MAP z → COOLEST 'MAP' JSON + MAP model-image FITS.
- Compute: r=4″ mask ≈80² @ss2 ≈ v2d size, diagonal cheaper than E2. 3 targets ≈ a few L4-h
  on PHOENIX. NO Perlmutter for Euclid. HST recipe run likely fits on phoenix native too.
  P3 ≤15 A100-h budget mostly UNUSED.
- Claim = "recovers a consistent θ_E + valid COOLEST export", NOT a digit-match (expect ~5–10%
  PyAutoLens-vs-gigalens convention offset on θ_E def + ell_comps/PA).

## Blockers: none hard. Required: prior re-centering; PSF-oversampling verify + renormalize;
## cutout ~0.1″ centering (recenter or loosen prior). Source-model mismatch expected/fine.
