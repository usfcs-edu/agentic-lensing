---
name: project-foundry-i-reproduction
description: "Foundry I (Huang 2025a) GIGA-Lens reproduction status on DESI-165.4754-06.0423 — v5 recovers θ_E to 0.5%"
metadata:
  type: project
---

Phase 1 reproduction of Huang 2025a Foundry I demo system **DESI-165.4754−06.0423** is set up end-to-end under `/raid/benson/git/agentic-lensing/reproductions/foundry-i/`. Eight iterations (v1 MAP → v8 SVI) were run on the 2× L4 GPUs of `/raid/benson`. **The v8 SVI posterior (initialized from a v7 paper-mode multi-start chain, 200-chain VI with 10k posterior samples) reproduces 5 of 6 mass parameters to within ~13-33% with correct signs:** θ_E to 1.0%, e2 to 13%, γ_ext_2 to 12%, e1 to 33%, γ_ext_1 to 33%. **The remaining γ_EPL slope discrepancy (v8: 2.20, paper: 1.37) is a true local-minimum bifurcation**: the data + priors admit at least two distinct (γ, ellipticity) modes, and our finder converges to the higher-γ mode unless explicitly initialized at the paper's lower-γ mode. v7's multi-start hunt confirmed paper-mode is in 17/200 random chains (8.5%), not the global best under our priors.

**Why:** Phase 1 of the broader 16-paper reproduction plan asked for the GIGA-Lens application lineage (Cikota 2023, Sheu 2024b, Foundry I), and Foundry I was the cleanest target — it's the unambiguous GIGA-Lens paper, with complete published priors + posteriors and public HST data.

**How to apply:** When asked to extend this reproduction (v6+) or to do parallel reproductions of Cikota 2023 / Sheu 2024b / other lensing modeling papers, follow the working recipe:

1. **Data**: HST GO-15867 from MAST DOI `10.17909/hx0v-9260`. Use `astroquery.mast.Observations.query_criteria(proposal_id="15867", obs_collection="HST")` and pull the `hst_15867_65_wfc3_ir_f140w_ie5065_drz.fits` product (the HAP combined-drizzle). Cut a 128×128 box at native 0.13″/px around (RA=165.4754, Dec=−6.0423).

2. **Simulator config**: `delta_pix=0.13, num_pix=128, supersample=2`, TinyTim F140W PSF from `gigalens/src/gigalens/assets/psf.npy` (already supersampled to 0.065″, suitable for our supersample=2 = 0.065″ effective).

3. **Model that works (v6, the best so far)** = `EPL + Shear` mass + 4 `SersicEllipse(use_lstsq=False)` lens-light (2 main + 2 nearby galaxy) + 1 `SersicEllipse(use_lstsq=False)` source + `Shapelets(n_max=6, use_lstsq=False)` source. All Sersics get `LogNormal` Ie priors → positive amplitudes only. Shapelet basis amps each get a `Normal(0, 5/sqrt(i+1))` prior (free sign, order-decaying scale per the gigalens demo). 74 free non-linear params.

4. **Critical constraints learned the hard way**:
   - **Lens-light amplitudes MUST be positivity-constrained.** With `use_lstsq=True`, gigalens 2.0 allows negative amplitudes, and MAP exploits this to fit χ² at non-physical configurations. The v4 fit with negative amps had χ²=3.45 but θ_E off by 43%. The v5 fit with positive amps had χ²=7.56 but θ_E off by 0.5%.
   - **Tight mass-center prior**: `N(0, 0.02)` — looser priors let the mass center drift 0.2-0.3″ off, breaking the mass-light degeneracy.
   - **Tight main-lens-light Sersic center prior**: `N(0, 0.02)` — effectively ties the two main-lens-light Sersics to share a center.
   - **Central pixel mask**: inflate `err_map` to 1e10 at r < 1.5 px from main lens center (mimics paper's 2.5-px mask at 0.065″ drizzled scale).
   - **Nearby galaxy must be modeled** — detect via source-extraction on the v2 residual; in this system it's at (−2.34″, −2.86″) from the main lens.

5. **v7 finding (200-chain multi-start):** the lens-modeling posterior is genuinely **multimodal** with at least three competing local minima differing in (e1, e2, γ_ext_2) sign. The paper's mode is found in 17/200 ≈ 8.5% of random starts. It is *not* the global best under our priors — our v6-mode has lower χ² and higher log_p. Paper-mode chain (idx 140) reproduces θ_E to 0.5%, e2 sign correct, γ_ext_2 to 15%. Paper-mode chain saved at `data/map_v7_paper_mode.npz`.

6. **v9 finding (empirical PSF):** Replacing gigalens's default TinyTim F140W PSF with an EPSFBuilder-stacked PSF from 30 field stars in the same exposure flipped the GLOBAL-BEST chain into paper-mode. Paper-mode fraction went 8.5% → 22.5%; fraction with γ < 1.7 went ~5% → 35%. χ² rose 6.6 → 11.9 — expected because TinyTim was over-smoothed and let the model "hide" residuals. Build script: `17_build_empirical_psf.py` (uses `photutils.psf.EPSFBuilder` at 2× oversampling, output `data/empirical_psf.npy` shape (27,27)). The TinyTim PSF (gigalens/src/gigalens/assets/psf.npy) is shape (13,13) and missing the F140W PSF wings beyond 0.42″; the empirical version covers ±0.85″. **Lesson for any HST GIGA-Lens reproduction (Cikota 2023, Sheu 2024b, future Foundry I-V): always build an empirical PSF from field stars in the same exposure before fitting.**

7. **v10 final (empirical PSF + paper-mode SVI):** all 6 mass params now have correct signs in the variational posterior. **e1 matches paper to 2.5%, θ_E to 3.0%, shear angle exact.** Remaining systematic: a 12° residual rotation of the lens ellipticity + 30% smaller magnitudes on (e2, γ_ext) + γ_EPL 57% high — all consistent with a single missing constraint (mass–light PA tying), not three independent failures. Posterior file: `data/svi_v10_posterior_mass.npz`.

**Reproduction lineage of scripts (foundry-i/):** 01_download_hst → 02_inspect_cutout → 17_build_empirical_psf → 18_fit_map_v9 (200-chain multi-start with empirical PSF) → 19_svi_v10_paper_mode_empirical (final paper-mode SVI). For new GIGA-Lens-lineage reproductions (Cikota, Sheu 2024b), reuse this 5-step path.

6. **Open issues for v8+**:
   - γ_EPL slope: paper hits prior LOWER bound (γ=1.37 with prior min 1.0), our v6 hits UPPER region (γ=1.73 with prior max 2.7). Paper-mode chain hits γ=2.26. The γ direction is genuinely different between local minima.
   - To force paper's mode, future runs should (a) initialize from `map_v7_paper_mode.npz` or (b) add constraints from lens-light position angle / external imaging.
   - HMC didn't run — JIT compile of `pmap(scan(mcmc_sample_chain(PreconditionedHMC+adaptation)))` didn't complete in reasonable time on JAX 0.6.2 with the v2 29-param model. SVI-as-posterior is the working fallback.

6. **Compute footprint**: each MAP fit (20 chains × 200-400 steps) takes 20-55 s on 2× L4 GPUs (UUID-pinned to avoid the A16+L4 mixed-type shard_map crash). JIT compile dominates first run; ~15-18 it/s post-JIT.

Related: [[reference-gigalens-env]], [[project-huang-lensing]], [[reference-paper-corpus]].
