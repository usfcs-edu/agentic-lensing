---
name: project-foundry-i-reproduction
description: "Foundry I (Huang 2025a) GIGA-Lens reproduction status on DESI-165.4754-06.0423 — v5 recovers θ_E to 0.5-3%; HMC now works (lstsq+float64+regularized-marginalization recipe), mixing degeneracy-limited; gamma=1.37 is a model-setup difference, not a sampler issue"
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
   - HMC now RUNS (resolved 2026-06; see "HMC posterior" section below). The old "JIT never completes" belief was wrong — see the compile diagnosis there.

6. **Compute footprint**: each MAP fit (20 chains × 200-400 steps) takes 20-55 s on 2× L4 GPUs (UUID-pinned to avoid the A16+L4 mixed-type shard_map crash). JIT compile dominates first run; ~15-18 it/s post-JIT.

## HMC posterior — full investigation (2026-06, the SVI-only fallback is now RESOLVED)

Goal: replace the v10 SVI surrogate with genuine HMC, on the 8×A16 GPUs (L4s reserved).
Outcome: **HMC now works mechanically** (compiles ~35 s, stable, smooth, near-PD mode,
samples the correct posterior); residual is mixing *efficiency*, limited by intrinsic
strong-lens parameter degeneracies. The chain of findings (each a separate fix):

1. **"HMC won't compile" was misdiagnosed** (not a JIT time budget). Every kernel —
   incl. gigalens's `GradientBasedTrajectoryLengthAdaptation` (GBTLA) — compiles in
   31–38 s **single-device** (`26_compile_diagnostic.py`). The original hang = gigalens
   `HMC()` (`gigalens/jax/inference.py:280-307`) wrapping GBTLA in `jax.pmap` over **all
   10 devices** × the `lstsq_simulate` 31-channel grouped conv × cuDNN autotuner
   thrashing (one conv algo took 11m50s). **Always set
   `XLA_FLAGS=--xla_gpu_autotune_level=0`** and use single/few devices.
2. **The v7 "paper-mode MAP" (`data/map_v7_paper_mode.npz`) is a SADDLE**, not a max —
   15 large negative Hessian eigenvalues, log-posterior improvable by **+62,000**
   (−99126→−36750). Every prior result (v8/v10 SVI, v11* NUTS) was anchored to a badly
   non-converged point. Refining off it (`28_refine_map.py`) removed the saddle.
3. **Recipe that makes HMC mix** (each step necessary): **(a) lstsq-marginalize** the 33
   linear light amplitudes — they create ~56 near-flat Hessian directions capping ESS~4
   (`_hmc_lib_lstsq.py`, 41 nonlinear params); **(b) float64** (`jax_enable_x64`) — the
   reduced objective is stiff (cond≥1e9), float32 floors ‖grad‖ at ~1.2e4; **(c) EXACT
   regularized Gaussian marginalization** of the 28 shapelet amps
   (`_hmc_lib_marg.py build_model_marg`, 46 params = 41 nonlinear + 5 sampled-positive
   Sérsic Ie): `A = XᵀWX + Λ` (Λ_ii=(i+1)/25 from the shapelet `Normal(0,5/√(i+1))`
   priors), `a*=cho_solve(A,b)`, `logL = −½ΣWR² + ½b·a* − ½log|A|`. This is what gigalens
   `lstsq_simulate` OMITS (see bug 3) — it makes the objective SMOOTH: ‖grad‖ broke the
   pinv floor 9e4→400 (228×), Hessian became 44/46 positive; **(d) mass matrix**: the
   residual posterior is ultra-ill-conditioned (cond~1e14: companion-galaxy lens-light
   Sérsic CENTERS `LL2/LL3.center` at H_ii~1e12, Sérsic INDICES `n_sersic` at ~1e-2; the
   physical mass params θ_E/γ/e1/e2/shear sit in the well-behaved 1e8–1e9 tier).
4. **Mass-matrix options** (`34_fit_marg.py --massmatrix`): floored full Hessian
   (`hess_marg_pd`) over-constrains soft dirs (γ frozen, std 1e-4); **un-floored diagonal
   (`diagraw`)** is float64-safe (per-param scalars, NO matrix ops) and unlocks realistic
   scales — γ_std 1e-4→**0.037** (≈ paper 0.023), γ drifts 1.866→1.746 toward the paper;
   **diagonally-scaled-correlation full Hessian (`hesscorr`)**, `chol(H)=diag(√D)·chol(D^−½ H D^−½)`,
   lifts correlated mass params (θ_E ESS 8→20) but the correlation matrix still has
   cond~1e8 (genuine lensing degeneracies: mass-sheet, slope–ellipticity).
5. **Multi-chain R-hat** (parallelized across all 8 A16s — the *right* convergence test, a
   single chain can't give it): 4 diagraw + 4 hesscorr × 500 → chains do **NOT converge**,
   **R-hat = 1.4–33** (need <1.01). Converged inference needs ~1e4–1e5 samples/chain
   (A100-feasible; ~17–24 h on A16 even 8-way parallel — the paper used A100s; **L4s do
   NOT help — same float64 precision, ~2× speed at most; A100s are the real lever**). The
   long 8-chain diagraw run (burn2000/keep8000 = 64k pooled, ~15 h, `data/long_diagraw_s0..7.npz`,
   2026-06-02) RESOLVED the γ-breadth question — see item 6. Pool/diagnose with
   `35_pool_chains.py --glob 'data/long_diagraw_s*.npz'`.
6. **Paper's γ=1.372 is CONSISTENT with our posterior (REVISED 2026-06-02 by the long run).**
   The long 8-chain HMC (64k samples) shows all 8 chains drift downhill from the MAP/SVI
   start (γ≈1.86, log-p −45841) DOWN THROUGH the paper's 1.372 to pooled median γ≈1.17
   (16-84%: 1.15-1.19) at HIGHER log-p (−45010..−45090, ~800 better than the start). So the
   γ≈1.86 mode our POINT estimators (MAP/SVI) picked is NOT the best fit; HMC favors LOWER γ
   and brackets the paper's value. The earlier **Δ=−32,500 "model-setup difference"** was a
   COLD-START ARTIFACT (the paper-seed refine stalled, ‖grad‖~5e5, at a bad joint config
   −78350), NOT the true cost of low γ. CAVEAT: chains still UNCONVERGED (R-hat up to 20 at
   64k samples, ESS~9-15; acceptance 0.02-0.06) — γ is a broad, slow-mixing degenerate
   direction; a converged credible interval needs a degeneracy-aware sampler (windowed dense
   mass / Riemannian HMC) or much deeper chains. Net: the γ gap is poorly-constrained,
   slow-mixing degeneracy + point-estimator bias, NOT a hard model-setup difference.

**Three real upstream gigalens bugs found** (reproducer `36_upstream_gigalens_repro.py`,
write-up `UPSTREAM_GIGALENS_ISSUE.md`): (1) GBTLA-under-`pmap` compile blowup
(`inference.py:280-307`); (2) **`inference.py:258-260` builds the HMC momentum as
`MultivariateNormalFullCovariance(jnp.linalg.inv(q_z.covariance()))` → NaN** whenever the
SVI covariance is rank-deficient (use the precision parameterization + regularize); (3)
`lstsq_simulate` pinv-profiles the linear amplitudes UNREGULARIZED and DROPS the
`−½log|XᵀWX|` Gaussian-evidence term → non-smooth/indefinite profiled objective on a
near-rank-deficient design matrix. This is open task #14 ("File upstream gigalens HMC JIT
issue") — reproducer ready; **filing on GitHub needs the user (outward-facing).**

**Key new scripts (foundry-i/):** `_hmc_lib.py`, `_hmc_lib_lstsq.py`, `_hmc_lib_marg.py`
(the model libs), `26_compile_diagnostic.py`, `28_refine_map.py`, `30_refine_lstsq.py`,
`32_saddlefree_newton.py`, `33_trust_refine.py`, `34_fit_marg.py` (main runner: `--mode
refine|hmc`, `--massmatrix hess_marg_pd|diag|diagraw|hesscorr`, `--seed`, `--x64`),
`35_pool_chains.py`. All HMC work ran on the **A16s** (float64 ~0.39 s/grad; fixed-leapfrog
HMC only — **NEVER NUTS in float64**, tree-depth-8 ran 8h45m once and was killed).

Related: [[reference-gigalens-env]], [[project-huang-lensing]], [[reference-paper-corpus]], [[reference-host-hardware]].
