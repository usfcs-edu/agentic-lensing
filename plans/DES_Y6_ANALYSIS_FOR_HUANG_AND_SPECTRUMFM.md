# DES Y6 Dynamical Dark Energy

## Implications for the Huang Strong-Lensing Program and SpectrumFM

*Working memo, 2026-05-26.*

Source paper: **T. M. C. Abbott et al. (DES Collaboration), "Constraints on Dynamical Dark Energy from Multiple Probes in the Full Dark Energy Survey," arXiv:2605.27221v1 [astro-ph.CO], 26 May 2026.** Report numbers DES-2026-0979, FERMILAB-PUB-26-0306-PPD. PDF stored locally at `papers/Constraints_on_Dynamical_Dark_Energy_Probes_2605.27221v1.pdf`.

---

## 1. Executive Summary

**The DES result.** Combining the full six-year DES dataset (cosmic shear + galaxy–galaxy lensing + galaxy clustering, type Ia supernovae, and baryon acoustic oscillations) with DESI DR2 BAO and primary CMB anisotropies (Planck + ACT DR6 + SPT-3G DR1) yields a CPL-parameterized dark-energy equation of state

> *w*₀ = −0.82 ± 0.05, &nbsp; *w*ₐ = −0.63⁺⁰·²¹₋₀.₁₈,

a **3.0 σ departure** from the cosmological constant (*w*₀, *w*ₐ) = (−1, 0). The DES-only multi-probe combination already shows 2.2 σ, and the low-redshift-only combination (no CMB) shows 2.3 σ. Leaving out any single probe class still produces 2.3 σ–3.2 σ; the best-fit always lives in the *w*₀ > −1, *w*ₐ < 0 quadrant — the so-called "phantom-crossing" region that single-field quintessence cannot reach. This is consistent with, and now appreciably tighter than, the earlier DESI DR1, DESI DR2, and DES-SN-only hints.

**Why it matters for Huang's program.** The DES paper is, in effect, a public demonstration that *multiplying late-time probes* is how the dark-energy question now advances: adding 3×2pt weak lensing to SN+DESI-BAO **doubles** the dark-energy figure of merit (FoM 61 → 110). Strong-lens cosmography is statistically independent of cosmic-shear 3×2pt and statistically independent of SN photometric calibration — the two probes that anchor the DES Y6 result. Huang's program is positioned to deliver three distinct late-time probes (lensed-quasar time delays, lensed-SN Ia time delays, and static-lens *θ*ₑ–*σᵥ* statistics), all of which were named as future contributions in the introductions of *Foundry I* (Huang et al. 2025a) and *GIGA-Lens* (Gu et al. 2022) but none of which has yet flowed through to a *w*₀wₐ constraint paper. The DES Y6 result raises the strategic value of closing that gap.

**Why it matters for SpectrumFM.** The SpectrumFM proposal (v7) opens with "Massive spectroscopic surveys … are the backbone of precision cosmology, providing the data to map the universe's structure and constrain dark energy" but then never names BAO, *w*₀, *w*ₐ, CPL, or SN cosmology again. After DES Y6, the proposal sits at the operational center of the most informative late-time dark-energy stack — LRG/QSO redshifts feed DESI-DR2 BAO; SN typing feeds the Stage-IV SN Ia pipeline; lens identification (already in the proposal via the Hsu 2025 pairwise method) feeds two of the three strong-lens cosmography paths. None of these connections are made explicit in the current narrative. They should be.

**Recommended near-term actions** (developed in §9):

1. Spin up a Li-et-al-2024-style static-lens *w*₀wₐ analysis on the existing ~3,500-candidate H20/H21/S24 catalog plus DESI σᵥ measurements — feasible *before* Rubin first light.
2. Convert GIGA-Lens into a hierarchical Bayesian framework that yields population-level cosmological posteriors.
3. Add a dark-energy section to the SpectrumFM v8 narrative covering BAO redshift purity, SN typing, σᵥ regression, and lens identification — with quantified expected gains.
4. Move the Sheu 2023 lensed-SN pipeline from retrospective to live triggering, gated by a SpectrumFM SN-type classifier on host-galaxy spectra.
5. Stand up the agentic coordination layer (§6) that turns these per-pipeline outputs into a continuously updated multi-probe cosmological constraint.

---

## 2. The DES Y6 Paper in Detail

This section is written assuming familiarity with general ML/statistics ideas but not with the specific cosmological vocabulary. Readers comfortable with the field can skip to §3.

### 2.1 What is being measured

Dark energy is, observationally, whatever component is driving the present-day accelerated expansion of the universe. In Einstein's equations it shows up as a pressure-to-density ratio called the **equation-of-state parameter** *w* ≡ *P*/*ρ*. A cosmological constant Λ has *w* = −1 exactly and at all times; quintessence-like scalar fields have *w* > −1; "phantom" fields have *w* < −1 and are problematic theoretically because they violate the null energy condition.

The simplest time-dependent generalization is the **Chevallier–Polarski–Linder (CPL) parameterization** (Chevallier & Polarski 2001; Linder 2003):

> *w*(*a*) = *w*₀ + *w*ₐ (1 − *a*),

where *a* = 1/(1+*z*) is the cosmic scale factor (*a* = 1 today, *a* → 0 at high redshift). *w*₀ is the present-day value; *w*ₐ controls the time evolution. ΛCDM is the single point (*w*₀, *w*ₐ) = (−1, 0). Any statistically significant excursion from that point would be the first concrete evidence that dark energy is dynamical — and would force theory beyond a vacuum-energy interpretation.

The DES Y6 best-fit (*w*₀ ≈ −0.82, *w*ₐ ≈ −0.63) is interesting because (i) it sits significantly off the Λ point and (ii) its trajectory *crosses w = −1*. The crossing happens because *w*(*a*) goes from *w* = *w*₀ + *w*ₐ = −1.45 in the distant past to *w* = *w*₀ = −0.82 today. Canonical single-field quintessence and k-essence models cannot make this transition smoothly (Hu 2005; Guo et al. 2005), so the result, if it holds, requires either modified gravity, dark-matter / dark-energy coupling, or an unidentified systematic.

### 2.2 The four probes

The DES Y6 analysis stacks four classes of cosmological observables:

- **3×2pt (weak lensing + galaxy clustering).** Three two-point correlation functions measured jointly: (i) cosmic shear (correlation of source-galaxy ellipticities across the sky — the integrated lensing distortion of background galaxies by foreground large-scale structure); (ii) galaxy–galaxy lensing (cross-correlation of foreground lens-galaxy positions with background-galaxy shear); and (iii) galaxy clustering (foreground galaxy auto-correlation). DES Y6 uses ~140 M shear-sample galaxies in 4 tomographic bins out to *z* ≲ 2 and ~9 M MAGLIM++ position-sample galaxies in 6 bins over 0.2 < *z* < 1.2, covering ~4,300 deg². This is the *growth* probe — it constrains how matter overdensities have grown under gravity, which depends on the expansion history that dark energy controls.

- **DES + DESI BAO.** The baryon acoustic oscillation feature is a fixed comoving scale (the sound horizon at the baryon drag epoch, *r*ₐ ≈ 150 Mpc) imprinted on the galaxy correlation function. Measuring its apparent angular and radial size at various redshifts gives the angular-diameter and Hubble distances *D*ₘ(*z*) and *H*(*z*) — the geometric probe of expansion history. DES Y6 contributes its own photometric BAO measurement (*D*ₘ(*z*=0.851)/*r*ₐ = 19.51 ± 0.41 from ~16 M LRGs; 2.1% precision, the tightest BAO from any photometric survey). DESI DR2 contributes spectroscopic BAO across 0.295 < *z* < 2.330 from galaxies, quasars, and the Lyman-α forest. To avoid double-counting in the ~1,000 deg² DES/DESI footprint overlap, the analysis uses a separate DES-BAO measurement (*D*ₘ/*r*ₐ = 19.74 ± 0.60) computed outside the overlap.

- **Type Ia supernovae (SNe).** The DES-Dovekie sample: 1,623 photometrically identified Type Ia SNe from the full five-year DES-SN survey (Sánchez et al. 2024) plus ~200 SNe from historical low-redshift samples (Hicken et al. 2009, 2012; Krisciunas et al. 2017; Foley et al. 2018), reanalyzed with updated photometric cross-calibration (Popovic et al. 2025). SNe Ia are "standardizable candles": after a light-curve-shape and color correction, their peak luminosities are tight enough to deliver luminosity distances at percent-level precision. Notably, the DES-Dovekie recalibration *reduced* the original DES-SN5YR dark-energy significance from 4.2 σ to 3.2 σ when combined with DESI DR2 BAO + CMB; this is the largest single source of significance drift in the analysis.

- **Cosmic microwave background (CMB).** Temperature and polarization power spectra (TT, EE, TE) from Planck PR3, ACT DR6, and SPT-3G DR1. The CMB anchors the expansion history at the recombination epoch (*z* ≈ 1100) and provides the long lever arm against the low-redshift probes. The analysis intentionally *excludes* CMB lensing because it cross-correlates with the 3×2pt observables, which would require modeling a joint covariance the authors prefer to avoid in this Letter.

### 2.3 Headline numbers and Table I

DES Y6's leave-one-out table (the paper's Table I) is the most informative single object in the paper, because it shows how much each probe contributes:

| Dataset combination | Distance to ΛCDM | FoM |
|---|---|---|
| 3×2pt only | 1.0 σ | 6 |
| SN + DES BAO | 1.5 σ | 11 |
| **All DES (3×2pt + SN + DES BAO)** | **2.2 σ** | **48** |
| **All data (All DES + DESI BAO + CMB)** | **3.0 σ** | **222** |
| SN + BAO | 2.4 σ | 61 |
| 3×2pt + DESI BAO | 1.7 σ | 32 |
| CMB + DESI BAO | 2.9 σ | 61 |
| All DES + DESI BAO (no CMB) | 2.3 σ | 110 |
| SN + BAO + CMB (no 3×2pt) | 3.2 σ | 202 |
| 3×2pt + BAO + CMB (no SN) | 2.6 σ | 66 |
| 3×2pt + SN + CMB (no BAO) | 2.4 σ | 96 |
| All DES + CMB (no DESI BAO) | 2.8 σ | 102 |

FoM ≡ 1 / √det(Cov<sub>w₀wₐ</sub>) is the standard dark-energy "figure of merit" introduced by Huterer & Turner (2001) and adopted by the Dark Energy Task Force (Albrecht et al. 2006): bigger means tighter joint constraint on (*w*₀, *w*ₐ).

Two observations are critical:

1. **3×2pt doubles the constraining power of the low-redshift combination** (FoM 61 → 110 when 3×2pt is added to SN + BAO). This is the central methodological message of the paper, and the multi-probe-stacking lesson Huang's group should internalize.
2. **No single probe is load-bearing.** Dropping any one of the four probe classes still gives 2.3 σ–3.2 σ; the best-fit (*w*₀, *w*ₐ) always lands in the same quadrant (*w*₀ > −1, *w*ₐ < 0). Excluding SN alone — the historical workhorse — still gives 2.6 σ. This is the robustness check that elevates the paper from "interesting hint" to "consistent multi-source evidence."

### 2.4 Phantom-crossing and what it means

The best-fit CPL trajectory crosses *w* = −1 at some redshift between 0 and 1 (depending on which data combination). As noted above, this cannot be realized in canonical single-field models without introducing pathologies. The paper cites two avenues that *can* produce a phantom-crossing effective *w*:

- **Non-minimal couplings between dark energy and dark matter** (Khoury, Lin & Trodden 2025; Bedroya et al. 2025). In these models the effective *w* inferred under the assumption of independent dark sectors can cross −1 even when the underlying scalar field has *w* > −1.
- **Modifications of gravity** (Ye et al. 2025; Pan & Ye 2026; Wolf et al. 2025, 2026; Tsujikawa 2026; Cataneo & Koyama 2025). Modified Friedmann equations can mimic a phantom-crossing equation of state without invoking exotic matter.

For the lensing program, the relevance is that **modified gravity also rewrites the growth-of-structure relation** that 3×2pt directly measures. A future analysis that compares Huang-strong-lensing-derived *w*(*z*) against DES-Y6-3×2pt growth at the same redshifts would constrain the gravity sector independently of the SN/BAO geometric channel.

### 2.5 Robustness and caveats

The paper goes to substantial length on robustness. Items the Huang group should note as methodologically transferable (see also §4.3):

- **Blinding.** Cosmological results were hidden until all analysis choices and robustness tests were finalized.
- **Nautilus nested sampler** (Lange 2023) is used throughout. Pure-Python, JAX-friendly, and a candidate drop-in for GIGA-Lens posterior sampling at population scale (§6).
- **Σm<sub>ν</sub> fixed at 0.06 eV** in the baseline; allowing it to vary shifts (*w*₀, *w*ₐ) to (−0.80, −0.74) and *increases* significance to 3.4 σ — i.e., the conservative choice is the fixed-mass baseline.
- **Intrinsic-alignment model: NLA instead of TATT.** Simpler model adopted to reduce projection effects; results show no significant sensitivity.
- **Analysis-choice tests** (TATT vs NLA, free baryon-feedback amplitude, including the dropped second tomographic bin) all shift *w*₀, *w*ₐ by ≤ 0.1 σ and the ΛCDM distance by ≤ 0.3 σ.
- **SN sample sensitivity.** DES-Dovekie, Pantheon+, and Union3.1 (after recent reanalyses) all give comparable significances (3.2 σ, 3.4 σ when combined with DESI DR2 BAO + CMB).

### 2.6 The S₈ "tension" addendum (Supplemental Material Fig. S1)

A separate long-standing puzzle is the offset between *S*₈ ≡ σ₈ √(Ω<sub>m</sub>/0.3) — the amplitude of matter clustering — measured from low-redshift 3×2pt vs predicted by the primary CMB. In ΛCDM, DES Y6 finds a 1.9 σ offset in the Ω<sub>m</sub>–*S*₈ plane and 2.0 σ along *S*₈. Under *w*₀*w*ₐCDM (after geometric anchoring with BAO+SN to break degeneracies) the offset is 1.2 σ in the plane and 1.7 σ in *S*₈ — essentially **unchanged**. So this dynamical-dark-energy result does *not* dissolve the S₈ tension; whatever causes that tension is unrelated to time-varying *w*. This is relevant because some literature had floated time-varying *w* as a candidate resolution.

---

## 3. Huang's Strong-Lensing Program

### 3.1 The program at a glance

For the cosmology context of this memo, the relevant deliverables of the Huang group are (paper citations point to PDFs in `papers/`):

- **Lens candidates from ResNet/EfficientNet searches.** ~3,500–5,500 cumulative grade-A/B candidates from H20, H21, S24, and the Inchausti 2025 two-architecture DR10 search. The Sheu 2023 lensed-SN pipeline uses a 5,807-system aggregated database of strong-lens candidates from H20/H21/Storfer plus published catalogs (Moustakas 2012, Carrasco 2017, Diehl 2017, Jacobs 2017/2019, Pourrahmani 2018, Sonnenfeld & Leauthaud 2018, Sonnenfeld 2020, Wong 2018).
- **Lensed-quasar candidates.** 436 from Dawes 2022 (autocorrelation method); the Sheu 2024a variable-quasar search targets a sample of 655 grade-A/B candidates compiled from D22 and H23, and isolates 20 systems (13 new, 7 previously known) with confirmed photometric variability.
- **Foundry I HST follow-up.** 51 HST SNAP-confirmed lenses (program GO-15867, PI Huang); all 51 of 51 selected candidates confirmed.
- **Foundry II / III / IV spectroscopic follow-up.** DESI (II), Keck NIRES (III), and VLT MUSE (IV) campaigns for lens-galaxy and source redshifts plus σᵥ.
- **Hsu 2025 pairwise spectroscopic search.** Friends-of-Friends grouping of DESI DR1 spectra with linking length 3″ and redshift ratio > 1.3 yields 13,218 groups (26,621 spectra), refined by visual inspection to ~11,837 systems for lens-candidacy review.
- **GIGA-Lens** (Gu et al. 2022). GPU-accelerated, differentiable, fully forward-modeling Bayesian framework in TensorFlow + JAX; multi-start gradient descent → variational inference → Hamiltonian Monte Carlo; ~105 seconds per system on four NVIDIA A100 GPUs.
- **Sheu 2023 lensed-SN pipeline.** Difference-imaging with Bramich 2008 + SFFT algorithms over 5,807 systems; SALT3 + 161 core-collapse models for typing; ~17–20 hours on 20 NERSC Cori nodes for full deployment.
- **Sheu 2024a variable lensed-quasar pipeline.** Same image-subtraction substrate plus PSF photometry, σ-magnitude variability threshold; 20 candidates returned.

### 3.2 The cosmographic outputs the program is positioned to deliver

The introductions of both *Foundry I* (Huang et al. 2025a §1) and *GIGA-Lens* (Gu et al. 2022 §1) lay out five cosmologically relevant outputs. Below, each is mapped to the corresponding DES Y6 probe class:

| Lensing probe | Observable | Maps to DES Y6 probe | Status in Huang program |
|---|---|---|---|
| **(a)** H₀ from lensed-quasar time delays | Δ*t* between QSO images | Independent late-time H₀ (cross-check on CMB+BAO inferred *H*₀) | Pipeline candidate set in hand (Sheu 2024a: 20 systems with variability; D22/H23: 655 grade A/B). Time-delay extraction not yet done. |
| **(b)** H₀ + dark-energy distance ratios from lensed SNe Ia | Δ*t* + magnification + standardizable peak | Independent low-z probe; complements SN Ia channel without sharing SN photometric-calibration systematics | Retrospective candidates from Sheu 2023; *live* triggering pipeline not yet stood up. Foundry I §1 explicitly cites Pierel et al. 2021 for the *w*-sensitivity argument. |
| **(c)** Static-lens statistics for *w*₀wₐ | *θ*ₑ–*σᵥ* ratios over an O(10⁴) lens sample | Direct *w*₀wₐ probe; statistically independent of 3×2pt and SN | Foundry I §1 cites Li et al. 2024 explicitly. Hsu 2025 §4.1 already computes *θ*ₑ from FastSpecFit *σᵥ* for individual candidates. The aggregated population analysis has not yet been published by the group. |
| **(d)** Compound (multi-source) lens cosmography | Multi-source distance ratios | Direct geometric *w*₀wₐ probe even for small samples | Sheu 2024b (carousel) plus literature discoveries (Dux 2024, Bolamperti 2024). Group is producing candidates; population analysis pending. |
| **(e)** Dark-matter substructure | Flux ratios + perturbations on lensed arcs | Not a *w*₀wₐ probe but a separate CDM test | Active program; Vegetti/Hezaveh/Gilman literature line. |

Channel (c) is the cleanest direct map to the DES Y6 paper. Li et al. (2024) showed that a sample of *O*(10⁴) galaxy-scale strong lenses with measured *σᵥ* values produces dark-energy constraints competitive with cosmic shear and SN Ia. Huang's group has the lens sample; DESI DR1/DR2 (via the same machinery Hsu 2025 already uses) provides the *σᵥ* values. The pipeline ingredients exist.

### 3.3 The current center of mass of the program is Discover + Characterize, not Cosmologize

This is the strategic observation behind the rest of the memo: the program's three pillars (Discover → Characterize → Cosmologize) are not currently balanced. The Discover pipeline produces ~10³–10⁴ candidates per data release; the Characterize pipeline (Foundry I HST + Foundry II–IV spectroscopy + GIGA-Lens) is healthy; the Cosmologize pipeline is *named* in every introduction and *not yet built*. The DES Y6 paper is a useful forcing function for prioritizing it.

---

## 4. Implications for Huang's Work

### 4.1 The DES result validates the strategic value of the lensing pillar

Three direct implications:

1. **Multi-probe stacking is the dominant operational lesson of DES Y6.** 3×2pt doubled the FoM when added to SN + DESI-BAO. Strong-lens cosmography — channels (a)–(d) above — is statistically independent of cosmic-shear 3×2pt (different observable, different systematics, different lens-galaxy populations). It is therefore structurally able to play the same FoM-doubling role in a future analysis that strong lensing replaces or augments one of the DES probe classes.
2. **Strong-lens H₀ is independent of SN photometric calibration.** The DES-Dovekie recalibration alone shifted SN-driven significance from 4.2 σ to 3.2 σ. The community's current confidence interval on dark energy is therefore sensitive to a single instrument's flux calibration. Lensed-quasar and lensed-SN time delays are insensitive to absolute photometric calibration (they depend on a *relative* time difference between images). A few-percent-precision strong-lens H₀, independently obtained, is a load-bearing cross-check.
3. **The leave-one-out robustness pattern is replicable for lensing.** DES Y6's strongest result is not the headline 3.0 σ but the leave-one-out table showing every subset still rejects ΛCDM at ≥ 2.3 σ. The corresponding template for the Huang program — show *w*₀, *w*ₐ posteriors with and without each lensing probe class included — is a deliverable in its own right.

### 4.2 Specific actions the group should consider

Each action is tied to a paper already in the program's corpus, so the lift is "extend an existing analysis" rather than "start from scratch."

- **Li-et-al-2024-style static-lens *w*₀wₐ analysis** on the ~3,500-candidate H20/H21/S24 catalog combined with FastSpecFit *σᵥ* values from DESI DR1. Foundry I §1 already cites Li et al. (2024) as the framework. Hsu 2025 §4.1 already computes *θ*ₑ from *σᵥ* per-candidate (equation 1 of that paper). What is missing is the population-level Bayesian inference layer that turns the per-candidate (*θ*ₑ, *σᵥ*, *z*<sub>l</sub>, *z*<sub>s</sub>) tuples into a (*w*₀, *w*ₐ) posterior. The Hsu sample size (~11,837 systems) is in the right order of magnitude. A constraint paper "Strong-Lens Statistics from the DESI Spectroscopic Pair Search: an Independent Test of Dynamical Dark Energy" is well within reach *before Rubin first light* (≲ 2027) and would be the first cosmology paper from the Foundry program.

- **Hierarchical GIGA-Lens.** Gu et al. 2022 §2.5 already implements HMC + variational inference inside TensorFlow / JAX. Extending this to a two-level hierarchical Bayesian model — per-system posteriors at the leaf level, population-level (*w*₀, *w*ₐ, lens-population priors) at the root — is a natural extension of the existing pipeline, not a redesign. The same multi-start-gradient-descent + VI-warmstart pattern is applicable; the dominant cost is the per-system inference, which is already 105 s on 4 × A100 — embarrassingly parallel across the candidate catalog. The /raid/benson 8 × A16 + 2 × L4 cluster (per project memory) can host a population-scale run.

- **Move lensed-SN Ia from retrospective to live triggering.** Sheu 2023's pipeline currently runs on archived DECaLS exposures. In the Rubin era, the same difference-imaging logic must run on alert streams in real time. The bottleneck is candidate filtering: live alert volume is too high for human VI. SpectrumFM SN typing on the host-galaxy spectrum (when available) plus a photometric prior is the natural filter. A live-triggering minimum viable pipeline could be stood up on the existing 5,807-system database against ZTF alerts as a Rubin precursor — useful both for the lensed-SN cosmology channel *and* as a real-time tooling demonstrator for the agentic layer (§6).

- **Time-delay extraction for Sheu 2024a's 20 variable lensed-quasar candidates.** The 13 *new* candidates in particular are unstudied by TDCOSMO / H0LiCOW. Even partial time delays on a handful of these would advance the channel (a) deliverable. Coordination with the existing time-delay cosmography community (Suyu / Treu / Birrer / Wong) is the obvious path.

- **Compound lenses as a flagship.** Sheu 2024b (carousel), Dux 2024, and Bolamperti 2024 each give one or a few systems. Per Foundry I §1, even a small sample of compound lenses is a powerful cosmological probe (Collett & Auger 2014; Linder 2016; Sharma & Linder 2022; Sharma, Collett & Linder 2023). Reframing this from "interesting curiosity" to "lead cosmology deliverable" repositions the program.

### 4.3 Methodological transfers the Foundry team can adopt today

DES Y6 demonstrates a handful of methodological practices the Foundry's cosmology paper(s) should match:

- **Posterior-difference tension metric.** Quantify the "distance to ΛCDM" by the posterior-mass fraction enclosed by the iso-likelihood contour through (*w*₀, *w*ₐ) = (−1, 0). This is more interpretable than Δ*χ*² when posteriors are non-Gaussian and is robust to prior choices when reported alongside the Δ*χ*²-based number for cross-validation. JAX implementation is straightforward.
- **Pre-unblinding analysis-choice freeze.** All analysis choices (scale cuts, intrinsic-alignment model, neutrino prior) were frozen before unblinding. This is industry-standard for high-stakes cosmology and should be the default for any Foundry cosmology paper.
- **Leave-one-out probe robustness table.** Replicate the DES Y6 Table I format for any lensing-cosmology analysis. The format is the artifact.
- **Nautilus nested sampler.** Drop-in for GIGA-Lens posterior sampling; pure-Python; JAX-compatible. Replaces dynesty / multinest in the existing stack with no API friction.

---

## 5. Implications for the SpectrumFM Proposal

### 5.1 The strategic gap

SpectrumFM v7 opens with the line, *"Massive spectroscopic surveys, such as the Dark Energy Spectroscopic Instrument (DESI), are the backbone of precision cosmology, providing the data to map the universe's structure and constrain dark energy."* It never returns to dark energy in any concrete sense — the words "BAO," "*w*₀," "*w*ₐ," "CPL," and "supernova cosmology" do not appear in the narrative as quantitative targets. Downstream tasks are listed as (i) the six DESI target classes (LRG / ELG / QSO / MWS as label-rich; LBG / LAE as few-shot for DESI-II / Spec-S5); (ii) strong-lens identification via the Hsu 2025 pairwise method; (iii) supernova typing. The connection to cosmology is structural but unstated.

After DES Y6, the structural argument tightens to a quantitative one: SpectrumFM's primary downstream tasks each map to a probe in the DES Y6 stack. The proposal should say so explicitly. This is the single highest-leverage edit to v8 / Phase II.

### 5.2 Concrete cosmological angles to add to the narrative

| SpectrumFM downstream task | DES Y6 probe | Mechanism of cosmological gain |
|---|---|---|
| LRG / ELG / QSO redshifts and per-spectrum confidence | DESI DR2 BAO (which the DES Y6 paper relies on) | Tighter, calibrated confidence reduces *N*<sub>eff</sub>-weighting and contamination in the BAO correlation-function fits → directly reduces σ(*D*ₘ/*r*ₐ) |
| Supernova typing on host-galaxy spectra | DES-Y5 SN Ia sample (1,623 photometric SNe) | Better SN-Ia / non-Ia separation reduces SN sample contamination → reduces dominant low-redshift systematic (the same systematic that moved 4.2 σ → 3.2 σ after the DES-Dovekie recalibration) |
| **Velocity-dispersion regression on lens-galaxy spectra** *(not in v7; new task to add)* | None directly in DES Y6 — *adds a new probe* | Provides the *σᵥ* values needed for the Li-et-al-2024 strong-lens *w*₀wₐ pipeline (§4.2). The Foundry's static-lens cosmography channel becomes a fully spectroscopic SpectrumFM downstream output. |
| Pairwise lens identification (Hsu 2025) | Strong-lens cosmography channels (a), (c), (d) | Currently in v7 as a *validation* task. Reframe as production accelerator: a ~100× speed-up on Hsu's manual VI step makes the population-level static-lens analysis tractable at DESI-II / Spec-S5 scale. |
| LBG / LAE few-shot redshifts | Future BAO at *z* > 2 | Provides the high-redshift BAO leverage Spec-S5 needs to extend *w*(*a*) into the redshift range where DES Y6's CPL trajectory predicts *w* ≪ −1. |

The single most impactful addition is **the new velocity-dispersion (*σᵥ*) regression head**. It is a small architectural extension (an additional regression output on the SpectrumFM decoder), it has clean training data (DESI FastSpecFit *σᵥ* values for ~10⁶+ galaxies), and it is the missing piece that takes channel (c) of §3.2 from "in principle" to "in production." If SpectrumFM adopts this, the proposal can say truthfully that the model directly outputs the spectroscopic inputs to an independent *w*₀wₐ measurement.

### 5.3 Phase II Rubin / Euclid / Roman alignment

The DES Y6 Conclusion ends, *"This Letter marks a significant step towards completing the multi-probe dark-energy program laid out in the early 2000s, setting the stage for the next chapter of the program with Stage-IV imaging surveys such as Rubin, Euclid, and Roman."* The SpectrumFM Phase II vision explicitly addresses Focus Area 14-A's call to *"combine different modalities of data across multiple space- and ground-based large-scale sky surveys (e.g., LSST, Roman, and Euclid)."* The natural rhetorical bridge: SpectrumFM is the spectroscopic-side foundation model for the same multi-survey dark-energy program the DES Y6 paper just closed Stage III on. Saying so explicitly is the simplest way to land the cosmological framing.

### 5.4 What the Stephen-Bailey LBNL connection makes possible

Bailey leads the DESI Redrock pipeline (the production redshift fitter) and the DESI corpus access — meaning SpectrumFM's redshift outputs can plug directly into the DESI BAO analysis chain as an alternative or augmented redshift source. This connection is in the v7 team list but not framed in cosmological terms. A v8 paragraph quantifying expected BAO σ(*D*ₘ/*r*ₐ) improvement under SpectrumFM redshift confidence weighting — even with rough Fisher-matrix scaling — would be the kind of concrete deliverable DOE Genesis reviewers look for under Topic 14.

---

## 6. The Agentic-AI Layer

This section is the one most directly aligned with Greg's funded role on SpectrumFM (*"Benson (USF; $33,309) leads agentic-AI tooling"*) and with the agentic-coding expertise he brings to the Foundry program. Per the v7 proposal it is also the role for which v7 itself provides the least architectural detail. The DES Y6 result motivates a specific architecture.

### 6.1 Why agentic coordination is the missing layer

The DES Y6 paper was won, methodologically, by *combining* four probes (3×2pt, SN, BAO, CMB) under a shared blinding protocol, a unified covariance, and an explicit leave-one-out auditing framework. Huang's program currently runs Discover (ResNet / EfficientNet / autocorrelation / pairwise / difference-imaging finders), Characterize (HST + DESI + Keck + MUSE + GIGA-Lens), and Cosmologize (the not-yet-built population-level pipelines) as *separately human-coordinated* workflows. To turn lens discoveries into a *w*₀wₐ constraint at the DES-Y6 level, the team needs an orchestration layer that:

- routes new candidates to the right downstream pipeline by science target,
- maintains a live multi-probe constraint as new data arrives,
- triggers real-time follow-up for transient channels (lensed SNe, variable lensed QSOs),
- pre-grades human-VI tasks at the scale Rubin will demand,
- forecasts the constraint impact of upcoming survey data so the program can prioritize.

Each of these is a natural agent. Their concrete designs follow.

### 6.2 Five agent designs

In each case, the agent is given a name, a clearly bounded role, a tool surface (the functions the agent calls), and a memory store (what state survives between invocations). Tool surfaces use illustrative dot-notation; the corresponding tools can be either Python functions exposed via the SpectrumFM tooling layer or MCP-style tool servers wrapping existing pipelines.

#### (i) Lens Triage Agent

**Role.** Ingests new candidate cutouts and metadata from the ResNet / EfficientNet / autocorrelation finders. Classifies each candidate by likely science target (galaxy-galaxy lens / lensed quasar / lensed-SN host / compound multi-source / cluster lens). Dispatches to the appropriate downstream pipeline.

**Tool surface.**
```
legacy_surveys.cutout(ra, dec, bands, size_arcsec) -> Cutout
gigalens.fit(cutout, priors) -> LensPosterior
hsu_pairwise.score(ra, dec, dr) -> {is_pair: bool, group_id: str|None}
sheu_imgsub.run(system_id, mode) -> {detections: list, light_curves: dict}
sheu_variability.score(system_id) -> {sigma_mag: float, candidate: bool}
spectrumfm.embed(spectrum) -> Embedding
spectrumfm.classify(spectrum, head) -> Label
candidate_db.upsert(system_id, fields) -> None
```

**Memory.** Candidate-state DB keyed by (RA, Dec, discovery-DR, grade-history, dispatched-pipelines, results).

**Decision logic.** Heuristics from H21 morphology + cutout color + presence of nearby spectroscopic pair → dispatch table.

#### (ii) Cosmology Audit Agent

**Role.** Maintains a continuously updated joint posterior over (*H*₀, *w*₀, *w*ₐ) from all available Foundry inputs (per-system GIGA-Lens posteriors, lensed-quasar time delays, lensed-SN time delays + magnifications, static-lens *θ*ₑ–*σᵥ* counts). Recomputes when any input arrives. Emits a log entry every time the posterior shifts by more than some threshold (e.g. 0.1 σ in the distance to ΛCDM). Replicates the DES Y6 leave-one-out table as the standard reporting artifact.

**Tool surface.**
```
nautilus.run(loglike, prior, ndim) -> Chain
lens_likelihood.static(theta_e, sigma_v, z_l, z_s) -> float
lens_likelihood.time_delay(delta_t, model_posterior) -> float
lens_likelihood.lensed_sn(...) -> float
posterior_difference(chain, point) -> float  # the DES Y6 tension metric
constraint_db.append(timestamp, summary) -> None
dashboard.render(chain) -> URL
```

**Memory.** Chain artifacts (HDF5); constraint-shift log; rolling leave-one-out table.

#### (iii) Live Transient Follow-up Agent

**Role.** Streams the Sheu 2023 / Sheu 2024a difference-imaging outputs (in production: ZTF alerts as a Rubin precursor). Classifies detections using the SpectrumFM SN-typing head on the host-galaxy spectrum (when DESI/SDSS exists) and a photometric prior otherwise. Auto-drafts ToO observation requests for Keck NIRES / Subaru / Lick when classification confidence and lensing-plausibility cross thresholds.

**Tool surface.**
```
sheu_imgsub.stream() -> AsyncIterator[Detection]
spectrumfm.classify_sn(host_spectrum) -> {type: str, conf: float}
salt3.fit(light_curve) -> LightCurveFit
sheu_lensing_plausibility(system_id, detection) -> float
keck.too_draft(target, justification, exposure_spec) -> ToO
slack.notify(message) -> None
```

**Memory.** Detection-state DB; ToO ledger.

#### (iv) VI Pre-grading Agent

**Role.** Scales human visual-inspection capacity by pre-grading candidate spectra and cutouts. For each candidate produces (a) a SpectrumFM-embedding-based similarity score against the validated lens corpus, (b) an explanation paragraph identifying which spectral features (e.g. partner's-Z lines, Einstein-radius consistency) drove the grade, (c) a queue position for human review. Reduces VI workload by triaging out the obviously-not-lensed and obviously-lensed extremes, leaving humans to review the borderline.

**Tool surface.**
```
desi.dr1.spectrum(target_id) -> Spectrum
spectrumfm.embed(spectrum) -> Embedding
vi_corpus.similarity(embedding, k) -> list[(corpus_id, score)]
hsu_pairwise.features(target_id) -> {sigma_v, theta_e, partner_z, ...}
explainer.generate(features, similarity) -> Markdown
vi_queue.enqueue(target_id, prior, explanation) -> None
```

**Memory.** VI corpus embeddings (a small vector DB); grade-decision history.

#### (v) DESI-II / Spec-S5 Forecasting Agent

**Role.** Runs Fisher-forecast or end-to-end mock-data ensembles against simulated DESI-II / Spec-S5 lens samples to project the (*w*₀, *w*ₐ) constraint that adding the next survey will provide. Used to prioritize agent-2 inputs and proposal-narrative figures.

**Tool surface.**
```
survey.sim(survey_spec) -> SimulatedCatalog
forecast.fisher(model, data, fiducial) -> CovMatrix
forecast.mock_mcmc(model, data, fiducial, n_realizations) -> ChainEnsemble
fom(cov) -> float
report.compare(baseline, with_extension) -> Markdown
```

**Memory.** Cached survey simulations; forecast-result registry.

### 6.3 Shared substrate

All five agents share:

- A **Parquet / TileDB feature store** keyed by (system_id, modality) holding cutouts, spectra, light curves, embeddings, σᵥ regressions, lens-model posteriors.
- A **SpectrumFM head registry**: a small adapter layer exposing the encoder's downstream heads (redshift, classification, SN-typing, σᵥ regression, lens-id) with consistent input/output schemas. Each agent depends on this registry rather than on the encoder directly.
- **JAX likelihood pipelines** re-used between GIGA-Lens per-system inference and the Cosmology Audit Agent's population-level posterior — i.e. the same forward model serves both per-lens fits and the joint cosmological constraint.

### 6.4 Phasing

Phase I (during the SpectrumFM grant period) should deliver agents (i), (iv), and (v) — these are the spectroscopic-pipeline agents and they double as demonstrations of the agentic-AI tooling deliverable in the proposal. Agent (ii) and agent (iii) are Rubin-era; they should be designed in Phase I and stood up in Phase II.

---

## 7. What We Can Directly Use From the DES Y6 Result

A short, action-oriented list:

- **Posterior-difference tension metric.** Re-implement in JAX as part of the Foundry's Cosmology Audit Agent (§6.2.ii). The DES Y6 paper gives the algorithm and confirms it tracks Δ*χ*² to within 0.1–0.3 σ in their analyses.
- **Pre-unblinding analysis-choice freeze checklist.** Adopt verbatim from the DES Y6 / DES Y6-3×2PT methodology papers for the first Foundry cosmology paper.
- **Leave-one-out probe robustness table.** Make this the standard final-section artifact for any multi-probe lensing-cosmology paper from the group.
- **Nautilus nested sampler.** Drop into GIGA-Lens as a sampler option; benchmark against the current HMC implementation.
- **DES Y6 public chains.** When released (the paper promises a Y6-EXT companion), import them and combine with Foundry constraints. The combined posterior is the cleanest cosmology paper the group can publish in the short term.
- **DES-Dovekie SN reanalysis pipeline (Popovic 2025).** Relevant for Sheu 2023's L-SN classification step — same SALT2/3 + photometric-calibration machinery, with improved cross-calibration that should propagate to lensed-SN photometry.
- **DES Y6 covariance modeling code.** The DES collaboration's open-source 3×2pt likelihood (CosmoSIS / TXPipe lineage) provides templates for building the Foundry's static-lens population likelihood.

---

## 8. Risks and Caveats

- **3 σ is not 5 σ.** The DES Y6 result is "weak preference" in the careful language of the abstract. A future SN reanalysis (the in-progress Camilleri 2026 combining DES-Dovekie with Pantheon+) or improved CMB-lensing covariance could erode it. The Foundry group should not bet on dynamical dark energy being a settled fact; they should bet on contributing the next probe that either confirms or refutes it.
- **Strong-lens systematics are real.** Mass-sheet degeneracy, line-of-sight perturbations, and microlensing (for lensed SNe) all introduce systematic floors in time-delay cosmography that the TDCOSMO program has spent a decade quantifying. The Foundry's H₀ from lensed QSOs and lensed SNe should be cross-validated against TDCOSMO methodology before being combined with DES-Y6 chains.
- **Static-lens cosmography needs *O*(10⁴) lenses for *w*₀wₐ precision.** The Hsu 2025 sample is in the right order of magnitude but selection effects in DESI fiber assignment are not yet quantified. A selection-function paper (or the equivalent forward-model selection injection) is a prerequisite for the cosmology paper.
- **SpectrumFM's expected cosmological gain is currently unquantified.** Until a Fisher-forecast pass exists, the cosmology framing of v8 will be qualitative. The DESI-II / Spec-S5 Forecasting Agent (§6.2.v) is the natural deliverable that closes this gap.
- **Compute budget.** Population-level hierarchical GIGA-Lens runs over 10⁴ lenses at 105 s/system on 4 × A100 are about 1 GPU-month per inference pass. The /raid/benson 8 × A16 + 2 × L4 cluster is suitable for development but a population-scale production run probably wants Perlmutter A100 allocation — which the SpectrumFM grant already provisions.
- **Modified-gravity vs DE/DM-coupling degeneracy.** The phantom-crossing best-fit cannot be uniquely interpreted from the geometric + 3×2pt data alone; degeneracies persist across model classes. The Foundry's contribution will be a measurement, not yet a discrimination — but the measurement is the prerequisite for the discrimination.

---

## 9. Recommended Near-Term Actions

In priority order:

1. **Plan and execute the static-lens *w*₀wₐ paper.** Use the H20 + H21 + S24 catalog plus DESI DR1 FastSpecFit *σᵥ* values plus the Hsu 2025 spectroscopic confirmations. Target a constraint paper within 12–18 months, before Rubin first light. This becomes the Foundry's first multi-probe cosmology deliverable and answers the DES Y6 result with an independent late-time probe.
2. **Add a dark-energy section to the SpectrumFM v8 narrative.** Cover BAO redshift purity (with a Fisher estimate of expected σ(*D*ₘ/*r*ₐ) improvement), SN typing for Stage-IV SN cosmology, and — most importantly — propose the new *σᵥ* regression head as a downstream task. Cite DES Y6 explicitly as the motivating context.
3. **Extend GIGA-Lens to hierarchical population inference.** Two-level Bayesian model with population-level (*w*₀, *w*ₐ) priors. Re-use the existing JAX HMC + VI infrastructure. Initial demonstration on a synthetic-lens population, then on a subset of the Foundry I HST-confirmed lenses.
4. **Stand up the agentic coordination layer.** Deliver the Lens Triage Agent (§6.2.i), VI Pre-grading Agent (§6.2.iv), and DESI-II / Spec-S5 Forecasting Agent (§6.2.v) as Phase-I SpectrumFM deliverables. These are the agents whose tool surfaces are buildable today against existing pipelines.
5. **Begin live lensed-SN triggering** against ZTF alerts (Rubin precursor) over the existing 5,807-system database, gated by SpectrumFM SN typing on host-galaxy spectra.
6. **Coordinate time-delay extraction** for the 13 new Sheu 2024a variable lensed-quasar candidates with TDCOSMO / H0LiCOW. Even partial delays advance the channel-(a) deliverable.
7. **Adopt the DES Y6 methodology checklist** (posterior-difference metric, pre-unblinding freeze, leave-one-out table, Nautilus sampler) as the default protocol for any future Foundry cosmology paper.

---

*Greg Benson, USF — working memo prepared for the Huang strong-lensing group and the SpectrumFM team, 2026-05-26.*
