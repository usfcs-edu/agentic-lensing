# NEXT DIRECTIONS — after the claude-giga-lens campaign (2026-07-15)

Post-campaign scoping note. Derived from a 10-agent structured review (5 readers over
CAMPAIGN.md / papers/main.tex / the campaign plan / foundry-i open items / a 2025–26
literature sweep; 4 independent proposal lenses producing 20 candidate directions; 1
adversarial critic that checked every proposal against the ledger, merged duplicates,
and ranked by information-per-A100-hour). Workflow run `wf_86912111-f2f`, session
`3d232f85`. All campaign numbers below trace to CAMPAIGN.md gate rows.

**Status of this note:** proposals only. Nothing here is started, budgeted, or
committed. Pre-registration of gates happens if/when a direction is funded.

---

## 0. The organizing question

The campaign's verdict — *correlated noise is necessary but not sufficient* — leaves one
central mystery: **why does the correlated likelihood over-correct?** The corrected
slopes *bracket* the diagonal-native anchor across scales instead of unifying onto it:

| product / likelihood | γ | vs anchor |
|---|---|---|
| fine, correlated, steep basin | 1.816 | above |
| native, diagonal (anchor) | 1.433 [1.400, 1.468] | — |
| binned, correlated, low basin | **1.103 ± 0.008** | ~17σ below |

The report attributes the residual to source-model/PSF systematics — but that
attribution is currently **hypothesis, not measurement** (ledger open item: "No
experiment designed yet"). Every direction below is ranked by how much it constrains
this residual per unit cost, with the cheapest decisive discriminators first.

Compute context: ~56 A100-h Perlmutter reserve (shared-QOS single-GPU jobs backfill
same-day and bill fractionally); phoenix 8×A16 + 2×L4 effectively free. Correlated SMC
costs ~400 MB/particle (AD tape inflated ~2.8× by the whitening conv) — 128 particles
≈ one 80 GB A100; ~1.5–2 A100-h per converged run.

---

## 1. Novelty verification (literature sweep, 2026-07-15)

Both campaign headline results appear **genuinely novel** as of mid-2026:

- **No published drizzle-covariance lens-modeling likelihood exists.** The substructure
  power-spectrum line (arXiv:2302.00480) deliberately models unstacked `flt` frames *to
  avoid* drizzle covariance; Galan et al.'s JWST work (arXiv:2402.18636) flags strongly
  correlated noise in drizzled JWST data and recommends re-assessing the uncorrelated
  assumption, but implements nothing. Our GPU whitened likelihood + the real-data
  necessary-but-not-sufficient demonstration has no published counterpart.
- **No sampler benchmark on real lens posteriors exists.** TDCOSMO IX (arXiv:2202.11101)
  compares modeling *codes*, not samplers. Closest generic work: GGNS
  (arXiv:2312.03911), MCLMC ensembles (arXiv:2502.06335), nautilus-in-TinyLensGpu
  (arXiv:2503.08586). Our 9-method × 370-cell benchmark + two-stage preconditioning +
  per-basin SMC evidence recipe is the first of its kind.
- **GIGA-Lens 2.0 (arXiv:2606.30633) is scaling-only** (see
  `notes/gigalens2-positioning.md`): unchanged diagonal likelihood, no sampler work.
  Complementary, not competing. Its real system DESI J238.5690+04.7276 is a natural
  second testbed (§4).
- **The open gap nobody has assembled:** joint PSF + pixelated-source + mass inference
  under a correlated-noise likelihood. The pieces now exist separately — our whitened
  likelihood; GP sources on ray-transformed grids (arXiv:2606.30620); pixellated PSF
  posteriors with diffusion priors (arXiv:2511.19594, PSF-only, not joint) — and the
  JWST substructure debates ("DM substructure or source-model systematics?",
  arXiv:2502.18571, 2410.12987, 2511.07513) confirm this exact confusion is the
  contested frontier. This is the natural successor campaign (§6).

---

## 2. Tier 0 — certify what we have (≈15–20 A100-h + CPU, ~1 week)

These run before any new physics is added. Each either certifies a load-bearing
campaign number or exposes it — and several are prerequisites for everything below.

### T0.1 The E3 kernel/λ-robustness scan *(the campaign's own unrun pre-registered gate)*
- **Why:** E3 is the one deliberately-unrun gate ("γ shift < 0.5σ across kernel
  variants — follow-on"). Until it runs, the source/PSF attribution of the 1.103
  over-correction is not defensible: the headline could still be (partly) a
  covariance-kernel artifact. Also gives the 191-nat H1 flip its missing robustness
  certificate.
- **What:** write the never-written `11_kernel_ablation.py`. Whitener variants on v3b:
  fitted two-component kernel (control), analytic-drizzle-only kernel,
  `binned_kernel_from_fine(cov_fine)`, δ-reg λ ∈ {0.03, 0.1 (prod), 0.3, 1.0}. Each
  must pass e_op ≤ 0.02; rerun the exact P1c recipe (MAP polish → Laplace check →
  128-particle SMC) per variant. **Cross-whitener logZ comparisons require the exact
  log|C| constant** (Szegő gap +179.21 nats — the rule binds here, *not* for
  fixed-whitener model ladders, see §5 warnings).
- **Gate:** pre-registered E3: γ posterior-mean shift < 0.5σ across variants.
- **Cost:** whitener builds CPU-local; ~5–8 A100-h (shared QOS).
- **Fork:** robust → covariance model exonerated, systematics campaign licensed.
  Drifts → the kernel joins the systematics budget and re-scopes everything downstream.

### T0.2 Seed-repeat the two linchpin numbers + gradient-checkpoint the likelihood
- **Why:** the 17σ tension and the −28.9-nat basin preference are **single-run numbers
  with no seed scatter**. Referee-proof them before anyone tries to explain them.
- **What:** first wrap the grouped-conv whitened log-density in `jax.checkpoint`
  (verify logp/grad parity vs `01_parity_harness.py` gates ≤1e-12; jaxlib-0.6.2
  remat × priority-fusion interaction untested — workaround flags catalogued in
  CAMPAIGN.md). Then 2 extra SMC seeds per basin at the frozen production config.
- **Gates:** σ_seed(ΔlogZ) < 5 nats (keeps the basin preference > 5σ);
  σ_seed(γ median) ≤ 0.008.
- **Cost:** ~1 day impl + ~6–8 A100-h. The checkpointing (target ≤200 MB/particle)
  roughly halves the cost of every SMC-based direction below — do it first.
- **Note:** this closes the held P2c confirmatory item (σ on w_steep) at the same time.

### T0.3 Companion-galaxy misfit discriminator *(critic's catch — absent from all 20 proposals)*
- **Why:** foundry-i's designated "natural next increment": a localized χ²≈9–15
  companion misfit (~290 px) sits on exactly the binned product where γ=1.103 lives and
  sets its χ²_ν=1.23 floor. A correlated likelihood **reweights spatial scales**, so it
  can transmit a localized real-space misfit into a global slope bias — a concrete,
  testable over-correction mechanism.
- **What:** mask the companion region (and/or add 1–2 Sérsic components) in the
  correlated v3b-low likelihood; rerun one 128-particle SMC.
- **Gate:** does γ move off 1.103 (≥3σ_stat)?
- **Cost:** ~2 A100-h. Cheapest mechanism test available; slot alongside T0.1–T0.2.

### T0.4 Free CPU/forward-pass checks (launchable today, zero A100-h)
- **Stationarity test:** the real skycell is *proven shift-variant* (NDRIZIM=3; render
  check 2.1–26.7σ, P1a Deviation 2), yet every whitener is a stationary kernel. 3×3
  per-block ACFs + `fit_kernel2` with bootstrap bands on model-subtracted v3/v3b
  residuals. A rejection reframes the over-correction as noise-model-**class**
  misspecification before any expensive source campaign.
- **λ-arm on the fine whitener** (feeds T0.1): rebuild v3 fine whiteners at
  λ ∈ {0.01…1.0}; at three fixed parameter points evaluate whitened logp **with** the
  exact log|C(λ)| (dense-Cholesky-anchored) + real-space χ²_pp. Resolves the fine-low
  "gaming" mechanism (polish railed γ 1.369→1.021 while whitened logp improved and
  real-space χ²_pp climbed 30.7→71.9) and may recover the missing fine-low cell that
  currently forces the bracket interpretation.
- **Real-space head-to-head:** render the 1.433-anchor model and the 1.103 model on the
  *same* binned pixels (forward passes only); compare χ²_pp. A free ordering check on
  which solution the binned data prefers in real space.
- **Ledger closures:** re-read the parked orientation-frame residuals (6.2°
  ellipticity / 12.8° shear rotations, 21%/29% amplitude deficits) from the on-disk P1c
  SMC particles — does the correlated likelihood move them? Harvest the niced glnt/PT
  T3 cells and the stricter-e_op native-whitener L4 check whose outcomes never reached
  the ledger.

### T0.5 Ops prerequisite: dead-man watchdog (half a day, before the first new batch)
The impl-agent death stalled P1c for 4 days. Every multi-run direction below (E3 scan,
SBC batches, pilot) inherits that risk until a simple watchdog (job-state poll + alert
+ fallback resubmit) exists. Build it once, before the first new SMC submission.

---

## 3. Tier 1 — the sharp forks (each redirects all downstream spend)

### T1.1 Injection-recovery on the *real* drizzle noise field (~6–10 A100-h)
The single sharpest experiment available. Inject EPL(γ=1.433, θ_E=2.655 — the anchor
posterior mean) + the fitted shapelet source, rendered through the convention-correct
ePSF, into the **model-subtracted real v3b residual field** (real correlated,
shift-variant noise — not mock noise). Run the unmodified P1c pipeline blind.
- **Gate (per critic):** state as **bias in γ units with per-injection z-scores** —
  n=2–3 injections cannot measure 68% coverage (the same low-n trap flagged on the
  E1c amended PASS). Growing to SBC scale merges this with T2.3.
- **Fork:** unbiased recovery → the likelihood is *certified on real noise*; 1.103 must
  be model misfit; source/PSF campaign licensed. Biased low (~1.1) → the
  stationary-kernel approximation is convicted; **cancel the source/PSF spend** and
  redirect at the noise model class (with T0.4-stationarity as corroboration).

### T1.2 Evidence-scored source-complexity ladder (~6–10 A100-h)
Cheapest test of the campaign's own prime suspect. Shapelet n_max ∈ {6 (frozen prod),
8, 10} × per-basin 128-particle SMC on v3b-low under the correlated likelihood
(foundry-i already saw n_max 6→8 + a 3rd Sérsic move γ 1.41→1.31, *not plateaued*;
warm starts on disk, `map_v11_v2d_nm8c3.npz` analog).
- **Gates:** seed-pair σ(logZ) ≤ 2 nats on the leading rung; then the fork —
  logZ prefers flexibility AND γ moves ≥3σ toward 1.433 → source systematics
  **demonstrated**, pixelated pillar (T2.1) funded. γ static → source exonerated at
  parametric level; PSF (T2.2) promoted.
- Whitener fixed across rungs → log|C| cancels; no Szegő bookkeeping needed here.
- Establishes ladder-plus-per-basin-logZ as the routine model-comparison stage.

### T1.3 Supersampled (ss=2) forward rendering at native + binned (~8–12 A100-h)
The **only hypothesis that predicts the bracket's sign pattern** (opposite-sign pixel
integration errors at different scales), and the only direction that can move the
**anchor itself**. GIGA-Lens 2.0 fit its real system at supersample=2; our production
fits are ss=1.
- **Trap (critic-flagged, must fix first):** the on-disk 0.065″ ePSF is *not* delta_pix/2
  for native — 0.12825″/2 = 0.064125″, a 1.4% sampling mismatch, exactly the PSF-
  convention defect class foundry-i regression-guards. Resample the ePSF (or write an
  explicit tolerance argument) and re-run parity gates before either arm.
- **Arms:** (a) v2d native diagonal refit at ss=2 from the `hmc_v13_v2d` warm state —
  does 1.433 move? (b) binned correlated ss=2 — does 1.103 move toward it?
- **Fork:** any anchor motion reframes the 17σ tension entirely; a double-null retires
  undersampling and sharpens the source-vs-PSF dichotomy.

Tier 0 + Tier 1 total: **≈35–50 A100-h**, inside the ~56 h reserve. Everything in
Tier 2 needs either new allocation or a decision to spend the remainder.

---

## 4. Tier 2 — the big builds (fund only what Tiers 0–1 license)

### T2.1 Pixelated/GP source under the correlated likelihood *(un-park the P3 pillar)*
Swap the 28 shapelet columns in `cgl/marg.py` for a ~30×30 gradient/curvature-
regularized pixel grid — the ridge marginalization + Occam logdet machinery composes
unchanged behind the whitening conv. Alternative: bridge to herculens wavelets or the
June-2026 RTU-GP source (arXiv:2606.30620) via the array-only likelihood API.
- **Dependency (critic):** requires T0.2's checkpointing first — 128 particles ×
  400 MB already sat at the 80 GB ceiling with a 28-column solve; a ~900-pixel grid
  grows the solve and the AD tape substantially. Not a footnote.
- **Run only if T1.2 moves γ.** Fund exactly ONE of {in-house marg-grid, herculens
  bridge} — they answer the same question (see §5 dedupe).
- ~1–2 weeks impl + ~10 A100-h.

### T2.2 PSF-uncertainty marginalization in the correlated likelihood
PSF is the co-suspect never manipulated: ePSF uncertainty was never propagated in ANY
fit (foundry-i open item); the binned product carries the documented PSF-representation
ceiling (round-trip cos ≤ 0.872); foundry-i history proves γ is violently
PSF-sensitive (the delta_pix defect). Build an ~8–12-dim PSF perturbation basis (PCA of
per-star residuals over the 30 in-field EPSFBuilder stars), sample the coefficients
with star-scatter priors (74-d → ~84-d), rerun correlated SMC on v3b-low.
- **Literature position:** pixellated PSF posteriors exist (arXiv:2511.19594) but
  *nobody* has put PSF uncertainty inside a lens likelihood — open gap, publishable
  either way (γ moves toward 1.433, or the error bar honestly inflates to cover it).
- ~1 week impl + ~5–10 A100-h. The right follow-on to whichever tier exonerates the source.

### T2.3 Certify the pipeline: SBC of the full correlated+SMC chain + source-mismatch mocks
Merge three proposals into ONE shared mock campaign (common truths, shared whitener,
one slurm template) so each SMC run triple-counts as calibration + mechanism +
coverage evidence: (a) staged SBC of the *SMC* pipeline (E1c only certified PHMC) —
16 binned drizzle mocks, γ from prior, production config @96 particles; (b) the
**source-complexity-mismatch mock**: truth rendered with structure outside the fitting
basis (n_max=12 truth fit with prod n_max; clumpy two-component variant) + matched
representable controls — can source misspecification *reproduce the bracket signature*
(binned-corr biased low AND fine-diag biased high) at known truth γ=1.433? (c) the
injection arm from T1.1 if grown to n≥16.
- Pre-register the two mismatch families before running (one family failing to
  reproduce the signature bounds, not proves, the negative).
- ~16 A100-h (halvable by prioritizing the mismatch arm) + free phoenix fine-scale fits.

### T2.4 The endgame arbiter: drizzle-free `flt`-level joint likelihood (~1–2 wks + 10–15 A100-h)
Fit the 3 WFC3/IR F140W exposures of DESI-165 (MAST DOI 10.17909/hx0v-9260) **jointly**:
one lens+source model rendered onto each exposure's native grid with per-exposure
sub-pixel offsets + delta_pix ePSF; diagonal noise *per exposure* — **no resampled-noise
assumption at all**. Validate the renderer on the existing 3-frame mocks first
(gate: |z(γ)| < 1). Can demote either 1.103 *or* 1.433 to a processing artifact, and
prototypes the exposure-level architecture Rubin/Roman will need. Highest information,
highest cost/risk (WCS registration, PSF-undersampling ceiling) → runs **after**
Tiers 0–1 have resolved or sharpened the question. This is also the field's stated
avoidance strategy (arXiv:2302.00480) done right on a GPU.

---

## 5. Critic's implementation warnings (read before building)

Recorded so future implementers don't trip on settled findings:

1. **MAP-level verdicts are inadmissible for the γ question.** MAP γ is unstable along
   the degeneracy valley (1.17–2.58 across configs at Δχ²~0.06); only posteriors
   arbitrate. Any cross-code arm (herculens/JAXtronomy) that falls back to MAP+Laplace
   "for speed" cannot answer whether γ moves off 1.103.
2. **log|C| bookkeeping:** the exact-constant rule (Szegő gap +179.21 nats) binds for
   **cross-whitener** logZ comparisons (T0.1, T0.4-λ) and **cancels** for fixed-whitener
   model ladders (T1.2). Getting this backwards either invalidates the comparison or
   wastes a dense-Cholesky pass.
3. **Memory is a design input, not a risk bullet.** ~400 MB/particle at 28 columns;
   any larger linear basis (T2.1) or parameter extension (T2.2) needs T0.2's
   checkpointing landed first. An L4-23GB cannot hold a meaningful correlated-SMC
   particle count — posterior-level work is A100 work.
4. **PSF sampling conventions:** anything touching supersampling (T1.3) must respect
   `psf_pixel_scale == delta_pix/ss` exactly (0.064125″, not 0.065″) — the guard
   exists because this class of defect produced foundry-i's 2×-broadening.
5. **Low-n coverage gates are vacuous** (n=2–3 cannot measure 68% coverage) — state
   small-n gates as bias/z-scores; graduate to SBC scale for coverage claims.
6. **Population-pilot budgets:** ≤3 A100-h/system is ~2× optimistic (2 basins × 2
   likelihoods ≈ 4 SMC runs + per-system PSF/kernel/whitener/MAP prep; P1c spent
   ~15 A100-h on ONE well-understood lens). Basin enumeration (T0-style census) is an
   unstated dependency of any "per-basin" pilot.

**Dedupe map** (the 20 raw candidates → this program): systematics#1+rigor#15 = T0.1;
rigor#18-stage1+inference#6-seeds+held-P2c = T0.2; systematics#2 = T1.1;
inference#9 = T1.2; systematics#4 = T1.3; systematics#3+scale#13-armB = T2.1 (fund one);
systematics#5 = T2.2; rigor#16+rigor#17+T1.1-grown = T2.3; scale#14 = T2.4;
rigor#19 = T0.4; inference#5 = T0.2's checkpointing + default-stage engineering.

---

## 6. Parallel tracks (no A100 contention)

- **Publish + upstream (zero compute, highest external leverage).** Two papers are
  extraction-ready from `papers/main.pdf` with novelty verified (§1): Paper A — the
  drizzle-covariance likelihood + necessary-but-not-sufficient real-data result
  (motivate with 2302.00480's avoidance + 2402.18636's unheeded warning); Paper B —
  the lens-posterior sampler benchmark + two-stage/per-basin-SMC recipe (GGNS
  2312.03911 and MCLMC-ensemble 2502.06335 are cheap benchmark rows to add if
  extending). Separately: foundry-i documents five actionable upstream gigalens fixes
  (delta_pix assert, NNLS, ridge+logdet in `lstsq_simulate`, full-rank-SVI gate,
  windowed adaptation) + the owed gu-2022 `phys_labels` flag — offer to
  multinode-2025 / GIGA-Lens 2.0.
- **A second real system at N=1 before any pilot** (~5–8 A100-h): GIGA-Lens 2.0's
  DESI J238.5690+04.7276 (published diagonal reference numbers to compare against) or
  one JWST drizzled lens (El Anzuelo — the community-flagged unsolved correlated-noise
  case). Tests whether the over-correction generalizes at all, before pilot-scale
  spend. The N=5 Foundry population pilot (~15–18 A100-h + pipeline hardening) waits
  for this and for the N=1 verdict.
- **Survey coadd noise audit (phoenix-only, ~2 weeks):** measure noise ACFs on
  empty-sky JWST NIRCam drizzled mosaics, an LSST DP1/ComCam coadd patch, and a Roman
  OpenUniverse coadd with `cgl/noise.py`; feed each kernel into an E1a-style mock
  bias arm. Caveat: LSST/Roman use lanczos warping, not tent kernels — the
  two-component family may not fit (the 0.05 ACF gate will catch it). Turns the
  one-instrument result into a field-wide "when do you need this" statement — directly
  actionable for the Euclid SLDE-A pipeline (336 diagonal-noise fits, arXiv:2503.15324).
- **TDCOSMO relevance, step 1 only (<1 GPU-h):** push the two existing DESI-165
  posteriors (diagonal-native anchor vs correlated binned-low SMC particles) through
  the differentiable deflection to Fermat-potential differences for synthetic image
  pairs; report the fractional Δt shift between noise models. Illustrative only
  (DESI-165 is not a TD lens) — but TDCOSMO 2025 (arXiv:2506.03023) has **no error-budget
  line item for coadd noise covariance**, and all TD-lens imaging is drizzled. A real
  TD-lens refit is a separate, later decision.
- **SBI misspecification result (phoenix-only, ~2–4 days, novel + publishable):** no
  published SBI simulator includes drizzle-correlated noise, and E1a *is* the
  demonstration that this misspecification biases γ (median |z|=5.84, cov68=0). Train
  a modest NPE on white-noise mocks, test on correlated mocks with the existing E1
  SBC harness, retrain on correlated, then importance-correct with the exact whitened
  likelihood. Doubles as a calibrated amortized initializer for the SMC pipeline.
- **Inference engineering (phoenix + ~4–8 A100-h confirmation):** (a) saddle-aware
  pre-flight: automate the Laplace-definiteness triage (it predicted PHMC convergence
  across ALL products: +0.108 mixed / −14.85 froze / −6.6e23 catastrophic) +
  negative-curvature-directed basin census — completes the recipe into a mechanical
  pipeline and resolves the un-quotable Euclid θ_E; (b) flow-within-SMC: use the NSF
  as an SMC **mutation/independence proposal** at high-λ anneal stages (the flows were
  never the diagnosed culprit in GL-NT's failure — flow-space trajectory adaptation
  was); train per-basin on the converged on-disk draws, gate acceptance ≥20% at λ=1 on
  T2 before spending A100 time. If it passes, SMC cost halves again and the benchmark
  paper's neural-transport negative upgrades to a constructive recipe.
- **Multi-band archival check (free, an afternoon):** foundry-i states single-band
  F140W understates uncertainty by a factor of a few, and the 2026 substructure
  literature names multi-band as THE accepted source/PSF-vs-mass disentangler. Search
  MAST/ESA archives for any second band on DESI-165 (other HST programs, JWST, Euclid
  coverage) before assuming F140W-only.
- **Multipole bound (near-free):** the second parked P3 pillar (m=3/4 angular
  complexity). Fair to deprioritize — ~1% γ effects cannot explain a 0.33 bracket —
  but a MAP-level multipole ablation on v2d bounds its contribution for near-zero cost
  and closes the parked item honestly.

---

## 7. Strategic synthesis

The decision tree, compressed:

```
T0.1 E3 scan ──not robust──▶ covariance kernel joins the budget; re-scope everything
     │ robust
T0.4 stationarity ──rejected──▶ noise-model CLASS misspecified ─┐
     │ ok                                                        ├─▶ T2.4 flt-level arbiter
T1.1 injection ──biased low──▶ stationary approx convicted ─────┘
     │ unbiased (likelihood certified on real noise → residual IS model misfit)
T1.2 source ladder ──γ moves──▶ T2.1 pixelated/GP source (+ T2.3 mechanism mocks)
     │ γ static
T1.3 ss2 ──anchor moves──▶ 17σ tension reframed; re-derive the bracket
     │ null
T2.2 PSF marginalization (the last suspect standing) ──▶ if γ unifies: DONE;
     if not: T2.4 flt-level arbiter settles which γ is the artifact
```

Whatever branch survives feeds the natural successor campaign — **joint
PSF + flexible-source + mass inference under the correlated likelihood** — which the
literature confirms nobody has assembled and which the JWST substructure controversy
is actively waiting for. Papers A and B (§6) are extraction-ready now and do not
depend on any of this tree.

---

*Full structured provenance: 20 raw candidate directions + critique JSON preserved in
the session workflow output (`wf_86912111-f2f`); readers covered CAMPAIGN.md,
papers/main.tex, the campaign plan, foundry-i, and a 2025–26 ADS/arXiv sweep. Raw
proposal text is deliberately not duplicated here — this note is the deduplicated,
conflict-checked program.*
