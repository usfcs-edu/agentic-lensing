# T0.4 free checks — stationarity / λ-arm / real-space head-to-head

**Campaign**: claude-giga-lens-linus, PLAN §6 P1 (NEXT_DIRECTIONS T0.4). **Cost**: 0 A100-h
(phoenix only: CPU numpy/scipy + GPU 9 L4 forward renders & two L-BFGS polishes,
GIGALENS_X64=1, old validated stack `/raid/benson/.venvs/cgl`). **Date**: 2026-07-15.

**Context folded in (X1-G0)**: fine & binned constrain the slope at the SAME effective
radius (Δr_eff ≈ 0.008″) yet differ by Δγ = 0.71 — the bracket driver acts AT FIXED
RADIUS. These three checks probe exactly the remaining suspects: (1) the noise-model
CLASS (stationary conv kernel), (2) whitener regularization / information discard at
spectral zeros, (3) which solution the binned data prefers in plain real space.

---

## Verdicts (one line each)

1. **STATIONARITY: REJECTED.** The money product v3b rejects the stationary-kernel
   class at calibrated p = 0.010 (max|z| = 3.67); the arc-excluded v3 arm also rejects
   at p = 0.010 (max|z| = 3.66); all four arms (2 products × 2 pixel sets) fail the
   naive pre-registered 2σ gate and share one coherent, cross-product-replicated
   spatial pattern (top row over-correlated, mid-left under-correlated). Drizzle
   sub-pixel registration variation cannot explain it (predicted block-to-block
   spread ±0.008 in ρ(0,1); observed ~0.28 peak-to-peak on v3b). The noise-model
   CLASS (one stationary conv kernel per product) is misspecified on the field that
   produced γ_binned = 1.103.

2. **λ-ARM: flooring does NOT cure the fine-low gaming — the bias is a genuine
   property of the (stationary) correlated model class, and it lives in the
   DOWN-WEIGHTING of high-power large-scale modes, not at the spectral zeros.** The
   whitened-vs-real-space ordering inversion (railed B beats start A by ~4200 nats
   whitened while real space prefers A by Δχ²_pp = 37; B within 260 nats of the steep
   MAP C while 16× worse in real space) persists essentially unchanged from the
   unfloored production whitener through λ = 1.0 (85% of modes floored); meanwhile the
   exact-normalized likelihood (log|C(λ)| Szegő + dense-Cholesky anchor, correction
   ≤ 7e-4 nats/px) says the data REJECT the floored kernels by thousands of nats per
   step. Information-discard-at-spectral-zeros is falsified as the mechanism;
   "genuine bias of the misspecified stationary correlated likelihood" stands — and
   the guilty modes (high-S, large-scale, the fitted correlated-background component)
   are exactly the ones check 1 shows are nonstationary.

3. **REAL-SPACE HEAD-TO-HEAD: the binned data in real space does NOT prefer the 1.103
   correlated-low solution.** On the same v3b pixels (diag-solved shapelet amps,
   forward renders only): the home-product diagonal-low model (γ=1.293) wins everywhere
   (χ²_pp 1.58 full / 1.93 arc-band); the 1.103 corr-low model is real-space-poor
   (8.32 / 2.76); the 1.433 anchor transfer renders at 7.44 / 3.62. The production
   correlated whitened metric inverts this ordering (corr-low best, −4599 vs −4662
   diag-low vs −5442 anchor) — the correlated likelihood buys its γ=1.103 point by
   sacrificing real-space fit, corroborating the metric/noise-model suspect at fixed
   radius.

---

## 1. Stationarity test (3×3 block grid, two-component kernel fits, null-calibrated)

### Method

- Residual fields: exact production convention (02_fit_noise_kernels.py):
  v = (img − model_map)/err, Background2D-detrended on the production sky set
  (keep & r>4.5″ & |img|<5·med(err); box 26 px v3 / 13 px v3b). **Provenance anchor:
  the production global masked ACF (`noise_kernel_{v3,v3b}.npz: rho_meas_ext`) is
  reproduced bit-for-bit by this pipeline before any block work (both products).**
- Block pixel set: keep & |img| < 5·med(err) (the production faint-pixel cut WITHOUT
  the r>4.5″ exclusion — otherwise the interior blocks are empty; the |img| cut already
  excises the bright object). Center block (1,1) has 0 valid px and is excluded by
  construction; the test runs on the 8 outer blocks.
- Per block: masked, mask-deconvolved ACF (`cgl.noise.masked_acf_2d`, window L = 8 v3 /
  4 v3b = production fit windows) + two-component kernel fit (`cgl.noise.fit_kernel2`,
  production family, full multi-start grid, drizzle anchor `rho_drz` from the
  production kernel npz).
- **Bootstrap bands (residual resampling under the stationary null)**: B = 200 exact
  FFT draws of stationary fields with the GLOBAL fitted kernel
  (`cgl.exact_ref.sample_stationary_batch`, `rho_kernel` 129², grid 512), pushed
  through the identical block masks → ACF → WLS refit (3 starts incl. the global
  params). This calibrates per-block fit scatter including mask geometry and
  finite-block noise.
- Gate quantities: stable functionals of the fitted kernel — w_tot = w_d+w_b and
  ρ_fit at lags (0,1), (1,0), (1,1) (the raw 8-param vector is fit-degenerate; the
  functionals are what the whitener consumes). z_b,f = (F_obs(b) − precision-weighted
  cross-block mean) / σ_null(b). Pre-registered gate: all |z| ≤ 2 ⇒ not rejected.
  Multiplicity-honest verdict: the same max|z| statistic computed on every null sim
  → calibrated p.

### Results

| arm | max\|z\| | naive 2σ gate | calibrated p (B=200) | verdict |
|---|---|---|---|---|
| v3 fine 260² (kernel of whitener_v3) | 2.55 | FAIL (2/32 comparisons > 2) | 0.170 | not rejected alone |
| **v3b binned 130² (kernel of whitener_v3b, the γ=1.103 product)** | 3.67 | FAIL | **0.010** | **REJECTED** |
| v3, arc annulus 1.2–4.2″ excluded (robustness) | 3.66 | FAIL | **0.010** | **REJECTED** |
| v3b, arc annulus excluded (robustness) | 3.08 | FAIL | 0.055 | borderline |

The four arms are not independent (same sky, overlapping pixel sets), but they share
ONE coherent spatial pattern: the top row of blocks is over-correlated (ρ(0,1) z up to
+2.8) and the middle-left block under-correlated (z −3.0 … −3.7) — the SAME sign
pattern in v3 and v3b, with and without the arc band. Excluding the arc annulus makes
the v3 rejection STRONGER (p 0.17 → 0.010): arc residuals were diluting, not driving,
the signal. Cross-product replication on the same underlying sky is exactly what a
real shift-variant noise field predicts and a fit fluke does not.

Structure of the rejection (figs/t04_stationarity_v3b.png and the _noarc variants): a
coherent large-scale pattern, not one bad block — the top row is over-correlated
(v3b ρ(0,1) up to 0.71 vs global 0.60; z up to +2.8) while the middle-left block is
strongly under-correlated (ρ(0,1) = 0.43, z = −3.67; w_tot z = −2.8). The identical
sign pattern appears in v3, in both pixel-set arms. Block variance of the
σ-normalized residual co-varies with it (v3b 0.65 → 1.44 across blocks; informational —
that is D-side shift-variance on top of the K-side kernel variation).

Drizzle phase-structure comparison (the `cgl.noise` overlap-matrix enumeration): the
spread of ρ(0,1) over 200 random 3-frame sub-pixel registrations is ±0.008 (v3:
0.757–0.784; v3b: 0.474–0.491). The observed cross-block spread is ~0.28 peak-to-peak
on v3b — **~17× beyond what drizzle registration variation alone could produce**. The
nonstationarity must come from something else that varies across the skycell (coverage/
weight-map structure — NDRIZIM boundaries — and/or residual astrophysical background),
consistent with the P1a render-check finding that the skycell is shift-variant
(2.1–26.7σ).

### Reading

The pre-registered fork triggers: **the noise-model CLASS (one stationary conv kernel
per product) is misspecified on the money product.** The γ_binned = 1.103 ± 0.008
over-correction sits on a whitener built from a single global kernel that provably does
not describe all regions of the field it whitens. This reframes the over-correction
before any source/PSF spend: T1.1 injection-recovery on the real noise field (already
first in the D7 queue) directly measures the γ bias this induces; a block-diagonal /
locally-stationary kernel class (kernel per region, PSD-by-construction mixture) is the
natural next likelihood iteration, and the correlated `LikelihoodTerm` port (cgl2)
should keep its whitener interface pluggable for exactly this.

Caveats: (i) the block pixel set includes r<4.5″ sky the production kernel never saw —
the no-arc arm controls the arc-band part of that (v3 rejection strengthens, v3b
softens to p=0.055; the shared spatial pattern is unchanged); no single arm is the
whole case — the case is the replicated pattern; (ii) blocks with valid-px counts
816–1802 (v3b) have correspondingly wider null bands — the calibration accounts for
this; (iii) per-block fit quality (max|resid| up to 0.088) is worse than the global
fit's 0.033, as expected at block size; (iv) the four arms share pixels and are not
independent tests.

Artifacts: `data/t04_stationarity.json`, `data/t04_stationarity_noarc.json`,
`figs/t04_stationarity_{v3,v3b}.png` (+ `_noarc` variants).

---

## 2. λ-arm (whitener regularization ordering test) — v3 fine

### Design (stated before reading the λ table)

The fine-low "gaming" pathology (P1c ledger, reproduced here to the nat): L-BFGS under
the production correlated fine likelihood drives γ 1.369 → 1.02 while whitened logp
IMPROVES −11410 → −6407 and real-space χ²_pp WORSENS 30.7 → 71.9. Mechanism fork:
(a) **information-discard/re-weighting at spectral zeros** — the near-singular fine
whitener (raw s_min/mean = 0.0527, ×4.4 amplitude amplification at the most-amplified
modes) rewards fitting the amplified low-S modes and tolerates arbitrarily bad misfit
in the down-weighted high-S (large-scale) modes, where the railed point dumps it; or
(b) **genuine correlated-model preference** that survives regularization. Discriminant:
at large λ the whitener flattens toward diagonal (λ→∞ limit IS real space), so under
(a) the whitened ordering B-vs-C must converge to the real-space ordering as λ grows;
under (b) B stays competitive at all λ. Structural facts fixed before evaluation:
production v3 s_floor = 0.05 (whitener_report.json; the task brief's "0.1
(production)" label corrected), and at 0.05 the floor does NOT engage (floor_frac = 0)
— λ ∈ {0.01, 0.03, 0.05} are the same unfloored whitener, so the informative grid is
{0.05 ≡ unfloored/production, 0.1, 0.3, 1.0}.

Fixed points (exact production machinery, `cgl.e2`; polish settings = prod defaults
rounds=4, iters=200; L4, f64):

| point | provenance | γ | corr logp | ledger check |
|---|---|---|---|---|
| A: v3b-low start | make_start('v3','low'): map_v11_v3cold light + map_v11_v3b_cold2d mass | 1.3694 | −11409.9 | matches P1c preflight (−11410, χ²_pp 30.7) |
| B: railed fine-low MAP | map_polish(A) | **1.0192** | **−6407.1** | matches P1c guard (1.021, −6408) |
| C: fine-steep MAP | map_polish(make_start('v3','steep')) | 2.3228 | −6160.2 | matches e2_v3 logp_map −6159.4 (γ_map 2.281) |

### Results (figs/t04_lambda_arm.png; data/t04_lambda_arm.json)

Whitener rebuilds: `cgl.whiten.build_whitener(rho_kernel_v3, M=20, s_floor=λ,
grid=512)`; λ = 0.05 reproduces the production `whitener_v3.npz` taps to
max|Δh| = 1.0e-9 (build-determinism anchor). Whitened-dof count fixed across λ
(same eroded keep_w, n_e = 37519). log|C(λ)| per pixel: Szegő mean log S_λ, anchored
by dense Cholesky on 16²–48² subgrids with 1/n extrapolation (independent numpy path,
`cgl.exact_ref` style); the anchored correction is ≤ 7e-4 nats/px at every λ — Szegő
is exact at our precision needs.

| λ | e_op | floor frac | ldpp (anchored) | logL_data A | B | C | χ²_pp A | B | C | B−C | B−A |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.01 (≡0.03) | 0.016 | 0.000 | −1.8286 | −10589.8 | −6370.2 | −6110.5 | 30.70 | 67.46 | 4.24 | −259.7 | +4219.6 |
| **0.05 (prod, ≡ unfloored)** | 0.016 | 0.000 | −1.8286 | −10589.8 | −6370.2 | −6110.5 | 30.70 | 67.46 | 4.24 | −259.7 | +4219.6 |
| 0.1 | 0.055 | 0.557 | −1.5670 | −8870.9 | −4792.1 | −4508.9 | 29.52 | 77.21 | 4.05 | −283.2 | +4078.8 |
| 0.3 | 0.055 | 0.781 | −0.7859 | −6745.0 | −2849.8 | −2527.6 | 28.72 | 79.93 | 3.90 | −322.2 | +3895.2 |
| 1.0 | 0.035 | 0.849 | +0.1946 | −5693.6 | −1862.1 | −1522.1 | 28.41 | 83.63 | 3.85 | −339.9 | +3831.6 |

Normalized (cross-λ comparable) logL_norm = logL_data − ½ n_e·ldpp_anchored(λ), Δ vs
production: −3187 (A) / −3328 (B) / −3304 (C) nats at λ=0.1; ≈ −33,060 / −33,450 /
−33,370 nats at λ=1.0. u²/n_e (whitened residual before source marginalization):
A 1.93 / B 6.61 / C 1.08 at production.

### Reading

- **The gaming ordering does not flip at any λ.** Whitened metric: B ≫ A by ~4200
  nats (production) and still +3832 nats at λ=1.0. Real space: A ≫ B (χ²_pp 30.7 vs
  67.5–83.6). Flooring more than half the spectrum (λ=0.1) through 85% of it (λ=1.0)
  shrinks the inversion by only ~9%. **If the pathology lived at the spectral zeros
  (information-discard hypothesis), flooring them would have re-aligned the orderings;
  it does not.** The λ→∞ (diagonal) limit must re-align eventually — but the data
  reject that direction long before it helps (thousands of nats of normalized logL per
  flooring step), i.e. "cure by flooring" costs more evidence than any plausible model
  can repay.
- **Where the pathology actually lives**: the floored modes are the LOW-power ones;
  what flooring never touches is the down-weighting of the high-S large-scale modes
  (S_max/mean ≈ 99 — the fitted w_b ≈ 0.27 correlated-background component). B's
  misfit is real-space-large-scale (u²/n_e = 6.6 pre-marginalization; the ridge-solved
  shapelet source then absorbs ~1e5 nats of whitened residual, leaving B within 260
  nats of C) — the whitener prices large-scale residual structure as "probably
  correlated noise" at ~1/10 amplitude weight, and the railed solution spends exactly
  that currency. Check 1 shows the correlated-background structure varies across the
  field, so this pricing is wrong region-by-region: a coherent mechanism for γ bias
  at fixed radial information (the X1-G0 constraint).
- Fine-low ledger phenomenology reproduced to the nat at production (A: −11409.9
  logp / χ²_pp 30.70; B: −6407.1; C: χ²_pp 4.24 exactly); B's χ²_pp 67.5 vs the
  ledger's 71.9 is regeneration variance of the flat railed plateau (same γ≈1.02,
  same logp to 1 nat).
- Caveats: e_op of the mid-λ diagnostic whiteners is 0.035–0.055 (>the 0.02
  production gate — M=20 taps cannot ripple-match the kinked floored spectrum); the
  quadratic form approximates R'C_λ⁻¹R to that ripple, negligible against the
  hundreds-to-thousands-of-nats gaps. Points are frozen (no re-polish per λ), which
  is the design — σ(logL) sensitivity to the point, not the optimum shift.

Artifacts: `data/t04_lambda_arm.json`, `figs/t04_lambda_arm.png` (+ whitener build
meta in the scratchpad, `t04_lambda_whiteners_meta.json`; the three fixed points in
the scratchpad `t04_lambda_points.npz`).

---

## 3. Real-space head-to-head on the SAME v3b binned pixels

### Method

- Forward renders only, exact production machinery (`cgl.e2.build_target('v3b')` for
  the correlated metric; `cgl.likelihood.build_marg_model(whiten_fn=None)` for the
  diagonal metric). The 28 shapelet source amps are marginalized parameters, so each
  model point gets its ridge-solved amps on the v3b data under each metric — the
  diagonal solve is the real-space-optimal source for that mass+light, which is the
  fair real-space comparison. All 46 other params frozen at posterior medians.
- Points (all on-disk, transforms validated):
  - **anchor 1.433**: `foundry-i/hmc_v13_v2d.npz` per-dim median of the 74-dim paper
    chains → 46-dim marg z via `e2.paper_z_to_marg_z` (ie_scale = cf_v3b, the e2/
    fermat-teaser-validated path). γ round-trips to 1.43298 (=stored chain median to
    5e-8). **Home-product validation: the same point rendered on v2d gives
    χ²_pp = 1.251** (the gated v2d MAP was 1.234) — the cross-product transform is
    sane.
  - **corr-low 1.103**: per-dim median of the 128 P1c converged SMC particles
    (`e2_v3b_low_smc_canary_fix.npz`); γ forwards to 1.10320 (money number 1.1032).
  - **diag-low 1.293** (home-product reference row): `hmc_v13_v3b.npz` low basin
    (γ<1.7 selection, the fermat-teaser convention) per-dim median; γ = 1.29325.
- Metrics: masked real-space χ²_pp on the 16653 keep px; arc-band χ²_pp on the 7825 px
  with r ∈ [1.2, 4.2]″ (where the slope information lives, X1-G0 convention);
  production whitened logL_data for reference.

### Results (figs/t04_headtohead_residuals.png)

| model (γ) | χ²_pp full, diag-solve | χ²_pp arc band | whitened logL_data (prod corr) | logpost (corr) |
|---|---|---|---|---|
| anchor 1.433 (from v2d) | 7.436 | 3.622 | −5442.1 | −5615.7 |
| **corr-low 1.103 (P1c SMC)** | 8.321 | 2.756 | **−4599.2** | **−4683.6** |
| diag-low 1.293 (home ref) | **1.578** | **1.929** | −4662.3 | −5184.3 |

- **Real space vs whitened metric invert**: the correlated whitened likelihood prefers
  the corr-low point over the γ=1.293 diag-low point by +63 nats (logL_data) / +501
  nats (logpost), while plain real space on the same pixels prefers diag-low by
  Δχ²_pp = 6.74 (Δχ²_total ≈ 1.1e5). The 1.103 solution is a whitened-metric
  optimum, not a real-space one — on its OWN product.
- Between the two named models: full-field real-space χ²_pp prefers the anchor
  (7.44 < 8.32; Δχ²_total = −14733), but the arc band prefers corr-low (2.76 < 3.62).
  The anchor row carries a cross-product handicap (its light/source params were fit at
  0.13″ native resolution; rendering at 0.08″ exposes structure v2d could not
  constrain — home-product χ²_pp 1.25 vs 7.44 transferred), so the full-field number
  should not be over-read; the honest summary is that **neither extreme-γ model is the
  binned data's real-space preference — the γ≈1.29 diagonal solution is**, which
  brackets the real-space-preferred slope well above 1.103.
- Residual maps: the corr-low render leaves strong coherent lens-center residuals in
  real space (χ²_pp full ≫ arc band) — the correlated whitener tolerates real-space
  center misfit (same signature class as the fine-low gaming finding).

### Reading

A free ordering check, not a calibrated test (χ² differences on correlated pixels).
It says: the 1.103 over-corrected posterior is not recoverable as "what the binned
pixels actually prefer" in real space — it is manufactured by the whitened metric.
Together with check 1 (that metric's kernel class is rejected on this product) the
over-correction now has a coherent mechanism candidate: a misspecified stationary
whitener re-weights spatial structure in a way real space does not endorse.

Artifacts: `data/t04_realspace_headtohead.json`, `figs/t04_headtohead_residuals.png`.

---

## Synthesis — what the three free checks say together

One mechanism candidate now explains all three results AND the X1-G0 constraint
(bracket driver acts at fixed radius):

1. The per-product stationary kernels contain a large correlated-background component
   (w_b ≈ 0.25–0.27, σ_b ≈ 3–8 px) fitted from sky that check 1 shows is NOT
   stationary across the field (calibrated p ≈ 0.01, pattern replicated across v3/v3b
   and pixel-set arms; drizzle registration structure is ~17× too small to produce it).
2. The whitener prices large-scale residual structure at ~1/10 amplitude weight
   everywhere. Check 2 shows this pricing — not the spectral floor — is what lets
   real-space-garbage solutions stay whitened-competitive (ordering inversion immune
   to flooring 85% of modes).
3. Check 3 shows the production money posterior (γ = 1.103) exploits the same
   currency on its own product: whitened metric ranks it best while the binned pixels
   in real space prefer γ ≈ 1.29 (home diagonal median) by Δχ²_pp = 6.7, with the
   corr-low residual dominated by a smooth large-scale center misfit.

Implications (no thresholds moved; recommendations only):
- The T1.1 injection-recovery on real drizzle noise (already first in the D7 queue)
  is now sharply posed: it measures the γ bias of a stationary-kernel correlated
  likelihood on a certifiably nonstationary noise field.
- The cgl2 `CorrelatedImageData` port should keep the whitener bundle pluggable for a
  locally-stationary (per-region kernel) class; check 1's block machinery is the
  natural estimator for a 2-region or blocked kernel variant.
- The 1.103 number's standing caveat sharpens: it is a whitened-metric optimum under
  a rejected noise-model class — consistent with claude-giga-lens's "necessary but
  not sufficient" verdict and 17σ over-correction.
- E3 (kernel-hyperparameter scan) remains deferred; the λ-arm's normalized-logL
  column already shows the data strongly prefer the measured (unfloored) kernel
  within this family — the interesting scan axis is now the STATIONARITY class, not
  the floor.

---

## Inputs (all on-disk, old validated stack)

- `foundry-i/data/cutout_{v3,v3b,v2d}.npz`, `model_map_v3cold.npy`,
  `model_map_v3b_cold.npy`, `hmc_v13_{v2d,v3b}.npz`, `map_v11_*.npz`
- `claude-giga-lens/data/noise_kernel_{v3,v3b}.npz`, `whitener_v3{,b}.npz`,
  `data/results/e2_v3b_low_smc_canary_fix.npz`, `e2_v3.{json,npz}`
- Machinery: `cgl.noise` (masked ACF, fit_kernel2, drizzle enumeration),
  `cgl.exact_ref` (stationary sampling, dense Cholesky), `cgl.whiten`
  (build_whitener), `cgl.e2` (targets, starts, MAP polish, transforms),
  `cgl.likelihood` (diagonal marg model)
- Analysis scripts (scratchpad, reproducible from the listed inputs):
  `t04_stationarity.py`, `t04_lambda_whiteners.py`, `t04_lambda_arm.py`,
  `t04_headtohead.py`, plotting `t04_plots_1_3.py`, `t04_plot_lambda.py` at
  `/tmp/claude-1306/-raid-benson-git-agentic-lensing/3d232f85-b99e-4114-ab38-6fc594956452/scratchpad/t04/`
