# B2′ — re-registered B2 minor-arm-mass gate (scored from existing artifacts)

**Date:** 2026-07-21 (P5 synthesis; experimental program COMPLETE, zero GPU, nothing in flight).
**DECLARATION — evaluated from existing artifacts only:** no new runs, no new samples, no
re-opening of particle files. Every count and cross-check below is taken from
`data/b2_gate_eval.json` (recovery-lane harvest 2026-07-19: orig arm = job 55985447 arm 1,
COMPLETED in-wall; ratio control = job 56006065, COMPLETED 01:49:02), whose indicator counts
were cross-checked against the stored particles AT HARVEST (recompute max|d| 5.6e-17 orig /
2.2e-16 control-θ_E-proxy, both PASS). Particle npz:
`data/results-perlmutter/b2_dspl20_{orig,ratio}_s1_seed2.npz` (untouched by this evaluation).
**File ownership:** this checkpoint + exactly two CAMPAIGN.md gate-record rows (B2′ and the
B4 final-disposition row of the same closure sweep). Cost: 0 A100-h.

---

## 1. Provenance — why a re-registration exists (no silent move)

The original B2 gate (PLAN §6 B2, pre-registered): prior-seeded S1 in the ORIGINAL
pathological (Om0,w0)+NormalCDF coordinates recovers their Run-A arm mass,
`|m̂(Om0<0.146) − 0.103| ≤ 0.045`. Scored 2026-07-16/19:

- Orig arm: **m̂ = 0.000** (0/512 particles) — **FAIL AS WRITTEN** (deviation −0.103).
- Pre-registered falsifier control (dspl20_ratio, exact reparameterization — immune to
  coordinate mode-death BY CONSTRUCTION) landed 2026-07-19:
  **NOT-DECIDABLE-AS-REGISTERED — the 0.103 ± 0.045 Run-A band is UNCALIBRATED for this
  realization** (data_seed=0 is a fresh lenstronomy realization; their baseline is
  unreproducible per their own docstring). The control's OWN minor-arm mass on the SAME
  realization: **m̂_ctrl = 5/512 = 0.0098 ± 0.0043** — itself far outside the Run-A band.
- The ledgered B2 gate row therefore mandated: *"Gate must be RE-REGISTERED against the
  control's own arm mass (B2′, ledgered amendment — not a silent move)."* The wave-1 reading
  against that target (Fisher **one-sided** p = 0.031, two-prop z = 2.24) was explicitly
  labeled *"SUGGESTIVE … NOT conclusive (~2σ, n=1 seed each, below the campaign's 3σ
  convention)"* — i.e. exploratory, not a registered decision.

This file is that amendment: the registered B2′ decision rule (§2) followed by the score (§3).

## 2. Re-registered decision rule (stated in full BEFORE the §3 score was computed)

- **Gate statement:** the orig-coordinate arm's minor-arm mass is CONSISTENT with the
  control's measured minor-arm mass. Both arms sample the SAME realization's posterior
  (identical data, data_seed=0); the control differs only by the exact reparameterization,
  so under the null (no coordinate-driven sampling pathology) the two indicator counts are
  draws from the same arm-mass proportion.
- **Test:** Fisher exact test on the 2×2 table [[0, 512], [5, 507]]
  (rows = orig, control; columns = minor-arm Om0<0.146, rest), **two-sided**.
  Two-sided because after the band recalibration no direction is pre-specified: the
  NormalCDF-ridge pathology could in principle suppress OR inflate apparent arm mass; the
  wave-1 one-sided reading inherited its direction from the now-retired 0.103 band and is
  superseded here, not double-counted.
- **Threshold:** reject consistency at **α = 0.01** (the campaign's ~3σ convention; chosen
  deliberately stricter than the earlier exploratory one-sided 0.031, which was already
  called suggestive-underpowered on the record).
- **Attainable-significance clause (declared as part of the rule, before scoring):**
  Fisher conditions on the margins. With only **5 total minor-arm particles across 1024**,
  the MOST EXTREME possible table (0 vs 5) attains
  p_min = P(X=0) + P(X=5) = 2 × 0.030945 = **0.0619 > α** —
  i.e. **no orig-arm outcome could FAIL this gate at α = 0.01** (nor at 0.05) given
  k_ctrl = 5; falsifiability at α = 0.01 would require k_ctrl ≥ 8 (p = 0.0076 at 0-vs-8).
  Therefore the only honest passing label this evaluation can emit is
  **PASS-UNDERPOWERED (CONSISTENT)** — committed to in advance so a PASS is not oversold.

## 3. Score

| Quantity | Value | Cross-check vs ledgered record |
|---|---|---|
| Fisher exact two-sided p (0/512 vs 5/512) | **0.0619** | — (this is the registered test) |
| Fisher one-sided p | 0.0309 | = the ledgered "0.031" ✓ |
| Two-proportion z | 2.2415 | = json `orig_vs_control_arm_mass_z` ✓ |
| Orig arm m̂ | 0/512 = 0.000; rule-of-three 95% upper **0.0059** | json ✓ |
| Control m̂_ctrl | 5/512 = 0.0098 ± 0.0043 (binomial σ); Clopper–Pearson 95% [0.0032, 0.0226] | json ✓ |
| Hypergeometric pmf at margins (N=1024, K=5, n=512) | P(0)=P(5)=0.0309, P(1)=P(4)=0.1559, P(2)=P(3)=0.3131 | — |

**VERDICT: B2′ = PASS-UNDERPOWERED (CONSISTENT).** p = 0.0619 ≥ α = 0.01 — the orig-arm
minor-arm mass is statistically consistent with the control's measured mass, at a test that
(per the §2 clause, known before scoring) could not have rejected at α for ANY outcome of
this dataset. Verdict label is exactly the pre-committed one.

## 4. Interpretation bounds (what this PASS does and does not license)

**Licensed:**
- No DETECTED coordinate-driven minor-arm suppression at N=512, n=1 seed per arm. Combined
  with the already-ledgered dominant-arm identity (Om0 med 0.4701 orig vs 0.4702 control,
  statistically identical) the prior-seeded S1 sampler is indistinguishable from the
  exact-reparameterization control at every level this realization can measure.
- The B2 headline claim ("the sampler handles what previously required a bespoke exact
  reparameterization") stands at **"no detected pathology"** strength.

**NOT licensed:**
- Proven equivalence. The orig arm's 95% upper bound (rule of three) is 0.59% vs the
  control's 0.98% point estimate: residual minor-arm under-coverage of up to ~×2–3 is fully
  compatible with the data, as is exact consistency.
- A naive reading "P(0 of 512 | m = 0.0098) = (1−5/512)^512 ≈ 0.0066 < 0.01 ⇒ reject" is
  WRONG and anticipated here: it treats the control's estimate as exact truth. Fisher
  propagates the control's own sampling error — that is precisely why p = 0.062, and why the
  earlier 2.24σ two-prop z stays "suggestive", never conclusive.
- A decisive B2′ needs the control-side expected minor count ≥ 8, i.e. roughly ≥ 2× the
  per-arm sampling (more seeds and/or larger N) at this realization's ~1% arm mass — already
  ledgered as "beyond the standing P2 cap"; with the experimental program complete this is a
  FOLLOW-ON, not fundable here.

## 5. Ledger

Zero A100-h (artifact-only evaluation). Gate-record row appended to CAMPAIGN.md by this
sweep (the one row this task owns for B2′). Companion closure sweep:
`research/gate_ledger_final.md`.
