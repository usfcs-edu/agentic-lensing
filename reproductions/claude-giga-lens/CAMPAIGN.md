# claude-giga-lens campaign ledger

Goal: advance lens-modeling quality beyond GIGA-Lens on two pillars — P1 correlated-noise
(drizzle) likelihood + generalized ridge marginalization; P2 sampler/multimodality benchmark
on a lens-posterior zoo → a demonstrated-better "CGL recipe".
Bar: the PRE-REGISTERED verification thresholds in README.md §Verification (frozen at P0,
before any science runs). Retractions are recorded inline in the stage log, never deleted.

Decisions (locked 2026-07-06):
- Scope: P1+P2 depth-first; P3 (GP/pixelated source) parked as follow-on.
- Substrate: vendored gigalens-sean@multinode-2025 ref 58ec9a7 (same as foundry-i), UNPATCHED;
  all campaign code in `cgl/` with array-only likelihood APIs.
- Environments: /raid/benson/.venvs/cgl (aarch64, pins matching the blessed gigalens venv);
  /raid/benson/.venvs/cgl-torch for pocoMC only. Blessed gigalens venv untouched.
- Perlmutter: user cap ≤100 A100-h — commit 90 (P1c 30 / P2c 45 / P3 15), HARD STOP 100,
  ≥56 h of the repo-wide 200-h allocation held in reserve.
  **ALL jobs charge account `deepsrch_g` (NOT cosmo_g — Greg, 2026-07-06). Every slurm
  template hardcodes `#SBATCH -A deepsrch_g`.**
- Phoenix (8×A16 15GB + 2×L4 23GB) is budget-free but ledgered weekly below.
- The lensing likelihood NEVER runs on CPU (216× slower; unit-test 16×16 toys excepted).

## A100-hour ledger  (append BEFORE reading results)

| job id | phase | nodes × walltime | A100-h | cumulative | purpose |
|---|---|---|---|---|---|
| 55600587 | P0/staging | 1 × 00:18:03 debug (exclusive 4-GPU node) | 1.20 | 1.20 | staging smoke: env PASS; A100 parity A–E PASS (A 1.2e-16, B 0.0, C 4.9e-16, D 0.0, E 2.0e-13); priority-fusion livelock CONFIRMED on A100 (600s timeout without flag; 8/8 in 118.6s with) |
| 55678657 | P1c smoke | CANCELLED (debug backlog ~75 min, 0 charged) | 0.0 | 1.20 | replaced by 55680126 on shared QOS |
| 55680126 | P1c smoke | 1 × shared QOS, single A100, 8:26 | 0.14 (actual) | 1.34 | E2 pre-flight DONE: all 6 basin starts valid (logp finite, χ²_pp 1.4–2.5 good; v3-fine-LOW 30.7 FLAGGED); timings v3 1.35/v3b 1.00/v2d 0.22 s/step (8ch) |

| 55683612 | P1c preflight | 1 × shared QOS, 3m05s | 0.05 (actual) | ~1.39 | fine-low guard — result FAIL (see finding below); the guard did its job (production NOT spent on a bad start) |
| ~~55688871~~ | P1c prod (Job 1) | CANCELLED (never ran, 0 charged) | 0.0 | — | VOIDED: resubmitted as 55703707 on cosmo_g (deepsrch_g congested by LensJudge) |
| 55688960 | P1c fine-low disc (Job 2) | 1 × shared QOS deepsrch_g | ~0.05 | ~1.44 | discriminator (done): GENUINE PATHOLOGY verdict |
| ~~55703707~~ | P1c prod (node) | CANCELLED unrun (queue-stuck ~07-17) | 0.0 | — | VOIDED: re-architected into 4 shared jobs below (8 days faster + cheaper) |
| 55712711 | P1c v3b (SHARED) | 1 GPU shared cosmo_g, -t 4:00 | ~3.5 | — | **MONEY product**: v3b binned BOTH basins k300 → γ_binned vs 1.433 (submitted first) |
| 55712712 | P1c v3-fine (SHARED) | 1 GPU shared, -t 3:30 | ~3.0 | — | v3 fine STEEP only k500 (corr vs diagonal 2.585 artifact) |
| 55712713 | P1c v2d (SHARED) | 1 GPU shared, -t 2:30 | ~2.0 | — | v2d native RELAXED both k2000 (H1/H2/H3) |
| 55712714 | P1c v2d-strict (SHARED) | 1 GPU shared, -t 1:30 | ~1.0 | ~24 | v2d STRICT whitener low k500 (strict-vs-relaxed info-cost) |

**Account strategy (user-directed 2026-07-08)**: deepsrch_g is over-subscribed at the account
level (RawUsage 427M vs cosmo_g 35M) + flooded by LensJudge (4+ ljv5 jobs) → gdbenson fairshare
0.145 there vs 0.298 on cosmo_g. So Perlmutter GPU jobs now go to **cosmo_g** while deepsrch_g is
congested (reverses the earlier "deepsrch_g for more hours" default — a scheduling-latency
exception). P1c on cosmo_g. **P2c → cosmo_g CONFIRMED** (user approved 2026-07-08). Balance verified via iris:
cosmo_g gdbenson ~936 node-h user-remaining / ~3620 project-remaining; the ENTIRE remaining
campaign (P1c ~3.5 + P2c ~11 + P3 ~0 node-h ≈ 15) is ~60× under that margin. deepsrch_g has
more total (~1473 user / ~7943 project) but cosmo_g is uncongested + ample. Both P1c and P2c
run on cosmo_g; diagnostics (preflight/discriminator, ~0.1 A100-h) already ran on deepsrch_g,
left as-is. Note: NERSC does not kill running jobs for balance (overdrawn = reduced priority),
so no mid-run-failure risk.

| 55712538 | P2c pre-flight | 1 node × 4 A100 cosmo_g, -t 03:30 | ~14 (16 cap) | ~29.4 | P2C_KIND=all seed 0: 4 T2 (s0/bj_mclmc ×{A,B}) + 4 T3 (s0/remc_pt/nautilus/glnt) + 1 PT-ref + 1 SMC-evidence. Calibrates A100 timings before the full matrix. Queued behind P1c (prio 67679 < P1c 68143 → P1c first) |

**QOS RE-ARCHITECTURE (2026-07-09, verified via sbatch --test-only)**: the 4-GPU NODE
reservations were queue-stuck — worst-case start ~2026-07-17 (8 days out) under gpu_regular
congestion. Single-GPU **shared** QOS jobs start ~TONIGHT 23:09 (backfill into partial nodes)
AND bill fractionally (cheaper). So both P1c and P2c re-submitted as PER-PRODUCT / PER-CELL
`-q shared -C gpu --gpus 1` jobs on cosmo_g: P1c → 4 shared jobs (v3b-money first, then
fine-steep, v2d-relaxed, v2d-strict; ~11 A100-h total), P2c pre-flight → per-cell shared jobs.
Node jobs 55703707 + 55712538 cancelled (never ran, 0 lost). Products/cells independent →
partial results as they land. Lesson: during GPU-partition congestion, unpack node jobs into
shared single-GPU jobs — faster (backfill) AND cheaper (fractional bill).

Committed (pre-rearchitecture): P1c diagnostics ~1.44 + P1c prod ~11 (shared) + P2c pre-flight
~11 (shared) ≈ ~23 / 90. New shared job IDs ledgered as the babysitters report them. Budget lesson applied: shared QOS for small
single-GPU work bills fractionally (0.14 vs the debug node's ~2 exclusive-4-GPU charge).

### P1c production plan FINALIZED (2026-07-08; pending fine-low gate 55683612)
- Prod extrapolation (conservative linear 8→24 chain × leapfrog): v3 fine 7.64 s/step,
  v3b 5.66, v2d 1.23. Budgets (two-stage re-precond PHMC, 24ch×16leap×step 0.1, lean
  300/300 stages): v3 fine 2 basins KEEP=500 ~3.04h/basin (money = 12000 draws); v3b KEEP=900;
  v2d KEEP=2000 (cheap); GPU3 = v3-fine-LOW seed-7 cross-seed replicate (same budget, 1 basin).
- Walltime W = GPU0 v3 (2 basins) ~6.08h; submit `-t 06:45:00` → HARD-CAPS worst-case A100-h
  at 4×6.75 = 27 regardless of extrapolation error. Projected 4×6.08 = 24.3 A100-h; reserve
  5.7 predicted / 3.0 worst-case. Within the ≤30 P1c envelope; P1c total after prod ~25.7.
- Gate: PASS (χ²_pp<3, γ<1.8 post-polish) → babysitter submits production immediately + LEDGER
  message; FAIL → STOP, no fine-budget spend, report fallbacks (v3b-low MAP→fine start / longer
  multi-restart polish / GPU3→strict-whitener native).
- **GATE FAILED 2026-07-08 → DECISION (main)**: run two jobs in parallel —
  JOB 1 (bank healthy node NOW): v3-fine STEEP + v3b BOTH + v2d BOTH + GPU3=strict-native-low;
  secures H1(v3b,v2d), H2(binned vs 1.433), H3, binned/native corr-vs-diag figures,
  fine-steep, strict/relaxed native comparison; ~14 A100-h (fine now 1 basin), -t caps ≤16.
  JOB 2 (fine-low diagnostic, parallel, ~0.05 A100-h): rebuild fine-low from v3b-low polished
  MAP mass (scale-independent) + fine light/source → re-polish → gate. DISCRIMINATES genuine
  fine-whitener pathology (sane start ALSO rails → report fine-low as characterized limitation,
  no fine-low production) vs bad-start (polishes to γ~1.4 → green-light a separate fine-low run).
- **PRE-REGISTERED HEADLINE REFRAME (2026-07-08, before any production result)**: money
  comparison = **γ_binned(corr) vs diagonal-native 1.433** (binned 2× vs fine 3.2×-upsampled;
  fine now MAP-level suspect — a whitener property independent of posteriors). γ_fine(steep) =
  secondary characterization panel. Consistent w/ the H1 fork + foundry-i. This is
  discovery-driven, recorded before the numbers; original γ_fine-vs-1.433 framing stands in the
  log above (ledger discipline).

### P1c smoke findings (2026-07-08)
- Per-step (8ch): v3 fine 1.347s (37519 px) = walltime-setter, v3b 0.998s (9273), v2d
  0.217s (1466). Production 24ch×16leap×2 basins comfortably fits 27 A100-h.
- Basin start sanity: v3 steep χ²_pp 1.39 / v3b steep 2.17 low 2.47 / v2d low 1.87 — all good.
  v2d steep 5.33 (steep disfavored at native — expected). **v3 fine LOW χ²_pp = 30.70 — BAD
  start** (mass-override: v3cold steep-γ light/source + v3b_cold2d low mass = mismatched).
  Feeds both primary fine-low AND GPU3 seed-2 replicate = the γ_fine(low) money number.
  GUARD (instructed to babysitter): after map_polish assert χ²_pp<~3 AND γ<1.8 before
  spending HMC budget; STOP+report if it drifts to steep or won't relax (fallbacks: fine-low
  from v3b-low MAP mapped to fine, longer polish, or GPU3→strict-whitener native).
Charging note: debug QOS allocates the node exclusively (4 A100s billed even at
--gpus-per-node=1). Use shared QOS for small jobs to bill fractionally.

## Phoenix GPU ledger  (weekly rollup)

| week | phase | script family | GPU-h (A16/L4) |
|---|---|---|---|

## Gate record

| gate | date | verdict | numbers | notes/retractions |
|---|---|---|---|---|
| P0 env gate | 2026-07-06 | PASS | 147 pkgs frozen, jax 0.6.2 GPU, all sampler libs import | jax-upgrade incident fixed via constraints.txt |
| P0 parity A–E | 2026-07-06 | PASS | A 6.2e-17 (thr 1e-12), B 0.0 (1e-8), C 0.0 (1e-8), D 0.0 (1e-10), E 0.0 (1e-10) | cgl vs foundry-i `_hmc_lib_marg`, both on vendored lib, L4, f64; data/parity_report.json. INFO: stored pip-era MAP logp reproduced to 6.5e-11; gu-2022 f32 forward image 1.5e-6 rel |
| P0 sampler smoke | 2026-07-06 | PASS 8/8 | NUTS+window 12.6s, MCLMC(un+adj), adaptive tempered SMC (λ→1, finite logZ), TFP REMC, flowMC RQSpline_MALA, flowjax MAF fit, nautilus logZ within 1.0 | blackjax 1.3 fine on jax 0.6.2 — 1.2.5 fallback NOT needed. See XLA defect below |
| E1a artifact | 2026-07-07 | **PASS** | median \|z(γ)\|=5.84 (gate >2), cov68=0 (gate <40%) | diag likelihood on drizzle-correlated fine mocks: ARTIFACT REPRODUCED — the pillar's motivating effect, demonstrated with known truth |
| E1b recovery | 2026-07-07 | **FAIL (confounded)** | fine z̄=−0.654 / binned +1.22 (gate \|z̄\|<0.5); cov68 0.57/0.375 | outlier z's coincide with UNHEALTHY fits (R̂ ≤2.1, minESS 54–238 at reduced budget) — gate confounded by sampler depth; diagnosis pass launched before any Perlmutter spend |
| E1c SBC | 2026-07-07 | **FAIL (γ only)** | γ rank p=6.5e-5 (16/44 first bin), γ cov68=0.34; other 5 params PASS rank gate; pooled cov68 0.49 vs [55,80] | candidate causes: under-mixing vs delta-regularized near-singular fine-kernel calibration (K_reg=(K+0.1δ)/1.1 QC remediation) — analytic-kernel arm + deep rerun will separate them |
| E1d whitener arbitration | 2026-07-07 | **NOT RUN** | no fits in any arm | UNDECIDED; included in diagnosis pass |
| E1b AMENDED (depth-controlled) | 2026-07-08 | **PASS** | z̄(γ) fine −0.359 / binned −0.331 / native −0.045 (gate <0.5); cross-scale 7/8 | two-stage PHMC (re-preconditioned from pooled stage-1 draws) is the production recipe: R̂ 2.11→1.003, ESS 13–22k. Original FAIL row stands above (ledger discipline) |
| E1c AMENDED (healthy-only, n=13) | 2026-07-08 | **PASS (low-n caveat)** | all 6 params rank p ≥ 0.19 (γ p=0.53); pooled cov68 0.615 ∈ [55,80] | γ pathology was sampler-induced (stuck chains → U-shaped ranks); definitive full-64 staged re-run (~35 GPU-h) queued post-P2b on phoenix |
| E1 D2 kernel attribution | 2026-07-08 | **NEITHER** (kernels exonerated) | fitted arm z̄(γ)=−0.14 cov 0.83; analytic arm z̄=−0.47 cov 0.75 — both calibrate | original failures were sampler depth/metric, not kernel fitting or δ-regularization |
| E1d AMENDED | 2026-07-08 | **RELAXED ADOPTED** | max\|z̄\|=0.492 (thin margin, flagged); cov68 0.594; kept 982 px = 4.9× strict | diag FAILS under real v2d kernel (z̄(γ)=−6.1) — artifact strong even at native scale. E2c uses relaxed whitener |
| P1c fine-low guard | 2026-07-08 | **FAIL → FINDING** | polish railed γ 1.369→1.021, χ²_pp 30.7→**71.9**, corr logp −11410→**−6408** (as good as healthy steep −6163); fine-steep self-check polished sane (χ²_pp 1.39→4.24, γ stays 2.44) | The near-singular FINE (3.2×-upsampled) δ-reg whitener is GAMED: small whitened residuals at a real-space-garbage γ≈1.0 model → the correlated likelihood cannot constrain the fine low basin. Manifestation of the E1b σ_fine/σ_native=2.45 spectral-zeros characterization as an outright MAP pathology; consistent w/ foundry-i "native/binned is the defensible headline". Diagnostic (v3b-low→fine-scale start) queued to confirm genuine-pathology vs bad-start |
| P1c fine-low discriminator | 2026-07-08 | **GENUINE PATHOLOGY (confirmed)** | v3b-low MAP sane (χ²_pp 2.47→4.07, γ→**1.277** — validates binned money product); spliced fine-low from that SANE mass STILL rails: χ²_pp 79.8→4.51 (relaxes) but γ 1.277→**1.137** (below binned-consistent 1.28, toward the ~1.0 rail; hits exclusion γ≤1.15). **Two independent sane starts (γ 1.02 & 1.137) both rail low ⇒ NOT a bad-start artifact.** | Intrinsic property of the near-singular fine whitener: systematic low-γ bias at the MAP, low basin unconstrained. **FINE-LOW EXCLUDED — no fine-low production.** Validates: (a) headline reframe γ_binned primary; (b) v3b as money product; (c) Job-1 fine-steep-only design. Evidence: data/results/e2_fine_low_{preflight,discriminator}.json |
| E1b width-ratio sub-gate | 2026-07-08 | **FAIL (characterized)** | median σ_fine/σ_native = 2.45 (gate [0.7,1.5]), incl. fully-healthy pairs | fine posterior is CONSERVATIVE, not biased (z̄+cross-scale pass): δ-reg whitener (λ=0.1) discards information at the tent kernel's spectral zeros. Pre-registered amendment: λ-sensitivity arm added to E3; H3 (real-data honesty gate) unchanged |

## Stage log

### P3 Euclid converged results — 2026-07-09 (phoenix, diagonal ss1, secondary deliverable)
Mixed/HONEST outcome (not a clean 3/3), a native-resolution characterization:
- 102157952 (θ_E,pub 1.11″, q 0.82, high S/N): CONVERGED (R̂_θE 1.08, ESS 449) → **−1.4%** CLEAN ✓
- 102157958 (θ_E,pub 0.85″, q 0.56 elliptical): **CONVERGED** (R̂_θE 1.04, ESS 2905) but **−32%** →
  genuine per-system discrepancy, NOT non-convergence (converged to a biased/degenerate θ_E)
- 102020061 (θ_E,pub 1.24″): NON-converged even at 48 chains (R̂_θE 18.8) → multimodal θ_E posterior
- Interpretation (to verify): native-resolution ss1 single-band VIS modeling works for
  well-resolved round high-S/N systems, but is biased/multimodal for small/elliptical ones —
  the PSF/pixel-undersampling (foundry-i R0c) + source-model-choice (2406.08484) themes.
  Recipe + COOLEST export are the WORKING P3 deliverables; Euclid θ_E is an honest limitation.
- Refit budget over-specified by finalizer (48ch × 3400 steps × 12 leap ≈ 11× working config
  → ~4-5h on A16; secondary, not re-run). Fresh finalizer diagnosing why 1&3 fail + closing P3.

### P3 recon — 2026-07-08 (de-risk complete; plan at research/notes/p3-euclid-coolest-recon.md)
- **SCOPING DECISION**: Euclid Q1 VIS is native 0.1″/px (NOT drizzle-resampled) → diagonal
  RMS. So **P3-Euclid demonstrates the Pillar-2 SAMPLER recipe on independent real data, NOT
  the P1 correlated likelihood** (correlated machinery applies only to resampled NIR, which we
  don't fit). Cleaner story; runs on phoenix L4 — **no Perlmutter for Euclid**; P3 ≤15 A100-h
  mostly unused. Correlated likelihood's real-data test remains the HST cross-scale P1c.
- 185 grade-A Euclid systems with published SIE models; flagship trio picked (θ_E,eff
  0.85/1.12/1.24″, clean single-plane, low shear, S/N 241–425). No multi-plane needed.
- Only real code change: parameterize build_marg_model priors (theta_E_med≈1.0,
  mass_center_sig~0.1) — defaults preserve HST parity bit-for-bit. PSF renormalize + verify
  not oversampled. Comparison at MASS level (einstein_radius_effective; ~5–10% PyAutoLens-vs-
  gigalens convention offset expected, not a digit-match).
- COOLEST 0.1.11 API mapped (COOLEST('MAP') container, PEMD+ExternalShear+Sersic+Shapelets);
  ridge-marginalized amps → MAP-mode file + chains sidecar + MAP model-image FITS.
- Scripts: cgl/euclid_io.py, 30_recipe_e2e.py (HST recipe, reuse P1c infra), 31_fit_euclid.py
  (diagonal), 32_coolest_export.py. No hard blockers.

## Stage log

### P1c — 2026-07-08 (RUNNING on Perlmutter; smoke job 55678657)
- Relaxed v2d whitener built: data/whitener_v2d_relaxed.npz M=10 e_op=0.0312, keeps 1466 px
  (3.0× strict; MC whiteness Var 1.0013 offdiag 0.0029). E1d mock kernel == real v2d kernel
  bit-identical (validated).
- 74→46 start mapping (foundry-i R8): forward map_v11 z through paper bijector, drop 28
  shapelet amps, scale 5 Sérsic Ie by conversion_factor=delta_pix²; γ round-trips exact,
  logp finite, diagnostic χ²_pp 2.2–2.5.
- **DEVIATION (start labels, real error caught)**: pre-reg brief mislabeled the v3b basins —
  map_v11_v3b_cold measures γ=1.465 (LOW), cold2d γ=1.369 (LOW); the genuinely STEEP v3b MAP
  is map_v11_v3b_WARM (γ=2.672). E2b now seeds warm=steep / cold2d=low. Fine (v3) low basin
  has no own-scale MAP → built from v3cold light/source + cold2d mass override (mass is
  scale-independent). Noted in cgl/e2.py.
- **STACK DEFECT #7 (found + fixed)**: correlated-marg batched grad (vmap over chains of 28
  per-column conv-whitening ops) LIVELOCKS XLA compile (100% CPU, GPU idle, 20+ min;
  priority-fusion-off alone did NOT fix). FIX: collapse 28 per-column convs → ONE
  depthwise/grouped conv (build_marg_model_grouped in cgl/e2.py) — logp BIT-IDENTICAL
  (absdiff 0.0), batched grad compiles in 13.8s. Both XLA flags set in e2 slurm templates.
- Sampler: fixed-leapfrog PHMC (num_leapfrog=16) + DualAveraging, two-stage re-preconditioned
  (P1b recipe); GBTLA dropped (meta-grad livelocked); metric = Laplace Hessian at seed via
  1/|eig| (near-zero eig flooring blew R̂ to 1e31 — must NOT floor to zero).
- Agent reports smoke A100 timings + production plan (per-job A100-h) BEFORE production submit.
- Production plan APPROVED: 1 node × 4 A100 regular QOS; GPU0=v3 fine / GPU1=v3b binned /
  GPU2=v2d native (2 basins sequential each); GPU3 (else idle, node bills 4×) = 2nd
  independent seed of v3 fine LOW basin → cross-seed robustness on γ_fine(corr), the
  campaign's central number (fallback GPU3 = v2d strict-whitener native). Sized so
  4×walltime ≤ 27 A100-h, ≥3 reserve. Gates H1/H2/H3 per 11_pool_e2.py, H2 anchored on
  diagonal-native 1.433.
- WHITENER CHECK (L4, budget-free): native prior-pull is INFORMATION-limited, not
  whitener-misspecification — strict (487 px) γ_std=0.109 vs relaxed (1466 px) γ_std=0.051;
  more px → tighter → closer to 1.433. Relaxed (D3-adopted) correctly the more informative
  native choice; the real-data analog of E1b σ_fine/σ_native=2.45. Direction robust
  (test-budget R̂ 5–25; converged runs give absolute γ).
- **PRE-REGISTERED AMENDMENT (2026-07-08, before converged runs)**: the native relaxed
  whitener (1466 kept px) is likelihood-weak → correlated-native γ prior-pulled toward ~2.0.
  Therefore H2/H3 anchor on the DIAGONAL native fit (γ=1.433 [1.400,1.469], σ_native,diag),
  NOT the correlated-native fit. Money comparison = γ_fine(corr) vs 1.433 (does corr pull the
  fine artifact γ=2.585 back to the trustworthy native anchor). H3 denominator re-spec'd to
  σ_γ(native,diag); correlated-native σ reported separately as pixel-loss information-cost
  diagnostic. Rationale: native is where the diagonal likelihood is least wrong (ρ(1)≈0.5 vs
  0.8 fine); corr's value-add is largest on fine (37519 px) / binned (9273 px). Local L4
  check queued: does stricter-e_op/larger-M native whitener recover data-drivenness.

### P1b diagnosis — 2026-07-08 (COMPLETE; P1c GREEN-LIT)
- ROOT CAUSE of E1b/E1c failures: floored-SVI covariance is too poor a momentum metric for
  near-degenerate 22-dim posteriors (3× depth alone still R̂=3.11). FIX (production recipe,
  pre-registered for P1c): **two-stage PHMC** — stage-2 re-preconditioned from pooled
  cross-chain stage-1 draws; same fit → R̂ 1.003, ESS 13–22k, ~1.9× cost (34 min fine/L4).
  63/63 deep re-runs clean.
- Depth-controlled gates: E1b PASS all scales; E1c healthy-only PASS all params; D2 kernels
  exonerated (both arms calibrate); D3 relaxed v2d whitener ADOPTED (4.9× pixels).
- Honest standing finding: σ_fine/σ_native = 2.45 — the δ-regularized fine whitener is
  conservative (information discarded at spectral zeros), not biased. λ-sensitivity arm
  added to E3 (pre-registered BEFORE P1c). Report will present this as a characterized
  cost of near-singular drizzle covariances, with exact-GLS information loss quantified.
- Mid-run QC (documented in e1_report deviations): δ-reg fallback extended to e_op-gate
  failure; diag-shapelet marg crash fixed; quarantines re-run; evidence in e1_quarantine/.
- P1c conditions adopted: (1) staged sampler default; (2) width gate re-specified as above;
  (3) definitive full-64 E1c staged re-run queued on phoenix post-P2b (non-gating).
- Diagnosis cost ≈95 phoenix GPU-h (campaign local total ≈165 GPU-h; Perlmutter still 1.2).

### Perlmutter staging — 2026-07-06 (COMPLETE)
- Layout: /global/cfs/cdirs/deepsrch/gdbenson/claude-giga-lens (venv 6.2G + repo 187M with
  tree-shape reproductions/{claude-giga-lens,foundry-i-data-subset,gu-2022-subset});
  ~/claude-giga-lens is a SYMLINK to it ($HOME was over 40GiB quota — pip cache purged,
  claudenet/foundry-i untouched). Venv pins exact, zero resolver drift vs phoenix.
- **A100 parity A–E PASS** (fresh run, job 55600587): A 1.2e-16, B 0.0, C 4.9e-16, D 0.0,
  E 2.0e-13; logp(z_ref) identical both stacks; |vs stored MAP logp| 7.3e-12. The campaign
  stack is validated on BOTH architectures.
- **Priority-fusion livelock CONFIRMED on A100** (not aarch64/L4-specific): sampler smoke
  hung 600s-timeout without the flag, 8/8 in 118.6s with it. POLICY: the flag is DEFAULT in
  all Perlmutter sampler processes (P2c). Watch item CLOSED.
- Slurm templates: remote copies point cd at repo/reproductions/claude-giga-lens (one level
  down); mirrored locally this commit. Remote smoke_staging.slurm kept remote-only.

### P2b — 2026-07-08 (COMPLETE; benchmark delivered)
- Track A 345/351 (6 = pre-registered structured failures) + Track B 24 cells. Pool = 370
  cells, data/bench_report.json + figs/bench_matrix.png. 202.4 phoenix GPU-h; Perlmutter 0.
- **Headline (two-sided, publishable):** nautilus dominates ESS/grad everywhere it applies
  (2.6×–307×, best logZ, recovers modes) BUT is DISQUALIFIED on cond-1e14 (t0_illcond46
  shells collapse) — i.e. on the real-lens 46-dim marg regime; its ESS is importance-weight
  ESS + weak pseudo-R̂ (semantics caveat recorded). Under budget parity NO gradient method
  beats S0 on eval-T1. Under until-converged, reliability inverts: remc_pt (PT-HMC) converges
  5/6 incl. sys006 where S0 fails; S0 uniquely OWNS cond-1e14 (only healthy gradient method
  there besides bj_mclmc).
- **flowMC = structural failure on lens posteriors** (scalar MALA, no per-dim precond in
  0.4.5): R̂ 1.9–2.1, converges 0/6 even at 4× budget, inverts modes on t0_mix22. Benchmark
  datum, not a bug.
- **GL-NT recipe verdict @ Track-A budget: FAILS its own bars** — T1-hard 0.03–0.05×
  (target ≥3×), easy-target regression, R̂ 1.4–2.0 (flow-space ChEES is the weak stage).
  Mode recovery real but partial (clears 0.05 bar only in f64) and not unique. T2/T3 not yet
  evaluated (P2c). Lowest P2c priority; flow-space HMC needs re-tuning or A100 budget.
- Mode collapse reproduced (S0/NUTS/neutra → 1.000/0.000 on t0_mix2); nautilus + remc_pt
  most reliable recoverers; flowMC inverts on t0_mix22.
- Freeze GENERALIZES: no dev→eval overfit (ratios ~1.0; flowmc/glnt consistently poor, not
  overfit) — validates the pre-registration protocol.
- **Contender-pick honesty flag**: Track-B set {flowmc,neutra,remc_pt} ≠ "3 best by ESS/grad"
  {remc_pt, bj_mclmc, neutra}; not re-run (valid data, deliver-now). bj_mclmc (only
  cond-1e14-safe gradient method) added to P2c picks below.
- **P2c picks (corrected):** T3 bimodal → nautilus (within-mode ESS + mode sensitivity, NOT
  R̂) + remc_pt (RE-TUNE β-ladder first — it blew up R̂ 1e16 on cond-1e14) + S0 reference;
  T2 cond-1e14 → bj_mclmc (auto SVI-diag mass = honest "no hand-built matrix" challenger) +
  S0 (incumbent, owns it). glnt = recipe-under-test on T3 only, drop-first if time-tight.

### P2b checkpoint — 2026-07-06T22:08Z (POLICY FREEZE — committed before any eval read)
- data/policies_frozen.json + data/policy_tuning_log.json (33+ trials, ≤4 configs/method,
  dev split only). Budgets: T0 2e5 / T1 1.5e6 grads (+692k billed init-cache for consumers).
- **REGISTERED FAILURE**: flowMC 0.4.5 on T1 — scalar MALA step (no per-dim preconditioning
  in 0.4.5) gives acceptance ≈0 on the ill-conditioned z-space at both 0.05 and 0.003 step
  sizes. T0 works well (mix2 occupancy 0.816/0.184, 410 round trips). Benchmark datum.
- MCLMC tuner produces NEGATIVE inverse-mass entries on cond-1e14 → all-NaN; frozen policy
  uses svi_diag mass on T1 (ESS 6001, R̂ 1.004 in dev) with deterministic fallback on T0.
- **GL-NT tier split (the recipe finding so far)**: T0 anneal-from-PRIOR recovers the
  mixture (0.778/0.222 vs true 0.8/0.2 — inside the 0.05 bar, vs plain NeuTra total collapse
  1.000/0.000 and baseline 0.951/0.049); T1 uses SVI→posterior anneal path (prior-path
  infeasible: λ=0.198 after 100 steps). Flow-space ChEES still the weak stage on T1
  (trial R̂ 1.65) — the matrix judges.
- NUTS budget honesty: uncapped tree depth blows the grad budget 1.9× on T1 → frozen at
  2^6 cap (weak on illcond by construction; recorded).
- Stack findings #4-6: priority-fusion livelock ALSO in f32 (MCLMC tuner/NSF pipelines;
  flag extended in 22_run_cell for bj_mclmc/neutra/glnt); SMC particle-with-grad memory
  ceiling ~21MB/particle (384 particles on L4; 1200 = 24GiB OOM); nautilus jitted
  log_like_x TracerArrayConversionError on new batch sizes → batch padding + bijector
  template probe (the e2/e1 reorder trap again).
- S5 pocoMC skipped (pre-approved). Matrix launching: 135 T0 + 54 dev-final + 162 eval
  cells, frozen policies, resumable queues on CUDA 9 + A16 0-3.
- Interim (94/351): structured failures all diagnosed — SMC/glnt PRIOR-anneal cannot bridge
  cond-1e14 (~1e16-nat range) on t0_illcond46 (pre-registered failure flag, not a bug);
  nautilus shells defeated by cond-1e14 (<8 equal-weight pts in 74 min → failure-flag guard).
  Signal: nautilus dominates T0 ESS/grad (2.9–169×), mclmc 17× on funnel, S0-precond owns
  illcond46; ON EVAL T1 NOTHING BEATS S0 YET under budget parity (mclmc-svidiag closest).
  Mixture mode recovery: remc/flowmc/nautilus/glnt recover both modes; S0 collapses.
- **DEVIATION (accepted 2026-07-07)**: Track-A wall-time ~12–16h (frozen T1 budgets ≈25–40
  min/cell on A16 × 351 cells / 5 GPUs); Track B scope-reduced to S0 + 3 best contenders,
  seed 0 only, T1 eval systems. Rationale: 3-seed ESS/grad medians already come from Track A;
  Track B's until-converged story on the targets that matter most (T2/T3) is P2c's A100 job.
  Perlmutter budget untouched.

### P2a — 2026-07-06 (COMPLETE, all 4 gates PASS)
- Zoo frozen: 19 targets (5 T0 + 12 T1 + T2 + T3; T4 stub), 3 seeded z-points each with logp
  + sha256 in data/zoo_freeze.json; cross-process re-checks bit-identical incl. f64 A16↔L4.
- Validation gates: (i) T1 zoo vs direct construction BIT-LEVEL 0.0 (sys000+003, 8 pts);
  (ii) T2 logp(qz_refined) = −45840.984005998456 = parity value exactly (L4); equals stored
  npz bit-for-bit on A16 (the 6.5e-11 was pure L4↔A16 drift — now fully explained);
  (iii) T3 mass_* bijector reproduction 0.0; basins measured from stored chains:
  45 low (γ̄=1.2939) / 3 steep (γ̄=**2.4159** — CORRECTION: briefed "≈1.5" was wrong; zoo
  Reference records the measured value), zero γ=1.8 crossings/round trips;
  (iv) T0 analytics all pass (logZ to 4.2e-14 f64); (v) prior+like==prob worst 4.9e-8 f32.
- S0 baseline validated end-to-end; documented pathologies visible: t0_mix2 occupancy
  0.951/0.049 vs true 0.8/0.2 (the mode-collapse the recipe must fix); sys003 γ min-ESS 179
  (ChEES) / 99 (stored-fit config) vs light-params ~1255.
- **STACK DEFECT #3**: jaxlib 0.6.2 XLA triton GEMM aborts on tiny f32 dots (dim-2 SVI) on
  L4 @ autotune-0 → all zoo processes set --xla_gpu_enable_triton_gemm=false
  (cgl/zoo/runtime.py, f64 immune). Freeze rebuilt under final flags.
- **METRICS BUG CAUGHT (would have corrupted the whole benchmark)**: arviz axes fed
  (draw,chain) as (chain,draw) silently hides stuck chains (known-R̂-2.07 fit came back
  1.00). Fixed in cgl/metrics.py + regression-pinned; all cell metrics recomputed.
- **gu-2022 archive finding**: stored fit phys_labels are per-block REVERSED vs true z-leaf
  order (block-contiguous, so archived mass-set aggregates fine; per-param attribution within
  blocks reversed). Zoo uses probed labels. Flag upstream to gu-2022/foundry eventually.
- sys003 pathology attribution: archived severity partly the SHORT-SVI preconditioner
  (Bug-2 class); under guard-mandated schedule the pathology persists but milder — the P2b
  baseline is therefore the HONEST (stronger) variant.
- T2 reference wired: long_diagraw_s0..7 (8×8000, per-chain ess_min 3.5–7.4, mixing caveats
  recorded). Adapter API frozen for P2b (run_cell + freeze-point fidelity assert; batch-size
  warmup contract; single-dtype-per-process via cgl/zoo/runtime.setup_process_env).
- Cost: ~2.3 L4-h + ~10 A16-min. Tests: 73 CPU + 10 GPU green.

### P1a — 2026-07-06 (COMPLETE, all gates PASS)
- 02 kernels (model-subtracted, guard-enforced): fit residual v2d 0.0448 / v3 0.0270 /
  v3b 0.0326 (gate ≤0.05). Drizzle anchor: enumerated t(1)=0.76799 vs closed form 0.76805
  at r=3.2075. Block-sum cross-check 0.0308 PASS (commuting processing); product-level 0.0595
  informational (Background2D detrend does not commute with binning).
- **DEVIATION 1 (accepted)**: plan's 2-param kernel family cannot pass (best 0.090–0.311);
  residuals carry a medium-scale correlated pedestal + anisotropic core + v2d column stripes.
  Adopted minimal PSD-by-construction extension: (1−w_d−w_b)δ + w_d·(ρ_drz⊛G₂) + w_b·G₂
  (two-component bivariate). Single-family failure numbers on record in noise_kernel_report.
- 03 whiteners: v2d M=14 e_op=0.0177; v3 M=20 e_op=0.0160; v3b M=10 e_op=0.0124.
  MC/dense whiteness all PASS (Var(u)=1.000, off-diag ≤4e-5). Construction fixes: kernels
  stored analytic to half-width 64 (truncation ringing); ADAPTIVE s_floor (hard 0.05 floor on
  a PSD kernel biases Var(u) — 0.981 FAIL → adaptive PASS).
- **OPEN FLAG → P1b decision**: v2d erosion loss 91.7% (5865→487 px; border-14 + mask blobs
  × 29² stencil). DECISION (campaign lead, 2026-07-06): P1b builds a RELAXED v2d whitener
  (target e_op ≤ 0.05, M≈6–8) alongside the strict one; mock recovery/coverage arbitrates;
  E2c uses the relaxed one iff mock calibration holds. Ledgered as a pre-registered-exception
  candidate, evidence-driven.
- 04 HARD GATE: |ΔlogL| conv-whitened GPU vs dense-C CPU reference at 20 prior draws:
  v2d worst 2.79e-9 nat, v3b worst 6.26e-7 (gate <0.1) — PASS. Gate semantics documented:
  this certifies implementation-exactness of the whitened functional (C⁻¹ := G_eᵀG_e);
  physical-C misspecification is bounded separately by e_op (0.1-nat exact-GLS equivalence
  is mathematically unattainable at prior draws — analysis in 04 docstring). Constant
  accounting closed: exact logdetC vs Szegő gap +27.30 (v2d) / +179.21 (v3b) — cross-whitener
  evidence comparisons MUST use exact constants.
- 05 mocks: 8 trios, render gate worst 2.1e-12σ (<0.05σ). Exact analytic covariances per
  product (fine tent ρ(1)=2/3, binned 0.4, native iid).
- **DEVIATION 2 (accepted, report-worthy)**: 3-dither Latin stack {(0,0),(1,2),(2,1)} is
  provably SHIFT-VARIANT (no convolutional effective PSF exists; render check 2.1–26.7σ —
  preserved at data/mocks_report_3frame.json). Mocks use all 9 phases → exact separable
  3×3-tent convolution. NOTE FOR REPORT: the REAL v3 skycell (NDRIZIM=3) is likewise
  shift-variant — its "effective PSF" is an approximation; discuss as a real-data caveat.
- Real-data teaser (informational, NOT a result): on the v3 MAP residual,
  RᵀC⁻¹R/n_kept = 0.458 under the correlated C (CG, rel-resid 1e-8) vs diagonal χ²_pp 0.4515.
- Perf: conv-whitened logpost/grad ≤1.6× diagonal (v3 40/85ms vs 25/55ms on L4) — no
  grouped-conv refactor needed. photutils 3.0.0 installed under constraints (Background2D).
- Tests: 57 CPU + 10 GPU green; parity A–E re-verified fresh post-change.

### P0 — 2026-07-06 (campaign start)
- Branch `claude-giga-lens`; work dir scaffolded per approved plan
  (`/home/benson/.claude/plans/this-repo-containes-astrophysics-radiant-sprout.md`).
- Vendored gigalens-sean copied bit-identical from foundry-i vendor (ref verified 58ec9a7);
  `cgl.paths.bootstrap_vendor()` asserts the ref on every import.
- venv /raid/benson/.venvs/cgl building (core pins = blessed stack: jax 0.6.2, tfp[jax] 0.25.0,
  numpy 2.4.6, scipy 1.17.1, optax 0.2.8, objax 1.8.0, astropy 7.2.0, lenstronomy 1.14.0;
  samplers: blackjax 1.3.*, flowMC 0.4.5, flowjax 19.0.0, equinox 0.13.8, nautilus 1.0.6).
- `cgl/guards.py` written: six guards, each encoding a prior real incident (PSF delta_pix
  convention ×2 incidents, pmap-all-devices hang, SVI cov rank deficiency, f32 grad floor,
  wing-contaminated sky calibration, CPU ban).
- venv gate incident (recorded): the first sampler install silently upgraded jax 0.6.2 →
  0.10.2 (culprit: unconstrained chex 0.1.92), breaking tfp 0.25.0. Fixed by re-resolving
  under `constraints.txt` (chex pinned back to 0.1.90, lineax 0.1.0); ALL future pip
  installs into cgl venvs must pass `-c constraints.txt`. Env gate now PASSES
  (00_env_check.py; freeze at data/env_freeze_phoenix.txt, 147 pkgs).
- GPU inventory confirmed: CUDA indices 0–7 = A16 15GB, 8–9 = L4 23GB (nvidia-smi
  authoritative; jax device_kind mislabels all 10 as "NVIDIA L4").
- 12/12 CPU unit tests green (guards, paths, vendor shadowing).
- requirements-{aarch64,perlmutter}.txt written; slurm/{smoke,prod}.slurm hardcode
  `#SBATCH -A deepsrch_g`.
- In flight (3 agents): sampler runtime smoke (GPU 9); marg-core port + parity harness A–E
  (GPU 8); GIGA-Lens-2.0 positioning note (web).
- GIGA-Lens 2.0 scope check DONE (research/notes/gigalens2-positioning.md): arXiv 2606.30633
  verified = multi-node scaling of the UNCHANGED diagonal-likelihood MAP→SVI→HMC recipe
  (128 nodes/512 A100s; real system DESI J238.5690+04.7276, 38 params, R̂<1.01, χ²ν=0.8954).
  Zero overlap with P1 (no correlated noise, no marginalization/Occam) or P2 (no
  multimodality/tempering/flows). Risk-register item 7 RETIRED. Their real-system result is
  a natural P2 baseline; their linear-inversion step is exactly the machinery P1 upgrades.
  Watch-list (nearest, non-competing): 2406.08484 (flags correlated noise as open, no
  implementation), 2511.04792 (score-based SBI), 2410.22573 (flow-matching NPE).
- Parity A–E PASS (gate record); cgl marg core is bit-identical to the validated foundry-i
  stack on the vendored lib.
- Sampler smoke 8/8 PASS (tests/test_sampler_smoke.py is the API seed for cgl/samplers/).
  **NEW STACK DEFECT (guard-worthy):** jaxlib 0.6.2 XLA `priority-fusion` pass LIVELOCKS
  (infinite compile, ~600 threads) fusing f64 `jax.random.normal` (erf_inv) with a reduction
  on L4/aarch64 — exactly blackjax MCLMC's `partially_refresh_momentum`. Workaround:
  `XLA_FLAGS=--xla_disable_hlo_passes=priority-fusion` (set pre-import; XLA_FLAGS parsed
  once). f32 unaffected; TFP-PHMC/NUTS f64 unaffected (foundry-i history confirms). Policy:
  sampler adapters that draw f64 normals inside jitted kernels (MCLMC family) set the flag
  in their own process (22_run_cell.py runs one process per cell); measure A100 impact
  during the first Perlmutter smoke before making it default there.
- flowMC 0.4.5 adapter requirement: re-init the bundle optimizer as
  optax.chain(clip_by_global_norm(1.0), adam(lr)) over eqx.filter(model, is_inexact_array)
  (upstream inits adamw over bool spline masks → jax 0.6.2 tree mismatch).
- flowjax 19 requires new-style typed keys (jax.random.key, not PRNGKey).
- Full-suite livelock incident (recorded): the combined `pytest tests -m gpu` session hit the
  priority-fusion livelock (698 threads, 40 min) because an earlier test module's imports
  initialized the JAX backend before test_sampler_smoke.py's import-time XLA_FLAGS append.
  Fix: 00_run_tests.sh now sets the full XLA_FLAGS in the process env. Lesson: XLA flags are
  process-level, set them in the runner/launcher, never per-module.
- **P0 EXIT GATE MET 2026-07-06**: full suite green — 18 CPU + 9 GPU passed (5m34s);
  parity A–E PASS; sampler smoke 8/8 PASS; env gate PASS; ledger live; positioning note done.
  grad-norm at z_ref 400.5806 reproduces map_marg_pd exactly; cond(A)=1.37e4.
- P1a watch-outs from the parity port (carry into cgl/whiten.py work): (1) vmap-per-column
  convs may need the grouped-conv refactor (gate D is the regression); (2) mask smearing —
  build sqrt_d_inv from the masked err map, trim keep_w by kernel half-width at masks/edges;
  (3) dropped log|C| constant is INVALID for E3 cross-whitener logL comparisons (add back or
  compare posteriors only); (4) 'SAME' conv zero-pads borders — account in e_op; (5) never
  route whiten_fn=None through the conv path (keeps gate B exactly 0.0); (6) parity product
  uses the legacy PSF deliberately; science products must declare psf_pixel_scale so
  guards.assert_psf_sampling engages.
