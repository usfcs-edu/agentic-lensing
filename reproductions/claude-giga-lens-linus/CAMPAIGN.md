# CAMPAIGN LEDGER — claude-giga-lens-linus

Authoritative record. Every gate, number, deviation, and retraction lands here with
provenance (script + artifact + commit). A100-h rows are appended BEFORE results are read.
Plan of record: `plans/PLAN.md` (approved 2026-07-15). Their-format handoffs: `papers/handoff/`.

## Locked decisions

| # | Decision | Provenance |
|---|---|---|
| D1 | Substrate = vendored gigalens-linus @ `80916d24f3e616edecf9fb66b041c716fa111c29`, UNPATCHED, `--no-deps`; re-pin only via PLAN §3 procedure | PLAN §3, 2026-07-15 |
| D2 | venv `/raid/benson/.venvs/cgl2`: py3.13.13, jax 0.6.2, blackjax 1.3, tfp 0.25.0, numpy 2.4.6 (KNOWN DEVIATION vs their 2.1.3 — covered by gate battery), NO tensorflow (verified inert) | PLAN §3; env smoke 2026-07-15 |
| D3 | Cross-stack parity via REFERENCE ARTIFACTS, not same-process dual import: both stacks are package `gigalens` (name collision, per their api-split.md warning). Old stack dumps reference npz in the cgl venv; scene API compares in cgl2. Same jax 0.6.2 both sides preserves 1e-12 comparability. | P0 finding, 2026-07-15 (supersedes the single-process wording in PLAN §3) |
| D4 | Carousel cells INCLUDED (user decision), incl. minimal flow-MAMS arm S7; results to the team first; publication sign-off-gated | user, 2026-07-15 |
| D5 | Budget cap 100 A100-h (commit ~82), shared-QOS single-GPU on cosmo_g | user, 2026-07-15 |
| D6 | Bright lines §8 of PLAN verbatim (no unimodal-efficiency publications; nothing from their unpublished repos external without sign-off; Vela untouched; "validated" reserved for the old stack) | PLAN §8 |
| D7 | **P4 (X1 profile-class fork) RETIRED at zero GPU cost** — pre-registered entry gate X1-G0 FAILED (see gate record). Its 10 A100-h returns to the pool; per PLAN §6 stretch priority order, PSF-marginalization MVP (old stack) is promoted toward core and the evidence-scored source ladder (already in P3's migrate list) absorbs the source-track question. Kill criterion executed as written — not a goalpost move. NOTE the finding's positive content: fine & binned constrain the slope at the SAME radius yet disagree by 0.71 — the bracket driver differs between products AT FIXED RADIUS, which points at the noise/likelihood treatment (whitener) and PSF representation, NOT radial mass structure. First claim on the freed budget: NEXT_DIRECTIONS T1.1 injection-recovery on real drizzle noise (design checkpoint before run), then PSF-marg MVP. | X1-G0, 2026-07-15 |

## A100-hour ledger (append BEFORE reading results)

| Date | Job | Phase | Est. h | Actual h | Cumulative |
|---|---|---|---|---|---|
| 2026-07-15 | 55951082 cgl2-t02-low-s3 (v3b-low SMC p128 seed3, slurm/t02_smc_v3b_low_seed3.slurm) | P1 T0.2 | 1.5 | 0.51 (COMPLETED 00:30:40, 1×A100 shared) | 0.51 |
| 2026-07-15 | 55951083 cgl2-t02-low-s4 (v3b-low SMC p128 seed4, slurm/t02_smc_v3b_low_seed4.slurm) | P1 T0.2 | 1.5 | 0.51 (COMPLETED 00:30:30) | 1.02 |
| 2026-07-15 | 55951084 cgl2-t02-steep-s3 (v3b-steep SMC p96 seed3, slurm/t02_smc_v3b_steep_seed3.slurm) | P1 T0.2 | 2.0 | 0.29 (COMPLETED 00:17:39) | 1.31 |
| 2026-07-15 | 55951085 cgl2-t02-steep-s4 (v3b-steep SMC p96 seed4, slurm/t02_smc_v3b_steep_seed4.slurm) | P1 T0.2 | 2.0 | 0.31 (COMPLETED 00:18:23) | 1.62 |
| 2026-07-15 | 55951086 cgl2-t03-compmask (v3b-low SMC p128 seed2, companion-eroded whitener, slurm/t03_smc_v3b_low_compmask.slurm) | P1 T0.3 | 2.0 | 0.59 (COMPLETED 00:35:40) | 2.21 |
| 2026-07-15 | 55952480 cgl2-t11-i1 (inj1 shift(0,0): svicov prep + SMC p128 seed2, slurm/t11_inj1.slurm) | T1.1 (D7) | 2.0 | 1.89 (COMPLETED 01:53:29, 1×A100 shared, nid008221) | 4.21 (est) |
| 2026-07-15 | 55952481 cgl2-t11-i2 (inj2 shift(+.030,−.014)″: svicov prep + SMC p128 seed2, slurm/t11_inj2.slurm) | T1.1 (D7) | 2.0 | 2.00 (COMPLETED 02:00:02, nid008221) | 6.21 (est) |
| 2026-07-15 | 55952482 cgl2-t11-i3 (inj3 shift(−.022,+.034)″: svicov prep + SMC p128 seed2, slurm/t11_inj3.slurm) | T1.1 (D7) | 2.0 | 1.49 (**FAILED** 01:29:15 — step-2 GPU OOM on hbm40g node; prep COMPLETED, artifacts valid; see stage log 2026-07-15 inj3 diagnosis) | 8.21 (est) |
| 2026-07-15 | 55952483 cgl2-t11-i1d (inj1 DIAGONAL control via delta whitener, slurm/t11_inj1_diagctl.slurm) | T1.1 (D7) | 2.0 | 1.99 (COMPLETED 01:59:32, nid008193) | 10.21 (est) |
| 2026-07-16 | **P1 T0.2/T0.3 actuals harvested**: 2.21 A100-h vs 9.0 est (shared-QOS single-GPU; sacct -X Elapsed × 1 GPU, AllocTRES gres/gpu=1 each) | P1 | — | 2.21 total | 2.21 actual + 8.0 T1.1 est |
| 2026-07-15 | 55958518 cgl2-t11-i3 **T1.1 inj3 resubmit** of 55952482 (SMC-only via SKIP_PREP=1, reuses the COMPLETED production prep npz on $PSCRATCH; fix = `-C gpu&hbm80g` pin + PYTHONUNBUFFERED=1, NO numerics change; slurm/t11_inj3.slurm) | T1.1 (D7) | 2.0 | 0.43 (COMPLETED 00:25:47, nid008193; SKIP_PREP=1) | 2.21 + 1.49 (failed 55952482) actual + 6.0 T1.1 est outstanding + 2.0 resubmit est |
| 2026-07-15 | **T1.1 actuals harvested**: 7.80 A100-h vs 10.0 est (1.89 + 2.00 + 1.49 FAILED + 1.99 + 0.43; sacct -X Elapsed × 1 GPU each) | T1.1 (D7) | — | 7.80 total | **10.01 actual** (2.21 P1 + 7.80 T1.1) of 100 h cap |

## Gate record

| Gate | Statement | Threshold | Status | Artifact |
|---|---|---|---|---|
| F1 | forward image (simulate + stacked M_det) | ≤1e-12 rel | **PASS 5.7e-15** | data/parity_report_scene.json |
| F2 | design columns (old `_design_ret` vs `return_stacked`) | ≤1e-12 rel | **PASS 3.9e-15** | data/parity_report_scene.json |
| F3 | diagonal masked loglik+χ² vs stock ImageLikelihoodTerm | ≤1e-8 | **PASS 7.5e-9** | data/parity_report_scene.json |
| F4 | grad of marg loglik wrt constrained params | ≤1e-8 rel-L2 | **PASS 1.5e-11** | data/parity_report_scene.json |
| F5 | delta-kernel CorrelatedTerm ≡ stock (fwd) [+F5b conv≡multiply] | ≤1e-10 | **PASS 5.8e-11 / 0.0** | data/parity_report_scene.json |
| F6 | Occam −½logdetA vs **fp128 truth** (RESTATED per signed exception 2026-07-15; originally vs numpy slogdet ≤1e-10, which FAILED 1.31e-10 on v3b — measured f64 cross-algorithm noise floor at cond(A)=7e7) | ≤ max(1e-10, 5·eps·cond(A)·1e-2) | **PASS** — v2d 1.19e-11 ≤ 1e-10; v3b 4.48e-10 ≤ 7.76e-10 | data/parity_report_scene.json |
| F7 | flat-z roundtrip + z_param_names audit | exact (info) | **PASS 0.0** (46 names, bijection clean, perturbation identity ok) | data/parity_report_scene.json |
| F8 | harness under NERSC jax-0.10 env | report-only | TEMPLATE READY (slurm/parity_f8_nersc.slurm, not run) | — |
| 03-A | corr term vs dense-Cholesky exact ref (32² toy) | ≤0.1 nat | **PASS 5.5e-12** | data/correlated_term_validation.json |
| 03-B | delta identities (fwd vs stock; conv vs multiply) | ≤1e-10 | **PASS 9.1e-13 / 0.0** | data/correlated_term_validation.json |
| W | whitener bundles re-validated (e_op reproduction + erosion + hashes) | e_op ≤0.02 strict | **PASS** (v2d/v3b/v3 admissible; v2d_relaxed inadmissible-by-design, e_op 0.0312 vs its own e_target 0.05) | data/whitener_manifest.json |
| B0 | MC-SMC correctness (adapters, mix2/funnel/illcond, MCLMC bias screen) | PLAN §5 | PENDING | data/smc_b0_report.json |
| X1-G0 | profile-curvature mechanism entry gate: r_eff ordering must admit the bracket's sign pattern | monotone ordering exists | **FAIL — hypothesis structurally dead** (24/24 robustness variants non-monotone; fine/binned constrain slope at the SAME radius, Δr_eff≈0.008″ < ¼ px, yet Δγ=0.71 ⇒ would need \|dγ_loc/dln r\|≈226 vs O(1) physical) | data/x1_g0_effective_radii.json, research/x1_g0_mechanism_check.md, figs/x1_g0_*.png |
| T1.1 | injection-recovery on real drizzle noise: does the production stationary-whitened correlated likelihood recover γ_truth=1.433 injected on the REAL v3b residual field? | pre-registered (finalized): median(γ_rec−1.43298) < −0.078 confirms LOW bias; \|median bias\| < 0.026 exonerates; between = partial, quantified; control (diag likelihood, same data) predicted 1.29–1.43 | **NO-CONFIRM / NO-EXONERATE — CONFOUNDED (positive-signed result outside both zones)**: γ_rec med 1.5151/1.5719/1.5076, biases +0.0822/+0.1389/+0.0747, own-σ z +2.25/+3.23/+1.82; **median bias +0.0822** (without dup-cluster-flagged inj2: +0.0784 — same zone); **control FAILS HIGH 1.5677 ∉ [1.29,1.43] AND SICK** (total resample collapse: 1/128 unique particles, γ_σ=4e-16; srcS.Ie railed at 10.09≈58× truth) ⇒ per pre-registered honesty clause the INJECTION CONSTRUCTION is implicated (bright-object scene-subtraction residue; recovered source ×1.8–3 bigger, ×2.4 brighter than truth in all 3 corr runs). Whitener-isolating differential corr−diag on same data: −0.0526 (sign consistent w/ mechanism, ≈16% of the 0.33 gap; no error bar — diag leg degenerate). T0.4-1's stationarity rejection UNREFUTED (in-class scene ⇒ mechanism's misfit lever arm absent by construction). n=3, no coverage claims | data/results-perlmutter/t11_*, figs/t11_recovery_overlay.png, data/t11_gate_eval.json, research/t11_injection_recovery.md |
| Fermat teaser | noise-model Δφ sensitivity (illustrative; NOT a TD lens; synthetic pairs; corr posterior is the known over-correcting product) | report-only | median \|frac shift\| **88%** anchor→corr (10.7σ); same-product diag→corr arm **61%** (17σ) — vs the ~1% TDCOSMO-relevant scale | data/fermat_dt_teaser.json, research/fermat_dt_teaser.md |
| T0.2 | seed-repeat certification of the P1c money numbers: σ_seed(γ) per basin over seeds {2 (production), 3, 4}; σ_seed(ΔlogZ) | σ_seed(γ)≤0.008 both basins; σ_seed(ΔlogZ)<5 nats; KILL σ_seed(γ)>0.024 | **PASS (both gates; kill not tripped)** — γ_med low {1.1032, 1.0967, 1.1005} → σ_seed=0.00325; steep {2.6393, 2.6522, 2.6485} → σ_seed=0.00664; logZ low {−4771.08, −4769.12, −4771.37} → σ_seed=1.22; steep {−4799.96, −4801.39, −4802.56} → σ_seed=1.30 → σ_seed(ΔlogZ)=1.79 nats; ΔlogZ(steep−low) per matched seed −28.88/−32.27/−31.19 — LOW-basin preference SEED-STABLE (all seeds, ≥16σ_seed). n=3 σ estimates carry ±46% χ-dist sampling error (quoted with every use). γ_binned(corr,low)=1.1032 CERTIFIED at stated significance: σ_tot=√(σ_stat²+σ_seed²)=0.0086 ≈ 1.08×σ_stat | data/t02_t03_gate_eval.json, figs/t02_seed_overlay.png, data/results-perlmutter/ |
| T0.3 | companion-mask discriminator: does the LL2/LL3 companion misfit transmit into global γ via the whitened likelihood? (whiten-then-drop, keep_w 9273→8247, production seed 2) | UPWARD shift ≥0.024 (3σ_stat) ⇒ mechanism REAL; static within 0.024 ⇒ companion EXONERATED | **COMPANION EXONERATED** — γ_med 1.1011 vs production 1.1032: shift −0.0021 (slightly DOWN, 0.26σ_stat, 0.65σ_seed — statistically static). σ_stat widened 0.0080→0.0085 (+6%, consistent with −11.1% whitened dof). logZ −4339.16 vs −4771.08 NOT comparable (different data: 1026 fewer whitened dof). Convergence indistinguishable from production (λ-steps 28=28, w_ess 127.4 vs 118.1, n_uniq 84 vs 77). The 1.103 over-correction is NOT companion-driven — consistent with the P1 synthesis (noise-model-CLASS misspecification), which stays the prime suspect for T1.1 | data/t02_t03_gate_eval.json, figs/t03_compmask_overlay.png |
| σ_seed FINALIZATION | downstream provisional thresholds inherit P1's measured σ_seed (README frozen-gates note: a ledgered finalization, not a goalpost move) | — | **FINALIZED**: (1) B5-G2 basin-ΔlogZ agreement = 3·√(σ_boot² + 1.79²) nats (floor 5.36 at σ_boot=0); (2) T1.1 σ floor = σ_tot = √(0.008² + 0.00325²) = 0.0086 → exonerate \|median(γ_rec−1.433)\| < 0.026, confirm < −0.078 (supersedes the provisional 0.024/−0.072 that used σ_stat alone; bands move <8%, interpretation zones unchanged in kind); (3) X1-G1's ~15-nat placeholder RETIRED with P4 (D7) — never finalized | data/t02_t03_gate_eval.json |
| T0.4-1 | per-block kernel homogeneity (stationarity of the noise-model class) | 2σ blockwise + calibrated p | **REJECTED** — money product v3b max\|z\|=3.67, calibrated p=0.010; arc-excluded v3 p=0.010; replicated spatial pattern; observed cross-block ρ(0,1) spread 16.4× the drizzle-registration envelope. **The stationary kernel class behind γ=1.103 is provably violated by the field.** Verifier CLEAN (all z/p recomputed exactly; power check confirms informative nulls). | data/t04_stationarity*.json, figs/t04_stationarity_*.png, research/t04_free_checks.md |
| T0.4-2 | λ-arm: does spectrum-flooring cure the fine-low gaming? | ordering table w/ per-λ exact log\|C\| | **NO — "information-discard-at-spectral-zeros" FALSIFIED**; data reject flooring by 3.3k–33k nats; pathology localized to down-weighting of high-S large-scale modes (the w_b≈0.27 background component — exactly the nonstationary component of T0.4-1). Production s_floor=0.05 confirmed (plan's "0.1" corrected); production taps reproduced to 1e-9. Verifier CLEAN (per-λ Szegő anchors verified — no shared constant). | data/t04_lambda_arm.json, figs/t04_lambda_arm.png |
| T0.4-3 | real-space head-to-head on the SAME v3b pixels | report-only ordering | Binned data in real space prefers **γ≈1.29** (diag-low, χ²_pp 1.58) over BOTH the anchor 1.433 (7.44) and corr-low 1.103 (8.32); the production whitened metric INVERTS this (+501 nats for corr-low). Corr-low's residual = smooth lens-center misfit (same currency as fine-low gaming). Anchor's full-field number carries a cross-product resolution handicap (honest caveat). | data/t04_realspace_headtohead.json, figs/t04_headtohead_residuals.png |

**P1 synthesis (2026-07-15): one mechanism spans T0.4-1/2/3 + X1-G0** — a NONSTATIONARY
correlated-background component priced as stationary lets the whitened metric discount
large-scale real-space misfit, biasing γ low. The 1.103 over-correction is now most plausibly
noise-model-CLASS misspecification, not source/PSF. Confirmatory experiment = T1.1
injection-recovery on the real noise field (D7 queue, mechanism-backed directional prediction).
Design implication for P3: CorrelatedImageData keeps a pluggable whitener seam for a
locally-stationary (per-region) class. Ops lesson: parallel build_whitener needs
OPENBLAS_NUM_THREADS=4 on aarch64 (default threading livelocks ~100×).

**Gate exception F6 — ENACTED (Benson sign-off, 2026-07-15):** F6 restated as
"|jax-chol logdetA − fp128-truth logdetA| ≤ max(1e-10, 5·eps·cond(A)·1e-2)" —
compares to actual ground truth instead of another noisy f64 algorithm, with the
tolerance scaling as floating-point error analysis requires (Benson: "the better
option scientifically"). Enacted in 01_parity_scene.py + README; re-run on L4:
**F6 PASS** — v2d err 1.19e-11 ≤ tol 1e-10 (cond 7.9e6, floor governs); v3b err
4.48e-10 ≤ tol 7.76e-10 (cond 6.99e7). **ALL HARD GATES NOW PASS (67 s re-run);
the P0 exit F6 arm is closed** (funnel10 remains the one recorded B0 FAIL,
impact-assessed above). Original recommendation + fp128 evidence preserved in
data/parity_report_scene.json → products.*.F6.
Verifier wording correction adopted: the defensible claim is "the gate REFERENCE
(numpy slogdet) itself errs vs fp128 truth by ~7e-10 at cond(A)=7e7, so 1e-10
cross-algorithm agreement is not meaningful there" — not "unreachable by any f64
implementation."

**P0 exit decision (2026-07-15, producer verdict — UNCERTIFIED, flagged for
Benson):** PLAN P0 exit criterion "F1–F7 + B0 pass" is NOT met as written: F6
FAIL (noise floor, above) and B0 funnel10 FAIL (logZ deficit −0.4..−1.0, all
kernels incl. the HMC baseline; N-independent; REPRODUCED in the old validated
stack's on-disk bj_smc funnel results at N=1000 → pre-existing adaptive-tempered-
SMC limitation, NOT a port defect). Campaign proceeds to P1/P2 with both FAILs
recorded (no threshold moved): impact assessment — the lens-relevant B0 content
(mix2 multimodality evidence, illcond46 accuracy, MAMS-vs-MCLMC bias screen,
adapter builds 11/11) all PASS; funnel-geometry evidence deficits are a known
caveat carried into B2 (whose pre-registered arm-mass gate tests ridge geometry
directly on their DSPL target). MCLMC formally DEMOTED to cost-frontier-only
(bias screen 30σ) per the pre-registered rule; MAMS is the evidence kernel.
Verifier-corrected provenance: the B0 run executed on an A16 (not L4 as the
build reported — CUDA device-order defect, since fixed); numerics unaffected
(f64, hardware-independent gates, reproduced on rerun). Known issue carried:
σ_boot understates evidence error on ill-conditioned targets (mams illcond logZ
−118 vs −146 across N — P1's σ_seed measurement is load-bearing for every
downstream ΔlogZ gate, as planned).

## Perlmutter ops

- Remote staging: `/global/cfs/cdirs/deepsrch/gdbenson/cgl2-linus/{code,data,results,slurm-logs}`
  (created 2026-07-15; user-designated disk). Scratch: `/pscratch/sd/g/gdbenson` for hot job I/O,
  results archived back to CFS (their results-storage pattern). Remote is a NON-GIT rsync copy →
  md5-audit every campaign `.py` before production (the stale-remote lesson).
- sshproxy refreshed 2026-07-15 (user). Jobs charge `cosmo_g` (D5), single-GPU shared QOS.

## Stage log (newest first)

### 2026-07-15 — T1.1 PRE-REGISTERED READOUT (jobs 55952480/81/83 + 55958518; 7.80 A100-h actual): NO-CONFIRM / NO-EXONERATE — CONFOUNDED BY SCENE RESIDUE

**Harvest ops:** all 24 t11 result files (4 SMC sets + 4 prep sets, npz+json+run.log) +
5 slurm logs pulled from CFS to `data/results-perlmutter/`; sacct actuals in the ledger
(1.89/2.00/1.49-FAILED/1.99/0.43 h, single A100 each; T1.1 total 7.80 vs 10.0 est;
campaign 10.01 A100-h actual of the 100 h cap). Provenance CLEAN: slurm-log md5 echoes
match the local build report exactly (datafiles 98b2b825/175166b8/81271c7f, delta
whitener fa167fe2, e2.py 782a268a); the inj3 resubmit (55958518) confirmed SKIP_PREP=1
+ same datafile md5 + warm start `q from t11_inj3_canary_svicov.npz` = the prep the
failed 55952482 completed (CFS 17:27). Analysis = `08_harvest_t11.py` (OLD cgl venv,
bijector-only, exact P1-harvest conventions; json weighted quantiles authoritative,
eqw cross-check ≤0.005). **Plot inspected BEFORE gate math**
(figs/t11_recovery_overlay.png); plot and numbers agree. Full numbers:
data/t11_gate_eval.json; full analysis: research/t11_injection_recovery.md.

**Sanity:** all four runs reached λ=1 (λ-steps 27/34/20/37 ≪ 400), n_floored_q=0,
basin purity clean (frac_γ>1.9 = 0 everywhere), w_ess 107.9–128/128. Saved-particle
diversity vs the certified production family (14–37 unique rows/128): inj1 52, inj3 81
(healthier than production), inj2 18 (in-family; flag: dominant duplicate cluster at
the TOP quantile, γ_med==γ_q84 — same class as T0.2 seed3-low, reported not sick).
**Diag control SICK:** total resample collapse — 1/128 unique particles (a point mass
copied 128×), γ_σ=4.4e-16, prep Rhat_max 307, srcS.Ie railed at 10.09 (≈58× truth) +
LL0/LL2/LL3.Ie + LL2.center_x railed. Mechanism: the diagonal likelihood is ~3× sharper
in logp scale and the frozen production SMC moves (step 0.1, whitened-geometry metric)
had ~zero late-tempering acceptance → systematic resampling degenerated. Gates
evaluated with and without the sick run (verdict unchanged in kind).

**GATES (finalized thresholds, NOT moved — confirm < −0.078, exonerate |·| < 0.026,
σ_inj = own posterior σ, n=3, no coverage claims):**
γ_rec = 1.5151 [1.4877,1.5543] / 1.5719 [1.4796,1.5719] / 1.5076 [1.4685,1.5482];
biases +0.0822/+0.1389/+0.0747; z = +2.25/+3.23/+1.82; logZ −4654.84/−4644.91/−4634.52
(different data files — reported, not compared). **median bias (n=3) = +0.0822;
without inj2 = +0.0784** — OUTSIDE both pre-registered zones, POSITIVE-signed
(opposite the predicted direction; identical verdict under the superseded provisional
±0.024/−0.072 bands). **Control gate FAIL:** γ_rec(diag) = 1.5677 ∉ [1.29,1.43],
biased HIGH by +0.135 (farther from truth than the corr runs), logZ −12799.37 (diag
dof, not comparable).

**INTERPRETATION (per the pre-registered honesty clause, which governs):** the control
failing HIGH alongside all three injections implicates the INJECTION CONSTRUCTION —
the real residual field's bright-object scene-subtraction residue (measured before
submission: G1 decomposition χ²_pp bright 2.71 / center 5.46) — not the whitener. The
nuisance readout shows the absorption signature in all three corr runs: recovered
source ×1.8–3 bigger (srcS.R_sersic +3.4σ/+2.6σ/+3.7σ) and ×2.4 brighter than the
injected truth, LL-block distortions in sympathy; the fits eat the residue and γ
steepens (+0.07..+0.14 common-mode across BOTH likelihood classes ⇒ data-driven).
The only whitener-isolating number — same-data differential γ(corr)−γ(diag) on inj1 =
**−0.0526** — has the mechanism's predicted sign and would explain ≈16% of the 0.330
real-data gap, but carries no defensible error bar (degenerate diag leg): indicative
only; the CONFIRM-level ≥24% is NOT supported. Crucial scope note: the injected scene
is exactly in-class (truth source = fitted ridge shapelets), so the T0.4 mechanism —
stationary whitening discounting large-scale real-space MISFIT — had little lever arm
BY CONSTRUCTION; even a clean EXONERATE could not have refuted T0.4-1's direct
stationarity rejection (p=0.010), which STANDS. **Verdict: T1.1
INCONCLUSIVE-BY-CONFOUND; the 1.103 over-correction diagnosis (noise-model-CLASS
misspecification) continues to rest on T0.4's direct evidence — neither confirmed at
injection level nor exonerated.**

**Implications:** (1) P3 CorrelatedImageData keeps the pluggable locally-stationary
whitener seam at priority (T0.4-1 untouched); new requirement — a residue-free
injection path (kernel-sampled noise-only / sky-set bootstrap, or deeper multi-start
scene subtraction) before any whitener-bias number is quotable. (2) P2 benchmark
framing STRENGTHENED: the production two-stage+SMC recipe does not transfer to the
sharper diagonal likelihood (total particle collapse at p128) — per-target tuning is
part of the thesis; a future diag arm needs its own step-size/metric. (3) Engagement
memo line: injection-recovery on the real residual field is residue-confounded
(+0.08..+0.14 common-mode); corr-vs-diag differential −0.05 (mechanism-signed, small
vs 0.33 at in-class scene specification); definitive test = residue-free injections +
locally-stationary arm. No headline change from T1.1.

**Housekeeping:** 55952480/81/83 + 55958518 deregistered from the watchdog (all four
T1.1 registrations — 480/481 were also still registered; all harvested), watchdog now
empty, loop alive (PID 118755); data/WATCHDOG_ALERT deleted (the 4 COMPLETED_NO_ARTIFACT
alerts were the designed harvest reminders — artifacts confirmed on CFS + pulled).
NOT committed (house rule: user commits).

### 2026-07-15 — T1.1 inj3 FAILURE DIAGNOSIS + RESUBMIT (55952482 FAILED → 55958518; ledger row appended BEFORE any readout)

**Symptom:** 55952482 (t11-i3) FAILED ExitCode 1:0 at Elapsed 01:29:15 on nid003112; the
slurm log showed only the [t11-i3] header echoes. This is expected under failure BY
DESIGN: both python steps redirect stdout/stderr to `$PSCRATCH/.../t11_inj3_*_run.log`,
and `set -e` aborts before the end-of-job summary grep — the slurm .out never gets
python output. The initial "no partial artifacts" triage read was STALE: on inspection,
step 1 (prep) had COMPLETED cleanly (t11_inj3_canary_svicov.{npz,json,run.log} on
$PSCRATCH 17:27 AND cp'd to CFS results/), and t11_inj3_smc_run.log (2975 B vs 435 B
healthy siblings) held the full traceback.

**ROOT CAUSE (from the step-2 run.log, not a hypothesis):** GPU OOM in the frozen
production SMC (p128) — `XlaRuntimeError: RESOURCE_EXHAUSTED ... 49,861,018,216 bytes`
in `common.run_adaptive_tempered_smc` step; XLA rematerialization reported it could not
reduce the working set below **39.05 GiB**. nid003112 is an **hbm40g** (40 GB A100) node:
0.95×40 GB cannot hold a ≥39 GiB floor. The successful siblings inj1/inj2 (and ALL prior
successful SMC runs this campaign: T0.2 low/steep + T0.3 on nid008221/nid008193) ran on
**hbm80g** nodes by scheduler luck — every script requested only `-C gpu`, while the OLD
campaign's own e2_smc slurm comment states the production memory budget outright:
"XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 (give JAX 76 GB of the 80 GB card)". Latent defect,
not an inj3 pathology.

**Ruled out:** (1) datafile corruption — md5 81271c7f… identical across local
data/t11_inj3_v3b.npz, the build report, and the CFS-staged copy; (2) an inj3-specific
prep NaN — prep COMPLETED with healthy numbers (MAP γ_map=1.4266 in-basin; SVI full_rank;
warm-start gamma(loc)=1.4995, in family with inj2's 1.5183); (3) host OOM — MaxRSS 3.6 GB
of 57 GB; (4) lost-buffered-stdout — the traceback flushed fine on exception.

**Fix applied (minimal, no numerics change):** slurm/t11_inj3.slurm gains
`#SBATCH -C gpu&hbm80g` (pins the GPU class the production config was budgeted for and
every successful SMC run actually used) + `export PYTHONUNBUFFERED=1` (observability).
Truth params / whitener / mask / SMC config UNTOUCHED. **Resubmitted with SKIP_PREP=1**
— the script's documented resilience path — reusing the completed, unmodified-production
prep npz (t11_inj3_canary_svicov.npz, seed 2) from $PSCRATCH; SMC-only rerun.

**Ops:** fixed script staged to CFS code/, md5 1e1e32aa… verified identical both sides;
submitted as **55958518** (cosmo_g shared, 1 GPU, SKIP_PREP=1 exported); watchdog:
55952482 deregistered, 55958518 registered (max_pending 24 h / max_run 6 h /
expect_artifact CFS t11_inj3_smc.npz / on_stall resubmit) — loop alive (PID 118755).
inj1/inj2/control results NOT touched; their COMPLETED_NO_ARTIFACT harvest-reminder
alerts left in place. **Carried flag:** the `-C gpu` under-pin affects all campaign
slurm templates — pin `gpu&hbm80g` on any future ≥40 GB-working-set (SMC/HMC prod)
submission. Transparency: during triage the inj2 SMC run.log tail was displayed
(a γ line was on screen); it was not interpreted and no gate math was done — the T1.1
readout remains a separate pre-registered phase. **NO results read this session.**

### 2026-07-15 — P1 T0.2/T0.3 HARVEST (jobs 55951082–86, all COMPLETED; 2.21 A100-h actual vs 9.0 est)

**Harvest ops:** all 15 result files (npz+json+run logs) + 5 slurm logs pulled from CFS
`/global/cfs/cdirs/deepsrch/gdbenson/cgl2-linus/{results,slurm-logs}/` to
`data/results-perlmutter/`. sacct -X actuals in the ledger (0.51/0.51/0.29/0.31/0.59 h,
single A100 each). Analysis = `07_harvest_t02_t03.py` under the OLD cgl venv (CPU,
bijector only — no likelihood evals): the per-run JSON summaries are the authoritative
extraction (the P1c convention — weighted quantiles via cgl.e2._weighted_quantile inside
run_correlated_smc); saved equal-weight particles transformed via
build_target('v3b').model.to_physical_mass for the plots, cross-checked against the
weighted medians (agree ≤0.003; the 0.0027 seed3-low delta is median discreteness from
its duplicate cluster, below). Plots BEFORE metrics: figs/t02_seed_overlay.png,
figs/t03_compmask_overlay.png. Full numbers: data/t02_t03_gate_eval.json.

**T0.2 PASS (gate record):** σ_seed(γ) = 0.0033 (low) / 0.0066 (steep), both ≤ 0.008;
σ_seed(ΔlogZ) = 1.79 nats < 5; kill (>0.024) not tripped; ΔlogZ(steep−low) =
−28.88/−32.27/−31.19 for seeds 2/3/4 — the ~29-nat low-basin preference is seed-stable
in sign AND magnitude (spread 3.4 nats). The P1c money number is now quotable as
γ_binned(corr,low) = 1.1032 ± 0.0080 (stat) ± 0.0033 (seed), σ_tot = 0.0086; the 17σ
anchor tension stands at its stated significance. n=3 caveat: σ estimates carry ±46%
χ-distribution sampling error — thresholds derived from them say so.

**T0.3 verdict (gate record): COMPANION EXONERATED.** With the companion disk
(r<1.2″ @ (−2.34,−2.86)″) whiten-then-dropped, γ_med = 1.1011 vs production 1.1032:
shift −0.0021, i.e. DOWN 0.26σ_stat, not the pre-registered ≥+0.024 upward move. The
localized companion misfit does NOT transmit into the global slope; the over-correction
driver stays with the T0.4/P1-synthesis nonstationarity mechanism now under direct test
in T1.1. logZ is not comparable across the mask change (9273→8247 whitened dof) and is
reported, not interpreted.

**Convergence sanity (all 7 runs compared, incl. the 2 production baselines):** every
run reached λ=1 by construction (run_adaptive_tempered_smc RAISES otherwise; artifact
written ⇒ terminated at λ=1); λ-step counts homogeneous per config (low 28/28/28 +
compmask 28; steep 21/21/22, all ≪ cap 400); n_floored_q=0 everywhere; basin purity
clean (frac_γ>1.9 = 0.000 low / 1.000 steep). Final-weights ESS: new runs 127.4–127.7/128
(low) and 90.3–96.0/96 (steep) — the NEW steep runs are markedly healthier than the
production seed-2 steep (w_ess 36.0/96), so σ_seed is not inflated by a sick repeat.
One flag, not sick: seed3-low's final population carries a dominant duplicate cluster
(γ_q16 == γ_med exactly ⇒ ~⅓ of weight on one γ value; resampling-duplicate survival,
w_ess still 127.7) — visible as the tall narrow peak in the overlay; its median enters
σ_seed as-is (reported exactly as computed). Provenance note: the two T0.2-low +
compmask jobs ran on nid008221, steep pair on nid008193; T0.2 jobs executed pre-patch
e2.py (md5 9a6d4488…), the compmask job (started after the T1.1 truing) executed the
T1.1-patched e2.py (md5 782a268a…) whose default data_file='' path is the documented
bit-for-bit production deviation (logp identity verified in the T1.1 record; its json
config confirms data_file='').

**Downstream thresholds FINALIZED from measured σ_seed (gate-record row):** B5-G2 →
3·√(σ_boot² + 1.79²) nats; T1.1 σ floor → σ_tot = 0.0086, exonerate |median bias| <
0.026, confirm < −0.078; X1-G1's ~15-nat placeholder retired with P4 (D7). Per the
README frozen-gates note these finalizations are themselves this ledger row.

**Housekeeping:** jobs 55951082–86 deregistered from the watchdog; the 5
COMPLETED_NO_ARTIFACT alerts were exactly the designed harvest reminders (artifacts
were on CFS all along, checked on phoenix's filesystem) — all accounted for,
data/WATCHDOG_ALERT deleted, heartbeat touched; loop alive (PID 118755). The 4 T1.1
registrations (55952480–83) left in place; T1.1 results NOT read (separate
pre-registered readout).

### 2026-07-15 — T1.1 DESIGN CHECKPOINT + submission (injection-recovery on real drizzle noise; est 8.0 A100-h committed in ledger, D7 freed pool)

**T1.1 DESIGN CHECKPOINT (pre-registered, appended BEFORE submission):**
- **Hypothesis (mechanism-backed, P1 synthesis / T0.4):** the production stationary-whitened
  correlated likelihood is biased LOW in γ on the real v3b noise field — a NONSTATIONARY
  correlated-background component priced as stationary (T0.4-1 rejection, p=0.010) lets the
  whitened metric discount large-scale real-space misfit (T0.4-2/3), dragging γ down at fixed
  radial information (X1-G0).
- **Design (n=3 injections + 1 control):** inj_i = (real v3b cutout img − model_map_v3b_cold,
  i.e. the production converged-model residual field = the field the whitener was fit on)
  + (synthetic full scene at the ANCHOR truth). Truth = hmc_v13_v2d per-dim-median 74-dim
  paper point → 46-dim marg z via the e2/fermat/t04-validated transform (ie_scale = cf_v3b);
  **γ_truth = 1.43298**; the 28 shapelet amps FROZEN at a_truth = the production correlated
  ridge solve of that point on the REAL v3b data ("the fitted shapelet source"); render =
  M_det(z_i) + ret(z_i)·a_truth via the parity-certified reference builder (cross-builder
  identity 4.4e-16 vs the grouped production model). Between-injection variation ONLY a
  rigid sub-pixel source-center shift (srcS+srcShp together): inj1 (0,0)″, inj2
  (+0.030,−0.014)″, inj3 (−0.022,+0.034)″ (v3b pixel = 0.08″); truth γ + mass sector
  identical (verified to 1e-9). err_map/keep_mask/psf/meta byte-identical to cutout_v3b.
  Truth provenance per injection: data/t11_inj{i}_truth.json (z46, named physical params,
  a_truth, shifts, input md5s, gate numbers).
- **Fit (unmodified production v3b-low correlated config, per injection, chained in ONE
  slurm job):** step 1 reproduces the production warm-start prep = the
  e2_v3b_low_canary_svicov.npz generation step, EXACT preserved config (scratchpad
  e2_v3b_low_canary_svicov.json: --mode prod --basins low --metric svi_cov --svi-steps 7000
  --svi-particles 16 --chains 24 --num-leapfrog 16 --step-size 0.1 --stage1-burn 200
  --stage1-keep 200 --burn 50 --keep 100 --map-rounds 4 --map-iters 200 --seed 2) ON the
  injection; step 2 = the frozen production correlated SMC (p128, cov-inflate 3.0,
  mcmc-steps 4, integration-steps 8, step 0.1, target-ess 0.7, max-lambda 400, seed 2 = THE
  production seed) warm-started from step 1's npz. **Whitener = PRODUCTION whitener_v3b,
  unchanged — testing it IS the experiment.** Job 4 (control, D7-optional exercised because
  turnkey): injection 1 under the DELTA whitener bundle (h=[[1]], M=0, keep_w=keep_mask;
  makes the production code path compute exactly the diagonal masked marg likelihood —
  identity gated at 0.0 nats on 3 test points, build report G4).
- **Prediction (pre-registered):** median(γ_rec − 1.433) < −0.072 (≥3× the 0.024 = 3·σ_stat
  floor, σ_stat = 0.008; mechanism scale suggests 0.1–0.3). Control prediction: γ_rec(diag)
  ≈ 1.29–1.43, NOT dragged to ~1.1 (also bounds any bias contributed by the residual field's
  scene-subtraction residue: same data, likelihood is the only change).
- **Falsifier:** |median(γ_rec − 1.433)| < 0.024 ⇒ the stationary-whitener approximation is
  EXONERATED on real noise ⇒ source/PSF reinstated as prime suspects. Intermediate
  (0.024–0.072): partial contribution, quantified. Per-injection z-scores
  (γ_rec,i − 1.433)/σ_i reported; **NO coverage claims at n=3.**
- **Build sanity gates (all measured BEFORE submission; data/t11_injection_build_report.json,
  builder 06_build_injections.py):** G2 t04-check-3 reproduction PASS (anchor diag-solve
  χ²_pp 7.4364 vs 7.436; corr logL_data −5442.1 vs −5442.1); G3 transform roundtrip PASS
  (γ 1.43298 = stored chain median, drift 0.0); cross-builder render identity PASS (4.4e-16);
  G4 delta-whitener identity PASS (0.0 nats); G5 production-code-path probe on each injection
  PASS (logp finite at truth and at the production low warm start; ridge amps on injected
  data within 35–36% rel-L2 of a_truth — noise-driven, recorded). **G1 as briefed
  (full-keep χ²_pp vs own truth ∈ [0.9, 1.25]): FAIL at 1.598 — root-caused and RESTATED
  here BEFORE submission** (F6-restatement pattern): the briefed range is v2d-native
  calibration lore (gated MAP 1.234, honest 0.92); on v3b NO model achieves it — the
  ledgered field SOTA is 1.578 (t04 check-3 diag-low home render). Decomposition: sky
  (production kernel-fit set) 0.952, faint 0.973, bright-object 2.71, center r<1.2″ 5.46 —
  103% of the excess over 1.0 is bright-object scene-subtraction residue that every REAL
  fit on this product also faces (it makes the injection MORE faithful to the real
  inference, and the diagonal control bounds its γ effect). Restated arms, both hard:
  **G1a sky-set χ²_pp ∈ [0.9,1.25]: PASS 0.952** (the intended noise-consistency test, on
  the pixels the whitener was actually fit from); **G1b full-keep = ledgered field level
  1.578 ± 0.15: PASS 1.598** (assembly-bug guard). All three injections identical in these
  numbers by construction (same residual field).
- **Documented deviation (the only code change):** OLD 10_run_e2.py + cgl/e2.py gained a
  keyword-only `--data-file`/`cutout_file` override — an exact clone of the existing
  `--whitener`/`whitener_file` pattern (the brief's assumed data-file flag did NOT exist;
  verified). Default reproduces production bit-for-bit: logp at the v3b low start
  −5200.305720610074 identical pre/post patch (scratchpad t11/baseline_{pre,post}patch.json).
  Same precedent as the accepted P3 keyword-only prior overrides in likelihood.py.
- **Budget:** 4 × est 2.0 A100-h = 8.0 from the D7-freed pool (rows above, marked
  "T1.1 (D7)"). Walltime 02:30/job; two-step structure gives a SKIP_PREP=1 resubmit path if
  step 2 is walltime-killed (canary npz persists on $PSCRATCH).

**Perlmutter ops record (this session, T1.1 submission):**
- md5-audit of the remote exec tree (`~gdbenson/claude-giga-lens/repo/.../claude-giga-lens/`,
  the same 71-file list as the T0.2 audit) BEFORE truing: 69/71 identical; the ONLY diffs were
  the two files this session intentionally patched locally (10_run_e2.py, cgl/e2.py), whose
  remote md5s matched the pre-patch local GIT state exactly (e8ce6b16…, 9a6d4488…) — i.e. the
  P1 truing held; remote was CLEAN. Patched files rsync-trued; **re-audit CLEAN 71/71** vs the
  local working tree (scratchpad t11/{local,remote_md5_post}.txt).
- Staged to `/global/cfs/cdirs/deepsrch/gdbenson/cgl2-linus/code/` with md5 verification
  **12/12 identical**: t11_inj{1,2,3}_v3b.npz (569 KB each), t11_inj{1,2,3}_truth.json,
  whitener_v3b_delta_diag.npz, t11_injection_build_report.json, and the four t11 slurm files.
  Absolute-path flags used for --data-file/--whitener (t03 precedent). OLD campaign's staged
  data tree untouched (only the two trued .py files changed there).
- **SUBMITTED 2026-07-15 14:15 PT (all 4, cosmo_g shared QOS, -t 02:30, ledger rows above):**
  55952480 (t11-i1), 55952481 (t11-i2), 55952482 (t11-i3), 55952483 (t11-i1d diag control).
  All 4 registered with the watchdog (max_pending 24 h / max_run 6 h / expect_artifact = CFS
  t11_*_smc.npz / on_stall resubmit:<CFS slurm path>); loop verified alive (PID 118755);
  manual pass clean for all 4 (PENDING ok). NOTE: the pass shows the EXPECTED fail-loud
  COMPLETED_NO_ARTIFACT alert for T0.2 job 55951082 (CFS artifact checked on phoenix's
  filesystem — the designed harvest reminder); left in place for the harvest session.
- Results path policy: hot I/O to `$PSCRATCH/cgl2-linus/results`, `cp` to CFS results/ at the
  end of each step (prep npz copied BEFORE the SMC step so a walltime kill of step 2 preserves
  the warm start; SKIP_PREP=1 resubmit path in each slurm file).
- **NO results read this session** (submission+setup only; harvest is a later phase).

### 2026-07-15 — P1 T0.2/T0.3 design checkpoints + Perlmutter submission ops (task #15; est 9.0 A100-h committed in ledger)

**T0.2 DESIGN CHECKPOINT (pre-registered, appended BEFORE submission):**
- **Hypothesis**: the P1c single-run SMC money numbers (γ_binned(corr,low)=1.1032±0.0080 stat;
  logZ_low=−4771.08, logZ_steep=−4799.96, ΔlogZ=−28.9 nats) carry unquantified seed-to-seed
  scatter — particle init, tempering path, and equal-weight resampling are all keyed to the
  single production `--seed 2`.
- **Predicted direction/magnitude**: σ_seed(γ) ≤ 0.008 (i.e. ≤1× σ_stat) and σ_seed(ΔlogZ)
  < 5 nats. Derived thresholds (frozen, PLAN §6 P1): 5 nats keeps the 28.9-nat basin
  preference > 5σ_seed, and σ_seed ≤ 0.008 keeps the 17σ anchor tension quotable at its
  stated significance (σ_tot ≤ √2·σ_stat).
- **Falsifier / kill (as pre-registered)**: σ_seed(γ) > 0.024 (3σ_stat) UNCERTIFIES 1.103 →
  X1-class real-lens claims re-scope to mocks. Intermediate zone (0.008, 0.024]: γ quoted
  with σ_tot=√(σ_stat²+σ_seed²); downstream provisional thresholds (e.g. ~15-nat ΔlogZ
  decisiveness) finalized from the measured σ_seed — a ledgered finalization, not a move.
- **Design**: 2 NEW seeds (3, 4) × {v3b-low @128 particles, v3b-steep @96} = 4 single-GPU
  shared-QOS jobs, EXACT production code path (frozen invocations cloned from the old
  campaign's `e2_smc_canary_v3b_{low,steep}.slurm`; deltas = seed + output paths ONLY; no
  checkpointing retrofits). σ_seed per basin computed over n=3 (production seed 2 + new 3, 4);
  n=3 σ estimates carry their own (χ-distribution) sampling error — will be quoted with it.

**T0.3 DESIGN CHECKPOINT (pre-registered, appended BEFORE submission):**
- **Hypothesis**: the localized companion-galaxy misfit (the LL2/LL3 Sérsic pair at
  (−2.34, −2.86)″, localized χ²~9–15 region in the v3b residuals) transmits into the GLOBAL
  slope via the whitened likelihood's spatial-frequency reweighting, contributing to the
  1.103 over-correction below the 1.433 native anchor.
- **Predicted direction/magnitude**: with the companion region excluded, γ_binned(corr,low)
  moves UPWARD (toward the anchor) by ≥ 3σ_stat = 0.024 if the mechanism is real.
- **Falsifier**: γ static within 3σ_stat ⇒ companion misfit EXONERATED as an over-correction
  driver (bracket question stays with the source/PSF track).
- **Implementation (whiten-then-drop)**: variant whitener bundle
  `data/whitener_v3b_companion_eroded.npz` built+validated locally in the cgl venv by
  `05_build_companion_whitener.py`: kernel h / e_op / M / rho_kernel / logdet_per_pix
  byte-identical to production `whitener_v3b.npz` (asserted; e_op is a kernel property —
  unchanged by construction); ONLY keep_w shrinks: erode_keep(keep_mask & ~disk, M) with
  disk = r<1.2″ @ (−2.34,−2.86)″ (X1-G0 geometry, research/x1_g0_mechanism_check.md; center
  = foundry-i nearby_galaxy_loc.npz LL2/LL3 prior center; grid convention validated against
  cutout brightness), asserted == keep_w ∧ ¬dilate(disk,(2M+1)²) (duality) and ⊆ keep_w.
  keep_w: 9273 → 8247 (−1026, 11.1% of whitened dof). Run = frozen v3b-low production config
  at the PRODUCTION seed 2; only deltas = `--whitener <abs CFS path>` (build_target's
  `np.load(DATA / wname)` honors absolute paths — old campaign staged tree untouched) +
  output names. 1 job.
- **Pre-registered caveat**: dropping 11.1% of whitened dof widens σ_stat somewhat; the
  ≥0.024 discriminator is on the shift of the γ median vs the seed-2 production 1.1032,
  and will be sanity-checked against the T0.2 measured σ_seed before interpretation.

**Perlmutter ops record (this session, before submission):**
- md5-audit (10_run_e2.py + full cgl tree + full vendored gigalens-sean src + VENDORED_REF,
  71 files) local git vs `~gdbenson/claude-giga-lens/repo/.../claude-giga-lens/`: 67/71
  identical; STALE: cgl/e1.py, cgl/likelihood.py, cgl/samplers/common.py (+ cgl/euclid_io.py
  missing remotely) → rsync-trued to local git state, re-audit **CLEAN 71/71**.
- Executed-path semantics verified UNCHANGED by the truing: likelihood.py delta = P3
  keyword-only prior overrides whose defaults reproduce production bit-for-bit (IEEE
  x·1.0==x, same literals); common.py delta = a particles_to_chains guard NOT called by
  run_correlated_smc/weight_ess; e1.py/euclid_io.py not imported by the SMC path. So the
  seed-repeats run the exact production computation.
- Fit-npz inputs verified present remotely (`data/results/e2_v3b_low_canary_svicov.npz`,
  `data/results/e2_v3b.npz`). New-campaign staging verified:
  `/global/cfs/cdirs/deepsrch/gdbenson/cgl2-linus/{code,data,results,slurm-logs}`.
- Results path policy: hot I/O to `$PSCRATCH/cgl2-linus/results`, `cp` to CFS results at
  job end (new-campaign storage rule); the OLD campaign's staged tree receives no new files.
- **SUBMITTED 2026-07-15 13:09 PT (all 5, cosmo_g shared QOS, ledger rows above):**
  55951082 (t02-low-s3), 55951083 (t02-low-s4), 55951084 (t02-steep-s3), 55951085
  (t02-steep-s4), 55951086 (t03-compmask). Slurm files staged + md5-verified at
  `/global/cfs/cdirs/deepsrch/gdbenson/cgl2-linus/code/` (7/7 identical, incl. the variant
  whitener 13bfaf18… and its builder). All 5 registered with the watchdog (max_pending 24 h /
  max_run 6 h / expect_artifact = CFS result npz / on_stall alert); loop started on phoenix
  under nohup, PID 118755 (data/watchdog_loop.pid), verified reparented to init + first pass
  clean. NOTE (fail-loud by design): expect_artifact paths are CFS paths checked on phoenix's
  filesystem, so job COMPLETION will raise COMPLETED_NO_ARTIFACT until the harvest session
  pulls results and deregisters — that alert doubles as the harvest reminder.
- **NO results read this session** (house rule: submission+setup only; harvest is a later
  phase). T0.4 free CPU checks not part of this submission batch.

### 2026-07-15 — Cross-stack parity harness F1–F8 + correlated-noise port (task #12, phoenix L4, 0 A100-h)
- **F1–F5, F7 PASS; F6 documented FAIL (noise floor); F8 template staged.** Artifacts:
  `data/parity_refs.npz` (old-stack refs, 01a in the cgl venv, v2d+v3b, z_ref+3 perts),
  `data/parity_report_scene.json`, `data/whitener_manifest.json`,
  `data/correlated_term_validation.json`. Wall: 01a 111 s + 01 69 s + 03 14 s on L4 GPU 9.
- **First external certification of the scene-API forward model** for the EPL+Shear +
  4×Sersic + Sersic+Shapelets(n_max=6) config class: forward image and design columns match
  the validated 58ec9a7 stack to ≤6e-15 rel (v2d AND v3b), constrained-space gradients to
  ≤1.5e-11 rel-L2 — **given three documented convention reconciliations** (scene_build
  docstring items 1–3): (1) Sersic bn approximant differs between stacks (old 1.9992n−0.3271
  vs their exp(0.6950+ln n−0.1789/n)) — a real ~2e-3 model-level difference, measured and
  reported (informational native_profiles arm); (2) old shapelet Hermite prefactor is f32
  (~1e-8 basis delta); (3) old coordinate grids are f32-valued + old subgrid PSF kernel is
  NOT re-normalized (~1e-4/2e-6 image deltas at v2d/v3b). Parity runs use cgl2-side
  subclasses/instance-overrides reproducing the old conventions; the VENDOR IS UNPATCHED.
- **Correlated-noise LikelihoodTerm ported** (cgl2/correlated.py: CorrelatedImageData +
  CorrelatedImageLikelihoodTerm, upstream-shaped on their documented Dataset/LikelihoodTerm
  seam): ONE lstsq_simulate(return_stacked=True) render per eval, grouped-depthwise-conv
  whitening (jax.checkpoint, default ON), generalized-ridge marginalization with the
  −½logdetA Occam term, reports_chi2=True (whitened χ²), event_size = kept whitened dof,
  delta-kernel limit ≡ stock ImageLikelihoodTerm (F5, ≤5.9e-11), dense-Cholesky exact
  reference agrees to 5.5e-12 nats (03 gate A vs its 0.1-nat threshold).
- **F6 honest failure:** threshold 1e-10 is below the measured f64 cross-algorithm noise
  floor at v3b's cond(A)=7.0e7 — pure-numpy chol-vs-slogdet on the IDENTICAL matrix differs
  by 7.2e-10, and vs an fp128 truth our jax-chol Occam term (err 4.5e-10) is MORE accurate
  than the numpy-slogdet gate reference (err 7.1e-10) and than the old stack's own value
  (err 1.3e-9). v2d passes (2.4e-11). Gate not moved; exception recommendation above.
- F1 gate scoping note: the pre-registered F1 statement ("forward image, old vs
  SceneSimulator.simulate") is gated as written; the full model image with each stack's own
  SOLVED amplitudes is reported informationally (worst 1.1e-12) — it compounds F1×F2 through
  the cond(A)≈7e7 amplitude solve, a path the likelihood itself never takes (b·a* quadratic
  form is what enters logL, gated via F3/F4).
- Whitener bundles imported by path + re-validated (e_op reproduced to 0.0 diff; keep_w ==
  erode_keep(product mask, M) for all 4; v2d_relaxed correctly inadmissible under the strict
  0.02 gate — it was built as the ledgered relaxed arm). 54 CPU unit tests green
  (tests/test_{param_map,guards,correlated_term}.py + P0 suite); ./00_run_tests.sh wired.

### 2026-07-15 — X1-G0 + Fermat teaser (free checks, both complete, 0 A100-h)
- **X1-G0 FAIL → P4 retired (D7).** The gate worked exactly as designed: the profile-curvature
  mechanism cannot produce the bracket (no r_eff ordering in 24/24 variants; magnitude kill
  \|dγ/dln r\|≈226 required). BPL evidence could still differ for OTHER reasons, but the
  pre-registered mechanism is excluded — no GPU spend is justified on it. Source/PSF track
  re-inherits the bracket question.
- **Fermat Δφ teaser: 60–90% noise-model shift** (~10–17σ) — the motivation number for
  correlated noise in any future TD work; prominently disclaimed as illustrative.
- **DATA PRESERVATION: the P1c money-number SMC particles were ONLY on Perlmutter**
  (`~gdbenson/claude-giga-lens/repo/.../data/results/`); pulled (~22 MB) and preserved to local
  `../claude-giga-lens/data/results/` (e2_v3b_low_smc_canary_fix.npz md5 db4cc221…, + steep p96,
  + e2_{v2d,v3,v3b}.npz correlated-HMC). Machinery validated en route: numpy EPL vs vendored jax
  EPL to 1.3e-15; all three posterior transforms reproduce known γ medians.

### 2026-07-15 — P0 open
- Branch `claude-giga-lens-linus` created; plan + engagement memo committed (2f67083).
- Vendor @80916d2 archived (15 MB, UNPATCHED); venv cgl2 built; full import smoke PASS
  (scene API + MCLMC/MAMS kernels + EPL/Shear/BPL/PIEMD/PIEP profiles under jax 0.6.2 CPU).
  Two missing runtime deps found (lenstronomy, objax+tqdm) — installed under constraints;
  gigalens pip metadata complains (numpy 2.1.3, tensorflow) — expected, D2.
- D3 recorded: reference-artifact parity design (gigalens package-name collision).
- cgl2 skeleton: paths.py (vendor bootstrap + jax pin), guards.py (carried + new fences),
  pyproject, x64 bootstrap __init__.
