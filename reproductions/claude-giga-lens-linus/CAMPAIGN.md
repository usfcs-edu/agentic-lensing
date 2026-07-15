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
| 2026-07-15 | 55951082 cgl2-t02-low-s3 (v3b-low SMC p128 seed3, slurm/t02_smc_v3b_low_seed3.slurm) | P1 T0.2 | 1.5 | — | 1.5 (est) |
| 2026-07-15 | 55951083 cgl2-t02-low-s4 (v3b-low SMC p128 seed4, slurm/t02_smc_v3b_low_seed4.slurm) | P1 T0.2 | 1.5 | — | 3.0 (est) |
| 2026-07-15 | 55951084 cgl2-t02-steep-s3 (v3b-steep SMC p96 seed3, slurm/t02_smc_v3b_steep_seed3.slurm) | P1 T0.2 | 2.0 | — | 5.0 (est) |
| 2026-07-15 | 55951085 cgl2-t02-steep-s4 (v3b-steep SMC p96 seed4, slurm/t02_smc_v3b_steep_seed4.slurm) | P1 T0.2 | 2.0 | — | 7.0 (est) |
| 2026-07-15 | 55951086 cgl2-t03-compmask (v3b-low SMC p128 seed2, companion-eroded whitener, slurm/t03_smc_v3b_low_compmask.slurm) | P1 T0.3 | 2.0 | — | 9.0 (est) |

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
| Fermat teaser | noise-model Δφ sensitivity (illustrative; NOT a TD lens; synthetic pairs; corr posterior is the known over-correcting product) | report-only | median \|frac shift\| **88%** anchor→corr (10.7σ); same-product diag→corr arm **61%** (17σ) — vs the ~1% TDCOSMO-relevant scale | data/fermat_dt_teaser.json, research/fermat_dt_teaser.md |
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
