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
| — | — | — | — | — | 0.0 |

## Gate record

| Gate | Statement | Threshold | Status | Artifact |
|---|---|---|---|---|
| F1 | forward image (simulate + stacked M_det) | ≤1e-12 rel | **PASS 5.7e-15** | data/parity_report_scene.json |
| F2 | design columns (old `_design_ret` vs `return_stacked`) | ≤1e-12 rel | **PASS 3.9e-15** | data/parity_report_scene.json |
| F3 | diagonal masked loglik+χ² vs stock ImageLikelihoodTerm | ≤1e-8 | **PASS 7.5e-9** | data/parity_report_scene.json |
| F4 | grad of marg loglik wrt constrained params | ≤1e-8 rel-L2 | **PASS 1.5e-11** | data/parity_report_scene.json |
| F5 | delta-kernel CorrelatedTerm ≡ stock (fwd) [+F5b conv≡multiply] | ≤1e-10 | **PASS 5.8e-11 / 0.0** | data/parity_report_scene.json |
| F6 | Occam −½logdetA vs numpy slogdet | ≤1e-10 | **FAIL 1.31e-10 (v3b; v2d passes 2.4e-11)** — measured f64 cross-algorithm noise floor at cond(A)=7e7: numpy-chol−vs−slogdet on the IDENTICAL matrix = 7.2e-10; vs fp128 truth our jax-chol errs 4.5e-10 < slogdet's 7.1e-10 < old stack's 1.3e-9. Implementation verified correct; threshold unreachable by ANY f64 implementation on this product. Gate NOT moved; exception recommendation filed (below). | data/parity_report_scene.json |
| F7 | flat-z roundtrip + z_param_names audit | exact (info) | **PASS 0.0** (46 names, bijection clean, perturbation identity ok) | data/parity_report_scene.json |
| F8 | harness under NERSC jax-0.10 env | report-only | TEMPLATE READY (slurm/parity_f8_nersc.slurm, not run) | — |
| 03-A | corr term vs dense-Cholesky exact ref (32² toy) | ≤0.1 nat | **PASS 5.5e-12** | data/correlated_term_validation.json |
| 03-B | delta identities (fwd vs stock; conv vs multiply) | ≤1e-10 | **PASS 9.1e-13 / 0.0** | data/correlated_term_validation.json |
| W | whitener bundles re-validated (e_op reproduction + erosion + hashes) | e_op ≤0.02 strict | **PASS** (v2d/v3b/v3 admissible; v2d_relaxed inadmissible-by-design, e_op 0.0312 vs its own e_target 0.05) | data/whitener_manifest.json |
| B0 | MC-SMC correctness (adapters, mix2/funnel/illcond, MCLMC bias screen) | PLAN §5 | PENDING | data/smc_b0_report.json |
| X1-G0 | profile-curvature mechanism entry gate: r_eff ordering must admit the bracket's sign pattern | monotone ordering exists | **FAIL — hypothesis structurally dead** (24/24 robustness variants non-monotone; fine/binned constrain slope at the SAME radius, Δr_eff≈0.008″ < ¼ px, yet Δγ=0.71 ⇒ would need \|dγ_loc/dln r\|≈226 vs O(1) physical) | data/x1_g0_effective_radii.json, research/x1_g0_mechanism_check.md, figs/x1_g0_*.png |
| Fermat teaser | noise-model Δφ sensitivity (illustrative; NOT a TD lens; synthetic pairs; corr posterior is the known over-correcting product) | report-only | median \|frac shift\| **88%** anchor→corr (10.7σ); same-product diag→corr arm **61%** (17σ) — vs the ~1% TDCOSMO-relevant scale | data/fermat_dt_teaser.json, research/fermat_dt_teaser.md |

**Gate-exception recommendation (F6, NOT enacted — needs Benson's dated
sign-off):** restate F6 as "|jax-chol logdetA − fp128-truth logdetA| ≤ max(1e-10,
5·eps·cond(A)·1e-2)" or gate at the v2d product only; evidence: fp128 manual
Cholesky truth in data/parity_report_scene.json → products.*.F6.err_vs_truth.
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
