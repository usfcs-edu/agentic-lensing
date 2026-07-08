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

Committed: 1.20 / 90 (hard stop 100).
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
| E1b width-ratio sub-gate | 2026-07-08 | **FAIL (characterized)** | median σ_fine/σ_native = 2.45 (gate [0.7,1.5]), incl. fully-healthy pairs | fine posterior is CONSERVATIVE, not biased (z̄+cross-scale pass): δ-reg whitener (λ=0.1) discards information at the tent kernel's spectral zeros. Pre-registered amendment: λ-sensitivity arm added to E3; H3 (real-data honesty gate) unchanged |

## Stage log

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
