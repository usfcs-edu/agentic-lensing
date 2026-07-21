# L0 design checkpoints — P3-L0 anchor arbitration (phoenix GPU 9 L4 only, 0 A100-h)

Ledgered per the CAMPAIGN.md multi-front mobilization entry (2026-07-15 evening):
this file carries the L0 front's pre-run design checkpoints; gate rows and
GPU-h actuals fold into CAMPAIGN.md at harvest. Owner files: 10_anchor_arbitration*.py,
research/checkpoints_l0.md, research/l0_*.md, figs/l0_*, data/l0_*.

---

## 2026-07-15 — L0-ARB DESIGN CHECKPOINT (pre-registered, written BEFORE any run)

**Question (PLAN §6 P3-L0, §9 risk row "1.103/1.433 premise itself artifactual"):**
does the scene API, fitting the SAME v2d native product with the DIAGONAL
likelihood, reproduce the 1.433 anchor at posterior level? Falsify the premise
before explaining it.

**Hypothesis:** the anchor is STACK-ROBUST — the v2d-native diagonal
(masked-err down-weighted, ridge-marginalized 46-dim) posterior on the scene
API sits where foundry-i's validated fit sat. Mechanically this SHOULD hold:
the two stacks' marg likelihood and gradients agree to ≤2.5e-8 / 1e-13
(F1–F7, data/parity_report_scene.json) and the priors are the same
construction (scene_build verbatim hyperparameters; prior logp cross-stack
abs diff 2.6e-6, informational) — so a posterior-level failure would isolate
sampler/posterior-geometry systematics, not likelihood math.

**Anchor reference (frozen):** gamma_anchor = 1.433, 68% CI [1.400, 1.468]
(foundry-i hmc_v13_v2d headline, reproductions/foundry-i/PERLMUTTER_CAMPAIGN.md;
sigma_anchor := half-width = 0.034).

**Design (required arm):**
- Model: cgl2.scene_build.build_scene_model(view="marg", near_xy=(-2.34, -2.86)
  from the parity-refs provenance) — the foundry-i 46-dim config class,
  foundry-i priors verbatim, PARITY-CERTIFIED configuration
  (assert_scene_config_certified; parity profiles + old f32 grids +
  unrenormalized subgrid PSF from data/parity_refs.npz; Lambda_diag from refs).
  Rationale: the certified config is the one F1–F7 tie to the validated stack;
  running the native-convention profiles would contaminate the arbitration
  with the documented ~2e-3 bn model delta.
- Likelihood: CorrelatedImageData + delta bundle on masked_err (the old
  stack's down-weighting convention bit-for-bit — the F2/F4 parity anchor),
  corr_mode='marg', Lambda_diag = refs v2d — i.e. the diagonal masked
  ridge-marg likelihood on cutout_v2d (80x80, 0.13", ss2). ProbModel supplies
  the blackjax (log_prior, log_like) split (B0-verified seam).
- Sampler: MC-SMC MAMS (B0-validated cgl2.samplers.smc_micro.run_smc,
  kernel='mams', frozen P0 protocol: target_ess 0.7, 4 draws/stage,
  per-lambda ensemble preconditioning), PRIOR-SEEDED (model.prior.sample ->
  bijector.inverse), N=512, seed 2 (the campaign production seed). Per-stage
  logging + particle checkpointing via a DELEGATING kernel wrapper in
  10_anchor_arbitration.py (validated driver and kernel code untouched).
- Optional arms (stretch, run ONLY if the required arm converges with >=2.5 h
  margin under the cap, in this priority order): (a) second SMC seed (3) for a
  worst-param seed-consistency check; (b) classic recipe (vendor
  MAP->SVI->HMC) on the same ProbModel — two arbitration arms beat one.

**Prediction (pre-registered):** gamma_med(scene, diag, v2d) within
2*sigma_comb of 1.433, where sigma_comb = sqrt(sigma_anchor^2 + sigma_scene^2),
sigma_anchor = 0.034, sigma_scene = (q84-q16)/2 of the scene posterior;
AND the two 68% intervals OVERLAP.

**Falsifier:** DISJOINT 68% intervals => the anchor is NOT stack-robust at
posterior level => the 17-sigma premise partially dissolves into
implementation/sampling systematics — a devastating-but-cheap catch, reported
as a finding (no silent fixing, no threshold moves).

**Multimodality clause (pre-registered):** the marg-MAP z_ref
(foundry-i map_marg_pd qz_refined) sits at gamma = 1.8655 while the anchor
posterior is 1.433 — a prior-seeded SMC may populate basins the anchor's
warm-started HMC never visited. Readout rule: PLOT the gamma histogram FIRST
(figs/l0_*, before any gate math); if multimodal, the arbitration gate applies
to the DOMINANT basin (equal-weight occupancy; per-basin gamma + occupancy
reported; a dominant far basin fires the falsifier), and secondary-basin mass
is a separate reported finding. Continuity stat frac(gamma > 1.9) reported
(the campaign's v3b basin-split convention).

**Wiring sanity gates (run before production; a failure means fix wiring,
never thresholds):**
- S1: this script's term logL at v2d z_ref + 3 perts vs refs logL_data,
  abs diff <= 1e-7 (proves the sampled object IS the F-gate-certified
  likelihood; parity itself already certified — worst recorded marg_dlogL
  2.4e-8).
- S2: pm.log_prior at z_ref vs refs v2d:z_ref:log_prior, abs diff <= 1e-5
  (parity-info recorded cross-stack prior delta is 2.6e-6).
- S3: N=512 prior draws give finite logprior + loglik (the driver's stage-0
  finiteness assertion enforces this fail-loud).

**Runtime / memory protocol (pre-registered):**
- Timing smoke first: max_stages=2 probe at N=512 (RuntimeError caught),
  extrapolate with an expected full schedule of ~40 stages (T0.2 warm-started
  low ran 28; prior-seeded runs longer; B0 zoo targets ran ~15–40). If
  projected wall > 5.0 h at N=512 -> single pre-registered halving to N=256
  (median SE stays ~0.003 << sigma_anchor). If OOM: halve N (512->256->128),
  record measured memory.
- Hard cap ~6 wall-hours on GPU 9 (L4 23 GB, CUDA_DEVICE_ORDER=PCI_BUS_ID,
  GIGALENS_X64=1). On cap: keep the newest per-stage particle checkpoint,
  report lambda_reached + PARTIAL posterior, NO gate math on a non-lambda=1
  ensemble.
- Convergence reporting (campaign SMC conventions): reached lambda=1,
  n_stages << 400, final-weights ESS, unique-particle trace, acceptance
  trace; worst-param diagnostic = max over the 46 params of |median shift
  between the final two stage ensembles| / final posterior sigma (expected
  < 0.5), upgraded to a proper 2-seed worst-param check if stretch arm (a)
  runs.

**L0-G2 memory smoke (the v3b-low correlated refit start; feasibility, not
science):**
- Build the v3b CorrelatedImageData with the ported whitener_v3b bundle
  (manifest sha 8816745c..., sqrt_d_inv = 1/masked_err_v3b, keep_w 9273,
  checkpoint=True) on the scene marg model; measure peak device memory of
  vmapped value_and_grad(loglik) at 8 and 16 particles (linearity check).
  MB/particle = slope; target from the old campaign's fix is ~200 MB/particle.
- Decision rule: N_corr_feasible = floor((23034 * 0.90 - measured overhead) /
  MB_per_particle). If N_corr_feasible >= 96 AND the arbitration arm finished
  with >= 2.5 h margin -> start L0-G2 (v3b-low, prior-seeded is NOT the
  production recipe — record that any L4 run here is a scoping run; the
  certified L0-G2 gate remains gamma_binned(corr,low) = 1.1032 within
  2*sqrt(sigma_stat^2+sigma_seed^2) = 0.0172 + low-basin logZ sign, and would
  normally run warm-started at p128 on Perlmutter). Else: DEFER L0-G2 to the
  Perlmutter deployment (front C) with the measured MB/particle documented —
  a finding, not a failure.

**Budget:** 0 A100-h (phoenix-only). Artifacts: data/l0_*.json,
figs/l0_*.png, research/l0_anchor_arbitration.md; per-stage checkpoints under
data/l0_smc_checkpoints/ (gitignored bulk).

---

## 2026-07-15 22:50 PT — L0-ARB DEVIATION ENTRY (written BEFORE the retry run)

**Observed (recorded, no results read):** the MAMS mutation OOMs on the L4 at
every step of the pre-registered halving sequence — N=512 (49.3 GB request),
N=256 (23.3 GB), N=128 (12.57 GB request with ~12.6 GB already live: the
mutate working set is ~2x the single big buffer, ~190 MB/particle total).
The registered sequence is exhausted; N=64 would sit below the campaign's
p128 SMC convention and thin the basin-occupancy readout.

**Deviation (ledgered, F6-restatement pattern):** enable the VENDOR's own
memory fix for exactly this regime — ``SimulatorConfig.remat_basis=True``
(gigalens-linus C-16/C-17 compute-profiling: batched MAP is memory-bound at
ss=2, remat cuts peak ~46%; "Exactness: remat re-runs identical ops", gated
upstream in their wip/validate_remat_default.py). No numerics change by
construction; the vendor stays UNPATCHED (config flag only, via
dataclasses.replace at build time behind env L0_REMAT=1).

**Pre-registered exactness gate for the deviation (must pass BEFORE the
retry):** diagonal marg term logL at v2d z_ref + 3 perts, remat ON vs OFF,
abs diff <= 1e-10 (same-device identical-op re-execution; roundoff-free in
expectation). Fail -> abandon remat, fall back to N=64 under the original
halving rule and record the power loss.

**Retry plan:** N=128 (the campaign-standard SMC particle count, in-family
with every certified SMC number of P1), seed 2, remat ON, all other frozen
protocol constants unchanged. If remat+128 still OOMs -> N=64 remat ON.
All arbitration THRESHOLDS unchanged (they never referenced N).

**Execution record 2 (2026-07-16 03:10 PT) — RESUME DEVIATION (written BEFORE
the resume run):** the N=64+remat production run (relaunched detached 00:22
after the task harness killed the first attempt at stage 4; deterministic
seed-2 stage rows reproduced bit-for-bit) progressed to stage 19
(lambda=6.5e-4, ~200-600 s/stage, accept 0.77-0.91, eps 0.04->0.4, n_int
64->24) and then OOM'd at the STAGE-20 PILOT on a new n_int compile
(10.23 GB request; accumulated compiled n_int variants + BFC fragmentation
— an ops failure, not a numerics one; third memory lesson for the P2
deployment plan). Per-stage particle checkpoints stage_000-019 intact.
DEVIATION: resume-from-checkpoint driver in 10_anchor_arbitration.py
(`resume` stage; loop copied with attribution from the validated
common.run_tempered_smc, identical per-stage semantics/constants, same
kernel adapter, eps warm-started from the stage-19 record, FRESH jax key
chain — SMC invariance does not depend on key continuation). CONSEQUENCE
(recorded before results): logZ is NOT quotable for this run (pre-resume
increments lost with the process) — the pre-registered arbitration gate
never referenced logZ (it gates on the posterior gamma), so the gate is
unaffected; occupancy remains the multimodality readout. A fresh process
also resets the compile-cache memory pressure (only the remaining 1-3
n_int variants get compiled). Resume-leg wall cap 3.0 h.

**Execution record 1 (2026-07-15 23:15 PT):** remat exactness gate PASS —
abs diff exactly 0.0 at z_ref + 3 perts (data/l0_remat_gate.json), and the
remat-off values reproduce the sanity S1 values bit-for-bit. N=128+remat
STILL OOM (11.89 GB request vs 12.57 GB without remat — only ~5% saved:
the dominant tape is the vmapped gradient of the full render, which
remat_basis does not cover; a P2-deployment-relevant measurement in itself:
~93 MB/particle irreducible grad tape at v2d ss2 f64 on this stack).
Pre-registered fallback FIRED: production run = N=64, remat ON, seed 2.
Power note (recorded before readout): final-ensemble median SE ~ 0.16*sigma
~ 0.006 if sigma~0.034 — small vs the 2*sigma_comb >= 0.068 gate band;
basin-occupancy resolution 1/64.

**Execution record 3 (2026-07-16 04:50 PT) — session close-out:**
- Resume leg 2 (fusion workaround `--xla_disable_hlo_passes=priority-fusion`
  — the campaign's sanctioned XLA knob; the 10.23 GB stage-20 pilot compile
  reproduced IDENTICALLY in a fresh process, so it was a compiler-pass
  pathology, not fragmentation) CLEARED the killer stage and ran stages
  20-26 cleanly (accept 0.82-0.91, eps 0.16-0.55, n_int 16-48).
- lambda growth measured ~1.2x/stage => ~32 stages remained from
  lambda=2.92e-3; the pre-registered ~6 h L4 envelope cannot reach lambda=1
  => **ARBITRATION VERDICT THIS SESSION = PARTIAL per the pre-registered
  runtime protocol. NO gate math, NO gamma numbers read.** Sampler health
  through stage 26 is clean (figs/l0_smc_traces_partial.png, plotted before
  any readout; lambda schedule smooth, acceptance around the 0.9 target).
- Leg 2 stopped AT stage 26 (checkpoint stage_026.npz, lam=2.9247e-3) to
  reclaim GPU 9 for the mandatory L0-G2 smoke; then **completion leg 3
  launched detached** (PID 265886, resume from stage 26, wall cap 12 h,
  same frozen protocol + fusion workaround). On reaching lambda=1 it writes
  data/l0_arbitration_smc.{npz,json} status=COMPLETE; harvest = plots FIRST
  then the UNCHANGED pre-registered gates via
  `10_anchor_arbitration.py harvest`.
- **L0-G2 memory smoke (data/l0_v3b_memory_smoke.json): 367.6 MB/particle**
  (8->16 slope, linear; upper-bound 357.7; checkpointed whitening ON;
  value+grad FINITE at 8 and 16 particles — the ported correlated v3b term
  is functional end-to-end on the scene API). Feasible ensemble on the L4
  ~56 particles < 96 => **L0-G2 DEFERRED to Perlmutter** per the
  pre-registered rule (a finding, not a failure). Cross-check: 128 x 367.6
  MB ~ 47 GB — independently consistent with the T1.1 inj3 lesson that
  production correlated SMC needs hbm80g A100s.
- Ops findings for the P2 deployment plan (three measured memory-failure
  classes on the L4): (a) v2d diag MAMS mutate grad tape ~93 MB/particle
  irreducible (49.3/23.3/12.57 GB at N=512/256/128; remat_basis saves ~5%);
  (b) new-n_int pilot compile requesting a pathological 10.23 GB temp under
  priority-fusion (fixed by disabling the pass); (c) v3b correlated 367.6
  MB/particle. L4 f64 stage cost at N=64: ~190-950 s depending on n_int.

---

## 2026-07-20 — L0-ARB PROTOCOL AMENDMENT: classic-recipe arm PROMOTED to primary arbitration vehicle (written BEFORE the classic-arm run; gates and bands UNCHANGED)

**Observed (ops record only — no gate math, no gamma read):** three MC-SMC
arbitration attempts have now ended at wall caps without reaching lambda=1:
1. phoenix L4 (N=64 + remat, legs 1-3): lambda ~0.45 at stage 67 after ~18
   wall-h total (data/l0_arbitration_smc.json status=PARTIAL_WALL_CAP,
   lambda_reached=0.418 at the resume-leg cap trigger, last trace row
   lambda=0.450; checkpoints data/l0_smc_checkpoints/stage_000-067).
2. Perlmutter A100 plain-gpu (56006048): OOM before stage 0 (hardware,
   ledgered 2026-07-19 — never a lambda datum).
3. Perlmutter A100 hbm80g (56168446, N=128): HEALTHY MAMS (accept ~0.88, no
   OOM, peak 22.7 GB) but PARTIAL_WALL_CAP at stage 76, lambda=0.587 inside
   the 3.5-h cap (CFS results/l0arb/l0_arbitration_smc.json + checkpoints
   stage_000-075).
The prior-seeded MC-SMC lambda-schedule on this target needs a multiple of
any wall this campaign can budget (~1.2x lambda growth per stage measured on
both machines). The VEHICLE is budget-infeasible here — the sampler is not
sick, and nothing in this amendment reinterprets its partial output.

**Amendment (vehicle only — every gate, band, and clause UNCHANGED):** the
original 2026-07-15 checkpoint listed optional arm (b): "classic recipe
(vendor MAP->SVI->HMC) on the same ProbModel — two arbitration arms beat
one." That arm is hereby PROMOTED to PRIMARY arbitration vehicle:
- Classic scene-API recipe MAP -> SVI -> HMC on the SAME v2d diagonal
  scene-API target (build_pm("v2d", diagonal=True) — the F-gate-certified
  likelihood, verbatim), via the vendored ModellingSequence recipe
  REPLICATED single-device with attribution per the B3 checkpoint amendment
  (the vendored MAP raises the known jax-0.6.2 shard_map cotangent
  TypeError; 22_run_b3.py's reference-arm replication ran 4/4 on hs2).
- Their standard settings = the B3 reference-arm FROZEN constants:
  MAP adabelief(1e-2, nesterov) x 350 steps; SVI full-rank MVN-TriL,
  250 draws x 500 steps, adabelief(1e-4, b1=.95, b2=.99); HMC 50 chains,
  250 burn + 750 results, init_eps 0.3, init_l 3, max_leapfrog 30, with the
  single pre-registered escalation to 500/1500 on Rhat/ESS failure
  (Rhat < 1.05, ESS >= 200); stage seeds 0/1/2 (B3 convention).
- WARM start: MAP stage starts from the MAPPED foundry-i MAP —
  param_map.scene_z_from_old_labels(model, labeled_from_vec46(
  refs["v2d:z_ref:x46"])) — the same object S1 wiring-certified. The
  multi-start MAP is replaced by a single-start warm refine (multi-start
  from a fixed warm point is degenerate; declared, not hidden). This is the
  SAME algorithm class, warm-started the same way, that produced the 1.433
  anchor on the old stack (hmc_v13_v2d warm-chained from map_marg_pd).
- The MC-SMC partials are RETAINED as SECONDARY, prior-seeded evidence with
  their PARTIAL status honestly recorded: L4 checkpoints stage_000-067
  (lambda ~0.45, N=64) and A100 checkpoints stage_000-075 (lambda 0.587,
  N=128). The standing rule is unchanged: NO gate math on a non-lambda=1
  ensemble; their lambda<1 occupancy content is context, never a gate input.
- GATES AND BANDS UNCHANGED, verbatim from 2026-07-15: dominant-basin
  gamma_med within 2*sigma_comb of 1.433 (sigma_anchor 0.034), AND
  overlapping 68% intervals; multimodality clause (plot FIRST, dominant
  basin by equal-weight occupancy, frac(gamma>1.9) continuity stat) and the
  falsifier (disjoint 68% intervals => anchor not stack-robust) as written.
  The arbitration GATE is about the POSTERIOR, not the sampler.

**Trade-off stated honestly (before the run):** the classic arm INHERITS the
warm-start basin. That is acceptable FOR ARBITRATION because the old-stack
anchor fit inherited it identically — the arm asks exactly "does the same
algorithm class, warm-started the same way, on the parity-certified same
data + likelihood, reproduce the anchor posterior on the new stack?" What
the warm arm CANNOT provide is prior-seeded coverage of alternative basins
— that was the MC-SMC arm's job and it REMAINS OPEN; the retained partials
are the honest record of how far that coverage got.

**Vehicle script:** 11_arbitration_classic.py (new operator; imports
10_anchor_arbitration.py's certified builders by path; refuses to run
without data/l0_sanity_report.json sanity_ok). Writes
data/l0_arbitration_classic.{npz,json}: gamma posterior samples + per-param
Rhat/ESS incl. worst-param + stage walls; artifacts written stage-wise
(post-SVI, post-HMC, post-escalation) so a walltime kill loses at most one
stage. NO gate math in the script — harvest stays plots-first.

**Runtime protocol:** Perlmutter plain `-C gpu` (40 GB) on purpose: the
56006048 21-GiB OOM was the N=128 MAMS mutate tape; this arm's tapes are
MAP n=1 (trivial), SVI <= 250-draw ELBO grad with the B3 OOM halving ladder
250->128->64 (ladder hits = recorded fallbacks, not failures), HMC 50-chain
batch (~50 x ~93 MB ~ 4.7 GB). Wall 2:00, est 1.5 A100-h, P3 ledger row
BEFORE results (slurm/p3_l0arb_classic.slurm, $PSCRATCH audited-copy
pattern + PYTHONPATH vendor pin from p3_l0arb.slurm).
