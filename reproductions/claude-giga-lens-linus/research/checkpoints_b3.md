# B3 design checkpoints (Front C — P2 B3 cells on phoenix)

Ledgered per the 2026-07-15 multi-front mobilization entry in CAMPAIGN.md: this file
carries the B3 front's pre-run design checkpoints (their-format: hypothesis + predicted
direction/magnitude + falsifier + derived threshold), WRITTEN BEFORE the runs. Gate rows
+ GPU-h actuals fold into CAMPAIGN.md at harvest (T1.1b agent owns that file this wave).

---

## B3 DESIGN CHECKPOINT (pre-registered 2026-07-15, before any cell run)

### Scope & deviations from PLAN §6 B3 (declared up front)
- **Cells: hs2_sys{0..3}** — 4 of the 8 adapters (A16 budget realism; assigned brief).
- **Reference = the classic scene-API MAP→SVI→HMC** via the VENDORED
  `gigalens.jax.inference.ModellingSequence` (diagonal ForwardProbModel likelihood).
  PLAN's "their MCLMC reference" is IMPOSSIBLE here: their observed-image npz + truth
  yaml are absent from the local mirror (zoo.py BLOCKED note), so the hs2 targets are
  OUR seeded mocks and no external reference numbers exist for them. The classic recipe
  is the GIGA-Lens BASELINE and OUR reference — explicitly NOT a benchmark row against
  their unpublished work (bright line §8.1-adjacent; these are unimodal easy targets, so
  every published number attaches to the reference-vs-SMC accuracy/cost question, never
  an MCLMC-vs-HMC comparison).
- Both arms run f64 (venv x64 bootstrap; hs2 SimulatorConfig leaves
  likelihood_precision=None → simulator defaults f64). Same-precision comparison.

### Hypothesis
On easy, unimodal, 22-dim single-lens targets (the honest easy-target row), prior-seeded
MC-SMC (MAMS mutation, N=512, frozen P0 protocol) reproduces the classic-recipe posterior
without a warm start — but has NO structural advantage (no multimodality, no cold-start
failure, evidence not needed), so it should NOT win the efficiency comparison; it buys
logZ that the classic recipe cannot produce at all.

### Predictions (direction AND magnitude, both cost directions — pre-registered)
1. **Accuracy**: all 4 cells pass worst-param |z| < 3 (definition below). Predicted
   worst-param |z| ~ O(1) (0.5–2.5).
2. **Cost, grad-evals (SMC predicted to LOSE, PLAN band)**: SMC/REF ∈ [1, 5] against the
   reference midpoint billing. Sizing arithmetic: SMC ≈ 512·(1+8·n_int)·n_stages + tune
   ≈ 0.6–1.8 M grad-evals (n_stages 15–35, n_int 3–12 on dim-22); REF = MAP 350×500 +
   SVI 500×250 (=300 k) + HMC 50×1000×L̄, L̄ ∈ [init_l=3, max_leapfrog=30] (tfp
   GradientBasedTrajectoryLengthAdaptation does not expose the adapted L̄ through their
   API) → REF ∈ [0.45 M, 1.8 M], midpoint ≈ 1.1 M.
3. **Cost, wall-clock same-device (SMC may WIN — the particle-vectorization direction)**:
   SMC/REF wall ∈ [0.3, 1.5]. SMC batches 512-wide; the classic HMC stage is a 50-chain
   batch × ≥1000 sequential steps × L̄ leapfrogs (latency-bound at batch 50: measured
   0.22 s/batched-grad on L4 vs 4.6 ms/particle-grad at batch 250).

### Frozen gate definitions (derived thresholds)
- **B3-accuracy (hard, per cell)**: worst-param |z| < 3 with
  z_j = (mean_SMC,j − mean_REF,j) / sqrt(var_REF,j/ESS_REF,j + var_SMC,j/ESS_SMC,j),
  j over the 22 unconstrained z-dims (both samplers' native space; bijector monotone).
  ESS_REF,j = tfp.mcmc.effective_sample_size(chains, cross_chain_dims) per param;
  ESS_SMC = final-stage unique-particle count (B0 illcond convention — duplicate
  degeneracy WIDENS the se, harder to pass, not easier).
- **B3-cost (two-sided report, per cell)**: grad-ratio quoted as the RANGE
  [SMC/REF_upper, SMC/REF_lower] (HMC L̄ bounds above); wall-ratio measured same-device.
  Bands (B1-mirror): WIN < 0.7, PARITY [0.7, 2), LOSS ≥ 2. If the grad-ratio range
  straddles a band boundary it is reported PARITY-AMBIGUOUS (no midpoint cherry-pick).
- **Reference acceptance (else cell = BLOCKED-BY-REFERENCE, not skipped)**: worst-param
  split-R̂ < 1.05 AND min per-param ESS ≥ 200 on the HMC stage. One pre-registered
  escalation allowed: num_burnin 250→500, num_results 750→1500 (their hundredsystems.py
  production values); if still failing, the cell is reported blocked-by-reference.
- **SMC sanity**: λ=1 reached (driver raises otherwise); final unique particles ≥ 32/512
  (else reported SICK, T1.1-control lesson); all logliks finite.
- **Falsifier for the hypothesis**: any accuracy FAIL on a converged reference kills the
  "SMC transfers to easy targets at N=512 prior-seeded" claim as stated and is reported
  as a FAILED gate (no goalpost move); cost outside BOTH predicted bands in the same
  direction (e.g. loses wall-clock >2× AND grads >5×) = the honest "two-stage recipe is
  simply better here" row.

### Design (frozen before launch)
- **Reference arm (per system)**: vendored `ModellingSequence(prob_model)` with the
  inference.py classic defaults = GIGA-Lens-paper recipe: MAP(optax.adam(-?), defaults:
  n_samples=500, num_steps=350, seed=0) → SVI(start=MAP best, n_vi=250, num_steps=500,
  seed=1) → HMC(q_z, n_hmc=50, num_burnin_steps=250, num_results=750, init_eps=0.3,
  init_l=3, max_leapfrog_steps=30, seed=2). Optimizers = THEIR pipeline defaults
  (GIGALens-Code src/gigalens_research/inference_utils/pipeline.py:1328-1336, replicated
  not imported): MAP optax.adabelief(1e-2, nesterov=True-if-supported), SVI
  optax.adabelief(1e-4, b1=0.95, b2=0.99). [Pre-launch amendment: an earlier draft of
  this checkpoint named adamw tutorial values; corrected to their pipeline defaults
  BEFORE any run — recorded here for transparency.]
  Their hundredsystems.py production settings (MAP 1000×2000, SVI 5000×1000, HMC
  64×(500+1500)) are HEAVIER; using classic defaults is the cheaper-reference choice and
  is billed as such (it can only make the reference LOOK cheaper, i.e. bias the cost
  comparison AGAINST SMC — the conservative direction for our bet).
  OOM fallback (pre-registered, auto): halving chain on RESOURCE_EXHAUSTED — MAP
  n_samples 500→256→128, SVI n_vi 250→128→64 (A16 15.3 GB headroom unknown at batch
  ≥256); recorded in cell json, billed at actuals. HMC n_hmc stays 50.
- **MC-SMC arm (per system)**: prior-seeded z0 = target.prior_sample_fn(PRNGKey(4000+sys),
  512); `cgl2.samplers.common.run_tempered_smc` with the frozen P0 protocol via a
  CHUNKED-MUTATION MAMS subclass (chunk=128): identical math and identical per-particle
  PRNG keys as smc_micro._MAMSKernel — the vmap over 512 particles is executed as 4
  sequential 128-chunks inside one jit (memory fit: batched grad at 512 measured OOM on
  L4 23 GB [28.4 GB request] and would OOM harder on A16 15.3 GB; chunk-128 grad measured
  1.00 s A16 / fits both). Execution-order-only change, pre-registered here; the frozen
  constants (NUM_MCMC_STEPS=4, TARGET_ESS=0.7, L_FACTOR=1.0, MAMS_DA_TARGET=0.90,
  PILOT_ITERS=10, pilot_size=64, eps0=0.5, n_boot=200, precondition=True) are UNTOUCHED.
  seed: run key PRNGKey(4100+sys), boot_seed 20260715+sys. max_stages 400.
- **Devices (same-device cost pairing, pre-registered)**: cell rows = A16 GPU i runs
  hs2_sys i (reference arm THEN SMC arm sequentially, same device → wall-clock ratio is
  same-device by construction). L4 GPU 8 re-runs the sys0 cell (both arms) as the
  cross-device check + wall-time reference (reported, not a cell row; f64 device
  invariance expected to ~1e-12 modulo reduction order — informational).
  CUDA_DEVICE_ORDER=PCI_BUS_ID everywhere; XLA priority-fusion disabled (aarch64
  workaround); one process per GPU.
- **Order of operations**: figs BEFORE gate math at harvest (house rule); cell JSONs are
  written by the runner with raw traces; the harvest step draws overlays first, then
  evaluates gates.

### Pre-run amendment (2026-07-15, machinery smoke — recorded BEFORE any cell run)
The tiny-budget machinery smoke (hs2_sys0, L4, MAP 30 steps × 64 — NOT a gate run, no
posterior read) exposed a SUBSTRATE DEFECT: the vendored `gigalens.jax.inference.
ModellingSequence.MAP` raises `TypeError: cotangent type does not match function output
... complex128[64,1,189,95] ... {V:device}` under jax 0.6.2 on a single GPU — the
shard_map-wrapped `value_and_grad` through the FFT PSF convolution hits the
varying-axis cotangent rules; inference.py's own comments show it was reworked for
jax 0.10, and it evidently no longer runs under the library's declared 0.6.2 pin.
Vendor stays UNPATCHED (D1). **Amendment**: the reference arm REPLICATES the classic
MAP→SVI→HMC recipe in 22_run_b3.py with attribution, byte-faithful math minus the
multi-device wrappers (shard_map/pmap over a 1-device mesh are semantically identity;
their per-step best-tracking order, ELBO construction FillScaleTriL(Exp, shift 1e-6),
init_scales=1e-3, and the tfp PreconditionedHMC + GradientBasedTrajectoryLengthAdaptation
+ DualAveragingStepSizeAdaptation stack are copied exactly; RNG streams equivalent to
their dev_cnt=1 path). Gate definitions, budgets, predictions UNCHANGED. The defect
itself is a deliverable finding (handoff/memo channel; UNCERTIFIED external): the
scene-API inference path requires their jax-0.10 container runtime — logged for the P2
deployment plan (research/p2_deployment_plan.md) and F8.

### Sizing-probe evidence (instrumentation only, run 2026-07-15 pre-checkpoint; no
posterior read, no gate math)
- L4 (GPU 8): batched grad 220 ms @50 / 291 ms @64 / 1156 ms @250 (≈4.6 ms/particle-grad);
  batch-500/512 grad OOM (28.4 GB request; XLA remat floor 22.3 GiB > 23 GB card).
  batched loglik 45 ms @50 … 255 ms @250.
- A16 (GPU 0): batched grad 255 ms @32 / 504 ms @64 / 1003 ms @128 (≈7.8 ms/particle-grad,
  linear); batched loglik @512 = 1.04 s (fits).
- Derived budget estimate: SMC/system ≈ 2–3 h (A16) / 1.1–1.7 h (L4); reference/system ≈
  1–4 h (A16, HMC L̄-dominated). Total ≈ 15–31 GPU-h vs the ~30–40 GPU-h envelope.
- **Budget fence (kill criterion)**: any single arm exceeding 8 h wall (A16) or 5 h (L4)
  is killed and its cell reported BLOCKED-BY-BUDGET with partial artifacts. Total front
  cap 40 GPU-h.

### Artifacts
data/b3_cells.json (all cells: configs, traces, timings, gate inputs),
data/b3_<arm>_<sys>.npz (samples/particles), figs/b3_sys{i}_overlay.png +
figs/b3_cost.png, research/b3_readout.md (gate table + verdicts).

---

## RESTART ENTRY (2026-07-16, before any re-run; gates/seeds/recipe UNTOUCHED)

**What happened (2026-07-15 22:26–22:33, logs data/b3_lane_*.log):**
1. **A16 OOM — the reference arm does not fit the A16 at ANY pre-registered
   size.** All four a16 lanes OOM'd the MAP stage at every step of the
   documented halving ladder: n_samples 500 → 26.9 GiB, 256 → 13.85 GiB,
   128 → 12.61 GiB requested; all RESOURCE_EXHAUSTED on the 15.3 GB A16
   (XLA remat floor ~10.4 GiB, but actual peak allocations stayed above the
   card). The `_oom_retry` ladder exhausted → runner error, no artifacts.
2. **Parent-exit kill.** The lanes were launched as background children of the
   agent session (no setsid); when the parent agent exited, the l4-8 sys0 lane
   died mid-fallback (log ends right after the first OOM at batch 500 — the
   batch-256 retry, which fits the 23 GB L4, never got to run). No completed
   b3_run_*.json / b3_*.npz exist; nothing is salvageable.

**Fix (execution-plan-only changes; hypothesis, predictions, gate definitions,
frozen constants, seeds, and the classic-recipe settings are all UNCHANGED):**
- **Device deviation (documented):** all 4 cells (hs2_sys{0..3}) now run
  SEQUENTIALLY on the L4 (GPU 8, 23 GB, CUDA_DEVICE_ORDER=PCI_BUS_ID
  CUDA_VISIBLE_DEVICES=8), ref arm THEN smc arm per system — the same-device
  wall-clock pairing is preserved per cell, devtag l4-8. The pre-registered
  "A16 GPU i runs sys i" layout is IMPOSSIBLE (point 1); the A16-vs-L4
  cross-device check is DROPPED for the same reason (no A16 reference can
  exist). GPU 9 (L4) carries the L0 leg and is not touched.
- **Expected pre-registered fallback on record:** on the L4 the MAP stage will
  OOM at n_samples=500 (28.1 GiB request, measured) and proceed at the
  documented 256 fallback (13.85 GiB, fits) — this is the checkpoint's own
  halving rule, billed at actuals in the cell json (oom_fallbacks field), not
  a new setting. SVI 250 and HMC 50 chains are expected to fit unchanged.
- **Detachment discipline:** one driver script 22_run_b3_lanes.sh chains all
  8 runs; launched via nohup + setsid (survives parent exit, reparented to
  PID 1, verified at launch); logs data/b3_lane2_sys<i>_<arm>.log; driver PID
  in data/b3_lane2_driver.pid; phoenix watchdog entry (max_run_h 20,
  expect_artifact data/b3_cells.json). Budget fence per arm enforced as
  timeout 18000 s (= the pre-registered 5 h L4 kill criterion).
- **Harvest:** 22_harvest_b3.py (supersedes 22_run_b3_harvest.py — cell tags
  now l4-8, cross-device section dropped) runs AFTER the driver finishes
  (figs first, then gates — house rule); it writes data/b3_cells.json.
- **Budget outlook:** expected ~9–21 GPU-h on the L4 (ref 1–3.5 h + smc
  1.1–1.7 h per system), within the 40 GPU-h front cap; worst case bounded by
  8 × 5 h = 40 h wall via the per-arm fence, watchdog alarm at 20 h.
