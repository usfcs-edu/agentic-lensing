# P3-L0 anchor arbitration — session report (2026-07-16, phoenix GPU 9 L4, 0 A100-h)

Pre-registered design + deviations: `research/checkpoints_l0.md` (all entries
written BEFORE the corresponding runs). Script: `10_anchor_arbitration.py`
(stages: sanity / rematgate / run / resume / v3bsmoke / harvest).

## Status at a glance

| Item | Verdict | Where |
|---|---|---|
| Wiring gates S1/S2/S3/S4 (scene fit object == F-gate-certified likelihood) | **PASS** | data/l0_sanity_report.json |
| Remat-deviation exactness gate | **PASS (diff exactly 0.0)** | data/l0_remat_gate.json |
| Arbitration SMC (v2d diag, MAMS, prior-seeded, seed 2) | **PARTIAL — lambda 2.92e-3 @ stage 26; pre-registered ~6 h L4 cap; NO gate math, NO gamma read** | data/l0_smc_checkpoints/, figs/l0_smc_traces_partial.png |
| Completion leg (detached, cap 12 h) | RUNNING (PID 265886); writes data/l0_arbitration_smc.* on lambda=1 | data/l0_run.log |
| L0-G2 (v3b correlated refit on scene API) | **DEFERRED to Perlmutter** by the pre-registered memory rule: 367.6 MB/particle -> ~56 feasible < 96 | data/l0_v3b_memory_smoke.json |

## What was certified this session (usable now)

1. **The sampled object IS the certified likelihood.** The scene-API
   ProbModel assembled for the arbitration (marg view + delta bundle on
   masked_err + parity grid/PSF conventions) reproduces the old-stack
   reference logL at z_ref + 3 perturbations to <= 2.5e-8 abs (S1), the
   cross-stack prior logp to 2.6e-6 (S2, matches the parity-info level), and
   the name-keyed gamma column identity to < 1e-9 (S4). 512 prior draws give
   finite (logprior, loglik) everywhere (S3).
2. **The ported correlated v3b term runs end-to-end on the scene API**
   (value+grad finite under vmap at 8/16 particles, checkpointed whitening,
   whitener_v3b bundle) — the L0 port itself is functional; only the L4's
   memory stops the refit here.
3. **`remat_basis` is exactly neutral numerically** (0.0 logL delta at 4
   points) and saves only ~5% of the mutation grad-tape on this workload —
   the vendor's documented ~46% applies to batched MAP, not to the
   MAMS-mutation gradient, whose tape is dominated by the render+whiten
   backward pass (~93 MB/particle at v2d ss2 f64).

## Why the arbitration is PARTIAL (and exactly per protocol)

Prior-seeded adaptive tempering on the real 5865-kept-pixel v2d diagonal
target needs ~58 ESS-0.7 stages (measured lambda growth ~1.2x/stage from
1e-7), and the L4 delivers ~190-950 s/stage at N=64 f64. The pre-registered
runtime protocol ("if the SMC hasn't converged in ~6 wall-hours on the L4,
checkpoint state, record, and report partial") fired at stage 26,
lambda=2.92e-3. Sampler health through stage 26 is clean — smooth lambda
schedule, MAMS acceptance 0.77-0.92 around the 0.9 target, dual-averaging eps
0.04->0.55, n_int laddering 64->12->48 (figs/l0_smc_traces_partial.png,
plotted before any readout). The run is stopped-not-broken: per-stage
particle checkpoints stage_000..026 are on disk, and the proven
resume-from-checkpoint leg (ledgered deviation, execution record 2) is
running detached with a 12 h cap. **No posterior gamma has been read; the
pre-registered bands are untouched** (dominant-basin median within
2*sigma_comb of 1.433 with overlapping 68% intervals; falsifier = disjoint
intervals).

### Harvest instructions (next session)
1. Check `data/l0_arbitration_smc.json` — status must be COMPLETE
   (reached_lambda1). If PARTIAL again: resume once more (`resume` stage,
   same env) or move the completion to Perlmutter (see below).
2. `10_anchor_arbitration.py harvest` — writes figs/l0_gamma_hist.png +
   figs/l0_smc_traces.png FIRST, then data/l0_gate_eval.json with the
   pre-registered gate math (including the data-driven basin split and the
   1.9-frac continuity stat). Inspect the plots before the json.
3. Fold the gate row into CAMPAIGN.md. Note logZ is NOT quotable for this
   run (resume deviation, execution record 2) — the gate never used it.

## Memory + runtime findings (deliverables for the P2 deployment plan)

Three distinct, measured memory-failure classes on the 23 GB L4 (all with
XLA_PYTHON_CLIENT_PREALLOCATE=false, autotune 0):

| # | What | Number | Fix / consequence |
|---|---|---|---|
| a | MAMS mutation grad tape, v2d 80x80 ss2 diag marg, f64 | ~93 MB/particle irreducible (OOM 49.3/23.3/12.57 GB at N=512/256/128); remat_basis -5% only | N=64 is the L4 ceiling for this config |
| b | NEW-n_int pilot compile at stage 20 | pathological 10.23 GB temp request, IDENTICAL in a fresh process | `--xla_disable_hlo_passes=priority-fusion` (campaign's sanctioned knob) clears it |
| c | v3b correlated marg term (whitener_v3b, checkpointed whitening) | **367.6 MB/particle** (linear 8->16) | ~56 particles feasible on L4 => L0-G2 to Perlmutter; 128 particles ~ 47 GB => hbm80g pin (matches the T1.1 inj3 lesson independently) |

Wall-clock: L4 f64 stage cost at N=64 ranges 190-950 s with n_int (12-64);
a full prior-seeded anneal here is a ~6-10 h L4 job vs ~30-40 min on one
hbm80g A100. Recommendation for front C: if leg 3 has not delivered
COMPLETE by the next session, run the arbitration fresh on Perlmutter
(p128 fits an 80 GB card even without remat; ~2 A100-h est) and treat the
L4 particles as the cross-check.

## Ops chronicle (for the ledger)

- 22:09-22:21 sanity: S1-S4 PASS; N=512 smoke OOM -> registered halving.
- 22:21-23:12 N=256 and N=128 runs OOM (halving sequence exhausted).
- 22:50 ledgered remat deviation; 22:59 exactness gate PASS (0.0).
- 23:14 N=128+remat OOM (11.89 GB) -> pre-registered fallback N=64.
- 23:16-03:15 N=64 legs: task-harness kill at stage 4 (relaunched detached
  via nohup — stage rows reproduced bit-for-bit, deterministic seed 2);
  stages 0-19 clean; stage-20 pilot OOM (class b).
- 03:10 ledgered resume deviation; 03:16 resume leg 1 hit the same compile
  OOM; 03:19 resume leg 2 with priority-fusion disabled cleared it; stages
  20-26 clean.
- 04:45 leg 2 stopped at the stage-26 checkpoint (protocol cap reached in
  cumulative L4 wall); v3bsmoke run (class c measured); 04:50 completion
  leg 3 launched detached (12 h cap).

## Artifacts

- data/l0_sanity_report.json — S1-S4 + N-decision record
- data/l0_remat_gate.json — remat exactness gate (0.0)
- data/l0_v3b_memory_smoke.json — L0-G2 feasibility measurement
- data/l0_smc_checkpoints/stage_000..026.npz — per-stage ensembles (z, lam, eps, n_int)
- data/l0_run.log — full stage-row log (all legs)
- figs/l0_smc_traces_partial.png — sampler diagnostics through stage 26
- data/l0_arbitration_smc.{npz,json} — written by leg 3 at lambda=1 (pending)
- data/l0_gate_eval.json + figs/l0_gamma_hist.png — written by `harvest` (pending)
