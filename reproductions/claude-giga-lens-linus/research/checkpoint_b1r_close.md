# B1r CAROUSEL CELL CLOSE-OUT at PARTIAL + S6br budget-matched submission (D9)

Checkpoint of record for closing the P2b B1-REDUCED S1r arm at PARTIAL and
running the budget-matched S6br baseline. WRITTEN BEFORE the S6br sbatch and
BEFORE any CAMPAIGN.md edit (checkpoint-before-run; concurrency rule — the
B5-steep/arbitration front is active in this tree, CAMPAIGN.md edits happen
only after mtime quiescence). Amends nothing in the pre-registered design
`research/checkpoint_b1_reduced.md` (md5 865247b16e90061803c9f4cf67fa1c8f)
except the two D9-authorized items stated below.

## D9 — USER DECISION (2026-07-20): stop at PARTIAL, no leg 4

The amendment's own fence fired exactly as pre-registered: after leg 3 the
chain STOPS regardless of lambda, and a PARTIAL S1r makes S6br an
"orchestrator decision point" (no auto-burn against a failed primary arm).
The user resolved that decision point on 2026-07-20:

1. **The S1r cell CLOSES at PARTIAL. No leg 4.** The 12-h fence is final.
2. **S6br IS run, Track-A budget-matched at S1r's REALIZED total grad
   budget** — B\* = `grad_evals.total` accumulated by the PARTIAL run
   (the amendment's B\*-from-COMPLETE-json wording is extended to the
   PARTIAL's realized ledger BY USER DECISION, not silently). Cap 5 A100-h,
   wall 4:55, in-job fences unchanged (grad fence at B\* + 4.5-h graceful
   wall fence).
3. Mechanical consequence (ledgered here, comment in the script): the S6br
   in-script S1r-marker guard was written expecting the COMPLETE json
   `b1r128_carousel33_s1_seed2.json`; the realized marker is
   `b1r128_carousel33_s1_seed2.PARTIAL.json`. The guard test is adjusted to
   accept the PARTIAL json as the marker — a MECHANICAL amendment that
   changes no numerics, no fence, no threshold.

## S1r verdict: PARTIAL-BY-BUDGET at lambda = 0.1506

- Jobs 56170216 / 56170219 / 56170221 (legs 1-3, all COMPLETED exit 0 under
  the exit-3-PARTIAL in-job protocol; sacct -X): walls 3:50:10 + 3:43:34 +
  3:45:59 = **11.33 A100-h** (vs 12-h hard cap, 11.75 worst case).
- End state: **36 stages, lambda_reached = 0.150575548505617**, checkpoints
  contiguous (stage_000..035 + ll_000..035 + run_meta, 73 files, ckpt dir
  kept on CFS remote). Harvested artifacts:
  `data/results-perlmutter/b1r128_carousel33_s1_seed2.PARTIAL.json`,
  `..._run.log`, and the checkpoint-replay sidecar
  `..._partial_replay.json` (exact ckpt.py pre-resume replay semantics,
  identity fence PASS 36/36 replayed lambdas == checkpointed).
- **The pre-registered 12-h fence executed as designed.** This is the
  descoped cell's PARTIAL branch, anticipated in the amendment ("real
  PARTIAL risk if the anneal lands in the upper half of the band") — except
  the anneal was far SLOWER than the R5c-derived 10-12 h-to-lambda=1 anchor:
  12 h bought lambda = 0.15, not 1.
- **Falsifier: n/a — NOT ARMED.** The unique-particle < N/4 = 32 falsifier
  is defined at lambda >~ 0.8; lambda never exceeded 0.151. No falsifier
  claim in either direction.
- **Sampler health (figs/b1r_s1_partial_traces.png, plotted BEFORE this gate
  text):** healthy as far as it got. Unique particles after resample 92-104
  of 128 across all 36 stages (min 92, never within a factor 2.8 of the
  falsifier line); MAMS accept 0.871-0.934 (mean 0.899); ESS pinned at
  ~89.6 = 0.7 N by the adaptive lambda solver; eps stable 0.08-0.19 after
  stage ~3; n_int 32-64. No pathology — the run was killed by cost, not by
  sampler failure.
- **NO gate math on the non-lambda=1 ensemble** (l0 protocol): no logZ row
  (partial-history logZ is not an evidence estimate), no efficiency R, no
  cold-start z/width gates from S1r draws.

## Grad ledger (fills S6br)

From the per-stage checkpoints (grad_tune + grad_mutate, grads only —
weight-step logp evals excluded identically in both arms per the amendment):

- **B\* = grad_evals.total = 3,078,912** (tune 1,689,856 + mutate 1,389,056;
  36 stages, N=128, chunk 32/fwd 64).
- S6br is submitted with `GRAD_BUDGET=3078912`; predicted binding fence per
  the amendment arithmetic remains the 4.5-h wall (in that case the
  efficiency row is labeled "S6br at fence-truncated budget
  (realized/B\* quoted)" with the burn-dominated caveat).

## The cell's evaluable content — the COST ROW

**Prior-seeded MC-SMC on the 33-dim multi-plane carousel scene target
(REAL MUSE cutouts, N=128, frozen P0 protocol): ~11.33 A100-h bought
lambda = 0.15.** The lambda schedule is near-geometric (log-lambda roughly
linear in stage, ~0.076 dex/stage at ~19 min/stage late in the run);
reaching lambda = 1 needs ~0.82 more dex, i.e. on the order of another
10-11 stages of widening posterior mass PLUS whatever the hard
0.15 -> 1 tempering regime costs — extrapolation aside, the measured point
alone (1.3% of the anneal per A100-h by log-lambda in the LAST leg) puts
lambda = 1 far beyond ANY budget this campaign could authorize (>> the 18-h
P2b cap, >> the remaining global headroom). This corroborates, on real
data, the B3/B4/wave-1 finding: cold-start SMC cost on carousel-class
targets is not marginally over budget — it is structurally out of reach.

## Honest framing for the decision matrix (pre-stated, before S6br results)

At MATCHED budget on carousel-class targets: **prior-seeded SMC delivers NO
posterior samples (lambda = 0.15, no evaluable ensemble), while warm MAMS
(S6br) delivers whatever ESS it measures — any nonzero ESS makes this a
DECISIVE LOSS row for cold-start on this target class.** That is the
two-sided result the benchmark was designed to produce, and it is stated
here BEFORE reading S6br. The cold-start WINS recorded/possible on
B4/B5/DSPL-class targets stay where they are, pending those readouts —
this row narrows the claim to target class, it does not overturn the others.
S6br caveats carried from the amendment: at fence-truncated budget the
comparison is burn-dominated (bias TOWARD S1r on the rate metric, stated);
its posterior may itself be unconverged ("vs unconverged warm reference"
labeling if so); descoped gates (2-seed self-consistency, <=2-nat logZ
repeatability) stay descoped.

## S6br submission mechanics (this session)

- `slurm/p2b_b1r_s6b.slurm`: GRAD_BUDGET filled = 3078912; marker guard
  adjusted to accept the PARTIAL json (comment cites D9); hbm80g pin, wall
  4:55, shared QOS cosmo_g, -c 32 — all per template.
- deploy.md5 line updated BOTH sides (local + Perlmutter WORK copy),
  `md5sum -c` clean both sides before sbatch.
- Submitted-snapshot verification via `scontrol show job <id> --Batch`
  (grep GRAD_BUDGET + chunk) — the B5-steep script-snapshot-ordering lesson.
- Watchdog: leg-3 entry 56170221 deregistered (harvested; its COMPLETE-npz
  expect-artifact is unreachable by design at PARTIAL); S6br registered
  max_run 5.2 h, expect artifact CFS b1r128_carousel33_s6b_seed2.json,
  on_stall=alert.
- Budget: P2b actual 11.33 (S1r) + 5.0 S6br cap = **16.33 <= 18 cap**;
  no other P2b spend exists or is authorized. The 1.67-h residual is NOT
  spendable without a user decision (amendment rule carried).
- No results read beyond the S1r PARTIAL harvest above; S6br results are
  NOT read by this session.
