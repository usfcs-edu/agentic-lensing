# B1-REDUCED (P2b) DESCOPE AMENDMENT — real-carousel budget-matched cell

Pre-registered design checkpoint (their-format: hypothesis + predicted
direction/magnitude + falsifier + derived threshold), WRITTEN BEFORE any
submission (2026-07-19). This AMENDS the B1 design checkpoint (CAMPAIGN.md
2026-07-16) by DESCOPE ONLY: **no gate threshold is moved**; arms are reduced
and every reduction is stated here, not silently dropped. Gate rows + A100-h
actuals fold into CAMPAIGN.md at submission/harvest per house rule (this
front makes no CAMPAIGN.md edit and runs no sbatch).

## Funding & fences (USER DECISION 2026-07-18 — the governing authority)

- **P2b sub-budget: cap 18 A100-h**, funded by reallocation: 10 h from the
  freed-P4 pool (ledger D7) + 8 h of the 20-h stretch pool. P2b sits OUTSIDE
  the exhausted P2 cap (24, spent 18.73 + 4.85 recovery est); the global
  100-h HARD STOP is UNCHANGED.
- **S1r hard cap 12 A100-h** — 3 chained legs, walls 3:55 each (3 x 3.917 =
  11.75 worst case <= 12 even on triple TIMEOUT). The chain STOPS after leg 3
  regardless of lambda: a non-lambda=1 end state is PARTIAL per the l0
  protocol (checkpoints kept, lambda_reached reported, NO gate math on a
  non-lambda=1 ensemble). **No fourth leg without a NEW user decision.**
- **S6br cap 5 A100-h** — wall 4:55 (worst 4.92), in-job graceful fence 4.5 h.
- **Worst-case P2b spend 11.75 + 4.92 = 16.67 <= 18; expected 15.5-16.5**
  (cost arithmetic below). The 1.3-h worst-case headroom is NOT spendable by
  any front without a user decision.

## Data declaration (supersedes the mock declaration of the original B1 row)

REAL carousel data: the team's MUSE cutouts at `newnewcutouts/`
(source4-5.fits md5 4a4f7c0216450e0bad49a7f1c9c3b4ea, source9.fits md5
220b5677ffb802f87c9b978a9380eca9; UNPUBLISHED team data — never committed,
D4/D6: results to the team first). Builder: `cgl2.zoo.build_carousel33`
real-data mode via `CGL2_CAROUSEL_DATA` (raise-never-default; loading
conventions verbatim from their build_model.py::dataset_from_dir). Smoke
record (2026-07-16, phoenix A16): real build 2.2 s, dim=32, 8 prior draws
finite, **adapter-vs-native log_prob parity max|d| = 0.0**. Every job echoes
the cutout md5s at start; run jsons carry them in provenance.

**SUPERSEDED FILES:** `slurm/p2_b1r_s1_seed2.slurm` (c98276fd...),
`p2_b1r_s1_seed3.slurm` (96a03c40...), `p2_b1r_s6b.slurm` (9b185606...) were
designed for the ORIGINAL full-scope arms (N=512 x 2 seeds + until-converged
S6b) and stay HELD FOREVER — do not submit them. The P2b cell is
`slurm/p2b_b1r_s1_leg{1,2,3}.slurm` + `slurm/p2b_b1r_s6b.slurm` only.

## Hypothesis (unchanged from B1)

Prior-seeded MC-SMC (MAMS mutation, frozen P0 protocol) is a structurally
useful sampler on the team's hardest ESS-limited real system (carousel
multi-plane lstsq class): it buys cold-start capability + logZ at an
efficiency not catastrophically below the classic warm-start MAMS baseline.

## Arms (the DESCOPE)

- **S1r** — MAMS-SMC prior-seeded, **N=128** (descope from the declared
  N=512; the pre-registered fallback-ladder precedent L0 512->256->128;
  campaign-standard SMC count), **seed 2 ONLY** (descope from {2,3}), REAL
  carousel data, mutation chunk 32 + weight-step chunk 64 (unchanged
  execution-order-only pins; bit-identity proven), frozen P0 protocol
  constants untouched (target_ess 0.7, 4 draws/stage, pilot 64, eps0 0.5,
  boot_seed 20260715+2, max_stages 400). Checkpoint/resume retrofit ON
  (cgl2/samplers/ckpt.py: per-stage full-state checkpoints, BIT-IDENTICAL
  resume incl. full logZ + bootstrap; 59-test suite green): leg 1 fresh with
  in-job wall cap 3.55 h (exit-3 PARTIAL protocol), legs 2-3 submitted
  `--dependency=afterany:<prev>` with `--resume` (no-op fast-exit when the
  COMPLETE marker — the final `<out>.json` — exists).
- **S6br** — MAMS-alone-warm baseline, **Track-A BUDGET-MATCHED** (replaces
  the until-converged 30+120-round design, whose measured true cost 17-20 h
  is unaffordable): S6br is given the SAME total gradient budget S1r
  ACTUALLY SPENT. **B\* := `grad_evals.total` from S1r's COMPLETE result
  json** (tune+mutate; grads only — the gate metric is per-grad; weight-step
  logp evals are ledgered separately and excluded from both arms
  identically). MAP warm-start cost is BILLED against B\* (pre-registered
  billing, unchanged): frozen MAP multistart 64x350 adabelief(1e-2), then 64
  chains, ensemble-preconditioned MAMS at lambda=1, **30 burn rounds frozen**
  (moving the burn protocol would change the baseline's tuning quality —
  not descoped), sampling rounds until a fence stops the loop at a round
  boundary: (i) grad fence at B\*, (ii) graceful in-job wall fence 4.5 h
  (both via the new `ckpt.round_fence_reason`, regression-locked; env seams
  `CGL2_S6B_GRAD_BUDGET` / `CGL2_S6B_WALL_CAP_H` / `CGL2_S6B_SAMPLE_ROUNDS`
  default OFF — unset env reproduces the frozen 30+120 design exactly).
  **S6br is submitted ONLY after S1r is COMPLETE** (its json defines B\*);
  if S1r ends PARTIAL at the 12-h cap, S6br stays HELD and the unspent P2b
  balance is reported to the orchestrator (no auto-burn against a failed
  primary arm).

Not run (unchanged bright lines): S4/S5 LAPS arms (PLAN §8.2), S7 flow-MAMS
(post-B1 as before), seed-3 S1 (descoped, above).

## Evaluable gates (threshold numbers VERBATIM from PLAN §6 B1 — none moved)

1. **Efficiency** — R = (ESS_est per 10^6 grads)_S1r / (same)_S6br, each
   arm's own realized grad ledger, ESS_est at harvest (tfp
   effective_sample_size on S6br draws; final-stage unique-particle
   convention for S1r, the B0/B3 convention): **WIN R >= 2, PARITY
   [0.7, 2), LOSS < 0.7 — evaluated AT MATCHED BUDGET**, where the matched
   budget is B\* fence-capped at 5 A100-h. Pre-declared arithmetic: B\*
   (~10-11 h of N=128 grads) exceeds what the 5-h fence buys at the measured
   >=7 min/round, so the wall fence is PREDICTED to truncate S6br at ~40-50%
   of B\*. Then the row is labeled "S6br at fence-truncated budget
   (realized/B\* quoted)" and carries the burn-dominated-rate caveat
   (MAP+burn are a larger billed fraction at truncated budget, which biases
   the rate comparison TOWARD S1r — stated, and quantified by the
   supplementary REPORT-ONLY sampling-segment rate ESS/grads_sampling; no
   threshold attaches to the supplement).
2. **Cold-start** — worst-param |z| < 3 AND width-ratio in [0.7, 1.4] for
   prior-seeded S1r vs the S6br posterior as warm reference. **Pre-stated
   caveat:** S6br at matched/truncated budget may itself be unconverged —
   in that case the gate is reported against whatever S6br achieved,
   labeled "vs unconverged warm reference", and is a caveated row, not a
   clean PASS/FAIL.
3. **Falsifier (unchanged)** — unique particles < N/4 = **32** at
   lambda >~ 0.8 (rotating-ridge geometry defeating affine per-lambda
   preconditioning). Evaluable from the per-stage checkpoints even on a
   PARTIAL S1r, provided lambda reached ~0.8.
4. **logZ** — reported WITH sigma_boot (full-history bootstrap survives
   resume by construction). **The <=2-nat repeatability gate is DESCOPED at
   1 seed — stated plainly, not silently dropped**; the sigma_boot
   understatement caveat on ill-conditioned targets (P0 record) is carried
   verbatim.
5. **Self-consistency (2 seeds, 0.27 sigma at ESS>=128) — DESCOPED, NOT
   EVALUABLE at 1 seed.** Stated plainly. Re-arming it requires a funded
   seed-3 arm (a NEW user decision, ~10-12 h more).
6. **SMC sanity (fail-loud, unchanged):** lambda=1 reached, finite logliks,
   unique particles >= N/4 at lambda=1.
7. **Comparison row vs the team's carousel numbers: 'summary-stats only'**
   (recorded 2026-07-16: their posterior arrays are globally gitignored and
   absent from the mirror; percentile/summary JSONs are the only readable
   product). Any draw-level comparison waits on a team transfer.

## Pre-registered honest prediction (direction + magnitude, unchanged)

Parity-to-3x: S1r does NOT win raw efficiency; its structural value is
logZ + cold-start. Added descope-specific predictions: (a) S1r@N=128
completes lambda=1 in 10-12 h (post-mortem R5c measured anchor) — i.e. it
NEEDS legs 2-3 and has real PARTIAL risk if the anneal lands in the upper
half of the band; (b) S6br inside the 4.5-h fence delivers MAP (~0.09 h) +
compile (~0.4 h) + 30 burn rounds (~3.5 h at the measured >=7 min/round) +
**only ~4-10 sampling rounds (~1000-2500 draws)** — the ESS_est will carry a
small-sample caveat; if sampling rounds < 8 the efficiency row is reported
with "burn-dominated" flagged prominently.

## Cost math (worst case <= 18 P2b)

| Job | Wall | In-job fence | Est (expected) | Worst case |
|---|---|---|---|---|
| S1r leg 1 (fresh) | 3:55 | ckpt cap 3.55 h | 3.7 | 3.92 |
| S1r leg 2 (afterany, --resume) | 3:55 | ckpt cap 3.55 h | 3.7 | 3.92 |
| S1r leg 3 (afterany, --resume) | 3:55 | ckpt cap 3.55 h | 3.7 (or fast-exit ~0.1 if COMPLETE earlier) | 3.92 |
| S6br (HELD template until B\* filled) | 4:55 | round fence 4.5 h + grad fence B\* | 4.7 | 4.92 |
| **Total** | | | **~15.8** | **16.67 <= 18** |

Ledger convention: legs are CHECKPOINTED, so est < wall is permitted (ops
rule 2); S6br's round loop is fence-protected (graceful artifact write at
4.5 h), so its wall binds only if the fence machinery itself fails. Est rows
are appended BY THE ORCHESTRATOR at submission, per house rule.

## Ops (standing rules, all mandatory)

`#SBATCH -C gpu&hbm80g` (SMC production pin) + shared QOS on cosmo_g +
`-c 32`; `md5sum -c deploy.md5` before GPU work in the executed copy;
PYTHONUNBUFFERED=1; cutout-md5 echo per job; hot I/O on $PSCRATCH (the ckpt
dir persists there across legs), results + ckpt dir + logs copied to CFS
`results/` (EXIT trap: even a hard TIMEOUT leaves the log + checkpoints
audit-trail on CFS); watchdog registration at submission (on_stall=alert —
never auto-resubmit: leg 1 would trip the ckpt no-mixing fence by design);
NO results read by the run session; harvest = figs FIRST then gates.
