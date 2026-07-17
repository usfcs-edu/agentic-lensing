# P2 wave-1 post-mortem + redesign proposal (2026-07-16 harvest session)

PROPOSAL ONLY — nothing here is executed. No sbatch, no code changes applied,
no ledger est rows appended. Decisions flagged below belong to the
orchestrator (arm redesign inside caps) or the user (any cap change).
Companion gate/ledger record: CAMPAIGN.md (this session's stage-log entry).

## 1. What the wave produced (sacct 2026-07-16, single A100 shared QOS each)

| Job | Cell | State | Elapsed | A100-h | Artifacts |
|---|---|---|---|---|---|
| 55980038 | P2 deploy verify | COMPLETED | 00:02:22 | 0.04 | F8 parity report (gate row, PASS) |
| 55985444 | B1 S1 seed2 (mock) | TIMEOUT | 04:00:18 | 4.01 | NONE |
| 55985445 | B1 S1 seed3 (mock) | TIMEOUT | 04:00:23 | 4.01 | NONE |
| 55985446 | B1 S6b (mock) | TIMEOUT | 03:00:26 | 3.01 | NONE (run log: MAP done, round 0 only) |
| 55985447 | B2 orig+ratio | TIMEOUT | 02:30:33 | 2.51 | orig arm COMPLETE (npz+json); ratio arm LOST |
| 55985448 | B4 marg46 S1 N=256 | TIMEOUT | 03:30:27 | 3.51 | NONE (log: build+warmup only) |
| 55985449 | B5 S1 low+steep | FAILED | 01:01:12 | 1.02 | NONE (writer defect; see §3) |
| 55985450 | B5 S2 MCLMC | FAILED | 00:38:10 | 0.64 | NONE (same defect) |
| 55985451 | L0-G2 (P3) | COMPLETED | 00:31:37 | 0.53 | full; gate PASSED (orchestrator, commit 1993cc6) |
| 56004205 | l0arb (P3 front's cell) | FAILED | 00:00:24 | 0.01 | none — vendor-guard path-identity trip, see stage log |

P2-wave spend 18.69 h (+0.04 deploy = 18.73 of the 24-h P2 cap). Of the
18.69 h, exactly 1.12 h (the B2 orig arm) produced readable science — 94% of
the wave's GPU time yielded zero artifacts. That is REAL spend and is counted.

## 2. Root causes (three, independent)

**RC1 — no progress checkpointing/printing in the SMC runners.**
`23_run_p2_scene.py` and `24_run_p2_oldstack.py` call the frozen driver
`common.run_tempered_smc`, which prints nothing per stage and writes nothing
until λ=1. A TIMEOUT therefore destroys 100% of progress AND 100% of
observability (B1: 11.02 h, zero stages' evidence; B4: 3.51 h, cannot even
say how many stages ran). The cure already exists in-repo:
`10_anchor_arbitration.py`'s delegating `LoggingKernel` (per-stage z
checkpoint npz + progress print + pre-registered wall-cap raise) + its
`stage_resume` pattern. See §5a.

**RC2 — est-hours calibrated on the wrong target class.** The 2.5/2.0/1.5/3.0
est rows came from the old-stack v3b experience (28 λ-stages, p128, 46–74-dim
on a small cutout). The scene cells are a different class: 300×300 forwards,
N=512, carousel grad ≈ 837 MB/particle, and B2 — the CHEAPEST scene cell —
measured 64 λ-stages × 62 s at N=512 (1.11 h/arm). Measured carousel anchor
(s6b log): MAP 302 s, then round 0 ≈ 25 min (incl. compile) and <20 rounds in
the remaining 2.48 h ⇒ ≥7 min/round at 64 chains, i.e. the S6b design
(30+120 rounds) alone costs ≈ 17–20 h, and an S1 stage at N=512 (8× the
64-chain mutation) costs ≈ 30–45 min ⇒ ≈ 25–40 h per S1 carousel arm. The 4-h
walls bought ~5–8 early stages (λ ≲ 1e-4), i.e. <5% of the anneal. B4: 3.5 h
was a lower bound with zero observability. Proposed ops rule (§5c): first
submission on any NEW target class is a ≤2-h wall-capped CHECKPOINTED pilot
whose measured stage rate calibrates the production est row.

**RC3 — artifact-writer defect in `24_run_p2_oldstack.py` (Path B cells B4+B5).**
`out = dict(..., kernel=kern.name, ..., **_jsonable(res))` at line ~205, but
`run_tempered_smc` returns `kernel=kernel.name` in `res` and only
`particles` was popped ⇒ `TypeError: dict() got multiple values for keyword
argument 'kernel'` — AFTER the sampler returned. `23_run_p2_scene.py` pops
`res["kernel"]` (line 165), which is why B2-orig wrote fine. Consequences:
both B5 runs COMPLETED their science (the driver raises before returning if
λ<1; the crash site is downstream of the return, so λ=1 + bootstrap are
confirmed) and lost everything at the write. B4 sits on the SAME code path —
even inside an adequate wall it could not have written artifacts. Fix is one
line (`res.pop("kernel")` before the dict build); it must land before ANY
Path-B resubmission.

## 3. B5 diagnosis detail (INFRA-FAIL, not science failure)

- 55985449 (S1): built 68 s, warmup logp 38474.656 finite, low@128 MAMS ran
  ~55 min to the write line, crashed; `set -e` killed the job before the
  steep leg. 55985450 (S2 MCLMC): same crash at ~36 min.
- Both legs reached λ=1 (structural: the TypeError line is after the driver
  return, and the driver raises RuntimeError if λ<1). No OOM (hbm80g pin
  held), no import/data-path issue (md5 audit PASS, build clean, warmup
  finite). Purely the RC3 writer defect.
- Rerun cost after the one-line fix (measured, not guessed): S1 low ≈ 1.0 h +
  steep@96 ≈ 0.75 h ⇒ est 1.9 (wall 2:30); S2 MCLMC est 0.65 (wall 1:00).

## 4. B2 status after this harvest

- orig arm: gate evaluated (CAMPAIGN gate row): m̂ = 0.000 (0/512 particles
  below Om0=0.146; 95% rule-of-three bound < 0.006) vs |m̂−0.103| ≤ 0.045 ⇒
  OUTSIDE the band, FAIL AS WRITTEN — but the pre-registered falsifier needs
  the control to pass on the same data, so interpretation is PENDING-CONTROL:
  (i) sampler mode-death in the pathological coords, vs (ii) this fresh
  lenstronomy realization genuinely holds ~zero minor-arm mass (their 0.103
  was measured on THEIR unreproducible realization). SMC sanity clean.
- ratio control: build completed (+64 s) then 79 min of SMC lost to the wall
  (needs MORE than orig's 66.5 min — the tasking's "cheap, 1 h" is
  contradicted by the log). Rerun: ratio arm alone, est 2.3 (wall 3:00,
  ideally checkpointed per §5a). If the control reproduces Run A r2 ≈
  N(1.32417, 6.7e-4) AND shows minor-arm mass ≈ 0.103, the falsifier fires
  (honest negative row: the sampler does NOT fix the coordinate pathology);
  if the control also shows ~zero minor-arm mass, the 0.103 band is
  uncalibrated for this realization and the B2 gate must be re-registered
  against the control's own arm mass (ledgered amendment, not a silent move).

## 5. Redesign proposal

### (a) Stage-checkpoint/resume retrofit (23_run_p2_scene.py; mirror for 24)

Transplant the accepted `10_anchor_arbitration.py` pattern — NO driver edit
needed for the checkpoint half:
1. `LoggingKernel`-style delegating wrapper (build/tune/mutate verbatim):
   per-stage `stage_NNN.npz` (z ensemble, λ, eps, n_int) + one progress print
   (stage, λ, accept, t_tune, t_mutate, wall) + `WallCapReached` raised AFTER
   the checkpoint at a pre-registered `--wall-cap-h` inside the slurm wall
   (exit 3 + PARTIAL json, no gate math on a non-λ=1 ensemble — the l0
   protocol verbatim).
2. Weight-step recording via the EXISTING keyword-only `loglik_batch_fn`
   seam: wrap the (chunked or stock full-vmap) evaluator in a python-side
   recorder that stores each stage's ll vector into the checkpoint dir.
   Zero numerics impact (it only copies the output); this is what lets a
   RESUMED run keep the full logZ + per-stage bootstrap (the l0 resume's
   `logZ_partial_from_resume` limitation, solved) — required because logZ
   repeatability ≤2 nats is a B1 gate.
3. `--resume` stage in the runner cloning `10_anchor_arbitration.stage_resume`
   (start from newest checkpoint's (z, λ), splice pre-resume traces).
   Documented ledgered deviation, same class as the l0 one; RNG caveat
   stated: resumed runs are protocol-identical but not bit-identical to an
   uninterrupted run (fresh key/rng segment from the resume point) — same
   caveat the l0 resume carries.
   Validation before use: 54-test suite re-run + toy stock-vs-wrapped
   bit-identity (wrapper is delegation-only, so exact-zero is the expected
   result — prove it anyway, house pattern).
4. Same wrapper wired into `24_run_p2_oldstack.py` for B4 (identical driver).
   And RC3's one-line fix (`res.pop("kernel")`) lands first in any case.

### (b) Rerun matrix (honest walls/N; est = expected burn, wall = worst case)

| # | Arm | Design | Est h | Wall h | Notes |
|---|---|---|---|---|---|
| R1 | B5-S1 rerun | unchanged + RC3 fix | 1.9 | 2.5 | measured legs; central-bet linchpin (multimodality certificate) |
| R2 | B5-S2 MCLMC rerun | unchanged + RC3 fix | 0.65 | 1.0 | B5-G3 diagnostic |
| R3 | B2 ratio control (alone) | unchanged, own job, checkpointed if (a) lands | 2.3 | 3.0 | decides the B2 falsifier; "1 h" est is refuted by the 79-min partial |
| R4 | B4 @N=256 checkpointed | resumable 4-h legs until λ=1 | 6–10 (unknown tail) | 4/leg | true cost unmeasurable today; leg 1 doubles as the §5c pilot |
| R4' | B4 @N=128 checkpointed | half mutation cost | 3–5 | 6 | leaves the DECLARED N=256 design ⇒ pre-registered checkpoint amendment required (fallback-ladder precedent exists: L0's 512→256→128) |
| R4x | B4 two-stage-seeded | warm-start from the production two-stage recipe | ~2–3 | 4 | **OFF-PROTOCOL — answers a different question** (the cell's hypothesis is that prior-seeded SMC REPLACES the two-stage recipe). If wanted, pre-register as a NEW arm (B4b), never as "B4" |
| R5a | B1-real S1 ×2 @N=512 | as declared, wall 12 h | 25–40 EACH | 12/leg resumable | measured-anchor estimate (§2 RC2); 12 h ≈ half an arm — only sane WITH checkpointing, and still 50–80 h for the pair |
| R5b | B1-real S1 ×2 @N=256 | amendment (N) | 17–20 each | 12 + resume | still 34–40 h for the pair |
| R5c | B1-real S1 ×2 @N=128 | amendment (N; campaign-standard SMC count) | 10–12 each | 12 + resume | the smallest self-consistency-capable pair: ~20–24 h |
| R5d | B1-real S6b | as declared (150 rounds, N-independent: 64 chains) | 17–20 | 12 + resume | the efficiency-gate + cold-start baseline; unchanged by any S1 N choice |
| R5d' | B1-real S6b-lite | 30 burn + 32 sample rounds | 8–9 | 10 | amendment; ESS/grad gate then carries "shorter chain" caveat |

Mock B1 arms: DO NOT resubmit — real data landed (B1-REAL-DATA-LANDED
amendment); the mock's machinery-certification value is already partially
covered by B2-orig (same runner+driver at N=512, healthy traces, λ=1).

### (c) Budget arithmetic and what must be dropped

P2: 18.73 spent of 24 ⇒ **5.27 h headroom**. Campaign: 29.27 of 100 ⇒ 70.73
remaining (P3 has used 0.53 of its 17).

- Recovery lane R1+R2+R3: est 4.85 ≤ 5.27, worst-case walls 6.5 ⇒ a 1.2-h
  cap-breach RISK if everything hits its wall. Strictly-cap-safe variant:
  R1+R2 only (est 2.55, worst case 3.5), R3 deferred to the cap decision.
  B5 rerun risk is genuinely low (completion times MEASURED, walls 1.3–1.5×).
- **No B1-real or B4 arm fits inside the standing 24-h P2 cap** after the
  recovery lane — not at any N. Every row R4/R5 therefore requires an
  orchestrator/user decision: raise/reallocate the P2 cap (campaign-level
  affordability exists: 70.7 h free; e.g. the 10 h freed by D7/P4 retirement
  was returned to the pool, and PLAN's contingency column priced P2 at up to
  35), or drop B4 and/or B1-real from P2 (B1-real could ship later as its own
  capped mini-campaign once checkpointed runners exist).
- If a cap raise to PLAN's contingency 35 were granted (USER decision):
  recovery lane (4.85) + R4' B4@N=128 (4) + R5c S1-pair@N=128 (22) ≈ 31 h
  total P2 ≈ the contingency envelope, WITHOUT S6b — the B1 efficiency and
  cold-start gates would be reported NOT EVALUABLE (S6b unaffordable at
  ~17–20 h; S6b-lite would add 8–9 more and an amendment). Stated plainly:
  the full pre-registered B1 gate set is not reachable under ≤35 h P2 on
  this hardware at any N; the orchestrator must either amend the S6b design
  (pre-registered), fund ~50 h, or re-scope B1's claims to
  self-consistency + logZ repeatability + falsifier.

### (d) Ops rules proposed (ledger on adoption)

1. Pilot-first estimation: new target class ⇒ ≤2-h checkpointed pilot before
   any production est row (the pilot's stage rate is the estimate's source,
   quoted in the row).
2. Wall = est only for checkpointed jobs; for non-checkpointed jobs the LEDGER
   EST MUST EQUAL THE WALL (TIMEOUT burns the full wall — this wave's lesson,
   twice).
3. Runner artifact-write paths get a smoke test (the 54-test suite covers the
   driver, not the runners' summary assembly — RC3 lived exactly in that gap).
