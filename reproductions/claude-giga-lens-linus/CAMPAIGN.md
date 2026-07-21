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
| D8 | **B1-REDUCED FUNDED via P2b sub-budget, cap 18 A100-h** — user decision 2026-07-18: the budget-matched reduced B1 carousel cell (S1r = prior-seeded MAMS-SMC N=128 seed 2 on REAL carousel cutouts, 3 chained checkpointed legs; S6br = Track-A budget-matched MAMS-alone-warm baseline) is funded by reallocation: **10 h from the freed-P4 pool (D7) + 8 h of the 20-h stretch pool (stretch 20→12)**. P2b sits OUTSIDE the exhausted P2 cap (24, spent 22.05 + 0.8 est in flight); the global **100 A100-h HARD STOP is UNCHANGED**. Sub-fences: S1r hard cap 12 (3 × 3:55 walls — chain STOPS after leg 3 regardless of λ, NO leg 4 without a NEW user decision), S6br cap 5 (submitted ONLY after S1r COMPLETE fills B\* = grad_evals.total; no auto-burn against a failed primary arm); worst case 16.67 ≤ 18, and the 1.3-h headroom is NOT spendable without a user decision. Design = descope-only amendment `research/checkpoint_b1_reduced.md` (md5 865247b16e90061803c9f4cf67fa1c8f, pre-registered 2026-07-19 BEFORE submission): **no gate threshold moved**; 2-seed self-consistency + ≤2-nat logZ repeatability DESCOPED and stated plainly; original full-scope files p2_b1r_s1_seed{2,3}.slurm + p2_b1r_s6b.slurm HELD FOREVER, superseded by slurm/p2b_b1r_*.slurm | user, 2026-07-18; amendment 2026-07-19 |
| D9 | **B1r CAROUSEL CELL CLOSED AT PARTIAL; S6br RUNS BUDGET-MATCHED TO THE REALIZED LEDGER** — user decision 2026-07-20: (1) S1r STOPS at its PARTIAL end state (λ=0.1506, 36 stages, 11.33 A100-h) — the 12-h fence is final, **NO leg 4**; (2) S6br IS run, Track-A budget-matched at S1r's **REALIZED** total grad budget B\* = grad_evals.total = **3,078,912** (tune 1,689,856 + mutate 1,389,056, from the PARTIAL checkpoints via the identity-fenced replay sidecar — the amendment's B\*-from-COMPLETE-json wording is extended to the PARTIAL's realized ledger BY THIS DECISION, not silently), cap 5 A100-h / wall 4:55 / fences unchanged; (3) mechanical amendment: the S6br in-script S1r-marker guard accepts the PARTIAL json (comment in-script; no numerics/fence/threshold change). S1r verdict = **PARTIAL-BY-BUDGET** (falsifier n/a — λ never approached 0.8; sampler healthy: unique particles 92–104/128, accept 0.871–0.934); the cell's evaluable content = the **COST ROW** (prior-seeded MC-SMC on the 33-dim multi-plane real-carousel target: ~11.33 A100-h → λ=0.15 at N=128 ⇒ λ=1 infeasible under ANY campaign budget — corroborates B3/B4/wave-1 on real data) + the S6br budget-matched comparison. Decision-matrix framing pre-stated BEFORE S6br results: at matched budget on carousel-class targets prior-seeded SMC delivers NO posterior samples while warm MAMS delivers its measured ESS — a decisive LOSS row for cold-start on THIS target class (the two-sided result the benchmark was designed to produce); cold-start WINS stay on B4/B5/DSPL-class targets pending those readouts. Close-out checkpoint of record: `research/checkpoint_b1r_close.md` | user, 2026-07-20; close-out checkpoint 2026-07-20 |

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
| 2026-07-16 | 55980038 cgl2-deploy-verify (P2 Path-A deploy verify: md5 audit + 00_env_check + 01_parity_scene NATIVE gate battery under the new cgl2-pm venv, slurm/deploy_f8_verify.slurm; plain `-C gpu`, 30-min cap, shared 1×GPU) | P2 deploy | 0.2 | 0.04 (COMPLETED 00:02:22; F8 PASS) | 10.05 actual |
| 2026-07-16 | 55985444 cgl2-b1-s1-s2 (B1 carousel33 S1 prior-seeded MAMS-SMC N=512 seed 2, mock stand-in DECLARED, chunk 32/fwd 64, slurm/p2_b1_s1_seed2.slurm) | P2 B1 | 2.5 | 4.01 (**TIMEOUT** 04:00:18 — ZERO artifacts; ~5–8 of ~60+ stages, post-mortem below) | 14.06 actual |
| 2026-07-16 | 55985445 cgl2-b1-s1-s3 (B1 carousel33 S1 seed 3, ditto, slurm/p2_b1_s1_seed3.slurm) | P2 B1 | 2.5 | 4.01 (**TIMEOUT** 04:00:23 — ZERO artifacts) | 18.06 actual |
| 2026-07-16 | 55985446 cgl2-b1-s6b (B1 carousel33 S6b MAP-multistart-billed + ensemble-preconditioned MAMS-alone λ=1, seed 2, slurm/p2_b1_s6b.slurm) | P2 B1 | 2.0 | 3.01 (**TIMEOUT** 03:00:26 — ZERO artifacts; MAP done 302 s, <20 of 150 MAMS rounds) | 21.07 actual |
| 2026-07-16 | 55985447 cgl2-b2-dspl (B2 dspl20_orig + dspl20_ratio control, S1 N=512 seed 2 each, one job, slurm/p2_b2_dspl.slurm) | P2 B2 | 1.5 | 2.51 (**TIMEOUT** 02:30:33 — orig arm COMPLETED+wrote at +68 min; ratio control arm lost at 79 min mid-SMC) | 23.58 actual |
| 2026-07-16 | 55985448 cgl2-b4-t2 (B4 T2 foundry_marg46 prior-seeded S1 N=256 f64 seed 2, Path B, slurm/p2_b4_t2.slurm) | P2 B4 | 3.0 | 3.51 (**TIMEOUT** 03:30:27 — ZERO artifacts; log shows build+warmup only, zero stage observability) | 27.09 actual |
| 2026-07-16 | 55985449 cgl2-b5-s1 (B5 T3 foundry_v3b74 per-basin S1 MAMS low@128 + steep@96, f32, seed 2, Path B, slurm/p2_b5_s1.slurm) | P2 B5 | 3.0 | 1.02 (**FAILED** 01:01:12 — INFRA-FAIL: low leg reached λ=1 then lost everything to the 24_run_p2_oldstack.py writer defect; steep leg never ran) | 28.11 actual |
| 2026-07-16 | 55985450 cgl2-b5-s2 (B5-G3 T3 low@128 S2 MCLMC-mutation diagnostic arm, f32, seed 2, Path B, slurm/p2_b5_s2_mclmc.slurm) | P2 B5 | 1.5 | 0.64 (**FAILED** 00:38:10 — same writer defect, λ=1 reached, artifacts lost) | 28.74 actual |
| 2026-07-16 | 55985451 cgl2-l0g2 (P3 L0-G2 scene-API v3b CORRELATED per-basin refit low@128 + steep@96, production-recipe mirror, seed 2, slurm/p3_l0g2.slurm) | P3 L0 | 2.0 | 0.53 (COMPLETED 00:31:37; harvested + gate PASSED by orchestrator, commit 1993cc6) | 29.27 actual |
| 2026-07-16 | 56004205 cgl2-l0arb (P3-L0 anchor arbitration FRESH Perlmutter run per the pre-registered fallback: `10_anchor_arbitration.py run` UNCHANGED, v2d diagonal scene-API target, MC-SMC MAMS prior-seeded, seed 2, N=128 via the documented L0_N_OVERRIDE fallback ladder; wall-cap 3.5 h inside -t 04:00; plain `-C gpu` on purpose (~93 MB/particle grad ⇒ ~12 GB — don't burn hbm80g); slurm/p3_l0arb.slurm) | P3 L0 | 2.5 | 0.01 (**FAILED** 00:00:24 — require_vendor_ref path-identity trip in the $PSCRATCH-copy execution pattern; ops note in the 2026-07-16 harvest stage-log entry, diagnosis/resubmit owed by the P3 front) | 29.27 actual + resubmit est TBD (P3) |
| 2026-07-16 | **P2 WAVE-1 ACTUALS HARVESTED (sacct -X, Elapsed × 1 A100 each):** deploy 0.04 + wave jobs 4.01+4.01+3.01+2.51+3.51+1.02+0.64 = **18.73 P2-phase actual of the 24 cap (headroom 5.27)**; of the wave's 18.69 h, only the B2-orig arm's 1.12 h produced readable science (94% zero-artifact burn — REAL spend, counted). Campaign total **29.27 actual of 100** (P1 2.21 + T1.1 7.80 + P2 18.73 + P3 0.53 l0g2 + 0.01 l0arb); ~70.7 h remain campaign-wide. Post-mortem + costed rerun matrix: research/p2_wave1_postmortem_redesign.md (NOTHING resubmitted this session) | P2 | — | 18.73 P2 total | **29.27 actual** of 100 |
| 2026-07-16 | 56006048 cgl2-l0arb **RESUBMIT of 56004205** — protocol UNCHANGED (`10_anchor_arbitration.py run`, v2d diag scene-API, MC-SMC MAMS prior-seeded, seed 2, N=128 via L0_N_OVERRIDE, L0_WALL_CAP_H=3.5 in -t 04:00, plain `-C gpu` on purpose); FIX = JOB-ENV ONLY: PYTHONPATH points `import gigalens` at the audited $PSCRATCH copy's vendor (the venv's gigalens.pth targets the CFS tree — the path-identity trip; guard UNCHANGED, preflight-proven PASS in a real copy); slurm/p3_l0arb.slurm | P3 L0 | 2.5 | 0.17 (**FAILED** 00:10:27 — PYTHONPATH fix WORKED, guard passed, gigalens resolved to the $PSCRATCH copy; NEW failure = GPU OOM at the first mutate: single 21.04 GiB alloc, RESOURCE_EXHAUSTED on the plain-gpu 40 GB A100. The "~93 MB/particle ⇒ 12 GB" sizing was an L4 measurement taken WITH L0_REMAT; no-remat A100 tape refutes it. Diagnosis + hbm80g resubmit below, 2026-07-19 harvest) | 29.44 actual |
| 2026-07-16 | 56006049 cgl2-b5-s1 **B5-S1 RERUN of 55985449** (recovery lane R1): science UNCHANGED (T3 v3b74 per-basin S1 MAMS low@128 + steep@96, f32, seed 2, Path B); RC3 writer fix + RC1 checkpoint retrofit ENABLED (per-leg wall caps 1.30/0.85 h, stage ckpts + PARTIAL/exit-3 + bit-identical resume); wall 2:30 vs measured legs ~1.0+0.75; slurm/p2_b5_s1.slurm | P2 B5 | 1.9 | 0.92 (**FAILED 00:55:01 — LOW LEG COMPLETED AND WROTE** b5_v3b74_low_mams_seed2.{npz,json}+ckpt to CFS at +49 min (λ=1 @31 stages, logZ 38376.33±0.33 — the RC3 fix works); STEEP leg died at its FIRST mutate on `AssertionError: (96, 64)` at 22_run_b3.py:135 — chunked MAMS requires n % chunk == 0 and steep is N=96 with --chunk 64. LATENT bug: original 55985449's set -e died on the low-leg writer before steep ever executed, so this assert had never run. Steep-only resubmit below, 2026-07-19 harvest) | 30.36 actual (56006048's 0.17 landed first) |
| 2026-07-16 | 56006052 cgl2-b5-s2 **B5-S2 MCLMC RERUN of 55985450** (recovery lane R2): science UNCHANGED (B5-G3 low@128 MCLMC diagnostic, f32, seed 2); RC3 fix + retrofit ENABLED (wall cap 1.15 h); wall 1:30 vs measured ~0.65 completion; slurm/p2_b5_s2_mclmc.slurm | P2 B5 | 0.65 | 0.58 (COMPLETED 00:34:51 — λ=1 @36 stages, wrote b5_v3b74_low_mclmc_seed2.{npz,json}+ckpt; logZ 38380.36±0.36; harvested 2026-07-19, B5-G3 evaluation in the gate row) | 30.94 actual |
| 2026-07-16 | 56006065 cgl2-b2-ratio **B2 ratio-control STANDALONE** (recovery lane R3, decides the B2 falsifier): dspl20_ratio S1 N=512 seed 2 chunk 128, design of the arm lost in 55985447; honest wall 3:00 per the post-mortem (the "1 h" est is REFUTED by the 79-min partial); retrofit ENABLED (wall cap 2.70 h); **LAST-submitted job — if recovery-lane actuals threaten the 24-h P2 cap, THIS job's overflow is the flagged breach (never absorbed silently)**; slurm/p2_b2_ratio.slurm | P2 B2 | 2.3 | 1.82 (COMPLETED 01:49:02 — λ=1 @64 stages, wrote b2_dspl20_ratio_s1_seed2.{npz,json}+ckpt; logZ 17290.95±0.24; NO cap breach; harvested 2026-07-19, falsifier decision in the B2 gate row) | 32.76 actual |
| 2026-07-19 | **RECOVERY-LANE ACTUALS HARVESTED (sacct -X, Elapsed × 1 A100 each):** 56006049 0.92 + 56006052 0.58 + 56006065 1.82 = **3.32 P2 (vs est 4.85) ⇒ P2 total 22.05 of 24 (headroom 1.95)**; 56006048 0.17 P3 ⇒ **P3 total 0.71 of 17** (0.53 l0g2 + 0.01 + 0.17 l0arb). Campaign total **32.76 actual of 100**. 3 of 4 recovery jobs produced full readable artifacts (the checkpoint retrofit + RC3 fix did their job); the 2 failures (l0arb OOM, B5-steep chunk-assert) are diagnosed with in-protocol fixes + resubmits below | P2+P3 | — | 3.49 recovery total | **32.76 actual** of 100 |
| 2026-07-19 | 56168443 cgl2-b5-s1st **B5-S1 STEEP-ONLY resubmit** (low leg NOT rerun — its artifacts are on CFS): science UNCHANGED (T3 v3b74 steep@96 S1 MAMS, f32, seed 2, Path B, ckpt retrofit, wall cap 0.95); FIX = --chunk 48 (divides 96; 56006049's steep leg died on the latent `assert n % chunk == 0` with N=96/chunk 64 — chunking is EXECUTION ORDER ONLY per the runner's own help + per-particle keys split before chunking, "== parent keys", bit-identity vs stock verified in the deployment plan ⇒ design unchanged); stale ckpt dir wiped in-job (held only run_meta + ll_000, no stage ckpts); wall 1:15; slurm/p2_b5_s1_steep.slurm | P2 B5 | 0.8 | 0.00 (**FAILED** 00:00:12 — ORIGINAL (96, 64) assert signature: sbatch snapshots the script at submission and this job was submitted BEFORE the chunk-48 edit landed in the CFS file, so the fix never rode along; script-snapshot root cause + ops lesson in the 2026-07-19 late stage entry; actual filled 2026-07-20) | 32.76 actual + 0.8 P2 est (22.05+0.8 = 22.85 ≤ 24; worst-case wall 23.30 ≤ 24 — SAFE, no breach flag needed) |
| 2026-07-19 | 56168446 cgl2-l0arb **RESUBMIT of 56006048** — protocol UNCHANGED (`10_anchor_arbitration.py run`, v2d diag scene-API, MC-SMC MAMS prior-seeded, seed 2, N=128, L0_WALL_CAP_H=3.5 in -t 04:00, L0_REMAT stays UNSET); FIX = HARDWARE ONLY: `-C gpu&hbm80g` (56006048 REFUTED the plain-gpu sizing by measurement — first mutate requested a single 21.04 GiB alloc and OOM'd the 40 GB card; the "~93 MB/particle ⇒ 12 GB" figure was an L4 measurement taken WITH the L0_REMAT deviation active, which this run pre-registeredly omits). hbm80g chosen over enabling L0_REMAT as the smaller deviation (pure hardware, zero numerics/graph change; restores the standing SMC pin); slurm/p3_l0arb.slurm | P3 L0 | 2.5 | 3.58 (**PARTIAL_WALL_CAP** — sacct FAILED 03:34:46, ExitCode 3:0 = the pre-registered exit-3 wall-cap protocol, NOT a crash; the hbm80g fix WORKED: guard passed, no OOM (peak 22.7 GB), MAMS healthy (accept ~0.88), capped at stage 76 **λ=0.587** of the 3.5-h in-job cap; json + run.log + ckpts stage_000–075 on CFS results/l0arb/; λ-reached readout is ops telemetry per the wall-cap protocol — NO gate math on a non-λ=1 ensemble; third consecutive wall-capped MC-SMC attempt ⇒ vehicle amendment 2026-07-20 in research/checkpoints_l0.md; actual filled 2026-07-20) | 32.76 actual + 0.8 P2 est + 2.5 P3 est (P3 0.71+2.5 = 3.21 ≤ 17) |
| 2026-07-19 | 56170216 cgl2-p2b-b1r-s1-L1 (**P2b B1-REDUCED S1r leg 1/3**, D8 + amendment research/checkpoint_b1_reduced.md: carousel33 **REAL MUSE cutouts** — md5s 4a4f7c02…/220b5677… echoed in-job, D4/D6 — prior-seeded MAMS-SMC **N=128 seed 2**, chunk 32/fwd 64, frozen P0 protocol; ckpt retrofit ON: in-job wall cap 3.55 h → exit-3 PARTIAL, per-stage ckpts on $PSCRATCH → CFS via EXIT trap; slurm/p2b_b1r_s1_leg1.slurm) | P2b B1r | 4.0 | 3.84 (COMPLETED 03:50:10, fresh leg, exit-3→PARTIAL protocol in-job; actual filled 2026-07-20 D9 harvest) | 32.76 actual + 0.8 P2 est + 2.5 P3 est + **4.0 P2b est** |
| 2026-07-19 | 56170219 cgl2-p2b-b1r-s1-L2 (S1r leg 2/3, `--dependency=afterany:56170216`, `--resume` from newest stage ckpt — BIT-IDENTICAL incl. full logZ+bootstrap; NO-OP fast-exit if COMPLETE marker b1r128_carousel33_s1_seed2.json exists; slurm/p2b_b1r_s1_leg2.slurm) | P2b B1r | 4.0 (conditional-resume; ~0.1 if fast-exit) | 3.73 (COMPLETED 03:43:34, resumed leg; actual filled 2026-07-20 D9 harvest) | … + **8.0 P2b est** |
| 2026-07-19 | 56170221 cgl2-p2b-b1r-s1-L3 (S1r leg 3/3, `--dependency=afterany:56170219`, `--resume`; after this leg the chain **STOPS regardless of λ** — S1r hard cap 12 = 3 × 3:55, worst 11.75; a non-λ=1 end state is PARTIAL per the l0 protocol, NO leg 4 without a NEW user decision; slurm/p2b_b1r_s1_leg3.slurm) | P2b B1r | 4.0 (conditional-resume; ~0.1 if fast-exit) | 3.77 (COMPLETED 03:45:59 — resumed from stage 23 λ=0.0200, ended stage 35 **λ=0.1506 PARTIAL-BY-BUDGET**; chain STOPPED per the D8 fence; D9 close-out + harvest rows below; actual filled 2026-07-20) | … + **12.0 P2b est ≤ 18 cap** |
| 2026-07-19 | S6br **TEMPLATE-HELD — NOT SUBMITTED** (slurm/p2b_b1r_s6b.slurm, GRAD_BUDGET unfilled by design; in-script guards refuse unfilled template or missing S1r COMPLETE marker; the S1r HARVEST fills B\* = grad_evals.total then updates deploy.md5 both sides before any sbatch; NOT submitted if S1r ends PARTIAL — orchestrator decision point) | P2b B1r | 5.0 (reserved, HELD) | — (no job exists) | **P2b committed est 17.0 ≤ 18 cap** (worst case 16.67); campaign 32.76 actual + 20.3 est in flight ≤ 100 |
| 2026-07-19 | 56170614 cgl2-b5-s1st **B5-steep resubmit #3** — **RETRO-RECORDED TABLE ROW (deviation, declared):** the est 0.8 + cap math (worst-case P2 23.65 ≤ 24) were declared in the 2026-07-19 late stage entry BEFORE sbatch, but the previous wave omitted the A100-ledger table row; row added 2026-07-20 at actuals fill. Science identical to 56168443 with the chunk-48 file verified on CFS pre-submission | P2 B5 | 0.8 | 0.03 (**FAILED** 00:01:52 — `AssertionError: (64, 48)` at 22_run_b3.py:135: chunk 48 divides the FULL pass (96 % 48 = 0) but NOT the 64-particle dual-averaging PILOT subset (common.run_tempered_smc pilot_size=64, frozen at 24_run_p2_oldstack.py:215) which mutates through the SAME chunked kernel; the divisibility constraint is chunk \| N AND chunk \| pilot_size; #4 fix = chunk 32, row below) | 32.76 actual + in-flight ests |
| 2026-07-20 | **FIX-WAVE ACTUALS HARVESTED (sacct -X, Elapsed × 1 A100 each):** 56168443 0.00 + 56170614 0.03 ⇒ **P2 total 22.08 of 24 (headroom 1.92)**; 56168446 3.58 ⇒ **P3 total 4.29 of 17** (l0g2 0.53 + l0arb 0.01 + 0.17 + 3.58). Campaign **36.37 actual of 100**. NOT harvested here (B1r front's lane, per tasking): P2b legs 56170216/219/221 actuals (chain has ENDED — leg 3 COMPLETED_NO_ARTIFACT watchdog alert left ACTIVE for the B1r front; est 12.0 committed + 5.0 S6br held stand) | P2+P3 | — | 3.61 fix-wave total | **36.37 actual** of 100 |
| 2026-07-20 | 56251555 cgl2-b5-s1st **B5-S1 STEEP-ONLY attempt #4** (science UNCHANGED: T3 v3b74 steep@96 S1 MAMS, f32, seed 2, Path B, ckpt retrofit, wall cap 0.95 in -t 01:15): FIX = **--chunk 32** — chunk must divide BOTH N=96 (full pass, 3 chunks) AND pilot_size=64 (tuning pilot, 2 chunks); 32 ≤ 48 ≤ 64 so memory is even safer; chunking remains execution-order-only (per-particle keys split before chunking, bit-identity vs stock verified in the deployment plan). Submitted-SNAPSHOT verified (the 07-19 ops lesson; `--Batch` prints nothing on this slurm, so `scontrol write batch_script 56251555 -` was used): snapshot md5 **51c7ae1bca9c294c50d00d3cf593c774 = local = CFS**, carries `--chunk 32` (line 64 command + line 29 constraint comment); remote deploy self-check PASS 223/223 pre-sbatch; stale ckpt dir wiped in-job | P2 B5 | 0.8 | 0.37 (COMPLETED 00:22:20 — chunk-32 fix WORKED: λ=1 @22 stages, wrote b5_v3b74_steep_mams_seed2.{npz,json}+ckpt to CFS; logZ 38518.23 ± 0.31_boot, γ_eqw med 2.5552, retention 1.000; harvested 2026-07-21, B5 gate row updated) | 36.37 actual + 0.8 P2 est (P2 22.08+0.8 = 22.88 ≤ 24; **worst-case wall 22.08+1.25 = 23.33 ≤ 24 — SAFE, no breach flag needed**) |
| 2026-07-20 | **P2b S1r ACTUALS HARVESTED (sacct -X, Elapsed × 1 A100 each): 56170216 3.84 + 56170219 3.73 + 56170221 3.77 = 11.33 P2b actual (vs est 12.0; hard cap 12 respected)** — chain ended **PARTIAL-BY-BUDGET at λ=0.150576, 36 stages** (D9: cell CLOSED, no leg 4). Harvest: PARTIAL json + run.log + checkpoint-replay sidecar (identity fence PASS 36/36; **grad_evals.total = 3,078,912** = tune 1,689,856 + mutate 1,389,056) in data/results-perlmutter/b1r128_carousel33_s1_seed2*, traces figs/b1r_s1_partial_traces.png (plotted BEFORE gate text per house rule); ckpt dir stays on CFS remote (73 files, stages 000–035 contiguous). Sampler healthy to λ=0.15 (uniq 92–104/128, accept 0.871–0.934, ESS pinned 0.7N); NO gate math on the non-λ=1 ensemble. Leg-3 COMPLETED_NO_ARTIFACT watchdog alert = resolved-by-design (PARTIAL end state), entry deregistered. Close-out: research/checkpoint_b1r_close.md | P2b B1r | — | 11.33 P2b total | **47.70 actual** of 100 (36.37 + 11.33 P2b) |
| 2026-07-20 | 56252401 cgl2-p2b-b1r-s6b (**P2b S6br Track-A BUDGET-MATCHED MAMS-alone-warm baseline**, D9: `GRAD_BUDGET=3078912` = S1r REALIZED grad ledger; marker guard accepts the PARTIAL json — mechanical amendment, in-script comment; frozen MAP 64×350 billed + 30 burn rounds frozen + sampling to the grad-fence-at-B\* OR 4.5-h graceful wall fence; hbm80g pin, wall 4:55, shared cosmo_g, -c 32; deploy.md5 updated BOTH sides + full remote audit PASS pre-sbatch; **submitted-SNAPSHOT verified**: `scontrol write batch_script 56252401 -` md5 **c65d3a719cb68c55acabaa9151f19a7f = local = CFS**, carries GRAD_BUDGET=3078912 (line 47) + --chunk 32 (line 73); watchdog registered max_run 5.2 h, expect CFS b1r128_carousel33_s6b_seed2.json, on_stall=alert; slurm/p2b_b1r_s6b.slurm) | P2b B1r | 5.0 | 4.52 (COMPLETED 04:31:25 — the pre-registered 4.5-h WALL fence fired at burn round 11 of 30, `wall_cap: 16204s > 16200s`; **SAMPLING NEVER STARTED**: sample_rounds_run=0, draws (0,64,32), grads realized 1,277,632 = 41.5% of B\*=3,078,912 (map 22,400 + tune 894,080 + mutate 361,152); MAP healthy (302 s, best logpost −309,600.96), MAMS burn healthy (accept 0.873–0.902, eps ~9e-4, n_int 64) — killed by cost, not pathology; harvested 2026-07-21, decision-matrix row in gate record + data/b1r_decision_matrix.json, figs/b1r_s6br_traces.png plotted FIRST) | **P2b 11.33 actual + 5.0 est = 16.33 ≤ 18 cap** (worst-case wall 11.33 + 4.92 = 16.25); campaign 47.70 actual + 5.8 est in flight (0.8 B5-steep#4 + 5.0 S6br) = 53.50 ≤ 100 |
| 2026-07-20 | 56252932 cgl2-l0arbc **P3-L0 anchor arbitration — CLASSIC-RECIPE ARM (PRIMARY vehicle per the 2026-07-20 protocol amendment, research/checkpoints_l0.md, appended BEFORE this run):** 3 MC-SMC attempts wall-capped below λ=1 (L4 λ≈0.45 @~18 h, ckpts stage_000–067; A100 hbm80g 56168446 λ=0.587 @3.5 h, ckpts stage_000–075 — healthy sampler, budget-infeasible VEHICLE), so pre-registered optional arm (b) is PROMOTED: classic scene-API MAP→SVI→HMC (B3 reference-arm replication, frozen settings: MAP adabelief(1e-2)×350 warm single-start; SVI 250×500 w/ OOM ladder 250→128→64; HMC 50 chains 250/750 + single 500/1500 escalation, R̂<1.05/ESS≥200; seeds 0/1/2), WARM-started from the mapped foundry-i MAP (param_map of refs v2d:z_ref:x46, the S1-certified path) on the SAME v2d diagonal parity-certified target — the SAME algorithm class that produced the 1.433 anchor on the old stack. **Gates and bands UNCHANGED; MC-SMC partials retained as SECONDARY evidence, PARTIAL status recorded; trade-off declared: the warm arm inherits the warm-start basin (as the old-stack anchor fit did) — prior-seeded alternative-basin coverage was the SMC arm's job and REMAINS OPEN.** Vehicle 11_arbitration_classic.py (LOCAL CPU TOY TEST PASS — driver-only synthetic target, exercised escalation + blocked-by-reference paths, lensing likelihood never run on CPU; stage-wise artifact writes so a walltime kill loses ≤1 stage; outputs γ posterior + per-param R̂/ESS worst-param; NO in-job gate math) via slurm/p3_l0arb_classic.slurm — plain `-C gpu` ON PURPOSE (56006048's 21-GiB OOM was the N=128 MAMS mutate tape; this arm's tapes: MAP n=1, SVI ladder, HMC ~4.7 GB), wall 2:00, $PSCRATCH audited-copy + PYTHONPATH vendor pin; staged md5 3f4787fc (script) / 2a042788 (slurm) identical local/CFS, deploy.md5 → **225 files** (merged with the concurrent B1r front's s6b line c65d3a71 after a cross-front clobber was caught and repaired — stage entry), remote self-check PASS 225/225 pre-sbatch | P3 L0 | 1.5 | 0.02 (**FAILED** 00:01:21 — `IndexError: 0-dimensional array indexed with 1 regular index` at `_map_warm_refine.one_step` (`chisq[i]`), AFTER the warm-start fence passed (log_prior=−851.509, log_like=−62566.298, γ_warm=1.8655 = z_ref); root cause = cgl2 CorrelatedImageData's bs==1 squeeze (correlated.py `return ll[0], chi2[0]`) makes chisq 0-d for THIS arm's single-row warm batch while lp re-broadcasts to (1,) via the batched log_prior — the toy test missed it because ToyPM returns batched chisq, and B3 never hit it because its multi-start MAP is n>1; REPRODUCED LOCALLY on phoenix L4 GPU 9 against the real v2d target at tiny budgets (identical signature), minimal fix + documented `--smoke` real-path mode added, local smoke CLEAN EXIT 0; resubmit = attempt #2 row below, 2026-07-21 stage entry) | 47.70 actual + **1.5 P3 est** (P3 4.29+1.5 = 5.79 ≤ 17; worst-case wall 4.29+2.0 = 6.29 ≤ 17); campaign 47.70 + 7.3 est in flight (0.8 B5#4 + 5.0 S6br + 1.5 l0arbc) = 55.00 ≤ 100 |
| 2026-07-21 | **FINAL-WAVE ACTUALS HARVESTED (sacct -X, Elapsed × 1 A100 each):** 56251555 0.37 (B5-steep #4 COMPLETED) + 56252401 4.52 (S6br COMPLETED, wall-fence-truncated) + 56252932 0.02 (l0arbc FAILED, fixed+resubmitted below) ⇒ **P2 total 22.45 of 24 (headroom 1.55)**; **P2b total 15.85 of 18 cap** (11.33 S1r + 4.52 S6br — sub-fences respected, cell now fully closed); **P3 total 4.31 of 17**. Campaign **52.61 actual of 100** | P2+P2b+P3 | — | 4.91 wave total | **52.61 actual** of 100 |
| 2026-07-21 | 56267678 cgl2-l0arbc **CLASSIC-ARM RESUBMIT of 56252932 (attempt #2)** — science/protocol/settings UNCHANGED (warm MAP→SVI→HMC, B3 frozen settings, v2d diagonal parity-certified target, slurm/p3_l0arb_classic.slurm byte-identical md5 2a042788, plain `-C gpu`, wall 2:00). FIX = shape normalization only in `_map_warm_refine.loss`: `jnp.atleast_1d(lp), jnp.atleast_1d(chisq)` — identity for every batched caller (B3 n>1 unchanged), unsqueezes the cgl2 CorrelatedImageData bs==1 0-d chisq that killed #1 at `chisq[i]`. RULE FOLLOWED: crash REPRODUCED LOCALLY FIRST (phoenix L4 GPU 9, real v2d target, identical IndexError signature), then fixed, then the new documented `--smoke` real-path mode (tiny budgets MAP 5 / SVI 16×50 / HMC 4×5/5; exercises escalation + blocked-by-reference; writes *_smoke stems only) ran to CLEAN EXIT 0 with warm-start values bit-matching the Perlmutter log (lp −851.509 / ll −62566.298 / γ 1.8655). Staged: script md5 **52e8df0378a4bbb60d679c089657327d** identical local/CFS; deploy.md5 UNION-MERGED per the 07-20 ops rule (remote pulled first — only divergence was our own line; 225 files, manifest md5 120ac030), remote self-check **PASS 225/225** pre-sbatch; submitted-SNAPSHOT verified `scontrol write batch_script 56267678 -` md5 = 2a042788 = local = CFS; watchdog: 56252932/56251555/56252401 deregistered (terminal+harvested), 56267678 registered (max_run 2.5, expect CFS l0arb/l0_arbitration_classic.npz, on_stall=alert) | P3 L0 | 1.5 | — (submitted; row appended BEFORE results) | 52.61 actual + **1.5 P3 est** (P3 4.31+1.5 = 5.81 ≤ 17; worst-case wall 4.31+2.0 = 6.31 ≤ 17); campaign 52.61 + 1.5 est in flight = 54.11 ≤ 100 |

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
| F8 | full parity battery under the Perlmutter NATIVE pinned env (cgl2-pm, jax 0.6.2 x86 A100 — Path A superseded the jax-0.10 container wording) | report-only | **ALL HARD GATES PASS** (job 55980038, 58 s on A100; F1 5.682e-15 bit-consistent with phoenix, F4 1.314e-11, F6 3.562e-10 within restated tol) — scene-API stack certified on BOTH machines; Perlmutter scene-API cells (B1/S7/L0-G2) UNBLOCKED | data/results-perlmutter/parity_report_scene_pmnative_55980038.json |
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
| T1.1b-G0 | residue-masked refit entry gate: whiten-then-drop dof budget — erode the production v3b whitener keep_w by the T1.1 residue regions (bright-object ∪ center, definitions pinned by exact reproduction of the ledgered G1 decomposition 0.952/0.973/2.71/5.46) | kept-dof loss ≤ 40% (pre-declared in the T1.1b tasking BEFORE the build ran) | **FAIL — STOPPED, information-starved (no jobs submitted, 0 A100-h)**: loss **85.8%** (keep_w 9273→1320); the region drop ALONE (before the 21×21 kernel-support erosion) already loses 47.6% — no whiten-then-drop variant of this region set can pass; arc-band (1.2–4.2″) whitened px 6373→**108**, survivors at r 3.9–6.2″ (median 4.9″) = pure outer sky, essentially zero lensed-arc signal; diag-control arm alone loses 35.9% (would pass) but the corr arm IS the experiment. Positive content: fit-side residue masking is STRUCTURALLY incompatible with this injection design — the residue region is the bright scene, which is exactly where the injected arcs live, and M=10 whitening support dilates the drop over the rest. Residue-free injections must be built DATA-side (kernel-sampled noise-only / sky-set bootstrap, or deeper multi-start scene subtraction), as the T1.1 implications already queued for P3. | data/t11b_residue_mask_report.json, 09_build_residue_masked_whitener.py |
| T0.4-1 | per-block kernel homogeneity (stationarity of the noise-model class) | 2σ blockwise + calibrated p | **REJECTED** — money product v3b max\|z\|=3.67, calibrated p=0.010; arc-excluded v3 p=0.010; replicated spatial pattern; observed cross-block ρ(0,1) spread 16.4× the drizzle-registration envelope. **The stationary kernel class behind γ=1.103 is provably violated by the field.** Verifier CLEAN (all z/p recomputed exactly; power check confirms informative nulls). | data/t04_stationarity*.json, figs/t04_stationarity_*.png, research/t04_free_checks.md |
| T0.4-2 | λ-arm: does spectrum-flooring cure the fine-low gaming? | ordering table w/ per-λ exact log\|C\| | **NO — "information-discard-at-spectral-zeros" FALSIFIED**; data reject flooring by 3.3k–33k nats; pathology localized to down-weighting of high-S large-scale modes (the w_b≈0.27 background component — exactly the nonstationary component of T0.4-1). Production s_floor=0.05 confirmed (plan's "0.1" corrected); production taps reproduced to 1e-9. Verifier CLEAN (per-λ Szegő anchors verified — no shared constant). | data/t04_lambda_arm.json, figs/t04_lambda_arm.png |
| T0.4-3 | real-space head-to-head on the SAME v3b pixels | report-only ordering | Binned data in real space prefers **γ≈1.29** (diag-low, χ²_pp 1.58) over BOTH the anchor 1.433 (7.44) and corr-low 1.103 (8.32); the production whitened metric INVERTS this (+501 nats for corr-low). Corr-low's residual = smooth lens-center misfit (same currency as fine-low gaming). Anchor's full-field number carries a cross-product resolution handicap (honest caveat). | data/t04_realspace_headtohead.json, figs/t04_headtohead_residuals.png |
| L0-G2 | scene-API v3b-low CORRELATED refit reproduces the money number + logZ sign (P3 license for X1-class real-lens claims) | \|γ−1.1032\| ≤ 0.017 AND ΔlogZ(steep−low) < 0 | **PASS** (job 55985451, 31 min; harvested + gate-evaluated by the ORCHESTRATOR, commit 1993cc6; row recorded here at the wave harvest): γ_low = 1.1005 [1.0992, 1.1065], Δ = 0.0027 ≤ 0.017; logZ_low = −4770.97 vs old-stack production −4771.08 (0.11 nats across stacks/machines); ΔlogZ(steep−low) = −29.36, sign + magnitude in the P1 seed family (−28.9/−32.3/−31.2). CorrelatedImageData port LICENSED; γ=1.103 reproduced on two independent stacks | data/results-perlmutter/l0g2_v3b_scene_seed2.{json,npz} |
| B2 | prior-seeded S1 in ORIGINAL pathological (Om0,w0)+NormalCDF coords recovers their pre-registered arm mass; control dspl20_ratio reproduces Run A r2 ≈ N(1.32417, 6.7e-4) | \|m̂(Om0<0.146) − 0.103\| ≤ 0.045 | **ORIG ARM EVALUATED — m̂ = 0.000 (0/512 particles; 95% rule-of-three < 0.006), OUTSIDE the band ⇒ FAIL AS WRITTEN** (job 55985447 TIMEOUT cut the in-job ratio arm at 79 min mid-SMC). **CONTROL LANDED 2026-07-19 (job 56006065, COMPLETED 01:49) — FALSIFIER DOES NOT DECIDE AS REGISTERED: the 0.103 band is UNCALIBRATED for this realization (pre-registered branch 2 of postmortem §4, written before the control ran, executed as written).** Control r2 = u_fn(Om0,w0): mean 1.32364, σ 5.32e-4 → width ratio 0.79 vs their 6.7e-4 (inside [0.67,1.5] — shape reproduces; operationalization DECLARED AT HARVEST, flagged in the json, the frozen row named the reference not a tolerance); mean offset −0.78σ_RunA from 1.32417 and −0.41σ from THEIR data u* 1.32392 (location within fresh-realization scatter; not a hard gate — data_seed=0 realization ≠ their unreproducible baseline). BUT the control's OWN minor-arm mass m̂_ctrl(Om0<0.146) = 5/512 = 0.0098 ± 0.0043, far OUTSIDE their Run-A band 0.103±0.045: the exact-reparameterization control — which cannot suffer coordinate mode-death — measures THIS realization's true arm mass at ~1%, not ~10.3% ⇒ the wave-1 fresh-realization caveat is CONFIRMED and the orig arm's m̂=0.000 was scored against a band that does not transfer across realizations. Gate must be RE-REGISTERED against the control's own arm mass (B2′, ledgered amendment — not a silent move). Against that target: orig 0/512 vs control 5/512 ⇒ Fisher one-sided p = 0.031 (two-prop z 2.24) — SUGGESTIVE of residual minor-arm under-coverage in the original coords, NOT conclusive (~2σ, n=1 seed each, below the campaign's 3σ convention). Dominant-arm AGREEMENT is striking: control Om0 med 0.4702 [0.333, 0.515] vs orig 0.4701 [0.352, 0.515] — statistically identical ⇒ the orig posterior's Om0 0.470-vs-truth-0.3 offset is REALIZATION, not sampler. Also recorded: ΔlogZ(orig−ratio) = +3.06 ± 0.35 nats on identical data (exact reparameterization; truncation excludes ~1e-138 posterior mass ⇒ true Δ≈0) — an evidence-error datum: σ_boot understates cross-parameterization logZ error (B0 known issue, now measured on DSPL). Control SMC sanity CLEAN: λ=1 @64 stages, min unique 360 ≥ 128, accept 0.82–0.93. NET B2 VERDICT: headline as written NOT ESTABLISHED, honest MIXED row — dominant arm: sampler ≡ control; ~1% minor arm: 0/512 vs 5/512 (2σ, underpowered); a decisive B2′ needs seeds/N beyond the standing P2 cap. Posterior in ORIGINAL coords: Om0 med 0.470 [0.352, 0.515] eqw (truth 0.3); logZ = 17294.01 ± 0.25 (recorded for repeatability context). SMC sanity CLEAN: λ=1 at 64 stages, min unique 358 ≥ N/4=128 (no resample collapse — if the minor arm died it died by weight decay, not collapse), final incr-ESS 486/512, accept 0.80–0.94. Harvest recompute cross-checks: orig max\|d\| = 5.6e-17; control θ_E proxy 2.2e-16 (grouped ratio prior emits no per-cosmo readout; plot inspected BEFORE gate math per house rule) | data/b2_gate_eval.json (combined orig+control+falsifier), figs/b2_om0_posterior.png, figs/b2_ratio_control.png, data/results-perlmutter/b2_dspl20_{orig,ratio}_s1_seed2.{npz,json}, 26_harvest_b2.py (plot-ratio/gates-ratio stages) |
| B5 (G1/G2/G3) | both basins at λ=1; mode weight within binomial of P2c; basin ΔlogZ within 3·√(σ_boot²+1.79²) (floor 5.36); G3 MCLMC distortion 1.5–3× | PLAN §6 B5 + finalized σ row | Wave-1 (55985449/50): INFRA-FAIL ×2 (RC3 writer defect, regression-locked); recovery low legs 56006049/52 harvested 2026-07-19; **steep leg LANDED 2026-07-21 (56251555 attempt #4, chunk 32) — figs/b5_basin_overlay.png plotted BEFORE this text. G1 PASS (both basins)** — low λ=1 @31 stages, retention 1.000, min unique 88, accept 0.86–1.00, logZ 38376.33 ± 0.33_boot, γ_eqw med 1.0919; steep λ=1 @22 stages, retention 1.000 (frac_low = 0.0), min unique 70 ≥ N/4=24, accept 0.886–0.998, logZ 38518.23 ± 0.31_boot, γ_eqw med 2.5552. **G2 REFERENCE-COMPARABILITY RESOLVED, then FAIL AS WRITTEN.** Comparability (the flagged +25.2-nat/γ-1.24 investigation, done BEFORE gate math): the P2c reference IS on the same target definition AND normalization by construction — both estimators run `cgl.zoo.get_target("foundry_v3b74")` in the SAME validated old stack (our runner executes in the old venv), and both use the identical 24_basin_evidence_v3b q_k construction (logprior:=log q_k, loglike:=log_prob−log q_k, n_fit 8000, cov_inflate 2.0, rng offset 24_20260709, split γ=1.8) so the evidence integrand is exp(log_prob) on BOTH sides and q_k cancels — there is NO normalization/prior-const to reconcile (verified line-by-line 24_run_p2_oldstack.py vs the old repo's 24_basin_evidence_v3b.py; the old repo's own on-disk smoke v3b_s0.json reproduces Δ=162.7, γ_low 1.279 under their kernel). Gate math therefore licensed: **Δ_meas(steep−low) = +141.90 ± 0.45_boot vs P2c +162.2 ± 1.8 ⇒ deviation −20.30 nats = 11.0× the gate σ-scale (band 5.54) — FAIL.** Honest decomposition (recorded, no goalpost move): per-basin offsets are +25.16 (low) and +4.86 (steep), BOTH positive — we find MORE evidence than P2c in BOTH basins — and both λ=1 ensembles land γ-DISJOINT from the hmc_v13_v3b chains P2c seeded AND mutated from (low: ours [1.071,1.126] vs chains [1.265,1.381]; steep: ours [2.514,2.600] vs chains med 2.423/q95 2.438), with TWO independent kernels agreeing on the low-basin location (MAMS 1.0919 / MCLMC 1.0935). The 1.0919-vs-1.24 flag is thereby explained: 1.24–1.29 is the reference-CHAIN support, and the adaptive anneals migrate off it in both basins toward higher-log_prob shelves the P2c fixed-step HMC(0.1, L=8, 2–4 steps) never reached ⇒ the discrepancy is within-basin COVERAGE (the FAIL indicts the frozen reference's coverage at least as much as the S1 vehicle); n=1 seed, attribution decidable only by a seed-repeat outside the P2 cap. Mode-weight clause DEGENERATE: w_low ours 2.4e-62 vs P2c 3.6e-71 — both predict 0 low particles at any feasible global N, binomially vacuous (ΔlogZ is the operative clause). **G3 EVALUATED — AMBIGUOUS AT FROZEN THRESHOLDS (the two frozen clauses CONFLICT on this measurement).** Construction: LOW is the minor mode on this diagonal target (P2c w_low ≈ 3.6e-71) and the minor-mode odds ratio cancels Z_steep, so G3 is evaluable without the steep leg: ΔlogZ_low(MCLMC−MAMS) = +4.03 ± 0.49_boot nats (8.3σ_boot from zero — real vs bootstrap noise) ⇒ MCLMC INFLATES the minor-mode weight ×56 [CI95_boot 22–146], EXCEEDING the pre-registered 1.5–3× prediction band (6.0σ_boot above the ln3 ceiling) — resampling does NOT launder the bias at the point estimate; BUT under the imported σ_seed=1.79 convention (cross-target: measured on cgl2-v3b f64) the 4.03-nat shift sits INSIDE the 5.56-nat MAMS repeatability band, so the frozen laundering falsifier ("MCLMC ΔlogZ within MAMS repeatability ⇒ launders") technically fires. 4.03 lies between the 1.10-nat prediction ceiling and the 5.56-nat band; n=1 seed per kernel — decidable only with a MAMS seed-repeat on THIS target. Within-basin γ marginal essentially untouched (MCLMC 1.0935 vs MAMS 1.0919, Δ 0.0016): the unadjusted-MCLMC bias lands in the EVIDENCE, not the within-basin location — consistent with the B0 30σ bias screen + MCLMC's demotion to diagnostic. **G3 occupancy completion (checkpoint item, steep in hand, 2026-07-21):** w_low(MAMS) = 2.36e-62, w_low(MCLMC-low/MAMS-steep) = 1.33e-60 (ratio ×56.4 — the G3 point estimate re-expressed as occupancy), P2c w_low = 3.6e-71; construction-vs-P2c moves occupancy ×6.5e8 (=e^20.3, the G2 deviation) — every estimator leaves the low mode utterly negligible (≤1.3e-60), so the minor-mode OCCUPANCY conclusion is robust across estimators; what is NOT robust at the ±5-nat gate scale is absolute evidence calibration | data/b5_gate_eval.json, figs/b5_basin_overlay.png, data/results-perlmutter/b5_v3b74_{low_mams,low_mclmc,steep_mams}_seed2.{npz,json}, research/p2_wave1_postmortem_redesign.md |
| B1 (mock arms) | self-consistency / logZ repeatability / efficiency vs S6b / cold-start | PLAN §6 B1 | **NOT EVALUABLE — all 3 arms TIMEOUT with ZERO artifacts (11.02 A100-h)**: runner writes only at λ=1 and prints no stage progress (RC1); est-hours were calibrated on the old-stack v3b class, not scene N=512 carousel (RC2). Measured anchors: S6b MAP 302 s, round 0 ≈ 25 min, <20/150 rounds in 2.48 h ⇒ S6b true cost ≈ 17–20 h; S1@N=512 ≈ 25–40 h/arm (the 4-h walls bought ~5–8 early stages, λ ≲ 1e-4). Mock arms will NOT be resubmitted (real data landed — B1-REAL-DATA-LANDED amendment); B1-real needs the checkpoint retrofit + a cap decision (rerun matrix in the redesign doc) | data/results-perlmutter/b1_*_run.log, slurm-55985444/45/46.out, research/p2_wave1_postmortem_redesign.md |
| B1r decision matrix (S1r vs S6br, D8/D9) | carousel33 REAL-data cell: prior-seeded MC-SMC (S1r) vs warm MAMS-alone (S6br) at matched grad budget B\*=3,078,912 | research/checkpoint_b1_reduced.md + checkpoint_b1r_close.md (framing pre-stated BEFORE S6br results) | **NEITHER CONVERGES AT THIS BUDGET** (figs/b1r_s6br_traces.png plotted BEFORE this text). S1r: 11.33 A100-h → λ=0.1506 @36 stages, 0 posterior samples, ESS/R̂ n/a (healthy sampler, killed by cost — D9 cost row stands). S6br (56252401): the pre-registered 4.5-h wall fence bound FIRST, at burn round 11 of 30 — **sampling never started** (0 draws), realized 1,277,632 grads = **41.5% of B\*** (map 22,400 + tune 894,080 + mutate 361,152), **ESS = 0, worst-param R̂ NOT COMPUTABLE**; burn itself healthy (accept 0.873–0.902 mean 0.887, eps ~9e-4, n_int 64 — ~24 min/round on the 33-dim multi-plane real-MUSE target). Per-grad throughput comparable across arms (~284k vs ~272k grads/h): the truncation is a target-cost fact, not arm inefficiency; matching B\* by grads needed ~10.9 h vs the 4.5-h fence, and until-converged S6b remains est 17–20 h (wave-1 anchors). The close-out's pre-stated "any nonzero ESS ⇒ decisive LOSS for cold-start" branch did NOT fire — the row is the equally-honest anticipated alternative: **carousel-class real-data targets are out of reach for BOTH vehicles at any campaign-affordable budget** (row narrows claim to target class; cold-start WINS/LOSSES on B4/B5/DSPL-class stand separately). Pre-registered caveats carried: burn-dominated comparison (biases the rate metric TOWARD S1r — moot, both ESS exactly 0); descoped gates stay descoped; no science read from either non-converged ensemble | data/b1r_decision_matrix.json, figs/b1r_s6br_traces.png, data/results-perlmutter/b1r128_carousel33_s6b_seed2.{json,npz,_run.log}, data/results-perlmutter/b1r128_carousel33_s1_seed2.PARTIAL.json |
| B4 | prior-seeded S1 replaces the two-stage recipe on cond~1e14: worst-param \|z\|<3 vs P2c ref; ESS ≥ 0.3N | PLAN §6 B4 | **NOT EVALUABLE — TIMEOUT 03:30:27, ZERO artifacts, ZERO stage observability** (log: build 78 s + finite warmup, then silence — RC1). True cost at N=256 unknown, lower-bounded at 3.5 h. CARRIED DEFECT: B4 runs the SAME writer code path as B5 — even on completion it could not have written artifacts; the one-line fix is a precondition for ANY B4 resubmission. Two-stage-seeded variant flagged OFF-PROTOCOL (answers a different hypothesis; would need pre-registration as a NEW arm B4b) | data/results-perlmutter/b4_marg46_s1_seed2_run.log, slurm-55985448.out, research/p2_wave1_postmortem_redesign.md |

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

### 2026-07-21 — FINAL-WAVE HARVEST: B5 gates closed (G1 PASS / G2 comparability-resolved-then-FAIL / G3 occupancy completed) + B1r decision matrix = "NEITHER converges" + classic-arm 0-d-chisq root cause fixed, locally smoke-proven, resubmitted (56267678); P2 22.45/24, P2b 15.85/18 CLOSED, P3 4.31/17, campaign 52.61/100

- **Session ops:** cert VALID (tested first); mid-session ~25-min Perlmutter
  ssh outage (auth handshake stall on login node x3116c0s5b0n0 — banner
  printed, publickey auth hung; earlier + later sessions on other login
  nodes fine; recovered on its own, all local work continued meanwhile).
  Final-wave actuals sacct'd into the ledger (56251555 0.37 COMPLETED,
  56252401 4.52 COMPLETED, 56252932 0.02 FAILED).
- **HARVEST 1 — B5 steep (56251555, attempt #4 chunk 32 WORKED):**
  scp'd steep {npz,json,run.log} to data/results-perlmutter/;
  figs/b5_basin_overlay.png plotted FIRST. λ=1 @22 stages, retention
  1.000, logZ 38518.23 ± 0.31_boot, γ_eqw med 2.5552. **The two flagged
  observations were investigated BEFORE gate math** (comparability question
  in the tasking): P2c reference numbers ARE same-target/same-normalization
  BY CONSTRUCTION — both estimators run `cgl.zoo.get_target('foundry_v3b74')`
  in the same validated old stack with the identical q_k evidence
  construction (q_k cancels; integrand = exp(log_prob) both sides; verified
  line-by-line 24_run_p2_oldstack.py vs old-repo 24_basin_evidence_v3b.py;
  old repo's on-disk v3b_s0.json smoke reproduces Δ=162.7/γ_low 1.279 under
  THEIR kernel). So G2 is scoreable, and **FAILS as written** (Δ 141.90 ±
  0.45 vs 162.2 ± 1.8; −20.30 nats = 11.0σ of the gate scale). The honest
  physics: both basin offsets are POSITIVE (+25.16 low, +4.86 steep) and
  both our λ=1 ensembles are γ-DISJOINT from the hmc_v13_v3b chain support
  P2c seeded AND mutated from (low: [1.071,1.126] vs [1.265,1.381]; steep:
  [2.514,2.600] vs med 2.423/q95 2.438) with MAMS+MCLMC agreeing — the
  γ-1.0919-vs-1.24 flag is thereby EXPLAINED (1.24–1.29 is reference-chain
  support; adaptive anneals migrate off it to higher-log_prob shelves the
  P2c fixed-step HMC never reached): a within-basin-COVERAGE discrepancy
  indicting the frozen reference at least as much as the vehicle. G1 PASS
  both basins; G3 occupancy completion recorded (w_low MAMS 2.36e-62 /
  MCLMC 1.33e-60 / P2c 3.6e-71 — minor-mode negligibility robust, absolute
  evidence calibration not). Gate row + data/b5_gate_eval.json updated.
- **HARVEST 2 — S6br (56252401):** scp'd s6b {json,npz,run.log};
  figs/b1r_s6br_traces.png plotted FIRST. The 4.5-h wall fence fired IN
  BURN (round 11/30), **sampling never started**: 0 draws, ESS = 0, R̂ not
  computable, grads 1,277,632 = 41.5% of B\*=3,078,912 (wall fence bound
  before the grad fence, as the amendment anticipated — "fence-truncated
  budget" labeling applied). Burn healthy (accept mean 0.887, eps ~9e-4,
  n_int 64); per-grad throughput ≈ S1r's (~284k vs ~272k grads/h) — the
  truncation is target cost, not arm inefficiency. **Decision-matrix row
  (gate record + data/b1r_decision_matrix.json): NEITHER CONVERGES AT THIS
  BUDGET** — the close-out's pre-stated "any nonzero ESS ⇒ decisive
  cold-start LOSS" branch did NOT fire; the equally-honest anticipated
  alternative holds: carousel-class real-data targets are out of reach for
  BOTH vehicles at campaign-affordable budgets. P2b cell fully closed at
  15.85 of 18.
- **FIX — classic arbitration arm (56252932 → 56267678, attempt #2):** root
  cause of the 1:21 IndexError: `_map_warm_refine.one_step` does `chisq[i]`
  but cgl2 CorrelatedImageData mirrors the stock bs==1 squeeze
  (correlated.py `return ll[0], chi2[0]`), so the single-row warm batch gets
  a 0-d chisq while lp re-broadcasts to (1,) through the batched log_prior —
  `lp[i]` works, `chisq[i]` raises. The CPU toy missed it (ToyPM returns
  batched chisq); B3 never hit it (multi-start MAP n>1 — this arm is the
  first n=1 caller). **Per the tasking rule the crash was reproduced
  LOCALLY BEFORE resubmission** (phoenix L4 GPU 9, CUDA_DEVICE_ORDER=
  PCI_BUS_ID, GIGALENS_X64=1, REAL v2d scene target, tiny budgets —
  identical signature at the same line), fixed minimally
  (`jnp.atleast_1d` on both aux outputs in `loss`; identity for batched
  callers), the invocation kept as the documented `--smoke` mode (module
  docstring + argparse; *_smoke stems only), and the smoke rerun to CLEAN
  EXIT 0 (135 s pipeline: MAP 19.7 s → SVI 23.9 s → HMC + escalation +
  blocked-by-reference path — the tiny-budget R̂/ESS fail is the EXPECTED
  path coverage; warm-start fence values bit-match Perlmutter). Restage +
  resubmit mechanics in the 56267678 ledger row (union-merge clean — only
  our own line diverged; remote audit 225/225; snapshot verified;
  watchdog swapped). Smoke artifacts kept at
  data/l0_arbitration_classic_smoke.{json,npz} (local evidence, not staged).
- **Queue at close:** 56267678 (l0arbc #2) PENDING (Priority), wall 2:00;
  watchdog 1 job, heartbeat fresh.
- **Budget statement:** P2 **22.45 of 24** actual (headroom 1.55, nothing
  in flight); P2b **15.85 of 18** actual (cell CLOSED — S1r 11.33 + S6br
  4.52; the 2.15-h residual is NOT spendable without a user decision); P3
  **4.31 of 17** actual + 1.5 est in flight = 5.81 (worst-case wall 6.31);
  campaign **52.61 actual of 100** + 1.5 est in flight = **54.11 ≤ 100**.
  NOT committed (house rule: user commits).

### 2026-07-20 — TWO PERLMUTTER FIXES: B5-steep attempt #4 SUBMITTED (56251555, chunk 32 — the PILOT divisibility root cause) + L0-arb VEHICLE AMENDMENT executed (classic-recipe arm PRIMARY, 56252932 submitted); cross-front deploy.md5 clobber caught + repaired; P2 22.08/24, P3 4.29/17, campaign 47.70/100

- **Session ops:** cert FRESH; WATCHDOG_ALERT triaged then deleted (JOB_FAILED
  56170614 + 56168446 = exactly this wave's two fixes; COMPLETED_NO_ARTIFACT
  56170221 = the B1r front's lane — resolved by their concurrent D9 harvest,
  see their rows). Fix-wave actuals sacct'd into the ledger (56168443 0.00,
  56170614 0.03, 56168446 3.58); the 56170614 table row is RETRO-RECORDED
  (deviation declared: the previous wave declared est+cap math in its stage
  entry but omitted the table row).
- **FIX 1 — B5-steep attempt #4 = 56251555 (P2, est 0.8, wall 1:15).** Root
  cause of #3 (56170614, `AssertionError: (64, 48)` at 1:52): chunk 48 fixed
  the FULL pass (96 % 48 = 0) but broke the PILOT — common.run_tempered_smc's
  dual-averaging step-size pilot (pilot_size=64, frozen at
  24_run_p2_oldstack.py:215) mutates a 64-particle subset through the SAME
  chunked kernel, and 64 % 48 ≠ 0 trips the same 22_run_b3.py:135 assert.
  The real constraint: **chunk must divide BOTH N and pilot_size** ⇒ chunk 32
  (3 chunks at N=96, 2 at the pilot; ≤ the proven chunk-64 memory). Edited
  slurm/p2_b5_s1_steep.slurm (both occurrences + comment now states the
  constraint) LOCALLY AND ON CFS, md5 **51c7ae1bca9c294c50d00d3cf593c774**
  identical both sides, THEN sbatch, THEN the 07-19 ops lesson executed:
  **submitted-SNAPSHOT verified** — `scontrol show job --Batch` prints
  nothing on this slurm build, so `scontrol write batch_script 56251555 -`
  was used: snapshot md5 = 51c7ae1b… (bit-identical to local/CFS), `--chunk
  32` present at the command line. Watchdog: 56251555 registered (max_run
  2.0, expect CFS steep npz, on_stall=alert), 56170614 deregistered.
- **FIX 2 — L0-arb protocol amendment + classic arm = 56252932 (P3, est 1.5,
  wall 2:00, plain gpu).** AMENDMENT appended to research/checkpoints_l0.md
  (dated 2026-07-20, BEFORE the run; pushed to CFS with the tree): after
  THREE wall-capped MC-SMC attempts (L4 λ≈0.45 @ ~18 h, ckpts stage_000–067;
  A100 plain-gpu OOM; A100 hbm80g 56168446 λ=0.587 @ 3.5 h at stage 76,
  ckpts stage_000–075 on CFS — sampler HEALTHY, accept ~0.88, no OOM: the
  VEHICLE is budget-infeasible, ~1.2×/stage λ growth), the pre-registered
  optional arm (b) — classic vendor-recipe MAP→SVI→HMC — is PROMOTED to
  PRIMARY arbitration vehicle. λ readouts above are wall-cap-protocol ops
  telemetry; NO gate math was done on any non-λ=1 ensemble, NO γ read.
  **Gates/bands UNCHANGED** (dominant-basin γ_med within 2σ_comb of 1.433
  [1.400,1.468] + 68% overlap + multimodality clause verbatim — the GATE is
  about the POSTERIOR, not the sampler); MC-SMC partials retained as
  SECONDARY prior-seeded evidence with PARTIAL status recorded; trade-off
  declared: the warm classic arm inherits the warm-start basin (as the
  old-stack anchor fit did — that is exactly the arbitration question);
  prior-seeded alternative-basin coverage REMAINS OPEN. Build: NEW
  11_arbitration_classic.py — imports 10_anchor_arbitration's certified
  build_pm/γ machinery + 22_run_b3's replicated MAP/SVI/HMC by path; warm
  single-start MAP refine from param_map(refs v2d:z_ref:x46) (multi-start
  from one fixed point is degenerate — declared); B3 frozen settings incl.
  R̂/ESS acceptance + single escalation + SVI OOM ladder; stage-wise artifact
  writes (post-SVI/post-HMC/final) so a walltime kill loses ≤1 stage;
  outputs γ posterior + per-param R̂/ESS worst-param; no in-job gate math.
  **Local CPU toy test PASS** (synthetic quadratic target only — the lensing
  likelihood was never run on CPU; exercised escalation AND
  blocked-by-reference paths; posterior median within 0.02 of truth).
  Staged: script md5 3f4787fc, slurm 2a042788 identical local/CFS;
  submitted-SNAPSHOT verified (`scontrol write batch_script 56252932 -` md5
  = 2a042788… bit-identical, `-C gpu` plain + wall 2:00 confirmed).
  Watchdog: 56252932 registered (max_run 2.5, expect CFS
  results/l0arb/l0_arbitration_classic.npz — note stage-wise json may exist
  even when the npz is missing); 56168446 deregistered (terminal, artifacts
  + ckpts already on CFS).
- **CROSS-FRONT deploy.md5 CLOBBER (caught + repaired):** the B1r front was
  staging its D9 S6br concurrently; this front's deploy.md5 push overwrote
  the B1r manifest update while their 56252401 was PENDING (remote audit
  then FAILED on slurm/p2b_b1r_s6b.slurm — which is how it was caught).
  Repair: their s6b line (c65d3a71, file verified bit-identical local/CFS)
  merged into THIS front's manifest ⇒ deploy.md5 is now the **UNION manifest
  (225 files: +11_arbitration_classic.py, +slurm/p3_l0arb_classic.slurm,
  updated p2_b5_s1_steep.slurm, B1r's s6b hash kept)**, pushed, **remote
  self-check PASS 225/225** — both fronts' pending jobs' in-job audits
  green. OPS RULE for concurrent fronts: deploy.md5 is a shared
  single-writer file — pull/merge the CURRENT remote manifest before
  pushing, and re-verify the remote audit AFTER every push while another
  front is active. (Local self-check transiently shows the other front's
  in-flight files by design — the REMOTE audit is binding, per the 07-19
  precedent.)
- **Queue at close:** 56251555 (B5-steep#4) PENDING, 56252932 (l0arbc)
  PENDING, 56252401 (B1r S6br, theirs) PENDING; watchdog 3 jobs, heartbeat
  touched, tmux loop alive.
- **Budget statement:** P2 **22.08 of 24** actual (+0.8 est in flight =
  22.88; worst-case wall 23.33 ≤ 24 — no breach flag); P3 **4.29 of 17**
  (+1.5 est = 5.79; worst-case 6.29); P2b 11.33 + 5.0 est (B1r's lane);
  campaign **47.70 actual of 100** + 7.3 est in flight = 55.00. NO results
  read (λ/exit telemetry only, per the wall-cap protocol); NOT committed
  (house rule: user commits).

### 2026-07-19 (late) — B5-steep resubmit #2 (script-snapshot root cause)
56168443 failed in 12 s with the ORIGINAL assert signature (96, 64): sbatch snapshots the
script at submission, and the first resubmit was submitted before the chunk-48 fix landed in
the file — the fix never rode along. Closure/code path verified sound (make_chunked_mams
captures the CLI chunk; 96 % 48 = 0 passes). Current CFS script verified to carry --chunk 48
(×2 occurrences) → resubmitted as **56170614** (est 0.8 A100-h, P2; worst-case P2 23.65 ≤ 24 —
no breach). Watchdog swapped (56168443 out, 56170614 in); alert flag cleared. OPS LESSON for
the ledger: fix-then-submit ordering must be verified via `scontrol show job --Batch` or by
md5-ing the submitted script, not by later file state.

### 2026-07-19 (late) — P2b B1-REDUCED SUBMITTED: S1r 3-leg chain live (D8 funding executed); S6br stays TEMPLATE-HELD; amendment checkpoint of record = research/checkpoint_b1_reduced.md

- **AMENDMENT CHECKPOINT (referenced per the concurrent-agent discipline, not
  copied):** `research/checkpoint_b1_reduced.md`, md5
  **865247b16e90061803c9f4cf67fa1c8f**, pre-registered 2026-07-19 BEFORE this
  submission. Contents of record: descope-only amendment of the 2026-07-16 B1
  design checkpoint (below) — S1 → S1r N=128 seed-2-only on REAL carousel
  cutouts; S6b until-converged → S6br Track-A budget-matched (B\* =
  S1r `grad_evals.total`, MAP billed, 30 burn rounds FROZEN, grad + 4.5-h
  graceful round fences via ckpt.round_fence_reason, env seams default OFF);
  **no gate threshold moved** (efficiency ≥2/[0.7,2)/<0.7, cold-start |z|<3 +
  width [0.7,1.4], falsifier unique<N/4=32 at λ≳0.8, SMC sanity — verbatim);
  2-seed self-consistency + ≤2-nat logZ repeatability **DESCOPED, stated
  plainly**; fence-truncation labeling + burn-dominated caveat pre-registered
  with bias direction (toward S1r) stated before any run. Verifier verdict on
  the amendment: CLEAN (all thresholds verbatim, budget arithmetic checked,
  chain semantics verified in code, real-data no-mock-fallback confirmed).
- **D8 row + P2b ledger rows appended ABOVE, BEFORE sbatch** (est 4.0 × 3 legs
  + 5.0 S6br reserved = 17.0 ≤ 18 cap; worst case 16.67).
- **Deploy audit (the re-audit this front owed):** LOCAL self-check PASS
  223/223 (deploy.md5 includes the p2b_b1r quartet + retrofitted
  23_run_p2_scene.py/cgl2/samplers/ckpt.py/tests) AND REMOTE CFS audit PASS
  223/223 with p2b file md5s bit-identical to local
  (e1fe0d09/cd6af1e1/f0673040/bb14a736). Stale-artifact check: NO b1r ckpt
  dir / marker on $PSCRATCH or CFS results (leg-1 fresh fence clear).
- **SUBMISSION (chain, per README_p2b.md; sbatch 2026-07-19 23:06 PT from the
  CFS slurm/ dir):** **L1 = 56170216** (no dependency, PENDING/Priority),
  **L2 = 56170219** (squeue Dependency = `afterany:56170216(unfulfilled)`),
  **L3 = 56170221** (`afterany:56170219(unfulfilled)`) — all three PENDING
  with walls 3:55, dependency links verified in squeue output at submission.
  Watchdog: all 3 legs registered (max_run 4.5 h; expect_artifact on the
  LAST leg only = CFS results/b1r128_carousel33_s1_seed2.npz — legs 1-2
  intentionally carry none, since exit-3 PARTIAL legs end COMPLETED by
  design and the ckpt dir is the progress signal; on_stall=alert — NEVER
  auto-resubmit, a blind resubmit without --resume trips the leg-1 no-mixing
  fence); watchdog loop RESTARTED on phoenix (tmux `watchdog`, 15-min
  passes — it was found dead since 2026-07-18) + heartbeat touched.
- **S6br NOT submitted** (template guards refuse unfilled GRAD_BUDGET or
  missing S1r marker; awaits the S1r grad ledger — orchestrator decision
  point if S1r ends PARTIAL).
- **NOT committed** (house rule: user commits).

### 2026-07-19 — RECOVERY-LANE HARVEST: B2 falsifier decided (band uncalibrated — pre-registered branch 2), B5-G3 evaluated (ambiguous at frozen thresholds), 2 failures root-caused + resubmitted (56168443 B5-steep chunk fix, 56168446 l0arb hbm80g); P2 22.05/24, campaign 32.76/100

- **Harvest ops:** cert FRESH; sacct -X actuals for 56006048/49/52/65 (ledger
  cells filled above; recovery lane cost 3.49 vs est 7.35). Pulled to
  data/results-perlmutter/: b2_dspl20_ratio_s1_seed2.{npz,json}+run.log,
  b5_v3b74_low_{mams,mclmc}_seed2.{npz,json}+run.logs,
  b5_v3b74_steep_mams_seed2_run.log, l0arb_56006048_run.log,
  slurm-5600604{8,9}/52/65.out, AND all four ckpt dirs (small: 7.1M/2.5M/
  2.9M/2K — the "keep remote unless small" clause allowed the pull; CFS
  copies remain the on-machine originals). 3 of 4 recovery jobs wrote full
  artifacts — the RC3 fix + checkpoint retrofit did their job.
- **B2 FALSIFIER DECISION (the headline — full numbers in the gate row):**
  the ratio control COMPLETED healthy and its r2 posterior reproduces the
  Run-A SHAPE (width ratio 0.79; mean −0.78σ from their 1.32417, −0.41σ from
  their data u*), but its OWN minor-arm mass is 5/512 = 0.0098 ± 0.0043 —
  the 0.103 ± 0.045 band is UNCALIBRATED for the fresh data_seed=0
  realization, so the falsifier DOES NOT DECIDE AS REGISTERED (pre-registered
  branch 2 of the postmortem executed as written: B2′ re-registration against
  the control's own arm mass is owed, ledgered — not silent). The orig arm is
  NOT convicted of mode-death: dominant-arm posteriors are statistically
  identical across coordinate systems (Om0 med 0.4702 vs 0.4701 — the
  0.470-vs-truth-0.3 offset is REALIZATION); the minor-arm comparison is
  0/512 vs 5/512, Fisher one-sided p = 0.031 — suggestive under-coverage at
  ~2σ, underpowered to convict. Bonus datum: ΔlogZ(orig−ratio) = +3.06 ±
  0.35 nats on identical data where the true Δ≈0 ⇒ σ_boot understates
  cross-parameterization evidence error (B0 known issue, measured on DSPL).
  Plot inspected before gate math (figs/b2_ratio_control.png); harvest =
  26_harvest_b2.py new plot-ratio/gates-ratio stages (bijector + u_fn only,
  CPU); combined artifact data/b2_gate_eval.json.
- **B5 (gate row updated):** G1 PARTIAL (low-MAMS PASS: λ=1 @31 stages,
  retention 1.000, logZ 38376.33±0.33; steep PENDING-RERUN); G2 NOT
  EVALUABLE without steep; **G3 evaluated WITHOUT steep** (minor-mode odds
  cancel Z_steep): MCLMC inflates the minor(low)-mode weight ×56
  (ΔlogZ_low +4.03 ± 0.49_boot, 8.3σ_boot) — outside the 1.5–3× prediction
  band, but INSIDE the 5.56-nat imported-σ_seed repeatability band ⇒ the two
  frozen G3 clauses conflict; verdict AMBIGUOUS AT FROZEN THRESHOLDS, needs
  a MAMS seed-repeat on this target to decide (n=1 seed/kernel). Within-basin
  γ untouched (Δ 0.0016): the bias lands in the EVIDENCE. Flagged
  observations for the steep harvest: +25.2-nat absolute logZ_low offset vs
  P2c; basin γ 1.0919 vs P2c's quoted 1.24. Numbers: data/b5_gate_eval.json.
- **FAILURE 1 — 56006049 B5-S1 steep leg (0:55, low leg SUCCEEDED first):**
  died at the FIRST mutate on `AssertionError: (96, 64)` (22_run_b3.py:135
  chunked-MAMS divisibility; steep N=96 with --chunk 64). LATENT since
  wave 1: the original 55985449 never reached the steep leg (set -e on the
  low-leg writer crash), so the assert had never executed. NOT dtype/OOM/
  writer (build 62 s, warmup 38616.379 finite). FIX = --chunk 48 (divides
  96, ≤ chunk-64 memory; chunking is execution-order-only: per-particle keys
  split before chunking, runner help says so, bit-identity vs stock verified
  in the deployment plan) ⇒ steep-only resubmit **56168443**
  (slurm/p2_b5_s1_steep.slurm; low leg NOT rerun; stale ckpt dir wiped
  in-job — it held no stage checkpoints, resume-from-checkpoint N/A since
  the crash predates stage 0). Cap math BEFORE sbatch: P2 22.05 + est 0.8 =
  22.85 ≤ 24; worst-case wall 23.30 ≤ 24 — SAFE.
- **FAILURE 2 — 56006048 l0arb (0:10, PAST the vendor guard — the PYTHONPATH
  fix worked, gigalens resolved to the $PSCRATCH copy):** NEW failure mode =
  GPU OOM at the first mutate — a SINGLE 21.04 GiB allocation
  (RESOURCE_EXHAUSTED, BFC pool exhausted) on the plain-gpu 40 GB A100. The
  "~93 MB/particle ⇒ N=128 ≈ 12 GB, don't burn hbm80g" sizing is REFUTED BY
  MEASUREMENT: it was an L4 figure taken WITH the L0_REMAT deviation active,
  and this run pre-registeredly keeps L0_REMAT unset. FIX = `-C gpu&hbm80g`
  (pure hardware, zero numerics/protocol change — smaller deviation than
  enabling remat, and it restores the standing SMC pin) ⇒ resubmit
  **56168446** (est 2.5; P3 0.71 + 2.5 = 3.21 ≤ 17 — SAFE). Checkpoint
  resume N/A (l0_smc_checkpoints empty — crash predates stage 0).
- **Ops/deploy:** deploy.md5 → 219 files (p3_l0arb.slurm hash updated,
  p2_b5_s1_steep.slurm added); remote self-check **PASS 219/219** BEFORE
  sbatch. **Concurrency note:** the B1-REDUCED front is active in this tree
  (commit 7be8bcb tonight + in-flight edits to 23_run_p2_scene.py /
  cgl2/samplers/ckpt.py / tests/test_ckpt_resume.py timestamped during this
  session) ⇒ the LOCAL md5 self-check fails on exactly that trio by design
  of in-flight work; the binding audit is the REMOTE one (PASS — remote
  still runs the committed recovery-lane versions; the B1r front owes its
  own re-audit at its deploy). Per tasking, NOTHING B1/B4 touched here.
  Watchdog: 56168443/46 registered (max_run per wall + slack,
  expect_artifact on CFS, on_stall=alert — auto-resubmit without --resume
  would trip the ckpt no-mixing fence); 56006048/49/52/65 + stale phoenix
  273079 deregistered (terminal, harvested; B3 artifacts confirmed on
  disk); WATCHDOG_ALERT triaged (4× cert-window UNREACHABLE + 2× stale B3)
  and deleted; heartbeat touched. NOT committed (house rule: user commits).
- **Budget statement:** P2 **22.05 of 24** spent (+0.8 est in flight =
  22.85); P3 **0.71 of 17** (+2.5 est in flight); campaign **32.76 of 100**
  actual. The B5-steep resubmit fits without touching the pre-declared
  overflow flag; no cap decision needed for anything submitted here.

### 2026-07-18 — B3 harvested (partial-by-budget); recovery-lane state; CERT EXPIRED
- **B3 outcome:** 4/4 reference posteriors (classic scene-API MAP→SVI→HMC) COMPLETE on the L4;
  **0/4 MC-SMC arms inside the pre-registered 5 h/arm fence** (all BLOCKED-BY-BUDGET — driver
  log; the fence worked as designed). Gates not evaluable without the SMC arms; the readable
  result is the COST row: scene-API MC-SMC at N=512 on hs2-class targets exceeds 5 h/target on
  an L4 (consistent with the B1/B4 true-cost post-mortem). 12.38 phoenix GPU-h (free tier).
  Rerun path when funded: retrofit checkpointing (now landed, 58 tests green incl. bit-identical
  resume) + N=256 or A100. Artifacts: data/b3_cells.json, figs/b3_cost.png.
- **Recovery-lane agent** was killed by a session exit AFTER completing its work: RC3 fix +
  checkpoint retrofit (cgl2/samplers/ckpt.py; runners 23/24; bit-identity + regression tests),
  l0arb resubmit 56006048 (PYTHONPATH job-env fix, guard unchanged), recovery jobs 56006049
  (B5-S1) / 56006052 (B5-S2) / 56006065 (B2-control) — all submitted + ledgered pre-results.
  Verified post-mortem: 58/58 tests pass locally; watchdog registrations present.
- **OPS: sshproxy cert EXPIRED ~2026-07-17 morning.** The 4 Perlmutter jobs finished server-side
  (walls 1:30–4:00, submitted 07-16 evening); results presumed on CFS; HARVEST BLOCKED pending
  cert refresh. Watchdog correctly degraded to UNREACHABLE alerts (designed graceful state).
  Pending readouts on CFS: l0arb = the ANCHOR ARBITRATION verdict; B5-S1/S2 = the B5 gates +
  MCLMC bias-laundering test; B2-control = the B2 falsifier decision.

### 2026-07-16 (recovery lane + runner hardening) — RC3 fixed + regression-locked; RC1 checkpoint/resume retrofit landed (bit-identity gates PASS, 58-test suite green both sides); l0arb path-identity trip fixed job-env-only (guard UNCHANGED) + resubmitted; B5-S1/B5-S2/B2-ratio reruns submitted CHECKPOINTED (P2 est 4.85 ≤ 5.27 headroom); NO B1/B4 arm (user decision pending)

- **Executes** research/p2_wave1_postmortem_redesign.md parts (a) fix+retrofit +
  the recovery lane R1/R2/R3 ONLY. The B1/B4 options (§5b R4/R5) are NOT
  touched — cap decision belongs to the user; nothing here presumes it.
- **RC3 FIX (24_run_p2_oldstack.py):** summary assembly now goes through the
  shared `cgl2.samplers.ckpt.summarize_res`, which pops `res['kernel']` (and
  defensively `'particles'`) from a COPY before the `**`-merge — the exact
  `dict() got multiple values for keyword argument 'kernel'` crash of
  55985449/50 is a regression-locked class
  (tests/test_ckpt_resume.py::test_rc3_summary_kernel_collision_unreproducible
  asserts the old pattern raises AND the new path succeeds on a fake res
  carrying 'kernel'). Runner 23's s1 arm routed through the same shared
  assembly (behavior unchanged — it already popped).
- **RC1 CHECKPOINT RETROFIT (new module cgl2/samplers/ckpt.py; frozen driver
  common.run_tempered_smc UNTOUCHED):** transplant of the accepted
  10_anchor_arbitration.py pattern, upgraded per the approved plan —
  (i) delegating `CheckpointKernel` (build/tune/mutate verbatim): per-stage
  FULL-STATE `stage_NNN.npz` (post-mutation z, λ, eps, n_int, accept,
  per-stage grad/logp ledger) + one progress print + pre-registered
  `--wall-cap-h` raised AFTER the checkpoint (exit-3 PARTIAL json protocol,
  no gate math on a non-λ=1 ensemble — l0 verbatim); (ii) weight-step ll
  recording via the EXISTING keyword-only `loglik_batch_fn` seam
  (`LoglikRecorder`, delegation-only) — this is what lets a resumed run keep
  the FULL logZ + per-stage bootstrap (B1's ≤2-nat gate requirement; solves
  the l0 `logZ_partial_from_resume` limitation); (iii) `--resume`: driver
  loop copied WITH ATTRIBUTION, jax key chain + numpy Generator
  FAST-FORWARDED BY REPLAY of the recorded lls ⇒ a resumed run is
  **BIT-IDENTICAL to an uninterrupted run** — STRONGER than the l0 resume's
  RNG caveat. Wired into 23_run_p2_scene.py (s1 arm; s6b has no weight steps
  — its round loop is a separate design pending the B1 decision) AND
  24_run_p2_oldstack.py (b4+b5). Documented dtype caveat (module docstring):
  under GIGALENS_X64=0 (B5 strict-f32 only) the checkpointed ensemble is the
  wrapper's f64 reconstruction from the f32 (mu,chol,U) the kernel sees ⇒
  resume is protocol-identical with an f32-rounding-level perturbation at the
  resume point; under x64 (everything else + the CPU gates) it is bit-exact.
- **VALIDATION (house pattern — prove the expected exact-zero anyway):**
  tests/test_ckpt_resume.py — (1) RC3 regression; (2) stock-vs-instrumented
  BIT-IDENTITY (particles/logZ/boot-σ/all traces/ledgers exactly equal);
  (3) **the retrofit gate: fresh vs interrupted(2 stages)+resumed
  BIT-IDENTICAL incl. full logZ + bootstrap** + config-desync fence;
  (4) resume-after-completion rebuilds the identical result with ZERO new
  forward evals (= the exact B5 lost-at-the-write recovery path). Suite
  **58 passed** (54 + 4) on phoenix (cgl2 venv, CPU) AND on Perlmutter
  login (cgl2-pm venv, 85.5 s). Path-B vendor-free import of ckpt verified
  under the OLD venv on both hosts (gigalens never enters sys.modules).
- **L0ARB DIAGNOSIS + FIX (owed by this front):** 56004205 failed because the
  cgl2-pm venv's `gigalens.pth` appends the CFS vendor to sys.path, while the
  job executes from the audited $PSCRATCH copy (cgl2 resolves from script
  dir) ⇒ require_vendor_ref's import-identity check tripped exactly as
  designed. FIX = smallest change preserving the guard's intent, JOB-ENV
  ONLY (slurm/p3_l0arb.slurm): `export PYTHONPATH=$RUN/campaign/vendor/
  gigalens-linus/src` — PYTHONPATH precedes site-packages .pth entries, so
  the AUDITED COPY's vendor wins and the UNCHANGED guard now certifies ref +
  import identity against that copy. Preflight-PROVEN in a real $PSCRATCH
  tree copy (login CPU): without fix → CFS resolution (the trip); with fix →
  copy resolution + `require_vendor_ref: PASS`. Guard code untouched;
  deploy audit stays mandatory in the copy.
- **Deploy/audits:** deploy.md5 regenerated **218 files** (adds
  cgl2/samplers/ckpt.py, tests/test_ckpt_resume.py, slurm/p2_b2_ratio.slurm)
  — self-check PASS locally AND remotely (218/218) after rsync of the 9
  changed/new files.
- **PRE-SUBMISSION CHECKPOINT (frozen BEFORE sbatch, designs unchanged from
  the pre-registered cells + post-mortem matrix):** R1 B5-S1 rerun est 1.9 /
  wall 2:30 / per-leg caps 1.30+0.85 h, legs independent (steep runs even if
  low caps); R2 B5-S2 est 0.65 / wall 1:30 / cap 1.15; R3 B2-ratio
  STANDALONE est 2.3 / wall 3:00 / cap 2.70 (honest wall; "1 h" refuted).
  All three: `-C gpu&hbm80g` (standing SMC pin), checkpointing ON, PARTIAL
  json + ckpt dirs copied to CFS ALWAYS (a wall hit preserves progress by
  construction). l0arb resubmit est 2.5 / -t 04:00 / L0_WALL_CAP_H=3.5,
  plain `-C gpu` (own-cell rationale unchanged). Budget: P2 18.73 + 4.85 est
  = 23.58 ≤ 24 (headroom 0.42 vs worst-case walls 7.0 ⇒ the pre-declared
  overflow flag sits on the LAST-submitted job, B2-ratio — see its ledger
  row); P3 4.5 + 2.5 = 7.0 est of cap 17.
- **Ops:** watchdog — 56004205 deregistered (diagnosis + resubmit done),
  4 new registrations (max_run per wall + slack, expect_artifact on CFS,
  on_stall=alert — auto-resubmit without `--resume` would trip the ckpt-dir
  no-mixing fence by design), heartbeat touched. NO results read this
  session (submissions + hardening only; harvest is a later phase). NOT
  committed (house rule: user commits). NO B1/B4 submission of any kind.

- **Harvest ops:** sacct -X actuals for all 9 wave-adjacent jobs in the ledger
  (rows updated above); pulled to data/results-perlmutter/: the completed B2-orig
  npz+json, all B1/B2/B4/B5 run logs from $PSCRATCH, slurm-5598544*.out +
  slurm-56004205.out. Timezone note: slurm CANCELLED stamps are UTC; run-log
  mtimes PT (used together to reconstruct per-leg timings below).
- **B2 readout (gate row above; plot inspected BEFORE gate math,
  figs/b2_om0_posterior.png; plot and numbers agree):** the ORIGINAL-coords arm
  finished healthy (λ=1 in 64 stages, min unique 358, no collapse) but puts
  ZERO of 512 equal-weight particles below the Om0=0.146 arm split: m̂ = 0.000
  vs the pre-registered band 0.103±0.045 ⇒ outside the band. The posterior sits
  entirely in the upper arm (Om0 med 0.470 [0.352,0.515]; truth 0.3;
  logZ 17294.01±0.25). Because job 55985447's wall killed the in-job
  dspl20_ratio control at 79 min mid-SMC, the pre-registered falsifier ("orig
  outside WHILE control passes") is NOT yet decidable — mode-death vs
  fresh-realization-difference stays open until the control reruns on the same
  seeded recipe. Extraction: 26_harvest_b2.py (cgl2 venv, CPU, BIJECTOR ONLY;
  authoritative recompute from z via the deterministic builder; saved readout
  cross-check 5.6e-17). Numbers: data/b2_gate_eval.json.
- **B5 root cause (from the run logs, not a hypothesis):** both jobs died at
  the artifact write — `TypeError: dict() got multiple values for keyword
  argument 'kernel'` (24_run_p2_oldstack.py:205 builds the summary dict with
  `kernel=kern.name` AND `**res` without popping res['kernel'];
  23_run_p2_scene.py pops it — B2-orig wrote fine). INFRASTRUCTURE failure:
  both SMC runs completed (λ=1 + bootstrap done — the crash is after the
  driver return), then everything was lost. Low-MAMS ~55 min, MCLMC ~36 min,
  both healthy to the end (build 68 s, warmup finite, hbm80g held). The steep
  leg of 55985449 never ran (set -e). B4 shares this code path — its rerun
  requires the fix regardless of wall/N.
- **B1/B4 timeout post-mortem (RC1 + RC2, research/p2_wave1_postmortem_redesign.md
  §2):** RC1 = no progress checkpoint/print in the runners (the frozen driver
  is write-only-at-λ=1), so 11.02 h (B1) + 3.51 h (B4) burned with zero
  artifacts AND zero observability. RC2 = est rows calibrated on the old-stack
  v3b class; the scene class is ~10–30× heavier. Measured anchors: B2 (the
  cheapest scene cell) = 64 stages × 62 s at N=512; B1-s6b MAP 302 s + round 0
  ≈ 25 min + <20/150 rounds in 2.48 h ⇒ ≥7 min/round ⇒ S6b ≈ 17–20 h true
  cost, S1@N=512 ≈ 25–40 h/arm (4-h walls ≈ 5–8 stages, λ ≲ 1e-4); B4 true
  cost unknown (>3.5 h at N=256, no stage evidence exists). The wave's 18.69 h
  bought exactly 1.12 h of readable science (B2-orig) — 94% zero-artifact burn.
- **Budget statement (plain):** P2 = 18.73 of 24 ⇒ **5.27 h headroom**; the
  B5 rerun (est 2.55) + B2-control rerun (est 2.3) alone ≈ fill it (worst-case
  walls 6.5 h would breach by ~1.2). **No B1-real or B4 arm fits the standing
  P2 cap at any N** — every such option needs an orchestrator/user cap
  decision (campaign-wide affordability exists: 29.27 of 100 used). Full
  costed matrix incl. N-reduction ladders, the S6b problem (its 17–20 h cost
  is N-independent), and the OFF-PROTOCOL two-stage-seeded B4 flag:
  research/p2_wave1_postmortem_redesign.md §5.
- **Redesign PROPOSED (not executed):** (a) transplant the accepted
  10_anchor_arbitration.py pattern into 23_run_p2_scene.py (delegating
  LoggingKernel: per-stage z checkpoint + progress print + wall-cap exit-3
  PARTIAL protocol; + weight-step ll recording through the existing
  keyword-only loglik_batch_fn seam so resumed runs keep full logZ/bootstrap —
  B1's ≤2-nat logZ gate needs that); mirror into 24_run_p2_oldstack.py for B4;
  (b) the RC3 one-liner (`res.pop("kernel")`); (c) ops rules: pilot-first est
  calibration on new target classes; ledger est := wall for non-checkpointed
  jobs; runner write-path smoke test (the 54-test suite covers the driver,
  not runner summary assembly — the gap RC3 lived in). NOTHING submitted, no
  code changed, no est rows appended this session.
- **Observed in passing (P3 front's cell, diagnosis owed there):** 56004205
  l0arb FAILED in 24 s — guards.require_vendor_ref raised because `import
  gigalens` resolved to the CFS campaign vendor (the cgl2-pm venv's editable
  install) while the job executed from the audited $PSCRATCH tree copy
  (deploy.md5 215/215 PASSED in that copy); a path-identity trip of the
  copy-tree execution pattern, not a wrong-code import (both trees are
  audit-identical). Actual 0.01 h ledgered; registration LEFT IN PLACE as the
  P3 front's reminder.
- **Housekeeping:** watchdog registrations 55985444–451 deregistered (all
  terminal, harvested/diagnosed here); 56004205 + phoenix 273079 left;
  data/WATCHDOG_ALERT deleted after triage (9 alerts = exactly this harvest's
  designed reminders + the l0arb failure, all triaged above); heartbeat
  touched. L0-G2 gate row + 55985451 actual folded into this ledger (numbers
  were previously only in commit 1993cc6's message). NOT committed (house
  rule: user commits); NOTHING submitted.

### 2026-07-16 (late) — P3-L0 ANCHOR ARBITRATION: phoenix L4 leg wall-capped at λ=0.418 PER PROTOCOL (not a failure); FRESH Perlmutter run 56004205 submitted per the pre-registered fallback (est 2.5 A100-h, ledger row above BEFORE results)

- **L4 completion-leg outcome (data/l0_arbitration_smc.json, status
  PARTIAL_WALL_CAP):** leg 3 (resumed from stage 26) ran stages 27–67 and hit its
  pre-registered 12 h wall cap (wall_h_resume_leg 12.26) at λ_reached = 0.4183,
  N=64, seed 2. This is the runtime protocol executing AS WRITTEN
  (research/checkpoints_l0.md: "On cap: keep the newest per-stage particle
  checkpoint, report lambda_reached + PARTIAL posterior, NO gate math on a
  non-λ=1 ensemble") — sampler health through stage 67 stayed clean (accept
  0.85–0.94 in the endgame; NO γ numbers read). The L4 f64 endgame cost
  ~20–26 min/stage (t_tune+t_mutate 1222–1573 s over stages 65–67), so λ→1 was
  out of reach in any sane L4 envelope — a device-class finding, not a sampler
  one.
- **Checkpoints RETAINED as cross-check (68 files, stage_000–067,
  data/l0_smc_checkpoints/):** md5 stage_000 0a9c9638258e78d7509c039a535273e3,
  stage_026 74d9bb41fc84bf7f673e61dc46e9509c, stage_067
  5128615550e840e9c5b3fa391b3cb330 (z (64,46), λ=0.4501, eps 0.0934, n_int 64);
  aggregate digest (md5 of the sorted 68-line per-file md5 list)
  e7ca59d48fba2e975c4fc81f1678094c. Any future λ≤0.45 diagnostic can be
  cross-checked against these ensembles.
- **Fresh Perlmutter run (Front B's pre-registered fallback) SUBMITTED:
  job 56004205** (cosmo_g shared QOS, 1×A100, -t 04:00, plain `-C gpu` on
  purpose — the L4-measured ~93 MB/particle irreducible v2d diag grad tape ⇒
  N=128 ≈ 12 GB, hbm80g not wasted). Protocol REUSED EXACTLY:
  `10_anchor_arbitration.py run` (frozen per-λ tuning, MC-SMC MAMS,
  PRIOR-SEEDED, seed 2, thresholds/bands UNCHANGED: dominant-basin γ median
  within 2σ_comb of 1.433 [1.400,1.468] + 68% overlap; falsifier disjoint);
  the ONLY deltas are the script's own documented seams — L0_N_OVERRIDE=128
  (the pre-registered 512→256→128 fallback ladder; campaign-standard SMC
  count) and device (A100). L0_REMAT stays UNSET (the remat deviation was an
  L4 memory measure; A100 doesn't need it). L0_WALL_CAP_H=3.5 inside the 4 h
  walltime so a slow run exits via the same wall-cap protocol
  (checkpoint + PARTIAL json + exit 3), never a silent slurm kill. NOTE: this
  is a FRESH run from prior draws (stage `run`, NOT `resume`) — logZ will be
  quotable again, and the L4 checkpoints stay untouched on phoenix (remote has
  no l0 checkpoints; the job runs in an audited $PSCRATCH tree copy).
- **Ops:** job executes from a $PSCRATCH copy of the audited tree
  (campaign + foundry-i + claude-giga-lens siblings), `md5sum -c deploy.md5`
  IN THE EXECUTED COPY (215/215; manifest updated this session — see the B1
  entry below), hot I/O on $PSCRATCH, results copied to CFS
  results/l0arb/ (json always; npz only on COMPLETE — a PARTIAL run therefore
  fires the watchdog artifact alert by design). data/l0_sanity_report.json
  (md5 5569fc1be3a3ae20b54d160abfe13bec, sanity_ok=true from the 2026-07-15
  phoenix wiring gates S1/S2/S3/S4 — device-independent) staged to CFS
  campaign/data/ and added to the manifest; stage_run refuses to start
  without it. Watchdog: 56004205 registered (max_pending 24 h / max_run 4.5 h /
  expect_artifact CFS results/l0arb/l0_arbitration_smc.npz / on_stall
  resubmit:p3_l0arb.slurm), heartbeat touched, loop alive (PID 118755).
  **NO results read this session** (harvest = plots FIRST then the unchanged
  pre-registered gates via `10_anchor_arbitration.py harvest`). Budget: P3
  rows now 2.0 (l0g2) + 2.5 (this) = 4.5 est of cap 17. NOT committed (house
  rule: user commits).

### 2026-07-16 (late) — B1 REAL DATA LANDED: cutouts arrived + builder extended + smoke PASS + staged to CFS; **the 3 real-data arms are STAGED READY BUT NOT SUBMITTED — P2 budget-overflow flag to the orchestrator (never self-authorized)**

- **THE MISSING PIECE ARRIVED:** the team's real MUSE cutouts are now on phoenix
  at `newnewcutouts/` (gitignored; exactly what the B1 design checkpoint named:
  source4-5.fits + source9.fits with DATA/STAT/PSF/MASK). linusu's Perlmutter
  copy remains permission-blocked (re-verified this session). Verified content:
  300×300 DATA/STAT/MASK + 19×19 PSF (sum=1.000000) per file; STAT strictly
  positive (min 0.384 / 1.058); masks keep 89350/90000 and 89550/90000; md5
  source4-5 4a4f7c0216450e0bad49a7f1c9c3b4ea, source9
  220b5677ffb802f87c9b978a9380eca9. UNPUBLISHED team data: never committed,
  staged only inside the campaign CFS area (D4/D6 honored — results to the
  team first).
- **Builder extension (cgl2/zoo.py, the only code change):**
  `build_carousel33` gains a REAL-data mode selected by `data_dir` arg or env
  `CGL2_CAROUSEL_DATA` — loading conventions VERBATIM from
  GIGALens-Code@eb2a09b6 build_model.py::dataset_from_dir (err = sqrt(STAT),
  per-dataset FITS PSF kernel, bool mask; delta_pix 0.2, num_pix 300, ss 1,
  likelihood f64 / conv f32). **RAISE-NEVER-DEFAULT:** once real data is
  requested, ANY missing/ill-formed input raises (missing file/extension,
  non-finite DATA, non-positive STAT, empty mask, wrong shapes) — no silent
  fallback to mock. The MOCK path is byte-for-byte unchanged (the RUNNING…
  now ended… mock arms stay the declared mock-arm record; registry default
  still mock). FITS md5s land in target provenance + meta (run jsons carry
  them). Runner 23_run_p2_scene.py UNTOUCHED (env seam only — the b1r slurm
  clones differ from the mock arms ONLY in CGL2_CAROUSEL_DATA + names, as
  tasked).
- **Phoenix smoke (FREE A16 GPU 0, CUDA_DEVICE_ORDER=PCI_BUS_ID; GPU 8 left to
  B3, GPU 9 left free for B3 spillover; scratchpad b1r_smoke.py): PASS** —
  raise-never-default probe raises FileNotFoundError on a bad dir (no mock
  fallback); real build 2.2 s, dim=32; 8 prior draws (seed 2): logprior /
  loglik / native ALL finite (loglik range [−363246.04, −347010.02]);
  **adapter-vs-native ProbModel.log_prob parity max|d| = 0.0** — the same
  exact-zero the mock pre-flight recorded, now on the real cutouts. (Known
  cosmetic: tfp WeakStructRef teardown KeyError at exit + the vendor shear
  plausibility UserWarning — both in-family with the mock arms' logs.)
- **Staging + audits:** source4-5.fits + source9.fits →
  CFS `code/campaign/newnewcutouts/` (md5 VERIFIED both sides, values above);
  updated cgl2/zoo.py, data/l0_sanity_report.json, and 4 new slurm files
  (p2_b1r_s1_seed2/p2_b1r_s1_seed3/p2_b1r_s6b + p3_l0arb) staged; deploy.md5
  regenerated to **215 files — self-check PASS locally AND remotely
  (215/215)**. Every b1r job additionally echoes the cutout md5s at start.
- **WHY THE 3 REAL ARMS ARE NOT SUBMITTED (deviation from the tasking,
  declared; the tasking's own overflow clause governs):** the tasking
  (+7.0 est: S1 seeds 2+3 @2.5 + S6b @2.0, P2 16.2→23.2 of cap 24) assumed the
  mock arms were healthy and RUNNING. Observed at session start (sacct, ops
  metadata only — no results read): **all three mock B1 arms hit their
  walltime — 55985444 TIMEOUT 04:00:18, 55985445 TIMEOUT 04:00:23, 55985446
  TIMEOUT 03:00:26 — with ZERO artifacts** (23_run_p2_scene.py writes
  npz/json only at completion; nothing on $PSCRATCH or CFS but run logs).
  Ops evidence from the logs (health lines only): the s6b log shows MAP done
  in 302 s but MAMS rounds at >6 min/round ⇒ 150 rounds ≈ **15 h ≫ the 3 h
  wall**; the s1 logs contain no progress lines at all (stock driver prints
  only at completion; last write ~17 min in). (Also observed in passing, other
  fronts' cells, left to their harvests: 55985447/48 TIMEOUT, 55985449/50
  FAILED, 55985451 l0g2 COMPLETED with artifacts.) Honest arithmetic for
  same-pin real arms: expected burn = the full walls, 4.0+4.0+3.0 = **11.0
  A100-h for near-zero artifact probability**, i.e. P2 est 16.2+11.0 = 27.2 >
  cap 24 — and the mock arms' own actuals (to be harvested) already exceed
  their est rows. That is an OVERFLOW + redesign decision, which this front
  is explicitly NOT authorized to make. **Held: no sbatch, no ledger rows
  (the +7.0 est was NOT appended — rows are appended at submission), mock
  arms NOT cancelled (they ended on their own walls; their watchdog
  registrations left in place — the TIMEOUT alerts double as the harvest
  reminders).**
- **Ready-to-fire state for the orchestrator:** slurm/p2_b1r_s1_seed2.slurm
  (md5 c98276fdb76f81620648566c8787f68e), p2_b1r_s1_seed3.slurm
  (96a03c402b93f48691830efa51053e3f), p2_b1r_s6b.slurm
  (9b18560645630e07d89484efd4f998a7) — staged on CFS, audit-covered,
  identical to the mock arms except CGL2_CAROUSEL_DATA + names, each carrying
  a loud SUBMISSION-HELD header. Decision points flagged (options, not
  choices): (i) walltime ≥ ~2× S1 / ~5–6× S6b (needs new est rows ⇒ P2 cap
  decision), (ii) N reduction (leaves the declared N=512 design), (iii) a
  progress-print/checkpoint retrofit to the runner BEFORE burning more wall
  (the runner's write-only-at-end behavior is what turned 11 GPU-h into zero
  artifacts), (iv) rescope B1. The real cutouts + builder + audits will be
  ready whichever way the call goes.
- **Comparison-row record (tasked question — are their carousel POSTERIOR
  artifacts readable in GIGALens-Code?): 'summary-stats only'.** The local
  mirror's sim_carousel pipeline outputs (messy_tests/*/{map,mams,mclmc,
  diag_qz}/) contain ONLY manifest.json files: the manifests declare array
  payloads (e.g. "arrays": ["samples_z"]) but ALL *.npy/*.npz are globally
  gitignored (GIGALens-Code .gitignore:48–49) and ABSENT from the mirror;
  file-type census of experiments/sim_carousel = {py, png, json, log, txt,
  sh, md, ipynb} — no array formats at all. The why_hard_to_sample
  results_carousel JSONs are percentile/summary tables, not draws. Their
  Perlmutter originals (linusu home) are permission-blocked. ⇒ The B1
  comparison row vs their carousel runs can only be built at
  summary-statistics level unless the team also transfers posterior arrays —
  recorded as the next precise missing piece for a draw-level comparison.

**Wave scope (tasking of record):** B1 carousel33 (S1 ×2 seeds + S6b), B2 dspl20
orig+ratio, B4 T2 foundry_marg46, B5 T3 foundry_v3b74 (S1 per-basin + S2 MCLMC arm),
P3 L0-G2 scene-API v3b correlated refit. NOT this wave (pre-declared): **S7
flow-MAMS** (prep note in the B1 checkpoint below) and **S4/S5 LAPS arms** (their
unpublished research code, not deployed — bright line §8.2; stated honestly: this
benchmark's baselines are S6b and published-class kernels only). Budget: est 18.0
A100-h this wave — P2 rows 16.0 (cap 24; +0.2 deploy-verify already committed),
P3 row 2.0 (cap 17). Campaign after this wave: 10.01 actual + 18.2 est of 100.

**New code this wave (runners call the FROZEN drivers with FROZEN constants; no
frozen-protocol numerics touched):** `23_run_p2_scene.py` (B1/B2 arms),
`24_run_p2_oldstack.py` (B4/B5, Path B), `25a_export_v3b_basin_x46.py` +
`25_run_l0g2_scene.py` (L0-G2), `slurm/p2_*.slurm`, `slurm/p3_l0g2.slurm`.
**One documented driver deviation (T1.1 keyword-only precedent):**
`cgl2/samplers/common.py::run_tempered_smc` gains keyword-only
`loglik_batch_fn=None`; default None reproduces the validated full-vmap weight
step BIT-FOR-BIT (54-test suite re-run green; toy bit-identity stock-vs-chunked:
max particle diff 0.0, identical logZ digits). Needed because the B1 weight step
at N=512 does not fit any A100 (measured below).

**PRE-FLIGHT (login A100-40G, this session — wiring checks, no results read):**
- Scene adapters under cgl2-pm: carousel33 / dspl20_orig / dspl20_ratio ALL build;
  logp finite at 8 prior draws; **adapter-vs-native ProbModel.log_prob parity
  max|d| = 0.0** (all three) — the B0 parity gate content reproduced on PM native.
- **Carousel sizing (B1):** vmapped value_and_grad **837 MB/particle** (8→16
  slope); forward loglik ~158 MB/particle (@128 requested 20.2 GB, OOM on 40 GB).
  ⇒ N=512: weight step ~81 GB > 76 GB usable on hbm80g ⇒ mutation chunk 32
  (26.8 GB) + **chunked weight step chunk 64** (~10 GB transient) via the
  loglik_batch_fn seam. dspl fits stock at chunk 128.
- Path B (OLD venv + PYTHONPATH): T2 built 73 s, logp finite at 4 prior draws
  (f64); T3 built 55 s, q_k fits reproduce the 45/3 chain split, logp+logq finite
  at 2 q draws per basin, gamma(q draws) low {1.289,1.309} / steep {2.408,2.477}
  (f32, GIGALENS_X64=0). Ops catch: cgl.zoo.runtime DEFAULT_GPU="9" (phoenix)
  breaks any process where slurm/user leaves CUDA_VISIBLE_DEVICES unset — all 8
  slurm scripts gained `CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}`.
- L0-G2 step 1 executed (login, OLD venv, CPU, bijector-only):
  `25a_export_v3b_basin_x46.py` → CFS `data/l0g2_basin_x46.npz` (md5 e5daca7413…),
  low 2400 draws γ_med 1.0997 (source e2_v3b_low_canary_svicov.npz md5 551be806…),
  steep 4096 draws γ_med 2.6696 (source e2_v3b.npz md5 1d2ee2c4…, == local git).
- Cross-stack mapping validated on phoenix CPU beforehand: x46→scene-z q fit
  round-trips gamma to 4e-5 at the v3b z_ref; scene correlated logp finite there.

**MD5 audits (recorded):** campaign deploy.md5 **208 files PASS both sides**
(includes the 4 new runners, 8 new slurm files, and
`../claude-giga-lens/data/whitener_v3b.npz` staged for L0-G2, md5 4242fd98…);
OLD exec tree **71/71 CLEAN** vs local git (the T0.2/T1.1 list);
foundry-i `_data_lib.py` identical (00d75458…); **B4/B5/L0-G2 data inputs 12/12
identical** local↔PM (svi_v12_v3br, hmc_v13_v3b, map_v11_v3b_{cold,cold2d,warm},
map_marg_pd, hess_marg_pd, cutout_v3b, cutout_F140W, empirical_psf,
long_diagraw_s0/s7 spot pair) + e2_v3b.npz fit npz identical; every job runs
`md5sum -c deploy.md5` before touching the GPU.

---

**B1 DESIGN CHECKPOINT (carousel33; 8 A100-h cap, est 7.0; cgl2-pm venv):**
- **DATA DECLARATION (honesty rule, governs every B1 claim):** the real MUSE
  cutouts remain permission-blocked (re-verified this session:
  `/global/u1/l/linusu/GIGALens-Code/experiments/sim_carousel/newnewcutouts/`
  Permission denied). B1 therefore runs on zoo.py's documented SEEDED MOCK
  stand-in (their model/priors VERBATIM; data = in-model render + noise, seed 33).
  The posterior geometry is NOT theirs: this cell certifies sampler MACHINERY,
  self-consistency, and cost accounting on a carousel-CLASS multi-plane lstsq
  system. Any "hardest ESS-limited system" benchmark claim is DEFERRED until the
  team transfers the cutouts (memo request stands — the precise missing piece:
  newnewcutouts/source4-5.fits + source9.fits with DATA/STAT/PSF/MASK).
- **Arms:** S1 prior-seeded MAMS-SMC N=512, seeds {2, 3} (mutation chunk 32,
  weight-step chunk 64 — execution order only, bit-identity proven); S6b
  MAMS-alone-warm ×1 (seed 2): MAP multistart 64×350 optax.adabelief(1e-2) (their
  pipeline default, replicated w/ attribution — the vendored inference.py MAP is
  broken under jax 0.6.2, ledgered defect E3) with **MAP cost billed in the grad
  ledger**, then 64 chains from the best MAP endpoints, ensemble-preconditioned
  MAMS at λ=1 (same preconditioning class as the SMC kernel ⇒ the comparison
  isolates tempering+resampling): 30 burn rounds (re-tune+re-precondition) + 120
  frozen sampling rounds × 4 draws.
- **Gates (PLAN §6 B1 verbatim, harvest-phase):** self-consistency 2 seeds,
  worst-param diff < 0.27σ at ESS≥128; logZ repeatability ≤ 2 nats; efficiency
  ESS/10⁶ grads vs S6b: WIN ≥2×, PARITY [0.7,2), LOSS <0.7; cold-start gate:
  prior-seeded S1 width-ratio vs the S6b warm reference ∈ [0.7,1.4] and |z|<3.
  conv=float64 (x64 process).
- **Pre-registered honest prediction:** parity-to-3× (S1 does not WIN raw
  efficiency on a unimodal-ish mock); S1's structural value here is logZ +
  cold-start. **Falsifier:** unique particles < N/4 = 128 at λ≳0.8
  (rotating-ridge geometry defeating affine per-λ preconditioning).
- **S7 PREP NOTE (not this wave, pre-declared):** minimal flow-preconditioned
  MAMS needs a flow library absent from the certified cgl2-pm pins; the minimal
  compliant build is a tfp-0.25-bijector IAF/RealNVP preconditioner trained on
  the S1 λ=1 ensemble (keeps Path A pins untouched — deployment-plan item 3),
  wired as a bijector wrap of the MAMS logdensity; anything heavier (flowjax) is
  a ledgered dep decision. S7 runs AFTER B1 artifacts exist (its comparison arm
  needs them); results to the team first (D4).
- **S4/S5 OMISSION (bright line, stated plainly):** LAPS warm/cold arms are NOT
  run — their unpublished research code is not deployed here and stays internal
  reference only per PLAN §8.2. The benchmark's baselines are S6b (classic
  MAP→warm-MAMS) and published-class kernels.

**B2 DESIGN CHECKPOINT (dspl20_orig + dspl20_ratio control; 2 A100-h cap, est 1.5):**
- **Hypothesis:** prior-seeded MAMS-SMC in the ORIGINAL pathological
  (Om0,w0)+NormalCDF coordinates recovers the correct arm mass without their
  bespoke exact reparameterization — per-λ ensemble preconditioning + tempering
  handles the ridge their LAPS/MCLMC runs needed ratio coordinates for.
- **Design:** one job, both targets sequential (same GPU ⇒ same seeded-mock
  realization class; their own generation recipe, data_seed 0 — their baseline
  realization is unreproducible per their note, recorded in zoo provenance);
  S1 N=512 seed 2, chunk 128, stock weight step.
- **Gates (PLAN §6 B2 verbatim):** |m̂(Om0<0.146) − 0.103| ≤ 0.045 on dspl20_orig;
  control dspl20_ratio must reproduce their Run A r2 ≈ N(1.32417, 6.7e-4)
  (weighted r2 = u_fn(Om0,w0) evaluated at harvest via the copied ratio-coords
  module). Falsifier: orig-coords arm mass outside the band while the control
  passes ⇒ the sampler does NOT fix the coordinate pathology (honest negative row).
- Both runs also gate on SMC sanity: λ=1 reached, unique particles ≥ N/4, finite
  logliks (fail-loud in the driver).

**B4 DESIGN CHECKPOINT (T2 foundry_marg46; 3 A100-h cap, est 3.0; Path B):**
- **Hypothesis (PLAN §6 B4):** per-λ ensemble preconditioning replaces the
  two-stage recipe on the cond~1e14 real marg posterior — prior-seeded SMC (no
  warm start, no Laplace metric) lands on the P2c reference posterior.
- **Design:** prior-seeded S1, N=256, float64, seed 2, MAMS chunk 64, stock
  weight step; target = the P0-gated 46-dim marg model (parity logp
  −45840.984005998456 provenance in the zoo builder); inputs verified present +
  md5-identical on PM (map_marg_pd, hess_marg_pd, cutout_F140W, empirical_psf,
  long_diagraw_s0..7).
- **Gates (verbatim):** worst-param |z| < 3 vs the P2c reference; ESS ≥ 0.3N
  (=77). Falsifier: resample-collapse (T1.1-control class: unique ≪ N/4) or
  worst-param blowout ⇒ the two-stage recipe stays necessary on brutal
  conditioning — reported as such.

**B5 DESIGN CHECKPOINT (T3 foundry_v3b74; 6 A100-h cap, est 4.5; Path B):**
- **DTYPE DECLARATION:** the T3 target is strict-float32 BY CONSTRUCTION
  (assert_dtype_env: x64 would silently promote the tfp prior constants off the
  stored-chain posterior). Both jobs run GIGALENS_X64=0; the MH-acceptance f32
  noise floor on this diagonal likelihood is O(1e-3..1e-2) nats — recorded as a
  caveat on ΔlogZ at the ~0.01-nat level, irrelevant at the gate's 5.36-nat floor.
- **Design:** basin-local q_k → posterior anneal, EXACTLY the
  24_basin_evidence_v3b (P2c) evidence construction (q_k from the stored
  hmc_v13_v3b chains split at γ=1.8 by chain mean — 45 low / 3 steep; n_fit 8000,
  cov_inflate 2.0, guard-floored; logprior:=log q_k, loglike:=log_prob−log q_k ⇒
  logZ_k = basin-restricted evidence), mutation kernel swapped to OURS through
  the generalized frozen driver. Job 1 (S1): MAMS low@128 then steep@96 (separate
  processes). Job 2 (S2, B5-G3): MCLMC low@128 — MCLMC is DEMOTED
  (B0 bias screen 30σ) — this arm is DIAGNOSTIC BY DESIGN, never evidence.
- **Gates (PLAN §6 B5 + finalized thresholds):** both basins converged at λ=1
  with basin retention (frac_gamma on the right side of 1.8; the harvested
  mode weight w_low = σ(logZ_low−logZ_steep) within binomial agreement of the
  P2c mode-weight reference); basin ΔlogZ within **3·√(σ_boot² + 1.79²) nats
  (floor 5.36)** of the P2c reference (the P1-finalized σ_seed row); **B5-G3
  pre-registered prediction:** the MCLMC-mutation arm distorts the minor-mode
  weight by 1.5–3× (resampling does NOT launder unadjusted-kernel bias);
  falsifier of G3: MCLMC ΔlogZ within the MAMS repeatability band ⇒ resampling
  DOES launder the bias (a genuinely useful negative).
- **Deviation note:** PLAN's B5 modal wording ("minor-basin occupancy") maps to
  the per-basin-evidence construction exactly as in P1c/P2c — occupancy is
  DERIVED from per-basin logZ, not from a single prior-seeded run (same
  convention as the certified diagonal basin-evidence linchpin).

**L0-G2 DESIGN CHECKPOINT (P3 budget; ~2 A100-h est 2.0; cgl2-pm venv, hbm80g):**
- **Gate (PLAN §6 P3 + finalized σ row, verbatim):** the scene-API v3b-low
  CORRELATED refit reproduces γ_binned(corr,low) = 1.1032 within
  2√(σ_stat²+σ_seed²) = **0.017**, AND the low-basin logZ preference keeps its
  sign (ΔlogZ(steep−low) < 0; old-stack per-matched-seed value −28.9). L0-G2
  licenses all X1-class real-lens claims on the new substrate. Falsifier:
  γ outside the band or sign flip ⇒ the ported likelihood is NOT
  substrate-faithful at posterior level ⇒ P3 payload stays old-stack (PLAN kill).
- **Design:** production sampler recipe mirrored parameter-for-parameter from
  cgl.e2.run_correlated_smc (HMC mutation 4×8 @ step 0.1, metric = q cov,
  target_ess 0.7, max 400; the VERBATIM driver copy run_adaptive_tempered_smc) —
  sampler held fixed so the cell isolates the SUBSTRATE question. Likelihood =
  build_pm('v3b', diagonal=False): parity-certified marg scene model +
  CorrelatedImageData with ported whitener_v3b (checkpointed whitening), parity
  conventions applied. Legs: low@128 (γ gate) then steep@96 (logZ-sign
  comparator), seed 2. Memory: 128×367.6 MB ≈ 47 GB ⇒ hbm80g pin.
- **Documented deviation (cross-stack q hand-off):** the production q lives in
  OLD-stack z; the scene bijector differs, so q is REBUILT in scene-z from the
  production q-fit draws exported in CONSTRAINED x46 (25a artifact above),
  mapped via param_map, Gaussian-refit with the production cov_inflate 3.0 +
  guard floor. q only defines the λ=0 reference of the basin anneal — logZ_basin
  and the λ=1 posterior are q-independent in exact arithmetic; declared, not
  silent. Inputs: l0g2_basin_x46.npz (e5daca74…), whitener_v3b.npz (4242fd98…),
  cutout_v3b.npz (a91ad318…), parity_refs.npz (manifest).

**Ops:** all 8 jobs pin `#SBATCH -C gpu&hbm80g` (SMC-wave rule), shared QOS 1×GPU
on cosmo_g, `md5sum -c deploy.md5` before GPU work, hot I/O on $PSCRATCH with cp
to CFS results/ per step, watchdog registration at submission, **no results read
this session** (harvest is a later pre-registered phase; pre-flight logp/sizing
numbers above are wiring checks, and the 25a export prints only warm-start
PROVENANCE quantities, T1.1 precedent).

**SUBMITTED 2026-07-16 ~10:40 PT (all 8, cosmo_g shared QOS, ledger rows above,
checkpoints above written BEFORE sbatch):** 55985444 (b1-s1-s2), 55985445
(b1-s1-s3), 55985446 (b1-s6b), 55985447 (b2-dspl), 55985448 (b4-t2), 55985449
(b5-s1), 55985450 (b5-s2-mclmc), 55985451 (l0g2). All 8 registered with the
watchdog (max_pending 24 h / max_run = walltime+0.5 h / expect_artifact = CFS
result npz / on_stall resubmit:<CFS slurm path>); loop verified alive (PID
118755). NOTE (fail-loud by design, P1 pattern): expect_artifact paths are CFS
paths checked on phoenix's filesystem, so COMPLETION raises
COMPLETED_NO_ARTIFACT until the harvest session pulls results and deregisters —
that alert doubles as the harvest reminder. Budget confirmation: P2 rows this
wave 16.0 est (cap 24; cumulative P2 incl. deploy-verify 16.2), P3 row 2.0 est
(cap 17); campaign 10.01 actual + 18.2 est of the 100-h HARD STOP. NOT
committed (house rule: user commits).

### 2026-07-16 — P2 DEPLOYMENT Path A + Path B on Perlmutter (scout plan research/p2_deployment_plan.md executed; verify job 55980038, 0.2 A100-h est)

- **Path A venv BUILT**: `/global/cfs/cdirs/deepsrch/gdbenson/cgl2-linus/cgl2-pm-venv`
  (`module load python` → 3.13.11; login-node pip with `--no-cache-dir`). Wheels-only
  (`--only-binary :all:`) under `-c constraints.txt` for the whole set EXCEPT
  **lenstronomy 1.14.0, which is sdist-only on PyPI** (plan E2's "wheel ✓" was
  release-existence; pure-python build, installed from sdist, trivial). Resolved core
  pins EXACT: jax/jaxlib/jax-cuda12-{plugin,pjrt} 0.6.2, numpy 2.4.6, scipy 1.17.1,
  tfp 0.25.0, blackjax 1.3, optax 0.2.8, objax 1.8.0, astropy 7.2.0, lenstronomy
  1.14.0 (+ arviz 1.2.0, corner 2.3.0, pytest 9.1.1, matplotlib 3.11.0 — figs now
  possible PM-side); **NO tensorflow** (D2 held; objax pulled tensorboard only).
  `pip install -e vendor/gigalens-linus --no-deps` + `pip install -e . --no-deps` OK;
  import smoke: gigalens resolves INSIDE the pinned vendor, jax-pin guard PASS.
- **Code staged non-git**: `cgl2-linus/code/campaign/` (cgl2/ + vendor@80916d2 +
  operator scripts + tests + tools + slurm). **md5 manifest `deploy.md5` (195 files:
  every .py, slurm scripts, pyproject/constraints, parity_refs.npz, whitener manifest
  + campaign whitener npz, ../foundry-i/data/cutout_{v2d,v3b}.npz) verified BOTH
  sides** (local self-check + remote `md5sum -c`: PASS; local copy
  data/deploy_p2_perlmutter.md5). Inputs staged for 01_parity_scene: parity_refs.npz
  → campaign/data/, cutouts → code/foundry-i/data/ (paths.py REPRO_ROOT-relative).
- **Login CPU tests (new venv)**: `pytest tests/` → **54 passed, 25.29 s**
  (CGL2_ALLOW_CPU=1 JAX_PLATFORMS=cpu) — matches the phoenix 54-green suite.
- **Path B PROVEN ON PERLMUTTER** (B4/B5 prep): OLD venv `~/claude-giga-lens/venv` +
  `PYTHONPATH=.../code/campaign` → `import cgl2.samplers.smc_micro` vendor-free
  (gigalens never enters sys.modules); `22_run_b3_vendorfree_test.py` →
  **logZ=−3.5272 true=−3.4813 err=0.0459 sig=0.0550 stages=3 — digit-identical to
  the plan's E1 record from phoenix** (x86 A100 login vs aarch64: same printed digits).
- **Verify job 55980038 submitted** (`slurm/deploy_f8_verify.slurm`, clone of
  parity_f8_nersc.slurm's shape but NATIVE pins: CGL2_F8 UNSET so require_jax_pin
  stays ACTIVE; plain `-C gpu` on purpose — don't burn hbm80g on a light job; -A
  cosmo_g -q shared). Ops lesson: **Perlmutter shared QOS rejects `-c 16` — requires
  32 cores per GPU** (first submit failed; script fixed to `-c 32`, manifest updated,
  re-audited both sides). Runs 00_env_check + the full 01_parity_scene battery
  (F1–F7 restated-F6) on A100 at the campaign pins. Watchdog: registered.
- **No P2 benchmark cells run** (deployment + verification only, per tasking).
  RESULT (appended post-completion): see gate row "PM-A100" below / job log.

### 2026-07-15 evening — T1.1b STOPPED AT THE PRE-DECLARED BUILD GATE (residue-masked refits are information-starved; NO submission, 0 A100-h)

**Task:** fix the T1.1 confound in the FIT — whiten-then-drop the residue-dominated
regions (bright-object + center, from the T1.1 G1 decomposition) so the production
correlated likelihood sees only the sky-dominated field where the injection is clean;
then resubmit inj1/2/3 corr + inj1 diag control. **The tasking pre-declared the
falsifier BEFORE the build: if > 40% of whitened dof would be lost, STOP — the
experiment would be information-starved, itself a finding about injection methodology.**

**Build (09_build_residue_masked_whitener.py, OLD cgl venv, CPU-only, mirrors
05_build_companion_whitener.py):** kernel h / e_op / logdet_per_pix / rho_kernel
copied verbatim (kernel properties, unchanged by construction — asserted); ONLY
keep_w eroded: keep_w_new = erode_keep(keep & ~drop, M=10), asserted equal to
keep_w_old & ~dilate(drop, 21×21) (duality) and ⊆ keep_w_old. Region definitions
PINNED before use (gate R): recomputed residual-field decomposition reproduces the
ledgered numbers exactly — sky 0.952239 (1e-9 vs build report), faint(=|img|<5·med
err, on keep) 0.9733, bright-object(=keep&~faint) 2.7118, center(=keep&r<1.2″)
5.4644 vs quoted 0.973/2.71/5.46. Noted + asserted: center ⊂ bright-object (every
r<1.2″ keep pixel is non-faint), so drop = (~faint)|(r<1.2″) and the bright-only
mask is identical to the union. Production keep_w reproduction gate (erode_keep(keep,
M) == on-disk keep_w) PASS.

**T1.1b-G0 readout (gate record): FAIL — STOPPED.**
- corr keep_w 9273 → **1320**: **85.8% of whitened dof lost** vs the 40% line.
- Attribution: the region drop alone (no 21×21 halo — not a valid correlated
  whitening, attribution only) already loses **47.6%** > 40%; the kernel-support
  erosion adds the rest. No whiten-then-drop variant of this region set can pass.
- Information content, not just count: arc-band (1.2–4.2″) whitened px 6373 → **108**;
  survivors live at r 3.88–6.17″ (median 4.86″) — pure outer sky. The masked corr
  likelihood would carry essentially zero lensed-arc signal; a γ recovery number
  from it would be prior/sky-driven, not a whitener test.
- Diag-control arm alone (delta whitener, M=0, keep & ~drop) would lose 35.9% —
  passes its own budget — but the corr refit IS the experiment; STOP governs.
- Center-only variant would lose 15.2% (informational; the region set is fixed by
  the confound analysis — the bright-object region is where 103% of the residue
  excess lives — so shrinking it would be a goalpost move, not taken).

**Finding (the STOP clause's positive content):** fit-side residue masking is
STRUCTURALLY incompatible with this injection design. The residue lives in the
bright-scene footprint, and the injected synthetic arcs live in (essentially) the
same footprint — masking the residue masks the signal, and the M=10 whitening
support (21×21) dilates the drop over most of what remains. This holds for ANY
region set that covers the bright scene, i.e. for any honest residue mask. The
residue-free injection path must therefore be DATA-side, exactly as the T1.1
implications queued: kernel-sampled noise-only injections / sky-set bootstrap
fields, or a deeper multi-start scene subtraction — now the ONLY viable routes to
a quotable whitener-bias number (P3 seam requirement unchanged).

**Discipline record:** threshold (40%) written before the build (tasking + script
header + report JSON `threshold_provenance`); gate evaluated as written; no
goalpost move. Because the entry gate tripped, the step-2 design checkpoint for
submission was never written and NOTHING was submitted: no slurm/t11b_* files, no
CFS staging, no md5 audits (nothing to audit), no ledger A100-h rows (est 6.0 h
NOT committed), watchdog untouched (loop alive, PID 118755). The planned readout
thresholds (confirm < −0.078; exonerate |median bias| < max(0.026, 3·median
σ_inj); masked diag control predicted unbiased as the residue-story falsifier)
are recorded here for the successor design but were never armed. Cost: CPU-only,
~1 min. Artifacts: data/t11b_residue_mask_report.json (verdict
STOPPED_INFO_STARVED; full dof/attribution/diagnostics),
09_build_residue_masked_whitener.py (re-runnable; exits nonzero on STOP, writes
no bundles). NOT committed (house rule: user commits).

### 2026-07-15 evening — multi-front mobilization (T1.1b + P3-L0 + P2-scout/B3 + X2)
Concurrent-agent discipline: the T1.1b ops agent owns CAMPAIGN.md edits this wave; the
P3-L0 / P2-B3 / X2 agents write their pre-run design checkpoints to
`research/checkpoints_{l0,b3,x2}.md` BEFORE their runs (same their-format content; referenced
here to preserve the checkpoint-before-run property without ledger file contention). Their gate
rows and A100/GPU-h actuals are folded into this ledger at harvest. Phoenix device assignment:
L0 → GPU 9 (L4), B3 → GPU 8 (L4) + A16s 0–3, X2 → A16s 4–7.

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
