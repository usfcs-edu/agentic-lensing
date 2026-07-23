# E2-O4 — The R̂-blindness demonstration: DESIGN CHECKPOINT (frozen BEFORE any run)

Written 2026-07-23, before smoke or production. Reference file per the E2 concurrency
discipline: the O1 agent owns CAMPAIGN.md this wave — this checkpoint is the pre-registered
record for O4 and will be transcribed/linked into the ledger at the E2 harvest.
Plan of record: `/home/benson/.claude/plans/subsequently-xiaosheng-sent-this-spicy-dream.md` §O4.

## Framing bright lines (read first)

- **Service, not critique.** The demonstration is a property of EVERYONE'S two-stage
  reruns — ours included (our two-stage recipe is empirically the same move as Evan's
  rescue). All wording that could go external ships through Greg for review first; O4 can
  be dropped from external artifacts without affecting O1–O3.
- No characterization of the team's unpublished results anywhere in text or figures
  (campaign bright line, held throughout).
- This experiment touches NONE of Evan's data (`odell/` untouched): it runs entirely on
  OUR stored E1 chains. Nothing here quotes a science γ — the v3b posterior location
  remains NO-QUOTE per E1; every γ below is demonstration telemetry.

## Context

Evan's rescue for "very migratory" first-stage MCLMC chains: precondition a second MCLMC
on the last ~half of the first; his stage 2 converged R̂ ≪ 1.01 (~8–16 chains,
~5000+5000). We hold, on disk, a first stage of exactly the kind his rescue is applied
to — the E1 v3b run, where all 64 chains (4 independent seed groups) were warm-started in
the physical basin (γ ≈ 1.29) and every one migrated during burn-in to the low-γ shelf
(E1-G2 FAIL 64/64; stage-1 R̂_worst 1.379, min-ESS 139) — plus a healthy first stage
(E1 v2d: R̂_worst 1.0031, min-ESS 30,334, γ = 1.4683 [1.4343, 1.5048]) as the control.

Stage-1 tail facts (recomputed from the stored npz for this checkpoint, reads only):

| arm | stage-1 chains | tail (last 50%) γ med [q16, q84] | per-chain tail medians | nearby references |
|---|---|---|---|---|
| v3b | 64 (4×16) | **1.1043** [1.0893, 1.1198] | 1.0807–1.1183 | corr-low 1.1032±0.0086; band edge 1.15 |
| v2d | 64 (2×32) | **1.4684** [1.4342, 1.5053] | 1.4644–1.4722 | E1 quotable 1.4683; anchor 1.4330 |

## Hypothesis

A stage-2 MCLMC preconditioned on a migrated stage-1 tail converges cleanly AT the
migrated location: stage-2 convergence diagnostics cannot detect stage-1 basin selection.

## Predictions (pre-registered)

- **v3b arm (the demonstration):** stage-2 R̂_worst ≪ 1.01 with γ settling at the
  migrated shelf — point prediction γ_med ≈ 1.10 (within ~0.02 of the stage-1 tail median
  1.1043), i.e. pristine convergence metrics on the solution the PI calls unphysical.
- **v2d arm (the control):** the identical recipe on the healthy product converges with
  γ_med ≈ 1.47 (within 1σ_comb of E1's 1.4683 [1.4343, 1.5048]) — the recipe works fine
  when stage 1 lands physically; the property being demonstrated is stage-1-conditional.

## Demonstration criteria (frozen; thresholds = the frozen E1-G1 numbers, no new knobs)

- **O4-D1 (blindness shown, v3b):** stage-2 R̂_worst < 1.01 AND min-ESS ≥ 1000
  (arviz rank-normalized split-R̂ / bulk ESS over 16 chains × 5000 draws, worst-param over
  46 scene-z + 8 physical mass params — identical construction to E1-G1) AND stage-2
  γ_med < 1.15 (still outside the E1-G2 band, at the shelf).
- **O4-D2 (control, v2d):** same convergence thresholds AND γ_med within 1σ_comb of the
  E1 v2d posterior (σ_comb from the two CI68 half-widths, E1 side 0.0353).
- **Falsifiers — each reported AS LOUDLY as a pass:**
  - F1: v3b stage-2 FAILS convergence (R̂_worst ≥ 1.01 or min-ESS < 1000) ⇒ the low-γ
    shelf stays glassy even under tail preconditioning ⇒ NO blindness demonstration —
    the diagnostics DO flag something. (Weakens the demo, not the E1 migration finding.)
  - F2: v3b stage-2 γ_med ≥ 1.15 (returns toward 1.29) ⇒ **weakens our migration
    interpretation itself** — escalate prominently to the E2 synthesis.
  - F3: v2d control fails to converge ⇒ the recipe-parallel claim is undermined; the demo
    is inconclusive and is reported as such.
- **Escalation: NONE pre-authorized.** This is a demonstration at Evan's stated scale;
  whatever happens at that scale is the result. Descriptive extras recorded either way:
  E1-G2 band [1.15, 2.0] containment fraction per stage-2 chain (report, never drop),
  and a 2×8 split-half R̂/ESS readout (chains 0–7 vs 8–15) for extra rigor at zero cost.

## Implementation (reuse maximal; vehicle `51_run_rhat_blindness.py`)

**Stage-1 tails** (pure numpy, deterministic): from `data/mclmc_diag_{v3b,v2d}.npz`
(md5s recorded in the run json), per chain take the last 2000 of the 4000 kept draws,
pool over ALL chains and seed groups → (128,000 × 46) scene-z tail per arm.

**Preconditioning — exactly the Evan-described move; stage 2 sees ONLY stage-1 output**
(the E1 warm x46 cloud is never touched):
- init positions = 16 rows drawn without replacement from the pooled tail
  (rng: v3b 1447, v2d 1448);
- inverse-mass init = `cgl2.samplers.common.regularize_cov(cov(pooled tail), n=128000)`
  — the same code path E1 used for its warm cloud; at n = 128,000 the shrinkage weight
  n/(n+5) ≈ 1, so this is PSD-flooring hygiene on the tail's own covariance;
- svi_mean = pooled tail mean.

**Stage-2 sampler = byte-identical E1 machinery**, imported from `50_run_mclmc_diag.py`
(`_mclmc_run` / `_build` / `_apply_vma_workaround` / `_mass_of` / `_hist_np`): vendored
`full_mclmc_with_adapt_sharded` + `isokinetic_mclachlan_smart`, MCLMC_JIT bypassed, VMA
check_vma=False runtime rebind (ledgered E1 amendment; single-device mesh), f64,
desired_energy_var 5e-4, num_effective_samples 100, windowed_mass_matrix True,
regularize_mass_matrix True (E1 amendment 2), psmile False, init_L √46,
init_step_size 0.25·√46, svi_mass_matrix_weight 10·16 = 160 (the wrapper's own rule, as
in the E1 v3b 16-chain groups; set via MCLMC_CHAINS=16 before import).

**Scale (Evan's stated scale): 16 chains, 5000 burn (frac_tune 0.2/0.6/0.2) + 5000
draws, ONE seed group per arm** (rng_key: v3b 230723, v2d 230724); R̂/ESS computed over
the 16 chains; the 2×8 split-half readout provides the extra-rigor split without extra
compute. Targets = the parity-certified DIAGONAL scene-API likelihoods via
`10_anchor_arbitration.build_pm(tag, refs, diagonal=True)` — identical to E1.

**Smoke (per arm, foreground, REQUIRED PASS before the lane launches):** 8 chains ×
300 burn + 100 draws from the same tail preconditioning (E1 amendment-2 geometry: every
mass-matrix window ≫ 46 samples; rng 991 / init-row 992). Gates: all draws finite,
kernel_nonan ≥ 0.99, nonan ≥ 0.99, γ draws inside a loose sanity window (0.8, 2.5) —
NOT the E1-G2 band, which the v3b tail violates by construction. Plus the E1 grad-timing
horizon probe at 16 chains; if the projected v3b+v2d total exceeds the 10 h watchdog
horizon, the launch is HELD and the geometry revisited as a ledgered amendment here.

**Cost projection** (from measured E1 walls: v3b 16-chain group 2.29 h / 6000 steps;
v2d 32-chain group 0.75 h / 6000 steps): v3b ≈ 3.8 h + v2d ≈ 1.0–1.4 h ⇒ ≈ 5 h total,
phoenix L4 free tier, 0 A100-h. Memory: E1 v3b 16-chain peak 6.8 GB at 6000 steps;
10,000-step histories add ~1.1 GB device-side ⇒ ~8 GB ≪ 23 GB L4.

## Ops (E1-attempt-3 hardened discipline)

- **GPU 9 ONLY** (asserted in-script; GPU 8 is the O2 lane's, untouched; A16s unused —
  the lensing likelihood never runs on them or CPU).
- Both arms SEQUENTIAL (v3b first) in ONE detached lane process:
  `setsid nohup ... </dev/null >> data/o4_lane.log 2>&1 & disown`, PPID=1 verified,
  `XLA_PYTHON_CLIENT_PREALLOCATE=false`; each arm runs in its own child process so the
  CUDA context is fully released between arms.
- Watchdog: phoenix PID registered, max_run_h 10, expect_artifact
  `data/o4_v2d_stage2.npz` (the LAST artifact in the chain — its existence proves both
  arms completed); note names both artifacts.
- Artifacts: `data/o4_{v3b,v2d}_stage2_smoke.json`, `data/o4_{v3b,v2d}_stage2.{npz,json}`
  (npz: kept positions (16,5000,46) + mass params (16,5000,8) + init rows/indices +
  cov_reg/svi_mean + per-step histories + final inverse-mass; json: config, provenance
  md5s, diagnostics incl. split-half, γ quantiles, containment readout, criteria verdicts
  EVALUATED AT HARVEST ONLY — the run json records the numbers, the verdict language is
  written at harvest, plots first, per house rule).
- NO harvest this session; NO git commit; CAMPAIGN.md / odell files untouched.

## What this buys (for the memo, wording via Greg)

If O4-D1 + O4-D2 hold: a cheap, reproducible demonstration that stage-2 convergence
metrics are blind to stage-1 basin selection — a property of the shared two-stage
recipe (Evan's rescue ≡ our recipe), which (a) explains how an unphysical solution can
acquire pristine R̂, (b) independently supports the PI's physicality-prior practice, and
(c) is a genuinely new, publishable methodological point for the team's sampling paper.
If any falsifier fires, that is reported with equal prominence per the criteria above.
