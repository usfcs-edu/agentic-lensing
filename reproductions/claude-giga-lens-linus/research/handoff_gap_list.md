# Handoff-package gap list — papers/handoff/ vs PLAN §6 P5 promises

**Date:** 2026-07-21 (P5 synthesis open; experimental program COMPLETE, 53.51/100 A100-h, nothing in flight)
**Inventory basis:** `papers/handoff/` on-disk contents + full-tree searches (CAMPAIGN.md, cgl2/, research/, slurm/).

## On-disk inventory of papers/handoff/ (before this task)

| File | Status |
|---|---|
| `ENGAGEMENT_MEMO.md` | EXISTS (P0 deliverable, drafted 2026-07-15; addendum appended 2026-07-21 by this task) |

That was the ONLY file. Everything else P5 promised was undelivered at P5 open.

## PLAN §6 P5 promise-by-promise

| # | Promised (PLAN §6 P5) | Status at P5 open | Substance that exists elsewhere | Remaining work |
|---|---|---|---|---|
| 1 | `CLAIMS.md` in their claims-register format, every verdict `proposed (UNCERTIFIED — external)` | **MISSING → DRAFTED 2026-07-21** (`papers/handoff/CLAIMS.md`, this task, 14 claims) | Gate record + research/*.md checkpoints carried all content | Benson review; graders are theirs |
| 2 | `SMCStage` adapter spec for their `pipeline.py` + tempered-SMC-with-MAMS implementation | **SPEC MISSING** | Implementation EXISTS and is exercised: `cgl2/samplers/smc_micro.py` (tempered SMC, MAMS/MCLMC/HMC mutations, logZ + per-basin evidence), `common.py::run_tempered_smc`, `ckpt.py` (checkpoint/resume, bit-identity gated) | Write the adapter spec (their StagedInference seam), point at the B0/B5/L0-G2 validation record |
| 3 | `CorrelatedImageData` + ridge/Occam-lstsq fix (closes their owed foundry-i item) | **HANDOFF DOC MISSING** | Code EXISTS certified: `cgl2/correlated.py` + `cgl2/marg.py` (F5 5.8e-11, F6 restated-PASS, 03-A 5.5e-12, L0-G2 cross-stack PASS). RFC text is in the memo §3 | Package as handoff doc + the lstsq −½logdetA fix as an upstream-shaped diff |
| 4 | SBC harness adapter | **NOT PACKAGED** (substance essentially complete) | `30_sbc_gift.py` (turnkey run+harvest), `research/x2_sbc_gift.md` ALREADY in their lab-notebook claims-register format, `data/x2_sbc.json`, `figs/x2_rank_hist_*.png` | Copy/point x2_sbc_gift.md into papers/handoff/ (or reference from CLAIMS.md — done) |
| 5 | DSPL note | **MISSING** | Full B2 material: gate record B2 row (orig m̂=0.000 vs control 5/512; band-transfer failure = pre-registered branch 2; dominant-arm agreement; ΔlogZ +3.06 exact-reparam evidence-error datum), `data/b2_gate_eval.json`, `figs/b2_*.png` | Write the short note (B2′ re-registration proposal included) |
| 6 | PR-ready diffs vs 80916d2 | **MISSING — no diff files exist anywhere in the tree** | Vendor deliberately UNPATCHED. Candidate diffs identified: (a) correlated-noise LikelihoodTerm (upstream-shaped already), (b) Occam term in their lstsq, (c) X2 Deviation 1 runtime `_shard_map check_vma=False` jax-0.6.2 workaround, (d) scene_build convention notes (Sérsic b_n) | Produce actual `git diff`-format patches vs 80916d2 |
| 7 | Hessian restore (X-track: restore `HessianSurrogateStage` from b82397c) | **UNDELIVERED — NEVER STARTED.** Zero mentions of "hessian" in the campaign tree outside PLAN.md/ENGAGEMENT_MEMO.md; no code, no ledger row, no checkpoint. The X-track item was planned but never picked up (definiteness triage never became load-bearing; the classic arbitration arm used MAP→SVI→HMC instead) | Either do the ~1-day restore or strike it from the offer list in the memo (addendum discloses the gap) |

## Related P5 items outside papers/handoff/ (for the orchestrator)

- `papers/main.tex` — NOT STARTED (papers/ contains only handoff/).
- REPRODUCTIONS.md row + NEXT_DIRECTIONS delta — NOT DONE.
- Bright-line reminder for whoever packages items 2–6: everything derived from
  gigalens-linus / GIGALens-Code (incl. carousel B1r rows, B2 DSPL rows, X2 frozen-set
  finding) is publication-gated on the team's sign-off (PLAN §8.2, D6). CLAIMS.md marks
  the gated claims explicitly.
