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

## A100-hour ledger (append BEFORE reading results)

| Date | Job | Phase | Est. h | Actual h | Cumulative |
|---|---|---|---|---|---|
| — | — | — | — | — | 0.0 |

## Gate record

| Gate | Statement | Threshold | Status | Artifact |
|---|---|---|---|---|
| F1–F8 | cross-stack parity battery | PLAN §5 | PENDING | data/parity_report_scene.json |
| B0 | MC-SMC correctness (adapters, mix2/funnel/illcond, MCLMC bias screen) | PLAN §5 | PENDING | data/smc_b0_report.json |
| X1-G0 | profile-curvature mechanism entry gate: r_eff ordering must admit the bracket's sign pattern | monotone ordering exists | **FAIL — hypothesis structurally dead** (24/24 robustness variants non-monotone; fine/binned constrain slope at the SAME radius, Δr_eff≈0.008″ < ¼ px, yet Δγ=0.71 ⇒ would need \|dγ_loc/dln r\|≈226 vs O(1) physical) | data/x1_g0_effective_radii.json, research/x1_g0_mechanism_check.md, figs/x1_g0_*.png |
| Fermat teaser | noise-model Δφ sensitivity (illustrative; NOT a TD lens; synthetic pairs; corr posterior is the known over-correcting product) | report-only | median \|frac shift\| **88%** anchor→corr (10.7σ); same-product diag→corr arm **61%** (17σ) — vs the ~1% TDCOSMO-relevant scale | data/fermat_dt_teaser.json, research/fermat_dt_teaser.md |

## Perlmutter ops

- Remote staging: `/global/cfs/cdirs/deepsrch/gdbenson/cgl2-linus/{code,data,results,slurm-logs}`
  (created 2026-07-15; user-designated disk). Scratch: `/pscratch/sd/g/gdbenson` for hot job I/O,
  results archived back to CFS (their results-storage pattern). Remote is a NON-GIT rsync copy →
  md5-audit every campaign `.py` before production (the stale-remote lesson).
- sshproxy refreshed 2026-07-15 (user). Jobs charge `cosmo_g` (D5), single-GPU shared QOS.

## Stage log (newest first)

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
