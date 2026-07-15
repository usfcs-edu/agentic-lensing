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

## A100-hour ledger (append BEFORE reading results)

| Date | Job | Phase | Est. h | Actual h | Cumulative |
|---|---|---|---|---|---|
| — | — | — | — | — | 0.0 |

## Gate record

| Gate | Statement | Threshold | Status | Artifact |
|---|---|---|---|---|
| F1–F8 | cross-stack parity battery | PLAN §5 | PENDING | data/parity_report_scene.json |
| B0 | MC-SMC correctness (adapters, mix2/funnel/illcond, MCLMC bias screen) | PLAN §5 | PENDING | data/smc_b0_report.json |

## Stage log (newest first)

### 2026-07-15 — P0 open
- Branch `claude-giga-lens-linus` created; plan + engagement memo committed (2f67083).
- Vendor @80916d2 archived (15 MB, UNPATCHED); venv cgl2 built; full import smoke PASS
  (scene API + MCLMC/MAMS kernels + EPL/Shear/BPL/PIEMD/PIEP profiles under jax 0.6.2 CPU).
  Two missing runtime deps found (lenstronomy, objax+tqdm) — installed under constraints;
  gigalens pip metadata complains (numpy 2.1.3, tensorflow) — expected, D2.
- D3 recorded: reference-artifact parity design (gigalens package-name collision).
- cgl2 skeleton: paths.py (vendor bootstrap + jax pin), guards.py (carried + new fences),
  pyproject, x64 bootstrap __init__.
