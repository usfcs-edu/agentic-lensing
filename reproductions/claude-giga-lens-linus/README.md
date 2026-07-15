# claude-giga-lens-linus — bridge campaign onto the next-gen GIGALens (scene API)

Campaign (2026-07) building on BOTH the completed `../claude-giga-lens` program AND the Strong
Lens team's next-gen GIGALens (`gigalens-linus @ linusu-dev-merge`, vendored UNPATCHED at
`80916d2`; research patterns from `GIGALens-Code`, read-only). Three pillars + gifts:

- **MC-SMC evidence layer** — tempered SMC with MAMS mutation kernels (prior-seeded,
  logZ-producing, multimodality-capable: the wrapper their own laps-spec names as missing),
  benchmarked on their targets (carousel, DSPL, hundred-systems) and our real zoo (T2/T3).
- **Correlated-noise likelihood as a scene-API `LikelihoodTerm`** — the validated drizzle
  whitening + ridge/Occam marginalization ported to their documented `Dataset`/`LikelihoodTerm`
  seam; the diagonal-limit gate doubles as the first external certification of the scene API.
  Includes the two-stack ANCHOR ARBITRATION (does 1.433 reproduce on their stack?).
- **X1 profile-class fork** — is the EPL single-power-law assumption the residual systematic
  behind the 1.816/1.433/1.103 bracket? BPL/dPIE (their profile library) vs EPL by per-basin
  evidence, compared on the local slope γ_loc(θ_E).
- **Gifts (phoenix)** — first formal SBC of the GIGALens pipeline class; evidence-scored Vela
  ladder prototype; Fermat Δt noise-model sensitivity; HessianSurrogateStage restore.

Plan of record: `plans/PLAN.md` (approved 2026-07-15). Ledger: `CAMPAIGN.md`. Engagement memo
(for Benson to send): `papers/handoff/ENGAGEMENT_MEMO.md`. Perlmutter: cap 100 A100-h,
shared-QOS single-GPU, account cosmo_g.

## Operator quickstart

```bash
source /raid/benson/.venvs/cgl2/bin/activate      # aarch64 phoenix venv (jax 0.6.2 pinned)
python 00_env_check.py                            # asserts pins/vendor-ref/GPU/imports
./00_run_tests.sh                                 # CPU unit toys + GPU parity tests
```

Conventions (house style): numbered `NN_verb_noun.py` operator scripts at root; shared logic in
`cgl2/` (`pip install -e . --no-deps`); bulk artifacts in gitignored `data/`; committed figures
in `figs/`; report in `papers/`. Copy-vs-import: code we modify is COPIED with attribution
(whiten/marg/noise/samplers-common from `../claude-giga-lens/cgl/`); frozen artifacts and both
vendored libraries are IMPORTED BY PATH (`cgl2/paths.py`, which asserts the vendor ref).
The vendored library is UNPATCHED; all mitigations live in `cgl2/guards.py`.

Their-format discipline (adopted voluntarily): design checkpoint (hypothesis + predicted
direction/magnitude + falsifier + derived threshold) logged in CAMPAIGN.md BEFORE every
consequential run; plots before metrics; worst-parameter convergence only; verdicts on their
substrate labeled UNCERTIFIED (external). Bright lines: PLAN §8 verbatim — no publishable
unimodal-efficiency comparisons; nothing derived from their unpublished repos leaves this repo
without the group's sign-off; Vela staged campaign untouched; "validated" reserved for the
old 58ec9a7 stack.

## PRE-REGISTERED verification gates (FROZEN at P0, 2026-07-15)

Any change requires a written, dated gate exception in CAMPAIGN.md.

### Cross-stack parity battery (01_parity_scene.py; reference-artifact design, ledger D3)
Old validated 58ec9a7 stack (reference npz, generated in the cgl venv) vs vendored scene API
(cgl2 venv), foundry-i v2d/v3b inputs, z_ref + 3 seeded perturbations, compared in
CONSTRAINED space via `cgl2/param_map.py` (keyed on z_param_names, never insertion order):

- **F1** forward image ≤ 1e-12 rel max-abs
- **F2** design-matrix columns (old `_design_ret` vs `lstsq_simulate(return_stacked=True)`,
  conversion conventions reconciled) ≤ 1e-12 rel
- **F3** diagonal masked loglik + χ² ≤ 1e-8
- **F4** grad of loglik wrt constrained params (chain rule through each stack's bijector)
  ≤ 1e-8 rel-L2, all 4 points
- **F5** delta-kernel CorrelatedImageLikelihoodTerm ≡ stock ImageLikelihoodTerm ≤ 1e-10
- **F6** Occam −½logdet A vs numpy slogdet ≤ 1e-10
- **F7** unconstrained(constrained(z)) == z + z_param_names audit (exact; informational)
- **F8** harness under a NERSC jax-0.10 env, 1 shared-QOS cell (report-only)

### MC-SMC correctness (04_smc_micro_validation.py, gate B0)
- adapter logp parity vs native ProbModel.log_prob at 64 matched prior draws ≤ 1e-8
- t0_mix2: |logZ − analytic| ≤ 3σ_boot AND minor-mode weight |ŵ−0.2| ≤ 0.053 (3× binomial, N=512)
- t0_funnel10: |logZ| ≤ 3σ_boot ; t0_illcond46: worst-param |z| < 3
- MCLMC-vs-MAMS ΔlogZ bias screen: > 3σ ⇒ MCLMC demoted to cost-frontier-only

### Science gates
P1/P2/P3/P4 gates (σ_seed certification, B1–B5 cells, L0-G2 anchor+1.103 reproduction, L1 SBC,
L2 decomposition, X1-G0/G1/G2/G3) as pre-registered in `plans/PLAN.md` §6 — PLAN §6 numbers are
the frozen thresholds; provisional values (e.g. X1-G1's ~15 nats) are finalized from P1's
measured σ_seed and the finalization is itself a ledger row, not a goalpost move.

## Honest status / what is a proxy / what is blocked

- 2026-07-15: campaign open. Env + vendor + skeleton up; import smoke PASS. Parity battery,
  MC-SMC v0, and adapters under construction. No science claims yet.
