# claude-giga-lens — advancing lens-modeling quality beyond GIGA-Lens

Campaign (2026-07) building on the GIGA-Lens lineage reproductions (`gu-2022`, `cikota-2023`,
`sheu-2024b`, `foundry-i`). Two science pillars, one synthesis:

- **P1 — Correlated-noise likelihood.** Foundry-I's one open methodological item: drizzled
  HST products have strongly correlated pixel noise (measured ρ(1)=0.795 on the 0.04″ fine
  product, 0.52 native; N_eff/N ≈ 0.017) that the diagonal likelihood ignores, producing
  scale-dependent γ (1.29 binned / 1.43 native / 2.585-artifact fine) and a two-basin γ
  posterior on the binned product. We build a convolutional-whitening correlated likelihood
  (C = D^{1/2} K D^{1/2}, drizzle-anchored stationary kernel fit on model-subtracted
  residuals) integrated with the ridge marginalization of the shapelet amplitudes, validate
  it on drizzle mocks with known truth (bias + SBC coverage), then test the pre-registered
  cross-scale unification hypothesis on the real system DESI-165.4754−06.0423.
- **P2 — Sampler/multimodality benchmark → the CGL recipe.** A "lens-posterior zoo"
  (synthetic analytics; gu-2022 mocks; the foundry-i 46-dim cond-1e14 real posterior; the
  genuinely bimodal v3b posterior; Euclid Q1) benchmarked under budget-matched, tuning-honest
  protocol across: baseline SVI→ChEES-HMC, window-adapted NUTS, MCLMC, flowMC, adaptive
  tempered SMC, nautilus, PT-HMC, neural-transport (NSF) HMC. Ranked recipe bet: MAP → SVI →
  tempered-SMC anneal → NSF flow → ChEES-HMC in flow space.

Approved plan: `/home/benson/.claude/plans/this-repo-containes-astrophysics-radiant-sprout.md`.
Ledger + stage log: `CAMPAIGN.md`. Perlmutter jobs charge **deepsrch_g** only.

## Status

**P0 (scaffold) — IN PROGRESS.** Nothing science-bearing has run yet.

## Operator quickstart

```bash
source /raid/benson/.venvs/cgl/bin/activate       # aarch64 phoenix venv (NOT the blessed gigalens venv)
python 00_env_check.py                            # asserts pins/GPU/imports, writes data/env_freeze_*.txt
./00_run_tests.sh                                 # CPU unit tests + GPU parity tests
GIGALENS_X64=1 CUDA_VISIBLE_DEVICES=8 python 01_parity_harness.py   # L4 recommended
```

Conventions: numbered `NN_verb_noun.py` operator scripts at dir root; shared logic in the
`cgl/` package (`pip install -e . --no-deps`); bulk artifacts in gitignored `data/`;
committed figures in `figs/`; report in `papers/` (`make -C papers pdf`).

Copy-vs-import rule: code we modify is **copied with attribution** (marg core from
`../foundry-i/_hmc_lib_marg.py`; truth sampler from `../gu-2022/01_gen_mocks.py`);
artifacts and the frozen vendored library are **imported by path** (`cgl/paths.py`, which
asserts the vendored ref 58ec9a7). The vendored library is UNPATCHED; all mitigations live
in `cgl/guards.py`, one per prior real incident.

## PRE-REGISTERED verification thresholds (frozen at P0, before any science runs)

The repo culture retracts celebrated-but-wrong numbers (foundry-i's χ²=0.451). These
goalposts are fixed NOW; outcomes are reported against them with no post-hoc moves.
Any change requires a written, dated gate exception in `CAMPAIGN.md`.

### Parity harness (01, gate for everything downstream)
Evaluated vs foundry-i `_hmc_lib_marg` at `map_marg_pd.npz` + 3 seeded perturbations,
and vs gu-2022 `system_000` truth:
- **A** forward image: max|Δ|/max|img| ≤ 1e-12 (float64)
- **B** log-posterior: |Δ logp| ≤ 1e-8
- **C** gradient: relative L2 ≤ 1e-8
- **D** diagonal limit: the correlated likelihood with C = diag(err²) equals the validated
  diagonal stack to ≤ 1e-10 (the new machinery must reduce EXACTLY to the validated one)
- **E** Occam term −½logdet(A) vs numpy slogdet ≤ 1e-10

### P1 gates
- Whitener: e_op = max_ω|S·|ĥ_M(ω)|²−1| ≤ 0.02; Monte-Carlo whiteness Var(u_kept) ∈ 1±0.02,
  mean |off-diag corr| at lags ≤3 below 0.01; |ΔlogL| < 0.1 nat vs dense-C exact reference
  on the 80²/130² grids at 20 prior draws.
- Kernel fits: max|ρ_fit − ρ_meas| ≤ 0.05 over the fit window; kernels fit on
  MODEL-SUBTRACTED residuals only (guard-enforced).
- Mock generator: noiseless render-vs-drizzle agreement < 0.05σ on every kept pixel.
- E1a (artifact): diagonal likelihood on fine mocks shows median |z(γ)| > 2 OR 68% coverage
  < 40% (reproduces the real-data pathology in miniature).
- E1b (recovery): correlated likelihood: |z̄(γ)| < 0.5 per scale; cross-scale
  |γ_fine − γ_native| < 1σ_comb on ≥ 6/8 mocks; width ratio σ_fine/σ_native ∈ [0.7, 1.5].
- E1c (SBC-lite, 64 mocks): rank-uniformity χ² p > 0.01 for each of
  {θ_E, γ, e1, e2, γ1, γ2}; empirical 68% coverage ∈ [55, 80]%.
- E2 real data, PRE-REGISTERED FORK (both branches publishable):
  - **H1** the v3/v3b steep basin: posterior mass < 10% under the correlated likelihood,
    or corrected Δlogℓ(steep−low) ≤ 0 at basin MAPs → bimodality is a likelihood artifact.
    ALTERNATIVE (written down now): bimodality survives a validated correlated likelihood
    → genuine/PSF-systematic multimodality; v3b becomes the flagship P2 target.
  - **H2** cross-scale unification: |γ_fine − γ_native| and |γ_binned − γ_native| < 2σ_comb
    (anchor: γ_native(diag) = 1.433 [1.400, 1.469]); stronger: overlapping 68% CIs.
  - **H3** honesty: σ_γ(fine, corr) ≥ σ_γ(native)/1.5 (same photons — no fake information).
- E3 robustness: γ posterior-mean shift < 0.5σ between the fitted kernel and each of
  {nonparametric PSD-projected, ±20% hyperparam perturbations, alternate-basin fit,
  M ∈ {6,9,12}, s_floor ∈ {0.02,0.05,0.10}} variants.

### P2 gates
- Zoo: every target exposes one jitted logp closure; adapter logp equality asserted at 3
  reference points before any benchmark run; zoo frozen (hashes) before the eval matrix.
- Reference (T3 v3b): PT with ≥100 β=1 round trips, adjacent-swap acceptance 20–40%,
  within-mode R̂ < 1.005; per-basin SMC evidence must agree within 2σ or no reference is
  claimed (metrics then honestly restricted to within-mode ESS + mode-finding sensitivity).
- Win criterion (per target): median-over-3-seeds ESS/grad ≥ 2× baseline AND rank-norm
  R̂ < 1.01 AND (multimodal targets) every reference mode recovered with occupancy ±10 pts.
- "CGL recipe" claim requires ALL of: T3 both basins in 3/3 seeds with mode-weight error
  < 0.05 within the Track-A budget; T1-hard ≥ 3× min-ESS/grad vs baseline; T2 matches the
  diagraw-HMC posterior (γ, θ_E within 0.2σ) at ≥ 2× ESS/s WITHOUT hand-built mass
  matrices; no easy-target regression below 0.7× baseline ESS/grad.
  Anything less = benchmark paper, not recipe paper (still publishable, stated as such).

### Budget gates
Perlmutter commit 90 A100-h (P1c ≤30, P2c ≤45, P3 ≤15), HARD STOP 100, account deepsrch_g,
ledger row appended BEFORE results are read.

## Honest status / what is a proxy / what is blocked

- Nothing yet — updated as gates read out. (This section is the campaign's conscience;
  every claim in the final report traces to a gate row in CAMPAIGN.md.)
