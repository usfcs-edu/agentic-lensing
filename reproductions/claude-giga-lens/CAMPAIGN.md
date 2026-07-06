# claude-giga-lens campaign ledger

Goal: advance lens-modeling quality beyond GIGA-Lens on two pillars — P1 correlated-noise
(drizzle) likelihood + generalized ridge marginalization; P2 sampler/multimodality benchmark
on a lens-posterior zoo → a demonstrated-better "CGL recipe".
Bar: the PRE-REGISTERED verification thresholds in README.md §Verification (frozen at P0,
before any science runs). Retractions are recorded inline in the stage log, never deleted.

Decisions (locked 2026-07-06):
- Scope: P1+P2 depth-first; P3 (GP/pixelated source) parked as follow-on.
- Substrate: vendored gigalens-sean@multinode-2025 ref 58ec9a7 (same as foundry-i), UNPATCHED;
  all campaign code in `cgl/` with array-only likelihood APIs.
- Environments: /raid/benson/.venvs/cgl (aarch64, pins matching the blessed gigalens venv);
  /raid/benson/.venvs/cgl-torch for pocoMC only. Blessed gigalens venv untouched.
- Perlmutter: user cap ≤100 A100-h — commit 90 (P1c 30 / P2c 45 / P3 15), HARD STOP 100,
  ≥56 h of the repo-wide 200-h allocation held in reserve.
  **ALL jobs charge account `deepsrch_g` (NOT cosmo_g — Greg, 2026-07-06). Every slurm
  template hardcodes `#SBATCH -A deepsrch_g`.**
- Phoenix (8×A16 15GB + 2×L4 23GB) is budget-free but ledgered weekly below.
- The lensing likelihood NEVER runs on CPU (216× slower; unit-test 16×16 toys excepted).

## A100-hour ledger  (append BEFORE reading results)

| job id | phase | nodes × walltime | A100-h | cumulative | purpose |
|---|---|---|---|---|---|

Committed: 0.0 / 90 (hard stop 100).

## Phoenix GPU ledger  (weekly rollup)

| week | phase | script family | GPU-h (A16/L4) |
|---|---|---|---|

## Gate record

| gate | date | verdict | numbers | notes/retractions |
|---|---|---|---|---|
| P0 env gate | 2026-07-06 | PASS | 147 pkgs frozen, jax 0.6.2 GPU, all sampler libs import | jax-upgrade incident fixed via constraints.txt |
| P0 parity A–E | 2026-07-06 | PASS | A 6.2e-17 (thr 1e-12), B 0.0 (1e-8), C 0.0 (1e-8), D 0.0 (1e-10), E 0.0 (1e-10) | cgl vs foundry-i `_hmc_lib_marg`, both on vendored lib, L4, f64; data/parity_report.json. INFO: stored pip-era MAP logp reproduced to 6.5e-11; gu-2022 f32 forward image 1.5e-6 rel |
| P0 sampler smoke | 2026-07-06 | PASS 8/8 | NUTS+window 12.6s, MCLMC(un+adj), adaptive tempered SMC (λ→1, finite logZ), TFP REMC, flowMC RQSpline_MALA, flowjax MAF fit, nautilus logZ within 1.0 | blackjax 1.3 fine on jax 0.6.2 — 1.2.5 fallback NOT needed. See XLA defect below |

## Stage log

### P0 — 2026-07-06 (campaign start)
- Branch `claude-giga-lens`; work dir scaffolded per approved plan
  (`/home/benson/.claude/plans/this-repo-containes-astrophysics-radiant-sprout.md`).
- Vendored gigalens-sean copied bit-identical from foundry-i vendor (ref verified 58ec9a7);
  `cgl.paths.bootstrap_vendor()` asserts the ref on every import.
- venv /raid/benson/.venvs/cgl building (core pins = blessed stack: jax 0.6.2, tfp[jax] 0.25.0,
  numpy 2.4.6, scipy 1.17.1, optax 0.2.8, objax 1.8.0, astropy 7.2.0, lenstronomy 1.14.0;
  samplers: blackjax 1.3.*, flowMC 0.4.5, flowjax 19.0.0, equinox 0.13.8, nautilus 1.0.6).
- `cgl/guards.py` written: six guards, each encoding a prior real incident (PSF delta_pix
  convention ×2 incidents, pmap-all-devices hang, SVI cov rank deficiency, f32 grad floor,
  wing-contaminated sky calibration, CPU ban).
- venv gate incident (recorded): the first sampler install silently upgraded jax 0.6.2 →
  0.10.2 (culprit: unconstrained chex 0.1.92), breaking tfp 0.25.0. Fixed by re-resolving
  under `constraints.txt` (chex pinned back to 0.1.90, lineax 0.1.0); ALL future pip
  installs into cgl venvs must pass `-c constraints.txt`. Env gate now PASSES
  (00_env_check.py; freeze at data/env_freeze_phoenix.txt, 147 pkgs).
- GPU inventory confirmed: CUDA indices 0–7 = A16 15GB, 8–9 = L4 23GB (nvidia-smi
  authoritative; jax device_kind mislabels all 10 as "NVIDIA L4").
- 12/12 CPU unit tests green (guards, paths, vendor shadowing).
- requirements-{aarch64,perlmutter}.txt written; slurm/{smoke,prod}.slurm hardcode
  `#SBATCH -A deepsrch_g`.
- In flight (3 agents): sampler runtime smoke (GPU 9); marg-core port + parity harness A–E
  (GPU 8); GIGA-Lens-2.0 positioning note (web).
- GIGA-Lens 2.0 scope check DONE (research/notes/gigalens2-positioning.md): arXiv 2606.30633
  verified = multi-node scaling of the UNCHANGED diagonal-likelihood MAP→SVI→HMC recipe
  (128 nodes/512 A100s; real system DESI J238.5690+04.7276, 38 params, R̂<1.01, χ²ν=0.8954).
  Zero overlap with P1 (no correlated noise, no marginalization/Occam) or P2 (no
  multimodality/tempering/flows). Risk-register item 7 RETIRED. Their real-system result is
  a natural P2 baseline; their linear-inversion step is exactly the machinery P1 upgrades.
  Watch-list (nearest, non-competing): 2406.08484 (flags correlated noise as open, no
  implementation), 2511.04792 (score-based SBI), 2410.22573 (flow-matching NPE).
- Parity A–E PASS (gate record); cgl marg core is bit-identical to the validated foundry-i
  stack on the vendored lib.
- Sampler smoke 8/8 PASS (tests/test_sampler_smoke.py is the API seed for cgl/samplers/).
  **NEW STACK DEFECT (guard-worthy):** jaxlib 0.6.2 XLA `priority-fusion` pass LIVELOCKS
  (infinite compile, ~600 threads) fusing f64 `jax.random.normal` (erf_inv) with a reduction
  on L4/aarch64 — exactly blackjax MCLMC's `partially_refresh_momentum`. Workaround:
  `XLA_FLAGS=--xla_disable_hlo_passes=priority-fusion` (set pre-import; XLA_FLAGS parsed
  once). f32 unaffected; TFP-PHMC/NUTS f64 unaffected (foundry-i history confirms). Policy:
  sampler adapters that draw f64 normals inside jitted kernels (MCLMC family) set the flag
  in their own process (22_run_cell.py runs one process per cell); measure A100 impact
  during the first Perlmutter smoke before making it default there.
- flowMC 0.4.5 adapter requirement: re-init the bundle optimizer as
  optax.chain(clip_by_global_norm(1.0), adam(lr)) over eqx.filter(model, is_inexact_array)
  (upstream inits adamw over bool spline masks → jax 0.6.2 tree mismatch).
- flowjax 19 requires new-style typed keys (jax.random.key, not PRNGKey).
- Full-suite livelock incident (recorded): the combined `pytest tests -m gpu` session hit the
  priority-fusion livelock (698 threads, 40 min) because an earlier test module's imports
  initialized the JAX backend before test_sampler_smoke.py's import-time XLA_FLAGS append.
  Fix: 00_run_tests.sh now sets the full XLA_FLAGS in the process env. Lesson: XLA flags are
  process-level, set them in the runner/launcher, never per-module.
- **P0 EXIT GATE MET 2026-07-06**: full suite green — 18 CPU + 9 GPU passed (5m34s);
  parity A–E PASS; sampler smoke 8/8 PASS; env gate PASS; ledger live; positioning note done.
  grad-norm at z_ref 400.5806 reproduces map_marg_pd exactly; cond(A)=1.37e4.
- P1a watch-outs from the parity port (carry into cgl/whiten.py work): (1) vmap-per-column
  convs may need the grouped-conv refactor (gate D is the regression); (2) mask smearing —
  build sqrt_d_inv from the masked err map, trim keep_w by kernel half-width at masks/edges;
  (3) dropped log|C| constant is INVALID for E3 cross-whitener logL comparisons (add back or
  compare posteriors only); (4) 'SAME' conv zero-pads borders — account in e_op; (5) never
  route whiten_fn=None through the conv path (keeps gate B exactly 0.0); (6) parity product
  uses the legacy PSF deliberately; science products must declare psf_pixel_scale so
  guards.assert_psf_sampling engages.
