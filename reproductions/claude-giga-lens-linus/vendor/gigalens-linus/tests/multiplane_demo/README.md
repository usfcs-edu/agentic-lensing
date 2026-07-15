# `tests/multiplane_demo/` — 3-lens-plane simulate→recover demo (lenstronomy oracle)

**Status: UNCERTIFIED (Mode B).** The forward model is validated against lenstronomy
(below). Recovery: Stage 0 (noiseless identity) and Stage 1 (MAP point estimate) are
done and UNCERTIFIED — the **dominant lens recovers** at the point estimate while the source
shape / weak z2 looked off (later shown to be MAP under-convergence, not degeneracy; see
*Recovery*). **MCLMC (MAP→MCLMC, no SVI) then recovers the full truth** — 24/24 params
inside their 95% CIs — at a **borderline-converged** Max R-hat 1.105 (one slow param just
over the 1.1 accept line; a longer-burn-in rerun should clear it). This is a **proposed**
recovery, **not** a certified PASS — the human grades it.

## What this is

A deliberately physical **3-lens-plane** strong-lens system, simulated with
**lenstronomy** (the validation oracle) so that gigalens can be tested end-to-end:

1. **Forward model** — does gigalens render the same image as lenstronomy? (done)
2. **Recovery** — fed the lenstronomy image + noise, does gigalens inference recover
   the truth parameters? (Stage 0 + MAP done; MCMC next)

Eventually this is meant to be a **demo test anyone can run**.

## The system (all numbers in `config.py`)

Planes in order of increasing redshift:

| z | mass | light |
|---|---|---|
| 0.3 | EPL (θ_E=1.0″, γ=2.00), offset at (0.30, 0.25)″ — **dominant lens** | — |
| 0.6 | EPL (θ_E=0.2″, γ=2.05) — **weak perturber** | Sérsic lens light (n=4, R=1.0″) |
| 2.0 | — | Sérsic source (n=1, R=0.15″) |

- **z3 source** is lensed mainly by the **dominant z1 mass** into a large Einstein ring,
  **weakly perturbed by z2** and mildly asymmetric because z1 is off-axis. The
  `(with z2) − (no z2)` panel shows z2's contribution as a thin dipole on the ring.
- **z2 lens light is itself foreground-lensed by the z1 plane** (gigalens renders *all*
  light at its traced position; light emitted at z2 passes through z1's potential). z1's
  effective Einstein radius for a z2 source, θ_E,eff = θ_E,z1 · √[(D_ds/D_s)(z1,z2) /
  (D_ds/D_s)(z1,z3)] (< θ_E,z1 = 1.0″), is large enough that the bright n=4 core is
  multiply imaged — the lens galaxy shows a **main image + an offset second image** rather
  than a smooth galaxy. z1 is deliberately offset so this is **not** a concentric ring.
- **Identifiability caveat (confirmed at MAP).** The dominant **z1 mass is well
  constrained** (θ_E, γ, centers recover to ~0.01 at the MAP). The **weak z2 perturber
  (θ_E=0.2″)** imprints only faintly on the z1-dominated ring, and the **compact source's
  (R_sersic, n_sersic)** trade off against the lstsq amplitude — both are off by ~2–3× at
  the point estimate *despite* χ²≈1 and structureless residuals. That is a **degeneracy**
  (a posterior-width question), not a fit failure; MCMC must quantify the widths, and per
  method-discipline §1 the source shape may only be reportable as an identifiable
  *combination* (effective size / magnified flux), not as R and n individually.
- A **z2-plane-only reference** (z1 deleted) is rendered alongside: smooth unlensed lens
  galaxy + a clean single-plane z3 ring from the weak z2 alone.

Observation: 0.05″/pix, 80×80 (4″ FOV), supersample 3, Gaussian PSF FWHM 0.10″, float64.

## Key conventions (load-bearing)

- **EPL `theta_E` is referenced to the FINAL source plane (z3)** — one value per
  deflector for the whole system, matching gigalens' multi-plane convention. Building
  per-plane sub-models with different `z_source` would silently reinterpret `theta_E`
  and change the physical mass — **do not** do that. Intermediate-plane light is placed
  via `MultiPlane.ray_shooting_partial(0→z_p)` on the single z3 model.
- **lenstronomy 1.13.1 `ray_shooting_partial` returns the angular β at `z_stop`
  directly** (no `T_xy` normalization). Guarded by self-check A.
- **Sérsic `Ie` == lenstronomy `amp`** (surface brightness at `R_sersic`).
- Cosmology: flat wCDM (H0=70, Ωm=0.3, w0=−1); the astropy/lenstronomy side uses a
  radiation-matched `FlatwCDM(Tcmb0=2.7255, Neff=3.046)` so cosmology is out of the
  error budget (same setup as the R1 trace test).

## Validation status (forward model)

From `build_demo.py` (float64, container run):

- **check A** — `ray_shooting_partial(0→z3)` vs full `ray_shooting`: `0.0` (β convention).
- **check B** — gigalens scene trace vs lenstronomy β: **1.7e-8 @ z2, 2.7e-6 @ z3**
  (independent confirmation of the multi-plane recursion; same order as the existing R1
  trace test).
- **forward-model identity** — gigalens render vs lenstronomy render: **2.3e-3 rms**,
  residual is smooth and core-concentrated with **no structure beyond the Sérsic `b_n`
  floor**. gigalens' `b_n` approximation differs from lenstronomy's exact value by
  ~0.07–0.2%; this is an accepted, quantified convention offset (decision 2a), not a
  bug. **Consequence (now quantified, retired):** at the chosen recovery noise the b_n
  systematic contributes a reduced-χ² bias at truth of Σ(bn/σ)²/N = **9.6e-5** (≈0.005×
  the χ² spread √(2/N)) — i.e. it sits far below the noise floor and **cannot bias the
  recovery**. The fit is noise-limited, not systematic-limited (see the model card).

## Recovery status (UNCERTIFIED, Mode B)

Noise model (`noise_model.py`): background_rms=0.005, exp_time=2000 (user-selected,
deeper-than-single-orbit HST), seed=20260624 →
`σ = √(bg² + clip(img,0,∞)/exp_time)` (gigalens `ImageData` §3.4 — explicit, no default).
**Peak SNR 41, source-ring integrated SNR 58.** Priors are deliberately *not* centered on
truth (positive-support dists for positive params); lstsq solves the 2 Sérsic `Ie`, so
24 nonlinear params are sampled. Cosmology + redshifts fixed.

- **Stage 0 — noiseless identity (`recover.py`).** lstsq model at truth vs noiseless
  truth: max|Δ|/peak = **1.3e-3** (< b_n floor 3.3e-3; < 5e-3 falsifier), reduced χ² ≈
  1e-5. PROPOSE PASS. The likelihood/lstsq wiring is correct.
- **Stage 1 — MAP (`recover.py`; way-station, NOT a recovery certification).** adabelief
  (1e-2, 0.95, 0.99), 100 starts × 500 steps, seed=1, 1×A100. Reduced χ² at MAP =
  **1.016**, in the pre-committed band 1 ± 3√(2/N) = 1 ± 0.053. **Dominant z1 mass
  recovers** (θ_E 1.000→0.999, γ exact, centers ≤0.008). **Weak z2 mass and source
  (R_sersic, n_sersic) are off ~2–3×.** Diagnostic (`diag_map.py`): reduced χ²(MAP)=1.016
  vs χ²(truth)=0.997 differ by only ~1σ, and the MAP normalized-residual map is clean
  N(0,1) (no arcs/dipoles/point-in-ring) → the off params are a **degeneracy** (the data
  can't distinguish them at this noise), not a fit failure or convergence bug.
- **Stage 2 — MCMC (`mcmc.py`; SVI→HMC): run 1 done, NOT converged → recovery
  UNCERTIFIED, claim withheld.** SVI (n_vi=100, 500 steps) → HMC (50 chains, 200 burnin,
  400 draws, ~800 s on 1×A100; q_z sanitized to dodge a JAX-0.10 pmap/mesh bug).
  *Coverage looks promising but is NOT trustworthy yet:* 24/24 truth params fall in their
  95% CI with |pull|<1.9 — **however Max R-hat = 1.92** (worst `p1.L.center_x`; target
  <1.01, accept <1.1) and Min ESS = 105, so the chains have **not converged** and the
  coverage cannot be certified (project-standards §2). Trace diagnosis (`diag_mcmc.py`,
  `demo_3plane_mcmc_diag.png`): the ~6–8 elevated-R-hat params (z2 lens-light position,
  z1 θ_E, z2 mass slope) show **slow low-frequency wandering** — high autocorrelation /
  under-sampling, *not* multimodality (no discrete clusters) and *not* frozen chains;
  other params mix well (e.g. `p1.L.e2` ESS 6071). Pre-registered as the "degenerate
  directions sample poorly" risk. **Next:** longer HMC (more burnin+draws) and/or
  `HMC_alt_multi` (burn-in-recomputed mass matrix). Prediction: ~4–8× sampling drops Max
  R-hat below ~1.1 and lifts Min ESS to several hundred; falsifier — if it stalls ≥1.2,
  the problem is geometry/conditioning (needs reparametrization), not sampling length.
- **Stage 2 (take 2) — MAP → MCLMC, no SVI (`mclmc_run.py`): near-pass, recovery
  PROPOSED (UNCERTIFIED).** Replaced SVI→HMC with MCLMC (self-tunes step size / L / mass
  matrix in burn-in; surrogate = MAP point + small default cov 1e-6·I). 32 chains, 1000
  burn-in, 2000 results, ~412 s on 1×A100 (faster *and* better than HMC). **Max R-hat =
  1.105** (worst `p1.L.center_x`, fractionally over the 1.1 accept line), **Min ESS = 280**
  — borderline converged (HMC was 1.92 / 105). **Coverage: 24/24 truth params inside their
  95% CIs, all |pull|<2**, now credible because R-hat is near-converged. Findings:
  (a) the **source shape is well recovered** (R_sersic 0.156±0.006, n_sersic 0.97±0.06,
  truth-centered) — the 2–3× MAP miss was MAP **under-convergence**, *not* a real
  degeneracy (earlier read revised); (b) the **weak z2 mass is the loose component**
  (θ_E 0.219±0.018, γ 2.21±0.11 — wide but containing truth); (c) the **z2 lens-light
  center is slowest-mixing** (R-hat 1.105, tight posterior), likely coupled to z1's
  foreground deflection.
- **Stage 2 confirmation — MCLMC 8 chains / 2000 burn-in / 2000 results (`mclmc_run.py`
  default): accept-level convergence + full recovery, PROPOSED PASS (UNCERTIFIED).**
  ~183 s on 1×A100. **Max R-hat = 1.076 — now under the 1.1 accept line** (worst still
  `p1.L.center_x`; not yet the ≤1.01 ideal), Min ESS = 79.5 (≫ 8 chains; lower absolute
  than the 32-chain run, just fewer total draws). **Coverage: 24/24 truth params inside
  95% CIs, all |pull| < 1.7.** Corner plot `demo_3plane_corner.png` (`cornerplot.py`):
  red truth crosshairs fall inside every posterior contour. **Verdict (proposed):
  gigalens recovers the truth of the full 3-plane multi-redshift system at accept-level
  convergence; the dominant z1 mass is tight, the weak z2 mass is wide-but-containing, the
  source shape is well constrained.** Residual slow direction (z2 lens-light center) would
  tighten further with more chains / burn-in if the ≤1.01 ideal is wanted. **The human
  grades before this becomes a certified PASS.**

## Files

- `config.py` — single source of truth for all parameters (edit here only).
- `build_demo.py` — builds the gigalens scene + lenstronomy oracle, runs the
  self-checks, renders, prints physicality, writes artifacts. `python -m
  tests.multiplane_demo.build_demo` from `~/gigalens`.
- `artifacts/demo_3plane.png` — 8-panel figure: row 0 = full 3-plane system (total
  sqrt/linear, z1-lensed lens light, z3 ring); row 1 = z2-plane-only reference (no z1,
  sqrt/linear), the `(with z2) − (no z2)` z2 contribution (z2's lensing of the z3 ring;
  the lens light is unchanged by removing z2's mass), and the gigalens−lenstronomy
  residual.
- `artifacts/demo_3plane.npz` — image arrays (incl. `img_noz1`, `img_noz2`, `z2_effect`)
  + kernel.
- `noise_model.py` — adds the explicit observational noise model + model card; writes the
  fixed-seed noisy data (`demo_3plane_noisy.{png,npz}`). Deterministic, runs no fit.
- `recover.py` — Stage 0 (noiseless identity) + Stage 1 (MAP). Priors, truth lookup,
  recovery model. Writes `demo_3plane_map.npz`. **GPU** (the MAP batches all starts; 100
  starts ≈ 19 GiB on one 40 GB A100 — 500 OOM'd).
- `diag_map.py` — MAP diagnostics (no re-fit): χ²(MAP) vs χ²(truth), normalized-residual
  map (`demo_3plane_map_resid.png`). The "plots before metrics" check on the MAP.

## Running

Inside the canonical Shifter container (see `GIGALens-Code/docs/env_setup.md`):

```bash
export PYTHONPATH=/global/homes/l/linusu/sidecar_jax_upgrade:\
/global/u1/l/linusu/gigalens/src:$HOME/.conda/envs/gigalens_multinode_env/lib/python3.12/site-packages
cd ~/gigalens
/usr/bin/python3 -m tests.multiplane_demo.build_demo
```

## Next (not done)

1. **Stage 2 — MCMC** (SVI → HMC) per `GIGALens-Code/docs/inference-diagnostics.md`:
   posterior widths + R-hat/ESS, coverage of the *identifiable* params/combinations. This
   is the step that turns the MAP's degeneracy finding into a certifiable (or withdrawn)
   recovery claim. Memory: HMC/SVI also batch chains — keep n_chains modest (OOM risk).
2. Decide whether the source `(R_sersic, n_sersic)` degeneracy is acceptable for the demo
   or whether to make the source slightly more extended / brighter so it is individually
   identifiable (a scene choice for the human).
3. Promote the pipeline into a pytest (`test_demo.py`) with derived tolerances, following
   `tests/validation/` conventions.
