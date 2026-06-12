# Foundry I (Huang 2025a, GIGA-Lens) reproduction

Public-data reproduction of Huang et al. 2025a, *DESI Strong Lens Foundry I:
HST Observations and Modeling with GIGA-Lens* (arXiv:2502.03455), system
DESI-165.4754-06.0423. Working scripts `01_*` ... `46_*` live in this directory;
data products in `data/`. Two reports live in `papers/`: **`main.pdf`** (13 pp,
the final-state report — final posterior, corrected treatment, lessons for
gigalens; served on the project site) and **`evolution.pdf`** (33 pp, the
complete development record).

The headline Phase-1 result is the SVI surrogate posterior in
`data/svi_v10_posterior_mass.npz` (a 10,000-sample Gaussian variational
posterior that recovers all six mass-parameter sign quadrants of Huang+2025a
Table 3, `theta_E` to 3.0%, `e1` to 2.5%, shear PA within 1 degree). A v11 NUTS
chain (`data/nuts_v11f_posterior_mass.npz`) gave a second-mode point estimate.
The section below documents the subsequent push to **genuine HMC** posterior
inference.

> **June 2026 update — the paper-scale Perlmutter campaign + rigor revision
> (scripts `40_*` ... `46_*`, `slurm/`, log in `PERLMUTTER_CAMPAIGN.md`,
> report sections "Paper-Scale Campaign" and "Rigor revision" in
> `papers/main.pdf`) supersede the older gamma-discrepancy conclusions
> below.** The campaign implemented the group's data-treatment feedback at
> published compute scale (vendored gigalens-sean `multinode-2025`,
> `vendor/`); the rigor revision then audited it, finding (a) the noise
> sky-term was over-calibrated by diffuse lens-wing flux (the celebrated
> chi2=0.451 was an artifact — recalibrated: 0.92, honestly under the
> group's <1.1 bar), and (b) a **PSF sampling-convention defect** (gigalens
> kernels must be sampled at delta_pix; the 0.065"-sampled ePSF fed at
> 0.13"/ss=2 broadened the effective PSF 2x in every native-scale fit).
> With both fixed (`data/cutout_v2d.npz`), the native MAP floor drops
> 3.4 -> 1.05 (PSF fix alone, identical noise) and the **final headline
> posterior is gamma = 1.433 [1.400, 1.468]** (published 1.372 +/- 0.023,
> consistent <2 sigma; ~0.1 model-class systematic), theta_E = 2.655
> (0.33% from published), R-hat 1.077, gamma ESS 5,714 (the old gamma-ESS
> saturation was the broadened kernel, not the sampler), inner critical
> curve recovered (`figs/ours_foundry-i_fig8_v2d.png`). A 2x2-binned 0.08"
> refit (`data/cutout_v3b.npz`) exposes an explicit low/steep-slope
> bimodality (delta-chi2 = 0.078 favoring steep; HMC chains split 45/3
> with zero migrations; gamma scale-stable only at +/-0.1) — resolving it
> needs the correlated-noise likelihood, the one open methodological item.
> Cost: ~44 A100-hours of the 200 budgeted.

Environment: `/raid/benson/.venvs/gigalens/bin/python` (JAX 0.6.2, TFP 0.25.0),
gigalens at `/raid/benson/lensing-repos/gigalens`.

---

## HMC posterior inference

The Phase-1 reproduction had only an SVI **surrogate** posterior. The goal here
was genuine HMC on the real (non-Gaussian) posterior. Naively wrapping the
gigalens `HMC()` helper hung; the investigation below traces every pathology to
its fix. **Net: HMC now works mechanically** (compiles ~35 s, stable, smooth,
near-PD mode, samples the correct posterior scale). The remaining limitation is
**mixing efficiency**, not correctness.

### Why it hung, and the recipe that fixed it

The fixes compose; each is necessary.

1. **Compile (not a time budget).** Every kernel — including gigalens
   `GradientBasedTrajectoryLengthAdaptation` — compiles in 31-38 s on a single
   device (`26_compile_diagnostic.py`). The original hang was gigalens `HMC()`
   wrapping GBTLA in `jax.pmap` over **all 10 devices** plus the lstsq 31-channel
   grouped convolution thrashing the cuDNN autotuner (one conv algo took
   11 m 50 s). **Fix:** single/few devices + `XLA_FLAGS=--xla_gpu_autotune_level=0`.

2. **TFP wiring.** `PreconditionedHamiltonianMonteCarlo` nests `step_size` under
   `MetropolisHastingsKernelResults.accepted_results` (not top-level like NUTS);
   the dual-averaging step-size getter/setter must navigate that. (Fixed in our
   runner.)

3. **Preconditioner direction.** TFP convention: the momentum **covariance** must
   equal `Sigma_post^-1` (mass matrix `M = Sigma^-1 ~= H`). The wrong direction
   collapses the step.

4. **Start from a real mode, not a saddle.** The v7 "paper-mode MAP" that all
   prior runs (v8/v10 SVI, v11f NUTS) started from is a **saddle** — 15 large
   negative Hessian eigenvalues, log-posterior improvable by +62,000
   (-99126 -> -36750). `28_refine_map.py` removes it.

5. **lstsq-marginalize the linear light amplitudes.** The full 74-param model
   samples 33 **linear** light amplitudes (5 Sersic `Ie` + 28 shapelet) that
   create ~56 near-flat Hessian directions, capping ESS ~4. Profiling them out
   (`_hmc_lib_lstsq.py build_model_lstsq`, 41 nonlinear params) removes the flat
   directions and moves gamma 2.18 -> 1.90.

6. **float64.** The reduced objective is stiff (cond >= 1e9); float32 floors
   `||grad||` at ~1.2e4. `jax_enable_x64` is required.

7. **Regularized Gaussian marginalization (the key methodological fix).**
   gigalens `lstsq_simulate` solves the linear amplitudes via
   `pinv(X^T W X)` **unregularized** and **omits** the Gaussian-evidence term
   `-0.5*log|X^T W X|`. On the near-rank-deficient shapelet design matrix the
   pinv singular-value truncation makes the profiled log-posterior
   **non-smooth**: `||grad||` floors at ~1e5 even in float64, curvature stays
   indefinite, no PD mode (confirmed: saddle-free Newton `32_saddlefree_newton.py`
   and scipy trust-exact `33_trust_refine.py` both stall at `||grad||~1e5`).

   `_hmc_lib_marg.py build_model_marg` (ndim = 46 = 41 nonlinear + 5
   sampled-positive Sersic `Ie`) marginalizes the 28 shapelet amps **exactly**
   with their Gaussian priors as a ridge:

   ```
   A    = X^T W X + Lambda        (Lambda_ii = (i+1)/25 from Normal(0, 5/sqrt(i+1)) priors)  -> PD
   a*   = cho_solve(A, b)         (smooth Cholesky solve; never pinv)
   logL = -0.5*sum(W R^2) + 0.5*b.a* - 0.5*logdet(A)   (the -0.5*logdet is the Occam term gigalens omits)
   ```

   Cholesky + slogdet of the PD `A` is smooth: `||grad||` breaks the floor
   9e4 -> 400 (228x), Hessian becomes 44/46 positive.

8. **Mass matrix.** The residual posterior is ultra-ill-conditioned (cond ~1e14):
   lens-light companion Sersic centers at `H_ii~1e12`, Sersic indices `n_sersic`
   at `~1e-2`; the physical mass params (`theta_E/gamma/e1/e2/shear`) sit in the
   well-behaved 1e8-1e9 tier. Two float64-safe preconditioners work; the floored
   **full** Hessian over-constrains soft directions (freezes gamma) and is *not*
   recommended:

   - `diagraw` — diagonal mass matrix from the **un-floored** Hessian diagonal
     `|H_raw_ii|` (per-param scalars, no matrix ops). Unlocks realistic scales:
     `gamma_std ~ 0.037` (paper 0.023), gamma drifts 1.866 -> 1.746 toward paper.
   - `hesscorr` — full Hessian via the diagonally-scaled correlation matrix
     `chol(H) = diag(sqrt(D)) @ chol(D^-1/2 H D^-1/2)`. Lifts correlated mass
     params (theta_E ESS 8 -> 20, gamma2 3 -> 14). The correlation matrix itself
     still has cond ~1e8 — empirically, a broad valley coupling gamma, e1/e2,
     and gamma_ext under the pre-correction likelihood (earlier "mass-sheet /
     slope-ellipticity degeneracy" labels retracted: nothing in this single-band
     analysis invokes the mass-sheet transformation; the valley largely closed
     once the June 2026 campaign corrected the likelihood).

### How to run

All commands are GPU-pinned single-device with the autotuner off. Use one A16
per chain. (The L4s were released for general use on 2026-06-10; the old
"reserved" restriction no longer applies. L4 is ~5.9x an A16 on the f64
marginalized gradient: 57 ms vs 336 ms.)

**Step 1 — refine to the PD mode** (scipy trust-exact on the smooth marginal
target). Writes `data/map_marg_pd.npz` + `data/hess_marg_pd.npz`:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID XLA_FLAGS=--xla_gpu_autotune_level=0 \
  CUDA_VISIBLE_DEVICES=4 /raid/benson/.venvs/gigalens/bin/python 34_fit_marg.py \
    --mode refine --x64 --maxiter 80 --out data/map_marg_pd.npz
```

**Step 2 — HMC** (`PreconditionedHamiltonianMonteCarlo` + dual-averaging
step-size adaptation), one chain per process. Vary `--seed` and the output
device/file across chains for parallel multi-chain runs:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID XLA_FLAGS=--xla_gpu_autotune_level=0 \
  CUDA_VISIBLE_DEVICES=0 /raid/benson/.venvs/gigalens/bin/python 34_fit_marg.py \
    --mode hmc --x64 --massmatrix diagraw --seed 0 \
    --num-leapfrog 16 --burn 500 --keep 800 \
    --start-file data/map_marg_pd.npz --mass-file data/hess_marg_pd.npz \
    --out data/prod_diagraw_s0.npz
```

Key flags:
- `--massmatrix {diagraw, hesscorr, hess_marg_pd, diag, identity}` — momentum
  covariance source. **`diagraw` and `hesscorr` are the working choices**;
  `hess_marg_pd` (floored full Hessian) freezes the soft mass directions.
- `--seed N` — PRNG seed; **use a distinct seed per parallel chain** so split
  R-hat / cross-chain ESS are meaningful.
- `--num-leapfrog`, `--target-accept`, `--step-size`, `--burn`, `--keep`.

Each process writes one npz with raw `samples` (n, 46) and the
bijector-mapped physical mass params `mass_theta_E`, `mass_gamma`, `mass_e1`,
`mass_e2`, `mass_gamma1`, `mass_gamma2`.

**Step 3 — pool the chains** (CPU-only post-processing — keep the A16s free):

```bash
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
  /raid/benson/.venvs/gigalens/bin/python 35_pool_chains.py \
    --glob 'data/prod_diagraw_s*.npz'
```

`35_pool_chains.py` pools the 6 physical mass params across all matched chains
and prints per-param **split R-hat** (`tfp.mcmc.potential_scale_reduction`,
`independent_chain_ndims=1, split_chains=True`), **cross-chain ESS**
(`tfp.mcmc.effective_sample_size`, `cross_chain_dims=1`), and pooled
median + 16/84% vs the paper values. It accepts any `--glob`, handles chains of
differing length (truncates to the min), and `--burn N` drops leading samples.

### Honest status

- **Mechanically: HMC works.** Compiles in ~35 s, stable, smooth target
  (`||grad||` ~400, near-PD mode 44/46 positive), correct posterior scale. Every
  prior pathology (saddle start, flat amplitude directions, pinv non-smoothness,
  float32 floor, wrong preconditioner direction) is fixed.
- **Mixing is degeneracy-limited.** Per-chain ESS ~3-20 / 300; multi-chain split
  R-hat = 1.4 to 6+ at 500 samples (need < 1.01). The smoke-test pool of
  `data/prod_diagraw_s*.npz` (4 x 500) gives gamma R-hat ~5, gamma1 R-hat ~6 —
  chains have **not** mixed. This is intrinsic strong-lens parameter degeneracy
  (correlation-matrix cond ~1e8 even after diagonal scaling), not a sampler bug.
  Converged inference needs ~1e4-1e5 samples/chain (A100-feasible; ~17-24 h on
  A16 even 8-way parallel). A **long 8-chain `diagraw` run** (burn 2000 / keep
  8000) is **in progress**, writing `data/long_diagraw_s0..7.npz`, to test
  whether the broad gamma posterior (diagraw gamma chains span 1.37-1.86)
  reconciles with the paper. Pool it once it lands:
  `35_pool_chains.py --glob 'data/long_diagraw_s*.npz'`.
- **The paper's gamma = 1.372 is a model-setup difference, NOT a sampler issue.**
  In our setup the paper's mode is a distinct, far-worse fit: a paper-seed refine
  stayed at gamma = 1.374 / theta_E = 2.640 with log-p = -78350 vs our basin
  -45841 (Delta = -32500). The gamma gap is an **unpublished model-setup
  difference** (PSF / masking / source complexity / priors), not something the
  sampler can close.

### Key files

| File | Purpose |
| ---- | ------- |
| `_hmc_lib.py` | full 74-param model (5 Sersic Ie + 28 shapelet amps sampled). |
| `_hmc_lib_lstsq.py` | `build_model_lstsq`, 41 nonlinear params (gigalens pinv lstsq profiling; non-smooth). |
| `_hmc_lib_marg.py` | **`build_model_marg`, 46 params** — regularized Gaussian marginalization (the fix). |
| `26_compile_diagnostic.py` | per-kernel single-device compile timing. |
| `28_refine_map.py` | refine off the v7 saddle. |
| `30_refine_lstsq.py`, `32_saddlefree_newton.py`, `33_trust_refine.py` | lstsq-target refinement / stall diagnostics. |
| `34_fit_marg.py` | **main runner**: `--mode refine\|hmc`, `--massmatrix`, `--seed`, `--x64`. |
| `35_pool_chains.py` | **CPU multi-chain pooler / R-hat + ESS diagnostic.** |
| `data/map_marg_pd.npz`, `data/hess_marg_pd.npz` | refined PD mode + Hessian (HMC start + mass-matrix source). |
| `data/prod_diagraw_s0-3.npz`, `data/prod_hesscorr_s0-3.npz` | 4x500 parallel production chains. |
| `data/long_diagraw_s0-7.npz` | long 8-chain diagraw run (in progress). |
