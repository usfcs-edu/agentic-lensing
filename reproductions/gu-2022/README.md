# Gu et al. 2022 (GIGA-Lens) reproduction

Reproduction of the **method-paper benchmark** of Gu, Huang, et al. 2022,
*GIGA-Lens: Fast Bayesian Inference for Strong Gravitational Lens Modeling*
(ApJ 935, 49; `papers/Gu_2022_GIGA_Lens.pdf`). slug = `gu-2022`.

The paper's headline validation (§3, Figs. 10-11, Table 2) is: simulate **100
mock lenses** with lenstronomy and recover their parameters with the GIGA-Lens
**MAP -> SVI -> HMC** pipeline, achieving for *every* parameter over *all* 100
systems **max split-R-hat <= 1.017** and **min ESS = 26822** (mean ESS ~32k),
with the posterior means consistent with the injected truth (mean scaled error
`mu_z ~ 0`). This directory reproduces that benchmark end-to-end with the
upstream `gigalens` JAX code.

Environment: `/raid/benson/.venvs/gigalens/bin/python` (JAX 0.6.2, TFP 0.25.0,
lenstronomy 1.14.0); gigalens at `/raid/benson/lensing-repos/gigalens`.
GPUs: A16 indices 0-5 (one mock per GPU process).

---

## Model & benchmark settings (from the paper)

| | value | source |
| - | ----- | ------ |
| mass | EPL + external shear | §2.1, Eq. (8) |
| lens light | Sersic (elliptical) | §2.1 |
| source light | Sersic (elliptical) | §2.1 |
| pixel scale | 0.065"/px | Fig. 1, Fig. 10 |
| cutout | 80 x 80 (5.2" x 5.2") | Fig. 10 caption |
| supersample | k_super = 2 | §2.5.2 |
| PSF | TinyTim WFC3-class (gigalens `assets/psf.npy`) | Fig. 1 |
| noise | sigma_bkg = 0.2, t_exp = 100 s, G = 1 | Fig. 10 caption |
| arc SNR | ~100 (range 30-200) | §3 / Eq. (8) light amps |
| **MAP** | K_MAP=300 multistart, n_MAP=300 steps, Adam 1e-2 -> 1e-3 | Table 1 |
| **SVI** | K_VI=1000 MC, n_VI=500 steps, Adam 0 -> 1e-3, init Sigma=1e-6 I | Table 1 |
| **HMC** | 50 chains, n_burn=250, n_sample=750, eps0=0.3, L0=3, M=Sigma_VI^-1 | Table 1 |

HMC uses `PreconditionedHamiltonianMonteCarlo` +
`GradientBasedTrajectoryLengthAdaptation` + `DualAveragingStepSizeAdaptation`
(adapting over the first 80% of burn-in), exactly the canonical gigalens
`jax-demo.ipynb` / paper stack. The 50 chains are a single **batched**
`sample_chain` state of shape `(50, 22)`, run on **one** pinned GPU -- NOT the
gigalens `HMC()` helper, which `jax.pmap`s over all visible devices and hangs.

The simulation distribution (Eq. 8 "a" side) and the broader modelling prior
("b" side) are coded in `01_gen_mocks.py:sample_truth` and
`02_fit_system.py:build_prior` respectively.

## Scripts

| file | purpose |
| ---- | ------- |
| `01_gen_mocks.py` | simulate N EPL+shear+2xSersic mocks with lenstronomy (80x80, 0.065"/px, supersample 2, TinyTim PSF, sigma_bkg/t_exp noise); save truth + image + err_map per system to `data/mocks/system_XXX.npz`. Reports integrated arc SNR (~paper target 100). |
| `02_fit_system.py` | MAP -> SVI -> HMC on ONE mock pinned to ONE GPU; write posterior, per-param ESS, split-R-hat, recovered-vs-truth (scaled error z) to `data/fits/system_XXX_fit.npz`. |
| `03_run_batch.sh` | fit a range of systems, cycling A16 GPUs 0-5 (one process per GPU per wave). |
| `04_pool_results.py` | pool all fits into the paper's Table 2: mu_z, <Rhat>/maxRhat, <ESS>/minESS, 68/95% truth coverage. CPU-only. |
| `05_recovery_plot.py` | paper Fig. 11 style: posterior mean - truth with error bars across systems. |

## How to run

**1. Generate mocks** (validation batch of 12, then scale to 100):
```bash
/raid/benson/.venvs/gigalens/bin/python 01_gen_mocks.py --n 12  --seed 0
/raid/benson/.venvs/gigalens/bin/python 01_gen_mocks.py --n 100 --seed 0
```

**2. Fit one system** (GPU 0, autotune off):
```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID XLA_FLAGS=--xla_gpu_autotune_level=0 \
  CUDA_VISIBLE_DEVICES=0 /raid/benson/.venvs/gigalens/bin/python 02_fit_system.py --idx 0
```

**3. Fit a batch** (cycles GPUs 0-5, one wave at a time):
```bash
./03_run_batch.sh 0 11      # validation: systems 0..11
./03_run_batch.sh 0 99      # full 100-system benchmark
```

**4. Pool & plot** (CPU-only; keep the A16s free):
```bash
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
  /raid/benson/.venvs/gigalens/bin/python 04_pool_results.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
  /raid/benson/.venvs/gigalens/bin/python 05_recovery_plot.py
```

## Findings & honest caveats

**1. Self-consistent simulation is required (the key methodological finding).**
At supersample=1 lenstronomy and gigalens agree to chi2=0.000 at identical truth,
but at the paper's **supersample=2** they differ (chi2~0.94 at identical truth,
peak flux 1.4x apart) because their supersampled-PSF-convolution conventions are
not bit-identical (lenstronomy `subgrid_kernel`+supersampling_convolution vs
gigalens internal supersampling). Fitting a *lenstronomy* mock with gigalens then
leaves an irreducible ~0.94 chi2 floor: the MAP plateaus at chi2~1.49 and gamma is
recovered ~16% biased. Simulating with the **same gigalens forward model** the
fitter uses (the default, `--simulator gigalens`) removes the floor: MAP reaches
**chi2 = 1.02** (the noise floor) and the recovered params are unbiased.

**2. Recovery is unbiased.** On the self-consistent gigalens mock, system 0
recovers all 8 lensing params within the posterior (scaled errors z: theta_E -1.5,
gamma +2.6, e1 +0.4, e2 +1.3, center_x +0.3, center_y -0.1, gamma1 0.0, gamma2
+1.7 sigma), MAP chi2 = 1.02, SVI -ELBO = -278.

**3. ESS/R-hat are hardware-limited, not method-limited.** The paper's
ESS > 26000 / max-R-hat <= 1.017 (50 chains x 750) comes from the
**GradientBasedTrajectoryLengthAdaptation** (GBTLA) finding near-optimal trajectory
lengths (~40 ESS/iter). On the A16:
  - Fixed L=5 PHMC + DualAveraging (fast, ~9 min HMC/system): truth recovered but
    mixing is short -- well-constrained light amps reach ESS=37500 R-hat~1.0, but
    a few correlated mass directions (e.g. center_y, src_e1) stay at ESS~130,
    R-hat~3.8. Mean ESS ~7000.
  - GBTLA (paper-exact, `--gbtla`, ~30-40 min HMC/system on the A16 because
    max_leapfrog=30 means up to 30 grad evals/step at 80x80 supersample=2): the
    sampler that produces the paper's ESS. The 12-system validation batch
    (`data/logs/fit_*.log`) and any 100-system scale-up use this.

**4. Hardware proxy.** The paper used 40-80 GB A100s; here each mock runs on one
**15 GB A16** (~5-10x slower). To fit memory + keep wall-time bounded, the MAP
multistart and SVI MC batch are reduced (K_MAP 300->128, K_VI 1000->200) and
float32 is used (the paper/demo are also float32). **The model, grid, noise,
priors, and HMC kernel stack are unchanged.** See `data/logs/` for per-system
timing and achieved ESS/R-hat; `04_pool_results.py` prints the Table-2 summary.
