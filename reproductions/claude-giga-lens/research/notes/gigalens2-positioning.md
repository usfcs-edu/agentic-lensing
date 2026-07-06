# GIGA-Lens 2.0 (arXiv:2606.30633) — positioning note

**ID check:** arXiv 2606.30633 is confirmed to be the correct paper ("GIGA-Lens 2.0: Strong-Lens
Modeling on Multiple GPU Nodes", submitted 2026-06-29). No correction needed. All numbers below
were verified against the arXiv HTML full text (v1) on 2026-07-06.

## (a) Summary

GIGA-Lens 2.0 (Huang et al. 2026, arXiv:2606.30633) is a multi-node/multi-GPU upgrade of the
GIGA-Lens Bayesian strong-lens modeling framework, demonstrated on up to **128 nodes / 512 A100
GPUs** on NERSC Perlmutter. It distributes the unchanged three-stage GIGA-Lens recipe — multi-start
**MAP → SVI (Gaussian surrogate) → HMC** — using JAX: `pmap` intra-node, `shard_map` inter-node,
`multihost_utils.process_allgather` for HMC (TFP-on-JAX; NCCL; Shifter containers). MAP and HMC
parallelize trivially (independent particles/chains per device); SVI averages per-device ELBO
gradients. Amdahl-fit parallel fractions at the largest workloads are **p = 0.983 (MAP,
n_MAP = 16000), 0.979 (SVI, n_VI = 16000), 0.968 (HMC)**. Models are EPL + external shear with
Sérsic lens light and Sérsic/shapelet sources; amplitudes are either free parameters ("forward")
or solved by linear inversion ("backward"); the statistical formulation explicitly "follow[s] the
same SVI formulation as in G22" (Gu et al. 2022), i.e., the standard per-pixel diagonal-Gaussian
likelihood — no likelihood changes. They fit **100 simulated systems** (n_MAP = 2000, n_VI = 1000,
n_HMC = 64 chains, 1500 samples + 500 burn-in) with truth-recovery residual plots (1σ posterior
bars), and one real system, **DESI J238.5690+04.7276** (HST imaging, 38 parameters, supersample = 2):
**128 min 24 s on 1 node → 25 min 12 s on 8 nodes (~5×)**, with identical reduced **χ²_ν = 0.8954**
to 4 decimals across node counts and, for the first time on a real system, **R̂ < 1.01 for all 38
parameters**. Other changes: AdaBelief (lr = 1e-4, β₁ = 0.95, β₂ = 0.99) replaces Adam as the
default SVI optimizer, and a new HMC mass-matrix adaptation uses ~10× longer burn-in instead of
relying on the SVI covariance alone. Stated future direction: use the speed to re-model systems
under varied assumptions to quantify systematics, and to tackle large-cutout, high-dimensional
systems (e.g., the Cosmic Carousel cluster lens); no survey-scale deployment or cosmography plan
is detailed.

## (b) Overlap check vs. our campaign pillars

| Topic | In GL2.0? | Notes |
|---|---|---|
| Correlated / drizzle noise likelihood (P1) | **No** | No mention of noise covariance, drizzling, or resampled-image noise anywhere; likelihood inherited unchanged from G22 (diagonal Gaussian). |
| Linear-parameter marginalization / Occam / ridge (P1) | **No** (touches machinery only) | "Backward" mode solves amplitudes by linear inversion but there is no analytic marginalization, no Occam factor, no regularization/ridge discussion. Our ridge-marginalized likelihood extends exactly this solve-only step. |
| PSF sampling conventions | **No** (footnote only) | One footnote: image grid "can be supersampled by some integer factor, typically 1 or 2"; first real-HST run at supersample = 2. Nothing on pixel-integrated vs. δ-sampled PSF kernels (the Foundry-I defect class). |
| Multimodal posteriors / tempering / flow-assisted sampling (P2) | **No** | Single Gaussian SVI surrogate seeds HMC — implicitly unimodal; no tempering, nested sampling, normalizing flows, or NeuTra. |
| Cross-pixel-scale consistency | **No** | Single-instrument fits only; no multi-band/multi-instrument joint modeling. |
| Uncertainty calibration | **Partial** | Truth-vs-posterior residual plots with 1σ bars on 100 sims; no formal coverage statistics or calibration analysis. |

**Verdict: orthogonal, as expected.** They scale the same diagonal-likelihood MAP→SVI→HMC recipe;
we change the likelihood (P1) and the sampler (P2). The only honest contact points: (i) their
backward linear-inversion step is the machinery P1's ridge marginalization upgrades; (ii) their
R̂ < 1.01-on-a-real-system convergence result is a natural baseline for P2's sampler benchmark;
(iii) the supersample = 2 footnote brushes past — but does not address — PSF kernel-sampling
conventions.

## (c) Positioning paragraph (liftable)

GIGA-Lens 2.0 (Huang et al. 2026) demonstrates that the MAP→SVI→HMC strong-lens inference recipe
scales essentially unmodified to 512 A100 GPUs, achieving R̂ < 1.01 on all 38 parameters of a real
DESI lens in ~25 minutes on eight nodes. That work scales the computation while leaving the
statistical model untouched: the likelihood remains the per-pixel diagonal Gaussian of Gu et al.
(2022), source amplitudes are solved (not marginalized) in linear inversion, and the single
Gaussian SVI surrogate presumes a unimodal posterior. Our campaign is complementary on both axes:
Pillar 1 replaces the diagonal likelihood with a correlated-noise (drizzle-aware) likelihood and
analytically ridge-marginalizes the linear parameters with the attendant Occam term, and Pillar 2
benchmarks samplers on the multimodal posteriors that a single-surrogate SVI→HMC pipeline cannot
represent, supplying a neural-transport recipe where needed. Faster wrong likelihoods converge
faster to the wrong posterior; our contribution is to fix what is being scaled.

## (d) Citation

- **Title:** GIGA-Lens 2.0: Strong-Lens Modeling on Multiple GPU Nodes
- **Authors:** Xiaosheng Huang, Linus Upson, Nicolas Ratier-Werbin, Harry Lu, Sean Xu, Elden Yap,
  Evan Odell, Ansel Parke, Harsh Ambardekar, Saul Baltasar, Nestor Demeure, Bradley Richardson,
  Andi Gu, Yuan-Ming Hsu, Junyi Liu
- **arXiv:** 2606.30633 [astro-ph.CO] (cross-list astro-ph.GA), submitted 29 June 2026
- **Comments:** 21 pages, 8 figures, 3 tables; preprint (no journal venue stated as of 2026-07-06)
- **Compute:** NERSC Perlmutter, up to 128 nodes / 512 A100s

## Adjacent-work watch-list

No 2025–2026 paper was found that directly implements a correlated-noise strong-lens likelihood or
a multimodality/sampler benchmark — both pillars appear open. Nearest-adjacent:

1. **arXiv:2406.08484** — "Exploiting the diversity of modeling methods to probe systematic biases
   in strong lensing analyses" (2024). Closest published statement of P1's motivation: explicitly
   flags that JWST imaging shows strongly correlated noise (citing Rigby et al. 2023) and that the
   uncorrelated-Gaussian assumption in lens modeling "should be reassessed" — motivation only, no
   implementation. Watch for a follow-up that builds the likelihood.
2. **arXiv:2511.04792** — "Blind Strong Gravitational Lensing Inversion: Joint Inference of Source
   and Lens Mass with Score-Based Models" (Nov 2025). Score-based (diffusion) joint posterior over
   source + lens; adjacent to P2 but simulation-based inference, not a likelihood-based sampler
   benchmark or neural transport for MCMC.
3. **arXiv:2410.22573** — "Flow Matching for Posterior Inference with Simulator Feedback" (2024,
   lensing showcase). Flow-based NPE with gradients from a differentiable lens simulator, compared
   against MCMC; nearest published relative of P2's flow-assisted direction, but amortized NPE
   rather than transport-assisted sampling of an exact likelihood.
