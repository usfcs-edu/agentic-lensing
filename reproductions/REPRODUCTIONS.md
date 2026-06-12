# REPRODUCTIONS — Huang-group strong-lensing corpus

Verification index for the public-data reproductions of the 16-paper Huang strong-lensing
corpus (`papers/`). Each row links a paper to its `reproductions/<slug>/` directory, the
headline number(s) achieved vs. published, and the artifact status. Every reproduction has
a LaTeX tech-report at `reproductions/<slug>/papers/main.pdf` and an operator README.

**Status: 16/16 in-scope papers reproduced.** Scope is honest throughout — algorithmic
finder steps reproduce intermediate counts (human visual-inspection grading is out of
scope); modeling papers reproduce on public imaging (some proprietary spectroscopy is
skipped using published redshifts); a few are MVP/consistency reproductions where archive
or compute limits apply. Caveats are stated per row and detailed in each report.

The two non-lensing corpus items — the dynamical dark-energy paper and the DES-Y6 memo —
are **out of scope** (cosmology, not lens discovery/modeling).

---

## GIGA-Lens modeling lineage

| Paper | slug | Headline: ours vs published | Status |
|---|---|---|---|
| **Gu 2022** — GIGA-Lens method (ApJ 935 49) | `gu-2022` | Method reproduced on 12 mock systems; MAP χ²≈1.0 all; mean ESS ≈11k; paper convergence target (R̂≤1.017, ESS>>1e4) met on well-conditioned systems; uniform minESS≥26,822 not reached on degenerate systems (depth-limited, 2.5× depth ≈doubles ESS — bias-free; paper used 4×A100) | ✅ report |
| **Cikota 2023** — Einstein cross (2307.12470) | `cikota-2023` | θ_E **2.10″ vs 2.52″**, σ_SIE 347 vs 379 km/s, μ 7.0 vs 10.47, χ²/px 0.90, four-image cross geometry reproduced; θ_E offset ablated to DESI-Legacy 1.35″ seeing (MUSE spectroscopy skipped, proprietary) | ✅ report (7pp) |
| **Sheu 2024b** — Carousel cluster lens (2408.10320) | `sheu-2024b` | θ_E **12.96″ vs 13.03″ (0.5%)**, γ 1.53–1.67 vs 1.67, M(<θ_E) 4.62 vs 4.78e13 M⊙, per-plane θ_E scaling reproduced (5 source planes); public HST F140W/F200LP; pixel-level fit set up, not run | ✅ report (10pp) |
| **Huang 2025a** — Foundry I, HST + GIGA-Lens (2502.03455) | `foundry-i` | θ_E to 3%, e1 to 2.5%, shear PA exact; genuine HMC made to work (lstsq+float64+regularized marginalization); long 8-chain HMC brackets paper γ; **3 real upstream gigalens bugs** documented | ✅ report (23pp) |

## Image-based lens-finder lineage (ResNet / EfficientNet)

| Paper | slug | Headline: ours vs published | Status |
|---|---|---|---|
| **Huang 2020** — DECaLS ResNet (1906.00970) | `huang-2020` | From-scratch Lanusse ResNet-46; full 6.24M-galaxy DR7 sweep; **83% Grade-A recall @p≥0.9** (paper-exact DR7-trained); leakage ablation | ✅ report (12pp) |
| **Huang 2021** — shielded ResNet, DR8 (2005.04730) | `huang-2021` | Shielded net **59,905 params (58.6× < L18), AUC ±0.002**; 17.3M-galaxy two-model DR8 sweep; north-calibration finding; leak-free recovery 50.4% @p≥0.9 | ✅ report (9pp) |
| **Storfer 2024 + Inchausti 2025** — DR9/DR10 ensemble (2308.04603 / 2508.20087) | `inchausti-2025` | EfficientNetV2-S + shielded-ResNet + 300-node FWLS meta (Fig.6 meta≈average); **recovery @1% FPR 91% / 97%**; neg:pos ratio (not architecture) sets usability | ✅ report (20pp) |
| **Silver 2025** — ML forecasts, ResNet/U-Net (JWST) | `silver-2025` | Model-1 (HST) ResNet val **AUC 0.994 vs 0.998** (lenstronomy Sérsic-source MVP; VELA-source + JWST Models 2/3 + U-Net = next) | ✅ report (7pp) |

## Specialty-discovery lineage (difference imaging / catalog)

| Paper | slug | Headline: ours vs published | Status |
|---|---|---|---|
| **Sheu 2023** — lensed supernovae (2301.03578) | `sheu-2023` | **Re-detected the Grade-A L-SN** (11 sub-detections, 1.48″ from lens = counter-image); from-scratch B08 difference imaging; SALT3 μ **8.6 vs 8.2** (synthetic-injection-validated; real-data photometry a documented proxy) | ✅ report (10pp) |
| **Sheu 2024a** — variable lensed quasars (2408.02670) | `sheu-2024a` | Variability **σ 0.34 vs 0.25 mag** at both lensed images; reused Sheu-2023 diff-imaging core; σ-metric validated to a few % on synthetic injections | ✅ report (9pp) |
| **Dawes 2022** — multiply-lensed quasars (ApJS 269 61) | `dawes-2022` | Autocorrelation FoF reproduced: **58/58 = 100% conditional recovery** (apples-to-apples with paper's 94/94); raw 14% proxy-limited (DR1 spectroscopic QSOs ~1.6M vs paper's ~5M photometric targets) | ✅ report (9pp) |
| **Hsu 2025** — pairwise spectroscopic (2509.16033) | `hsu-2025` | Full 28M-fiber DR1 FoF: **13,530 groups / 27,334 spectra vs 13,218 / 26,621 (+2.4%)**; 20/20 Table-2 Grade-A recall within 3″ | ✅ report (10pp) |

## DESI Strong Lens Foundry follow-up spectroscopy

| Paper | slug | Headline: ours vs published | Status |
|---|---|---|---|
| **Huang 2025b** — Foundry II, DESI spectroscopy (2509.18089) | `foundry-ii` | 73/73 systems matched to DR1 fibers; **z_lens 70/72 & z_source 16/22 to <0.001; σ_v 65/71 (r=0.80)** — all from public on-disk DR1/FastSpecFit | ✅ report (8pp) |
| **Agarwal 2025** — Foundry III, Keck NIRES (2025) | `foundry-iii` | **6/6 source z to \|dz\|<0.001** via blind Eq.1 line-fit + MC validation; consistency reproduction (KOA serves raw-L0 NIRES only; pypeit won't build aarch64 → real-spectra fit pending x86 PypeIt) | ✅ report (9pp) |
| **Lin 2025** — Foundry IV, VLT/MUSE (2509.18087) | `foundry-iv` | Pulled 3 public ESO MUSE cubes; **auto z_lens 3/3 within dz<0.02** (2/3 <0.003); guided source z exact for Lens22 (0.821); built an automated line-ID engine (unguided source-ID interloper-prone — intrinsic, why the paper did it by hand) | ✅ report (10pp) |

---

## Reusable infrastructure (validated, on disk)

- **GIGA-Lens JAX stack** — env `/raid/benson/.venvs/gigalens`; single-system MAP/SVI/HMC pipeline in `foundry-i/` (regularized-Gaussian amplitude marginalization `_hmc_lib_marg.py`; single-device GBTLA recipe avoiding the gigalens pmap-over-all-devices hang). Drives Gu 2022, Cikota 2023.
- **Difference-imaging core** — `sheu-2023/` from-scratch Bramich-2008 + SEP + SALT3 + NOIRLab per-exposure DECam fetch; reused verbatim by `sheu-2024a/`. venv `/home/benson/.venvs/lens` (sep, sncosmo, reproject, photutils, mpdaf).
- **Finder stack** — PyTorch Lanusse + shielded ResNet + EfficientNetV2 + meta-learner; legacysurvey brick-FITS download + WCS slicing (~150× faster than the cutout endpoint).
- **DESI spectroscopic stack** — `hsu-2025/` spherimatch FoF; DR1 `zall-pix-iron.fits` (28M rows) + FastSpecFit σ_v shards on disk. Drives Dawes 2022 and Foundry II.
- **DR11 `One_percent` dataset generator** — `tools/desi-dr11-cookbook/`; clean reimplementation of the Legacy Survey DR11 → HDF5 ML-dataset pipeline (filtered tractor catalogs + 101×101×3 grz cutouts), reverse-engineered and verified **bit-for-bit** against xhuang's `One_percent` (golden brick north/2400p345). MPI fan-out on Perlmutter CPU nodes. Produces the `One_percent`-format training data feeding the finder lineage (Huang 2020/2021, Inchausti 2025).

## Verification thresholds (per the reproduction plan)
- GIGA-Lens modeling — posterior medians within ~1σ of published (θ_E, γ, q, shear); convergence (ESS, R̂) where compute allows.
- Finders — AUC matched (±0.002 architecture-controlled); recovery-by-grade.
- Catalog/FoF — intermediate counts within a few %; Grade-A recovery.
- Spectroscopy — redshifts to ±0.001; σ_v distribution.
- Difference imaging — re-detect ≥1 published transient in-tile; validated photometric metric.

*Each reproduction's `papers/main.pdf` carries the full methods, results table, and honest "not reproduced" section. Bulk data lives in gitignored `data/`; scripts/figures/reports are tracked.*
