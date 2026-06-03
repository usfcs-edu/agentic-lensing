# Plan: Map the 5 local repos to the 16 Huang-group papers and chart a reproduction roadmap

## Context

Greg Benson is joining Xiaosheng Huang's strong-gravitational-lensing ML group and wants to reproduce as many of the 16 published-paper results as possible. The repo working tree contains five repos under `/raid/benson/git/agentic-lensing/lensing-repos/` (symlink to `/raid/benson/lensing-repos/`) plus a 16-paper PDF corpus under `papers/` and an existing onboarding document at `plans/AGENTIC_LENSING_ONBOARDING_PLAN.md`. The user explicitly chose: (a) document the four non-gigalens repos as a separate "foundation-model / simulation" track, (b) assume Local + NERSC Perlmutter + DESI Collaboration data access for compute/data, (c) cover all four paper lineages (GIGA-Lens, lens-finder, discovery-modality, Foundry).

**Key surprise to flag up-front:** only **one** of the five local repos directly implements one of the 16 papers (`gigalens` → Gu 2022). The other four are SpectrumFM-adjacent infrastructure (codecs, redshifty) and downstream-tooling efforts (galaxy-search, agentic-cosmic-webb-sim) rather than re-implementations of the discovery papers. The Huang group's image-based lens-finder code (Huang 2020/2021, Storfer 2024, Inchausti 2025) does **not appear to be publicly released** — they distribute candidate catalogs via `sites.google.com/usfca.edu/neuralens/` but not the training code. Reproducing those papers will require either re-implementing from the methods sections or obtaining the code through collaboration access.

The deliverable from this plan is a written reproduction roadmap, not code; the implementation phase will be future work whose shape depends on which papers Greg chooses to prioritize.

---

## 1. Local repo inventory

| Repo | Origin | Last commit | What it actually is | Paper it implements |
|---|---|---|---|---|
| `gigalens/` | `giga-lens/gigalens` on `multi-node` branch (HEAD = "update to GIGA-Lens 2.0 release version") | Active upstream | **GIGA-Lens 2.0** — current development line including multi-node training, float64 throughout, JAX-first. Other available branches: `master`, `multinode-2025`, `single-node`, `tf-demo-update`, `xh-dev` (Xiaosheng Huang's personal dev branch). Demo notebooks under `notebooks/`. | **Gu 2022 lineage**; the right baseline for reproductions targeting Cikota 2023, Sheu 2024b, Huang 2025a. |
| `gigalens-archive/` | `seanxuseanxu/gigalens` (master) | 2022-05-26 frozen fork | Original GIGA-Lens code as released with the Gu+2022 paper. Mass profiles (SIS, SIE, EPL, NFW, shear) + light profiles (Sérsic, shapelets) on TF & JAX backends. Demo notebooks: `tf-demo.ipynb`, `jax-demo.ipynb`, `shapelets-demo.ipynb`. | **Gu 2022** as-published (reproduce-the-original-paper baseline only). |
| `codecs/` | `cosmologyfoundation/codecs` (main) | 2026-04-26 | Discrete-token spectrum codec (Mamba + Residual FSQ) trained on DESI DR1 at NERSC. Successor to AION-1's ConvNeXt + Flat LFQ spectrum tokenizer. | **None of the 16.** SpectrumFM-adjacent. |
| `galaxy-search/` | `cosmologyfoundation/galaxy-search` (main) | 2026-04-09 | CLIP-style semantic search over galaxy images: AION image-encoder + LLM (SGLang) caption generation + projection-head training, evaluated on GZ-DECaLS. **Not a ResNet lens finder.** | **None of the 16.** |
| `redshifty/` | `cosmologyfoundation/redshifty` (main) | 2026-05-14 | Unimodal transformer foundation model for DESI spectra with an auxiliary-redshift head (Approaches A/B). Critiques AION-1's "redshift token treated identically to spectral tokens." Directly prefigures the **SpectrumFM proposal** (`proposals/doe_genesis_spectrumfm_project_narrative_v7.docx`). | **None of the 16.** SpectrumFM Phase-I prototype. |
| `agentic-cosmic-webb-sim/` | `kvinneslandn-ML-AI/agentic-cosmic-webb-sim` (greg/pipeline-setup) | 2026-05-22 | Multi-band JWST/COSMOS-Web lens-simulation pipeline (4 NIRCam bands, SIE+shear, 125×125 px), CVAE generative modeling. Calibrated to Nightingale et al. 2025 (COWLS I) and Mahler et al. 2025 (COWLS II). Greg has an active branch. | **None of the 16** (closest analogue is **Silver 2025**, which uses simulated JWST images for the same end). |

**Setup note:** as of this plan revision, the local working tree has the active `giga-lens/gigalens` repo at `lensing-repos/gigalens/` on the `multi-node` branch (GIGA-Lens 2.0), with the historical 2022 fork preserved at `lensing-repos/gigalens-archive/`. The multi-node branch is the right baseline for any 51-system Foundry-I-scale reproductions; the `xh-dev` branch is worth comparing against since it likely carries Xiaosheng Huang's own work-in-progress.

---

## 2. Paper-by-paper mapping and reproduction status

Grouped by the four lineages Greg asked to cover. Each entry lists: the *named software* in the paper, the *primary public repo* (or "unreleased"), the *data source*, and a one-line reproduction note.

### 2.1 GIGA-Lens lineage (Gu 2022 → applications)

| Paper | Local repo | Public repo(s) | Public data | Reproduction note |
|---|---|---|---|---|
| **Gu 2022** (arXiv:2202.07663) | ✅ `gigalens/` | `giga-lens/gigalens` (active); Zenodo DOI `10.5281/zenodo.3743493` referenced from PDF | Synthetic systems generated with `lenstronomy`; demo `.npy` ships in repo | **Fully reproducible locally.** Run the three demo notebooks. Benchmarks (105 s on 4 A100s) need GPU. |
| **Cikota 2023** (Einstein cross, arXiv:2307.12470) | ⚠️ `gigalens/` (modeling only) | `giga-lens/gigalens`; lens-finder code unreleased | DESI-253.2534+26.8843 candidate from Huang 2021; VLT/MUSE archival (ESO Prog. ID 0111.B-0400); HST archival | **Partially reproducible.** The single-system GIGA-Lens fit is reproducible if you can get the HST cutout. The MUSE redshift confirmation requires re-running PypeIt on archival MUSE data. |
| **Sheu 2024b** (Carousel, arXiv:2408.10320) | ⚠️ `gigalens/` (modeling only) | `giga-lens/gigalens` | HST + VLT/MUSE archival; lens at z_L=0.49 | **Partially reproducible.** Lens-modeling is reproducible; the discovery itself came from Huang 2021's catalog. |
| **Huang 2025a — Foundry I** (HST, arXiv:2502.03455) | ⚠️ `gigalens/` (modeling only) | `giga-lens/gigalens` | HST GO-15867 SNAPshot imaging (public MAST); 51 confirmed candidates from Huang/Storfer | **Reproducible at single-system level.** Run GIGA-Lens on DESI-165.4754−06.0423 as published. Full 51-system reproduction is multi-GPU work but mechanically the same. |

### 2.2 Lens-finder lineage (ResNet/EfficientNet image classifiers)

| Paper | Local repo | Public repo(s) | Public data | Reproduction note |
|---|---|---|---|---|
| **Huang 2020** (DECaLS, arXiv:1906.00970) | ❌ none | **Code unreleased.** Candidate catalogs at `sites.google.com/usfca.edu/neuralens/publications/lens-candidates-huang-2020b`. Reference implementation: Lanusse et al. 2018 CMU DeepLens (`github.com/McWilliamsCenter/CMU_DeepLens`). | DECaLS DR3/DR5 imaging at `legacysurvey.org`; ~700 training lenses + 13k negatives | **Method reproducible from paper**, code is not. Re-train a ResNet on DECaLS cutouts + the published candidate catalog as positive training data. |
| **Huang 2021** (DESI Legacy, arXiv:2005.04730) | ❌ none | Code unreleased; catalog on NeuraLens site | DESI Legacy Surveys DR7/DR8 (`legacysurvey.org`) | Same as Huang 2020 but scaled to 14k deg² and with "shielding" (1×1 conv) layers — needs re-implementation from §3 of the paper. |
| **Storfer 2024** (DR9, arXiv:2309.18089, ApJS 274 16) | ❌ none | Code unreleased; catalog likely on NeuraLens / via collaboration | DESI Legacy DR9 (`legacysurvey.org`); ~19k deg² | Builds on Huang 2021 — same shielded ResNet on DR9, refined A/B/C/D grading. Re-implementation feasible if Huang 2021 is in hand. |
| **Inchausti 2025** (DR10, arXiv:2508.20089/20087) | ❌ none | Code unreleased | DR10 (legacysurvey.org); training on NERSC Perlmutter (4-GPU) | Dual ResNet + EfficientNetV2 ensemble with 300-node meta-learner. AUC 0.9989. Re-implementation needs both the trained backbones and the meta-learner architecture from the paper. |

**Strategy for the lens-finder lineage:** since the code is closed, the practical reproduction path is either (a) request the code through Huang's group directly, or (b) re-implement the architecture from the paper using EfficientNet/ResNet from `torchvision` plus the published candidate catalogs as positive training data. Collaboration data access (option chosen) makes (a) the right move.

### 2.3 Specialty-discovery lineage

| Paper | Local repo | Public repo(s) | Public data | Reproduction note |
|---|---|---|---|---|
| **Sheu 2023** (lensed SNe, arXiv:2301.03578) | ❌ none | `thomasvrussell/sfft` (SFFT image subtraction); B08 (Bramich 2008) is referenced but no standard repo; `kbarbary/sep`; SALT3 via `sncosmo` | DESI Legacy multi-epoch coadds | Pipeline = SFFT + B08 difference imaging → SEP detection → SALT3 light-curve fit. Glue code is custom and not released; rebuild from §3 of the paper using the public SFFT/SEP/SALT3 stack. |
| **Sheu 2024a** (variable lensed quasars, arXiv:2408.02670) | ❌ none | Same as Sheu 2023 plus color/PSF/variability cuts (custom) | Same multi-epoch coadds + Dawes 2022 catalog | Variant of the same pipeline. Cross-references Dawes 2022 outputs as a seed catalog. |
| **Dawes 2022** (multiply-lensed quasars, ApJS 269 16) | ❌ none | Custom autocorrelation pipeline, **not released**. Uses `astropy.coordinates`, `Gaia` DR3 archive, spatial autocorrelation on quasar coords. | DESI quasar sample; Gaia DR3 (`gea.esac.esa.int`) | Pure catalog-level analysis (no ML training). Cleanest reproduction path: rebuild from §3–4 with publicly available DESI QSO catalog and Gaia DR3 proper motions. |
| **Hsu 2025** (pairwise spectroscopic / dimple lenses, arXiv:2509.16033) | ❌ none (`redshifty/` is adjacent but not this paper) | ✅ `technic960183/spherimatch` — explicitly named in the paper for the friends-of-friends spatial clustering | DESI DR1 28M spectra (`data.desi.lbl.gov`); DR10 imaging | **Highest-value scientific target.** Pipeline = `spherimatch` (sky-coord FoF, 3″ link) → redshift-ratio cut → visual inspection. Reproducible at the candidate-generation step; the VI grading step requires team participation. |

### 2.4 DESI Strong Lens Foundry (confirmation papers)

| Paper | Local repo | Public repo(s) | Public data | Reproduction note |
|---|---|---|---|---|
| **Huang 2025a — Foundry I** (HST, arXiv:2502.03455) | ⚠️ `gigalens/` | `giga-lens/gigalens` | HST GO-15867 (MAST public) | Already covered in §2.1. |
| **Huang 2025b — Foundry II** (DESI spectroscopy, arXiv:2509.18089) | ❌ none | DESI public stack: `desihub/redrock`, `desihub/fastspecfit`, `desihub/desispec` | DESI EDR/DR1 + Secondary Target Program metadata | Run Redrock + FastSpecFit on the Foundry candidate fibers. Requires DESI collaboration metadata identifying which fibers belong to the Secondary Target Program. |
| **Agarwal 2025 — Foundry III** (Keck NIRES, arXiv:2501.08066) | ❌ none | `pypeit/PypeIt` (`pypeit.readthedocs.io`) | Keck/NIRES raw frames at KOA (`koa.ipac.caltech.edu`); public after PI's proprietary window | Reduce NIRES echellette frames with PypeIt → fit [O II]/Hα/Hβ to get z_s for 8 systems. Mechanically reproducible if KOA access is granted (collaboration channel). |
| **Lin 2025 — Foundry IV** (VLT/MUSE, arXiv:2509.18078) | ❌ none | MUSE DRS or `pypeit` for MUSE mode; `astropy`, `mpdaf`, `marz` for emission-line fitting | ESO MUSE archive (public after proprietary window) | Same shape as Foundry III but IFU. 75 candidates → 48 fully confirmed. Reduce + measure redshifts; lens modeling via GIGA-Lens. |

---

## 3. External public software dependencies (consolidated)

Pin these as the public-dependency surface for any reproduction work:

| Tool | Purpose | URL | Used in |
|---|---|---|---|
| `giga-lens/gigalens` | Differentiable Bayesian lens modeling | github.com/giga-lens/gigalens | Gu 2022; Cikota 2023; Sheu 2024b; Foundry I |
| `lenstronomy` | Reference simulator + non-differentiable lens modeling | github.com/lenstronomy/lenstronomy | Gu 2022 (comparison); Silver 2025 (simulator); training-data synthesis everywhere |
| `thomasvrussell/sfft` | Saccadic Fast Fourier Transform image subtraction | github.com/thomasvrussell/sfft | Sheu 2023; Sheu 2024a |
| `kbarbary/sep` | Source Extractor in Python | github.com/kbarbary/sep | Sheu 2023; Sheu 2024a; Hsu 2025 |
| `technic960183/spherimatch` | FoF spatial clustering for fiber spectra | github.com/technic960183/spherimatch | **Hsu 2025 pairwise/dimple** |
| `desihub/redrock` | DESI template-fit redshift pipeline | github.com/desihub/redrock | Foundry II; Hsu 2025 inputs |
| `desihub/fastspecfit` | DESI derived spectrophotometric quantities | github.com/desihub/fastspecfit | Hsu 2025 σ_v; Foundry II |
| `desihub/desispec` | DESI raw-to-spectra pipeline | github.com/desihub/desispec | Foundry II baseline |
| `pypeit/PypeIt` | Echelle/IFU spectroscopic reduction | github.com/pypeit/PypeIt | Foundry III (NIRES); some MUSE work |
| `sncosmo` + SALT3 | Type-Ia SN light-curve fitting | github.com/sncosmo/sncosmo | Sheu 2023 |
| `astropy`, `photutils`, `astroquery` | Foundational | astropy.org | All |
| `Gaia DR3` | Stellar proper motion / parallax filters | gea.esac.esa.int | Dawes 2022 |
| `The Tractor` (Lang+2016) | Survey-level model photometry — already baked into the Legacy Surveys catalogs | github.com/dstndstn/tractor | Upstream of every paper (DR8/9/10 catalog source) |

## 4. External public data sources

| Survey/instrument | Endpoint | Access tier | Used in |
|---|---|---|---|
| DESI Legacy Imaging Surveys (DR8/9/10) | `legacysurvey.org` | Public | Huang 2020/21, Storfer 2024, Inchausti 2025, Sheu 2023/24a, Dawes 2022, Hsu 2025 |
| DESI DR1 spectra | `data.desi.lbl.gov` | Public | Hsu 2025, Foundry II |
| Pre-DR1 / VI campaigns | NERSC `/global/cfs/cdirs/desi/...` | Collaboration | Foundry II, SpectrumFM prep |
| HST archives | `mast.stsci.edu` (GO-15867) | Public | Foundry I, Cikota 2023, Sheu 2024b |
| Keck Observatory Archive (NIRES) | `koa.ipac.caltech.edu` | Public after PI proprietary | Foundry III |
| ESO MUSE archive | `archive.eso.org` | Public after proprietary | Foundry IV, Cikota 2023, Sheu 2024b |
| Gaia DR3 | `gea.esac.esa.int` | Public | Dawes 2022 |
| COSMOS-Web NIRCam mosaics (for `agentic-cosmic-webb-sim`) | `exchg.calet.org/cosmosweb-public/DR0.5/NIRCam/Jan23/` | Public | Not a Huang paper; relevant to Silver 2025 analogue |
| Lens-candidate catalogs (Huang/Storfer/etc.) | `sites.google.com/usfca.edu/neuralens/` | Public | All Huang lens-finder papers |

## 5. Code that is missing or not publicly released

These are the gaps any serious reproduction effort has to either re-implement from the paper or obtain via collaboration:

1. **Lens-finder training code** (Huang 2020/2021, Storfer 2024, Inchausti 2025) — the shielded ResNet, the dual ResNet+EfficientNetV2 ensemble, and the 300-node meta-learner are described in the methods sections but no GitHub release was found. **Path forward:** request from Huang/Storfer/Inchausti; otherwise re-implement using `torchvision` ResNet + `timm` EfficientNetV2 with the published catalogs as positive training data.
2. **Difference-imaging glue** (Sheu 2023 / Sheu 2024a) — SFFT + B08 + SEP + SALT3 are public, but the orchestration code and the variability/morphology cuts are not. **Path forward:** rebuild from the papers' §3.
3. **Dawes 2022 autocorrelation pipeline** — pure analysis code on top of public catalogs; not released. **Path forward:** rebuild from §3–4 (small project).
4. **Inchausti 2025 meta-learner weights** — even the architecture is published, but trained weights are not. Re-training is feasible on the NERSC Perlmutter allocation Greg is targeting.
5. **Silver 2025 U-Net for substructure localization** — no public repo found. Adjacent public code includes `dangilman/samana` for substructure lensing, but the architecture and training data (VELA + Cosmodc2 synthesized JWST images) need to be reconstructed. The local `agentic-cosmic-webb-sim/` is the closest local asset — it produces JWST-like images that could be adapted as training data.
6. **Foundry follow-up reduction recipes** (III, IV) — PypeIt config files and emission-line line-lists used by Agarwal 2025 and Lin 2025 are not packaged; methods sections describe the parameters.

## 6. Recommended reproduction roadmap

Phased so that the easiest, highest-leverage reproductions land first.

**Phase 0 — local sanity check (1 day).** Set up the local `gigalens/` repo (already pinned to `multi-node`, GIGA-Lens 2.0) and run the demo notebooks. Confirm the local GPU + JAX environment works end-to-end. Optionally also run the same demo from `gigalens-archive/` to confirm the as-published 2022 numbers reproduce. This validates the GPU+JAX environment that the rest of the GIGA-Lens lineage needs.

**Phase 1 — GIGA-Lens application papers (1–2 weeks).** Run the published single-system fits from Cikota 2023, Sheu 2024b, and Foundry I against archival HST and MUSE data. This is the most-mechanical-reproducible cluster of papers: same code, three different inputs.

**Phase 2 — Hsu 2025 pairwise pipeline (2–4 weeks).** This is the highest-scientific-upside reproduction. Clone `technic960183/spherimatch`, pull DESI DR1 spectra (collaboration access already granted), run the FoF grouping + redshift-ratio cut. Output should match the paper's 26,621 candidate spectra in 11,848 fiber pairs. Visual-inspection step requires team coordination.

**Phase 3 — Lens-finder lineage (4–8 weeks).** Approach via two tracks running in parallel: (a) ask Huang/Storfer/Inchausti for the training code through the collaboration; (b) start a re-implementation in `torchvision`/`timm` against the published Huang 2021 candidate catalog. Train on NERSC Perlmutter (the paper's own training rig). Target reproducing Huang 2021's AUC = 0.992 first as the canonical milestone.

**Phase 4 — Difference-imaging discoveries (Sheu 2023/2024a, 4 weeks).** Build the SFFT+B08+SEP+SALT3 pipeline. Re-run on a small DR9 sky tile and compare against the 7 lensed-SN candidates from Sheu 2023.

**Phase 5 — Catalog-level Dawes 2022 (1 week).** Smallest project; reproduce the autocorrelation-based quasar lens search on the DESI quasar sample.

**Phase 6 — Foundry follow-up reductions (variable, depends on archive access).** PypeIt reductions for NIRES (Foundry III) and MUSE (Foundry IV); cross-check with the source-redshift tables in the papers.

**Phase 7 — Silver 2025 / JWST forecasts (open-ended).** Adapt `agentic-cosmic-webb-sim/` to generate the substructure-localization training set, then build a U-Net classifier head. This dovetails with Greg's existing branch on that repo and could become an original contribution rather than a pure reproduction.

## 7. Verification

For each reproduced result, the verification recipe is:

- **GIGA-Lens reproductions** — compare posterior parameter medians (θ_E, γ_EPL, q, γ_ext) to the published values; agreement to within ~1σ is the threshold.
- **Lens-finder reproductions** — compute AUC on the held-out test split published with each paper; match the published number (0.98 / 0.992 / 0.9997 / 0.9989).
- **Hsu 2025** — count candidate spectra, fiber pairs, and final lens-candidate counts; should match the paper's 26,621 / 11,848 / 2,046 / 318 to within the stochastic FoF threshold.
- **Difference imaging** — re-detect at least one of the 7 published lensed-SN candidates from Sheu 2023 in the same DR9 tile.
- **Dawes 2022** — match the count of 436 multiply-lensed/binary-quasar candidates on the same DESI quasar input catalog.
- **Foundry reductions** — match source redshifts in the published tables to within ±0.001 in z.

All reproductions should be logged in a `reproductions/` subdirectory under `agentic-lensing/`, with one notebook or script per paper and a top-level `REPRODUCTIONS.md` index recording the verification numbers achieved.

---

## Critical files / repos referenced

- Active GIGA-Lens working tree: `/raid/benson/git/agentic-lensing/lensing-repos/gigalens/` (branch `multi-node`, GIGA-Lens 2.0; alternate branches incl. `xh-dev`)
- Frozen Gu-2022 baseline: `/raid/benson/git/agentic-lensing/lensing-repos/gigalens-archive/`
- Local repos *not* used for the 16-paper reproduction (separate SpectrumFM track): `codecs/`, `galaxy-search/`, `redshifty/`, `agentic-cosmic-webb-sim/`
- Active upstream GIGA-Lens: `github.com/giga-lens/gigalens` (PyPI: `pip install gigalens`)
- Key external public repos: `technic960183/spherimatch` (Hsu 2025), `thomasvrussell/sfft` (Sheu 2023/24a), `desihub/redrock`, `desihub/fastspecfit`, `desihub/desispec`, `pypeit/PypeIt`, `kbarbary/sep`, `sncosmo/sncosmo`, `lenstronomy/lenstronomy`, `dstndstn/tractor`
- Candidate catalogs (public, paper outputs): `sites.google.com/usfca.edu/neuralens/`
- Onboarding context already on disk: `plans/AGENTIC_LENSING_ONBOARDING_PLAN.md`
- Existing memory entries to consult when implementing: `memories/memory/project_huang_lensing.md`, `memories/memory/reference_paper_corpus.md`

When implementation begins, expect to also update `MEMORY.md` with a new entry recording which papers ended up fully reproduced vs. only catalog-verified, since that state is exactly the kind of thing future sessions will need.
