# Agentic Lensing Onboarding Report

**For:** Prof. Greg Benson, University of San Francisco
**Subject:** Prof. Xiaosheng Huang's strong-gravitational-lensing ML research program, the SpectrumFM DOE Genesis proposal, and concrete agentic-AI techniques to apply to both
**Date prepared:** May 25, 2026
**Status:** Internal onboarding report; not for circulation outside the collaboration

---

## Abstract

This report brings Prof. Greg Benson — a computer scientist with deep expertise in agentic coding and multi-agent systems, joining Prof. Xiaosheng Huang's gravitational-lensing ML group as the agentic-AI Co-I on the DOE Genesis SpectrumFM proposal — up to speed on the scientific, mathematical, and machine-learning context of the program. It surveys all sixteen of Huang's published lens-discovery and follow-up papers, summarizes the SpectrumFM proposal, traces the current (May 2026) state of agentic / "AI scientist" research methods, and proposes seven concrete agentic-AI architectures (LensAgent, AutoFoundry, DimpleScout, SpectrumFM Autopilot, GIGA-LensAgent, FoundryScribe, LensLit) that could be prototyped against the existing pipeline and the SpectrumFM training program. All agentic-landscape citations have been verified via WebFetch against authoritative sources.

---

## Reader's Guide

This document is built in three layers.

1. **Tutorial background** — §1–§4 give a CS-reader-friendly introduction to strong gravitational lensing, the lens equation and its math, and the machine-learning techniques used in the program (CNNs, ResNets, EfficientNet, U-Net, differentiable forward modeling, transformer foundation models). If you are already fluent in lensing physics, skip to §5.
2. **Program survey** — §5–§7 walk through Huang's sixteen papers (thematically, with one-page per-paper deep dives in Appendix A), the SpectrumFM DOE Genesis proposal, and the datasets/instruments/computing infrastructure the work depends on.
3. **Agentic synthesis** — §8 maps the current agentic-research landscape; §9 proposes seven concrete agentic architectures targeted at Huang's program; §10 prioritizes which to build first, and §11–§12 close with a glossary and references.

If you want to read this in 30 minutes: read §1, §2, the executive paragraphs of §6 and §8, and all of §9–§10. If you want the deep version, read end-to-end.

---

## Contents

  - [Abstract](#abstract)
  - [Reader's Guide](#readers-guide)
- [1. Executive Summary](#1-executive-summary)
  - [1.1 The program in one paragraph](#11-the-program-in-one-paragraph)
  - [1.2 Three pillars](#12-three-pillars)
  - [1.3 SpectrumFM and agentic AI](#13-spectrumfm-and-agentic-ai)
  - [1.4 Seven concrete proposals (forward reference)](#14-seven-concrete-proposals-forward-reference)
- [2. The Research Program at a Glance](#2-the-research-program-at-a-glance)
  - [2.1 The five discovery modalities](#21-the-five-discovery-modalities)
  - [2.2 DESI Strong Lens Foundry: the confirmation pipeline](#22-desi-strong-lens-foundry-the-confirmation-pipeline)
  - [2.3 Timeline of cumulative candidates](#23-timeline-of-cumulative-candidates)
  - [2.4 Key instruments and facilities](#24-key-instruments-and-facilities)
  - [2.5 Software stack](#25-software-stack)
- [3. Physics Background — Gravitational Lensing for Computer Scientists](#3-physics-background--gravitational-lensing-for-computer-scientists)
  - [3.1 What gravitational lensing is](#31-what-gravitational-lensing-is)
  - [3.2 The lens equation](#32-the-lens-equation)
  - [3.3 The Einstein radius](#33-the-einstein-radius)
  - [3.4 Image configurations](#34-image-configurations)
  - [3.5 Magnification, parity, and time delay](#35-magnification-parity-and-time-delay)
  - [3.6 Why strong lenses are scientifically valuable](#36-why-strong-lenses-are-scientifically-valuable)
  - [3.7 Useful field-specific terms (light glossary, full one in §11)](#37-useful-field-specific-terms-light-glossary-full-one-in-11)
- [4. Mathematical &amp; Machine-Learning Foundations](#4-mathematical--machine-learning-foundations)
  - [4.1 Lens mass models](#41-lens-mass-models)
  - [4.2 Source light models](#42-source-light-models)
  - [4.3 Bayesian inference for lens parameters](#43-bayesian-inference-for-lens-parameters)
  - [4.4 Why differentiability changes the math](#44-why-differentiability-changes-the-math)
  - [4.5 CNN/ResNet primer for lens classification](#45-cnnresnet-primer-for-lens-classification)
  - [4.6 U-Net for substructure localization](#46-u-net-for-substructure-localization)
  - [4.7 Transformer foundation models for spectra (SpectrumFM context)](#47-transformer-foundation-models-for-spectra-spectrumfm-context)
- [5. Paper-by-Paper Walkthrough (themed)](#5-paper-by-paper-walkthrough-themed)
  - [5.1 Discovery lineage (Huang 2020 → Inchausti 2025)](#51-discovery-lineage-huang-2020--inchausti-2025)
  - [5.2 Modeling tooling (Gu 2022, Silver 2025)](#52-modeling-tooling-gu-2022-silver-2025)
  - [5.3 Specialty discovery modes (Sheu, Dawes, Hsu)](#53-specialty-discovery-modes-sheu-dawes-hsu)
  - [5.4 Single-system science (Cikota 2023, Sheu 2024b)](#54-single-system-science-cikota-2023-sheu-2024b)
  - [5.5 The DESI Strong Lens Foundry series (Huang 2025a/b, Agarwal 2025, Lin 2025)](#55-the-desi-strong-lens-foundry-series-huang-2025ab-agarwal-2025-lin-2025)
- [6. The SpectrumFM Proposal Deep Dive](#6-the-spectrumfm-proposal-deep-dive)
  - [6.1 Scientific motivation](#61-scientific-motivation)
  - [6.2 The "one model, six classes" claim](#62-the-one-model-six-classes-claim)
  - [6.3 Architecture](#63-architecture)
  - [6.4 Training data](#64-training-data)
  - [6.5 Human Alignment Training (the RLHF analogue)](#65-human-alignment-training-the-rlhf-analogue)
  - [6.6 Downstream validation](#66-downstream-validation)
  - [6.7 Phase-I 9-month timeline](#67-phase-i-9-month-timeline)
  - [6.8 Phase-I go/no-go metrics](#68-phase-i-gono-go-metrics)
  - [6.9 Compute, infrastructure, and budget](#69-compute-infrastructure-and-budget)
  - [6.10 Team and roles](#610-team-and-roles)
  - [6.11 Connection to gravitational lensing](#611-connection-to-gravitational-lensing)
  - [6.12 Phase-II vision](#612-phase-ii-vision)
- [7. Datasets, Instruments, and Computing Infrastructure](#7-datasets-instruments-and-computing-infrastructure)
  - [7.1 Imaging — DESI Legacy Imaging Surveys](#71-imaging--desi-legacy-imaging-surveys)
  - [7.2 Spectroscopy — DESI](#72-spectroscopy--desi)
  - [7.3 Follow-up — HST, Keck, VLT](#73-follow-up--hst-keck-vlt)
  - [7.4 Computing — NERSC Perlmutter (and the road to Doudna)](#74-computing--nersc-perlmutter-and-the-road-to-doudna)
  - [7.5 Software stack summary](#75-software-stack-summary)
- [8. Agentic-Research Landscape (May 2026)](#8-agentic-research-landscape-may-2026)
  - [8.1 Code/experiment loops](#81-codeexperiment-loops)
  - [8.2 Literature / scientific QA](#82-literature--scientific-qa)
  - [8.3 Hypothesis generation &amp; evolution](#83-hypothesis-generation--evolution)
  - [8.4 General research orchestration](#84-general-research-orchestration)
  - [8.5 Astronomy-specific agentic systems](#85-astronomy-specific-agentic-systems)
  - [8.6 Benchmarks](#86-benchmarks)
  - [8.7 Cross-cutting patterns (synthesis)](#87-cross-cutting-patterns-synthesis)
- [9. Proposed Agentic-AI Applications](#9-proposed-agentic-ai-applications)
  - [9.1 LensAgent — Multi-agent candidate-triage orchestrator](#91-lensagent--multi-agent-candidate-triage-orchestrator)
  - [9.2 AutoFoundry — autoresearch-style training-loop driver](#92-autofoundry--autoresearch-style-training-loop-driver)
  - [9.3 DimpleScout — hypothesis-driven spectroscopic search agent](#93-dimplescout--hypothesis-driven-spectroscopic-search-agent)
  - [9.4 SpectrumFM Autopilot — Phase-I training-pipeline orchestrator](#94-spectrumfm-autopilot--phase-i-training-pipeline-orchestrator)
  - [9.5 GIGA-LensAgent — Bayesian lens-modeling at scale](#95-giga-lensagent--bayesian-lens-modeling-at-scale)
  - [9.6 FoundryScribe — manuscript-drafting agent for confirmation papers](#96-foundryscribe--manuscript-drafting-agent-for-confirmation-papers)
  - [9.7 LensLit — literature-surveillance agent](#97-lenslit--literature-surveillance-agent)
- [10. Recommendations and Next Steps](#10-recommendations-and-next-steps)
  - [10.1 First-90-day priorities](#101-first-90-day-priorities)
  - [10.2 Tooling stack](#102-tooling-stack)
  - [10.3 Evaluation](#103-evaluation)
  - [10.4 Open-source the agentic pipelines](#104-open-source-the-agentic-pipelines)
  - [10.5 Risk register](#105-risk-register)
  - [10.6 Strategic horizon](#106-strategic-horizon)
- [11. Glossary](#11-glossary)
- [12. References](#12-references)
  - [12.1 Huang-group papers (the corpus)](#121-huang-group-papers-the-corpus)
  - [12.2 Lensing background and reviews](#122-lensing-background-and-reviews)
  - [12.3 Foundation models and ML methods](#123-foundation-models-and-ml-methods)
  - [12.4 Agentic-research projects and benchmarks (all WebFetch-verified May 25, 2026)](#124-agentic-research-projects-and-benchmarks-all-webfetch-verified-may-25-2026)
- [Appendix A — Per-paper deep dives](#appendix-a--per-paper-deep-dives)
  - [A.1 Huang et al. 2020 — DECaLS lens search (arXiv:1906.00970)](#a1-huang-et-al-2020--decals-lens-search-arxiv190600970)
  - [A.2 Huang et al. 2021 — DESI Legacy Surveys (arXiv:2005.04730)](#a2-huang-et-al-2021--desi-legacy-surveys-arxiv200504730)
  - [A.3 Gu, Huang et al. 2022 — GIGA-Lens (arXiv:2202.07663)](#a3-gu-huang-et-al-2022--giga-lens-arxiv220207663)
  - [A.4 Dawes, Storfer, Huang et al. 2022 — autocorrelation lensed quasars (ApJS 269, 16)](#a4-dawes-storfer-huang-et-al-2022--autocorrelation-lensed-quasars-apjs-269-16)
  - [A.5 Sheu, Huang et al. 2023 — lensed supernovae (arXiv:2301.03578)](#a5-sheu-huang-et-al-2023--lensed-supernovae-arxiv230103578)
  - [A.6 Cikota, Toro Bertolla, Huang et al. 2023 — DESI-253 Einstein cross (arXiv:2307.12470)](#a6-cikota-toro-bertolla-huang-et-al-2023--desi-253-einstein-cross-arxiv230712470)
  - [A.7 Sheu, Huang et al. 2024a — variable lensed quasars (arXiv:2408.02670)](#a7-sheu-huang-et-al-2024a--variable-lensed-quasars-arxiv240802670)
  - [A.8 Sheu, Cikota, Huang et al. 2024b — Carousel cluster lens (arXiv:2408.10332)](#a8-sheu-cikota-huang-et-al-2024b--carousel-cluster-lens-arxiv240810332)
  - [A.9 Storfer, Huang et al. 2024 — DR9 lens search (arXiv:2309.18089)](#a9-storfer-huang-et-al-2024--dr9-lens-search-arxiv230918089)
  - [A.10 Inchausti, Storfer, Huang et al. 2025 — DR10 dual architectures (arXiv:2508.20089)](#a10-inchausti-storfer-huang-et-al-2025--dr10-dual-architectures-arxiv250820089)
  - [A.11 Hsu, Huang et al. 2025 — pairwise spectroscopic search (arXiv:2509.16033)](#a11-hsu-huang-et-al-2025--pairwise-spectroscopic-search-arxiv250916033)
  - [A.12 Silver, Wang, Huang et al. 2025 — ML-driven JWST forecasts (arXiv:2207.09431)](#a12-silver-wang-huang-et-al-2025--ml-driven-jwst-forecasts-arxiv220709431)
  - [A.13 Huang et al. 2025a — DESI Strong Lens Foundry I (arXiv:2502.03455)](#a13-huang-et-al-2025a--desi-strong-lens-foundry-i-arxiv250203455)
  - [A.14 Huang et al. 2025b — DESI Strong Lens Foundry II (arXiv:2509.18089)](#a14-huang-et-al-2025b--desi-strong-lens-foundry-ii-arxiv250918089)
  - [A.15 Agarwal, Huang et al. 2025 — DESI Strong Lens Foundry III (arXiv:2501.08066)](#a15-agarwal-huang-et-al-2025--desi-strong-lens-foundry-iii-arxiv250108066)
  - [A.16 Lin, Toro Bertolla, Cikota, Huang et al. 2025 — DESI Strong Lens Foundry IV (arXiv:2509.18078)](#a16-lin-toro-bertolla-cikota-huang-et-al-2025--desi-strong-lens-foundry-iv-arxiv250918078)
- [Appendix B — Lens-equation derivation and GIGA-Lens math](#appendix-b--lens-equation-derivation-and-giga-lens-math)
  - [B.1 Deflection angle from a point mass](#b1-deflection-angle-from-a-point-mass)
  - [B.2 The scaled lens equation](#b2-the-scaled-lens-equation)
  - [B.3 Lens potential](#b3-lens-potential)
  - [B.4 Magnification](#b4-magnification)
  - [B.5 Time delay (Fermat potential)](#b5-time-delay-fermat-potential)
  - [B.6 GIGA-Lens EPL surface mass density](#b6-giga-lens-epl-surface-mass-density)
  - [B.7 Three-stage inference flow](#b7-three-stage-inference-flow)
- [Appendix C — Verified agentic-research citation table](#appendix-c--verified-agentic-research-citation-table)

---


# 1. Executive Summary

## 1.1 The program in one paragraph

Prof. Xiaosheng Huang's research group at the University of San Francisco builds machine-learning pipelines to discover strong gravitational lenses in wide-area imaging surveys (primarily the DESI Legacy Imaging Surveys, ~19,000 deg² in *griz*), then characterizes the discoveries with high-resolution imaging (HST), optical and near-infrared spectroscopy (DESI, Keck NIRES, VLT/MUSE), and differentiable Bayesian lens modeling (GIGA-Lens). The cumulative output across DR8/9/10 is roughly **3,500–5,500 lens candidates** (depending on how lensed quasars and dimple lenses are counted), an order of magnitude expansion of the field. The DOE Genesis Mission SpectrumFM proposal then proposes to build a transformer foundation model on ~60M DESI spectra, with strong-lens identification as one of its downstream-validation tasks. Greg Benson joins as the agentic-AI Co-I — the architect responsible for bringing agentic tooling and multi-agent systems into both the existing lens-discovery program and the SpectrumFM training loop.

## 1.2 Three pillars

The program organizes naturally into three pillars that the rest of this report follows:

- **Discover.** Find lens candidates in imaging and/or spectroscopic data. The group has built five complementary discovery modalities — ResNet on imaging cutouts (Huang 2020, 2021), a ResNet+EfficientNet ensemble with meta-learner (Storfer 2024; Inchausti 2025), difference-imaging for variable lensed transients (Sheu 2023 SNe; Sheu 2024a quasars), an autocorrelation method for binary/lensed quasars (Dawes 2022), and a **pair-wise spectroscopic search** over DESI fiber spectra that has now also revealed a brand-new class of low-mass-halo systems called "dimple lenses" (Hsu 2025).
- **Characterize.** Confirm the candidates, measure their lens/source redshifts, and extract physical parameters (Einstein radius, mass-slope, velocity dispersion). This is the *DESI Strong Lens Foundry*: Paper I (HST imaging + GIGA-Lens modeling), Paper II (DESI spectroscopy), Paper III (Keck NIRES near-IR for high-z sources), Paper IV (VLT/MUSE integral-field spectroscopy). GIGA-Lens (Gu 2022) — a GPU-accelerated, differentiable Bayesian lens-modeling framework in TensorFlow + JAX — is the workhorse.
- **Cosmologize.** Use the characterized sample to constrain H₀ via time-delay cosmography, test CDM at low halo mass via dark-matter substructure, and probe galaxy-scale mass profiles. This is the long-run scientific payoff; the program is positioned for the LSST/Rubin / Euclid / Roman era when O(10⁵) lenses are expected.

## 1.3 SpectrumFM and agentic AI

**SpectrumFM** extends this stack one layer deeper. Rather than train one CNN per discovery modality, the proposal builds a transformer foundation model on the full ~60M DESI spectral corpus, with a key innovation — an *auxiliary redshift head* trained jointly with masked-spectrum modeling on every pretraining step, plus *human alignment* (RLHF-style preference learning over DESI Visual-Inspection campaigns). The same encoder is intended to serve LRG/ELG/QSO/MWS at production quality plus few-shot LBG/LAE extensibility for DESI-II, with downstream validation on strong-lens identification (via Hsu 2025's pairwise method) and supernova typing. The proposal explicitly names Benson's role: *"Benson (USF; $33,309) leads agentic-AI tooling"*.

**Agentic AI** is therefore not an afterthought in the program — it is a named work package. The rest of this report grounds that work package: it surveys the current state of agentic / "AI scientist" research methods (Karpathy autoresearch, Sakana AI Scientist v2, Google AI Co-Scientist, Anthropic multi-agent research system, FutureHouse/PaperQA2/Aviary, Denario/CMBAgent, RE-Bench/MLE-bench) and proposes seven concrete agentic architectures targeted at specific bottlenecks in Huang's pipeline.

## 1.4 Seven concrete proposals (forward reference)

§9 elaborates each of the following. The naming is provisional and serves to make them distinct enough to discuss.

1. **LensAgent** — a multi-agent candidate-triage orchestrator that augments human visual-grading of ResNet/EfficientNet outputs.
2. **AutoFoundry** — a Karpathy-style autoresearch loop driving iteration on the dual-architecture lens-finder code.
3. **DimpleScout** — a Co-Scientist-style hypothesis-evolution loop for the new dimple-lens spectroscopic search.
4. **SpectrumFM Autopilot** — an orchestrator for the SpectrumFM Phase-I 9-month training plan, owned by Benson.
5. **GIGA-LensAgent** — an agent wrapper around GIGA-Lens to scale Bayesian lens modeling from O(10²) to O(10⁴) systems.
6. **FoundryScribe** — a Sakana-v2-style writing agent for confirmation-paper sections in the Foundry series.
7. **LensLit** — a PaperQA2-style literature-surveillance agent over NASA ADS strong-lensing literature.

§10 recommends starting with **LensLit** (fastest payoff), **DimpleScout** (highest scientific upside), and **SpectrumFM Autopilot** (strategic anchor since Benson already owns that piece).

---

# 2. The Research Program at a Glance

## 2.1 The five discovery modalities

| Modality | Lead paper(s) | Input | Method | Output |
|---|---|---|---|---|
| ResNet on imaging cutouts | Huang 2020; Huang 2021 | *grz* (and later *griz*) 101×101 px cutouts | Residual CNN trained on observed lenses + curated negatives | 335 (2020) + 1,210 (2021) candidates |
| ResNet + EfficientNet ensemble + meta-learner | Storfer 2024; Inchausti 2025 | *griz* DR9/DR10 cutouts | Dual-architecture ensemble; 300-node neural meta-learner over the two probabilities | 1,512 (DR9) + 811 (DR10) candidates |
| Difference-imaging for variable lensed transients | Sheu 2023 (SNe); Sheu 2024a (quasars) | Multi-epoch *griZy* imaging | Two image-subtraction algorithms (B08, SFFT), SEP detection, light-curve fits with SALT3 | 7 lensed-SN candidates; 13 variable lensed quasar candidates (3 quads) |
| Autocorrelation over quasar catalogs | Dawes 2022 | DESI quasar sample (~5M objects) | Color cuts, PSF morphology, Gaia proper-motion/parallax filters; autocorrelation by position | 436 multiply-lensed/binary-quasar candidates |
| Pair-wise spectroscopic search | Hsu 2025 | DESI DR1 fiber spectra (28M) + DR10 imaging | Friends-of-Friends grouping on sky position (3″ link); redshift-ratio cut; visual inspection | 2,164 new candidates (1,906 conventional + 318 *dimple lenses*) |

The combined cumulative catalog through May 2026 is on the order of **5,500 candidates** across galaxy-scale lenses, lensed quasars, lensed transients, cluster-scale arcs, and the new dimple-lens class — and roughly an order of magnitude more candidates than were known to the global lensing community a decade earlier. The pipeline is now feeding into the LSST/Rubin era, where O(10⁵) lenses are projected.

## 2.2 DESI Strong Lens Foundry: the confirmation pipeline

Once candidates exist, the Foundry sub-program turns them into characterized lens systems with measured redshifts, lens models, and (eventually) cosmological constraints. The current paper sequence is:

- **Foundry I** (Huang 2025a). HST WFC3/F140W SNAPshot imaging of 51 of the most promising candidates via program GO-15867 (PI: Huang). All 51 were confirmed as strong lenses. Includes the first full multi-GPU forward-modeling Bayesian fit (GIGA-Lens) ever applied to HST data, demonstrated on DESI-165.4754−06.0423.
- **Foundry II** (Huang 2025b). DESI Strong Lensing Secondary Target Program — DESI multi-fiber spectroscopy of candidates from H20/H21/S24. Early Data Release: 73 candidates, 20 confirmed strong lenses, 4 confirmed not lenses, remainder pending source spectra. Bridges to Foundry III for sources beyond the DESI optical window.
- **Foundry III** (Agarwal 2025). Keck-2 NIRES (0.94–2.45 μm, R~2700) near-infrared spectroscopy for source galaxies at z > ~1.5 whose [O II] λ3727 doublet has redshifted out of the DESI optical range. Eight confirmed systems, source redshifts up to z_s = 3.33.
- **Foundry IV** (Lin 2025). VLT/MUSE 60″×60″ integral-field spectroscopy (R = 2000–5500) of 75 candidates, yielding 48 fully confirmed lens systems (both lens and source redshift) and 21 with source-only redshifts. Adds spatial spectroscopy for the cluster-scale and multi-source configurations.

The four Foundry papers are mutually self-citing and form the integrated confirmation pipeline that turns raw ML candidates into Bayesian-modelable systems with the redshifts needed for H₀ time-delay analyses and mass-sheet-degeneracy-breaking.

## 2.3 Timeline of cumulative candidates

```
Year   Cumulative new candidates (galaxy + cluster-scale)
2020    335     Huang_2020 (DECaLS)
2021  1,545     +Huang_2021 (DR7 expanded)
2024  3,057     +Storfer_2024 (DR9)
2025  3,868     +Inchausti_2025 (DR10 dual-architecture)
2025  5,068     +Dawes_2022 lensed quasar autocorrelation (436) +Sheu_2024 variable QSO (13) + lensed SNe (7) + cluster-scale work
2025  5,500+    +Hsu_2025 pairwise spectroscopic (2,164 new)
```

A cleaner statement: by mid-2025 Huang's group had discovered approximately **5,000–5,500 distinct new strong-lensing candidates** across all modalities — comparable in scale to the total number of confirmed strong lenses known to the field at the start of the decade.

## 2.4 Key instruments and facilities

- **DESI Legacy Imaging Surveys** — DECaLS (Blanco 4m), BASS (Bok 2.3m), MzLS (Mayall 4m). DR8: 14,000 deg²; DR9: ~19,000 deg²; DR10: extends with reprocessed DECam data south of δ = −18°. *grz* (and *iY* in DR10), pixel scale 0.262″, z-band median depth ~22.5–23.5 AB mag.
- **DESI spectroscopy** — Mayall 4m / Kitt Peak, 5,000-fiber multi-object spectrograph, 3,600–9,800 Å in three arms, R ~ 2,000–5,500. DR1 hosts ~28M spectra.
- **Hubble Space Telescope** — Program GO-15867 SNAPshots in WFC3/F140W (near-IR), 51 successful targets.
- **Keck-2 / NIRES** — Cross-dispersed echellette, 0.94–2.45 μm, R ~ 2,700, slit 0.55″×18″.
- **VLT / MUSE** — Wide-Field-Mode, 60″×60″ FoV, 0.2″ spaxel, 4750–9350 Å, R ~ 2,000–5,200.
- **NERSC Perlmutter** — HPE Cray EX with NVIDIA A100 GPU partition. The training facility for GIGA-Lens, the dual-architecture finder, and (proposed) SpectrumFM. Doudna (NERSC-10) enters production late 2026 with ~10× Perlmutter performance.

## 2.5 Software stack

The group writes primarily in Python. Key dependencies and where they show up:

- **TensorFlow + TensorFlow Probability + JAX** — GIGA-Lens (Gu 2022) end-to-end; differentiable forward model, VI, HMC.
- **PyTorch** — used in some recent CNN training (Inchausti 2025).
- **lenstronomy** — reference simulator and the basis on which GIGA-Lens improves; used for ground-truth synthetic systems.
- **Astropy / Photutils / SEP / sep / spherematch** — catalog and photometry tooling.
- **The Tractor** — model-based PSF/galaxy photometry, the basis for DESI Legacy source catalogs.
- **PyPelt** — spectroscopic reduction for Keck/NIRES and VLT/MUSE follow-up.
- **FastSpecFit** — derived spectro-photometric quantities (velocity dispersions, emission-line fluxes) on DESI spectra.
- **Redrock** — DESI's template-fitting redshift pipeline (the baseline SpectrumFM is built to replace).

---

# 3. Physics Background — Gravitational Lensing for Computer Scientists

## 3.1 What gravitational lensing is

In general relativity, mass curves spacetime, and light follows geodesics in that curved spacetime. When a massive object (the *lens*, almost always a galaxy or cluster) sits along the line of sight between us and a more distant source (the *source*, often a background galaxy or quasar), light from the source is deflected as it passes the lens. From our point of view, the source appears at a different position on the sky — and if the geometry is right, may even appear in multiple positions simultaneously.

The qualitative picture you should hold in your head: imagine a sheet of rubber that gets stretched and dimpled wherever there is mass. Light rays travel along straight lines on the rubber sheet, but those straight lines look curved when projected onto a flat sky. The lensing regime is determined by how strong the deflection is relative to the angular size of the source:

- **Strong lensing.** Multiple images, arcs, Einstein rings. The lens is massive enough to produce qualitatively new images on the sky. This is what Huang's group works on. Typical deflection angles are arcseconds.
- **Weak lensing.** Small statistical distortions to the shapes of background galaxies, only detectable in large-sample statistics. This is the regime cosmic-shear surveys (KiDS, DES, Euclid, LSST) operate in.
- **Microlensing.** A foreground star deflects light from a distant point source (another star or quasar accretion disk), producing time-variable magnification. Microlensing dominates inside individual lensed quasar images.

For comprehensive treatments at three levels of depth: P. Schneider, J. Ehlers, E. E. Falco, *Gravitational Lenses* (Springer 1992) — the textbook; Bartelmann & Schneider, *Phys. Rep.* 340, 291 (2001) — the canonical review; Treu, *Annu. Rev. Astron. Astrophys.* 48, 87 (2010) — accessible introduction to strong-lensing science.

## 3.2 The lens equation

Under the *thin-lens approximation* (the lens is much smaller in line-of-sight depth than the distances involved), the geometry collapses to a 2D problem on the sky. Let:

- **β** — the true angular position of the source (where it would be if there were no lens), a 2-vector on the sky.
- **θ** — the observed angular position of an image (a 2-vector).
- **α(θ)** — the deflection angle as a function of the image position. This is the scaled deflection (folding in the distance ratios).

Then the *lens equation* is:

> **β = θ − α(θ)**

This is the fundamental equation of lensing. Given a lens mass distribution (which determines α(θ)) and a source position β, you solve for the image positions θ. Note this is nonlinear in θ — the deflection α depends on where the image is — so for a given β there can be multiple θ that satisfy the equation. Those multiple solutions are exactly the multiple images you see in a strong lens.

## 3.3 The Einstein radius

For a point-mass lens M_L at distance D_L from us, with the source at distance D_S and a lens-source distance D_LS, the angular Einstein radius is:

> **θ_E = √( 4GM_L / c² × D_LS / (D_L D_S) )**

For an isothermal-sphere lens (a much better approximation to a galaxy), with velocity dispersion σ_v:

> **θ_E = 4π (σ_v / c)² × D_LS / D_S**

This is precisely equation (1) in Hsu et al. 2025 — they use this formula to assign Einstein radii to dimple-lens candidates from FastSpecFit velocity dispersions.

The Einstein radius sets the angular scale of the strong-lensing phenomenon. A typical massive elliptical galaxy at z ~ 0.5 has σ_v ~ 250 km/s and θ_E ~ 1–2″. Cluster-scale lenses can have θ_E up to ~10–30″.

## 3.4 Image configurations

The number and shape of lensed images depends on the lens mass profile and the source position:

- **Two images** ("doubles"). The source lies just outside the lens's tangential caustic. Most common for galaxy-galaxy lensing.
- **Four images** ("quads", including Einstein crosses). The source lies inside the inner tangential caustic. Rarer but information-rich because there are more observables to constrain the lens model. Cikota 2023's DESI-253.2534+26.8843 is an Einstein cross.
- **Einstein rings**. The source is almost exactly behind the lens. The four images merge into a single ring.
- **Cluster-scale arcs and tangential arcs**. The source lies near a cluster's tangential critical curve. Magnifications can reach 50–100×. Sheu 2024b's "Carousel" cluster lens is in this regime, with seven distinct lensed source galaxies confirmed by VLT/MUSE.

## 3.5 Magnification, parity, and time delay

Each lensed image is magnified (or de-magnified) by a factor μ — the Jacobian determinant of the lens equation evaluated at θ. Some images have positive parity (same handedness as the source) and others negative.

Light traversing different image paths arrives at the observer at different times. The relative *time delay* between images depends on:

1. The geometric path-length difference between the two ray paths.
2. The gravitational (Shapiro) delay from passing through the lens potential.

For a variable source (a supernova, a quasar with intrinsic variability), this time delay is measurable directly: see one image flicker, wait days to years, watch the same flicker in another image. Refsdal (1964) first proposed using this as a route to the Hubble constant H₀: combining the measured time delay with a lens model gives the distance scale of the universe directly. This is the modern *time-delay cosmography* program (H0LiCOW, TDCOSMO, STRIDES), and it is the long-run cosmological payoff of having a large, well-modeled lens sample.

## 3.6 Why strong lenses are scientifically valuable

Three high-value applications motivate the whole program:

1. **H₀ via time-delay cosmography.** Multiply-imaged variable sources (quasars, supernovae) yield direct H₀ measurements that are *independent* of the local distance ladder (Cepheids, SNe Ia) and the CMB-inferred value (Planck). With ~50–100 well-modeled systems, the program can produce a 1% H₀ measurement — directly relevant to the present "Hubble tension" between the local and CMB-inferred values.
2. **Dark-matter substructure detection.** CDM predicts an abundance of dark subhalos in the M_halo ~ 10⁸–10⁹ M_⊙ range. These are invisible directly, but their gravity perturbs lens images at the few-milliarcsecond level. With high-resolution imaging (HST, JWST) and forward-modeling, one can detect or rule out their presence — testing CDM at the smallest probed scales. Silver 2025 explicitly extends this to θ_E ~ 0.03″ and M_halo ~ 10¹¹ M_⊙; Hsu 2025's dimple lenses are at M_halo ≲ 10¹³ M_⊙.
3. **Magnified high-redshift galaxies.** A lens magnification of 10–100× turns a faint, distant galaxy into something observable in detail with current instruments. The "Carousel" lens (Sheu 2024b) magnifies seven distinct sources, including a predicted source at z ~ 4.5.

## 3.7 Useful field-specific terms (light glossary, full one in §11)

- **Convergence κ(θ)** — the surface mass density of the lens normalized to a critical density. κ > 1 indicates a region capable of producing multiple images.
- **Shear γ(θ)** — the tidal field of the lens; produces image distortion.
- **Critical curve** — the locus on the image plane where the magnification is formally infinite.
- **Caustic** — the corresponding locus on the source plane. A source crossing a caustic gains or loses image pairs.
- **Mass-sheet degeneracy** — a fundamental degeneracy in lens modeling: adding a uniform sheet of mass plus rescaling the source preserves all image positions and brightness ratios but changes the inferred H₀. Velocity-dispersion measurements (DESI fiber spectra, Keck/MUSE follow-up) break this degeneracy — which is why Foundry II/III/IV exist.

---

# 4. Mathematical & Machine-Learning Foundations

## 4.1 Lens mass models

A handful of parameterized mass profiles cover most of the field:

- **SIS — Singular Isothermal Sphere.** Spherical, isothermal. Two parameters (lens center, velocity dispersion). Cheap and analytically tractable; the right zeroth approximation.
- **SIE — Singular Isothermal Ellipsoid.** Adds ellipticity to SIS. The default galaxy-scale lens model in many papers, including Cikota 2023's GIGA-Lens fit of the Einstein cross.
- **EPL — Elliptical Power-Law.** Used in GIGA-Lens (Gu 2022). Surface mass density κ(x,y) = ½ (3 − γ_epl) × ( θ_E / √(q x² + y²/q) )^(γ_epl − 1) with mass-slope γ_epl (γ_epl = 2 recovers SIE) and axial ratio q. Equivalent to the PEMD (power-law elliptical mass density) profile.
- **NFW (Navarro-Frenk-White)** — cosmological halo profile for cluster-scale lensing.
- **External shear** — a constant added shear from foreground/background mass not in the lens galaxy. Two parameters (γ_ext,1; γ_ext,2). Always included in serious modeling.
- **Multipoles** — higher-order angular expansions of the potential, used when the simple parameterizations don't fit (rare but important).

The choice of mass model is a real source of systematic uncertainty (the *modeling degeneracy*). Modern lens modeling pipelines (lenstronomy, GIGA-Lens) make it easy to compare models via Bayesian evidence.

## 4.2 Source light models

You need to model the light, not just the mass:

- **Sérsic profile.** Empirical I(r) = I₀ exp(−b_n × ((r/R_eff)^(1/n) − 1)) with index n (n=1 exponential, n=4 de Vaucouleurs, n~2–6 typical for ellipticals). Used for lens light and (often) source light.
- **Shapelets / Hermite basis.** Linear expansion in a Cartesian/polar basis; can fit irregular source morphology.
- **Pixelated source.** Reconstruct the source on a regularized pixel grid; most flexible and most prone to overfitting.

For the demonstration in Gu 2022, both lens and source use elliptical Sérsic profiles. The five lens-light/source-light parameters per component (center, ellipticity, R_eff, n, I₀) plus the EPL mass parameters and external shear give ~17–22 free parameters for a single-source galaxy-galaxy lens.

## 4.3 Bayesian inference for lens parameters

The forward model takes parameters Θ and produces a predicted image *I_model*(x, y; Θ). The likelihood compares to the observed image *I_data* (with per-pixel uncertainties σ²(x,y) including sky, read, and Poisson contributions):

> L(Θ) ∝ exp( −½ Σ_{pixels} ( (I_data − I_model)² / σ² ) )

Priors are chosen from physical reasonable ranges (positive R_eff, Einstein radius within plausible bounds, etc.). The posterior π(Θ | data) is then the target of inference.

For high-dimensional, non-convex posteriors, three sampling techniques are universal:

- **Importance sampling / particle swarm optimization (PSO).** Used by older lenstronomy pipelines. No convergence guarantees in high dimensions.
- **Variational inference (VI).** Approximate the posterior by a tractable family (Gaussian, mean-field, or normalizing flow) and minimize KL(q || π). Fast — gives you a covariance matrix.
- **Hamiltonian Monte Carlo (HMC) / NUTS.** Use Hamiltonian dynamics to make long-range, low-rejection moves through parameter space. Gold standard for high-D posteriors, but requires gradients of log π with respect to Θ.

GIGA-Lens uses all three in sequence: multi-start gradient descent to find global modes, VI to estimate the posterior covariance, HMC (initialized from the VI posterior) to draw exact samples. The combination is the headline contribution of Gu 2022.

## 4.4 Why differentiability changes the math

Older lens-modeling codes (lenstronomy with PSO + emcee) treat the forward model as a black box: you can evaluate it, but you can't differentiate through it. That forces you to use gradient-free MCMC like emcee, which scales poorly in high dimensions.

GIGA-Lens is implemented in **TensorFlow + JAX**, so the entire forward model — ray-tracing through the EPL deflection field, evaluating the Sérsic source profile, applying the PSF, computing the pixel-wise likelihood — is differentiable via automatic differentiation. That unlocks:

- Gradient-informed optimization (faster, more reliable global mode-finding).
- VI with the reparameterization trick.
- HMC, which fundamentally requires ∇ log π.

The wall-clock impact: GIGA-Lens fits a system in 105 s on four NVIDIA A100 GPUs, vs. ~4.3 hours for a comparable lenstronomy + emcee fit on the same systems (Rojas et al. 2021 baseline). For LSST-era O(10⁵) lenses, only the differentiable approach is computationally feasible.

This is the same pattern that has transformed simulation-based inference across scientific machine learning. Foundation reading: Cranmer, Brehmer, & Louppe, *PNAS* 117, 30055 (2020), "The frontier of simulation-based inference".

## 4.5 CNN/ResNet primer for lens classification

The image-based lens finders use the standard supervised-classification CNN template:

- Input: a small (101×101 px) multi-band cutout centered on a candidate.
- Backbone: a Residual Network (ResNet, He et al. 2016), originally adapted by Lanusse et al. 2018 for lensing.
- Output: scalar probability p(lens | cutout).
- Loss: binary cross-entropy.
- Training data: a few hundred to ~2,000 known lenses + tens of thousands of non-lenses (curated to balance depth bins).

Huang's group has iterated on three pieces:

1. **"Shielding" layers** — 1×1 convolutions inserted in the ResNet head to reduce parameter count by ~50× (Huang 2021). Improves training time without hurting AUC.
2. **EfficientNetV2** — Inchausti 2025 adds a pretrained EfficientNetV2 (20M params) alongside the ResNet (~200K params), then trains a 300-node neural meta-learner over the two output probabilities. AUC improves from ~0.998 → ~0.9989 — the marginal gain is small but materially reduces the false-positive count at the operating threshold.
3. **Depth-balanced training samples** — explicitly bin training negatives by z-band depth to prevent the classifier from learning depth as a proxy for lens-likeliness (Huang 2021, §3.2).

The grading scheme is human-mediated: candidates above a probability threshold are reviewed by team members and graded A/B/C/D. This human-in-the-loop step is the natural target for agentic AI assistance (§9.1).

## 4.6 U-Net for substructure localization

Silver 2025 uses a U-Net (Ronneberger et al. 2015), originally designed for medical-image segmentation, to *localize* small Einstein-radius perturbers in lensing images. Where the ResNet says "this cutout contains a lens," the U-Net says "here is the (x,y) location of a small subhalo near this arc." Trained on hydrodynamical-simulation-based synthetic lenses (VELA + Cosmodc2), it forecasts that JWST will be able to detect O(17/deg²) systems with θ_E ~ 0.03″ and M_halo ~ 10¹¹ M_⊙ — opening a CDM substructure test at unprecedented mass scales.

## 4.7 Transformer foundation models for spectra (SpectrumFM context)

The SpectrumFM proposal builds on the wave of *omnimodal foundation models for astronomy*. Key 2024–2025 systems for context:

- **AstroCLIP** — contrastive learning between spectra and images.
- **AION-1** — Parker et al. 2025 (Polymathic-AI Collaboration), arXiv:2510.17960. 39-modality omnimodal model; SpectrumFM is positioned as a successor specialized to spectra at full resolution.
- **AstroLLM** — retrieval-grounded astronomy LLM (https://astrollm.org/), launched 2026.

The architectural choice SpectrumFM bets on is *encoder–decoder transformer with an auxiliary redshift head trained jointly on every step*, rather than AION-1's approach of treating redshift as one of 39 modalities masked uniformly. The proposal argues this is the right inductive bias: redshift is the most important downstream quantity, and baking it into pretraining (rather than fine-tuning a frozen backbone) gives a stronger encoder.

The proposed scale: ~100M to ~1B parameters trained on ~60M DESI spectra (the corpus is 60× larger than AION-1's spectral component). Three-stage training: Masked Spectrum Modeling pretraining → Supervised Fine-Tuning + preference learning over Visual Inspection campaigns (RLHF-style) → few-shot fine-tuning for novel classes (LBGs, LAEs). Go-deeper reading on foundation-model-for-science: Bommasani et al., "On the Opportunities and Risks of Foundation Models" (2021); Subramanian et al., "AION-1" (2025); and the SpectrumFM proposal itself.

---

# 5. Paper-by-Paper Walkthrough (themed)

The full per-paper write-ups are in Appendix A. This section summarizes the themes; jump to A for the deep version on any single paper.

## 5.1 Discovery lineage (Huang 2020 → Inchausti 2025)

Four papers, one continuously evolving pipeline:

- **Huang 2020 (DECaLS)**, arXiv:1906.00970. *First systematic ResNet lens search of the 9,000 deg² DECaLS footprint.* 335 new candidates, ROC-AUC 0.98. Key methodological choice (and the one that distinguishes Huang's program from competing groups using simulated training data): train on *real* observed lenses + curated real non-lenses. The "we made simulated data work, but observed data works better and is cleaner" thesis.
- **Huang 2021 (DESI Legacy Surveys)**, arXiv:2005.04730. *Scaled to the full 14,000 deg² across DECaLS + BASS + MzLS, with "shielding" layers reducing the parameter count 50×.* 1,210 new candidates, AUC 0.992. Introduces the depth-balanced negative-sampling that became standard for the rest of the program.
- **Storfer 2024 (DR9)**, arXiv:2309.18089. *Continued scaling to DR9, refined visual-inspection grading and built the candidate database that feeds the Foundry confirmation pipeline.* 1,512 new candidates, AUC 0.9997.
- **Inchausti 2025 (DR10 two architectures)**, arXiv:2508.20089. *Introduces the ResNet + EfficientNetV2 ensemble + 300-node meta-learner.* Trains on NERSC Perlmutter with 4 GPU nodes. 811 new candidates from DR10. AUC 0.9989. Combined I–IV total: 3,868 candidates.

Methodological progression in one sentence per paper: real-data training (2020) → architectural compression (2021) → expanded survey + grading infrastructure (2024) → ensemble + meta-learner (2025). The fully unified ML pipeline is then ready for LSST/Rubin in the next phase.

## 5.2 Modeling tooling (Gu 2022, Silver 2025)

Two complementary modeling efforts:

- **Gu 2022 (GIGA-Lens)**, arXiv:2202.07663. *A gradient-informed, GPU-accelerated Bayesian framework for forward-modeling strong lenses in TensorFlow + JAX.* Three-stage inference: multi-start gradient descent → VI → HMC. 105 seconds per system on 4 A100 GPUs. Headlined as the modeling pipeline ready for O(10⁵) lenses in the LSST/Rubin/Euclid/Roman era. Used in Foundry I, Cikota 2023, Sheu 2024b.
- **Silver 2025 (ML-driven discoveries at θ_E ~ 0.03″)**, arXiv:2207.09431. *Combines a ResNet for lens classification with a U-Net for subhalo localization, trained on simulated JWST observations.* Forecasts JWST will discover O(17/deg²) lenses at M_halo ~ 10¹¹ M_⊙ — the dark-matter substructure regime. Demonstrates "superhuman" lens detection (discovering two real HST lenses missed by crowdsourced classification).

## 5.3 Specialty discovery modes (Sheu, Dawes, Hsu)

The five-modality picture from §2 is built largely from this cluster of papers:

- **Sheu 2023 (lensed SNe)**, arXiv:2301.03578. *Difference-imaging pipeline searching for variable lensed transients in DESI Legacy multi-epoch coadds.* 7 new lensed-SN candidates. Establishes the SFFT + B08 image-subtraction stack.
- **Sheu 2024a (variable lensed quasars)**, arXiv:2408.02670. *Same pipeline adapted to quasar variability rather than SNe.* 13 new candidates, 3 quads. Distinct from Dawes 2022 in being temporally-driven rather than morphologically-driven.
- **Dawes 2022 (autocorrelation for lensed/binary quasars)**, ApJS 269 16. *Color cuts + PSF morphology + Gaia proper-motion filters + spatial autocorrelation on the DESI quasar sample.* 436 multiply-lensed/binary-quasar candidates. The current 436 number is what propagates into Foundry II's stated total of "~3,500 candidates plus 436 lensed quasar candidates".
- **Hsu 2025 (pairwise spectroscopic search)**, arXiv:2509.16033. *A genuinely new lens-discovery modality: pair fiber spectra that are close on the sky and have inconsistent redshifts, then visually inspect.* From 28M DESI DR1 spectra they obtain 26,621 candidate spectra in 11,848 fiber pairs/triplets/quartets; visual inspection yields **2,046 conventional lens candidates** (1,906 new) and **318 dimple-lens candidates** — a new class of low-mass (M_halo ≲ 10¹³ M_⊙) foreground lenses producing surface-brightness indentations rather than arcs. The dimple class is potentially the most important new science target in the program because it probes CDM at the dwarf-galaxy scale.

## 5.4 Single-system science (Cikota 2023, Sheu 2024b)

Two showcase systems modeled in detail with GIGA-Lens:

- **Cikota 2023 (DESI-253.2534+26.8843 Einstein cross)**, arXiv:2307.12470. *A quadruply imaged blue source around a massive elliptical at z_L = 0.630, source z_S = 2.597, originally found in Huang 2021.* Spectroscopic confirmation with VLT/MUSE; lens model with GIGA-Lens. Einstein radius θ_E = 2.52″.
- **Sheu 2024b ("Carousel" cluster lens DESI-090.9854−35.9683)**, arXiv:2408.10332. *A cluster-scale lens at z_L = 0.49 with **seven** spectroscopically confirmed lensed sources, plus a model-predicted z ~ 4.5 high-z source.* Modeled with GIGA-Lens. Mass interior to the Einstein radius: 4.78 × 10¹³ M_⊙.

These are the prototypes for the kind of detailed science papers that the larger Foundry sample will support at scale. Each is also a demonstration that GIGA-Lens handles real (rather than simulated) data, with HST imaging and multi-source spectroscopy.

## 5.5 The DESI Strong Lens Foundry series (Huang 2025a/b, Agarwal 2025, Lin 2025)

The integrated confirmation pipeline. The four-paper sequence is structured to publish the four follow-up modalities in parallel:

- **Foundry I — HST imaging + GIGA-Lens** (Huang 2025a, arXiv:2502.03455). HST GO-15867 SNAP, 51 of 51 targets confirmed strong lenses; GIGA-Lens applied to DESI-165.4754−06.0423 as a demonstration. First multi-GPU forward-modeling Bayesian fit to HST data.
- **Foundry II — DESI Strong Lensing Secondary Target Program** (Huang 2025b, arXiv:2509.18089). DESI optical spectroscopy of candidate systems. EDR: 73 systems, 20 confirmed lenses, 4 ruled out. Notes that ~30% of candidates have source redshifts beyond the DESI optical range and so need Foundry III (Keck NIRES).
- **Foundry III — Keck NIRES** (Agarwal 2025, arXiv:2501.08066). Near-IR follow-up of 8 systems whose source redshifts (z_s = 1.68–3.33) are beyond DESI's optical window. The bridge from optical-only to high-z source characterization.
- **Foundry IV — VLT/MUSE** (Lin 2025, arXiv:2509.18078). Integral-field optical spectroscopy of 75 candidate systems. 48 fully confirmed (lens + source redshifts), 21 with source-only redshifts. Spatial spectroscopy is the right tool for complex (multi-source, cluster-scale) systems.

The four together turn ML candidates into Bayesian-modelable systems ready for cosmological analysis.

---

# 6. The SpectrumFM Proposal Deep Dive

This section relies on the project narrative at `proposals/doe_genesis_spectrumfm_project_narrative_v7.docx` (Topic 14, Focus Area A, DE-FOA-0003612).

## 6.1 Scientific motivation

DESI's production redshift pipeline, **Redrock**, fits hand-crafted templates per object class. This works adequately for the four label-rich classes (LRG, ELG, QSO, MWS). It does not extend to new target classes without major engineering work: even extending Redrock to QSOs required two bolted-on afterburners (QuasarNET — a supervised CNN — and a broad-MgII module; see Alexander et al. 2023). DESI-II will add Lyman-Break Galaxies (LBGs) and Lyman-Alpha Emitters (LAEs), and Spec-S5 will add more. Each new class requires more template engineering.

The proposal argues this is a *structural* bottleneck, not a fix-up issue: a foundation model that learns a transferable spectral representation can serve all six classes (and future ones) with the same encoder, by fine-tuning a small head per class. This is the "one model, six classes" thesis.

## 6.2 The "one model, six classes" claim

| Regime | Classes | Phase-I Approach |
|---|---|---|
| Label-rich | LRG, ELG, QSO (extragalactic); MWS (stars) | Pretraining with auxiliary redshift head; SFT + preference learning on VI data |
| Extensibility (the AI advantage) | LBG, LAE (extragalactic) | Few-shot fine-tuning with ≤5,000 examples per class |

Note these are explicit Phase-I deliverables, with go/no-go decision metrics.

## 6.3 Architecture

- **Transformer encoder-decoder** operating at substantially higher spectral resolution than AION-1. Initial compression ~10:1 with a path to full 7,081-pixel resolution. (AION-1 by contrast uses 273 tokens.)
- **Auxiliary redshift head** — a supervised redshift regression head on the encoder's pooled output, trained jointly with the masked-spectrum-modeling loss on *every pretraining step*. This is the architectural commitment that distinguishes SpectrumFM from AION-1: redshift is the most important downstream quantity, so bake it into the encoder rather than train it as a downstream head on a frozen backbone.
- **Masked Spectrum Modeling (MSM)** — random spectral windows masked at every step; model predicts the missing flux. Standard self-supervised pretraining objective adapted to 1D spectra.
- **Physical Prior Regularization** — explicit loss terms encoding physical relationships (redshift equivariance: a coherent shift of all features; known emission/absorption line ratios). The proposal's bid for physics-informed inductive bias.

## 6.4 Training data

- ~50M extragalactic DESI spectra (DR1 and subsequent internal releases)
- ~10M stellar DESI spectra
- ~60M total — a corpus 60× larger than any prior spectroscopic foundation model
- Already hosted at NERSC, co-located with Perlmutter, eliminating data egress
- Pre-public data handled entirely within NERSC behind DOE authentication; checkpoints, embeddings, and derived catalogs released aligned with DESI publication policy

Labels:
- ~1.6M QSO redshifts (Redrock + QuasarNET + broad-MgII)
- DESI Visual Inspection campaigns (Lan et al. 2023; Alexander et al. 2023; Hsu et al. 2026; Weaverdyck et al. in prep) — curated expert classifications and redshifts
- 12,000 strong-lens candidates for downstream validation
- Visually-inspected SN spectra for SN typing
- LBG/LAE test observations for the few-shot extensibility test

## 6.5 Human Alignment Training (the RLHF analogue)

The single most novel element of the proposal. Inspired by Ouyang et al. 2022 (RLHF in LLMs), the proposal applies SFT + preference learning to the spectroscopic domain:

- **SFT**: fine-tune SpectrumFM on DESI spectra with VI redshifts as expert demonstrations. Corrects pretraining biases inherited from imperfect Redrock labels and brings label-rich performance to baseline.
- **Preference learning**: SpectrumFM's auxiliary head outputs ranked candidate redshifts; preference pairs (correct redshift top-ranked vs. lower-ranked) train per-spectrum confidence calibration. Downstream cosmology analyses can then consume per-spectrum confidence scores, weighting/thresholding similarly to ZWARN flags today.

Phase II extends to **rationale-based supervision**: the natural-language VI comment fields become supervision for short model-generated rationales (e.g., "broad Mg II emission near 4200 Å implies z ≈ 0.5 quasar"). This mirrors the LLM alignment shift from preference-only to rationale-based training (e.g., constitutional AI, reasoning-trace fine-tuning).

## 6.6 Downstream validation

Three explicit Phase-I downstream tasks demonstrate that the encoder has learned a transferable representation:

1. **Strong gravitational lens identification** via Hsu et al. 2025's pairwise spectroscopic search. This is the direct connection to Huang's lens program: SpectrumFM is asked to identify lensed systems from spectral features alone.
2. **Supernova typing** — classify SNe Ia / Ib / Ic / II from spectra. A scientifically distinct downstream task.
3. **Scaling behavior** — empirical scaling curves (model performance vs. data, vs. compute) showing positive power-law trends, justifying Phase II investment.

## 6.7 Phase-I 9-month timeline

| Months | Focus | Milestone |
|---|---|---|
| 1–3 | Data curation & architecture | DESI corpus + VI labels assembled; auxiliary-redshift-head + physical-prior + MSM architecture implemented and validated on small-scale runs |
| 4–6 | Pretraining & label-rich | Full-scale pretraining on Perlmutter executed; scaling curves established; Objective 2a complete |
| 7–9 | Extensibility, alignment, validation | Human Alignment Training executed; Objectives 1, 2b, 3 complete; Phase I report + Phase II proposal submitted |

## 6.8 Phase-I go/no-go metrics

1. **Performance advancement.** Match or exceed Redrock (LRG/ELG) and Redrock+QuasarNET+broad-MgII (QSO) across the four label-rich classes, with no class degrading by more than 5% relative to baseline.
2. **Extensibility (the AI advantage).** Few-shot fine-tuning on LAE/LBG with ≤5,000 examples must achieve classification accuracy within 10% of a specialized algorithm, without degrading the original targets.
3. **Scaling behavior.** Positive power-law trends in performance vs. training data and compute.

## 6.9 Compute, infrastructure, and budget

- **Perlmutter (NERSC, LBNL).** A100 GPU partition, HPE Slingshot interconnect. Director's Discretionary Reserve for Months 1–3; ERCAP FY26 allocation thereafter.
- **Doudna (NERSC-10).** Next flagship system entering production late 2026, ~10× Perlmutter performance — the natural Phase II target.
- **Genesis Mission data platform.** Hosted at NERSC; allows compliant sharing of trained checkpoints with other DOE Consortium teams.
- **Year-1 budget:** $605,922 total. USF lead: $407,283. LBNL subaward: $116,283. NOIRLab subaward: $82,356. FSU: $0 (in-kind).

## 6.10 Team and roles

| Institution | Personnel | Role |
|---|---|---|
| USF (lead) | Xiaosheng Huang (PI; $42,478) | Scientific direction, architecture design (with Wang), strong-lens task, SN typing |
| USF | Shan Wang ($27,119) | Co-leads architecture design and testing |
| USF | **Greg Benson ($33,309)** | **Leads agentic-AI tooling** |
| USF | Postdoc ($96,320) + 3 grad students ($45,111) | Execute pretraining, scaling, fine-tuning, ablations, downstream validation |
| LBNL | Stephen Bailey | DESI corpus access, Redrock-baseline evaluation, VI failure-mode taxonomy |
| LBNL | Green (postdoc) | VI data curation, preference-pair construction |
| NOIRLab | Stephanie Juneau + intern ($82,356) | Foundation-model prototyping and evaluation |
| FSU | Nao Suzuki, Eric Hsiao | Cross-survey generalization, SN spectroscopy expertise |

## 6.11 Connection to gravitational lensing

The lensing connection is direct and concrete:

- Hsu et al. 2025's pairwise spectroscopic search is *the* downstream validation task. SpectrumFM is expected to identify lensed systems from spectral features alone (the spectra of multiply-imaged systems contain superposed emission lines at two distinct redshifts).
- Phase II's multimodal extension brings imaging into the encoder, which would let SpectrumFM serve as the encoder for image-based lens detection (replacing or augmenting the ResNet/EfficientNet pipeline).
- The dimple-lens class identified in Hsu 2025 is exactly the kind of system where small spectral features (a faint, low-mass foreground galaxy producing a "partner's Z" line in a target's spectrum) require a high-resolution spectroscopic foundation model rather than a template-fitting algorithm.

## 6.12 Phase-II vision

Three threads:

1. Multimodal extension — DESI spectra + LSST/Roman/Euclid imaging + Gaia astrometry, addressing Focus Area 14-A's call for cross-modality cosmic-physics foundation models.
2. Rationale-based alignment — incorporating free-text VI comments as natural-language supervision for short model-generated explanations.
3. Generalization to Spec-S5 and successor surveys — the encoder built in Phase I is the same encoder that will scale.

---

# 7. Datasets, Instruments, and Computing Infrastructure

## 7.1 Imaging — DESI Legacy Imaging Surveys

The primary discovery substrate:

- **DECaLS** — Dark Energy Camera Legacy Survey on the Blanco 4m at Cerro Tololo. *grz* bands, FWHM ~1.18″, pixel scale 0.262″, declinations ≤ +32°. Covers ~9,000 deg².
- **BASS** — Beijing-Arizona Sky Survey on the Bok 2.3m. *gr* bands. Northern complement to DECaLS.
- **MzLS** — Mayall *z*-band Legacy Survey on the Mayall 4m at Kitt Peak. *z* band only.
- Together: ~14,000 deg² in DR8, ~19,000 deg² in DR9, ~19,000 deg² in DR10 (with reprocessed DECam south of −18°).
- 5σ depths: *g* ≈ 24.0, *r* ≈ 23.4, *z* ≈ 22.5 AB mag.
- Photometric catalogs from **The Tractor** (Lang et al. 2016) — PSF and galaxy-profile based model photometry; the deblending engine that DESI target selection itself relies on.

Data portal: https://www.legacysurvey.org/

## 7.2 Spectroscopy — DESI

- **DESI**, the Dark Energy Spectroscopic Instrument, on the Mayall 4m at Kitt Peak. 5,000 robotically-actuated fibers across a 3.2° focal plane.
- Three-arm spectrograph: blue (3,600 Å) to red (9,800 Å). R = 2,000 (blue) to 5,500 (red).
- DR1: 18M+ unique objects.
- Pipeline: **Redrock** (template fitting) → optional QuasarNET (CNN re-typing) → broad-MgII module.
- Target classes: BGS (bright galaxy survey), LRG (luminous red galaxies, 0.4 ≲ z ≲ 1.0), ELG (emission-line galaxies, 0.6 ≲ z ≲ 1.6), QSO (broad-z range), MWS (Milky Way stellar survey). DESI-II adds LBGs and LAEs.

## 7.3 Follow-up — HST, Keck, VLT

- **HST WFC3/F140W** — wide near-IR filter, ~ 4× tighter PSF than DESI imaging. The Huang group's SNAP program GO-15867 is the source of the Foundry I imaging.
- **Keck-2 NIRES** — near-IR (0.94–2.45 μm) echellette, R ~ 2,700. Used in Foundry III for high-z source redshifts.
- **VLT/MUSE** — integral-field unit, 60″×60″ FoV, 0.2″ spaxel, 4,750–9,350 Å, R = 2,000–5,200. Used in Foundry IV and the Cikota 2023, Sheu 2024b single-system papers.

## 7.4 Computing — NERSC Perlmutter (and the road to Doudna)

- **Perlmutter** — HPE Cray EX. The GPU partition uses NVIDIA A100 accelerators interconnected with HPE Slingshot.
- Already used for: Inchausti 2025 dual-architecture training (4 nodes); GIGA-Lens production runs (4 A100s per fit); SpectrumFM Phase-I (proposed).
- **Doudna (NERSC-10)** — late-2026 production, ~10× Perlmutter performance. The Phase-II SpectrumFM target.

## 7.5 Software stack summary

A consolidated picture of what's used and where:

| Tool | Used in | Role |
|---|---|---|
| TensorFlow + JAX + TFP | GIGA-Lens | Differentiable forward modeling, VI, HMC |
| PyTorch | Inchausti 2025 (some training) | CNN training |
| lenstronomy | Cikota 2023, simulation studies | Reference simulator; baseline lens modeler |
| Astropy / SEP / spherematch | Sheu 2023/24, Hsu 2025 | Catalog matching, photometry |
| The Tractor | Survey level | Source extraction, photometric catalogs |
| PyPelt | Foundry III, IV | Spectroscopic data reduction |
| FastSpecFit | Hsu 2025, Foundry II | Spectro-photometric derived quantities (velocity dispersion) |
| Redrock | DESI baseline | Template-fitting redshifts (the SpectrumFM baseline-to-beat) |

---

# 8. Agentic-Research Landscape (May 2026)

This section is the verification-pass output: every URL and headline claim was checked via WebFetch against authoritative sources on May 25, 2026. The map below organizes systems by *what they automate*, not by vendor.

## 8.1 Code/experiment loops

### 8.1.1 Karpathy autoresearch

- URL: https://github.com/karpathy/autoresearch (verified 2026-05-25)
- 83.3k stars, 12.1k forks. Active.
- **Premise:** "AI agents running research on single-GPU nanochat training automatically." Karpathy frames it as: *"give an AI agent a small but real LLM training setup and let it experiment autonomously overnight."*
- **Design pattern:** three files. `prepare.py` (fixed; data + utilities), `train.py` (agent modifies this — full GPT model, Muon+AdamW optimizer, training loop), `program.md` (human-edited; instructions to the agent).
- **Hard constraints:** 5-minute training window per experiment (gives ~12 experiments/hour, ~100 experiments overnight). Single GPU. Single validation metric (val_bpb, validation bits-per-byte). Agent only modifies `train.py`. All experiments committed to git for auditability.
- **Why it matters for Huang's program:** the design pattern (scoped budget, single editable file, single metric, git ledger) is exactly the discipline you want for autonomous ML research, and maps directly to AutoFoundry (§9.2).

### 8.1.2 Sakana AI Scientist (v1, v2)

- v2 paper: *Yamada et al., "The AI Scientist-v2: Workshop-Level Automated Scientific Discovery via Agentic Tree Search,"* arXiv:2504.08066 (April 10, 2025). Verified 2026-05-25.
- v2 repo: https://github.com/SakanaAI/AI-Scientist-v2. Verified 2026-05-25; license is the AI Scientist Source Code License (derivative of Responsible AI License).
- **v1 vs v2:** v1 required human-authored templates (well-defined objectives); v2 removes that dependence and uses *progressive agentic tree search* with an experiment manager agent that orchestrates exploration. v2 trades v1's higher per-attempt success rate for broader exploratory reach.
- **Milestone:** v2 submitted three fully-autonomous manuscripts to a peer-reviewed ICLR 2025 workshop; one passed peer review (first AI-generated paper to do so in this venue). Caveat: workshop track (~32% acceptance rate), withdrawn by prior agreement.
- **Loop automated:** literature → hypothesis → code → experiment → analysis → manuscript → review. The full research lifecycle.

## 8.2 Literature / scientific QA

### 8.2.1 FutureHouse PaperQA2

- URL: https://github.com/Future-House/paper-qa (verified 2026-05-25)
- Apache 2.0 license. CalVer versioning (current: v2026.03.18). The repo transitioned from SemVer to CalVer in December 2025.
- High-accuracy retrieval-augmented generation on PDFs, text files, Office docs, and source code. Designed for scientific literature.
- Reportedly achieves "superhuman performance on scientific tasks" per the maintainers' 2024 research publication.
- Underpins FutureHouse's Crow product (literature Q&A) and ContraCrow (contradiction finder).

### 8.2.2 FutureHouse Aviary

- Paper: *Narayanan, Braza, Griffiths, et al., "Aviary: training language agents on challenging scientific tasks,"* arXiv:2412.21154 (December 30, 2024). Verified 2026-05-25.
- Repo: https://github.com/Future-House/aviary (environments); https://github.com/Future-House/ldp (agent code).
- "An extensible gymnasium for language agents" with five environments, three scientific (DNA manipulation, literature research, protein engineering).
- Key result: open-source, non-frontier LLMs (Llama 3.1 8B trained agents) match and exceed frontier LLMs and human PhD experts on lab-bench tasks, at dramatically lower compute cost.

### 8.2.3 FutureHouse organization

- https://www.futurehouse.org/ — nonprofit, "automating scientific discovery." Founders Sam Rodriques (CEO) and Andrew White (Head of Science).
- Current 2026 product slate emphasizes DISCO (enzyme invention), OXtal (crystal structure prediction), and Edison Scientific (a new AI-scientist initiative co-developed with Google DeepMind per genengnews.com).

## 8.3 Hypothesis generation & evolution

### 8.3.1 Google DeepMind AI Co-Scientist

- Blog: https://deepmind.google/blog/co-scientist-a-multi-agent-ai-partner-to-accelerate-research/ (verified 2026-05-25; dated May 19, 2026).
- Paper: *Gottweis et al., "Towards an AI co-scientist,"* arXiv:2502.18864 (Feb 26, 2025). Verified 2026-05-25. Built with Gemini 2.0.
- Nature publication ~May 19, 2026 (DOI s41586-026-10644-y).
- **Five specialized agents:**
  - **Generation** — proposes initial focus areas and novel hypotheses grounded in literature.
  - **Proximity** — maps/clusters hypotheses to ensure diverse exploration.
  - **Reflection** — virtual peer reviewer; evaluates correctness and novelty.
  - **Ranking** — orchestrates a pairwise-comparison "idea tournament" (Elo-style).
  - **Evolution** — refines and combines top-ranked hypotheses.
- **Loop pattern:** *generate → debate → evolve*, inspired by the scientific method. Test-time-compute scaling.
- Applications demonstrated: drug repurposing, target discovery, bacterial evolution.
- Available via Hypothesis Generation tool in Google Labs (May 2026 onward).

## 8.4 General research orchestration

### 8.4.1 Anthropic multi-agent research system (Claude)

- URL: https://www.anthropic.com/engineering/built-multi-agent-research-system (verified 2026-05-25; dated June 13, 2025).
- **Pattern:** lead orchestrator agent + 3–5 parallel subagents.
- Subagents are "intelligent filters" — each gets an objective, output format, tools, and clear boundaries.
- Individual subagents do 3+ parallel tool calls.
- Headline: "cut research time by up to 90% for complex queries."
- **Lessons quoted in the post that translate to lens-program work:**
  - *"Delegation requires detail — each subagent needs an objective, an output format, guidance on the tools and sources to use, and clear task boundaries."*
  - *"Effort scaling matters — embedding explicit rules prevents overinvestment; simple queries require one agent, complex research needs 10+ subagents."*
  - *"Tool design is critical — using the right tool is efficient; often, it's strictly necessary."*
  - Extended thinking serves as a "controllable scratchpad."

### 8.4.2 Claude Code

- Product: https://claude.com/product/claude-code (Anthropic's agentic coding tool).
- Launched Feb 24, 2025 (preview with Claude 3.7 Sonnet); general availability May 22, 2025 with Claude 4.
- The natural daily-driver for Benson's agentic-AI tooling work. Reads codebases, plans across files, edits, runs tests, iterates.
- 2026 features: multi-agent orchestration, "outcomes" (branch-and-compare), MCP integrations.

### 8.4.3 OpenAI Deep Research

- Multi-step web-research agent, launched early 2025. Powered by OpenAI's web-optimized model. Verified citations, source links.
- Useful as a comparison point for autonomous web research; less applicable to local-data lensing work.

### 8.4.4 AutoGen / AG2

- Original: https://github.com/microsoft/autogen — Microsoft's conversational multi-agent framework. Moved to maintenance mode 2025.
- Community fork: AG2 (https://github.com/ag2-Ai/ag2) — active continuation. Streaming, event-driven actor model, typed tools, multi-provider LLMs, first-class testing.
- Microsoft Agent Framework (2026) — official successor merging AutoGen + Semantic Kernel.

### 8.4.5 LangGraph

- LangChain's stateful, cyclic graph orchestration framework. Production-grade since LangGraph 1.0 (2025).
- Native multi-agent collaboration with checkpoints (pause/resume) and human-in-the-loop interrupts.
- The most general-purpose orchestration substrate, frequently combined with AG2 (per the Denario architecture).

## 8.5 Astronomy-specific agentic systems

### 8.5.1 Denario / CMBAgent (AstroPilot-AI)

- Repo: https://github.com/AstroPilot-AI/Denario (verified 2026-05-25; v1.0.0 released November 3, 2025).
- Paper: *Villaescusa-Navarro et al., "The Denario project: Deep knowledge AI agents for scientific discovery,"* arXiv:2510.26887 (October 30, 2025). Verified 2026-05-25.
- **Description:** *"Denario, an AI multi-agent system designed to serve as a scientific research assistant. Denario can perform many different tasks, such as generating ideas, checking the literature, developing research plans, writing and executing code, making plots, and drafting and reviewing a scientific paper."*
- **Orchestration framework:** combines AG2 and LangGraph. CMBAgent is the research-analysis backend.
- **Award:** CMBAgent won first place at the NeurIPS 2025 Fair Universe Competition (Phase 1, December 7, 2025).
- **Why it matters:** Denario is the closest existing precedent for a domain-specialized multi-agent science assistant. The pattern (AG2 + LangGraph + specialized backend) is the most directly transferable to a "LensAgent" (§9.1).

### 8.5.2 AstroLLM

- URL: https://astrollm.org/ (verified 2026-05-25).
- Domain-specialized retrieval-grounded LLM for astronomy. Retrieves from 15M+ papers, 20.5M astronomical objects, 5,700+ exoplanets via NASA ADS, SIMBAD.
- Phase 1 development as of 2026. Four planned model sizes (Nano through Ultra).
- *Not* an agent itself — a foundation/RAG layer that agents can sit on top of.

### 8.5.3 NASA-IBM INDUS / Surya

- https://science.nasa.gov/open-science/ai-language-model-science-research/
- INDUS: LLM family fine-tuned on scientific corpora.
- Surya: NASA-IBM solar-storm prediction model. Production at NASA SWPC since 2025.
- Astronomy-adjacent, space-physics focused. Less directly applicable to lensing but worth tracking.

## 8.6 Benchmarks

### 8.6.1 RE-Bench

- Paper: *Wijk et al., "RE-Bench: Evaluating frontier AI R&D capabilities of language model agents against human experts,"* arXiv:2411.15114 (November 22, 2024; revised May 2025). Verified 2026-05-25.
- 7 open-ended ML research-engineering environments. 71 8-hour attempts by 61 human experts as the baseline.
- Headline finding: AI agents score 4× higher than humans at 2-hour budgets; humans recover the lead at longer time budgets.
- The right benchmark to think about if Huang's group ever wants to evaluate agent vs. expert at lens-modeling or lens-finder iteration.

### 8.6.2 MLE-bench

- Paper: *Chan et al., "MLE-bench: Evaluating Machine Learning Agents on Machine Learning Engineering,"* arXiv:2410.07095 (October 9, 2024). Verified 2026-05-25.
- 75 ML-engineering competitions sourced from Kaggle.
- Best agent setups achieve at least bronze-medal performance in 16.9% of competitions.

### 8.6.3 ICLR 2025 Workshop: Agentic AI for Science

- https://iclragenticai.github.io/
- The venue where Sakana's first AI-generated paper passed peer review. A growing community around rigorous evaluation of agentic science.

## 8.7 Cross-cutting patterns (synthesis)

Five patterns recur across every successful system in §8.1–§8.6:

1. **Orchestrator–worker decomposition.** Lead agent + N parallel specialized subagents. Anthropic's pattern, Google Co-Scientist's pattern, Sakana v2's experiment manager, Denario's multi-agent backend.
2. **Generate → debate → evolve.** Hypothesis generation isn't one-shot; it's a tournament with ranking and iterative refinement. Google AI Co-Scientist is the canonical recent expression.
3. **Tool-use specialization is destiny.** Domain-specialized tool suites beat general-purpose agents. ChemCrow (chemistry), Aviary (biology), Denario (astronomy via ADS/SIMBAD). The lensing analogue is a tool layer over lenstronomy, GIGA-Lens, DESI cutout servers, Tractor catalogs, etc.
4. **Human alignment is a research-tool, not just an LLM-safety tool.** SpectrumFM's RLHF analogue is a leading-edge instance of this. Co-Scientist, Sakana v2, FutureHouse all integrate expert ranking/preference data.
5. **Scoped budgets + ledgers.** Karpathy's 5-minute experiments + git ledger pattern. Sakana's tree-search bounded by compute. The discipline that distinguishes "agents that produce research" from "agents that loop forever."

The May-2026 economics: Claude Code is at $2.5B+ annual run-rate; OpenAI Deep Research is monetized; Google's Gemini-for-Science is in lab rollout; Sakana, FutureHouse/PaperQA, LangGraph, AutoGen/AG2 are mature open-source. Adoption cost is low.

What is *not* (yet) in the public landscape, as of May 25, 2026: a dedicated agentic system for gravitational lensing. Denario is the closest, but lensing-specific tooling and lens-modeling-as-tool have not been integrated. **This is the whitespace Huang's group can move into.**

---

# 9. Proposed Agentic-AI Applications

Each subsection below proposes a concrete agentic system, named for clarity. For each: what it automates, why this program needs it, the architecture and role decomposition, the tool surface, the suggested orchestration framework, success metrics, prototype effort, and risks. These are designs, not implementations; §10 sequences which to build first.

## 9.1 LensAgent — Multi-agent candidate-triage orchestrator

**What it automates.** The visual-inspection step where ResNet/EfficientNet outputs (~5,000–10,000 above-threshold cutouts per DR-scale search) get graded A/B/C/D by team members. This is currently a wall-clock bottleneck.

**Why it fits.** The grading step is *structured judgment*: multiple specific factors (arc morphology, color, source-galaxy plausibility, prior-catalog overlap, lens-galaxy properties) get integrated. That's exactly what a multi-agent orchestrator does well — specialized agents each evaluate one factor, then an arbitrator integrates.

**Architecture (orchestrator-worker, Denario / Anthropic pattern):**

```
LensAgent (orchestrator)
├── VisionAgent          — frozen ResNet+EfficientNet ensemble reused; outputs p(lens)
├── MorphologyAgent      — checks for arc-like curvature, multi-image counterparts; can call
│                          a quick lenstronomy fit to test "is there a plausible lens model?"
├── ColorAgent           — checks lens vs. source colors against population priors;
│                          looks for the lens-red / source-blue signature
├── CrossmatchAgent      — queries NASA ADS, MAST, prior-Huang Lens-DB; flags overlaps
├── ContaminantAgent     — checks for known false-positive morphologies (cosmic rays,
│                          rings around bright stars, satellite trails, spiral structure)
├── GradeArbitratorAgent — integrates the above; produces A/B/C/D grade with rationale
└── HumanInLoopGate      — for borderline cases, escalates to human reviewer with
                            a structured pre-filled review form
```

**Tool surface.**

- DESI cutout server (https://www.legacysurvey.org/viewer/) — `get_cutout(ra, dec, bands, pixscale)`.
- Tractor catalog API — `get_photometry(ra, dec, radius)`.
- `quick_lensmodel(cutout, prior)` — wrapper on GIGA-Lens VI-only fit (no HMC), ~5 s budget per attempt.
- NASA ADS API (for cross-matching against published lenses).
- The local Lens-DB (Huang group's accumulated candidate database).

**Orchestration framework.** AG2 + LangGraph (the Denario pattern). MCP for tool exposure so the same tools work with multiple LLM backbones.

**Success metrics.** Agreement with the human-grader baseline (Cohen's κ) on a held-out set of 500 systems graded by ≥2 humans. Throughput in cutouts/hour at fixed κ. False-positive rate at the operating threshold.

**Prototype effort.** Estimated 4–8 weeks for a single grad student + Benson's tooling. Re-uses existing ResNet weights, lenstronomy/GIGA-Lens, and the existing grading rubric.

**Risks.** (a) Cohen's κ may be lower than human-vs-human agreement on borderline systems; mitigate by requiring agent agreement *plus* human review for any system intended for HST/Keck/MUSE allocation. (b) Cost runaway if the agent calls quick_lensmodel too often — bound by per-system budget. (c) The agent can hallucinate "lens-like" features in noisy cutouts; require evidence from at least two of (Morphology, Color, Crossmatch) for A grade.

## 9.2 AutoFoundry — autoresearch-style training-loop driver

**What it automates.** Iteration on the dual-architecture lens-finder code (Inchausti 2025 ResNet + EfficientNet + meta-learner). Hyperparameter sweeps, architectural ablations, training-data curation experiments, threshold tuning.

**Why it fits.** Karpathy's autoresearch design — single editable training file, scoped time budget, single metric, git-tracked experiments — maps directly onto the lens-finder iteration cycle. The current cycle is *months* of student-led tuning between papers.

**Architecture (Karpathy autoresearch + Anthropic-style budget control):**

```
AutoFoundry agent
├── reads:       lensfinder/train.py   (the training script — editable)
│                lensfinder/data.py    (data utilities — fixed)
│                lensfinder/program.md (instructions to the agent)
│                lensfinder/baseline_metrics.json
├── iterates:    propose change → train (NERSC budget: 30-min wall-clock) → evaluate AUC
│                + FP rate at threshold → commit-or-discard → repeat
├── ledger:      git commits + a structured experiment log
└── escalation:  promotes top-K experiments through a tournament for human review
```

**Tool surface.** Slurm job submission on Perlmutter. Git. A small monitoring server for live AUC curves. A simple `bench/evaluate.py` that takes a checkpoint and returns the headline metrics.

**Orchestration framework.** Claude Code as the agent driver; either custom or LangGraph for the budgeted loop.

**Success metrics.** AUC improvement on the held-out HST-confirmed lens set. Number of experiments run per week. Time-to-promote-a-winner from idea to merge.

**Prototype effort.** 2–4 weeks. Most of the heavy lifting is wrapping the existing training code with a clean `train.py` + `program.md` interface.

**Risks.** (a) Compute cost. Mitigated by hard NERSC allocation budget; agent treats compute as a finite resource. (b) Overfitting to held-out validation. Mitigate with a separate "blind" test set (HST-confirmed lenses *not* used for tuning) revealed quarterly. (c) Reproducibility — agents introduce non-determinism; pin all seeds, log everything, require any "winner" to be re-runnable from a single commit.

## 9.3 DimpleScout — hypothesis-driven spectroscopic search agent

**What it automates.** The Hsu 2025 pairwise spectroscopic pipeline currently flags ~318 dimple-lens candidates from DESI DR1. Scaling to DR2 and beyond means: refining the selection criteria, exploring new redshift-ratio cuts, integrating photometric priors, and ranking the resulting candidates for visual inspection. The agent runs a Co-Scientist-style generate-debate-evolve loop over the *selection criteria themselves*.

**Why it fits.** The dimple-lens class is brand new; the right criteria are not yet locked in. This is genuinely a hypothesis-evolution task: propose a refinement (e.g., "add a velocity-dispersion cut at σ_v > X"), test it on the held-out set, debate against the current best, and evolve.

**Architecture (generate–debate–evolve, Google Co-Scientist pattern):**

```
DimpleScout (orchestrator)
├── GenerationAgent      — proposes selection-criterion refinements grounded in lensing physics
├── EvaluationAgent      — applies the proposed criteria to held-out DESI fiber pairs,
│                           reports precision/recall against the known dimple set
├── ReflectionAgent      — virtual peer reviewer; checks for physics violations, sample bias,
│                           selection effects
├── RankingAgent         — pairwise tournament among the surviving criteria
└── EvolutionAgent       — recombines top-ranked criteria, mutates parameters, returns to
                            GenerationAgent for next round
```

**Tool surface.** DESI DR1+ spec server. FastSpecFit velocity-dispersion catalog. spherematch FoF. DR10 imaging cutouts via Legacy Survey API. Lightweight quick-lensmodel via GIGA-Lens VI.

**Orchestration framework.** AG2 for the agent debate; LangGraph for stateful tournament checkpoints.

**Success metrics.** Number of new dimple-lens candidates per round of agent-driven criterion refinement. Precision/recall on the held-out set. Wall-clock to reach a stable criterion set.

**Prototype effort.** 6–10 weeks (a half-semester project for a motivated grad student + Benson). Depends on Hsu having documented the current criteria and held-out set.

**Risks.** (a) The dimple-lens "ground truth" set is small (318 systems); agents may overfit. Mitigate with leave-one-pipeline-version-out validation. (b) Physics correctness — agents could propose criteria that violate lensing geometry. Mitigate by requiring ReflectionAgent to flag any rule that contradicts the Einstein-radius formula or related physics.

## 9.4 SpectrumFM Autopilot — Phase-I training-pipeline orchestrator

**What it automates.** The SpectrumFM Phase-I 9-month milestone plan. Data curation QC, MSM-pretraining sweeps, redshift-head ablations, scaling-curve fits, preference-pair construction from VI archives, alignment training, downstream-validation runs.

**Why it fits.** This is the work package Benson explicitly owns ("Benson — leads agentic-AI tooling"). The proposal calls out a structured 3×3-month timeline with explicit milestones; that maps cleanly onto an autonomous loop with human checkpoints at month boundaries.

**Architecture (autoresearch + multi-agent for stages with multiple sub-tasks):**

```
SpectrumFM Autopilot (top-level orchestrator)
├── M1-M3 phase:  DataCurationAgent          — DESI DR1+ corpus assembly, VI label QC
│                 ArchitectureAgent          — implements auxiliary redshift head,
│                                              physical-prior loss, MSM
│                 SmokeTestAgent             — small-scale validation runs
├── M4-M6 phase:  PretrainingAgent           — full-scale Perlmutter runs; monitors loss
│                 ScalingCurveAgent          — runs the scaling sweep
│                 BaselineComparisonAgent    — head-to-head vs. Redrock for LRG/ELG/QSO
├── M7-M9 phase:  AlignmentAgent             — SFT + preference learning over VI data
│                 ExtensibilityAgent         — few-shot LBG/LAE fine-tuning
│                 DownstreamValidationAgent  — strong-lens identification, SN typing
│                 ReportAgent                — drafts Phase-I report
└── checkpoints:  human reviews at end of each 3-month phase before authorizing the next
```

**Tool surface.** NERSC Slurm. Weights & Biases or MLflow for experiment tracking. DESI Visual Inspection corpus access. Cluster-side data loaders. The strong-lens validation set (Hsu 2025 + 12,000 candidates referenced in the proposal). The SN spectroscopic typing benchmark.

**Orchestration framework.** Claude Code driving the agent; either custom or LangGraph for the stateful month-by-month workflow. MCP wraps Slurm and the data tools so the same agents work across Perlmutter and (eventually) Doudna.

**Success metrics.** Phase-I go/no-go criteria as stated in the proposal: ≥ Redrock performance with no class degrading >5%; few-shot LAE/LBG within 10% of specialized algorithm; positive power-law scaling.

**Prototype effort.** Substantial — this *is* the Phase-I work plan. But the *autopilot harness* can be prototyped in 8–12 weeks; the runs themselves take the full 9 months.

**Risks.** (a) Compute budget — Phase-I has a finite NERSC allocation. The autopilot must respect it; Benson controls budget gates. (b) Run failures — distributed pretraining is fragile; agent must handle restart-from-checkpoint, NaN losses, OOM. (c) Premature optimization — the agent must not collapse to the easy fixes; needs an explicit exploration budget separate from exploitation.

## 9.5 GIGA-LensAgent — Bayesian lens-modeling at scale

**What it automates.** Per-system Bayesian modeling with GIGA-Lens. Currently this requires expert judgment: which mass model? which priors? when to stop at VI vs. escalate to HMC? when to add a secondary lens galaxy or a multipole? The agent encodes this judgment.

**Why it fits.** Foundry-scale physical characterization needs O(10³–10⁴) lens models in the LSST/Rubin era. That is not human-feasible. The current ~100 systems modeled by hand will scale 1,000×.

**Architecture (wrapper agent + tool-using LLM):**

```
GIGA-LensAgent
├── PriorSelectorAgent      — picks priors from candidate type, redshift, image multiplicity
├── ModelSelectorAgent      — SIE vs. EPL vs. EPL+multipole; with/without external shear;
│                              with/without secondary lens
├── FitDriverAgent          — runs multi-start gradient descent → VI → (conditionally) HMC
├── DiagnosticAgent         — checks chain convergence, posterior predictive residuals
├── EscalationAgent         — decides "this is hard, escalate to a human modeler"
└── CardWriterAgent         — produces a per-system summary card with figures, parameters,
                               quality flags
```

**Tool surface.** GIGA-Lens API (model construction, fitting). lenstronomy as fallback for unusual systems. Posterior diagnostic plotters (corner plots, residual maps). Slurm.

**Orchestration framework.** Claude Code or AG2; MCP for tool wrappers.

**Success metrics.** Agreement of agent-derived posterior parameters (θ_E, mass-slope, ellipticity) with hand-modeled posteriors on a held-out set of 20 well-studied systems. Throughput in systems/day. Fraction of systems escalated to humans.

**Prototype effort.** 6–10 weeks. Most of the difficulty is encoding the prior/model-selection judgment.

**Risks.** (a) Convergence failures — HMC chains can fail silently; DiagnosticAgent must be rigorous. (b) Mass-model misspecification — agent picks the wrong family of models. Mitigate by reporting Bayesian evidence across families. (c) Garbage-in/garbage-out — if the candidate cutout is actually a non-lens, GIGA-Lens will produce nonsense; LensAgent (§9.1) is upstream of this.

## 9.6 FoundryScribe — manuscript-drafting agent for confirmation papers

**What it automates.** Sections of the DESI Strong Lens Foundry-style confirmation papers. Object tables, redshift tables, lens-model summary tables, the introduction section ("X strong lenses confirmed via Y"), the conclusions. Humans retain final-pass control over interpretation.

**Why it fits.** Foundry-series papers have a stable, repeatable structure: candidate table, observations table, spectroscopic results table, lens-model table, discussion. The agent template-writes the predictable pieces and leaves science interpretation to humans.

**Architecture (Sakana v2 writing-agent pattern, scoped to a paper template):**

```
FoundryScribe
├── ContextAgent       — reads candidate metadata, observations, GIGA-Lens posteriors
├── TableAgent         — produces LaTeX tables for object/redshift/lens-model summaries
├── IntroAgent         — drafts an introduction (cite predecessors, summarize sample size)
├── MethodsAgent       — drafts the methods section (instrument config, reduction pipeline)
├── ResultsAgent       — drafts the results section (per-class statistics, headline numbers)
├── DiscussionStubAgent — leaves a stub with prompts for human interpretation
└── ReferencesAgent    — auto-cites against NASA ADS bibcodes
```

**Tool surface.** NASA ADS API. The Foundry LaTeX template. The Lens-DB. A LaTeX compiler for round-trip validation.

**Orchestration framework.** Claude Code; AG2 for the multi-section parallelization.

**Success metrics.** Time-to-first-complete-draft, human-edit count per section, agreement of auto-generated tables with manually-curated reference tables.

**Prototype effort.** 4–6 weeks. Reuses existing Foundry template prose.

**Risks.** (a) Citation hallucination — agents can invent ADS bibcodes. Mitigate by requiring every citation to be ADS-verified before insertion. (b) Over-claiming — agents may overstate the scientific significance. Mitigate by sandboxing discussion to human-only sections. (c) Authorship norms — publish-ready text must disclose AI assistance per ICLR-2025-workshop-style norms.

## 9.7 LensLit — literature-surveillance agent

**What it automates.** Daily NASA ADS surveillance for new lensing-related preprints. Cross-matching newly-published lens systems against the Huang Lens-DB to detect overlap. Weekly digests sent to the team.

**Why it fits.** Smallest scope, highest weekly payoff. Fastest to prototype. Builds the team's familiarity with the agent stack before larger investments.

**Architecture (PaperQA2 + Crow pattern):**

```
LensLit
├── ADSWatcherAgent    — daily ADS query: "strong gravitational lens" + recent
├── SummarizerAgent    — per-paper TL;DR, with quoted methodology
├── CrossmatchAgent    — extracts any published RA/Dec of new lens systems;
│                         matches against Lens-DB
├── DigestAgent        — assembles the weekly digest with priority flags
└── DistributionAgent  — posts to a team Slack channel / email list
```

**Tool surface.** NASA ADS API. The Huang Lens-DB. Slack or email.

**Orchestration framework.** PaperQA2 + LangGraph; or Claude Code + MCP for the simpler version.

**Success metrics.** Daily uptime. Percentage of human-flagged-as-relevant papers also surfaced by the agent. False-positive rate in the digest.

**Prototype effort.** 1–2 weeks. The smallest possible useful agent.

**Risks.** (a) ADS rate limits — bound the query frequency. (b) Spam noise — keep the digest curated. (c) Privacy of the Lens-DB — keep the cross-match local; do not send candidate coordinates to external APIs.

---

# 10. Recommendations and Next Steps

## 10.1 First-90-day priorities

1. **Ship LensLit (§9.7).** 1–2 weeks. Smallest scope, highest weekly payoff. Builds team familiarity with the agent stack.
2. **Stub SpectrumFM Autopilot (§9.4) harness.** Begin the harness even before the DOE Genesis funding hits — that way, the moment Phase-I starts, the autopilot is ready. Benson owns this.
3. **Scope DimpleScout (§9.3).** The new dimple-lens class is the highest-upside scientific target; getting a Co-Scientist-style loop running on Hsu's selection criteria is the natural pilot of generate-debate-evolve in this domain.

## 10.2 Tooling stack

Concrete recommendations, picking from the verified May-2026 landscape:

- **IDE/coding agent:** Claude Code. The daily driver for agent development. Supports MCP, multi-agent orchestration, branch-and-compare.
- **Multi-agent orchestration:** AG2 (Denario precedent) for inter-agent conversation. LangGraph for stateful workflows with checkpoints.
- **Tool exposure:** MCP (Model Context Protocol). Wrap GIGA-Lens, the DESI cutout server, NASA ADS, the Lens-DB, etc. — each as an MCP server. Then any LLM (Claude, Gemini, GPT) can use the same tools.
- **Literature layer:** PaperQA2 (Future-House/paper-qa). The mature open-source layer for scientific RAG.
- **Experiment tracking:** Weights & Biases for SpectrumFM training; lighter-weight git ledgers per Karpathy autoresearch for the lens-finder iteration.
- **Compute:** NERSC Perlmutter (current), Doudna (Phase II, late 2026).

## 10.3 Evaluation

A "LensBench" inspired by RE-Bench would be valuable: a held-out evaluation set of (a) HST-confirmed strong lenses, (b) known non-lenses with hard-to-distinguish morphology, and (c) historical dimple-lens candidates. Each candidate agent (LensAgent, DimpleScout, etc.) is then evaluated on this set. The benchmark itself becomes a publication — both because rigorous agent benchmarks are scarce, and because publishing the benchmark commits the field to a shared evaluation.

## 10.4 Open-source the agentic pipelines

Following the Sakana, FutureHouse, Denario model, open-source the agentic pipelines as they mature. This both builds community visibility for Huang's group and forces the engineering hygiene that production-grade research code needs.

## 10.5 Risk register

| Risk | Mitigation |
|---|---|
| Hallucinated lens classifications enter the candidate database | Require evidence from ≥2 independent agents *plus* human review for any candidate intended for HST/Keck/MUSE allocation |
| Compute-cost runaway in iterative agent loops | Strict NERSC allocation budgets per workflow; agent must surface its remaining budget at each step |
| Reliance on closed-model APIs (Claude, Gemini) for production research | Open-source models (Llama, Mistral) as fallback; MCP makes the model swappable; AION-1 / SpectrumFM open the path to a self-hosted scientific encoder |
| Reproducibility of agent-driven results in publications | Per-experiment git commits + pinned seeds + Karpathy-style ledger; publication clearly discloses agent involvement |
| Agent debate degrades into mutual confirmation | Reflection agent must be a *critic* (different prompt, different model if affordable), not a sycophant |
| Privacy of unpublished candidate coordinates | Lens-DB cross-matching stays local; nothing pre-publication leaves NERSC or USF infrastructure |

## 10.6 Strategic horizon

Within 12 months the group should aim to:

1. Have a production LensLit + DimpleScout in the team's daily workflow.
2. Have at least one Foundry-series paper that lists agentic-AI tooling in its acknowledgements.
3. Have SpectrumFM Phase-I in motion with the Autopilot harness running.
4. Have a publishable LensBench evaluation suite.
5. Have an open-source release of the LensAgent triage pipeline.

Within 24 months (~end of Phase I, possibly into Phase II of SpectrumFM):

1. Run GIGA-LensAgent at Foundry scale (~1,000 modeled systems).
2. Publish a methods paper on the Co-Scientist-style hypothesis evolution that produced new dimple-lens selection criteria.
3. Stand up a multi-modal SpectrumFM in collaboration with the Polymathic-AI / AION-1 community.

---

# 11. Glossary

**Lensing physics**

- **Einstein radius (θ_E)** — angular scale of strong lensing; for an isothermal lens, θ_E = 4π (σ_v/c)² × D_LS/D_S.
- **Convergence (κ)** — surface mass density normalized to a critical density; κ > 1 → multiple images possible.
- **Shear (γ)** — tidal lensing field; produces image stretch and rotation.
- **Critical curve** — image-plane locus where magnification formally diverges.
- **Caustic** — source-plane locus corresponding to a critical curve; source crossings change image multiplicity.
- **Time delay** — relative arrival-time difference between multiple images; gives H₀ via Refsdal's method.
- **Mass-sheet degeneracy** — uniform mass sheet + source rescaling preserves image positions; broken by velocity-dispersion measurements.
- **Dimple lens** — Hsu 2025's new class: low-mass foreground galaxy producing surface-brightness indentations rather than arcs; M_halo ≲ 10¹³ M_⊙.

**Machine learning**

- **ResNet** — Residual Network (He et al. 2016); CNN with skip connections; lens-finding workhorse.
- **EfficientNetV2** — second-generation EfficientNet with improved training efficiency.
- **U-Net** — segmentation CNN (Ronneberger et al. 2015); used in Silver 2025 for subhalo localization.
- **Transformer** — attention-based architecture (Vaswani et al. 2017); SpectrumFM backbone.
- **VI — Variational Inference** — posterior approximation by minimizing KL to a tractable family.
- **HMC — Hamiltonian Monte Carlo** — gradient-informed posterior sampler; gold standard in high dimensions.
- **MSM — Masked Spectrum Modeling** — self-supervised spectroscopic pretraining; SpectrumFM's primary objective.
- **SFT — Supervised Fine-Tuning** — alignment phase using expert demonstrations.
- **RLHF — Reinforcement Learning from Human Feedback** — alignment via preference-pair learning.
- **MCP — Model Context Protocol** — Anthropic-driven standard for exposing tools to LLMs.

**Astronomy / survey**

- **LRG / ELG / QSO / MWS** — DESI extragalactic target classes (Luminous Red Galaxies, Emission-Line Galaxies, Quasars) plus the Milky Way Stellar survey.
- **LBG / LAE** — Lyman-Break Galaxies, Lyman-α Emitters; DESI-II target classes.
- **VI campaign** — DESI Visual Inspection campaign — human-validated redshifts/classifications.
- **FoF — Friends-of-Friends** — spatial clustering algorithm; Hsu 2025 uses 3″ linking length.
- **DESI / DR1 / DR8 / DR9 / DR10** — Dark Energy Spectroscopic Instrument; its imaging-survey data releases.
- **HST / WFC3 / F140W** — Hubble Space Telescope's Wide Field Camera 3 in its wide near-IR filter.
- **NIRES** — Keck-2 Near-Infrared Echellette Spectrometer.
- **MUSE** — Multi-Unit Spectroscopic Explorer on VLT; integral-field optical spectrograph.
- **The Tractor** — model-based source extraction and photometry code; underpins DESI catalogs.

**DOE / infrastructure**

- **DOE Genesis Mission** — Office of Science initiative for foundation models in particle and cosmic physics; Topic 14, Focus Area A.
- **NERSC / Perlmutter** — National Energy Research Scientific Computing Center; HPE Cray EX with NVIDIA A100 GPU partition.
- **Doudna (NERSC-10)** — NERSC's next flagship system, production late 2026, ~10× Perlmutter.

---

# 12. References

## 12.1 Huang-group papers (the corpus)

1. Huang, X., et al. 2020. *Finding Strong Gravitational Lenses in the DESI DECam Legacy Survey.* ApJ. arXiv:1906.00970.
2. Huang, X., et al. 2021. *Discovering New Strong Gravitational Lenses in the DESI Legacy Imaging Surveys.* ApJ. arXiv:2005.04730.
3. Gu, A., Huang, X., et al. 2022. *GIGA-Lens: Fast Bayesian Inference for Strong Gravitational Lens Modeling.* ApJ. arXiv:2202.07663.
4. Dawes, C., Storfer, C., Huang, X., et al. 2022. *Finding Multiply Lensed and Binary Quasars in the DESI Legacy Imaging Surveys.* ApJS 269, 16.
5. Sheu, W., Huang, X., et al. 2023. *Retrospective Search for Strongly Lensed Supernovae in the DESI Legacy Imaging Surveys.* arXiv:2301.03578.
6. Cikota, A., Toro Bertolla, I., Huang, X., et al. 2023. *DESI-253.2534+26.8843: A New Einstein Cross Spectroscopically Confirmed with VLT/MUSE and Modeled with GIGA-Lens.* arXiv:2307.12470.
7. Sheu, W., Huang, X., et al. 2024a. *A Targeted Search for Variable Gravitationally Lensed Quasars.* arXiv:2408.02670.
8. Sheu, W., Cikota, A., Huang, X., et al. 2024b. *The Carousel Lens: A Well-Modeled Strong Lens with Multiple Sources Spectroscopically Confirmed by VLT/MUSE.* arXiv:2408.10332.
9. Storfer, C., Huang, X., et al. 2024. *New Strong Gravitational Lenses from the DESI Legacy Imaging Surveys Data Release 9.* ApJS 274, 16. arXiv:2309.18089.
10. Huang, X., et al. 2025a. *DESI Strong Lens Foundry I: HST Observations and Modeling with GIGA-Lens.* arXiv:2502.03455.
11. Agarwal, S., Huang, X., et al. 2025. *DESI Strong Lens Foundry III: Keck Spectroscopy for Strong Lenses Discovered Using Residual Neural Networks.* arXiv:2501.08066.
12. Silver, E., Wang, R., Huang, X., et al. 2025. *ML-Driven Strong Lens Discoveries: Down to θ_E ∼ 0.03″ and M_halo ∼ 10¹¹ M_⊙.* arXiv:2207.09431.
13. Hsu, Y.-M., Huang, X., et al. 2025. *A New Way to Discover Strong Gravitational Lenses: Pair-wise Spectroscopic Search from DESI DR1.* arXiv:2509.16033.
14. Lin (Lis), E., Toro Bertolla, I., Cikota, A., Huang, X., et al. 2025. *DESI Strong Lens Foundry IV: Spectroscopic Confirmation of DESI Lens Candidates with VLT/MUSE.* arXiv:2509.18078.
15. Inchausti, J. C., Storfer, C., Huang, X., et al. 2025. *Strong Lens Discoveries in DESI Legacy Imaging Surveys DR10 with Two Deep Learning Architectures.* arXiv:2508.20089.
16. Huang, X., Inchausti, J. C., Storfer, C., et al. 2025b. *DESI Strong Lens Foundry II: DESI Spectroscopy for Strong Lens Candidates.* arXiv:2509.18089.

## 12.2 Lensing background and reviews

- Bartelmann, M., & Schneider, P. 2001. *Weak Gravitational Lensing.* Physics Reports 340, 291.
- Treu, T. 2010. *Strong Lensing by Galaxies.* Annu. Rev. Astron. Astrophys. 48, 87.
- Refsdal, S. 1964. *On the possibility of determining Hubble's parameter and the masses of galaxies from the gravitational lens effect.* MNRAS 128, 307.
- Birrer, S., & Amara, A. 2018. *lenstronomy: Multi-purpose gravitational lens modeling software.* Phys. Dark Universe 22, 189.
- Lanusse, F., et al. 2018. *CMU DeepLens: deep learning for automatic image-based galaxy-galaxy strong lens finding.* MNRAS 473, 3895.
- Metcalf, R. B., et al. 2019. *The Strong Gravitational Lens Finding Challenge.* A&A 625, A119.

## 12.3 Foundation models and ML methods

- He, K., et al. 2016. *Deep Residual Learning for Image Recognition.* CVPR.
- Ronneberger, O., Fischer, P., & Brox, T. 2015. *U-Net: Convolutional Networks for Biomedical Image Segmentation.* MICCAI.
- Vaswani, A., et al. 2017. *Attention Is All You Need.* NeurIPS.
- Ouyang, L., et al. 2022. *Training Language Models to Follow Instructions with Human Feedback.* NeurIPS 35, 27730.
- Parker, L., et al. (Polymathic-AI) 2025. *AION-1: Omnimodal Foundation Model for Astronomical Sciences.* arXiv:2510.17960.

## 12.4 Agentic-research projects and benchmarks (all WebFetch-verified May 25, 2026)

- **Karpathy autoresearch**: https://github.com/karpathy/autoresearch.
- **Sakana AI Scientist v2**: Yamada et al. arXiv:2504.08066. Repo: https://github.com/SakanaAI/AI-Scientist-v2.
- **FutureHouse PaperQA2**: https://github.com/Future-House/paper-qa (Apache 2.0, CalVer).
- **FutureHouse Aviary**: Narayanan et al. arXiv:2412.21154. Repo: https://github.com/Future-House/aviary; agent code https://github.com/Future-House/ldp.
- **FutureHouse organization**: https://www.futurehouse.org/.
- **Google AI Co-Scientist**: blog https://deepmind.google/blog/co-scientist-a-multi-agent-ai-partner-to-accelerate-research/; paper Gottweis et al. arXiv:2502.18864.
- **Anthropic multi-agent research system**: https://www.anthropic.com/engineering/built-multi-agent-research-system (June 13, 2025).
- **Claude Code**: https://claude.com/product/claude-code.
- **AutoGen** (legacy): https://github.com/microsoft/autogen. **AG2** (community): https://github.com/ag2-Ai/ag2.
- **LangGraph**: https://langchain-ai.github.io/langgraph/ (LangChain Academy course at https://academy.langchain.com/courses/deep-research-with-langgraph).
- **Denario / CMBAgent**: https://github.com/AstroPilot-AI/Denario; paper Villaescusa-Navarro et al. arXiv:2510.26887.
- **AstroLLM**: https://astrollm.org/.
- **NASA-IBM INDUS / Surya**: https://science.nasa.gov/open-science/ai-language-model-science-research/.
- **RE-Bench**: Wijk et al. arXiv:2411.15114.
- **MLE-bench**: Chan et al. arXiv:2410.07095.
- **ICLR 2025 Workshop on Agentic AI for Science**: https://iclragenticai.github.io/.

---

# Appendix A — Per-paper deep dives

The 16 papers in the corpus, in roughly chronological order. Each entry: science question, dataset, method, headline results, takeaway for Benson's onboarding.

## A.1 Huang et al. 2020 — DECaLS lens search (arXiv:1906.00970)

**Question.** Can a ResNet trained on real (observed) lenses, not simulations, find new strong lenses at scale in DECaLS?

**Dataset.** 9,000 deg² DECaLS *grz*, 101×101 px cutouts. ~700 observed lenses + ~13,000 non-lens cutouts. Train/val/test 70/20/10.

**Method.** ResNet adapted from Lanusse et al. 2018, re-implemented in TensorFlow. Cross-entropy loss. Trained 120 epochs on 3 Haswell nodes, 17 hours.

**Result.** 335 new lens candidates. ROC-AUC = 0.98 on validation.

**Takeaway.** Establishes the "real-data wins over simulated data" thesis that underwrites the entire program. The training-data composition (curated real lenses + diverse real negatives including spirals, cosmic rays, artifacts) is the key novelty, not the architecture itself.

## A.2 Huang et al. 2021 — DESI Legacy Surveys (arXiv:2005.04730)

**Question.** Can the pipeline scale to the full ~14,000 deg² DESI Legacy footprint (DECaLS + BASS + MzLS) without losing performance to varying survey depth?

**Dataset.** As above plus BASS and MzLS. 632 lenses, 21,000 non-lens cutouts. 5σ depths *g* ≈ 24.0, *r* ≈ 23.4, *z* ≈ 22.5.

**Method.** ResNet with 1×1 "shielding" layers — dimensionality reduction that drops total parameter count from 3M to 60K. Trained on Google Colab V100. Depth-balanced non-lens sampling to prevent the classifier from learning depth as a lens-likelihood proxy.

**Result.** 1,210 new lens candidates. AUC = 0.992. Combined H20+H21 = 1,545 candidates.

**Takeaway.** Architectural compression (50× parameter reduction) + depth-balanced negative sampling generalize the H20 result across surveys. The depth-balance lesson is one Benson should remember when designing any new candidate finder: imaging-depth proxies are everywhere and need to be controlled at training-data construction time, not at evaluation.

## A.3 Gu, Huang et al. 2022 — GIGA-Lens (arXiv:2202.07663)

**Question.** Can the Bayesian lens-modeling pipeline be accelerated to handle O(10⁵) systems for LSST/Rubin/Euclid/Roman?

**Dataset.** Simulated lens systems via lenstronomy, 22 parameters per system. Reference EPL + external shear mass model with elliptical Sérsic source + lens light.

**Method.** TensorFlow + JAX implementation. Three-stage inference: (a) multi-start gradient descent for global mode-finding, (b) Variational Inference (mean-field Gaussian) for posterior covariance, (c) Hamiltonian Monte Carlo (NUTS) initialized from the VI posterior for exact sampling. All gradients via automatic differentiation; all computation on GPUs.

**Result.** 105 seconds per system on 4 NVIDIA A100 GPUs (vs. ~4.3 hours for lenstronomy + emcee). Scales to O(10⁵) projected.

**Takeaway.** This is the workhorse modeling tool. Three things Benson should know: (a) it's differentiable, which is what makes HMC tractable; (b) the three-stage flow (GD → VI → HMC) is a general template for high-D Bayesian inference, not specific to lensing; (c) any agent that wraps lens modeling will treat GIGA-Lens as a primary tool (§9.5).

## A.4 Dawes, Storfer, Huang et al. 2022 — autocorrelation lensed quasars (ApJS 269, 16)

**Question.** Can a non-CNN approach (autocorrelation on the DESI quasar catalog) find lensed and binary quasar candidates missed by morphological CNNs?

**Dataset.** ~5M objects in the DESI quasar sample. SDSS DR16 + SupaNova IFS for follow-up redshifts.

**Method.** Color cuts (W1−W2 vs r−g, z−W vs g−z), PSF morphology, Gaia proper-motion / parallax filters. Spatial autocorrelation finds pairs of quasar-like objects close on the sky.

**Result.** 436 new multiply-lensed / binary-quasar candidates, classified into A/B/C groups by significance.

**Takeaway.** Complementary discovery modality to the imaging-CNN approach. The 436 number propagates into Foundry II's "~3500 candidates + 436 lensed quasars" headline.

## A.5 Sheu, Huang et al. 2023 — lensed supernovae (arXiv:2301.03578)

**Question.** Can a retrospective difference-imaging search of multi-epoch DESI Legacy coadds find lensed supernovae among known lens candidates?

**Dataset.** 5,807 strong-lens candidates in *grizY* DESI Legacy multi-epoch coadds.

**Method.** Image-subtraction pipeline using both B08 (Bramich 2008) and SFFT (Hu et al. 2022) algorithms. SEP for source detection. SALT3 light-curve fitting. Selection criteria: strong-lensing plausibility, asteroid filtering, location relative to the lens, light-curve quality, amplification/Hubble-diagram-residual analysis.

**Result.** 7 new lensed-SN candidates.

**Takeaway.** Defines the difference-imaging stack the group uses for all variable-transient searches. The 7-SN candidate set is small but each is enormously scientifically valuable for time-delay H₀ work.

## A.6 Cikota, Toro Bertolla, Huang et al. 2023 — DESI-253 Einstein cross (arXiv:2307.12470)

**Question.** Spectroscopically confirm and model a candidate Einstein cross from H21.

**Dataset.** VLT/MUSE 4×700s observations, May 2023, Prog. ID 0111.B-0400(H). Plus HST archival.

**Method.** Spectroscopic redshift via Ca H&K (lens) and Hγ, Hδ, [O II] (sources). GIGA-Lens modeling with two SIE profiles (main lens + secondary L2 at z=0.386).

**Result.** Lens at z_L = 0.630 ± 0.001; source at z_S = 2.597 ± 0.001. Einstein radius θ_E = 2.52″; velocity dispersion σ_v = 379 ± 2 km/s.

**Takeaway.** First single-system showcase combining spectroscopy + GIGA-Lens modeling on real (non-simulated) data, well before Foundry I formalized this combination.

## A.7 Sheu, Huang et al. 2024a — variable lensed quasars (arXiv:2408.02670)

**Question.** Adapt the Sheu 2023 SN difference-imaging pipeline to find variable lensed quasars.

**Dataset.** Same 5,807 strong-lens candidates plus the known lensed-quasar candidates (D22, H23).

**Method.** B08/SFFT difference-imaging. Selection by location (core of source), color (blue/white), variability (σ-threshold on per-epoch photometry), and PSF photometry via SEP.

**Result.** 13 new variable lensed quasar candidates, including 3 quads. Extends the total lensed-quasar candidate count to 655.

**Takeaway.** Time-domain finding complements the static morphological CNNs and the static autocorrelation method. The future LSST cadence makes time-domain methods particularly powerful.

## A.8 Sheu, Cikota, Huang et al. 2024b — Carousel cluster lens (arXiv:2408.10332)

**Question.** Detailed modeling of a cluster-scale lens with multiple sources.

**Dataset.** VLT/MUSE 4×700s in WFM-NOAO-N. Plus HST F140W/F200LP archival.

**Method.** MUSE spectroscopy: Ca H&K (lens galaxies), [O II] doublet (sources). GIGA-Lens modeling with multiple source planes.

**Result.** Cluster at z_L = 0.49 with **seven** spectroscopically confirmed lensed sources at z_S = 0.962, 0.962, 1.160, 1.432, 1.432, plus predicted z_S ≈ 4.5 source. Einstein radius θ_E = 2.52″; enclosed mass M(<θ_E) = 4.78 × 10¹³ M_⊙.

**Takeaway.** The Carousel is the showcase cluster-scale system. It also illustrates GIGA-Lens's capacity for multi-source modeling — directly relevant to cluster-scale H₀ and dark-matter tests.

## A.9 Storfer, Huang et al. 2024 — DR9 lens search (arXiv:2309.18089)

**Question.** Apply the H21 pipeline to DR9 (~19,000 deg²) and document the visual-inspection grading rubric.

**Dataset.** 1,961 known lenses (from H20 + H21 + literature) + ~64,000 validation/test non-lenses. 100:1 non-lens-to-lens ratio.

**Method.** "Shielded" ResNet from H21. Batch 128, learning rate 5×10⁻⁴ with 1/5 decay every 40 epochs.

**Result.** 1,895 candidates in DR9 (115 A, 526 B, 1,254 C). Total H20+H21+S24 = 3,057.

**Takeaway.** Establishes the visual-inspection grading framework (A/B/C/D) that subsequent papers use. Sets the baseline for the dual-architecture extension in Inchausti 2025.

## A.10 Inchausti, Storfer, Huang et al. 2025 — DR10 dual architectures (arXiv:2508.20089)

**Question.** Extend to DR10 with two architectures and a learned meta-learner over them.

**Dataset.** DR10 ~19,000 deg² with reprocessed DECam south of −18°. 1,372 known lenses (~869 from H20–S24). 134,182 contamination-removed non-lenses.

**Method.** ResNet "shielded" (194,433 params) + EfficientNetV2 pretrained and fine-tuned (20.5M params). 300-node neural meta-learner over the two output probabilities. Trained on NERSC Perlmutter, 4 GPU nodes. Threshold set at p ≥ 0.9367 (top 0.01 percentile).

**Result.** 811 new candidates from DR10. AUC: ResNet 0.9984, EfficientNet 0.9987, meta-learner 0.9989. Combined cumulative I–IV = 3,868. 20 of 51 HST-SNAP-imaged candidates are confirmed strong lenses in the EDR.

**Takeaway.** First multi-architecture ensemble in the program. The 0.9984 → 0.9989 AUC jump is marginal but materially reduces false positives at the operating threshold. AutoFoundry (§9.2) is the natural next step.

## A.11 Hsu, Huang et al. 2025 — pairwise spectroscopic search (arXiv:2509.16033)

**Question.** Can a *spectroscopic* method find lenses missed by imaging searches?

**Dataset.** 28M DESI DR1 spectra; 11,848 fiber-pair/triplet/quartet groups after FoF clustering; 26,621 spectra visually inspected.

**Method.** Pre-filter (ZWARN flags, target type, longest exposure per coadd) → FoF clustering with 3″ linking length → redshift-ratio filter (z_max / z_min ≥ 1.3 to ensure significant differences) → manual visual inspection with three quality grades (High / Moderate / Reject) → cross-match with DR10 imaging.

**Result.** 2,046 conventional lens candidates (1,906 new). 318 "dimple lens" candidates — a new class with low-mass (M_halo ≲ 10¹³ M_⊙) foreground galaxies producing surface-brightness indentations. Total 2,164 new candidates. Largest spectroscopically-identified lens-candidate sample to date.

**Takeaway.** The fifth discovery modality, and the one that opens a new science target (dimple lenses → dwarf-galaxy CDM tests). The pairwise-spectroscopic method is also the downstream-validation task for SpectrumFM. DimpleScout (§9.3) is the natural agentic next step.

## A.12 Silver, Wang, Huang et al. 2025 — ML-driven JWST forecasts (arXiv:2207.09431)

**Question.** Can ML push lens detection to θ_E ~ 0.03″ and M_halo ~ 10¹¹ M_⊙?

**Dataset.** Simulated JWST observations using VELA hydrodynamical simulations + Cosmodc2 lens/source catalogs. HST archival images for validation.

**Method.** ResNet for lens classification + U-Net for subhalo localization. Three-part pipeline: (1) JWST detectability forecasts, (2) hydrodynamical-simulation training, (3) U-Net localization.

**Result.** Forecasts O(17/deg²) JWST detectable lenses at θ_E ~ 0.03″, M_halo ~ 10¹¹ M_⊙. Two new HST strong-lens candidates discovered (missed by Garvin et al. 2022 crowdsourced classification).

**Takeaway.** Positions the program for JWST-scale follow-up at the CDM-substructure frontier. "Superhuman" lens detection is demonstrated on archival data.

## A.13 Huang et al. 2025a — DESI Strong Lens Foundry I (arXiv:2502.03455)

**Question.** Confirm and model strong-lens candidates with HST imaging + GIGA-Lens modeling.

**Dataset.** HST SNAP GO-15867 (PI: Huang); 51 successful WFC3/F140W observations from 112 submitted candidates. Plus DESI + Keck NIRES spectroscopy.

**Method.** HST imaging confirmation. First multi-GPU forward-modeling Bayesian fit (GIGA-Lens) to HST data: demonstration on DESI-165.4754−06.0423, with lens z_L = 0.4461, source z_S = 0.8100, σ_v = 152 ± 100 km/s.

**Result.** All 51 HST targets confirmed as strong lenses. ~3,500 cumulative candidates from the Legacy Surveys.

**Takeaway.** The headline confirmation paper for the Foundry series. The Bayesian-modeling demonstration sets the template for what GIGA-LensAgent (§9.5) needs to automate at scale.

## A.14 Huang et al. 2025b — DESI Strong Lens Foundry II (arXiv:2509.18089)

**Question.** Spectroscopic follow-up of strong-lens candidates with DESI itself.

**Dataset.** ~2,157 candidate systems from H20 + H21 + S24, targeted via the DESI Strong Lensing Secondary Target Program. EDR: 73 unique systems.

**Method.** DESI multi-fiber spectroscopy. Redshift determination via Redrock + visual inspection. FastSpecFit for velocity dispersions.

**Result.** 73 candidates in EDR: 20 confirmed strong lenses, 34 pending source confirmation, 4 confirmed not lenses. ~30% of candidates have source redshifts beyond DESI's optical range → escalate to Foundry III (Keck NIRES).

**Takeaway.** Confirms that DESI spectroscopy is the cheapest first-pass for lens confirmation, and bridges naturally to NIR follow-up (Foundry III) for high-z sources.

## A.15 Agarwal, Huang et al. 2025 — DESI Strong Lens Foundry III (arXiv:2501.08066)

**Question.** Near-IR spectroscopic follow-up of high-z source galaxies whose optical features have redshifted out of DESI's range.

**Dataset.** 8 candidate lens systems whose source redshifts are inaccessible to DESI optical.

**Method.** Keck-2 NIRES cross-dispersed echellette (0.94–2.45 μm, R ~ 2,700). PyPelt reduction; emission-line fits.

**Result.** 8 systems with confirmed source redshifts z_s = 1.68–3.33. Quality grades Q_z = 1 (Robust), 2 (Probable), 3 (Possible).

**Takeaway.** Closes the source-redshift gap for high-z sources. Without it, ~30% of Foundry II candidates would remain unconfirmed.

## A.16 Lin, Toro Bertolla, Cikota, Huang et al. 2025 — DESI Strong Lens Foundry IV (arXiv:2509.18078)

**Question.** Integral-field spectroscopic characterization of the most interesting lens candidates.

**Dataset.** 75 candidates observed with VLT/MUSE in Wide-Field Mode.

**Method.** MUSE 60″×60″ IFU, R = 2,000–5,200. PyPelt-style reduction. Gaussian fits to emission lines for redshifts. Three-level quality grading (Robust / Probable / Possible).

**Result.** 48 systems with fully confirmed (lens + source) redshifts. 21 with source-only redshifts.

**Takeaway.** Integral-field spectroscopy enables 3D spectroscopic mapping of complex (cluster-scale, multi-source) systems and adds spatial-spectral richness beyond DESI/NIRES single-fiber data.

---

# Appendix B — Lens-equation derivation and GIGA-Lens math

## B.1 Deflection angle from a point mass

A photon passing a point mass M at impact parameter ξ experiences a deflection (in the small-angle limit) of α̂ = 4GM / (c² ξ). For an extended mass distribution with surface density Σ(ξ⃗), the deflection is the sum:

> α̂(ξ⃗) = (4G/c²) ∫ Σ(ξ⃗') (ξ⃗ − ξ⃗') / |ξ⃗ − ξ⃗'|² d²ξ'

(This is the gravitational analogue of the Coulomb-gauge potential in 2D electrostatics.)

## B.2 The scaled lens equation

Define dimensionless quantities. Let D_L, D_S, D_LS be angular diameter distances to lens, source, and from lens to source. The *critical surface density* is

> Σ_cr = c² / (4πG) × D_S / (D_L D_LS)

Define the *convergence* κ(θ⃗) = Σ(D_L θ⃗) / Σ_cr (dimensionless surface mass density). Define the *scaled deflection*

> α(θ⃗) = (D_LS / D_S) × α̂(D_L θ⃗)

Then the lens equation simplifies to the dimensionless form quoted in §3.2:

> β⃗ = θ⃗ − α(θ⃗)

## B.3 Lens potential

α is the gradient of a potential: α(θ⃗) = ∇ψ(θ⃗), where ψ satisfies ∇²ψ = 2κ. The shear γ is the trace-free part of the second derivative ∂²ψ/∂θ_i ∂θ_j.

## B.4 Magnification

The Jacobian of the lens equation is A = δ_ij − ∂α_i/∂θ_j = δ_ij(1 − κ) − γ_ij. The magnification is the inverse determinant:

> μ(θ⃗) = 1 / det A = 1 / [(1 − κ)² − γ²]

Critical curves are loci where det A = 0 (μ → ∞).

## B.5 Time delay (Fermat potential)

The arrival time at the observer for a ray through θ⃗ relative to the unperturbed path is

> t(θ⃗) = (1 + z_L) / c × D_L D_S / D_LS × [ ½ |θ⃗ − β⃗|² − ψ(θ⃗) ]

The first term is geometric (longer path = later arrival); the second is gravitational (Shapiro delay). The time delay between two images at θ⃗_A and θ⃗_B is Δt_AB = t(θ⃗_A) − t(θ⃗_B), measurable directly for variable sources. Combined with a lens model, this yields D_Δt ∝ 1/H₀ — Refsdal's method.

## B.6 GIGA-Lens EPL surface mass density

For the elliptical power-law mass model used in GIGA-Lens (Gu 2022, eq. 1):

> κ(x_lens, y_lens) = ( (3 − γ_epl) / 2 ) × ( θ_E / √(q x_lens² + y_lens²/q) )^(γ_epl − 1)

with axial ratio q, mass slope γ_epl (= 2 for SIE), and aligned along the lens-light major axis after a rotation by position angle φ. External shear is added as a contribution to the deflection α_ext(x, y) = (γ_ext,1 x + γ_ext,2 y, γ_ext,2 x − γ_ext,1 y).

The Sérsic source profile is

> I(x_light, y_light) = I₀ exp(−b_n × [( √(q x_light² + y_light²/q) / R_eff )^(1/n) − 1])

with b_n = 1.9992n − 0.3271 (an approximation to the Sérsic-index normalization).

## B.7 Three-stage inference flow

GIGA-Lens runs three stages back-to-back:

1. **Multi-start gradient descent.** Initialize N candidate solutions across prior support, take Adam steps minimizing −log π(Θ | data). Keep the K best modes. Identifies the global posterior mode region; required because the posterior is non-convex.
2. **Variational Inference.** Approximate π by a multivariate Gaussian q_φ centered at the best mode, with full covariance learned by minimizing KL(q_φ || π) via the reparameterization trick. Yields a Gaussian-approximate posterior covariance.
3. **Hamiltonian Monte Carlo (NUTS).** Initialize from the VI posterior. Run NUTS with the gradient ∇ log π provided by automatic differentiation. Adaptive step-size and trajectory-length tuning. Returns posterior samples.

Wall-clock: ~105 s per system on 4 A100 GPUs (Gu 2022).

---

# Appendix C — Verified agentic-research citation table

| Project | URL | What it does | Verified |
|---|---|---|---|
| Karpathy autoresearch | https://github.com/karpathy/autoresearch | Single-GPU autonomous ML research loop; agent edits train.py | 2026-05-25 |
| Sakana AI Scientist v2 (paper) | https://arxiv.org/abs/2504.08066 | Agentic tree-search for full research lifecycle | 2026-05-25 |
| Sakana AI Scientist v2 (repo) | https://github.com/SakanaAI/AI-Scientist-v2 | Open-source impl. of v2 | 2026-05-25 |
| FutureHouse PaperQA2 | https://github.com/Future-House/paper-qa | High-accuracy RAG for scientific literature | 2026-05-25 |
| FutureHouse Aviary (paper) | https://arxiv.org/abs/2412.21154 | Gymnasium for scientific language agents | 2026-05-25 |
| FutureHouse Aviary (env) | https://github.com/Future-House/aviary | Aviary environments | 2026-05-25 |
| FutureHouse Aviary (agent) | https://github.com/Future-House/ldp | Agent code | 2026-05-25 |
| Google AI Co-Scientist (blog) | https://deepmind.google/blog/co-scientist-a-multi-agent-ai-partner-to-accelerate-research/ | 5-agent generate-debate-evolve; Gemini 2.0 | 2026-05-25 |
| Google AI Co-Scientist (paper) | https://arxiv.org/abs/2502.18864 | Towards an AI co-scientist | 2026-05-25 |
| Anthropic multi-agent research system | https://www.anthropic.com/engineering/built-multi-agent-research-system | Orchestrator-worker; ~90% time reduction | 2026-05-25 |
| Claude Code | https://claude.com/product/claude-code | Agentic IDE/coding tool | 2026-05-25 |
| AutoGen (legacy) | https://github.com/microsoft/autogen | Conversational multi-agent framework | 2026-05-25 |
| AG2 (community) | https://github.com/ag2-Ai/ag2 | Active continuation of AutoGen | 2026-05-25 |
| LangGraph | https://langchain-ai.github.io/langgraph/ | Stateful cyclic-graph orchestration | 2026-05-25 |
| Denario (repo) | https://github.com/AstroPilot-AI/Denario | Multi-agent astronomy assistant; AG2+LangGraph | 2026-05-25 |
| Denario (paper) | https://arxiv.org/abs/2510.26887 | Deep-knowledge AI agents for scientific discovery | 2026-05-25 |
| AstroLLM | https://astrollm.org/ | Retrieval-grounded astronomy LLM | 2026-05-25 |
| NASA-IBM INDUS | https://science.nasa.gov/open-science/ai-language-model-science-research/ | Scientific-corpus-fine-tuned LLMs | 2026-05-25 |
| RE-Bench | https://arxiv.org/abs/2411.15114 | 7 ML R&D tasks, 71 human attempts | 2026-05-25 |
| MLE-bench | https://arxiv.org/abs/2410.07095 | 75 Kaggle ML competitions | 2026-05-25 |
| ICLR 2025 Agentic-AI-for-Science | https://iclragenticai.github.io/ | Workshop where Sakana v2 passed peer review | 2026-05-25 |

---

*End of report. Comments, corrections, and proposals for additional sections welcome.*
