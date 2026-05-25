# Agentic Lensing Onboarding Report — Execution Plan

## Context

Prof. Greg Benson (CS / agentic-systems expertise) is joining Prof. Xiaosheng Huang's strong-gravitational-lensing ML research group. Benson is already named Co-I on the DOE Genesis Mission **SpectrumFM** proposal at USF, where he **leads agentic-AI tooling**. He needs a single, self-contained Word document that brings him up to speed on:

1. The problems Huang's program addresses, the physics/math/ML behind them, and the lineage of his 16 papers.
2. The SpectrumFM proposal's goals, architecture, and connection to lensing.
3. The current (May 2026) state of agentic / "AI scientist" research methods.
4. Concrete proposals for applying agentic-AI techniques to both the existing lens-discovery pipeline and the SpectrumFM work.

Deliverable: a comprehensive (~30-50 page) Word document at `plans/agentic_lensing_onboarding_plan.docx`, with verified citations, hyperlinks, and concrete agentic-architecture sketches.

The Phase 1 Explore agents have already produced detailed structured summaries of (a) the methodology-lineage papers (Huang_2020 → Huang_2025b, GIGA-Lens), (b) the discovery/follow-up papers (Sheu, Cikota, Dawes, Hsu, Silver, Lin, Agarwal), and (c) the SpectrumFM proposal + agentic-research landscape. Execution now proceeds to verification, synthesis, and Word-document generation.

---

## Approach (high-level)

1. **Verify agentic-landscape citations** with WebFetch (per user choice). The agent-supplied landscape included 2025-2026 dates that may include fabrications; every URL/claim needs grounding before it goes into the doc.
2. **Re-read 4-6 pivotal source PDFs directly** to confirm Phase-1 summaries on the most important methodological claims (don't trust agent summaries blindly for the load-bearing parts).
3. **Draft the Word doc as Markdown first**, then convert via `python-docx` (preferred — gives full styling control) or Pandoc as fallback.
4. **Generate the .docx**, sanity-check by opening it, and place at `plans/agentic_lensing_onboarding_plan.docx`.
5. **Save memory entries** about Benson's role (agentic-AI tooling Co-I on SpectrumFM), the project context, and useful references.

---

## Document Structure (target Table of Contents)

The doc is built in three layers: tutorial background (sections 1-4), program survey (sections 5-7), and agentic synthesis (sections 8-10). Appendix holds per-paper deep dives.

### Front matter
- Title, author (Greg Benson), date, abstract (~150 words).
- Reader's-guide one-pager: "How to read this — start with §1 for the program at a glance; jump to §8-9 if you already know lensing."

### 1. Executive Summary (1-2 pp)
Three-pillar framing of Huang's program: **Discover** (ML lens-finders), **Characterize** (HST/Keck/VLT spectroscopy + GIGA-Lens modeling), **Cosmologize** (H₀ via time delays, dark-matter substructure). Tee up SpectrumFM as the next foundation-model layer and agentic-AI as the integration fabric.

### 2. The Research Program at a Glance (3-4 pp)
- Map of the five discovery modalities, with one-line method + lead paper:
  - ResNet on imaging cutouts (Huang_2020, 2021)
  - ResNet + EfficientNet ensemble + meta-learner (Storfer_2024, Inchausti_2025)
  - Difference-imaging for variable transients (Sheu_2023 SNe, Sheu_2024a quasars)
  - Autocorrelation for lensed/binary quasars (Dawes_2022)
  - **Pairwise spectroscopic search** (Hsu_2025 — discovers "dimple lenses", a new low-mass-halo class)
- DESI Foundry papers I-IV: imaging (I), DESI spec (II), Keck NIRES (III), VLT/MUSE (IV).
- Timeline graphic: candidates discovered cumulatively (335 → 1545 → 3057 → 3868 → ~5500 including spectroscopic).
- Key collaborators and instruments (DESI, HST GO-15867, Keck-2 NIRES, VLT/MUSE, NERSC Perlmutter).

### 3. Physics Background — Gravitational Lensing for Computer Scientists (4-6 pp)
Written for a CS reader: build intuition from the geodesic-deflection picture, then layer in equations.
- 3.1 What lensing is: light deflection by mass; weak vs strong vs micro
- 3.2 The thin-lens approximation and the lens equation: β = θ − α(θ)
- 3.3 Einstein radius θ_E and how it scales with mass and redshift
- 3.4 Image multiplicity (doubles, quads, Einstein crosses, Einstein rings, cluster-scale arcs)
- 3.5 Magnification, parity, and time delay
- 3.6 Why lenses matter: H₀ via time-delay cosmography (Refsdal 1964; H0LiCOW; TDCOSMO); dark-matter substructure detection (subhalo-induced perturbations)
- 3.7 Glossary: convergence κ, shear γ, critical/caustic curves
- "Go deeper" links: Bartelmann & Schneider 2001 review, Saas-Fee lecture notes, lenstronomy docs, Treu 2010 ARA&A review.

### 4. Mathematical & ML Foundations (5-7 pp)
- 4.1 Mass models: SIS, SIE, EPL (the one GIGA-Lens uses), NFW, generic multipole expansions
- 4.2 Light models: Sérsic profile, pixelated source reconstruction
- 4.3 Bayesian inference for lens parameters: prior choice, likelihood, posterior shape
- 4.4 Sampling: multi-start gradient descent (initialization), Variational Inference for covariance, Hamiltonian Monte Carlo (NUTS) for the posterior — exactly the GIGA-Lens stack
- 4.5 Differentiable forward modeling: why JAX/TensorFlow change everything (gradients through ray-tracing)
- 4.6 CNN/ResNet primer for the astro context (Lanusse et al. 2018 → Huang adaptation)
- 4.7 EfficientNetV2 and ensemble meta-learners
- 4.8 U-Net for subhalo localization (Silver_2025)
- 4.9 Transformer-based foundation models for spectra (SpectrumFM context: AION-1, AstroCLIP, others)
- "Go deeper" links: Birrer & Amara lenstronomy paper, MCMC textbook (Betancourt's NUTS), Foundation Models For Science survey.

### 5. Paper-by-Paper Walkthrough — Themed (8-10 pp main body, full per-paper detail in Appendix A)
Theme groupings (main body summarizes each theme; per-paper detail in appendix):
- 5.1 **Discovery lineage** (Huang_2020, Huang_2021, Storfer_2024, Inchausti_2025) — methodological evolution, ROC-AUC progression, infrastructure
- 5.2 **Modeling tooling** (Gu_2022 GIGA-Lens, Silver_2025 forecasts + U-Net for substructure)
- 5.3 **Specialty discovery modes** (Sheu_2023 SNe, Sheu_2024a variable quasars, Dawes_2022 autocorrelation, Hsu_2025 pairwise-spectroscopic — including dimple lenses)
- 5.4 **Single-system science** (Cikota_2023 DESI-253 Einstein cross, Sheu_2024b "Carousel" cluster lens)
- 5.5 **The DESI Strong Lens Foundry** (Huang_2025a/b, Agarwal_2025 III, Lin_2025 IV) — the integrated confirmation pipeline

### 6. The SpectrumFM Proposal (4-5 pp)
- 6.1 The problem with Redrock: hand-engineered templates don't scale to new target classes (LAEs, LBGs in DESI-II, future Spec-S5)
- 6.2 One model, six classes — the foundation-model thesis
- 6.3 Architecture: transformer encoder/decoder; auxiliary redshift head as the key differentiator vs AION-1
- 6.4 Training data: ~60M DESI spectra (~50M extragalactic + ~10M stellar)
- 6.5 Three-stage pipeline: Masked Spectrum Modeling → SFT + preference learning on VI data → few-shot extensibility
- 6.6 Downstream validation: **strong-lens identification** (via Hsu_2025 pairwise method) + SN typing
- 6.7 Compute: Perlmutter A100s → Doudna (NERSC-10) Phase II
- 6.8 Phase-I go/no-go metrics, team composition (highlight Benson's agentic-tooling lead role), budget summary
- 6.9 Phase-II vision: multimodal extension to LSST/Roman/Euclid imaging + astrometry

### 7. Datasets, Instruments, and Computing Infrastructure (2-3 pp)
- DESI Legacy Imaging Surveys (DECaLS, BASS, MzLS, DR9/10): pixel scale, depth, bands
- DESI spectroscopy (Mayall/Kitt Peak, 4m, 5000-fiber)
- Follow-up: HST WFC3/F140W, Keck/NIRES NIR, VLT/MUSE IFU
- NERSC Perlmutter A100 partition; future Doudna
- Software stack: TensorFlow, JAX, lenstronomy, Astropy, Pandas, PyTorch

### 8. Agentic-Research Landscape, May 2026 (5-7 pp)
**All claims verified via WebFetch before inclusion.** Organized by what they automate, not by vendor:
- 8.1 Code/experiment loops: **Karpathy autoresearch** (5-minute loop, single GPU, direct AST edits); **Sakana AI Scientist v1 → v2** (agentic tree search, ICLR-2025-workshop acceptance)
- 8.2 Literature/QA: **FutureHouse PaperQA2, Crow, ChemCrow, Aviary**
- 8.3 Hypothesis generation & evolution: **Google AI Co-Scientist** (generate-debate-evolve loop, tournament-Elo)
- 8.4 General research orchestration: **OpenAI Deep Research**, **Anthropic Claude / Computer Use / Claude Code**, **AutoGen-AG2**, **LangGraph**
- 8.5 Astronomy-specific: **Denario / CMBAgent** (NeurIPS 2025 FairUniverse winner; built on AG2 + LangGraph), **AstroLLM**, **NASA-IBM INDUS / Surya**
- 8.6 Benchmarks: **RE-Bench**, **MLE-bench**, **FML-Bench**; ICLR 2025 *Agentic AI for Science* workshop
- 8.7 Cross-cutting patterns: orchestrator-worker, generate-debate-evolve, tool-use specialization, human alignment (SFT + preference + rationale), open-source vs commercial economics

### 9. Proposed Agentic-AI Applications to the Lensing Program (8-12 pp) — **CONCRETE ARCHITECTURES**

For each, provide: (a) what it automates, (b) why it fits this program, (c) architecture diagram + role decomposition, (d) tool surface, (e) suggested orchestration framework, (f) success metrics, (g) prototype effort estimate, (h) risks.

- **9.1 LensAgent — Multi-agent candidate-triage orchestrator.** Replaces (or augments) human visual-grading of ResNet+EfficientNet outputs. Roles: vision-model agent, morphology-checker agent, color/photometry agent, prior-catalog cross-matcher agent, grade-arbitrator agent. Framework: AG2 + LangGraph (Denario pattern). Tool surface: Tractor catalog API, lenstronomy quick-model, DESI cutout server.

- **9.2 AutoFoundry — Autoresearch-style training-loop driver for next-gen finders.** Karpathy-style autoresearch agent that iterates on `train.py` for the dual-architecture (ResNet + EfficientNetV2) meta-learner, scoped to NERSC-Perlmutter 30-minute budgets. Tracks every experiment in a Git ledger; promotes winners through a tournament. Hooks into the existing Inchausti_2025 codebase.

- **9.3 DimpleScout — Hypothesis-driven spectroscopic search agent.** Targets the Hsu_2025 "dimple lens" class. Co-Scientist–style generate-debate-evolve loop proposes selection-criterion refinements over DESI fiber pairs; runs lightweight lens-modeling tests; ranks candidates for VI. Tool surface: DESI DR1+ spec server, spherematch FoF, FastSpecFit velocity dispersions.

- **9.4 SpectrumFM Autopilot — Training-pipeline orchestrator for SpectrumFM.** Agent automates: data-curation QC, masked-spectrum-modeling pretraining sweeps, redshift-head ablations, scaling-curve fits, preference-pair construction from VI archives. Implements the Phase-I 9-month milestone plan as an autonomous loop with human checkpoints at month boundaries. Framework: Claude Code + Anthropic multi-agent orchestration; logs to NERSC's Genesis Mission data-platform.

- **9.5 GIGA-LensAgent — Bayesian-modeling agent for high-throughput lens characterization.** Wraps GIGA-Lens (Gu_2022) as a tool; agent selects priors, picks mass model, decides when to escalate to HMC vs stop at VI, writes per-system summary cards. Critical for scaling lens modeling from ~100 systems (current) to O(10⁴) in LSST era.

- **9.6 FoundryScribe — Manuscript-drafting agent for confirmation papers.** Sakana-v2-style writing agent that drafts confirmation-paper sections (object table, redshift table, lens-model summary, conclusions). Specialized for the Foundry-series template. Human stays in the loop for science interpretation.

- **9.7 LensLit — Literature-surveillance agent.** PaperQA2 + Crow pattern over the NASA ADS strong-lensing literature. Daily digest of preprints, automated detection of newly published lens systems that overlap the DESI footprint (cross-matchable to existing candidates). Reduces the "did we already know about this?" load on the team.

Then a synthesis pass: which 2-3 of these to prototype first (likely **LensLit** for fastest payoff, **DimpleScout** for highest scientific upside, **SpectrumFM Autopilot** as the strategic anchor since Benson already owns that piece).

### 10. Recommendations & Next Steps (2-3 pp)
- Tooling pick: Claude Code as IDE/coding agent, AG2 for multi-agent orchestration (per Denario precedent), LangGraph for stateful workflows, **MCP** for tool exposure to multiple LLMs.
- Evaluation: build a lens-finding mini-benchmark (RE-Bench inspired) using held-out HST-confirmed lenses; measure agent vs. human-grader agreement on a sample.
- First 90 days: prototype LensLit + a stub of SpectrumFM Autopilot; instrument with telemetry; report at the next group meeting.
- Open-source the agentic pipelines (Sakana / FutureHouse model) to build community visibility.
- Risk register: hallucinated lens classifications, compute-cost runaway, reliance on closed-model APIs, reproducibility of agent-driven results in publications.

### 11. Glossary (1-2 pp)
Lensing terms (Einstein radius, convergence, shear, caustic, time delay), ML terms (ResNet, EfficientNet, U-Net, transformer, VI, HMC), astronomy terms (LRG, ELG, QSO, LAE, LBG, VI campaign, FoF), DOE terms (Genesis Mission, NERSC, Perlmutter, Doudna).

### 12. References (4-6 pp)
- All 16 Huang-group papers with arXiv IDs/DOIs
- Background reviews (Treu 2010, Bartelmann & Schneider 2001, lenstronomy paper)
- Agentic-research papers and project URLs (all WebFetch-verified)
- Tool documentation (LangGraph, AG2, Claude Code, AION-1, etc.)

### Appendix A — Per-paper deep dives
One ~1-page write-up per paper for the 16 in the corpus, structured: science question, dataset, method, architecture, result, takeaway. Re-uses the structured material the Phase-1 agents produced.

### Appendix B — Lens-equation derivations & GIGA-Lens math
The full lens equation, magnification matrix, time-delay surface, and a worked example of the GIGA-Lens VI → HMC inference flow.

### Appendix C — Verified agentic-research citation table
Project → URL → what it does → last-verified date.

---

## Critical Files / Inputs

- **Papers** (16): `/Users/gbenson/sync/research/agentic-lensing/papers/*.pdf`
- **Proposal**: `/Users/gbenson/sync/research/agentic-lensing/proposals/doe_genesis_spectrumfm_project_narrative_v7.docx`
- **Phase 1 outputs** (already in conversation): methodology-papers summary, discovery-papers summary (parts A and B), SpectrumFM + landscape summary at `/Users/gbenson/.claude/projects/-Users-gbenson-sync-research-agentic-lensing/95b34e74-4971-42df-affe-07a9467ad83a/tool-results/toolu_01TSFcLBmxiDXbZc9nCyxN9X.json`
- **Output**: `/Users/gbenson/sync/research/agentic-lensing/plans/agentic_lensing_onboarding_plan.docx` (note: `plans/` dir does not yet exist under the project; will be created by `mkdir -p`)
- **Plan file (this file)**: `/Users/gbenson/.claude/plans/agentic-lensing-research-prompt-glowing-sundae.md`

---

## Execution Steps (after ExitPlanMode is approved)

1. **Verify agentic citations.** WebFetch each URL in the landscape (Karpathy autoresearch, Sakana AI Scientist v2 paper, FutureHouse, Denario GitHub, Google Co-Scientist blog, Anthropic multi-agent research-system blog, RE-Bench/MLE-bench arXiv, AstroLLM, NASA-IBM INDUS, Aviary paper). Drop or correct any that don't ground.
2. **Spot-check the load-bearing papers** by re-reading specific page ranges directly (not via agent): Huang_2025a (Foundry I abstract + GIGA-Lens application section), Huang_2025b (Foundry II abstract + EDR results table), Hsu_2025 (dimple-lens section + Einstein-radius formula + counts), Gu_2022 (GIGA-Lens architecture section), SpectrumFM proposal (Phase-I metrics + Benson's role).
3. **Choose docx generation route.** Default: install `python-docx` and write a generator script (`scripts/build_onboarding_docx.py`) that consumes a structured Markdown source. Fallback if install blocked: Pandoc `md → docx`.
4. **Author the Markdown source** in `plans/agentic_lensing_onboarding_plan.md` (intermediate, kept in repo for reproducibility). Hyperlinks as inline Markdown.
5. **Generate the .docx**: run the script, produce `plans/agentic_lensing_onboarding_plan.docx`. Embed a few figures (timeline of candidate counts; agentic-loop diagrams for §9). Use a simple sans-serif heading style; readable line spacing.
6. **Verification of the deliverable**:
   - Open the .docx with a programmatic sanity check (word count > 8,000; section count == 12 main + 3 appendices; references count > 50; all hyperlinks parse).
   - Manually inspect 3 random sections in the docx with `unzip -p ... word/document.xml | xmllint --xpath ...` or open in macOS Preview.
   - Confirm all 16 papers are cited and all 5 discovery modalities are covered.
7. **Save memory entries** (after exiting plan mode): user role (Benson is agentic-AI Co-I on SpectrumFM, not a junior onboarding student); project (Huang group at USF, DESI Strong Lens Foundry, SpectrumFM DOE Genesis); references (paper directory, proposal location); feedback (user wants deep + concrete agentic proposals + verified citations).

---

## Verification (how we'll know the doc works)

- **End-to-end smoke**: open the `.docx` in macOS Preview / Word and skim Table of Contents, executive summary, §9 (concrete proposals), and references.
- **Citation grounding**: every URL in §8/§9/§12 must have a "verified DATE" tag from a WebFetch in step 1.
- **Coverage**: a grep over the source Markdown finds the filename stem of all 16 papers and the SpectrumFM proposal.
- **Self-consistency**: the five discovery modalities listed in §2 each cite the lead paper introduced in §5/§A.
- **Reader test**: read the executive summary cold and decide whether it would be intelligible to a CS-trained reader with no astronomy background. If not, revise.

---

## Out of Scope (for this deliverable)

- Actually building any of the proposed agents (§9 is design only; prototyping comes later).
- Modifying any paper code, the SpectrumFM training scripts, or any GIGA-Lens internals.
- Producing slides, posters, or talk versions of the report (could be a follow-on).
- New scientific analysis on the lens corpus (the doc surveys; it does not introduce new findings).
