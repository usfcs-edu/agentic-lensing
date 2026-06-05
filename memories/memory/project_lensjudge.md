---
name: project-lensjudge
description: "LensJudge — agentic lens-candidate VI-grading system built on the Claude Agent SDK; renamed from the onboarding plan's 'LensAgent' to avoid an arXiv name clash; staged lean->robust build under reproductions/lensjudge/"
metadata: 
  node_type: memory
  type: project
  originSessionId: 158a07c5-5b95-480b-b8f4-09a9b3caf60c
---

**LensJudge** is the agentic system that automates the human visual-inspection (VI)
grading (A/B/C/D) of strong-lens candidates from the Huang group's ResNet/EfficientNet
finders, built with the **Claude Agent SDK for Python** (`pip install claude-agent-sdk`).
It is the system the onboarding report calls **"LensAgent" (§9.1)** / **"VI Pre-grading
Agent" (DES-Y6 memo §6.2.iv)**.

**Why the name LensJudge:** the user (2026-06-03) requires it NOT be called "LensAgent"
because **a paper titled "LensAgent" already exists on arXiv**. Always use **LensJudge**
for our system; keep "LensAgent" only as a citation to the onboarding-plan §9.1 design.

**How to apply:** code/dirs live at `reproductions/lensjudge/`; venv `.venvs/lensjudge`
(Python 3.13; the SDK rides on the installed `claude` CLI v2.1.x for auth, so no
ANTHROPIC_API_KEY needed). Plan file: `~/.claude/plans/i-want-to-continue-expressive-honey.md`.

**Build (user-approved 2026-06-03):** staged — (1) lean single multimodal grader
baseline + shared LensBench-VI eval harness, (2) upgrade to CNN-gated perspective-diverse
judge panel + confidence-gated human-in-loop, (3) spectroscopic VI pre-grader (Hsu
discordant-z pairs / Foundry-II), with the §9.1 multi-agent factor-decomposition as an
ablation arm. Both imaging AND spectroscopic modalities.

**Key SDK facts:** vision is TOOL-ONLY (a `fetch_cutout` tool must render FITS->PNG and
return an image content block; no image-in-prompt); Python `@tool` `structuredContent` is
NOT forwarded -> enforce JSON via system prompt + parse `ResultMessage` with Pydantic + 1
repair retry; subagents via `AgentDefinition` (isolated context, static tools, can't nest);
hooks + `ResultMessage.total_cost_usd` = the eval harness.

**Data verdict (from the inventory sweep):** imaging is fully testable today — ~2,706
graded A/B/C cutouts on disk (Storfer+Inchausti) + per-candidate ML scores, 2,769 Grade-D
human-rejected hard negatives (RA/Dec only, fetch via brick-slice), ~73K random negatives,
Silver 4,000 balanced sims, confirmed gold (Foundry/Cikota/Carousel). Two gaps: (a) NO
multi-grader labels exist anywhere -> eval is consensus+gold-referenced, every kappa
labeled "no human ceiling" (a ~250-cutout 2-3 grader study is the noted future fix); (b)
spectroscopic needs DESI fiber FLUX streamed by TARGETID (only redshift catalog + sigma_v
are local). Reuses [[project-lensing-repro-sprint-2026-06]] assets (inchausti-2025 renderer
`16_build_inspection_viewer.py`, brick fetch `20_build_negatives_brick_dr9.py`, `_scorelib.py`).

**Status — BUILT + EVALUATED (2026-06-03):** full system at `reproductions/lensjudge/`
(substrate, MCP tool bundle, 4 graders, LensBench-VI harness, README, tech-report
`papers/main.pdf` 4pp). SDK `claude_agent_sdk 0.2.89`; vision works (tool returns image
blocks); `permission_mode="bypassPermissions"`; all 22 modules import clean. Cost/candidate:
lean Sonnet $0.06, lean Opus $0.13, panel $0.25, multiagent $0.20.

**KEY RESULTS (honest):** (1) **Spectroscopy strong** — 20/20 Hsu Table-2 Grade-A pairs
graded plausible, 4/4 Foundry-II non-lenses rejected (clean physical features + confirmed
gold). (2) **Imaging near-chance vs single-consensus labels, and MORE COMPUTE DOESN'T HELP**
— on the CNN's hard high-p_meta pool, binary AUC ≈0.34–0.51 and QWK≈0 for ALL of
{lean-Sonnet, lean-Opus, panel, multiagent}; all uniformly skeptical (p_lens≈0.02–0.03 even
on consensus-A). Verified NOT a bug (cutouts real + correctly resolved; obvious arcs DO get
high p_lens; the n=16 sanity AUC 0.85 was an unrepresentative easy sample). (3) **It's a
measurement problem, CONFIRMED by a physical check** — added the Foundry-I "does it model
as a lens?" criterion as `quick_lensmodel` (real GIGA-Lens MAP fit on grz, ~45s/cutout,
subprocesses into .venvs/gigalens, JAX-free SDK; `tools/quicklens.py`+`outputs/quicklens_proto.py`;
agent arm `run_batch --mode lean --modelability`; ablation `eval/run_modelability.py`). Its
lens_score separates real lenses from ORDINARY galaxies (AUC ~0.77) but NOT from hard Grade-D
human-rejects (AUC ~0.44) — same wall as vision/Opus/panel/multiagent. So the imaging
criteria the agents use are Huang-2020 (discovery VI) for imaging + Hsu-2025/physics for
spectro; the Foundry-I modelability criterion is now wired too. A/B/C are mostly *unconfirmed*
candidates and the C/D boundary is one rater's subjective call, so agent-vs-consensus on
imaging is **unadjudicable** without
multi-grader/spectroscopic truth. The no-human-ceiling gap is THE finding, not a footnote.

**RepresentationKit (2026-06-04):** built engineered image transforms that make lensing
signal explicit — `common/representations.py` (Tier-1 scipy/numpy: lens-light subtraction,
polar/tangential, 180° symmetry, blue-excess color-iso, DoG+Hessian arcness; +blue-coupled
variants), `tools/representations.py` (`lens_representations` @tool: scalars-first JSON + 5
view images, in-process), `tools/representations_proto.py` (Tier-2 in .venvs/lens: photutils
isophote, skimage frangi, SEP geometry), `eval/run_representations.py` (gate, bootstrap CIs)
+ `eval/run_representations_tier2.py`, `run_batch --representations` arm, rubric section.
**RESULT (decisive, closes the question):** the SAME features score **~0.80 on confirmed
GOLD labels and ~0.80 on noiseless SILVER sims** (so signal extraction WORKS) but **~0.51 on
the hard consensus pool**; cleaner Tier-2 ≤0.57; agent+representations 0.40 vs lean 0.44
(+43% cost). Easy-regime pre-screen ~0.70 (~50ms vs 45s GIGA-Lens). So vision/Opus/panel/
multiagent/GIGA-Lens/Tier-1-features/Tier-2-features/agent+features ALL hit the same wall on
the soft Grade-D pool → the wall is a **label/data limit, not an algorithm limit**, definitively.
Next: the ~250-cutout 2–3-grader study; push agentic effort to the spectroscopic channel.

**External validation — BETTER DATA BREAKS THE WALL (2026-06-04):** found + tested the "better
data / better labels" the report flagged as future work. Network from the sandbox reaches Zenodo /
MAST / VizieR / IRSA (Python DNS works; IRSA was a transient blip). Pulled **Euclid Q1 Strong
Lensing Discovery Engine** (Zenodo 10.5281/zenodo.15025832, ~3.6GB → `reproductions/euclid-q1/data/`,
gitignored; tracked `euclid-q1/README.md`): 2,584 candidates (A309/B267/C2008) with `expert_score`
+ `grade` + **`expert_total_votes`~10** (a multi-grader 0.1" catalog); 539 objects ship 4-band
**VIS+NIR Y/J/H FITS @ 0.1"/px (300×300, 30")** + full SIE/MGE/Sérsic models. `unsuccess/`=modeling
failed NOT lens-rejected → ALL 539 are positives (no released non-lens cutouts). Built the Euclid
arm: `common/euclid.py` (loader + RGB/VIS/vis_sub renderers), `tools/euclid_cutout.py`
(`fetch_euclid_cutout`, registered in server._lazy), `eval/run_euclid.py` (modes paired|rank),
`eval/crossmatch_external.py` (master table + Euclid xmatch). **Crossmatch (4,354 unique cands →
291 have external high-res / multi-grader / confirmation; 89 are DESI grade-C):** SuGOHI/HSC 104
(0.6", ~9-grader committee + spec-z), archival HST/MAST 148 of 1414 grade-A+B (0.05-0.1"; 99/481
grade-A, 49/933 grade-B), AGEL DR2 77
Keck-confirmed, Euclid Q1 24, BELLS-Gallery/CASTLES/SpaceWarps 8 (ran as a 3-agent Workflow:
`lensjudge-external-xmatch`). **RESULTS:** (1) 53% of DESI grade-C cands in Euclid Q1 are graded
A/B by 10 experts (6/17 → A) — "C" = unresolved, not weak. (2) **Paired within-object** (same 10
objects, DESI 1.3" vs Euclid 0.1", lean Sonnet): mean p_lens 0.14→0.75 (up 9/10); **DESI grade-C
subset 0.05→0.90** — the agent rejects them as blobs at ground res, grades them clear Einstein
rings at 0.1" (montage `outputs/euclid_grade_flip_montage.png`). DETECTION wall is resolution+label
limited, NOT algorithm. (3) But fine A/B/C grade stays ~chance even at 0.1"/10-votes (A-vs-C AUC
0.51, Spearman 0.08) — the confidence grade is irreducibly soft; replace it with binary
detect/escalate anchored to confirmed/high-res truth. Per-rater κ still needs the in-house
~250-cutout study (only consensus is publicly downloadable; per-classifier votes = Zooniscience).
Tech report now 7pp (§"External validation"); IRSA random-galaxy fetch = path to a field-realistic
0.1" lens-vs-nonlens AUC (not done).

**Human ceiling — ESTIMATED, two ways (2026-06-04, `eval/human_ceiling.py`):** the "missing human
ceiling" caveat is now quantified. Per-classifier raw votes are NOT public anywhere (Tier-2 workflow
`lensjudge-human-ceiling-tier2` verified: Space Warps releases only skill-weighted SWAP posteriors;
SuGOHI/Euclid only aggregated consensus). So (a) **inter-team QWK** on crossmatched objects: DESI vs
SuGOHI ~9-grader committee **0.29** [0.12-0.44] n=103; DESI vs Euclid ~10-expert panel 0.17 n=24 —
independent expert teams agree only "fairly". (b) **Literature** (`outputs/_ceiling_scratch/`):
Rojas+2023 (arXiv 2301.03670; 55 graders, 1489 imgs) 73% intra-rater exact / QWK~0.8 (upper bound),
>=6 graders needed; Petrillo+2019 only 4.5% of cands flagged by all 7; Cañameras+2021 30% zero-disp
among 5; Euclid Q1 (Walmsley+2025 arXiv 2503.15324) 10-expert panel decisive on only ~38%. → A/B/C
is intrinsically SOFT. (c) **Agent vs ceiling:** graded all 104 SuGOHI-matched at DESI 1.3" (lean
Sonnet); κ(agent,DESI)=**0.003** n=104 — BELOW the human ceiling, essentially uncorrelated:
over-skeptical (66/104 graded D, 0 A on HSC-confirmed cands b/c arcs unresolved). Reaches human level
only with resolution (Euclid paired). So 3 limits: fine grade barely reproducible by anyone (κ~0.3); agent underperforms
even that at ground res; high-res closes the gap for DETECTION. Bib + §"The human ceiling is low" added
(report now 8pp). NOTE: the SuGOHI grading first stalled on slow ls-dr9 legacysurvey fetches → FIXED by adding
`MAX_FETCH_WALL=120s` total wall-clock cap per candidate in common/fetch.py; resumed to full 104/104. Refs added: rojas2023vi, euclidq1engine,
petrillo2019links, canameras2021holismokes, sonnenfeld2018sugohi.

Related: [[project-huang-lensing]], [[project-lensing-repro-sprint-2026-06]],
[[project-inchausti-2025-reproduction]], [[reference-host-hardware]].
