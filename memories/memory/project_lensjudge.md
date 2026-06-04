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

Related: [[project-huang-lensing]], [[project-lensing-repro-sprint-2026-06]],
[[project-inchausti-2025-reproduction]], [[reference-host-hardware]].
