# LensJudge — agentic strong-lens candidate VI grader (Claude Agent SDK)

LensJudge automates the **human visual-inspection (VI) grading** (A/B/C/D) of strong
gravitational-lens candidates produced by the Huang group's ResNet/EfficientNet finders.
It is the system the onboarding report calls **"LensAgent" (§9.1)** / **"VI Pre-grading
Agent" (DES-Y6 memo §6.2.iv)** — renamed **LensJudge** to avoid a clash with the existing
*LensAgent* paper on arXiv — re-targeted onto the **Claude Agent SDK for Python**.

This is a research prototype + an evaluation suite (**LensBench-VI**) that answers
*"can we build and test such a system on the data we have?"* — **yes for imaging today**,
with two documented gaps (no multi-grader labels → consensus-referenced eval; DESI fiber
flux not local → spectroscopic grader runs on catalog features + imaging).

## Does the data support testing this? (the inventory verdict)
- **~2,706 human-graded A/B/C candidates** as 101×101×3 grz FITS on disk (Storfer DR9 +
  Inchausti DR10) with per-candidate CNN scores — in `../inchausti-2025/data/`.
- **2,345 Grade-D human-*rejected* hard negatives** (parsed from the published Google-Sheet
  exports; cutouts fetched on demand) + ~65K random-galaxy negatives on disk.
- **Foundry-II gold**: 21 confirmed/known lenses + 4 confirmed non-lenses (blind tier).
- **Hsu-2025**: 13,530 discordant-redshift fiber pairs + 20 Table-2 Grade-A (spectro).
- **Gaps**: (1) every grade is a *single consensus* — no per-rater labels, so all κ are
  **consensus-referenced, no human ceiling**; (2) DESI fiber **flux is not local**.

## Architecture (staged: lean → robust, + multi-agent ablation, + spectroscopic)
The Agent SDK has **no image-in-prompt**, so the only way the model sees pixels is the
`fetch_cutout` tool returning a rendered PNG image block. Three imaging graders share one
tool/eval substrate:

| Mode | What it is | Cost/candidate |
|---|---|---|
| `lean` | one multimodal call: cutout views + 5-criterion rubric + CNN scores → one `ImageGrade` JSON | ~$0.06 |
| `panel` (Option C) | 4 perspective-diverse judges (Advocate/Skeptic/Morphologist/Contaminant) in parallel → quorum + skeptic-veto + 2-of-N A-rule | ~$0.27 |
| `multiagent` (§9.1) | factor specialists (morphology/color/contaminant) → GradeArbitrator fuses | ~$0.33 |
| `spectro` | discordant-z pair: catalog features + SIS θ_E/sep check + DR10 imaging → `SpecGrade` (lens/dimple/not_lens) | ~$0.1 |

The lean grader is the baseline the others are measured against in the same harness.

## Layout
```
config.py            paths, survey layers, render params, model tiers, budgets
common/   render.py  FITS grz cube -> multi-view Lupton-RGB PNG (full/zoom/residual/highcontrast)
          fetch.py   cube acquisition: on-disk first, else legacysurvey fits-cutout endpoint (cached)
          schemas.py Pydantic ImageGrade / JudgeVote / SpecGrade / ReviewForm (lenient/clamping)
          parse.py   robust JSON-in-text -> Pydantic (the SDK returns free text)
          io.py       graded-candidate + Grade-D + Foundry-II gold loaders
          hooks.py    PreToolUse/PostToolUse/SubagentStop -> JSONL trace + cost (the eval substrate)
tools/    cutout.py   fetch_cutout  (the vision tool: returns image blocks)
          photometry.py get_photometry (aperture g-r/r-z colors from the cube)
          crossmatch.py crossmatch_local (prior published/confirmed lens overlap)
          spectrum.py  get_specfit (SIS θ_E/sep check) + fetch_spectrum (best-effort flux)
          quicklens.py quick_lensmodel — Foundry-I "does it model as a lens?" GIGA-Lens MAP
                       fit (subprocesses into .venvs/gigalens; JAX kept out of the SDK)
          server.py    create_sdk_mcp_server bundle + (mcp_servers, allowed_tools) builder
          # future (not built; server._lazy tolerates absence): cnn_score CNN pre-filter
eval/     run_modelability.py  GPU-sharded modelability ablation (lens_score AUC vs consensus)
imaging/  grader_lean.py  Phase-1 lean grader      run_batch.py  --mode {lean,panel,multiagent}
          judges.py + aggregate.py   Option-C panel
          orchestrator.py            §9.1 factor-decomposition
spectro/  grader.py + run.py         spectroscopic VI pre-grader
eval/     build_eval_set.py  frozen LensBench manifest    score.py / report.py  metrics + report
prompts/  rubric_imaging.md, advocate/skeptic/morphology/contaminant.md, rubric_spectro.md
```

## Setup
```bash
python3.13 -m venv ~/.venvs/lensjudge
~/.venvs/lensjudge/bin/pip install -r requirements.txt
```
The SDK rides on the installed `claude` CLI (≥2.1) for transport + auth — no
`ANTHROPIC_API_KEY` needed when the CLI is logged in. Run from the `reproductions/` dir
(scripts self-bootstrap that path).

## Run
```bash
# 1. build the frozen held-out LensBench manifest
python lensjudge/eval/build_eval_set.py --n-graded 120 --n-grade-d 50 --n-random 30

# 2. grade it (lean baseline), resumable, bounded concurrency
python lensjudge/imaging/run_batch.py --manifest lensjudge/outputs/lensbench_manifest.csv \
       --mode lean --concurrency 8 --out lensjudge/outputs/preds_lensbench_lean.parquet

# 3. robust panel / multi-agent on a comparison slice
python lensjudge/imaging/run_batch.py --manifest <slice>.csv --mode panel --out preds_panel.parquet
python lensjudge/imaging/run_batch.py --manifest <slice>.csv --mode multiagent --out preds_ma.parquet

# 4. spectroscopic gold
python lensjudge/spectro/run.py

# 5. score + report
python lensjudge/eval/report.py --out lensjudge/outputs/lensbench_v1.md \
       lean=...preds_lensbench_lean.parquet panel=...preds_panel.parquet multiagent=...preds_ma.parquet
```

## Key findings (honest)
1. **Spectroscopy works.** With clean physical features and spectroscopically confirmed
   gold, the agent re-grades **20/20** Hsu Table-2 Grade-A pairs as plausible and rejects
   **4/4** Foundry-II confirmed non-lenses.
2. **Imaging on the hard pool is near-chance vs single-consensus labels, and more compute
   doesn't help.** On the CNN's high-`p_meta` pool, *no* configuration — lean Sonnet
   (AUC 0.51 on n=200; 0.44 on the slice), lean **Opus** (0.32), **panel** (0.42), or the
   §9.1 **multi-agent** (0.38) — reproduces the consensus A/B/C grades; all are uniformly
   skeptical (p_lens≈0.02–0.03 even on consensus-A), and 2–4× the cost buys nothing.
3. **It's a measurement problem, not (only) a model one.** Cutouts are verified real and
   correctly resolved; obvious arcs *do* get high p_lens. But A/B/C are mostly *unconfirmed*
   candidates and the C/D boundary is one rater's subjective call, so agent-vs-consensus
   disagreement on imaging **cannot be adjudicated** without the multi-grader labels (or
   spectroscopic confirmation) the corpus lacks. The missing human ceiling is the result,
   not a footnote. See `papers/main.pdf` and `outputs/lensbench_v1.md`.
4. **A physical Foundry-I lens-model fit hits the same wall.** `quick_lensmodel` (a real
   GIGA-Lens MAP fit, the Foundry-I "does it model as a lens?" criterion) separates real
   lenses from *ordinary* galaxies (AUC ~0.77) but **not** from the hard Grade-D human-rejects
   (AUC ~0.44) — they admit an EPL fit just as well. So vision, a stronger backbone, a judge
   panel, multi-agent deliberation, *and* a physical model all fail on the same hard pool —
   convergent evidence the limit is the data + soft labels. Run:
   `python lensjudge/eval/run_modelability.py --gpus 0,1,2,3,4,5`; agent arm:
   `run_batch.py --mode lean --modelability`.
5. **Engineered representations (RepresentationKit) confirm it's a label/data limit, not an
   algorithm one.** `common/representations.py` + the `lens_representations` tool extract
   physically-motivated transforms (lens-light subtraction, polar/tangential, 180° symmetry,
   blue-excess color-isolation, DoG+Hessian arcness) as both images and scalar features. They
   **demonstrably extract lensing signal** — AUC ~0.80 on noiseless sims (SILVER) and ~0.80 on
   spectroscopically *confirmed* labels (GOLD) — and give a cheap easy-regime pre-screen
   (~0.70, ~50 ms vs the 45 s GIGA-Lens). But on the *hard consensus pool* they collapse to
   ~0.51, unchanged by cleaner Tier-2 implementations (photutils-isophote/skimage-Frangi/SEP,
   `tools/representations_proto.py`, all ≤0.57) **and** when fed to the agent (0.40 vs lean's
   0.44, +43% cost). The *same features* score 0.80 on confirmed and 0.51 on consensus — the
   wall is the soft Grade-D labels, not the method. Gate: `python lensjudge/eval/run_representations.py`;
   Tier-2: `python lensjudge/eval/run_representations_tier2.py`; agent: `run_batch.py --mode lean --representations`.

## Honest caveats
- **Consensus-referenced, no human ceiling**: labels are a single 2-author consensus; a
  ~250-cutout 2–3-grader study is the noted future fix to compute a human-vs-human κ.
- **Grade-D negatives are lens-*like*** (CNN-flagged) hard negatives — binary AUC is lower
  (and more honest) than against random galaxies; the strata are scored separately.
- **Spectroscopic** runs on catalog features + imaging; raw DESI fiber flux streaming
  (SPARCL / public healpix coadds) is the documented enhancement (`fetch_spectrum`).
- Orchestration of the §9.1 specialists is **programmatic** (Python fans out query() calls)
  rather than LLM-dispatched `AgentDefinition` subagents — more reliable/controllable and
  avoids the SDK's no-nested-subagents / static-tools / JSON-in-text caveats.
```
