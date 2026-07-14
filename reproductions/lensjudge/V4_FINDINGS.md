# LensJudge v4 — open-weight backend: findings

**Goal.** Replace proprietary Claude with **open-weight** models for LensJudge's two roles (vision grading +
agentic tool-calling), runnable small on local GPUs / Mac and large on Perlmutter, with Claude **retained as
the default backend and the regression oracle**. Branch `lensjudge-v4`.

## What shipped (infrastructure — all working)
- **Selectable backend** (`common/llm_client.py`): one OpenAI-compatible client behind the grader seam,
  env-switched `LENSJUDGE_BACKEND=anthropic|openai` (lazy `openai` import; the default Anthropic path is
  byte-identical). `grader_direct` ported; the agentic `run_tool_loop` is written (Phase-3 wiring pending).
- **Serving** (`serving/`): vLLM recipe validated on gpu3 (7× Titan RTX, sm_75). Turing gotchas encoded:
  `--dtype float16` (no bf16), `--attention-backend TRITON_ATTN` (FlashInfer attn crashes on Turing;
  `VLLM_ATTENTION_BACKEND` env was removed in vLLM 0.23), `VLLM_USE_FLASHINFER_SAMPLER=0`, `--enforce-eager`,
  `--limit-mm-per-prompt '{"image":N}'` (JSON).
- **LensBench-VI gate** (`eval/lensbench_gate.py`): splits **lens-vs-random** (detection) and
  **lens-vs-mimic** AUC (which `eval/score.py` lumps), + open-vs-oracle PASS/FAIL + INT4/FP16 flip check.
  Frozen 270-row manifest (150 lens / 60 random / 60 mimic).
- **Levers**: arcsinh-RGB render toggle; in-context few-shot (`LENSJUDGE_FEWSHOT_MANIFEST`).
- **Distillation** (`finetune/`): leakage-free SFT builder + ms-swift QLoRA train script (validated on Turing).
- 28 no-network unit tests pass.

## Headline result — detection AUC on LensBench-VI (lens A/B/C vs random)

| grader | detection AUC | parse | cost/cand |
|---|---|---|---|
| Qwen3-VL-8B off-the-shelf (Lupton) | 0.476 | 1.0 | $0 |
| Qwen3-VL-8B + arcsinh-RGB | 0.503 | 1.0 | $0 |
| Qwen3-VL-8B + few-shot | 0.406 | 1.0 | $0 |
| Qwen3-VL-32B off-the-shelf (fp16, TP=4) | **0.559** | 1.0 | $0 |
| Qwen3-VL-32B + few-shot (90-subset) | 0.542 | 1.0 | $0 |
| Qwen3-VL-8B QLoRA-distill **v1** (grade-buckets, token-acc select) | 0.416 | 1.0 | $0 |
| **Qwen3-VL-8B QLoRA-distill v2** (continuous p_lens, AUC-select) | **0.511** | 1.0 | $0 |
| **Claude sonnet (oracle)** | **0.663** | 1.0 | $0.016 |

`recovery@1%FPR = 0.0` for **every** config including Claude (the resolution wall).

## Conclusion
**No open-weight configuration matched Claude at DESI 1.3″ / 0.26″-px detection.** Scale is the only lever
that helped (8B→32B: +0.08); rendering was marginal; few-shot was unhelpful/harmful; and **first-pass QLoRA
distillation made detection *worse*** (0.416, anti-correlated): it graded 8/11 real grade-A lenses as "D" and
110/120 negatives as lens-grade. The fine-tune faithfully reproduced the target *format* (eval_token_acc
0.93 — misleading, dominated by JSON-structure tokens) but learned **spurious shortcuts** that inverted on
held-out objects. The bottleneck is the model's **visual discrimination of faint lens features at this
resolution**, not the training signal — consistent across off-the-shelf, few-shot, and distillation.

This mirrors the program's standing thesis: even Claude is resolution-limited here (0.66, 0 recovery@1%FPR);
the productive lever for *net-new* science is **higher resolution** (HSC/Euclid tier-2), not a better
DESI-resolution detector.

## Distillation v2 — continuous targets + AUC-based selection (levers 1 & 2, DONE)
The v1 failure had two fixable causes: grade-**bucketed** targets, and selection by **token-acc** (which
measures JSON structure, not the grade). v2 fixed both:
1. **Continuous targets** — relabelled 1,000 train cutouts (3-way-disjoint from the 270 eval) with Claude's
   *actual* p_lens (~$16). Teacher signal is monotonic but **compressed**: mean p_lens A 0.22 / B 0.15 / C 0.12 /
   D 0.10 (median 0.05) — Claude is resolution-limited, so its soft scores cluster near 0.
2. **AUC-based checkpoint selection** — frequent checkpoints, each scored by **detection AUC** on a held-out
   val-select set (`eval_checkpoints.py`). The curve peaked then declined — **25:0.476 · 50:0.485 · 75:0.511 ·
   100:0.543 · 125:0.493** — i.e. the overfit-then-degrade that v1 (train-to-end + token-acc) walked off.

Result: v2 (best ckpt-100) = **0.511 on the 270 eval** (val 0.543). This **fixed v1's inversion** (0.42→0.51,
recovery@1%FPR 0→0.013) and the methodology is now sound — but the distilled 8B only reached
**off-the-shelf-8B level** and did **not** close the gap to 32B (0.56) or Claude (0.66). p_lens stays
collapsed (truth A 0.08 ≈ D 0.10): you cannot distill a separator the teacher itself lacks at 0.26″/px.

**Firmed conclusion:** the limiter is *resolution*, not the training recipe. The distillation+AUC-selection
pipeline is correct and **reusable where the teacher signal is strong** (higher-resolution surveys) — but at
DESI resolution no open approach (off-the-shelf, few-shot, or distillation) reaches Claude, and Claude itself
is weak (0.66, ~0 recovery@1%FPR).

## 32B LoRA distillation (lever a, DONE — didn't help)
Same v2 pipeline, 32B QLoRA model-parallel on 2× Titan RTX (7 h). AUC-select **plateaus at ~0.54**
(25:0.516 / 50:0.539 / 75:0.542 / 100:0.541 / 125:0.542) — **≈ the 8B distill (0.543) and BELOW
off-the-shelf 32B (0.559)**. Distilling Claude's compressed-near-0 p_lens pulls the 32B toward the
collapsed regime, slightly *hurting* it vs off-the-shelf. Scale does not help distillation: the
resolution-limited teacher is the ceiling, independent of student size.

## Phase 3 — open-weight agentic tool-loop (DONE, mechanism transfers)
Wired the OpenAI tool-calling loop into the agentic graders (`tools/openai_tools.py` + `grader_lean`
branch on `LENSJUDGE_BACKEND`; Claude Agent SDK path retained). Live-validated on off-the-shelf
Qwen3-VL-8B (vLLM `--enable-auto-tool-choice --tool-call-parser hermes`): the model **emits tool calls
and loops** (mean 3.0 turns: fetch_cutout → get_photometry → grade), 270/270 parse.

| 8B config | detection AUC | **lens-vs-mimic AUC** |
|---|---|---|
| direct (no loop) | 0.476 | 0.39 |
| **agentic + v2 rubric (loop)** | 0.409 | **0.562** |
| Claude oracle | 0.663 | 0.520 |

The loop **lifts lens-vs-mimic 0.39 → 0.56** (above the Claude oracle's 0.52 on this set) while *not*
helping detection (0.48 → 0.41) — the **exact agency-ablation signature**: the loop/v2 config is
mimic-tuned and over-skeptical, trading lens-vs-random for lens-vs-mimic. The open-weight loop thus
reproduces Claude's qualitative behavior; the cascade design (cheap direct triage → agentic mimic
adjudication on escalations) ports to open weights.

## Phase 5 — Perlmutter A100 scale-out (DONE, validated offline)
Provisioned Perlmutter (login node): uv → `$SCRATCH/venvs/vllm` (vLLM) + `$SCRATCH/venvs/lensjudge`
(direct-mode deps, **no claude-agent-sdk**), Qwen3-VL-8B staged to `$SCRATCH` HF cache, lensjudge code
rsync'd, cutouts staged as `{name}.fits` (new `LENSJUDGE_CUTOUT_DIR` override). A self-contained slurm
job (`serving/perlmutter_vllm.slurm`) serves the model on **1 A100 (bf16)** and grades via the local
endpoint. **Validated (job 55339203): server ready ~90 s, 12/12 parse, ~5.8 s/candidate (~3× gpu3),
fully offline** (weights from `$SCRATCH`, no internet, SDK-free).

Two fixes this required (both committed): (1) **all vLLM/Triton caches + TMPDIR must go to `$SCRATCH`** —
Perlmutter `$HOME` is quota-tiny (~40 GB, was 101% full) and compute `/tmp` is small, else the weight
loader dies with `Errno 122 Disk quota exceeded`; (2) the Claude Agent SDK is now **optional** (guarded
imports across grader_lean/hooks/tools) so the open path imports with no `claude_agent_sdk` installed.

## Bigger model on Perlmutter A100 — EMPIRICAL (validated on-cluster)
Perlmutter has 1,536× 4×A100-40GB + 256× 4×A100-80GB nodes (direct all-to-all NVLink3; TP≤4 fast;
`-C "gpu&hbm80g"` for 80GB). A100=Ampere → no FP8 → INT4 (AWQ-marlin or compressed-tensors) for big
models. Measured on the same LensBench-VI cutouts, offline, SDK-free:

| model (A100) | quant | GPUs | metric | value | s/cand |
|---|---|---|---|---|---|
| Qwen3-VL-8B (direct) | bf16 | 1×40GB | detection AUC | 0.476 | 5.8 |
| **Qwen3-VL-32B (direct)** | AWQ INT4 | 1×40GB | detection AUC | **0.534** | 13.4 |
| Qwen3-VL-8B (agentic v2) | bf16 | 1× | lens-vs-mimic | 0.562 | ~15 |
| **GLM-4.6V-106B (agentic v2)** | comp-tensors INT4 | 2×40GB TP2 | lens-vs-mimic | **0.560** | ~110 |
| Claude oracle | — | — | detection AUC | 0.663 | — |

32B-AWQ costs ~2.5 pt AUC vs 32B-fp16 (0.559) — 8.5% lens-call flips — acceptable for the single-GPU fit.
**Conclusion: at DESI 0.26"/px, bigger ≈ same accuracy** — the 106B agentic ties the 8B agentic (0.56) at
~7× the latency and 84% parse; the 32B is a modest +0.06 over 8B. Parameters don't clear the resolution
wall. **Practical picks:** DR11 bulk sweep → **Qwen3-VL-32B-AWQ**, 4 data-parallel replicas per 40GB node
(~3.4 s/cand aggregate); reserve GLM-4.6V-106B / 235B-A22B for **tier-2 (resolved HSC/Euclid)** where
capacity should actually pay off — NOT for tier-1 DESI. Serving gotchas learned: QuantTrio Qwen3-VL AWQ →
`--quantization awq_marlin`; cyankiwi GLM-4.6V is **compressed-tensors** (omit `--quantization`, auto-detect);
GLM tool parser = `glm47` (not glm47_moe); route all caches to `$SCRATCH`.

## Tier-2 (HSC/Euclid) open-weight port — DONE (infrastructure; SDK-free, no-network tested)
The resolution lever for *net-new* science now runs on the open backend. `fetch_hsc_cutout` /
`fetch_euclid_cutout` are exposed as OpenAI tool schemas + executor branches (`tools/openai_tools.py`,
reusing the SDK tools' loaders/renderers/`VIEW_DESC` so the open agent sees byte-identical HSC/Euclid
evidence); `eval/run_hsc.py` + `eval/run_euclid.py` gained an `LENSJUDGE_BACKEND=openai` branch
(`run_tool_loop`) with the Claude-SDK path retained; the SDK is now optional across the tier-2 modules
(`hsc_cutout`/`euclid_cutout`/`run_hsc`/`run_euclid`) so the escalate cascade no longer silently drops
to tier-1 in an SDK-free deploy. 17/17 no-network unit tests pass (`tests/test_tier2_openai.py` +
extended `test_openai_tools.py`).

**Where it runs — the decoupled fetch → stage → grade flow** (answers the tier-2 API/location question):
- **Euclid needs no API** — static Zenodo Q1 dataset; `LENSJUDGE_EUCLID_ROOT` points at a staged copy →
  runs **entirely offline on Perlmutter**.
- **HSC needs the das_cutout service** (creds `HSC_USER`/`HSC_PASSWORD`, env-only). Perlmutter compute
  nodes have no internet, so **fetch** on an internet host (gpu3 or a Perlmutter **login** node) via the
  new `eval/stage_hsc.py`, rsync the warm cache, then **grade offline** on the A100. Enabled by three
  small offline-cache changes: `hsc_fetch.fetch_hsc_cutout` serves a **warm cache credential-free** (auth
  only needed to fetch a *missing* band) + a new `cached()` helper; `highres._resolve_hsc` treats a warm
  cache as coverage without creds; `LENSJUDGE_HSC_CACHE` overrides the cache root (→ `$SCRATCH`). See
  `serving/README.md` for the exact commands.

## Tier-2 Euclid rank gate — first open-vs-Claude comparison (2026-07-01)
Ran `eval/run_euclid --mode rank` on **89 Euclid Q1 objects** (34 grade-A / 34 B / 21 C, deterministic
seed → identical objects) through the agentic `fetch_euclid_cutout` tool loop, on BOTH backends:
open **Qwen3-VL-8B** (served on gpu3, tools) and the **Claude sonnet oracle**. Data verified healthy
(87/89 clean FITS; a corrupt-FITS `OSError` in `load_euclid` was found + hardened to graceful None).

| backend | p_lens by expert grade (A / B / C) | agent grades | Spearman(p_lens,expert) | AUC(A vs C) | parse | cost |
|---|---|---|---|---|---|---|
| **Qwen3-VL-8B** (agentic) | 0.05 / 0.03 / 0.00 | **86/89 = D** | 0.13 (n.s.) | 0.529 | 1.0 | $0 |
| **Claude sonnet** (oracle) | 0.52 / 0.46 / 0.46 | 22A/30B/18C/19D | 0.108 (n.s.) | 0.556 | 1.0 | $6.62 |

open-vs-oracle p_lens Spearman 0.275; exact grade agreement 22%.

**Two findings.** (1) **The Euclid A/B/C "grade" is a subtle expert-CONFIDENCE ranking among already-selected
discovery-engine candidates** (expert_score A 2.43 / B 1.71 / C 1.47), *not* a lens-vs-non-lens axis — grade-A
appears across the `lens`, `unsuccess`, and `group` subsets alike, and only 21 grade-C are staged (the full
catalog is 78% C, unstaged). So **even Claude is at chance here (AUC 0.556, Spearman n.s.)** → this task
**cannot gate open-vs-Claude** (no oracle bar to clear). (2) The decision-relevant signal: the **open 8B's
p_lens collapses (all-D, ~0) even with resolved 0.1″ arcs**, while Claude stays calibrated (mean ~0.49,
correctly ordered A>B,C). The 8B is **unusable at tier-2 as-is** — over-skeptical *and* unfamiliar with the
Euclid VIS+NIR color rendering (more collapsed than at DESI tier-1, where it gave p_lens 0.04–0.09). This is
the same agency-ablation over-skepticism, worse on out-of-distribution Euclid imagery.

**Implication for the plan:** a meaningful tier-2 open-vs-Claude gate needs BOTH (a) a task where the oracle
clearly succeeds — **HSC SuGOHI known-lens recovery** (`run_hsc_validation`; Claude v3 pilot recovered 3/3)
or the **Euclid paired DESI→Euclid p_lens lift** (the README's actual resolution finding) — and (b) a
**bigger open model** (GLM-4.6V-106B / Qwen3-VL-235B on Perlmutter, per the tier-2 model rec); the 8B is too
weak/OOD for tier-2. Artifacts: `outputs/euclid_rank_{open_8b,oracle}.parquet` (gitignored).

## Tier-2 HSC known-lens gate — bigger open model (GLM-4.6V-106B) vs Claude (2026-07-01)
The Euclid rank task couldn't gate (oracle at chance), so we ran the clean **oracle-succeeds** gate:
grade **26 SuGOHI-matched known lenses** (real, confirmed lenses; storfer+inchausti) at **HSC-SSP PDR3
0.168″** through the agentic `fetch_hsc_cutout` loop. This exercised the full decoupled flow end-to-end:
`eval/stage_hsc.py` fetched 73/76 HSC cutouts on gpu3 (creds+internet) → rsync warm cache to Perlmutter →
grade **offline on A100s** (credential-free warm cache) with **GLM-4.6V-106B-AWQ** (TP2, tools).

| grader | recovery (A/B) | grades | mean p_lens | parse |
|---|---|---|---|---|
| **Claude sonnet** (oracle) | **92%** (23/25 escalated) | mostly A/B | 0.618 | — |
| **GLM-4.6V-106B-AWQ** (agentic) | **0%** | **all D (25/25)** | 0.010 | 25/26 |

GLM's rationales are **genuine and per-object** (it accurately describes each image — "red elliptical with a
nearby compact blue source, round/point-like, no tangential curvature" → D; one correctly IDs a ring but
calls it "a ring galaxy, not a lens"). So this is not a plumbing/parse artifact: **GLM-4.6V-106B sees the
images but is systematically over-skeptical, grading confirmed lenses as non-lenses (D, p_lens 0.01).**

**Verdict — scale does NOT fix tier-2.** Across both tier-2 experiments the open models collapse: Qwen3-VL-8B
graded 86/89 D at Euclid 0.1″, and GLM-4.6V-106B graded **25/25 D** at HSC 0.168″ (0% recovery) where Claude
recovers 92%. A 13× larger model fails as hard as the 8B. The bottleneck is the open models' **calibration /
domain knowledge for lens vetting** (they demand textbook arcs and reject everything else), not resolution or
capacity. This is the tier-1 over-skepticism (the agency-ablation signature) taken to the extreme at tier-2.

**Serving gotchas learned (all real, cost cycles):** (1) **GLM-4.6V is a reasoning model** — in the agentic
loop its `<think>` blocks never terminate with JSON; *more* `max_tokens` made parsing WORSE (12/26→6/26).
Fixed with `LENSJUDGE_NOTHINK=1` (→ `chat_template_kwargs.enable_thinking=false`), which lifted parse to
25/26. (2) **KV-cache OOM** at `--max-model-len 32768` on 2×40 GB (util 0.92 → max ≈23520); use ≤22528.
(3) The **fetch→stage→grade decoupling works**: `stage_hsc.py` on gpu3 + offline credential-free warm-cache
grade on the A100 (the tier-2 port's core design). Artifact: `outputs/hsc_gate_glm106_nothink.parquet`.

**Actionable next step (the real path to open tier-2):** unlike DESI tier-1 (where Claude's teacher signal is
weak, so distillation didn't help), **at HSC/Euclid Claude's signal is STRONG** (92% recovery, p_lens 0.62) —
exactly the regime the v4 distillation pipeline was built for ("reusable where the teacher signal is strong").
Fine-tuning/distilling an open VLM on Claude's HSC/Euclid grades is the promising route to a usable open-weight
tier-2 grader; off-the-shelf open VLMs (8B–106B) are not.

## Tier-2 Euclid distillation PoC — DISTILLATION WORKS (2026-07-02)
The gates above showed off-the-shelf open VLMs (8B, GLM-106B) collapse to all-D at tier-2. This PoC tests
the proposed fix: LoRA-distill **Claude's DIRECT Euclid grades** into Qwen3-VL-8B. Pipeline
(`finetune/distill_euclid.py`, fully offline): grade-stratified split (train 388 / valsel 64 / test 80,
disjoint) → Claude-label all three DIRECT (no tool loop) at Euclid 0.1″ (~$8; balanced signal: train
52A/106B/108C/121D) → SFT (349 ex, 4 Euclid views + Claude JSON targets) → QLoRA on Qwen3-VL-8B (3 epochs,
2 h, gpu3) → **AUC-select** (valsel curve 20:0.70 · 40:0.77 · **60:0.89** · 80:0.85 · 120:0.88 — classic
overfit-then-plateau; best ckpt-60) → merge → serve → gate on held-out test.

**Held-out test (80 obj), reproducing the Claude teacher:**

| metric | off-the-shelf 8B | **distilled (ckpt-60)** | Claude |
|---|---|---|---|
| recovery of Claude-A/B | 47% | **87%** | — |
| AUC (Claude A/B vs D) | 0.625 | **0.704** | — |
| Spearman(student, Claude p_lens) | 0.21 | **0.41** | — |
| grade dist | D56/B17/A7 | **B52/C15/D10/A3** | D34/B19/C16/A11 |
| mean p_lens | 0.25 | 0.51 | 0.35 |

**Distillation FIXED the tier-2 collapse:** the student stops defaulting to D — recovery of Claude's lenses
47%→**87%**, agreement with Claude's p_lens doubled (Spearman 0.21→0.41), separation of Claude-A/B-vs-D
0.625→**0.704**, all on held-out data. This is the first positive tier-2 result and validates the v4 thesis:
distillation pays off **where the teacher signal is strong** (HSC/Euclid), unlike the failed DESI tier-1
attempt (weak teacher). The v2 distillation methodology (continuous targets + AUC-select) ports directly.

**Honest caveats:** small PoC (349 train / 80 test; noisy). The distilled student now slightly *over*-calls
(mean p_lens 0.51 > Claude 0.35, B-heavy) — high recovery but an elevated false-positive rate; it learned
"don't collapse" but overshot. Held-out AUC 0.704 < valsel 0.887 (selection optimism → true generalization
~0.70). Teacher = Claude's own Euclid grades (imperfect on fine A/B/C). **To productionize:** scale train +
hard negatives to sharpen calibration, add HSC, and gate on REAL confirmed lenses (SuGOHI-style) not just
teacher agreement. But the core question — can distillation lift an open VLM out of the tier-2 collapse? —
is answered **yes**. Artifacts (gitignored): `outputs/distill_euclid/` (labels, sft, ckpt-60-merged, gate
parquets).

## Tier-2 productionization attempt (HSC hard-neg + real-lens gate) — NEGATIVE, well-diagnosed (2026-07-02)
The Euclid PoC distilled Claude's *soft* grades on all-candidate data and improved (0.625→0.704) but
over-called for lack of true negatives. This pass tried to productionize: add REAL HSC hard negatives
(132 covered mimics/randoms, fetched via parallel `stage_hsc`) + SuGOHI-confirmed positives, train a
**combined HSC+Euclid** student on **ground-truth** labels (SuGOHI→A, mimic/random→D), and gate on a
HELD-OUT **real-lens** test (26 confirmed SuGOHI lenses vs 46 true non-lenses) — not teacher agreement.

**Gate (held-out HSC real-lens test):**

| grader | real-lens AUC | recovery | rejection | mean p_lens L / N |
|---|---|---|---|---|
| off-the-shelf 8B | 0.730 | 8% | 96% | 0.15 / 0.09 |
| **distilled (combined, ckpt-118)** | **0.502** | 15% | 85% | **0.18 / 0.18** |
| Claude (oracle) | 0.823 | 54% | 89% | 0.46 / 0.12 |

**The combined ground-truth recipe FAILED — it flattened discrimination to chance (AUC 0.50), *below*
off-the-shelf (0.73).** The distilled student calls lens-like features indiscriminately: 4 true-positives
but **7 false-positives** (mimics look arc-like), identical mean p_lens for lenses and non-lenses.

**Root causes (diagnosed, not guessed):**
1. **Ground-truth-A on ambiguous positives.** Claude itself grades only 54% of these SuGOHI-at-HSC lenses
   A/B — ~46% look genuinely non-lensy even to Claude. Forcing them all to target A taught the student to
   call ambiguous/arc-like things lenses → it can't separate real arcs from mimics.
2. **Cross-survey selection masking.** The AUC-select valset was 38 Euclid / 13 HSC, so its 0.833 "best"
   was Euclid-dominated and hid the HSC-only failure (0.50). Per-survey selection is mandatory.
3. **Too few / noisy HSC positives** (47 train) and mimics that are visually lens-like — a hard, small set.

**The Euclid teacher-distillation result (0.625→0.704) stands; this ground-truth+combined+small-HSC recipe
does not transfer to real HSC lenses.** Corrected recipe for a future iteration: distill Claude's **soft**
HSC grades (calibrated, captures visual difficulty) — NOT ground-truth-A — on a larger, cleaner HSC
positive set; train **HSC-only** and **AUC-select on an HSC-only** valset; keep the hard negatives.
Artifacts (gitignored): `outputs/distill_hsc/`, `outputs/distill_combined/`.

## Corrected recipe (soft Claude HSC labels + HSC-only selection) — PARTIAL, data-limited (2026-07-02)
Applied the two diagnosed fixes: (1) distill Claude's **soft** HSC grades (positives spread 13A/8B/10C/7D,
mean p_lens 0.48 — ambiguous ones correctly C/D, NOT forced to A) via `distill_hsc sft --claude-preds`;
(2) train **HSC-only** (95 ex) + **AUC-select on an HSC-only** valset (`distill_hsc evalset`) — no
cross-survey masking. Same held-out real-lens test.

| grader | real-lens AUC | recovery | rejection | TP / FP |
|---|---|---|---|---|
| off-the-shelf 8B | 0.730 | 8% | 96% | 2 / 2 |
| combined ground-truth (failed) | 0.502 | 15% | 85% | 4 / 7 |
| **HSC-only SOFT (corrected)** | **0.617** | **19%** | 83% | 5 / 8 |
| Claude (oracle) | 0.823 | 54% | 89% | 14 / 5 |

**The corrected recipe is methodologically VALIDATED but data-limited.** It fixed the catastrophic
ground-truth failure (0.502→0.617, real discrimination restored: L mean p_lens 0.20 > N 0.16) and **more
than doubled recovery** (8%→19%, finds 5 real lenses vs off-the-shelf's 2), with a calibrated grade spread
(D31/C28/B12/A1) instead of collapse. **But 95 HSC training examples is too thin to reach a usable grader:**
it does not beat off-the-shelf's *AUC* (whose 0.730 is a rejection artifact — 8% recovery, useless as a
finder) and is far below Claude (0.823 / 54%). The AUC-select curve peaked early (16:0.71) then overfit —
the signature of too little data.

**Firm conclusion:** the soft-distillation recipe is CORRECT (restores discrimination + lifts recovery toward
Claude); the binding constraint at tier-2 is now **HSC training-data volume**, not method. The clear next
lever is **more real HSC lens positives** — the full SuGOHI catalog (hundreds, not just the 73 DESI-matched)
and/or Euclid-confirmed lenses (higher resolution, larger sample) — then re-run this exact recipe.

## Summary of the tier-2 distillation arc
- Off-the-shelf open VLMs (8B, GLM-106B) **collapse** at tier-2 (all-D, ~0% recovery).
- **Euclid soft-distillation** of Claude's grades **works** (0.625→0.704) — first positive result.
- HSC combined **ground-truth** recipe **fails** (0.50, flattened) — ambiguous-A labels + cross-survey masking.
- HSC-only **soft** recipe (both bugs fixed) **restores discrimination + doubles recovery** (0.617, 19%) but
  is **data-limited** at 95 examples → below Claude. Method validated; needs more HSC/Euclid lens data.

## Remaining (not done)
- **Scale the HSC/Euclid positive set** (full SuGOHI catalog / Euclid-confirmed lenses) + re-run the soft recipe.
- **Run** the tier-2 open-weight vetting at scale (stage the HSC shortlist / Euclid subset, grade on the
  A100) — a science-campaign run, not a "does it work" step; the path is now proven & tested offline.
- A full production DR11 tier-1 vetting sweep on Perlmutter with Qwen3-VL-32B-AWQ — likewise a
  campaign run; the path is proven.
