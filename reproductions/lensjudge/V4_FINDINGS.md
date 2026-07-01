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

## Remaining (not done)
- **Run** the tier-2 open-weight vetting at scale (stage the HSC shortlist / Euclid subset, grade on the
  A100) — a science-campaign run, not a "does it work" step; the path is now proven & tested offline.
- A full production DR11 tier-1 vetting sweep on Perlmutter with Qwen3-VL-32B-AWQ — likewise a
  campaign run; the path is proven.
