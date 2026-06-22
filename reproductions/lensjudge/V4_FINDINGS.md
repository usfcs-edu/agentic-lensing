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

## Next-iteration levers (not tried)
1. **32B LoRA** distillation (scale helped most off-the-shelf; may transfer to fine-tune).
2. **Phase 3** — agentic tool-loop + tier-2 HSC/Euclid escalation, where lens-vs-mimic and *net-new* value
   actually live and the resolution lever applies; independent of the DESI-resolution detection gap.
