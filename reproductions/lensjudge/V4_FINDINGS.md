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
| **Qwen3-VL-8B QLoRA-distilled** | **0.416** | 1.0 | $0 |
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

## Next-iteration levers (not yet tried)
1. Distill Claude's **continuous p_lens** (soft scores), not grade buckets.
2. Add a **detection-AUC validation metric during training** + early-stop (token-acc is useless as a gate).
3. Larger student (**32B LoRA**); more data; regularize to curb shortcut overfit.
4. **Phase 3** (agentic tool-loop + tier-2 HSC/Euclid) — where lens-vs-mimic and net-new value actually live;
   independent of the detection gap.
