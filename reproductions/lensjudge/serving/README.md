# LensJudge v4 — open-weight serving recipes

LensJudge v4 makes the LLM backend **selectable**. The Anthropic/Claude path is the default and is
untouched; to run on an **open-weight** model you (1) start an OpenAI-compatible server with one of the
recipes here, then (2) point LensJudge at it with three env vars. Nothing else in the harness changes —
the same `eval/run_cascade.py`, `imaging/run_batch.py`, rubric prompts, `ImageGrade` schema, and
`eval/calibrate.py` all work as-is.

## Point LensJudge at an open server

```bash
export LENSJUDGE_BACKEND=openai
export LENSJUDGE_BASE_URL=http://localhost:8000/v1     # the server you start below
export LENSJUDGE_MODEL_GRADER=Qwen/Qwen3-VL-8B-Instruct # the served model id (flows through config.MODELS)
# optional:
export LENSJUDGE_API_KEY=EMPTY            # local servers accept anything (default EMPTY)
export LENSJUDGE_GUIDED_JSON=1            # vLLM/SGLang: constrain output to the ImageGrade schema
# export LENSJUDGE_JSON_MODE=1            # portable JSON nudge (some servers)
# export LENSJUDGE_TEMPERATURE=0.0        # deterministic grading (default)

# then run LensJudge exactly as before, e.g. the cheap Stage-1 detection grader:
python -m lensjudge.imaging.run_batch --mode direct ...
# or the cascade (Stage-1 direct + Stage-2 escalate):
python -m lensjudge.eval.run_cascade ...
```

Per-role overrides use the existing seam (`LENSJUDGE_MODEL_{GRADER,JUDGE,WORKER,ARBITRATOR,SPECTRO}`), so
you can **mix** backends — e.g. an open Stage-1 grader with a Claude Stage-2 adjudicator: set
`LENSJUDGE_BACKEND=openai` for the open run, or keep `=anthropic` and override only one role once the
agentic loop port (Phase 3) lands. **To return to Claude** (the regression oracle): `unset
LENSJUDGE_BACKEND` (default `anthropic`).

## Hardware tiers (this lab)

| Box | GPU | Recommended grader | Notes |
|---|---|---|---|
| **gpu3** | Titan RTX 24 GB (Turing) | Qwen3-VL-8B fp16 (~17 GB) or 30B-A3B **INT4** | **no FP8** on Turing → AWQ/GPTQ INT4 only |
| **phoenix** | L4 24 GB (Ada) [+8×A16 16 GB] | Qwen3-VL-8B / 30B-A3B | L4 supports FP8; A16 pool = slow PCIe TP, batch only |
| **Mac Studio** | M4 Max 128 GB | Qwen3-VL-8B→32B 4-bit (MLX) | also gpt-oss-120b / GLM-4.5-Air as text controller |
| **Perlmutter** | A100 40 GB ×4 (Ampere) | 1 GPU: 32B/30B-A3B INT4 · TP=4: 235B INT4 | **no FP8 on Ampere** → AWQ/GPTQ INT4; offline (`HF_HUB_OFFLINE=1`) |

## Recipes
- `serve_vllm_qwen3vl.sh` — single-GPU vLLM (gpu3 / phoenix L4), OpenAI endpoint + tool calling.
- `serve_mlx_qwen3vl.sh` — Mac Studio via MLX (mlx-vlm), OpenAI endpoint.
- `perlmutter_vllm.slurm` — A100 batch server in a podman-hpc container, INT4, tensor-parallel, offline.

> These are **templates** — verify the exact model id / quant repo on Hugging Face at deploy time
> (the open-model frontier moves fast). The correctness-critical flags (tool parser, no-FP8 on
> Turing/Ampere, `min_pixels` for tiny cutouts) are noted inline in each script.

## Why `min_pixels` matters
Cutouts are 101 px upsampled to 400 px. A VLM's default image preprocessor may **re-downsample** a 400 px
image and destroy the faint arc. Raise `min_pixels` (Qwen3-VL) / the visual-token budget (Gemma) so the
cutout keeps enough visual tokens. Tune empirically and pin it; Qwen3-VL grades are prompt/resolution
sensitive (a 71-pt recall swing across prompt variants was reported), so **calibrate p_lens thresholds**
on a labeled set per config (`eval/calibrate.py`).
