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

## Tier-2 (HSC / Euclid high-res) — the decoupled fetch → stage → grade flow

Tier-2 re-grades ambiguous DESI candidates at higher resolution (HSC-SSP PDR3 0.168″, Euclid Q1 0.1″).
The open-weight port drives the SAME `fetch_hsc_cutout` / `fetch_euclid_cutout` tools through the OpenAI
tool loop (`eval/run_hsc.py`, `eval/run_euclid.py`, `tools/openai_tools.py`), SDK-free.

The two steps need different things, so **decouple** them:

| step | needs | where |
|---|---|---|
| **FETCH** HSC cutouts | internet + `HSC_USER`/`HSC_PASSWORD` (env-only, never committed) | gpu3, **or a Perlmutter LOGIN node** (compute nodes have no internet) |
| **GRADE** (the VLM) | the GPU; fully offline | gpu3, or the Perlmutter A100 |

**Euclid needs no API** — it's a static staged dataset (Zenodo Q1 15025832). Stage it once and point at it
with `LENSJUDGE_EUCLID_ROOT=$SCRATCH/.../euclid-q1/data`; then Euclid tier-2 runs entirely offline.

**HSC needs the das_cutout service**, so pre-fetch on an internet host, then grade offline from a warm
cache (a warm cache serves credential-free):

```bash
# 1) FETCH/STAGE on an internet host (gpu3 or a Perlmutter login node), creds set:
HSC_USER=... HSC_PASSWORD=... \
  python -m lensjudge.eval.stage_hsc --manifest cands.csv --cache $SCRATCH/ljv5/hsc_cache
#    cands.csv has ra,dec[,name] columns. Idempotent (already-cached rows skipped).

# 2) rsync the warm cache to the offline grade host:
rsync -a $SCRATCH/ljv5/hsc_cache/  <grade-host>:$SCRATCH/ljv5/hsc_cache/

# 3) GRADE offline (no creds, no internet) — the escalate cascade or the tier-2 runners:
export LENSJUDGE_BACKEND=openai LENSJUDGE_BASE_URL=http://localhost:8000/v1
export LENSJUDGE_MODEL_GRADER=<served-id>  LENSJUDGE_HSC_CACHE=$SCRATCH/ljv5/hsc_cache
python -m lensjudge.eval.run_hsc --ra 0.078839 --dec 0.271578        # single object
python -m lensjudge.eval.run_euclid --mode rank --n 90               # Euclid batch
```

Tier-2 is the resolution lever for *net-new* science — the V4 finding is that no open (or Claude) config
clears the DESI resolution wall at tier-1, so the ≥32B / GLM-4.6V-106B capacity is reserved for tier-2
where resolved arcs should actually pay off.

## Why `min_pixels` matters
Cutouts are 101 px upsampled to 400 px. A VLM's default image preprocessor may **re-downsample** a 400 px
image and destroy the faint arc. Raise `min_pixels` (Qwen3-VL) / the visual-token budget (Gemma) so the
cutout keeps enough visual tokens. Tune empirically and pin it; Qwen3-VL grades are prompt/resolution
sensitive (a 71-pt recall swing across prompt variants was reported), so **calibrate p_lens thresholds**
on a labeled set per config (`eval/calibrate.py`).
