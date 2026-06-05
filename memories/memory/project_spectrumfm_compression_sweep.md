---
name: project-spectrumfm-compression-sweep
description: Part-A finer-tokenizer compression sweep — 16x/544-token tokenizer does NOT improve good-z (ties/marginally worse than 32x V1); compression is NOT the precision bottleneck either; NERSC uses V1 32x
metadata:
  type: project
---
Part A of the build-out (2026-06-03/05) DONE. Tested whether the spectrum tokenizer's 32x spatial compression (272 tokens, ~23.5 A/token, localizes a line only to dλ/λ~0.0039 ~ the 0.0033 good-z threshold) caps redshift precision — the one lever Phase-15 never varied (V1/V2/codecs all 32x).

Made `src/tokenizers/spectrum.py` downsampling configurable (`downsample_strides`, default (1,2,2,2)=32x; persisted in ckpt; DECODER UpsampleBlock mirrors encoder via reversed strides — the build agent's first cut only did the encoder and crashed recon 17408 vs 8704, caught by a full-forward smoke). Added `--downsample-strides` (pretrain_tokenizer), `--max-seq-len` (train_transformer, default 512; eval tools default 1024 — RoPE position-absolute so V1 reproduces exactly). Loaders read strides from ckpt (eval_per_class/probe_six_class/measure_equivariance/train_transformer). All additive/default-preserving, build+verify via Workflow.

Trained finer 16x/544-token tokenizer (`--downsample-strides 1,2,2,1`, codebook-health PASS 5.18 bits/102 codes) + matched 15k Approach-A arm (eff-batch 64, --max-seq-len 1024, seed 42; needed `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` — batch 32 at seq 547 OOM'd via fragmentation at step ~2200 without it).

**RESULT (definitive 15k-vs-15k):** finer does NOT close the gap. Per-class catastrophic rates statistically identical to V1; aggregate good-z 23.9% vs 24.1%; galaxy median |dz|/(1+z) marginally WORSE with finer (LRG 0.088 vs 0.063, ELG 0.153 vs 0.111) — never better, the opposite of what finer localization predicts if compression were the limit. **Compression is NOT the precision bottleneck.** Finer = 2x encoder sequence for zero gain → NERSC uses V1 32x.

**Net across Phases 15-17:** every locally-accessible precision lever is ruled out — tokenizer architecture, equivariance prior, AND compression ratio — leaving data/model/training SCALE as the sole lever (the proposal's Phase-I premise). Folded into the tech report (Phase 17 + Conclusions fix) + `reproductions/redshifty/NERSC_SCALING_SPEC.md` (Part B: full-DR1, 104M→1.07B param ladder verified, eval_per_class per-class Metric-1 as the completion gate). See [[project-spectrumfm-ws1-per-class]], [[project-spectrumfm-ws4a-equivariance]], [[project-spectrumfm-v2-tokenizer-collapse]], [[project-spectrumfm-buildout]].
