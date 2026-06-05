# SpectrumFM — NERSC full-scale run spec (DOE Genesis Phase-I go/no-go)

**Status:** ready (tokenizer choice decided — V1 32×, §5).
**Audience:** the DOE Genesis SpectrumFM Phase-I scale-up. This is the
proposal-facing spec; it reuses the implementation substrate in the redshifty
repo (a course project) but replaces its internal success criteria (aggregate
TF/AR accuracy) with the **proposal's per-class go/no-go metrics**.

## 1. Purpose

The local 2×L4 prototype settled the architecture questions and built the
evaluation framework, and every result points to **scale** as the lever for
redshift *precision*:

- **Metric 1 (per-class parity vs Redrock):** FAIL at prototype scale — only
  stars reach DESI good-z; galaxies/QSO are catastrophic at the 0.0033
  threshold (`tools/spectrumfm/eval_per_class.py`).
- **Six-class capability:** strong — frozen-encoder probe macro-F1 0.81–0.85 vs
  0.10 no-skill (`probe_six_class.py`).
- **Tokenizer architecture / equivariance prior:** do not move precision
  (V1≈V2-noskip≈codecs; equivariance instilled but precision unchanged).
- **Spectrum-tokenizer compression (16× vs 32×):** tested locally (§5) — result
  decides whether the full-scale runs use the finer tokenizer.

This spec defines the NERSC runs that deliver the **actual precision verdict**
(tens of millions of spectra, 250M–1B params) that 2×L4 cannot.

## 2. Reused scaffolding (already in the redshifty repo)

- `nersc/train_transformer_ddp.slurm`, `nersc/pretrain_tokenizer_ddp.slurm` — 1
  node × 4 A100, `--ntasks=4 --gpus=4 --gpus-per-task=1`, NCCL rendezvous
  (`MASTER_ADDR`/`MASTER_PORT` from SLURM), `srun python … --amp`.
- `nersc/build_dr1_index.py` — walks `/global/cfs/cdirs/desi/public/dr1`, writes
  the JSONL manifest; `--surveys sv3 main --programs bright dark`, `--max-healpix`
  cap (uncap for full DR1).
- `nersc/stage_to_scratch.py` (+ `.slurm`) — optional CFS→SCRATCH staging.
- **Eval harnesses (the go/no-go instruments):** `eval_per_class.py`,
  `probe_six_class.py`, `eval_redshift_dz.py`, `measure_equivariance.py` — all
  take `--checkpoints`, all run on a single GPU in minutes.

## 3. Data — full DR1

- Build the full manifest: `build_dr1_index.py --surveys sv3 main --programs
  bright dark` with **no `--max-healpix` cap** (the local mix used ~200 healpix /
  1.8M spectra; full DR1 is the entire iron healpix tree — target the proposal's
  ~tens of millions of spectra). The **four-way survey×program mix is
  load-bearing** (it is what ignites Approach-A; do not subset to one combo).
- Path: `/global/cfs/cdirs/desi/public/dr1/spectro/redux/iron/healpix/{sv3,main}/{bright,dark}/`.
- Stream from CFS by default; stage to SCRATCH only if GPU SM-util < ~90%
  (Lustre I/O bound) via `stage_to_scratch.slurm`.
- Holdout: keep the existing healpix-level split (`--healpix-holdout-frac 0.05`,
  seed 42) so validation has no same-pointing leakage — **the same split the
  per-class evaluator assumes.**

## 4. Model param ladder (proposal's 100M → 1B)

Knobs are decoupled (`--d-model`, `--n-encoder-layers`, `--n-decoder-layers`,
`--n-heads`; `n_redshift_classes` and vocab are independent of `d_model`). Param
counts below were **verified by constructing each config** (the baseline
103.67M matches the training script's own `[model] params=` print):

| tier | d_model | enc/dec layers | heads | params | node config |
|------|---------|----------------|-------|--------|-------------|
| baseline (local) | 768 | 6 / 6 | 12 | 103.67 M | 1 node × 4 A100 |
| P-250M | 1024 | 8 / 8 | 16 | 245.7 M | 1–2 nodes |
| P-500M | 1280 | 12 / 12 | 16 | 575.2 M | 2–4 nodes |
| P-1B | 1536 | 16 / 16 | 16 | 1.069 B | 4–8 nodes |

Note: the redshifty course plan is **1-node/4-GPU only**; the ≥250M tiers need
**multi-node** (`--nodes=N`, NCCL over Slingshot) — a scaffolding addition
(DistributedSampler already handles >1 node; the `*_ddp.slurm` headers need
`--nodes`, and the rendezvous already derives `MASTER_ADDR` from the nodelist).

## 5. Tokenizer choice — DECIDED: V1 32×

The local compression sweep is complete. A finer 16× / 544-token tokenizer
(`--downsample-strides 1,2,2,1`, codebook-health PASS at 5.18 bits) with a matched
15k Approach-A arm (`--max-seq-len 1024`, eff-batch 64, seed 42) was compared
head-to-head with the 32× V1 via `eval_per_class.py` on the identical 2048-spectrum
val set. **Result: no benefit.** Per-class catastrophic rates are statistically
identical, the aggregate good-z is unchanged (23.9% vs 24.1%), and the galaxy
median |Δz|/(1+z) is marginally *worse* with finer (LRG 0.088 vs 0.063; ELG 0.153
vs 0.111). Spatial compression is not the precision bottleneck.

**→ Use the V1 32× tokenizer for all full-scale runs.** The finer tokenizer
doubles the encoder sequence (547 vs 275) — strictly more compute for zero gain.
The precision burden rests entirely on data/model/training scale.

## 6. DDP launch protocol (per tier)

- Per-GPU batch 32 (baseline); reduce for larger `d_model` / `max_seq_len` to
  fit 40/80 GB A100, keep global batch constant via more GPUs or grad-accum.
- LR linear-scaled from the single-GPU 2e-4 by global-batch ratio (the course
  plan uses 8e-4 at 4×; fall back to √-scaling if it diverges early).
- **Always run the debug-QOS smoke first** (`STEPS=200 MAX_HEALPIX=20`, ~10 min)
  before each long run — catches NCCL/rendezvous/shape issues cheaply.
- Steps: 100k baseline; the ≥250M tiers ~300–500k (the proposal's "longer
  training" axis). Checkpoint every ~2.5k; `best.pt` by val.
- Checkpoints to `/global/cfs/cdirs/deepsrch/$USER/checkpoints/$RUN_NAME/`.

## 7. Run-completion GATE = the proposal's go/no-go (NOT aggregate accuracy)

A tier "passes" only when, on the held-out split:

1. **Metric 1 — per-class parity:** `eval_per_class.py --checkpoints <best.pt>` —
   no class (LRG/ELG/QSO/MWS, + BGS) degrades >5pp catastrophic vs Redrock on
   the ZWARN==0 set. This is the primary gate (the local prototype FAILs it; the
   scale-up exists to pass it).
2. **Six-class capability:** `probe_six_class.py` macro-F1 ≥ the local 0.85
   (must not regress with scale).
3. **Metric 3 — scaling curve:** each tier is a point on per-class catastrophic
   rate vs (params × tokens-seen); the **local ladder points are the
   low-compute anchors of the same curve** (same evaluators, same metric), so
   the NERSC points extend a curve that already has its low end pinned. Pass =
   a monotone-improving (power-law-like) trend toward DESI parity.

Report all three per tier; escalate to the next tier only while Metric-1
catastrophic rate is still falling with scale.

## 8. Estimated allocation

Baseline (104M, 100k steps, 4×A100): ~12 h wall / ~120 A100-h (per the course
plan's DDP4 estimate). The ≥250M multi-node tiers scale roughly with
params×steps; budget the curve (4 tiers) at a few thousand A100-hours, gated so
a flat Metric-1 trend halts escalation early (don't burn 1B-param compute if
250M→500M shows no per-class improvement).

## 9. Open decisions

- [x] Tokenizer: **V1 32×** (finer 16× tested locally — no benefit, §5).
- [ ] Multi-node scaffolding for ≥250M (add `--nodes` to the `*_ddp.slurm`).
- [ ] Full-DR1 manifest size / staging vs stream (confirm via `wc -l` + SM-util).
- [ ] Allocation cap and the Metric-1-trend halt rule.
