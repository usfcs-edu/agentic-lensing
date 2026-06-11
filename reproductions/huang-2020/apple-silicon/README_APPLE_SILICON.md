# Huang-2020 reproduction — Apple Silicon (M4 Max / MPS) port

A self-contained re-run of the `huang-2020` lens-finding pipeline on Apple
Silicon, validating the Mac Studio (M4 Max, 128 GB, MPS) as a from-scratch model
workhorse. **All computation** (catalog filtering, negatives, parent-sample
selection, both ResNet trainings, and inference) runs locally on MPS; the **base
survey data** is pulled from `phoenix` rather than re-downloaded from the original
web services.

The original 21 scripts in `../` are untouched. This directory copies them and
applies only the minimal edits needed for MPS + the nested path layout (see
`device.py` and the diff against `../`).

## Result (2026-06-02)

Full from-scratch run on the M4 Max reproduces phoenix; `verify_against_reference.py`
passes **13/13** gated port-correctness checks (`data/REPRODUCTION_MPS_COMPARE.md`):

| metric | MPS (M4 Max) | phoenix (CUDA) |
| :--- | ---: | ---: |
| DR9 ResNet test AUC | 0.9988 | 0.9991 |
| DR7 ResNet test AUC | 0.9945 | 0.9943 |
| MPS-vs-CUDA inference, max\|Δ\| (same ckpt) | 4.1e-4 | — |
| parent sample | 6,242,507 | 6,242,507 |
| DR9-trained recovery @p≥0.9 (A/B/C/ALL) | 90.0/68.9/43.8/59.6 | 90.0/68.9/43.8/59.6 |
| DR7-trained recovery @p≥0.9 (A/B/C/ALL) | 83.3/67.0/31.8/51.8 | 83.3/64.2/27.8/48.8 |

Each ResNet trains in ~12 min on MPS (≈ the L4's 25 min); the full 6.24M-galaxy DR7
sweep ran at ~1.4 bricks/s (~24 h). The DR9-trained column is reproduced exactly
(it is recomputed from the phoenix scores, validating the analysis code on Mac). The
DR7-trained column comes from an **independent from-scratch MPS retrain** whose model
landed at a marginally higher val AUC, so it is slightly more sensitive (≈+3 pp recall,
38,483 vs 25,792 candidates at p≥0.9). The two DR7 runs are strongly rank-correlated
(Spearman 0.85); the tail disagreement is the expected variance of retraining on
uniformly-random negatives (caveat 3), not a backend difference. Grade-A recall (the
most robust signal) matches phoenix exactly. **Conclusion: the Mac Studio is a sound
from-scratch train+infer workhorse for these reproductions.**

## Layout

- Base survey **inputs** are symlinked from the canonical `../data/` and
  `../../hsu-2025/data/` (one shared copy, never duplicated):
  `data/dr7_sweep`, `data/cutouts_fits_dr9`, `data/cutouts_fits_dr7_train`,
  `data/cutouts_fits_dr7`, `data/zall-pix-iron.fits`.
- All **computed outputs** are real files under `data/` (checkpoints, parquets,
  JSON, CSVs) — so the committed phoenix reference numbers in `../data/` are never
  overwritten.
- Reference baseline (phoenix checkpoints + full score parquets + committed
  CSV/JSON) lives in `data/ref/`. Paper PDFs in `data/papers/`.

## Setup

```bash
/opt/homebrew/bin/python3.12 -m venv .venv
.venv/bin/python -m pip install -U pip wheel
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -c "import torch; assert torch.backends.mps.is_available()"
```

## Sync base data from phoenix

```bash
./sync_from_phoenix.sh priority   # training cutouts, 21GB zcat, PDFs, ref ckpts/scores (~22 GB)
./sync_from_phoenix.sh bulk       # dr7_sweep 476 GB + cutouts_fits_dr7 28 GB
```
Resumable (GNU rsync `--partial-dir`); uses the `ssh phoenix` alias.

## Run (all on MPS)

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1   # also set inside device.py
PY=.venv/bin/python

$PY 02_filter_catalog.py
$PY 04_build_negatives.py
$PY 05_train_resnet.py            # DR9 training  -> checkpoint_best.pt
$PY 05b_train_resnet_dr7.py       # DR7 training  -> checkpoint_best_dr7.pt
$PY 06_write_reproduction_report.py
$PY 07_plot_training_curves.py
$PY 10_select_parent_sample.py    # 476 GB sweeps -> parent_dr7.parquet
$PY 13_extract_huang2020_catalog.py

# inference: single MPS device (--n-shards 1 is the default here)
$PY 08_smoketest_dr7.py
$PY 11b_brick_inference_dr7.py --ckpt data/checkpoint_best.pt        # ~20 h, resumable
$PY 12_merge_shards.py && mv data/inference_scores.parquet data/inference_scores_dr9trained.parquet
# (repeat 11b + merge with checkpoint_best_dr7.pt -> inference_scores_dr7trained.parquet)

$PY 14b_recovery_comparison.py
$PY 15_diagnose_missing_seven.py
$PY 16_build_inspection_viewer.py --top-n 2000 --per-page 50
$PY 17_extended_crossmatch.py
$PY verify_against_reference.py   # writes data/REPRODUCTION_MPS_COMPARE.md
```

### Orchestrators (resumable; re-run to resume)

- `run_full_inference.sh` — full sweep for BOTH checkpoints (~40 h) + analysis + verify.
- `run_dr7_sweep.sh` — full sweep for the **DR7 checkpoint only** (~24 h on this M4
  Max @ ~1.4 bricks/s), fills the DR9-trained column from the transferred phoenix
  scores, then analysis + verify. This is the configuration actually run here.

## Apple Silicon notes

- `device.py::pick_device()` selects `mps` > `cuda` > `cpu`. The 2-GPU / 2-shard
  design of `11`/`11b` is collapsed to a single MPS device (`--n-shards 1`,
  `--shard 0`); `--brick-workers` is raised to 8 because the inference bottleneck
  is NERSC brick downloads, not the GPU.
- `pin_memory` is disabled on MPS (no pinned host memory); training DataLoaders use
  `persistent_workers` + `prefetch_factor` over 120 short epochs.
- MPS↔CUDA float drift is immaterial to the headline AUC / recovery numbers;
  training stochasticity (RNG) dominates. See `verify_against_reference.py` for the
  tolerances and the same-checkpoint MPS-vs-phoenix score cross-check.
