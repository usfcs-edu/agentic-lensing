# Huang-2021 reproduction — Apple Silicon (M4 Max / MPS) port

A self-contained re-run of the `huang-2021` DESI Legacy Survey (DR8) lens-finding
pipeline on Apple Silicon, continuing the validation of the Mac Studio (M4 Max,
128 GB, MPS) as a from-scratch model workhorse begun with the sibling `huang-2020`
port. **Tier 1** runs all the *modeling* from scratch on MPS — the novel shielded
ResNet, the L18-vs-shielded comparison, and the north-augmentation retrain — plus
the leak-aware recovery analysis, and cross-checks two-model MPS inference against
phoenix. The 724 GB / ~2.5-day full DR8 deployment sweep (**Tier 2**) is left
runnable but not run (it mainly re-confirms the already-proven MPS==CUDA inference
fidelity).

The original numbered scripts in `../` are untouched. This directory copies them and
applies only the minimal edits for MPS + the nested path layout (`device.py` plus the
diff against `../`). What is unique to huang-2021 vs huang-2020: two architectures
(L18 `01` + shielded `01b`), a two-model ensemble (`11b`), the north-aug workflow
(`18` → `05c`), and the leak-aware recovery (`14`).

## Result (2026-06-02)

Full Tier-1 run on the M4 Max reproduces phoenix; `verify_against_reference.py`
passes **29/29** gated checks (`data/REPRODUCTION_MPS_COMPARE.md`). Whole pipeline
(4 from-scratch trainings + north-cutout download + bounded inference + analysis)
ran in **~1h33m** on a single MPS device.

| metric | MPS (M4 Max) | phoenix (CUDA) |
| :--- | ---: | ---: |
| shielded test AUC (DR9 / DR7) | 0.9996 / 0.9944 | 0.9988 / 0.9955 |
| L18 / shielded north-aug test AUC | 0.9998 / 0.9985 | 0.9985 / 0.9996 |
| shielded params (== L18 3,508,833 / 58.6×) | 59,905 | 59,905 |
| MPS-vs-CUDA two-model inference, p99.9\|Δ\| (max) | 3.9e-4 / 4.7e-4 (≤1.1e-3) | — |
| north non-lens score≥0.1 (pre → post north-aug L18) | 65.4% → **0.3%** | 91% → 0.8% |
| two-model recovery, all 1,312 (p≥0.1/0.5/0.9) | 83.2 / 81.8 / 76.1 % | 83.2 / 81.8 / 76.1 % |
| leak-free honest 363 (p≥0.9) | 50.4 % | 50.4 % |
| published catalog (A / B / C / total) | 216 / 199 / 897 / 1312 | 216 / 199 / 897 / 1312 |

Each shielded net trains in ~12 min on MPS; both north-aug retrains likewise. The
recovery/catalog analysis is recomputed on the Mac from the phoenix full DR8 scores
(validating the leak-aware crossmatch code), and MPS two-model inference is verified
bit-faithful to CUDA on a bounded 300-brick run (median \|Δ\| ~1e-7; the shielded
net's 1×1 "shields" are the most MPS-sensitive op, landing at p99.9 ~5e-4). The
north-augmentation false-positive collapse — the headline Phase-4b science result —
reproduces (south-trained L18 over-fires on 65% of held-out north non-lenses;
north-calibrated L18 drops to 0.3%). **Conclusion: the Mac Studio reproduces the
shielded architecture, the north-aug fix, and the leak-aware recovery from scratch.**

### Apple Silicon training gotcha (important)

`05`/`05c` set `non_blocking=pin` (i.e. `False` on MPS) for host→device transfers:
`.to(device, non_blocking=True)` on MPS races the copy against the forward pass and
yields **NaN logits from epoch 1**. It's timing-sensitive (masked by DataLoader
workers, exposed by `num_workers=0`), so a CUDA loop that ports "cleanly" can still
diverge. Also: run exactly **one** MPS process (two concurrent processes corrupt each
other), and use `PYTHONUNBUFFERED=1` (Python block-buffers stdout to a file, so a
healthy run can look hung). `run_tier1.sh` sets both.

## Layout

- Base **inputs** symlinked from the canonical `../../huang-2020/data/` (one shared
  copy): `cutouts_fits_dr9`, `cutouts_fits_dr7_train`, `positives_huang2020.parquet`,
  `positives_all.parquet`, `negatives.parquet`, `neuralens_catalog.csv`. The
  **L18 checkpoints** (`checkpoint_best{,_dr7}.pt`) symlink from the Mac-trained
  `../../huang-2020/apple-silicon/data/`.
- All **computed outputs** are real files under `data/` (the four checkpoints,
  training/test JSON, `arch_comparison.csv`, north parquets + cutouts, recovery +
  crossmatch CSVs) — the committed phoenix reference numbers in `../data/` stay
  pristine.
- `data/ref/` holds the phoenix baseline pulled for verification: the northaug
  deployment checkpoints and the full DR8 two-model score parquets.

## Setup

```bash
/opt/homebrew/bin/python3.12 -m venv .venv
.venv/bin/python -m pip install -U pip wheel && .venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -c "import torch; assert torch.backends.mps.is_available()"
./sync_from_phoenix.sh priority   # northaug ckpts + full score parquets + parent_dr8 (~2 GB) + symlinks
```

## Run (Tier 1 — all modeling + analysis on MPS)

```bash
./run_tier1.sh        # resumable; each step skips if its output exists
```
which runs: `05 --dr dr9`, `05 --dr dr7`, `06`, `07` (4a) → `18`, `05c --arch l18`,
`05c --arch shielded`, `northaug_fp_check` (4b) → bounded two-model `11b` (phoenix
northaug ckpts, 300 bricks) + `xcheck_mps_inference` → `13`, `14`,
`15 --model shielded` from the phoenix full scores → `verify_against_reference`.

### Tier 2 (deferred) — full DR8 deployment sweep

```bash
./sync_from_phoenix.sh bulk   # 724 GB dr8_sweep
./run_dr8_sweep.sh            # 10 → 11b (~298,844 bricks, ~55-60 h, resumable) → 12 → 16 → analysis
```

## Apple Silicon notes

- `device.py::pick_device()` selects `mps > cuda > cpu`. The 4-shard / 2-L4 design of
  `11b` collapses to one MPS device (`--n-shards 1 --shard 0`); `--brick-workers` is
  raised to 8 because the bottleneck is NERSC brick downloads, not the GPU. The
  two-model ensemble scores both nets per downloaded brick (doubles GPU work, not
  download cost).
- The shielded net (~60K params, four 1×1 shields) is the most MPS-numerically-
  sensitive case; the xcheck gates it explicitly (≤1e-3) and reports its delta apart
  from L18.
- macOS `spawn` portability: only `07`'s ROC DataLoader uses `num_workers=0` (its
  `LensDataset` comes from the dynamically-imported `05_train_shielded`). `05`/`05c`
  keep `num_workers=8` (their `LensDataset` is local to `__main__`).
- `pin_memory` is disabled on MPS; training DataLoaders use `persistent_workers` +
  `prefetch_factor` over the 120 short epochs.
