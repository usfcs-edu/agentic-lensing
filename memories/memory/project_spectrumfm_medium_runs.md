---
name: project-spectrumfm-medium-runs
description: SpectrumFM Track 3 medium-real local runs — data layout, headline numbers, what's reproducible vs what needs NERSC, non-obvious gotchas
metadata:
  type: project
---

# SpectrumFM medium-real local training — Track 3 completion

Lives at `/raid/benson/data/desi_dr1_medium/` (9.1 GB) + `experiments/runs/*medium*/`. The Track 2 harness (`[[project-spectrumfm-tooling]]`) drives all three trainings end-to-end on the aarch64 box.

**Why:** Track 3 first project from `plans/SPECTRUMFM_NEXT_STEPS.md`. Goal was to prove the pipeline works at medium scale before any architectural integration (Track 1). It does — code paths are exercised, metrics flow through the harness, all three models learn from random init. Absolute targets (`val_recon ≤ 2`, `z_acc ≥ 30%`, `val_r2 > 0.5`) were missed by data-proportional margins; the pipeline is correct, the dataset is just 33× smaller than the redshifty author's 200-healpix baseline.

**How to apply:** when proposing a Track 1 (codecs ↔ redshifty integration) experiment or any "validate locally before NERSC" idea — these numbers are the local floors you're comparing against.

## Headline numbers (2026-05-26)

| Run | wallclock | data | best metric |
|---|---|---|---|
| V1 tokenizer medium | 44 min | 11.5k spectra / 33 px | `val_recon` 7.07 → 4.08 (step 4250) |
| codecs medium | 13 min | 7.1k qcut / RFSQ Mamba3 | `val_r2` 0.20 → 0.44; `val_nll` 23.5 → 2.85 |
| Approach A medium | 15 min | 11.5k via V1-medium tokenizer | `val_redshift_acc` peak 3.8%; `val_loss` 278 → 237 |
| **V1 tokenizer large** | **3.5 h** | **219k spectra / 154 px** | **`val_recon` 7.07 → 1.38 (step 13000) — matches NERSC V1 baseline (1.35)** |
| **codecs large** | **28 min** | **139.9k qcut** | **`val_r2` 0.27 → 0.46; `val_perplexity` 0.06 → 0.38 (codebook diversified)** |
| **Approach A large** | **74 min** | **219k via V1-large tokenizer** | **`val_spectrum_acc` 31% → 41% (peak), beats NERSC ~30%; `val_redshift_acc` peak 3.9% — STILL no ignition** |

Large scale (2026-05-27) brought the tokenizer to NERSC-baseline parity (val_recon=1.38 vs author's 1.35 on half the data) and pushed Approach A's spectrum reconstruction above the author's NERSC number. **But Approach A's redshift pathway still failed to ignite at `weight=50, mask=0.50, batch=8, lr=2e-4`** — the cross-attention-copy mechanism the author saw at step ~6500 didn't fire even at 10k steps + 19× more data than our medium run. See "Hypothesis on missing ignition" below.

## Non-obvious facts

1. **`redshifty/nersc/dr1_dataset.py:101–102` opens FITS with `memmap=True`**, which astropy refuses for the DESI coadd files because the B/R/Z mask images have BZERO/BSCALE/BLANK headers. The crash is silent inside the DataLoader worker and surfaces as "Cannot load a memory-mapped image: BZERO/BSCALE/BLANK header keywords present" only when the first batch tries to read mask data. **Fix lives in the local working tree**; if redshifty is re-cloned, re-apply `memmap=False` to both lines (the `# NOTE: memmap=False — DESI coadd FITS files have BZERO/BSCALE...` comment marks the spot).

2. **EDR/fuji SV3-bright healpix is dominated by large coadds.** 47 of 80 pixels exceed 200 MiB; only 20 are smaller. Local download at 500 MiB cap yields ~25% pass-through; ~9 GB total disk for 33 pixels / 11.5k spectra. Lifting the cap to 1 GB roughly doubles disk and triples spectra.

3. **DESI public mirror download is fast.** 33 healpix coadd+redrock pairs in ~10 seconds. I/O is not the bottleneck; FITS parsing during pretrain_tokenizer.py's first batch is (~2 minutes wallclock from "params=24.3M" to first metric line at step 0).

4. **Codec Mamba3 first-step CUDA autotune is ~2.5 minutes** even with `TORCHDYNAMO_DISABLE=1`. Inductor cache survives across runs; second-and-later runs start much faster (~few seconds for first step).

5. **Both `pretrain_tokenizer.py` and `train_transformer.py` print metrics as `[step N] k=v k=v ...`** with optional `[AR]` tag on AR-train steps. The Track 2 parser now handles both via a single permissive regex (`RED_PRETRAIN_STEP_RE`) — see `[[project-spectrumfm-tooling]]` fact 6. Originally the parser was specific to the pretrain triple `(loss, recon, quant)`; refactored to harvest every `key=val` and prefix with `train_`/`val_`.

6. **Build a synthetic iron zcatalog from EDR fibermaps** — codecs/scripts/data.py needs a zcatalog at `zcatalog/v1/zall-pix-iron.fits` with columns ZCAT_PRIMARY, COADD_FIBERSTATUS, OBJTYPE, SURVEY, PROGRAM, HEALPIX, TARGETID. The EDR/fuji data doesn't ship this catalog (DR1's iron release does, 22 GB), so we manufacture one from each coadd's FIBERMAP HDU. Inline script in `plans/SPECTRUMFM_TRACK3_COMPLETION.md`. ~250 rows are filtered out per pixel by `COADD_FIBERSTATUS != 0`; 11,483 catalog rows → 7,125 cache rows.

7. **Worker scaling tuning required.** `pretrain_tokenizer.py` defaults to `--num-workers 4`. On L4 + 4 dataloader workers we get 1.9 step/s sustained. `train_transformer.py` at `--num-workers 4 --batch-size 8` gets 3.0 step/s — faster because each step processes less data per worker. Increasing workers above 4 didn't help (FITS parsing is the limit).

8. **GPU allocation pattern that works:** tokenizer on `cuda:8` (L4), codecs on `cuda:9` (L4) — they run in parallel without I/O contention because the codecs dataset is an HDF5 cache (memory-mapped, no FITS parsing) and the tokenizer/transformer datasets parse FITS at runtime. Disk doesn't saturate.

## Approach-A ignition diagnostic — current best understanding (2026-05-27)

The original "weight=50 too low" hypothesis was wrong. An Explore-agent diagnostic + a Phase-10-match rerun pinned the actual issue as **two compounding causes**:

**Cause 1 (necessary, confirmed): hparam mismatch in `_large` spec.** The `_large` spec accidentally combined Phase 9's `batch=8 lr=2e-4` (which the author had only used with `mask=0.0`) with Phase 10's `mask=0.50`. Author's Phase 10 final — the run with ignition — used `batch=32 lr=4e-4` (sqrt-scaled for 4× batch). Fixing this in the `_phase10` spec doubled peak z_acc 3.9% → **8.2%** and dropped val_loss min from 225 → 199.

**Cause 2 (TESTED 2026-05-28, RULED OUT): pure data scale.** Pulled the full sv3-bright tree (373 pixels / 729k spectra — 1.85× author's 394k) and reran `_phase10` config. Result: peak val_z_acc=6.18% (slightly WORSE than the small-data 8.2%); val_loss_redshift descended 4.87→3.96 (same descent rate); val_spec_acc climbed to 45% (best so far). **Spectrum side benefits from more data, redshift side does not.** Honest AR z_acc (now n=194) still at noise floor (~4.6% peak).

**Cause 3 (TESTED 2026-05-28, RULED OUT): seed luck.** Ran 3 additional seeds {1337, 7, 12345} parallel to existing seed=42 on the xlarge data. Per-seed peak z_acc: 6.18 / 8.76 / 3.76 / 6.84 — variance ~1–3 pp std, much smaller than the 70 pp gap to author's 73.8%.

**Cause 4 (TESTED 2026-05-29, CONFIRMED): data-mix diversity.** Pulled the missing sv3-dark + main-bright + main-dark (using `download_desi_subset.py --release {edr,dr1}` after extending the script to support DR1/iron + main survey). Combined manifest: 1137 pixels / 1.82M raw spectra / ~750 GiB. **The mix run is the first to ignite:** val_z_acc 5.5 → 8.1 → 9.2 → 10.85 → 14.86 across steps 6500–9500; val_loss_redshift 4.88 → 3.69 (cumulative drop 1.19); AR z_acc 7.96% at step 9000 (ratio 0.73 of TF, matches author's NERSC 0.74). All three pass criteria from the plan met. val_loss min 190.67 (best of any local run).

**Diagnostic conclusion:** the author's NERSC ignition is reproducible on local hardware with three things together:
1. **Correct hparams** (batch=32, lr=4e-4, mask=0.50, weight=50) — Phase 10 final config exactly.
2. **Data MIX, not just scale** — sv3+main × bright+dark, all four survey×program combinations.
3. **Frozen tokenizer at val_recon ≤ ~1.4** — `tokenizer_v1_large/best.pt` at val_recon=1.38 is enough.

The hidden requirement that caused months of frustration in our diagnostic was #2 — author's `200 healpix files (sv3+main, bright+dark)` was buried in RESEARCH_LOG.md and we mistook it as 200 sv3-bright pixels. Every prior local run was sv3-bright only, which produces good spectrum reconstruction (spec_acc 40%) but never igniting redshift.

**Cheaper-but-less-likely-to-help alternative:** my original `--redshift-loss-weight 100` arm. The hparam fix already moved peak z_acc 2.1× so the weight is probably fine; data is the bigger lever.

**Definitively ruled out as the primary cause:** weight=50 being too low (now-confirmed via Phase-10 rerun that fixed hparams without changing weight).

The honest AR z_acc was unreadable at n=94 samples (1–4 correct ≈ pure noise). If we want a real AR signal we'd need to bump `--ar-eval-batches` to ~32+ or refactor the eval to use the full val set.

## Spec lineage (which is canonical)

- `redshifty_approach_a_phase10_mix.yaml` — **canonical Track 1 baseline (2026-05-29).** batch=32, lr=4e-4, mask=0.50, weight=50, **mix manifest** (sv3+main bright+dark, 1.82M spectra). **First local run to ignite redshift.** Peak val_z_acc 14.86% at step 9500, val_loss min 190.67, AR 7.96% at step 9000 (ratio 0.73 of TF, matches author Phase 10 final 0.74). Use for any head-to-head comparison vs codecs tokenizer.
- `redshifty_approach_a_phase10_xlarge.yaml` — same hparams on sv3-bright-only 729k. Best val_loss 203.3, peak z_acc 6.18% (no ignition). Demonstrates that more data alone doesn't fix it.
- `redshifty_approach_a_phase10.yaml` — same hparams on 219k sv3-bright. Best val_loss 199.2, peak z_acc 8.2%.
- `redshifty_approach_a_large.yaml` — historical, cautionary. Wrong hparam combination (Phase 9 batch/lr + Phase 10 mask). Don't use unless reproducing the diagnostic.
- `redshifty_approach_a_medium.yaml` — historical, smallest data run.

## Where the checkpoints live

- `desi_dr1_medium/checkpoints/tokenizer_v1_medium/best.pt` — 292 MB w/optim, step 4250 val_total=4.197 (medium)
- `desi_dr1_medium/checkpoints/tokenizer_v1_medium/final.pt` — 97 MB state-dict (medium)
- `desi_dr1_medium/checkpoints/tokenizer_v1_large/best.pt` — 292 MB w/optim, **step 13000 val_total=1.7047 (matches NERSC V1)**
- `desi_dr1_medium/checkpoints/tokenizer_v1_large/final.pt` — 97 MB state-dict (large)
- `desi_dr1_medium/checkpoints/approach_a_medium/` + `approach_a_large/` — analogous; neither hit ignition
- `desi_dr1_medium/codecs_output/model.pt` — codecs at step 2000 (medium), val_r2=0.44
- `desi_dr1_medium/codecs_output_large/model.pt` — codecs at step 5000 (large), val_r2=0.46

The **`tokenizer_v1_large/best.pt`** is the recommended frozen tokenizer for Track 1 (codecs ↔ redshifty integration) — it sits at the NERSC baseline.

Related: [[project-spectrumfm]], [[project-spectrumfm-tooling]], [[reference-spectrumfm-local-env]].
