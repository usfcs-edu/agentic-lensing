---
name: project-huang-2020-reproduction
description: Phase 3a reproduction of Huang+2020 DECaLS ResNet lens finder — surprising facts about architecture, training data, and AUC parity
metadata:
  type: project
---

# Huang et al. 2020 (arXiv:1906.00970) reproduction — Phase 3a

Lives at `/raid/benson/git/agentic-lensing/reproductions/huang-2020/` (scripts
01-06, mirroring foundry-i + hsu-2025 numbered pattern). Trains a PyTorch port
of the Lanusse 2018 *CMU DeepLens* ResNet-46 from paper + public code only —
NO Huang training code, NO Bologna challenge data. Test AUC = 0.9991 vs
Huang+2020 published 0.98.

**Why:** Phase 3 of the 8-phase Huang-group reproduction roadmap is the
ResNet/EfficientNet lens-finder lineage. Phase 3a = baseline ResNet on real
DECaLS, validating the pipeline. Phase 3b = full DECaLS deployment + recover
the 335 candidates (deferred). Per-paper plan: see
[[project-foundry-i-reproduction]] and [[project-hsu-2025-reproduction]] for
the surrounding context.

**How to apply:** when extending to Huang 2021 (shielded ResNet) or
Inchausti 2025 (dual ResNet + EfficientNetV2 + meta-learner), the
architecture, training loop, and data-loading infrastructure in this
directory are directly reusable.

## Non-obvious facts

1. **AUC 0.99+ is reachable in 25 minutes on a single L4** with the Lanusse
   ResNet, 949 positives + 5000 negatives, batch=128, 120 epochs. The
   architecture is small (3.5 M params) and converges fast on 101×101 grz.
   The plan-doc estimate of "weeks on NERSC Perlmutter" was 1000× too
   conservative for the *training* step alone; deployment-time inference on
   millions of DECaLS DEV/COMP galaxies is the actual long pole.

2. **The NeuraLens catalog has 1,312 rows split L18 / shielded.** A common
   misread of Huang+2020 is "335 candidates" — that's the *new* candidate
   count, not the training-positive count. The NeuraLens release table
   (`drive.google.com/file/d/1_KbEHWhl8LeeTyXpXkWFbLRxt6o42wBg`) has 949
   L18-model rows (Huang 2020-era) + 363 shielded rows (Huang 2021-era).
   For Phase 3a training we use all 949 L18 rows as positives; this is the
   group's *output* used as our *input*, which is acceptable because Phase 3a
   is about reproducing the methodology + AUC, not the candidate discovery.

3. **DR1 fibers make excellent fast non-lens training negatives.** Sampling
   5000 random galaxies from the already-downloaded
   `reproductions/hsu-2025/data/zall-pix-iron.fits` with filter
   (ZWARN=0, !STAR, Z>0.05, Dec in [-68, +32.5]) gives realistic DECaLS-
   footprint galaxies and avoids needing to query NOIRLab DataLab or
   download Tractor sweeps. Lens contamination is statistically negligible
   (lens rate ~10⁻⁴ per Huang §3.1). We exclude 10″ around any positive.

4. **Lanusse 2018 ResNet-46 architecture is 5 stages × 3 pre-activated
   bottleneck blocks** (1×1 → 3×3 → 1×1), ELU activation, channel
   progression 32 → 32 → 64 → 128 → 256 → 512 with /2 downsample at every
   stage except the first. Initial 7×7 conv + ELU + BN; AdaptiveAvgPool +
   FC(512→1) + sigmoid head. 3,508,833 params at 3-channel input. See
   `01_lanusse_resnet.py` for the PyTorch port; smoke-test at input sizes
   45×45 (L18 native) and 101×101 (Huang 2020) both work.

5. **FITS columns are big-endian → torch is little-endian** — already
   documented in [[project-hsu-2025-reproduction]] but worth re-emphasizing:
   wrap every cutout-read in `np.ascontiguousarray(arr).astype(np.float32)`.
   `astropy.io.fits` returns `>f4` (big-endian float32); pyarrow can't
   accept big-endian arrays. The Lanusse-2018 input pipeline silently
   ignores this and trains on byte-swapped data with awful AUC.

6. **DECam pixel scale is 0.262″/px**, not the 0.27 that Phase 2's Hsu
   reproduction used. For Huang-2020-faithful cutouts use
   `pixscale=0.262`. At 101×101 the FoV is 26.5″, comfortable for typical
   Einstein-radius systems (1-3″).

7. **`ls-dr9` is the correct layer name** for the modern DECaLS imaging
   accessible via `legacysurvey.org/viewer/fits-cutout`. Huang+2020 used
   DR5; the morphology of DEV/COMP elliptical galaxies has not changed,
   only the photometric depth and reductions. DR9 is the closest available
   analogue.

## Layout

```
reproductions/huang-2020/
  01_lanusse_resnet.py            # PyTorch port + smoke test
  02_filter_catalog.py            # parse NeuraLens CSV → positives parquet
  03_download_decals_cutouts.py   # resumable ls-dr9 cutout puller
  04_build_negatives.py           # sample 5000 DR1 galaxies as non-lenses
  05_train_resnet.py              # Adam(1e-3, /10 @ 40 epochs), 120 epochs
  06_write_reproduction_report.py # papers/REPRODUCTION.md template
  data/cutouts_fits_dr9/          # 929 + 4908 = 5837 grz FITS cutouts (gitignored)
  data/positives_huang2020.parquet
  data/negatives.parquet
  data/training_history.json      # per-epoch loss/AUC/lr
  data/test_result.json           # final AUC numbers
  data/checkpoint_best.pt         # 14 MB ResNet state-dict (gitignored)
  papers/REPRODUCTION.md
```

New venv: `/raid/benson/.venvs/lensfinder` (torch 2.12 cu126 + torchvision +
astropy stack; needs ~5 GB; lives on /raid not /home to dodge quota).

## Out of scope (Phase 3b)

- Full DECaLS deployment: query Tractor sweeps for all DEV/COMP galaxies in
  DECaLS footprint (~3 × 10⁶ rows), pull cutouts, infer ResNet scores, rank,
  and visually inspect the top scorers to recover the 335 published
  candidates.
- Curated by-eye hard negatives (Huang+2020 §3.1 mentions spirals, galaxy
  groups, cosmic rays, unusual configs, artifacts) — would lift Phase 3a's
  trivial 0.999 AUC into the regime where the published 0.98 was actually
  challenged.
- Comparison against Huang's own code once obtained via collaboration.
