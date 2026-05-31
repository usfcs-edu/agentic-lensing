# Phase 5 reproduction — Inchausti 2025 ensemble + Storfer 2024 baseline

Internal reproduction of the two successor papers in the DESI Legacy Surveys
strong-lens series, building directly on the [`huang-2021`](../huang-2021)
shielded-ResNet reproduction (Phase 4):

- **Storfer et al. 2024** (Paper III, DR9, ApJS 274:16, [arXiv:2308.04603](https://arxiv.org/abs/2308.04603))
  — the *same* shielded ResNet as Huang 2021, scaled up to 1,961 training lenses
  and deployed on DR9 (~19,000 deg², 45.26M cutouts) → 1,895 candidates
  (1,512 new; A=115/B=526/C=1254).
- **Inchausti et al. 2025** (Paper IV, DR10, [arXiv:2508.20087](https://arxiv.org/abs/2508.20087))
  — the genuinely new content: a **two-architecture ensemble** combining a
  shielded ResNet and EfficientNetV2 through a feature-weighted-stacking
  meta-learner, deployed on DR10 (~14,000 deg², ~43M cutouts) → 811 new
  candidates (A=90/B=104/C=617).

The full tech-report PDF is [`papers/main.pdf`](papers/main.pdf); this is the
operator's guide.

## Scope (this reproduction)

**Ensemble + targeted recovery — not a full survey sweep.** We reproduce the
papers' central *methodological* claim — does an EfficientNetV2 + shielded-ResNet
stacking ensemble improve on a single shielded ResNet? — and recover the
published catalogues by *targeted* scoring, deliberately **without** re-running
the full ~45M (DR9) / ~43M (DR10) parent-sample sweeps (that would be two more
Phase-4-scale deployments). Concretely:

1. **Controlled architecture comparison** (the headline). Train the three models
   — the 194,433-param shielded ResNet, EfficientNetV2-S, and the 300-node
   meta-learner — on the *identical* DR9 cutouts / positives / negatives / seed /
   split as the Phase-4 baseline, and reproduce Inchausti Fig. 6
   (val AUC: ResNet 0.9984, EfficientNet 0.9987, meta 0.9989 = average 0.9989).
2. **Targeted recovery** of the published catalogues:
   - *track (i)* re-score the on-disk DR8 candidate pool (~319K cutouts kept at
     p≥0.1 in Phase 4) with the ensemble — zero new downloads;
   - *track (ii)* directly score freshly-fetched DR9/DR10 cutouts of every
     published Storfer (1,895) and Inchausti (811) candidate — the honest "would
     our reproduction have flagged this published lens" signal, leak-bucketed.

## The three models

| Model | Params (ours / paper) | Role | File |
| :-- | :-- | :-- | :-- |
| Shielded ResNet (194K) | 194,501 / 194,433 | base 1 | `01b_shielded_resnet.py` (Phase-4 arch, new config) |
| EfficientNetV2-S | 20,543,145 / 20,542,883 | base 2 | `02_efficientnet.py` (timm, pretrained, fine-tuned) |
| Meta-learner (FWLS) | 1,201 | stacking ensemble | `03_meta_learner.py` (2→300→1 MLP) |

The shielded ResNet is the same `ShieldedDeepLens` from Phase 4, instantiated at
the Inchausti parameter count via constructor args (`stage_out=52, stage_mid=32,
shield_ch=12, final_out=24` → 194,501, +68 of the published 194,433). The
EfficientNetV2-S backbone (20,177,488 params) + a `Linear(1280→285)→ReLU→
Linear(→2)` fine-tuning head lands at 20,543,145 (+262 of 20,542,883). The
meta-learner is the faithful minimal reconstruction of Inchausti's "one-layer,
300-node" feature-weighted-stacking net (Coscrato et al. 2020) over the two base
probabilities.

## Pipeline

| Script | Purpose | Phase |
| :-- | :-- | :-: |
| `01b_shielded_resnet.py`, `01_lanusse_resnet.py` | shielded + L18 models (symlinked from huang-2021) | — |
| `02_efficientnet.py` | EfficientNetV2-S grz wrapper + param smoke test | 5a |
| `03_meta_learner.py` | FWLS 2→300→1 MLP + simple-average baseline | 5a |
| `04_scaffold_symlinks.py` | data/ symlinks + EffNet weight pre-fetch + env report | 5a |
| `05_train_shielded194k.py` | train shielded ResNet @194K (`--config/--recipe`) | 5a |
| `06_train_efficientnet.py` | fine-tune EfficientNetV2 on the same split | 5a |
| `07_train_meta_learner.py` | train FWLS over base probs; stacking vs average | 5a |
| `08_compare_models.py` | controlled AUC table + Fig-6-style figure | 5a |
| `09_build_storfer_catalog.py` | published Storfer 1,895 catalog (NeuraLens) | 5c |
| `10_build_inchausti_catalog.py` | published Inchausti 811 catalog (+ per-model probs) | 5c |
| `11_rescore_dr8_ensemble.py` | track (i): ensemble re-score of on-disk DR8 cutouts | 5b |
| `12_download_candidate_cutouts.py` | DR9/DR10 cutouts of published candidates | 5b |
| `13_score_candidates_direct.py` | track (ii): score candidates with all 3 models | 5b |
| `14_crossmatch_recovery.py` | recovery by grade × model, leak-bucketed | 5c |
| `15_extended_crossmatch.py` | provenance of top-N ensemble candidates | 5c |
| `16_build_inspection_viewer.py` | paginated Lupton-RGB viewer of top-N ensemble | 5c |
| `17_assemble_literature_catalogs.py` | Stage B: harvest literature lens catalogs (VizieR) | 5B |
| `18_download_litpos_cutouts.py` | Stage B: DR9 cutouts for literature positions | 5B |
| `19_train_stageb.py` | Stage B: retrain on enlarged set + report AUC/recovery shift | 5B |
| `20_build_negatives_brick_dr9.py` | Stage C: brick-slice 45K DR9 random-galaxy negatives | 5C |
| `21_train_stagec.py` | Stage C: retrain at ~1:25 + FPR before/after | 5C |
| `22_fpr_operating_point.py` | Stage C: recovery at matched false-positive rate | 5C |

Large training inputs (cutouts, positives, negatives, baseline checkpoints, DR8
parent sample + scores) are **symlinked** from `../huang-2020` and `../huang-2021`
by `04_scaffold_symlinks.py` — nothing is re-downloaded.

## Reproducing from scratch

```bash
# 5a — models + controlled comparison (uses huang-2020/2021 DR9 cutouts; ~1 h)
./02_efficientnet.py        # smoke test: ~20.54M params
./03_meta_learner.py        # smoke test: 300-node MLP, meta ≈ average
./04_scaffold_symlinks.py   # data/ symlinks + pre-fetch EfficientNetV2 weights
# train the two base models concurrently on two A16s (NOT the L4s):
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 ./05_train_shielded194k.py --config 194k --recipe inchausti &
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 ./06_train_efficientnet.py &
wait
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 ./07_train_meta_learner.py
./08_compare_models.py

# 5c — published catalogues
./09_build_storfer_catalog.py
./10_build_inchausti_catalog.py

# 5b — targeted recovery
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 ./11_rescore_dr8_ensemble.py
./12_download_candidate_cutouts.py --catalog both
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 ./13_score_candidates_direct.py --catalog both
./14_crossmatch_recovery.py
./15_extended_crossmatch.py
./16_build_inspection_viewer.py
make -C papers pdf
```

> **GPU note.** This host has 8×A16 (15 GB, nvidia-smi indices 0–7) + 2×L4 (24 GB,
> indices 8,9). CUDA defaults to `FASTEST_FIRST`, so always export
> `CUDA_DEVICE_ORDER=PCI_BUS_ID`. Phase 5 runs on the A16s (the L4s were reserved
> for other work); the wide 194K shielded net's full-101×101 stage-1 activations
> need batch ≤128 on 15 GB, and EfficientNetV2 uses batch 128 + gradient
> accumulation. All steps fit comfortably in 15 GB.

## Caveats

1. **No code released** by either paper — reconstructed from the methods sections.
2. **Param-count matches are reconstructions.** EfficientNetV2 head width and the
   shielded net's intermediate channels are unpublished; our fits land within
   0.0013% (EffNet) and 0.035% (shielded) of the published counts.
3. **EfficientNetV2 input.** The paper does not specify the variant or input
   adaptation. We use EfficientNetV2-S at the native 101×101 FoV with the same
   per-band (astronomical, not ImageNet) normalisation as the ResNet so all three
   models share one Dataset.
4. **Training-set leakage (carried from Phase 4).** Our positives are the
   NeuraLens Huang 2020/2021 candidates, which overlap the Storfer/Inchausti
   literature positives. Recovery is reported separately for leaked vs leak-free
   candidates; the leak-free numbers are the honest signal. **Stage B**
   (`17`–`19`) harvested 6,302 literature lens positions (11 VizieR catalogs;
   5,561 in the DR9 footprint), sampled to the Storfer scale (1,961 positives),
   and retrained: the enlarged set **lowers** the internal test AUC
   (0.9999→0.9881, the narrow Stage-A test was easiness-inflated) but **improves**
   recovery of the independent published catalogs (Storfer 90.7%→93.5%, Inchausti
   93.2%→96.9%) — more diverse positives generalize better to held-out discoveries.
   **Stage C** (`20`–`22`) addresses the biggest remaining gap: the
   negative:positive ratio. We brick-sliced **45,000** random DR9 galaxies as
   negatives (300 brick coadds, ~4 min, vs ~8 h via the cutout endpoint) and
   retrained at ~1:25 (toward Storfer's ~1:33). The AUC barely moves (meta
   0.9876), but on 4,991 held-out random galaxies the Stage-B models flag
   37–51% as lenses at p≥0.5 — so Stage-B's "93–97% recovery" was at a
   meaningless threshold. At a matched **1% false-positive rate**, meta recovery
   of the published catalogs goes from 11.8%/19.1% (Stage B) to **83.6%/88.5%**
   (Stage C, Storfer/Inchausti). The negative ratio — not the architecture or
   AUC — sets whether the finder is usable at a real operating point.
5. **Meta-learner training.** As in the paper, the meta-learner is trained on the
   base models' in-sample probabilities; we report held-out test AUC and a
   simple-average baseline (the correlated bases make stacking ≈ averaging).
