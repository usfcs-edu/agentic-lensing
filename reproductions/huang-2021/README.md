# Huang 2021 reproduction (Phase 4)

Internal reproduction of \[Huang et al. 2021, *Discovering New Strong
Gravitational Lenses in the DESI Legacy Imaging Surveys*, ApJ 909:27,
[arXiv:2005.04730](https://arxiv.org/abs/2005.04730)\]. Builds directly on the
[`huang-2020`](../huang-2020) reproduction (Phases 3a + 3b). The full tech-report
PDF lives at [`papers/main.pdf`](papers/main.pdf); this is the operator's guide.

## What Huang 2021 adds over Huang 2020

1. **The "shielding" architecture** — the headline novelty. The Lanusse-2018
   (L18) ResNet gets 1×1 conv "shield" layers inserted between every three
   residual blocks, and the final block reduced to 32 channels. This cuts
   trainable parameters ~50× (3.5M → ~60K) while *raising* validation AUC
   (paper: 0.992 → 0.997). See `01b_shielded_resnet.py`.
2. **DR8 deployment over ~14,000 deg²**, adding the northern BASS/MzLS footprint
   (δ ≳ +32°) that Huang 2020's DECaLS-only DR7 search lacked. ~22M cutouts
   scored at a low threshold (p ≥ 0.1) with a **two-model ensemble** (L18 +
   shielded).
3. **1,312 published candidates** (216 A + 199 B + 897 C), 1,210 of them new.

## Phase 4a — shielded architecture (controlled comparison)

We isolate the architecture as the only variable: the shielded net is trained on
the *same* cutouts / positives / negatives / seed / split as the L18 baseline
from the huang-2020 reproduction, for both DR9 and DR7 cutouts.

`01b_shielded_resnet.py` smoke test (verified): **59,905 params, 4 shields,
15 residual blocks, 58.6× reduction** vs L18's 3,508,833, with the final-channel
sweep ladder (512→…→16) reproducing the paper's experiment.

| Run | params | val AUC | test AUC |
| :-- | --: | --: | --: |
| L18 / DR9 (huang-2020) | 3,508,833 | 0.9983 | 0.9991 |
| L18 / DR7 (huang-2020) | 3,508,833 | 0.9890 | 0.9943 |
| **shielded / DR9** | **59,905** | **0.9989** | **0.9988** |
| **shielded / DR7** | **59,905** | **0.9875** | **0.9955** |

The shielded net **matches the L18 baseline to within ±0.001–0.002 AUC at 59×
fewer parameters** — reproducing the paper's central claim (shielding cuts
parameters ~50× without hurting AUC; on DR9 the shielded val AUC actually edges
L18, +0.0006). Our absolute AUCs (~0.99) run higher than the paper's 0.992/0.997
because of the documented training-set leakage; the *relative* L18-vs-shielded
result is the faithful reproduction. `06_compare_architectures.py` builds the
table + the param-vs-AUC figure (`papers/figures/arch_comparison.png`).

## Phase 4b — DR8 deployment (both models, north + south)

DR8 sweeps live in two regions on the NERSC portal:
`dr8/south/sweep/8.0/` (437 files, DECaLS/DECam) and `dr8/north/sweep/8.0/`
(286 files, BASS+MzLS). Each parent-sample row is tagged with a `footprint`
column so brick images are pulled from the matching `dr8/{south,north}/coadd/`.
Both the L18 and shielded checkpoints score every brick in a single download
pass (`11b_brick_inference_dr8.py`).

## Phase 4c — recovery + leakage

`13` rebuilds the published 1,312-candidate catalog from the NeuraLens release
table (reproduces Table 3 **exactly**: 216 A / 199 B / 897 C; 185 MzLS-north).
`14` cross-matches both models against it, separating the **leaked** ~949
training positives from the **leak-free** 363 shielded-model discoveries — the
honest test of whether the architecture generalises.

## Pipeline

| Script | Purpose | Phase |
| :-- | :-- | :-: |
| `01_lanusse_resnet.py` | L18 ResNet (copied from huang-2020) | 4a |
| `01b_shielded_resnet.py` | **Shielded ResNet** (`ShieldedDeepLens` + `Shield`) | 4a |
| `05_train_shielded.py` | Train shielded net (`--dr {dr9,dr7}`) on existing cutouts | 4a |
| `06_compare_architectures.py` | L18-vs-shielded AUC table + param/AUC figure | 4a |
| `07_plot_training_curves.py` | Curves + shielded ROC + architecture schematic | 4a |
| `08_smoketest_dr8.py` | Gate: score known south+north lenses; verify north routing | 4b |
| `09_download_dr8_sweeps.py` | DR8 south+north sweep catalogs (~1.5 TB) | 4b |
| `10_select_parent_sample_dr8.py` | z<20 + DEV/COMP + NOBS≥3 cuts + footprint tag | 4b |
| `11b_brick_inference_dr8.py` | **Footprint-aware two-model brick sweep** | 4b |
| `12_merge_shards.py` | Merge per-shard, per-model score parquets | 4b |
| `13_extract_huang2021_catalog.py` | Build the 1,312-candidate published catalog | 4c |
| `14_crossmatch_recovery_dr8.py` | Recovery by grade × model × threshold + leakage split | 4c |
| `15_extended_crossmatch.py` | Provenance of top-N (leak / published / new) | 4c |
| `16_build_inspection_viewer.py` | Paginated Lupton-RGB viewer of top-N per model | 4c |

Large training inputs (cutouts, positives, negatives, L18 checkpoints, NeuraLens
catalog) are **symlinked** from `../huang-2020/data/` — not re-downloaded.

## Reproducing from scratch

```bash
# Phase 4a — architecture (uses huang-2020 cutouts; ~30 min)
./01b_shielded_resnet.py                                   # smoke test
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=8 ./05_train_shielded.py --dr dr9 &
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=9 ./05_train_shielded.py --dr dr7 &
wait
./06_compare_architectures.py
./07_plot_training_curves.py

# Phase 4b — DR8 deployment (~1.5 TB download + ~20-35 h two-model inference)
./09_download_dr8_sweeps.py --workers 6
./10_select_parent_sample_dr8.py
./08_smoketest_dr8.py                                       # gate (north routing)
# 4 shards across the 2 L4s (2 procs/GPU); each downloads bricks once, scores both nets
for s in 0 1; do CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=8 ./11b_brick_inference_dr8.py --shard $s --n-shards 4 --gpu 0 & done
for s in 2 3; do CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=9 ./11b_brick_inference_dr8.py --shard $s --n-shards 4 --gpu 0 & done
wait
./12_merge_shards.py

# Phase 4c — recovery + report
./13_extract_huang2021_catalog.py
./14_crossmatch_recovery_dr8.py
./15_extended_crossmatch.py --model shielded
./16_build_inspection_viewer.py --model shielded
make -C papers pdf
```

> **Note on GPU device ordering.** This host has 8×A16 + 2×L4. CUDA defaults to
> `FASTEST_FIRST` ordering, so `CUDA_VISIBLE_DEVICES=8` does *not* select
> nvidia-smi's GPU 8. Always export `CUDA_DEVICE_ORDER=PCI_BUS_ID` so device
> indices 8,9 map to the two L4s.

## Caveats

1. **No Huang code released** — from-scratch off the published methodology.
2. **Shield channel counts are a reconstruction.** The paper gives only "1×1
   between every three blocks", "final block 32 best", and "~60K params"; our
   `SHIELD_CH=16` lands at 59,905 — the closest principled fit.
3. **Training-set leakage.** Our nets train on the 949 L18-model NeuraLens rows,
   which are themselves Huang+2021 candidates. The honest recovery metric is on
   the 363 shielded-model discoveries that were never in training.
