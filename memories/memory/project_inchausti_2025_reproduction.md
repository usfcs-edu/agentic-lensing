---
name: project-inchausti-2025-reproduction
description: Phase 5 — Storfer 2024 (DR9) + Inchausti 2025 (DR10 two-architecture ensemble) reproduction; EfficientNetV2 + shielded ResNet + meta-learner
metadata:
  type: project
---

# Phase 5 — Storfer 2024 + Inchausti 2025 reproduction (DONE 2026-05-30)

Lives at `/raid/benson/git/agentic-lensing/reproductions/inchausti-2025/`. The
successor to [[project-huang-2021-reproduction]] in the lens-finder lineage
(Papers I–IV = Huang 2020 / Huang 2021 / Storfer 2024 / Inchausti 2025). Training
inputs (DR9 cutouts, positives, negatives, baseline checkpoints, DR8 parent
sample + scores, published Huang-2021 catalog) are **symlinked** from the
huang-2020/huang-2021 dirs by `04_scaffold_symlinks.py` — nothing re-downloaded.

## Scope (user-chosen): "Ensemble + targeted recovery", NOT a full sweep
Reproduce the methodology (the ensemble + its AUC behaviour) and recover the
published catalogs by **targeted** scoring; deliberately NO full DR9/DR10
parent-sample sweep (~45M/~43M cutouts = two more Phase-4 deployments).
Training set (user-chosen): "reuse existing first, then literature" — Stage A
reuses the Huang-2020/2021 NeuraLens positives (leaky, documented); Stage B
(`17_…`, literature-catalog enlargement via VizieR) still PENDING.

## The two-architecture ensemble (Inchausti 2025, Paper IV, arXiv:2508.20087)
Three models, all trained on the IDENTICAL DR9 split (SEED 2026, build_split):
- **Shielded ResNet @194K**: the Phase-4 `ShieldedDeepLens` at a new config
  `stage_out=52, stage_mid=32, shield_ch=12, final_out=24` → **194,501** params
  (paper 194,433; +68 — closest 4-shield/15-block fit, no model-file edit).
- **EfficientNetV2-S** (`02_efficientnet.py`, timm `tf_efficientnetv2_s`,
  pretrained backbone 20,177,488 + `Linear(1280→285)→ReLU→Linear(→2)` head) →
  **20,543,145** params (paper 20,542,883; +262). grz fed as 3 channels at native
  101×101; per-band astronomical normalisation (NOT ImageNet) so all 3 models
  share one Dataset.
- **Meta-learner** (`03_meta_learner.py`): faithful FWLS (Coscrato 2020,
  arXiv:1906.09735) = `Linear(2→300)→ReLU→Linear(→1)`, **1,201** params, input =
  the two base probabilities. Simple-average baseline alongside.

## Headline result — controlled comparison reproduces Inchausti Fig. 6
On the identical split (our val AUC, vs paper): ResNet 0.9992 (0.9984), EffNet
0.9989 (0.9987), **meta 0.9996 = average 0.9996** (paper 0.9989=0.9989). The
**meta-learner ≈ simple average** (Δ=+0.00002 val) — exactly the paper's finding:
the two bases are trained on the same data so they are correlated, and stacking
buys ~nothing over averaging. Our absolute AUCs run high (leakage), as in
Phases 3–4; the relative result is the faithful reproduction. Phase-4a 60K
shielded (0.9988) and L18 (0.9991) sit on the params-vs-AUC curve.

## Targeted recovery (honest, leak-free, direct scoring of published cutouts)
`12` fetched DR9/DR10 cutouts of all published candidates (1895 Storfer + 811
Inchausti, 0 fail); `13` scored them; `14` did recovery + leakage. All-grade meta
recovery @p≥0.5: **Storfer 90.7%, Inchausti 93.2%** (all 811 Inchausti are
leak-free DR10 discoveries; EffNet is the single strongest recoverer at 92.6 /
94.2%). Track (i): `11` re-scored the 319,015 on-disk DR8 cutouts (Phase-4 p≥0.1
pool) with the ensemble — no new downloads.

## Non-obvious facts
1. **Published catalogs come from NeuraLens Google Sheets** (not VizieR/arXiv).
   `09`/`10` download them (Storfer combined Drive CSV; Inchausti per-grade CSV
   export) and reproduce the grade counts EXACTLY (115/526/1254=1895;
   90/104/617=811). The Inchausti sheet uniquely ships the published per-model
   probabilities (ResNet/EffNet/meta) and RA/Dec are parsed from the legacysurvey
   viewer URLs (DESI name is only 4-decimal).
2. **corr(our scores, published Inchausti scores) ≈ 0.08** — LOW, but it's a
   range-restriction artefact (the published set is censored to high-prob
   objects, so both score distributions pile up near 1.0), NOT a disagreement.
   The agreement that matters is recovery (does our model flag it), which is high.
3. **Ran on the 8×A16 GPUs (indices 0–7), not the L4s** — the user reserved the
   L4s; see [[reference-host-hardware]] (`CUDA_DEVICE_ORDER=PCI_BUS_ID`). The wide
   194K shielded net's full-101×101 stage-1 needs batch ≤128 on 15 GB (OOM at
   256); EffNet uses batch 128 + grad-accum. Both base models train in <1 h each.
4. Storfer 2024 (Paper III, DR9, ApJS 274:16, arXiv:2308.04603) adds NO new
   architecture — same shielded ResNet as Huang 2021; reproduced as the in-house
   single-model baseline inside the Inchausti ensemble study (one shared dir).

Tech report `papers/main.pdf` (6 pp, natbib like [[project-huang-2021-reproduction]]).
venv `/raid/benson/.venvs/lensfinder` (timm 1.0.27 + torch 2.12).
