# ClaudeNet — improving ML strong-lens finding beyond ResNet/EfficientNet

ClaudeNet is a research program that improves machine-learning **strong gravitational
lens finding** on DESI Legacy Survey *grz* imaging, building directly on the
Huang-group lineage reproduced in this repo (`huang-2020/21`, `storfer-2024` /
`inchausti-2025`, `silver-2025`) and reusing the `aion-1` omnimodal foundation-model
harness.

## The premise (why not just a bigger backbone)

The repo's own reproductions proved a hard negative result: **at the deployment
operating point, architecture is not the bottleneck.** A 194 K-param shielded ResNet
≈ a 20.5 M-param EfficientNetV2-S within ±0.003 AUC, and the published
Inchausti-2025 **meta-learner collapsed to a simple average** because its two base
models were trained on byte-identical data, so their errors are correlated and a
combiner has nothing to exploit. A deep literature survey (8 technique families,
adversarially fact-checked) converged with this: no transformer / SSL / equivariant
/ foundation method cleanly beats a well-tuned EfficientNet at matched FPR for DECaLS
lens finding. The demonstrated wins are all on the axes the repo diagnosed but never
fixed — **ensemble diversity, negative quality / operating-point calibration, label
efficiency, domain shift, uncertainty.** ClaudeNet attacks those.

**Primary metric everywhere: recovery @ matched false-positive rate** (1 % and
0.1 % FPR), reusing `inchausti-2025/22_fpr_operating_point.py`'s exact arithmetic so
every number is directly comparable to the reproduced baseline. The honest baseline
to beat is the reproduced Stage-D Inchausti **meta-learner** (storfer@1% 0.908,
@0.1% 0.755; inchausti@1% 0.968, @0.1% 0.845).

## Results by phase

| Phase | Direction | Headline result |
|---|---|---|
| 0 | AION decorrelation gate | **VALIDATE** — AION↔EfficientNet score correlation Pearson **0.10** (vs repo's ~1.0); a learned combiner already edges the best member |
| 1 | **Engineered-diversity ensemble** | **SHIP** — beats the published meta-learner on 4/4 matched-FPR metrics |
| 2 | Hard-negative mining | QUALITY-HELPS (modest): hard − random = +0.008 storfer@1%, **+0.031 inchausti@1%** at fixed count |
| 3 | Label-efficiency curves | AION frozen-probe wins **only <150 labels** (5 %: 0.344 vs 0.279); supervised CNN dominates ≥300 labels (0.862 vs 0.529) and at full labels (0.894 vs 0.659) |
| 4 | Conformal selection | Certified FDR control: empirical FDR ≤ nominal at every α (.05→.03, .25→.16), completeness .96–.999 |
| 5 | Domain adaptation (north↔south, MMD) | **Negative result**: real domain gap confirmed (north@1% 0.688 vs south 0.874, −0.186), but naive MMD (λ=1) *hurt* both (−0.06) — over-regularizes |
| 6 | Uncertainty + triage | Deep-ensemble disagreement → selective error 0.022 → **0.0002 @50 % coverage** |
| 7 | Equivariance | Test-time D4 pooling lifts storfer@1% +0.032–0.049 (mean **+0.041**) |

### Phase 1 — the flagship (recovery @ matched FPR)

Five deliberately **decorrelated** members (one per GPU, independent jobs): three
EfficientNet-family backbones (EfficientNetV2-S ×2 + EfficientNet-B3, diversified by
bootstrap-negative subsets + augmentation seeds), an **AION-1 frozen-embedding MLP
probe** (different objective / data / bands ⇒ maximal decorrelation), and two ResNets
(shielded-194 K + Lanusse-46). Each member is isotonic-calibrated, then combined
(average / logistic / random-forest).

| metric | published meta | ClaudeNet ensemble | Δ |
|---|---|---|---|
| Storfer @1 % FPR | 0.908 | **0.938** | +0.030 |
| Storfer @0.1 % FPR | 0.755 | **0.853** | **+0.090** |
| Inchausti @1 % FPR | 0.968 | **0.980** | +0.012 |
| Inchausti @0.1 % FPR | 0.845 | **0.935** | **+0.090** |

Member score correlation is Pearson ~0.31 / Spearman ~0.45 (vs the repo's collapsed
~1.0). The lesson: **diversity, not combiner cleverness, is the lever** — with
decorrelated strong members even a naive *average* beats the published
correlated-base meta-learner, especially at the strict 0.1 % FPR point that controls
purity. (The ensemble matches/beats the single best member everywhere except a 0.002
tie at the loose 1 % FPR.)

## Pipeline

```
# bootstrap / sanity
00_env_check  01_sync_data_from_phoenix  02_smoke_reuse  03_reproduce_baseline   # SANITY GATE
# phase 0 — decorrelation gate
10_build_aion_inputs  11_embed_aion(aion venv)  12_probe_aion  13_decorrelation_gate
# phase 1 — flagship ensemble
19_build_member_subsets  20_train_member(xN)  22_member_aion
25_calibrate_members  26_fit_combiner  27_correlation_report  28_eval_flagship
# phase 2..7
30_hard_negative_mining   40_label_efficiency   50_conformal_selection
60_domain_adapt   70_uncertainty   80_equivariance
# report
90_make_tables   91_make_figures
```

Shared libs: `_clib.py` (config/reuse loaders, GPUS={0,2,3,4,5,6}), `_train.py`
(recipe-faithful trainer + DihedralPool), `_minelib.py` (in-RAM cutout cache + fast
cached trainer), `_ensemble.py` (matched-FPR recovery copied verbatim from
`22_fpr_operating_point.py` + calibration / combiners / diversity / AUPRC / ECE),
`_combine.py`, `_aion_lens.py`. The inchausti finder libs and model files are
symlinked; the aion-1 harness (`_aion_embed`/`_probe`/`_ls_cutout`) is reused on path.

## Compute & data

- **Trainer:** this box's 7× TITAN RTX (24 GB, Turing sm_75); use GPUs **{0,2,3,4,5,6}**,
  exclude GPU 1 (thermal). No NVLink ⇒ **independent per-GPU jobs** (also why a
  thermally-throttling card — observed on GPUs 2/3/5 under sustained load — only slows
  its own member, never poisoning others). Env: `/home2/benson/.venvs/claudenet`
  (torch 2.6 cu124 + timm + lenstronomy).
- **Data:** FITS cutouts live on **phoenix** (`/raid/...`); `01_sync` rsyncs ~9.2 GB
  locally. AION embeddings run under `/home2/benson/.venvs/aion` (only it has `aion`);
  the on-disk `.npy` is the venv boundary.
- **NERSC Perlmutter** (A100/H100) reserved for the scale-up follow-ups below.

## Honest scope / caveats

- **AION member uses a degraded input** for speed: grz→griz by resize 101→160 + a
  synthetic i = ½(r+z) band (real native-griz fetch via `_ls_cutout` is the upgrade).
  Its standalone recovery is weak; its ensemble value is *decorrelation*, which holds.
- **Phase 2** mines within the existing 45 K random-galaxy pool with the weak shielded
  model → a *lower bound*. The deployment-scale version (fresh ~200 K brick pool +
  EfficientNet miner) is a NERSC follow-up, likely a larger effect.
- **Phase 5** is a negative result: the north↔south gap is real (−0.186) but naive
  MMD (λ=1, aligning only negatives) erased lens-discriminative features and hurt both
  domains; λ-tuning / joint-distribution alignment / the repo's north-negative
  augmentation are the levers. Only ~80 held-out north lenses → north recovery noisy.
- **Phase 7** demonstrates equivariance via *test-time* D4 pooling; the full
  equivariant-*training* label-efficiency study (escnn C4/D4) is deferred (the 8×
  forward made a trained D4 member impractical at this scale).
- Conformal guarantees are marginal and assume calibration/test exchangeability
  (violated under domain shift → use group-conformal).

## NERSC scale-up follow-ups

1. Native-griz AION member + AION-large/xlarge frozen embeddings.
2. Deployment-scale hard-negative mining (fresh ~200 K DR9 brick pool, EfficientNet miner).
3. Full equivariant-training label-efficiency study (escnn).
4. Full DR9/DR10 sky sweep with the diversity ensemble + conformal-controlled selection.

*Built on branch `reproductions/claudenet`. Bulk data is gitignored; scripts, result
JSON/CSV, figures, and this report are tracked.*
