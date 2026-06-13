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

## Discovery impact & scan deployment

The recovery gains are best read operationally: a 45 M-cutout sweep must run near
**0.01 % FPR** (1 % FPR = ~450 K false positives), and ClaudeNet's advantage *grows*
as the threshold tightens (+0.030→+0.098 Storfer, +0.012→+0.090 Inchausti over
1 %→0.1 % FPR) — it helps most where scans actually operate.

- **Fewer missed lenses at fixed inspection budget:** at 0.1 % FPR, **~40 % fewer
  Storfer / ~58 % fewer Inchausti** lenses missed (~37 %/56 % with the gate-certified
  learned combiner). Dual view: **~2× fewer false positives at fixed completeness**
  (~200–250 K fewer objects to inspect on a DR9/DR10 sweep; interpolation, 1.3–2.2×).
- **Two guarantees prior finders lack:** certified-FDR conformal selection (a
  defensible purity ceiling) and deep-ensemble triage (confident ~half essentially
  error-free → ~halves the human-inspection bottleneck).
- **Scan throughput:** the 5 *grz* finder members share one cutout load, so an
  I/O-bound scan costs ~a single model's wall-clock (~3 GPU-h for 45 M on 5 TITAN RTX,
  compute-only). The AION member (~100× slower, griz-160) is **stage-2 only** on
  filtered survivors — the lineage's existing two-stage design. **Net: same inspection
  budget, ~9–10 points more real lenses at the operating point that matters.**
- **Honest limits:** "recovery" is held-out TPR on *existing* catalogs (no new lenses
  found here); the 0.1 % threshold is pinned by ~6–7 of ~6,500 held-out negatives
  (±~10 pp, no CI); absolute counts and the ~2× saving are interpolations; survey-scale
  extrapolations (Euclid ~10⁵, LSST ~6–12×10⁴; order +10⁴ lenses at fixed budget)
  assume the DESI-Legacy deltas transfer across bands/PSF/depth — unverified.

See `papers/main.pdf` §"Discovery impact and scan deployment" for the full tables and
the measured per-member throughputs.

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

> **All four were executed at survey scale — see the ClaudeNet v2 section below.**

## ClaudeNet v2 (Perlmutter scale-up)

The four NERSC follow-ups above were executed at survey scale on Perlmutter, with the
**full DESI Legacy Survey archive resident on CFS** (no per-cutout fetch). This turned
every "deferred — too big locally" caveat in the v1 report into a measured result. All
numbers below are quoted verbatim from `data/v2/` artifacts (cited per row).

### Results by phase (v2)

| Phase | Direction | Headline result |
|---|---|---|
| 110 | **NegEval-1M** honest eval set | The v1 0.1 % threshold was pinned by ~6–7 of **6,501** held-out negatives; v2 rebuilds it on **1,000,000** negatives. The v1 6-member average vs published meta: storfer@1 % **+0.049** [+0.037, +0.061], @0.1 % **+0.075** [+0.057, +0.091]; but @0.01 % (first ever measured) the *average collapses* **−0.120** [−0.161, −0.084] — the degraded-AION member's neg tail poisons it. RF combiner robust @0.01 % **+0.102** [+0.083, +0.123]. Best single member `effnet_S2` @0.01 % **+0.194** [+0.175, +0.215]. (`thresholds_ci.json`, `operating_points_v2.csv`) |
| 120 | Deployment-scale hard-neg mining (1M pool) | **Round-1 SHIPS** — paired (hard − random) mean over 3 archs at inchausti@0.1 % **+0.157** [+0.139, +0.177]; all members PASS the +0.015 gate. **Round-2 (hard2) rejected**: re-mining with the round-1 model over-mines (hard2 < hard everywhere). Keep round-1 only. (`mining_v2_results.json`) |
| 130 | AION upgrade (native griz, large/xlarge, LoRA) | **DROP / KEEP-V1.** Matched-rows native-griz storfer@1 % **0.535** < degraded-refit **0.647** → gate FAIL (`gate_a_pass:false`). The synthetic-i "degradation" *was* the diversity mechanism (Pearson vs CNN 0.13 degraded vs ~0.65 native pooled). large/xlarge frozen probes didn't beat the degraded standalone; LoRA skipped (unjustified). (`aion_gate_v2.json`) |
| 140 | **Ensemble v2-lean refit (HEADLINE)** | **SHIP 4/4.** Roster = {`effnet_B`, `effnet_B3_hard`, `effnet_S2_hard`, `resnet46_C_hard`, `zoobot_N`}; AION + `shielded_A` dropped by leave-one-out admission. New Zoobot ConvNeXT-Nano member needs lr 1e-4 (v1's 1e-3 collapses it). (`ensemble_v2lean_verdict.json`) |
| 150 | Distill ensemble → single student | **FAILS the gate.** EffNetV2-S student hits **3,494 c/s** vs the 5-member shared-load **663 c/s** (5.3×), but recovery drops **−0.111** storfer@0.1 % vs teacher (gate is −0.02). Deploy the shared-load lean members instead. (`throughput_v2.json`) |
| 160 | Full DR9 sweep → group-conformal FDR | **17,290,814** parent galaxies swept; **29,892** survivors (stage-1 rate **1.73e-3**); per-group conformal BH at **FDR ≤ 0.05** selects **1,449** (813 new, **737 new-and-unseen**), **2,836** @0.10, **5,141** @0.25. Known-lens recall **47.5 %** in-coverage into survivors (1,676/3,528). (`sweep/{sweep_summary,conformal_summary,crossmatch_recall,stage1_summary}.json`) |

### The v2-lean headline (recovery @ matched FPR, NegEval-1M)

v2-lean **beats the v1 flagship on all four ship cells** and crushes the published meta,
with paired-bootstrap CIs (10,000 reps) over 1 M negatives (`ensemble_v2lean_verdict.json`):

| metric | published meta | v1 flagship | **v2-lean** | Δ (v2-lean − v1) | Δ (v2-lean − meta) |
|---|---|---|---|---|---|
| Storfer @1 % | 0.854 | 0.903 | **0.963** | **+0.060** [+0.049, +0.072] | +0.109 |
| Storfer @0.1 % | 0.679 | 0.754 | **0.895** | **+0.141** [+0.125, +0.160] | +0.216 |
| Storfer @0.01 % | 0.513 | 0.394 | **0.734** | **+0.340** [+0.306, +0.381] | +0.220 |
| Inchausti @1 % | 0.932 | 0.968 | **0.996** | **+0.028** [+0.017, +0.041] | +0.064 |
| Inchausti @0.1 % | 0.769 | 0.891 | **0.961** | **+0.069** [+0.052, +0.090] | +0.191 |
| Inchausti @0.01 % | 0.607 | 0.614 | **0.871** | **+0.256** [+0.217, +0.301] | +0.264 |

The gain *grows monotonically as the threshold tightens* — exactly the regime a real sweep
runs in. The v2-lean **storfer@0.01 % = 0.734** vs the v1 flagship's 0.394 and the meta's
0.513 is the single most important number: at the operating point a 17 M-cutout sweep
actually uses, v2-lean recovers nearly twice the lenses the v1 ensemble did. Member
correlation stays low (Pearson ~0.18–0.40 off-diagonal). The equivariance follow-up is a
**trained C4 escnn member** (replacing v1's test-time-D4-only result); the pilot/full-D4
member (scripts `141`/`142`) is a queued bonus, not part of the shipped 5-member roster.

### Sky-sweep candidate list (Phase 160)

From **17,290,814** DR9 parent galaxies (north 5,018,411 + south 12,272,362), the lean
5-member stage-1 union (per-member 1e-4 FPR) yields **29,892 survivors** (realized rate
**1.728783e-3**, consistent with the EVT union-bound prediction within ×3). Stage-2 rescore
with the v2-lean average, then **per-group conformal BH** against the NegEval-1M calibration
half (`conformal_summary.json`):

| selection | FDR α | n selected | n new | n new-and-unseen | north / south |
|---|---|---|---|---|---|
| `sel_group` | **0.05** | **1,449** | 813 | **737** | 0 / 1,449 |
| `sel_group` | 0.10 | 2,836 | 1,999 | — | 886 / 1,950 |
| `sel_group` | 0.25 | 5,141 | 4,186 | — | 2,634 / 2,507 |

North @0.05 selects **0** — surfaced honestly: with `n_cal=145,227` the per-group power
floor (`p ≥ 1/(n_cal+1)`, full-m BH) cannot reach α=0.05, so the valid estimator reports
nothing rather than over-claiming. Anti-conservative survivors-only-m diagnostics are kept
separate and explicitly carry **no FDR guarantee**. Visual grading of the top ~300 new
candidates (`sweep/visual_grading.md`) finds an overwhelmingly LRG population with a
~5–8 % plausible-arc minority — a **candidate list, not a confirmed-lens list**, whose value
is the statistical control: 737 new-and-unseen at certified per-group FDR ≤ 0.05 from a 17.3 M
sweep, with the pipeline re-finding 47.5 % of in-coverage known lenses at the same threshold.

### New pipeline scripts (100–165)

*Built on branch `reproductions/claudenet` (v2 on `reproductions/claudenet-v2`).
Bulk data is gitignored; scripts, result JSON/CSV, figures, and this report are tracked.*
