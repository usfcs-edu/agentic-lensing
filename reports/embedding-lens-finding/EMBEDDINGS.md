# Embedding-Based Strong-Lens Finding — Research & Design

*Can we improve lens finding with high-dimensional embeddings instead of conventional
supervised CNNs (ResNet / EfficientNet)?*

**Scope.** This report (1) inventories the lens-finding neural networks currently in this
repo, (2) surveys the embedding / self-supervised / foundation-model literature for lens
finding (2021–2026), and (3) proposes concrete, repo-grounded ways to use embeddings —
each with an honest, adversarially-verified assessment. It was produced by a multi-agent
workflow (6 repo readers + 9 web-research analysts + diverse-framing proposal generators +
a two-lens adversarial verification panel per proposal + a completeness critic). Every
proposal below carries the verifiers' caveats inline; the conclusions are deliberately
sober.

---

## TL;DR

1. **Every lens finder in this repo is a supervised CNN binary classifier** trained on the
   same ~949 positives + curated negatives over 3-band grz 101×101 DECaLS cutouts:
   Lanusse ResNet-46 (huang-2020, 3.5M params), the "shielded" ResNet (huang-2021, 60k
   params), an EfficientNetV2-S + shielded-ResNet + FWLS meta-ensemble (inchausti-2025,
   ~20.5M), and a ResNet/U-Net forecast model (silver-2025). The only embedding-style model
   in the repo is **SpectrumFM/redshifty — spectroscopic, not imaging, and a failing
   prototype.** There is no image-domain self-supervised model here today.

2. **Embeddings are a strong *complement*, not a drop-in replacement.** The honest signal
   from the literature and the repo's own results: the field's current best lens *ranker*
   is a **supervised** model (fine-tuned Zoobot in Euclid Q1), individual-ML F1 in DES is
   only 0.31–0.54, and this repo's own finding is that **neg:pos ratio — not architecture —
   sets usability.** A backbone swap will not move the real bottleneck (false positives and
   visual-inspection load). Where embeddings *do* win is: **label-efficiency, out-of-
   distribution recall, "build-once / query-forever" economics at survey scale, cross-survey
   transfer, and automating the human-grading bottleneck.**

3. **There is one load-bearing risk that recurs in every proposal:** a galaxy-galaxy lens
   embeds like its bright elliptical deflector — the arc is a tiny perturbation — and
   generic SSL augmentations (color-jitter, blur, noise, crop) are trained to be *invariant*
   to exactly the faint-blue-arc signal that defines a lens. So embeddings cleanly separate
   lenses from *random* galaxies (the flattering "1-in-10⁴" framing) but **not** from the
   far more common *look-alike ellipticals* (Stein's ~0.20-precision wall). This is a
   make-or-break empirical question, and it is **cheap to test first.**

4. **Recommended first action (Tier 0): a 1–2 day go/no-go probe** — download public frozen
   embeddings (AstroPT DR8 / AION / Stein DR9), embed a *leak-free* graded lens set + matched
   non-lens elliptical controls + the CNN-scored-low subset, and measure whether lenses are
   even separable, *restricted to the CNN's blind spot*. Everything else is gated on this.

5. **Highest-leverage real builds (Tier 1), all reusing on-disk assets, no retraining:**
   a build-once embedding index over the existing DR7/DR8 brick sweeps → (a) similarity-
   search discovery, (b) **CNN-residual / embedding-disagreement** OOD mining (lenses the
   net scored *low* but the embedding ranks *high*), (c) **embedding-space hard-negative
   mining** to attack the FP bottleneck the repo itself identified, and (d) a de-risked
   **multimodal re-ranker for the hsu-2025 funnel** (13,530 → 2,046) that automates a real
   in-repo human-VI bottleneck the CNNs do not even address.

---

# Part I — The lens finders currently in this repo

All four families share one paradigm: **supervised binary classification of grz cutouts**,
trained on the same tiny positive set, deployed by sweeping a survey and thresholding a
probability. They differ only in backbone and scale.

| Finder | Slug | Backbone | Params | Input | Labels | Deployment | Headline |
|---|---|---|---|---|---|---|---|
| **Lanusse ResNet-46** (CMU DeepLens) | `huang-2020` | Pre-act bottleneck ResNet-46 | **3,508,833** | grz 101×101 (0.262″/px) | 949 pos + 5k neg | 6.24M DR7 sweep | test AUC 0.9991; 83% Grade-A recall @p≥0.9 |
| **Shielded ResNet** | `huang-2021` | L18 + 4 "shield" 1×1 bottlenecks | **59,905** (58.6× smaller) | grz 101×101 | 949 pos + rand neg | 17.3M DR8 (12.27M S + 5.02M N) | AUC parity (±0.002); north-calibration finding |
| **EfficientNetV2-S + shielded + FWLS meta** | `inchausti-2025` | timm `tf_efficientnetv2_s` (ImageNet-pretrained, fine-tuned) + shielded ResNet + 300-node FWLS | ~**20.5M** (effnet) | grz 101×101 | shared split | DR9/DR10 ensemble | recovery @1% FPR **91% / 97%**; meta ≈ average |
| **ResNet (+U-Net planned)** | `silver-2025` | ResNet classifier | — | space-based (HST/JWST) | lenstronomy/VELA sims | forecasts | val AUC 0.994 vs 0.998 (sim-to-real MVP) |

**Common training recipe** (Lanusse §3.4, reused by Huang): Adam, lr 1e-3 (/10 every 40
epochs), batch 128, 120 epochs, BCE/CE loss, per-band mean/std normalization + clamp ±250σ
(deliberately *not* ImageNet stats), augmentation = random rotation [-90,90°] + flips + zoom
[0.9–1.0]. 70/20/10 split.

### Documented limitations an embedding approach could address

These are taken from the repo's own reports (not speculation):

- **Label hunger & leakage.** Only ~949 positives; the positive set overlaps the model's own
  discoveries (test-set leakage ≈ 10 pp recall when isolated). The whole pipeline is a tiny-
  positive-class, ~1-in-10⁴ problem.
- **False positives dominate at the operating point.** The Inchausti reproduction's binding
  constraint is FP rate (37–51% of random galaxies flagged @p≥0.5 in the Stage-B
  reproduction), and the paper's own conclusion is **neg:pos ratio (not architecture) sets
  usability** — fixed by 1:33 ratio scaling + positive curation, not a bigger net.
- **Domain shift.** A south-trained net over-fired on the BASS/MzLS north (91% of north
  non-lenses scored ≥0.1 for L18; 10% for the shielded net), curable only by retraining with
  northern data. DR9→DR7 transfer also degrades. silver-2025 carries a sim-to-real gap.
- **Per-survey re-sweep cost.** The repo *explicitly skipped* full DR9 (~45M) / DR10 (~43M)
  parent sweeps because each is a multi-GPU-day job — a cost embeddings amortize.
- **Black box, binary only.** No θ_E, no morphological explanation, no follow-up ranking; the
  human Grade-A/B/C visual-inspection queue is entirely manual.

> ⚠️ **Correction to keep honest:** the often-quoted "27.8% Grade-C recovery @p≥0.9" is the
> *weak DR7-reproduction trained on random negatives*. The repo's own recovery tables show
> better models (DR9-trained ResNet, EfficientNet) reach **60–90%** Grade-C recovery. So the
> "lenses the CNN structurally misses" headroom is real but **smaller than a naïve reading
> suggests** — proposals must be benchmarked against the *strongest* in-repo baseline (the
> EfficientNet/meta ensemble), not the weakest.

### What embedding infrastructure already exists here

- **SpectrumFM / redshifty** — a DESI-*spectrum* transformer (d_model 768; encoder embeddings
  are mean-pooled in `probe_six_class.py`). It is a **prototype that fails per-class redshift
  parity** (galaxy catastrophic-z 96–99%; only stars reach DESI good-z) and is redshift
  *non-equivariant* (slope ≈ 0). Useful as a *class* discriminator (probe macro-F1 0.81–0.85),
  not as a lens-grade signal. **No image-domain SSL exists in the repo.**
- The only locally-available image "embedding" is the **supervised** EfficientNet/ResNet
  penultimate layer — which reproduces the CNN's own rankings (the meta-learner already
  stacks `p_resnet + p_effnet`). **It is not a substitute for a self-supervised embedding.**
- **Reusable infra that any embedding build should ride on:** the brick-FITS download + WCS-
  slicing loop (`huang-2020/11b_brick_inference_dr7.py`, ~150× faster than the cutout
  endpoint); **on-disk CNN scores for all 6.24M DR7 + 17.3M DR8 galaxies**; the
  LensJudge agentic grader; the GIGA-Lens JAX stack; the DESI/hsu-2025 FoF + FastSpecFit
  stack. Hardware: 8× A16 (16 GB) + 2× L4 (23 GB), **986 GB RAM**, no A100s.

---

# Part II — The embedding landscape for lens finding (2021–2026)

Nine research threads, distilled. Citations are in the [Appendix](#appendix--bibliography).

### 1. Self-supervised contrastive embeddings + similarity search — *the proven path*
**Stein et al. 2022** (arXiv:2110.00023) trained a MoCo-v2 ResNet50 on 76M unlabeled DESI
Legacy DR9 grz cutouts → a **2048-d** vector per galaxy, then found lenses by cosine
nearest-neighbour from **1–3 seed lenses** (~**500× enrichment** over random) plus a CPU-
minute linear probe; campaign yield **1,192 new candidates** (404 A / 788 B) from ~18k
inspections. Built on Hayat et al. 2021 (SSL matches supervised with 2–4× fewer labels).
**Public weights + 2048-d HDF5 vectors + a Streamlit similarity tool are released.** Caveat
the authors themselves flag: linear-probe precision plateaus at ~0.20 at 1:4000 imbalance —
this is a human-in-the-loop discovery channel, not a finished classifier.

### 2. Multimodal image+spectrum embeddings (AstroCLIP)
**AstroCLIP** (Parker, Lanusse et al. 2024, arXiv:2310.03024) CLIP-aligns a galaxy-image
encoder and a DESI-spectrum transformer into a shared space; frozen-embedding heads beat
single-modal SSL ~2× on property regression and match a supervised ResNet18 on photo-z, with
zero-shot cross-modal retrieval. Successors: **AION-1** (arXiv:2510.17960, omnimodal, 39
modalities, *lists strong-lens ID as a downstream task*, public weights), AstroM3, SpecCLIP,
Maven. *Caveat (important for lensing — see Part III): the lens signal lives in the image,
while the single DESI fiber captures the deflector; the cross-modal gain on z/mass does not
automatically transfer to lens finding.* Also note AstroCLIP partially **fine-tunes** the
image encoder (it is not a fully-frozen + MLP recipe).

### 3. In-domain image foundation models (AstroPT, DINOv2/MAE, Zoobot)
- **AstroPT** (Smith et al. 2024, arXiv:2405.14930): autoregressive image FM on **8.6M grz
  Legacy DR8** stamps — *the same footprint as huang-2021* — 1M–2.1B params, MIT weights.
- **Zoobot** (Walmsley, arXiv:2404.02973): *supervised* morphology pretraining; galaxy-
  pretrained encoders fine-tune **+31%** vs ImageNet, especially low-label. ConvNeXT/MaxViT
  on HuggingFace.
- **GraViT** (Parlange 2025): ImageNet ViT/MLP-Mixer transfer reaches AUROC 0.88–1.00 on HSC
  but needs deep-layer unfreezing (lens task diverges from natural images).

### 4. Anomaly / novelty detection for OOD lenses
**Astronomaly** (Lochner & Bassett 2021) and **Astronomaly-at-Scale** (Etsebeth 2024: ~8
lenses in the top ~2000 anomalies of 4M DECaLS) show lenses *are* embedding-space outliers —
but they are a tiny fraction of all anomalies, so **raw novelty ranking is artifact/merger-
dominated.** **Protege** (Lochner & Rudnick 2025) fixes this with Gaussian-process *relevance*
steering + active learning. *Correction: Protege argues interesting sources occupy an
**extended region** (not "isolated outliers"), which is why blind isolation-forest scoring
underperforms — the "deep space is too uniform" phrasing is imprecise.*

### 5. Metric learning, few-shot, active learning, label-efficiency
The "few-shot" evidence in lensing is really **similarity search** (Stein), not episodic
meta-learning (ProtoNets/MAML are essentially unexplored). **LenSiam** (Chang 2023,
arXiv:2311.10100) is the key lens-specific lesson: standard zoom/crop augmentations **corrupt
the Einstein radius**, so it uses a *lens-preserving* augmentation (fix the lens model, vary
the source). Active learning (Walmsley 2020, BALD) cuts labels 35–60%; Euclid Q1 iterative
retraining on real lenses + FPs roughly **doubled purity to ~94%.**

### 6. Domain shift / OOD robustness (directly relevant to the repo's north/south failure)
SSL features are empirically more shift-robust than supervised CNNs (Hendrycks 2019). And the
repo's *exact* failure has been studied: **Ye et al. 2024** (arXiv:2412.15533) do unsupervised
domain adaptation from **GZ-DECaLS → BASS/MzLS**, recovering near-source performance with
**zero northern labels**; **Parul et al. 2024** (arXiv:2410.01203) do ADDA/WDGRL + an
equivariant encoder for sim→HSC lens finding; DeepAstroUDA (Ćiprijanović 2023) +40% target
accuracy. CKA (Gondhalekar 2023) is a useful *diagnostic*, **not** a quantitative predictor.

### 7. Vector search at survey scale — *the clearest economic win*
Embed once → every future query is a ms-latency ANN lookup, vs a multi-GPU-day CNN re-sweep
per criterion. FAISS / ScaNN / HNSW deliver sub-ms, >90%-recall queries at 10⁹ scale.
Storage: 2048-d fp32 = 8 KB/vec → 17M ≈ 139 GB raw, **≈1 GB PQ-compressed**; 20B LSST ≈ 1.3 TB
PQ (vs ~15 PB catalog). Collett 2015 forecasts ~170k discoverable Euclid lenses. *Caveat:
the 10¹⁰ end is aspirational on this hardware; 10⁷–10⁸ (DR7/8/9) is realistic with faiss-cpu
today (the 986 GB RAM even fits a brute-force flat index over 17.3M×1280-d ≈ 88 GB → exact
search, no approximation confound).*

### 8. Embeddings for downstream tasks (grading, follow-up triage, parameters)
One frozen vector can feed many cheap heads: A/B/C grade regression, θ_E/ellipticity/shear
regression (Schuldt HOLISMOKES IV/IX; **Gawade 2024 ~10–20% θ_E error on real HSC**), NPE/SBI
posteriors (Swierc 2024 domain-adaptive NPE), and similarity-to-confirmed-lenses ranking for
DESI/Keck/MUSE queue ordering. Euclid Q1 used a fine-tuned Zoobot to rank 1M objects.

### 9. State of the art, 2024–2026 (Euclid)
The frontier is Euclid. The Q1 **Strong Lensing Discovery Engine** (arXiv:2503.15324/.15326/
.15328): **497 lenses (250 A, 243 new) from 63 deg²**, projected **>100,000** for the full
survey. The decisive ML finding: **the best automated ranker is a foundation model — a *fine-
tuned* Zoobot** (Lines et al., Engine C: 122 A + 41 B in the top-1000 of 1M), and the winning
*production* recipe is a **synergy** (DL ensemble → citizen science → expert vetting →
modelling), with ensemble purity ~52% / completeness ~50%. Pure-SSL AstroPT beats supervised
on photo-z with 1% of labels but was **not** used to build the lens catalogue (yet).

---

# Part III — The honest core: why "embeddings instead of CNNs" needs reframing

The adversarial verification panel converged on one conclusion: **embeddings are
complementary, not a replacement.** Six load-bearing reasons, each of which a real build must
confront:

1. **The deflector dominates the embedding.** A global-pooled embedding over a ~26″ FoV is
   dominated by the central galaxy's light; the arc is a tiny fraction of the flux/pixels. So
   a galaxy-galaxy lens embeds ≈ like its bright elliptical/LRG deflector, and cosine-NN from
   lens seeds preferentially returns **look-alike non-lens ellipticals.** Embeddings separate
   lenses from *random* galaxies (easy, flattering) but not from the *common confusers* — this
   is Stein's ~0.20-precision ceiling, and it is structural, not a tuning issue.

2. **Augmentations can erase the very signal you want.** Generic SSL augmentations (color-
   jitter, Gaussian blur, per-band noise, random crop) train *invariance* to exactly the cues
   that define a lens: blue-source-vs-red-deflector color contrast, faint low-surface-
   brightness arc curvature, small angular scale. LenSiam exists *because* standard crops
   corrupt the Einstein radius. **Mitigation: lens-preserving augmentations; ablate color-
   jitter / crop-scale against lens recall before trusting any frozen encoder.**

3. **Seed leakage.** The repo's 949 positives *are* the CNNs' training set, so similarity-NN
   from them is ≈ a 1-NN classifier on the same labels — *not* an orthogonal/OOD channel.
   **Evaluate leak-free** (use the disjoint graded catalogs as targets, the 949 only as
   seeds; leave-one-out for self-retrieval).

4. **"Replace the CNN" misreads the bottleneck.** Euclid Q1 → *supervised* (fine-tuned
   Zoobot) is the best ranker; DES comparative (arXiv:2510.23782) → individual ML F1 0.31–0.54,
   ensembling + human VI essential; the repo's own finding → **neg:pos ratio, not
   architecture, sets usability.** The binding constraint is **false positives and VI load**,
   which a backbone swap does not address. Frame embeddings around what they *do* move:
   label-efficiency, OOD recall, build-once economics, cross-survey transfer, VI-load
   reduction.

5. **Supervised penultimate features ≠ SSL embedding.** The only image embedding *in* the repo
   is the supervised EfficientNet/ResNet penultimate, which re-derives the CNN's own decision
   surface (and the meta-learner already stacks those). A genuine SSL/foundation embedding
   must come from **public weights** (AstroPT MIT / AION / Stein via Globus / Zoobot HF) or be
   pretrained — both are net-new dependencies, not "reuse."

6. **Input-pipeline parity is a silent failure mode.** Public encoders expect specific
   preprocessing (Stein: arcsinh-RGB stretch, 96-px crop, band order; AstroPT: 512-px / 16-px
   tokens). Feeding the repo's raw ±250-clamped float cubes produces **garbage embeddings and
   a silently-broken index.** Validate embeddings against the encoder's released vectors on a
   handful of known objects *before* any sweep.

**Net:** the right question is not "embeddings *instead of* CNNs" but "embeddings *alongside*
CNNs, where they are differentially strong." And the make-or-break uncertainty (#1+#2) is
empirical and **cheap to resolve first** — which is exactly Tier 0 below.

---

# Part IV — Concrete proposals for this repo, tiered

Each proposal folds in the verification panel's verdict: ✅ what's sound, ⚠️ what to fix, and
the falsification experiment. Effort = S/M/L. All Tier-0/1 work reuses on-disk assets and
needs **no model retraining**.

## Tier 0 — The gating go/no-go probe *(do this first; ~1–2 days; S)*

**Before building anything**, answer the make-or-break question: *do lenses even separate in a
frozen public embedding, restricted to the CNN's blind spot?*

- Pull a public frozen encoder (start with **AstroPT** — MIT, DR8 footprint, `astropt` on
  PyPI — and/or Stein's DR9 MoCo weights).
- Embed: (a) the *leak-free* graded lens catalogs (disjoint from the 949 training positives),
  (b) a **matched-deflector** non-lens elliptical control set (same type/magnitude/redshift),
  (c) a random parent sample, (d) the subset the CNN scored **low (p<0.5)**.
- Measure: linear-probe AUC (minutes on CPU, per Stein); **kNN purity restricted to the CNN-
  missed subset**; an **augmentation-ablation** (does color-jitter/crop wash out recall?); and
  the decisive one — *does the embedding beat the EfficientNet/meta ensemble **on the CNN-
  missed subset**?*
- **Decision rule:** if lenses are not separable from matched ellipticals here, frozen-
  embedding similarity search will not beat the supervised ensemble, and you should pivot to
  the hard-negative-mining / VI-automation uses (which don't depend on this) rather than
  invest in a DR7 SSL pretrain.

This single experiment de-risks every Tier-1/2 build and costs almost nothing.

## Tier 1 — High-leverage builds, no retraining *(reuse on-disk CNN scores + brick sweep)*

### 1.1 Build-once embedding index over the DR7/DR8 sweeps *(M)*
Re-run the existing brick-inference loop with the supervised scorer swapped for a **frozen
public encoder**, writing one vector per galaxy (float16 memmap / HDF5) instead of a scalar.
Stand up a **faiss-cpu** index. This single asset then serves similarity search, linear-probe
scoring, anomaly detection, and hard-negative mining — every future criterion becomes a ms
query instead of a fresh multi-GPU-day re-sweep.
- ✅ *Sound:* reuses the brick-FITS WCS-slicing path verbatim; 986 GB RAM fits an exact flat
  index (no PQ approximation needed at repo scale); economics quantified (17M ≈ 1 GB PQ; 20B
  LSST ≈ 1.3 TB PQ fits in one node's RAM, ms-latency >90%-recall queries).
- ⚠️ *Fixes:* **faiss-gpu is not pip-installable on aarch64 — use faiss-cpu** (sufficient).
  Must replicate the encoder's input preprocessing (Part III #6) and validate on released
  vectors first. **Frame the index as a candidate-generator / re-ranker that feeds the
  *existing* CNN + meta-ensemble — not a standalone detector** (the verifiers were firm: drop
  any "frozen-embedding parity with the supervised net" claim; the economics thesis is sound,
  the standalone-detection thesis is not, for a 1-in-10⁴ subtle class). **Economics
  correction:** the ~45M (DR9) / ~43M (DR10) sweeps the repo *skipped* are the real win — the
  DR8 sweep (17.3M) actually *ran* in ~16 h, so the amortization story is "reuse across future
  tasks/surveys + makes the skipped DR9/DR10 runnable once + ms queries," not "avoids DR8."
  No public weights are DR7-native, so you re-embed (cheap with a released encoder) and re-
  embed again if depth/footprint changes.

### 1.2 CNN-residual / embedding-disagreement OOD mining *(S, given 1.1)*
Join the new embeddings to the **on-disk CNN scores** and surface objects ranked **high by
embedding similarity to confirmed lenses but scored low by the CNN** — candidate OOD lenses.
- ✅ *Sound & cheap:* both ingredients exist; gives an interpretable, auditable retrieval.
- ⚠️ *Fixes (the verifiers were sharp here):* NN-to-known-seeds is the *most in-distribution*
  operation, so this measures **embedding-vs-CNN disagreement**, not true OOD — frame it as a
  *complementarity diagnostic + SSL second opinion*, not "lenses the CNN structurally
  misses." Evaluate **leak-free**, against the **strongest** baseline (EfficientNet/meta,
  not the weak DR7 model), and **pre-register the vetting budget** (e.g. top-2000 by
  similarity, expected lenses-per-100-inspected vs the Astronomaly ~0.4% prior) — the human-
  vetting cost is real and must be stated, not hidden.

### 1.3 Embedding-space hard-negative mining *(S–M)* — attacks the FP bottleneck directly
The repo's own conclusion is that **neg:pos ratio + negative quality set usability.** Use the
index to retrieve the *confusers* — ring galaxies, mergers, spirals, tidal features near the
lens manifold — and feed them as curated hard negatives to the existing finders. This targets
the actual binding constraint (false positives) rather than the recall side.
- ✅ *Sound:* directly operationalizes the repo's strongest empirical finding; no new
  paradigm; improves the *existing* CNNs.
- ⚠️ *Fix:* measure as recall-at-fixed-FPR improvement on the leak-free set.

### 1.4 De-risked multimodal re-ranker for the hsu-2025 funnel *(M)*
The hsu-2025 pipeline stops at 13,530 FoF groups; the 13,530 → 2,046 reduction is **pure
human visual inspection** — a bottleneck the imaging CNNs *don't even address*. Rank groups by
**[image embedding] + [on-disk FastSpecFit / pairwise-FoF tabular features]**, seed with the
20 tabulated Grade-A + funnel-eligible Foundry systems, route top-N to LensJudge.
- ✅ *Sound & uniquely repo-grounded:* automates a *verified* in-repo bottleneck; the spectral
  tabular features already exist; no heavy training; **no SpectrumFM dependency in v1.**
- ⚠️ *Fixes (verifiers found blockers):* the on-disk DR10 cutouts are **8-bit JPEGs**, not
  FITS — re-run the downloader with `--format fits` (~1.1 GB) for the low-SB dimple class and
  the LensJudge FITS path. The full published 2,046+318 catalog is **not yet released** — so
  for v1 use **leave-one-out CV on the 20 Grade-A + funnel-eligible Foundry positives** and
  report the VI-volume-reduction-at-fixed-recall curve; never seed and evaluate on the same
  20. Expand seeds via the on-disk Foundry-II DR1 crossmatch (only ~7–14 are funnel-eligible —
  verify membership). For the 9,292 σ_v-less "dimple" groups the method is **image-only by
  construction** — scope the "multimodal" claim to the ~4,238 σ_v-bearing subset.

## Tier 2 — Strong, gated on Tier 0/1

### 2.1 Label-efficiency curve: frozen embedding + logistic probe vs the ensemble *(S→M)*
If Tier 0 passes, fit a logistic/MLP probe on a frozen in-domain encoder at shrinking label
budgets N ∈ {10, 30, 100, 300, 949, 1312} and chart **recall@1% FPR vs #labels**, overlaying
the reproduced EfficientNet/meta ensemble's operating point as the supervised reference. This
is the canonical label-efficiency demonstration (Hayat/Stein: 2–4× fewer labels; AstroPT: 1%
of labels) — and the cleanest, most defensible single experiment after Tier 0: the literature-
feasibility panel rated this proposal **"accept / proceed" (conf 0.85)** — "the rare proposal
whose every literature claim survives spot-checking against the primary source and whose
compute/storage assumptions fit the actual hardware." Reuse `inchausti-2025/22_fpr_operating_
point.py` as the metric; train the heads in CPU-minutes.
- ⚠️ *Honest caveats:* the probe curve will very likely **plateau *below* the ensemble line** —
  frozen features trail end-to-end fine-tuned CNNs on subtle fine-grained discrimination
  (Stein's frozen probe: ~0.20 precision, ~0.5 recall at 1:4000). So the deliverable is the
  *label-efficiency curve* (match-with-fewer-labels), **not** "beats the CNN at the operating
  point." Same preprocessing-parity requirement as Tier 0/1.1. Use **lens-preserving
  augmentations** (LenSiam) if you pretrain — generic crops corrupt θ_E.

### 2.2 Dual-footprint / domain-adapted index for the north/south failure + cross-survey transfer *(M–L)*
Embed **both** DECaLS-south and BASS/MzLS-north with one encoder so the index spans the
northern PSF/depth before any label; discover by NN from confirmed lenses; abstain on
distributionally-far bricks. The repo's north/south failure is a ready-made UDA testbed (Ye
2024 studies this exact shift).
- ✅ *Well-founded premise:* Stein's released encoder was pretrained on **3.5M DR9-*south*
  galaxies only** (then *applied* to all 76M N+S), so it is genuinely south-biased — meaning
  "continue-pretrain on actual BASS/MzLS north" is a principled fix, not rhetoric. The repo's
  north/south experiment is a ready-made in-place benchmark. The defensible deliverable is
  **north-FPR reduction + embedding-space abstention** — something the binary-logit CNNs
  structurally cannot do.
- ⚠️ *Fixes (verifiers were emphatic):* the north negatives in `05c` were **randomly sampled,
  not hand-curated** — so the "cure the curation cost" premise is wrong; benchmark against the
  **deployed shielded-northaug** model (~10% / 0.8% FPR), *not* the never-deployed 91% L18
  strawman. Setting a north operating point still needs *some* north labels (calibration ≠
  zero labels). Report **recall-by-grade at matched FPR with the abstention rate** — and
  forbid the trivial "abstain on all north → 0 FPR" win. Prefer a *genuine SSL/equivariant*
  encoder + an explicit DA method (DANN/WDGRL/CORAL); the supervised EfficientNet "fallback"
  is south-trained and would re-import the bias it's meant to cure. CKA is a diagnostic, not a
  go/no-go predictor. **Position this as an amortized multi-task embedding catalog + ANN
  retrieval with abstention** (where the value is real), not primarily as a calibration patch
  (where the cheap random-negative retrain already wins).

### 2.3 Relevance-steered active learning with LensJudge as the oracle *(S–M)*
Astronomaly-Protege-style: novelty over [engineered lensing features + frozen embedding], with
GP relevance steering seeded by confirmed lenses, LensJudge grading each round's top-K. The
repo *uniquely* has an automated grader — the ingredient other groups replace with citizen
science.
- ⚠️ *Fixes (the verifiers found a hard blocker):* **LensJudge's 4-grade agreement with the
  DESI team is QWK ≈ 0.003** (near-zero) on `human_ceiling.csv` — using its A/B/C grades as
  the AL oracle injects noise, not signal. **Restrict LensJudge to coarse *binary* triage and
  validate binary agreement first.** Concatenating ~14 engineered scalars with a 1280–2048-d
  embedding **drowns** the lens-salient axes (≈1% of the distance budget) — use supervised
  feature weighting / metric learning, not raw concat+standardize. Keep an isolation-forest
  baseline as the negative control.

## Tier 3 — Longer-horizon / research

- **AstroCLIP-style multimodal, reframed.** Align a frozen image encoder to the spectrum side
  — but the verifiers showed the **lens signal is modality-asymmetric** (the single DESI fiber
  captures the *deflector*, a red elliptical indistinguishable from a non-lens at the same z),
  so "image↔spectrum consistency cuts FPs" and "spectrum→image two-redshift retrieval"
  **collapse.** Keep only the genuinely useful modes: image-only similarity search + deflector
  **photo-z** for follow-up ranking. First run the one-day falsification: do confirmed-lens
  deflector spectra separate from matched non-lens ellipticals at all? If not, the cross-modal
  lens payoff is nil.
- **θ_E-magnitude triage head for follow-up prioritization.** A regression head (Schuldt/
  Gawade-style) to rank candidates for GIGA-Lens / DESI-Keck-MUSE time. *Drop* the
  "warm-start fixes foundry-i" and "cross-modal source-z" claims — verifiers showed foundry-i's
  bottleneck was light-amplitude marginalization / pinv conditioning (not the mass-param start
  point), rotation-invariant embeddings can't carry ellipticity/shear PA, and the fiber/image
  both encode the *lens* z (not the hard *source* z). Build θ_E as an **end-to-end** regressor
  on injected mocks (only ~13 fitted labels exist), and mind the HST(0.065″)-vs-DECaLS(0.26″)
  resolution gap.
- **Close the silver-2025 sim-to-real gap with real-image SSL + adversarial DA.** silver-2025
  trains *only* on analytic Sérsic simulations and validates on simulations — the regime where
  sim-trained finders collapse from ~92% purity on sims to ~11–24% on real Euclid imaging.
  Instead: pretrain the encoder on **real** imaging, then domain-*align* the simulated
  positives into the real embedding space (Parul 2024; Swierc 2024 NPE-DA). Stage the sim-to-
  real core first (it stands alone and fixes a *named* silver-2025 gap); attach the cross-modal
  layer only if the spectrum encoder proves CLIP-alignable. *Caveat:* HST/JWST resolution
  (0.065″) is far from the DECaLS-trained encoders (0.26″), so the encoder likely needs space-
  based pretraining or fine-tuning.
- **Equivariant / E(2)-steerable backbone** (escnn) as an augmentation-free, tiny replacement
  for the 60k-param shielded net — bakes in the rotation/reflection symmetry of arcs and pairs
  best with WDGRL UDA (Parul 2024). The completeness critic flagged this as a notable gap.
- **Knowledge distillation** of a foundation-model embedding into the repo's tiny shielded net
  — keep the deployable-small-net story while importing FM representations.

## Tier 4 — Drop / defer (with reasons)

- **Spectral-anomaly gate via SpectrumFM** — *drop in current form.* SpectrumFM is redshift-
  non-equivariant (slope ≈ 0), so it is largely *invariant* to the exact line-position
  perturbation a lens+source blend creates; and single-fiber blended-spectrum lens detection
  is a **mature, high-precision, ~20-year-old method** (SLACS/Bolton 2006; BELLS/Brownstein
  2012; Talbot 2021) that a failing prototype cannot beat. *If* FP-triage via spectra is
  wanted, use the **classical template-residual emission-line search + Redrock second-peak
  ΔΧ²** — feasible now, zero GPU, on the existing DESI stack. Calibrate on Holwerda-2015 /
  SLACS blends, **not** on hsu's two-fiber pairs (a different physical regime).
- **Cross-modal lens *confirmation*** — defer; the modality-asymmetry above guts the value.
- **GIGA-Lens embedding warm-start** — defer; misdiagnoses the documented bottleneck.

---

# Part V — How to evaluate (apples-to-apples, gaming-resistant)

The repo already has the right yardsticks; embedding experiments must use them:

- **Recovery-by-grade** and **AUC within ±0.002** (the repo's architecture-controlled
  convention); the **hsu-2025 20/20 Table-2 Grade-A** set; the leak-free graded catalogs.
- **Net-new over the strongest baseline, not over random.** Report gains vs the EfficientNet/
  meta ensemble (recall@1% FPR 91–97%), *not* the 500×-over-random enrichment figure.
- **Gaming-resistant metrics:** recall-by-grade at *matched* FPR **with the abstention rate
  reported** (forbid abstain-everything); **lenses-per-100-inspected** at a *pre-registered*
  vetting budget; leak-free / leave-one-out so seeds can't be re-retrieved as "discoveries."
- Add an **embedding-experiment row to `REPRODUCTIONS.md`** so it's tracked like every other
  result here.

---

# Part VI — Implementation notes (concrete)

- **Encoders to try (public, ranked by fit):** AstroPT (MIT, DR8 grz, `astropt` PyPI) →
  AION-1 (omnimodal, lists lens ID) → Stein MoCo ResNet50 (DR9, 2048-d, Globus) → Zoobot 2.0
  (HF, ConvNeXT/MaxViT — but it's *supervised* morphology, and Euclid's win was the *fine-
  tuned* variant, so plan to fine-tune).
- **Pipeline:** reuse `huang-2020/11b_brick_inference_dr7.py` WCS-slicing; run extraction on
  the L4s (`CUDA_VISIBLE_DEVICES`, `CUDA_DEVICE_ORDER=PCI_BUS_ID`) + A16s for extra throughput;
  store embeddings as a **float16 memmap**; join to on-disk CNN scores by brickname/objid.
- **Index:** **faiss-cpu** (faiss-gpu has no aarch64 wheel); at repo scale a flat exact index
  fits in the 986 GB RAM (no PQ approximation confound); use IVF-PQ only when you go to
  LSST scale.
- **Preprocessing parity is mandatory** (Part III #6): match each encoder's crop size / band
  order / flux scaling and validate against its released vectors before sweeping.
- **Always run Tier 0 first** — it's minutes-on-CPU and gates the rest.

---

# Appendix — Bibliography

*(arXiv IDs verified by the research + literature-feasibility agents where possible; a few
flagged "confirm authorship/ID against ADS" are noted.)*

**Self-supervised / similarity-search lens finding**
- Stein, Blaum, Harrington, Medan, Lukić 2022 — *Mining for Strong Gravitational Lenses with
  Self-supervised Learning* — arXiv:2110.00023, ApJ 932, 107
- Stein et al. 2021 — *Self-supervised similarity search for large scientific datasets* —
  arXiv:2110.13151 (NeurIPS 2021 ML4PS)
- Hayat, Stein, Harrington, Lukić, Mustafa 2021 — arXiv:2012.13083, ApJL 911, L33
- Chang, Huang, Fagin, Chan, Lin 2023 — *LenSiam* — arXiv:2311.10100

**Foundation models / multimodal**
- Parker, Lanusse, Golkar, Sarra, Cranmer et al. 2024 — *AstroCLIP* — arXiv:2310.03024,
  MNRAS 531, 4990
- Smith, Roberts, Angeloudi, Huertas-Company 2024 — *AstroPT* — arXiv:2405.14930
- Parker, Lanusse, Shen et al. 2025 — *AION-1* — arXiv:2510.17960
- Walmsley et al. 2024 — *Scaling Laws for Galaxy Images (Zoobot 2.0)* — arXiv:2404.02973;
  Zoobot (JOSS 2023, arXiv:2303.00366)
- Rizhko & Bloom 2025 — *AstroM3* — arXiv:2411.08842
- Parlange et al. 2025 — *GraViT* — arXiv:2509.00226

**Anomaly / OOD discovery**
- Lochner & Bassett 2021 — *Astronomaly* — arXiv:2010.11202
- Etsebeth, Lochner, Walmsley, Grespan 2024 — *Astronomaly at Scale* — arXiv:2309.08660
- Lochner & Rudnick 2025 — *Astronomaly Protege* — arXiv:2411.04188
- Walmsley & Scaife 2023 — *Rare Galaxy Classes in Foundation Model Representations* —
  arXiv:2312.02910
- Storey-Fisher et al. 2021 — HSC GAN anomalies — arXiv:2105.02434

**Domain adaptation / robustness**
- Ye, Shen et al. 2024 — *GZ-DECaLS → BASS/MzLS UDA* — arXiv:2412.15533
- Parul, Gleyzer, Reddy, Toomey 2024 — *Domain adaptation for lens finding* — arXiv:2410.01203
- Ćiprijanović et al. 2023 — *DeepAstroUDA* — arXiv:2302.02005; 2022 *DeepAdversaries* —
  arXiv:2112.14299
- Hendrycks et al. 2019 — arXiv:1906.12340 (NeurIPS)
- Gondhalekar et al. 2023 — CKA / OOD generalization — arXiv:2311.18007

**Few-shot / active learning / challenges / baselines**
- Metcalf et al. 2019 — *Bologna Lens Finding Challenge* — arXiv:1802.03609, A&A 625, A119
- Walmsley et al. 2020 — Bayesian CNN + active learning — arXiv:1905.07424, MNRAS 491, 1554
- Thuruthipilly et al. 2022 — transformers for lens finding — arXiv:2110.09202, A&A 664, A4
- *Does ML Work? Comparative SGL searches in DES* 2025 — arXiv:2510.23782 (F1 0.31–0.54)
- *Bayesian lens finding* — arXiv:2311.07455; FP reduction via augmentation+ensemble —
  arXiv:2502.14936; transformer FP reduction — arXiv:2212.12915

**Vector search at scale**
- Johnson, Douze, Jégou 2017 — *FAISS* — arXiv:1702.08734; Douze et al. 2024 — arXiv:2401.08281
- Guo et al. 2020 — *ScaNN* — arXiv:1908.10396; Malkov & Yashunin 2018 — *HNSW* —
  arXiv:1603.09320
- Collett 2015 — lens population forecast — arXiv:1507.02657, ApJ 811, 20

**Downstream tasks (parameters / triage)**
- Schuldt et al. — *HOLISMOKES IV* (A&A 646, A126, 2021) / *IX* (arXiv:2206.11279, A&A 671,
  A147, 2023)
- Gawade et al. 2024 — θ_E regression on HSC — arXiv:2404.18897
- Swierc, Tamargo-Arizmendi, Çiprijanović, Nord et al. 2024 — domain-adaptive NPE —
  arXiv:2410.16347

**Euclid 2024–2026 (state of the art)**
- Euclid Q1 Strong Lensing Discovery Engine — A (arXiv:2503.15324), C/Lines (arXiv:2503.15326),
  E/Holloway (arXiv:2503.15328)
- Pearce-Casey et al. 2024 — Euclid ERO Perseus CNN lens search — arXiv:2411.16808 *(confirm
  ID/authorship vs ADS)*
- Siudek et al. 2025 — *Euclid Q1 AstroPT multimodal FM* — arXiv:2503.15312
- *From simulations to sky* (sim-to-real) — arXiv:2512.05899
- Lines, Li, Collett et al. 2025 — *The revolution in strong lensing discoveries from Euclid*
  — arXiv:2508.14624 (Nature Astronomy)
- DES interactive ML + ViT + Space Warps — arXiv:2501.15679

**Completeness-critic additions** (gaps worth pursuing): equivariant E(2)-CNNs (escnn);
lens-preserving SSL augmentations (LenSiam); UDA as a concrete method (DANN/ADDA/WDGRL/CORAL);
calibrated/Bayesian ranking + conformal thresholds; lens-specific MAE / joint find+model
pretraining (arXiv:2512.06642); image-text CLIP for LLM-queryable grading (PAPERCLIP
arXiv:2403.08851); foundation-model→tiny-net knowledge distillation; embedding-space hard-
negative mining; generative/density OOD scoring; embedding-based selection-function
characterization.

---

*Generated by a multi-agent research + design + adversarial-verification workflow over this
repo (`reports/embedding-lens-finding/`). Proposal assessments reflect a two-lens skeptical
review panel; treat the inline ⚠️ caveats as requirements, not footnotes.*
