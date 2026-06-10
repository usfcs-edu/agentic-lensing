# ClaudeNet — a research program to improve ML strong-lens finding

## Context

The repo's existing reproductions (Huang 2020/2021 → Storfer 2024 → Inchausti 2025 → Silver 2025)
faithfully rebuilt the ResNet/EfficientNet lens-finder lineage **and proved a hard, useful negative
result**: at the deployment operating point, *architecture is not the bottleneck*. Shielded-ResNet
(194 K params) ≈ EfficientNetV2-S (20.5 M) within ±0.003 AUC, and the Inchausti FWLS meta-learner
collapsed to a simple average because both base models trained on byte-identical data (correlated
errors). A deep literature survey (8 technique families, 25 agents, adversarially fact-checked)
**converges with this**: no transformer / SSL / equivariant / foundation-model method cleanly beats a
well-tuned EfficientNet at matched false-positive rate for DECaLS *grz* lens finding. The demonstrated
wins in the literature are all on the axes the repo diagnosed but never fixed:

- **Ensemble diversity** — DES (arXiv:2510.23782) raised completeness 70→82 % and precision ~5.5× by
  combining *different-architecture* finders, with **naive averaging the only combiner that failed**;
  KiDS/Rezaei (2502.14936) cut FP-rate 11× at 88 % completeness via augmentation+ensemble.
- **Negative quality / operating-point calibration** under extreme class imbalance (the repo's own
  Stage-C/D finding: neg:pos ratio, not architecture, moved recovery@1 %FPR from ~9–19 % to ~84–88 %).
- **Label efficiency** (frozen foundation/SSL features; real morphology number is 2–4× fewer labels,
  *not* the folklore "10×"), **domain shift**, and **uncertainty** (absent from the repo).

**Goal.** Stand up `reproductions/claudenet/` as a new research program — **ClaudeNet** — that attacks
these proven levers with controlled experiments, validated on this machine's TITAN RTX GPUs first and
scaled at NERSC. The honest primary metric throughout is **recovery@1 % FPR** (matched false-positive
rate), reusing the repo's own `22_fpr_operating_point.py` arithmetic verbatim so every result drops
straight into the existing comparison tables. The user selected the **full seven-direction program**
with **autonomous execution through decision gates, reporting at each**.

### Baseline numbers to beat (already on disk, `inchausti-2025/data/`)
From `meta_metrics_staged.json` (Stage D, 1961 pos : 65 010 neg ≈ 1:33), recovery@1 % FPR:
`effnet` storfer **0.912** / inchausti 0.959; `meta` 0.908 / 0.968; `resnet` 0.640 / 0.829.
The meta-learner **does not beat its best single member on Storfer (0.908 < 0.912)** — the exact
collapse the flagship targets. Stage-C (≈1:25): `meta` storfer 0.836 / inchausti 0.885.
**A ClaudeNet result is "significant" only if it beats the best single member AND the naive average at
recovery@1 % FPR on the held-out Storfer/Inchausti positives vs held-out Stage-C/D negatives.**

## Branch & compute ground truth
- **Branch:** cut `reproductions/claudenet` from **current HEAD (`reproductions/aion-1`)**, *not* `main`
  — `main` has the finder reproductions but **not `aion-1`**, and the flagship reuses the AION-1
  harness. (`git branch reproductions/claudenet && git switch reproductions/claudenet`.)
- **Primary trainer (this box):** 7× TITAN RTX (24 GB, Turing sm_75, fp16 AMP+tensor cores,
  `torch.compile` OK). **Use GPUs `{0,2,3,4,5,6}` — exclude GPU 1 (thermal throttler).** No NVLink
  (PCIe PHB/PIX) ⇒ **independent per-GPU jobs**, one training pinned per card via `CUDA_VISIBLE_DEVICES`
  (also why a thermal straggler can't poison other jobs). 64 cores, 251 GB RAM, `/home2` 3.7 TB free.
- **Data lives on phoenix**, not here. `phoenix.cs.usfca.edu:/raid/benson/git/agentic-lensing/
  reproductions/.../data/cutouts_fits_*` (a few GB total). phoenix also has 8× A16 (15 GB) + 2× L4
  (busy) and runs hot (load ~9) → **data source + optional secondary inference node only**.
- **Envs:** `uv` at `/home/benson/.local/bin/uv`. The `aion` venv (`/home2/benson/.venvs/aion`,
  torch 2.6 cu124) has `aion`+`sklearn`+`astropy` but **no `timm`/`lenstronomy`**. ClaudeNet gets its
  own `/home2/benson/.venvs/claudenet`. **Venv split rule:** AION embedding (`11_…`) runs under the
  `aion` venv and writes `.npy`; everything downstream reads those `.npy` under the `claudenet` venv —
  the on-disk embedding file *is* the boundary.
- **NERSC Perlmutter** (A100/H100, account `deepsrch_g`, shared QOS, DDP templates in
  `lensing-repos/redshifty/nersc/`) for: full DR9/DR10 sky sweeps, AION-backbone fine-tuning
  (frozen-only locally), and large escnn (D16) groups.

## Reusable assets (confirmed by reading the files — reuse, do not rewrite)
| Asset | Path | What to reuse |
|---|---|---|
| Dataset/split/preproc | `inchausti-2025/_trainlib.py` | `LensDataset` (grz FITS loader, per-band norm+clamp±250, rot/flip/zoom aug, honors per-row `fits_dir`), `build_split` (70/20/10, SEED=2026), `compute_band_stats`, `model_prob`, `DR_TO_FITS` |
| Batch scoring | `inchausti-2025/_scorelib.py` | `load_checkpoint_model`, `score_paths` (NaN-safe) |
| Models | `huang-2021/01b_shielded_resnet.py`, `01_lanusse_resnet.py`, `inchausti-2025/02_efficientnet.py`, `03_meta_learner.py` | `ShieldedDeepLens` (configurable), `CMUDeepLens`, `EfficientNetV2Lens` (timm), `MetaLearner` + `simple_average` |
| Supervised trainer | `inchausti-2025/19_train_stageb.py` | **`CFG194`** = `dict(stage_out=52,stage_mid=32,shield_ch=12,final_out=24)`, **`band_stats`**, **`train_base(arch,df,mean,std,device,epochs,batch,lr,decay_ep,accum)`** (import by path; `LensDataset`'s per-row `fits_dir` makes it read ClaudeNet data despite its hardcoded `DATA`) |
| **Eval harness** | `inchausti-2025/22_fpr_operating_point.py` | matched-FPR recovery; copy the ~6-line threshold arithmetic (`thr=np.quantile(neg,1-fpr); recovery=(cand>=thr).mean()`) into `_ensemble.recovery_at_fpr` |
| Brick negatives | `inchausti-2025/20_build_negatives_brick_dr9.py` | `download_brick`/`load_brick`/`extract`/`lens_mask` (45 K negs in ~4 min) |
| Staged trainers | `inchausti-2025/21_train_stagec.py`, `24_train_staged.py` | ratio-controlled retraining |
| Triage viewer | `inchausti-2025/16_build_inspection_viewer.py` | Lupton-RGB paginated viewer |
| Sims | `silver-2025/01_gen_sims.py` (tunable θ_E lenstronomy), `02_train_resnet.py` | small-θ_E / sim-to-real source |
| **AION harness** | `aion-1/_aion_embed.py`, `_probe.py`, `_ls_cutout.py`, `_config.py` | `multi_gpu_extract(specs,variant,out,pool,gpus)`, `image_spec("LegacySurveyImage",flux_npy,["DES-G","DES-R","DES-I","DES-Z"])`; `LinearHead`/`MLPHead`/`train_classification`; `fetch_one(ra,dec,layer,size=160,pixscale=0.262,bands="griz")` (returns `None` for northern grz-only fields). Models `polymathic-ai/aion-{base,large,xlarge}` cached at `$HF_HOME=/home2/benson/.cache/huggingface`. Existing GZ10/GZ-DECaLS scripts (`06`,`12`,`44`) are the proven **160 px griz** embed template. |

## Critical files to CREATE in `reproductions/claudenet/`
- `_clib.py` — config hub: `SEED=2026`, `GPUS=[0,2,3,4,5,6]`, paths, `PHOENIX="phoenix.cs.usfca.edu"`,
  `PHOENIX_RAID`, `gpu_env(id)`, and `importlib`-by-path loaders for the reused inchausti/aion funcs.
- `_ensemble.py` — calibration (Platt/isotonic), combiners (naive-avg / logistic / random-forest),
  **`recovery_at_fpr`** (verbatim `22` arithmetic, generalized to N members), correlation/Q-statistic,
  ECE, AUPRC.
- `_aion_lens.py` — grz→**griz 160 px** flux builder (i-band: real fetch via `_ls_cutout`, else
  `dup_r` with an `i_synth` flag — never silently drop northern lenses), wrapping
  `_aion_embed.multi_gpu_extract` + `_probe`.
- `_watch_gpu.sh` — watchdog **keyed to PCI bus-id** (not CUDA ordinal), excludes GPU 1, kills a
  member job if its pinned card crosses a temp threshold.
- Symlinks (no edits): `_trainlib.py`, `_scorelib.py`, `01_lanusse_resnet.py`, `01b_shielded_resnet.py`,
  `02_efficientnet.py`, `03_meta_learner.py` → `../inchausti-2025/…`.
- `.gitignore` mirroring inchausti (ignore `data/cutouts_fits_*`, `*.pt`, `data/emb/`, `*.npy`, `*.log`).

## Phases (numbered scripts; autonomous gates)

**Bootstrap & sanity**
- `00_env_check.py` — build/validate `/home2/benson/.venvs/claudenet` (uv: torch 2.6 cu124, timm,
  scikit-learn, astropy, lenstronomy, pandas, pyarrow, matplotlib; `escnn` optional). Assert sm_75,
  ≥6 visible cards, GPU-1 bus-id resolved & excluded; assert the `aion` venv launches `import aion`.
- `01_sync_data_from_phoenix.py` — rsync (read-only pull, `--ssh-user`/`--host` args + `ssh true`
  preflight) the cutout dirs (`cutouts_fits_{candidates_storfer,candidates_inchausti,curated_dr9,
  litpos_dr9,neg_dr9}`, huang-2020 `cutouts_fits_dr9`), tabular (positives/negatives parquets, published
  catalogs, `training_split_stage{c,staged}.parquet`, `operating_point.csv`, `meta_metrics_staged.json`),
  and Stage-C/D checkpoints. Verify file counts + md5 spot-checks; abort on mismatch.
- `02_smoke_reuse.py` — import symlinked libs; assert `ShieldedDeepLens(**CFG194)`≈194 433 and
  `EfficientNetV2Lens`≈20 542 883 params; `score_paths` returns finite probs on 8 FITS.
- `03_reproduce_baseline.py` — **non-negotiable sanity gate.** Re-run matched-FPR eval on the rsynced
  Stage-C/D checkpoints; assert reproduction of the on-disk numbers (Stage-D `meta` storfer≈0.908 /
  inchausti≈0.968; `effnet` storfer≈0.912). **No improvement is claimed until this passes.**

**Phase 0 — decorrelation gate (<1 day; KILL/VALIDATE the flagship)**
`10_build_aion_inputs.py` (build 160 px griz arrays for train pool + Storfer/Inchausti held-out
positives + Stage-C/D held-out negatives) → `11_embed_aion.py` (**aion venv**, `multi_gpu_extract`
base, `.npy`) → `12_probe_aion.py` (claudenet venv, `MLPHead`/`LinearHead` via `train_classification`)
→ `13_decorrelation_gate.py`: report recovery@1 %/0.1 % FPR for {AION-probe, EfficientNet (cached
scores), 50/50 avg} + Pearson/Spearman between the two score vectors. **KILL** if r>0.9 *and* average
doesn't beat best single member; **VALIDATE** if r≲0.9 *and* average already beats best single member.

**Phase 1 — flagship engineered-diversity ensemble**
`19_build_member_subsets.py` (deterministic disjoint negative subsets A/B + per-member aug seeds +
PU-guard: drop negatives within 10″ of any known-lens catalog via `lens_mask`). Train 5 decorrelated
members, one per card (independent processes):

| GPU | Script | Member | Diversity lever |
|---|---|---|---|
| 0 | `20_train_member_shielded_A.py` | Shielded-ResNet (`CFG194`), neg-subset A | arch + negatives A |
| 2 | `21_train_member_effnet_B.py` | EfficientNetV2-S, neg-subset B + aug-seed B | arch + negs B + aug |
| 3 | `22_member_aion_probe.py` | AION-1 frozen-embedding probe (real 160 px griz) | different objective/data/bands (max decorrelation) |
| 4 | `23_train_member_equivariant.py` | Cn/Dn-equivariant (dihedral pooling first; escnn) | inductive bias |
| 5 | `24_train_member_zoobot_b0.py` | EfficientNet-B0 morphology-transfer | pretraining corpus |

→ `25_calibrate_members.py` (Platt/isotonic on val; pre/post ECE) → `26_fit_combiner.py`
(out-of-fold member probs; naive-avg vs logistic vs **random-forest** — DES found tree combiners beat
averaging; select on val recovery@1 %FPR) → `27_correlation_report.py` (pairwise error-corr + Q-stat
= the diversity proof) → `28_eval_flagship.py` (matched-FPR table: members / avg / logistic / RF vs the
reproduced `effnet` & `meta`). **Gate:** ship positive only if the best combiner beats *both* best
single member and naive average on Storfer; else report the (still-informative) negative result.

**Phase 2 — iterative hard-negative mining (ratio-fixed)**
`30_score_dr9_negpool.py` (brick-slice a fresh DR9 negative pool, score with Stage-C model) →
`31_select_hard_negatives.py` (top-scoring non-lenses = rings/spirals/edge-on/mergers/blends/artifacts;
**PU-guard**: drop within-10″ known-lens matches; morphology-stratify) → `32_retrain_ratio_fixed.py`
(swap easy→hard negatives **holding neg:pos fixed**, recalibrate, rounds 1–3) →
`33_control_random_negatives.py` (random negs at same ratio = the control isolating *quality*) →
`34_eval_mining_rounds.py` (recovery@1 %FPR per round, hard vs control). The mined model also becomes an
extra base in the Phase-1 combiner.

**Phase 3 — label-efficiency curves** `40_label_efficiency_embed.py` / `41_label_efficiency_curves.py`:
recovery@1 %FPR & labels-to-target-AUC at {5,10,25,50,100}% positive labels for AION-probe vs
Zoobot-probe vs supervised-from-scratch. Deliverable = how many labels each needs to hit the Stage-D
operating point (quantifies the labor saving; honest 2–4× expectation, not folklore 10×).

**Phase 4 — conformal selection** `50_conformal_selection.py` (CPU-only, ~100 lines): split-conformal /
conformal-risk-control on combiner scores → FDR-controlled operating point; validate empirical vs
nominal FDR; note the marginal-exchangeability caveat under north/south shift (use group-conformal).

**Phase 5 — domain adaptation** `60_gen_smalltheta_sims.py` (silver sims, θ_E<0.3″ slice) /
`61_train_dann_mmd.py` (DANN gradient-reversal / MMD alignment; source = sims or DECaLS-south, target =
unlabeled north/real; heteroscedastic mean-variance head for UQ) / `62_eval_domain_shift.py`
(target-domain recovery@1 %FPR + θ_E-stratified completeness vs source-only control). Precedent:
Roncoli/Khullar 2024 ~2× target accuracy on lensing imagery (regression).

**Phase 6 — uncertainty & triage** `70_uncertainty.py` (deep-ensemble variance + MC-dropout epistemic
UQ from the Phase-1 members) / `71_triage_viewer.py` (rank candidates by uncertainty-aware acquisition,
feed `16_build_inspection_viewer.py`). Sold on **inspection efficiency** (lenses per inspected object)
and OOD flagging, not recovery@1 %FPR.

**Phase 7 — equivariance study** `80_train_equivariant_sweep.py` / `81_label_efficiency_equiv.py`:
label-fraction curve for equivariant vs plain CNN (the *unmeasured-on-lenses* data-efficiency claim) +
its correlation with other members (diversity contribution). Framed as data-efficiency/diversity, not
peak accuracy (HOLISMOKES XI: ResNet ≥ G-CNN at the operating point). Large D16 groups → NERSC.

**Reporting** `90_make_tables.py` / `91_make_figures.py` + `papers/` LaTeX tech report
(`../tech-report.sty`), README operator guide, and a `REPRODUCTIONS.md`-style results row.

## Evaluation protocol (single source of truth)
All phases call `_ensemble.recovery_at_fpr(neg_scores, cand_scores, fprs=(0.01,0.001))` — the verbatim
`22_fpr_operating_point.py` quantile-threshold arithmetic. **Held fixed everywhere:** the same
Storfer/Inchausti held-out positives and the same Stage-C/D **test-split** held-out negatives;
calibration uses only the val split. **Primary:** recovery@1 %FPR (Storfer = headline). **Secondary:**
recovery@0.1 %FPR, AUPRC, ECE, labels-to-target-AUC, pairwise error-correlation. Output schema matches
`operating_point.csv` so ClaudeNet rows append to existing tables.

## NERSC scale-up (triggers)
Validate on TITAN RTX first; move to Perlmutter only when GPU-memory- or sky-scale-bound: (1) full
DR9/DR10 sweeps for mining pools & deployment candidate generation; (2) AION-backbone fine-tuning
(frozen-only locally); (3) large escnn D16 groups. `claudenet/nersc/` adapts redshifty's
`ddp_template.slurm` + `export_ddp_vars.sh` + `setup_env.sh`; checkpoints rsync back and evaluate under
the identical harness for direct comparability.

## Risks & mitigations (key)
- **phoenix ssh user/host** unconfirmed (no Host alias) → args + `ssh true` preflight; abort on failure.
- **AION griz unavailable in DR9 north** (`fetch_one`→`None`) → per-object `dup_r` fallback with
  `i_synth` flag; report with/without synthesized-i; never drop northern lenses.
- **Members stay correlated** → `27` correlation report is itself a gate; if Q-stats are high, report
  the negative result (confirms architecture-isn't-the-bottleneck) rather than overclaim.
- **PU-learning trap** (real lenses mined as negatives) → 10″ known-lens exclusion before every retrain.
- **quality/quantity confound** in mining → ratio held fixed + random-negative control.
- **escnn install/compile friction on Turing** → equivariant member optional; proceed with 4 members,
  defer D16 to NERSC.
- **GPU-1 thermal** → never assigned + bus-id watchdog; independent per-card processes contain any
  straggler.

## Verification (end-to-end, before any claim)
1. `00_env_check.py` — both venvs sane; 6 cards at sm_75; GPU-1 excluded by bus-id.
2. `01` count/md5 check vs phoenix (e.g. huang-2020 dr9 ≈ 5837 files); abort on mismatch.
3. `02_smoke_reuse.py` — param-count asserts pass; `score_paths` finite on 8 FITS.
4. Tiny 2-member dry run (`20`+`21`, ~2 epochs, 1 000-row subset, one card each) → confirms per-GPU
   pinning, checkpoint schema, and that `28` emits a CSV end-to-end.
5. **`03_reproduce_baseline.py` passes** (Stage-D `meta` storfer≈0.908/inchausti≈0.968) — proves the
   harness+data+checkpoints are wired correctly so any later delta is real.
6. Then run Phase 0 → 7 autonomously, acting on each gate's criteria and reporting results at each.
