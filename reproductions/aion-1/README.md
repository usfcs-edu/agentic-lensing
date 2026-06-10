# AION-1 reproduction (Parker et al. 2025, arXiv:2510.17960)

Public-data reproduction of the **downstream experiments** of *AION-1: Omnimodal
Foundation Model for Astronomical Sciences* (Polymathic AI). AION-1 is a frozen
omnimodal transformer over 39 astronomical modalities from five surveys. We do
**not** pretrain (the paper used 64–288 H100s for 1.5–3.5 days); we evaluate the
**released checkpoints** `polymathic-ai/aion-{base,large,xlarge}`
(0.3B / 0.9B / 3B params) with lightweight probes on our 7× TITAN RTX, and
compare against the paper's printed AION numbers.

The live headline table is regenerated into [`data/results/REPORT.md`](data/results/REPORT.md)
by `60_make_tables.py`; figures in `figs/` by `61_make_figures.py`.

## Headline result

The flagship reproduces cleanly. With a frozen encoder + the paper's
attentive-pooling head on **PROVABGS galaxy property estimation**, redshift R²
climbs **0.81 (photometry) → 0.98 (+spectrum) → 0.985 (+image+spectrum)**,
landing on the paper's multimodal column (z=1.00, M⋆=0.96):

| config (AION-base) | z | log M⋆ | age | log Z | sSFR |
|---|---|---|---|---|---|
| photometry | 0.81 | 0.75 | 0.35 | 0.38 | 0.61 |
| + image | 0.86 | 0.84 | 0.33 | 0.44 | 0.63 |
| + spectrum | 0.98 | 0.94 | 0.46 | 0.57 | 0.69 |
| + image + spectrum | **0.985** | **0.94** | 0.43 | 0.57 | 0.69 |
| paper (phot+im+spec) | 1.00 | 0.96 | 0.53 | 0.61 | 0.72 |

**Stellar parameters from Gaia XP (task 3)** match/beat the paper's residuals
(Teff 89–95 K vs 94.6; logg 0.19–0.20 vs 0.206; [Fe/H] 0.105–0.11 vs 0.115).

## The 11 experiments

| # | task | metric | status |
|---|------|--------|--------|
| 1 | Galaxy property estimation (PROVABGS) | R² (5 props) | ✅ B/L/XL, all 4 configs |
| 2 | Stellar params (DESI×DD-Payne, 26k) | R² (4 params) | ✅ spectrum; +parallax config added |
| 3 | Stellar params (APOGEE×Gaia XP) | residual σ | ✅ matches paper |
| 4 | Galaxy morphology (Galaxy10) | accuracy | ✅ (corpus-limited, see below) |
| 5 | Galaxy segmentation (GZ3D) | IoU | ✅ spiral/bar conv upsampler |
| 6 | Low-data regime | R²/acc vs N | ✅ saturation matches |
| 7 | Retrieval: spirals | nDCG@10 | ⚠️ best-effort corpus |
| 8 | Retrieval: mergers | nDCG@10 | ⚠️ best-effort corpus |
| 9 | Strong-lens retrieval (SuGOHI) | nDCG@10 | ✅ grade-A/B lenses |
| 10 | Redshift posterior (generative) | contraction | ✅ phot→+spec |
| 11 | Spectral super-resolution (generative) | line recovery | ✅ |

## Data

The AION tutorials' pre-joined convenience files are gone (404); the canonical
replacement is the **Multimodal Universe** HF datasets, assembled by our fetch
scripts:

- `desi_provabgs` (labels+photometry) × `desi` spectra — key-join on TARGETID →
  20.7k galaxies with image+spectrum (task 1). Legacy Survey g,r,i,z images come
  from the LS cutout service by RA/Dec (`_ls_cutout.py`), since MMU `legacysurvey`
  carries no RA/Dec.
- DD-Payne DESI stellar labels (Zhang+2024) × `desi` spectra → 26k stars; Gaia DR3
  parallax via CDS XMatch (97% matched) for the `+parallax` config (task 2).
- `MultimodalUniverse/gaia` × APOGEE for task 3; `gz10` / Galaxy10 DECaLS for tasks 4/7/8.
- SuGOHI lens master list (Oguri/U-Tokyo) for task 9; GZ3D VAC (Masters+2021) masks
  for task 5.

## How to run

```bash
PY=/home2/benson/.venvs/aion/bin/python        # py3.11, torch 2.6 cu124
export HF_HOME=/home2/benson/.cache/huggingface

# M0 smoke (env, codecs, all 3 checkpoints, torch.compile on sm_75)
$PY 00_env_check.py && $PY 01_smoke_codecs.py && $PY 02_smoke_model.py

# Task 1 (flagship): fetch -> embed -> probe
$PY 03_fetch_provabgs.py && $PY 04_fetch_desi_spectra.py && $PY 05_fetch_ls_images.py --index spec
for cfg in phot phot_spec phot_image phot_image_spec; do
  $PY 10_embed_provabgs.py --config $cfg --gpus 0,2,3,4,5,6
  $PY 20_probe_provabgs.py --config $cfg
done
$PY 50_redshift_posterior.py --variant base   # task 10
$PY 25_lowdata.py --variant base              # task 6

# Stellar (tasks 2,3,11)
$PY 08_fetch_ddpayne_desi.py && $PY 08b_fetch_parallax.py
$PY 13_embed_ddpayne.py --config desi      && $PY 23_probe_ddpayne.py --config desi
$PY 13_embed_ddpayne.py --config desi_plx  && $PY 23_probe_ddpayne.py --config desi_plx
$PY 07_xmatch_gaia_apogee.py && $PY 11_embed_gaia_xp.py && $PY 21_probe_gaia_apogee.py
$PY 09_xmatch_gaia_desi.py && $PY 51_spectral_superres.py

# Morphology / retrieval / segmentation (tasks 4,7,8,9,5)  [LS cutouts are slow/rate-limited]
$PY 06_fetch_gz10_images.py && $PY 06b_finalize_gz10.py && $PY 12_embed_gz10.py && $PY 22_probe_gz10.py && $PY 40_retrieve_gz10.py
$PY 41_fetch_sugohi.py && $PY 42_retrieve_lenses.py
$PY 30_fetch_gz3d.py && $PY 31_seg_gz3d.py

# Report
$PY 60_make_tables.py && $PY 61_make_figures.py
make -C papers pdf
```

The autonomous driver chain is `run_main_gpu.sh` / `run_gz10_gpu.sh` (sequenced
via `_watch.sh`), with `run_fixes.sh` / `run_remaining.sh` for the M4 tail.

## Honest status / limitations

- **xlarge ≤ base on photometry-only** — expected; 4 scalars is too little signal
  for a 3B frozen model. Scaling helps in the image+spectrum configs.
- **Tasks 4 / 7 / 8 are corpus-limited by the LS cutout service** (~20 cutouts/min,
  per-IP). Morphology (task 4) grows toward the full 17.7k Galaxy10 set. For
  retrieval (tasks 7/8) we run **two tiers**: a quick Galaxy10 best-effort
  (`40_retrieve_gz10.py`) and a **faithful GZ-DECaLS reproduction**
  (`43_fetch_gzdecals_campaign.py` + `44_retrieve_gzdecals.py`) that uses the
  published Walmsley+2022 vote fractions to define high-confidence spirals
  (~24k) and the rare mergers (~3.4k), fetched priority-first (mergers → spirals
  → distractors) into a 63k-image corpus over a multi-day, resumable campaign —
  matching the paper's rare-positive setup. The campaign runs on **two IPs in
  parallel** (this host + phoenix via `phoenix_fetch.py`, cache-key-compatible,
  rsync-merged by `run_phoenix_sync.sh`) for ~2× griz throughput.

  Pre-imaged shortcuts were investigated and ruled out for tasks 7/8: MMU
  `ssl_legacysurvey` is grz-only with no RA/Dec; phoenix has ~550k DECaLS cutouts
  but they are (a) grz-only — though AION's masked-modeling image codec *does*
  accept a band subset (verified: 3-band grz encodes fine, i-band masked) — and
  (b) a lens-search sample that overlaps GZ-DECaLS by only ~1.4k galaxies, with
  no brick coadds on disk to generate new cutouts (`phoenix_overlap_check.py`).
  Phoenix's value is therefore as a second cutout-fetching IP, not a pre-imaged
  source.
- **Task 9 positives** are SuGOHI grade-A/B candidates (human-graded, not all
  spectroscopically confirmed); distractors are PROVABGS galaxies.
- **Undocumented paper details** (seeds, exact splits, head hyperparams) → we fix
  `SEED=2026`, AdamW head-only, token-aware batch. Absolute numbers are *targets*,
  not pass/fail; we always print ours next to the paper's.

## Key files

| file | purpose |
|---|---|
| `_config.py` | seeds, model ids, paths, `paper_targets()` |
| `_aion_embed.py` | frozen-encoder embedding, file-backed multi-GPU sharding |
| `_probe.py` | `CrossAttnHead` (attentive pooling), `MLPHead`, `LinearHead`, `SegHead` + train loops |
| `_retrieval.py` / `_metrics.py` | nDCG@10; R²/accuracy/IoU/residual-σ |
| `_data_mmu.py` / `_ls_cutout.py` | MMU loaders; LS cutout fetcher (429-backoff) |
| `NN_*.py` | numbered fetch / embed / probe / retrieve / seg / report scripts |
| `data/results/*.json`, `REPORT.md` | per-task numbers + comparison table |
| `papers/main.tex` | tech report (`make -C papers pdf`) |
