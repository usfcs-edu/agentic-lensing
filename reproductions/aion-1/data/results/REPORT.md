# AION-1 reproduction — headline results

Frozen released checkpoints (`polymathic-ai/aion-{base,large,xlarge}`), our probes vs the paper's printed AION numbers. SEED=2026.

## Task 1 — Galaxy property estimation (PROVABGS), R²

| config / variant | z | logmass | age | logZ | sSFR |
|---|---|---|---|---|---|
| phot / base | 0.807 | 0.752 | 0.346 | 0.378 | 0.614 |
| phot / large | 0.789 | 0.755 | 0.335 | 0.370 | 0.601 |
| phot / xlarge | 0.756 | 0.699 | 0.277 | 0.319 | 0.558 |
| phot_image / base | 0.857 | 0.842 | 0.333 | 0.440 | 0.625 |
| phot_image / large | 0.887 | 0.842 | 0.341 | 0.445 | 0.600 |
| phot_image / xlarge | 0.864 | 0.815 | 0.315 | 0.418 | 0.620 |
| phot_spec / base | 0.980 | 0.940 | 0.459 | 0.571 | 0.686 |
| phot_spec / large | 0.979 | 0.942 | 0.452 | 0.582 | 0.693 |
| phot_spec / xlarge | 0.973 | 0.921 | 0.413 | 0.556 | 0.665 |
| phot_image_spec / base | 0.985 | 0.937 | 0.432 | 0.570 | 0.685 |
| phot_image_spec / large | 0.976 | 0.940 | 0.408 | 0.591 | 0.687 |
| phot_image_spec / xlarge | 0.963 | 0.913 | 0.304 | 0.538 | 0.651 |
| **paper (phot+im+spec, B)** | 1.00 | 0.96 | 0.53 | 0.61 | 0.72 |

## Task 3 — APOGEE×GaiaXP stellar params (residual std)

| variant | Teff (K) | logg (dex) | [Fe/H] (dex) | N |
|---|---|---|---|---|
| base | 108.4 | 0.231 | 0.108 | 1261 |
| large | 89.5 | 0.240 | 0.110 | 1261 |
| xlarge | 97.1 | 0.190 | 0.112 | 1261 |
| **paper (B)** | 94.6 | 0.206 | 0.115 | ~10000 |

## Task 4 — Galaxy morphology (Galaxy10 DECaLS), accuracy

| variant | accuracy | N |
|---|---|---|
| base | 0.732 | 11381 |
| large | 0.749 | 11381 |
| xlarge | 0.704 | 11381 |
| **paper (B)** | 0.840 | ~8000 |

## Tasks 7/8 — Morphology retrieval (Galaxy10 DECaLS), nDCG@10

_Corpus differs from paper's full GZ-DECaLS; best-effort._

| variant | spirals | mergers |
|---|---|---|
| base | 0.873 | 0.677 |
| large | 0.874 | 0.691 |
| xlarge | 0.865 | 0.668 |
| **paper (B)** | 0.938 | 0.892 |

## Task 10 — Redshift posterior (generative)

| variant/config | point R² | mean post. std | NLL@true |
|---|---|---|---|
| base/phot | -9.341 | 0.6126 | 4.414 |
| base/phot_spec | 0.937 | 0.0339 | 7.464 |

## Task 2 — Stellar params (DESI×DD-Payne), R²

| config / variant | Teff | logg | FeH | vmic |
|---|---|---|---|---|
| desi / base | 0.957 | 0.889 | 0.802 | 0.458 |
| desi / large | 0.953 | 0.882 | 0.804 | 0.455 |
| desi / xlarge | 0.957 | 0.860 | 0.788 | 0.449 |
| desi_plx / base | 0.959 | 0.892 | 0.803 | 0.458 |
| desi_plx / large | 0.952 | 0.878 | 0.804 | 0.461 |
| desi_plx / xlarge | 0.960 | 0.873 | 0.779 | 0.428 |
| **paper (DESI+Plx, B)** | 0.99 | 0.98 | 0.94 | 0.89 |

_Note: +parallax gives ~no logg gain on the frozen encoder (paper's number likely needs finetuning)._

## Task 5 — Galaxy structure segmentation (GZ3D), IoU

| variant | spiral arms | bar | N |
|---|---|---|---|
| base | 0.521 | 0.295 | 2000 |
| large | 0.528 | 0.304 | 2000 |
| xlarge | 0.522 | 0.275 | 2000 |
| **paper (B)** | 0.6 | 0.31 | ~2800 |

## Task 9 — Strong-lens retrieval (SuGOHI), nDCG@10

_LegacySurvey lenses (paper uses HSC); corpus less rare than paper._

| variant | nDCG@10 | corpus | lenses |
|---|---|---|---|
| base | 0.802 | 13436 | 1436 |
| large | 0.796 | 13436 | 1436 |
| xlarge | 0.804 | 13436 | 1436 |
| **paper (B, HSC)** | 0.968 | — | — |

## Tasks 7/8 (faithful) — GZ-DECaLS retrieval, nDCG@10

_Walmsley+2022 vote-fraction positives in a 63k rare-positive corpus (real griz, two-machine campaign)._

| variant | spirals | mergers | corpus |
|---|---|---|---|
| base | 0.830 | 0.515 | 63448 |
| large | 0.832 | 0.517 | 63448 |
| xlarge | 0.817 | 0.518 | 63448 |
| **paper (B)** | 0.938 | 0.892 | ~171k |

## Task 11 — Spectral super-resolution (Gaia XP→DESI)

| variant | median corr | mean corr | N |
|---|---|---|---|
| base | 0.828 | 0.787 | 300 |

_Qualitative in paper (line recovery); high corr = good reconstruction._

## Task 6 — Low-data regime (PROVABGS), z R² vs #labels

- **phot** (base): N=[100, 300, 1000, 3000, 10000] → z R²=[0.56, 0.66, 0.75, 0.77, 0.80]
- **phot_spec** (base): N=[100, 300, 1000, 3000, 10000] → z R²=[0.78, 0.92, 0.94, 0.97, 0.98]

_Paper: performance saturates by 10³–10⁴ labels (reproduced)._
