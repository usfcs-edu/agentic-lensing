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
| phot_spec / base | 0.980 | 0.940 | 0.459 | 0.571 | 0.686 |
| phot_spec / large | 0.979 | 0.942 | 0.452 | 0.582 | 0.693 |
| phot_spec / xlarge | 0.973 | 0.921 | 0.413 | 0.556 | 0.665 |
| phot_image_spec / base | 0.985 | 0.937 | 0.432 | 0.570 | 0.685 |
| phot_image_spec / large | 0.976 | 0.940 | 0.408 | 0.591 | 0.687 |
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
| base | 0.718 | 5181 |
| large | 0.747 | 5181 |
| xlarge | 0.706 | 5181 |
| **paper (B)** | 0.840 | ~8000 |

## Tasks 7/8 — Morphology retrieval (Galaxy10 DECaLS), nDCG@10

_Corpus differs from paper's full GZ-DECaLS; best-effort._

| variant | spirals | mergers |
|---|---|---|
| base | 0.821 | 0.685 |
| large | 0.828 | 0.692 |
| xlarge | 0.815 | 0.680 |
| **paper (B)** | 0.938 | 0.892 |
