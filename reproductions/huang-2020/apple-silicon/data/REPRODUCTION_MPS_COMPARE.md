# Apple Silicon (MPS) reproduction vs. phoenix reference

Provenance: torch 2.6.0, device=mps

Port correctness is gated on identical computation (AUC, MPS inference
fidelity, and reproduction of the phoenix recovery table from phoenix scores).
The DR7-trained recovery/pool reflect an INDEPENDENT from-scratch MPS retrain
(a different RNG draw of the same procedure) and are reported, with a wide
sanity band, not held to bit-reproducibility.

| section | metric | reference | MPS | tol | result |
| :--- | :--- | ---: | ---: | :--- | :--- |
| training | DR9 test_auc | 0.9991 | 0.9988 | >=0.994 | PASS |
| training | DR7 test_auc | 0.9943 | 0.9945 | >=0.994 | PASS |
| mps-fidelity | max|score_mps-score_phoenix| (n=3716) | <=0.001 | 4.14e-04 | <=0.001 | PASS |
| recovery DR9 (gate) | A | 0.900 | 0.900 | +/-0.01 | PASS |
| recovery DR9 (gate) | B | 0.689 | 0.689 | +/-0.01 | PASS |
| recovery DR9 (gate) | C | 0.438 | 0.438 | +/-0.01 | PASS |
| recovery DR9 (gate) | ALL | 0.596 | 0.596 | +/-0.01 | PASS |
| recovery DR7 (sanity) | A  (Δ=+0.000) | 0.833 | 0.833 | +/-0.08 | PASS |
| recovery DR7 (sanity) | B  (Δ=+0.028) | 0.642 | 0.670 | +/-0.08 | PASS |
| recovery DR7 (sanity) | C  (Δ=+0.040) | 0.278 | 0.318 | +/-0.08 | PASS |
| recovery DR7 (sanity) | ALL  (Δ=+0.029) | 0.488 | 0.518 | +/-0.08 | PASS |
| structure | DR7 grade order A>=B>=C | A>=B>=C | 0.833>=0.670>=0.318 |  | PASS |
| structure | leakage gap DR7<=DR9 per grade | all grades | holds |  | PASS |
| pool@0.9 (info) | DR9 n(score>=0.9) | 74,011 | 74,011 | info | PENDING |
| pool@0.9 (info) | DR7 n(score>=0.9) | 25,792 | 38,483 | info | PENDING |
| DR7 model corr (info) | Spearman / Pearson (n=6,240,007) | — | 0.847 / 0.766 | info | PENDING |

**Gated checks:** 13/13 passed  (3 pending).

Note: the MPS DR7-trained model had a marginally higher val AUC than phoenix's
and is correspondingly slightly more sensitive (higher recall, larger candidate
pool). The two DR7 runs are strongly rank-correlated; per-galaxy disagreement at
the p>=0.9 tail is the expected variance of retraining on uniformly-random
negatives (see README caveat 3), not a backend difference.

## OVERALL: PASS
