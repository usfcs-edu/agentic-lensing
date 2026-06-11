# Apple Silicon (MPS) Huang-2021 reproduction vs. phoenix reference

Provenance: torch 2.6.0, device=mps

Port correctness is gated on identical computation: the four from-scratch test
AUCs clearing their floors, MPS-vs-CUDA inference fidelity for the same
checkpoints, reproduction of the leak-aware recovery table from the phoenix
scores, the published-catalog grade counts, and the north-augmentation
false-positive collapse. The absolute AUC values reflect an INDEPENDENT
from-scratch MPS retrain and are reported with their deltas.

| section | metric | reference | MPS | tol | result |
| :--- | :--- | ---: | ---: | :--- | :--- |
| training | shielded_dr9 test_auc  (Δ=+0.0008) | 0.9988 | 0.9996 | >=0.995 | PASS |
| training | shielded_dr7 test_auc  (Δ=-0.0011) | 0.9955 | 0.9944 | >=0.99 | PASS |
| training | l18_northaug test_auc  (Δ=+0.0013) | 0.9985 | 0.9998 | >=0.995 | PASS |
| training | shielded_northaug test_auc  (Δ=-0.0011) | 0.9996 | 0.9985 | >=0.995 | PASS |
| mps-fidelity | l18 p99.9|Δ|  (max=9.8e-04, n=17506) | <=0.001 | 3.87e-04 | <=0.001 | PASS |
| mps-fidelity | shielded p99.9|Δ|  (max=1.1e-03, n=17506) | <=0.001 | 4.69e-04 | <=0.001 | PASS |
| recovery (gate) | all/combined A p>=0.1 | 0.861 | 0.861 | +/-0.01 | PASS |
| recovery (gate) | all/combined A p>=0.5 | 0.847 | 0.847 | +/-0.01 | PASS |
| recovery (gate) | all/combined A p>=0.9 | 0.764 | 0.764 | +/-0.01 | PASS |
| recovery (gate) | all/combined B p>=0.1 | 0.834 | 0.834 | +/-0.01 | PASS |
| recovery (gate) | all/combined B p>=0.5 | 0.829 | 0.829 | +/-0.01 | PASS |
| recovery (gate) | all/combined B p>=0.9 | 0.749 | 0.749 | +/-0.01 | PASS |
| recovery (gate) | all/combined C p>=0.1 | 0.825 | 0.825 | +/-0.01 | PASS |
| recovery (gate) | all/combined C p>=0.5 | 0.808 | 0.808 | +/-0.01 | PASS |
| recovery (gate) | all/combined C p>=0.9 | 0.764 | 0.764 | +/-0.01 | PASS |
| recovery (gate) | all/combined ALL p>=0.1 | 0.832 | 0.832 | +/-0.01 | PASS |
| recovery (gate) | all/combined ALL p>=0.5 | 0.818 | 0.818 | +/-0.01 | PASS |
| recovery (gate) | all/combined ALL p>=0.9 | 0.761 | 0.761 | +/-0.01 | PASS |
| structure | leaked recall >= honest (combined p>=0.9) | all grades | holds |  | PASS |
| honest recall (info) | leak-free combined ALL p>=0.9 | 0.504 | 0.504 | info | PENDING |
| catalog | grade A | 216 | 216 | +/-10 | PASS |
| catalog | grade B | 199 | 199 | +/-9 | PASS |
| catalog | grade C | 897 | 897 | +/-44 | PASS |
| catalog | grade total | 1312 | 1312 | +/-65 | PASS |
| northaug (gate) | post north non-lens >=0.1  (pre=65.4%, n=295) | <=5% | 0.3% | <=5% | PASS |
| structure | shielded params | 59,905 | 59,905 | == | PASS |
| structure | |shielded-L18| AUC DR9 | <=0.005 | 0.0005 | <=0.005 | PASS |
| structure | L18 params (DR9) | 3,508,833 | 3,508,833 | == | PASS |
| structure | |shielded-L18| AUC DR7 | <=0.005 | 0.0001 | <=0.005 | PASS |
| structure | L18 params (DR7) | 3,508,833 | 3,508,833 | == | PASS |

**Gated checks:** 29/29 passed  (1 pending).

## OVERALL: PASS
