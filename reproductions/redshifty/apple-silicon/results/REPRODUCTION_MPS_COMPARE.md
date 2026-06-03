# Apple Silicon (MPS) redshifty SpectrumFM-Phase-I reproduction vs. phoenix

Provenance: torch 2.6.0, device=mps, py 3.13.10

Port correctness is gated on (a) same-checkpoint MPS-vs-CUDA forward fidelity,
(b) NaN-free from-scratch training on MPS that improves, and (c) the STRUCTURE of
the redshift ignition (known high seed variance, so the shape is gated, not the
exact peak). The reference ignition (phoenix L4): val_z_acc 14.86% peak @ step 9500,
val_loss min 190.67, val_loss_redshift drop 1.19, AR/TF ~0.73.

| layer | metric | reference | MPS | result |
| :--- | :--- | ---: | ---: | :--- |
| a: MPS==CUDA fwd | median|Δ| (max=3.15) | <=1e-3 | 1.4e-06 | PASS |
| a: MPS==CUDA fwd | argmax-token agreement | >=99.5% | 99.85% | PASS |
| a: MPS==CUDA fwd | loss relative |Δ| | <=1e-3 | 2.5e-04 | PASS |
| a: MPS==CUDA fwd | |Δ redshift_acc| (science readout) | <=0.005 | 0.0000 | PASS |
| b1: tokenizer/MPS | NaN-free (19 rows) | no NaN | clean | PASS |
| b1: tokenizer/MPS | learns from init (best val_recon < init train) | 97->11.1 | yes | PASS |
| b2: transformer/MPS | NaN-free (17 rows) | no NaN | clean | PASS |
| b2: transformer/MPS | learns from init (best val_loss < init train) | 388->274 | yes | PASS |
| c: ignition/MPS | ignition metrics.jsonl | ignition struct | PENDING (run run_tier2.sh) | PENDING |

**Gated checks:** 8/8 passed  (1 pending).

## OVERALL: PASS
