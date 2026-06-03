# Apple Silicon (MPS) redshifty SpectrumFM-Phase-I reproduction vs. phoenix

Provenance: torch 2.6.0, device=mps, py 3.13.10

Gated on (a) same-checkpoint MPS-vs-CUDA forward fidelity, (b) NaN-free from-scratch
training on MPS that improves, and (c) the reproducible STRUCTURE of the redshift
ignition — the back-half phase transition, the honest AR readout igniting, and a peak
within the phoenix seed-sweep variance band (3.76-8.76%). The exact peak is NOT gated:
it is a high-variance, hardware-path-dependent quantity (bf16 kernels diverge the
10k-step trajectory), reported informationally against the phoenix L4 mix draw
(val_z_acc 14.86% peak @ 9500, val_loss 190.67, val_loss_redshift drop 1.19).

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
| c: ignition/MPS | NaN-free (134 rows) | no NaN | clean | PASS |
| c: ignition/MPS | back-half phase transition (late peak vs early floor) | >=2.5x | 6.0x (1.3%->7.9%) | PASS |
| c: ignition/MPS | AR readout ignites (AR peak vs TF there) | >=0.5xTF | AR=5.8% (1.14xTF) | PASS |
| c: ignition/MPS | peak within phoenix seed-sweep band | >=3.76% | 7.88% | PASS |
| c: ignition/MPS | val_loss_redshift descends | >=0.8 | 0.92 | PASS |
| c: ignition (info) | peak val_z_acc vs phoenix mix draw 14.86% (band 3.76-8.76%) | 14.86% | 7.88% | info |
| c: ignition (info) | doc full-ignition bar: >=10% sustained | >=10% | 7.88% peak (not crossed) | info |
| c: ignition (info) | val_loss min vs phoenix 190.67 | 190.67 | 210.8 | info |

**Gated checks:** 13/13 passed.

## OVERALL: PASS
