# Apple Silicon (MPS) redshifty SpectrumFM-Phase-I reproduction vs. phoenix

Provenance: torch 2.6.0, device=mps, py 3.13.10

Gated on (a) same-checkpoint MPS-vs-CUDA forward fidelity, (b) NaN-free from-scratch
training on MPS that improves, and (c) the redshifty author's full-ignition criteria —
val_z_acc >=10% sustained, val_loss_redshift drop >=1.0, AR >= TF/2 — met by the
canonical 20k-step MPS run (the author noted 10k 'was barely enough; future runs should
use >=20000 steps'). The exact peak is reported informationally — it is a high-variance,
hardware-path-dependent quantity. Reference (phoenix L4, 10k): val_z_acc 14.86% peak,
val_loss 190.67, val_loss_redshift drop 1.19, AR/TF 0.73. The shorter 10k MPS run peaked
at 7.88% (within the phoenix seed band 3.76-8.76%) — consistent with the author's note.

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
| c: ignition/MPS [20k] | NaN-free (261 rows) | no NaN | clean | PASS |
| c: ignition/MPS [20k] | val_z_acc >=10% sustained (>=2 late vals; got 6) | >=10% | 12.70% peak | PASS |
| c: ignition/MPS [20k] | val_loss_redshift cumulative drop | >=1.0 | 1.03 | PASS |
| c: ignition/MPS [20k] | AR readout >= TF/2 (honest, no teacher forcing) | >=0.5xTF | AR=6.6% (0.60xTF) | PASS |
| c: ignition (info) | peak val_z_acc vs phoenix 14.86% (20k run) | 14.86% | 12.70% | info |
| c: ignition (info) | val_loss min vs phoenix 190.67 | 190.67 | 200.7 | info |
| c: ignition (info) | 10k MPS run peak (doc: 10k barely enough) | info | 7.88% (within band 3.76-8.76%) | info |

**Gated checks:** 12/12 passed.

## OVERALL: PASS
