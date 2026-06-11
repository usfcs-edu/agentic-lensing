# Apple Silicon (M4 Max / MPS) ports

Can a Mac Studio (M4 Max, 128 GB unified memory, PyTorch MPS backend) serve as a
from-scratch model workhorse for this corpus? Three ports say yes: two
image-CNN lens-finding pipelines and one transformer / foundation-model
pipeline, each re-run end-to-end on Apple Silicon and gated against the CUDA
reference (phoenix) with automated port-correctness checks.

[:material-github: View on GitHub](https://github.com/usfcs-edu/agentic-lensing/tree/main/reproductions){ .md-button }

## The three ports

**[Huang 2020 port](huang-2020.md)** — the DECaLS ResNet lens finder, all
computation (catalog filtering, negatives, both ResNet trainings, inference) on
MPS. Passes **13/13** gated port-correctness checks: DR9/DR7 test AUC 0.9988/0.9945
vs phoenix's 0.9991/0.9943, recovery-by-grade identical, and each ResNet trains in
~12 min on MPS (≈ half the L4's 25 min). The full 6.24M-galaxy DR7 sweep ran at
~1.4 bricks/s (~24 h).

**[Huang 2021 port](huang-2021.md)** — the shielded-ResNet DR8 pipeline: four
from-scratch trainings (L18 + 59,905-parameter shielded net + the north-augmentation
retrains), two-model ensemble inference, and leak-aware recovery. Passes **29/29**
gated checks with the whole Tier-1 pipeline in **~1h33m** on a single MPS device;
two-model recovery and leak-free recovery match phoenix exactly (83.2/81.8/76.1%
and 50.4%).

**[Redshifty port](redshifty.md)** — the SpectrumFM Phase-I transformer
(ConvNeXt+LFQ tokenizer → Approach-A transformer), the first run of this stack
off CUDA. Three validation layers: same-checkpoint MPS-vs-CUDA forward fidelity
(median |Δ| logits 1.4e-6, redshift_acc identical), NaN-free from-scratch
training with bf16 genuinely engaging on MPS, and the capstone — the **full-mix
redshift ignition reproduced** on MPS in a 20k-step run (`val_z_acc` 12.70% peak
with six late validations ≥ 10%, meeting all of the reference run's own gates).

## Why it matters

The ports validate a commodity desktop as a credible third compute tier next to
the lab's CUDA hosts: identical science numbers where determinism allows,
documented MPS-specific pitfalls where it doesn't (the `non_blocking` host→device
race that NaNs training from step 1, `autocast("cuda")` silently running fp32 on
Mac, fp16 vs bf16 divergence). Each port directory carries the patch set, the
verification script, and a machine-checked comparison report.
