# Phase-14: V1 vs V2 tokenizer on 2×L4 (Approach-A, eff batch 64, bf16)

| arm | z-bins | TF z_acc peak | AR z_acc peak | codebook entropy | fair DESI good-z (<0.0033) |
|---|---|---|---|---|---|
| V1 (ConvNeXt+LFQ) | 256 | 0.55 | 0.57 | 5.00 bits (163 codes) | 0.192 |
| V2 +skips | 1024 | 0.00 | 0.00 | **0.00 bits (1 code, collapsed)** | n/a |
| V2 no-skip | 1024 | 0.50 | 0.48 | 5.24 bits (113 codes) | 0.192 |

**Findings.** (1) The 2×L4 rerun (bf16, eff batch 64) drives V1 to 55% TF / 57% AR — 3.7× the single-A16 mix run (14.86%). (2) The V2 tokenizer's U-Net skip connections route reconstruction info around the quantizer; its discrete codebook collapses to a single code, so the transformer (which sees only the codes) cannot learn redshift (0%). (3) Removing the skips restores a healthy codebook and redshift ignition, but on the binning-fair |dz|/(1+z) metric V2-no-skip merely TIES V1 (0.192 DESI good-z both). The V2 tokenizer offers no downstream redshift benefit over V1; its 8.6× reconstruction advantage was a skip-connection artifact. Reconstruction quality is not a valid proxy for tokenizer quality when skips bypass the discrete bottleneck.
