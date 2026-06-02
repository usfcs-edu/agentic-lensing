# V1 vs V2 vs codecs tokenizer on 2×L4 (Approach-A, eff batch 64, bf16, 15k)

| arm | tokenizer | z-bins | TF z_acc peak | AR z_acc peak | codebook | verdict |
|---|---|---|---|---|---|---|
| V1 | ConvNeXt+LFQ | 256 | 0.55 | 0.57 | 5.00 bits (163 codes) | baseline |
| V2 +skips | +U-Net skips/cross-attn | 1024 | 0.00 | 0.00 | 0.00 bits (collapsed) | dead (skip bypass) |
| V2 no-skip | +tophat/entropy, no skips | 1024 | 0.50 | 0.48 | 5.24 bits (113 codes) | ties V1 |
| codecs | Mamba3+RFSQ (layer0) | 256 | 0.53 | 0.52 | layer0 6.27 bits (233/625) | ties V1 |

**Findings.** (1) The 2×L4 rerun (bf16, eff batch 64) drives V1 to 55% TF / 57% AR — 3.7× the single-A16 mix run (14.86%). (2) The full V2 tokenizer's U-Net skips route reconstruction around the quantizer; its discrete codebook collapses to a single code, so the transformer learns 0% redshift. (3) Skip-free V2 restores a healthy codebook and ignites, but only TIES V1 on the binning-fair |dz|/(1+z) metric. (4) The codecs Mamba3+RFSQ tokenizer (layer0, 256-lvl z so directly V1-comparable) starts ~2.7× slower but catches up by ~step 6k and plateaus at 0.53 TF / 0.52 AR — also a TIE with V1 (its spec_acc plateaus ~0.26, i.e. the layer0 codes resist AR prediction yet still carry the redshift signal).

**Conclusion.** Three structurally different healthy tokenizers (ConvNeXt+LFQ, ConvNeXt+LFQ+tophat, Mamba3+RFSQ) all converge to ~0.50–0.55 redshift accuracy. The tokenizer architecture is NOT the bottleneck at this scale once the discrete codebook is healthy; the lever is data / model size / steps. Reconstruction quality and codebook entropy gate usability, but do not by themselves buy downstream accuracy. (codecs caveat: layer0-only — the 2 residual RFSQ layers were dropped to fit the 1024-code slot; a multi-token expansion could test them.)
