# approach_a_mix_l4x2_codecs (Track 1: codecs Mamba3+RFSQ tokenizer)

Codecs tokenizer (layer0 RFSQ, 625 codes; DESI->codecs grid resample) as the
Approach-A spectrum tokenizer; 256-level RedshiftTokenizer (V1-comparable).
2xL4 DDP, eff batch 64, 15k steps, bf16. exit=0, no NaN.

- TF redshift_acc peak 0.5346 @ step 11500 (best.pt); final 0.517
- AR redshift_acc up to ~0.52
- spec_acc plateaus ~0.26 (layer0 codes hard to AR-predict; redshift still ignites)

Verdict: codecs ~ TIES V1 (0.55 TF / 0.57 AR). The Mamba3+RFSQ sequence-model
tokenizer does NOT beat ConvNeXt+LFQ here. Combined with V2-no-skip (also tying
V1), the tokenizer architecture is not the bottleneck at this scale.
