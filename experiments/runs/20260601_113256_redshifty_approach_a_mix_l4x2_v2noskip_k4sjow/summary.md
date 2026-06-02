# redshifty_approach_a_mix_l4x2_v2noskip

_V2-no-skip transformer arm: Approach-A on the 4-way mix using the skip-free V2
tokenizer (checkpoints/tokenizer_v2noskip_l4x2/best.pt, from
redshifty_tokenizer_v2noskip_l4x2). Identical transformer config to the _v1
control and the (failed) _v2 arm — DDP on both L4s, eff batch 64, 15k steps,
bf16, --ar-eval-batches 8 — so the only deliberate change vs V1 is the
tokenizer architecture (ConvNeXt+LFQ+tophat+entropy, no skips/cross-attn).
Tests whether a healthy-codebook V2 variant beats V1's 55% TF z_acc. The full
skip+cross-attn _v2 arm got 0% because its codebook collapsed.

train_transformer.py auto-detects the (no-skip, no-cross-attn) architecture
from the checkpoint keys, so --tokenizer-kind v2 loads this variant correctly.
NOTE: --tokenizer-kind v2 also uses RedshiftTokenizerV2 (1024 levels) vs V1's
256, so redshift_acc is not directly comparable to V1 by exact-bin match —
use the Δz-tolerance metric in analysis.
_

- **run dir:** `/raid/benson/git/agentic-lensing/experiments/runs/20260601_113256_redshifty_approach_a_mix_l4x2_v2noskip_k4sjow`
- **repo:** redshifty
- **started:** 2026-06-01 11:32:56 UTC
- **finished:** 2026-06-01 20:47:41 UTC
- **wallclock:** 33284.2s
- **exit code:** 0
- **tags:** xlarge-scale, transformer, approach-a, track3, l4-ddp, v2-noskip, phase14

## Command
```
/home/benson/.venvs/redshifty/bin/python -m torch.distributed.run --standalone --nproc_per_node=2 nersc/train_transformer.py --manifest /raid/benson/data/desi_dr1_medium/manifest_mix.jsonl --tokenizer-ckpt /raid/benson/data/desi_dr1_medium/checkpoints/tokenizer_v2noskip_l4x2/best.pt --tokenizer-kind v2 --approach a --steps 15000 --batch-size 32 --lr 4e-4 --warmup 500 --healpix-holdout-frac 0.05 --val-every 500 --save-every 2500 --log-every 100 --num-workers 4 --amp --redshift-loss-weight 50.0 --encoder-mask-ratio 0.50 --ar-eval-batches 8 --run-name approach_a_mix_l4x2_v2noskip --scratch-out /raid/benson/data/desi_dr1_medium/checkpoints --wandb-mode disabled --no-push-wandb-artifact
```

## Trajectory

| approach | epoch | train_loss | train_acc | train_redshift_acc | val_loss | val_acc | val_redshift_acc |
|---|---|---|---|---|---|---|---|
|  |  | 393.7913 |  |  |  |  |  |
|  |  | 319.4130 |  |  |  |  |  |
|  |  | 318.3962 |  |  |  |  |  |
|  |  | 311.0014 |  |  |  |  |  |
|  |  | 254.7286 |  |  |  |  |  |
|  |  | 286.0052 |  |  |  |  |  |
|  |  |  |  |  | 316.9580 |  | 0.0045 |
|  |  | 275.0059 |  |  |  |  |  |
|  |  | 262.9690 |  |  |  |  |  |
|  |  | 236.3303 |  |  |  |  |  |
|  |  | 254.4772 |  |  |  |  |  |
|  |  | 289.8646 |  |  |  |  |  |
|  |  |  |  |  | 298.9857 |  | 0.0050 |
|  |  | 245.0254 |  |  |  |  |  |
|  |  | 259.3274 |  |  |  |  |  |
|  |  | 229.4294 |  |  |  |  |  |
|  |  | 249.4493 |  |  |  |  |  |
|  |  | 201.7510 |  |  |  |  |  |
|  |  |  |  |  | 280.1981 |  | 0.0226 |
|  |  | 271.8201 |  |  |  |  |  |
|  |  | 211.1060 |  |  |  |  |  |
|  |  | 256.8301 |  |  |  |  |  |
|  |  | 242.5443 |  |  |  |  |  |
|  |  | 283.1130 |  |  |  |  |  |
|  |  |  |  |  | 264.0324 |  | 0.0255 |
|  |  | 234.0137 |  |  |  |  |  |
|  |  | 181.4241 |  |  |  |  |  |
|  |  | 183.0576 |  |  |  |  |  |
|  |  | 236.3184 |  |  |  |  |  |
|  |  | 181.4032 |  |  |  |  |  |
|  |  |  |  |  | 246.6605 |  | 0.0446 |
|  |  | 205.4951 |  |  |  |  |  |
|  |  | 220.9864 |  |  |  |  |  |
|  |  | 201.1662 |  |  |  |  |  |
|  |  | 228.0038 |  |  |  |  |  |
|  |  | 142.9482 |  |  |  |  |  |
|  |  |  |  |  | 229.8776 |  | 0.1067 |
|  |  | 157.0609 |  |  |  |  |  |
|  |  | 186.1091 |  |  |  |  |  |
|  |  | 184.4464 |  |  |  |  |  |
|  |  | 217.4499 |  |  |  |  |  |
|  |  | 178.2359 |  |  |  |  |  |
|  |  |  |  |  | 216.5187 |  | 0.1403 |
|  |  | 158.3422 |  |  |  |  |  |
|  |  | 192.3103 |  |  |  |  |  |
|  |  | 169.4120 |  |  |  |  |  |
|  |  | 179.7314 |  |  |  |  |  |
|  |  | 211.1362 |  |  |  |  |  |
|  |  |  |  |  | 203.3909 |  | 0.2169 |
|  |  | 153.2951 |  |  |  |  |  |
|  |  | 153.8361 |  |  |  |  |  |
|  |  | 156.4885 |  |  |  |  |  |
|  |  | 185.1228 |  |  |  |  |  |
|  |  | 166.0243 |  |  |  |  |  |
|  |  |  |  |  | 200.0249 |  | 0.2135 |
|  |  | 134.4805 |  |  |  |  |  |
|  |  | 123.0833 |  |  |  |  |  |
|  |  | 181.6986 |  |  |  |  |  |
|  |  | 163.2215 |  |  |  |  |  |
|  |  | 197.1384 |  |  |  |  |  |
|  |  |  |  |  | 186.3101 |  | 0.2968 |
|  |  | 149.5541 |  |  |  |  |  |
|  |  | 177.2920 |  |  |  |  |  |
|  |  | 141.4787 |  |  |  |  |  |
|  |  | 178.5780 |  |  |  |  |  |
|  |  | 187.0423 |  |  |  |  |  |
|  |  |  |  |  | 191.0082 |  | 0.2657 |
|  |  | 177.4365 |  |  |  |  |  |
|  |  | 140.8935 |  |  |  |  |  |
|  |  | 111.0031 |  |  |  |  |  |
|  |  | 229.7743 |  |  |  |  |  |
|  |  | 174.9261 |  |  |  |  |  |
|  |  |  |  |  | 176.4222 |  | 0.3340 |
|  |  | 144.4667 |  |  |  |  |  |
|  |  | 188.8853 |  |  |  |  |  |
|  |  | 207.7705 |  |  |  |  |  |
|  |  | 85.8149 |  |  |  |  |  |
|  |  | 147.4277 |  |  |  |  |  |
|  |  |  |  |  | 170.8771 |  | 0.3500 |
|  |  | 130.9796 |  |  |  |  |  |
|  |  | 117.1719 |  |  |  |  |  |
|  |  | 120.7230 |  |  |  |  |  |
|  |  | 128.6552 |  |  |  |  |  |
|  |  | 192.1052 |  |  |  |  |  |
|  |  |  |  |  | 178.9186 |  | 0.3473 |
|  |  | 145.7193 |  |  |  |  |  |
|  |  | 128.4224 |  |  |  |  |  |
|  |  | 148.4000 |  |  |  |  |  |
|  |  | 183.1177 |  |  |  |  |  |
|  |  | 179.7445 |  |  |  |  |  |
|  |  |  |  |  | 170.7342 |  | 0.3678 |
|  |  | 183.8250 |  |  |  |  |  |
|  |  | 131.1993 |  |  |  |  |  |
|  |  | 114.0866 |  |  |  |  |  |
|  |  | 87.0079 |  |  |  |  |  |
|  |  | 159.7851 |  |  |  |  |  |
|  |  |  |  |  | 164.5554 |  | 0.4068 |
|  |  | 119.0570 |  |  |  |  |  |
|  |  | 151.6854 |  |  |  |  |  |
|  |  | 162.2221 |  |  |  |  |  |
|  |  | 87.7953 |  |  |  |  |  |
|  |  | 136.1005 |  |  |  |  |  |
|  |  |  |  |  | 161.3058 |  | 0.4302 |
|  |  | 112.9597 |  |  |  |  |  |
|  |  | 145.9932 |  |  |  |  |  |
|  |  | 179.8119 |  |  |  |  |  |
|  |  | 196.2041 |  |  |  |  |  |
|  |  | 153.6128 |  |  |  |  |  |
|  |  |  |  |  | 160.2432 |  | 0.4319 |
|  |  | 171.0496 |  |  |  |  |  |
|  |  | 164.7930 |  |  |  |  |  |
|  |  | 102.6571 |  |  |  |  |  |
|  |  | 171.5386 |  |  |  |  |  |
|  |  | 135.8910 |  |  |  |  |  |
|  |  |  |  |  | 162.3707 |  | 0.4120 |
|  |  | 149.9193 |  |  |  |  |  |
|  |  | 140.6632 |  |  |  |  |  |
|  |  | 150.6809 |  |  |  |  |  |
|  |  | 172.1371 |  |  |  |  |  |
|  |  | 214.3223 |  |  |  |  |  |
|  |  |  |  |  | 148.6760 |  | 0.4598 |
|  |  | 207.1721 |  |  |  |  |  |
|  |  | 145.2905 |  |  |  |  |  |
|  |  | 131.5448 |  |  |  |  |  |
|  |  | 194.0958 |  |  |  |  |  |
|  |  | 182.0580 |  |  |  |  |  |
|  |  |  |  |  | 144.7737 |  | 0.4864 |
|  |  | 134.6235 |  |  |  |  |  |
|  |  | 137.8404 |  |  |  |  |  |
|  |  | 117.8875 |  |  |  |  |  |
|  |  | 128.5009 |  |  |  |  |  |
|  |  | 118.1553 |  |  |  |  |  |
|  |  |  |  |  | 146.3690 |  | 0.4828 |
|  |  | 140.5660 |  |  |  |  |  |
|  |  | 135.0030 |  |  |  |  |  |
|  |  | 108.5185 |  |  |  |  |  |
|  |  | 156.4028 |  |  |  |  |  |
|  |  | 146.4078 |  |  |  |  |  |
|  |  |  |  |  | 147.2824 |  | 0.4878 |
|  |  | 94.7935 |  |  |  |  |  |
|  |  | 112.7092 |  |  |  |  |  |
|  |  | 109.0978 |  |  |  |  |  |
|  |  | 145.8863 |  |  |  |  |  |
|  |  | 100.2926 |  |  |  |  |  |
|  |  |  |  |  | 146.0395 |  | 0.4879 |
|  |  | 124.8992 |  |  |  |  |  |
|  |  | 118.0367 |  |  |  |  |  |
|  |  | 132.0135 |  |  |  |  |  |
|  |  | 136.7812 |  |  |  |  |  |
|  |  | 144.5341 |  |  |  |  |  |
|  |  |  |  |  | 150.9896 |  | 0.4661 |
|  |  | 136.6976 |  |  |  |  |  |
|  |  | 109.7738 |  |  |  |  |  |
|  |  | 103.9569 |  |  |  |  |  |
|  |  | 112.6761 |  |  |  |  |  |
|  |  | 157.7493 |  |  |  |  |  |
|  |  |  |  |  | 142.7804 |  | 0.4971 |
|  |  | 114.4844 |  |  |  |  |  |
|  |  | 139.9775 |  |  |  |  |  |
|  |  | 131.7598 |  |  |  |  |  |
|  |  | 129.8457 |  |  |  |  |  |
|  |  | 149.8256 |  |  |  |  |  |
|  |  |  |  |  | 146.9985 |  | 0.4753 |
|  |  | 163.0871 |  |  |  |  |  |
|  |  | 150.1234 |  |  |  |  |  |
|  |  | 96.4046 |  |  |  |  |  |
|  |  | 153.6447 |  |  |  |  |  |
|  |  | 136.3389 |  |  |  |  |  |
|  |  |  |  |  | 143.8608 |  | 0.4905 |
|  |  | 161.8277 |  |  |  |  |  |
|  |  | 189.1353 |  |  |  |  |  |
|  |  | 163.7511 |  |  |  |  |  |
|  |  | 167.9665 |  |  |  |  |  |
|  |  | 157.4687 |  |  |  |  |  |
|  |  |  |  |  | 146.5475 |  | 0.4802 |
|  |  | 146.5171 |  |  |  |  |  |
|  |  | 121.7872 |  |  |  |  |  |
|  |  | 109.4500 |  |  |  |  |  |
|  |  | 154.8105 |  |  |  |  |  |

## Summary stats

```yaml
n_records: 179
step_max: 14900
step_n: 179
train_all_r2_final: 0.774
train_all_r2_first: 0
train_all_r2_max: 0.8
train_all_r2_min: 0
train_loss_final: 154.81
train_loss_first: 393.791
train_loss_max: 393.791
train_loss_min: 85.8149
train_mask_r2_final: 0.777
train_mask_r2_first: 0
train_mask_r2_max: 0.801
train_mask_r2_min: 0
train_masked_acc_final: 0.481
train_masked_acc_first: 0
train_masked_acc_max: 0.539
train_masked_acc_min: 0
train_rz_masked_acc_final: 0.083
train_rz_masked_acc_first: 0
train_rz_masked_acc_max: 0.222
train_rz_masked_acc_min: 0
train_spec_acc_final: 0.475
train_spec_acc_first: 0
train_spec_acc_max: 0.543
train_spec_acc_min: 0
train_spec_loss_final: 1.726
train_spec_loss_first: 7.695
train_spec_loss_max: 7.695
train_spec_loss_min: 1.523
train_z_acc_final: 0.478
train_z_acc_first: 0
train_z_acc_max: 0.682
train_z_acc_min: 0
train_z_loss_final: 3.053
train_z_loss_first: 7.603
train_z_loss_max: 7.603
train_z_loss_min: 1.681
val_loss_final: 146.548
val_loss_first: 316.958
val_loss_max: 316.958
val_loss_min: 142.78
val_loss_redshift_final: 2.8797
val_loss_redshift_first: 6.2707
val_loss_redshift_max: 6.2707
val_loss_redshift_min: 2.8046
val_loss_spectrum_final: 2.283
val_loss_spectrum_first: 3.0199
val_loss_spectrum_max: 3.0199
val_loss_spectrum_min: 2.283
val_loss_total_final: 2.2851
val_loss_total_first: 3.0318
val_loss_total_max: 3.0318
val_loss_total_min: 2.2851
val_masked_spec_acc_final: 0.3403
val_masked_spec_acc_first: 0.22
val_masked_spec_acc_max: 0.3403
val_masked_spec_acc_min: 0.22
val_overall_acc_final: 0.3445
val_overall_acc_first: 0.2181
val_overall_acc_max: 0.3445
val_overall_acc_min: 0.2181
val_redshift_acc_final: 0.4802
val_redshift_acc_first: 0.0045
val_redshift_acc_masked_final: 0.0049
val_redshift_acc_masked_first: 0
val_redshift_acc_masked_max: 0.0161
val_redshift_acc_masked_min: 0
val_redshift_acc_max: 0.4971
val_redshift_acc_min: 0.0045
val_spectrum_acc_final: 0.344
val_spectrum_acc_first: 0.2189
val_spectrum_acc_max: 0.344
val_spectrum_acc_min: 0.2189
```

