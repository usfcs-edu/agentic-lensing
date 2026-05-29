# redshifty_tokenizer_v1_medium

_V1 ConvNeXt+LFQ tokenizer pretrain on local 33-pixel/11.5k-spectrum medium subset_

- **run dir:** `/raid/benson/git/agentic-lensing/experiments/runs/20260526_230118_redshifty_tokenizer_v1_medium_2yozeb`
- **repo:** redshifty
- **started:** 2026-05-26 23:01:19 UTC
- **finished:** 2026-05-26 23:45:22 UTC
- **wallclock:** 2643.1s
- **exit code:** 0
- **tags:** medium-real, tokenizer, track3

## Command
```
/home/benson/.venvs/redshifty/bin/python nersc/pretrain_tokenizer.py --manifest /raid/benson/data/desi_dr1_medium/manifest.jsonl --steps 5000 --batch-size 16 --lr 3e-4 --warmup 200 --val-frac 0.05 --val-every 250 --save-every 1000 --log-every 50 --num-workers 4 --amp --run-name tokenizer_v1_medium --scratch-out /raid/benson/data/desi_dr1_medium/checkpoints --wandb-mode disabled --no-push-wandb-artifact
```

## Trajectory

| approach | epoch | train_loss | train_acc | train_redshift_acc | val_loss | val_acc | val_redshift_acc |
|---|---|---|---|---|---|---|---|
|  |  | 146.3657 |  |  |  |  |  |
|  |  | 3.3894 |  |  |  |  |  |
|  |  | 20.2218 |  |  |  |  |  |
|  |  | 5.0853 |  |  |  |  |  |
|  |  | 17.6512 |  |  |  |  |  |
|  |  | 4.6453 |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
|  |  | 2.9697 |  |  |  |  |  |
|  |  | 2.2907 |  |  |  |  |  |
|  |  | 7.8251 |  |  |  |  |  |
|  |  | 17.1978 |  |  |  |  |  |
|  |  | 8.6023 |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
|  |  | 2.6417 |  |  |  |  |  |
|  |  | 2.7208 |  |  |  |  |  |
|  |  | 3.1073 |  |  |  |  |  |
|  |  | 9.7328 |  |  |  |  |  |
|  |  | 5.9280 |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
|  |  | 2.0505 |  |  |  |  |  |
|  |  | 2.6793 |  |  |  |  |  |
|  |  | 2.5951 |  |  |  |  |  |
|  |  | 2.9848 |  |  |  |  |  |
|  |  | 3.7521 |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
|  |  | 5.0623 |  |  |  |  |  |
|  |  | 5.3682 |  |  |  |  |  |
|  |  | 3.6104 |  |  |  |  |  |
|  |  | 3.5000 |  |  |  |  |  |
|  |  | 6.0576 |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
|  |  | 2.9569 |  |  |  |  |  |
|  |  | 8.6004 |  |  |  |  |  |
|  |  | 7.1080 |  |  |  |  |  |
|  |  | 2.8861 |  |  |  |  |  |
|  |  | 4.1808 |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
|  |  | 2.1630 |  |  |  |  |  |
|  |  | 3.1681 |  |  |  |  |  |
|  |  | 16.1033 |  |  |  |  |  |
|  |  | 4.4913 |  |  |  |  |  |
|  |  | 5.4743 |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
|  |  | 2.4682 |  |  |  |  |  |
|  |  | 3.9922 |  |  |  |  |  |
|  |  | 2.3097 |  |  |  |  |  |
|  |  | 2.8990 |  |  |  |  |  |
|  |  | 2.6365 |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
|  |  | 2.8835 |  |  |  |  |  |
|  |  | 35.3732 |  |  |  |  |  |
|  |  | 2.1972 |  |  |  |  |  |
|  |  | 4.5870 |  |  |  |  |  |
|  |  | 3.3411 |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
|  |  | 2.3431 |  |  |  |  |  |
|  |  | 2.3164 |  |  |  |  |  |
|  |  | 2.4248 |  |  |  |  |  |
|  |  | 8.0367 |  |  |  |  |  |
|  |  | 4.4466 |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
|  |  | 2.5977 |  |  |  |  |  |
|  |  | 2.6637 |  |  |  |  |  |
|  |  | 12.3423 |  |  |  |  |  |
|  |  | 2.5884 |  |  |  |  |  |
|  |  | 3.6480 |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
|  |  | 8.9229 |  |  |  |  |  |
|  |  | 2.6613 |  |  |  |  |  |
|  |  | 3.2811 |  |  |  |  |  |
|  |  | 2.5174 |  |  |  |  |  |
|  |  | 3.1832 |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
|  |  | 6.3700 |  |  |  |  |  |
|  |  | 2.7302 |  |  |  |  |  |
|  |  | 3.6356 |  |  |  |  |  |
|  |  | 2.1366 |  |  |  |  |  |
|  |  | 6.6483 |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
|  |  | 4.2182 |  |  |  |  |  |
|  |  | 2.8255 |  |  |  |  |  |
|  |  | 3.6472 |  |  |  |  |  |
|  |  | 7.5015 |  |  |  |  |  |
|  |  | 1.8975 |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
|  |  | 2.9624 |  |  |  |  |  |
|  |  | 4.1053 |  |  |  |  |  |
|  |  | 1.8416 |  |  |  |  |  |
|  |  | 18.3071 |  |  |  |  |  |
|  |  | 3.6998 |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
|  |  | 2.4726 |  |  |  |  |  |
|  |  | 3.4949 |  |  |  |  |  |
|  |  | 2.8888 |  |  |  |  |  |
|  |  | 2.8497 |  |  |  |  |  |
|  |  | 2.7331 |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
|  |  | 2.5756 |  |  |  |  |  |
|  |  | 2.2389 |  |  |  |  |  |
|  |  | 4.8054 |  |  |  |  |  |
|  |  | 2.1993 |  |  |  |  |  |
|  |  | 2.6041 |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
|  |  | 4.8548 |  |  |  |  |  |
|  |  | 2.2967 |  |  |  |  |  |
|  |  | 4.5701 |  |  |  |  |  |
|  |  | 2.1990 |  |  |  |  |  |
|  |  | 2.7978 |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
|  |  | 2.6188 |  |  |  |  |  |
|  |  | 5.6675 |  |  |  |  |  |
|  |  | 3.7604 |  |  |  |  |  |
|  |  | 9.4631 |  |  |  |  |  |
|  |  | 4.0947 |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
|  |  | 2.5288 |  |  |  |  |  |
|  |  | 2.3372 |  |  |  |  |  |
|  |  | 2.8648 |  |  |  |  |  |
|  |  | 5.1295 |  |  |  |  |  |

## Summary stats

```yaml
lr_final: 3.01e-05
lr_first: 1.5e-06
lr_max: 0.0003
lr_min: 1.5e-06
n_records: 119
step_max: 4950
step_n: 119
steps_per_sec_final: 1.9
steps_per_sec_first: 0.1
steps_per_sec_max: 1.9
steps_per_sec_min: 0.1
train_loss_final: 5.1295
train_loss_first: 146.366
train_loss_max: 146.366
train_loss_min: 1.8416
train_loss_quant_final: 0.1191
train_loss_quant_first: 0.2429
train_loss_quant_max: 0.2429
train_loss_quant_min: 0.0429
train_loss_recon_final: 5.0104
train_loss_recon_first: 146.123
train_loss_recon_max: 146.123
train_loss_recon_min: 1.7425
val_quant_final: 0.1119
val_quant_first: 0.0532
val_quant_max: 0.1139
val_quant_min: 0.0532
val_recon_final: 4.5442
val_recon_first: 7.0728
val_recon_max: 7.0728
val_recon_min: 4.0827
val_total_final: 4.6561
val_total_first: 7.126
val_total_max: 7.126
val_total_min: 4.1966
```

