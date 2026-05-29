# redshifty_approach_a_phase10_xlarge

_Approach A transformer matching Phase 10 final NERSC hparams (batch=32, lr=4e-4)
on the full sv3-bright tree (373 pixels / 729,898 spectra — 1.85× the author's
394k Phase 10 dataset). Tests whether the residual ignition gap (peak 8.2% in
phase10 on 219k vs author's 73.8% on 394k) closes with more data.
_

- **run dir:** `/raid/benson/git/agentic-lensing/experiments/runs/20260528_082256_redshifty_approach_a_phase10_xlarge_zytk0w`
- **repo:** redshifty
- **started:** 2026-05-28 08:22:56 UTC
- **finished:** 2026-05-28 15:42:00 UTC
- **wallclock:** 26344.3s
- **exit code:** 0
- **tags:** xlarge-scale, transformer, approach-a, track3, diagnostic, phase10-match

## Command
```
/home/benson/.venvs/redshifty/bin/python nersc/train_transformer.py --manifest /raid/benson/data/desi_dr1_medium/manifest.jsonl --tokenizer-ckpt /raid/benson/data/desi_dr1_medium/checkpoints/tokenizer_v1_large/best.pt --tokenizer-kind v1 --approach a --steps 10000 --batch-size 32 --lr 4e-4 --warmup 500 --healpix-holdout-frac 0.05 --val-every 500 --save-every 2500 --log-every 100 --num-workers 4 --amp --redshift-loss-weight 50.0 --encoder-mask-ratio 0.50 --ar-eval-batches 8 --run-name approach_a_phase10_xlarge --scratch-out /raid/benson/data/desi_dr1_medium/checkpoints --wandb-mode disabled --no-push-wandb-artifact
```

## Trajectory

| approach | epoch | train_loss | train_acc | train_redshift_acc | val_loss | val_acc | val_redshift_acc |
|---|---|---|---|---|---|---|---|
|  |  | 406.3311 |  |  |  |  |  |
|  |  | 275.0110 |  |  |  |  |  |
|  |  | 249.5240 |  |  |  |  |  |
|  |  | 249.3813 |  |  |  |  |  |
|  |  | 254.3291 |  |  |  |  |  |
|  |  | 246.3288 |  |  |  |  |  |
|  |  |  |  |  | 250.8532 |  | 0.0164 |
|  |  | 237.0973 |  |  |  |  |  |
|  |  | 253.2334 |  |  |  |  |  |
|  |  | 230.7695 |  |  |  |  |  |
|  |  | 237.1033 |  |  |  |  |  |
|  |  | 241.4588 |  |  |  |  |  |
|  |  |  |  |  | 245.3104 |  | 0.0163 |
|  |  | 240.3466 |  |  |  |  |  |
|  |  | 236.3184 |  |  |  |  |  |
|  |  | 233.5105 |  |  |  |  |  |
|  |  | 254.9757 |  |  |  |  |  |
|  |  | 257.2112 |  |  |  |  |  |
|  |  |  |  |  | 235.7724 |  | 0.0170 |
|  |  | 251.5319 |  |  |  |  |  |
|  |  | 227.4759 |  |  |  |  |  |
|  |  | 224.8875 |  |  |  |  |  |
|  |  | 245.3814 |  |  |  |  |  |
|  |  | 237.5078 |  |  |  |  |  |
|  |  |  |  |  | 237.7129 |  | 0.0227 |
|  |  | 248.1000 |  |  |  |  |  |
|  |  | 218.4108 |  |  |  |  |  |
|  |  | 228.6216 |  |  |  |  |  |
|  |  | 221.0413 |  |  |  |  |  |
|  |  | 237.1484 |  |  |  |  |  |
|  |  |  |  |  | 231.7156 |  | 0.0339 |
|  |  | 225.8908 |  |  |  |  |  |
|  |  | 214.4593 |  |  |  |  |  |
|  |  | 235.4943 |  |  |  |  |  |
|  |  | 223.3303 |  |  |  |  |  |
|  |  | 240.1622 |  |  |  |  |  |
|  |  |  |  |  | 226.1350 |  | 0.0346 |
|  |  | 211.1089 |  |  |  |  |  |
|  |  | 226.3052 |  |  |  |  |  |
|  |  | 248.8919 |  |  |  |  |  |
|  |  | 206.1540 |  |  |  |  |  |
|  |  | 240.8747 |  |  |  |  |  |
|  |  |  |  |  | 222.9447 |  | 0.0349 |
|  |  | 227.1586 |  |  |  |  |  |
|  |  | 221.4743 |  |  |  |  |  |
|  |  | 222.9349 |  |  |  |  |  |
|  |  | 226.7993 |  |  |  |  |  |
|  |  | 214.8574 |  |  |  |  |  |
|  |  |  |  |  | 219.9667 |  | 0.0300 |
|  |  | 202.3362 |  |  |  |  |  |
|  |  | 204.4443 |  |  |  |  |  |
|  |  | 228.3107 |  |  |  |  |  |
|  |  | 196.5008 |  |  |  |  |  |
|  |  | 219.8951 |  |  |  |  |  |
|  |  |  |  |  | 218.2479 |  | 0.0323 |
|  |  | 206.8985 |  |  |  |  |  |
|  |  | 215.1167 |  |  |  |  |  |
|  |  | 219.3429 |  |  |  |  |  |
|  |  | 214.0219 |  |  |  |  |  |
|  |  | 219.0863 |  |  |  |  |  |
|  |  |  |  |  | 214.4091 |  | 0.0492 |
|  |  | 213.7138 |  |  |  |  |  |
|  |  | 204.8956 |  |  |  |  |  |
|  |  | 207.2859 |  |  |  |  |  |
|  |  | 217.5671 |  |  |  |  |  |
|  |  | 226.0080 |  |  |  |  |  |
|  |  |  |  |  | 213.0241 |  | 0.0384 |
|  |  | 198.5270 |  |  |  |  |  |
|  |  | 216.7324 |  |  |  |  |  |
|  |  | 214.2199 |  |  |  |  |  |
|  |  | 197.1515 |  |  |  |  |  |
|  |  | 204.8974 |  |  |  |  |  |
|  |  |  |  |  | 210.6671 |  | 0.0467 |
|  |  | 239.6234 |  |  |  |  |  |
|  |  | 198.4538 |  |  |  |  |  |
|  |  | 215.8504 |  |  |  |  |  |
|  |  | 206.7373 |  |  |  |  |  |
|  |  | 203.1835 |  |  |  |  |  |
|  |  |  |  |  | 209.9188 |  | 0.0455 |
|  |  | 211.7383 |  |  |  |  |  |
|  |  | 232.1571 |  |  |  |  |  |
|  |  | 200.3219 |  |  |  |  |  |
|  |  | 229.4768 |  |  |  |  |  |
|  |  | 212.9244 |  |  |  |  |  |
|  |  |  |  |  | 209.9994 |  | 0.0462 |
|  |  | 225.3715 |  |  |  |  |  |
|  |  | 227.4416 |  |  |  |  |  |
|  |  | 235.4655 |  |  |  |  |  |
|  |  | 215.6451 |  |  |  |  |  |
|  |  | 199.9538 |  |  |  |  |  |
|  |  |  |  |  | 208.0504 |  | 0.0519 |
|  |  | 204.4279 |  |  |  |  |  |
|  |  | 233.0319 |  |  |  |  |  |
|  |  | 228.9861 |  |  |  |  |  |
|  |  | 194.4361 |  |  |  |  |  |
|  |  | 207.9659 |  |  |  |  |  |
|  |  |  |  |  | 206.0282 |  | 0.0537 |
|  |  | 207.0172 |  |  |  |  |  |
|  |  | 207.6921 |  |  |  |  |  |
|  |  | 210.0362 |  |  |  |  |  |
|  |  | 202.5488 |  |  |  |  |  |
|  |  | 203.4375 |  |  |  |  |  |
|  |  |  |  |  | 206.1303 |  | 0.0542 |
|  |  | 222.8615 |  |  |  |  |  |
|  |  | 200.6667 |  |  |  |  |  |
|  |  | 204.9772 |  |  |  |  |  |
|  |  | 196.1125 |  |  |  |  |  |
|  |  | 195.2689 |  |  |  |  |  |
|  |  |  |  |  | 203.2701 |  | 0.0527 |
|  |  | 205.5834 |  |  |  |  |  |
|  |  | 203.3507 |  |  |  |  |  |
|  |  | 228.5571 |  |  |  |  |  |
|  |  | 171.6465 |  |  |  |  |  |
|  |  | 218.6660 |  |  |  |  |  |
|  |  |  |  |  | 203.8297 |  | 0.0618 |
|  |  | 177.2102 |  |  |  |  |  |
|  |  | 201.4367 |  |  |  |  |  |
|  |  | 187.7786 |  |  |  |  |  |
|  |  | 183.0413 |  |  |  |  |  |

## Summary stats

```yaml
n_records: 119
step_max: 9900
step_n: 119
train_all_r2_final: 0.703
train_all_r2_first: 0.032
train_all_r2_max: 0.766
train_all_r2_min: 0.032
train_loss_final: 183.041
train_loss_first: 406.331
train_loss_max: 406.331
train_loss_min: 171.647
train_mask_r2_final: 0.697
train_mask_r2_first: 0.033
train_mask_r2_max: 0.767
train_mask_r2_min: 0.033
train_masked_acc_final: 0.354
train_masked_acc_first: 0.007
train_masked_acc_max: 0.488
train_masked_acc_min: 0.007
train_rz_masked_acc_final: 0.167
train_rz_masked_acc_first: 0
train_rz_masked_acc_max: 0.2
train_rz_masked_acc_min: 0
train_spec_acc_final: 0.358
train_spec_acc_first: 0.006
train_spec_acc_max: 0.483
train_spec_acc_min: 0.006
train_spec_loss_final: 2.266
train_spec_loss_first: 7.386
train_spec_loss_max: 7.386
train_spec_loss_min: 1.788
train_z_acc_final: 0.048
train_z_acc_first: 0
train_z_acc_max: 0.174
train_z_acc_min: 0
train_z_loss_final: 3.542
train_z_loss_first: 7.867
train_z_loss_max: 7.867
train_z_loss_min: 3.323
val_loss_final: 203.83
val_loss_first: 250.853
val_loss_max: 250.853
val_loss_min: 203.27
val_loss_redshift_final: 3.9567
val_loss_redshift_first: 4.8682
val_loss_redshift_max: 4.8682
val_loss_redshift_min: 3.9459
val_loss_spectrum_final: 1.9833
val_loss_spectrum_first: 2.7452
val_loss_spectrum_max: 2.7452
val_loss_spectrum_min: 1.9833
val_loss_total_final: 1.9905
val_loss_total_first: 2.7529
val_loss_total_max: 2.7529
val_loss_total_min: 1.9905
val_masked_spec_acc_final: 0.4583
val_masked_spec_acc_first: 0.3486
val_masked_spec_acc_max: 0.4583
val_masked_spec_acc_min: 0.3486
val_overall_acc_final: 0.4525
val_overall_acc_first: 0.3404
val_overall_acc_max: 0.4525
val_overall_acc_min: 0.3404
val_redshift_acc_final: 0.0618
val_redshift_acc_first: 0.0164
val_redshift_acc_masked_final: 0.0322
val_redshift_acc_masked_first: 0.017
val_redshift_acc_masked_max: 0.0373
val_redshift_acc_masked_min: 0.0169
val_redshift_acc_max: 0.0618
val_redshift_acc_min: 0.0163
val_spectrum_acc_final: 0.454
val_spectrum_acc_first: 0.3416
val_spectrum_acc_max: 0.454
val_spectrum_acc_min: 0.3416
```

