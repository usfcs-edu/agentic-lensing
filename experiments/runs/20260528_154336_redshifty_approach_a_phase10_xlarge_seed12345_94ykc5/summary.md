# redshifty_approach_a_phase10_xlarge_seed12345

_Approach A xlarge with seed=12345 — seed sweep arm #4. Same data and hparams as
_phase10_xlarge; tests initialization sensitivity. Runs on A16 (cuda:0); will
be slower wallclock than the L4 arms.
_

- **run dir:** `/raid/benson/git/agentic-lensing/experiments/runs/20260528_154336_redshifty_approach_a_phase10_xlarge_seed12345_94ykc5`
- **repo:** redshifty
- **started:** 2026-05-28 15:43:36 UTC
- **finished:** 2026-05-28 22:02:02 UTC
- **wallclock:** 22706.6s
- **exit code:** 0
- **tags:** xlarge-scale, transformer, approach-a, track3, seed-sweep

## Command
```
/home/benson/.venvs/redshifty/bin/python nersc/train_transformer.py --manifest /raid/benson/data/desi_dr1_medium/manifest.jsonl --tokenizer-ckpt /raid/benson/data/desi_dr1_medium/checkpoints/tokenizer_v1_large/best.pt --tokenizer-kind v1 --approach a --steps 10000 --batch-size 32 --lr 4e-4 --warmup 500 --healpix-holdout-frac 0.05 --val-every 500 --save-every 2500 --log-every 100 --num-workers 4 --amp --redshift-loss-weight 50.0 --encoder-mask-ratio 0.50 --ar-eval-batches 8 --seed 12345 --run-name approach_a_phase10_xlarge_seed12345 --scratch-out /raid/benson/data/desi_dr1_medium/checkpoints --wandb-mode disabled --no-push-wandb-artifact
```

## Trajectory

| approach | epoch | train_loss | train_acc | train_redshift_acc | val_loss | val_acc | val_redshift_acc |
|---|---|---|---|---|---|---|---|
|  |  | 409.8332 |  |  |  |  |  |
|  |  | 279.6678 |  |  |  |  |  |
|  |  | 264.3346 |  |  |  |  |  |
|  |  | 246.2655 |  |  |  |  |  |
|  |  | 233.2407 |  |  |  |  |  |
|  |  | 237.3748 |  |  |  |  |  |
|  |  |  |  |  | 248.9584 |  | 0.0263 |
|  |  | 236.3937 |  |  |  |  |  |
|  |  | 233.6977 |  |  |  |  |  |
|  |  | 239.3557 |  |  |  |  |  |
|  |  | 223.0202 |  |  |  |  |  |
|  |  | 228.3349 |  |  |  |  |  |
|  |  |  |  |  | 233.5138 |  | 0.0174 |
|  |  | 204.5467 |  |  |  |  |  |
|  |  | 245.3837 |  |  |  |  |  |
|  |  | 252.9568 |  |  |  |  |  |
|  |  | 223.3019 |  |  |  |  |  |
|  |  | 228.8979 |  |  |  |  |  |
|  |  |  |  |  | 231.6908 |  | 0.0283 |
|  |  | 224.0359 |  |  |  |  |  |
|  |  | 235.1757 |  |  |  |  |  |
|  |  | 215.0984 |  |  |  |  |  |
|  |  | 222.2887 |  |  |  |  |  |
|  |  | 224.3872 |  |  |  |  |  |
|  |  |  |  |  | 230.2040 |  | 0.0229 |
|  |  | 216.2038 |  |  |  |  |  |
|  |  | 229.1604 |  |  |  |  |  |
|  |  | 230.7928 |  |  |  |  |  |
|  |  | 239.7004 |  |  |  |  |  |
|  |  | 221.9670 |  |  |  |  |  |
|  |  |  |  |  | 228.1052 |  | 0.0365 |
|  |  | 219.5125 |  |  |  |  |  |
|  |  | 226.0404 |  |  |  |  |  |
|  |  | 230.9301 |  |  |  |  |  |
|  |  | 238.5100 |  |  |  |  |  |
|  |  | 235.2851 |  |  |  |  |  |
|  |  |  |  |  | 227.3187 |  | 0.0159 |
|  |  | 223.3714 |  |  |  |  |  |
|  |  | 238.2141 |  |  |  |  |  |
|  |  | 251.0626 |  |  |  |  |  |
|  |  | 232.0142 |  |  |  |  |  |
|  |  | 260.8510 |  |  |  |  |  |
|  |  |  |  |  | 227.2352 |  | 0.0261 |
|  |  | 223.8362 |  |  |  |  |  |
|  |  | 230.0500 |  |  |  |  |  |
|  |  | 241.4464 |  |  |  |  |  |
|  |  | 221.5625 |  |  |  |  |  |
|  |  | 237.5938 |  |  |  |  |  |
|  |  |  |  |  | 224.2052 |  | 0.0182 |
|  |  | 228.5570 |  |  |  |  |  |
|  |  | 241.8282 |  |  |  |  |  |
|  |  | 215.9328 |  |  |  |  |  |
|  |  | 217.2668 |  |  |  |  |  |
|  |  | 212.9364 |  |  |  |  |  |
|  |  |  |  |  | 222.3110 |  | 0.0276 |
|  |  | 236.5439 |  |  |  |  |  |
|  |  | 207.9843 |  |  |  |  |  |
|  |  | 225.8832 |  |  |  |  |  |
|  |  | 218.1627 |  |  |  |  |  |
|  |  | 210.5169 |  |  |  |  |  |
|  |  |  |  |  | 222.4501 |  | 0.0236 |
|  |  | 208.1114 |  |  |  |  |  |
|  |  | 210.9615 |  |  |  |  |  |
|  |  | 223.0663 |  |  |  |  |  |
|  |  | 189.4618 |  |  |  |  |  |
|  |  | 206.0971 |  |  |  |  |  |
|  |  |  |  |  | 220.2290 |  | 0.0274 |
|  |  | 201.6908 |  |  |  |  |  |
|  |  | 212.8680 |  |  |  |  |  |
|  |  | 209.6026 |  |  |  |  |  |
|  |  | 213.4803 |  |  |  |  |  |
|  |  | 234.2817 |  |  |  |  |  |
|  |  |  |  |  | 222.4764 |  | 0.0321 |
|  |  | 217.8998 |  |  |  |  |  |
|  |  | 210.6123 |  |  |  |  |  |
|  |  | 249.7453 |  |  |  |  |  |
|  |  | 205.0058 |  |  |  |  |  |
|  |  | 205.5509 |  |  |  |  |  |
|  |  |  |  |  | 215.8484 |  | 0.0243 |
|  |  | 220.6110 |  |  |  |  |  |
|  |  | 192.0936 |  |  |  |  |  |
|  |  | 211.2951 |  |  |  |  |  |
|  |  | 213.2142 |  |  |  |  |  |
|  |  | 192.9028 |  |  |  |  |  |
|  |  |  |  |  | 214.3413 |  | 0.0242 |
|  |  | 191.2782 |  |  |  |  |  |
|  |  | 192.7069 |  |  |  |  |  |
|  |  | 215.0592 |  |  |  |  |  |
|  |  | 197.1276 |  |  |  |  |  |
|  |  | 222.0482 |  |  |  |  |  |
|  |  |  |  |  | 211.6010 |  | 0.0450 |
|  |  | 211.4892 |  |  |  |  |  |
|  |  | 207.9143 |  |  |  |  |  |
|  |  | 217.4458 |  |  |  |  |  |
|  |  | 204.5167 |  |  |  |  |  |
|  |  | 212.4855 |  |  |  |  |  |
|  |  |  |  |  | 209.2530 |  | 0.0501 |
|  |  | 199.9017 |  |  |  |  |  |
|  |  | 194.2187 |  |  |  |  |  |
|  |  | 200.5374 |  |  |  |  |  |
|  |  | 215.5564 |  |  |  |  |  |
|  |  | 202.8879 |  |  |  |  |  |
|  |  |  |  |  | 209.2474 |  | 0.0369 |
|  |  | 182.8575 |  |  |  |  |  |
|  |  | 196.4005 |  |  |  |  |  |
|  |  | 198.0925 |  |  |  |  |  |
|  |  | 190.0674 |  |  |  |  |  |
|  |  | 212.4425 |  |  |  |  |  |
|  |  |  |  |  | 207.6117 |  | 0.0474 |
|  |  | 219.1018 |  |  |  |  |  |
|  |  | 211.8257 |  |  |  |  |  |
|  |  | 210.0910 |  |  |  |  |  |
|  |  | 205.1667 |  |  |  |  |  |
|  |  | 206.7026 |  |  |  |  |  |
|  |  |  |  |  | 204.0125 |  | 0.0684 |
|  |  | 215.5104 |  |  |  |  |  |
|  |  | 208.9226 |  |  |  |  |  |
|  |  | 208.5981 |  |  |  |  |  |
|  |  | 210.4320 |  |  |  |  |  |

## Summary stats

```yaml
n_records: 119
step_max: 9900
step_n: 119
train_all_r2_final: 0.716
train_all_r2_first: 0
train_all_r2_max: 0.747
train_all_r2_min: 0
train_loss_final: 210.432
train_loss_first: 409.833
train_loss_max: 409.833
train_loss_min: 182.857
train_mask_r2_final: 0.712
train_mask_r2_first: 0
train_mask_r2_max: 0.744
train_mask_r2_min: 0
train_masked_acc_final: 0.391
train_masked_acc_first: 0
train_masked_acc_max: 0.472
train_masked_acc_min: 0
train_rz_masked_acc_final: 0
train_rz_masked_acc_first: 0
train_rz_masked_acc_max: 0.333
train_rz_masked_acc_min: 0
train_spec_acc_final: 0.391
train_spec_acc_first: 0
train_spec_acc_max: 0.473
train_spec_acc_min: 0
train_spec_loss_final: 2.167
train_spec_loss_first: 7.843
train_spec_loss_max: 7.843
train_spec_loss_min: 1.932
train_z_acc_final: 0.095
train_z_acc_first: 0
train_z_acc_max: 0.188
train_z_acc_min: 0
train_z_loss_final: 4.083
train_z_loss_first: 7.927
train_z_loss_max: 7.927
train_z_loss_min: 3.544
val_loss_final: 204.012
val_loss_first: 248.958
val_loss_max: 248.958
val_loss_min: 204.012
val_loss_redshift_final: 3.9606
val_loss_redshift_first: 4.8296
val_loss_redshift_max: 4.8296
val_loss_redshift_min: 3.9606
val_loss_spectrum_final: 1.986
val_loss_spectrum_first: 2.7756
val_loss_spectrum_max: 2.7756
val_loss_spectrum_min: 1.986
val_loss_total_final: 1.9932
val_loss_total_first: 2.7831
val_loss_total_max: 2.7831
val_loss_total_min: 1.9932
val_masked_spec_acc_final: 0.443
val_masked_spec_acc_first: 0.3179
val_masked_spec_acc_max: 0.4439
val_masked_spec_acc_min: 0.3179
val_overall_acc_final: 0.4437
val_overall_acc_first: 0.3168
val_overall_acc_max: 0.4437
val_overall_acc_min: 0.3168
val_redshift_acc_final: 0.0684
val_redshift_acc_first: 0.0263
val_redshift_acc_masked_final: 0.0367
val_redshift_acc_masked_first: 0.0266
val_redshift_acc_masked_max: 0.0367
val_redshift_acc_masked_min: 0.0117
val_redshift_acc_max: 0.0684
val_redshift_acc_min: 0.0159
val_spectrum_acc_final: 0.4451
val_spectrum_acc_first: 0.3179
val_spectrum_acc_max: 0.4451
val_spectrum_acc_min: 0.3179
```

