# redshifty_approach_a_phase10_xlarge_seed7

_Approach A xlarge with seed=7 — seed sweep arm #3. Same data and hparams as
_phase10_xlarge; tests initialization sensitivity of redshift ignition.
_

- **run dir:** `/raid/benson/git/agentic-lensing/experiments/runs/20260528_154334_redshifty_approach_a_phase10_xlarge_seed7_xhyfaw`
- **repo:** redshifty
- **started:** 2026-05-28 15:43:34 UTC
- **finished:** 2026-05-28 23:21:05 UTC
- **wallclock:** 27450.1s
- **exit code:** 0
- **tags:** xlarge-scale, transformer, approach-a, track3, seed-sweep

## Command
```
/home/benson/.venvs/redshifty/bin/python nersc/train_transformer.py --manifest /raid/benson/data/desi_dr1_medium/manifest.jsonl --tokenizer-ckpt /raid/benson/data/desi_dr1_medium/checkpoints/tokenizer_v1_large/best.pt --tokenizer-kind v1 --approach a --steps 10000 --batch-size 32 --lr 4e-4 --warmup 500 --healpix-holdout-frac 0.05 --val-every 500 --save-every 2500 --log-every 100 --num-workers 4 --amp --redshift-loss-weight 50.0 --encoder-mask-ratio 0.50 --ar-eval-batches 8 --seed 7 --run-name approach_a_phase10_xlarge_seed7 --scratch-out /raid/benson/data/desi_dr1_medium/checkpoints --wandb-mode disabled --no-push-wandb-artifact
```

## Trajectory

| approach | epoch | train_loss | train_acc | train_redshift_acc | val_loss | val_acc | val_redshift_acc |
|---|---|---|---|---|---|---|---|
|  |  | 405.7924 |  |  |  |  |  |
|  |  | 251.9834 |  |  |  |  |  |
|  |  | 238.1506 |  |  |  |  |  |
|  |  | 267.2147 |  |  |  |  |  |
|  |  | 255.0557 |  |  |  |  |  |
|  |  | 236.2357 |  |  |  |  |  |
|  |  |  |  |  | 253.0376 |  | 0.0315 |
|  |  | 264.1200 |  |  |  |  |  |
|  |  | 232.2329 |  |  |  |  |  |
|  |  | 247.3848 |  |  |  |  |  |
|  |  | 251.3333 |  |  |  |  |  |
|  |  | 227.1691 |  |  |  |  |  |
|  |  |  |  |  | 254.1108 |  | 0.0181 |
|  |  | 232.6771 |  |  |  |  |  |
|  |  | 242.3517 |  |  |  |  |  |
|  |  | 232.1837 |  |  |  |  |  |
|  |  | 245.5819 |  |  |  |  |  |
|  |  | 241.0336 |  |  |  |  |  |
|  |  |  |  |  | 250.9469 |  | 0.0100 |
|  |  | 211.6845 |  |  |  |  |  |
|  |  | 230.1715 |  |  |  |  |  |
|  |  | 232.6846 |  |  |  |  |  |
|  |  | 227.7053 |  |  |  |  |  |
|  |  | 238.9148 |  |  |  |  |  |
|  |  |  |  |  | 247.6729 |  | 0.0116 |
|  |  | 227.2153 |  |  |  |  |  |
|  |  | 228.5011 |  |  |  |  |  |
|  |  | 243.5823 |  |  |  |  |  |
|  |  | 229.1084 |  |  |  |  |  |
|  |  | 236.0536 |  |  |  |  |  |
|  |  |  |  |  | 250.2753 |  | 0.0099 |
|  |  | 238.5292 |  |  |  |  |  |
|  |  | 235.6819 |  |  |  |  |  |
|  |  | 239.4016 |  |  |  |  |  |
|  |  | 227.4157 |  |  |  |  |  |
|  |  | 222.2072 |  |  |  |  |  |
|  |  |  |  |  | 245.0802 |  | 0.0114 |
|  |  | 231.3784 |  |  |  |  |  |
|  |  | 228.7403 |  |  |  |  |  |
|  |  | 234.2955 |  |  |  |  |  |
|  |  | 219.7712 |  |  |  |  |  |
|  |  | 218.3570 |  |  |  |  |  |
|  |  |  |  |  | 243.1165 |  | 0.0094 |
|  |  | 229.5521 |  |  |  |  |  |
|  |  | 233.9241 |  |  |  |  |  |
|  |  | 235.9794 |  |  |  |  |  |
|  |  | 241.5091 |  |  |  |  |  |
|  |  | 261.5384 |  |  |  |  |  |
|  |  |  |  |  | 245.3920 |  | 0.0062 |
|  |  | 246.3871 |  |  |  |  |  |
|  |  | 227.5516 |  |  |  |  |  |
|  |  | 215.6646 |  |  |  |  |  |
|  |  | 230.3437 |  |  |  |  |  |
|  |  | 238.9785 |  |  |  |  |  |
|  |  |  |  |  | 240.1884 |  | 0.0137 |
|  |  | 238.6579 |  |  |  |  |  |
|  |  | 252.6277 |  |  |  |  |  |
|  |  | 224.5271 |  |  |  |  |  |
|  |  | 233.3214 |  |  |  |  |  |
|  |  | 214.5763 |  |  |  |  |  |
|  |  |  |  |  | 241.6920 |  | 0.0146 |
|  |  | 227.5109 |  |  |  |  |  |
|  |  | 235.7344 |  |  |  |  |  |
|  |  | 230.0195 |  |  |  |  |  |
|  |  | 214.2109 |  |  |  |  |  |
|  |  | 203.4496 |  |  |  |  |  |
|  |  |  |  |  | 238.3814 |  | 0.0184 |
|  |  | 230.9487 |  |  |  |  |  |
|  |  | 210.8615 |  |  |  |  |  |
|  |  | 219.9207 |  |  |  |  |  |
|  |  | 248.9249 |  |  |  |  |  |
|  |  | 237.0400 |  |  |  |  |  |
|  |  |  |  |  | 232.0088 |  | 0.0242 |
|  |  | 217.6645 |  |  |  |  |  |
|  |  | 199.4386 |  |  |  |  |  |
|  |  | 205.2720 |  |  |  |  |  |
|  |  | 230.4901 |  |  |  |  |  |
|  |  | 226.8821 |  |  |  |  |  |
|  |  |  |  |  | 226.4320 |  | 0.0209 |
|  |  | 209.8103 |  |  |  |  |  |
|  |  | 208.4036 |  |  |  |  |  |
|  |  | 222.8530 |  |  |  |  |  |
|  |  | 212.6232 |  |  |  |  |  |
|  |  | 222.1576 |  |  |  |  |  |
|  |  |  |  |  | 224.1127 |  | 0.0285 |
|  |  | 213.1645 |  |  |  |  |  |
|  |  | 204.8160 |  |  |  |  |  |
|  |  | 185.6519 |  |  |  |  |  |
|  |  | 185.2860 |  |  |  |  |  |
|  |  | 202.1976 |  |  |  |  |  |
|  |  |  |  |  | 224.3476 |  | 0.0184 |
|  |  | 220.3494 |  |  |  |  |  |
|  |  | 201.2154 |  |  |  |  |  |
|  |  | 217.8566 |  |  |  |  |  |
|  |  | 209.4329 |  |  |  |  |  |
|  |  | 191.9839 |  |  |  |  |  |
|  |  |  |  |  | 222.5456 |  | 0.0376 |
|  |  | 194.9245 |  |  |  |  |  |
|  |  | 193.3540 |  |  |  |  |  |
|  |  | 209.8763 |  |  |  |  |  |
|  |  | 215.4976 |  |  |  |  |  |
|  |  | 191.3906 |  |  |  |  |  |
|  |  |  |  |  | 220.3312 |  | 0.0226 |
|  |  | 225.7040 |  |  |  |  |  |
|  |  | 207.3818 |  |  |  |  |  |
|  |  | 209.8249 |  |  |  |  |  |
|  |  | 210.5587 |  |  |  |  |  |
|  |  | 216.6567 |  |  |  |  |  |
|  |  |  |  |  | 219.4620 |  | 0.0169 |
|  |  | 236.0527 |  |  |  |  |  |
|  |  | 216.5502 |  |  |  |  |  |
|  |  | 206.8423 |  |  |  |  |  |
|  |  | 207.7430 |  |  |  |  |  |
|  |  | 218.1483 |  |  |  |  |  |
|  |  |  |  |  | 217.8431 |  | 0.0350 |
|  |  | 203.6515 |  |  |  |  |  |
|  |  | 213.1785 |  |  |  |  |  |
|  |  | 203.4817 |  |  |  |  |  |
|  |  | 210.0523 |  |  |  |  |  |

## Summary stats

```yaml
n_records: 119
step_max: 9900
step_n: 119
train_all_r2_final: 0.746
train_all_r2_first: 0
train_all_r2_max: 0.754
train_all_r2_min: 0
train_loss_final: 210.052
train_loss_first: 405.792
train_loss_max: 405.792
train_loss_min: 185.286
train_mask_r2_final: 0.745
train_mask_r2_first: 0
train_mask_r2_max: 0.753
train_mask_r2_min: 0
train_masked_acc_final: 0.467
train_masked_acc_first: 0
train_masked_acc_max: 0.477
train_masked_acc_min: 0
train_rz_masked_acc_final: 0.111
train_rz_masked_acc_first: 0
train_rz_masked_acc_max: 0.222
train_rz_masked_acc_min: 0
train_spec_acc_final: 0.47
train_spec_acc_first: 0
train_spec_acc_max: 0.479
train_spec_acc_min: 0
train_spec_loss_final: 1.935
train_spec_loss_first: 7.645
train_spec_loss_max: 7.645
train_spec_loss_min: 1.878
train_z_acc_final: 0.143
train_z_acc_first: 0
train_z_acc_max: 0.19
train_z_acc_min: 0
train_z_loss_final: 4.081
train_z_loss_first: 7.849
train_z_loss_max: 7.849
train_z_loss_min: 3.587
val_loss_final: 217.843
val_loss_first: 253.038
val_loss_max: 254.111
val_loss_min: 217.843
val_loss_redshift_final: 4.2334
val_loss_redshift_first: 4.9093
val_loss_redshift_max: 4.936
val_loss_redshift_min: 4.2334
val_loss_spectrum_final: 1.8527
val_loss_spectrum_first: 2.8019
val_loss_spectrum_max: 2.8019
val_loss_spectrum_min: 1.8527
val_loss_total_final: 1.8614
val_loss_total_first: 2.8096
val_loss_total_max: 2.8096
val_loss_total_min: 1.8614
val_masked_spec_acc_final: 0.4604
val_masked_spec_acc_first: 0.2927
val_masked_spec_acc_max: 0.4604
val_masked_spec_acc_min: 0.2927
val_overall_acc_final: 0.4597
val_overall_acc_first: 0.2917
val_overall_acc_max: 0.4597
val_overall_acc_min: 0.2917
val_redshift_acc_final: 0.035
val_redshift_acc_first: 0.0315
val_redshift_acc_masked_final: 0.0138
val_redshift_acc_masked_first: 0.0274
val_redshift_acc_masked_max: 0.0274
val_redshift_acc_masked_min: 0.0046
val_redshift_acc_max: 0.0376
val_redshift_acc_min: 0.0062
val_spectrum_acc_final: 0.4612
val_spectrum_acc_first: 0.2927
val_spectrum_acc_max: 0.4612
val_spectrum_acc_min: 0.2927
```

