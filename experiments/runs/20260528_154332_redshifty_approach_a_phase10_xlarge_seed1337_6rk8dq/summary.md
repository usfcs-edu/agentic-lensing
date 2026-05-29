# redshifty_approach_a_phase10_xlarge_seed1337

_Approach A xlarge with seed=1337 — seed sweep arm #2 (arm #1 = the existing
xlarge run @ seed=42). Tests whether redshift-ignition is initialization-
sensitive. Same data (729k spectra) and hparams as _phase10_xlarge.
_

- **run dir:** `/raid/benson/git/agentic-lensing/experiments/runs/20260528_154332_redshifty_approach_a_phase10_xlarge_seed1337_6rk8dq`
- **repo:** redshifty
- **started:** 2026-05-28 15:43:32 UTC
- **finished:** 2026-05-28 22:18:32 UTC
- **wallclock:** 23699.3s
- **exit code:** 0
- **tags:** xlarge-scale, transformer, approach-a, track3, seed-sweep

## Command
```
/home/benson/.venvs/redshifty/bin/python nersc/train_transformer.py --manifest /raid/benson/data/desi_dr1_medium/manifest.jsonl --tokenizer-ckpt /raid/benson/data/desi_dr1_medium/checkpoints/tokenizer_v1_large/best.pt --tokenizer-kind v1 --approach a --steps 10000 --batch-size 32 --lr 4e-4 --warmup 500 --healpix-holdout-frac 0.05 --val-every 500 --save-every 2500 --log-every 100 --num-workers 4 --amp --redshift-loss-weight 50.0 --encoder-mask-ratio 0.50 --ar-eval-batches 8 --seed 1337 --run-name approach_a_phase10_xlarge_seed1337 --scratch-out /raid/benson/data/desi_dr1_medium/checkpoints --wandb-mode disabled --no-push-wandb-artifact
```

## Trajectory

| approach | epoch | train_loss | train_acc | train_redshift_acc | val_loss | val_acc | val_redshift_acc |
|---|---|---|---|---|---|---|---|
|  |  | 402.3175 |  |  |  |  |  |
|  |  | 264.2588 |  |  |  |  |  |
|  |  | 242.5795 |  |  |  |  |  |
|  |  | 264.1084 |  |  |  |  |  |
|  |  | 233.5312 |  |  |  |  |  |
|  |  | 245.8643 |  |  |  |  |  |
|  |  |  |  |  | 242.7735 |  | 0.0308 |
|  |  | 228.7700 |  |  |  |  |  |
|  |  | 247.4131 |  |  |  |  |  |
|  |  | 223.0058 |  |  |  |  |  |
|  |  | 222.9444 |  |  |  |  |  |
|  |  | 233.8103 |  |  |  |  |  |
|  |  |  |  |  | 231.4415 |  | 0.0211 |
|  |  | 239.9087 |  |  |  |  |  |
|  |  | 229.0044 |  |  |  |  |  |
|  |  | 228.9561 |  |  |  |  |  |
|  |  | 229.3567 |  |  |  |  |  |
|  |  | 215.8358 |  |  |  |  |  |
|  |  |  |  |  | 230.3437 |  | 0.0181 |
|  |  | 248.7902 |  |  |  |  |  |
|  |  | 219.2318 |  |  |  |  |  |
|  |  | 242.4427 |  |  |  |  |  |
|  |  | 232.2686 |  |  |  |  |  |
|  |  | 234.7676 |  |  |  |  |  |
|  |  |  |  |  | 224.4937 |  | 0.0398 |
|  |  | 217.2574 |  |  |  |  |  |
|  |  | 214.0184 |  |  |  |  |  |
|  |  | 226.9152 |  |  |  |  |  |
|  |  | 230.1560 |  |  |  |  |  |
|  |  | 221.6841 |  |  |  |  |  |
|  |  |  |  |  | 216.9449 |  | 0.0876 |
|  |  | 211.0126 |  |  |  |  |  |
|  |  | 250.0843 |  |  |  |  |  |
|  |  | 233.6358 |  |  |  |  |  |
|  |  | 212.0930 |  |  |  |  |  |
|  |  | 239.3260 |  |  |  |  |  |
|  |  |  |  |  | 218.8726 |  | 0.0133 |
|  |  | 225.0365 |  |  |  |  |  |
|  |  | 204.5870 |  |  |  |  |  |
|  |  | 219.5647 |  |  |  |  |  |
|  |  | 225.8802 |  |  |  |  |  |
|  |  | 240.3468 |  |  |  |  |  |
|  |  |  |  |  | 224.6163 |  | 0.0175 |
|  |  | 220.4090 |  |  |  |  |  |
|  |  | 222.2844 |  |  |  |  |  |
|  |  | 217.7210 |  |  |  |  |  |
|  |  | 217.5046 |  |  |  |  |  |
|  |  | 227.2293 |  |  |  |  |  |
|  |  |  |  |  | 214.7856 |  | 0.0563 |
|  |  | 219.9332 |  |  |  |  |  |
|  |  | 231.8711 |  |  |  |  |  |
|  |  | 213.1869 |  |  |  |  |  |
|  |  | 205.5500 |  |  |  |  |  |
|  |  | 222.3338 |  |  |  |  |  |
|  |  |  |  |  | 225.3549 |  | 0.0333 |
|  |  | 221.4764 |  |  |  |  |  |
|  |  | 203.6221 |  |  |  |  |  |
|  |  | 215.9173 |  |  |  |  |  |
|  |  | 236.5785 |  |  |  |  |  |
|  |  | 221.3673 |  |  |  |  |  |
|  |  |  |  |  | 227.3311 |  | 0.0197 |
|  |  | 209.9910 |  |  |  |  |  |
|  |  | 234.1653 |  |  |  |  |  |
|  |  | 224.4392 |  |  |  |  |  |
|  |  | 219.2506 |  |  |  |  |  |
|  |  | 218.1800 |  |  |  |  |  |
|  |  |  |  |  | 223.2577 |  | 0.0290 |
|  |  | 226.5048 |  |  |  |  |  |
|  |  | 235.3232 |  |  |  |  |  |
|  |  | 233.4149 |  |  |  |  |  |
|  |  | 243.3839 |  |  |  |  |  |
|  |  | 219.3857 |  |  |  |  |  |
|  |  |  |  |  | 215.4756 |  | 0.0322 |
|  |  | 211.8878 |  |  |  |  |  |
|  |  | 214.2844 |  |  |  |  |  |
|  |  | 218.2515 |  |  |  |  |  |
|  |  | 205.9297 |  |  |  |  |  |
|  |  | 205.4926 |  |  |  |  |  |
|  |  |  |  |  | 213.7717 |  | 0.0607 |
|  |  | 212.0950 |  |  |  |  |  |
|  |  | 198.9301 |  |  |  |  |  |
|  |  | 254.1834 |  |  |  |  |  |
|  |  | 216.4661 |  |  |  |  |  |
|  |  | 211.3334 |  |  |  |  |  |
|  |  |  |  |  | 213.9827 |  | 0.0390 |
|  |  | 230.4028 |  |  |  |  |  |
|  |  | 214.7401 |  |  |  |  |  |
|  |  | 214.0409 |  |  |  |  |  |
|  |  | 209.6030 |  |  |  |  |  |
|  |  | 200.1330 |  |  |  |  |  |
|  |  |  |  |  | 214.1412 |  | 0.0343 |
|  |  | 207.4151 |  |  |  |  |  |
|  |  | 215.6108 |  |  |  |  |  |
|  |  | 200.3104 |  |  |  |  |  |
|  |  | 231.1977 |  |  |  |  |  |
|  |  | 205.3009 |  |  |  |  |  |
|  |  |  |  |  | 214.7732 |  | 0.0352 |
|  |  | 218.1030 |  |  |  |  |  |
|  |  | 236.3407 |  |  |  |  |  |
|  |  | 210.1401 |  |  |  |  |  |
|  |  | 195.7598 |  |  |  |  |  |
|  |  | 223.8802 |  |  |  |  |  |
|  |  |  |  |  | 217.1903 |  | 0.0453 |
|  |  | 214.8495 |  |  |  |  |  |
|  |  | 203.0653 |  |  |  |  |  |
|  |  | 215.7075 |  |  |  |  |  |
|  |  | 201.9583 |  |  |  |  |  |
|  |  | 222.6385 |  |  |  |  |  |
|  |  |  |  |  | 209.5275 |  | 0.0542 |
|  |  | 230.3643 |  |  |  |  |  |
|  |  | 207.6827 |  |  |  |  |  |
|  |  | 221.6288 |  |  |  |  |  |
|  |  | 231.4348 |  |  |  |  |  |
|  |  | 230.8016 |  |  |  |  |  |
|  |  |  |  |  | 211.8311 |  | 0.0427 |
|  |  | 207.9164 |  |  |  |  |  |
|  |  | 209.4997 |  |  |  |  |  |
|  |  | 216.0710 |  |  |  |  |  |
|  |  | 209.9792 |  |  |  |  |  |

## Summary stats

```yaml
n_records: 119
step_max: 9900
step_n: 119
train_all_r2_final: 0.706
train_all_r2_first: 0
train_all_r2_max: 0.757
train_all_r2_min: 0
train_loss_final: 209.979
train_loss_first: 402.317
train_loss_max: 402.317
train_loss_min: 195.76
train_mask_r2_final: 0.708
train_mask_r2_first: 0
train_mask_r2_max: 0.758
train_mask_r2_min: 0
train_masked_acc_final: 0.371
train_masked_acc_first: 0
train_masked_acc_max: 0.487
train_masked_acc_min: 0
train_rz_masked_acc_final: 0
train_rz_masked_acc_first: 0
train_rz_masked_acc_max: 0.2
train_rz_masked_acc_min: 0
train_spec_acc_final: 0.371
train_spec_acc_first: 0
train_spec_acc_max: 0.477
train_spec_acc_min: 0
train_spec_loss_final: 2.243
train_spec_loss_first: 7.756
train_spec_loss_max: 7.756
train_spec_loss_min: 1.856
train_z_acc_final: 0.042
train_z_acc_first: 0
train_z_acc_max: 0.16
train_z_acc_min: 0
train_z_loss_final: 4.073
train_z_loss_first: 7.779
train_z_loss_max: 7.779
train_z_loss_min: 3.799
val_loss_final: 211.831
val_loss_first: 242.774
val_loss_max: 242.774
val_loss_min: 209.528
val_loss_redshift_final: 4.1135
val_loss_redshift_first: 4.7101
val_loss_redshift_max: 4.7101
val_loss_redshift_min: 4.0684
val_loss_spectrum_final: 2.012
val_loss_spectrum_first: 2.7207
val_loss_spectrum_max: 2.7207
val_loss_spectrum_min: 2.0115
val_loss_total_final: 2.0197
val_loss_total_first: 2.728
val_loss_total_max: 2.728
val_loss_total_min: 2.019
val_masked_spec_acc_final: 0.4405
val_masked_spec_acc_first: 0.3335
val_masked_spec_acc_max: 0.4405
val_masked_spec_acc_min: 0.3335
val_overall_acc_final: 0.4406
val_overall_acc_first: 0.327
val_overall_acc_max: 0.4406
val_overall_acc_min: 0.327
val_redshift_acc_final: 0.0427
val_redshift_acc_first: 0.0308
val_redshift_acc_masked_final: 0.0369
val_redshift_acc_masked_first: 0.0252
val_redshift_acc_masked_max: 0.0806
val_redshift_acc_masked_min: 0.009
val_redshift_acc_max: 0.0876
val_redshift_acc_min: 0.0133
val_spectrum_acc_final: 0.4421
val_spectrum_acc_first: 0.328
val_spectrum_acc_max: 0.4421
val_spectrum_acc_min: 0.328
```

