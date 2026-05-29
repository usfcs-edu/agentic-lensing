# redshifty_approach_a_medium

_Approach A transformer on local medium subset with V1 tokenizer (depends on redshifty_tokenizer_v1_medium completing)_

- **run dir:** `/raid/benson/git/agentic-lensing/experiments/runs/20260526_234553_redshifty_approach_a_medium_ejhjea`
- **repo:** redshifty
- **started:** 2026-05-26 23:45:53 UTC
- **finished:** 2026-05-27 00:01:07 UTC
- **wallclock:** 913.9s
- **exit code:** 0
- **tags:** medium-real, transformer, approach-a, track3

## Command
```
/home/benson/.venvs/redshifty/bin/python nersc/train_transformer.py --manifest /raid/benson/data/desi_dr1_medium/manifest.jsonl --tokenizer-ckpt /raid/benson/data/desi_dr1_medium/checkpoints/tokenizer_v1_medium/best.pt --tokenizer-kind v1 --approach a --steps 2000 --batch-size 8 --lr 2e-4 --warmup 200 --healpix-holdout-frac 0.10 --val-every 200 --save-every 500 --log-every 25 --num-workers 4 --amp --redshift-loss-weight 50.0 --encoder-mask-ratio 0.50 --ar-eval-batches 2 --run-name approach_a_medium --scratch-out /raid/benson/data/desi_dr1_medium/checkpoints --wandb-mode disabled --no-push-wandb-artifact
```

## Trajectory

| approach | epoch | train_loss | train_acc | train_redshift_acc | val_loss | val_acc | val_redshift_acc |
|---|---|---|---|---|---|---|---|
|  |  | 396.0460 |  |  |  |  |  |
|  |  | 375.2480 |  |  |  |  |  |
|  |  | 399.7938 |  |  |  |  |  |
|  |  | 344.5505 |  |  |  |  |  |
|  |  | 301.9713 |  |  |  |  |  |
|  |  | 238.0056 |  |  |  |  |  |
|  |  | 319.4971 |  |  |  |  |  |
|  |  | 270.2380 |  |  |  |  |  |
|  |  | 320.8556 |  |  |  |  |  |
|  |  |  |  |  | 278.3829 |  | 0.0200 |
|  |  | 306.7122 |  |  |  |  |  |
|  |  | 267.4397 |  |  |  |  |  |
|  |  | 270.7953 |  |  |  |  |  |
|  |  | 303.4459 |  |  |  |  |  |
|  |  | 335.9524 |  |  |  |  |  |
|  |  | 240.3089 |  |  |  |  |  |
|  |  | 316.6715 |  |  |  |  |  |
|  |  | 277.4863 |  |  |  |  |  |
|  |  |  |  |  | 262.3202 |  | 0.0033 |
|  |  | 287.2116 |  |  |  |  |  |
|  |  | 286.9443 |  |  |  |  |  |
|  |  | 218.0839 |  |  |  |  |  |
|  |  | 255.8660 |  |  |  |  |  |
|  |  | 249.0489 |  |  |  |  |  |
|  |  | 270.0996 |  |  |  |  |  |
|  |  | 231.3614 |  |  |  |  |  |
|  |  | 250.6282 |  |  |  |  |  |
|  |  |  |  |  | 259.0348 |  | 0.0080 |
|  |  | 249.0505 |  |  |  |  |  |
|  |  | 268.7144 |  |  |  |  |  |
|  |  | 238.3989 |  |  |  |  |  |
|  |  | 246.7787 |  |  |  |  |  |
|  |  | 279.7640 |  |  |  |  |  |
|  |  | 239.5145 |  |  |  |  |  |
|  |  | 235.8866 |  |  |  |  |  |
|  |  | 261.1456 |  |  |  |  |  |
|  |  |  |  |  | 243.7267 |  | 0.0247 |
|  |  | 268.4837 |  |  |  |  |  |
|  |  | 231.2076 |  |  |  |  |  |
|  |  | 244.9177 |  |  |  |  |  |
|  |  | 285.1768 |  |  |  |  |  |
|  |  | 267.1986 |  |  |  |  |  |
|  |  | 320.8773 |  |  |  |  |  |
|  |  | 211.1115 |  |  |  |  |  |
|  |  | 234.2571 |  |  |  |  |  |
|  |  |  |  |  | 248.8230 |  | 0.0150 |
|  |  | 234.2895 |  |  |  |  |  |
|  |  | 306.4820 |  |  |  |  |  |
|  |  | 290.8089 |  |  |  |  |  |
|  |  | 267.6693 |  |  |  |  |  |
|  |  | 237.4092 |  |  |  |  |  |
|  |  | 217.8858 |  |  |  |  |  |
|  |  | 223.5294 |  |  |  |  |  |
|  |  | 235.5223 |  |  |  |  |  |
|  |  |  |  |  | 241.6763 |  | 0.0380 |
|  |  | 228.9389 |  |  |  |  |  |
|  |  | 240.9389 |  |  |  |  |  |
|  |  | 240.2019 |  |  |  |  |  |
|  |  | 248.9255 |  |  |  |  |  |
|  |  | 252.1296 |  |  |  |  |  |
|  |  | 231.4217 |  |  |  |  |  |
|  |  | 224.5527 |  |  |  |  |  |
|  |  | 257.1517 |  |  |  |  |  |
|  |  |  |  |  | 242.2776 |  | 0.0217 |
|  |  | 260.6297 |  |  |  |  |  |
|  |  | 249.9684 |  |  |  |  |  |
|  |  | 234.8646 |  |  |  |  |  |
|  |  | 229.6855 |  |  |  |  |  |
|  |  | 221.3505 |  |  |  |  |  |
|  |  | 211.8367 |  |  |  |  |  |
|  |  | 235.4646 |  |  |  |  |  |
|  |  | 248.0062 |  |  |  |  |  |
|  |  |  |  |  | 238.4111 |  | 0.0330 |
|  |  | 217.3476 |  |  |  |  |  |
|  |  | 271.6669 |  |  |  |  |  |
|  |  | 246.0113 |  |  |  |  |  |
|  |  | 228.7162 |  |  |  |  |  |
|  |  | 245.3940 |  |  |  |  |  |
|  |  | 238.6650 |  |  |  |  |  |
|  |  | 255.1995 |  |  |  |  |  |
|  |  | 240.1372 |  |  |  |  |  |
|  |  |  |  |  | 236.9062 |  | 0.0123 |
|  |  | 229.0744 |  |  |  |  |  |
|  |  | 226.3177 |  |  |  |  |  |
|  |  | 246.9409 |  |  |  |  |  |
|  |  | 236.0177 |  |  |  |  |  |
|  |  | 246.6839 |  |  |  |  |  |
|  |  | 243.3278 |  |  |  |  |  |
|  |  | 268.3068 |  |  |  |  |  |

## Summary stats

```yaml
n_records: 89
step_max: 1975
step_n: 89
train_all_r2_final: 0.61
train_all_r2_first: 0
train_all_r2_max: 0.652
train_all_r2_min: 0
train_loss_final: 268.307
train_loss_first: 396.046
train_loss_max: 399.794
train_loss_min: 211.112
train_mask_r2_final: 0.623
train_mask_r2_first: 0
train_mask_r2_max: 0.658
train_mask_r2_min: 0
train_masked_acc_final: 0.191
train_masked_acc_first: 0
train_masked_acc_max: 0.271
train_masked_acc_min: 0
train_rz_masked_acc_final: 0
train_rz_masked_acc_first: 0
train_rz_masked_acc_max: 0.333
train_rz_masked_acc_min: 0
train_spec_acc_final: 0.179
train_spec_acc_first: 0
train_spec_acc_max: 0.262
train_spec_acc_min: 0
train_spec_loss_final: 2.976
train_spec_loss_first: 7.853
train_spec_loss_max: 7.853
train_spec_loss_min: 2.652
train_z_acc_final: 0
train_z_acc_first: 0
train_z_acc_max: 0.2
train_z_acc_min: 0
train_z_loss_final: 5.206
train_z_loss_first: 7.653
train_z_loss_max: 7.739
train_z_loss_min: 4.077
val_loss_final: 236.906
val_loss_first: 278.383
val_loss_max: 278.383
val_loss_min: 236.906
val_loss_redshift_final: 4.5815
val_loss_redshift_first: 5.3896
val_loss_redshift_max: 5.3896
val_loss_redshift_min: 4.5815
val_loss_spectrum_final: 3.2553
val_loss_spectrum_first: 3.793
val_loss_spectrum_max: 3.793
val_loss_spectrum_min: 3.2553
val_loss_total_final: 3.2601
val_loss_total_first: 3.7989
val_loss_total_max: 3.7989
val_loss_total_min: 3.2601
val_masked_spec_acc_final: 0.1388
val_masked_spec_acc_first: 0.0948
val_masked_spec_acc_max: 0.144
val_masked_spec_acc_min: 0.0948
val_overall_acc_final: 0.1417
val_overall_acc_first: 0.0965
val_overall_acc_max: 0.1431
val_overall_acc_min: 0.0965
val_redshift_acc_final: 0.0123
val_redshift_acc_first: 0.02
val_redshift_acc_masked_final: 0.0092
val_redshift_acc_masked_first: 0.0273
val_redshift_acc_masked_max: 0.0273
val_redshift_acc_masked_min: 0
val_redshift_acc_max: 0.038
val_redshift_acc_min: 0.0033
val_spectrum_acc_final: 0.1422
val_spectrum_acc_first: 0.0968
val_spectrum_acc_max: 0.1435
val_spectrum_acc_min: 0.0968
```

