# redshifty_approach_a_phase10

_Approach A transformer matching the redshifty author's Phase 10 final NERSC config
exactly — batch=32, lr=4e-4 (the combo we mistakenly skipped in _large by using
Phase 9's batch=8 lr=2e-4 together with Phase 10's mask=0.50). 10k steps on the
same 219k-spectrum/V1-large-tokenizer setup. Diagnostic: should fire redshift
ignition by step ~4000-6500 if the hparam-mismatch hypothesis is right.
_

- **run dir:** `/raid/benson/git/agentic-lensing/experiments/runs/20260527_213001_redshifty_approach_a_phase10_0rvxp4`
- **repo:** redshifty
- **started:** 2026-05-27 21:30:01 UTC
- **finished:** 2026-05-28 02:29:32 UTC
- **wallclock:** 17970.6s
- **exit code:** 0
- **tags:** large-scale, transformer, approach-a, track3, diagnostic, phase10-match

## Command
```
/home/benson/.venvs/redshifty/bin/python nersc/train_transformer.py --manifest /raid/benson/data/desi_dr1_medium/manifest.jsonl --tokenizer-ckpt /raid/benson/data/desi_dr1_medium/checkpoints/tokenizer_v1_large/best.pt --tokenizer-kind v1 --approach a --steps 10000 --batch-size 32 --lr 4e-4 --warmup 500 --healpix-holdout-frac 0.05 --val-every 500 --save-every 2500 --log-every 100 --num-workers 4 --amp --redshift-loss-weight 50.0 --encoder-mask-ratio 0.50 --ar-eval-batches 4 --run-name approach_a_phase10 --scratch-out /raid/benson/data/desi_dr1_medium/checkpoints --wandb-mode disabled --no-push-wandb-artifact
```

## Trajectory

| approach | epoch | train_loss | train_acc | train_redshift_acc | val_loss | val_acc | val_redshift_acc |
|---|---|---|---|---|---|---|---|
|  |  | 397.1953 |  |  |  |  |  |
|  |  | 267.3769 |  |  |  |  |  |
|  |  | 255.1395 |  |  |  |  |  |
|  |  | 242.0085 |  |  |  |  |  |
|  |  | 246.3256 |  |  |  |  |  |
|  |  | 226.4912 |  |  |  |  |  |
|  |  |  |  |  | 259.2867 |  | 0.0133 |
|  |  | 239.4887 |  |  |  |  |  |
|  |  | 235.4094 |  |  |  |  |  |
|  |  | 256.6625 |  |  |  |  |  |
|  |  | 228.0605 |  |  |  |  |  |
|  |  | 234.3386 |  |  |  |  |  |
|  |  |  |  |  | 240.1921 |  | 0.0387 |
|  |  | 237.3472 |  |  |  |  |  |
|  |  | 238.0385 |  |  |  |  |  |
|  |  | 252.6652 |  |  |  |  |  |
|  |  | 237.2688 |  |  |  |  |  |
|  |  | 207.0959 |  |  |  |  |  |
|  |  |  |  |  | 235.6251 |  | 0.0128 |
|  |  | 239.6496 |  |  |  |  |  |
|  |  | 245.0338 |  |  |  |  |  |
|  |  | 239.5143 |  |  |  |  |  |
|  |  | 225.9373 |  |  |  |  |  |
|  |  | 216.2968 |  |  |  |  |  |
|  |  |  |  |  | 229.4143 |  | 0.0196 |
|  |  | 236.7647 |  |  |  |  |  |
|  |  | 220.1670 |  |  |  |  |  |
|  |  | 219.7551 |  |  |  |  |  |
|  |  | 215.6822 |  |  |  |  |  |
|  |  | 226.0609 |  |  |  |  |  |
|  |  |  |  |  | 228.7941 |  | 0.0262 |
|  |  | 227.6323 |  |  |  |  |  |
|  |  | 222.3173 |  |  |  |  |  |
|  |  | 218.8260 |  |  |  |  |  |
|  |  | 236.3952 |  |  |  |  |  |
|  |  | 241.8738 |  |  |  |  |  |
|  |  |  |  |  | 222.5971 |  | 0.0538 |
|  |  | 220.9179 |  |  |  |  |  |
|  |  | 211.9953 |  |  |  |  |  |
|  |  | 224.9931 |  |  |  |  |  |
|  |  | 220.3752 |  |  |  |  |  |
|  |  | 208.5065 |  |  |  |  |  |
|  |  |  |  |  | 225.8249 |  | 0.0207 |
|  |  | 232.6343 |  |  |  |  |  |
|  |  | 209.3653 |  |  |  |  |  |
|  |  | 223.6041 |  |  |  |  |  |
|  |  | 233.3709 |  |  |  |  |  |
|  |  | 224.2420 |  |  |  |  |  |
|  |  |  |  |  | 222.3368 |  | 0.0258 |
|  |  | 214.2259 |  |  |  |  |  |
|  |  | 227.9593 |  |  |  |  |  |
|  |  | 228.4843 |  |  |  |  |  |
|  |  | 228.8735 |  |  |  |  |  |
|  |  | 216.4708 |  |  |  |  |  |
|  |  |  |  |  | 220.2091 |  | 0.0199 |
|  |  | 213.4667 |  |  |  |  |  |
|  |  | 236.3966 |  |  |  |  |  |
|  |  | 214.5169 |  |  |  |  |  |
|  |  | 214.6493 |  |  |  |  |  |
|  |  | 222.5305 |  |  |  |  |  |
|  |  |  |  |  | 227.1649 |  | 0.0352 |
|  |  | 226.7017 |  |  |  |  |  |
|  |  | 209.2211 |  |  |  |  |  |
|  |  | 216.2583 |  |  |  |  |  |
|  |  | 212.4796 |  |  |  |  |  |
|  |  | 209.0607 |  |  |  |  |  |
|  |  |  |  |  | 214.6426 |  | 0.0292 |
|  |  | 224.0879 |  |  |  |  |  |
|  |  | 232.6547 |  |  |  |  |  |
|  |  | 200.0692 |  |  |  |  |  |
|  |  | 231.4743 |  |  |  |  |  |
|  |  | 208.0479 |  |  |  |  |  |
|  |  |  |  |  | 218.6088 |  | 0.0448 |
|  |  | 234.0695 |  |  |  |  |  |
|  |  | 209.8465 |  |  |  |  |  |
|  |  | 205.9493 |  |  |  |  |  |
|  |  | 228.1884 |  |  |  |  |  |
|  |  | 223.8891 |  |  |  |  |  |
|  |  |  |  |  | 210.2943 |  | 0.0554 |
|  |  | 197.8991 |  |  |  |  |  |
|  |  | 234.0487 |  |  |  |  |  |
|  |  | 220.9275 |  |  |  |  |  |
|  |  | 217.3273 |  |  |  |  |  |
|  |  | 207.8836 |  |  |  |  |  |
|  |  |  |  |  | 208.8761 |  | 0.0767 |
|  |  | 217.6550 |  |  |  |  |  |
|  |  | 211.9690 |  |  |  |  |  |
|  |  | 188.5432 |  |  |  |  |  |
|  |  | 218.6016 |  |  |  |  |  |
|  |  | 208.9302 |  |  |  |  |  |
|  |  |  |  |  | 209.9433 |  | 0.0437 |
|  |  | 212.7922 |  |  |  |  |  |
|  |  | 216.4926 |  |  |  |  |  |
|  |  | 221.9972 |  |  |  |  |  |
|  |  | 184.4729 |  |  |  |  |  |
|  |  | 203.2513 |  |  |  |  |  |
|  |  |  |  |  | 204.5496 |  | 0.0665 |
|  |  | 202.6063 |  |  |  |  |  |
|  |  | 199.9576 |  |  |  |  |  |
|  |  | 202.9480 |  |  |  |  |  |
|  |  | 202.6008 |  |  |  |  |  |
|  |  | 190.8375 |  |  |  |  |  |
|  |  |  |  |  | 205.0205 |  | 0.0359 |
|  |  | 219.3103 |  |  |  |  |  |
|  |  | 210.4536 |  |  |  |  |  |
|  |  | 192.3613 |  |  |  |  |  |
|  |  | 196.6411 |  |  |  |  |  |
|  |  | 192.0846 |  |  |  |  |  |
|  |  |  |  |  | 199.8001 |  | 0.0816 |
|  |  | 190.5042 |  |  |  |  |  |
|  |  | 192.9104 |  |  |  |  |  |
|  |  | 210.2710 |  |  |  |  |  |
|  |  | 210.1778 |  |  |  |  |  |
|  |  | 188.8503 |  |  |  |  |  |
|  |  |  |  |  | 199.2430 |  | 0.0643 |
|  |  | 188.1454 |  |  |  |  |  |
|  |  | 202.1836 |  |  |  |  |  |
|  |  | 218.1502 |  |  |  |  |  |
|  |  | 190.7082 |  |  |  |  |  |

## Summary stats

```yaml
n_records: 119
step_max: 9900
step_n: 119
train_all_r2_final: 0.737
train_all_r2_first: 0
train_all_r2_max: 0.751
train_all_r2_min: 0
train_loss_final: 190.708
train_loss_first: 397.195
train_loss_max: 397.195
train_loss_min: 184.473
train_mask_r2_final: 0.74
train_mask_r2_first: 0
train_mask_r2_max: 0.751
train_mask_r2_min: 0
train_masked_acc_final: 0.441
train_masked_acc_first: 0
train_masked_acc_max: 0.479
train_masked_acc_min: 0
train_rz_masked_acc_final: 0
train_rz_masked_acc_first: 0
train_rz_masked_acc_max: 0.182
train_rz_masked_acc_min: 0
train_spec_acc_final: 0.432
train_spec_acc_first: 0
train_spec_acc_max: 0.493
train_spec_acc_min: 0
train_spec_loss_final: 2.009
train_spec_loss_first: 8.141
train_spec_loss_max: 8.141
train_spec_loss_min: 1.896
train_z_acc_final: 0.045
train_z_acc_first: 0
train_z_acc_max: 0.188
train_z_acc_min: 0
train_z_loss_final: 3.699
train_z_loss_first: 7.67
train_z_loss_max: 7.67
train_z_loss_min: 3.575
val_loss_final: 199.243
val_loss_first: 259.287
val_loss_max: 259.287
val_loss_min: 199.243
val_loss_redshift_final: 3.8625
val_loss_redshift_first: 5.0265
val_loss_redshift_max: 5.0265
val_loss_redshift_min: 3.8625
val_loss_spectrum_final: 2.1859
val_loss_spectrum_first: 3.0305
val_loss_spectrum_max: 3.0305
val_loss_spectrum_min: 2.1856
val_loss_total_final: 2.192
val_loss_total_first: 3.0378
val_loss_total_max: 3.0378
val_loss_total_min: 2.1918
val_masked_spec_acc_final: 0.4113
val_masked_spec_acc_first: 0.3051
val_masked_spec_acc_max: 0.4113
val_masked_spec_acc_min: 0.3019
val_overall_acc_final: 0.4005
val_overall_acc_first: 0.2891
val_overall_acc_max: 0.4014
val_overall_acc_min: 0.2865
val_redshift_acc_final: 0.0643
val_redshift_acc_first: 0.0133
val_redshift_acc_masked_final: 0.0441
val_redshift_acc_masked_first: 0.0132
val_redshift_acc_masked_max: 0.0473
val_redshift_acc_masked_min: 0.0066
val_redshift_acc_max: 0.0816
val_redshift_acc_min: 0.0128
val_spectrum_acc_final: 0.4017
val_spectrum_acc_first: 0.2901
val_spectrum_acc_max: 0.4026
val_spectrum_acc_min: 0.2875
```

