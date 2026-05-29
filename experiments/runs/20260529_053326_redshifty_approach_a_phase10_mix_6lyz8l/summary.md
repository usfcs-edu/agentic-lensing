# redshifty_approach_a_phase10_mix

_Approach A with Phase 10 final hparams on the 4-way data mix matching the
redshifty author's NERSC setup: sv3+main × bright+dark (vs all prior runs
which used sv3-bright only). 1137 pixels, ~1.82M raw spectra, ~750 GiB on
disk. Tests whether the ignition gap closes with author's data diversity.
_

- **run dir:** `/raid/benson/git/agentic-lensing/experiments/runs/20260529_053326_redshifty_approach_a_phase10_mix_6lyz8l`
- **repo:** redshifty
- **started:** 2026-05-29 05:33:26 UTC
- **finished:** 2026-05-29 13:24:31 UTC
- **wallclock:** 28265.9s
- **exit code:** 0
- **tags:** xlarge-scale, transformer, approach-a, track3, data-mix-test

## Command
```
/home/benson/.venvs/redshifty/bin/python nersc/train_transformer.py --manifest /raid/benson/data/desi_dr1_medium/manifest_mix.jsonl --tokenizer-ckpt /raid/benson/data/desi_dr1_medium/checkpoints/tokenizer_v1_large/best.pt --tokenizer-kind v1 --approach a --steps 10000 --batch-size 32 --lr 4e-4 --warmup 500 --healpix-holdout-frac 0.05 --val-every 500 --save-every 2500 --log-every 100 --num-workers 4 --amp --redshift-loss-weight 50.0 --encoder-mask-ratio 0.50 --ar-eval-batches 8 --run-name approach_a_phase10_mix --scratch-out /raid/benson/data/desi_dr1_medium/checkpoints --wandb-mode disabled --no-push-wandb-artifact
```

## Trajectory

| approach | epoch | train_loss | train_acc | train_redshift_acc | val_loss | val_acc | val_redshift_acc |
|---|---|---|---|---|---|---|---|
|  |  | 398.8898 |  |  |  |  |  |
|  |  | 247.9704 |  |  |  |  |  |
|  |  | 194.9174 |  |  |  |  |  |
|  |  | 226.5921 |  |  |  |  |  |
|  |  | 198.7148 |  |  |  |  |  |
|  |  | 211.9413 |  |  |  |  |  |
|  |  |  |  |  | 251.9119 |  | 0.0278 |
|  |  | 207.8342 |  |  |  |  |  |
|  |  | 267.0298 |  |  |  |  |  |
|  |  | 200.6091 |  |  |  |  |  |
|  |  | 203.3260 |  |  |  |  |  |
|  |  | 174.7903 |  |  |  |  |  |
|  |  |  |  |  | 246.2352 |  | 0.0101 |
|  |  | 201.1380 |  |  |  |  |  |
|  |  | 214.7069 |  |  |  |  |  |
|  |  | 187.9801 |  |  |  |  |  |
|  |  | 183.3422 |  |  |  |  |  |
|  |  | 197.9589 |  |  |  |  |  |
|  |  |  |  |  | 244.0955 |  | 0.0093 |
|  |  | 196.2838 |  |  |  |  |  |
|  |  | 191.3194 |  |  |  |  |  |
|  |  | 210.7063 |  |  |  |  |  |
|  |  | 195.2622 |  |  |  |  |  |
|  |  | 207.0366 |  |  |  |  |  |
|  |  |  |  |  | 238.5405 |  | 0.0232 |
|  |  | 185.5438 |  |  |  |  |  |
|  |  | 202.6467 |  |  |  |  |  |
|  |  | 178.1214 |  |  |  |  |  |
|  |  | 201.1233 |  |  |  |  |  |
|  |  | 164.2863 |  |  |  |  |  |
|  |  |  |  |  | 235.5367 |  | 0.0187 |
|  |  | 172.1626 |  |  |  |  |  |
|  |  | 187.2399 |  |  |  |  |  |
|  |  | 222.3298 |  |  |  |  |  |
|  |  | 164.2382 |  |  |  |  |  |
|  |  | 193.1043 |  |  |  |  |  |
|  |  |  |  |  | 231.9634 |  | 0.0153 |
|  |  | 205.9321 |  |  |  |  |  |
|  |  | 188.9821 |  |  |  |  |  |
|  |  | 213.2840 |  |  |  |  |  |
|  |  | 195.3101 |  |  |  |  |  |
|  |  | 186.3017 |  |  |  |  |  |
|  |  |  |  |  | 227.1584 |  | 0.0227 |
|  |  | 160.7126 |  |  |  |  |  |
|  |  | 169.4833 |  |  |  |  |  |
|  |  | 188.0811 |  |  |  |  |  |
|  |  | 179.7526 |  |  |  |  |  |
|  |  | 157.7194 |  |  |  |  |  |
|  |  |  |  |  | 225.5049 |  | 0.0231 |
|  |  | 208.3035 |  |  |  |  |  |
|  |  | 203.1719 |  |  |  |  |  |
|  |  | 160.5602 |  |  |  |  |  |
|  |  | 151.7242 |  |  |  |  |  |
|  |  | 199.0769 |  |  |  |  |  |
|  |  |  |  |  | 219.0447 |  | 0.0602 |
|  |  | 194.5015 |  |  |  |  |  |
|  |  | 131.5074 |  |  |  |  |  |
|  |  | 161.5064 |  |  |  |  |  |
|  |  | 170.7838 |  |  |  |  |  |
|  |  | 156.2203 |  |  |  |  |  |
|  |  |  |  |  | 218.6275 |  | 0.0342 |
|  |  | 162.1351 |  |  |  |  |  |
|  |  | 196.6264 |  |  |  |  |  |
|  |  | 210.0102 |  |  |  |  |  |
|  |  | 151.1678 |  |  |  |  |  |
|  |  | 164.8531 |  |  |  |  |  |
|  |  |  |  |  | 213.9090 |  | 0.0424 |
|  |  | 150.6210 |  |  |  |  |  |
|  |  | 176.5351 |  |  |  |  |  |
|  |  | 167.4390 |  |  |  |  |  |
|  |  | 191.4691 |  |  |  |  |  |
|  |  | 169.4075 |  |  |  |  |  |
|  |  |  |  |  | 217.9772 |  | 0.0338 |
|  |  | 155.6940 |  |  |  |  |  |
|  |  | 161.5879 |  |  |  |  |  |
|  |  | 151.1809 |  |  |  |  |  |
|  |  | 157.6722 |  |  |  |  |  |
|  |  | 139.6216 |  |  |  |  |  |
|  |  |  |  |  | 210.2586 |  | 0.0547 |
|  |  | 110.9721 |  |  |  |  |  |
|  |  | 177.6776 |  |  |  |  |  |
|  |  | 137.1347 |  |  |  |  |  |
|  |  | 153.7931 |  |  |  |  |  |
|  |  | 137.1570 |  |  |  |  |  |
|  |  |  |  |  | 212.0104 |  | 0.0524 |
|  |  | 151.0296 |  |  |  |  |  |
|  |  | 155.6335 |  |  |  |  |  |
|  |  | 153.3435 |  |  |  |  |  |
|  |  | 152.7583 |  |  |  |  |  |
|  |  | 152.8644 |  |  |  |  |  |
|  |  |  |  |  | 204.5809 |  | 0.0814 |
|  |  | 163.2928 |  |  |  |  |  |
|  |  | 156.2283 |  |  |  |  |  |
|  |  | 160.3027 |  |  |  |  |  |
|  |  | 119.4518 |  |  |  |  |  |
|  |  | 142.4314 |  |  |  |  |  |
|  |  |  |  |  | 197.8641 |  | 0.0897 |
|  |  | 167.4297 |  |  |  |  |  |
|  |  | 134.4013 |  |  |  |  |  |
|  |  | 155.8996 |  |  |  |  |  |
|  |  | 172.4122 |  |  |  |  |  |
|  |  | 140.2558 |  |  |  |  |  |
|  |  |  |  |  | 196.6068 |  | 0.0922 |
|  |  | 212.0666 |  |  |  |  |  |
|  |  | 163.1533 |  |  |  |  |  |
|  |  | 129.2372 |  |  |  |  |  |
|  |  | 116.6142 |  |  |  |  |  |
|  |  | 174.6935 |  |  |  |  |  |
|  |  |  |  |  | 192.5806 |  | 0.1085 |
|  |  | 98.1675 |  |  |  |  |  |
|  |  | 147.4444 |  |  |  |  |  |
|  |  | 143.7235 |  |  |  |  |  |
|  |  | 152.3563 |  |  |  |  |  |
|  |  | 115.2885 |  |  |  |  |  |
|  |  |  |  |  | 190.6703 |  | 0.1486 |
|  |  | 147.5023 |  |  |  |  |  |
|  |  | 168.9507 |  |  |  |  |  |
|  |  | 162.9640 |  |  |  |  |  |
|  |  | 137.7842 |  |  |  |  |  |

## Summary stats

```yaml
n_records: 119
step_max: 9900
step_n: 119
train_all_r2_final: 0.824
train_all_r2_first: 0
train_all_r2_max: 0.886
train_all_r2_min: 0
train_loss_final: 137.784
train_loss_first: 398.89
train_loss_max: 398.89
train_loss_min: 98.1675
train_mask_r2_final: 0.821
train_mask_r2_first: 0
train_mask_r2_max: 0.884
train_mask_r2_min: 0
train_masked_acc_final: 0.644
train_masked_acc_first: 0
train_masked_acc_max: 0.735
train_masked_acc_min: 0
train_rz_masked_acc_final: 0.308
train_rz_masked_acc_first: 0
train_rz_masked_acc_max: 0.4
train_rz_masked_acc_min: 0
train_spec_acc_final: 0.644
train_spec_acc_first: 0
train_spec_acc_max: 0.738
train_spec_acc_min: 0
train_spec_loss_final: 1.342
train_spec_loss_first: 7.891
train_spec_loss_max: 7.891
train_spec_loss_min: 0.873
train_z_acc_final: 0.429
train_z_acc_first: 0
train_z_acc_max: 0.579
train_z_acc_min: 0
train_z_loss_final: 2.67
train_z_loss_first: 7.707
train_z_loss_max: 7.707
train_z_loss_min: 1.9
val_loss_final: 190.67
val_loss_first: 251.912
val_loss_max: 251.912
val_loss_min: 190.67
val_loss_redshift_final: 3.6914
val_loss_redshift_first: 4.8821
val_loss_redshift_max: 4.8821
val_loss_redshift_min: 3.6914
val_loss_spectrum_final: 2.2306
val_loss_spectrum_first: 3.0401
val_loss_spectrum_max: 3.0401
val_loss_spectrum_min: 2.2306
val_loss_total_final: 2.2359
val_loss_total_first: 3.0468
val_loss_total_max: 3.0468
val_loss_total_min: 2.2359
val_masked_spec_acc_final: 0.3992
val_masked_spec_acc_first: 0.307
val_masked_spec_acc_max: 0.3992
val_masked_spec_acc_min: 0.307
val_overall_acc_final: 0.3988
val_overall_acc_first: 0.3
val_overall_acc_max: 0.3988
val_overall_acc_min: 0.3
val_redshift_acc_final: 0.1486
val_redshift_acc_first: 0.0278
val_redshift_acc_masked_final: 0.0524
val_redshift_acc_masked_first: 0.0272
val_redshift_acc_masked_max: 0.0612
val_redshift_acc_masked_min: 0.0061
val_redshift_acc_max: 0.1486
val_redshift_acc_min: 0.0093
val_spectrum_acc_final: 0.3998
val_spectrum_acc_first: 0.301
val_spectrum_acc_max: 0.3998
val_spectrum_acc_min: 0.301
```

