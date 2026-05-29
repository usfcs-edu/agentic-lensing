# redshifty_approach_a_large

_Approach A transformer on large subset with V1-large tokenizer (10k steps targeting redshift-ignition point ~6.5k seen by author)_

- **run dir:** `/raid/benson/git/agentic-lensing/experiments/runs/20260527_050908_redshifty_approach_a_large_7onz0u`
- **repo:** redshifty
- **started:** 2026-05-27 05:09:08 UTC
- **finished:** 2026-05-27 06:23:31 UTC
- **wallclock:** 4462.5s
- **exit code:** 0
- **tags:** large-scale, transformer, approach-a, track3

## Command
```
/home/benson/.venvs/redshifty/bin/python nersc/train_transformer.py --manifest /raid/benson/data/desi_dr1_medium/manifest.jsonl --tokenizer-ckpt /raid/benson/data/desi_dr1_medium/checkpoints/tokenizer_v1_large/best.pt --tokenizer-kind v1 --approach a --steps 10000 --batch-size 8 --lr 2e-4 --warmup 500 --healpix-holdout-frac 0.05 --val-every 500 --save-every 2500 --log-every 100 --num-workers 4 --amp --redshift-loss-weight 50.0 --encoder-mask-ratio 0.50 --ar-eval-batches 4 --run-name approach_a_large --scratch-out /raid/benson/data/desi_dr1_medium/checkpoints --wandb-mode disabled --no-push-wandb-artifact
```

## Trajectory

| approach | epoch | train_loss | train_acc | train_redshift_acc | val_loss | val_acc | val_redshift_acc |
|---|---|---|---|---|---|---|---|
|  |  | 397.7187 |  |  |  |  |  |
|  |  | 332.9998 |  |  |  |  |  |
|  |  | 263.1015 |  |  |  |  |  |
|  |  | 250.4893 |  |  |  |  |  |
|  |  | 310.8342 |  |  |  |  |  |
|  |  | 263.0152 |  |  |  |  |  |
|  |  |  |  |  | 264.8587 |  | 0.0107 |
|  |  | 277.6508 |  |  |  |  |  |
|  |  | 270.9641 |  |  |  |  |  |
|  |  | 266.3197 |  |  |  |  |  |
|  |  | 238.4145 |  |  |  |  |  |
|  |  | 257.4832 |  |  |  |  |  |
|  |  |  |  |  | 245.5628 |  | 0.0157 |
|  |  | 236.7704 |  |  |  |  |  |
|  |  | 246.8712 |  |  |  |  |  |
|  |  | 266.9263 |  |  |  |  |  |
|  |  | 215.2750 |  |  |  |  |  |
|  |  | 230.0598 |  |  |  |  |  |
|  |  |  |  |  | 249.5944 |  | 0.0033 |
|  |  | 257.5986 |  |  |  |  |  |
|  |  | 242.0312 |  |  |  |  |  |
|  |  | 244.5138 |  |  |  |  |  |
|  |  | 294.0906 |  |  |  |  |  |
|  |  | 249.9919 |  |  |  |  |  |
|  |  |  |  |  | 250.6769 |  | 0.0062 |
|  |  | 233.2572 |  |  |  |  |  |
|  |  | 242.8917 |  |  |  |  |  |
|  |  | 246.5809 |  |  |  |  |  |
|  |  | 248.9994 |  |  |  |  |  |
|  |  | 235.3711 |  |  |  |  |  |
|  |  |  |  |  | 249.2367 |  | 0.0073 |
|  |  | 214.6982 |  |  |  |  |  |
|  |  | 250.7541 |  |  |  |  |  |
|  |  | 259.1668 |  |  |  |  |  |
|  |  | 242.2013 |  |  |  |  |  |
|  |  | 224.7384 |  |  |  |  |  |
|  |  |  |  |  | 237.4164 |  | 0.0247 |
|  |  | 252.0863 |  |  |  |  |  |
|  |  | 229.3920 |  |  |  |  |  |
|  |  | 211.9153 |  |  |  |  |  |
|  |  | 257.0052 |  |  |  |  |  |
|  |  | 218.4138 |  |  |  |  |  |
|  |  |  |  |  | 243.3045 |  | 0.0073 |
|  |  | 250.0016 |  |  |  |  |  |
|  |  | 231.8688 |  |  |  |  |  |
|  |  | 313.7935 |  |  |  |  |  |
|  |  | 266.1844 |  |  |  |  |  |
|  |  | 213.3425 |  |  |  |  |  |
|  |  |  |  |  | 235.3016 |  | 0.0150 |
|  |  | 221.9728 |  |  |  |  |  |
|  |  | 214.3765 |  |  |  |  |  |
|  |  | 267.1687 |  |  |  |  |  |
|  |  | 249.7871 |  |  |  |  |  |
|  |  | 245.6300 |  |  |  |  |  |
|  |  |  |  |  | 232.4514 |  | 0.0389 |
|  |  | 231.8676 |  |  |  |  |  |
|  |  | 299.5697 |  |  |  |  |  |
|  |  | 214.8687 |  |  |  |  |  |
|  |  | 224.8764 |  |  |  |  |  |
|  |  | 224.1605 |  |  |  |  |  |
|  |  |  |  |  | 241.2416 |  | 0.0129 |
|  |  | 233.7159 |  |  |  |  |  |
|  |  | 211.3230 |  |  |  |  |  |
|  |  | 209.1450 |  |  |  |  |  |
|  |  | 224.7744 |  |  |  |  |  |
|  |  | 248.7213 |  |  |  |  |  |
|  |  |  |  |  | 234.7057 |  | 0.0067 |
|  |  | 243.9689 |  |  |  |  |  |
|  |  | 225.3428 |  |  |  |  |  |
|  |  | 251.6003 |  |  |  |  |  |
|  |  | 212.6290 |  |  |  |  |  |
|  |  | 230.9493 |  |  |  |  |  |
|  |  |  |  |  | 235.3854 |  | 0.0285 |
|  |  | 234.8915 |  |  |  |  |  |
|  |  | 225.1004 |  |  |  |  |  |
|  |  | 259.4066 |  |  |  |  |  |
|  |  | 230.2383 |  |  |  |  |  |
|  |  | 224.5517 |  |  |  |  |  |
|  |  |  |  |  | 235.0113 |  | 0.0173 |
|  |  | 224.6291 |  |  |  |  |  |
|  |  | 226.5077 |  |  |  |  |  |
|  |  | 219.2450 |  |  |  |  |  |
|  |  | 221.1422 |  |  |  |  |  |
|  |  | 243.9213 |  |  |  |  |  |
|  |  |  |  |  | 233.9141 |  | 0.0273 |
|  |  | 219.5493 |  |  |  |  |  |
|  |  | 240.0063 |  |  |  |  |  |
|  |  | 246.1665 |  |  |  |  |  |
|  |  | 223.9080 |  |  |  |  |  |
|  |  | 202.8065 |  |  |  |  |  |
|  |  |  |  |  | 229.1376 |  | 0.0240 |
|  |  | 223.9664 |  |  |  |  |  |
|  |  | 214.4435 |  |  |  |  |  |
|  |  | 211.0203 |  |  |  |  |  |
|  |  | 209.6039 |  |  |  |  |  |
|  |  | 222.5695 |  |  |  |  |  |
|  |  |  |  |  | 231.2419 |  | 0.0270 |
|  |  | 231.3251 |  |  |  |  |  |
|  |  | 235.1194 |  |  |  |  |  |
|  |  | 227.1070 |  |  |  |  |  |
|  |  | 219.3611 |  |  |  |  |  |
|  |  | 282.0374 |  |  |  |  |  |
|  |  |  |  |  | 228.1703 |  | 0.0247 |
|  |  | 206.0900 |  |  |  |  |  |
|  |  | 252.7978 |  |  |  |  |  |
|  |  | 210.6288 |  |  |  |  |  |
|  |  | 236.1558 |  |  |  |  |  |
|  |  | 217.6834 |  |  |  |  |  |
|  |  |  |  |  | 224.6743 |  | 0.0240 |
|  |  | 216.5815 |  |  |  |  |  |
|  |  | 180.7315 |  |  |  |  |  |
|  |  | 208.9405 |  |  |  |  |  |
|  |  | 254.0366 |  |  |  |  |  |
|  |  | 192.7831 |  |  |  |  |  |
|  |  |  |  |  | 226.9932 |  | 0.0370 |
|  |  | 247.7426 |  |  |  |  |  |
|  |  | 228.6787 |  |  |  |  |  |
|  |  | 239.4453 |  |  |  |  |  |
|  |  | 227.7386 |  |  |  |  |  |

## Summary stats

```yaml
n_records: 119
step_max: 9900
step_n: 119
train_all_r2_final: 0.678
train_all_r2_first: 0
train_all_r2_max: 0.75
train_all_r2_min: 0
train_loss_final: 227.739
train_loss_first: 397.719
train_loss_max: 397.719
train_loss_min: 180.732
train_mask_r2_final: 0.683
train_mask_r2_first: 0
train_mask_r2_max: 0.747
train_mask_r2_min: 0
train_masked_acc_final: 0.363
train_masked_acc_first: 0
train_masked_acc_max: 0.512
train_masked_acc_min: 0
train_rz_masked_acc_final: 0
train_rz_masked_acc_first: 0
train_rz_masked_acc_max: 0.5
train_rz_masked_acc_min: 0
train_spec_acc_final: 0.35
train_spec_acc_first: 0
train_spec_acc_max: 0.51
train_spec_acc_min: 0
train_spec_loss_final: 2.457
train_spec_loss_first: 7.658
train_spec_loss_max: 7.658
train_spec_loss_min: 1.906
train_z_acc_final: 0
train_z_acc_first: 0
train_z_acc_max: 0.5
train_z_acc_min: 0
train_z_loss_final: 4.415
train_z_loss_first: 7.689
train_z_loss_max: 7.689
train_z_loss_min: 3.495
val_loss_final: 226.993
val_loss_first: 264.859
val_loss_max: 264.859
val_loss_min: 224.674
val_loss_redshift_final: 4.4087
val_loss_redshift_first: 5.1369
val_loss_redshift_max: 5.1369
val_loss_redshift_min: 4.3624
val_loss_spectrum_final: 2.189
val_loss_spectrum_first: 3.1245
val_loss_spectrum_max: 3.1245
val_loss_spectrum_min: 2.189
val_loss_total_final: 2.1971
val_loss_total_first: 3.1318
val_loss_total_max: 3.1318
val_loss_total_min: 2.1971
val_masked_spec_acc_final: 0.4091
val_masked_spec_acc_first: 0.3121
val_masked_spec_acc_max: 0.4097
val_masked_spec_acc_min: 0.3121
val_overall_acc_final: 0.4093
val_overall_acc_first: 0.3131
val_overall_acc_max: 0.4093
val_overall_acc_min: 0.3131
val_redshift_acc_final: 0.037
val_redshift_acc_first: 0.0107
val_redshift_acc_masked_final: 0.0342
val_redshift_acc_masked_first: 0.015
val_redshift_acc_masked_max: 0.042
val_redshift_acc_masked_min: 0
val_redshift_acc_max: 0.0389
val_redshift_acc_min: 0.0033
val_spectrum_acc_final: 0.4106
val_spectrum_acc_first: 0.3142
val_spectrum_acc_max: 0.4106
val_spectrum_acc_min: 0.3142
```

