# redshifty_approach_a_phase10_mix_mps_20k

_20k-step Apple Silicon / MPS ignition run — identical to
redshifty_approach_a_phase10_mix_mps.yaml but --steps 20000 (the redshifty author's own
recommendation: "10000 steps was barely enough to see ignition kinetics; future runs
should use >=20000 steps to give the post-ignition phase room"). Tests whether the MPS
trajectory climbs past the 10k run's 7.88% peak toward the reference's 14.86% / the
>=10%-sustained bar. Same frozen V1 tokenizer, same 4-way mix, same hparams, same seed.
Driven through the unmodified tools/spectrumfm/exp_run.py harness.
_

- **run dir:** `/Users/benson/sync-git/sync-lens/agentic-lensing/reproductions/redshifty/apple-silicon/data/runs/20260603_214732_redshifty_approach_a_phase10_mix_mps_20k_il5t7z`
- **repo:** redshifty
- **started:** 2026-06-03 21:47:32 UTC
- **finished:** 2026-06-04 13:53:13 UTC
- **wallclock:** 57941.3s
- **exit code:** 0
- **tags:** apple-silicon, mps, transformer, approach-a, ignition, data-mix, 20k

## Command
```
/Users/benson/.venvs/redshifty/bin/python nersc/train_transformer.py --manifest /Users/benson/sync-git/sync-lens/agentic-lensing/reproductions/redshifty/apple-silicon/_raid/benson/data/desi_dr1_medium/manifest_mix_local.jsonl --tokenizer-ckpt /Users/benson/sync-git/sync-lens/agentic-lensing/reproductions/redshifty/apple-silicon/_raid/benson/data/desi_dr1_medium/checkpoints/tokenizer_v1_large/best.pt --tokenizer-kind v1 --approach a --steps 20000 --batch-size 32 --lr 4e-4 --warmup 500 --healpix-holdout-frac 0.05 --val-every 500 --save-every 2500 --log-every 100 --num-workers 0 --amp --redshift-loss-weight 50.0 --encoder-mask-ratio 0.50 --ar-eval-batches 8 --run-name approach_a_phase10_mix_mps_20k --scratch-out /Users/benson/sync-git/sync-lens/agentic-lensing/reproductions/redshifty/apple-silicon/_raid/benson/data/desi_dr1_medium/checkpoints --wandb-mode disabled --no-push-wandb-artifact
```

## Trajectory

| approach | epoch | train_loss | train_acc | train_redshift_acc | val_loss | val_acc | val_redshift_acc |
|---|---|---|---|---|---|---|---|
|  |  | 402.9259 |  |  |  |  |  |
|  |  | 256.1150 |  |  |  |  |  |
|  |  | 210.6630 |  |  |  |  |  |
|  |  | 242.4478 |  |  |  |  |  |
|  |  | 216.7055 |  |  |  |  |  |
|  |  | 185.5761 |  |  |  |  |  |
|  |  |  |  |  | 253.9225 |  | 0.0386 |
|  |  | 222.4716 |  |  |  |  |  |
|  |  | 235.3088 |  |  |  |  |  |
|  |  | 205.5509 |  |  |  |  |  |
|  |  | 190.1127 |  |  |  |  |  |
|  |  | 227.4664 |  |  |  |  |  |
|  |  |  |  |  | 244.3944 |  | 0.0315 |
|  |  | 218.3940 |  |  |  |  |  |
|  |  | 191.8743 |  |  |  |  |  |
|  |  | 215.7424 |  |  |  |  |  |
|  |  | 240.2977 |  |  |  |  |  |
|  |  | 219.9216 |  |  |  |  |  |
|  |  |  |  |  | 243.9676 |  | 0.0173 |
|  |  | 197.7861 |  |  |  |  |  |
|  |  | 162.3226 |  |  |  |  |  |
|  |  | 174.5977 |  |  |  |  |  |
|  |  | 208.8303 |  |  |  |  |  |
|  |  | 189.3314 |  |  |  |  |  |
|  |  |  |  |  | 238.0917 |  | 0.0265 |
|  |  | 218.0942 |  |  |  |  |  |
|  |  | 183.4425 |  |  |  |  |  |
|  |  | 215.1239 |  |  |  |  |  |
|  |  | 207.0783 |  |  |  |  |  |
|  |  | 217.0397 |  |  |  |  |  |
|  |  |  |  |  | 239.9406 |  | 0.0144 |
|  |  | 189.0952 |  |  |  |  |  |
|  |  | 194.0718 |  |  |  |  |  |
|  |  | 203.8055 |  |  |  |  |  |
|  |  | 197.2500 |  |  |  |  |  |
|  |  | 211.7116 |  |  |  |  |  |
|  |  |  |  |  | 235.9486 |  | 0.0479 |
|  |  | 204.8945 |  |  |  |  |  |
|  |  | 221.0306 |  |  |  |  |  |
|  |  | 213.4310 |  |  |  |  |  |
|  |  | 190.7856 |  |  |  |  |  |
|  |  | 189.7443 |  |  |  |  |  |
|  |  |  |  |  | 238.1508 |  | 0.0316 |
|  |  | 220.5507 |  |  |  |  |  |
|  |  | 195.1134 |  |  |  |  |  |
|  |  | 199.5276 |  |  |  |  |  |
|  |  | 208.2026 |  |  |  |  |  |
|  |  | 180.0482 |  |  |  |  |  |
|  |  |  |  |  | 233.7210 |  | 0.0230 |
|  |  | 180.3511 |  |  |  |  |  |
|  |  | 214.7861 |  |  |  |  |  |
|  |  | 182.9202 |  |  |  |  |  |
|  |  | 194.9274 |  |  |  |  |  |
|  |  | 203.0770 |  |  |  |  |  |
|  |  |  |  |  | 235.6659 |  | 0.0094 |
|  |  | 206.6869 |  |  |  |  |  |
|  |  | 213.6997 |  |  |  |  |  |
|  |  | 219.1788 |  |  |  |  |  |
|  |  | 223.1128 |  |  |  |  |  |
|  |  | 182.7651 |  |  |  |  |  |
|  |  |  |  |  | 231.6743 |  | 0.0170 |
|  |  | 213.1816 |  |  |  |  |  |
|  |  | 189.7438 |  |  |  |  |  |
|  |  | 175.2728 |  |  |  |  |  |
|  |  | 212.3414 |  |  |  |  |  |
|  |  | 207.2122 |  |  |  |  |  |
|  |  |  |  |  | 236.9282 |  | 0.0347 |
|  |  | 194.6735 |  |  |  |  |  |
|  |  | 190.5978 |  |  |  |  |  |
|  |  | 202.3962 |  |  |  |  |  |
|  |  | 209.2785 |  |  |  |  |  |
|  |  | 203.3010 |  |  |  |  |  |
|  |  |  |  |  | 228.7924 |  | 0.0286 |
|  |  | 213.4561 |  |  |  |  |  |
|  |  | 212.1042 |  |  |  |  |  |
|  |  | 196.5278 |  |  |  |  |  |
|  |  | 186.0068 |  |  |  |  |  |
|  |  | 189.4772 |  |  |  |  |  |
|  |  |  |  |  | 224.5371 |  | 0.0502 |
|  |  | 176.9695 |  |  |  |  |  |
|  |  | 204.4797 |  |  |  |  |  |
|  |  | 202.1134 |  |  |  |  |  |
|  |  | 207.5157 |  |  |  |  |  |
|  |  | 160.8700 |  |  |  |  |  |
|  |  |  |  |  | 231.0496 |  | 0.0117 |
|  |  | 194.9974 |  |  |  |  |  |
|  |  | 168.1318 |  |  |  |  |  |
|  |  | 195.5913 |  |  |  |  |  |
|  |  | 178.5535 |  |  |  |  |  |
|  |  | 180.7177 |  |  |  |  |  |
|  |  |  |  |  | 225.2100 |  | 0.0554 |
|  |  | 188.5279 |  |  |  |  |  |
|  |  | 232.9172 |  |  |  |  |  |
|  |  | 197.7372 |  |  |  |  |  |
|  |  | 197.0580 |  |  |  |  |  |
|  |  | 185.5338 |  |  |  |  |  |
|  |  |  |  |  | 230.6422 |  | 0.0348 |
|  |  | 169.4353 |  |  |  |  |  |
|  |  | 184.6965 |  |  |  |  |  |
|  |  | 211.7560 |  |  |  |  |  |
|  |  | 184.1533 |  |  |  |  |  |
|  |  | 166.4443 |  |  |  |  |  |
|  |  |  |  |  | 224.7341 |  | 0.0175 |
|  |  | 181.4077 |  |  |  |  |  |
|  |  | 197.4464 |  |  |  |  |  |
|  |  | 150.0534 |  |  |  |  |  |
|  |  | 189.6929 |  |  |  |  |  |
|  |  | 151.6497 |  |  |  |  |  |
|  |  |  |  |  | 223.1158 |  | 0.0223 |
|  |  | 169.4953 |  |  |  |  |  |
|  |  | 146.6275 |  |  |  |  |  |
|  |  | 203.1768 |  |  |  |  |  |
|  |  | 152.3740 |  |  |  |  |  |
|  |  | 184.7857 |  |  |  |  |  |
|  |  |  |  |  | 227.1988 |  | 0.0360 |
|  |  | 177.7803 |  |  |  |  |  |
|  |  | 178.7448 |  |  |  |  |  |
|  |  | 167.8891 |  |  |  |  |  |
|  |  | 172.7902 |  |  |  |  |  |
|  |  | 175.9622 |  |  |  |  |  |
|  |  |  |  |  | 223.6077 |  | 0.0552 |
|  |  | 152.9077 |  |  |  |  |  |
|  |  | 190.6170 |  |  |  |  |  |
|  |  | 168.2487 |  |  |  |  |  |
|  |  | 184.3742 |  |  |  |  |  |
|  |  | 160.0282 |  |  |  |  |  |
|  |  |  |  |  | 220.1926 |  | 0.0373 |
|  |  | 182.5462 |  |  |  |  |  |
|  |  | 124.0001 |  |  |  |  |  |
|  |  | 180.0868 |  |  |  |  |  |
|  |  | 162.5130 |  |  |  |  |  |
|  |  | 198.0579 |  |  |  |  |  |
|  |  |  |  |  | 221.0233 |  | 0.0317 |
|  |  | 165.4919 |  |  |  |  |  |
|  |  | 158.7419 |  |  |  |  |  |
|  |  | 163.0419 |  |  |  |  |  |
|  |  | 165.8872 |  |  |  |  |  |
|  |  | 146.4038 |  |  |  |  |  |
|  |  |  |  |  | 222.1666 |  | 0.0511 |
|  |  | 147.9250 |  |  |  |  |  |
|  |  | 180.0376 |  |  |  |  |  |
|  |  | 123.6550 |  |  |  |  |  |
|  |  | 177.8301 |  |  |  |  |  |
|  |  | 174.7109 |  |  |  |  |  |
|  |  |  |  |  | 217.6765 |  | 0.0411 |
|  |  | 175.0297 |  |  |  |  |  |
|  |  | 170.9855 |  |  |  |  |  |
|  |  | 158.8436 |  |  |  |  |  |
|  |  | 190.3972 |  |  |  |  |  |
|  |  | 149.8009 |  |  |  |  |  |
|  |  |  |  |  | 216.1267 |  | 0.0772 |
|  |  | 194.1105 |  |  |  |  |  |
|  |  | 196.1591 |  |  |  |  |  |
|  |  | 165.3252 |  |  |  |  |  |
|  |  | 157.7029 |  |  |  |  |  |
|  |  | 198.0466 |  |  |  |  |  |
|  |  |  |  |  | 215.2981 |  | 0.0928 |
|  |  | 160.7177 |  |  |  |  |  |
|  |  | 175.2478 |  |  |  |  |  |
|  |  | 160.3389 |  |  |  |  |  |
|  |  | 159.7612 |  |  |  |  |  |
|  |  | 199.3233 |  |  |  |  |  |
|  |  |  |  |  | 213.0961 |  | 0.0708 |
|  |  | 190.7760 |  |  |  |  |  |
|  |  | 167.8542 |  |  |  |  |  |
|  |  | 194.4246 |  |  |  |  |  |
|  |  | 128.7061 |  |  |  |  |  |
|  |  | 153.6025 |  |  |  |  |  |
|  |  |  |  |  | 215.3958 |  | 0.0766 |
|  |  | 154.5121 |  |  |  |  |  |
|  |  | 175.5198 |  |  |  |  |  |
|  |  | 132.2872 |  |  |  |  |  |
|  |  | 178.8582 |  |  |  |  |  |
|  |  | 153.2599 |  |  |  |  |  |
|  |  |  |  |  | 211.7742 |  | 0.0973 |
|  |  | 178.3288 |  |  |  |  |  |
|  |  | 196.6132 |  |  |  |  |  |
|  |  | 175.9887 |  |  |  |  |  |
|  |  | 186.5171 |  |  |  |  |  |
|  |  | 142.5212 |  |  |  |  |  |
|  |  |  |  |  | 207.6828 |  | 0.0909 |
|  |  | 151.5344 |  |  |  |  |  |
|  |  | 175.2670 |  |  |  |  |  |
|  |  | 143.4419 |  |  |  |  |  |
|  |  | 177.9725 |  |  |  |  |  |
|  |  | 108.1362 |  |  |  |  |  |
|  |  |  |  |  | 207.0125 |  | 0.0971 |
|  |  | 176.7511 |  |  |  |  |  |
|  |  | 185.4308 |  |  |  |  |  |
|  |  | 139.8923 |  |  |  |  |  |
|  |  | 160.0369 |  |  |  |  |  |
|  |  | 188.0290 |  |  |  |  |  |
|  |  |  |  |  | 205.9537 |  | 0.0985 |
|  |  | 173.1296 |  |  |  |  |  |
|  |  | 167.3259 |  |  |  |  |  |
|  |  | 178.6435 |  |  |  |  |  |
|  |  | 175.7540 |  |  |  |  |  |
|  |  | 147.3861 |  |  |  |  |  |
|  |  |  |  |  | 207.1080 |  | 0.0855 |
|  |  | 147.8536 |  |  |  |  |  |
|  |  | 143.5772 |  |  |  |  |  |
|  |  | 168.8880 |  |  |  |  |  |
|  |  | 136.5082 |  |  |  |  |  |
|  |  | 152.7294 |  |  |  |  |  |
|  |  |  |  |  | 205.1833 |  | 0.1035 |
|  |  | 164.5792 |  |  |  |  |  |
|  |  | 148.4832 |  |  |  |  |  |
|  |  | 135.5109 |  |  |  |  |  |
|  |  | 157.0008 |  |  |  |  |  |
|  |  | 108.5395 |  |  |  |  |  |
|  |  |  |  |  | 206.6209 |  | 0.1080 |
|  |  | 154.5204 |  |  |  |  |  |
|  |  | 122.7062 |  |  |  |  |  |
|  |  | 178.3594 |  |  |  |  |  |
|  |  | 119.0244 |  |  |  |  |  |
|  |  | 167.8291 |  |  |  |  |  |
|  |  |  |  |  | 200.6893 |  | 0.1109 |
|  |  | 139.6037 |  |  |  |  |  |
|  |  | 135.6457 |  |  |  |  |  |
|  |  | 143.3859 |  |  |  |  |  |
|  |  | 206.4467 |  |  |  |  |  |
|  |  | 159.9982 |  |  |  |  |  |
|  |  |  |  |  | 202.1082 |  | 0.1071 |
|  |  | 147.0028 |  |  |  |  |  |
|  |  | 181.1608 |  |  |  |  |  |
|  |  | 133.2899 |  |  |  |  |  |
|  |  | 179.7421 |  |  |  |  |  |
|  |  | 200.5666 |  |  |  |  |  |
|  |  |  |  |  | 203.5291 |  | 0.1024 |
|  |  | 187.2866 |  |  |  |  |  |
|  |  | 161.1537 |  |  |  |  |  |
|  |  | 169.1874 |  |  |  |  |  |
|  |  | 198.2357 |  |  |  |  |  |
|  |  | 195.2773 |  |  |  |  |  |
|  |  |  |  |  | 201.6056 |  | 0.1270 |
|  |  | 193.7333 |  |  |  |  |  |
|  |  | 170.0678 |  |  |  |  |  |
|  |  | 167.9609 |  |  |  |  |  |
|  |  | 150.9824 |  |  |  |  |  |

## Summary stats

```yaml
n_records: 239
step_max: 19900
step_n: 239
train_all_r2_final: 0.847
train_all_r2_first: 0.016
train_all_r2_max: 0.89
train_all_r2_min: 0.016
train_loss_final: 150.982
train_loss_first: 402.926
train_loss_max: 402.926
train_loss_min: 108.136
train_mask_r2_final: 0.845
train_mask_r2_first: 0.016
train_mask_r2_max: 0.886
train_mask_r2_min: 0.016
train_masked_acc_final: 0.665
train_masked_acc_first: 0
train_masked_acc_max: 0.742
train_masked_acc_min: 0
train_rz_masked_acc_final: 0.125
train_rz_masked_acc_first: 0
train_rz_masked_acc_max: 0.444
train_rz_masked_acc_min: 0
train_spec_acc_final: 0.67
train_spec_acc_first: 0.001
train_spec_acc_max: 0.751
train_spec_acc_min: 0.001
train_spec_loss_final: 1.171
train_spec_loss_first: 7.51
train_spec_loss_max: 7.51
train_spec_loss_min: 0.843
train_z_acc_final: 0.333
train_z_acc_first: 0
train_z_acc_max: 0.5
train_z_acc_min: 0
train_z_loss_final: 2.935
train_z_loss_first: 7.796
train_z_loss_max: 7.796
train_z_loss_min: 2.093
val_loss_final: 201.606
val_loss_first: 253.923
val_loss_max: 253.923
val_loss_min: 200.689
val_loss_redshift_final: 3.91
val_loss_redshift_first: 4.9224
val_loss_redshift_max: 4.9224
val_loss_redshift_min: 3.8915
val_loss_spectrum_final: 2.1209
val_loss_spectrum_first: 3.0013
val_loss_spectrum_max: 3.0013
val_loss_spectrum_min: 2.1203
val_loss_total_final: 2.1275
val_loss_total_first: 3.0083
val_loss_total_max: 3.0083
val_loss_total_min: 2.1269
val_masked_spec_acc_final: 0.4188
val_masked_spec_acc_first: 0.3116
val_masked_spec_acc_max: 0.4197
val_masked_spec_acc_min: 0.3116
val_overall_acc_final: 0.42
val_overall_acc_first: 0.3079
val_overall_acc_max: 0.42
val_overall_acc_min: 0.3079
val_redshift_acc_final: 0.127
val_redshift_acc_first: 0.0386
val_redshift_acc_masked_final: 0.0494
val_redshift_acc_masked_first: 0.0363
val_redshift_acc_masked_max: 0.0503
val_redshift_acc_masked_min: 0.0064
val_redshift_acc_max: 0.127
val_redshift_acc_min: 0.0094
val_spectrum_acc_final: 0.4211
val_spectrum_acc_first: 0.3089
val_spectrum_acc_max: 0.4211
val_spectrum_acc_min: 0.3089
```

