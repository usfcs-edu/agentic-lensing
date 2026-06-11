# redshifty_approach_a_phase10_mix_mps

_Apple Silicon / MPS reproduction of the redshift-ignition run. Byte-identical to
redshifty_approach_a_phase10_mix.yaml EXCEPT: --num-workers 0 (macOS spawn + MPS),
MPS fallback/unbuffered env, and no CUDA_VISIBLE_DEVICES. Driven through the
unmodified tools/spectrumfm/exp_run.py harness — which also validates the Track-2
tooling on Apple Silicon. Requires the one-time `sudo ln -s _raid /raid` so the
absolute paths below + manifest_mix.jsonl resolve verbatim, and
~/.venvs/redshifty -> the port .venv (created by sync_from_phoenix.sh links).
_

- **run dir:** `/Users/benson/sync-git/sync-lens/agentic-lensing/reproductions/redshifty/apple-silicon/data/runs/20260603_111414_redshifty_approach_a_phase10_mix_mps_j9sukl`
- **repo:** redshifty
- **started:** 2026-06-03 11:14:14 UTC
- **finished:** 2026-06-03 19:59:58 UTC
- **wallclock:** 31543.6s
- **exit code:** 0
- **tags:** apple-silicon, mps, transformer, approach-a, ignition, data-mix

## Command
```
/Users/benson/.venvs/redshifty/bin/python nersc/train_transformer.py --manifest /Users/benson/sync-git/sync-lens/agentic-lensing/reproductions/redshifty/apple-silicon/_raid/benson/data/desi_dr1_medium/manifest_mix_local.jsonl --tokenizer-ckpt /Users/benson/sync-git/sync-lens/agentic-lensing/reproductions/redshifty/apple-silicon/_raid/benson/data/desi_dr1_medium/checkpoints/tokenizer_v1_large/best.pt --tokenizer-kind v1 --approach a --steps 10000 --batch-size 32 --lr 4e-4 --warmup 500 --healpix-holdout-frac 0.05 --val-every 500 --save-every 2500 --log-every 100 --num-workers 0 --amp --redshift-loss-weight 50.0 --encoder-mask-ratio 0.50 --ar-eval-batches 8 --run-name approach_a_phase10_mix_mps --scratch-out /Users/benson/sync-git/sync-lens/agentic-lensing/reproductions/redshifty/apple-silicon/_raid/benson/data/desi_dr1_medium/checkpoints --wandb-mode disabled --no-push-wandb-artifact
```

## Trajectory

| approach | epoch | train_loss | train_acc | train_redshift_acc | val_loss | val_acc | val_redshift_acc |
|---|---|---|---|---|---|---|---|
|  |  | 407.9583 |  |  |  |  |  |
|  |  | 244.2167 |  |  |  |  |  |
|  |  | 182.8760 |  |  |  |  |  |
|  |  | 234.5443 |  |  |  |  |  |
|  |  | 223.5376 |  |  |  |  |  |
|  |  | 212.6097 |  |  |  |  |  |
|  |  |  |  |  | 258.2269 |  | 0.0122 |
|  |  | 190.2021 |  |  |  |  |  |
|  |  | 204.3797 |  |  |  |  |  |
|  |  | 222.8543 |  |  |  |  |  |
|  |  | 216.6581 |  |  |  |  |  |
|  |  | 218.6298 |  |  |  |  |  |
|  |  |  |  |  | 245.1665 |  | 0.0101 |
|  |  | 191.9409 |  |  |  |  |  |
|  |  | 243.0613 |  |  |  |  |  |
|  |  | 215.8642 |  |  |  |  |  |
|  |  | 166.4075 |  |  |  |  |  |
|  |  | 222.5812 |  |  |  |  |  |
|  |  |  |  |  | 243.0036 |  | 0.0142 |
|  |  | 225.1306 |  |  |  |  |  |
|  |  | 200.8359 |  |  |  |  |  |
|  |  | 181.9149 |  |  |  |  |  |
|  |  | 208.6927 |  |  |  |  |  |
|  |  | 192.1845 |  |  |  |  |  |
|  |  |  |  |  | 241.3838 |  | 0.0158 |
|  |  | 202.9610 |  |  |  |  |  |
|  |  | 207.6850 |  |  |  |  |  |
|  |  | 192.3133 |  |  |  |  |  |
|  |  | 181.1425 |  |  |  |  |  |
|  |  | 184.0127 |  |  |  |  |  |
|  |  |  |  |  | 238.8075 |  | 0.0300 |
|  |  | 170.1098 |  |  |  |  |  |
|  |  | 194.1762 |  |  |  |  |  |
|  |  | 197.7179 |  |  |  |  |  |
|  |  | 155.9395 |  |  |  |  |  |
|  |  | 225.4723 |  |  |  |  |  |
|  |  |  |  |  | 242.2871 |  | 0.0132 |
|  |  | 188.3717 |  |  |  |  |  |
|  |  | 186.4217 |  |  |  |  |  |
|  |  | 205.3751 |  |  |  |  |  |
|  |  | 220.4630 |  |  |  |  |  |
|  |  | 187.0714 |  |  |  |  |  |
|  |  |  |  |  | 234.1379 |  | 0.0254 |
|  |  | 170.4920 |  |  |  |  |  |
|  |  | 177.6005 |  |  |  |  |  |
|  |  | 174.5892 |  |  |  |  |  |
|  |  | 176.8258 |  |  |  |  |  |
|  |  | 185.7773 |  |  |  |  |  |
|  |  |  |  |  | 228.3574 |  | 0.0403 |
|  |  | 176.3279 |  |  |  |  |  |
|  |  | 171.8222 |  |  |  |  |  |
|  |  | 178.5152 |  |  |  |  |  |
|  |  | 194.0333 |  |  |  |  |  |
|  |  | 219.1338 |  |  |  |  |  |
|  |  |  |  |  | 226.0816 |  | 0.0326 |
|  |  | 205.9700 |  |  |  |  |  |
|  |  | 187.3563 |  |  |  |  |  |
|  |  | 210.5403 |  |  |  |  |  |
|  |  | 193.5832 |  |  |  |  |  |
|  |  | 183.3453 |  |  |  |  |  |
|  |  |  |  |  | 227.2219 |  | 0.0382 |
|  |  | 185.0905 |  |  |  |  |  |
|  |  | 194.7482 |  |  |  |  |  |
|  |  | 203.6684 |  |  |  |  |  |
|  |  | 142.6281 |  |  |  |  |  |
|  |  | 186.4290 |  |  |  |  |  |
|  |  |  |  |  | 231.6894 |  | 0.0200 |
|  |  | 169.9066 |  |  |  |  |  |
|  |  | 141.0285 |  |  |  |  |  |
|  |  | 191.2814 |  |  |  |  |  |
|  |  | 189.2222 |  |  |  |  |  |
|  |  | 179.3997 |  |  |  |  |  |
|  |  |  |  |  | 220.0510 |  | 0.0455 |
|  |  | 173.8620 |  |  |  |  |  |
|  |  | 183.4070 |  |  |  |  |  |
|  |  | 134.6687 |  |  |  |  |  |
|  |  | 188.3145 |  |  |  |  |  |
|  |  | 195.7569 |  |  |  |  |  |
|  |  |  |  |  | 225.9336 |  | 0.0418 |
|  |  | 208.4065 |  |  |  |  |  |
|  |  | 152.1782 |  |  |  |  |  |
|  |  | 164.1935 |  |  |  |  |  |
|  |  | 156.9109 |  |  |  |  |  |
|  |  | 174.5417 |  |  |  |  |  |
|  |  |  |  |  | 217.7148 |  | 0.0710 |
|  |  | 177.2014 |  |  |  |  |  |
|  |  | 183.9410 |  |  |  |  |  |
|  |  | 192.3836 |  |  |  |  |  |
|  |  | 184.1050 |  |  |  |  |  |
|  |  | 168.2910 |  |  |  |  |  |
|  |  |  |  |  | 215.8368 |  | 0.0645 |
|  |  | 161.5747 |  |  |  |  |  |
|  |  | 214.2033 |  |  |  |  |  |
|  |  | 180.1682 |  |  |  |  |  |
|  |  | 178.9525 |  |  |  |  |  |
|  |  | 145.2353 |  |  |  |  |  |
|  |  |  |  |  | 218.5367 |  | 0.0459 |
|  |  | 174.7551 |  |  |  |  |  |
|  |  | 174.3000 |  |  |  |  |  |
|  |  | 179.6593 |  |  |  |  |  |
|  |  | 144.8144 |  |  |  |  |  |
|  |  | 221.5490 |  |  |  |  |  |
|  |  |  |  |  | 213.8273 |  | 0.0504 |
|  |  | 144.0027 |  |  |  |  |  |
|  |  | 112.8929 |  |  |  |  |  |
|  |  | 134.8555 |  |  |  |  |  |
|  |  | 172.4371 |  |  |  |  |  |
|  |  | 177.4109 |  |  |  |  |  |
|  |  |  |  |  | 211.2756 |  | 0.0535 |
|  |  | 183.5204 |  |  |  |  |  |
|  |  | 166.5432 |  |  |  |  |  |
|  |  | 218.9786 |  |  |  |  |  |
|  |  | 133.5050 |  |  |  |  |  |
|  |  | 170.1536 |  |  |  |  |  |
|  |  |  |  |  | 210.7992 |  | 0.0788 |
|  |  | 140.4141 |  |  |  |  |  |
|  |  | 180.0431 |  |  |  |  |  |
|  |  | 136.3708 |  |  |  |  |  |
|  |  | 141.7441 |  |  |  |  |  |

## Summary stats

```yaml
n_records: 119
step_max: 9900
step_n: 119
train_all_r2_final: 0.83
train_all_r2_first: 0
train_all_r2_max: 0.855
train_all_r2_min: 0
train_loss_final: 141.744
train_loss_first: 407.958
train_loss_max: 407.958
train_loss_min: 112.893
train_mask_r2_final: 0.827
train_mask_r2_first: 0
train_mask_r2_max: 0.858
train_mask_r2_min: 0
train_masked_acc_final: 0.627
train_masked_acc_first: 0
train_masked_acc_max: 0.697
train_masked_acc_min: 0
train_rz_masked_acc_final: 0
train_rz_masked_acc_first: 0
train_rz_masked_acc_max: 0.4
train_rz_masked_acc_min: 0
train_spec_acc_final: 0.635
train_spec_acc_first: 0
train_spec_acc_max: 0.697
train_spec_acc_min: 0
train_spec_loss_final: 1.3
train_spec_loss_first: 7.831
train_spec_loss_max: 7.831
train_spec_loss_min: 1.103
train_z_acc_final: 0.222
train_z_acc_first: 0
train_z_acc_max: 0.45
train_z_acc_min: 0
train_z_loss_final: 2.751
train_z_loss_first: 7.893
train_z_loss_max: 7.893
train_z_loss_min: 2.182
val_loss_final: 210.799
val_loss_first: 258.227
val_loss_max: 258.227
val_loss_min: 210.799
val_loss_redshift_final: 4.087
val_loss_redshift_first: 5.0065
val_loss_redshift_max: 5.0065
val_loss_redshift_min: 4.087
val_loss_spectrum_final: 2.2841
val_loss_spectrum_first: 3.1562
val_loss_spectrum_max: 3.1562
val_loss_spectrum_min: 2.2841
val_loss_total_final: 2.2907
val_loss_total_first: 3.163
val_loss_total_max: 3.163
val_loss_total_min: 2.2907
val_masked_spec_acc_final: 0.3923
val_masked_spec_acc_first: 0.3051
val_masked_spec_acc_max: 0.3923
val_masked_spec_acc_min: 0.3051
val_overall_acc_final: 0.3928
val_overall_acc_first: 0.302
val_overall_acc_max: 0.3928
val_overall_acc_min: 0.302
val_redshift_acc_final: 0.0788
val_redshift_acc_first: 0.0122
val_redshift_acc_masked_final: 0.056
val_redshift_acc_masked_first: 0.0124
val_redshift_acc_masked_max: 0.056
val_redshift_acc_masked_min: 0.0067
val_redshift_acc_max: 0.0788
val_redshift_acc_min: 0.0101
val_spectrum_acc_final: 0.3939
val_spectrum_acc_first: 0.303
val_spectrum_acc_max: 0.3939
val_spectrum_acc_min: 0.303
```

