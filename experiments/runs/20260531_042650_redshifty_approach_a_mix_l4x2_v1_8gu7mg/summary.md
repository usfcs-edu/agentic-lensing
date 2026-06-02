# redshifty_approach_a_mix_l4x2_v1

_Clean L4 reproduction of the Approach-A ignition on the 4-way data mix,
correctly mapped to BOTH L4 GPUs (indices 8,9 under PCI_BUS_ID). The prior
_mix run accidentally trained on a single A16 due to CUDA's FASTEST_FIRST
device reordering, so its 7h51m wallclock and hardware are wrong. DDP across
2 L4s via torchrun, effective batch 64 (32/GPU x 2), extended to 15k steps
(the mix-run trajectory was still descending at 10k; 15k comfortably passes
the NERSC reference's ~15k peak region). --ar-eval-batches kept at 8 (matches
the original mix run; n~226). NOTE: --ar-eval-batches 32 is DDP-incompatible —
the rank-0-only AR eval then runs >10min, and the other rank's DDP buffer
broadcast hits NCCL's default watchdog timeout and aborts the job (observed
SIGABRT on rank 1). train_transformer.py now also raises the PG timeout to
30min as a safety net. Frozen V1 tokenizer (val_recon=1.38). This is the
V1 CONTROL arm of the V1-vs-V2 tokenizer ablation.
_

- **run dir:** `/raid/benson/git/agentic-lensing/experiments/runs/20260531_042650_redshifty_approach_a_mix_l4x2_v1_8gu7mg`
- **repo:** redshifty
- **started:** 2026-05-31 04:26:50 UTC
- **finished:** 2026-05-31 13:24:24 UTC
- **wallclock:** 32254.8s
- **exit code:** 0
- **tags:** xlarge-scale, transformer, approach-a, track3, l4-ddp, v1-control, phase14

## Command
```
/home/benson/.venvs/redshifty/bin/python -m torch.distributed.run --standalone --nproc_per_node=2 nersc/train_transformer.py --manifest /raid/benson/data/desi_dr1_medium/manifest_mix.jsonl --tokenizer-ckpt /raid/benson/data/desi_dr1_medium/checkpoints/tokenizer_v1_large/best.pt --tokenizer-kind v1 --approach a --steps 15000 --batch-size 32 --lr 4e-4 --warmup 500 --healpix-holdout-frac 0.05 --val-every 500 --save-every 2500 --log-every 100 --num-workers 4 --amp --redshift-loss-weight 50.0 --encoder-mask-ratio 0.50 --ar-eval-batches 8 --run-name approach_a_mix_l4x2_v1 --scratch-out /raid/benson/data/desi_dr1_medium/checkpoints --wandb-mode disabled --no-push-wandb-artifact
```

## Trajectory

| approach | epoch | train_loss | train_acc | train_redshift_acc | val_loss | val_acc | val_redshift_acc |
|---|---|---|---|---|---|---|---|
|  |  | 411.0748 |  |  |  |  |  |
|  |  | 228.8124 |  |  |  |  |  |
|  |  | 245.0038 |  |  |  |  |  |
|  |  | 190.1321 |  |  |  |  |  |
|  |  | 210.9117 |  |  |  |  |  |
|  |  | 216.3074 |  |  |  |  |  |
|  |  |  |  |  | 242.8551 |  | 0.0279 |
|  |  | 209.5366 |  |  |  |  |  |
|  |  | 186.5395 |  |  |  |  |  |
|  |  | 156.2518 |  |  |  |  |  |
|  |  | 193.3444 |  |  |  |  |  |
|  |  | 178.3901 |  |  |  |  |  |
|  |  |  |  |  | 229.0179 |  | 0.0316 |
|  |  | 164.2629 |  |  |  |  |  |
|  |  | 168.0384 |  |  |  |  |  |
|  |  | 192.4283 |  |  |  |  |  |
|  |  | 150.3568 |  |  |  |  |  |
|  |  | 138.7499 |  |  |  |  |  |
|  |  |  |  |  | 174.0977 |  | 0.2510 |
|  |  | 187.3641 |  |  |  |  |  |
|  |  | 112.3532 |  |  |  |  |  |
|  |  | 110.8532 |  |  |  |  |  |
|  |  | 188.5788 |  |  |  |  |  |
|  |  | 119.3651 |  |  |  |  |  |
|  |  |  |  |  | 142.8736 |  | 0.3663 |
|  |  | 109.1378 |  |  |  |  |  |
|  |  | 95.6499 |  |  |  |  |  |
|  |  | 89.2205 |  |  |  |  |  |
|  |  | 206.5805 |  |  |  |  |  |
|  |  | 93.1596 |  |  |  |  |  |
|  |  |  |  |  | 127.9829 |  | 0.4659 |
|  |  | 93.0298 |  |  |  |  |  |
|  |  | 104.3881 |  |  |  |  |  |
|  |  | 126.1188 |  |  |  |  |  |
|  |  | 127.2509 |  |  |  |  |  |
|  |  | 94.8247 |  |  |  |  |  |
|  |  |  |  |  | 120.9418 |  | 0.4936 |
|  |  | 124.6338 |  |  |  |  |  |
|  |  | 62.7466 |  |  |  |  |  |
|  |  | 121.5649 |  |  |  |  |  |
|  |  | 54.6064 |  |  |  |  |  |
|  |  | 81.1698 |  |  |  |  |  |
|  |  |  |  |  | 119.2146 |  | 0.5008 |
|  |  | 93.6398 |  |  |  |  |  |
|  |  | 92.3963 |  |  |  |  |  |
|  |  | 123.2777 |  |  |  |  |  |
|  |  | 117.2434 |  |  |  |  |  |
|  |  | 101.8710 |  |  |  |  |  |
|  |  |  |  |  | 115.9322 |  | 0.5063 |
|  |  | 137.3177 |  |  |  |  |  |
|  |  | 129.8741 |  |  |  |  |  |
|  |  | 98.9078 |  |  |  |  |  |
|  |  | 107.7795 |  |  |  |  |  |
|  |  | 84.7162 |  |  |  |  |  |
|  |  |  |  |  | 122.7972 |  | 0.4695 |
|  |  | 97.6539 |  |  |  |  |  |
|  |  | 103.9351 |  |  |  |  |  |
|  |  | 146.6467 |  |  |  |  |  |
|  |  | 103.1612 |  |  |  |  |  |
|  |  | 123.0785 |  |  |  |  |  |
|  |  |  |  |  | 120.0727 |  | 0.4923 |
|  |  | 87.5830 |  |  |  |  |  |
|  |  | 92.5373 |  |  |  |  |  |
|  |  | 122.9194 |  |  |  |  |  |
|  |  | 104.0380 |  |  |  |  |  |
|  |  | 81.2166 |  |  |  |  |  |
|  |  |  |  |  | 115.6366 |  | 0.5039 |
|  |  | 62.3964 |  |  |  |  |  |
|  |  | 155.1500 |  |  |  |  |  |
|  |  | 56.6252 |  |  |  |  |  |
|  |  | 121.1941 |  |  |  |  |  |
|  |  | 119.9829 |  |  |  |  |  |
|  |  |  |  |  | 113.4883 |  | 0.5217 |
|  |  | 89.2189 |  |  |  |  |  |
|  |  | 106.6347 |  |  |  |  |  |
|  |  | 93.0493 |  |  |  |  |  |
|  |  | 52.0219 |  |  |  |  |  |
|  |  | 113.2829 |  |  |  |  |  |
|  |  |  |  |  | 116.1052 |  | 0.5175 |
|  |  | 92.0866 |  |  |  |  |  |
|  |  | 119.6607 |  |  |  |  |  |
|  |  | 80.5523 |  |  |  |  |  |
|  |  | 140.9321 |  |  |  |  |  |
|  |  | 127.1155 |  |  |  |  |  |
|  |  |  |  |  | 125.5765 |  | 0.4937 |
|  |  | 85.2146 |  |  |  |  |  |
|  |  | 116.6157 |  |  |  |  |  |
|  |  | 72.3995 |  |  |  |  |  |
|  |  | 128.9856 |  |  |  |  |  |
|  |  | 117.7087 |  |  |  |  |  |
|  |  |  |  |  | 112.7729 |  | 0.5087 |
|  |  | 104.8263 |  |  |  |  |  |
|  |  | 117.7076 |  |  |  |  |  |
|  |  | 91.3543 |  |  |  |  |  |
|  |  | 74.8273 |  |  |  |  |  |
|  |  | 108.9304 |  |  |  |  |  |
|  |  |  |  |  | 109.1903 |  | 0.5358 |
|  |  | 75.9191 |  |  |  |  |  |
|  |  | 107.4138 |  |  |  |  |  |
|  |  | 91.5358 |  |  |  |  |  |
|  |  | 66.8771 |  |  |  |  |  |
|  |  | 70.0334 |  |  |  |  |  |
|  |  |  |  |  | 104.5567 |  | 0.5543 |
|  |  | 94.2065 |  |  |  |  |  |
|  |  | 76.6286 |  |  |  |  |  |
|  |  | 107.3999 |  |  |  |  |  |
|  |  | 97.2057 |  |  |  |  |  |
|  |  | 80.1711 |  |  |  |  |  |
|  |  |  |  |  | 118.3975 |  | 0.4898 |
|  |  | 98.7576 |  |  |  |  |  |
|  |  | 160.9814 |  |  |  |  |  |
|  |  | 106.5790 |  |  |  |  |  |
|  |  | 76.1732 |  |  |  |  |  |
|  |  | 60.6947 |  |  |  |  |  |
|  |  |  |  |  | 110.7943 |  | 0.5284 |
|  |  | 64.0433 |  |  |  |  |  |
|  |  | 93.1996 |  |  |  |  |  |
|  |  | 90.8599 |  |  |  |  |  |
|  |  | 112.5915 |  |  |  |  |  |
|  |  | 149.8440 |  |  |  |  |  |
|  |  |  |  |  | 114.2000 |  | 0.5206 |
|  |  | 109.7268 |  |  |  |  |  |
|  |  | 94.1523 |  |  |  |  |  |
|  |  | 66.6871 |  |  |  |  |  |
|  |  | 140.4324 |  |  |  |  |  |
|  |  | 105.8157 |  |  |  |  |  |
|  |  |  |  |  | 112.2837 |  | 0.5208 |
|  |  | 36.2773 |  |  |  |  |  |
|  |  | 129.7425 |  |  |  |  |  |
|  |  | 74.4665 |  |  |  |  |  |
|  |  | 106.7165 |  |  |  |  |  |
|  |  | 106.1064 |  |  |  |  |  |
|  |  |  |  |  | 111.2774 |  | 0.5206 |
|  |  | 120.6571 |  |  |  |  |  |
|  |  | 128.8235 |  |  |  |  |  |
|  |  | 85.9714 |  |  |  |  |  |
|  |  | 73.5072 |  |  |  |  |  |
|  |  | 115.8146 |  |  |  |  |  |
|  |  |  |  |  | 114.0158 |  | 0.4992 |
|  |  | 78.8062 |  |  |  |  |  |
|  |  | 88.7091 |  |  |  |  |  |
|  |  | 80.6790 |  |  |  |  |  |
|  |  | 75.9990 |  |  |  |  |  |
|  |  | 93.3967 |  |  |  |  |  |
|  |  |  |  |  | 107.5960 |  | 0.5308 |
|  |  | 103.8986 |  |  |  |  |  |
|  |  | 112.7064 |  |  |  |  |  |
|  |  | 59.0804 |  |  |  |  |  |
|  |  | 142.3179 |  |  |  |  |  |
|  |  | 80.1484 |  |  |  |  |  |
|  |  |  |  |  | 106.4647 |  | 0.5378 |
|  |  | 65.7560 |  |  |  |  |  |
|  |  | 67.6889 |  |  |  |  |  |
|  |  | 97.0403 |  |  |  |  |  |
|  |  | 96.5962 |  |  |  |  |  |
|  |  | 145.1008 |  |  |  |  |  |
|  |  |  |  |  | 104.2089 |  | 0.5448 |
|  |  | 70.4875 |  |  |  |  |  |
|  |  | 108.7941 |  |  |  |  |  |
|  |  | 91.0175 |  |  |  |  |  |
|  |  | 86.3133 |  |  |  |  |  |
|  |  | 59.0163 |  |  |  |  |  |
|  |  |  |  |  | 109.0461 |  | 0.5137 |
|  |  | 114.9850 |  |  |  |  |  |
|  |  | 84.2722 |  |  |  |  |  |
|  |  | 48.6245 |  |  |  |  |  |
|  |  | 111.0488 |  |  |  |  |  |
|  |  | 83.6644 |  |  |  |  |  |
|  |  |  |  |  | 110.3402 |  | 0.5185 |
|  |  | 87.6012 |  |  |  |  |  |
|  |  | 87.7078 |  |  |  |  |  |
|  |  | 92.6235 |  |  |  |  |  |
|  |  | 117.4616 |  |  |  |  |  |
|  |  | 74.0313 |  |  |  |  |  |
|  |  |  |  |  | 105.2675 |  | 0.5287 |
|  |  | 112.0296 |  |  |  |  |  |
|  |  | 86.4511 |  |  |  |  |  |
|  |  | 75.4033 |  |  |  |  |  |
|  |  | 136.4449 |  |  |  |  |  |

## Summary stats

```yaml
n_records: 179
step_max: 14900
step_n: 179
train_all_r2_final: 0.818
train_all_r2_first: 0
train_all_r2_max: 0.877
train_all_r2_min: 0
train_loss_final: 136.445
train_loss_first: 411.075
train_loss_max: 411.075
train_loss_min: 36.2773
train_mask_r2_final: 0.818
train_mask_r2_first: 0
train_mask_r2_max: 0.881
train_mask_r2_min: 0
train_masked_acc_final: 0.597
train_masked_acc_first: 0
train_masked_acc_max: 0.735
train_masked_acc_min: 0
train_rz_masked_acc_final: 0.067
train_rz_masked_acc_first: 0
train_rz_masked_acc_max: 0.455
train_rz_masked_acc_min: 0
train_spec_acc_final: 0.599
train_spec_acc_first: 0
train_spec_acc_max: 0.736
train_spec_acc_min: 0
train_spec_loss_final: 1.39
train_spec_loss_first: 8.047
train_spec_loss_max: 8.047
train_spec_loss_min: 0.935
train_z_acc_final: 0.391
train_z_acc_first: 0
train_z_acc_max: 0.786
train_z_acc_min: 0
train_z_loss_final: 2.648
train_z_loss_first: 7.944
train_z_loss_max: 7.944
train_z_loss_min: 0.681
val_loss_final: 105.267
val_loss_first: 242.855
val_loss_max: 242.855
val_loss_min: 104.209
val_loss_redshift_final: 2.0227
val_loss_redshift_first: 4.7088
val_loss_redshift_max: 4.7088
val_loss_redshift_min: 2.0015
val_loss_spectrum_final: 2.0929
val_loss_spectrum_first: 2.8584
val_loss_spectrum_max: 2.8584
val_loss_spectrum_min: 2.0929
val_loss_total_final: 2.0926
val_loss_total_first: 2.8651
val_loss_total_max: 2.8651
val_loss_total_min: 2.0926
val_masked_spec_acc_final: 0.4246
val_masked_spec_acc_first: 0.3222
val_masked_spec_acc_max: 0.4246
val_masked_spec_acc_min: 0.3222
val_overall_acc_final: 0.4262
val_overall_acc_first: 0.3178
val_overall_acc_max: 0.4262
val_overall_acc_min: 0.3178
val_redshift_acc_final: 0.5287
val_redshift_acc_first: 0.0279
val_redshift_acc_masked_final: 0.0287
val_redshift_acc_masked_first: 0.0202
val_redshift_acc_masked_max: 0.0598
val_redshift_acc_masked_min: 0.0084
val_redshift_acc_max: 0.5543
val_redshift_acc_min: 0.0279
val_spectrum_acc_final: 0.4258
val_spectrum_acc_first: 0.3188
val_spectrum_acc_max: 0.4258
val_spectrum_acc_min: 0.3188
```

