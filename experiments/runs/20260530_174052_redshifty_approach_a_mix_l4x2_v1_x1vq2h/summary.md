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

- **run dir:** `/raid/benson/git/agentic-lensing/experiments/runs/20260530_174052_redshifty_approach_a_mix_l4x2_v1_x1vq2h`
- **repo:** redshifty
- **started:** 2026-05-30 17:40:52 UTC
- **finished:** 2026-05-31 02:42:32 UTC
- **wallclock:** 32499.9s
- **exit code:** 1
- **tags:** xlarge-scale, transformer, approach-a, track3, l4-ddp, v1-control, phase14

## Command
```
/home/benson/.venvs/redshifty/bin/python -m torch.distributed.run --standalone --nproc_per_node=2 nersc/train_transformer.py --manifest /raid/benson/data/desi_dr1_medium/manifest_mix.jsonl --tokenizer-ckpt /raid/benson/data/desi_dr1_medium/checkpoints/tokenizer_v1_large/best.pt --tokenizer-kind v1 --approach a --steps 15000 --batch-size 32 --lr 4e-4 --warmup 500 --healpix-holdout-frac 0.05 --val-every 500 --save-every 2500 --log-every 100 --num-workers 4 --amp --redshift-loss-weight 50.0 --encoder-mask-ratio 0.50 --ar-eval-batches 8 --run-name approach_a_mix_l4x2_v1 --scratch-out /raid/benson/data/desi_dr1_medium/checkpoints --wandb-mode disabled --no-push-wandb-artifact
```

## Trajectory

| approach | epoch | train_loss | train_acc | train_redshift_acc | val_loss | val_acc | val_redshift_acc |
|---|---|---|---|---|---|---|---|
|  |  | 407.2336 |  |  |  |  |  |
|  |  | 223.7785 |  |  |  |  |  |
|  |  | 245.3872 |  |  |  |  |  |
|  |  | 196.6822 |  |  |  |  |  |
|  |  | 206.3203 |  |  |  |  |  |
|  |  | 219.5909 |  |  |  |  |  |
|  |  |  |  |  | 240.0903 |  | 0.0354 |
|  |  | 202.9885 |  |  |  |  |  |
|  |  | 185.8608 |  |  |  |  |  |
|  |  | 155.9434 |  |  |  |  |  |
|  |  | 190.0499 |  |  |  |  |  |
|  |  | 180.6904 |  |  |  |  |  |
|  |  |  |  |  | 235.5647 |  | 0.0332 |
|  |  | 179.6055 |  |  |  |  |  |
|  |  | 186.1481 |  |  |  |  |  |
|  |  | 182.6949 |  |  |  |  |  |
|  |  | 156.4710 |  |  |  |  |  |
|  |  | 159.2720 |  |  |  |  |  |
|  |  |  |  |  | 204.7384 |  | 0.1060 |
|  |  | 189.7337 |  |  |  |  |  |
|  |  | 164.8019 |  |  |  |  |  |
|  |  | 207.7991 |  |  |  |  |  |
|  |  | 169.7055 |  |  |  |  |  |
|  |  | 199.8524 |  |  |  |  |  |
|  |  |  |  |  | 202.2553 |  | 0.1036 |
|  |  | 141.5400 |  |  |  |  |  |
|  |  | 136.1562 |  |  |  |  |  |
|  |  | 144.3030 |  |  |  |  |  |
|  |  | 173.1262 |  |  |  |  |  |
|  |  | 142.8011 |  |  |  |  |  |
|  |  |  |  |  | 188.3298 |  | 0.1485 |
|  |  | 158.2819 |  |  |  |  |  |
|  |  | 155.6501 |  |  |  |  |  |
|  |  | 131.6695 |  |  |  |  |  |
|  |  | 188.0462 |  |  |  |  |  |
|  |  | 106.2012 |  |  |  |  |  |
|  |  |  |  |  | 186.0968 |  | 0.1567 |
|  |  | 119.0766 |  |  |  |  |  |
|  |  | 162.3082 |  |  |  |  |  |
|  |  | 164.9340 |  |  |  |  |  |
|  |  | 180.3341 |  |  |  |  |  |
|  |  | 134.0050 |  |  |  |  |  |
|  |  |  |  |  | 187.5794 |  | 0.1383 |
|  |  | 101.9635 |  |  |  |  |  |
|  |  | 141.1240 |  |  |  |  |  |
|  |  | 149.8859 |  |  |  |  |  |
|  |  | 107.6056 |  |  |  |  |  |
|  |  | 172.1025 |  |  |  |  |  |
|  |  |  |  |  | 174.7830 |  | 0.2149 |
|  |  | 131.9167 |  |  |  |  |  |
|  |  | 132.2246 |  |  |  |  |  |
|  |  | 101.4722 |  |  |  |  |  |
|  |  | 164.0788 |  |  |  |  |  |
|  |  | 145.4768 |  |  |  |  |  |
|  |  |  |  |  | 166.5086 |  | 0.2396 |
|  |  | 108.9565 |  |  |  |  |  |
|  |  | 117.6995 |  |  |  |  |  |
|  |  | 156.9431 |  |  |  |  |  |
|  |  | 143.3566 |  |  |  |  |  |
|  |  | 127.8595 |  |  |  |  |  |
|  |  |  |  |  | 159.7874 |  | 0.2778 |
|  |  | 167.0235 |  |  |  |  |  |
|  |  | 141.7493 |  |  |  |  |  |
|  |  | 132.6996 |  |  |  |  |  |
|  |  | 157.7419 |  |  |  |  |  |
|  |  | 74.6114 |  |  |  |  |  |
|  |  |  |  |  | 159.2064 |  | 0.2670 |
|  |  | 103.6040 |  |  |  |  |  |
|  |  | 169.7139 |  |  |  |  |  |
|  |  | 43.9290 |  |  |  |  |  |
|  |  | 139.8171 |  |  |  |  |  |
|  |  | 134.6245 |  |  |  |  |  |
|  |  |  |  |  | 142.6098 |  | 0.3451 |
|  |  | 110.6263 |  |  |  |  |  |
|  |  | 122.2535 |  |  |  |  |  |
|  |  | 116.5711 |  |  |  |  |  |
|  |  | 97.4030 |  |  |  |  |  |
|  |  | 93.1240 |  |  |  |  |  |
|  |  |  |  |  | 151.3662 |  | 0.2863 |
|  |  | 99.4527 |  |  |  |  |  |
|  |  | 140.2205 |  |  |  |  |  |
|  |  | 134.7057 |  |  |  |  |  |
|  |  | 127.6090 |  |  |  |  |  |
|  |  | 121.7850 |  |  |  |  |  |
|  |  |  |  |  | 142.8605 |  | 0.3562 |
|  |  | 131.3522 |  |  |  |  |  |
|  |  | 113.0360 |  |  |  |  |  |
|  |  | 124.5545 |  |  |  |  |  |
|  |  | 170.9982 |  |  |  |  |  |
|  |  | 144.3539 |  |  |  |  |  |
|  |  |  |  |  | 140.1986 |  | 0.3927 |
|  |  | 128.4357 |  |  |  |  |  |
|  |  | 149.5244 |  |  |  |  |  |
|  |  | 75.3150 |  |  |  |  |  |
|  |  | 60.5412 |  |  |  |  |  |
|  |  | 120.5254 |  |  |  |  |  |
|  |  |  |  |  | 131.9185 |  | 0.4122 |
|  |  | 133.6005 |  |  |  |  |  |
|  |  | 134.1711 |  |  |  |  |  |
|  |  | 111.7582 |  |  |  |  |  |
|  |  | 127.8239 |  |  |  |  |  |
|  |  | 93.8972 |  |  |  |  |  |
|  |  |  |  |  | 130.4009 |  | 0.4232 |
|  |  | 134.3272 |  |  |  |  |  |
|  |  | 103.2372 |  |  |  |  |  |
|  |  | 182.1113 |  |  |  |  |  |
|  |  | 95.4384 |  |  |  |  |  |
|  |  | 125.9333 |  |  |  |  |  |
|  |  |  |  |  | 126.2737 |  | 0.4349 |
|  |  | 107.6606 |  |  |  |  |  |
|  |  | 86.2259 |  |  |  |  |  |
|  |  | 69.6817 |  |  |  |  |  |
|  |  | 117.2871 |  |  |  |  |  |
|  |  | 113.4169 |  |  |  |  |  |
|  |  |  |  |  | 121.1046 |  | 0.4588 |
|  |  | 78.5259 |  |  |  |  |  |
|  |  | 71.3393 |  |  |  |  |  |
|  |  | 117.1617 |  |  |  |  |  |
|  |  | 116.3004 |  |  |  |  |  |
|  |  | 89.9418 |  |  |  |  |  |
|  |  |  |  |  | 119.7564 |  | 0.4743 |
|  |  | 93.1663 |  |  |  |  |  |
|  |  | 88.3127 |  |  |  |  |  |
|  |  | 98.9817 |  |  |  |  |  |
|  |  | 113.3126 |  |  |  |  |  |
|  |  | 127.1523 |  |  |  |  |  |
|  |  |  |  |  | 114.2164 |  | 0.5196 |
|  |  | 155.7015 |  |  |  |  |  |
|  |  | 110.2156 |  |  |  |  |  |
|  |  | 120.3521 |  |  |  |  |  |
|  |  | 92.6333 |  |  |  |  |  |
|  |  | 100.9921 |  |  |  |  |  |
|  |  |  |  |  | 116.8451 |  | 0.4935 |
|  |  | 83.7779 |  |  |  |  |  |
|  |  | 95.1521 |  |  |  |  |  |
|  |  | 90.2626 |  |  |  |  |  |
|  |  | 101.5456 |  |  |  |  |  |
|  |  | 157.5860 |  |  |  |  |  |
|  |  |  |  |  | 119.2087 |  | 0.4741 |
|  |  | 70.7458 |  |  |  |  |  |
|  |  | 86.3211 |  |  |  |  |  |
|  |  | 108.4847 |  |  |  |  |  |
|  |  | 99.6740 |  |  |  |  |  |
|  |  | 45.3016 |  |  |  |  |  |
|  |  |  |  |  | 116.0113 |  | 0.4957 |
|  |  | 140.6614 |  |  |  |  |  |
|  |  | 103.6211 |  |  |  |  |  |
|  |  | 70.7120 |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
|  |  | 110.3011 |  |  |  |  |  |
|  |  |  |  |  | 119.2918 |  | 0.4902 |
|  |  | 106.9881 |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  | 0.0000 |
|  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  | 0.0000 |
|  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  | 0.0000 |
|  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  | 0.0000 |
|  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |

## Summary stats

```yaml
n_records: 179
step_max: 14900
step_n: 179
train_all_r2_final: 0.617
train_all_r2_first: 0
train_all_r2_max: 0.867
train_all_r2_min: 0
train_loss_final: 106.988
train_loss_first: 407.234
train_loss_max: 407.234
train_loss_min: 43.929
train_mask_r2_final: 0.625
train_mask_r2_first: 0
train_mask_r2_max: 0.87
train_mask_r2_min: 0
train_masked_acc_final: 0
train_masked_acc_first: 0
train_masked_acc_max: 0.721
train_masked_acc_min: 0
train_rz_masked_acc_final: 0
train_rz_masked_acc_first: 0
train_rz_masked_acc_max: 0.455
train_rz_masked_acc_min: 0
train_spec_acc_final: 0
train_spec_acc_first: 0
train_spec_acc_max: 0.715
train_spec_acc_min: 0
train_spec_loss_final: 2.924
train_spec_loss_first: 7.748
train_spec_loss_max: 7.748
train_spec_loss_min: 1.013
train_z_acc_final: 0
train_z_acc_first: 0
train_z_acc_max: 0.786
train_z_acc_min: 0
train_z_loss_final: 2.042
train_z_loss_first: 7.882
train_z_loss_max: 7.882
train_z_loss_min: 0.834
val_loss_final: 119.292
val_loss_first: 240.09
val_loss_max: 240.09
val_loss_min: 114.216
val_loss_redshift_final: 2.2632
val_loss_redshift_first: 4.6537
val_loss_redshift_max: 4.6537
val_loss_redshift_min: 2.1658
val_loss_spectrum_final: 3.7562
val_loss_spectrum_first: 2.8742
val_loss_spectrum_max: 3.7562
val_loss_spectrum_min: 2.2133
val_loss_total_final: 3.7507
val_loss_total_first: 2.8807
val_loss_total_max: 3.7507
val_loss_total_min: 2.2141
val_masked_spec_acc_final: 0
val_masked_spec_acc_first: 0.324
val_masked_spec_acc_max: 0.4011
val_masked_spec_acc_min: 0
val_overall_acc_final: 0
val_overall_acc_first: 0.3198
val_overall_acc_max: 0.4034
val_overall_acc_min: 0
val_redshift_acc_final: 0
val_redshift_acc_first: 0.0354
val_redshift_acc_masked_final: 0
val_redshift_acc_masked_first: 0.0322
val_redshift_acc_masked_max: 0.075
val_redshift_acc_masked_min: 0
val_redshift_acc_max: 0.5196
val_redshift_acc_min: 0
val_spectrum_acc_final: 0
val_spectrum_acc_first: 0.3208
val_spectrum_acc_max: 0.4033
val_spectrum_acc_min: 0
```

