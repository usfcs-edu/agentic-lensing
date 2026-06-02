# redshifty_approach_a_mix_l4x2_v1

_Clean L4 reproduction of the Approach-A ignition on the 4-way data mix,
correctly mapped to BOTH L4 GPUs (indices 8,9 under PCI_BUS_ID). The prior
_mix run accidentally trained on a single A16 due to CUDA's FASTEST_FIRST
device reordering, so its 7h51m wallclock and hardware are wrong. DDP across
2 L4s via torchrun, effective batch 64 (32/GPU x 2), extended to 15k steps
(the mix-run trajectory was still descending at 10k; 15k comfortably passes
the NERSC reference's ~15k peak region). --ar-eval-batches bumped to 32
(n~758 vs 226) for a
sharper AR z_acc signal. Frozen V1 tokenizer (val_recon=1.38). This is the
V1 CONTROL arm of the V1-vs-V2 tokenizer ablation.
_

- **run dir:** `/raid/benson/git/agentic-lensing/experiments/runs/20260530_165752_redshifty_approach_a_mix_l4x2_v1_urneae`
- **repo:** redshifty
- **started:** 2026-05-30 16:57:52 UTC
- **finished:** 2026-05-30 17:24:05 UTC
- **wallclock:** 1573.1s
- **exit code:** 1
- **tags:** xlarge-scale, transformer, approach-a, track3, l4-ddp, v1-control, phase14

## Command
```
/home/benson/.venvs/redshifty/bin/python -m torch.distributed.run --standalone --nproc_per_node=2 nersc/train_transformer.py --manifest /raid/benson/data/desi_dr1_medium/manifest_mix.jsonl --tokenizer-ckpt /raid/benson/data/desi_dr1_medium/checkpoints/tokenizer_v1_large/best.pt --tokenizer-kind v1 --approach a --steps 15000 --batch-size 32 --lr 4e-4 --warmup 500 --healpix-holdout-frac 0.05 --val-every 500 --save-every 2500 --log-every 100 --num-workers 4 --amp --redshift-loss-weight 50.0 --encoder-mask-ratio 0.50 --ar-eval-batches 32 --run-name approach_a_mix_l4x2_v1 --scratch-out /raid/benson/data/desi_dr1_medium/checkpoints --wandb-mode disabled --no-push-wandb-artifact
```

## Trajectory

| approach | epoch | train_loss | train_acc | train_redshift_acc | val_loss | val_acc | val_redshift_acc |
|---|---|---|---|---|---|---|---|
|  |  | 408.6034 |  |  |  |  |  |
|  |  | 228.9382 |  |  |  |  |  |
|  |  | 239.7914 |  |  |  |  |  |
|  |  | 194.5182 |  |  |  |  |  |
|  |  | 211.8825 |  |  |  |  |  |
|  |  | 219.6836 |  |  |  |  |  |
|  |  |  |  |  | 245.2362 |  | 0.0166 |

## Summary stats

```yaml
n_records: 7
step_max: 500
step_n: 7
train_all_r2_final: 0.737
train_all_r2_first: 0
train_all_r2_max: 0.737
train_all_r2_min: 0
train_loss_final: 219.684
train_loss_first: 408.603
train_loss_max: 408.603
train_loss_min: 194.518
train_mask_r2_final: 0.739
train_mask_r2_first: 0
train_mask_r2_max: 0.739
train_mask_r2_min: 0
train_masked_acc_final: 0.524
train_masked_acc_first: 0
train_masked_acc_max: 0.541
train_masked_acc_min: 0
train_rz_masked_acc_final: 0
train_rz_masked_acc_first: 0
train_rz_masked_acc_max: 0.125
train_rz_masked_acc_min: 0
train_spec_acc_final: 0.521
train_spec_acc_first: 0
train_spec_acc_max: 0.542
train_spec_acc_min: 0
train_spec_loss_final: 2.003
train_spec_loss_first: 7.728
train_spec_loss_max: 7.728
train_spec_loss_min: 2.003
train_z_acc_final: 0.053
train_z_acc_first: 0
train_z_acc_max: 0.1
train_z_acc_min: 0
train_z_loss_final: 4.273
train_z_loss_first: 7.903
train_z_loss_max: 7.903
train_z_loss_min: 3.775
val_loss_final: 245.236
val_loss_first: 245.236
val_loss_max: 245.236
val_loss_min: 245.236
val_loss_redshift_final: 4.7553
val_loss_redshift_first: 4.7553
val_loss_redshift_max: 4.7553
val_loss_redshift_min: 4.7553
val_loss_spectrum_final: 2.8966
val_loss_spectrum_first: 2.8966
val_loss_spectrum_max: 2.8966
val_loss_spectrum_min: 2.8966
val_loss_total_final: 2.9034
val_loss_total_first: 2.9034
val_loss_total_max: 2.9034
val_loss_total_min: 2.9034
val_masked_spec_acc_final: 0.3221
val_masked_spec_acc_first: 0.3221
val_masked_spec_acc_max: 0.3221
val_masked_spec_acc_min: 0.3221
val_overall_acc_final: 0.3173
val_overall_acc_first: 0.3173
val_overall_acc_max: 0.3173
val_overall_acc_min: 0.3173
val_redshift_acc_final: 0.0166
val_redshift_acc_first: 0.0166
val_redshift_acc_masked_final: 0.0195
val_redshift_acc_masked_first: 0.0195
val_redshift_acc_masked_max: 0.0195
val_redshift_acc_masked_min: 0.0195
val_redshift_acc_max: 0.0166
val_redshift_acc_min: 0.0166
val_spectrum_acc_final: 0.3184
val_spectrum_acc_first: 0.3184
val_spectrum_acc_max: 0.3184
val_spectrum_acc_min: 0.3184
```

