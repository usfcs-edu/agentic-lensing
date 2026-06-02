# redshifty_approach_a_mix_l4x2_v2

_V2-tokenizer arm of the V1-vs-V2 ablation: Approach-A on the 4-way data mix
using the locally-trained V2 tokenizer (checkpoints/tokenizer_v2_l4x2/best.pt
from the redshifty_tokenizer_v2_l4x2 spec). Identical transformer config to
the _v1 control (DDP on both L4s, effective batch 64, 15k steps,
--ar-eval-batches 8) so the only deliberate change is the tokenizer.
(--ar-eval-batches 8 not 32: under DDP, a >10min rank-0-only AR eval makes
the other rank's buffer broadcast hit NCCL's watchdog timeout and abort.)
Tests the central SpectrumFM claim that a sharper tokenizer (val_recon 0.157
vs 1.38) yields faster/higher downstream redshift ignition.

NOTE: --tokenizer-kind v2 also switches the redshift tokenizer from
RedshiftTokenizer(n_levels=256) to RedshiftTokenizerV2(n_levels=1024), so this
is a "tokenizer-kind" ablation, not a pure spectrum-tokenizer swap.
_

- **run dir:** `/raid/benson/git/agentic-lensing/experiments/runs/20260531_195809_redshifty_approach_a_mix_l4x2_v2_vj8n4y`
- **repo:** redshifty
- **started:** 2026-05-31 19:58:09 UTC
- **finished:** 2026-06-01 04:18:04 UTC
- **wallclock:** 29995.4s
- **exit code:** 0
- **tags:** xlarge-scale, transformer, approach-a, track3, l4-ddp, v2-revisit, phase14

## Command
```
/home/benson/.venvs/redshifty/bin/python -m torch.distributed.run --standalone --nproc_per_node=2 nersc/train_transformer.py --manifest /raid/benson/data/desi_dr1_medium/manifest_mix.jsonl --tokenizer-ckpt /raid/benson/data/desi_dr1_medium/checkpoints/tokenizer_v2_l4x2/best.pt --tokenizer-kind v2 --approach a --steps 15000 --batch-size 32 --lr 4e-4 --warmup 500 --healpix-holdout-frac 0.05 --val-every 500 --save-every 2500 --log-every 100 --num-workers 4 --amp --redshift-loss-weight 50.0 --encoder-mask-ratio 0.50 --ar-eval-batches 8 --run-name approach_a_mix_l4x2_v2 --scratch-out /raid/benson/data/desi_dr1_medium/checkpoints --wandb-mode disabled --no-push-wandb-artifact
```

## Trajectory

| approach | epoch | train_loss | train_acc | train_redshift_acc | val_loss | val_acc | val_redshift_acc |
|---|---|---|---|---|---|---|---|
|  |  | 406.0911 |  |  |  |  |  |
|  |  | 329.0302 |  |  |  |  |  |
|  |  | 346.5892 |  |  |  |  |  |
|  |  | 326.9626 |  |  |  |  |  |
|  |  | 301.0665 |  |  |  |  |  |
|  |  | 327.7303 |  |  |  |  |  |
|  |  |  |  |  | 358.7643 |  | 0.0032 |
|  |  | 316.5777 |  |  |  |  |  |
|  |  | 299.0747 |  |  |  |  |  |
|  |  | 307.6967 |  |  |  |  |  |
|  |  | 304.5110 |  |  |  |  |  |
|  |  | 306.0855 |  |  |  |  |  |
|  |  |  |  |  | 350.2750 |  | 0.0032 |
|  |  | 305.8689 |  |  |  |  |  |
|  |  | 314.8791 |  |  |  |  |  |
|  |  | 314.3104 |  |  |  |  |  |
|  |  | 316.2401 |  |  |  |  |  |
|  |  | 302.6221 |  |  |  |  |  |
|  |  |  |  |  | 346.9909 |  | 0.0000 |
|  |  | 321.4387 |  |  |  |  |  |
|  |  | 302.7597 |  |  |  |  |  |
|  |  | 333.3989 |  |  |  |  |  |
|  |  | 327.8717 |  |  |  |  |  |
|  |  | 336.4551 |  |  |  |  |  |
|  |  |  |  |  | 347.9114 |  | 0.0000 |
|  |  | 309.9774 |  |  |  |  |  |
|  |  | 298.9741 |  |  |  |  |  |
|  |  | 299.1454 |  |  |  |  |  |
|  |  | 325.3998 |  |  |  |  |  |
|  |  | 284.2949 |  |  |  |  |  |
|  |  |  |  |  | 347.9934 |  | 0.0000 |
|  |  | 320.6108 |  |  |  |  |  |
|  |  | 333.3199 |  |  |  |  |  |
|  |  | 295.8152 |  |  |  |  |  |
|  |  | 317.4337 |  |  |  |  |  |
|  |  | 287.2335 |  |  |  |  |  |
|  |  |  |  |  | 347.4790 |  | 0.0000 |
|  |  | 292.0005 |  |  |  |  |  |
|  |  | 298.3246 |  |  |  |  |  |
|  |  | 315.9028 |  |  |  |  |  |
|  |  | 322.5551 |  |  |  |  |  |
|  |  | 309.3406 |  |  |  |  |  |
|  |  |  |  |  | 348.3534 |  | 0.0000 |
|  |  | 285.3952 |  |  |  |  |  |
|  |  | 323.1873 |  |  |  |  |  |
|  |  | 304.6786 |  |  |  |  |  |
|  |  | 325.3857 |  |  |  |  |  |
|  |  | 334.4316 |  |  |  |  |  |
|  |  |  |  |  | 347.5632 |  | 0.0000 |
|  |  | 296.8164 |  |  |  |  |  |
|  |  | 293.8086 |  |  |  |  |  |
|  |  | 297.8799 |  |  |  |  |  |
|  |  | 325.4765 |  |  |  |  |  |
|  |  | 310.2454 |  |  |  |  |  |
|  |  |  |  |  | 346.5462 |  | 0.0000 |
|  |  | 303.5507 |  |  |  |  |  |
|  |  | 280.6282 |  |  |  |  |  |
|  |  | 331.6096 |  |  |  |  |  |
|  |  | 297.6476 |  |  |  |  |  |
|  |  | 306.2282 |  |  |  |  |  |
|  |  |  |  |  | 346.1194 |  | 0.0000 |
|  |  | 310.9392 |  |  |  |  |  |
|  |  | 321.4628 |  |  |  |  |  |
|  |  | 297.6851 |  |  |  |  |  |
|  |  | 324.7401 |  |  |  |  |  |
|  |  | 290.0026 |  |  |  |  |  |
|  |  |  |  |  | 347.2510 |  | 0.0000 |
|  |  | 301.1923 |  |  |  |  |  |
|  |  | 349.2834 |  |  |  |  |  |
|  |  | 278.2993 |  |  |  |  |  |
|  |  | 315.0055 |  |  |  |  |  |
|  |  | 309.5820 |  |  |  |  |  |
|  |  |  |  |  | 346.6773 |  | 0.0032 |
|  |  | 292.0609 |  |  |  |  |  |
|  |  | 314.2383 |  |  |  |  |  |
|  |  | 284.1528 |  |  |  |  |  |
|  |  | 253.3554 |  |  |  |  |  |
|  |  | 306.4100 |  |  |  |  |  |
|  |  |  |  |  | 347.0608 |  | 0.0032 |
|  |  | 291.4019 |  |  |  |  |  |
|  |  | 315.8611 |  |  |  |  |  |
|  |  | 300.6032 |  |  |  |  |  |
|  |  | 293.7393 |  |  |  |  |  |
|  |  | 343.4862 |  |  |  |  |  |
|  |  |  |  |  | 348.0960 |  | 0.0000 |
|  |  | 327.6913 |  |  |  |  |  |
|  |  | 295.2821 |  |  |  |  |  |
|  |  | 283.3655 |  |  |  |  |  |
|  |  | 297.4240 |  |  |  |  |  |
|  |  | 306.6038 |  |  |  |  |  |
|  |  |  |  |  | 346.7766 |  | 0.0032 |
|  |  | 307.7122 |  |  |  |  |  |
|  |  | 316.4216 |  |  |  |  |  |
|  |  | 310.8859 |  |  |  |  |  |
|  |  | 288.5640 |  |  |  |  |  |
|  |  | 323.8455 |  |  |  |  |  |
|  |  |  |  |  | 345.7583 |  | 0.0000 |
|  |  | 303.6338 |  |  |  |  |  |
|  |  | 313.7080 |  |  |  |  |  |
|  |  | 298.9927 |  |  |  |  |  |
|  |  | 301.7257 |  |  |  |  |  |
|  |  | 315.1041 |  |  |  |  |  |
|  |  |  |  |  | 346.1281 |  | 0.0000 |
|  |  | 293.0303 |  |  |  |  |  |
|  |  | 296.6109 |  |  |  |  |  |
|  |  | 317.4814 |  |  |  |  |  |
|  |  | 297.4611 |  |  |  |  |  |
|  |  | 303.2659 |  |  |  |  |  |
|  |  |  |  |  | 346.3444 |  | 0.0000 |
|  |  | 312.9294 |  |  |  |  |  |
|  |  | 327.6588 |  |  |  |  |  |
|  |  | 300.8224 |  |  |  |  |  |
|  |  | 298.4865 |  |  |  |  |  |
|  |  | 307.8676 |  |  |  |  |  |
|  |  |  |  |  | 344.6403 |  | 0.0000 |
|  |  | 310.6630 |  |  |  |  |  |
|  |  | 315.2422 |  |  |  |  |  |
|  |  | 317.5777 |  |  |  |  |  |
|  |  | 294.8031 |  |  |  |  |  |
|  |  | 323.1630 |  |  |  |  |  |
|  |  |  |  |  | 346.9059 |  | 0.0032 |
|  |  | 336.9127 |  |  |  |  |  |
|  |  | 313.5589 |  |  |  |  |  |
|  |  | 302.8843 |  |  |  |  |  |
|  |  | 336.7781 |  |  |  |  |  |
|  |  | 324.0368 |  |  |  |  |  |
|  |  |  |  |  | 345.8704 |  | 0.0000 |
|  |  | 305.9917 |  |  |  |  |  |
|  |  | 314.7740 |  |  |  |  |  |
|  |  | 312.9026 |  |  |  |  |  |
|  |  | 310.6757 |  |  |  |  |  |
|  |  | 275.9859 |  |  |  |  |  |
|  |  |  |  |  | 345.7788 |  | 0.0032 |
|  |  | 282.1494 |  |  |  |  |  |
|  |  | 313.4170 |  |  |  |  |  |
|  |  | 313.3544 |  |  |  |  |  |
|  |  | 304.5827 |  |  |  |  |  |
|  |  | 315.8559 |  |  |  |  |  |
|  |  |  |  |  | 345.9598 |  | 0.0000 |
|  |  | 312.8616 |  |  |  |  |  |
|  |  | 329.0249 |  |  |  |  |  |
|  |  | 298.4733 |  |  |  |  |  |
|  |  | 293.0298 |  |  |  |  |  |
|  |  | 313.9672 |  |  |  |  |  |
|  |  |  |  |  | 346.5692 |  | 0.0032 |
|  |  | 310.7857 |  |  |  |  |  |
|  |  | 330.4930 |  |  |  |  |  |
|  |  | 299.5995 |  |  |  |  |  |
|  |  | 325.7479 |  |  |  |  |  |
|  |  | 312.4247 |  |  |  |  |  |
|  |  |  |  |  | 346.3876 |  | 0.0032 |
|  |  | 305.4837 |  |  |  |  |  |
|  |  | 268.8756 |  |  |  |  |  |
|  |  | 315.0088 |  |  |  |  |  |
|  |  | 296.0293 |  |  |  |  |  |
|  |  | 306.9117 |  |  |  |  |  |
|  |  |  |  |  | 345.5542 |  | 0.0032 |
|  |  | 308.9650 |  |  |  |  |  |
|  |  | 289.6591 |  |  |  |  |  |
|  |  | 308.5822 |  |  |  |  |  |
|  |  | 338.7381 |  |  |  |  |  |
|  |  | 304.3102 |  |  |  |  |  |
|  |  |  |  |  | 345.7019 |  | 0.0032 |
|  |  | 282.8939 |  |  |  |  |  |
|  |  | 320.6758 |  |  |  |  |  |
|  |  | 296.9356 |  |  |  |  |  |
|  |  | 320.3188 |  |  |  |  |  |
|  |  | 277.2347 |  |  |  |  |  |
|  |  |  |  |  | 346.7456 |  | 0.0032 |
|  |  | 313.5329 |  |  |  |  |  |
|  |  | 326.8175 |  |  |  |  |  |
|  |  | 296.1132 |  |  |  |  |  |
|  |  | 319.5578 |  |  |  |  |  |
|  |  | 311.5394 |  |  |  |  |  |
|  |  |  |  |  | 346.8681 |  | 0.0032 |
|  |  | 307.6319 |  |  |  |  |  |
|  |  | 306.4483 |  |  |  |  |  |
|  |  | 309.8276 |  |  |  |  |  |
|  |  | 315.8457 |  |  |  |  |  |

## Summary stats

```yaml
n_records: 179
step_max: 14900
step_n: 179
train_all_r2_final: 1
train_all_r2_first: 0
train_all_r2_max: 1
train_all_r2_min: 0
train_loss_final: 315.846
train_loss_first: 406.091
train_loss_max: 406.091
train_loss_min: 253.355
train_mask_r2_final: 1
train_mask_r2_first: 0
train_mask_r2_max: 1
train_mask_r2_min: 0
train_masked_acc_final: 1
train_masked_acc_first: 0
train_masked_acc_max: 1
train_masked_acc_min: 0
train_rz_masked_acc_final: 0
train_rz_masked_acc_first: 0
train_rz_masked_acc_max: 0.286
train_rz_masked_acc_min: 0
train_spec_acc_final: 1
train_spec_acc_first: 0
train_spec_acc_max: 1
train_spec_acc_min: 0
train_spec_loss_final: 0
train_spec_loss_first: 8.871
train_spec_loss_max: 8.871
train_spec_loss_min: 0
train_z_acc_final: 0.043
train_z_acc_first: 0
train_z_acc_max: 0.158
train_z_acc_min: 0
train_z_loss_final: 6.3
train_z_loss_first: 7.824
train_z_loss_max: 7.824
train_z_loss_min: 5.066
val_loss_final: 346.868
val_loss_first: 358.764
val_loss_max: 358.764
val_loss_min: 344.64
val_loss_redshift_final: 6.9282
val_loss_redshift_first: 7.1631
val_loss_redshift_max: 7.1631
val_loss_redshift_min: 6.8829
val_loss_spectrum_final: 0.0003
val_loss_spectrum_first: 0.0631
val_loss_spectrum_max: 0.0631
val_loss_spectrum_min: 0.0003
val_loss_total_final: 0.0256
val_loss_total_first: 0.089
val_loss_total_max: 0.089
val_loss_total_min: 0.0256
val_masked_spec_acc_final: 1
val_masked_spec_acc_first: 0.9963
val_masked_spec_acc_max: 1
val_masked_spec_acc_min: 0.9963
val_overall_acc_final: 0.9963
val_overall_acc_first: 0.9891
val_overall_acc_max: 0.9963
val_overall_acc_min: 0.9891
val_redshift_acc_final: 0.0032
val_redshift_acc_first: 0.0032
val_redshift_acc_masked_final: 0.0033
val_redshift_acc_masked_first: 0.0034
val_redshift_acc_masked_max: 0.005
val_redshift_acc_masked_min: 0
val_redshift_acc_max: 0.0032
val_redshift_acc_min: 0
val_spectrum_acc_final: 1
val_spectrum_acc_first: 0.9927
val_spectrum_acc_max: 1
val_spectrum_acc_min: 0.9927
```

