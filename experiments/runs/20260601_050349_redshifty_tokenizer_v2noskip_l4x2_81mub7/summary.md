# redshifty_tokenizer_v2noskip_l4x2

_Skip-free V2 spectrum tokenizer. The full V2 (--use-skip --use-cross-attention,
see redshifty_tokenizer_v2_l4x2) reconstructs to val_recon 0.35 but its DISCRETE
codebook COLLAPSES to a single code (0 bits entropy) — the U-Net skips carry all
the reconstruction info around the quantizer, so the discrete codes the
transformer consumes are constant and redshift never ignites (0% vs V1's 55%).
This variant sets --no-skip --no-cross-attention so all information must flow
through the discrete codes (as in V1), keeping the V2 improvements that don't
bypass the bottleneck: tophat smoothing + codebook entropy regularization.
bf16 AMP, DDP on both L4s, 20k steps. Output: checkpoints/tokenizer_v2noskip_l4x2/best.pt.
VERIFY codebook health with tools/spectrumfm/diagnose_tokenizer_entropy.py before
spending a transformer run on it.
_

- **run dir:** `/raid/benson/git/agentic-lensing/experiments/runs/20260601_050349_redshifty_tokenizer_v2noskip_l4x2_81mub7`
- **repo:** redshifty
- **started:** 2026-06-01 05:03:49 UTC
- **finished:** 2026-06-01 11:31:17 UTC
- **wallclock:** 23248.0s
- **exit code:** 0
- **tags:** xlarge-scale, tokenizer, track3, v2, no-skip, l4-ddp, phase14

## Command
```
/home/benson/.venvs/redshifty/bin/python -m torch.distributed.run --standalone --nproc_per_node=2 nersc/pretrain_tokenizer_v2.py --manifest /raid/benson/data/desi_dr1_medium/manifest_mix.jsonl --steps 20000 --batch-size 32 --lr 1e-4 --warmup 1000 --val-frac 0.05 --val-every 500 --save-every 2500 --log-every 100 --num-workers 8 --amp --entropy-weight 0.1 --commitment-weight 0.05 --no-skip --no-cross-attention --run-name tokenizer_v2noskip_l4x2 --scratch-out /raid/benson/data/desi_dr1_medium/checkpoints --wandb-mode disabled --no-push-wandb-artifact
```

## Trajectory

| approach | epoch | train_loss | train_acc | train_redshift_acc | val_loss | val_acc | val_redshift_acc |
|---|---|---|---|---|---|---|---|
|  |  | 780.7595 |  |  |  |  |  |
|  |  | 11.0804 |  |  |  |  |  |
|  |  | 3.5702 |  |  |  |  |  |
|  |  | 3.5486 |  |  |  |  |  |
|  |  | 2.6987 |  |  |  |  |  |
|  |  | 4.1270 |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
|  |  | 4.1359 |  |  |  |  |  |
|  |  | 5.1351 |  |  |  |  |  |
|  |  | 9.7543 |  |  |  |  |  |
|  |  | 3.3764 |  |  |  |  |  |
|  |  | 3.5183 |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
|  |  | 1.6285 |  |  |  |  |  |
|  |  | 4.3169 |  |  |  |  |  |
|  |  | 1.7330 |  |  |  |  |  |
|  |  | 3.3423 |  |  |  |  |  |
|  |  | 1.7013 |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
|  |  | 4.0780 |  |  |  |  |  |
|  |  | 2.3273 |  |  |  |  |  |
|  |  | 1.7583 |  |  |  |  |  |
|  |  | 0.8821 |  |  |  |  |  |
|  |  | 1.4873 |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
|  |  | 1.4509 |  |  |  |  |  |
|  |  | 6.1946 |  |  |  |  |  |
|  |  | 3.1916 |  |  |  |  |  |
|  |  | 1.0656 |  |  |  |  |  |
|  |  | 1.5386 |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
|  |  | 2.3026 |  |  |  |  |  |
|  |  | 3.0092 |  |  |  |  |  |
|  |  | 6.1838 |  |  |  |  |  |
|  |  | 1.8348 |  |  |  |  |  |
|  |  | 1.8140 |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
|  |  | 1.5425 |  |  |  |  |  |
|  |  | 1.2226 |  |  |  |  |  |
|  |  | 4.4924 |  |  |  |  |  |
|  |  | 3.8533 |  |  |  |  |  |
|  |  | 2.6575 |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
|  |  | 1.0526 |  |  |  |  |  |
|  |  | 0.9049 |  |  |  |  |  |
|  |  | 1.1019 |  |  |  |  |  |
|  |  | 0.9430 |  |  |  |  |  |
|  |  | 2.6024 |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
|  |  | 1.4736 |  |  |  |  |  |
|  |  | 2.4468 |  |  |  |  |  |
|  |  | 4.3939 |  |  |  |  |  |
|  |  | 1.9204 |  |  |  |  |  |
|  |  | 1.8311 |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
|  |  | 1.0383 |  |  |  |  |  |
|  |  | 3.2661 |  |  |  |  |  |
|  |  | 1.9413 |  |  |  |  |  |
|  |  | 2.4099 |  |  |  |  |  |
|  |  | 1.5672 |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
|  |  | 1.0554 |  |  |  |  |  |
|  |  | 0.8995 |  |  |  |  |  |
|  |  | 2.5800 |  |  |  |  |  |
|  |  | 1.6777 |  |  |  |  |  |
|  |  | 2.0159 |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
|  |  | 2.7076 |  |  |  |  |  |
|  |  | 1.4126 |  |  |  |  |  |
|  |  | 1.3990 |  |  |  |  |  |
|  |  | 1.9974 |  |  |  |  |  |
|  |  | 0.8631 |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
|  |  | 2.3044 |  |  |  |  |  |
|  |  | 2.0615 |  |  |  |  |  |
|  |  | 3.9286 |  |  |  |  |  |
|  |  | 1.4207 |  |  |  |  |  |
|  |  | 2.0497 |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
|  |  | 2.0581 |  |  |  |  |  |
|  |  | 1.1920 |  |  |  |  |  |
|  |  | 0.9240 |  |  |  |  |  |
|  |  | 2.1429 |  |  |  |  |  |
|  |  | 1.5423 |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
|  |  | 1.2928 |  |  |  |  |  |
|  |  | 1.2171 |  |  |  |  |  |
|  |  | 1.3881 |  |  |  |  |  |
|  |  | 0.6905 |  |  |  |  |  |
|  |  | 1.9911 |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
|  |  | 0.9510 |  |  |  |  |  |
|  |  | 2.2587 |  |  |  |  |  |
|  |  | 1.0791 |  |  |  |  |  |
|  |  | 1.2888 |  |  |  |  |  |
|  |  | 1.1805 |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
|  |  | 0.6917 |  |  |  |  |  |
|  |  | 1.9697 |  |  |  |  |  |
|  |  | 3.6089 |  |  |  |  |  |
|  |  | 1.0117 |  |  |  |  |  |
|  |  | 1.5754 |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
|  |  | 1.3948 |  |  |  |  |  |
|  |  | 1.7374 |  |  |  |  |  |
|  |  | 1.2627 |  |  |  |  |  |
|  |  | 1.1565 |  |  |  |  |  |
|  |  | 2.5857 |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
|  |  | 0.9058 |  |  |  |  |  |
|  |  | 1.4569 |  |  |  |  |  |
|  |  | 0.8843 |  |  |  |  |  |
|  |  | 17.6020 |  |  |  |  |  |
|  |  | 0.9137 |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
|  |  | 0.7936 |  |  |  |  |  |
|  |  | 0.9172 |  |  |  |  |  |
|  |  | 1.2222 |  |  |  |  |  |
|  |  | 1.3115 |  |  |  |  |  |
|  |  | 0.9508 |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
|  |  | 2.2192 |  |  |  |  |  |
|  |  | 1.7770 |  |  |  |  |  |
|  |  | 1.2904 |  |  |  |  |  |
|  |  | 0.8356 |  |  |  |  |  |
|  |  | 1.3261 |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
|  |  | 2.0204 |  |  |  |  |  |
|  |  | 2.5050 |  |  |  |  |  |
|  |  | 0.9617 |  |  |  |  |  |
|  |  | 2.0552 |  |  |  |  |  |
|  |  | 11.0414 |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
|  |  | 1.0596 |  |  |  |  |  |
|  |  | 1.2850 |  |  |  |  |  |
|  |  | 0.7854 |  |  |  |  |  |
|  |  | 0.9006 |  |  |  |  |  |
|  |  | 1.4251 |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
|  |  | 2.8290 |  |  |  |  |  |
|  |  | 0.9043 |  |  |  |  |  |
|  |  | 1.1315 |  |  |  |  |  |
|  |  | 2.8010 |  |  |  |  |  |
|  |  | 1.7871 |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
|  |  | 2.2661 |  |  |  |  |  |
|  |  | 1.2774 |  |  |  |  |  |
|  |  | 0.9876 |  |  |  |  |  |
|  |  | 1.1267 |  |  |  |  |  |
|  |  | 1.3735 |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
|  |  | 2.0625 |  |  |  |  |  |
|  |  | 2.2962 |  |  |  |  |  |
|  |  | 3.5792 |  |  |  |  |  |
|  |  | 1.1196 |  |  |  |  |  |
|  |  | 0.9161 |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
|  |  | 1.0544 |  |  |  |  |  |
|  |  | 0.7526 |  |  |  |  |  |
|  |  | 1.8657 |  |  |  |  |  |
|  |  | 0.8159 |  |  |  |  |  |
|  |  | 1.7498 |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
|  |  | 1.5536 |  |  |  |  |  |
|  |  | 0.6334 |  |  |  |  |  |
|  |  | 1.3255 |  |  |  |  |  |
|  |  | 1.6821 |  |  |  |  |  |
|  |  | 1.1443 |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
|  |  | 1.1468 |  |  |  |  |  |
|  |  | 1.9052 |  |  |  |  |  |
|  |  | 0.6109 |  |  |  |  |  |
|  |  | 11.0881 |  |  |  |  |  |
|  |  | 1.0652 |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
|  |  | 0.7113 |  |  |  |  |  |
|  |  | 0.6500 |  |  |  |  |  |
|  |  | 0.9838 |  |  |  |  |  |
|  |  | 0.9317 |  |  |  |  |  |
|  |  | 1.9735 |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
|  |  | 0.7927 |  |  |  |  |  |
|  |  | 0.9003 |  |  |  |  |  |
|  |  | 1.9395 |  |  |  |  |  |
|  |  | 1.0607 |  |  |  |  |  |
|  |  | 819.5482 |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
|  |  | 0.8595 |  |  |  |  |  |
|  |  | 1.0292 |  |  |  |  |  |
|  |  | 0.9226 |  |  |  |  |  |
|  |  | 0.8478 |  |  |  |  |  |
|  |  | 0.8915 |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
|  |  | 1.0009 |  |  |  |  |  |
|  |  | 0.9718 |  |  |  |  |  |
|  |  | 0.8252 |  |  |  |  |  |
|  |  | 0.7905 |  |  |  |  |  |
|  |  | 0.9650 |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
|  |  | 0.9750 |  |  |  |  |  |
|  |  | 1.2796 |  |  |  |  |  |
|  |  | 1.1317 |  |  |  |  |  |
|  |  | 2.0672 |  |  |  |  |  |
|  |  | 1.3077 |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
|  |  | 1.0339 |  |  |  |  |  |
|  |  | 0.8956 |  |  |  |  |  |
|  |  | 1.3958 |  |  |  |  |  |
|  |  | 0.9082 |  |  |  |  |  |
|  |  | 0.9578 |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
|  |  | 0.5493 |  |  |  |  |  |
|  |  | 1.2777 |  |  |  |  |  |
|  |  | 0.8407 |  |  |  |  |  |
|  |  | 1.7641 |  |  |  |  |  |
|  |  | 1.0440 |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
|  |  | 0.6967 |  |  |  |  |  |
|  |  | 0.7799 |  |  |  |  |  |
|  |  | 1.0215 |  |  |  |  |  |
|  |  | 1.5627 |  |  |  |  |  |
|  |  | 1.3446 |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
|  |  | 1.3685 |  |  |  |  |  |
|  |  | 1.2545 |  |  |  |  |  |
|  |  | 0.7824 |  |  |  |  |  |
|  |  | 0.9991 |  |  |  |  |  |
|  |  | 1.8187 |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
|  |  | 1.6478 |  |  |  |  |  |
|  |  | 1.2642 |  |  |  |  |  |
|  |  | 1.1067 |  |  |  |  |  |
|  |  | 1.0652 |  |  |  |  |  |
|  |  | 1.3166 |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
|  |  | 2.2546 |  |  |  |  |  |
|  |  | 1.3554 |  |  |  |  |  |
|  |  | 0.9761 |  |  |  |  |  |
|  |  | 1.6724 |  |  |  |  |  |

## Summary stats

```yaml
n_records: 239
step_max: 19900
step_n: 239
train_commit_final: 0.679
train_commit_first: 0.0481
train_commit_max: 0.7294
train_commit_min: 0.0139
train_ent_final: 0.0432
train_ent_first: 0.0659
train_ent_max: 0.0852
train_ent_min: 0.0429
train_loss_final: 1.6724
train_loss_first: 780.76
train_loss_max: 819.548
train_loss_min: 0.5493
train_lr_final: 1e-05
train_lr_first: 1e-07
train_lr_max: 0.0001
train_lr_min: 1e-07
train_quant_final: 0.7222
train_quant_first: 0.114
train_quant_max: 0.7723
train_quant_min: 0.0721
train_recon_final: 0.9501
train_recon_first: 780.645
train_recon_max: 818.907
train_recon_min: 0.331
val_commit_final: 0.3534
val_commit_first: 0.022
val_commit_max: 0.5155
val_commit_min: 0.0147
val_entropy_final: 0.0448
val_entropy_first: 0.0617
val_entropy_max: 0.0617
val_entropy_min: 0.0445
val_quant_final: 0.3982
val_quant_first: 0.0837
val_quant_max: 0.5604
val_quant_min: 0.0737
val_recon_final: 1.3936
val_recon_first: 5.0267
val_recon_max: 5.0267
val_recon_min: 1.3936
val_total_final: 1.7918
val_total_first: 5.1104
val_total_max: 5.1104
val_total_min: 1.7918
```

