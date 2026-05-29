# codecs_smoke

_20-step codecs Mamba3+RFSQ training on the 247-spectrum mini-HDF5 cache_

- **run dir:** `/raid/benson/git/agentic-lensing/experiments/runs/20260526_222055_codecs_smoke_vx7ngz`
- **repo:** codecs
- **started:** 2026-05-26 22:20:55 UTC
- **finished:** 2026-05-26 22:21:33 UTC
- **wallclock:** 38.8s
- **exit code:** 0
- **tags:** smoke, track2-mvp

## Command
```
/home/benson/.venvs/codecs/bin/torchrun --nproc_per_node=1 --master_port=29501 scripts/train.py --config /raid/benson/data/desi_dr1_mini/codecs_smoke.yaml
```

## Trajectory

| step | train_loss | val_nll | val_r2 | val_mask_bce | val_perplexity |
|---|---|---|---|---|---|
| 2 | 941.0000 | 161.8750 | -7.7109 | 0.9414 | 0.1740 |
| 4 | 134.2500 | 49.6250 | -2.3047 | 0.8428 | 0.1483 |
| 6 | 34.5000 | 16.8359 | -0.9868 | 0.7471 | 0.1630 |
| 8 | 45.2500 | 22.9336 | -0.9897 | 0.7207 | 0.1474 |
| 10 | 45.5000 | 29.5234 | -1.1313 | 0.7012 | 0.1571 |
| 12 | 26.4700 | 19.8164 | -0.7754 | 0.7109 | 0.1400 |
| 14 | 38.4500 | 13.2266 | -0.4761 | 0.7080 | 0.1373 |
| 16 | 27.4400 | 9.9473 | -0.3193 | 0.7061 | 0.1417 |
| 18 | 25.8100 | 8.4980 | -0.2515 | 0.7021 | 0.1435 |
| 20 | 12.3100 | 7.9355 | -0.2227 | 0.6992 | 0.1438 |

## Summary stats

```yaml
lr_final: 1e-05
lr_first: 0.0001
lr_max: 0.0001
lr_min: 1e-05
n_records: 10
step_max: 20
step_n: 10
train_loss_final: 12.31
train_loss_first: 941
train_loss_max: 941
train_loss_min: 12.31
val_mask_bce_final: 0.6992
val_mask_bce_first: 0.9414
val_mask_bce_max: 0.9414
val_mask_bce_min: 0.6992
val_nll_final: 7.9355
val_nll_first: 161.875
val_nll_max: 161.875
val_nll_min: 7.9355
val_perplexity_final: 0.1438
val_perplexity_first: 0.174
val_perplexity_max: 0.174
val_perplexity_min: 0.1373
val_r2_final: -0.2227
val_r2_first: -7.7109
val_r2_max: -0.2227
val_r2_min: -7.7109
```

