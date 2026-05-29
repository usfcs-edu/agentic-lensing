# codecs_medium

_2000-step codecs Mamba3+RFSQ training on local medium (~11.5k spectra) HDF5 cache_

- **run dir:** `/raid/benson/git/agentic-lensing/experiments/runs/20260526_230011_codecs_medium_dzi7jy`
- **repo:** codecs
- **started:** 2026-05-26 23:00:11 UTC
- **finished:** 2026-05-26 23:13:00 UTC
- **wallclock:** 769.0s
- **exit code:** 0
- **tags:** medium-real, codecs, track3

## Command
```
/home/benson/.venvs/codecs/bin/torchrun --nproc_per_node=1 --master_port=29502 scripts/train.py --config /raid/benson/data/desi_dr1_medium/codecs_medium.yaml
```

## Trajectory

| step | train_loss | val_nll | val_r2 | val_mask_bce | val_perplexity |
|---|---|---|---|---|---|
| 200 | 146.2700 | 23.4945 | 0.2002 | 0.0954 | 0.1211 |
| 400 | 20.4500 | 19.4286 | 0.2207 | 0.0449 | 0.1297 |
| 600 | 14.3600 | 8.2334 | 0.3840 | 0.0342 | 0.1542 |
| 800 | 12.1600 | 7.2459 | 0.3889 | 0.0284 | 0.1863 |
| 1000 | 7.4400 | 11.4929 | 0.3048 | 0.0251 | 0.1820 |
| 1200 | 6.8400 | 4.7624 | 0.4074 | 0.0245 | 0.2025 |
| 1400 | 3.8900 | 5.4590 | 0.3966 | 0.0204 | 0.2105 |
| 1600 | 6.2400 | 3.7961 | 0.4282 | 0.0189 | 0.2198 |
| 1800 | 3.0400 | 2.8540 | 0.4401 | 0.0177 | 0.2346 |
| 2000 | 2.5300 | 3.0258 | 0.4395 | 0.0172 | 0.2355 |

## Summary stats

```yaml
lr_final: 1e-05
lr_first: 9.94e-05
lr_max: 9.94e-05
lr_min: 1e-05
n_records: 10
step_max: 2000
step_n: 10
train_loss_final: 2.53
train_loss_first: 146.27
train_loss_max: 146.27
train_loss_min: 2.53
val_mask_bce_final: 0.0172
val_mask_bce_first: 0.0954
val_mask_bce_max: 0.0954
val_mask_bce_min: 0.0172
val_nll_final: 3.0258
val_nll_first: 23.4945
val_nll_max: 23.4945
val_nll_min: 2.854
val_perplexity_final: 0.2355
val_perplexity_first: 0.1211
val_perplexity_max: 0.2355
val_perplexity_min: 0.1211
val_r2_final: 0.4395
val_r2_first: 0.2002
val_r2_max: 0.4401
val_r2_min: 0.2002
```

