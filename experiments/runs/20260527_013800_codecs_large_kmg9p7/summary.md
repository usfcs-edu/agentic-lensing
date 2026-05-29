# codecs_large

_5000-step codecs Mamba3+RFSQ training on the larger DESI subset HDF5 cache_

- **run dir:** `/raid/benson/git/agentic-lensing/experiments/runs/20260527_013800_codecs_large_kmg9p7`
- **repo:** codecs
- **started:** 2026-05-27 01:38:00 UTC
- **finished:** 2026-05-27 02:05:45 UTC
- **wallclock:** 1665.1s
- **exit code:** 0
- **tags:** large-scale, codecs, track3

## Command
```
/home/benson/.venvs/codecs/bin/torchrun --nproc_per_node=1 --master_port=29503 scripts/train.py --config /raid/benson/data/desi_dr1_medium/codecs_large.yaml
```

## Trajectory

| step | train_loss | val_nll | val_r2 | val_mask_bce | val_perplexity |
|---|---|---|---|---|---|
| 500 | 134.1300 | 15.8483 | 0.2668 | 0.0634 | 0.0618 |
| 1000 | 12.0000 | 8.9542 | 0.3759 | 0.0282 | 0.0915 |
| 1500 | 7.3100 | 4.7588 | 0.4103 | 0.0217 | 0.1906 |
| 2000 | 5.3000 | 6.4654 | 0.4212 | 0.0173 | 0.2262 |
| 2500 | 4.5700 | 3.6504 | 0.4326 | 0.0154 | 0.2271 |
| 3000 | 3.4500 | 3.6652 | 0.4356 | 0.0131 | 0.2572 |
| 3500 | 2.5800 | 2.8933 | 0.4452 | 0.0105 | 0.3122 |
| 4000 | 2.1300 | 2.8758 | 0.4532 | 0.0096 | 0.3659 |
| 4500 | 2.0900 | 2.3787 | 0.4597 | 0.0092 | 0.3643 |
| 5000 | 1.8000 | 2.3207 | 0.4615 | 0.0094 | 0.3779 |

## Summary stats

```yaml
lr_final: 1e-05
lr_first: 9.94e-05
lr_max: 9.94e-05
lr_min: 1e-05
n_records: 10
step_max: 5000
step_n: 10
train_loss_final: 1.8
train_loss_first: 134.13
train_loss_max: 134.13
train_loss_min: 1.8
val_mask_bce_final: 0.0094
val_mask_bce_first: 0.0634
val_mask_bce_max: 0.0634
val_mask_bce_min: 0.0092
val_nll_final: 2.3207
val_nll_first: 15.8483
val_nll_max: 15.8483
val_nll_min: 2.3207
val_perplexity_final: 0.3779
val_perplexity_first: 0.0618
val_perplexity_max: 0.3779
val_perplexity_min: 0.0618
val_r2_final: 0.4615
val_r2_first: 0.2668
val_r2_max: 0.4615
val_r2_min: 0.2668
```

