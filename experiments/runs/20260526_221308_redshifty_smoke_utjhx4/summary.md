# redshifty_smoke

_3-epoch Approach A+B smoke test on 50-spectrum subset (mirrors scripts/smoke_test.py)_

- **run dir:** `/raid/benson/git/agentic-lensing/experiments/runs/20260526_221308_redshifty_smoke_utjhx4`
- **repo:** redshifty
- **started:** (see prior)
- **finished:** (see prior)
- **wallclock:** 429.2s
- **exit code:** 0
- **tags:** smoke, track2-mvp

## Command
```
(reparsed)
```

## Trajectory

| approach | epoch | train_loss | train_acc | train_redshift_acc | val_loss | val_acc | val_redshift_acc |
|---|---|---|---|---|---|---|---|
| a | 1 | 20.3680 | 0.1020 | 0.0000 | 20.0041 | 0.1270 | 0.0000 |
| a | 2 | 18.4135 | 0.1690 | 0.0210 | 19.6710 | 0.1380 | 0.0000 |
| a | 3 | 17.2396 | 0.1920 | 0.1040 | 19.3549 | 0.1380 | 0.0000 |
| b | 1 | 20.2961 | 0.0970 | 0.0000 | 20.3278 | 0.0720 | 0.0000 |
| b | 2 | 18.3663 | 0.1630 | 0.0210 | 20.0816 | 0.1070 | 0.0000 |
| b | 3 | 17.2467 | 0.1640 | 0.0620 | 19.8531 | 0.1070 | 0.0000 |

## Summary stats

```yaml
approach_a_epoch_max: 3
approach_a_epoch_n: 3
approach_a_train_acc_final: 0.192
approach_a_train_acc_first: 0.102
approach_a_train_acc_max: 0.192
approach_a_train_acc_min: 0.102
approach_a_train_loss_final: 17.2396
approach_a_train_loss_first: 20.368
approach_a_train_loss_max: 20.368
approach_a_train_loss_min: 17.2396
approach_a_train_redshift_acc_final: 0.104
approach_a_train_redshift_acc_first: 0
approach_a_train_redshift_acc_max: 0.104
approach_a_train_redshift_acc_min: 0
approach_a_val_acc_final: 0.138
approach_a_val_acc_first: 0.127
approach_a_val_acc_max: 0.138
approach_a_val_acc_min: 0.127
approach_a_val_loss_final: 19.3549
approach_a_val_loss_first: 20.0041
approach_a_val_loss_max: 20.0041
approach_a_val_loss_min: 19.3549
approach_a_val_redshift_acc_final: 0
approach_a_val_redshift_acc_first: 0
approach_a_val_redshift_acc_max: 0
approach_a_val_redshift_acc_min: 0
approach_b_epoch_max: 3
approach_b_epoch_n: 3
approach_b_train_acc_final: 0.164
approach_b_train_acc_first: 0.097
approach_b_train_acc_max: 0.164
approach_b_train_acc_min: 0.097
approach_b_train_loss_final: 17.2467
approach_b_train_loss_first: 20.2961
approach_b_train_loss_max: 20.2961
approach_b_train_loss_min: 17.2467
approach_b_train_redshift_acc_final: 0.062
approach_b_train_redshift_acc_first: 0
approach_b_train_redshift_acc_max: 0.062
approach_b_train_redshift_acc_min: 0
approach_b_val_acc_final: 0.107
approach_b_val_acc_first: 0.072
approach_b_val_acc_max: 0.107
approach_b_val_acc_min: 0.072
approach_b_val_loss_final: 19.8531
approach_b_val_loss_first: 20.3278
approach_b_val_loss_max: 20.3278
approach_b_val_loss_min: 19.8531
approach_b_val_redshift_acc_final: 0
approach_b_val_redshift_acc_first: 0
approach_b_val_redshift_acc_max: 0
approach_b_val_redshift_acc_min: 0
n_records: 6
```

