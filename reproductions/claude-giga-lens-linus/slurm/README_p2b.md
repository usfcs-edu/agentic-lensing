# P2b B1-REDUCED job chain (design: research/checkpoint_b1_reduced.md)

Funded 2026-07-18 (user): P2b cap 18 A100-h (10 freed-P4/D7 + 8 stretch),
OUTSIDE the exhausted P2 cap; global 100 unchanged. Fences: S1r <= 12
(3 x 3:55 walls), S6br <= 5, worst case 16.67.

## S1r chain (submit all three at once; ledger est rows appended at submission)

```bash
J1=$(sbatch --parsable p2b_b1r_s1_leg1.slurm)
J2=$(sbatch --parsable --dependency=afterany:$J1 p2b_b1r_s1_leg2.slurm)
J3=$(sbatch --parsable --dependency=afterany:$J2 p2b_b1r_s1_leg3.slurm)
```

Leg 1 fresh (refuses a non-empty ckpt dir); legs 2-3 `--resume` from the
newest $PSCRATCH stage checkpoint (bit-identical incl. full logZ) and no-op
fast-exit once the COMPLETE marker `b1r128_carousel33_s1_seed2.json` exists.
Exit 3 = PARTIAL_WALL_CAP per the l0 protocol (expected mid-chain, not a
failure). After leg 3 the chain STOPS regardless of lambda (hard cap 12) —
no leg 4 without a NEW user decision. Watchdog: register each leg at
submission (max_run 4.5 h, expect_artifact CFS
results/b1r128_carousel33_s1_seed2.json on the LAST leg only, on_stall=alert
— NEVER auto-resubmit: leg 1 would trip the ckpt no-mixing fence).

## S6br (HELD TEMPLATE — the S1r HARVEST fills it)

`p2b_b1r_s6b.slurm` is a TEMPLATE: the harvest session fills
`GRAD_BUDGET="TODO_FILL_FROM_S1R_HARVEST"` with **B\* = `grad_evals.total`
from the S1r COMPLETE result json** (Track-A budget-matched; MAP billed),
then updates the file's line in `deploy.md5` (both sides) before sbatch.
In-script guards refuse an unfilled template or a missing S1r marker.
S6br is NOT submitted if S1r ends PARTIAL (orchestrator decision point).

## Superseded

`p2_b1r_s1_seed2.slurm`, `p2_b1r_s1_seed3.slurm`, `p2_b1r_s6b.slurm` =
the ORIGINAL full-scope held designs (N=512 x 2 seeds + until-converged
S6b). Permanently HELD — never submit; kept only as the pre-descope record.
