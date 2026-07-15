# tools/ — dead-man watchdog operating card

## Why this exists: the P1c incident

Last campaign (`../claude-giga-lens/CAMPAIGN.md`, "P1c SMC OOM + multi-day stall"):
the correlated-SMC canary **55885270** OOM'd on Perlmutter (120.4 GB alloc with 300
particles) **and the babysitting agent died silently** during a multi-day gap. Its
`SMC_PARTICLES=200` fallback never fired. Nobody noticed. **P1c stalled 2026-07-10 →
07-14 — four days of dead air.** Two independent failures had to coincide, and both
were silent. This watchdog makes each one loud:

1. **job quietly dead** (OOM/timeout/vanished/stuck in queue) → alert conditions below;
2. **agent quietly dead** (nobody registering/harvesting jobs anymore) → `STATE_STALE`
   tripwire on the state file's heartbeat timestamp.

Plan of record: PLAN §2 ("Autonomous operation with the dead-man watchdog") and the
previous campaign's NEXT_DIRECTIONS T0.5. **No production sbatch without a watchdog
row** — register every job immediately after submission.

## The pieces

| File | Role |
|---|---|
| `tools/watchdog.py` | ONE poll pass per invocation. Stdlib-only (no jax/cgl2/venv needed — system `python3` is fine). Polls perlmutter via a single batched `ssh gdbenson@perlmutter.nersc.gov squeue+sacct` and phoenix PIDs via local `ps` (+ best-effort `nvidia-smi` on-GPU tag). |
| `tools/watchdog_add.py` | Register / deregister / list watched jobs; `touch` = heartbeat. The ONLY writer of `updated_at`. |
| `data/watchdog_state.json` | The watch list (gitignored). Schema documented in `watchdog.py`'s docstring. |
| `data/watchdog_alerts.log` | Append-only alert history (per-condition deduped for `realert_h`, default 6 h). |
| `data/WATCHDOG_ALERT` | **The flag file.** (Re)written on every pass with ≥1 active alert — even if the log line was deduped. The orchestrating agent checks for this file at the start of every session, triages via the log, and **deletes it after handling. The watchdog never deletes it.** |
| `tests/test_watchdog.py` | CPU-only unit tests, fake pollers injected (no ssh). |

## Running the loop

One pass ≈ one ssh round-trip; run it every 15 min under tmux (preferred — inspectable)
or nohup, from the campaign root:

```bash
cd /raid/benson/git/agentic-lensing/reproductions/claude-giga-lens-linus

# tmux (preferred)
tmux new -s watchdog -d \
  "while true; do python3 tools/watchdog.py; sleep 900; done"

# or nohup
nohup bash -c 'while true; do python3 tools/watchdog.py; sleep 900; done' \
  >> data/watchdog_loop.log 2>&1 &
```

Exit code: `0` clean, `2` alert(s) active (the loop ignores it; scripts can use it).
The loop itself can die too — that is what `STATE_STALE` + the session-start check of
`data/WATCHDOG_ALERT` are for: the agent's session ritual is
(1) look for `data/WATCHDOG_ALERT`, (2) `python3 tools/watchdog.py` once by hand,
(3) `python3 tools/watchdog_add.py touch`, (4) confirm the tmux/nohup loop is alive.

## Registering jobs (immediately after every sbatch)

```bash
# perlmutter slurm job (jobid from sbatch output; on-stall path is the REMOTE .slurm path)
python3 tools/watchdog_add.py add --jobid 55885270 --host perlmutter \
    --max-pending-h 24 --max-run-h 6 \
    --expect-artifact data/e2_v3b_low.npz \
    --on-stall 'resubmit:/global/homes/g/gdbenson/foundry-i/slurm/p1c_v3b.slurm' \
    --note 'P1c v3b money product'

# phoenix local run (jobid = PID; e.g. after: nohup python 04_smc.py & echo $!)
python3 tools/watchdog_add.py add --jobid 412233 --host phoenix --max-run-h 12 \
    --expect-artifact data/smc_b0_report.json --note 'B0 gates on L4'

python3 tools/watchdog_add.py list
python3 tools/watchdog_add.py remove --jobid 55885270   # after harvesting results
python3 tools/watchdog_add.py touch                     # heartbeat, every session
```

Set `--max-run-h` to the slurm `-t` walltime (+ a small margin) and `--max-pending-h`
from queue experience (shared-QOS cosmo_g jobs historically start within hours; a day
pending = something is wrong — recall the queue-stuck 55703707 node job).
Jobs that finish with their `--expect-artifact` present are auto-deregistered.

## Alert taxonomy — what each one means, what to do

| Alert | Meaning | Operator action |
|---|---|---|
| `PENDING_TOO_LONG` | Queued > `max_pending_h`. Queue-stuck (the 55703707 pattern) or priority problem. | `squeue --me` on perlmutter; consider cancel + resubmit shared-QOS / re-nice. No auto-resubmit (job still alive). |
| `RUNNING_TOO_LONG` | Running > `max_run_h`. Hung (74-dim live-point stall pattern) or walltime bomb burning budget. | Inspect job log; `scancel` if hung. No auto-resubmit (job still alive). |
| `JOB_FAILED` | sacct terminal-bad state — `OUT_OF_MEMORY` (the P1c signature), `FAILED`, `TIMEOUT`, `CANCELLED`, `NODE_FAIL`, ... | Read the detail (raw state rides along). Fix cause (e.g. particle count), then resubmit — command already emitted if `on_stall=resubmit:`. Ledger row BEFORE reading results. |
| `COMPLETED_NO_ARTIFACT` | Slurm says COMPLETED but the expected output file never appeared. | Silent in-job failure — read the job log before trusting anything. |
| `VANISHED_NO_ARTIFACT` | In neither `squeue` nor `sacct` (or PID gone) and no artifact. Typo'd jobid, sacct window expired, or process killed. | Verify jobid; check for partial outputs; resubmit if genuinely lost. |
| `UNREACHABLE` | Poll itself failed — **ssh cert expired**, network, timeout. Job status UNKNOWN (never treated as vanished; never auto-resubmitted). | Renew the NERSC cert (`sshproxy`), re-run a pass by hand. |
| `STATE_STALE` | Nobody ran add/remove/touch for > `stale_state_h` (default 48 h) while jobs are watched. **The dead-agent tripwire — this is the alert that would have caught P1c.** | Is the orchestrating agent alive? Resume the campaign session; `touch` after triage. |
| `STATE_CORRUPT` / `CONFIG_ERROR` | Unparseable state file / bad entry (unknown host, bad timestamp). | Fix `data/watchdog_state.json` by hand or re-register. |

Alerts print to stdout, append to `data/watchdog_alerts.log` (deduped per
(host, jobid, condition) for `realert_h`), and (re)write `data/WATCHDOG_ALERT` on
every pass while active. A job with an unresolved alert stays on the watch list and
re-flags forever until you fix it or `remove` it — that is the point.

## Resubmit semantics (dry-run by default)

`--on-stall 'resubmit:<path>'` attaches the exact resubmit command to **dead-job**
alerts only (`JOB_FAILED`, `VANISHED_NO_ARTIFACT`, `COMPLETED_NO_ARTIFACT` — never to
`*_TOO_LONG`, where the job still exists and resubmitting would double-submit):

- perlmutter → `ssh gdbenson@perlmutter.nersc.gov 'sbatch <remote .slurm path>'`
- phoenix → `bash <local script path>`

**By default the command is EMITTED, never executed** (an auto-resubmit spends
A100-h; the ledger row must exist before results are read — CAMPAIGN.md house rule).
Running the loop as `python3 tools/watchdog.py --execute` enables real resubmission,
capped at `max_resubmits` per entry (default **1**, to prevent OOM-crash loops — the
P1c canary would have OOM'd identically on a blind resubmit). On a successful
`--execute` resubmit the entry is rebound to the new jobid automatically.
Recommended posture: dry-run loop; let the agent execute emitted commands after triage.

## Tests

```bash
/raid/benson/.venvs/cgl2/bin/python -m pytest tests/test_watchdog.py -v   # 26 tests, no ssh
```

Fake pollers are injected via `run_pass(pollers=...)`; a subprocess "bomb" fixture
proves dry-run never executes anything. Covers: pending/running stalls, vanished ±
artifact, OOM signature, stale/corrupt state, resubmit emission + execute cap,
UNREACHABLE, log dedup + flag re-touch idempotence, slurm state/elapsed parsers.
