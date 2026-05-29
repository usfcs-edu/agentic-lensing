# tools/spectrumfm — experiment-runner harness

Track 2 MVP of the SpectrumFM next-steps plan (`plans/SPECTRUMFM_NEXT_STEPS.md`).
The two SpectrumFM repos under `lensing-repos/` (`redshifty/` and `codecs/`) each
have their own conventions for launching training and printing metrics. This
harness wraps both behind a single yaml-driven interface so the same workflow
applies whether you're running a 30-second smoke or a multi-hour DDP job.

## What's here

| File | Purpose |
|---|---|
| `exp_run.py` | Take a yaml spec, resolve the right venv, launch the training script, tee stdout, parse metric lines into `metrics.jsonl`, write `summary.md`. |
| `exp_analyze.py` | `--digest <run>` (paragraph summary) and `--compare <run...>` (markdown table + matplotlib PNG of val curves). |

Specs live under `experiments/specs/`; outputs land under `experiments/runs/<run_id>/`.

## Quick start

```bash
# from the repo root
cd /raid/benson/git/agentic-lensing

# Run the harness (uses redshifty venv — has yaml + matplotlib + json)
PY=/raid/benson/.venvs/redshifty/bin/python

$PY tools/spectrumfm/exp_run.py experiments/specs/redshifty_smoke.yaml
$PY tools/spectrumfm/exp_run.py experiments/specs/codecs_smoke.yaml

# Inspect the latest run of each
RD_RED=$(ls -td experiments/runs/*_redshifty_smoke_* | head -1)
RD_COD=$(ls -td experiments/runs/*_codecs_smoke_* | head -1)
cat $RD_RED/summary.md
cat $RD_COD/summary.md

# Digest + compare
$PY tools/spectrumfm/exp_analyze.py --digest $RD_RED
$PY tools/spectrumfm/exp_analyze.py --compare $RD_RED $RD_COD \
    --out experiments/runs/_comparisons/my_compare.md \
    --plot experiments/runs/_comparisons/my_compare.png
```

## Spec yaml schema

Minimum viable spec:

```yaml
name: redshifty_smoke         # used in run_id and summary heading
repo: redshifty | codecs      # selects ~/.venvs/<repo>/ and the cwd
description: free text        # appears in summary.md
tags: [smoke, track2-mvp]     # arbitrary; appears in summary.md

launch:
  command:                    # command[0] is resolved against <venv>/bin/
    - python                  #   so use bare 'python', 'torchrun', 'pytest'
    - scripts/smoke_test.py   #   subsequent args are passed verbatim
    - --epochs
    - "3"

env:                          # merged into subprocess env on top of os.environ
  WANDB_MODE: offline
  CUDA_VISIBLE_DEVICES: "0"
```

Notes:

- `repo: redshifty` ⇒ venv `~/.venvs/redshifty`, cwd `lensing-repos/redshifty`
- `repo: codecs` ⇒ venv `~/.venvs/codecs`, cwd `lensing-repos/codecs`
- For codecs, the typical first arg is `torchrun` (also in the venv's bin/)
- `--dry-run` on `exp_run.py` prints the resolved argv without launching

## Run-directory layout

Each invocation of `exp_run.py` writes:

```
experiments/runs/<run_id>/
  spec.yaml         # frozen copy of the input spec
  command.txt       # the exact argv that was launched
  stdout.log        # captured stdout+stderr stream
  metrics.jsonl     # one json line per parsed metric record
  summary.md        # human-readable summary (markdown)
```

The `run_id` is `<UTC-timestamp>_<spec.name>_<short_random>` unless you
override with `--run-id`.

## Metric parsing

Per-repo line parsers in `exp_run.py` (class `MetricParser`):

- **codecs** — looks for `^\s*step\s+N\s+train_loss=...\s+lr=...\s+val_nll=...` lines emitted by `codecs/scripts/train.py:323–340` once per `log_interval`. Each step produces one jsonl record.
- **redshifty** — looks for the three-line epoch summary `Epoch N/M (Xs)` / `Train: ...` / `Val: ...` emitted by `redshifty/scripts/train.py:373–390`. Also tracks the `Training Approach A|B` separator so that smoke_test.py's two-approach output ends up tagged with an `approach` field.

If you add a new training script that prints metrics in a different format,
extend `MetricParser` (the two `_emit_*` methods) accordingly.

## Adding a new spec

1. Drop a yaml under `experiments/specs/`.
2. Run `exp_run.py --dry-run experiments/specs/your_spec.yaml` to confirm
   the resolved argv looks right.
3. Run for real.

For a medium-real (Track 3) spec, the common moves are: bump `max_steps`,
swap the `--config` path to a `codecs_medium.yaml` (or for redshifty, point
`--data_dir` at a larger pull), and add a `tags: [medium-real]` line so
later comparisons can group runs by tier.

## Out of scope (Track 2 follow-ons)

- **Sweep scheduling** (lr × mask grid → N parallel runs across the 10 local GPUs).
- **Decision logging** that appends a "winner" entry to `RESEARCH_LOG.md`.
- **Hparam-proposal agent** that reads a RESEARCH_LOG and suggests the next experiment.

The MVP here is the substrate everything else builds on. See
`plans/SPECTRUMFM_NEXT_STEPS.md` for the full Track 2 roadmap.
