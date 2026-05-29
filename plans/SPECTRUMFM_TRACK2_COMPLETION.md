# Track 2 — Agentic-AI tooling MVP — completion record (2026-05-26)

Track 2 first project from `plans/SPECTRUMFM_NEXT_STEPS.md` is done. This file is the as-built record: what landed, what works, what was deferred.

## What landed

```
tools/spectrumfm/
  exp_run.py          # yaml → subprocess launcher + metric parser + summary writer
  exp_analyze.py      # --digest <run> and --compare <runs...> CLIs
  README.md           # spec schema, quick-start, parser extension points

experiments/
  specs/
    redshifty_smoke.yaml   # 3-epoch Approach A+B smoke
    codecs_smoke.yaml      # 20-step Mamba3+RFSQ smoke
  runs/
    20260526_221308_redshifty_smoke_utjhx4/   # 6 records, 429 s
    20260526_222055_codecs_smoke_vx7ngz/      # 10 records, 38.8 s
    _comparisons/
      smoke_redshifty_vs_codecs.{md,png}
```

## How to use it

```bash
PY=/raid/benson/.venvs/redshifty/bin/python
cd /raid/benson/git/agentic-lensing

# launch
$PY tools/spectrumfm/exp_run.py experiments/specs/<spec>.yaml

# analyze
$PY tools/spectrumfm/exp_analyze.py --digest experiments/runs/<run_id>
$PY tools/spectrumfm/exp_analyze.py --compare experiments/runs/<run_a> experiments/runs/<run_b> \
    --out experiments/runs/_comparisons/<label>.md \
    --plot experiments/runs/_comparisons/<label>.png
```

The harness itself runs in the **redshifty venv**, which has `pyyaml` and `matplotlib`. It then launches subprocesses into whichever venv (`~/.venvs/<repo>/bin/python` or `…/bin/torchrun`) the spec's `repo:` field selects.

## Verified behavior

Numbers below match the install-task baselines exactly, confirming the parser captures real metrics rather than re-deriving them.

| Run | Records | Wallclock | Headline trajectory |
|---|---|---|---|
| `redshifty_smoke` | 6 (3 epochs × 2 approaches) | 429 s | Approach A: `train_loss 20.4 → 17.2`. Approach B: `train_loss 20.3 → 17.2`. Both `val_redshift_acc = 0` (expected at this scale). |
| `codecs_smoke` | 10 (step 2…20) | 38.8 s | `train_loss 941 → 12.3`, `val_nll 162 → 7.9`, `val_r2 −7.7 → −0.22`. exp_analyze digest: *"Loss is dropping clearly — training is learning."* |

## Spec yaml schema

```yaml
name: <slug>                    # appears in run_id + summary heading
repo: redshifty | codecs        # selects venv + cwd
description: <free text>        # appears in summary.md
tags: [<tag>, ...]              # arbitrary; appears in summary.md

launch:
  command:                      # command[0] resolved against <venv>/bin/
    - python                    # bare 'python', 'torchrun', etc.
    - scripts/train.py
    - --approach
    - a

env:                            # merged into subprocess env on top of os.environ
  WANDB_MODE: offline
  CUDA_VISIBLE_DEVICES: "0"
  TORCHDYNAMO_DISABLE: "1"      # codecs needs this; see SPECTRUMFM_LOCAL_SETUP.md
```

## Parser internals (for future extension)

- **codecs**: stateless line regex `CODECS_STEP_RE` matches the `  step N train_loss=… lr=… val_nll=…` lines emitted by `codecs/scripts/train.py:329–337`. Every match is one jsonl record.
- **redshifty**: 3-state machine. Header `Epoch N/M (Xs)` sets state, then `  Train: loss=…, acc=…` accumulates, then `  Val:   loss=…, acc=…` emits the record. Separator line `Training Approach A|B` (from `train.py:357`, NOT `smoke_test.py`'s post-hoc `SMOKE TEST: Approach X`) updates an `approach` field on subsequent records.

If you add a new training script with a different print format, extend `MetricParser._emit_<repo>` accordingly. The current implementation is in `tools/spectrumfm/exp_run.py` around lines 50–130.

## Deferred (Track 2 follow-ons, in priority order)

1. **Sweep scheduler (Phase B).** Take a sweep spec (`lr ∈ {1e-4, 2e-4, 4e-4} × mask ∈ {0.30, 0.50}`), expand to N concrete runs, manage GPU allocation across the 10 local devices. Required before Track 1's multi-arm tokenizer comparison.
2. **Decision logger (Phase C).** After a sweep finishes, pick the winner against a stated criterion (e.g. peak `val/redshift_acc`) and append a structured "Decision" section to `redshifty/RESEARCH_LOG.md` (or our own equivalent). The first place the harness starts looking "agentic" rather than just "scripted".
3. **Hparam-proposal agent (Phase D).** Given a RESEARCH_LOG history, propose the next experiment to run. This is what the SpectrumFM proposal actually funds; everything before is the substrate.

Minor polish items (do when convenient, not blocking):

- `--dry-run` creates the run_dir before short-circuiting — leaves empty dirs if used to debug specs. Move the mkdir below the dry-run check.
- `--compare` PNG plots epoch-indexed redshifty runs together regardless of approach; could color-code by `approach` field.
- `_headline_metric` is hardcoded per repo; once we have an honest AR metric on the local box, switch redshifty's headline from `val_loss` to `val/redshift_acc` (or expose via CLI flag).

## Next

Per `plans/SPECTRUMFM_NEXT_STEPS.md` the recommended sequence is **Track 3 next** (medium-real training; produces the V1 tokenizer ckpt + Approach A baseline + a real codecs val_r2), then Track 1 (codecs ↔ redshifty integration adapter, which needs the Track 3 outputs).
