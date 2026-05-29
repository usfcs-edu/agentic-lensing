---
name: project-spectrumfm-tooling
description: SpectrumFM Track 2 experiment-runner harness — what's at tools/spectrumfm/, parser quirks, and what's deferred
metadata:
  type: project
---

# SpectrumFM agentic-AI tooling — Track 2 MVP

Lives at `/raid/benson/git/agentic-lensing/tools/spectrumfm/`. Two CLIs (`exp_run.py`, `exp_analyze.py`) and a README. Seed specs at `experiments/specs/{redshifty,codecs}_smoke.yaml`; outputs land at `experiments/runs/<run_id>/{spec.yaml,command.txt,stdout.log,metrics.jsonl,summary.md}`.

**Why:** Greg's named role on the SpectrumFM proposal is "agentic-AI tooling lead". The MVP is *not* an agent that proposes hparam changes — that's the eventual Track 2 Phase D destination. The MVP is **structured experiment metadata + a launcher used by humans now, agents later**. Every Track 1 and Track 3 experiment is meant to run through this harness.

**How to apply:** when launching any non-trivial training run on either repo, write a yaml spec under `experiments/specs/` and run it through `exp_run.py` rather than `python scripts/...` directly. The harness pays for itself the moment you want to compare two runs.

## Non-obvious facts

1. **The harness itself runs in the redshifty venv**, not the codecs venv or a third one. Reason: it needs `pyyaml` + `matplotlib`, both of which the redshifty venv has; codecs has them too but redshifty is the cheaper venv to keep warm. The harness then *launches* subprocesses into whichever venv (`~/.venvs/<repo>/bin/python` or `…/bin/torchrun`) the spec's `repo:` field selects.

2. **`command[0]` is resolved against `<venv>/bin/`**, not the system PATH. This is why specs use bare `python` or `torchrun` rather than absolute paths — keeps specs portable across machines as long as the venv layout is the same.

3. **redshifty approach-tagging keys on `Training Approach A|B` (from `train.py:357`), NOT `SMOKE TEST: Approach A|B`** (from `smoke_test.py`). The smoke_test.py header lines appear *after* each child subprocess exits, so by the time the parser sees them the metric records have already been emitted untagged. Both run as `tail -f`-style line streaming; the train.py header is the only one that lives at the start of each phase.

4. **codecs step metric format is fragile to the leading space**: lines from `codecs/scripts/train.py:329–337` are emitted via `pbar.write(msg)` and have a leading two-space indent. The `CODECS_STEP_RE` accounts for this with `^\s*step`. If the codecs upstream ever switches to plain `print()` the regex still works; if it switches to no leading whitespace it still works.

5. **`--dry-run` on `exp_run.py` still creates the run_dir** and writes `spec.yaml` + `command.txt` before short-circuiting. Minor cleanup pain if you use `--dry-run` to debug spec yaml — leaves an empty run dir per probe. Worth fixing if the pattern shows up often.

6. **Smoke wallclocks (measured 2026-05-26 on /raid/benson, 1×L4):**
   - `redshifty_smoke`: 429 s. Dominated by loading 2,400 SV3 spectra *twice* (once per approach). The actual training is only ~6 s of that.
   - `codecs_smoke` (re-run): 38.8 s. The first run took 2 min for CUDA autotune; subsequent runs reuse the inductor cache.
   - Implication for Track 3 timing estimates: the redshifty data-load overhead is constant — bigger configs pay the same fixed cost.

7. **Headline metric for `--compare` is hardcoded per repo:** `val_nll` for codecs, `val_loss` for redshifty. If we add a new repo or a meaningfully different metric (e.g. `val_ar/redshift_acc` for honest redshifty) the `_headline_metric` function in `exp_analyze.py` needs updating.

8. **`smoke_test.py`'s 2-approach output produces 6 records in one run** (3 epochs × 2 approaches). The harness handles this correctly — `MetricParser._epoch_header/_epoch_train` reset between approaches when the `Training Approach X` line fires. But the `--compare` PNG plots all 6 epochs on the same x-axis without color-coding by approach; the markdown table correctly carries an `approach` column. Plotting hygiene is a Track 2 Phase B follow-on.

## Status as of 2026-05-26

- **Done:** `exp_run.py`, `exp_analyze.py`, `experiments/specs/{redshifty,codecs}_smoke.yaml`, both smoke runs verified end-to-end and producing parseable metrics that match the install-task baselines (`Approach A loss 20.4→17.2`; `codecs val_nll 162→7.9`).
- **Track 2 Phase B (deferred):** sweep scheduler that fans `lr × mask` across the 10 local GPUs.
- **Track 2 Phase C (deferred):** decision logger that appends a "winner" entry to RESEARCH_LOG.md after each sweep.
- **Track 2 Phase D (deferred, proposal funds this):** hparam-proposal agent reading a RESEARCH_LOG history and suggesting the next experiment.

Related: [[project-spectrumfm]], [[reference-spectrumfm-local-env]], [[user-role-benson]].
