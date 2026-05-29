# Plan: Next steps for `redshifty` and `codecs`

## Context

Both repos are now installed and smoke-passing on the `/raid/benson` aarch64 host (recorded in `plans/SPECTRUMFM_LOCAL_SETUP.md` and `memories/memory/reference_spectrumfm_local_env.md`). What we actually have so far is just the toolchain — 247 spectra, 20 codecs training steps, 3 redshifty epochs at 50 spectra. None of that is enough to claim anything about the actual SpectrumFM architectural ideas.

Where the project itself stands (from reading `redshifty/PRODUCTION_RUN_PLAN.md` + the full `redshifty/RESEARCH_LOG.md` through Phase 12):

- **redshifty V2 tokenizer** trained to `val_recon=0.157` (8.6× better than V1) on 1.7M spectra. ConvNeXt + LFQ + U-Net skips + cross-attention + tophat + entropy loss. That checkpoint exists at NERSC, not on this box.
- **redshifty transformer** Approach A reached `val/redshift_acc=73.8% TF / 55.0% AR` at step 4000 with V1 tokenizer + `weight=50, mask=0.50, batch=32`. The author's `PRODUCTION_RUN_PLAN.md` lists "Phase 3: long transformer run with V2 tokenizer" as the next OPEN item.
- **codecs** (Mamba3 + Residual FSQ) is the cosmologyfoundation org's architectural successor to redshifty's V2 tokenizer (ConvNeXt + LFQ). It builds + runs on aarch64 (we verified) but has never been trained to a meaningful loss.
- The two repos are **not yet wired together** — redshifty's `--tokenizer_ckpt` expects a `SpectrumTokenizer` state-dict, not a codecs `Codec`. Their quantizer outputs are also structurally different (single 1024-vocab token vs. hierarchical RFSQ `[5,4,4]` codes).

Greg is **agentic-AI tooling lead** on the SpectrumFM proposal. His named responsibility is infrastructure that lets the team launch and analyze experiments — not running every experiment himself.

The user picked all three tracks (architectural integration, agentic-AI tooling, medium-real training) and asked for one concrete first project per track plus how they compose. NERSC bridge is explicitly out of scope for this plan.

---

## Track recommendation order

**Run them in this order** because each makes the next more valuable:

1. **Track 2 first (tooling MVP)** — 3–5 days. Pays off across every subsequent experiment in both other tracks. Without it, Tracks 1 and 3 generate ad-hoc shell history.
2. **Track 3 next (scale to medium-real)** — 3–5 days. Gives us credible local baselines (V1 tokenizer val_recon, Approach A z_acc) without needing NERSC. Required for Track 1 to be meaningful.
3. **Track 1 last (codecs ↔ redshifty integration)** — 1–2 weeks. The architectural comparison only makes sense once we have a real codecs checkpoint trained on real data (Track 3) and the harness to run + analyze it (Track 2).

If you'd rather run them in parallel: Tracks 2 and 3 can go simultaneously; Track 1 has to wait on both.

---

## Track 1 — Architectural integration: codecs ↔ redshifty

**Why:** Tests the central SpectrumFM Phase-I architectural claim — that a discrete-token Mamba3+RFSQ tokenizer produces better encoder representations than ConvNeXt+LFQ. This is exactly the kind of result the team needs to publish at the end of the 9-month Phase I.

### First project (1–2 weeks): Build the `CodecsTokenizerAdapter`

The adapter is the smallest viable bridge between the two repos. It lets `redshifty/scripts/train.py --tokenizer_ckpt` consume a checkpoint trained by `codecs/scripts/train.py`.

**Core design issue to resolve before coding** (~½ day, with the team):
- redshifty's `SpectrumTokenizer.encode()` returns shape `(B, T)` token indices over a single 1024-code vocabulary.
- codecs `Codec.encode()` returns `(z_q, codes_list, scalar_commit)` where `codes_list` is a list of three tensors (one per RFSQ level, vocab sizes `[5, 4, 4]`).
- Two reasonable mappings:
  - **(a) Flatten** the 3 RFSQ codes into a single compound 5×4×4=80-code "super-token" per position. Drops the residual-hierarchy structure but preserves the redshifty transformer interface verbatim.
  - **(b) Multi-token expansion**: treat each RFSQ level as its own token in the transformer sequence (sequence length triples). Preserves the hierarchy but means changing redshifty's tokenizer interface + retraining the embedding layer.
- Recommend (a) for the first comparison — minimal blast radius, head-to-head test of "is the codecs encoder representation better, all else equal." Reserve (b) as a Phase 2 ablation if (a) shows codecs is competitive.

**Code to write:**
- `tools/spectrumfm/codecs_adapter.py` — `CodecsTokenizerShim(nn.Module)` that wraps a loaded codecs `Codec`, exposes `.encode(spectrum) -> (B,T)` matching redshifty's `SpectrumTokenizer.encode()` interface. Internally calls `Codec.encode()` and applies the chosen flatten map.
- `tools/spectrumfm/load_codecs_checkpoint.py` — `load_codec(ckpt_path, config_path) -> Codec` that materializes a Codec from `codecs/configs/*.yaml` + a state-dict.
- One-line patch to `redshifty/scripts/train.py` (or a parallel `train_with_codecs.py`) to detect codecs-style checkpoints and instantiate via the adapter instead of `SpectrumTokenizer`.

**Critical files to read/modify:**
- `redshifty/src/tokenizers/spectrum.py:194` — `SpectrumTokenizer` reference interface
- `redshifty/src/tokenizers/spectrum_v2.py` — V2 reference (better baseline to compare against)
- `codecs/models/model.py:66` — `Codec` class (`encode`, `decode`, `forward`, `perplexity`)
- `redshifty/scripts/train.py:280–290` — where `--tokenizer_ckpt` is loaded; add a branch on file signature

**Verification:**
- Adapter unit test: pass a random `(B=2, T=8192, C=2)` flux tensor through the shim, confirm output shape matches `(B, T_redshifty)` and dtype matches redshifty's expectations.
- End-to-end: run `redshifty/scripts/train.py --approach a --tokenizer_ckpt <codecs.pt> --epochs 3` against a small data subset, confirm loss is finite and decreases.
- Comparative: same data + same transformer hparams, run with (i) random-init `SpectrumTokenizer`, (ii) V2 LFQ tokenizer if accessible, (iii) codecs RFSQ tokenizer via shim. Plot val_redshift_acc curves; the headline result is the relative ordering.

### Roadmap context (Track 1 follow-ons)

- **Phase B (1 week):** RFSQ multi-token expansion variant (option b above) — if Phase A's flatten loses too much, try preserving the hierarchy.
- **Phase C (1–2 weeks):** Multi-arm comparison study with consistent metrics: `val/redshift_acc` (TF + AR), `val/spectrum_acc` (TF + AR), `val_recon`, codebook utilization, perplexity. Feeds into the Phase-I writeup.
- **Phase D (open-ended):** Architectural ablations within codecs — vary `rfsq_levels`, `latent_dim`, Mamba `d_state`, etc. Only worth it once Phase C confirms RFSQ wins.

---

## Track 2 — Agentic-AI tooling harness

**Why:** Greg's named role on SpectrumFM. The MVP is *not* an "AI agent that proposes hparam changes" — that's the eventual destination. The MVP is structured experiment metadata + a launcher that's used by humans now and by agents later.

### First project (3–5 days): Experiment runner CLI + structured log

A wrapper that takes a yaml experiment spec, launches the underlying train script, captures structured metrics, writes a markdown summary in the style of `RESEARCH_LOG.md`. Same interface for redshifty and codecs.

**Code to write:**
- `tools/spectrumfm/exp_run.py`:
  - Reads an `experiment.yaml` (repo + script + script args + dataset spec + tags).
  - Resolves the right venv (`~/.venvs/redshifty` or `~/.venvs/codecs`) from the `repo` field.
  - Launches via `subprocess` (torchrun where needed), captures stdout to a per-run log file under `experiments/<run_id>/`.
  - Parses the standard metric-print lines both repos emit (`step <N> train_loss=... val_nll=...` for codecs; `Epoch N: train_loss=... val_loss=...` for redshifty).
  - Writes structured `metrics.jsonl` (one line per logged step or epoch).
  - On exit, emits `summary.md` with: run config, peak/final metric, runtime, GPU(s) used, link to log.
- `tools/spectrumfm/exp_analyze.py`:
  - `--compare <run_id> <run_id> ...` produces a side-by-side markdown table + a single matplotlib PNG of val curves.
  - `--digest <run_id>` produces a single-paragraph natural-language summary of the run (peak metric, time-to-best, whether it converged), formatted to drop into RESEARCH_LOG.md.
- `experiments/specs/` — checked-in yaml templates: `redshifty_smoke.yaml`, `redshifty_approach_a_medium.yaml`, `codecs_smoke.yaml`, `codecs_medium.yaml`.

**Critical files to read:**
- `redshifty/RESEARCH_LOG.md` lines 1–500 — the style + structure to emulate for `summary.md` output. The author has settled on a recognizable format (setup, trajectory table, analysis, files, next steps).
- `redshifty/nersc/_wandb_util.py` — `init_wandb` / `wlog` patterns to optionally integrate with.
- `redshifty/scripts/train.py:373–390` — what the metric print lines look like.
- `codecs/scripts/train.py:323–340` — same for codecs.

**Verification:**
- Run `exp_run.py experiments/specs/redshifty_smoke.yaml` and `exp_run.py experiments/specs/codecs_smoke.yaml`; both should produce a `summary.md` whose metrics match what we already saw in the smoke runs from the install task (Approach A loss 17.9→16.6, codecs train_loss 941→12).
- Run `exp_analyze.py --compare <redshifty_run_id> <codecs_run_id>`; output should be a one-page comparison.
- Manual: paste the `--digest` output into a scratch markdown file; it should be drop-in compatible with RESEARCH_LOG.md formatting.

### Roadmap context (Track 2 follow-ons)

- **Phase B (1 week):** Multi-run scheduler. Take a sweep spec (e.g. `lr ∈ {1e-4, 2e-4, 4e-4} × mask ∈ {0.30, 0.50}`), expand to N concrete runs, manage GPU allocation across the 10 local GPUs.
- **Phase C (1–2 weeks):** Decision-logging integration. After a sweep finishes, automatically pick the winner against a stated criterion (e.g. peak `val/redshift_acc`) and append a structured "Decision" section to RESEARCH_LOG.md. This is where "agentic" starts to mean something — the agent has to argue why one config won.
- **Phase D (open-ended):** Hparam-proposal agent — given a RESEARCH_LOG history, propose the next experiment to run. Phase D is what the proposal funds; everything before it is the substrate.

---

## Track 3 — Scale up local training to medium-real

**Why:** 247 spectra at 20 steps tells us the code compiles. It tells us nothing about whether the local box can produce a credible baseline. Track 1 needs real comparisons — that means real data, real training. NERSC isn't available yet; this is the closest local analogue.

### First project (3–5 days): 50–200 healpix → reproduce V1-scale baselines

Pull DESI EDR data up to the same scale Phase 8/9 used (~200 healpix, ~400k spectra) on the local box, then train both repos to credible loss numbers.

**Data work:**
- Extend `lensing-repos/redshifty/scripts/download_desi_batch.py` to support `--n-files 200` resumably (it currently uses a flat `requests.get`; add `if filepath.exists() and ok` early-skip + a `--resume` flag).
- A companion `download_redrock_batch.py` (or fold into the existing one) to fetch the matching `redrock-*.fits` files. Our install task did this inline in a one-liner shell loop; promote to a real script.
- Extend `/raid/benson/data/desi_dr1_mini/build_mini.py` to scan more pixels and keep the catalog → symlink invariant. Watch disk: 200 healpix at EDR scale is ~10–20 GB.

**Training work:**
- **redshifty V1 tokenizer reproduction:** Run `redshifty/nersc/pretrain_tokenizer.py` (it's single-GPU AMP, will work locally with minor path tweaks) for ~20k steps. Target: `val_recon ≤ 2.0` (V1 hit 1.35 at 16.5k steps on the same data scale). Wallclock estimate: 5–10h on one A16.
- **redshifty Approach A:** With the resulting tokenizer ckpt, run `train.py --approach a --epochs <N>` against the larger data, weight=50, mask=0.50, batch=32, lr=4e-4. Target: `val/redshift_acc ≥ 30%` (Phase 10 reached 73.8% at NERSC scale; 30% is the "this is working" floor on our local subset).
- **codecs medium run:** Bump the smoke yaml to `max_steps=10000, batch_size=8, val_size=500`. Target: `val_r2 > 0.5` (smoke run ended at val_r2=-0.22 after 20 steps, so the actual learning happens in the next ~9980 steps).

**Critical files to read/modify:**
- `redshifty/scripts/download_desi_batch.py` — extend
- `redshifty/nersc/pretrain_tokenizer.py` — runs locally with minor `--scratch-out` / `--cfs-out` path changes (defaults to `$SCRATCH` env var)
- `redshifty/nersc/dr1_dataset.py` — already manifest-based; build a local manifest with `nersc/build_dr1_index.py --root /raid/benson/data/desi_dr1_mini`
- `/raid/benson/data/desi_dr1_mini/codecs_smoke.yaml` — clone to `codecs_medium.yaml` with bumped step count

**Verification:**
- After the 200-healpix download: confirm disk usage <30 GB, `wc -l manifest.jsonl` matches what `build_dr1_index.py` claims.
- After tokenizer training: confirm `val_recon < 2.0` and the LFQ codebook utilization is non-degenerate (>10% of codes active).
- After Approach A training: confirm `val/redshift_acc ≥ 30%` and AR eval (`val_ar/redshift_acc`) is within 20 pp of the TF metric — same honesty check the redshifty author uses.
- After codecs medium: confirm `val_r2 > 0`, `val_rfsq_perplexity > 1`, reconstruction plots from `visualize.py` show recognizable spectrum shapes (not noise).

### Roadmap context (Track 3 follow-ons)

- **Phase B (variable):** 1000-healpix scale (~2M spectra) — matches redshifty V2 Stage 1 scale. Tests whether the local 10-GPU box can run DDP4 for these training runs in reasonable wallclock (~12h vs NERSC's 6h on 4× A100).
- **Phase C:** Full DR1 EDR pull (~9M spectra, ~300 GB) is probably too big for local — that's the moment to actually need NERSC. Knowing this boundary up front is itself useful for the Phase-II proposal narrative.

---

## How the three tracks compose

```
Track 2 (tooling MVP)
        │
        │  every experiment runs through exp_run.py / exp_analyze.py
        ▼
Track 3 (medium-real data + training) ─────► V1 tokenizer ckpt @ local
                                              local Approach A baseline
                                              codecs medium ckpt
        │
        │  Track 1 needs real codecs ckpt + a real redshifty baseline
        ▼
Track 1 (codecs ↔ redshifty adapter)  ─────► first comparison of
                                              RFSQ Mamba3 vs LFQ ConvNeXt
                                              as redshifty's frozen tokenizer
```

Once all three tracks have their first project done, the team has:
1. A working experimental harness that produces RESEARCH_LOG-compatible outputs (Track 2)
2. Credible local baselines for both tokenizers (Track 3)
3. The first head-to-head architectural comparison testing the SpectrumFM Phase-I claim (Track 1)

That's the substrate for everything else — agentic experiment-proposal (Track 2 follow-on), full-DR1 NERSC runs when ERCAP lands, and the Phase-II proposal narrative.

---

## Critical files / paths referenced across all three tracks

- `lensing-repos/redshifty/` — `src/tokenizers/{spectrum,spectrum_v2}.py`, `scripts/{train,download_desi_batch}.py`, `nersc/{pretrain_tokenizer,train_transformer,dr1_dataset,build_dr1_index}.py`, `RESEARCH_LOG.md`, `PRODUCTION_RUN_PLAN.md`
- `lensing-repos/codecs/` — `models/model.py`, `scripts/{train,data,visualize}.py`, `configs/{test,train}.yaml`, `data/desi.py`
- `/raid/benson/data/desi_dr1_mini/` — local data staging area, `build_mini.py`, `codecs_smoke.yaml`
- New tooling (Track 2 lands here): `tools/spectrumfm/{exp_run,exp_analyze,codecs_adapter,load_codecs_checkpoint}.py`
- New experiment specs: `experiments/specs/*.yaml`, `experiments/<run_id>/` for outputs

## Out of scope

- **NERSC bridge** — explicitly skipped per user scoping. Revisit when ERCAP allocation lands.
- **Galaxy-search repo** — already skipped during the install task; nothing changes here.
- **Hsu 2025 downstream lens-ID probe** — premature until we have a credibly-trained encoder. Natural Track 1 Phase D follow-on once the architectural comparison is settled.
- **Full DR1 pretraining** — local box can't hold ~300 GB of spectra; needs NERSC.
