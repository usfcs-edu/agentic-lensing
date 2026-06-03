---
name: project-spectrumfm-buildout
description: SpectrumFM Phase-I build-out roadmap — 4 workstreams toward the proposal's go/no-go metrics, hybrid local→NERSC
metadata:
  type: project
---
Approved 2026-06-02. Roadmap to "further build out and evaluate SpectrumFM" against the DOE Genesis proposal (one model, six classes: LRG/ELG/QSO/MWS label-rich + LBG/LAE few-shot). Full plan in `plans/fizzy-soaring-floyd.md`. Hybrid compute (prototype on 2×L4, emit NERSC-ready specs); balanced sequenced.

**Go/no-go metrics:** (1) per-class redshift parity vs Redrock, no class degrades >5%; (2) few-shot LBG/LAE within 10% of a specialist; (3) positive power-law scaling. Objective-3 downstream: strong-lens ID + SN typing.

**Workstreams (ordered):** WS1 per-class eval vs Redrock (LOCAL NOW — the one blocking change is extending `nersc/dr1_dataset.py` to carry SPECTYPE/ZWARN/ZERR/DELTACHI2 + DESI_TARGET/MWS_TARGET bits, which are confirmed present in local redrock files; fork `tools/spectrumfm/eval_redshift_dz.py` → `eval_per_class.py`). WS2 frozen-encoder linear probe / six-class capability. WS3 local scaling ladder + NERSC full-scale spec (extends PRODUCTION_RUN_PLAN.md). WS4 physical-prior reg (flag-gated) + alignment (DESI EDR VI VACs, public) + few-shot harness (proxy class; real LBG/LAE gated) + downstream (SN: Super-SNID/BSNIP; lens: SLACS/BELLS/lenscat — resample to DESI grid via the codecs_adapter pattern).

**Key prior finding to honor:** tokenizer architecture is NOT the bottleneck (V1≈V2-noskip≈codecs ~0.50-0.55) — do not re-open tokenizer work. New loss terms/capabilities are flag-gated default-off (no regression). See [[project-spectrumfm-v2-tokenizer-collapse]], [[feedback-spectrumfm-workflow-orchestration]], [[project-spectrumfm-tooling]].
