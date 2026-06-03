---
name: project-spectrumfm-ws2-probe
description: WS2 frozen-encoder six-class probe — encoder learns STRONG class structure (macro-F1 0.81-0.85 vs 0.10 no-skill) even though it can't pin galaxy redshift; classification and z-precision are distinct capabilities
metadata:
  type: project
---
WS2 of the build-out (2026-06-02) DONE. Built `tools/spectrumfm/probe_six_class.py` (frozen `model.encode()` features, RZ token masked, mean-pool over SPECTRUM positions only `feats[:,2:-1,:]` excluding SOS/RZ/EOS; healpix-disjoint train/test via split_records_by_healpix; torch nn.Linear probe — sklearn NOT in ~/.venvs/redshifty so the auto torch fallback engages; class-balanced CE; train-majority-on-test baseline; `--smoke` CPU honesty check). Built+verified via Workflow (1 build + 3 adversarial Explore verifiers: leakage clean, frozen-feature clean, fixed a baseline-definition high-severity finding).

**Result (1250 healpix-disjoint test spectra, both checkpoints):** frozen-encoder linear probe macro-F1 **0.813 (V1) / 0.847 (V2-noskip)**, accuracy 0.86/0.89, vs no-skill baseline macro-F1 0.097 / acc 0.321 (+0.72-0.75 macro-F1). Per-class F1 (V2-noskip): MWS 0.95, BGS 0.92, ELG 0.90, LRG 0.87, QSO 0.61. Confusions are physically sensible: QSO↔ELG (emission-line), LRG↔BGS (red galaxies). V2-noskip marginally beats V1 on class (esp QSO 0.61 vs 0.53) though they TIE on redshift.

**KEY INSIGHT:** classification and redshift-precision are DISTINCT capabilities. The encoder learns strong, linearly-separable CLASS representations (supports the proposal's "one encoder, six classes" claim) while [[project-spectrumfm-ws1-per-class]] shows it lacks DESI good-z PRECISION for galaxies. The gap is fine-grained z regression, the target of scaling (WS3) + physical priors (WS4) — not representation quality. NOT committed yet. See [[project-spectrumfm-buildout]].
