---
name: project-spectrumfm-ws1-per-class
description: WS1 per-class redshift eval (Metric 1) — prototype FAILs galaxy/QSO good-z parity; only stars pass; catastrophic ordering tracks spectroscopic difficulty
metadata:
  type: project
---
WS1 of the build-out (2026-06-02) DONE. Built the per-class Metric-1 harness (all additive, default-off): `nersc/dr1_dataset.py` `return_labels=True` path + `collate_dr1_with_labels` (survey-conditioned target columns — sv3 rows MUST read SV3_DESI_TARGET, their bare DESI_TARGET is ~0); `tools/spectrumfm/desi_targets.py` pure bit→class decoder (DESI_MASK bits LRG=0/ELG=1/QSO=2/BGS_ANY=60/MWS_ANY=61, verified 97% vs SPECTYPE); `tools/spectrumfm/eval_per_class.py` (forks eval_redshift_dz.py; honest enc[:,1]=MASK readout; per-class catastrophic rate |dz|/(1+z)>=0.0033 on the ZWARN==0 set; stratified strided sampling across all val records; `--smoke` CPU mode).

**Metric-1 VERDICT = FAIL at prototype scale** (V1 and V2-noskip identical, ~1719 val spectra). Per-class catastrophic rate / median|dz|: MWS 2%/0.0002 (PASS) ≪ BGS 96%/0.036 < LRG 97.5%/0.063 < ELG 98%/0.111 < QSO 97-99%/0.18 (all FAIL). The ordering tracks spectroscopic difficulty exactly (stars z≈0 trivial → bright galaxies → LRG absorption → ELG single-line → QSO broad-line over z→3.7), which validates the harness. Aggregate good-z ~19-24%, reproducing eval_redshift_dz.py.

**Takeaway:** the prototype learned only COARSE redshift (60% within Δz<0.05) but lacks DESI good-z precision (<0.0033) for everything except stars. This is the honest small-scale baseline that motivates WS3 (scaling) + WS4 (physical-prior/alignment). Verdict logic votes over {LRG,ELG,QSO,MWS,BGS} with N≥30 floor; an MWS-only "pass" is flagged NOT parity (stars are trivial). NOT committed yet. See [[project-spectrumfm-buildout]], [[project-spectrumfm-v2-tokenizer-collapse]].
