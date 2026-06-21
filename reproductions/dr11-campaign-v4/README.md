# DR11-South v4 Re-Sweep — report (scaffold)

Sibling to `../dr11-campaign/` (the v3 `v3blend8` *as-run, HSC-vetted* campaign). This report covers the
**ClaudeNet v4** release-native fine-tuned re-sweep of DR11-south.

**Status: SCAFFOLD.** `papers/main.tex` §1–3 are drafted from the completed re-sweep (mean combiner +
release-native fine-tune gate + full 53.8M re-sweep → v4 top-150k; held-out recall 0.87 / 0.825). §4
(candidate catalogue + LensJudge v3 HSC vetting of the v4 NEW shortlist) and §5 (conclusion) are **TODO**,
pending the vetting pass. Build: `cd papers && make pdf`.

The v4 method/recipe is documented in the ClaudeNet tech report (`../claudenet/papers/v4_section.tex`,
§"ClaudeNet v4"); this report is the DR11-south *application/catalogue*. Inputs (gitignored, regenerable):
`claudenet/data/v3/{survivors_dr11s_v4.parquet, resweep_v4_summary.json, dr11_finetune_gate.json}`
(scripts `claudenet/380`–`395`).
