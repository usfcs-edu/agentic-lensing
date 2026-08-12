# agentic-lens-discovery — Claude Code JWST archive strong-lens search (paper)

NOVEL (not a reproduction). Research paper describing the first archive-wide
galaxy-scale strong-gravitational-lens search of public JWST NIRCam imaging
conducted end-to-end by a general-purpose agentic AI (Claude Code): pipeline
design in plan mode, code, census, cutouts, vision inspection of all 5,391
targets, three-persona adversarial verification, blind Einstein-radius
measurement, literature crossmatch, and a ranked top-100 candidate list.

Run by Nathan Kvinnesland 2026-08-07 → 2026-08-09 (Claude Code v2.1.224,
claude-opus-5). Pipeline + results repo:
<https://github.com/kvinneslandn-ML-AI/jwst-strong-lens-search>.

## Layout

- `papers/` — the research article (tech-report style; AASTeX port planned for
  submission). `make pdf` builds `main.pdf`.
- `assets/` — source material (gitignored bulk): per-agent subagent transcripts
  (3 sessions, 1,281 JSONL files, 2.2 GB uncompressed; also as
  `claude_subagent_transcripts.tar.gz`), `PLAN.md` (the agent-authored search
  plan), `MANIFEST.csv` (per-transcript index), top-100 results files, the
  independent crossmatch audit findings, contact sheet, Slack export, and the
  PI paper plan.
- `assets/analysis/` — transcript-mining and figure scripts for the paper's
  "how the agent reasoned" section (§5) and cost accounting.

## Headline numbers (from the run repo + independent audit)

- 6,848 public NIRCam calib-3 obs (>1000 s), 916 fields, 4.48 deg² unique sky.
- 5,391 r<21 LS-DR10/DR9N elliptical targets (complete population), 100%
  inspected; 31 COWLS lenses injected blind, 15 recovered by inspection (48%).
- 2,024 flagged → 350 adversarially verified (3 personas) → grades A 5 / B 5 /
  C 12 / D 328.
- Independent crossmatch audit of the top-100: 10 published lenses (incl. rank 1
  = SL2S J02176-0513), 24 in known cluster strong-lensing fields, 66 with no
  prior identification; clean grade-A/B unknowns: ranks 2, 8, 10.
- ~50M billable tokens, 1,067 subagents; ~21% wasted on usage-limit kills.
