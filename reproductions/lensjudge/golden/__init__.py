"""lensjudge.golden — the golden single-grader dataset (Xiaosheng Huang, JWST NIRCam).

One named expert grades a curated, blind, shuffled frame of lens candidates (Paper-II 1-4
score + L/M/H sureness) with hidden byte-identical repeats, giving the program its first
measured intra-rater human ceiling and a per-rater label set for (i) in-context few-shot
exemplars, (ii) open-weight SFT, (iii) the pre-registered align/validate experiment.
Plan: ~/.claude/plans/i-want-to-explore-golden-seal.md (approved 2026-08-22).
"""
