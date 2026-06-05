# Euclid Q1 — Strong Lensing Discovery Engine (high-resolution benchmark for LensJudge)

External 0.1″ benchmark used by `reproductions/lensjudge` to test whether higher-resolution
imaging breaks the hard-pool grading wall. **The bulk data under `data/` is gitignored** (≈3.6 GB);
this README is the reproducibility pointer.

## Source
Euclid Collaboration, *Euclid Quick Data Release (Q1): The Strong Lensing Discovery Engine*,
Zenodo record **10.5281/zenodo.15025832** (<https://zenodo.org/records/15025832>).

- `q1_discovery_engine_lens_catalog.csv` — 2,584 lens candidates (grade A=309/B=267/C=2008),
  each with `right_ascension`, `declination`, `expert_score` (continuous), `grade`, and
  `expert_total_votes` (~10 independent expert votes/object). **This is a multi-grader, 0.1″ catalog.**
- `lens.zip` / `unsuccess.zip` / `group.zip` / `recenter.zip` — per-object cutouts (539 total with
  FITS). Each object: a multi-extension FITS with **VIS + NIR Y/J/H** (FLUX/PSF/RMS) on a common
  0.1″/px grid (300×300 = 30″ FoV), plus a full lens model (`result/`: SIE fit, MGE, Sérsic,
  source reconstruction). *Note: `unsuccess` = lens-modeling pipeline failed, NOT lens-rejected —
  all 539 are positive candidates; the release ships no non-lens cutouts.*
- `modeling_*.csv` — MGE / mass / Sérsic model tables.

## Fetch / rebuild
```bash
D=reproductions/euclid-q1/data/raw; mkdir -p "$D"; cd "$D"
for f in q1_discovery_engine_lens_catalog.csv lens.zip unsuccess.zip group.zip recenter.zip \
         modeling_lens_mge.csv modeling_lens_mass.csv modeling_lens_sersic.csv; do
  curl -C - -O "https://zenodo.org/records/15025832/files/${f}?download=1"
done
cd ..; for z in lens unsuccess group recenter; do python -c "import zipfile;zipfile.ZipFile('raw/$z.zip').extractall('.')"; done
```
Then from `reproductions/`:
`python lensjudge/eval/crossmatch_external.py` (DESI↔Euclid crossmatch) and
`python lensjudge/eval/run_euclid.py --mode paired|rank` (grade on Euclid pixels).

## Headline result (see lensjudge tech report §"External validation")
24 DESI candidates fall in this field; 53% of the DESI grade-C ones are graded A/B by the 10-expert
panel. Grading the *same* objects at DESI 1.3″ vs Euclid 0.1″, LensJudge's mean p_lens on the
grade-C subset rises **0.05 → 0.90** — the wall is resolution+label-limited, not algorithm-limited.
