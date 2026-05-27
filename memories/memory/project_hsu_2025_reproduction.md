---
name: project-hsu-2025-reproduction
description: Phase 2 reproduction of Hsu+2025 pair-wise spectroscopic DESI DR1 lens search — surprising facts about the algorithm, data layout, and what's reproducible
metadata:
  type: project
---

# Hsu et al. 2025 (arXiv:2509.16033) reproduction — non-obvious facts

Lives at `/raid/benson/git/agentic-lensing/reproductions/hsu-2025/` (scripts 01-08, mirroring foundry-i pattern). Algorithmic core (FoF + z-ratio cut) reproduces published intermediate counts within 2% on full 28M-fiber DR1; recovers 20/20 of Hsu Table 2 Grade A new candidates within 3″.

**Why:** Phase 2 of [[project-foundry-i-reproduction]]'s 8-phase roadmap; first time we reproduced one of Huang's lens-FINDER papers (vs. only modeling). Confirmation that we can drive the algorithmic side of the discovery program end-to-end on local /raid (no NERSC needed).

**How to apply:** when revisiting this work — paper write-up, scaling to DR2, or building DimpleScout (§9.3 of the onboarding plan) — these are the load-bearing facts that cost time to discover.

## Non-obvious facts

1. **The dimple class is MORPHOLOGICAL, not a σ_v cut.** §4.4 defines dimples by visual indentation features in DR10 imaging; Fig. 6 caption explicitly says "Velocity dispersion is not available for most of the dimple candidates". The 318 published count cannot be recovered algorithmically. The natural proxy (pairs lacking FastSpecFit σ_v on the lens) is a necessary but not sufficient condition.

2. **spherimatch is by Hsu himself.** github.com/technic960183/spherimatch is Y.-M. Hsu's own tool — they are reproducing their own data with their own released code. `pip install spherimatch==0.1`. API: `fof(catalog, tolerance_deg) -> FoFResult`. Tolerance is in **degrees** (3″ → 3.0/3600.0). The FoFResult's `get_group_dataframe()` returns a `MultiIndex(Group, Object)` DataFrame, not a column-keyed table — group_id is in the index level "Group", not a column.

3. **spherimatch scales sublinearly in our regime.** Synthetic 10k → 1M timing curve: t ∼ N^0.63. The full 15.8M-row FoF runs in **36 seconds**. Plan budgeted "8-24 h" — wildly conservative. Total pipeline (load + FoF) is ~2 min for 28M fibers.

4. **Algorithmic step ≠ published "11,848 pairs".** After FoF + z-ratio cut you get **13,218 groups / 26,621 spectra** (13,044 pairs + 165 triplets + 7 quartets + 2 quintets). The 11,848 number comes AFTER VI grading (§3.3) which converts triplets+ into representative pairs. Easy to misread the abstract as "11,848 pairs is the algorithmic output". The plan/onboarding-doc paraphrase had this conflated.

5. **DESI DR1 zcat layout gotchas:**
   - The catalog is `zall-pix-iron.fits` (22.4 GB), URL footnoted in Hsu §3.1.
   - Coordinate columns are `TARGET_RA` / `TARGET_DEC`, not `RA` / `DEC`.
   - `ZCAT_PRIMARY=True` is DESI's "longest-exposure coadd per TARGETID" flag — matches Hsu's §3.1 "retain only the spectrum with the longest effective exposure time" without writing custom logic.
   - `SPECTYPE != 'STAR'` (Redrock typing) is the right interpretation of Hsu's "TGT classified as star" filter — gives 15.8M (matches Hsu exactly). `OBJTYPE == 'TGT'` is wrong (gives 20.4M).
   - FITS columns are big-endian — wrap every column read in `np.ascontiguousarray(arr).astype(dtype, copy=False)` before pandas/pyarrow conversion, otherwise pyarrow raises "Byte-swapped arrays not supported".

6. **FastSpecFit DR1 v3.0 VAC layout:**
   - Lives at `data.desi.lbl.gov/public/dr1/vac/dr1/fastspecfit/iron/v3.0/catalogs/`.
   - Partitioned by survey × program × nside1 healpix → **36 files, ~79 GB total**.
   - VDISP lives in **HDU 2 (SPECPHOT)**, not HDU 1 (METADATA). TARGETID is in both, rows aligned 1:1.
   - `VDISP_IVAR == 0` indicates failed fit with default cap of 250 km/s — filter on `VDISP_IVAR > 0` for real measurements.
   - σ_v coverage on our 13,530-pair list: 31.3% (4,238 pairs). Lens σ_v median 217 km/s, IQR (152, 292) — matches Grade A range (242-485) in Table 2.

7. **Published catalog not yet released.** Paper Appendix A says machine-readable catalog is on "project website and Zenodo", but as of 2026-05-26 neither has it (paper is still in ApJS review). For now, recall validation uses the 20 Grade A candidates explicitly named in Table 2 of the paper text — 20/20 within 3″ in our pair list.

8. **Cosmology is FlatLambdaCDM(H0=70, Om0=0.3)** per §4.1 — not Planck18. Use `astropy.cosmology.FlatLambdaCDM` with `angular_diameter_distance` and `angular_diameter_distance_z1z2` for the SIS Einstein-radius formula (Hsu eq. 1).

## Files of record

- `reproductions/hsu-2025/data/dr1_pairs.parquet` — the 27,334-row / 13,530-group reproduced pair list
- `reproductions/hsu-2025/data/classified_pairs.parquet` — with σ_v + θ_E
- `reproductions/hsu-2025/papers/REPRODUCTION.md` — short markdown report
- `reproductions/hsu-2025/data/{smoketest_timings,sv3dark_stats,dr1_stats,xmatch_table2,classified_stats}.json` — all numerical artifacts
- New venv: `/home/benson/.venvs/hsu` (spherimatch + astropy + fitsio + healpy + pyarrow stack)

## What we did NOT reproduce

Hsu §3.3 (spectral VI quality grading H/M/R), §4 (imaging-based Grade A/B/C lens grading), and §4.4 (dimple morphological identification). Final counts of 2,046 conventional + 318 dimple require all three human-in-loop steps. Our "dimple proxy" column over-counts because it's necessary but not sufficient.
