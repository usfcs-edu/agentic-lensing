# Foundry II reproduction — DESI Strong Lens Foundry II (Huang et al. 2025b)

Paper: `papers/Huang_2025b_DESI_Foundry_II.pdf` (arXiv:2509.18089). venv: `hsu`.

**Goal.** Re-derive the lens/source redshifts and lens velocity dispersions for
the 73 Foundry II EDR systems (Table 2) from the *public* DESI DR1 (Iron)
products already on disk, and quantify how well they match the published values.

**Caveat (EDR vs DR1).** Foundry II used DESI EDR (Fuji). DR1/Iron is a superset
that re-reduces the same SV/EDR tiles **plus** Year-1 main-survey tiles. Most
EDR fibers reappear in Iron (often re-observed on main-survey tiles too), but a
few EDR-only / special-tile fibers and VI-corrected source redshifts are not
reproduced by the DR1 automated pipeline — these are the expected misses.

## Pipeline
- `01_parse_table2.py` — pdfplumber per-page extraction of Table 2 (73 rows),
  parse DESI names (RA/Dec are decimal degrees, 4 dp), z_d, z_s, sigma_v, VI
  flags, section. -> `data/foundry_ii_table2.csv`
- `02_crossmatch_dr1.py` — box-prefilter + astropy match to
  `zall-pix-iron.fits` (28.4M rows). Closest fiber within 1.5" = lens; wide 5"
  radius recovers the offset lensed-source fiber. -> `data/foundry_ii_dr1_crossmatch.csv`
- `03_sigmav_fastspecfit.py` — match lens-fiber TARGETIDs to the 40 on-disk
  FastSpecFit Iron `*.sigmav.parquet` shards, keep VDISP where VDISP_IVAR>0.
  -> `data/foundry_ii_sigmav.csv`
- `04_build_report.py` — merge + figure. -> `data/foundry_ii_master_comparison.csv`,
  `figs/foundry_ii_recovery.png`

## Results (parsed table reproduces the paper's stated targets exactly)
- 73 systems: 20 confirmed, 1 known, 34 pending-source, 13 pending-zs, 1 pending-both, 4 nonlens.
- Published: 72/73 z_lens, 22/36 source spectra (22 z_source values in table), 71 sigma_v.
- sigma_v range 132–609, median ~292, mode bin ~300 km/s (matches paper).

## Recovery from DR1
- **73/73** systems have a DESI fiber within 1.5" in DR1 (full positional coverage).
- **z_lens: 70/72** recovered to |dz|<0.005 (median |dz|=3e-5, all 70 within 0.001).
  The 2 misses are VI-corrected lenses (J218.5371: Redrock put the source z on the
  lens fiber; J239.5610b: ZWARN=518) — the published z came from visual inspection.
- **z_source: 16/22** recovered to |dz|<0.005 (all within 0.001). The 6 misses are
  physically expected: high-z Lyα/[OII]-desert sources needing VI (J188/J212/J215
  z_s>1.6), a Keck-NIRES source (J215.2654, the "c" flag), and EDR-only / >5" offset
  source fibers.
- **sigma_v: 65/71** recovered (FastSpecFit Iron, VDISP_IVAR>0). r=0.80, median
  ratio 0.96; median |Δσ|=22.5 km/s for well-measured systems (pub err<60).
  Residual scatter is Iron-vs-Fuji reduction + best-IVAR-fiber vs paper-spectrum.
  The 6 with no VDISP are mostly high-z lenses FastSpecFit does not fit.
