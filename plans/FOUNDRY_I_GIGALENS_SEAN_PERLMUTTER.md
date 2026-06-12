# Plan: Foundry-I at paper-level quality — autonomous campaign on NERSC Perlmutter

## Context

Goal: reach Huang 2025a "Fig. 8 / Table 3" quality for DESI-165.4754−06.0423. Published bar
(verified from the PDF): R̂ < 1.10, ESS 32,200–40,000, γ = 1.372 ± 0.023, reduced χ² ≈ 1, in
193 min on one 4×A100 Perlmutter node (MAP 69 / SVI 112 / HMC 12 min).

**Why this plan now:** (1) The research group's slide-9 feedback was validated point-by-point by
a code audit — the repro's χ²≈12–24 floor traces to a too-wide un-masked field (3 faint galaxies
+ arc-A object never masked), a drizzle-blind noise model, and **~23×/~30× under-invested
MAP/SVI** (v9 MAP loss still descending at stop; v10 SVI covariance rank 53/74). The γ "valley"
that consumed 119 local GPU-hours is plausibly a likelihood-misspecification artifact.
(2) `lensing-repos/gigalens-sean` (seanxuseanxu fork) is the group's **active production
infrastructure, developed on Perlmutter itself** — `multinode-2025` is the latest stable library
line (version-bumped 2026-02-11, full shard_map multinode MAP/SVI/HMC); `multinode-carousel`
holds the group's real-data application patterns (masks.h5, carousel{MAP,SVI,HMC}.py, job.slurm).
No Foundry-I-specific code exists in any repo — we build it. (3) Perlmutter access verified:
`ssh gdbenson@perlmutter.nersc.gov` non-interactive; Slurm reachable.

**User decisions:** account **deepsrch_g**; budget **≤ 200 A100-hours**; codebase **hybrid** —
library = `gigalens-sean @ multinode-2025`, drivers patterned on the carousel branch, our
Stage-A data-treatment fixes on top.

### Perlmutter facts (recon, verified)
- QOS: `gpu_debug` (30 min, fast queue) for smoke; `gpu_regular` (2 days) for production;
  nodes = 4× A100. Account `deepsrch_g` (fallback `cosmo_g` if rejected).
- Storage: `$HOME` = /global/homes/g/gdbenson (40 GiB, ~empty); `$SCRATCH` =
  /pscratch/sd/g/gdbenson (20 TiB, empty; 8-week purge → results synced back promptly).
- Software: `module load python/3.13` (default 3.13.11), CUDA 12.9 default; **JAX via pip**
  (login nodes have internet); TF 2.15 module exists but the jax backend + TFP-on-JAX needs no TF.
- Local 10-GPU machine (8×A16 + 2×L4) = dev/smoke platform; measured baselines already in hand
  (marg-f64 grad: 336 ms A16 / 57 ms L4; 74-dim f32 leapfrog: 3.5 ms A16).

## Architecture

```
LOCAL (dev + smoke)                      PERLMUTTER (production)
reproductions/foundry-i/                 $HOME/foundry-i/        (code + small data, persistent)
  new scripts 40-45 + slurm/   --rsync->   gigalens-sean/        (checkout @ multinode-2025)
  smoke on L4s (tiny configs)             $SCRATCH/foundry-i/runs/<stage>/  (job outputs)
  gate evaluation + Fig-8 render <--rsync-- logs + posterior npz (small)
```
Orchestration: this Claude session drives everything over ssh — rsync up, `sbatch`, poll
`squeue/sacct`, rsync back, evaluate gates locally, submit next stage. Long waits via scheduled
wake-ups (~20–30 min cadence). Campaign state checkpointed in
`reproductions/foundry-i/PERLMUTTER_CAMPAIGN.md` (job IDs, A100-hour ledger, gate verdicts) so
any future session can resume.

## Implementation

### W0 — local code (1–2 days dev, smoke on local L4s before any shipping)
New files in `reproductions/foundry-i/` (existing scripts 01–36 untouched):
1. `_data_lib.py` + `40_make_cutout_v2.py` — **Stage A fixes**: ~80×80 px crop (10.4″; verify
   arcs + companion fit using `10_detect_nearby.py` coords); masks for the 3 faint galaxies +
   arc-A object + companion core (DAOStarFinder pass + paper Fig. 7 positions), written
   carousel-style as `masks_v2.h5`; drizzle-corrected noise (rescale σ so source-free sky
   regions give reduced χ² = 1.00 ± 0.05); ONE error-map definition shared by fit and plots.
2. `41_map_paper_scale.py` — MAP on sean/multinode-2025 `ModellingSequence` (shard_map over
   4 GPUs): 500–1000 particles × up to 3000 steps with a loss-plateau stop, Table-2 priors,
   empirical PSF (`17_build_empirical_psf.py` output).
3. `42_svi_paper_scale.py` — SVI n_vi 500–1000, ≥1500 steps; logs ELBO every step and the
   covariance spectrum every 100 (rank + min-eigenvalue gate built in).
4. `43_hmc_paper_scale.py` — preconditioned HMC (momentum = inverse SVI covariance via the
   precision-factor parameterization), 50–100 chains sharded on 1 node, 250 burn + 750+ keep.
5. `44_diagnostics.py` — R̂/ESS via the `35_pool_chains.py` machinery + gate verdict JSON.
6. `45_fig8_panel.py` — data · model + critical curves · reduced residual · source + caustics
   (det(Jacobian)=0 contour; lenstronomy cross-check). Runs locally.
7. `slurm/{smoke,map,svi,hmc}.slurm` — patterned on carousel `job.slurm`:
   `-A deepsrch_g -C gpu -q <qos> --gpus-per-node=4 -t <cap>`, job names `foundry-<stage>`.
Smoke-test 41–43 locally at toy scale (20 particles × 50 steps; 4 chains) on the 2 L4s.

### P0 — Perlmutter setup (~½ day, queue-dependent)
1. rsync up: scripts + slurm + `data/` essentials (`cutout_F140W.fits` 141 MB,
   `empirical_psf.npy`, `nearby_galaxy_loc.npz`, v9/refined MAP starts, long-run npz ~30 MB)
   → `$HOME/foundry-i/`; gigalens-sean checkout at the `multinode-2025` ref → alongside.
2. Build venv: `module load python/3.13; python -m venv ~/foundry-i/venv;
   pip install -U "jax[cuda12]" tensorflow-probability optax astropy h5py matplotlib tqdm;
   pip install --no-deps -e gigalens-sean` (skip its TF pins; jax backend only). Import smoke.
3. `gpu_debug` smoke job (15 min, 1 node): JAX sees 4×A100; one grad eval of our 46-dim marg
   f64 target (expect ~5–15 ms) and a 50-step mini-MAP through the sean ModellingSequence.
   **Gate: clean run + sane timings.** (~1 A100-h)

### P1 — Stage A on the new data treatment + MAP at paper scale
Run `40` locally (no GPU), ship `cutout_v2` + `masks_v2.h5`. Then `41` on 1 node,
`gpu_regular`, 2 h cap, 2–3 configs (500/1000 particles; lr schedules) as separate jobs.
**Gate (the group's bar): masked reduced χ² < 1.1.** If stalled ≫1.1 with plateaued loss →
one model-flexibility retune (source shapelets n_max 6→8; 3rd companion Sérsic), max 2 retries.
(~16–24 A100-h)

### P2 — SVI at paper scale
`42` on 1 node, 3 h cap (paper spent 112 min). **Gate: ELBO smooth + flattened AND covariance
full-rank (min eig > 0 at f64)** — the group's stated precondition. Retune: longer/lower-lr,
larger n_vi; max 2 retries. (~12–24 A100-h)

### P3 — HMC, paper recipe
`43` on 1 node (paper: 12 min; cap 1 h), 50 chains × 1000 draws; if R̂ marginal, escalate once
to 2 nodes / 100 chains × 2000 (the multinode path sean's code exists for).
**Gate: R̂ < 1.1 all params, ESS ≥ 10³.** Then compare γ vs 1.372 ± 0.023. (~4–35 A100-h)

### P4 — only if γ still disagrees after the likelihood fix
Soft mass–light PA tie (~50 LOC, prior-level) and/or m=4 multipole (port from lenstronomy,
installed locally; ~150–250 LOC JAX) → rerun P3 config. Plus a small ablation sweep (mask
on/off, noise-rescale on/off, γ-prior variants per the paper's §4 test) as 1-node array jobs
to attribute the original discrepancy. (~30–60 A100-h)

### P5 — Fig-8 render + report (local, ~2 days)
`45` at the converged posterior (inner critical curve presence = γ<2 arbiter). Update
`papers/main.tex`: new sections (likelihood corrections, paper-scale pipeline on Perlmutter,
convergence at paper bar) + comparison appendix `ours_fig8` ↔ paper Fig. 8; rebuild PDF.
Update README (stale "L4s reserved" note; Perlmutter campaign section). Update memory files.

**Budget ledger: P0+P1+P2+P3 ≈ 35–85 A100-h; with P4 ≈ 65–145; hard stop at 180 A100-h
(sacct actuals logged per job in PERLMUTTER_CAMPAIGN.md). Calendar: ~1.5–2 weeks.**

## Autonomy protocol
- Every Perlmutter mutation is scoped to `$HOME/foundry-i/` + `$SCRATCH/foundry-i/`; jobs
  tagged `foundry-*`; never more than 2 nodes; walltime caps per stage as above.
- Poll cadence ~20–30 min (queue waits dominate); results rsynced back on completion; each
  gate evaluated locally before the next submission; max 2 retunes per stage then report back.
- Checkpoint after every action in `PERLMUTTER_CAMPAIGN.md`; resume-safe across sessions.
- Stop-and-ask triggers: budget 180 A100-h reached; submission rejected on `deepsrch_g`
  (ask before switching to `cosmo_g`); any gate failed after 2 retunes; anything anomalous
  in NERSC usage.

## Critical files
- New: `reproductions/foundry-i/{_data_lib.py,40_make_cutout_v2.py,41_map_paper_scale.py,
  42_svi_paper_scale.py,43_hmc_paper_scale.py,44_diagnostics.py,45_fig8_panel.py,slurm/*,
  PERLMUTTER_CAMPAIGN.md}`
- Library: `lensing-repos/gigalens-sean` @ `origin/multinode-2025` (do not touch its working
  tree — it has uncommitted local changes; rsync a clean `git archive` of the ref instead)
- Templates: carousel branch `carousel{MAP,SVI,HMC}.py`, `masks.h5`, `job.slurm` (read via
  `git show origin/multinode-carousel:<path>`)
- Reuse: `17_build_empirical_psf.py`, `35_pool_chains.py`, `10_detect_nearby.py`

## Verification
- Gates: sky χ² = 1.00±0.05 (A) → MAP reduced χ² < 1.1 (P1) → full-rank SVI + flat ELBO (P2)
  → R̂ < 1.1 / ESS ≥ 10³ (P3) → γ vs 1.372±0.023 (P3/P4).
- Final: Fig-8 four-panel vs paper (inner critical curve, structureless residual); Table-3
  parameter comparison; every report claim backed by a logged job ID + wall-clock; A100-hour
  ledger ≤ budget.
