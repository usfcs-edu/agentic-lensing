# P2 deployment plan — cgl2 stack on Perlmutter (Front C scout, 2026-07-15)

Scope: what P2 needs to run where. B1 carousel + S7 need the cgl2 stack (scene API) on
Perlmutter x86 GPUs; B4/B5 need MC-SMC against OLD-stack targets there. Everything below
is measured (commands run today on the login node / phoenix), not assumed.

## RECOMMENDATION (one paragraph)

**Build a native `cgl2-pm` venv on Perlmutter at the exact campaign pins (Path A) and run
B4/B5 out of the existing OLD venv via `PYTHONPATH` (Path B, zero new env work —
vendor-free import PROVEN today).** Every wheel the cgl2 venv pins is on PyPI for
x86_64/cp313, the login `module load python` is python 3.13.11 (same minor as phoenix's
3.13.13), and the OLD campaign already runs the identical jax 0.6.2 + cuda12-plugin
wheels on Perlmutter A100s (all of P1/T1.1) — so Path A is an assembly job, not a
porting job. The team's Shifter container (Path C) is the FALLBACK only: it is jax 0.10
+ tfp-nightly (tfp 0.25.0, our pin, refuses to import under jax ≥0.7 per their own
env_setup.md), its conda env + sidecar live in linusu's home and are
**permission-denied to us**, and using it flips gate F8 from informational to
load-bearing (full re-validation inside the container before any benchmark row).

## Evidence

### E1 — vendor-free `cgl2.samplers.smc_micro` (local test, both venvs): PASS, no refactor needed
Test: meta-path blocker raising ImportError for any `gigalens*` import, then
`import cgl2.samplers.smc_micro` + a full MAMS SMC run on a conjugate toy
(`22_run_b3_vendorfree_test.py`; run it with `CGL2_ALLOW_CPU=1 JAX_PLATFORMS=cpu
GIGALENS_X64=1 <venv-python> 22_run_b3_vendorfree_test.py`).
- cgl2 venv (vendor blocked): IMPORT OK, `gigalens` never enters sys.modules,
  logZ −3.5272 vs analytic −3.4813 (err 0.046 < 1σ_boot 0.055). PASS.
- **OLD cgl venv** (`/raid/benson/.venvs/cgl`, python 3.13.13, blackjax 1.3) with
  `PYTHONPATH=<campaign root>`: identical output **bit-for-bit** (same logZ digits). PASS.
- Why it works: `cgl2/__init__.py` only sets x64; `paths.bootstrap_vendor()` runs ONLY
  when an operator script calls `guards.require_vendor_ref()`. All `gigalens` imports in
  the package are function-deferred and live in {paths, guards, correlated, scene_build,
  zoo} only; {param_map, whiten, marg, noise, samplers/*, _ratio_coords_copy} are
  vendor-free. **No samplers-subpackage split is required** — the pre-briefed refactor
  contingency is NOT needed.

### E2 — Perlmutter login-node probe (2026-07-15 22:00 PT, login29; light commands only)
- OLD venv `~/claude-giga-lens/venv`: python **3.13.11**, jax/jaxlib/jax-cuda12-{plugin,pjrt}
  **0.6.2**, blackjax **1.3**, tfp **0.25.0**, numpy **2.4.6**, scipy 1.17.1, optax 0.2.8,
  lenstronomy 1.14.0, objax 1.8.0, tqdm 4.68.3 — i.e. the cgl2 pin set (D2) already
  proven on that machine's A100s. (matplotlib absent — harvest/figs stay on phoenix.)
- `module load python` → python 3.13.11 (nersc-python 26.1.0) → cp313 wheels apply.
- PyPI wheel availability (JSON API, exact pins): jaxlib 0.6.2 cp313 manylinux2014_x86_64
  ✓, jax-cuda12-plugin 0.6.2 cp313 ✓, jax-cuda12-pjrt 0.6.2 py3-none ✓, numpy 2.4.6
  cp313 ✓, scipy 1.17.1 cp313 ✓, blackjax 1.3 ✓, tfp 0.25.0 ✓, optax 0.2.8 ✓,
  lenstronomy 1.14.0 ✓, objax 1.8.0 ✓. **Wheels-only install possible — nothing
  compiles.**
- Shifter images BOTH present in the NERSC registry (`shifterimg lookup` returns ids):
  `ghcr.io/nvidia/jax:jax-2026-04-13` (their canonical, jax ≥0.10.0.dev) and
  `ghcr.io/nvidia/jax:jax-2025-06-07` (their legacy jax-0.6-era image).
- **Permission-denied**: `/global/u1/l/linusu/GIGALens-Code/experiments/sim_carousel/
  newnewcutouts/` (the REAL carousel MUSE cutouts B1 needs) and
  `/global/homes/l/linusu/sidecar_jax_upgrade` (the tfp-nightly overlay the container
  needs). Their conda env `gigalens_multinode_env` is in `~linusu/.conda` (same story).
- CFS staging `/global/cfs/cdirs/deepsrch/gdbenson/cgl2-linus/{code,data,results,slurm-logs}`
  and `$PSCRATCH/cgl2-linus` exist (P1/T1.1 pattern).

### E3 — substrate finding (B3 smoke, phoenix L4, jax 0.6.2, 1 GPU)
The VENDORED `gigalens.jax.inference.ModellingSequence.MAP` (@80916d2) raises
`TypeError: cotangent type does not match function output ... complex128[...]{V:device}`
— the shard_map-wrapped `value_and_grad` through the FFT PSF convolution breaks under
jax 0.6.2 (the library's own declared pin; the file's comments show it was reworked for
jax 0.10). Consequence for P2: any arm using their inference path under jax 0.6.2 must
use the single-device replicated recipe validated today in `22_run_b3.py`
(copy-with-attribution, math-identical at dev_cnt=1), or the jax-0.10 container.
Our SMC arms never touch inference.py. Handoff/memo item (UNCERTIFIED external).

## Path A (RECOMMENDED for B1 carousel + S7): native `cgl2-pm` venv

1. **Stage code** (phoenix → CFS; NON-GIT rsync + md5-audit, the stale-remote lesson):
   `rsync -a --exclude data/ --exclude figs/ --exclude __pycache__/
   reproductions/claude-giga-lens-linus/
   perlmutter:/global/cfs/cdirs/deepsrch/gdbenson/cgl2-linus/code/repo/`
   then md5-audit every `.py` that will execute (local vs remote) before submission.
   Also stage `data/parity_refs.npz` (11 MB) + `data/whitener_manifest.json` + the
   whitener bundles the payload needs — parity re-certification consumes them.
2. **Build the venv on a COMPUTE node** (no login-node installs; shared-QOS CPU or the
   first 10 min of a shared-GPU job): venv at
   `/global/cfs/cdirs/deepsrch/gdbenson/cgl2-linus/cgl2-pm-venv` (CFS, not $PSCRATCH —
   purge policy would eat it):
   ```
   module load python
   python3 -m venv $CFS_DIR/cgl2-pm-venv && source .../bin/activate
   pip install --only-binary :all: -c constraints.txt \
       jax==0.6.2 jaxlib==0.6.2 jax-cuda12-plugin==0.6.2 jax-cuda12-pjrt==0.6.2 \
       blackjax==1.3 tensorflow-probability==0.25.0 numpy==2.4.6 scipy==1.17.1 \
       optax==0.2.8 lenstronomy==1.14.0 objax==1.8.0 tqdm
   pip install -e $CODE/repo/vendor/gigalens-linus --no-deps   # D1: UNPATCHED, no-deps
   pip install -e $CODE/repo --no-deps                          # cgl2 package
   ```
   (`constraints.txt` is in the campaign root — the chex lesson.)
3. **Certify before any benchmark row** (one shared-GPU canary, ~0.2 A100-h, ledgered):
   `00_env_check.py` + `01_parity_scene.py` against the staged parity refs + `03` + `04
   --quick --skip-t0`-parity. Expected: F1–F7 reproduce (same jax/tfp/numpy pins; A100
   vs L4 is an XLA-reduction-order delta only — F-gate tolerances already absorb the
   hardware change, cf. the B0 A16 provenance note). Record a "PM-native gate battery"
   row. **F8 stays report-only** on this path.
4. **Run pattern**: identical to P1/T1.1 ops (cosmo_g shared QOS, 1 GPU, ledger row
   BEFORE results, watchdog_add.py registration, `#SBATCH -C gpu&hbm80g` for any
   SMC/HMC production step — the 55952482 OOM lesson; hot I/O on $PSCRATCH, `cp` to CFS).

## Path B (B4/B5 against OLD-stack targets): OLD venv + PYTHONPATH

The OLD venv already has every runtime dep of `cgl2.samplers` (E2) and the vendor-free
import + run is PROVEN in that exact pin set (E1). No install at all:
```
source ~/claude-giga-lens/venv/bin/activate
export PYTHONPATH=/global/cfs/cdirs/deepsrch/gdbenson/cgl2-linus/code/repo
python -c "import cgl2.samplers.smc_micro"   # sanity; gigalens never imported
```
B4/B5 wire `run_tempered_smc`/`make_kernel('mams')` to the OLD cgl targets exactly as
the T2/T3 zoo paths do on phoenix (`cgl` importable from the existing
`~/claude-giga-lens/repo/...` tree). GPU-memory realism from B3: the N-wide vmapped
gradient will not fit at N=512 on any card for image likelihoods — use the
chunked-mutation MAMS subclass pattern from `22_run_b3.py` (bit-identity vs the stock
kernel verified: max particle diff 2.8e-13, identical logZ/grad counts) with hbm80g.

## Path C (FALLBACK ONLY): their Shifter container, jax 0.10

Use only if Path A hits an XLA/driver wall on A100s (unlikely: the OLD campaign runs the
same jaxlib-cuda12 0.6.2 stack there daily). Costs if invoked:
- **F8 becomes load-bearing, not informational**: the full parity battery (F1–F7 + 03
  gates + B0 adapter parity) must re-run and PASS inside the container BEFORE any
  benchmark row (template exists: `slurm/parity_f8_nersc.slurm`); any FAIL = ledgered
  stop, not a tolerance bump.
- **tfp pin breaks**: tfp 0.25.0 will not import under jax ≥0.7 (their env_setup.md);
  the container path forces tfp-nightly — a different stack from every validated number
  this campaign has produced. blackjax 1.3 `adjusted_mclmc` under jax 0.10 is unverified.
- **Their conda env + sidecar are NOT reusable by us** (permission-denied, E2): we would
  rebuild both from scratch inside the container — more work than Path A, with weaker
  provenance.
- Positive use: the container IS the right venue for the F8 informational cell itself,
  and the only currently-known venue where the vendored `inference.py` MAP runs (E3).

## Open blockers / memo items (independent of environment)

1. **B1 real carousel cutouts are permission-blocked** (`newnewcutouts/source4-5.fits`,
   `source9.fits` + STAT/PSF/MASK): B1/S7 on the real system need a data transfer from
   the team (memo channel request; ~MBs). Until then B1 runs only on the mock stand-in
   (machinery value, zero benchmark value — zoo.py already says so).
2. **inference.py-on-0.6.2 defect (E3)** → any warm-reference arm on Perlmutter uses the
   replicated single-device recipe from `22_run_b3.py` (attribution kept) — flag in the
   handoff so the team knows their 0.6.2 pin no longer runs their own MAP.
3. **S7 minimal flow-MAMS**: no flow library in the pins; the minimal arm should build
   its preconditioner from tfp 0.25 bijectors already present (keeps Path A pins
   untouched); anything heavier is a ledgered dep decision.
4. Carousel memory sizing (B1): 300×300 grid, 32-dim z, lstsq shapelet basis — heavier
   per-eval than hs2; plan chunk 32–64 with the chunked kernel + `gpu&hbm80g`, and probe
   before the ledger row (B3 sizing-probe pattern).

## Risk table

| Risk | Path | Mitigation |
|---|---|---|
| PSCRATCH purge eats env/artifacts | A/B | venv + results archived on CFS; hot I/O only on scratch |
| Wheel resolution drifts (pip picks newer dep) | A | `-c constraints.txt` + `--only-binary :all:` + pip freeze diff vs phoenix `data/env_freeze_phoenix.cs.usfca.edu.txt` in the certify step |
| gigalens pip metadata demands tensorflow/numpy 2.1.3 | A | `--no-deps` install (D2 precedent; TF verified inert) |
| A100 OOM at production particle counts | A/B | chunked MAMS (bit-identity proven) + `-C gpu&hbm80g` pin (carried flag) |
| jax 0.10 container invalidates pins | C | don't take Path C except as ledgered fallback; F8 battery gates it |
| Stale remote code | A/B | md5-audit every executed .py (house rule; caught 3 stale files in P1) |
