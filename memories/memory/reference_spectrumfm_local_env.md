---
name: reference-spectrumfm-local-env
description: "Working venvs for the two SpectrumFM repos (redshifty, codecs) on /raid/benson aarch64 host — paths, versions, aarch64 source-build gotchas, smoke recipes"
metadata:
  type: reference
---

The two SpectrumFM-relevant repos in `lensing-repos/` are installed in separate venvs on the aarch64 host. Activate with `/raid/benson/.venvs/<name>/bin/python` directly or via the `~/.venvs/<name>` symlinks. Both pin Python 3.13.13 from `/home/benson/.local/bin/python3.13`.

## redshifty — `/raid/benson/.venvs/redshifty/`
Easy install. `pip install -e '.[dev]'` from `lensing-repos/redshifty/` works on Python 3.13 even though pyproject classifiers only list 3.10–3.12. `pytest` is 231 passed / 3 skipped (~5 min). Key pinned versions:
- torch 2.12.0+cu130 (aarch64 wheel exists at `https://download.pytorch.org/whl/cu130`)
- numpy 2.4.6, astropy 7.2.0, h5py 3.16, wandb 0.27.0
- All 10 GPUs visible (8× A16 SM 8.6 / 2× L4 SM 8.9)

Smoke recipe:
```bash
cd lensing-repos/redshifty
python scripts/download_desi_batch.py --n-files 5 --output-dir data/desi_raw
# also download matching redrock files (the batch script only fetches coadds):
for f in data/desi_raw/coadd-sv3-bright-*.fits; do
  pix=${f##*-}; pix=${pix%.fits}
  curl -sSL -o data/desi_raw/redrock-sv3-bright-$pix.fits \
    "https://data.desi.lbl.gov/public/edr/spectro/redux/fuji/healpix/sv3/bright/${pix%??}/$pix/redrock-sv3-bright-$pix.fits"
done
WANDB_MODE=offline python scripts/smoke_test.py  # 3 ep Approach A+B, ~1 s/epoch
```

## codecs — `/raid/benson/.venvs/codecs/`
Hard install: requires source builds for both `causal_conv1d` and `mamba_ssm` because the codecs README pins `causal_conv1d-1.6.1+cu13torch2.10cxx11abiTRUE-cp313-cp313-linux_x86_64.whl` — that wheel does not exist for aarch64. Mamba branch was also renamed `mamba3-release` → `mamba3-stable` upstream; the README is stale.

Working source-build recipe (causal_conv1d ~15 min; mamba_ssm ~30 min, dominated by selective_scan kernel compiles across SM 75/80/87/90/...):
```bash
export PATH=/usr/local/cuda-13.0/bin:$PATH CUDA_HOME=/usr/local/cuda-13.0
export TORCH_CUDA_ARCH_LIST="8.6;8.9"   # A16 + L4; build setup.py ignores this and builds for everything anyway
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu130   # aarch64 wheel exists
pip install packaging wheel ninja

git clone --depth 1 --branch v1.6.1.post4 https://github.com/Dao-AILab/causal-conv1d /raid/benson/src/causal-conv1d
cd /raid/benson/src/causal-conv1d
CAUSAL_CONV1D_FORCE_BUILD=TRUE pip install . --no-build-isolation -v

MAMBA_FORCE_BUILD=TRUE pip install --no-binary mamba-ssm \
  git+https://github.com/state-spaces/mamba.git@mamba3-stable --no-build-isolation -v

pip install -r lensing-repos/codecs/requirements.txt   # tilelang>=0.1.9 conflict warning is benign
```
Resulting key pins: torch 2.12.0+cu130 (mamba-ssm pulled the upgrade), mamba_ssm 2.3.1, causal_conv1d 1.6.1, tilelang 0.1.10. CUDA kernels JIT-fall back from compute_80 PTX to SM 8.6/8.9 at runtime — verified that `causal_conv1d_fn` and `Mamba2()` run on cuda:0 (L4).

**Critical runtime workaround:** `train.py` calls `torch.compile(model, mode="reduce-overhead")` which crashes inside the inductor graph on this stack. Workaround: set `TORCHDYNAMO_DISABLE=1` before `torchrun`. With that env var, 20-step smoke training runs end-to-end (train_loss 941→12, val_nll 162→7.9 on 247 spectra; ~6.5 s/step amortized including first-step CUDA autotune).

Smoke data pipeline (reuses the redshifty SV3 download):
- Synthetic iron-style zcatalog at `/raid/benson/data/desi_dr1_mini/zcatalog/v1/zall-pix-iron.fits` (built by `/raid/benson/data/desi_dr1_mini/build_mini.py`, skips the 1.1 GB pixel 10147)
- Healpix tree of symlinks at `/raid/benson/data/desi_dr1_mini/healpix/sv3/bright/...`
- HDF5 cache at `/raid/benson/data/desi_dr1_mini/codecs_cache/part1.h5` (247 spectra × 7958 pixels)
- Smoke config at `/raid/benson/data/desi_dr1_mini/codecs_smoke.yaml` (d_model=128, max_steps=20, batch_size=4)

Run command:
```bash
cd lensing-repos/codecs
CUDA_VISIBLE_DEVICES=0 TORCHDYNAMO_DISABLE=1 \
  torchrun --nproc_per_node=1 --master_port=29501 \
    scripts/train.py --config /raid/benson/data/desi_dr1_mini/codecs_smoke.yaml
```

Out of scope (deferred): `galaxy-search/` (NERSC `$CFS/deepsrch/data_gz` only, Python 3.11 unavailable on this box, 4-GPU SGLang server requirement).

Related: [[project-spectrumfm]], [[reference-gigalens-env]], [[user-role-benson]].
