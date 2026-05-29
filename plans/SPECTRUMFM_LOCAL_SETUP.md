# SpectrumFM local-setup runbook (redshifty + codecs)

What you need to recreate the two SpectrumFM venvs on the `/raid/benson` aarch64 host. Memory entry `reference_spectrumfm_local_env` is the quick-lookup version; this is the full repeatable recipe with all the gotchas explained.

## Host facts (locked in 2026-05-26)

- aarch64 Linux (AlmaLinux 9.7, kernel 5.14)
- 10 GPUs: 8× NVIDIA A16 16 GB (SM 8.6) + 2× NVIDIA L4 23 GB (SM 8.9). CUDA driver 13.2.
- CUDA toolkits at `/usr/local/cuda-{12,12.9,13,13.0,13.1,13.2}`; codecs source builds use `cuda-13.0`.
- Python 3.13.13 at `/home/benson/.local/bin/python3.13`. No system 3.10/3.11/3.12 (irrelevant for these repos but it kills galaxy-search which is deferred).
- gcc 11.5 system-wide. **Both repos build with this** — we verified causal_conv1d 1.6.1 and mamba_ssm 2.3.1 compile cleanly on gcc 11.5 + nvcc 13.0, despite the codecs README pinning gcc/12.2. gcc-toolset-12 (12.2.1) is in the AlmaLinux 9 appstream repo if needed later.
- Venvs live under `/raid/benson/.venvs/` with `~/.venvs/<name>` symlinks (matches the existing gigalens convention).

## redshifty (≈10 min wall time)

```bash
/home/benson/.local/bin/python3.13 -m venv /raid/benson/.venvs/redshifty
ln -s /raid/benson/.venvs/redshifty /home/benson/.venvs/redshifty
source /raid/benson/.venvs/redshifty/bin/activate

pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu130   # picks 2.12.0+cu130 aarch64
cd lensing-repos/redshifty
pip install -e '.[dev]'
pytest   # 231 passed / 3 skipped, ~5 min
```

Smoke data + run:
```bash
python scripts/download_desi_batch.py --n-files 5 --output-dir data/desi_raw
# the batch script only pulls coadd-*.fits; pull matching redrock-*.fits separately:
for f in data/desi_raw/coadd-sv3-bright-*.fits; do
  pix=${f##*-}; pix=${pix%.fits}
  curl -sSL -o data/desi_raw/redrock-sv3-bright-$pix.fits \
    "https://data.desi.lbl.gov/public/edr/spectro/redux/fuji/healpix/sv3/bright/${pix%??}/$pix/redrock-sv3-bright-$pix.fits"
done
WANDB_MODE=offline python scripts/smoke_test.py   # Approach A + B, 3 epochs each
```

Expected: 5 coadds (~1.2 GB; 10147 is the 1.1 GB one), Approach A train_loss ~17.9 → 16.6 over 3 epochs, Approach B ~19.3 → 16.3, ~1 s/epoch on cuda. Checkpoints under `checkpoints/approach_{a,b}/`.

Notes:
- pyproject classifiers only list Python 3.10–3.12, but 3.13 works fine.
- The repo's own `data/download_desi.py` produces `spectra-*.fits` files that the model doesn't consume — use `scripts/download_desi_batch.py` for `coadd-*.fits` instead.

## codecs (≈60 min wall time; aarch64 source builds dominate)

```bash
/home/benson/.local/bin/python3.13 -m venv /raid/benson/.venvs/codecs
ln -s /raid/benson/.venvs/codecs /home/benson/.venvs/codecs
source /raid/benson/.venvs/codecs/bin/activate

export PATH=/usr/local/cuda-13.0/bin:$PATH CUDA_HOME=/usr/local/cuda-13.0
export TORCH_CUDA_ARCH_LIST="8.6;8.9"   # setup.py ignores this and builds for everything; harmless

pip install --upgrade pip packaging wheel ninja
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu130   # aarch64 wheel exists

# causal_conv1d: README pins an x86_64-only prebuilt wheel; build from source instead (~15 min)
mkdir -p /raid/benson/src
git clone --depth 1 --branch v1.6.1.post4 \
  https://github.com/Dao-AILab/causal-conv1d /raid/benson/src/causal-conv1d
cd /raid/benson/src/causal-conv1d
CAUSAL_CONV1D_FORCE_BUILD=TRUE pip install . --no-build-isolation -v

# mamba-ssm: README points at branch `mamba3-release` which upstream renamed to `mamba3-stable` (~30 min)
MAMBA_FORCE_BUILD=TRUE pip install --no-binary mamba-ssm \
  git+https://github.com/state-spaces/mamba.git@mamba3-stable --no-build-isolation -v

# Rest of the codecs deps. tilelang/apache-tvm-ffi version conflict against mamba-ssm's pin is benign at runtime.
pip install -r lensing-repos/codecs/requirements.txt
```

Verify imports + a small GPU forward:
```python
import torch
from causal_conv1d import causal_conv1d_fn
from mamba_ssm import Mamba2
torch.cuda.set_device(0)
x = torch.randn(2, 16, 32, device='cuda', dtype=torch.float16)
w = torch.randn(16, 4, device='cuda', dtype=torch.float16); b = torch.randn(16, device='cuda', dtype=torch.float16)
print(causal_conv1d_fn(x, w, b, activation='silu').shape)
m = Mamba2(d_model=64, d_state=32, d_conv=4, expand=2, headdim=16).cuda().to(torch.bfloat16)
print(m(torch.randn(2, 16, 64, device='cuda', dtype=torch.bfloat16)).shape)
```

### Smoke data + run

The redshifty SV3 EDR download (above) doubles as the codecs data source. Codecs/scripts/data.py needs an "iron-style" zcatalog and a `<healspec>/<survey>/<program>/<hp//100>/<hp>/coadd-...fits` directory tree. Build both with the helper:

```bash
# Run once, after the redshifty data download.
/raid/benson/.venvs/redshifty/bin/python /raid/benson/data/desi_dr1_mini/build_mini.py
```

This produces:
- `/raid/benson/data/desi_dr1_mini/zcatalog/v1/zall-pix-iron.fits`  (synthetic; 310 rows)
- `/raid/benson/data/desi_dr1_mini/healpix/sv3/bright/<hp//100>/<hp>/coadd-...fits` (symlinks)
- (script skips the 1.1 GB pixel 10147 to keep the smoke run fast)

Then build the HDF5 cache and run the smoke training (`codecs_smoke.yaml` is co-located with the data; d_model=128, max_steps=20):

```bash
cd lensing-repos/codecs
/raid/benson/.venvs/codecs/bin/python scripts/data.py \
  --catalog /raid/benson/data/desi_dr1_mini/zcatalog/v1/zall-pix-iron.fits \
  --healspec-dir /raid/benson/data/desi_dr1_mini/healpix \
  --output /raid/benson/data/desi_dr1_mini/codecs_cache/part1.h5 \
  --workers 4 --chunk 1 --total-chunks 1 --chunk-rows 32

CUDA_VISIBLE_DEVICES=0 TORCHDYNAMO_DISABLE=1 \
  /raid/benson/.venvs/codecs/bin/torchrun --nproc_per_node=1 --master_port=29501 \
    scripts/train.py --config /raid/benson/data/desi_dr1_mini/codecs_smoke.yaml

/raid/benson/.venvs/codecs/bin/python scripts/visualize.py \
  --config /raid/benson/data/desi_dr1_mini/codecs_smoke.yaml \
  --checkpoint /raid/benson/data/desi_dr1_mini/outputs/smoke/model.pt \
  --n 3 --val-size 16 \
  --output_dir /raid/benson/data/desi_dr1_mini/outputs/smoke/viz
```

Expected: 247 spectra × 7958 wavelength pixels in the HDF5; training 20 steps in ~2 min (first step ~2 min for CUDA autotune, rest <0.4 s); train_loss 941→12, val_nll 162→7.9; `model.pt` + `checkpoint.pt` + `logs.txt` under `outputs/smoke/`; 3 PNG reconstructions in `outputs/smoke/viz/`.

### Codecs gotchas — exhaustive list

1. **README pins an x86_64-only prebuilt wheel** for causal_conv1d. The provided URL (`...-cp313-cp313-linux_x86_64.whl`) is unusable on aarch64; build from source with `CAUSAL_CONV1D_FORCE_BUILD=TRUE`.
2. **`mamba3-release` branch was renamed `mamba3-stable`.** The README's pip URL fails with `error: pathspec 'mamba3-release' did not match any file(s)`.
3. **mamba-ssm install upgrades torch to 2.12.0+cu130** (from the 2.10.0 README pin). Harmless on aarch64; the cu130 wheel exists.
4. **`torch.compile(model, mode="reduce-overhead")` crashes inductor.** Bypass with `TORCHDYNAMO_DISABLE=1` env var on the torchrun command. We did not patch `scripts/train.py` itself.
5. **`scripts/data.py` hardcodes NERSC paths** as defaults, but `--catalog` and `--healspec-dir` CLI flags override cleanly — no source edit needed.
6. **`val_size: 20000` in the upstream `test.yaml`** dwarfs a 247-spectrum smoke cache. Use the bespoke `codecs_smoke.yaml` (val_size=16, max_steps=20).
7. **DDP single-GPU works with `torchrun --nproc_per_node=1`**; we did not have to manually set MASTER_ADDR/PORT/RANK/WORLD_SIZE.
8. **`tilelang==0.1.8` vs `tilelang>=0.1.9` conflict** between mamba-ssm's pin and codecs' requirement is a pip-resolver warning only; both 0.1.8 and 0.1.10 import and run on this stack.
9. **NumPy 2.x is the default** (numpy 2.4.6) — both repos work fine with it; no NumPy 1.x fallback needed.
10. **First training step is slow (~2 min)** because of CUDA kernel autotune; steps 2+ are fast. Don't time the first step.

## Out of scope

- `galaxy-search/` — Python 3.11 not on this box; NERSC `$CFS/deepsrch/data_gz` required for eval; 4-GPU SGLang server side-service; tangential to the SpectrumFM Phase-I narrative. Skipped per the user's scoping question in the planning step.
- Full DESI DR1 pretraining — needs the real iron zcatalog and the full healpix tree at NERSC; out of scope for a local box.
- Human-alignment SFT / preference learning (Phase 2 of the SpectrumFM proposal).
