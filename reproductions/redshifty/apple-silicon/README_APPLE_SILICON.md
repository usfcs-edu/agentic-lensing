# redshifty (SpectrumFM Phase-I) — Apple Silicon (M4 Max / MPS) port

A from-scratch re-run of the `redshifty` DESI spectral-transformer pipeline on Apple
Silicon, extending the Mac Studio (M4 Max, 128 GB, MPS) validation begun with the
`huang-2020`/`huang-2021` lens-finding ports — but now in the **transformer / foundation-
model** regime that SpectrumFM actually targets (attention, bf16 mixed precision, a 1-D
spectral pipeline, multi-hour training), which the image-CNN ports never exercised.

`redshifty` is the SpectrumFM Phase-I prototype: a **ConvNeXt+LFQ spectrum tokenizer →
transformer encoder → cross-attention redshift head**. Its headline result is the
**redshift "ignition"** — the cross-attention pathway discovering redshift, reproduced on
phoenix (NVIDIA L4) at `val_redshift_acc` 14.86% (TF) / 7.96% (AR) using the 4-way DESI
data mix. Every prior SpectrumFM run was on phoenix's GPUs; **this port is the first on MPS.**

The codecs (Mamba3 + Residual-FSQ) tokenizer arm is deliberately **out of scope** — it is
CUDA/Triton-bound (`mamba_ssm`, `causal_conv1d`, `torch.compile`). The MPS-portable piece
is redshifty **V1 (ConvNeXt+LFQ) + the Approach-A transformer**, which are pure PyTorch + bf16.

## What the port is

The upstream redshifty source (`lensing-repos/redshifty` on phoenix) is **not committed
here** (third-party: *desi-foundation-model*); `./sync_from_phoenix.sh code` pulls it into
`src-redshifty/` and `mps_redshifty.patch` is the committed record of the MPS changes —
**6 files, +52 / −17**:

- device selection: the off-SLURM `else` branch in `nersc/{train_transformer,pretrain_tokenizer}.py`
  → a shared `nersc/_mps.py` `pick_device()` (mps>cuda>cpu). The DDP path self-bypasses
  off-SLURM, so that one line is the whole device change.
- `autocast("cuda", …)` → `autocast(device.type, …, dtype=bfloat16)` in the trainers +
  `src/training/eval.py`. **Mandatory, not cosmetic:** on a CUDA-less Mac `autocast("cuda")`
  does not error — it silently runs fp32 — so without this bf16 never engages and the
  computation diverges from the reference.
- `.to(device, non_blocking=True)` → `non_blocking=pin_ok(device)` (False on MPS) in
  `src/training/sequences.py` + `pretrain_tokenizer.py`. This is the exact host→device race
  that produced NaN-from-step-1 in the huang ports (see [[macos-mps-nonblocking-transfer-nan]]).
- `GradScaler` switched to the bf16 (disabled) path — bf16 has fp32's exponent range and
  needs no loss scaler; fp16 NaN-diverges.

`pin_memory` was already `device.type=="cuda"`-gated and the macOS-astropy `memmap=False`
fix was already upstream — confirmed, no edit. There is **no `torch.compile`** anywhere in
redshifty, attention is hand-rolled (`matmul + softmax + masked_fill`), and the tokenizer is
pure ConvNeXt-V2 + LFQ — all MPS-friendly.

## Result

Three validation layers (mirroring the huang ports' fidelity-gate + from-scratch + headline-
result structure). `verify_against_reference.py` writes `results/REPRODUCTION_MPS_COMPARE.md`.

**(a) Same-checkpoint MPS-vs-CUDA forward fidelity — PASS.** The frozen V1 tokenizer + the
ignition transformer (step 9500, `val_loss` 190.67) run on the SAME 4 byte-identical DESI
pixels on the Mac (MPS) and phoenix (CUDA), fp32:

| metric | MPS vs CUDA |
| :--- | ---: |
| median \|Δ\| logits (bulk bit-faithfulness) | **1.4e-6** |
| argmax-token agreement | **99.85%** |
| loss relative \|Δ\| | **2.5e-4** |
| redshift_acc (the cross-attention readout) | **identical** (Δ=0.0000) |
| informational: max \|Δ\| raw logits | 3.15 (heavy-tailed at near-tie/high-magnitude positions — benign; the bulk + science are faithful) |

**(b) From-scratch training on MPS — PASS.** The V1 tokenizer (24.3M) and the Approach-A
transformer (103.7M) train from random init on a bounded ~9 GB sv3-bright subset, **NaN-free**
(the non_blocking-NaN sentinel) with the val metric improving. bf16 autocast genuinely engages
on MPS (verified); throughput ~2 step/s (tokenizer) on a single MPS device.

**(c) Full-mix redshift ignition — capstone — REPRODUCED.** The exact ignition spec (frozen
tokenizer, `manifest_mix.jsonl` 1137 px / 1.72M train spectra, bf16) ran from scratch on a single
MPS device through the **unmodified `tools/spectrumfm/exp_run.py` harness** (validating the Track-2
tooling on Apple Silicon too). The **20k-step run reproduces the full ignition** — meeting all of
the redshifty author's own criteria:

| criterion | reference (phoenix L4) | MPS 20k | gate |
| :--- | ---: | ---: | :---: |
| `val_z_acc` ≥10% sustained | 14.86% peak | **12.70% peak, 6 late vals ≥10%** | PASS |
| `val_loss_redshift` drop ≥1.0 | 1.19 | **1.03** | PASS |
| AR ≥ TF/2 (honest, no teacher-forcing) | 0.73× | **0.60× (AR 6.6%)** | PASS |
| `val_loss` min | 190.67 | 200.7 | (info) |

The back-half climb continued past step 10000 — `val_z_acc` 7.72% @ 12500 → 10.35% @ 17000 →
**12.70% @ 19500** — landing close to the phoenix reference's 14.86%. This directly confirms the
author's note that *"10000 steps was barely enough to see ignition kinetics; future runs should
use ≥20000 steps to give the post-ignition phase room."* A shorter **10k-step MPS run** (8.76 h)
peaked at **7.88%** — within the phoenix seed-sweep band 3.76–8.76% but short of the ≥10% bar,
exactly as the author's note predicts. Both trajectories are committed
(`results/ignition_metrics_mps{,_20k}.jsonl`). The exact peak is high-variance and
hardware-path-dependent (bf16 kernels diverge the trajectory), so it is reported, not gated.

### Apple Silicon gotchas (important)

- **`autocast("cuda")` silently runs fp32 on a CUDA-less Mac** (no error) — you must pass
  `device.type` or bf16 never engages and you diverge from the reference computation.
- **`.to(device, non_blocking=True)` races the H2D copy on MPS → NaN from step 1.** Tie
  `non_blocking` to `pin_ok(device)` (CUDA-only).
- Checkpoints were saved under Python **3.13** on phoenix; load them with a **3.13** Mac venv
  (a 3.12 venv fails unpickling `pathlib._local`). Use `/opt/homebrew/bin/python3.13`.
- Run exactly **one** MPS process; use `PYTHONUNBUFFERED=1` (Python block-buffers stdout to a
  file, so a healthy run looks hung); `--num-workers 0` on MPS.

## Layout

- `src-redshifty/` — vendored upstream source (gitignored) + the MPS patch.
- `_raid/` (gitignored) — all synced DESI data + phoenix checkpoints, mirrored at the phoenix
  absolute path `/raid/benson/data/desi_dr1_medium/…` and exposed on the Mac via a one-time
  `sudo ln -s _raid /raid` (lets the unmodified manifest/spec/harness resolve verbatim).
- `results/` — committed small artifacts: `xcheck_compare.json` (the fidelity verdict) and
  `REPRODUCTION_MPS_COMPARE.md` (the verify report). `data/` (raw xcheck dumps, Tier-1 metrics,
  logs) and large ckpts/run-dirs are gitignored.

## Setup & run

```bash
/opt/homebrew/bin/python3.13 -m venv .venv
.venv/bin/python -m pip install -U pip wheel && .venv/bin/python -m pip install -r requirements.txt
./sync_from_phoenix.sh code priority links     # source + ckpts + manifests + harness symlinks
( cd . && .venv/bin/python -m pip install -e src-redshifty )
git apply mps_redshifty.patch                  # (already applied in src-redshifty if synced post-port)

./run_infer_fidelity.sh                        # layer (a): MPS-vs-CUDA forward fidelity
.venv/bin/python make_tier1_submanifest.py && ./run_tier1.sh   # layer (b): from-scratch training

# layer (c) — the ~15-30h capstone:
sudo ln -s "$PWD/_raid" /raid                  # one-time (reversible: sudo rm /raid)
./sync_from_phoenix.sh mix                      # ~760 GiB, resumable
./run_tier2.sh                                  # ignition via the unmodified harness
.venv/bin/python verify_against_reference.py
```
