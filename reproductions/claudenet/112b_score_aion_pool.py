#!/usr/bin/env python3
"""112b_score_aion_pool.py — Phase 110: score pool cutout shards with the v1
DEGRADED AION-1 member (frozen aion-base embeddings + the v1 MLP probe), one
GPU. Runs on Perlmutter (pytorch/2.8.0 module + `aion` installed --user) or
locally under the aion venv (/home2/benson/.venvs/aion/bin/python).

v1 pipeline replicated EXACTLY (this member is knowingly degraded; keep it so):
  1. 101px grz -> 160px griz: bilinear F.interpolate(align_corners=False) and a
     SYNTHETIC i band i = 0.5*(r+z), band order [g,r,i,z]  (to_griz160 copied
     verbatim from 10_build_aion_inputs.py).
  2. Frozen polymathic-ai/aion-<variant> encoder via CodecManager.encode ->
     AION.encode(tokens, num_encoder_tokens=<summed token count>), mean-pooled
     over tokens, stored as float16 (the aion-1/_aion_embed.py extract_range
     math, incl. the fp16 on-disk round-trip the v1 probe consumed). autocast
     fp16 follows aion-1/_config.DEFAULT_AMP (OFF for base — v1 embedded base
     in fp32; --amp on overrides).
  3. v1 MLP probe data/ckpt/aion_probe_<variant>.pt (Linear->GELU->Dropout->
     Linear, hidden/k inferred from the state_dict; class copied verbatim from
     12_probe_aion.py): standardise with the checkpoint xmu/xsd, p =
     softmax(head(x), 1)[:, 1].

Inputs: --cutout-root with cutouts_<k>.npy (n,3,101,101 float32) + index.parquet
[row_id,shard,idx_in_shard,ok,...]; ok=False rows (brick missing a grz band at
footprint edges, brick-edge crossers) -> NaN. Per-shard embeddings are cached
under <out-stem>_emb/ (resume-safe; kept for reuse). Output parquet:
row_id, ok, member_aion.

aion install on Perlmutter (LOGIN node; pyproject name `polymathic-aion`,
setuptools backend -> pip-installable straight from GitHub; pin the commit the
local editable install /home2/benson/lensing-repos/AION uses):
    module load pytorch/2.8.0
    python -m pip install --user \
        "git+https://github.com/PolymathicAI/AION.git@3434e316ec2a17335e301de24c20058f2ef57e92"
    # (alternative: rsync the local checkout, then
    #  python -m pip install --user /path/to/AION)
HF prefetch (login node — compute nodes have no internet; the aion-base HF repo
also hosts the codec checkpoints: aion/codecs/config.py HF_REPO_ID ==
'polymathic-ai/aion-base', so one snapshot covers model + codecs):
    HF_HOME=$SCRATCH/claudenet/hf python -c \
        "from huggingface_hub import snapshot_download as d; d('polymathic-ai/aion-base')"

    sbatch --export=ALL,CMD='HF_HOME=$SCRATCH/claudenet/hf python 112b_score_aion_pool.py \
        --cutout-root $SCRATCH/claudenet/cutouts/negeval \
        --out $SCRATCH/claudenet/scores/negeval_scores_aion.parquet' \
        nersc/shared_gpu.slurm
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import _clib as C

BANDS = ["DES-G", "DES-R", "DES-I", "DES-Z"]      # from 11_embed_aion.py
# copied from aion-1/_config.py (not importable on Perlmutter):
MODELS = {"base": "polymathic-ai/aion-base", "large": "polymathic-ai/aion-large",
          "xlarge": "polymathic-ai/aion-xlarge"}
DEFAULT_AMP = {"base": False, "large": False, "xlarge": True}


def to_griz160(grz_batch: np.ndarray) -> np.ndarray:
    """(B,3,101,101) grz -> (B,4,160,160) griz, bilinear resize + i=0.5*(r+z).
    Copied verbatim from 10_build_aion_inputs.py."""
    x = torch.from_numpy(grz_batch)                       # (B,3,101,101)
    x = F.interpolate(x, size=(160, 160), mode="bilinear", align_corners=False)
    g, r, z = x[:, 0], x[:, 1], x[:, 2]
    i = 0.5 * (r + z)
    out = torch.stack([g, r, i, z], dim=1)                # g,r,i,z order
    return out.numpy().astype(np.float32)


class MLPProbe(nn.Module):
    """mirror of 12_probe_aion.MLPProbe (= _probe.MLPHead with 2-class out)."""

    def __init__(self, dim, hidden=256, k=2, p=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(),
                                 nn.Dropout(p), nn.Linear(hidden, k))

    def forward(self, x):
        return self.net(x)


def _num_encoder_tokens(tokens) -> int:
    """copied from aion-1/_aion_embed.py (matches AION.forward's count)."""
    n = 0
    for v in tokens.values():
        n += v.shape[1] if v.dim() == 2 else 1
    return n


def load_aion(variant: str, device):
    """aion-1/_aion_embed.load_model: frozen AION + codec manager."""
    from aion.codecs import CodecManager
    from aion.model import AION

    torch.set_grad_enabled(False)
    cm = CodecManager(device=device)
    model = AION.from_pretrained(MODELS[variant]).to(device).eval()
    return model, cm


def embed_batch(model, cm, grz: np.ndarray, device, amp: bool) -> np.ndarray:
    """(B,3,101,101) grz -> (B,D) float16 mean-pooled embeddings (the
    _aion_embed.extract_range math on a to_griz160 input)."""
    from aion.modalities import LegacySurveyImage

    flux = torch.as_tensor(to_griz160(grz), device=device)
    img = LegacySurveyImage(flux=flux, bands=BANDS)
    tokens = cm.encode(img)
    net = _num_encoder_tokens(tokens)
    if amp:
        with torch.autocast("cuda", dtype=torch.float16):
            emb = model.encode(tokens, num_encoder_tokens=net)   # (B,T,D)
    else:
        emb = model.encode(tokens, num_encoder_tokens=net)
    emb = emb.float().mean(dim=1)
    return emb.cpu().numpy().astype(np.float16)   # v1 stored embeddings as fp16


def probe_ckpt_path(ckpt_dir: Path, variant: str) -> Path:
    for p in (ckpt_dir / "ckpt" / f"aion_probe_{variant}.pt",
              ckpt_dir / f"aion_probe_{variant}.pt"):
        if p.exists():
            return p
    raise FileNotFoundError(f"aion_probe_{variant}.pt not under {ckpt_dir}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cutout-root", required=True,
                    help="111 output dir (cutouts_<k>.npy + index.parquet)")
    ap.add_argument("--out", required=True, help="output parquet path")
    ap.add_argument("--batch", type=int, default=128)   # aion-1 DEFAULT_BATCH[base]
    ap.add_argument("--gpus", type=int, default=1,
                    help="only 1 supported (Perlmutter shared QOS gives one GPU)")
    ap.add_argument("--variant", default="base", choices=tuple(MODELS))
    ap.add_argument("--ckpt-dir", default=str(C.DATA),
                    help="dir holding ckpt/aion_probe_<variant>.pt")
    ap.add_argument("--amp", choices=("auto", "on", "off"), default="auto",
                    help="fp16 autocast; auto = v1 aion-1/_config.DEFAULT_AMP "
                         "(off for base)")
    args = ap.parse_args()
    if args.gpus != 1:
        ap.error("--gpus: only 1 is supported (shard the pool externally if needed)")
    amp = DEFAULT_AMP[args.variant] if args.amp == "auto" else (args.amp == "on")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(C.SEED)
    root, out = Path(args.cutout_root), Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    emb_dir = out.parent / f"{out.stem}_emb"
    emb_dir.mkdir(parents=True, exist_ok=True)

    index = (pd.read_parquet(root / "index.parquet")
             .sort_values(["shard", "idx_in_shard"]).reset_index(drop=True))
    print(f"[112b] {len(index):,} rows in {index.shard.nunique()} shards; "
          f"{int(index.ok.sum()):,} ok | variant={args.variant} amp={amp} "
          f"batch={args.batch} device={device}")

    # probe first (cheap; fails fast if the checkpoint is missing/odd)
    pk = probe_ckpt_path(Path(args.ckpt_dir), args.variant)
    ck = torch.load(str(pk), map_location="cpu", weights_only=False)
    sd = ck["state_dict"]
    dim = int(sd["net.0.weight"].shape[1])
    hidden = int(sd["net.0.weight"].shape[0])
    k = int(sd["net.3.weight"].shape[0])
    assert dim == int(ck["dim"]), f"{pk.name}: dim mismatch {dim} != {ck['dim']}"
    head = MLPProbe(dim, hidden=hidden, k=k).to(device)
    head.load_state_dict(sd)
    head.eval()
    xmu, xsd = np.asarray(ck["xmu"]), np.asarray(ck["xsd"])
    print(f"[112b] probe {pk.name}: dim={dim} hidden={hidden} k={k} "
          f"val_auc={float(ck.get('val_auc', float('nan'))):.4f}")

    model, cm = load_aion(args.variant, device)

    # 1. embed ok rows shard by shard (cached -> resume-safe; the cache is
    #    fingerprinted to the ok-row ids AND the source shard file so a
    #    re-extraction with the same shapes can never serve stale embeddings)
    t0 = time.time()
    shard_embs = {}
    for kk, sub in index.groupby("shard", sort=True):
        ok = sub[sub.ok]
        src = root / f"cutouts_{kk}.npy"
        st = src.stat()
        h = pd.util.hash_pandas_object(ok.row_id.astype(str), index=False).to_numpy()
        fp = f"{len(ok)}:{int(h.sum()) & 0xFFFFFFFFFFFF:x}:{st.st_size}:{int(st.st_mtime)}"
        cache = emb_dir / f"emb_{kk}.npz"
        if cache.exists():
            try:
                z = np.load(cache, allow_pickle=False)
                e, fp_stored = z["emb"], str(z["fp"])
            except Exception:
                e, fp_stored = None, None
            if e is not None and fp_stored == fp and e.shape == (len(ok), dim):
                shard_embs[kk] = e
                print(f"[112b] shard {kk}: resume {cache.name} {e.shape}")
                continue
            print(f"[112b] shard {kk}: stale embedding cache -> re-embedding")
        mm = np.load(src, mmap_mode="r")
        if mm.shape[1:] != (3, 101, 101):
            raise ValueError(f"cutouts_{kk}.npy shape {mm.shape}; expected (*,3,101,101)")
        sidx = ok.idx_in_shard.to_numpy()
        parts = []
        for s in range(0, len(ok), args.batch):
            grz = np.asarray(mm[sidx[s:s + args.batch]], dtype=np.float32)
            parts.append(embed_batch(model, cm, grz, device, amp))
        e = (np.concatenate(parts, 0) if parts
             else np.zeros((0, dim), np.float16))
        tmp = cache.with_suffix(".tmp")
        with open(tmp, "wb") as fh:                   # file handle: no .npz append
            np.savez(fh, emb=e, fp=fp)
        tmp.rename(cache)
        shard_embs[kk] = e
        del mm
        print(f"[112b] shard {kk}: embedded {len(e):,} rows "
              f"({(time.time() - t0) / 60:.1f} min elapsed)", flush=True)

    # 2. probe -> probabilities (12_probe_aion math: fp32 standardise, softmax[:,1])
    probs = np.full(len(index), np.nan, np.float32)
    for kk, sub in index.groupby("shard", sort=True):
        ok = sub[sub.ok]
        if not len(ok):
            continue
        X = shard_embs[kk].astype(np.float32)             # fp16 -> fp32 (as v1 load)
        Xs = torch.from_numpy(((X - xmu) / xsd).astype(np.float32)).to(device)
        with torch.no_grad():
            p = torch.softmax(head(Xs), 1)[:, 1].cpu().numpy()
        probs[ok.index.to_numpy()] = p

    df = pd.DataFrame({"row_id": index.row_id, "ok": index.ok, "member_aion": probs})
    tmp = out.with_suffix(out.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    tmp.rename(out)
    f = np.isfinite(probs)
    print(f"[112b] wrote {out} — n={int(f.sum()):,}/{len(df):,} scored, "
          f"mean_p={np.nanmean(probs):.4f} (embeddings cached in {emb_dir})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
