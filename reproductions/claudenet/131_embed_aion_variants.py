#!/usr/bin/env python3
"""131_embed_aion_variants.py — Phase 130: embed native 160px-griz cutout
shards with a frozen AION encoder (base/large/xlarge), one GPU. Runs on
Perlmutter (pytorch/2.8.0 module + `aion` --user; see 112b's docstring for the
install/HF-prefetch one-liners — models are cached under
HF_HOME=$SCRATCH/claudenet/hf) or locally under /home2/benson/.venvs/aion.

This is the NATIVE-input replacement for the v1 degraded AION path: input
shards are 111 output with --size 160 --bands griz (n,4,160,160 float32, band
order g,r,i,z — exactly the [g,r,i,z] order 112b's to_griz160 fed the
LegacySurveyImage codec, so NO conversion happens here; the to_griz160
bilinear-resize + synthetic-i step is deliberately absent). North shards have
a zero-filled i plane with i_ok=False in the index (DR9 north has no i band) —
recorded per row in the output index so downstream probes can condition on it.

Embedding math is 112b's embed_batch minus the conversion: CodecManager.encode
-> AION.encode(num_encoder_tokens) -> fp32 mean-pool over tokens -> float16.
autocast fp16 per v1 aion-1/_config.DEFAULT_AMP (base/large fp32, xlarge fp16)
and batch per DEFAULT_BATCH (base 128, large 96, xlarge 24); --amp/--batch
override. Per-shard embedding caches under <out-root>/emb_<variant>_cache/ are
fingerprinted to the variant + EFFECTIVE amp mode (the resolved on/off string)
+ ok-row ids + source shard file (112b's scheme, extended) -> resume-safe
across job preemptions, and a rerun with a different --amp can never reuse a
stale cache.

Output (per variant, embeddings for ok rows ONLY — extraction-failed rows and
rows whose embedding comes back non-finite, e.g. NaN coadd pixels, are
excluded, with the index recording everything):

    <out-root>/emb_<variant>.npy            float16 (N_ok, dim)
    <out-root>/emb_<variant>_index.parquet  row_id, ok, i_ok, nan_frac,
                                            emb_row (-1 = no embedding)

    sbatch --export=ALL,CMD='HF_HOME=$SCRATCH/claudenet/hf python 131_embed_aion_variants.py \\
        --cutout-root $SCRATCH/claudenet/cutouts/griz_south \\
        --out-root $SCRATCH/claudenet/emb/griz_south --variant base' \\
        nersc/shared_gpu.slurm
    # repeat per --variant {base,large,xlarge} x {griz_south, griz_north}
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import _clib as C

BANDS = ["DES-G", "DES-R", "DES-I", "DES-Z"]   # == shard band order g,r,i,z
# copied from aion-1/_config.py (not importable on Perlmutter):
MODELS = {"base": "polymathic-ai/aion-base", "large": "polymathic-ai/aion-large",
          "xlarge": "polymathic-ai/aion-xlarge"}
DEFAULT_BATCH = {"base": 128, "large": 96, "xlarge": 24}
DEFAULT_AMP = {"base": False, "large": False, "xlarge": True}


def _num_encoder_tokens(tokens) -> int:
    """copied from aion-1/_aion_embed.py (matches AION.forward's count)."""
    n = 0
    for v in tokens.values():
        n += v.shape[1] if v.dim() == 2 else 1
    return n


def load_aion(variant: str, device):
    """aion-1/_aion_embed.load_model: frozen AION + codec manager (as 112b)."""
    from aion.codecs import CodecManager
    from aion.model import AION

    torch.set_grad_enabled(False)
    cm = CodecManager(device=device)
    model = AION.from_pretrained(MODELS[variant]).to(device).eval()
    return model, cm


def embed_batch(model, cm, griz: np.ndarray, device, amp: bool) -> np.ndarray:
    """(B,4,160,160) native griz -> (B,D) float16 mean-pooled embeddings.
    112b's embed_batch WITHOUT to_griz160 (input is already native griz 160)."""
    from aion.modalities import LegacySurveyImage

    flux = torch.as_tensor(griz, device=device)
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cutout-root", required=True,
                    help="111 --size 160 --bands griz output dir "
                         "(cutouts_<k>.npy + index.parquet)")
    ap.add_argument("--out-root", required=True,
                    help="dir for emb_<variant>.npy + emb_<variant>_index.parquet")
    ap.add_argument("--variant", default="base", choices=tuple(MODELS))
    ap.add_argument("--batch", type=int, default=0,
                    help="0 = v1 aion-1/_config.DEFAULT_BATCH for the variant")
    ap.add_argument("--amp", choices=("auto", "on", "off"), default="auto",
                    help="fp16 autocast; auto = v1 aion-1/_config.DEFAULT_AMP")
    args = ap.parse_args()
    amp = DEFAULT_AMP[args.variant] if args.amp == "auto" else (args.amp == "on")
    amp_str = "on" if amp else "off"                 # EFFECTIVE amp mode
    batch = args.batch or DEFAULT_BATCH[args.variant]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(C.SEED)
    root, out_root = Path(args.cutout_root), Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    cache_dir = out_root / f"emb_{args.variant}_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    index = (pd.read_parquet(root / "index.parquet")
             .sort_values(["shard", "idx_in_shard"]).reset_index(drop=True))
    if "i_ok" not in index.columns:
        raise ValueError(f"{root}/index.parquet has no i_ok column — shards are "
                         f"not 111 --bands griz output")
    print(f"[131] {len(index):,} rows in {index.shard.nunique()} shards; "
          f"{int(index.ok.sum()):,} ok, {int((index.ok & ~index.i_ok).sum()):,} "
          f"ok-without-i | variant={args.variant} amp={amp} batch={batch} "
          f"device={device}")

    model, cm = load_aion(args.variant, device)

    # embed ok rows shard by shard (cache fingerprinted to variant + effective
    # amp + ok-row ids + source shard file, 112b's scheme extended ->
    # resume-safe, never stale, and a different --amp can never reuse a cache)
    t0, shard_embs, dim = time.time(), {}, None
    for kk, sub in index.groupby("shard", sort=True):
        ok = sub[sub.ok]
        src = root / f"cutouts_{kk}.npy"
        st = src.stat()
        h = pd.util.hash_pandas_object(ok.row_id.astype(str), index=False).to_numpy()
        fp = (f"{args.variant}:{amp_str}:{len(ok)}:"
              f"{int(h.sum()) & 0xFFFFFFFFFFFF:x}:{st.st_size}:{int(st.st_mtime)}")
        cache = cache_dir / f"emb_{kk}.npz"
        if cache.exists():
            try:
                z = np.load(cache, allow_pickle=False)
                e, fp_stored = z["emb"], str(z["fp"])
            except Exception:
                e, fp_stored = None, None
            if e is not None and fp_stored == fp and len(e) == len(ok) \
                    and (len(ok) == 0 or dim is None or e.shape[1] == dim):
                shard_embs[kk] = e
                dim = e.shape[1] if len(e) else dim
                print(f"[131] shard {kk}: resume {cache.name} {e.shape}")
                continue
            print(f"[131] shard {kk}: stale embedding cache -> re-embedding")
        mm = np.load(src, mmap_mode="r")
        if mm.shape[1:] != (4, 160, 160):
            raise ValueError(f"cutouts_{kk}.npy shape {mm.shape}; expected "
                             f"(*,4,160,160) — wrong shards for the native path?")
        sidx = ok.idx_in_shard.to_numpy()
        parts = []
        for s in range(0, len(ok), batch):
            griz = np.asarray(mm[sidx[s:s + batch]], dtype=np.float32)
            parts.append(embed_batch(model, cm, griz, device, amp))
        e = np.concatenate(parts, 0) if parts else np.zeros((0, dim or 1), np.float16)
        dim = e.shape[1] if len(e) else dim
        tmp = cache.with_suffix(".tmp")
        with open(tmp, "wb") as fh:                   # file handle: no .npz append
            np.savez(fh, emb=e, fp=fp)
        tmp.rename(cache)
        shard_embs[kk] = e
        del mm
        print(f"[131] shard {kk}: embedded {len(e):,} rows "
              f"({(time.time() - t0) / 60:.1f} min elapsed)", flush=True)
    if dim is None:
        raise RuntimeError("no ok rows anywhere — nothing to embed")

    # assemble: ok rows only, drop non-finite embeddings (NaN coadd pixels)
    emb_all = np.concatenate([shard_embs[kk] for kk in sorted(shard_embs)
                              if len(shard_embs[kk])] or [np.zeros((0, dim), np.float16)], 0)
    ok_pos = index.index[index.ok].to_numpy()         # index rows, shard-major order
    assert len(emb_all) == len(ok_pos)
    finite = np.isfinite(emb_all.astype(np.float32)).all(axis=1)
    emb_rows = np.full(len(index), -1, np.int64)
    emb_rows[ok_pos[finite]] = np.arange(int(finite.sum()))
    out_idx = pd.DataFrame({
        "row_id": index.row_id, "ok": emb_rows >= 0, "i_ok": index.i_ok,
        "nan_frac": index.nan_frac, "emb_row": emb_rows,
    })
    emb = emb_all[finite]

    npy = out_root / f"emb_{args.variant}.npy"
    tmp = npy.with_suffix(".npy.tmp")
    with open(tmp, "wb") as fh:        # file handle: np.save must not append .npy
        np.save(fh, emb)
    tmp.rename(npy)
    pq = out_root / f"emb_{args.variant}_index.parquet"
    tmp = pq.with_suffix(".parquet.tmp")
    out_idx.to_parquet(tmp, index=False)
    tmp.rename(pq)
    n_nonfin = int(len(ok_pos) - finite.sum())
    print(f"[131] wrote {npy} {emb.shape} float16 + {pq} — "
          f"{int(out_idx.ok.sum()):,}/{len(out_idx):,} rows embedded "
          f"({n_nonfin} non-finite excluded; "
          f"{int((out_idx.ok & ~out_idx.i_ok).sum()):,} embedded with zero-filled i); "
          f"{(time.time() - t0) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
