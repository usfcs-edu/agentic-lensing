#!/usr/bin/env python3
"""133_lora_finetune_aion.py — Phase 130: gated LoRA fine-tune of the AION
encoder on the v1 staged train split, native 160px griz shards (runs on
PERLMUTTER, 1x A100-80GB via nersc/shared_gpu.slurm; pytorch/2.8.0 module +
`aion` installed --user, HF models prefetched under HF_HOME — see the
112b_score_aion_pool.py docstring for the install/prefetch one-liners).
Only run if 132_probe_gate_v2.py said lora_justified; --variant comes from its
lora_variant field.

Architecture (verified against the local AION checkout
/home2/benson/lensing-repos/AION):
  frozen CodecManager tokenisation (aion/codecs/manager.py CodecManager.encode,
    discrete tokens -> no_grad)
  -> AION encoder with LoRA adapters: aion/fourm/lora_utils.py:144
    inject_trainable_LoRA(module, rank, scale, target_replace_modules=
    ATTENTION_MODULES) applied IN-PLACE to model.encoder. ATTENTION_MODULES
    (lora_utils.py:20-22) covers self AND cross attention class names
    {Attention, NormAttention, CrossAttention, NormCrossAttention}; the 4M
    encoder itself contains only self-attention (fm_utils.Block -> NormAttention
    since all three HF configs set qk_norm=true) — cross-attention lives in the
    DECODER, which AION.encode never touches, so injecting model.encoder covers
    every attention layer in the gradient path without dead decoder adapters.
    Packed qkv Linear(dim,3*dim) is auto-detected (num_packed_linear=3).
  -> AION.encode (aion/model.py:149, signature encode(input_dict,
    input_mask=None, num_encoder_tokens=256)) with num_encoder_tokens = summed
    token count (the AION.forward arithmetic, as in 112b) so all tokens are
    kept; output (B,T,D) includes the frozen decoder_proj_context+emb residual
    (model.py:126-130), exactly the v1 embedding feature.
  -> mean-pool over T -> trainable Linear(dim,2) head.
Trainable = LoRA(lora_down/lora_up) + head only; asserted < 5% of total params.
At rank 16 each encoder block adds 224*dim LoRA params (qkv 192*dim + proj
32*dim): base 12x224x768 + 1,538 head = 2,065,922 (~0.7% of ~300M); large
24x224x1024 + 2,050 = 5,507,074 (~0.7% of ~800M).

Training: CE with class weight [1, n0/n1] (12_probe_aion's class handling),
bf16 autocast (A100; --amp off for non-Ampere smoke runs), batch --batch 64
grad-accumulated to --eff-batch 256, AdamW lr 1e-4 (head) / 5e-5 (LoRA) wd
1e-4, grad-clip 1.0, --epochs 10 with early stop on val AUC (patience 3,
rank-based AUC — no sklearn on the pytorch module), seed C.SEED. Train/val
rows with nan_frac>0 are excluded (NaN flux would poison the loss); at scoring
time those rows get p=NaN (113-style self-exclusion), ok=False rows likewise.
Resume safety: an atomic per-epoch checkpoint <out-ckpt>.epoch.pt (LoRA+head
state, optimizer, epoch, best-val tracker, rng states; tmp+rename) is written
after every epoch and auto-loaded on start (--no-resume disables), so the 4h
shared-QOS wall is survivable.

Inputs:
  --cutout-root  comma list of 111 output roots (south+north 160px griz shards:
                 cutouts_<k>.npy (n,4,160,160) + index.parquet [row_id, shard,
                 idx_in_shard, ok, nan_frac, g_ok/r_ok/i_ok/z_ok])
  --labels       parquet [row_id, label, split] (data/v2/griz_labels.parquet,
                 built locally from training_split_staged + eval manifests and
                 rsynced up; split 'train'/'val' drive the fit, EVERY labelled
                 row is scored)
Outputs:
  --out-ckpt     data/v2/ckpt/aion_lora_<variant>.pt  (LoRA+head state_dicts,
                 config, per-epoch val log; + sidecar <out-ckpt>.log.json)
  --out-scores   data/v2/scores_aion_lora_<variant>_remote.parquet [row_id, p]
                 (ALL labelled rows; rsynced back for the local gate eval)

    sbatch --export=ALL,CMD='HF_HOME=$SCRATCH/claudenet/hf python \
        133_lora_finetune_aion.py \
        --cutout-root $SCRATCH/claudenet/cutouts/griz_south,$SCRATCH/claudenet/cutouts/griz_north \
        --labels data/v2/griz_labels.parquet --variant base' \
        nersc/shared_gpu.slurm
    # variant=large (or xlarge) needs the 80GB nodes: override the wrapper's
    # `-C gpu` with  sbatch -C 'gpu&hbm80g' --export=ALL,CMD='...' \
    #     nersc/shared_gpu.slurm
    # base fits on the 40GB cards.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import _clib as C

BANDS = ["DES-G", "DES-R", "DES-I", "DES-Z"]      # from 11_embed_aion.py / 112b
MODELS = {"base": "polymathic-ai/aion-base", "large": "polymathic-ai/aion-large"}


# ----- data -------------------------------------------------------------------
class ShardDataset(Dataset):
    """Streams (4,160,160) float32 cutouts from the 111 memmap shards.
    rows: DataFrame [root(int), shard(int), idx_in_shard(int), label(int)].
    Memmaps are opened lazily per (root, shard) inside each worker process."""

    def __init__(self, rows: pd.DataFrame, roots: list[Path]):
        self.root = rows.root.to_numpy(np.int64)
        self.shard = rows.shard.to_numpy(np.int64)
        self.idx = rows.idx_in_shard.to_numpy(np.int64)
        self.label = rows.label.to_numpy(np.int64)
        self.roots = roots
        self._mm: dict[tuple, np.memmap] = {}

    def __len__(self):
        return len(self.root)

    def _shard(self, root: int, shard: int):
        key = (root, shard)
        if key not in self._mm:
            self._mm[key] = np.load(self.roots[root] / f"cutouts_{shard}.npy",
                                    mmap_mode="r")
        return self._mm[key]

    def __getitem__(self, i):
        cut = np.array(self._shard(int(self.root[i]), int(self.shard[i]))
                       [int(self.idx[i])], dtype=np.float32)   # copy off the memmap
        return torch.from_numpy(cut), int(self.label[i]), i


def build_rows(roots: list[Path], labels: pd.DataFrame):
    """Join every root's index.parquet to the labels table."""
    assert labels.row_id.is_unique, \
        "--labels table has duplicate row_id (rebuild griz_labels.parquet)"
    parts = []
    for ri, root in enumerate(roots):
        idx = pd.read_parquet(root / "index.parquet")
        idx["root"] = ri
        parts.append(idx)
    index = pd.concat(parts, ignore_index=True)
    assert index.row_id.is_unique, "row_id collides across --cutout-root entries"
    if "nan_frac" not in index.columns:
        index["nan_frac"] = 0.0
    rows = labels.merge(
        index[["row_id", "root", "shard", "idx_in_shard", "ok", "nan_frac"]],
        on="row_id", how="left")
    miss = rows.root.isna()
    if miss.any():
        print(f"[133] WARNING {int(miss.sum()):,}/{len(rows):,} labelled rows "
              f"have no cutout -> p=NaN")
    return rows


# ----- model ------------------------------------------------------------------
def _num_encoder_tokens(tokens) -> int:
    """copied from aion-1/_aion_embed.py (matches AION.forward's count)."""
    n = 0
    for v in tokens.values():
        n += v.shape[1] if v.dim() == 2 else 1
    return n


def load_lora_model(variant: str, rank: int, scale: float, device):
    """Frozen AION + CodecManager, LoRA injected into the encoder attention.
    NOTE: unlike 112b/load_aion this must NOT torch.set_grad_enabled(False)."""
    from aion.codecs import CodecManager
    from aion.fourm.lora_utils import (ATTENTION_MODULES, inject_trainable_LoRA,
                                       unfreeze_all_LoRA_layers)
    from aion.model import AION

    cm = CodecManager(device=device)
    model = AION.from_pretrained(MODELS[variant]).to(device)
    model.requires_grad_(False)
    inject_trainable_LoRA(model.encoder, rank=rank, scale=scale,
                          target_replace_modules=ATTENTION_MODULES)
    unfreeze_all_LoRA_layers(model)          # belt-and-braces: requires_grad on
    model.to(device)                         # move fresh LoRA linears to device
    head = nn.Linear(model.dim, 2).to(device)
    n_lora = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad)
    n_head = sum(p.numel() for p in head.parameters())
    n_tot = sum(p.numel() for p in model.parameters()) + n_head
    frac = (n_lora + n_head) / n_tot
    print(f"[133] {variant} rank={rank}: trainable LoRA={n_lora:,} + head={n_head:,} "
          f"= {n_lora + n_head:,} / total {n_tot:,} ({100 * frac:.2f}%)")
    assert frac < 0.05, f"trainable fraction {frac:.3f} >= 5% — wrong injection?"
    for n, p in model.named_parameters():
        assert (not p.requires_grad) or "lora" in n, f"non-LoRA trainable: {n}"
    return model, cm, head


def forward_batch(model, cm, head, flux, device, amp: bool):
    """flux (B,4,160,160) float32 -> logits (B,2). Tokenisation is frozen/
    discrete (no_grad); grads flow through the LoRA'd encoder + head."""
    from aion.modalities import LegacySurveyImage

    with torch.no_grad():
        tokens = cm.encode(LegacySurveyImage(flux=flux.to(device), bands=BANDS))
    net = _num_encoder_tokens(tokens)
    ctx = (torch.autocast("cuda", dtype=torch.bfloat16) if amp
           else torch.autocast("cuda", enabled=False))
    with ctx:
        emb = model.encode(tokens, num_encoder_tokens=net)   # (B,T,D)
        logits = head(emb.mean(dim=1))
    return logits.float()


def auc_rank(y: np.ndarray, p: np.ndarray) -> float:
    """Mann-Whitney AUC (average ranks; no sklearn on the pytorch module)."""
    y = np.asarray(y); p = np.asarray(p, np.float64)
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p), np.float64)
    ranks[order] = np.arange(1, len(p) + 1)
    ps = p[order]
    # average ties
    i = 0
    while i < len(ps):
        j = i
        while j + 1 < len(ps) and ps[j + 1] == ps[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = 0.5 * (i + j) + 1
        i = j + 1
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def trainable_state(model, head):
    sd = {f"model.{n}": p.detach().cpu().clone()
          for n, p in model.named_parameters() if p.requires_grad}
    sd.update({f"head.{n}": p.detach().cpu().clone()
               for n, p in head.state_dict().items()})
    return sd


def load_trainable_state(model, head, sd):
    model.load_state_dict({k[len("model."):]: v for k, v in sd.items()
                           if k.startswith("model.")}, strict=False)
    head.load_state_dict({k[len("head."):]: v for k, v in sd.items()
                          if k.startswith("head.")})


@torch.no_grad()
def score_loader(model, cm, head, loader, n_rows, device, amp, tag):
    model.eval(); head.eval()
    p = np.full(n_rows, np.nan, np.float32)
    t0 = time.time()
    for bi, (flux, _, idxs) in enumerate(loader):
        logits = forward_batch(model, cm, head, flux, device, amp)
        p[idxs.numpy()] = torch.softmax(logits, 1)[:, 1].cpu().numpy()
        if bi % 50 == 0:
            print(f"[133] {tag}: batch {bi}/{len(loader)} "
                  f"({(time.time() - t0) / 60:.1f} min)", flush=True)
    return p


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cutout-root", required=True,
                    help="comma list of 111 shard roots (south,north)")
    ap.add_argument("--labels", required=True,
                    help="parquet [row_id,label,split] (griz_labels.parquet)")
    ap.add_argument("--variant", default="base", choices=tuple(MODELS))
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--lora-scale", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--eff-batch", type=int, default=256,
                    help="effective batch via grad accumulation")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--patience", type=int, default=3,
                    help="early stop on val AUC")
    ap.add_argument("--lr-head", type=float, default=1e-4)
    ap.add_argument("--lr-lora", type=float, default=5e-5)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--amp", choices=("bf16", "off"), default="bf16")
    ap.add_argument("--out-ckpt", default=None,
                    help="default data/v2/ckpt/aion_lora_<variant>.pt")
    ap.add_argument("--out-scores", default=None,
                    help="default data/v2/scores_aion_lora_<variant>_remote.parquet")
    ap.add_argument("--limit", type=int, default=0,
                    help="smoke: cap train/val/score rows (0 = all)")
    ap.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True,
                    help="resume from <out-ckpt>.epoch.pt if present "
                         "(--no-resume disables)")
    args = ap.parse_args()

    torch.manual_seed(C.SEED); np.random.seed(C.SEED)
    if not torch.cuda.is_available():
        raise SystemExit("[133] CUDA required (A100); refusing CPU run")
    device = torch.device("cuda")
    amp = args.amp == "bf16"
    accum = max(1, -(-args.eff_batch // args.batch))   # ceil
    v2 = C.DATA / "v2"
    out_ckpt = Path(args.out_ckpt or v2 / "ckpt" / f"aion_lora_{args.variant}.pt")
    out_scores = Path(args.out_scores or
                      v2 / f"scores_aion_lora_{args.variant}_remote.parquet")
    out_ckpt.parent.mkdir(parents=True, exist_ok=True)
    out_scores.parent.mkdir(parents=True, exist_ok=True)

    # --- rows ------------------------------------------------------------------
    roots = [Path(r) for r in args.cutout_root.split(",") if r]
    labels = pd.read_parquet(args.labels)
    rows = build_rows(roots, labels)
    have = rows[rows.root.notna() & rows.ok.astype("boolean").fillna(False)].copy()
    clean = have[have.nan_frac == 0]
    tr = clean[clean.split == "train"]
    va = clean[clean.split == "val"]
    sc = have                                           # score every ok row
    if args.limit:
        tr, va, sc = (d.sample(n=min(args.limit, len(d)), random_state=C.SEED)
                      for d in (tr, va, sc))
    print(f"[133] rows: labelled={len(rows):,} ok={len(have):,} "
          f"(nan_frac>0 excluded from fit: {len(have) - len(clean):,}) | "
          f"train={len(tr):,} (pos={int(tr.label.sum()):,}) val={len(va):,} "
          f"score={len(sc):,} | accum={accum} (eff batch {accum * args.batch})")
    assert len(tr) and len(va), "empty train/val — check --labels split values"

    cols = ["root", "shard", "idx_in_shard", "label"]
    g = torch.Generator(); g.manual_seed(C.SEED)
    dl_kw = dict(batch_size=args.batch, num_workers=args.num_workers,
                 pin_memory=True, persistent_workers=args.num_workers > 0)
    tl = DataLoader(ShardDataset(tr[cols], roots), shuffle=True, drop_last=True,
                    generator=g, **dl_kw)
    vl = DataLoader(ShardDataset(va[cols], roots), shuffle=False, **dl_kw)

    # --- model + optimiser -------------------------------------------------------
    model, cm, head = load_lora_model(args.variant, args.lora_rank,
                                      args.lora_scale, device)
    lora_params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(
        [{"params": head.parameters(), "lr": args.lr_head},
         {"params": lora_params, "lr": args.lr_lora}],
        weight_decay=args.weight_decay)
    y = tr.label.to_numpy()
    cw = torch.tensor([1.0, float((y == 0).sum() / max((y == 1).sum(), 1))],
                      device=device)                    # 12_probe_aion class weight
    lossf = nn.CrossEntropyLoss(weight=cw)
    print(f"[133] class weight pos={float(cw[1]):.2f} | lr head={args.lr_head} "
          f"lora={args.lr_lora} | amp={args.amp}")

    # --- train (resume-safe: the 4h shared-QOS wall must be survivable) -----------
    best_auc, best_state, best_ep, bad, log, start_ep = -1.0, None, -1, 0, [], 0
    epoch_ckpt = Path(f"{out_ckpt}.epoch.pt")
    if args.resume and epoch_ckpt.exists():
        rc = torch.load(epoch_ckpt, map_location="cpu", weights_only=False)
        load_trainable_state(model, head, rc["trainable_state"])
        opt.load_state_dict(rc["optimizer"])
        best_auc, best_state = rc["best_val_auc"], rc["best_state"]
        best_ep, bad, log = rc["best_epoch"], rc["bad"], rc["val_log"]
        torch.set_rng_state(rc["rng"]["torch"])
        torch.cuda.set_rng_state(rc["rng"]["cuda"])
        np.random.set_state(rc["rng"]["numpy"])
        g.set_state(rc["rng"]["loader_g"])
        start_ep = rc["epoch"] + 1
        print(f"[133] resumed {epoch_ckpt} -> continue at epoch {start_ep} "
              f"(best ep{best_ep} auc={best_auc:.4f}, bad={bad})")
        if bad >= args.patience:
            print(f"[133] resume: early stop already reached — skipping training")
            start_ep = args.epochs
    for ep in range(start_ep, args.epochs):
        model.train(); head.train()
        t0, tot, nb = time.time(), 0.0, 0
        opt.zero_grad()
        for bi, (flux, yb, _) in enumerate(tl):
            logits = forward_batch(model, cm, head, flux, device, amp)
            loss = lossf(logits, yb.to(device)) / accum
            loss.backward()
            tot += float(loss) * accum; nb += 1
            if (bi + 1) % accum == 0 or bi + 1 == len(tl):
                torch.nn.utils.clip_grad_norm_(
                    [p for grp in opt.param_groups for p in grp["params"]], 1.0)
                opt.step(); opt.zero_grad()
            if bi % 100 == 0:
                print(f"[133] ep{ep} {bi}/{len(tl)} loss={tot / max(nb, 1):.4f} "
                      f"({(time.time() - t0) / 60:.1f} min)", flush=True)
        pv = score_loader(model, cm, head, vl, len(va), device, amp, f"ep{ep}-val")
        m = np.isfinite(pv)
        auc = auc_rank(va.label.to_numpy()[m], pv[m])
        log.append({"epoch": ep, "train_loss": tot / max(nb, 1), "val_auc": auc,
                    "minutes": (time.time() - t0) / 60})
        print(f"[133] epoch {ep}: loss={log[-1]['train_loss']:.4f} "
              f"val_auc={auc:.4f} ({log[-1]['minutes']:.1f} min)", flush=True)
        if auc > best_auc:
            best_auc, best_ep, bad = auc, ep, 0
            best_state = trainable_state(model, head)
        else:
            bad += 1
        # atomic per-epoch resume checkpoint (tmp+rename)
        ec = {"trainable_state": trainable_state(model, head),
              "optimizer": opt.state_dict(), "epoch": ep,
              "best_val_auc": best_auc, "best_epoch": best_ep,
              "best_state": best_state, "bad": bad, "val_log": log,
              "rng": {"torch": torch.get_rng_state(),
                      "cuda": torch.cuda.get_rng_state(),
                      "numpy": np.random.get_state(),
                      "loader_g": g.get_state()}}
        etmp = epoch_ckpt.with_suffix(".tmp")
        torch.save(ec, etmp)
        etmp.rename(epoch_ckpt)
        print(f"[133] saved {epoch_ckpt} (ep{ep}, resume-safe)", flush=True)
        if bad >= args.patience:
            print(f"[133] early stop at epoch {ep} (best ep{best_ep} "
                  f"auc={best_auc:.4f})")
            break

    assert best_state is not None, "no epoch completed — check --epochs"
    load_trainable_state(model, head, best_state)
    ck = {"lora_head_state": best_state, "val_log": log,
          "best_val_auc": best_auc, "best_epoch": best_ep,
          "config": {"variant": args.variant, "lora_rank": args.lora_rank,
                     "lora_scale": args.lora_scale, "dim": int(model.dim),
                     "bands": BANDS, "seed": C.SEED, "batch": args.batch,
                     "eff_batch": accum * args.batch, "epochs": args.epochs,
                     "lr_head": args.lr_head, "lr_lora": args.lr_lora,
                     "weight_decay": args.weight_decay, "amp": args.amp,
                     "labels": str(args.labels),
                     "cutout_roots": [str(r) for r in roots]}}
    tmp = out_ckpt.with_suffix(".tmp")
    torch.save(ck, tmp); tmp.rename(out_ckpt)
    Path(f"{out_ckpt}.log.json").write_text(json.dumps(
        {k: ck[k] for k in ("val_log", "best_val_auc", "best_epoch", "config")},
        indent=2))
    print(f"[133] wrote {out_ckpt} (best val AUC={best_auc:.4f} @ ep{best_ep})")

    # --- score ALL labelled ok rows with the best state ---------------------------
    sl = DataLoader(ShardDataset(sc[cols].assign(label=0), roots),
                    shuffle=False, **dl_kw)
    ps = score_loader(model, cm, head, sl, len(sc), device, amp, "score")
    ps[(sc.nan_frac > 0).to_numpy()] = np.nan          # 113-style self-exclusion
    out = pd.DataFrame({"row_id": sc.row_id.to_numpy(), "p": ps})
    # rows without a usable cutout still appear, p=NaN (never silently dropped)
    rest = rows.loc[~rows.row_id.isin(out.row_id), ["row_id"]].copy()
    rest["p"] = np.nan
    out = pd.concat([out, rest], ignore_index=True)
    tmp = out_scores.with_suffix(out_scores.suffix + ".tmp")
    out.to_parquet(tmp, index=False); tmp.rename(out_scores)
    print(f"[133] wrote {out_scores} — n={len(out):,} "
          f"finite={int(np.isfinite(out.p).sum()):,} "
          f"mean_p={np.nanmean(out.p):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
