#!/usr/bin/env python3
"""390_finetune_dr11.py — DR11-native warm-start fine-tune of the 3 swappable members
(task D). Loads each member's south _b50 checkpoint, fine-tunes BRIEFLY at a low LR on the
DR11 table (389) so it adapts features to DR11 pixels + the DR11 hard negatives without drifting
far from the b50 init. effnet_B + zoobot_N stay frozen (the random-FPR / diversity anchors).

Reuses the exact model factory (20_train_member) + _train.train_supervised; differs from 121
only in (a) no negative-mining swap (trains the table as-is) and (b) an exposed low LR / fewer
epochs suited to fine-tuning rather than from-scratch.

  CUDA_VISIBLE_DEVICES=0 /home2/benson/.venvs/claudenet/bin/python 390_finetune_dr11.py \
      --member effnet_S2 --epochs 12 --lr 3e-4
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

import _clib as C

V2 = C.DATA / "v2"
MEMBERS = ("effnet_S2", "effnet_B3", "resnet46_C")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--member", required=True, choices=MEMBERS)
    ap.add_argument("--table", default=str(C.DATA / "member_dr11_train.parquet"))
    ap.add_argument("--init-ckpt", default=None, help="default data/v2/ckpt/member_<m>_b50.pt")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out-suffix", default="_dr11")
    args = ap.parse_args()

    import torch
    import _train as T
    M20 = C._load("cn390_m20", C.ROOT / "20_train_member.py")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    spec = {m["name"]: m for m in json.load(open(C.DATA / "members.json"))}[args.member]
    arch, aug_seed = spec["arch"], spec["aug_seed"]
    score_arch = "efficientnet" if arch == "efficientnet" else "shielded"
    df = pd.read_parquet(args.table)
    df["fits_dir"] = df["fits_dir"].astype(str)        # per-row dirs (honored by LensDataset)

    model = M20.build_model(arch, spec.get("variant"))
    init = args.init_ckpt or str(V2 / "ckpt" / f"member_{args.member}_b50.pt")
    ck = torch.load(init, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["state_dict"])
    print(f"[390] {args.member}: warm-start from {init} (b50 val_auc={ck.get('val_auc', 0):.4f}); "
          f"arch={arch} epochs={args.epochs} lr={args.lr:g} "
          f"n_train={(df.split=='train').sum():,} n_val={(df.split=='val').sum():,}")

    batch, accum = {"efficientnet": (128, 2), "dihedral": (32, 4)}.get(arch, (128, 1))
    t0 = time.time()
    model, val_auc, mean, std = T.train_supervised(
        model, arch, df, device, epochs=args.epochs, batch=batch, lr=args.lr,
        decay_ep=max(4, args.epochs // 3), accum=accum, aug_seed=aug_seed)
    print(f"[390] {args.member}{args.out_suffix} best_val_auc={val_auc:.4f} "
          f"({(time.time()-t0)/60:.1f}m)")

    out = V2 / "ckpt" / f"member_{args.member}_b50{args.out_suffix}.pt"
    torch.save({"state_dict": model.state_dict(), "arch": arch, "score_arch": score_arch,
                "mean": mean.tolist(), "std": std.tolist(), "val_auc": val_auc,
                "shielded_cfg": C.CFG194 if arch in ("shielded", "dihedral") else None,
                "variant": spec.get("variant"),
                "finetune": {"init": init, "epochs": args.epochs, "lr": args.lr,
                             "table": args.table}}, out)
    print(f"[390] saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
