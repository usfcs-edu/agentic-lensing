#!/usr/bin/env python3
"""20_train_member.py — train ONE supervised ensemble member and score the shared
eval manifests. Launch once per GPU (independent processes; no NVLink needed).

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
      /home2/benson/.venvs/claudenet/bin/python 20_train_member.py --name shielded_A

Writes:
  data/ckpt/member_<name>.pt
  data/scores_member_<name>.parquet   [split,row_id,label,p]   (val/testneg/storfer/inchausti)
"""
from __future__ import annotations

import argparse
import json
import time

import pandas as pd
import torch

import _clib as C
import _train as T

EPOCHS = {"shielded": 45, "l18": 45, "dihedral": 30, "efficientnet": 25}


def build_model(arch, variant=None):
    M = C.models()
    if arch == "shielded":
        return M["ShieldedDeepLens"](in_channels=3, **C.CFG194)
    if arch == "l18":
        return M["CMUDeepLens"](in_channels=3)
    if arch == "efficientnet":
        kw = {"variant": variant} if variant else {}
        return M["EfficientNetV2Lens"](pretrained=True, **kw)
    if arch == "dihedral":
        return T.DihedralPool(M["ShieldedDeepLens"](in_channels=3, **C.CFG194))
    raise ValueError(arch)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    args = ap.parse_args()
    roster = {m["name"]: m for m in json.load(open(C.DATA / "members.json"))}
    spec = roster[args.name]
    arch, aug_seed = spec["arch"], spec["aug_seed"]
    score_arch = "efficientnet" if arch == "efficientnet" else "shielded"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_parquet(C.DATA / f"member_{args.name}_train.parquet")
    model = build_model(arch, spec.get("variant"))
    n_params = sum(p.numel() for p in model.parameters())
    # dihedral holds 8 graphs (D4 pooling) -> small batch + accumulation to fit 24GB
    batch, accum = {"efficientnet": (128, 2), "dihedral": (32, 4)}.get(arch, (128, 1))
    print(f"[train] {args.name} arch={arch} params={n_params:,} epochs={EPOCHS[arch]} "
          f"batch={batch} accum={accum} aug_seed={aug_seed} n_train={(df.split=='train').sum()}")
    t0 = time.time()
    model, val_auc, mean, std = T.train_supervised(
        model, arch, df, device, epochs=EPOCHS[arch], batch=batch, lr=1e-3,
        decay_ep=max(8, EPOCHS[arch] // 3), accum=accum, aug_seed=aug_seed)
    print(f"[train] {args.name} best_val_auc={val_auc:.4f} ({(time.time()-t0)/60:.1f}m)")

    torch.save({"state_dict": model.state_dict(), "arch": arch, "score_arch": score_arch,
                "mean": mean.tolist(), "std": std.tolist(), "val_auc": val_auc,
                "shielded_cfg": C.CFG194 if arch in ("shielded", "dihedral") else None},
               C.CKPT / f"member_{args.name}.pt")

    # score shared eval manifests
    rows = []
    for sp in ("val", "testneg", "storfer", "inchausti"):
        d = pd.read_parquet(C.DATA / f"eval_{sp}.parquet").copy()
        d["p"] = T.score_df(model, score_arch, d, mean, std, device)
        d["split"] = sp
        rows.append(d[["split", "row_id", "label", "p"]])
        print(f"[score] {sp:10s} n={len(d)} mean_p={d['p'].mean():.3f}")
    pd.concat(rows, ignore_index=True).to_parquet(C.DATA / f"scores_member_{args.name}.parquet", index=False)
    print(f"[20] {args.name} done")


if __name__ == "__main__":
    main()
