#!/usr/bin/env python3
"""22_member_aion.py — assemble the AION-1 ensemble member's scores on all four
shared eval manifests (val/testneg/storfer/inchausti), reusing the Phase-0 probe.

The held-out eval sets (testneg/storfer/inchausti) were already AION-scored in the
gate (scores_aion_gate.parquet). Here we only add the `val` split (needed for
calibration + combiner fitting): build griz inputs, embed via the aion venv
(subprocess), then score with the frozen probe.

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 \
      /home2/benson/.venvs/claudenet/bin/python 22_member_aion.py --variant base
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import _clib as C
import _trainlib as TL

EMB = C.EMB


class MLPProbe(nn.Module):  # mirror of 12_probe_aion.MLPProbe
    def __init__(self, dim, hidden=256, k=2, p=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(),
                                 nn.Dropout(p), nn.Linear(hidden, k))

    def forward(self, x):
        return self.net(x)


def to_griz160(grz):  # (B,3,101,101) -> (B,4,160,160), i=0.5*(r+z)
    x = F.interpolate(torch.from_numpy(grz), size=(160, 160), mode="bilinear", align_corners=False)
    g, r, z = x[:, 0], x[:, 1], x[:, 2]
    return torch.stack([g, r, 0.5 * (r + z), z], 1).numpy().astype(np.float32)


def build_val_inputs():
    d = pd.read_parquet(C.DATA / "eval_val.parquet").copy()
    d["fits_path"] = d.apply(lambda r: str(Path(r.fits_dir) / f"{r.row_id}.fits"), axis=1)
    flux, keep, buf, bi = [], [], [], []

    def flush():
        if buf:
            flux.append(to_griz160(np.stack(buf).astype(np.float32))); keep.extend(bi)
            buf.clear(); bi.clear()
    for j, r in enumerate(d.itertuples()):
        try:
            a = TL.load_fits_cube(Path(r.fits_path))
        except Exception:
            continue
        if a.shape == (3, 101, 101):
            buf.append(a); bi.append(j)
            if len(buf) >= 256:
                flush()
    flush()
    arr = np.concatenate(flux, 0) if flux else np.zeros((0, 4, 160, 160), np.float32)
    man = d.iloc[keep].reset_index(drop=True)
    np.save(EMB / "aion_in_val.npy", arr)
    man.to_parquet(EMB / "aion_in_val_manifest.parquet", index=False)
    print(f"[22] val griz inputs {arr.shape}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="base")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. build + embed val (embed via aion venv subprocess)
    if not (EMB / f"aion_emb_val_{args.variant}.npy").exists():
        build_val_inputs()
        env = {**__import__("os").environ, "HF_HOME": C.HF_HOME, "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}
        subprocess.run([C.AION_PY, str(C.ROOT / "11_embed_aion.py"),
                        "--variant", args.variant, "--gpus", ",".join(map(str, C.GPUS)),
                        "--splits", "val"], check=True, env=env)

    # 2. score val with the frozen probe
    ck = torch.load(str(C.CKPT / f"aion_probe_{args.variant}.pt"), map_location="cpu", weights_only=False)
    head = MLPProbe(ck["dim"]).to(device); head.load_state_dict(ck["state_dict"]); head.eval()
    Xv = np.load(EMB / f"aion_emb_val_{args.variant}.npy").astype(np.float32)
    man = pd.read_parquet(EMB / "aion_in_val_manifest.parquet")
    Xs = torch.from_numpy(((Xv - ck["xmu"]) / ck["xsd"]).astype(np.float32)).to(device)
    with torch.no_grad():
        pv = torch.softmax(head(Xs), 1)[:, 1].cpu().numpy()
    val = man[["row_id", "label"]].copy(); val["p"] = pv; val["split"] = "val"

    # 3. reuse gate scores for the held-out eval sets
    gate = pd.read_parquet(C.DATA / "scores_aion_gate.parquet")[["split", "row_id", "label", "p_aion"]]
    gate = gate.rename(columns={"p_aion": "p"})

    out = pd.concat([val[["split", "row_id", "label", "p"]], gate], ignore_index=True)
    out.to_parquet(C.DATA / "scores_member_aion.parquet", index=False)
    for sp in ("val", "testneg", "storfer", "inchausti"):
        s = out[out.split == sp]
        print(f"[22] aion {sp:10s} n={len(s)} mean_p={s['p'].mean():.3f}")
    print("[22] wrote scores_member_aion.parquet")


if __name__ == "__main__":
    main()
