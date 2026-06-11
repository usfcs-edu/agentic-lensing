#!/usr/bin/env python3
"""80_equivariance.py — Phase 7: does enforcing the dihedral (D4) orientation
invariance lenses physically have help? We test it the cheap way — TEST-TIME D4
pooling on already-trained members (average the prediction over the 8 rotation/flip
symmetries, no retraining) — and report the matched-FPR recovery change.

This isolates the inductive-bias effect (orientation invariance) without the 8x
training cost that made a fully D4-equivariant member impractical here (Phase 1).
The full equivariant-TRAINING label-efficiency study (escnn C4/D4) is the documented
NERSC follow-up.

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 \
      /home2/benson/.venvs/claudenet/bin/python 80_equivariance.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import _clib as C
import _ensemble as E
import _minelib as ML


def d4_transforms(x):
    outs = []
    for k in range(4):
        xr = torch.rot90(x, k, dims=[2, 3])
        outs.append(xr)
        outs.append(torch.flip(xr, dims=[3]))
    return outs


@torch.no_grad()
def score(model, score_arch, rows, cache, mean, std, device, d4=False, batch=256):
    mt = torch.from_numpy(mean.reshape(3, 1, 1).astype(np.float32))
    st = torch.from_numpy(std.reshape(3, 1, 1).astype(np.float32))
    out = np.full(len(rows), np.nan, np.float32)
    buf, idx = [], []

    def prob(x):
        if score_arch == "efficientnet":
            return torch.softmax(model(x), 1)[:, 1]
        return torch.sigmoid(model(x))

    def flush():
        if not buf:
            return
        x = torch.clamp((torch.from_numpy(np.stack(buf)) - mt) / st, -250, 250).to(device)
        p = (torch.stack([prob(t) for t in d4_transforms(x)], 0).mean(0) if d4 else prob(x)).cpu().numpy()
        for j, ii in enumerate(idx):
            out[ii] = p[j]
        buf.clear(); idx.clear()

    for i, r in enumerate(rows.itertuples()):
        k = ML._key(r.fits_dir, r.row_id)
        if k in cache:
            buf.append(cache[k]); idx.append(i)
            if len(buf) >= batch:
                flush()
    flush()
    return out


def load_member(name, device):
    ck = torch.load(str(C.CKPT / f"member_{name}.pt"), map_location="cpu", weights_only=False)
    M = C.models()
    if ck["arch"] == "shielded":
        m = M["ShieldedDeepLens"](in_channels=3, **ck["shielded_cfg"])
    elif ck["arch"] == "l18":
        m = M["CMUDeepLens"](in_channels=3)
    else:
        raise ValueError(ck["arch"])
    m.load_state_dict(ck["state_dict"]); m.to(device).eval()
    return m, ck["score_arch"], np.array(ck["mean"]), np.array(ck["std"])


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ev = {}
    for sp, f in (("testneg", "eval_testneg.parquet"), ("storfer", "eval_storfer.parquet"),
                  ("inchausti", "eval_inchausti.parquet")):
        d = pd.read_parquet(C.DATA / f)[["row_id", "label", "fits_dir"]]
        if "cutouts" not in str(d.fits_dir.iloc[0]):
            d["fits_dir"] = d["fits_dir"].apply(lambda p: str(C.DATA / Path(str(p)).name))
        ev[sp] = d
    cache = ML.load_cache(ev["testneg"], ev["storfer"], ev["inchausti"])

    res = {}
    for name in ("shielded_A", "resnet46_C"):
        m, sa, mean, std = load_member(name, device)
        row = {}
        for d4 in (False, True):
            neg = score(m, sa, ev["testneg"], cache, mean, std, device, d4=d4)
            tag = "d4" if d4 else "plain"
            for cat in ("storfer", "inchausti"):
                cand = score(m, sa, ev[cat], cache, mean, std, device, d4=d4)
                row[f"{cat}_{tag}"] = E.recovery_at_fpr(neg, cand)[0.01]["recovery"]
        res[name] = row
        print(f"[equiv] {name}: storfer@1% plain={row['storfer_plain']:.3f} "
              f"D4={row['storfer_d4']:.3f} (delta {row['storfer_d4']-row['storfer_plain']:+.3f}); "
              f"inch plain={row['inchausti_plain']:.3f} D4={row['inchausti_d4']:.3f}")

    (C.DATA / "equivariance.json").write_text(json.dumps(res, indent=2))
    avg_delta = np.mean([res[n]["storfer_d4"] - res[n]["storfer_plain"] for n in res])
    print(f"\n[80] mean storfer@1% gain from test-time D4 pooling: {avg_delta:+.3f}")
    print("[80] wrote equivariance.json")


if __name__ == "__main__":
    raise SystemExit(main())
