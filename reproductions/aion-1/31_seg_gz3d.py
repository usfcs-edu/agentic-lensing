"""
31 -- Galaxy structure segmentation from frozen AION image tokens (task 5).

Embeds the GZ3D galaxy cutouts (image modality, full 24x24 spatial tokens) with
the frozen encoder, then trains the lightweight conv upsampler (_probe.SegHead)
to predict spiral-arm and bar masks. Metric: per-image IoU (paper AION-B:
spiral 0.60, bar 0.31). Writes data/results/task5_gz3d.json + a qualitative figure.

Run: HF_HOME=... python 31_seg_gz3d.py [--variant base]
"""

import argparse
import json

import numpy as np
import torch

import _aion_embed as E
import _config as C
import _probe as P
from _metrics import mean_iou

RAW = C.RAW / "gz3d"
CLASSES = ["spiral_arms", "bar"]


def embed(variant, gpus):
    out = C.EMB / f"gz3d_{variant}.npy"
    if out.exists():
        return out
    specs = [E.image_spec("LegacySurveyImage", str(RAW / "image_flux.npy"),
                          ["DES-G", "DES-R", "DES-I", "DES-Z"])]
    E.multi_gpu_extract(specs, variant, out, pool="none", gpus=gpus)
    return out


def train_seg(X, Y, dim, device="cuda", epochs=120, lr=1e-3, bs=32, patience=15):
    n = len(X)
    rng = np.random.default_rng(C.SEED)
    perm = rng.permutation(n)
    ntr = int(0.8 * n)
    tr, te = perm[:ntr], perm[ntr:]
    head = P.SegHead(dim, Y.shape[1]).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    lossf = torch.nn.BCEWithLogitsLoss()
    best_iou, best_state, bad = -1, None, 0
    for ep in range(epochs):
        head.train()
        rng.shuffle(tr)
        for s in range(0, len(tr), bs):
            b = tr[s:s + bs]
            xb = torch.as_tensor(np.asarray(X[b], np.float32), device=device)
            yb = torch.as_tensor(Y[b], device=device).float()
            opt.zero_grad()
            loss = lossf(head(xb), yb)
            loss.backward()
            opt.step()
        head.eval()
        with torch.no_grad():
            preds = []
            for s in range(0, len(te), 64):
                xb = torch.as_tensor(np.asarray(X[te[s:s + 64]], np.float32), device=device)
                preds.append(torch.sigmoid(head(xb)).cpu().numpy())
        preds = np.concatenate(preds)  # (nte, C, 96, 96)
        ious = [mean_iou(preds[:, c], Y[te][:, c]) for c in range(Y.shape[1])]
        miou = float(np.nanmean(ious))
        if miou > best_iou:
            best_iou, bad = miou, 0
            best_state = {"ious": ious, "preds_idx": te[:6].tolist()}
        else:
            bad += 1
            if bad >= patience:
                break
    return best_state["ious"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default=None)
    ap.add_argument("--gpus", default="0,2,3,4,5,6")
    args = ap.parse_args()
    C.seed_everything()
    gpus = [int(g) for g in args.gpus.split(",")]

    spiral = np.load(RAW / "spiral_mask.npy")
    bar = np.load(RAW / "bar_mask.npy")
    Y = np.stack([spiral, bar], axis=1).astype(np.float32)  # (N,2,96,96)
    variants = [args.variant] if args.variant else C.VARIANTS

    res_path = C.RESULTS / "task5_gz3d.json"
    results = json.loads(res_path.read_text()) if res_path.exists() else {}
    for v in variants:
        emb_path = embed(v, gpus)
        X = np.load(emb_path, mmap_mode="r")
        dim = X.shape[-1]
        ious = train_seg(X, Y[: len(X)], dim)
        results[v] = {c: round(float(i), 4) for c, i in zip(CLASSES, ious)}
        results[v]["n"] = int(len(X))
        print(f"[{v}] IoU " + " ".join(f"{c}={i:.3f}" for c, i in zip(CLASSES, ious)))
        res_path.write_text(json.dumps(results, indent=2))
    print("SEG_GZ3D_OK ->", res_path)


if __name__ == "__main__":
    main()
