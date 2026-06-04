"""
20 -- Probe PROVABGS galaxy properties from frozen AION embeddings (task 1).

Trains the paper's attentive-pooling head (CrossAttnHead = learned queries ->
NormCrossAttention -> per-target linear) on the frozen token embeddings, with an
80/20 split, and reports R^2 for the 5 targets (z, logmass, age, logZ, sSFR) per
config per variant. Also records a mean-pool + MLP number for reference. Writes
data/results/task1_provabgs.json.

Paper AION-B target (Photometry+Image+Spectrum): z=1.00 M*=0.96 age=0.53
logZ=0.61 sSFR=0.72.

Run: python 20_probe_provabgs.py [--config phot] [--variant base]
"""

import argparse
import json

import numpy as np
from sklearn.model_selection import train_test_split

import _config as C
import _probe as P
from _metrics import r2_per_target

TARGETS = ["z", "logmass", "age", "logZ", "sSFR"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="phot",
                    choices=["phot", "phot_spec", "phot_image", "phot_image_spec"])
    ap.add_argument("--variant", default=None)
    ap.add_argument("--force", action="store_true", help="re-probe even if variant already done")
    ap.add_argument("--heads", type=int, default=None, help="cross-attn heads")
    args = ap.parse_args()
    C.seed_everything()

    targets = np.load(C.RAW / "provabgs" / "targets.npy")  # (N,5)
    variants = [args.variant] if args.variant else C.VARIANTS

    results = {}
    res_path = C.RESULTS / "task1_provabgs.json"
    if res_path.exists():
        results = json.loads(res_path.read_text())

    for v in variants:
        if not args.force and v in results.get(args.config, {}):
            print(f"  [skip] {args.config}/{v} already in results")
            continue
        emb_path = C.EMB / f"provabgs_{args.config}_{v}.npy"
        if not emb_path.exists():
            print(f"  [missing] {emb_path.name}; run 10_embed_provabgs.py --config {args.config} --variant {v}")
            continue
        # memmap + index split: image/spec configs are up to ~70 GB, so avoid
        # the full-array copy that train_test_split(X, ...) would make.
        X = np.load(emb_path, mmap_mode="r")  # (M,T,D) full tokens
        idx_path = C.EMB / f"provabgs_{args.config}_index.npy"
        idx = np.load(idx_path) if idx_path.exists() else np.arange(len(X))
        y = targets[idx]
        assert len(y) == len(X), (len(y), len(X))
        tr_i, te_i = train_test_split(np.arange(len(X)), test_size=0.2, random_state=C.SEED)
        Xtr, Xte = np.ascontiguousarray(X[tr_i]), np.ascontiguousarray(X[te_i])
        ytr, yte = y[tr_i], y[te_i]
        dim = X.shape[-1]
        tok = X.shape[1]
        heads = args.heads or {"base": 12, "large": 16, "xlarge": 32}[v]
        # token-aware batch so big token x dim configs (e.g. xlarge phot_image,
        # 580x2048) don't OOM the cross-attention head on a 24 GB card.
        bs = int(np.clip(5e7 / (tok * dim), 12, 256))
        import gc
        import torch
        gc.collect()
        torch.cuda.empty_cache()

        # attentive pooling (paper head)
        preds, r2s, _ = P.train_regression(
            Xtr, ytr, Xte, yte,
            lambda d, o: P.CrossAttnHead(d, o, num_heads=heads),
            epochs=120, lr=1e-3, batch_size=bs, standardize_x=False, verbose=False)
        gc.collect(); torch.cuda.empty_cache()
        # mean-pool + MLP (reference)
        Xtr_m, Xte_m = Xtr.mean(1), Xte.mean(1)
        preds_m, r2m, _ = P.train_regression(
            Xtr_m, ytr, Xte_m, yte, lambda d, o: P.MLPHead(d, o, hidden=256),
            epochs=200, lr=1e-3, batch_size=256, verbose=False)

        rec = {
            "attn_R2": {t: round(r, 4) for t, r in zip(TARGETS, r2s)},
            "mlp_R2": {t: round(r, 4) for t, r in zip(TARGETS, r2m)},
            "n_train": len(Xtr), "n_test": len(Xte), "heads": heads, "dim": dim,
        }
        results.setdefault(args.config, {})[v] = rec
        print(f"[{v}/{args.config}] attn R2: " +
              " ".join(f"{t}={r:.3f}" for t, r in zip(TARGETS, r2s)))
        print(f"            mlp  R2: " +
              " ".join(f"{t}={r:.3f}" for t, r in zip(TARGETS, r2m)))
        res_path.write_text(json.dumps(results, indent=2))

    print("PROBE_PROVABGS_OK ->", res_path)


if __name__ == "__main__":
    main()
