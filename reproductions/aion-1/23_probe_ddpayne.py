"""
23 -- Probe stellar parameters (Teff, logg, [Fe/H], vmicro) from DESI-spectrum
AION embeddings (task 2). Metric: R² (paper AION-B, DESI+Parallax:
0.99/0.98/0.94/0.89; DESI-spectrum alone is close).

Attentive-pooling head on the frozen DESI-spectrum tokens, 80/20 split. Also a
mean-pool+MLP reference. Note: this is the DESI-spectrum-only config; we do not
add the Gaia parallax channel, so small shortfalls vs the paper's DESI+Parallax
column are expected. Writes data/results/task2_ddpayne.json.

Run: python 23_probe_ddpayne.py [--variant base]
"""

import argparse
import json

import numpy as np
from sklearn.model_selection import train_test_split

import _config as C
import _probe as P
from _metrics import r2_per_target

TARGETS = ["Teff", "logg", "FeH", "vmic"]
RAW = C.RAW / "ddpayne"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default=None)
    ap.add_argument("--config", default="desi", choices=["desi", "desi_plx"])
    ap.add_argument("--heads", type=int, default=None)
    args = ap.parse_args()
    C.seed_everything()

    targets = np.load(RAW / "targets.npy")  # (N,4)
    variants = [args.variant] if args.variant else C.VARIANTS
    results = {}
    res_path = C.RESULTS / "task2_ddpayne.json"
    if res_path.exists():
        results = json.loads(res_path.read_text())
    # migrate old flat {base,large,xlarge} -> {"desi": {...}}
    if results and "desi" not in results and "base" in results:
        results = {"desi": results}

    for v in variants:
        emb_path = C.EMB / f"ddpayne_{args.config}_{v}.npy"
        if not emb_path.exists():
            print(f"  [missing] {emb_path.name}; run 13_embed_ddpayne.py --config {args.config} --variant {v}")
            continue
        X = np.load(emb_path)
        assert len(X) == len(targets), (len(X), len(targets))
        Xtr, Xte, ytr, yte = train_test_split(X, targets, test_size=0.2, random_state=C.SEED)
        heads = args.heads or {"base": 12, "large": 16, "xlarge": 32}[v]
        preds, r2s, _ = P.train_regression(
            Xtr, ytr, Xte, yte,
            lambda d, o: P.CrossAttnHead(d, o, num_heads=heads),
            epochs=150, lr=1e-3, batch_size=256, standardize_x=False, verbose=False)
        preds_m, r2m, _ = P.train_regression(
            Xtr.mean(1), ytr, Xte.mean(1), yte,
            lambda d, o: P.MLPHead(d, o, hidden=256),
            epochs=250, lr=1e-3, batch_size=256, verbose=False)
        rec = {
            "attn_R2": {t: round(r, 4) for t, r in zip(TARGETS, r2s)},
            "mlp_R2": {t: round(r, 4) for t, r in zip(TARGETS, r2m)},
            "n_train": len(Xtr), "n_test": len(Xte), "heads": heads, "dim": X.shape[-1],
        }
        results.setdefault(args.config, {})[v] = rec
        print(f"[{args.config}/{v}] attn R2: " + " ".join(f"{t}={r:.3f}" for t, r in zip(TARGETS, r2s)))
        res_path.write_text(json.dumps(results, indent=2))
    print("PROBE_DDPAYNE_OK ->", res_path)


if __name__ == "__main__":
    main()
