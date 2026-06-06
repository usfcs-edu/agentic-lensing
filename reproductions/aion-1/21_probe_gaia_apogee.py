"""
21 -- Probe APOGEE stellar params (Teff, logg, [Fe/H]) from Gaia-XP AION
embeddings (task 3). Metric: residual std (paper AION-B: 94.6 K / 0.206 / 0.115 dex).

Attentive-pooling head (paper uses cross-attn/linear for stellar params) on the
frozen 110-token Gaia-XP embeddings, 80/20 split. Also a mean-pool+MLP reference.
Writes data/results/task3_gaia_apogee.json.

Run: python 21_probe_gaia_apogee.py [--variant base]
"""

import argparse
import json

import numpy as np
from sklearn.model_selection import train_test_split

import _config as C
import _probe as P
from _metrics import residual_std

TARGETS = ["Teff", "logg", "FeH"]
APO = C.RAW / "apogee"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default=None)
    ap.add_argument("--heads", type=int, default=None)
    args = ap.parse_args()
    C.seed_everything()

    targets = np.load(APO / "targets.npy")  # (N,3)
    variants = [args.variant] if args.variant else C.VARIANTS
    results = {}
    res_path = C.RESULTS / "task3_gaia_apogee.json"
    if res_path.exists():
        results = json.loads(res_path.read_text())

    for v in variants:
        emb_path = C.EMB / f"gaia_apogee_{v}.npy"
        if not emb_path.exists():
            print(f"  [missing] {emb_path.name}; run 11_embed_gaia_xp.py --variant {v}")
            continue
        X = np.load(emb_path)
        assert len(X) == len(targets), (len(X), len(targets))
        Xtr, Xte, ytr, yte = train_test_split(X, targets, test_size=0.2, random_state=C.SEED)
        heads = args.heads or {"base": 12, "large": 16, "xlarge": 32}[v]
        preds, _, _ = P.train_regression(
            Xtr, ytr, Xte, yte,
            lambda d, o: P.CrossAttnHead(d, o, num_heads=heads),
            epochs=200, lr=1e-3, batch_size=256, standardize_x=False, verbose=False)
        rstd = [residual_std(yte[:, k], preds[:, k]) for k in range(3)]
        # mean-pool + MLP reference
        preds_m, _, _ = P.train_regression(
            Xtr.mean(1), ytr, Xte.mean(1), yte,
            lambda d, o: P.MLPHead(d, o, hidden=256),
            epochs=300, lr=1e-3, batch_size=256, verbose=False)
        rstd_m = [residual_std(yte[:, k], preds_m[:, k]) for k in range(3)]
        rec = {
            "attn_residual_std": {t: round(float(r), 4) for t, r in zip(TARGETS, rstd)},
            "mlp_residual_std": {t: round(float(r), 4) for t, r in zip(TARGETS, rstd_m)},
            "n_train": len(Xtr), "n_test": len(Xte), "heads": heads, "dim": X.shape[-1],
        }
        results[v] = rec
        print(f"[{v}] attn residual std: " +
              " ".join(f"{t}={r:.3f}" for t, r in zip(TARGETS, rstd)))
        res_path.write_text(json.dumps(results, indent=2))
    print("PROBE_GAIA_APOGEE_OK ->", res_path)


if __name__ == "__main__":
    main()
