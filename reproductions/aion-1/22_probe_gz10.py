"""
22 -- Galaxy morphology classification from frozen AION image embeddings (task 4).

Mean-pool the 576 image tokens -> 2-layer MLP(256, dropout 0.1) classifier over
the 10 Galaxy10 DECaLS classes, stratified 80/20 split. Metric: accuracy
(paper AION-B: 84.0%). Writes data/results/task4_gz10.json.

Run: python 22_probe_gz10.py [--variant base]
"""

import argparse
import json

import numpy as np
from sklearn.model_selection import train_test_split

import _config as C
import _probe as P
from _metrics import accuracy

RAW = C.RAW / "gz10"
CLASS_NAMES = ["Disturbed", "Merging", "Round Smooth", "In-between Smooth",
               "Cigar Smooth", "Barred Spiral", "Unbarred Tight Spiral",
               "Unbarred Loose Spiral", "Edge-on no Bulge", "Edge-on Bulge"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default=None)
    args = ap.parse_args()
    C.seed_everything()

    labels_all = np.load(RAW / "labels.npy")
    idx = np.load(RAW / "image_index.npy")
    y = labels_all[idx].astype(np.int64)
    variants = [args.variant] if args.variant else C.VARIANTS
    results = {}
    res_path = C.RESULTS / "task4_gz10.json"
    if res_path.exists():
        results = json.loads(res_path.read_text())

    for v in variants:
        emb_path = C.EMB / f"gz10_{v}.npy"
        if not emb_path.exists():
            print(f"  [missing] {emb_path.name}; run 12_embed_gz10.py --variant {v}")
            continue
        X = np.load(emb_path).mean(1)  # mean-pool tokens
        assert len(X) == len(y), (len(X), len(y))
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.2, random_state=C.SEED, stratify=y)
        preds, acc, _ = P.train_classification(
            Xtr, ytr, Xte, yte,
            lambda d, o: P.MLPHead(d, o, hidden=256, dropout=0.1),
            n_classes=10, epochs=200, lr=1e-3, batch_size=256, verbose=False)
        rec = {"accuracy": round(float(acc), 4), "n_train": len(Xtr),
               "n_test": len(Xte), "dim": X.shape[-1]}
        results[v] = rec
        print(f"[{v}] morphology accuracy = {acc:.4f}")
        res_path.write_text(json.dumps(results, indent=2))
    print("PROBE_GZ10_OK ->", res_path)


if __name__ == "__main__":
    main()
