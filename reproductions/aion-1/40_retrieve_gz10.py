"""
40 -- Morphology retrieval from frozen AION image embeddings (tasks 7, 8).

Mean-pool the GZ10 image tokens, L2-normalise, and for each positive query rank
the corpus by cosine similarity -> mean nDCG@10 (paper protocol). The paper uses
the full GZ-DECaLS catalogue (24,622 spirals / 726 mergers in a ~310k corpus);
we only have Galaxy10 DECaLS (17,736 galaxies, 10 classes), so the corpus and
positive fractions differ. We therefore report this as a best-effort retrieval
quality check, not a direct match:
  - spirals  = classes {5 Barred, 6 Unbarred-tight, 7 Unbarred-loose}
  - mergers  = class  {1 Merging}
Writes data/results/task78_gz10_retrieval.json.

Run: python 40_retrieve_gz10.py [--variant base]
"""

import argparse
import json

import numpy as np

import _config as C
from _retrieval import retrieval_ndcg

RAW = C.RAW / "gz10"
GROUPS = {"spirals": {5, 6, 7}, "mergers": {1}}


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
    res_path = C.RESULTS / "task78_gz10_retrieval.json"
    if res_path.exists():
        results = json.loads(res_path.read_text())

    for v in variants:
        emb_path = C.EMB / f"gz10_{v}.npy"
        if not emb_path.exists():
            print(f"  [missing] {emb_path.name}; run 12_embed_gz10.py --variant {v}")
            continue
        X = np.load(emb_path).mean(1)
        rec = {}
        for name, classes in GROUPS.items():
            pos = np.isin(y, list(classes))
            out = retrieval_ndcg(X, pos, k=10)
            rec[name] = {k: (round(val, 4) if isinstance(val, float) else val)
                         for k, val in out.items()}
            print(f"[{v}] {name}: nDCG@10={out['ndcg@10']:.4f} "
                  f"(n_pos={out['n_positive']}/{out['corpus']})")
        results[v] = rec
        res_path.write_text(json.dumps(results, indent=2))
    print("RETRIEVE_GZ10_OK ->", res_path)


if __name__ == "__main__":
    main()
