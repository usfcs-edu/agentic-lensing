"""
42 -- Strong-lens retrieval (task 9).

Builds a retrieval corpus = SuGOHI grade-A/B lenses (positives, from
41_fetch_sugohi.py) + a large non-lens distractor pool (the already-cached
PROVABGS galaxy cutouts), embeds the g,r,i,z images with the frozen encoder
(mean-pooled), and ranks by cosine similarity. Metric: nDCG@10 with lenses as
the rare positive class (paper AION-B: 0.968). Writes data/results/task9_lenses.json.

Run: HF_HOME=... python 42_retrieve_lenses.py [--variant base] [--n_distract 12000]
"""

import argparse
import json

import numpy as np

import _aion_embed as E
import _config as C
from _retrieval import retrieval_ndcg

SUG = C.RAW / "sugohi"
PROV = C.RAW / "provabgs"


def build_corpus(n_distract):
    lens = np.load(SUG / "lens_image.npy")  # (L,4,160,160)
    dist = np.load(PROV / "image_flux.npy", mmap_mode="r")  # (D,4,160,160)
    rng = np.random.default_rng(C.SEED)
    di = np.sort(rng.choice(len(dist), min(n_distract, len(dist)), replace=False))
    dist_sel = np.asarray(dist[di])
    corpus = np.concatenate([lens, dist_sel], axis=0).astype(np.float32)
    is_lens = np.zeros(len(corpus), bool)
    is_lens[: len(lens)] = True
    np.save(SUG / "corpus_image.npy", corpus)
    np.save(SUG / "corpus_islens.npy", is_lens)
    print(f"corpus: {len(lens)} lenses + {len(dist_sel)} distractors = {len(corpus)} "
          f"(lens fraction {100*is_lens.mean():.2f}%)")
    return is_lens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default=None)
    ap.add_argument("--n_distract", type=int, default=12000)
    ap.add_argument("--gpus", default="0,2,3,4,5,6")
    args = ap.parse_args()
    C.seed_everything()
    gpus = [int(g) for g in args.gpus.split(",")]

    is_lens = build_corpus(args.n_distract)
    specs = [E.image_spec("LegacySurveyImage", str(SUG / "corpus_image.npy"),
                          ["DES-G", "DES-R", "DES-I", "DES-Z"])]
    variants = [args.variant] if args.variant else C.VARIANTS
    res_path = C.RESULTS / "task9_lenses.json"
    results = json.loads(res_path.read_text()) if res_path.exists() else {}
    for v in variants:
        emb_out = C.EMB / f"lenses_corpus_{v}.npy"
        E.multi_gpu_extract(specs, v, emb_out, pool="mean", gpus=gpus)
        emb = np.load(emb_out)
        r = retrieval_ndcg(emb, is_lens, k=10)
        results[v] = r
        print(f"[{v}] lens nDCG@10={r['ndcg@10']:.4f} "
              f"(corpus {r['corpus']}, {r['n_positive']} lenses)")
        res_path.write_text(json.dumps(results, indent=2))
    print("RETRIEVE_LENSES_OK ->", res_path)


if __name__ == "__main__":
    main()
