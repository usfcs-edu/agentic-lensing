"""
44 -- Faithful GZ-DECaLS morphology retrieval (tasks 7/8).

Builds the retrieval corpus from whatever GZ-DECaLS cutouts the campaign
(43_fetch_gzdecals_campaign.py) has cached, embeds the g,r,i,z images with the
frozen encoder (mean-pooled), and reports nDCG@10 for spiral and merger
queries against the full corpus. Unlike the Galaxy10 best-effort version
(40_retrieve_gz10.py), positives come from the published GZ-DECaLS vote
fractions over a large, rare-positive corpus (paper: spirals 0.938, mergers
0.892). Runs on the partial corpus mid-campaign and improves as it grows.

Writes data/results/task78_gzdecals_retrieval.json.

Run: HF_HOME=... python 44_retrieve_gzdecals.py [--variant base]
"""

import argparse
import json

import numpy as np

import _aion_embed as E
import _config as C
import _ls_cutout as LS
import pandas as pd
from _retrieval import retrieval_ndcg

OUT = C.RAW / "gzdecals"


def build_corpus():
    t = pd.read_parquet(OUT / "targets.parquet")
    imgs, sp, mg = [], [], []
    n_try = 0
    for ra, dec, label in zip(t["ra"], t["dec"], t["label"]):
        n_try += 1
        a = LS.fetch_one(float(ra), float(dec), layer="ls-dr10", size=160)  # cache hit only
        if a is None:
            continue
        imgs.append(a.astype(np.float32))
        sp.append(label == "spiral")
        mg.append(label == "merger")
    corpus = np.stack(imgs)
    np.save(OUT / "corpus_image.npy", corpus)
    np.save(OUT / "corpus_spiral.npy", np.array(sp))
    np.save(OUT / "corpus_merger.npy", np.array(mg))
    print(f"corpus: {len(corpus)} cached / {n_try} targets; "
          f"{int(np.sum(sp))} spirals, {int(np.sum(mg))} mergers")
    return np.array(sp), np.array(mg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default=None)
    ap.add_argument("--gpus", default="0,2,3,4,5,6")
    args = ap.parse_args()
    C.seed_everything()
    gpus = [int(g) for g in args.gpus.split(",")]

    is_sp, is_mg = build_corpus()
    specs = [E.image_spec("LegacySurveyImage", str(OUT / "corpus_image.npy"),
                          ["DES-G", "DES-R", "DES-I", "DES-Z"])]
    variants = [args.variant] if args.variant else C.VARIANTS
    res_path = C.RESULTS / "task78_gzdecals_retrieval.json"
    results = json.loads(res_path.read_text()) if res_path.exists() else {}
    for v in variants:
        emb_out = C.EMB / f"gzdecals_corpus_{v}.npy"
        E.multi_gpu_extract(specs, v, emb_out, pool="mean", gpus=gpus)
        emb = np.load(emb_out)
        rs = retrieval_ndcg(emb, is_sp, k=10)
        rm = retrieval_ndcg(emb, is_mg, k=10)
        results[v] = {"spirals": rs, "mergers": rm}
        print(f"[{v}] spirals nDCG@10={rs['ndcg@10']:.4f} (n={rs['n_positive']}/{rs['corpus']}); "
              f"mergers nDCG@10={rm['ndcg@10']:.4f} (n={rm['n_positive']}/{rm['corpus']})")
        res_path.write_text(json.dumps(results, indent=2))
    print("RETRIEVE_GZDECALS_OK ->", res_path)


if __name__ == "__main__":
    main()
