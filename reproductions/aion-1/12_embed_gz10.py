"""
12 -- Frozen AION embeddings for Galaxy10 DECaLS images (tasks 4, 7, 8).

Input modality is a single LegacySurveyImage (g,r,i,z), the cutouts fetched in
06_fetch_gz10_images.py. pool='none' keeps all 576 image tokens so the morphology
probe can mean-pool (+MLP, paper's setup) and the retrieval tasks can mean-pool
-> cosine.

Saves data/emb/gz10_<variant>.npy (M,576,D), aligned to image_index.npy.

Run: HF_HOME=... python 12_embed_gz10.py [--variant base]
"""

import argparse
import time

import numpy as np

import _aion_embed as E
import _config as C

RAW = C.RAW / "gz10"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default=None)
    ap.add_argument("--gpus", default="0,2,3,4,5,6")  # exclude GPU1 (thermal)
    ap.add_argument("--pool", default="none", choices=["none", "mean"])
    args = ap.parse_args()

    gpus = [int(g) for g in args.gpus.split(",")]
    variants = [args.variant] if args.variant else C.VARIANTS
    specs = [E.image_spec("LegacySurveyImage", str(RAW / "image_flux.npy"),
                          ["DES-G", "DES-R", "DES-I", "DES-Z"])]
    n = E._n_rows(specs)
    print(f"task4 gz10 n={n} variants={variants}")
    for v in variants:
        out = C.EMB / f"gz10_{v}.npy"
        if out.exists():
            print(f"  [skip] {out.name} {np.load(out, mmap_mode='r').shape}")
            continue
        t = time.time()
        arr = E.multi_gpu_extract(specs, v, out, pool=args.pool, gpus=gpus)
        print(f"  [{v}] {out.name} {arr.shape} in {time.time()-t:.1f}s")
    print("EMBED_GZ10_OK")


if __name__ == "__main__":
    main()
