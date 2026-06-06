"""
11 -- Frozen AION embeddings from Gaia XP BP/RP coefficients (task 3).

Reuses the multi-GPU harness. GaiaXpBp and GaiaXpRp are 55-token scalar
modalities (field 'value'), so we feed the (N,55) BP and RP coefficient arrays
from 07_xmatch_gaia_apogee.py and pool='none' to keep all 110 tokens for the
attentive-pooling probe.

Saves data/emb/gaia_apogee_<variant>.npy  (N, T, D).

Run: HF_HOME=... python 11_embed_gaia_xp.py [--variant base]
"""

import argparse
import time

import numpy as np

import _aion_embed as E
import _config as C

APO = C.RAW / "apogee"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default=None)
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6")
    ap.add_argument("--pool", default="none", choices=["none", "mean"])
    args = ap.parse_args()

    gpus = [int(g) for g in args.gpus.split(",")]
    variants = [args.variant] if args.variant else C.VARIANTS
    specs = [
        E.scalar_spec("GaiaXpBp", str(APO / "xp_bp.npy")),
        E.scalar_spec("GaiaXpRp", str(APO / "xp_rp.npy")),
    ]
    n = E._n_rows(specs)
    print(f"task3 gaia_apogee n={n} variants={variants}")
    for v in variants:
        out = C.EMB / f"gaia_apogee_{v}.npy"
        if out.exists():
            print(f"  [skip] {out.name} {np.load(out, mmap_mode='r').shape}")
            continue
        t = time.time()
        arr = E.multi_gpu_extract(specs, v, out, pool=args.pool, gpus=gpus)
        print(f"  [{v}] {out.name} {arr.shape} in {time.time()-t:.1f}s")
    print("EMBED_GAIA_APOGEE_OK")


if __name__ == "__main__":
    main()
