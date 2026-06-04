"""
13 -- Frozen AION embeddings from DESI spectra for the DD-Payne stellar set (task 2).

Single DESISpectrum modality (flux, ivar, mask, wavelength) from
08_fetch_ddpayne_desi.py. pool='none' keeps all spectrum tokens for the
attentive-pooling probe.

Saves data/emb/ddpayne_desi_<variant>.npy (N, T, D).

Run: HF_HOME=... python 13_embed_ddpayne.py [--variant base] [--gpus 0,2,3,4,5,6]
"""

import argparse
import time

import numpy as np

import _aion_embed as E
import _config as C

RAW = C.RAW / "ddpayne"


def build_specs(config):
    specs = [E.spectrum_spec(
        "DESISpectrum", str(RAW / "spec_flux.npy"), str(RAW / "spec_ivar.npy"),
        str(RAW / "spec_mask.npy"), str(RAW / "spec_wave.npy"))]
    if config == "desi_plx":
        # paper's "DESI+Parallax" config: add Gaia parallax + sky coords.
        plx = np.clip(np.load(RAW / "parallax.npy").reshape(-1, 1), 1e-3, None).astype(np.float32)
        np.save(RAW / "_plx_col.npy", plx)  # LogScalarCodec needs positive parallax
        np.save(RAW / "_ra_col.npy", np.load(RAW / "ra.npy").reshape(-1, 1).astype(np.float32))
        np.save(RAW / "_dec_col.npy", np.load(RAW / "dec.npy").reshape(-1, 1).astype(np.float32))
        specs += [E.scalar_spec("GaiaParallax", str(RAW / "_plx_col.npy")),
                  E.scalar_spec("Ra", str(RAW / "_ra_col.npy")),
                  E.scalar_spec("Dec", str(RAW / "_dec_col.npy"))]
    return specs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default=None)
    ap.add_argument("--config", default="desi", choices=["desi", "desi_plx"])
    ap.add_argument("--gpus", default="0,2,3,4,5,6")  # exclude GPU1 (thermal)
    ap.add_argument("--pool", default="none", choices=["none", "mean"])
    args = ap.parse_args()

    gpus = [int(g) for g in args.gpus.split(",")]
    variants = [args.variant] if args.variant else C.VARIANTS
    specs = build_specs(args.config)
    n = E._n_rows(specs)
    print(f"task2 ddpayne_{args.config} n={n} variants={variants}")
    for v in variants:
        out = C.EMB / f"ddpayne_{args.config}_{v}.npy"
        if out.exists():
            print(f"  [skip] {out.name} {np.load(out, mmap_mode='r').shape}")
            continue
        t = time.time()
        arr = E.multi_gpu_extract(specs, v, out, pool=args.pool, gpus=gpus)
        print(f"  [{v}] {out.name} {arr.shape} in {time.time()-t:.1f}s")
    print("EMBED_DDPAYNE_OK")


if __name__ == "__main__":
    main()
