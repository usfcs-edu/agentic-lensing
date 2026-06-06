"""
10 -- Extract frozen AION embeddings for PROVABGS galaxies (task 1).

Builds the per-config input modalities and runs the multi-GPU embedding harness
(``_aion_embed.multi_gpu_extract``) across the 7 TITAN RTX cards, for each
model variant. Saves full token embeddings (pool='none', shape (N,T,D)) so the
attentive-pooling probe (CrossAttnHead) in 20_probe_provabgs.py can run.

Configs (paper's three input settings for galaxy property estimation):
  phot              -- Legacy Survey g,r,z,W1 photometry (4 scalar tokens)
  phot_image        -- + Legacy Survey g,r,i,z image     (+576 tokens)
  phot_image_spec   -- + DESI spectrum                   (+273 tokens)

Run: HF_HOME=... python 10_embed_provabgs.py --config phot [--variant base]
"""

import argparse

import numpy as np

import _aion_embed as E
import _config as C

RAW = C.RAW / "provabgs"
PHOT_BANDS = ["G", "R", "Z", "W1"]
FLUX_CLS = {"G": "LegacySurveyFluxG", "R": "LegacySurveyFluxR",
            "Z": "LegacySurveyFluxZ", "W1": "LegacySurveyFluxW1"}


def config_index(config):
    """Which PROVABGS rows a config covers (some modalities exist only on a
    subset). Returns an int index array into the full PROVABGS arrays."""
    n_full = len(np.load(RAW / "flux.npy", mmap_mode="r"))
    needs_spec = "spec" in config
    needs_image = "image" in config
    idx = np.arange(n_full)
    if needs_spec:
        idx = np.intersect1d(idx, np.load(RAW / "spec_index.npy"))
    if needs_image:
        idx = np.intersect1d(idx, np.load(RAW / "image_index.npy"))
    return idx


def _field_dir(config):
    d = RAW / "_fields" / config
    d.mkdir(parents=True, exist_ok=True)
    return d


def build_specs(config, idx):
    """Write subset-aligned field files for `idx` and return modality specs."""
    fd = _field_dir(config)
    flux = np.load(RAW / "flux.npy")[idx]  # (M,4) G,R,Z,W1
    specs = []
    for j, b in enumerate(PHOT_BANDS):
        p = fd / f"flux_{b}.npy"
        np.save(p, flux[:, j : j + 1].astype(np.float32))
        specs.append(E.scalar_spec(FLUX_CLS[b], str(p)))
    if "image" in config:
        img = np.load(RAW / "image_flux.npy")  # aligned to image_index
        img_idx = np.load(RAW / "image_index.npy")
        pos = {r: k for k, r in enumerate(img_idx)}
        sel = np.array([pos[r] for r in idx])
        p = fd / "image_flux.npy"
        np.save(p, img[sel].astype(np.float32))
        specs.append(E.image_spec("LegacySurveyImage", str(p),
                                  ["DES-G", "DES-R", "DES-I", "DES-Z"]))
    if "spec" in config:
        spec_idx = np.load(RAW / "spec_index.npy")
        pos = {r: k for k, r in enumerate(spec_idx)}
        sel = np.array([pos[r] for r in idx])
        for field, src in [("flux", "spec_flux"), ("ivar", "spec_ivar"),
                           ("mask", "spec_mask"), ("wavelength", "spec_wave")]:
            arr = np.load(RAW / f"{src}.npy")[sel]
            p = fd / f"spec_{field}.npy"
            np.save(p, arr)
        specs.append(E.spectrum_spec(
            "DESISpectrum", str(fd / "spec_flux.npy"), str(fd / "spec_ivar.npy"),
            str(fd / "spec_mask.npy"), str(fd / "spec_wavelength.npy")))
    return specs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="phot",
                    choices=["phot", "phot_spec", "phot_image", "phot_image_spec"])
    ap.add_argument("--variant", default=None, help="base/large/xlarge; default all")
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6")
    ap.add_argument("--pool", default="none", choices=["none", "mean"])
    ap.add_argument("--batch", type=int, default=0, help="override per-variant embed batch")
    args = ap.parse_args()

    gpus = [int(g) for g in args.gpus.split(",")]
    variants = [args.variant] if args.variant else C.VARIANTS
    idx = config_index(args.config)
    np.save(C.EMB / f"provabgs_{args.config}_index.npy", idx)
    specs = build_specs(args.config, idx)
    n = E._n_rows(specs)
    print(f"config={args.config} n={n} variants={variants} pool={args.pool}")
    for v in variants:
        out = C.EMB / f"provabgs_{args.config}_{v}.npy"
        if out.exists():
            print(f"  [skip] {out.name} exists {np.load(out, mmap_mode='r').shape}")
            continue
        import time
        t = time.time()
        bs = args.batch or None
        arr = E.multi_gpu_extract(specs, v, out, pool=args.pool, gpus=gpus, batch_size=bs)
        print(f"  [{v}] {out.name} {arr.shape} in {time.time()-t:.1f}s")
    print("EMBED_PROVABGS_OK")


if __name__ == "__main__":
    main()
