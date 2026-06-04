"""
06b -- Assemble the GZ10 image arrays from whatever cutouts are already cached.

The Legacy Survey cutout service rate-limits hard, so fetching all ~7.5k GZ10
cutouts is slow. This decouples assembly from fetching: it reads the on-disk
cutout cache (written by _ls_cutout) for the stratified GZ10 coordinate set and
builds image_flux.npy / image_index.npy from the successful ones, so the
morphology + retrieval tasks can run on the available subset while the fetch
continues. Re-run later to pick up more.

Outputs (data/raw/gz10/): image_flux.npy (M,4,160,160), image_index.npy, labels.npy.

Run: python 06b_finalize_gz10.py [--subsample 8000]
"""

import argparse

import h5py
import numpy as np

import _config as C
import _ls_cutout as LS

RAW = C.RAW / "gz10"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subsample", type=int, default=8000)
    ap.add_argument("--size", type=int, default=160)
    args = ap.parse_args()
    C.seed_everything()

    with h5py.File(RAW / "Galaxy10_DECals.h5", "r") as f:
        ra = f["ra"][:].astype(np.float64)
        dec = f["dec"][:].astype(np.float64)
        ans = f["ans"][:].astype(np.int64)
    np.save(RAW / "labels.npy", ans)

    idx = np.arange(len(ra))
    if args.subsample and args.subsample < len(idx):
        rng = np.random.default_rng(C.SEED)
        per = args.subsample // 10
        chosen = []
        for c in range(10):
            ci = idx[ans == c]
            chosen.append(rng.choice(ci, min(per, len(ci)), replace=False))
        idx = np.sort(np.concatenate(chosen))

    imgs, good = [], []
    for i in idx:
        key = LS._key(float(ra[i]), float(dec[i]), "ls-dr10", args.size, 0.262)
        p = LS.CACHE / (key + ".npy")
        if p.exists():
            try:
                a = np.load(p)
                if a.ndim == 3 and a.shape[0] == 4:
                    imgs.append(a.astype(np.float32)); good.append(i)
            except Exception:
                pass
    if not imgs:
        print("NO CACHED GZ10 CUTOUTS YET"); return
    imgs = np.stack(imgs)
    good = np.array(good)
    np.save(RAW / "image_flux.npy", imgs)
    np.save(RAW / "image_index.npy", good)
    print(f"GZ10_FINALIZE_OK {imgs.shape} for {len(good)}/{len(idx)} "
          f"({100*len(good)/len(idx):.1f}% of stratified set cached)")


if __name__ == "__main__":
    main()
