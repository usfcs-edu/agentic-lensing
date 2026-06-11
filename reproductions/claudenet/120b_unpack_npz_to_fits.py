#!/usr/bin/env python3
"""120b_unpack_npz_to_fits.py — Phase 120 helper: unpack a 111b_dump_rows.py npz
(cutouts (n,3,101,101) float32, row_ids, ok) into per-row FITS files that the v1
loaders (_trainlib.LensDataset / _scorelib.score_paths, per-row fits_dir) consume
unchanged (runs LOCALLY, CPU).

Each ok row becomes <out-dir>/<row_id>.fits written with the EXACT byte format
of inchausti-2025/20_build_negatives_brick_dr9.to_bytes — fits.PrimaryHDU(
float32 cube).writeto — so mined negatives are file-format-identical to the v1
training negatives. ok=False rows are skipped; the written count is asserted.
Also writes a training manifest parquet [row_id (str), label=0, fits_dir
(absolute <out-dir>)] for 121_retrain_mined_members.py.

    /home2/benson/.venvs/claudenet/bin/python 120b_unpack_npz_to_fits.py \
        --npz data/v2/mined_hard.npz --out-dir data/v2/mined_hard_fits
    # -> data/v2/mined_hard_fits/<row_id>.fits
    #    data/v2/mined_hard_fits_manifest.parquet   (default --manifest-out)
"""
from __future__ import annotations

import argparse
import io
import time
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits


def to_bytes(cube: np.ndarray) -> bytes:
    """Verbatim inchausti-2025/20_build_negatives_brick_dr9.to_bytes."""
    bio = io.BytesIO()
    fits.PrimaryHDU(data=cube.astype(np.float32)).writeto(bio)
    return bio.getvalue()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--npz", required=True, help="111b output (cutouts/row_ids/ok)")
    ap.add_argument("--out-dir", required=True, help="per-row FITS output dir")
    ap.add_argument("--manifest-out", default=None,
                    help="manifest parquet path (default <out-dir>_manifest.parquet)")
    ap.add_argument("--expect", type=int, default=None,
                    help="assert exactly this many FITS get written")
    args = ap.parse_args()
    t0 = time.time()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_out = (Path(args.manifest_out) if args.manifest_out
                    else out_dir.parent / f"{out_dir.name}_manifest.parquet")

    z = np.load(args.npz, allow_pickle=False)
    cut, row_ids, ok = z["cutouts"], z["row_ids"].astype(str), z["ok"].astype(bool)
    assert cut.ndim == 4 and len(cut) == len(row_ids) == len(ok), \
        f"npz arrays misaligned: cutouts {cut.shape}, row_ids {row_ids.shape}, ok {ok.shape}"
    assert cut.shape[1] == 3, \
        f"expected 3-band grz cubes for the v1 loaders, got {cut.shape[1]} bands"
    assert pd.Index(row_ids).is_unique, "duplicate row_ids in npz"
    print(f"[unpack] {args.npz}: {len(row_ids):,} rows, cutouts {cut.shape}, "
          f"{int(ok.sum()):,} ok")

    n_nan = 0
    written = []
    for i in np.where(ok)[0]:
        cube = np.asarray(cut[i], dtype=np.float32)
        if not np.isfinite(cube).all():
            n_nan += 1
        (out_dir / f"{row_ids[i]}.fits").write_bytes(to_bytes(cube))
        written.append(row_ids[i])
    n_skip = int((~ok).sum())
    assert len(written) == int(ok.sum()), \
        f"wrote {len(written)} != ok count {int(ok.sum())}"
    if args.expect is not None:
        assert len(written) == args.expect, \
            f"wrote {len(written)} FITS, --expect {args.expect}"
    if n_nan:
        print(f"[unpack] WARNING: {n_nan:,} ok rows contain non-finite pixels "
              f"(NaNs propagate to NaN scores; 120 only selects finite-score rows, "
              f"so this should be 0 for mined sets)")

    pd.DataFrame({"row_id": pd.array(written, dtype="string"),
                  "label": 0,
                  "fits_dir": str(out_dir)}).to_parquet(manifest_out, index=False)
    print(f"[unpack] wrote {len(written):,} FITS -> {out_dir} "
          f"(skipped {n_skip} ok=False) + manifest {manifest_out} "
          f"({time.time() - t0:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
