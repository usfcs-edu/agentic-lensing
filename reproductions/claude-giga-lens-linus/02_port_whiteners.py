"""02_port_whiteners.py — import + re-validate the frozen whitener bundles.

Imports the claude-giga-lens whitener bundles BY PATH (never copied — README
copy-vs-import rule), re-validates each against its own stored construction
metadata, re-hashes, and writes data/whitener_manifest.json:

  * e_op re-computation: rebuild S = FFT(embed(rho_kernel)) on the stored
    grid, apply the stored s_floor, H = spectrum of the stored taps h, and
    check e_op = max|S H^2 - 1| reproduces the stored certificate
    (tolerance 1e-9) AND meets the frozen admissibility gate e_op <= 0.02.
    whitener_v2d_relaxed was BUILT as a relaxed arm (its own e_target was
    0.05, ledgered P1b D3); it is recorded admissible_strict=False by
    design, not as a validation failure.
  * keep-mask consistency: stored keep_w == cgl2.whiten.erode_keep(product
    keep_mask, M) for the bundle's product.
  * sha256 hashes of the file and of each load-bearing array (h, keep_w,
    rho_kernel) — the freeze fingerprint every downstream run can assert.

CPU-only (FFT bookkeeping, one of the sanctioned CPU tasks).

Run:  /raid/benson/.venvs/cgl2/bin/python 02_port_whiteners.py
"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("GIGALENS_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CGL2_ALLOW_CPU", "1")

HERE = Path(__file__).resolve().parent

from cgl2 import paths  # noqa: E402
from cgl2.whiten import _embed_kernel, _spectrum_of_taps, erode_keep  # noqa: E402

E_OP_GATE = 0.02          # frozen admissibility certificate
REPRO_TOL = 1e-9          # stored-vs-recomputed e_op reproduction tolerance

BUNDLES = {
    "v2d": dict(file="whitener_v2d.npz", cutout="cutout_v2d.npz",
                strict=True),
    "v3b": dict(file="whitener_v3b.npz", cutout="cutout_v3b.npz",
                strict=True),
    "v3": dict(file="whitener_v3.npz", cutout="cutout_v3.npz", strict=True),
    "v2d_relaxed": dict(file="whitener_v2d_relaxed.npz",
                        cutout="cutout_v2d.npz", strict=False),
}


def sha256_arr(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def main():
    t0 = time.time()
    manifest = dict(
        generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        script="02_port_whiteners.py",
        source_dir=str(paths.CGL1_DATA),
        e_op_gate=E_OP_GATE, repro_tol=REPRO_TOL, bundles={})
    all_ok = True

    for tag, cfg in BUNDLES.items():
        f = paths.CGL1_DATA / cfg["file"]
        z = np.load(f, allow_pickle=True)
        meta = json.loads(str(z["meta"]))
        h = np.asarray(z["h"], dtype=np.float64)
        keep_w = np.asarray(z["keep_w"]).astype(bool)
        rho = np.asarray(z["rho_kernel"], dtype=np.float64)
        M = int(z["M"])
        e_stored = float(z["e_op"])
        grid = int(meta["grid"])
        s_floor = float(meta["s_floor"])

        # --- e_op re-computation (independent of the stored scalar) ---------
        S_raw = np.real(np.fft.fft2(_embed_kernel(rho, grid)))
        S = np.maximum(S_raw, s_floor * float(S_raw.mean()))
        H = _spectrum_of_taps(h, grid)
        e_re = float(np.max(np.abs(S * H * H - 1.0)))
        d_e = abs(e_re - e_stored)
        repro_ok = d_e <= REPRO_TOL
        admissible = e_re <= E_OP_GATE

        # --- keep-mask consistency vs the product mask ----------------------
        zp = np.load(paths.FOUNDRY_DATA / cfg["cutout"], allow_pickle=True)
        keep_prod = np.asarray(zp["keep_mask"]).astype(bool)
        keep_expect = erode_keep(keep_prod, M)
        keep_ok = bool(np.array_equal(keep_w, keep_expect))

        row = dict(
            file=str(f), file_sha256=hashlib.sha256(f.read_bytes()).hexdigest(),
            h_sha256=sha256_arr(h), keep_w_sha256=sha256_arr(keep_w),
            rho_kernel_sha256=sha256_arr(rho),
            M=M, taps_shape=list(h.shape), grid=grid, s_floor=s_floor,
            e_op_stored=e_stored, e_op_recomputed=e_re,
            e_op_repro_absdiff=d_e, e_op_repro_ok=bool(repro_ok),
            admissible_strict=bool(admissible),
            strict_expected=bool(cfg["strict"]),
            logdet_per_pix=float(z["logdet_per_pix"]),
            n_keep_w=int(keep_w.sum()),
            keep_mask_erosion_consistent=keep_ok,
            product=cfg["cutout"],
            kernel_meta_tag=meta.get("tag"),
            note=("relaxed arm: built against e_target=0.05 (P1b D3); "
                  "admissible_strict=False BY DESIGN" if not cfg["strict"]
                  else ""),
        )
        manifest["bundles"][tag] = row
        ok = repro_ok and keep_ok and (admissible == cfg["strict"])
        all_ok &= ok
        print(f"{tag:12s} e_op stored={e_stored:.6f} recomputed={e_re:.6f} "
              f"|d|={d_e:.2e} admissible={admissible} keep_ok={keep_ok} "
              f"-> {'OK' if ok else 'FAIL'}", flush=True)

    manifest["all_ok"] = bool(all_ok)
    manifest["wall_s"] = time.time() - t0
    dest = HERE / "data" / "whitener_manifest.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(manifest, indent=2))
    print(f"wrote {dest}  ({'ALL OK' if all_ok else 'FAILURES PRESENT'})")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
