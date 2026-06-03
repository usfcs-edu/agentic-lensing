"""
04 -- Fetch DESI spectra and key-join them to PROVABGS (task 1, +spectrum).

MMU desi_provabgs has no spectrum; MultimodalUniverse/desi does, joined by
object_id (= DESI TARGETID, an exact integer-string key -- no positional
cross-match needed). Downloads the 7 EDR parquet shards with hf_hub_download
(robust), keeps only rows whose TARGETID is a PROVABGS galaxy, and saves the
spectrum field arrays aligned to a subset index into the PROVABGS arrays.

Outputs (in data/raw/provabgs/):
  spec_index.npy   -- indices into the PROVABGS arrays that have a spectrum
  spec_flux.npy / spec_ivar.npy / spec_mask.npy / spec_wave.npy  -- (M, L)

Run: HF_HOME=... python 04_fetch_desi_spectra.py
"""

import json

import numpy as np
import pyarrow.parquet as pq

import _config as C
import _data_mmu as D

REPO = "MultimodalUniverse/desi"
OUT = C.RAW / "provabgs"


def _struct_field(tab, struct, field):
    arr = tab.column(struct)
    try:
        return arr.field(field)
    except Exception:
        return arr.combine_chunks().field(field)


def main():
    tid = np.load(OUT / "targetid.npy")  # provabgs TARGETIDs (str), provabgs order
    pg_pos = {t: i for i, t in enumerate(tid)}
    files = [f for f in D.list_parquet(REPO) if "edr_sv3" in f] or D.list_parquet(REPO)
    print("desi parquet shards:", len(files))

    rows_idx, flux, ivar, mask, wave = [], [], [], [], []
    for fi, fn in enumerate(files):
        path = D.download_parquet(REPO, fn)
        t = pq.read_table(path, columns=["object_id", "spectrum"])
        oids = [str(x) for x in t.column("object_id").to_pylist()]
        keep = [j for j, o in enumerate(oids) if o in pg_pos]
        if not keep:
            print(f"  shard {fi}: 0 / {len(oids)} match"); continue
        sub = t.take(keep)
        fl = np.asarray(_struct_field(sub, "spectrum", "flux").to_pylist(), dtype=np.float32)
        iv = np.asarray(_struct_field(sub, "spectrum", "ivar").to_pylist(), dtype=np.float32)
        mk = np.asarray(_struct_field(sub, "spectrum", "mask").to_pylist(), dtype=bool)
        wv = np.asarray(_struct_field(sub, "spectrum", "lambda").to_pylist(), dtype=np.float32)
        for j, kj in enumerate(keep):
            rows_idx.append(pg_pos[oids[kj]])
        flux.append(fl); ivar.append(iv); mask.append(mk); wave.append(wv)
        print(f"  shard {fi}: matched {len(keep)} / {len(oids)} (cum {sum(len(x) for x in flux)})", flush=True)

    if not rows_idx:
        print("NO OVERLAP between PROVABGS and DESI EDR -- +spectrum config not available")
        (OUT / "spec_meta.json").write_text(json.dumps({"n": 0}))
        return
    idx = np.array(rows_idx)
    order = np.argsort(idx)  # keep ascending provabgs order
    idx = idx[order]
    flux = np.concatenate(flux)[order]
    ivar = np.concatenate(ivar)[order]
    mask = np.concatenate(mask)[order]
    wave = np.concatenate(wave)[order]
    np.save(OUT / "spec_index.npy", idx)
    np.save(OUT / "spec_flux.npy", flux)
    np.save(OUT / "spec_ivar.npy", ivar)
    np.save(OUT / "spec_mask.npy", mask)
    np.save(OUT / "spec_wave.npy", wave)
    (OUT / "spec_meta.json").write_text(json.dumps(
        {"n": int(len(idx)), "spec_len": int(flux.shape[1]),
         "wave_range": [float(wave.min()), float(wave.max())]}, indent=2))
    print(f"DESI_SPECTRA_OK matched {len(idx)} galaxies, spec_len {flux.shape[1]}")


if __name__ == "__main__":
    main()
