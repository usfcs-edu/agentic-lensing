"""
Multimodal Universe (MMU) data helpers for the AION-1 reproduction.

The AION tutorials' pre-joined convenience files (provabgs_desi_ls.hdf5 etc.)
are gone (404); the canonical replacement is the MMU HuggingFace datasets. This
module downloads the needed parquet shards with ``hf_hub_download`` (robust,
same path the model loader uses) and exposes small helpers to (a) pull columns
into numpy and (b) turn them into the typed ``aion.modalities`` field arrays our
embedding harness consumes.

Confirmed MMU sources (see README / data-sourcing sweep):
  * MultimodalUniverse/desi_provabgs -- PROVABGS labels + LS photometry + ra/dec
                                        + object_id(=DESI TARGETID).  (task 1 labels)
  * MultimodalUniverse/desi          -- DESI EDR spectra; join by object_id.   (task 1 spectrum)
  * MultimodalUniverse/legacysurvey  -- g,r,i,z images (no ra/dec; brick objid).
  * MultimodalUniverse/gaia          -- Gaia DR3 XP coeffs (110=55+55) + ra/dec. (tasks 2,3,11)
  * MultimodalUniverse/gz10          -- Galaxy10 morphology labels + RGB images. (task 4)
  * MultimodalUniverse/apogee        -- APOGEE stellar labels.                  (task 3)
"""

from __future__ import annotations

import numpy as np
from huggingface_hub import HfApi, hf_hub_download

import _config as C  # noqa: F401  (ensures HF_HOME is set)


def list_parquet(repo: str) -> list[str]:
    files = HfApi().list_repo_files(repo, repo_type="dataset")
    return [f for f in files if f.endswith(".parquet")]


def download_parquet(repo: str, filename: str) -> str:
    return hf_hub_download(repo_id=repo, filename=filename, repo_type="dataset")


def read_table(repo: str, filenames: list[str], columns=None):
    """Download the listed parquet shards and return a single pyarrow Table
    (optionally projected to `columns`)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    tabs = []
    for fn in filenames:
        path = download_parquet(repo, fn)
        tabs.append(pq.read_table(path, columns=columns))
    return pa.concat_tables(tabs)


def col(table, name) -> np.ndarray:
    """Materialise one column as a numpy array (handles fixed-size-list ->
    2D, and struct.subfield via dotted name)."""
    if "." in name:
        top, sub = name.split(".", 1)
        arr = table.column(top).to_pylist()
        return np.array([_dig(a, sub) for a in arr])
    return np.array(table.column(name).to_pylist())


def _dig(d, dotted):
    for k in dotted.split("."):
        d = d[k]
    return d


def first_present(names, available):
    for n in names:
        if n in available:
            return n
    upper = {a.upper(): a for a in available}
    for n in names:
        if n.upper() in upper:
            return upper[n.upper()]
    return None
