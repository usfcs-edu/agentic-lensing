"""
Frozen-encoder embedding extraction for AION-1, with file-backed multi-GPU
sharding across the 7 local TITAN RTX cards.

Design notes
------------
* Inputs are described by *modality specs* that reference on-disk ``.npy``
  field files (memmapped), so a 240k x (4,96,96) image array is never copied
  into every worker process -- each worker only reads its own shard rows.
* One worker process per GPU; each loads its own frozen model copy (one model
  per card, so even xlarge's ~12 GB fits). Workers write their shard to a tmp
  ``.npy``; the parent concatenates in rank order.
* ``num_encoder_tokens`` is computed exactly as ``AION.forward`` does it
  (sum of per-modality token counts), so ``model.encode`` keeps all tokens.

Reuses: ``aion.model.AION.from_pretrained``/``.encode`` (aion/model.py:149),
``aion.codecs.CodecManager.encode`` (aion/codecs/manager.py), and the modality
dataclasses in aion/modalities.py.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np

import _config as C

_BOOL_FIELDS = {"mask"}


def load_model(variant: str, device: str = "cuda"):
    import torch

    from aion.codecs import CodecManager
    from aion.model import AION

    torch.set_grad_enabled(False)
    cm = CodecManager(device=device)
    model = AION.from_pretrained(C.MODELS[variant]).to(device).eval()
    return model, cm


def _resolve(cls_name: str):
    import aion.modalities as M

    return getattr(M, cls_name)


def _build_batch(specs, rows, device):
    """Build a list of modality objects for the given absolute row indices."""
    import torch

    mods = []
    for spec in specs:
        cls = _resolve(spec["cls"])
        kwargs = dict(spec.get("const", {}))
        for arg, arr in spec["_fields"].items():  # arr is a memmap
            chunk = np.asarray(arr[rows])
            if arg in _BOOL_FIELDS:
                t = torch.as_tensor(chunk.astype(bool), device=device)
            else:
                t = torch.as_tensor(chunk.astype(np.float32), device=device)
            kwargs[arg] = t
        mods.append(cls(**kwargs))
    return mods


def _num_encoder_tokens(tokens) -> int:
    n = 0
    for v in tokens.values():
        n += v.shape[1] if v.dim() == 2 else 1
    return n


def extract_range(specs, variant, pool, batch_size, amp, device, lo, hi):
    """Extract embeddings for rows [lo, hi). Returns (n, D) if pool=='mean'
    else (n, T, D)."""
    import torch

    model, cm = load_model(variant, device)
    out = []
    for start in range(lo, hi, batch_size):
        rows = np.arange(start, min(start + batch_size, hi))
        mods = _build_batch(specs, rows, device)
        tokens = cm.encode(*mods)
        net = _num_encoder_tokens(tokens)
        ctx = torch.autocast("cuda", dtype=torch.float16) if amp else _nullctx()
        with ctx:
            emb = model.encode(tokens, num_encoder_tokens=net)  # (b, T, D)
        emb = emb.float()
        if pool == "mean":
            emb = emb.mean(dim=1)
        out.append(emb.cpu().numpy().astype(np.float16))
        del tokens, emb
    return np.concatenate(out, axis=0)


class _nullctx:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


def _attach_fields(specs):
    """Memmap the field .npy paths referenced by each spec."""
    out = []
    for spec in specs:
        s = dict(spec)
        s["_fields"] = {arg: np.load(p, mmap_mode="r") for arg, p in spec["fields"].items()}
        out.append(s)
    return out


def _n_rows(specs) -> int:
    first = next(iter(specs[0]["fields"].values()))
    return int(np.load(first, mmap_mode="r").shape[0])


def _worker(rank, specs, variant, pool, batch_size, amp, gpus, tmpdir, n):
    gpu = gpus[rank]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    import torch  # re-import inside process

    torch.cuda.set_device(0)
    bounds = np.linspace(0, n, len(gpus) + 1).astype(int)
    lo, hi = int(bounds[rank]), int(bounds[rank + 1])
    specs_m = _attach_fields(specs)
    res = extract_range(specs_m, variant, pool, batch_size, amp, "cuda", lo, hi)
    np.save(Path(tmpdir) / f"shard_{rank}.npy", res)


def multi_gpu_extract(specs, variant, out_file, pool="mean", gpus=(0, 1, 2, 3, 4, 5, 6),
                      batch_size=None, amp=None):
    """Extract embeddings across multiple GPUs and save to ``out_file`` (.npy).

    ``specs``: list of {"cls": <modality class name>, "fields": {arg: path.npy},
    "const": {...}}. Returns the assembled array.
    """
    import torch.multiprocessing as mp

    gpus = list(gpus)
    batch_size = batch_size or C.DEFAULT_BATCH[variant]
    amp = C.DEFAULT_AMP[variant] if amp is None else amp
    n = _n_rows(specs)
    gpus = gpus[: max(1, min(len(gpus), n))]

    with tempfile.TemporaryDirectory(dir=C.DATA) as td:
        if len(gpus) == 1:
            _worker(0, specs, variant, pool, batch_size, amp, gpus, td, n)
        else:
            mp.spawn(_worker, args=(specs, variant, pool, batch_size, amp, gpus, td, n),
                     nprocs=len(gpus), join=True)
        parts = [np.load(Path(td) / f"shard_{r}.npy") for r in range(len(gpus))]
    arr = np.concatenate(parts, axis=0)
    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    np.save(out_file, arr)
    return arr


# --- spec builders (convenience) --------------------------------------------
def scalar_spec(cls_name: str, path: str):
    return {"cls": cls_name, "fields": {"value": path}, "const": {}}


def image_spec(cls_name: str, flux_path: str, bands: list[str]):
    return {"cls": cls_name, "fields": {"flux": flux_path}, "const": {"bands": bands}}


def spectrum_spec(cls_name: str, flux, ivar, mask, wavelength):
    return {"cls": cls_name,
            "fields": {"flux": flux, "ivar": ivar, "mask": mask, "wavelength": wavelength},
            "const": {}}
